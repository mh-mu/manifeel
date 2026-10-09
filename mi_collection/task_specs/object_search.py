"""TaskSpec for TacSL object search (``TacSLTaskObjectSearch``).

A battery (sim actor ``plug``) rests inside an open box (sim actor ``socket``). The
gripper starts open above the box, descends into it, finds the battery -- the front
camera sees only the box wall, so the battery has to be found by touch -- closes on
it via ``action[6]`` and lifts it out.

Unlike peg insertion the gripper is not closed on the object at reset, so the
hand-object pairs switch on and off, and the fingers work *inside* the box, so the
hand-env pairs can fire. Each hand side is the finger link plus its elastomer: the
finger links carry collision geometry too (the peg-insertion incidence probe saw
``panda_*finger <-> plug`` live 100 % of steps), and while searching the box it is
the finger, not the gel, that is likely to meet a wall. ``panda_hand`` against the
box rim and the battery against the table are optional pairs; whether either ever
fires is a pilot question.

Success is stricter than upstream. ``_check_success`` is a band on the item-to-box
keypoint distance, which an item merely tipping over also passes through: 6 of 63
upstream successes in 100 instrumented vistac rollouts were tip-overs or pushes
with no finger on the battery, 4 of them at t=2-3, and the collector truncates at
the first success. So success here is the band AND both fingers on the item AND
the item lifted AND upright; that rule kept all 57 real lifts of those rollouts
and rejected all 6 false ones. The upstream flag is still recorded per step as
``success_upstream``, so the paper's metric stays recomputable.

Every episode also carries per-step motion-primitive labels
(:mod:`mi_collection.motion_primitives`).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from mi_collection import motion_primitives
from mi_collection.contact import DEFAULT_CONTACT_THRESHOLD, pair_descriptor
from mi_collection.task_spec import ContactPairSpec, TaskSpec

_LEFT_HAND: Tuple[str, ...] = ("panda_leftfinger", "elastomer_left")
_RIGHT_HAND: Tuple[str, ...] = ("panda_rightfinger", "elastomer_right")

#: ``state_full`` composition. ``state`` (the policy's own channel) is the first
#: seven columns; the gripper width is new relative to peg insertion because here
#: the policy actuates it.
_STATE_FULL_LAYOUT: Tuple[Tuple[str, int], ...] = (
    ("ee_pos", 3),
    ("ee_quat", 4),
    ("item_pos", 3),
    ("item_quat", 4),
    ("box_pos", 3),
    ("box_quat", 4),
    ("gripper_dof_pos", 2),
)


class ObjectSearchSpec(TaskSpec):
    """Contact pairs, success and world state for the object-search task."""

    name: str = "object_search"
    env_name: str = "manifeel_tacsl_object_search"

    def __init__(
        self,
        strict_success: bool = True,
        min_lift: float = 0.05,
        max_tilt_deg: float = 30.0,
        item_rest_height: float = 0.0048,
        grasp_threshold: float = DEFAULT_CONTACT_THRESHOLD,
        primitive_params: motion_primitives.PrimitiveParams | None = None,
    ) -> None:
        self.strict_success = strict_success
        self.min_lift = min_lift
        self.max_tilt_deg = max_tilt_deg
        #: Item centre above the box origin when resting upright: the settled
        #: value of ``plug_z - socket_z`` across 100 rollouts (median, t=30).
        self.item_rest_height = item_rest_height
        self.grasp_threshold = grasp_threshold
        self.primitive_params = primitive_params or motion_primitives.PrimitiveParams(
            item_rest_height=item_rest_height
        )

    def contact_pairs(self, include_optional: bool = True) -> List[ContactPairSpec]:
        pairs = [
            ContactPairSpec("contact_left_item", "hand-object", _LEFT_HAND, ("plug",)),
            ContactPairSpec(
                "contact_right_item", "hand-object", _RIGHT_HAND, ("plug",)
            ),
            ContactPairSpec("contact_left_box", "hand-env", _LEFT_HAND, ("socket",)),
            ContactPairSpec("contact_right_box", "hand-env", _RIGHT_HAND, ("socket",)),
            ContactPairSpec(
                "contact_hand_box",
                "hand-env",
                ("panda_hand",),
                ("socket",),
                optional=True,
            ),
            ContactPairSpec("contact_item_box", "object-env", ("plug",), ("socket",)),
            ContactPairSpec(
                "contact_item_table",
                "object-env",
                ("plug",),
                ("table",),
                optional=True,
            ),
        ]
        if include_optional:
            return pairs
        return [p for p in pairs if not p.optional]

    def body_index_map(self, task: object) -> Dict[str, int]:
        table = dict(task.franka_body_ids_env)
        table["plug"] = int(task.plug_body_id_env)
        table["socket"] = int(task.socket_body_id_env)
        table["table"] = int(task.table_body_id)
        return {name: int(index) for name, index in table.items()}

    def success(self, task: object) -> np.ndarray:
        """Upstream band AND held by both fingers AND lifted AND upright.

        The band (``close_error_thresh_lower`` < d < ``upper``) alone also reads
        true while the item tips through 42-74 degrees; see the module docstring.
        Success can switch off again (item carried past the band, or dropped);
        the accumulator keys on the first success, which is the event that matters.
        """
        upstream = task._check_success()
        if not self.strict_success:
            return _to_uint8(upstream)
        held = self._held(task)
        lift = task.plug_pos[:, 2] - task.socket_pos[:, 2] - self.item_rest_height
        cos_tilt = (_z_axis(task.plug_quat) * _z_axis(task.socket_quat)).sum(-1)
        upright = cos_tilt >= float(np.cos(np.deg2rad(self.max_tilt_deg)))
        return _to_uint8(upstream.bool() & held & (lift >= self.min_lift) & upright)

    def success_rule(self) -> Dict[str, object]:
        if not self.strict_success:
            return {"predicate": "upstream _check_success() keypoint band"}
        return {
            "predicate": (
                "upstream _check_success() keypoint band AND both hand groups "
                "(finger link + elastomer) touching the item AND item lifted AND "
                "item axis within max_tilt_deg of the box axis"
            ),
            "min_lift_m": self.min_lift,
            "max_tilt_deg": self.max_tilt_deg,
            "item_rest_height_m": self.item_rest_height,
            "grasp_threshold_n": self.grasp_threshold,
            "upstream_channel": "success_upstream",
        }

    def extra_step_channels(self, task: object) -> Dict[str, np.ndarray]:
        return {"success_upstream": _to_uint8(task._check_success()).reshape(-1, 1)}

    def annotate_episode(
        self, episode: Dict[str, np.ndarray], control_dt: float
    ) -> Dict[str, np.ndarray]:
        labels, phase = motion_primitives.label_episode(
            episode, control_dt, self.primitive_params
        )
        return {
            "motion_primitive": labels.reshape(-1, 1),
            "primitive_phase": phase.reshape(-1, 1),
        }

    def finalize_store(self, path: str) -> Dict[str, object]:
        block = motion_primitives.write_store_annotations(path, self.primitive_params)
        return {"motion_primitives": block}

    def collection_params(self) -> Dict[str, object]:
        return {
            "success_rule": self.success_rule(),
            "motion_primitives": {
                "version": motion_primitives.VERSION,
                "params": self.primitive_params.to_dict(),
            },
        }

    def _held(self, task: object) -> torch.Tensor:
        pairwise = task.contact_force_pairwise
        item = self.resolve_bodies(task, ("plug",))
        sides = [
            pair_descriptor(
                pairwise, self.resolve_bodies(task, hand), item, self.grasp_threshold
            )[:, 0]
            > 0.5
            for hand in (_LEFT_HAND, _RIGHT_HAND)
        ]
        return sides[0] & sides[1]

    def state_full_layout(self) -> List[Tuple[str, int]]:
        return list(_STATE_FULL_LAYOUT)

    def state_full(self, task: object) -> np.ndarray:
        blocks = [
            task.fingertip_midpoint_pos,
            task.fingertip_midpoint_quat,
            task.plug_pos,
            task.plug_quat,
            task.socket_pos,
            task.socket_quat,
            task.gripper_dof_pos,
        ]
        arrays = [b.detach().to("cpu").numpy().astype(np.float32) for b in blocks]
        return np.concatenate(arrays, axis=1)

    def analyzer_recommendations(self) -> Dict[str, object]:
        return {
            "force_channels": ["tactile_force_field_right"],
            "third_view_channel": "front",
            "wrist_channel": "wrist",
            "extra_image_channels": ["right_tactile_camera_taxim"],
            "stage_markers": True,
            "stage_order": "objsearch",
            "notes": (
                "meta/stage_* hold each motion primitive's first entry: run with "
                "--stage-order objsearch (fbsource stages.py); data/motion_primitive "
                "has every entry. The front camera sees only the outside of the "
                "box, so the battery is occluded from the policy's own view. Image "
                "encoders trained on peg insertion are NOT valid here -- train "
                "object-search encoders against contact_item_box, or use "
                "--camera-downsample-only."
            ),
        }


def _to_uint8(flags: torch.Tensor) -> np.ndarray:
    return flags.detach().to("cpu").numpy().astype(np.uint8)


def _z_axis(quat: torch.Tensor) -> torch.Tensor:
    """The body z axis in world frame, for Isaac Gym ``(x, y, z, w)`` quaternions."""
    x, y, z, w = quat.unbind(-1)
    return torch.stack(
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1
    )
