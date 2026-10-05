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
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

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
        """``_check_success()`` -- the item-to-box keypoint distance inside the band.

        The band (``close_error_thresh_lower`` < d < ``upper``) means "lifted clear
        of the box but not yet past the upper bound", so success can switch off
        again if the item is carried too far; the accumulator keys on the first
        success, which is the event that matters.
        """
        return task._check_success().detach().to("cpu").numpy().astype(np.uint8)

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
            "stage_markers": False,
            "notes": (
                "No stage labels yet (they need pilot data to calibrate): run with "
                "--no-stage-markers. The front camera sees only the outside of the "
                "box, so the battery is occluded from the policy's own view. Image "
                "encoders trained on peg insertion are NOT valid here -- train "
                "object-search encoders against contact_item_box, or use "
                "--camera-downsample-only."
            ),
        }
