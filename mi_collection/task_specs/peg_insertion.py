"""TaskSpec for TacSL peg insertion (``TacSLTaskInsertion``).

The gripper has two GelSight fingers, so the analogue of ShapeFilter's eleven
actor pairs is five: each elastomer against the plug (hand-object) and against
the socket (hand-env), plus plug against socket (object-env).  ``plug`` against
the table ``box`` is a sixth, optional pair -- it is the only other object-env
contact the scene can produce, and whether it ever fires is a pilot question.

Only ``elastomer_{left,right}`` carry collision geometry on the finger tips
(``elastomer_tip_*`` are massless frames for the tactile cameras), so the
hand-side groups are single-body.  They are still declared as tuples: the
descriptor sums over the group, which is what a multi-link receiver would need.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from mi_collection.task_spec import ContactPairSpec, TaskSpec

#: ``state_full`` composition.  ``state`` (the policy's own channel) is the
#: first seven columns, so a consumer can slice it out of ``state_full``.
_STATE_FULL_LAYOUT: Tuple[Tuple[str, int], ...] = (
    ("ee_pos", 3),
    ("ee_quat", 4),
    ("plug_pos", 3),
    ("plug_quat", 4),
    ("socket_pos", 3),
    ("socket_quat", 4),
)


class PegInsertionSpec(TaskSpec):
    """Contact pairs, success and world state for the peg-in-hole task."""

    name: str = "peg_insertion"
    env_name: str = "manifeel_tacsl_insertion"

    def contact_pairs(self, include_optional: bool = True) -> List[ContactPairSpec]:
        pairs = [
            ContactPairSpec(
                "contact_left_plug", "hand-object", ("elastomer_left",), ("plug",)
            ),
            ContactPairSpec(
                "contact_right_plug", "hand-object", ("elastomer_right",), ("plug",)
            ),
            ContactPairSpec(
                "contact_left_socket", "hand-env", ("elastomer_left",), ("socket",)
            ),
            ContactPairSpec(
                "contact_right_socket", "hand-env", ("elastomer_right",), ("socket",)
            ),
            ContactPairSpec(
                "contact_plug_socket", "object-env", ("plug",), ("socket",)
            ),
            ContactPairSpec(
                "contact_plug_table", "object-env", ("plug",), ("table",), optional=True
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
        """``_check_success()`` -- keypoint distance below the insertion tolerance.

        Read directly rather than off ``reset_buf`` so the collector never
        depends on the task's reset semantics (here ``reset_buf`` *is* the
        success flag, but that is a property of this task, not a contract).
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
                "No subtask machine in this task, so there are no meta/stage_* "
                "arrays: run with --no-stage-markers. The image encoders under "
                "mi_image_encoders/ were trained on Mochi/ShapeFilter imagery "
                "against the 8-d descriptor and are NOT valid here -- use "
                "--camera-downsample-only until ManiFeel encoders exist."
            ),
        }
