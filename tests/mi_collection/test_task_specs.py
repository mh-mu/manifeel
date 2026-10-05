"""Task-spec registry and object-search spec tests.  No GPU, no sim.

    python -m unittest discover -s tests/mi_collection -t . -v
"""

import unittest

import numpy as np
import torch

from mi_collection.contact import build_pair_indices, sidecar_contact_pairs
from mi_collection.task_spec import CONTACT_LABELS
from mi_collection.task_specs import get_task_spec, TASK_SPECS
from mi_collection.task_specs.object_search import ObjectSearchSpec
from mi_collection.task_specs.peg_insertion import PegInsertionSpec

from tests.mi_collection.test_contact import antisymmetric, FakeTask


class FakeObjectSearchTask(FakeTask):
    """Adds what ``ObjectSearchSpec.state_full`` reads on top of the body table."""

    def __init__(self, num_envs=2):
        super().__init__()
        self.franka_body_ids_env["panda_hand"] = 5
        self.fingertip_midpoint_pos = torch.full((num_envs, 3), 1.0)
        self.fingertip_midpoint_quat = torch.full((num_envs, 4), 2.0)
        self.plug_pos = torch.full((num_envs, 3), 3.0)
        self.plug_quat = torch.full((num_envs, 4), 4.0)
        self.socket_pos = torch.full((num_envs, 3), 5.0)
        self.socket_quat = torch.full((num_envs, 4), 6.0)
        self.gripper_dof_pos = torch.full((num_envs, 2), 7.0)


class RegistryTest(unittest.TestCase):
    def test_both_tasks_are_registered(self):
        self.assertEqual(set(TASK_SPECS), {"peg_insertion", "object_search"})
        self.assertIsInstance(get_task_spec("object_search"), ObjectSearchSpec)
        self.assertIsInstance(get_task_spec("peg_insertion"), PegInsertionSpec)

    def test_unknown_spec_lists_the_known_ones(self):
        with self.assertRaisesRegex(KeyError, "object_search"):
            get_task_spec("nonesuch")

    def test_env_names_are_distinct_and_peg_insertion_is_unchanged(self):
        # The published peg-insertion sidecars carry this exact string.
        self.assertEqual(PegInsertionSpec.env_name, "manifeel_tacsl_insertion")
        self.assertNotEqual(ObjectSearchSpec.env_name, PegInsertionSpec.env_name)


class ObjectSearchSpecTest(unittest.TestCase):
    def test_all_three_analyzer_labels_present_without_optional_pairs(self):
        spec = ObjectSearchSpec()
        for include_optional in (True, False):
            labels = {p.label for p in spec.contact_pairs(include_optional)}
            self.assertEqual(labels, set(CONTACT_LABELS))

    def test_channel_names_are_three_token_and_unique(self):
        names = [p.name for p in ObjectSearchSpec().contact_pairs()]
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            self.assertEqual(len(name.split("_")), 3, name)

    def test_optional_pairs(self):
        spec = ObjectSearchSpec()
        optional = {p.name for p in spec.contact_pairs(True)} - {
            p.name for p in spec.contact_pairs(False)
        }
        self.assertEqual(optional, {"contact_hand_box", "contact_item_table"})

    def test_hand_side_includes_the_finger_link(self):
        pairs = build_pair_indices(FakeObjectSearchTask(), ObjectSearchSpec())
        _, provenance = sidecar_contact_pairs(pairs)
        self.assertEqual(
            provenance["contact_left_item"]["bodies_a"],
            ["panda_leftfinger", "elastomer_left"],
        )
        self.assertEqual(provenance["contact_left_item"]["body_ids_a"], [6, 7])
        self.assertEqual(provenance["contact_item_box"]["body_ids_b"], [2])

    def test_finger_link_contact_is_seen_through_the_group(self):
        # Body 6 (panda_leftfinger) touches body 2 (socket): a wall hit during search.
        from mi_collection.contact import contact_descriptors, IN_CONTACT

        pairs = build_pair_indices(FakeObjectSearchTask(), ObjectSearchSpec())
        pairwise = antisymmetric(1, 10, [(0, 6, 2, [0.0, 1.5, 0.0])])
        stacked = contact_descriptors(pairwise, pairs)
        live = {p.name for p, v in zip(pairs, stacked[0, :, IN_CONTACT]) if v > 0}
        self.assertEqual(live, {"contact_left_box"})

    def test_state_full_matches_its_layout(self):
        spec = ObjectSearchSpec()
        state = spec.state_full(FakeObjectSearchTask(num_envs=2))
        widths = [w for _, w in spec.state_full_layout()]
        self.assertEqual(state.shape, (2, sum(widths)))
        self.assertEqual(state.dtype, np.float32)
        # Blocks land in layout order: 3+4+3+4+3+4 columns, then the gripper.
        np.testing.assert_allclose(state[0, -2:], [7.0, 7.0])
        np.testing.assert_allclose(state[0, :7], [1.0] * 3 + [2.0] * 4)
        self.assertEqual(sum(widths), 23)


if __name__ == "__main__":
    unittest.main()
