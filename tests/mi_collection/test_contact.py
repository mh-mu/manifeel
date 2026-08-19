"""Contact-descriptor unit tests.  No GPU, no sim.

Run from the repo root with the conda python and
``LD_LIBRARY_PATH=<env>/lib``::

    python -m unittest discover -s tests/mi_collection -t . -v
"""

import unittest

import numpy as np
import torch

from mi_collection.contact import (
    build_pair_indices,
    CONTACT_STATE_DIM,
    contact_descriptors,
    FORCE_SLICE,
    IN_CONTACT,
    pair_descriptor,
    sidecar_contact_pairs,
)
from mi_collection.task_spec import CONTACT_LABELS
from mi_collection.task_specs.peg_insertion import PegInsertionSpec


def antisymmetric(num_envs, num_bodies, entries):
    """A pairwise tensor with ``P[i,j] = -P[j,i]`` and a zero diagonal."""
    tensor = torch.zeros(num_envs, num_bodies, num_bodies, 3)
    for env, i, j, force in entries:
        value = torch.tensor(force, dtype=torch.float32)
        tensor[env, i, j] = value
        tensor[env, j, i] = -value
    return tensor


class FakeTask:
    """Just the attributes ``PegInsertionSpec.body_index_map`` reads."""

    def __init__(self):
        self.franka_body_ids_env = {
            "panda_hand": 5,
            "panda_leftfinger": 6,
            "elastomer_left": 7,
            "panda_rightfinger": 8,
            "elastomer_right": 9,
        }
        self.plug_body_id_env = 1
        self.socket_body_id_env = 2
        self.table_body_id = 0


class ContactDescriptorTest(unittest.TestCase):
    def test_force_is_on_group_a_from_group_b(self):
        # 19.7 N pushes body 1 (plug) up, away from body 0 (table).
        pairwise = antisymmetric(1, 3, [(0, 1, 0, [0.0, 0.0, 19.719])])
        on_plug = pair_descriptor(pairwise, [1], [0])
        on_table = pair_descriptor(pairwise, [0], [1])
        np.testing.assert_allclose(
            on_plug[0, FORCE_SLICE].numpy(), [0.0, 0.0, 19.719], rtol=1e-6
        )
        np.testing.assert_allclose(
            on_table[0, FORCE_SLICE].numpy(), [0.0, 0.0, -19.719], rtol=1e-6
        )

    def test_descriptor_width(self):
        pairwise = antisymmetric(2, 4, [(0, 1, 2, [1.0, 0.0, 0.0])])
        self.assertEqual(pair_descriptor(pairwise, [1], [2]).shape, (2, 4))
        self.assertEqual(CONTACT_STATE_DIM, 4)

    def test_multi_body_group_sums(self):
        pairwise = antisymmetric(
            1,
            5,
            [(0, 3, 1, [1.0, 0.0, 0.0]), (0, 4, 1, [0.0, 2.0, 0.0])],
        )
        grouped = pair_descriptor(pairwise, [3, 4], [1])
        np.testing.assert_allclose(
            grouped[0, FORCE_SLICE].numpy(), [1.0, 2.0, 0.0], rtol=1e-6
        )

    def test_in_contact_survives_a_cancelling_sum(self):
        """Two real contacts whose vectors cancel must still read in_contact."""
        pairwise = antisymmetric(
            1,
            5,
            [(0, 3, 1, [1.0, 0.0, 0.0]), (0, 4, 1, [-1.0, 0.0, 0.0])],
        )
        grouped = pair_descriptor(pairwise, [3, 4], [1])
        np.testing.assert_allclose(grouped[0, FORCE_SLICE].numpy(), [0.0, 0.0, 0.0])
        self.assertEqual(float(grouped[0, IN_CONTACT]), 1.0)

    def test_threshold_boundaries(self):
        below = antisymmetric(1, 3, [(0, 1, 2, [1e-4, 0.0, 0.0])])
        above = antisymmetric(1, 3, [(0, 1, 2, [2e-3, 0.0, 0.0])])
        self.assertEqual(
            float(pair_descriptor(below, [1], [2], 1e-3)[0, IN_CONTACT]), 0.0
        )
        self.assertEqual(
            float(pair_descriptor(above, [1], [2], 1e-3)[0, IN_CONTACT]), 1.0
        )
        # A looser threshold does admit the 1e-4 reading.
        self.assertEqual(
            float(pair_descriptor(below, [1], [2], 1e-6)[0, IN_CONTACT]), 1.0
        )

    def test_no_contact_is_all_zero(self):
        pairwise = torch.zeros(3, 4, 4, 3)
        descriptor = pair_descriptor(pairwise, [1], [2])
        np.testing.assert_allclose(descriptor.numpy(), np.zeros((3, 4)))

    def test_cpu_pipeline_none_tensor_raises(self):
        spec = PegInsertionSpec()
        pairs = build_pair_indices(FakeTask(), spec)
        with self.assertRaisesRegex(RuntimeError, "CPU pipeline"):
            contact_descriptors(None, pairs)

    def test_stacked_descriptor_shape_and_order(self):
        spec = PegInsertionSpec()
        task = FakeTask()
        pairs = build_pair_indices(task, spec)
        pairwise = antisymmetric(2, 10, [(0, 7, 1, [3.0, 0.0, 0.0])])
        stacked = contact_descriptors(pairwise, pairs)
        self.assertEqual(stacked.shape, (2, len(pairs), 4))
        names = [p.name for p in pairs]
        self.assertEqual(names[0], "contact_left_plug")
        np.testing.assert_allclose(
            stacked[0, 0, FORCE_SLICE].numpy(), [3.0, 0.0, 0.0], rtol=1e-6
        )
        # Everything else is untouched.
        self.assertEqual(float(stacked[0, 1:, IN_CONTACT].sum()), 0.0)
        self.assertEqual(float(stacked[1, :, IN_CONTACT].sum()), 0.0)


class PairSpecTest(unittest.TestCase):
    def test_all_three_analyzer_labels_present(self):
        spec = PegInsertionSpec()
        labels = {p.label for p in spec.contact_pairs()}
        self.assertEqual(labels, set(CONTACT_LABELS))

    def test_channel_names_are_three_token(self):
        for pair in PegInsertionSpec().contact_pairs():
            self.assertEqual(len(pair.name.split("_")), 3, pair.name)

    def test_optional_pair_can_be_dropped(self):
        spec = PegInsertionSpec()
        with_optional = [p.name for p in spec.contact_pairs(True)]
        without = [p.name for p in spec.contact_pairs(False)]
        self.assertIn("contact_plug_table", with_optional)
        self.assertNotIn("contact_plug_table", without)
        # Dropping the optional pair must not drop a required label.
        self.assertEqual(
            {p.label for p in spec.contact_pairs(False)}, set(CONTACT_LABELS)
        )

    def test_sidecar_carries_every_label_and_body_group(self):
        spec = PegInsertionSpec()
        pairs = build_pair_indices(FakeTask(), spec)
        labels, provenance = sidecar_contact_pairs(pairs)
        self.assertEqual(set(labels.values()), set(CONTACT_LABELS))
        self.assertEqual(
            provenance["contact_left_plug"]["bodies_a"], ["elastomer_left"]
        )
        self.assertEqual(provenance["contact_left_plug"]["body_ids_b"], [1])

    def test_unknown_body_name_fails_loudly(self):
        spec = PegInsertionSpec()
        with self.assertRaisesRegex(KeyError, "not in this task"):
            spec.resolve_bodies(FakeTask(), ["elastomer_middle"])

    def test_malformed_channel_name_rejected(self):
        from mi_collection.task_spec import ContactPairSpec

        with self.assertRaises(ValueError):
            ContactPairSpec("contact_left", "hand-object", ("a",), ("b",))
        with self.assertRaises(ValueError):
            ContactPairSpec("contact_a_b", "hand-thing", ("a",), ("b",))
