"""Motion-primitive labeller tests.  No GPU, no sim.

    python -m unittest tests.mi_collection.test_motion_primitives -v
"""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import zarr

from mi_collection import motion_primitives as mp
from mi_collection.writer import write_sidecar

DT = 0.1
OPEN, CLOSED = 0.04, 0.02


def scripted_episode(segments):
    """Build channels from ``(steps, ee_dz_per_step, item_dz_per_step, cmd, held,
    touching)`` segments; the EE starts 0.074 above the box, the item at rest."""
    rows = []
    ee, item = 0.074, mp.PrimitiveParams().item_rest_height
    width = 2 * OPEN
    for steps, ee_dz, item_dz, cmd, held, touching in segments:
        for _ in range(steps):
            ee += ee_dz
            item += item_dz
            width += (-0.004 if cmd == CLOSED else 0.004)
            width = float(np.clip(width, 0.03, 2 * OPEN))
            rows.append((ee, item, width, cmd, held, touching))
    n = len(rows)
    state = np.zeros((n, 23), dtype=np.float32)
    action = np.zeros((n, 7), dtype=np.float32)
    left = np.zeros((n, 4), dtype=np.float32)
    right = np.zeros((n, 4), dtype=np.float32)
    for t, (ee, item, width, cmd, held, touching) in enumerate(rows):
        state[t, 2], state[t, 9], state[t, 16] = ee, item, 0.0
        state[t, 21:23] = width / 2
        action[t, 6] = cmd
        left[t, 0] = float(held or touching)
        right[t, 0] = float(held)
    return {
        "state_full": state,
        "action": action,
        "contact_left_item": left,
        "contact_right_item": right,
    }


#: descend 10 -> search 10 -> probe 6 -> close 8 -> lift 10 -> hold 8
NOMINAL = [
    (10, -0.006, 0.0, OPEN, False, False),
    (10, 0.0, 0.0, OPEN, False, False),
    (6, 0.0, 0.0, OPEN, False, True),
    (8, 0.0, 0.0, CLOSED, True, False),
    (10, 0.01, 0.01, CLOSED, True, False),
    (8, 0.0, 0.0, CLOSED, True, False),
]


def names(labels):
    starts, _ = mp.run_bounds(labels)
    return [mp.PRIMITIVES[int(labels[s])] for s in starts]


class LabelTest(unittest.TestCase):
    def test_nominal_sequence(self):
        labels, _ = mp.label_episode(scripted_episode(NOMINAL), DT)
        self.assertEqual(
            names(labels), ["descend", "search", "probe", "close", "lift", "hold"]
        )

    def test_closed_hand_rising_without_the_item_is_a_failed_lift(self):
        episode = scripted_episode(
            NOMINAL[:3]
            + [(8, 0.0, 0.0, CLOSED, False, False), (10, 0.01, 0.0, CLOSED, False, False)]
        )
        labels, _ = mp.label_episode(episode, DT)
        self.assertEqual(names(labels)[-1], "failed_lift")

    def test_empty_closed_hand_above_the_box_is_a_failed_lift(self):
        # Close on nothing in the box, rise 0.2 m, then wait up there.
        episode = scripted_episode(
            NOMINAL[:2]
            + [(8, 0.0, 0.0, CLOSED, False, False), (10, 0.02, 0.0, CLOSED, False, False)]
            + [(10, 0.0, 0.0, CLOSED, False, False)]
        )
        labels, _ = mp.label_episode(episode, DT)
        self.assertEqual(names(labels)[-2:], ["close", "failed_lift"])

    def test_open_idle_hand_above_the_box_hovers(self):
        episode = scripted_episode(
            [(6, 0.0, 0.0, OPEN, False, False)] + NOMINAL[:2]
        )
        labels, _ = mp.label_episode(episode, DT)
        self.assertEqual(names(labels), ["hover", "descend", "search"])

    def test_regrasp_releases_then_closes_again(self):
        episode = scripted_episode(
            NOMINAL[:4]
            + [(8, 0.0, 0.0, OPEN, False, False)]
            + [(8, 0.0, 0.0, CLOSED, True, False)]
        )
        labels, _ = mp.label_episode(episode, DT)
        seq = names(labels)
        self.assertIn("release", seq)
        self.assertEqual(seq[-1], "close")
        self.assertGreater(seq.count("close"), 1)

    def test_phase_resets_at_every_run(self):
        labels, phase = mp.label_episode(scripted_episode(NOMINAL), DT)
        starts, lengths = mp.run_bounds(labels)
        for start, length in zip(starts, lengths):
            self.assertEqual(phase[start], 0.0)
            if length > 1:
                self.assertAlmostEqual(float(phase[start + length - 1]), 1.0)
        self.assertTrue(np.all((phase >= 0) & (phase <= 1)))


class MergeTest(unittest.TestCase):
    def test_short_run_is_absorbed_into_the_previous_one(self):
        labels = np.array([0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 1], dtype=np.uint8)
        out = mp.merge_short_runs(labels, 3)
        np.testing.assert_array_equal(out, [0] * 8 + [1] * 3)

    def test_leading_short_run_is_absorbed_into_the_next_one(self):
        labels = np.array([3, 1, 1, 1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(mp.merge_short_runs(labels, 3), [1] * 5)


class FirstEntryTest(unittest.TestCase):
    def test_reader_contract_index_is_one_before_the_first_step(self):
        # Two episodes, lengths 5 and 4.
        labels = np.array([0, 0, 1, 1, 0, 1, 2, 2, 2], dtype=np.uint8)
        ends = np.array([5, 9])
        entries, dropped = mp.first_entries(labels, ends, np.array([3, -1]))
        # Episode 0 enters `search` (1) at local 2 -> stored 1; episode 1 at
        # local 0 -> dropped. Episode 1 enters `probe` (2) at local 1 -> 5 + 1 - 1.
        self.assertEqual(entries["search"], [1])
        self.assertEqual(dropped["search"], 1)
        self.assertEqual(entries["probe"], [5])
        self.assertEqual(entries["success"], [2])
        # Every episode's first label starts at step 0, so it is never stored.
        self.assertEqual(entries["descend"], [])
        self.assertEqual(dropped["descend"], 1)

        # Round trip through the fbsource reader's arithmetic
        # (stages.read_stage_starts: local = global - episode_start + 1).
        starts = np.concatenate([[0], ends[:-1]])
        for stored in entries["probe"]:
            episode = int(np.searchsorted(ends, stored, side="right"))
            self.assertEqual(stored - starts[episode] + 1, 1)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = os.path.join(self.tmp, "s.zarr")
        root = zarr.open(self.store, mode="w")
        first = scripted_episode(NOMINAL)
        second = scripted_episode(NOMINAL[:3])
        n1, n2 = len(first["action"]), len(second["action"])
        data = root.create_group("data")
        for key in first:
            data.array(key, np.concatenate([first[key], second[key]]))
        data.array("motion_primitive", np.zeros((n1 + n2, 1), dtype=np.uint8))
        data.array("primitive_phase", np.zeros((n1 + n2, 1), dtype=np.float32))
        meta = root.create_group("meta")
        meta.array("episode_ends", np.array([n1, n1 + n2]))
        meta.array("first_success_step", np.array([n1 - 3, -1]))
        meta.array("stage_stale", np.array([1]))
        write_sidecar(self.store, {"control_dt": DT})

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_relabel_writes_dense_channels_stage_arrays_and_sidecar(self):
        mp.main(["relabel", self.store, "--param", "min_run=2"])
        root = zarr.open(self.store, mode="r")
        labels = root["data/motion_primitive"][:].reshape(-1)
        self.assertEqual(names(labels[: len(labels)])[:2], ["descend", "search"])
        meta = root["meta"]
        self.assertNotIn("stage_stale", meta)
        self.assertEqual(len(meta["stage_lift"][:]), 1)
        self.assertEqual(len(meta["stage_search"][:]), 2)
        self.assertEqual(meta["stage_success"].dtype, np.dtype("<i8"))
        with open(os.path.join(self.store, "mi_collect_info.json")) as handle:
            block = json.load(handle)["motion_primitives"]
        self.assertEqual(block["vocabulary"], list(mp.PRIMITIVES))
        self.assertEqual(block["params"]["min_run"], 2)
        self.assertEqual(block["meta_arrays"]["stage_search"], 2)


if __name__ == "__main__":
    unittest.main()
