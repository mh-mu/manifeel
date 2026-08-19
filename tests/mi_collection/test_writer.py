"""Shard writing, consolidation and sidecar tests.  No GPU, no sim."""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer

from mi_collection.writer import (
    channel_roles,
    consolidate,
    discard_shard,
    shard_is_reusable,
    shard_path,
    ShardWriter,
    write_sidecar,
)


def episode(length, value, width=4):
    return {
        "state": np.full((length, 7), value, dtype=np.float32),
        "wrist": np.full((length, 3, 2, 2), value % 255, dtype=np.uint8),
        "contact_left_plug": np.full((length, width), value, dtype=np.float32),
        "timestamp": (np.arange(length, dtype=np.float32) * 0.066668).reshape(-1, 1),
    }


PARAMS = {"num_envs": 3, "max_steps": 10, "success_tail_steps": 2}


class ShardWriterTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="mi_writer_test_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_shard(self, batch, seed, rows):
        path = shard_path(self.root, batch)
        writer = ShardWriter(path, batch, seed)
        for env_index, length, success, first in rows:
            writer.add(
                episode(length, env_index + 10 * batch),
                {
                    "env_index": env_index,
                    "success": success,
                    "first_success_step": first,
                    "length": length,
                },
            )
        writer.finalize(PARAMS)
        return path

    def test_round_trip_variable_length_episodes(self):
        path = self._write_shard(0, 100, [(0, 5, 1, 2), (1, 9, 0, -1), (2, 3, 1, 0)])
        buffer = ReplayBuffer.create_from_path(path, mode="r")
        self.assertEqual(buffer.n_episodes, 3)
        np.testing.assert_array_equal(buffer.episode_ends[:], [5, 14, 17])
        self.assertEqual(buffer["wrist"].dtype, np.uint8)
        self.assertEqual(buffer["state"].shape, (17, 7))

    def test_meta_is_stored_in_env_order_not_completion_order(self):
        # Completion order 2, 0, 1 -- as a real batch would produce.
        path = self._write_shard(0, 100, [(2, 3, 1, 0), (0, 5, 1, 2), (1, 9, 0, -1)])
        buffer = ReplayBuffer.create_from_path(path, mode="r")
        np.testing.assert_array_equal(buffer.meta["episode_env_index"][:], [0, 1, 2])
        np.testing.assert_array_equal(buffer.meta["episode_success"][:], [1, 0, 1])
        np.testing.assert_array_equal(
            buffer.meta["first_success_step"][:], [2, -1, 0]
        )
        np.testing.assert_array_equal(buffer.meta["episode_batch_seed"][:], [100] * 3)

    def test_consolidation_is_deterministic_in_batch_env_order(self):
        a = self._write_shard(0, 100, [(2, 3, 1, 0), (0, 5, 1, 2), (1, 9, 0, -1)])
        b = self._write_shard(1, 103, [(1, 4, 0, -1), (0, 6, 1, 1), (2, 2, 1, 0)])
        out = os.path.join(self.root, "merged.zarr")
        summary = consolidate([a, b], out)
        self.assertEqual(summary["num_episodes"], 6)
        merged = ReplayBuffer.create_from_path(out, mode="r")
        # value == env_index + 10 * batch, so this checks the exact ordering.
        first_rows = [
            float(merged.get_episode(i)["state"][0, 0]) for i in range(6)
        ]
        self.assertEqual(first_rows, [0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        np.testing.assert_array_equal(
            merged.meta["episode_env_index"][:], [0, 1, 2, 0, 1, 2]
        )
        np.testing.assert_array_equal(
            merged.meta["episode_batch_seed"][:], [100, 100, 100, 103, 103, 103]
        )
        np.testing.assert_array_equal(
            merged.meta["episode_success"][:], [1, 0, 1, 1, 0, 1]
        )
        # every data channel is as long as the last episode end
        ends = np.asarray(merged.episode_ends[:])
        for key in merged.keys():
            self.assertEqual(merged[key].shape[0], int(ends[-1]), key)

    def test_consolidation_limit_trims_the_tail(self):
        a = self._write_shard(0, 100, [(0, 5, 1, 2), (1, 9, 0, -1), (2, 3, 1, 0)])
        b = self._write_shard(1, 103, [(0, 6, 1, 1), (1, 4, 0, -1), (2, 2, 1, 0)])
        out = os.path.join(self.root, "merged_limited.zarr")
        summary = consolidate([a, b], out, limit=4)
        self.assertEqual(summary["num_episodes"], 4)
        merged = ReplayBuffer.create_from_path(out, mode="r")
        self.assertEqual(len(merged.meta["episode_success"][:]), 4)

    def test_resume_only_reuses_matching_complete_shards(self):
        path = self._write_shard(0, 100, [(0, 5, 1, 2)])
        self.assertTrue(shard_is_reusable(path, PARAMS))
        other = dict(PARAMS, max_steps=999)
        self.assertFalse(shard_is_reusable(path, other))
        discard_shard(path)
        self.assertFalse(shard_is_reusable(path, PARAMS))

    def test_incomplete_shard_is_not_reusable(self):
        path = shard_path(self.root, 7)
        writer = ShardWriter(path, 7, 1)
        writer.add(episode(4, 1.0), {"env_index": 0, "success": 0,
                                     "first_success_step": -1, "length": 4})
        # no finalize() -- the batch died
        self.assertFalse(shard_is_reusable(path, PARAMS))

    def test_sidecar_round_trips(self):
        out = os.path.join(self.root, "store.zarr")
        os.makedirs(out)
        path = write_sidecar(out, {"control_dt": 0.066668, "contact_pairs": {}})
        with open(path, encoding="utf-8") as handle:
            self.assertAlmostEqual(json.load(handle)["control_dt"], 0.066668)


class RolesTest(unittest.TestCase):
    def test_roles(self):
        roles = channel_roles(
            [
                "wrist",
                "front",
                "right_tactile_camera_taxim",
                "tactile_force_field_right",
                "state",
                "state_full",
                "contact_left_plug",
                "reward_total",
                "action",
                "timestamp",
                "success",
            ]
        )
        self.assertEqual(roles["wrist"], "sensor")
        self.assertEqual(roles["state_full"], "world_state")
        self.assertEqual(roles["contact_left_plug"], "contact")
        self.assertEqual(roles["reward_total"], "reward")
        self.assertEqual(roles["action"], "action")
        self.assertEqual(roles["timestamp"], "time")
        self.assertEqual(roles["success"], "outcome")
