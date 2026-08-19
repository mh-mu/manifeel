"""Recorder + accumulator unit tests against a fake env.  No GPU, no sim."""

import unittest

import numpy as np
import torch
from gym import spaces

from manifeel.gym_util.multistep_wrapper import MultiStepWrapper

from mi_collection.contact import PairIndex
from mi_collection.episode_accumulator import BatchAccumulator, EpisodeAccumulator
from mi_collection.recording_wrapper import (
    InnerStepRecorder,
    to_spatial_f32,
    to_uint8,
    validate_render_cache,
)


class FakeSpec:
    """Minimal TaskSpec surface the recorder uses."""

    name = "fake"

    def success(self, task):
        return np.asarray(task.success_flags, dtype=np.uint8)

    def state_full(self, task):
        return np.full((task.num_envs, 6), float(task.step_count), dtype=np.float32)


class FakeTask:
    def __init__(self, num_envs, num_bodies=4):
        self.num_envs = num_envs
        self.step_count = 0
        self.rew_buf = torch.zeros(num_envs)
        self.contact_force_pairwise = torch.zeros(num_envs, num_bodies, num_bodies, 3)
        self.success_flags = np.zeros(num_envs, dtype=np.uint8)


class FakeInnerEnv:
    """Stands in for MultipleIsaacEnvWrapper.

    ``render_cache`` is set on reset and after each step, so at the top of a
    recorder ``step()`` it always holds the *pre*-action observation -- exactly
    the ordering the real wrapper has.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, num_envs=3, success_schedule=None):
        self.num_envs = num_envs
        self.envs = FakeTask(num_envs)
        self.render_cache = {}
        self.success_schedule = success_schedule or {}
        self.observation_space = spaces.Dict(
            {"state": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)}
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.actions_seen = []

    def _write_cache(self):
        step = self.envs.step_count
        self.render_cache = {
            "state": np.full((self.num_envs, 2), float(step), dtype=np.float32),
            "wrist": np.full((self.num_envs, 3, 2, 2), 0.5, dtype=np.float32),
            "tactile_force_field_right": np.full(
                (self.num_envs, 3, 10, 14), float(step), dtype=np.float32
            ),
        }

    def reset(self):
        self.envs.step_count = 0
        self.envs.success_flags = np.zeros(self.num_envs, dtype=np.uint8)
        self.actions_seen = []
        self._write_cache()
        return {"state": self.render_cache["state"]}

    def step(self, action):
        self.actions_seen.append(np.array(action, copy=True))
        step = self.envs.step_count
        # reward is the *post*-step value: encode the step it belongs to.
        self.envs.rew_buf = torch.full((self.num_envs,), float(step) + 100.0)
        flags = np.zeros(self.num_envs, dtype=np.uint8)
        for env_index, first in self.success_schedule.items():
            if step >= first:
                flags[env_index] = 1
        self.envs.success_flags = flags
        self.envs.step_count += 1
        self._write_cache()
        done = np.array(flags, dtype=np.int64)
        return (
            {"state": self.render_cache["state"]},
            np.copy(done),
            done,
            {},
        )


def make_recorder(inner, sink=None, channels=None):
    return InnerStepRecorder(
        env=inner,
        inner_env=inner,
        task_spec=FakeSpec(),
        pair_indices=[PairIndex("contact_a_b", "hand-object", [1], [2], ("a",), ("b",))],
        control_dt=0.066668,
        contact_threshold=1e-3,
        obs_channels=channels
        or {
            "state": "lowdim_f32",
            "wrist": "image_u8",
            "tactile_force_field_right": "spatial_f32",
        },
        sink=sink,
    )


class ConversionTest(unittest.TestCase):
    def test_uint8_from_unit_float(self):
        out = to_uint8(np.array([[0.0, 0.5, 1.0]], dtype=np.float32))
        self.assertEqual(out.dtype, np.uint8)
        np.testing.assert_array_equal(out, [[0, 128, 255]])

    def test_uint8_passthrough(self):
        source = np.array([[0, 17, 255]], dtype=np.uint8)
        out = to_uint8(source)
        self.assertIs(out, source)

    def test_uint8_clips_out_of_range(self):
        out = to_uint8(np.array([[-0.5, 1.5]], dtype=np.float32))
        np.testing.assert_array_equal(out, [[0, 255]])

    def test_spatial_layout_round_trip(self):
        chw = np.arange(2 * 3 * 10 * 14, dtype=np.float32).reshape(2, 3, 10, 14)
        hwc = to_spatial_f32(chw)
        self.assertEqual(hwc.shape, (2, 10, 14, 3))
        np.testing.assert_array_equal(np.moveaxis(hwc, -1, 1), chw)


class RecorderTest(unittest.TestCase):
    def test_one_row_per_inner_step(self):
        """A single policy call of n_action_steps actions records that many rows."""
        inner = FakeInnerEnv(num_envs=2)
        rows = []
        recorder = make_recorder(inner, sink=lambda t, row: rows.append((t, row)))
        env = MultiStepWrapper(
            recorder, n_obs_steps=2, n_action_steps=8, max_episode_steps=None
        )
        env.reset()
        env.step(np.zeros((2, 8, 6), dtype=np.float32))
        self.assertEqual(len(rows), 8)
        self.assertEqual([t for t, _ in rows], list(range(8)))
        self.assertEqual(len(inner.actions_seen), 8)

    def test_pre_step_obs_with_post_step_reward(self):
        inner = FakeInnerEnv(num_envs=1)
        rows = []
        recorder = make_recorder(inner, sink=lambda t, row: rows.append(row))
        recorder.reset()
        for _ in range(3):
            recorder.step(np.zeros((1, 6), dtype=np.float32))
        for t, row in enumerate(rows):
            # obs and state_full are the pre-action state (step counter == t)
            self.assertEqual(float(row["state"][0, 0]), float(t))
            self.assertEqual(float(row["state_full"][0, 0]), float(t))
            # reward is what the step returned
            self.assertEqual(float(row["reward_total"][0, 0]), float(t) + 100.0)
            self.assertAlmostEqual(float(row["timestamp"][0, 0]), t * 0.066668, 5)

    def test_action_recorded_per_inner_step(self):
        inner = FakeInnerEnv(num_envs=2)
        rows = []
        recorder = make_recorder(inner, sink=lambda t, row: rows.append(row))
        env = MultiStepWrapper(
            recorder, n_obs_steps=1, n_action_steps=4, max_episode_steps=None
        )
        env.reset()
        actions = np.arange(2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6)
        env.step(actions)
        for t, row in enumerate(rows):
            np.testing.assert_allclose(row["action"], actions[:, t])

    def test_done_is_masked_upward(self):
        """MultiStepWrapper must not see success as an episode boundary.

        Without the mask its inner loop breaks the moment every env is
        successful, and every later call breaks immediately too -- the driver
        then spins without stepping the sim.
        """
        inner = FakeInnerEnv(num_envs=2, success_schedule={0: 0, 1: 0})
        rows = []
        recorder = make_recorder(inner, sink=lambda t, row: rows.append(row))
        env = MultiStepWrapper(
            recorder, n_obs_steps=1, n_action_steps=4, max_episode_steps=None
        )
        env.reset()
        env.step(np.zeros((2, 4, 6), dtype=np.float32))
        env.step(np.zeros((2, 4, 6), dtype=np.float32))
        self.assertEqual(len(rows), 8)
        # the true flag still reaches the recorder
        self.assertEqual(int(rows[0]["success"][0, 0]), 1)

    def test_missing_channel_is_detected(self):
        inner = FakeInnerEnv(num_envs=1)
        inner.reset()
        self.assertEqual(validate_render_cache(inner, ["state", "wrist"]), [])
        self.assertEqual(validate_render_cache(inner, ["side"]), ["side"])

    def test_missing_channel_raises_on_step(self):
        inner = FakeInnerEnv(num_envs=1)
        recorder = make_recorder(inner, channels={"side": "image_u8"})
        recorder.reset()
        with self.assertRaisesRegex(KeyError, "render_cache"):
            recorder.step(np.zeros((1, 6), dtype=np.float32))


class TruncationTest(unittest.TestCase):
    def _row(self, num_envs, success):
        return {
            "value": np.arange(num_envs, dtype=np.float32).reshape(num_envs, 1),
            "success": np.asarray(success, dtype=np.uint8).reshape(num_envs, 1),
        }

    def test_success_at_step_zero(self):
        accumulator = EpisodeAccumulator(0, max_steps=50, tail_steps=20)
        for step in range(30):
            if accumulator.push(step, {}, success=(step == 0)):
                break
        self.assertTrue(accumulator.success)
        self.assertEqual(accumulator.first_success_step, 0)
        self.assertEqual(accumulator.length, 21)

    def test_early_success_gets_its_full_tail(self):
        accumulator = EpisodeAccumulator(0, max_steps=50, tail_steps=20)
        for step in range(200):
            if accumulator.push(step, {}, success=(step >= 10)):
                break
        self.assertEqual(accumulator.first_success_step, 10)
        self.assertEqual(accumulator.length, 31)  # 10 + 1 + 20, under the cap

    def test_success_near_the_horizon_has_its_tail_clipped(self):
        """max_steps is a hard cap: success + tail OR the horizon, whichever first."""
        accumulator = EpisodeAccumulator(0, max_steps=50, tail_steps=20)
        for step in range(200):
            if accumulator.push(step, {}, success=(step >= 49)):
                break
        self.assertEqual(accumulator.first_success_step, 49)
        self.assertEqual(accumulator.length, 50)  # min(49 + 1 + 20, 50)
        self.assertTrue(accumulator.success)

    def test_tail_clips_exactly_at_the_boundary(self):
        boundary = EpisodeAccumulator(0, max_steps=50, tail_steps=20)
        for step in range(200):
            if boundary.push(step, {}, success=(step >= 29)):
                break
        self.assertEqual(boundary.length, 50)  # 29 + 1 + 20 == max_steps exactly
        just_under = EpisodeAccumulator(0, max_steps=50, tail_steps=20)
        for step in range(200):
            if just_under.push(step, {}, success=(step >= 28)):
                break
        self.assertEqual(just_under.length, 49)

    def test_failure_is_exactly_max_steps(self):
        accumulator = EpisodeAccumulator(0, max_steps=50, tail_steps=20)
        for step in range(200):
            if accumulator.push(step, {}, success=False):
                break
        self.assertFalse(accumulator.success)
        self.assertEqual(accumulator.first_success_step, -1)
        self.assertEqual(accumulator.length, 50)

    def test_late_success_does_not_upgrade_a_failure(self):
        """A success only in the tail (>= max_steps) must not relabel the episode."""
        accumulator = EpisodeAccumulator(0, max_steps=10, tail_steps=5)
        for step in range(50):
            if accumulator.push(step, {}, success=(step >= 10)):
                break
        self.assertFalse(accumulator.success)
        self.assertEqual(accumulator.length, 10)

    def test_envs_truncate_independently(self):
        completed = []
        batch = BatchAccumulator(
            num_envs=3,
            max_steps=20,
            tail_steps=2,
            on_complete=lambda i, ep, meta: completed.append((i, ep, meta)),
        )
        schedule = {0: 0, 1: 5}  # env 2 never succeeds
        for step in range(batch.physical_ceiling):
            if batch.done:
                break
            flags = [int(step >= first) for _, first in sorted(schedule.items())] + [0]
            batch.push(step, self._row(3, flags))
        batch.finish()
        by_env = {index: meta for index, _ep, meta in completed}
        self.assertEqual(by_env[0]["length"], 3)  # 0 + 1 + 2
        self.assertEqual(by_env[1]["length"], 8)  # 5 + 1 + 2
        self.assertEqual(by_env[2]["length"], 20)  # max_steps
        self.assertEqual(batch.physical_ceiling, 20)  # the cap, tail included
        self.assertEqual(by_env[2]["success"], 0)
        self.assertEqual(by_env[0]["first_success_step"], 0)

    def test_completion_order_is_recorded_not_assumed(self):
        completed = []
        batch = BatchAccumulator(
            num_envs=2,
            max_steps=20,
            tail_steps=0,
            on_complete=lambda i, ep, meta: completed.append(meta["env_index"]),
        )
        for step in range(batch.physical_ceiling):
            if batch.done:
                break
            batch.push(step, self._row(2, [int(step >= 4), int(step >= 1)]))
        # env 1 succeeds first, so the episodes arrive out of env order --
        # the writer sorts by env_index, it never trusts arrival order.
        self.assertEqual(completed, [1, 0])

    def test_stacked_episode_has_leading_time_axis(self):
        batch = BatchAccumulator(2, max_steps=3, tail_steps=0, on_complete=_collect())
        episodes = batch.on_complete.episodes
        for step in range(3):
            batch.push(step, self._row(2, [0, 0]))
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]["value"].shape, (3, 1))
        self.assertEqual(episodes[0]["success"].shape, (3, 1))


class _collect:
    def __init__(self):
        self.episodes = []

    def __call__(self, index, episode, meta):
        self.episodes.append(episode)
