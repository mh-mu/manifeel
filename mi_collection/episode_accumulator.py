"""Per-env buffering and first-success truncation.

One :class:`EpisodeAccumulator` per env slot.  Episodes end independently, at
the success tail or the horizon, **whichever comes first**:

* success at step ``s`` (with ``s < max_steps``) -> length ``min(s + 1 + tail, max_steps)``
* no success in the first ``max_steps`` steps -> length exactly ``max_steps``

So ``max_steps`` is a hard cap on every episode, and a success later than
``max_steps - 1 - tail`` keeps a shortened tail rather than overrunning it.  On
the pilot distribution that costs the tail of a handful of episodes: the latest
first-success seen in 300 pilot episodes was step 489 of 500.

``first_success_step`` is frozen at the first success inside the horizon; a
success that only appears in the tail never upgrades a failure, so the
``episode_success`` label and the recorded length always agree.

Completion is tracked here and not read off ``done``: in this task ``reset_buf``
is a success flag, not an episode boundary.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np


class EpisodeAccumulator:
    """Rows of one env slot, plus its truncation state."""

    def __init__(self, env_index: int, max_steps: int, tail_steps: int) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if tail_steps < 0:
            raise ValueError("success_tail_steps must be >= 0.")
        self.env_index = env_index
        self.max_steps = int(max_steps)
        self.tail_steps = int(tail_steps)
        self.rows: List[Dict[str, np.ndarray]] = []
        self.first_success_step: int = -1
        self.target_length: Optional[int] = None
        self.complete: bool = False

    @property
    def length(self) -> int:
        return len(self.rows)

    @property
    def success(self) -> bool:
        return self.first_success_step >= 0

    def push(self, step: int, row: Dict[str, np.ndarray], success: bool) -> bool:
        """Append one step's row.  Returns True when the episode just completed."""
        if self.complete:
            return False
        self.rows.append(row)
        if self.target_length is None and success and step < self.max_steps:
            self.first_success_step = int(step)
            # min(): the horizon is a hard cap, so a late success gets a
            # shortened tail rather than overrunning max_steps.
            self.target_length = min(int(step) + 1 + self.tail_steps, self.max_steps)
        limit = self.target_length if self.target_length is not None else self.max_steps
        if self.length >= limit:
            self.complete = True
            return True
        return False

    def stack(self) -> Dict[str, np.ndarray]:
        """The episode as ``channel -> (T, *shape)`` arrays."""
        if not self.rows:
            raise RuntimeError(f"Env {self.env_index} recorded no steps.")
        keys = list(self.rows[0])
        return {
            key: np.stack([row[key] for row in self.rows], axis=0) for key in keys
        }


class BatchAccumulator:
    """Fans one recorded step out to per-env accumulators and flushes finished ones.

    ``on_complete(env_index, episode, meta)`` is called the moment an episode
    finishes, so its rows can be written and freed instead of being held for the
    whole batch (a 50-env batch of 256x256 frames is tens of GB otherwise).
    Episodes therefore arrive in completion order; the writer records
    ``episode_env_index`` and consolidation restores the deterministic
    ``(batch, env_index)`` order.
    """

    def __init__(
        self,
        num_envs: int,
        max_steps: int,
        tail_steps: int,
        on_complete: Callable[[int, Dict[str, np.ndarray], Dict[str, int]], None],
        annotate: Optional[
            Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]
        ] = None,
    ) -> None:
        self.num_envs = int(num_envs)
        #: Adds whole-episode channels (e.g. motion-primitive labels) at flush.
        self.annotate = annotate
        self.max_steps = int(max_steps)
        self.tail_steps = int(tail_steps)
        self.on_complete = on_complete
        self.accumulators = [
            EpisodeAccumulator(i, max_steps, tail_steps) for i in range(self.num_envs)
        ]
        self.completed = 0

    @property
    def physical_ceiling(self) -> int:
        """Steps the batch has to be driven for.  Every episode ends by here."""
        return self.max_steps

    @property
    def done(self) -> bool:
        return self.completed >= self.num_envs

    def push(self, step: int, row: Dict[str, np.ndarray]) -> None:
        success_column = np.asarray(row["success"]).reshape(-1)
        for index, accumulator in enumerate(self.accumulators):
            if accumulator.complete:
                continue
            per_env = {key: np.asarray(value)[index] for key, value in row.items()}
            if accumulator.push(step, per_env, bool(success_column[index])):
                self._flush(accumulator)

    def finish(self) -> None:
        """Flush any env still open at the ceiling.

        Should be a no-op: every episode ends at ``max_steps`` at the latest,
        which is the ceiling.  Kept so a driver bug truncates loudly rather than
        dropping episodes.
        """
        for accumulator in self.accumulators:
            if not accumulator.complete and accumulator.length > 0:
                accumulator.complete = True
                self._flush(accumulator)

    def _flush(self, accumulator: EpisodeAccumulator) -> None:
        meta = {
            "env_index": accumulator.env_index,
            "success": int(accumulator.success),
            "first_success_step": int(accumulator.first_success_step),
            "length": accumulator.length,
        }
        episode = accumulator.stack()
        if self.annotate is not None:
            episode.update(self.annotate(episode))
        accumulator.rows = []
        self.completed += 1
        self.on_complete(accumulator.env_index, episode, meta)
