"""The single task-specific seam of the MI collector.

Everything that knows about *this* ManiFeel task -- which rigid bodies form a
contact pair, what counts as success, what goes into ``state_full`` -- lives
behind :class:`TaskSpec`.  The collector, the recorder, the accumulator and the
writer are task-agnostic, so supporting another TacSL task means writing one new
spec and nothing else.

Mirrors ``manipulation_mochi/policy_learning/envs_spec/base.py`` (the ShapeFilter
``EnvSpec`` ABC), which is the prior art for this pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Tuple

import numpy as np


class ContactPairSpec:
    """One ground-truth contact channel: force on group A from group B.

    ``name`` is the zarr channel name and must be ``contact_<a>_<b>`` -- the
    three-token form the fbsource analyzer's ``--contact-fingers`` parses.
    ``label`` is one of ``hand-object`` / ``hand-env`` / ``object-env``; the
    analyzer groups channels by it and requires all three to be present.
    """

    __slots__ = ("name", "label", "bodies_a", "bodies_b", "optional")

    def __init__(
        self,
        name: str,
        label: str,
        bodies_a: Sequence[str],
        bodies_b: Sequence[str],
        optional: bool = False,
    ) -> None:
        if len(name.split("_")) != 3 or not name.startswith("contact_"):
            raise ValueError(
                f"Contact channel {name!r} must be named contact_<a>_<b> so the "
                "analyzer can parse it."
            )
        if label not in CONTACT_LABELS:
            raise ValueError(f"Unknown contact label {label!r}; have {CONTACT_LABELS}.")
        self.name = name
        self.label = label
        self.bodies_a = tuple(bodies_a)
        self.bodies_b = tuple(bodies_b)
        self.optional = optional

    def __repr__(self) -> str:
        return (
            f"ContactPairSpec({self.name!r}, {self.label!r}, "
            f"{list(self.bodies_a)} <- {list(self.bodies_b)})"
        )


#: The three pair types the fbsource analyzer groups contact channels into.
CONTACT_LABELS: Tuple[str, ...] = ("hand-object", "hand-env", "object-env")


class TaskSpec(ABC):
    """Task-specific knowledge the collector needs.  One instance per task."""

    #: Short identifier, recorded in the sidecar and selected by ``--task-spec``.
    name: str = "base"
    #: The sidecar's ``env`` field, which the fbsource readers report as the source env.
    env_name: str = "manifeel_tacsl"

    @abstractmethod
    def contact_pairs(self, include_optional: bool = True) -> List[ContactPairSpec]:
        """Ordered contact-pair definitions.  Order fixes the channel order."""

    @abstractmethod
    def body_index_map(self, task: object) -> Dict[str, int]:
        """``body name -> DOMAIN_ENV rigid-body index`` for the live task."""

    @abstractmethod
    def success(self, task: object) -> np.ndarray:
        """Per-env success at the current sim state, as a ``(num_envs,)`` uint8."""

    @abstractmethod
    def state_full_layout(self) -> List[Tuple[str, int]]:
        """``(name, width)`` of each block of ``state_full``, in order."""

    @abstractmethod
    def state_full(self, task: object) -> np.ndarray:
        """Privileged world state, ``(num_envs, sum(widths))`` float32."""

    @abstractmethod
    def analyzer_recommendations(self) -> Dict[str, object]:
        """Channel roles the fbsource analyzers should be pointed at."""

    def resolve_bodies(self, task: object, names: Sequence[str]) -> List[int]:
        """Rigid-body indices for ``names``, failing loudly on an unknown name.

        A silently-dropped body name would record an all-zero contact channel
        that is indistinguishable from "these bodies never touch".
        """
        table = self.body_index_map(task)
        missing = [n for n in names if n not in table]
        if missing:
            raise KeyError(
                f"Body name(s) {missing} are not in this task; "
                f"have {sorted(table)}."
            )
        return [table[n] for n in names]
