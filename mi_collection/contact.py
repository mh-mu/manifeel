"""The 4-d ground-truth contact descriptor, ``[in_contact, force_world(3)]``.

Reduced from ShapeFilter's 8-d ``[in_contact, num_points, force(3), normal(3)]``:
the point count and the geometric normal need ``get_env_rigid_contacts``, which
Isaac Gym refuses under the GPU pipeline -- and the GPU pipeline is what the
tactile force field and the pairwise force tensor need.  Keeping four columns
also keeps the critic input narrow; constant columns are zeroed by the
analyzer's ``standardize_columns`` anyway, so padding to eight would only widen
it.

Convention, recorded in the sidecar: ``force`` is the force **on group A from
group B**.  ``contact_force_pairwise`` is exactly antisymmetric, so the
direction is a labelling choice, but it has to be written down.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

#: Column layout of the descriptor.
IN_CONTACT: int = 0
FORCE_SLICE: slice = slice(1, 4)
CONTACT_STATE_DIM: int = 4

#: Non-contacting pairs read below 1e-4 N on this scene (probed 2026-08-18), so
#: this behaves as "any nonzero" with an order of magnitude of headroom.
DEFAULT_CONTACT_THRESHOLD: float = 1e-3


class PairIndex:
    """A contact pair's rigid-body index sets, resolved against a live task."""

    __slots__ = ("name", "label", "ids_a", "ids_b", "bodies_a", "bodies_b")

    def __init__(
        self,
        name: str,
        label: str,
        ids_a: Sequence[int],
        ids_b: Sequence[int],
        bodies_a: Sequence[str],
        bodies_b: Sequence[str],
    ) -> None:
        self.name = name
        self.label = label
        self.ids_a = list(ids_a)
        self.ids_b = list(ids_b)
        self.bodies_a = tuple(bodies_a)
        self.bodies_b = tuple(bodies_b)


def build_pair_indices(task: object, spec: object, include_optional: bool = True):
    """Resolve every pair of ``spec`` to rigid-body indices on the live task."""
    resolved: List[PairIndex] = []
    for pair in spec.contact_pairs(include_optional=include_optional):
        resolved.append(
            PairIndex(
                name=pair.name,
                label=pair.label,
                ids_a=spec.resolve_bodies(task, pair.bodies_a),
                ids_b=spec.resolve_bodies(task, pair.bodies_b),
                bodies_a=pair.bodies_a,
                bodies_b=pair.bodies_b,
            )
        )
    return resolved


def pair_descriptor(
    pairwise: torch.Tensor,
    ids_a: Sequence[int],
    ids_b: Sequence[int],
    threshold: float = DEFAULT_CONTACT_THRESHOLD,
) -> torch.Tensor:
    """``(num_envs, 4)`` descriptor for one pair, computed on the tensor's device.

    ``in_contact`` thresholds the **constituent** pairwise magnitudes, not the
    norm of the summed vector: with a multi-body group two real contacts can
    cancel in the sum while both are live.
    """
    block = pairwise[:, ids_a][:, :, ids_b]  # (E, na, nb, 3)
    force = block.sum(dim=(1, 2))  # (E, 3)
    magnitudes = block.norm(dim=-1).flatten(1)  # (E, na*nb)
    in_contact = (magnitudes.amax(dim=1) > threshold).to(force.dtype)
    return torch.cat([in_contact.unsqueeze(-1), force], dim=-1)


def contact_descriptors(
    pairwise: torch.Tensor,
    pairs: Sequence[PairIndex],
    threshold: float = DEFAULT_CONTACT_THRESHOLD,
) -> torch.Tensor:
    """``(num_envs, n_pairs, 4)``, reduced on device before any host transfer."""
    if pairwise is None:
        raise RuntimeError(
            "contact_force_pairwise is None -- the sim is on the CPU pipeline, "
            "where Isaac Gym does not expose the pairwise contact tensor. Set "
            "pipeline=gpu; the collector cannot record contacts otherwise."
        )
    return torch.stack(
        [pair_descriptor(pairwise, p.ids_a, p.ids_b, threshold) for p in pairs],
        dim=1,
    )


def sidecar_contact_pairs(pairs: Sequence[PairIndex]) -> Tuple[dict, dict]:
    """``(channel -> label, channel -> body-group provenance)`` for the sidecar."""
    labels = {p.name: p.label for p in pairs}
    provenance = {
        p.name: {
            "bodies_a": list(p.bodies_a),
            "bodies_b": list(p.bodies_b),
            "body_ids_a": list(p.ids_a),
            "body_ids_b": list(p.ids_b),
        }
        for p in pairs
    }
    return labels, provenance
