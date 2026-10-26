"""Four-tier RNG derivation (FND-06 + Codex CR-3).

CRITICAL: uses ``hashlib.sha256`` — NOT Python ``hash()``. Python's built-in
``hash()`` of tuples containing strings is salted per-process when
``PYTHONHASHSEED`` is not fixed. Two fresh Python processes therefore produce
different hash values for the same input. ``hashlib.sha256`` is stable across
every process, Python version, and OS.

Three parallel factories (``py`` / ``np`` / ``torch``) share the same
seed-derivation rule with a per-factory namespace prefix. A namespace string
means that ``py_rng(s, u, r, p)`` and ``np_rng(s, u, r, p)`` are independent
streams even though all four inputs are identical.

Typical call sites
------------------
Server (per round):
    ``clients = server_rng(run_seed).sample(all_ids, k)``

Client (per round, per user):
    ``neg = np_rng(run_seed, user_idx, round_num, "train_neg").choice(items, 4)``
    ``g = torch_gen(run_seed, user_idx, round_num, "dataloader")``
    ``DataLoader(ds, shuffle=True, generator=g)``

Every downstream phase MUST pass a ``torch.Generator`` (via ``torch_gen(...)``)
into every ``DataLoader(..., generator=..., shuffle=True)`` to satisfy CR-3.
"""
from __future__ import annotations

import hashlib
import random
from typing import Optional

import numpy as np
import torch

# Closed set of legal purposes. Prevents typos from silently producing new
# (and otherwise-undiscoverable) RNG streams.
_ALLOWED_PURPOSES = frozenset(
    {
        "train_neg",
        "eval_neg",
        "model_init",
        "server_sample",
        "dataloader",
    }
)

# Max int accepted by ``torch.Generator.manual_seed`` (signed 64-bit positive
# range). The full sha256 digest is 256 bits, so we mod into this range for
# torch only (python random / numpy default_rng accept arbitrary ints).
_TORCH_SEED_MAX = 2 ** 63 - 1


def _derive_seed(
    namespace: str,
    run_seed: int,
    user_idx: int,
    round_num: int,
    purpose: str,
) -> int:
    """Derive a deterministic Python int from a namespaced payload.

    Parameters
    ----------
    namespace : str
        One of ``"py"``, ``"np"``, ``"torch"`` — disambiguates parallel RNG
        streams that share the same (run_seed, user_idx, round_num, purpose)
        tuple.
    run_seed : int
        Root seed for the run. Typically the ``run_seed`` field of the
        ``RunManifest`` and the ModeProfile.
    user_idx : int
        Global user index (from the canonical mapping). Use ``-1`` for
        server-level calls where there is no user.
    round_num : int
        Current FL round number (0-indexed). Use ``-1`` outside of a round
        (for example, ``"model_init"`` before round 1 starts).
    purpose : str
        One of :data:`_ALLOWED_PURPOSES`. Unknown values raise ``ValueError``.

    Returns
    -------
    int
        Full 256-bit int from ``SHA-256(payload)``. Never truncated (Codex
        review comment N-1: don't truncate to 8 bytes — full digest).

    Raises
    ------
    ValueError
        If ``purpose`` is not in :data:`_ALLOWED_PURPOSES`.
    """
    if purpose not in _ALLOWED_PURPOSES:
        raise ValueError(
            f"Unknown RNG purpose {purpose!r}. "
            f"Allowed: {sorted(_ALLOWED_PURPOSES)}"
        )
    payload = f"{namespace}:{run_seed}:{user_idx}:{round_num}:{purpose}".encode(
        "ascii"
    )
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def py_rng(
    run_seed: int,
    user_idx: int,
    round_num: int,
    purpose: str,
) -> random.Random:
    """Return a deterministic ``random.Random`` instance (namespace ``"py"``).

    Parameters
    ----------
    run_seed : int
    user_idx : int
        Use ``-1`` for server-level calls.
    round_num : int
        Use ``-1`` outside of a round.
    purpose : str

    Returns
    -------
    random.Random
        A fresh instance — not the global ``random`` module. Never seeds the
        process-global state.
    """
    return random.Random(_derive_seed("py", run_seed, user_idx, round_num, purpose))


def np_rng(
    run_seed: int,
    user_idx: int,
    round_num: int,
    purpose: str,
) -> np.random.Generator:
    """Return a deterministic ``numpy.random.Generator`` (namespace ``"np"``).

    Parameters
    ----------
    run_seed : int
    user_idx : int
    round_num : int
    purpose : str

    Returns
    -------
    numpy.random.Generator
        Constructed via ``np.random.default_rng(seed)`` — independent of any
        other numpy RNG in the process.
    """
    return np.random.default_rng(
        _derive_seed("np", run_seed, user_idx, round_num, purpose)
    )


def torch_gen(
    run_seed: int,
    user_idx: int,
    round_num: int,
    purpose: str,
) -> torch.Generator:
    """Return a deterministic ``torch.Generator`` (namespace ``"torch"``).

    Suitable for passing as ``generator=`` into ``DataLoader`` and into
    tensor-init functions (``torch.randn(..., generator=g)``). The seed is
    mod'd into the int64 positive range because ``torch.Generator.manual_seed``
    refuses values ``>= 2**63``.

    Parameters
    ----------
    run_seed : int
    user_idx : int
    round_num : int
    purpose : str

    Returns
    -------
    torch.Generator
        A fresh CPU generator. Callers that need a CUDA generator should
        create one with the same seed (use ``g.initial_seed()`` to extract).
    """
    g = torch.Generator()
    seed = _derive_seed("torch", run_seed, user_idx, round_num, purpose)
    g.manual_seed(seed % _TORCH_SEED_MAX)
    return g


def server_rng(run_seed: int) -> random.Random:
    """Top-level server RNG for per-round client selection.

    The server calls this once at startup and passes the returned instance to
    the node-selection sampler each round. Determines, together with
    ``run_seed``, the full sequence of sampled clients across all rounds.

    Parameters
    ----------
    run_seed : int

    Returns
    -------
    random.Random
    """
    return random.Random(run_seed)


# Back-compat alias matching the research file's earlier exposition.
def derive_rng(
    run_seed: int,
    user_id: int,
    round_num: int,
    purpose: str,
) -> random.Random:
    """Alias for :func:`py_rng` kept for call-site clarity.

    ``py_rng`` is the preferred spelling going forward; ``derive_rng``
    preserves backwards compatibility with code written from research-phase
    signatures.
    """
    return py_rng(run_seed, user_id, round_num, purpose)
