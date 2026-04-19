---
phase: 01-foundation-contract
plan: 04
type: execute
wave: 2
depends_on: [01-foundation-contract-01]
files_modified:
  - scripts/foundation/fedrec_foundation/rng.py
  - scripts/foundation/fedrec_foundation/manifest.py
  - scripts/foundation/tests/test_rng.py
  - scripts/foundation/tests/test_manifest.py
autonomous: true
requirements: [FND-06, FND-07]
must_haves:
  truths:
    - "Two fresh Python processes with DIFFERENT PYTHONHASHSEED values produce byte-identical RNG output streams for the same (run_seed, user_id, round, purpose) input — verified by a subprocess test."
    - "Same run_seed reproduces: (a) Python client selection order, (b) NumPy eval-negative samples, (c) Torch model-init weights, (d) DataLoader iteration order — four separate assertions (CR-3)."
    - "Run manifest includes all 18 D-16 fields plus IMP-2 fields: `mapping_sha256`, `exclusion_sha256`, `foundation_contract_sha256`."
    - "Run manifest writes are atomic (tempfile + os.replace) AND write twice (embedded in result JSON under `_manifest` AND sibling `<run_id>-manifest.json`) per D-15."
  artifacts:
    - path: "scripts/foundation/fedrec_foundation/rng.py"
      provides: "Three RNG factories + server_rng + _derive_seed (sha256-based, CR-3)"
      exports: ["py_rng", "np_rng", "torch_gen", "server_rng", "derive_rng", "_derive_seed", "_ALLOWED_PURPOSES"]
    - path: "scripts/foundation/fedrec_foundation/manifest.py"
      provides: "RunManifest dataclass, build_run_manifest, write_manifest_sibling, embed_manifest_in_result, generate_run_id"
      exports: ["RunManifest", "RUN_MANIFEST_SCHEMA_VERSION", "generate_run_id", "build_run_manifest", "write_manifest_sibling", "embed_manifest_in_result"]
  key_links:
    - from: "scripts/foundation/fedrec_foundation/rng.py::_derive_seed"
      to: "hashlib.sha256 (NOT Python hash())"
      via: "CR-3 correctness fix"
      pattern: "hashlib\\.sha256"
    - from: "scripts/foundation/fedrec_foundation/rng.py"
      to: "py_rng, np_rng, torch_gen"
      via: "three namespaced RNG factories (CR-3)"
      pattern: "def torch_gen"
    - from: "scripts/foundation/fedrec_foundation/manifest.py::RunManifest"
      to: "mapping_sha256, exclusion_sha256, foundation_contract_sha256"
      via: "IMP-2 composite hash fields"
      pattern: "foundation_contract_sha256"
---

<objective>
Implement FND-06 (four-tier seeding — THREE RNG factories, `hashlib.sha256`-based, with namespace prefixes per Codex CR-3) and FND-07 (run manifest with composite foundation_contract_sha256, embedded + sibling writes per D-15). These are the reproducibility + traceability backbone every downstream result artifact depends on.

Purpose: CR-3 flagged that only seeding Python `random` is insufficient — clients also use NumPy, PyTorch, and DataLoader generators. The foundation must expose THREE factories (`py_rng`, `np_rng`, `torch_gen`) derived from the same sha256 seed so Phases 2–5 can pass a `torch.Generator` into every DataLoader. IMP-2 requires the run manifest to carry all three fingerprints from the bundle index.

Output: Two fully-implemented modules; test_rng.py + test_manifest.py flip GREEN; cross-process subprocess test confirms CR-3 correctness.
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/01-foundation-contract/01-CONTEXT.md
@.planning/phases/01-foundation-contract/01-RESEARCH.md
@.planning/phases/01-foundation-contract/01-VALIDATION.md
@CLAUDE.md
@.planning/codebase/CONVENTIONS.md

<interfaces>
From scripts/foundation/fedrec_foundation/rng.py (CR-3 expanded from research Pattern 6):
```python
import hashlib, random
from typing import Literal
import numpy as np
import torch

_ALLOWED_PURPOSES = frozenset({"train_neg", "eval_neg", "model_init", "server_sample", "dataloader"})

def _derive_seed(namespace: str, run_seed: int, user_idx: int, round_num: int, purpose: str) -> int: ...
def py_rng(run_seed: int, user_idx: int, round_num: int, purpose: str) -> random.Random: ...
def np_rng(run_seed: int, user_idx: int, round_num: int, purpose: str) -> np.random.Generator: ...
def torch_gen(run_seed: int, user_idx: int, round_num: int, purpose: str) -> torch.Generator: ...
def server_rng(run_seed: int) -> random.Random: ...                 # top-level server selector RNG
def derive_rng(run_seed, user_id, round_num, purpose) -> random.Random: ...   # back-compat alias for py_rng
```

From scripts/foundation/fedrec_foundation/manifest.py (extended per IMP-2):
```python
from dataclasses import dataclass
from typing import Dict

RUN_MANIFEST_SCHEMA_VERSION = 1

@dataclass
class RunManifest:
    schema_version: int
    run_id: str
    mode: str
    num_supernodes: int
    partition_mode: str
    fraction_train: float
    fraction_eval: float
    weight_policy: str
    primary_evaluator: str
    num_train_negatives: int
    num_eval_negatives: int
    run_seed: int
    checkpoint_rule: str
    # Foundation fingerprints (IMP-2).
    mapping_sha256: str
    split_hash: str
    exclusion_sha256: str
    foundation_contract_sha256: str
    raw_data_hash: str
    builder_version: str
    # Overrides + module metadata.
    overrides: Dict[str, object]
    module: str                      # "baseline" | "personalized" | "adaptive" | "pfedrec"
    # Environment.
    flwr_version: str
    torch_version: str
    git_commit: str

def generate_run_id() -> str: ...                                 # "20260419-142301-a1b2c3"
def build_run_manifest(...) -> RunManifest: ...
def write_manifest_sibling(m: RunManifest, result_json_path) -> Path: ...
def embed_manifest_in_result(m: RunManifest, result_dict: dict) -> dict: ...
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Implement FND-06 three RNG factories (CR-3) + flip test_rng green</name>
  <files>
    scripts/foundation/fedrec_foundation/rng.py
    scripts/foundation/tests/test_rng.py
  </files>
  <read_first>
    - .planning/phases/01-foundation-contract/01-CONTEXT.md D-14 (four-tier RNG derivation; no global reseeding)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md §"Pattern 6: Four-Tier RNG Derivation (FND-06) — CRITICAL CORRECTNESS" (lines 739-820)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md CODEX PEER REVIEW §CR-3 (THREE factories + namespace prefixes LOCKED; full sha256 digest, not truncated)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md §"Pitfall 1: hash() is process-salted"
    - .planning/phases/01-foundation-contract/01-RESEARCH.md CODEX §N-1 (use full sha256 digest; don't truncate to 8 bytes)
    - .planning/phases/01-foundation-contract/01-VALIDATION.md (test IDs FND-06-a..e)
    - CLAUDE.md (Python 3.9 typing)
  </read_first>
  <behavior>
    - `_derive_seed(namespace, run_seed, user_idx, round_num, purpose) -> int`: returns `int.from_bytes(hashlib.sha256(payload).digest(), "big")` — FULL 32-byte digest, not truncated (N-1). Payload is ASCII-encoded `f"{namespace}:{run_seed}:{user_idx}:{round_num}:{purpose}"`. Raises `ValueError` on unknown purpose.
    - `py_rng(run_seed, user_idx, round_num, purpose) -> random.Random`: namespace `"py"`.
    - `np_rng(run_seed, user_idx, round_num, purpose) -> numpy.random.Generator`: namespace `"np"`; uses `np.random.default_rng(_derive_seed(...))`.
    - `torch_gen(run_seed, user_idx, round_num, purpose) -> torch.Generator`: namespace `"torch"`; `g = torch.Generator(); g.manual_seed(seed % (2**63 - 1)); return g`.
    - `server_rng(run_seed) -> random.Random`: `random.Random(run_seed)` — top-level selector.
    - `derive_rng(...)` alias for `py_rng(...)` (back-compat with the research file's earlier exposition).
    - `_ALLOWED_PURPOSES = frozenset({"train_neg", "eval_neg", "model_init", "server_sample", "dataloader"})`.
    - Tests flip green (all five): `test_derive_rng_stable_across_processes` (subprocess test, varying PYTHONHASHSEED), `test_tuple_uniqueness`, `test_all_three_rng_factories`, `test_torch_generator_reproducible`, `test_sample_reproducible`.
  </behavior>
  <action>
Create `scripts/foundation/fedrec_foundation/rng.py` starting from research Pattern 6 (lines 745-794) AND expanding with Codex CR-3's THREE-factory pattern (research file's CODEX PEER REVIEW code block at lines 38-58). Final structure:

```python
"""Four-tier RNG derivation (FND-06 + Codex CR-3).

CRITICAL: uses hashlib.sha256 — NOT Python hash(). Python's built-in
hash() of tuples containing strings is salted per-process when
PYTHONHASHSEED is not fixed. Two fresh Python processes therefore
produce different hash values for the same input. hashlib.sha256 is
stable across every process, Python version, and OS.

Three parallel factories (py/np/torch) share the same seed-derivation
rule with a per-factory namespace prefix. A namespace string means
that py_rng(s, u, r, p) and np_rng(s, u, r, p) are independent streams
even though all four inputs are identical.
"""
from __future__ import annotations

import hashlib
import random
from typing import Literal

import numpy as np
import torch

# Closed set of legal purposes. Prevents typos from silently producing
# new RNG streams.
_ALLOWED_PURPOSES = frozenset({
    "train_neg",
    "eval_neg",
    "model_init",
    "server_sample",
    "dataloader",
})

# Max int for torch.Generator.manual_seed (signed 64-bit positive range).
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
        One of "py", "np", "torch" — disambiguates parallel RNG streams.
    run_seed : int
    user_idx : int
        Use ``-1`` for server-level (no user).
    round_num : int
        Use ``-1`` outside of a round (e.g., model_init before round 1).
    purpose : str
        One of _ALLOWED_PURPOSES.

    Returns
    -------
    int
        Full 256-bit int from SHA-256(payload). Never truncated (Codex N-1).
    """
    if purpose not in _ALLOWED_PURPOSES:
        raise ValueError(
            f"Unknown RNG purpose {purpose!r}. Allowed: {sorted(_ALLOWED_PURPOSES)}"
        )
    payload = f"{namespace}:{run_seed}:{user_idx}:{round_num}:{purpose}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def py_rng(run_seed: int, user_idx: int, round_num: int, purpose: str) -> random.Random:
    """Return a deterministic `random.Random` instance (namespace=py)."""
    return random.Random(_derive_seed("py", run_seed, user_idx, round_num, purpose))


def np_rng(run_seed: int, user_idx: int, round_num: int, purpose: str) -> np.random.Generator:
    """Return a deterministic `numpy.random.Generator` instance (namespace=np)."""
    return np.random.default_rng(_derive_seed("np", run_seed, user_idx, round_num, purpose))


def torch_gen(run_seed: int, user_idx: int, round_num: int, purpose: str) -> torch.Generator:
    """Return a deterministic `torch.Generator` (namespace=torch).

    Suitable for passing as ``generator=`` to DataLoader and to tensor-init
    functions. The seed is mod'd into the int64 positive range because
    torch.Generator.manual_seed refuses values >= 2**63.
    """
    g = torch.Generator()
    g.manual_seed(_derive_seed("torch", run_seed, user_idx, round_num, purpose) % _TORCH_SEED_MAX)
    return g


def server_rng(run_seed: int) -> random.Random:
    """Top-level server RNG for per-round client selection.

    Server calls this once at startup and passes the instance to the
    node-selection sampler each round.
    """
    return random.Random(run_seed)


# Back-compat alias matching research file's earlier exposition.
def derive_rng(run_seed: int, user_id: int, round_num: int, purpose: str) -> random.Random:
    """Alias for `py_rng(...)` kept for clarity at call sites."""
    return py_rng(run_seed, user_id, round_num, purpose)
```

Flip `tests/test_rng.py` to GREEN. Remove the skip marker; replace test stubs with real bodies:

```python
"""Tests for fedrec_foundation.rng (FND-06 + CR-3)."""
from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fedrec_foundation.rng import (
    _derive_seed, py_rng, np_rng, torch_gen, server_rng, derive_rng,
    _ALLOWED_PURPOSES,
)


def test_derive_rng_stable_across_processes() -> None:
    """CR-3 anchor: sha256-based seed is stable across PYTHONHASHSEED values."""
    script = textwrap.dedent("""
        from fedrec_foundation.rng import _derive_seed
        print(_derive_seed("py", 42, 123, 7, "train_neg"))
    """)
    outputs = []
    import os
    for hashseed in ("0", "1", "random"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hashseed
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, env=env, check=True,
        )
        outputs.append(r.stdout.strip())
    assert len(set(outputs)) == 1, f"RNG seed unstable across PYTHONHASHSEED: {outputs}"


def test_tuple_uniqueness() -> None:
    """Different (user, round, purpose) tuples produce different seeds."""
    seeds = {
        _derive_seed("py", 42, u, r, p)
        for u in (1, 2, 3)
        for r in (0, 1, 2)
        for p in ("train_neg", "eval_neg", "model_init")
    }
    # 27 tuples -> 27 distinct seeds (collision-free by sha256).
    assert len(seeds) == 27


def test_all_three_rng_factories() -> None:
    """py_rng, np_rng, torch_gen produce INDEPENDENT streams for the same input tuple."""
    py = py_rng(42, 1, 0, "train_neg")
    np_r = np_rng(42, 1, 0, "train_neg")
    torch_g = torch_gen(42, 1, 0, "train_neg")
    # Each is a distinct type.
    import random as _r
    assert isinstance(py, _r.Random)
    assert isinstance(np_r, np.random.Generator)
    assert isinstance(torch_g, torch.Generator)
    # Same inputs, different outputs (namespaces disambiguate).
    py_val = py.random()
    np_val = float(np_r.random())
    torch_val = float(torch.rand(1, generator=torch_g).item())
    assert py_val != np_val
    assert np_val != torch_val


def test_torch_generator_reproducible() -> None:
    """Two torch.Generators from the same input produce the same tensor."""
    g1 = torch_gen(42, 1, 0, "model_init")
    g2 = torch_gen(42, 1, 0, "model_init")
    t1 = torch.randn(5, 5, generator=g1)
    t2 = torch.randn(5, 5, generator=g2)
    assert torch.equal(t1, t2)


def test_sample_reproducible() -> None:
    """py_rng + random.sample gives identical client selections across calls."""
    s1 = py_rng(42, -1, 0, "server_sample")
    s2 = py_rng(42, -1, 0, "server_sample")
    assert s1.sample(range(6040), 200) == s2.sample(range(6040), 200)


def test_dataloader_iteration_order() -> None:
    """CR-3 fourth assertion: DataLoader(generator=torch_gen(...)) order is deterministic."""
    ds = TensorDataset(torch.arange(100))
    g1 = torch_gen(42, 1, 0, "dataloader")
    g2 = torch_gen(42, 1, 0, "dataloader")
    loader1 = DataLoader(ds, batch_size=10, shuffle=True, generator=g1)
    loader2 = DataLoader(ds, batch_size=10, shuffle=True, generator=g2)
    batches1 = [b[0].tolist() for b in loader1]
    batches2 = [b[0].tolist() for b in loader2]
    assert batches1 == batches2


def test_unknown_purpose_raises() -> None:
    with pytest.raises(ValueError, match="Unknown RNG purpose"):
        _derive_seed("py", 42, 1, 0, "not_a_purpose")


def test_allowed_purposes_includes_dataloader() -> None:
    """dataloader purpose is REQUIRED for CR-3 DataLoader seeding."""
    assert "dataloader" in _ALLOWED_PURPOSES
    assert "server_sample" in _ALLOWED_PURPOSES


def test_server_rng_reproducible() -> None:
    s1 = server_rng(42)
    s2 = server_rng(42)
    assert s1.random() == s2.random()


def test_derive_rng_is_py_rng_alias() -> None:
    """derive_rng must produce identical stream to py_rng for same inputs."""
    d = derive_rng(42, 1, 0, "train_neg")
    p = py_rng(42, 1, 0, "train_neg")
    assert d.random() == p.random()
```
  </action>
  <verify>
    <automated>cd scripts/foundation &amp;&amp; pytest tests/test_rng.py -v</automated>
  </verify>
  <acceptance_criteria>
    - File `scripts/foundation/fedrec_foundation/rng.py` defines `py_rng`, `np_rng`, `torch_gen`, `server_rng`, `derive_rng`, `_derive_seed`, `_ALLOWED_PURPOSES`.
    - `grep "hashlib.sha256" scripts/foundation/fedrec_foundation/rng.py` matches (CR-3 correctness).
    - `grep "dataloader" scripts/foundation/fedrec_foundation/rng.py` matches ("dataloader" is in `_ALLOWED_PURPOSES`).
    - `grep "def torch_gen" scripts/foundation/fedrec_foundation/rng.py` matches.
    - `cd scripts/foundation && pytest tests/test_rng.py -v` prints 10+ passed, including `test_derive_rng_stable_across_processes`.
    - Subprocess test confirms seeds identical across `PYTHONHASHSEED=0,1,random`.
  </acceptance_criteria>
  <done>FND-06 + CR-3 implemented; three RNG factories are deterministic across processes.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Implement FND-07 run manifest with IMP-2 composite hash fields + flip test_manifest green</name>
  <files>
    scripts/foundation/fedrec_foundation/manifest.py
    scripts/foundation/tests/test_manifest.py
  </files>
  <read_first>
    - .planning/phases/01-foundation-contract/01-CONTEXT.md D-15 (embedded + sibling), D-16 (field list)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md §"Pattern 8: Run Manifest (FND-07)" (lines 991-1128)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md CODEX PEER REVIEW §IMP-2 (add `mapping_sha256`, `exclusion_sha256`, `foundation_contract_sha256` to the manifest)
    - .planning/phases/01-foundation-contract/01-VALIDATION.md (test IDs FND-07-a..c)
    - CLAUDE.md (dataclass-first)
  </read_first>
  <behavior>
    - `RunManifest` dataclass with the 18 D-16 fields AND the 3 IMP-2 fingerprints: `mapping_sha256`, `exclusion_sha256`, `foundation_contract_sha256` (all in addition to D-16's `split_hash` and `raw_data_hash`). Plus `schema_version` and `run_id` (so total 23 fields).
    - `generate_run_id()` returns `f"{YYYYMMDD}-{HHMMSS}-{uuid4hex[:6]}"` (UTC).
    - `build_run_manifest(run_id, mode_profile, run_seed, overrides, module, mapping_sha256, split_hash, exclusion_sha256, foundation_contract_sha256, raw_data_hash, builder_version)` assembles the manifest. Reads `flwr.__version__`, `torch.__version__`, `git rev-parse HEAD`.
    - `write_manifest_sibling(manifest, result_json_path)` writes `<parent>/<run_id>-manifest.json` via `atomic_write_json`. Returns the sibling path.
    - `embed_manifest_in_result(manifest, result_dict)` inserts `result_dict["_manifest"] = asdict(manifest)`. Returns the (mutated) dict.
    - Tests flip green: `test_all_fields_populated`, `test_both_writes`, `test_composite_foundation_hash`.
  </behavior>
  <action>
Start from research Pattern 8 (lines 996-1117) and EXTEND the `RunManifest` dataclass with the three IMP-2 fields. Also make the build function take them.

Create `scripts/foundation/fedrec_foundation/manifest.py`:
```python
"""Run manifest / protocol fingerprint (FND-07 + IMP-2).

Written twice per run (D-15):
  1. Embedded under '_manifest' key inside the result JSON.
  2. Sibling <run_id>-manifest.json next to the result file.

Carries all four foundation fingerprints (IMP-2):
  - mapping_sha256
  - split_hash
  - exclusion_sha256
  - foundation_contract_sha256 (composite)
"""
from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from fedrec_foundation.atomic import atomic_write_json

RUN_MANIFEST_SCHEMA_VERSION: int = 1


@dataclass
class RunManifest:
    """Run fingerprint.

    Notes
    -----
    D-16 field set AND IMP-2 composite-hash fields. Downstream writers
    call build_run_manifest(...) then both embed_manifest_in_result(...)
    and write_manifest_sibling(...) to satisfy D-15 (belt-and-suspenders).
    """
    schema_version: int
    run_id: str
    # Mode + locked config (from ModeProfile).
    mode: str
    num_supernodes: int
    partition_mode: str
    fraction_train: float
    fraction_eval: float
    weight_policy: str
    primary_evaluator: str
    num_train_negatives: int
    num_eval_negatives: int
    run_seed: int
    checkpoint_rule: str
    # Foundation fingerprints (IMP-2).
    mapping_sha256: str
    split_hash: str
    exclusion_sha256: str
    foundation_contract_sha256: str
    raw_data_hash: str
    builder_version: str
    # Overrides + module metadata.
    overrides: Dict[str, object]
    module: str
    # Environment.
    flwr_version: str
    torch_version: str
    git_commit: str


def generate_run_id() -> str:
    """Return a run id of the form '20260419-142301-a1b2c3' (UTC).

    Timestamp-slug + short uuid. Human-readable and sortable.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:6]}"


def _git_commit() -> str:
    """Best-effort git rev-parse; returns 'unknown' on failure."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def build_run_manifest(
    run_id: str,
    mode_profile,                            # fedrec_foundation.mode.ModeProfile; duck-typed to avoid circular import
    run_seed: int,
    mapping_sha256: str,
    split_hash: str,
    exclusion_sha256: str,
    foundation_contract_sha256: str,
    raw_data_hash: str,
    builder_version: str,
    overrides: Dict[str, object],
    module: str,
) -> RunManifest:
    """Assemble a RunManifest from a ModeProfile + foundation fingerprints.

    Reads environment versions from flwr and torch and git-rev-parse.
    """
    import flwr
    import torch
    return RunManifest(
        schema_version=RUN_MANIFEST_SCHEMA_VERSION,
        run_id=run_id,
        mode=mode_profile.mode,
        num_supernodes=mode_profile.num_supernodes,
        partition_mode=mode_profile.partition_mode,
        fraction_train=mode_profile.fraction_train,
        fraction_eval=mode_profile.fraction_eval,
        weight_policy=mode_profile.weight_policy,
        primary_evaluator=mode_profile.primary_evaluator,
        num_train_negatives=mode_profile.num_train_negatives,
        num_eval_negatives=mode_profile.num_eval_negatives,
        run_seed=run_seed,
        checkpoint_rule=mode_profile.checkpoint_rule,
        mapping_sha256=mapping_sha256,
        split_hash=split_hash,
        exclusion_sha256=exclusion_sha256,
        foundation_contract_sha256=foundation_contract_sha256,
        raw_data_hash=raw_data_hash,
        builder_version=builder_version,
        overrides=dict(overrides),
        module=module,
        flwr_version=getattr(flwr, "__version__", "unknown"),
        torch_version=torch.__version__,
        git_commit=_git_commit(),
    )


def write_manifest_sibling(manifest: RunManifest, result_json_path: Path) -> Path:
    """D-15 sibling file: write <run_id>-manifest.json next to result JSON.

    Returns the sibling path.
    """
    sibling = Path(result_json_path).parent / f"{manifest.run_id}-manifest.json"
    atomic_write_json(str(sibling), asdict(manifest))
    return sibling


def embed_manifest_in_result(manifest: RunManifest, result_dict: dict) -> dict:
    """D-15 embedded: inject '_manifest' key into an existing result dict."""
    result_dict["_manifest"] = asdict(manifest)
    return result_dict
```

Flip `tests/test_manifest.py` to GREEN:
```python
"""Tests for fedrec_foundation.manifest (FND-07 + IMP-2 + D-15)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fedrec_foundation.manifest import (
    RunManifest, RUN_MANIFEST_SCHEMA_VERSION,
    generate_run_id, build_run_manifest,
    write_manifest_sibling, embed_manifest_in_result,
)


class _StubProfile:
    """Duck-typed ModeProfile stand-in (avoids circular import during Plan 04)."""
    def __init__(self):
        self.mode = "benchmark_cross_device"
        self.num_supernodes = 6040
        self.partition_mode = "natural"
        self.fraction_train = 0.1
        self.fraction_eval = 1.0
        self.weight_policy = "num_positives"
        self.primary_evaluator = "sampled_loo_99"
        self.num_train_negatives = 4
        self.num_eval_negatives = 99
        self.checkpoint_rule = "best_round"


def _build(run_seed=42):
    return build_run_manifest(
        run_id=generate_run_id(),
        mode_profile=_StubProfile(),
        run_seed=run_seed,
        mapping_sha256="m" * 64,
        split_hash="s" * 64,
        exclusion_sha256="e" * 64,
        foundation_contract_sha256="c" * 64,
        raw_data_hash="r" * 64,
        builder_version="1.0.0",
        overrides={"num-supernodes": 100},
        module="baseline",
    )


def test_all_fields_populated() -> None:
    m = _build()
    d = m.__dict__
    # Every D-16 field + IMP-2 extension is present and non-empty.
    for key in (
        "schema_version", "run_id",
        "mode", "num_supernodes", "partition_mode",
        "fraction_train", "fraction_eval",
        "weight_policy", "primary_evaluator",
        "num_train_negatives", "num_eval_negatives",
        "run_seed", "checkpoint_rule",
        "mapping_sha256", "split_hash", "exclusion_sha256", "foundation_contract_sha256",
        "raw_data_hash", "builder_version",
        "overrides", "module",
        "flwr_version", "torch_version", "git_commit",
    ):
        assert key in d, f"missing {key}"
    assert m.schema_version == RUN_MANIFEST_SCHEMA_VERSION
    assert len(m.mapping_sha256) == 64
    assert m.overrides == {"num-supernodes": 100}


def test_both_writes(tmp_path: Path) -> None:
    """D-15: embed in result JSON AND write sibling <run_id>-manifest.json."""
    m = _build()
    result_path = tmp_path / f"{m.run_id}-results.json"
    result = {"final_metrics": {"ndcg@10": 0.42}}
    embed_manifest_in_result(m, result)
    with open(result_path, "w") as f:
        json.dump(result, f)
    sibling = write_manifest_sibling(m, result_path)

    # Embedded key exists in the result JSON.
    loaded = json.loads(result_path.read_text())
    assert "_manifest" in loaded
    assert loaded["_manifest"]["run_id"] == m.run_id

    # Sibling file exists and matches.
    assert sibling.exists()
    sibling_data = json.loads(sibling.read_text())
    assert sibling_data["run_id"] == m.run_id
    assert sibling_data["foundation_contract_sha256"] == "c" * 64


def test_composite_foundation_hash() -> None:
    """IMP-2: manifest carries the composite sha alongside the three inputs."""
    m = _build()
    assert m.mapping_sha256 == "m" * 64
    assert m.split_hash == "s" * 64
    assert m.exclusion_sha256 == "e" * 64
    assert m.foundation_contract_sha256 == "c" * 64
    # A manifest carrying only split_hash (pre-IMP-2) would fail this test.


def test_run_id_format() -> None:
    rid = generate_run_id()
    # Format: YYYYMMDD-HHMMSS-<6 hex>
    parts = rid.split("-")
    assert len(parts) == 3
    assert len(parts[0]) == 8   # date
    assert len(parts[1]) == 6   # time
    assert len(parts[2]) == 6   # short uuid hex


def test_atomic_sibling_write(tmp_path: Path) -> None:
    """Write failure does not leave a partial file."""
    m = _build()
    result_path = tmp_path / "res.json"
    result_path.write_text("{}")
    sibling = write_manifest_sibling(m, result_path)
    assert sibling.exists()
    # No .tmp-* leftovers.
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp-")] == []
```
  </action>
  <verify>
    <automated>cd scripts/foundation &amp;&amp; pytest tests/test_manifest.py -v</automated>
  </verify>
  <acceptance_criteria>
    - File `scripts/foundation/fedrec_foundation/manifest.py` defines `RunManifest`, `RUN_MANIFEST_SCHEMA_VERSION`, `generate_run_id`, `build_run_manifest`, `write_manifest_sibling`, `embed_manifest_in_result`.
    - `grep "foundation_contract_sha256" scripts/foundation/fedrec_foundation/manifest.py` matches (IMP-2 field).
    - `grep "mapping_sha256" scripts/foundation/fedrec_foundation/manifest.py` matches.
    - `grep "exclusion_sha256" scripts/foundation/fedrec_foundation/manifest.py` matches.
    - `cd scripts/foundation && pytest tests/test_manifest.py -v` prints 5 passed.
  </acceptance_criteria>
  <done>FND-07 implemented; manifest carries all four fingerprints per IMP-2; D-15 double-write verified.</done>
</task>

</tasks>

<verification>
- `cd scripts/foundation && pytest tests/test_rng.py tests/test_manifest.py -v` — all tests pass including cross-process subprocess test.
- `python -c "from fedrec_foundation.rng import py_rng, np_rng, torch_gen; print(py_rng(42, 1, 0, 'train_neg').random())"` prints a stable deterministic float.
- `python -c "from fedrec_foundation.manifest import generate_run_id; print(generate_run_id())"` prints the timestamp-slug format.
</verification>

<success_criteria>
- FND-06: Three RNG factories exist (py/np/torch); seed derivation is `hashlib.sha256`-based; `dataloader` is in `_ALLOWED_PURPOSES`; cross-process subprocess test confirms determinism.
- FND-07: RunManifest dataclass has all 23 fields (D-16 + IMP-2 + bookkeeping); `generate_run_id()` produces the timestamp-slug format; write_manifest_sibling uses atomic_write_json; embed_manifest_in_result mutates the result dict with a `_manifest` top-level key.
</success_criteria>

<output>
After completion, create `.planning/phases/01-foundation-contract/01-foundation-contract-04-SUMMARY.md` — document the three RNG factories' namespaces, the `_ALLOWED_PURPOSES` set (including `dataloader` + `server_sample`), the RunManifest field list with IMP-2 composite fields, and note that every downstream phase MUST pass a `torch.Generator` (via `torch_gen(...)`) into every `DataLoader(..., generator=..., shuffle=True)` to satisfy CR-3.
</output>
