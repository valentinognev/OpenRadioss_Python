"""
pyradioss.accel — optional accelerated compute backends (M7).

There is no Fortran origin for this package: OpenRadioss gets its speed
from compiled Fortran + OpenMP/MPI (out of scope for the port). This
package is the port's own answer to "the readable NumPy solver is slow":
an *optional* numba backend behind the exact same kernel API.

Backend architecture
--------------------
The element kernels and the TYPE7 contact keep their NumPy code inline —
that code IS the reference implementation and the readable narrative of
the port. Each hot kernel is split into "pre"/"post" blocks around its
Python-level material/failure loop (which stays pure Python — laws
dispatch on law objects; the one M39 exception is a handful of LAW70
tabulated-foam numeric LEAVES, whose mirrors are BITWISE-identical):

    solid_hexa8: hexa_pre / hexa_post (+ hexa_hgphys for LAW70 bricks)
    shell_bt4:   shell_pre / shell_post
    tetra10:     tetra10_pre / tetra10_post
    inter_type7: t7_narrow (+ the t24_narrow alias)
    law70:       law70_tab2d / enorm / snorm / elastic_stress (bitwise)

Each block is a plain function of arrays. ``accel.get(name)`` returns
the active backend's implementation, or ``None`` meaning "use the inline
NumPy code" — a reader can ignore the accelerated path entirely, and
the NumPy path never pays more than one dict lookup per block.

Selecting a backend
-------------------
The default is ``auto`` (M40): numba IS the default, but only where the
M39 benchmark showed it wins.  ``accel.auto_select_backend(model)`` — the
engine calls it once, after the restart is read and BEFORE the first
kernel call — picks numba when (a) numba imports cleanly and (b) the model
is big enough to amortise the JIT warm-up (>= ``_AUTO_MIN_ELEMENTS`` total
elements; see that constant for the derivation from the M39 A/B sweep).
Small models where warm-up dominates stay on NumPy.  P3.13 outcome (b):
even on a numba model, ``get`` withholds the QBAT/QEPH blocks listed in
``_AUTO_EXCLUDED_KERNELS`` — their mirrors are faster but not yet
bitwise-identical, so those groups run the NumPy reference until real
mirrors land.  Either way the engine
listing carries a ``COMPUTE BACKEND`` line naming the choice and the reason.

A user can still PIN a backend, which the auto path then leaves untouched:

* environment variable ``PYRADIOSS_BACKEND=numpy|numba`` (read lazily on
  the first kernel call, so it works for pytest and plain scripts; ``auto``
  or unset = the auto default), or
* the engine command line: ``pyradioss-engine -i RUN_0001.rad -backend
  numba`` (overrides the environment; ``-backend auto`` = the default), or
* programmatically: ``accel.select_backend("numba")``.

numba is an OPTIONAL dependency: requesting the numba backend without
numba installed prints a warning and falls back to NumPy — the base
install keeps working unchanged. The numba kernels are compiled with
``cache=True`` so the (one-time, ~10 s) JIT cost is paid on the first
run only; subsequent processes load the cached machine code.

The parity contract (the M7 hard requirement)
---------------------------------------------
A backend MUST NOT change results. Concretely:

* every numba kernel mirrors its NumPy block operation for operation —
  same formulas, same branch structure, same evaluation order — so the
  two backends agree bitwise wherever the NumPy expression maps to one
  scalar expression per element;
* the unavoidable differences are *reduction associations* (a numba
  scalar loop accumulates sequentially; NumPy einsum/matmul may use
  SIMD partial sums). These show up at machine precision (≲1e-15
  relative per operation) and are the documented per-kernel tolerance;
* ``tests/test_m7_backends.py`` enforces the contract at two levels:
  single-call kernel parity on real element groups (tight tolerance)
  and full starter+engine runs compared state-array by state-array;
* the M6 restart-chaining acceptance tests (chained == unchained,
  exactly) must hold WITHIN each backend — reordered reductions would
  break that immediately, which makes the chaining tests the regression
  canary for nondeterminism (e.g. accidentally enabling numba's
  ``parallel=True`` or ``fastmath=True``, both of which are forbidden
  here).

Why there is no JAX backend (deferred, see PORTING_GUIDE §5)
------------------------------------------------------------
JAX's value is jit-compiling the WHOLE cycle onto an accelerator, but
this engine's cycle is built around in-place scatter into shared force
arrays, per-cycle Python branching (sensors, deletion, EOS, /DT/NODA)
and stateful dict-of-arrays element buffers — none of which map to
jax.jit without rewriting the engine in functional style, i.e. a fork,
not a backend. A jax.numpy drop-in without jit would only add dispatch
overhead at these array sizes. Deferred until (if ever) the engine
grows a functional-core refactor.
"""

from __future__ import annotations

import os
import warnings

#: resolved backend: {"name": "numpy"|"numba"|"cupy", "mod": module or None,
#: "forced": bool}.  ``name = None`` means "not resolved yet" — the
#: environment variable is read lazily on the first ``get()`` so library
#: users, pytest and the CLIs all honour PYRADIOSS_BACKEND without extra
#: plumbing.  ``forced`` is True when the user PINNED a backend explicitly
#: (PYRADIOSS_BACKEND / -backend numpy|numba|cupy, or a direct select_backend
#: call); ``auto_select_backend`` overrides only when it is False (M40).
_state = {"name": None, "mod": None, "forced": False}


#: M40 auto threshold — the minimum TOTAL element count for which the numba
#: backend beats NumPy end-to-end.  DERIVED, not guessed, from the M39
#: numpy->numba A/B sweep (tools/validation_data/perf_m39_speed.json) joined
#: with the per-deck sizes (perf_m39.json) and the profile_cycle.py regimes:
#:
#:   deck            elements  nodes   numba speedup   verdict
#:   gas_piston            4     20        0.456x      LOSS  (sole regression)
#:   antenna_mast         10     12        1.124x      marginal (in contention
#:                                                     noise — whole sweep was
#:                                                     run under the user's MPI)
#:   tensile_bar          40     99        1.537x      robust win  <- smallest
#:   rubber_block         64    125        1.442x      robust win
#:   c04_E1000            99    137        1.732x      robust win
#:   rigid_impactor      118    170        2.382x      robust win
#:   notched_plate       194    231        2.494x      robust win
#:   box_beam_impact     200    220        1.961x      robust win
#:   edge_impact         256    330        1.838x      robust win
#:   spot_weld           300    372        1.869x      robust win
#:   c46_LAW70          1000   1333    1.81-2.89x      robust win
#:   c37_T1040         65439  72047        1.773x      robust win
#:
#: The ONE regression (gas_piston, 4 elem: JIT warm-up + per-call dispatch
#: swamp its 0.9 s NumPy baseline) and the ONE marginal case (antenna_mast,
#: 10 elem, inside the sweep's contention noise) both sit far below the
#: smallest ROBUST win (tensile_bar, 40 elem); every larger deck wins
#: 1.44-2.49x.  32 sits in the (10, 40) element gap — 8x the regression, >3x
#: the marginal, and below every robust win.  NODE count is deliberately NOT
#: the metric: gas_piston has MORE nodes (20) than antenna_mast (12) yet
#: loses, so nodes misclassify it — the element count is the causal workload
#: for the JIT kernels (bricks/tetras/shells/contact all scale with it).
#: Override with PYRADIOSS_BACKEND_AUTO_MIN_ELEMENTS (int) for tuning/tests.
_AUTO_MIN_ELEMENTS = 32


#: P3.13 outcome (b) — kernel blocks the ``auto`` rule must NOT serve, even
#: on a model big enough for numba. The QBAT/QEPH mirrors in
#: ``jit_kernels/shells_qbat.py`` / ``shells_qeph.py`` are faster than NumPy
#: at the kernel level but are NOT bitwise-identical to the reference
#: (measured: wrong condensed length, a different flat/warped classifier,
#: end-to-end forces off at rel ~1 — formula drift, not reduction noise),
#: so ``get`` withholds them on the auto path and those groups run NumPy.
#: A user PIN (forced) still serves them. Delete this set when outcome (a)
#: — mirrors that are BOTH bitwise-identical and faster — lands.
_AUTO_EXCLUDED_KERNELS = frozenset(
    {"qbat_pre", "qbat_post", "qeph_pre", "qeph_post"})


def _load_numba_module():
    """Import (hence probe the availability of) the numba kernel module.
    Factored out so the auto path and the tests share ONE seam — tests
    monkeypatch this to simulate numba being absent without touching the
    real import machinery."""
    import numba  # noqa: F401 — probe availability before loading kernels
    from . import jit_kernels  # import compiles/loads the mirrors
    return jit_kernels


def _load_cupy_module():
    """Import (hence probe the availability of) the CuPy GPU kernel module.
    Factored out so the auto path and the tests share ONE seam — tests
    monkeypatch this to simulate cupy being absent without touching the
    real import machinery."""
    import cupy  # noqa: F401 — probe availability before loading kernels
    if hasattr(cupy, "cuda") and hasattr(cupy.cuda, "is_available"):
        if not cupy.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
    from . import gpu_kernels
    return gpu_kernels


def select_backend(name=None, log=None) -> str:
    """Select the compute backend; returns the name actually activated.

    ``name = None`` reads ``PYRADIOSS_BACKEND`` (default ``"auto"`` since
    M40).  A concrete ``"numpy"`` / ``"numba"`` / ``"cupy"`` — whether from the env var,
    the ``-backend`` CLI flag, or a direct call — PINS that backend: it is
    recorded as *forced* so ``auto_select_backend`` leaves it untouched.
    ``"auto"`` (the default) resolves *provisionally* to NumPy — the safe
    choice while no model size is known — and is marked NOT forced so the
    engine can upgrade it to cupy or numba once the restart is read.  Unknown names,
    the deferred ``jax``, and a missing numba/cupy installation warn and fall
    back to NumPy — requesting acceleration must never break a run."""
    if name is None:
        name = os.environ.get("PYRADIOSS_BACKEND", "auto")
    name = (name or "auto").strip().lower()

    if name == "auto":
        # provisional NumPy; auto_select_backend decides once a model exists
        _state.update(name="numpy", mod=None, forced=False)
    elif name == "numba":
        try:
            _state.update(name="numba", mod=_load_numba_module(), forced=True)
        except (ImportError, RuntimeError, TypeError, NameError):
            msg = ("numba backend requested but numba is not installed — "
                   "falling back to NumPy (pip install numba)")
            warnings.warn(msg, stacklevel=2)
            if log is not None:
                log.warning(msg, "BACKEND")
            _state.update(name="numpy", mod=None, forced=True)
    elif name == "cupy":
        try:
            _state.update(name="cupy", mod=_load_cupy_module(), forced=True)
        except (ImportError, RuntimeError, TypeError, NameError):
            msg = ("cupy backend requested but cupy is not installed or "
                   "CUDA is unavailable — falling back to NumPy (pip install cupy)")
            warnings.warn(msg, stacklevel=2)
            if log is not None:
                log.warning(msg, "BACKEND")
            _state.update(name="numpy", mod=None, forced=True)
    elif name == "numpy":
        _state.update(name="numpy", mod=None, forced=True)
    elif name == "jax":
        msg = ("JAX backend is deferred (see the M7 notes in "
               "PORTING_GUIDE.md §5) — falling back to NumPy")
        warnings.warn(msg, stacklevel=2)
        if log is not None:
            log.warning(msg, "BACKEND")
        # not forced: fall through to the auto default on the next resolve
        _state.update(name="numpy", mod=None, forced=False)
    else:
        msg = f"unknown PYRADIOSS_BACKEND '{name}' — using NumPy"
        warnings.warn(msg, stacklevel=2)
        if log is not None:
            log.warning(msg, "BACKEND")
        # not forced: an invalid pin must not defeat the auto default
        _state.update(name="numpy", mod=None, forced=False)
    return _state["name"]


def _auto_min_elements() -> int:
    """The active auto threshold (``_AUTO_MIN_ELEMENTS``, env-overridable via
    ``PYRADIOSS_BACKEND_AUTO_MIN_ELEMENTS`` for tuning and tests)."""
    try:
        return int(os.environ.get("PYRADIOSS_BACKEND_AUTO_MIN_ELEMENTS",
                                  _AUTO_MIN_ELEMENTS))
    except (TypeError, ValueError):
        return _AUTO_MIN_ELEMENTS


def _model_element_count(model) -> int:
    """Total explicit-element count across the model's groups — the workload
    the numba kernels accelerate.  0 if the model exposes no groups."""
    try:
        return sum(int(g.n) for _, g in model.element_groups())
    except Exception:
        return 0


def _log_backend(name, reason, log) -> None:
    """Emit the engine-listing COMPUTE BACKEND line (no-op without a log)."""
    if log is not None:
        log.info(f" COMPUTE BACKEND  . . . . . . . . . . : {name} ({reason})")


def auto_select_backend(model, log=None, *, explicit=True) -> str:
    """M40 AUTO resolution — the engine calls this once, after the restart is
    read (so the element count is known) and BEFORE the first kernel call.
    Returns the activated backend name and logs it with the deciding reason.

    * A user-PINNED backend (forced via PYRADIOSS_BACKEND / -backend /
      select_backend) is respected untouched.
    * Otherwise cupy is chosen when available (fastest GPU acceleration)
      AND the model has at least ``_auto_min_elements()`` elements AND
      the run is ``explicit``.
    * Otherwise numba is chosen when it imports cleanly AND the model has at
      least ``_auto_min_elements()`` elements AND the run is ``explicit``
      (the M39 speed sweep that set the threshold covers the /RUN leap-frog
      path only — implicit runs stay on NumPy under auto unless pinned).
      Everything else falls back to NumPy; numba/cupy missing -> NumPy + warning.
    Small models where warm-up dominates therefore stay on NumPy — the
    documented threshold derivation (``_AUTO_MIN_ELEMENTS``) is the gate."""
    # resolve env/default the first time (sets name + forced)
    if _state["name"] is None:
        select_backend(log=log)

    if _state["forced"]:
        _log_backend(_state["name"], "forced by request", log)
        return _state["name"]

    nelem = _model_element_count(model)
    thr = _auto_min_elements()

    if not explicit:
        _state.update(name="numpy", mod=None)
        _log_backend("numpy", "auto: implicit run — accelerated auto-enable is "
                     "explicit-only (the M39 speed data covers /RUN)", log)
        return "numpy"

    if nelem < thr:
        _state.update(name="numpy", mod=None)
        _log_backend("numpy", f"auto: {nelem} elements < {thr} — JIT/GPU warm-up "
                     "dominates (see accel threshold)", log)
        return "numpy"

    try:
        _state.update(name="cupy", mod=_load_cupy_module())
        _log_backend("cupy", f"auto: {nelem} elements >= {thr} (GPU accelerated)", log)
        return "cupy"
    except (ImportError, RuntimeError, TypeError, NameError):
        pass

    try:
        _state.update(name="numba", mod=_load_numba_module())
        _log_backend("numba", f"auto: {nelem} elements >= {thr}", log)
        return "numba"
    except (ImportError, RuntimeError, TypeError, NameError):
        _state.update(name="numpy", mod=None)
        _log_backend("numpy", "auto: cupy and numba not installed "
                     "(pip install cupy or numba to enable)", log)
        return "numpy"


def backend_name(log=None) -> str:
    """The active backend name ("numpy", "numba", or "cupy"), resolving lazily
    (``log`` routes a fallback warning into the engine listing)."""
    if _state["name"] is None:
        select_backend(log=log)
    return _state["name"]


def get(kernel: str):
    """The active backend's implementation of a kernel block, or None
    meaning 'use the inline NumPy reference code'. This is the single
    dispatch point the element kernels call (one dict lookup + getattr
    per block per cycle — negligible). On the auto path (not user-pinned),
    the P3.13-excluded QBAT/QEPH blocks always resolve to None."""
    if _state["name"] is None:
        select_backend()
    if kernel in _AUTO_EXCLUDED_KERNELS and not _state["forced"]:
        return None
    mod = _state["mod"]
    return getattr(mod, kernel, None) if mod is not None else None


from .gpu_transfer import GPUArrayManager  # noqa: E402

