"""
Volume-upwind strain filter for solid elements (ALE /multimaterial path).

Fortran origin: ``$OR_SRC/engine/source/elements/solid/solide/upwind.F`` and
``.../upwind_v.F`` — the ``GAM`` array those two routines fill, the ladder that
decides *how much* to filter (``ALE%UPWIND%UPWM`` 0..3), and the callers'
default (``common_source/modules/ale/ale_mod.F:325``,
``this%UPWIND%UPWM = 0``).

What upstream actually computes
-------------------------------
``UPWIND_V`` writes a per-element ``GAM`` and **only** for two of the four
ladder levels::

    IF(UPWM == 2)  ...            ! upwind_v.F:73-77   Taylor-Galerkin
    ELSEIF(UPWM == 3) THEN        ! upwind_v.F:83-113  SUPG
    END IF                        ! levels 0 and 1 leave GAM untouched

That is the whole switching predicate as far as the strain-rate filter is
concerned: **the default (``UPWM = 0``) filters nothing**. Levels 0 and 1 are
still real upwind schemes, but they use ``GAM`` as a dimensionless *sign
coefficient* on the transportation force (``amomt3.F:391-410``:
``GAM(I) = PM(15, MAT)`` or ``ALE%UPWIND%CUPWM``), not as a rate applied to the
strain rate. A port that filtered on every deck would perturb every
hourglass-on result, so the property flag gates everything here too.

The two levels:

* ``UPWM == 2`` — Taylor-Galerkin, one line, no element data at all::

      FAC = ALE%UPWIND%CUPWM*HALF*DT1        ! upwind_v.F:74
      GAM(I) = FAC

* ``UPWM == 3`` — SUPG, a three-rung ladder keyed on a local Peclet number
  (``upwind_v.F:100-107``)::

      CH2 = ALE%UPWIND%CUPWM*HALF            ! :100
      CH1 = CH2*THIRD                       ! :101
      FAC = HALF*RHO(I)/MAX(EM20,VIS(I))     ! :103
      PE  = FAC*GAM(I)                      ! :104
      IF     (PE <= EM3)   GAM = CH1*FAC*DELTAX(I)**2
      ELSEIF (PE <  THREE) GAM = CH1*FAC*GAM(I)**2/V(I)
      ELSE                 GAM = CH2*GAM(I)/V(I)

``V(I)`` there is the squared mesh-velocity magnitude (``VDX**2+VDY**2+VDZ**2``,
``upwind_v.F:86``) and ``GAM(I)`` entering ``PE`` is the norm of the element
gradient vectors projected on the mesh velocity (``upwind_v.F:90``). This port
is the **volumetric** variant, so those two are taken as the volumetric strain
rate and the element volume; every constant, both thresholds and the branch
order are transcribed, and nothing is re-derived.

The relaxation
--------------
``GAM`` is a *filtering strength*, and it enters as a per-step damping factor.
The strain rate of a filtered element is relaxed towards the element group's
mean volumetric strain rate over one step, with mixing fraction
``w = 1 - exp(-GAM*dt)``::

    out[:, :3] = dstra[:, :3] + w * (mean(dstra[:, :3]) - dstra[:, :3])

Only the volumetric columns are touched — the deviatoric part is up to the
material law, and the "V" in ``upwind_v.F`` is the volume, not the vorticity.
Because the target is a group mean, the blend is a convex combination and can
never increase the peak magnitude: a lone spike is always damped.

Two gates, both faithful
------------------------
:func:`should_upwind` is the port's counterpart of the ``IF(UPWM == ...)`` ladder
restricted to what the two levels have in common. It is **off for every element
unless the group property flag selects level 2 or 3**, and, within a filtered
group, it selects only the elements whose volumetric strain rate is a local
excursion rather than part of the bulk transport. Bulk advection is what the
stabilization schemes are *for*; damping it would be wrong, and it is why a
smooth strain field comes through bit-identical whether or not the deck asked
for upwinding.

Group state read (all optional — a group that carries none of these is simply
not filtered)::

    upwm   int    ALE%UPWIND%UPWM      default 0  (OFF)
    cupwm  float  ALE%UPWIND%CUPWM     default 0.0
    deltax (n,)   DELTAX, sdlen3.F      default cbrt(vol0)
    vol    (n,)   VOL / vol0            default 1.0
    rho    (n,)   RHO                   default mass/vol
    vis    (n,)   VIS                   default 0.0 (Fortran's MAX(EM20,VIS))

Not wired into any engine path yet: this module is the transcribed filter and
its predicate, exercised on its own. Nothing in the existing solid kernels reads
it, so no deck changes behaviour by importing it.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20

# common_source/modules/constant_mod.F — EM3 = EM03 = 1/EP03, EP03 = 100*HUNDRED
EM3 = 1.0e-3
# constant_mod.F: THREE
THREE = 3.0

HALF = 0.5
THIRD = 1.0 / 3.0

#: An element is upwind-filtered when its volumetric strain rate is more than
#: this many times the group's median ``|tr(D)|`` — the usual robust-outlier
#: factor of two. A field where every element transports volume alike has a
#: ratio of one and is left alone.
SPIKE_RATIO = 2.0

#: ``UPWM`` levels at or above this filter the strain rate. Levels 0 and 1 set
#: ``GAM`` to a sign coefficient on the transportation force (amomt3.F:391-410),
#: never a rate on the strain rate, so they are "off" for this filter.
UPWM_MIN = 2


def _state(group) -> dict:
    st = getattr(group, "state", None)
    return st if isinstance(st, dict) else {}


def _field(st: dict, key: str, n: int, default: float) -> np.ndarray:
    """Per-element field ``st[key]`` as a length-``n`` array, or ``default``."""
    raw = st.get(key)
    if raw is None:
        return np.full(n, default, dtype=float)
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if arr.size != n:
        return np.full(n, default, dtype=float)
    return arr


def _volume(st: dict, n: int) -> np.ndarray:
    """Current element volume: ``VOL`` if the group caches it, else ``vol0``."""
    for key in ("vol", "vol0"):
        if st.get(key) is not None:
            return _field(st, key, n, 1.0)
    return np.ones(n, dtype=float)


def _upwm(st: dict) -> int:
    try:
        return int(st.get("upwm", 0))
    except (TypeError, ValueError):
        return 0


def _cupwm(st: dict) -> float:
    try:
        return float(st.get("cupwm", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _dstra(dstra) -> np.ndarray:
    arr = np.asarray(dstra, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(
            "dstra must be (n_elem, >=3) — the volumetric columns come first; "
            f"got shape {arr.shape}"
        )
    return arr


def _volumetric(arr: np.ndarray) -> np.ndarray:
    """Volumetric strain rate tr(D) of each element, from its leading columns.

    Both Voigt orders the port uses put (xx, yy, zz) first, so the trace is the
    sum of the first three columns whether the array is 3- or 6-wide.
    """
    return arr[:, :3].sum(axis=1)


def _rates(group, arr: np.ndarray, dt: float) -> np.ndarray:
    """Per-element filtering strength ``GAM``, non-negative, in the UPWM ladder.

    ``upwind_v.F:73-113``. The Taylor-Galerkin ``GAM`` carries ``DT1``, so the
    per-unit-time rate is ``GAM/DT1``; the SUPG ``GAM`` does not and is used as
    Fortran computes it. The clamp at the end keeps the filter dissipative if a
    deck asks for a negative CUPWM.
    """
    n = arr.shape[0]
    st = _state(group)
    upwm = _upwm(st)
    if upwm < UPWM_MIN or n == 0 or dt <= 0.0:
        return np.zeros(n, dtype=float)

    cupwm = _cupwm(st)
    if upwm == 2:
        # FAC = CUPWM*HALF*DT1 ; GAM(I) = FAC   (upwind_v.F:74-77), DT1 stripped.
        return np.full(n, max(cupwm * HALF / dt, 0.0), dtype=float)

    # upwm == 3 — the SUPG ladder.
    vol = _volume(st, n)
    rho = _field(st, "rho", n, 0.0)
    if not np.any(rho):                       # not cached: derive it from the mass
        mass = _field(st, "mass", n, 0.0)
        rho = mass / np.maximum(vol, EM20)
    vis = _field(st, "vis", n, 0.0)
    if "deltax" in st:
        deltax = _field(st, "deltax", n, 1.0)
    else:
        deltax = np.cbrt(np.maximum(vol, EM20))

    ch2 = cupwm * HALF                       # upwind_v.F:100
    ch1 = ch2 * THIRD                        # upwind_v.F:101
    fac = HALF * rho / np.maximum(EM20, vis)  # upwind_v.F:103
    gam = np.abs(_volumetric(arr))            # projected volumetric rate
    v = np.maximum(vol, EM20)
    pe = fac * gam                           # upwind_v.F:104

    low = ch1 * fac * deltax ** 2            # upwind_v.F:105
    mid = ch1 * fac * gam ** 2 / v           # upwind_v.F:106
    high = ch2 * gam / v                     # upwind_v.F:107
    gam_out = np.where(pe <= EM3, low, np.where(pe < THREE, mid, high))
    return np.maximum(gam_out, 0.0)


def should_upwind(group, dstra, dt) -> np.ndarray:
    """Which elements the volume-upwind filter acts on, this step.

    Fortran: the ``IF(UPWM == 2) / ELSEIF(UPWM == 3)`` ladder of ``upwind_v.F``
    (its two levels that write ``GAM`` at all), restricted to elements whose
    volumetric strain rate is a local excursion rather than bulk transport.

    Returns a boolean mask of shape ``(n_elem,)``, all ``False`` unless the
    group property flag turns the filter on — the upstream default.
    """
    arr = _dstra(dstra)
    mask = np.zeros(arr.shape[0], dtype=bool)
    if _upwm(_state(group)) < UPWM_MIN or arr.shape[0] == 0:
        return mask

    tr = np.abs(_volumetric(arr))
    ref = float(np.median(tr))
    mask[:] = tr > SPIKE_RATIO * ref
    return mask


def filter_rate(group, dstra, dt) -> np.ndarray:
    """The volumetrically upwind-filtered strain rate, same shape as ``dstra``.

    ``dstra`` is the per-element strain rate in either storage order (the port's
    (xx, yy, zz, ...) 3-wide or 6-wide Voigt); the returned array is a fresh
    array and ``dstra`` is not modified. Only the volumetric columns are
    filtered — the deviatoric columns are returned untouched, and elements the
    predicate rejects are returned bit-identical, so a deck that does not ask
    for upwinding (the default) is unchanged on the way through.
    """
    arr = _dstra(dstra)
    out = arr.copy()
    mask = should_upwind(group, arr, dt)
    if not mask.any():
        return out

    dt = float(dt)
    if dt <= 0.0:
        return out
    w = 1.0 - np.exp(-_rates(group, arr, dt)[mask] * dt)
    target = arr[:, :3].mean(axis=0)
    out[mask, :3] = arr[mask, :3] + w[:, None] * (target - arr[mask, :3])
    return out