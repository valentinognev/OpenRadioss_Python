"""Task P2.8 — the solid thermal strain: ``mmain``'s ``ETH`` block.

Fortran origin
--------------
The thermal strain of a **solid** is applied by the material cycle, in
``engine/source/materials/mat_share/mmain.F90:758-791``::

    if (iexpan > 0 .and. jthe < 0 .and. tt/=zero .and. mtn /= 137) then
      ...
      ifunc_alpha = mat_elem%mat_param(imat)%therm%func_thexp
      fscal_alpha = mat_elem%mat_param(imat)%therm%scale_thexp
      tempel0(i)  = lbuf%temp(i)
      alpha = finter(ifunc_alpha,tempel(i),npf,tf,bidon1)
      alpha = alpha * fscal_alpha
      eth(i)= alpha *(tempel(i)-tempel0(i))*off(i)          ! 778
      lbuf%forth(i) = lbuf%forth(i) + eth(i)
      epsth(i)= three*lbuf%forth(i)
      dxx(i)  = dxx(i)-eth(i)/max(dt1,em20)                 ! 781
      dyy(i)  = dyy(i)-eth(i)/max(dt1,em20)
      dzz(i)  = dzz(i)-eth(i)/max(dt1,em20)
      dvol(i) = dvol(i)-three*eth(i)*vol_avg(i)
      amu(i)  = amu(i) + epsth(i)
      sigkk(i)= lbuf%sig(1)+lbuf%sig(2)+lbuf%sig(3)         ! 786
      lbuf%eintth(i) = lbuf%eintth(i)-half*sigkk(i)*eth(i)   ! 787
    end if

``TEMPEL`` is the element temperature ``/HEAT//MAT`` produces and ``TEMPEL0``
is the reference stored on the element buffer, so the argument this module
calls ``dT`` is exactly ``TEMPEL - TEMPEL0`` — the temperature rise of THIS
step, not the absolute temperature.

What the five routines the plan names actually are
--------------------------------------------------
``plan/03_phase2_elements_solid.md`` names ``stherm.F``, ``s4therm.F``,
``s4therm-itet1.F``, ``sctherm.F`` and ``s6ctherm.F``.  All five were read in
full before this file was written, and all five are the **heat-conduction
source term** of their element family, not the thermal strain::

    stherm.F        KC = (AS + BS*TEMPEL)*VOL*DT1*THEACCFACT
    s4therm.F       KC = (CA + CB*TEL   )*DT1*VOL*THEACCFACT
    s4therm-itet1.F  the same, computing its own PX/PY/PZ from XX/YY/ZZ
    sctherm.F       the same, accumulating (FPHI = FPHI + ...) on a thickshell
    s6ctherm.F      the same on a 6-node wedge, HEAT/6

Each builds the conduction flux from PRE-COMPUTED shape gradients
``PX1..PZ4``, scales it by the temperature-dependent conductivity
``CA + CB*T`` times the volume and the time step, and adds the volumetric heat
source ``HEAT`` spread over its nodes.  **None of them reads ``ALPHA``, none
of them touches a stress, and none of them books a mechanical energy.**  The
thermal *strain* they are named for lives in ``mmain``, above; that is what
this module ports.  The conduction routines are a separate piece of work on
the heat side of the coupling and are not started here.

What the four entry points are
------------------------------

* :func:`thermal_strain` is ``mmain.F90:778`` — ``ETH`` — the free expansion
  an element *would* undergo, ``alpha * (TEMPEL - TEMPEL0) * OFF``.
* :func:`effective_strain_increment` is ``mmain.F90:781-783`` — the
  strain-rate subtraction, which upstream performs **before** the
  constitutive call at line 840+.
* :func:`forces` is that rule evaluated to the nodal force of a **fully
  constrained** element (``sfint3``'s ``f_i = -V sigma . gradN_i``, as
  ``solid_hexa8._post`` writes it), i.e. the reaction a blocked expansion
  produces.
* :func:`energy` is ``mmain.F90:786-787`` — the thermal-energy ledger entry
  ``-HALF * tr(sigma) * ETH``, times the volume so the caller books a work.

Four things a port must not "improve"
-------------------------------------

**The order is the physics.**  The thermal strain is subtracted from the
strain rate BEFORE the constitutive update, so the stress that reaches the
``EINTTH`` booking at line 787 is the stress the element *entered the step
with* — the pre-update one — and it enters at **half** weight.  The
consequence is sharp and is a test rather than a claim: at the first thermal
step of a body that was unstressed, the booking is **exactly zero**.  A port
that subtracted the thermal strain after the constitutive update (or read the
post-update stress) books a different number, differing by ``sigma . alpha .
dT`` — the plan's characterisation of the sensitivity this test pins.

**The expansion coefficient of a solid is ONE scalar.**  ``mmain.F90:776``
evaluates ``alpha`` once per element and subtracts the same ``ETH`` from
``DXX``, ``DYY`` and ``DZZ`` — isotropic expansion.  The per-direction
``ETHXX/ETHYY/ETHZZ`` of ``materials/mat_share/thermexpc.F`` belongs to the
SHELL path (and there only for layer-structured stacks), so a length-3
``alpha`` here would be an anisotropy no solid element has.  ``alpha`` is an
argument of every entry point and is never looked up: the ``iexpan > 0`` gate
is set by the Starter from ``IPM(218) > 0`` on the ``/MAT`` card
(``starter/source/elements/solid/solide/sgrtails.F:1556``), and reading
``alpha`` off a material law is Phase 6's
``materials.thermal_expansion(law)``.

**``OFF`` kills the thermal strain.**  ``ETH(I) = ... * OFF(I)``, so a deleted
element expands by nothing, exerts no thermal force and books no thermal
energy.  It is not a "small" contribution — it is exactly zero.

**``MAX(DT1, EM20)`` is a floor, not a tolerance.**  It is the only thing
between a zero time step and a division by zero, and it is reproduced rather
than replaced by ``dt or EM20`` (which would turn a negative dt into a
positive step).

Where the material constants come from
--------------------------------------
:func:`forces` needs the elastic operator's action on the isotropic tensor,
``C : m`` with ``m = [1, 1, 1, 0, 0, 0]``.  For an isotropic material that is
``3K`` exactly (``C_ijkl = lam*di j*dk l + G*(dik*djl + dil*djk)`` contracts
with the identity to ``3*lam + 2*G = 3K``), so the thermal stress of an
isotropic element is ``-3K * alpha * dT`` in every normal component — the
*only* place this module reads a material constant, and it reads it off the
group's own ``state["slices"]``, exactly as every other solid kernel does.
Nothing else here looks at a material.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from . import solid_hexa8

#: Voigt order of the three normal components — mmain's DXX/DYY/DZZ
_NORMAL = slice(0, 3)


def _off(group) -> np.ndarray:
    """``GBUF%OFF`` — 1 while the element is alive, 0 once deleted
    (``init_group`` allocates it; ``mmain.F90:778`` multiplies by it)."""
    n = group.n
    state = getattr(group, "state", None) or {}
    off = state.get("off")
    if off is None:
        return np.ones(n)
    off = np.asarray(off, dtype=np.float64).reshape(-1)
    if off.shape[0] != n:
        raise ValueError(f"off has {off.shape[0]} entries, group has {n}")
    return off


def _per_element(value, n: int, name: str) -> np.ndarray:
    """Broadcast a scalar or an already per-element array to ``(n,)``."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.full(n, arr[0])
    if arr.size != n:
        raise ValueError(
            f"{name} must be a scalar or one value per element "
            f"({n}), got {arr.size}.  The solid thermal expansion is "
            f"ISOTROPIC — upstream evaluates a single ALPHA per element "
            f"(mmain.F90:776) and subtracts the same ETH from all three "
            f"normals; the per-direction ETHXX/ETHYY/ETHZZ of "
            f"thermexpc.F is the SHELL path.")
    return arr


def _geometry(group):
    """``(vol, dndx)`` from the group's current nodal positions, through the
    ported brick geometry of ``solid_hexa8._geometry`` (``srcoor3.F`` /
    ``sderi3.F``).  The thermal force must use the SAME gradients as the
    element's own internal force or the two would not cancel."""
    model = getattr(group, "_model", None)
    if model is None:
        raise KeyError(
            "group has no _model: solid_thermal needs the nodal positions")
    x = getattr(model, "x", None)
    if x is None:
        x = model.x0
    xe = np.asarray(x)[group.conn]
    dndx, vol = solid_hexa8._geometry(xe)
    return vol, dndx


def _three_k(group) -> np.ndarray:
    """``C : m`` — the isotropic projection of the elastic operator, element for
    element, read off the group's material slices.  ``K`` when the material
    carries it, otherwise ``E / (3 (1 - 2 nu))``, which is the same number.

    A group with no slice raises ``KeyError``: a stiffness invented for a
    missing slice would put a stress nobody asked for into the cycle.
    """
    out = np.zeros(group.n)
    for sl, mat, _prop in group.state["slices"]:       # KeyError if absent
        k = getattr(mat, "K", None)
        if k is None:
            e = getattr(mat, "E", None)
            nu = getattr(mat, "nu", None)
            if e is None or nu is None:
                raise KeyError(
                    f"material {getattr(mat, 'law', '?')} carries neither K "
                    f"nor E/nu, so C:m = 3K is not defined")
            k = e / (3.0 * (1.0 - 2.0 * nu))
        out[sl] = 3.0 * k
    return out


def thermal_strain(group, dT, alpha) -> np.ndarray:
    """``ETH`` — the free thermal strain of every element in the group.

    ``mmain.F90:778``::

        eth(i) = alpha * (tempel(i) - tempel0(i)) * off(i)

    dT : the element temperature RISE of this step, ``TEMPEL - TEMPEL0`` —
         ``TEMPEL`` is the element temperature ``/HEAT//MAT`` produces and
         ``TEMPEL0`` the reference stored on the element buffer
         (``mmain.F90:775``).  Scalar, or one value per element.
    alpha: the expansion coefficient, one ISOTROPIC value per element
           (``mmain.F90:776-777``).  Scalar, or one value per element.

    Returns (nel,) — the strain the element would undergo **if nothing
    restrained it**.  Upstream accumulates it into ``GBUF%FORTH`` and sets
    ``EPSTH = 3*FORTH``; both of those belong to the caller that owns those
    buffers, so neither is written here.
    """
    n = group.n
    if n == 0:
        return np.empty(0)
    return (_per_element(alpha, n, "alpha") * _per_element(dT, n, "dT")
            * _off(group))


def effective_strain_increment(deps: np.ndarray,
                               eth: np.ndarray) -> np.ndarray:
    """``mmain.F90:781-783`` — the strain increment the constitutive law sees.

    Upstream subtracts ``ETH / MAX(DT1, EM20)`` from the strain RATE
    (``DXX``, ``DYY``, ``DZZ``) *before* the material-law call, which is the
    order the whole coupling depends on; integrated over ``DT1`` that is the
    increment ``deps - ETH`` on the three normals.  The engineering shears
    are untouched — thermal expansion has no shear part, which is another way
    of saying the coupling is isotropic.

    deps : (nel, 6) Voigt strain INCREMENT from the kinematics
           (``solid_hexa8._pre``'s ``deps``).
    eth  : (nel,) from :func:`thermal_strain`.

    Returns a new (nel, 6) array; the input is not modified.

    Upstream's other companions of the same block — ``DVOL(I) = DVOL(I) -
    3*ETH(I)*VOL_AVG(I)``, ``AMU(I) = AMU(I) + EPSTH(I)`` with
    ``EPSTH = 3*FORTH`` (``mmain.F90:780, 784-785``) — belong to the element
    cycle that owns ``DVOL`` and ``AMU``; they are recorded here rather than
    duplicated.
    """
    d = np.asarray(deps, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != 6:
        raise ValueError(f"deps must be (nel, 6) Voigt, got {d.shape}")
    e = np.asarray(eth, dtype=np.float64).reshape(-1)
    if e.shape[0] != d.shape[0]:
        raise ValueError(f"eth has {e.shape[0]} entries, deps has {d.shape[0]}")
    out = d.copy()
    out[:, _NORMAL] -= e[:, None]
    return out


def forces(group, dT, alpha, dt: float) -> np.ndarray:
    """The nodal force a **fully constrained** element feels when heated.

    "Fully constrained" is the ``deps = 0`` case of
    :func:`effective_strain_increment`: every node held, so the only strain
    the kinematics produce is the one the thermal expansion tried to produce
    and could not.  Upstream reaches the same state by the same route — the
    subtraction happens before the constitutive update, so the law sees a
    contraction of ``ETH`` and returns a compressive stress.

    The stress follows ``mmain.F90:781`` exactly, the strain rate first and
    the time step after it::

        deps_th = -ETH * DT / MAX(DT, EM20)

    so any positive ``dt`` reproduces ``-ETH`` (and hence the same stress —
    the time step cancels, as it must, because thermal expansion is a
    temperature effect and not a rate effect), and the ``MAX(..., EM20)``
    floor is what keeps ``dt = 0`` from dividing by zero.

    For an isotropic material the stress is ``C : deps_th`` with ``C : m =
    3K`` and ``m = [1,1,1,0,0,0]``, i.e. ``-3K * alpha * dT`` in each normal
    component: **compressive**, because a blocked expansion is a blocked
    increase in volume.  The nodal force is ``f_i = -V sigma . gradN_i`` —
    the ``sfint3`` form ``solid_hexa8._post`` uses, so this force and the
    element's own internal force cancel exactly when they should.

    dT, alpha : as in :func:`thermal_strain`.
    dt : the cycle time step.

    Returns (nel, 8, 3).  A deleted element (``OFF = 0``) contributes exactly
    zero, because its ``ETH`` is zero (``mmain.F90:778``).
    """
    n = group.n
    if n == 0:
        return np.zeros((0, 8, 3))
    eth = thermal_strain(group, dT, alpha)
    # mmain.F90:781 — the thermal strain is a RATE subtraction floored at EM20,
    # and the constitutive update integrates that rate back over DT.
    deps_th = -eth * float(dt) / max(float(dt), EM20)
    # C : m = 3K for an isotropic law, so the thermal stress on the three
    # normals is 3K * deps_th and the shears stay zero.
    sig_n = _three_k(group) * deps_th
    vol, dndx = _geometry(group)
    fe = dndx * (-vol * sig_n)[:, None, None]
    return np.where(_off(group)[:, None, None] > 0.0, fe, 0.0)


def energy(group, sig: np.ndarray, dT, alpha) -> float:
    """The thermal-energy ledger entry — ``mmain.F90:786-787``.

    ``lbuf%eintth(i) = lbuf%eintth(i) - half*sigkk(i)*eth(i)``, with
    ``SIGKK`` the sum of the three normal stresses **read before the
    constitutive update** (the call is 200 lines further down).  That is the
    order, and it is why the first thermal step of an unstressed element
    books exactly zero.

    Signed: a body held while it cools books a negative entry, one held while
    it heats a positive one — the constraint does work on the body, and the
    ledger has to carry it.  Multiplied by the element volume so the caller
    books a work and not a density (``mmain`` books per unit volume into
    ``GBUF%EINTTH``; the port's ``eint`` ledger is a work, exactly as
    ``solid_hexa8._post``'s ``deint0`` is ``vol * sigma_mid : deps``).

    sig : (nel, >=3) Voigt stress as it stands BEFORE this step's constitutive
          update.  Only the three normal components enter, as in ``mmain``.

    Returns the group's total, as a float.
    """
    n = group.n
    if n == 0:
        return 0.0
    s = np.asarray(sig, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] != n or s.shape[1] < 3:
        raise ValueError(f"sig must be (nel, >=3) Voigt with nel={n}, "
                         f"got {s.shape}")
    eth = thermal_strain(group, dT, alpha)
    vol, _ = _geometry(group)
    sigkk = s[:, 0] + s[:, 1] + s[:, 2]
    return float(np.sum(-0.5 * sigkk * eth * vol))