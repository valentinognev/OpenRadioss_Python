"""
Two-node Rivet connector element (/RIVET + /PROP/RIVET, /PROP/TYPE5).

Fortran origin:
  ``engine/source/elements/rivet/rivet1.F`` - engine step kernel (RIVET1).

The rivet element connects two nodes with a rigid-body-like constraint that
degrades and fails when the combined normal/tangential force exceeds a limit.

Element buffer consumed by the kernel - the GEO array read by RIVET1
(rivet1.F lines 99-100, 138-141, 352-354):
    GEO(1) = FN**2 - squared normal failure force
    GEO(2) = FT**2 - squared tangential failure force
    GEO(3) = DX**2 - squared maximum separation distance
    GEO(4) = IROT - rotation flag (0 = translation only, 1 = translation+rotation)
    GEO(5) = IMOD - formulation (1 = rigid body, 2 = old rigid link)

Engine loop (rivet1.F lines 93-415):
    1. Check separation; decrement OFF by 0.1 each step if DX2 > DMX2
    2. IMOD==1 (rigid body): mass-weighted CG velocity constraint
    3. IMOD==2 (old rigid link): reduced-mass impulse formulation
    4. Failure: sqrt(AN2/FN2 + AT2/FT2) capped at 1/OFF (elliptic criterion)
    5. Log failure when OFF reaches 0

Sign convention: accelerations A(3,*) are **accumulated** — the rivet adds
a corrective impulse force/mass to each node in equal-and-opposite pairs.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Small-number guard (rivet1.F: EM15 ≈ 1e-15, EM01 = 0.1)
_EM15 = 1.0e-15
_EM01 = 0.1  # damage decrement per step (rivet1.F line 105: OFF=OFF-EM01)
_EP15 = 1.0e15  # default for FN/FT/DX when the card field is blank or zero (unbounded)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RivetProp:
    """Kernel-side view of the /PROP/RIVET (TYPE5) parameters.

    Fields mirror the GEO array consumed by RIVET1 (rivet1.F lines 138-141):
        GEO(1) = fn_max**2
        GEO(2) = ft_max**2
        GEO(3) = d_max**2
        GEO(4) = irot  (+ 0.1 in Fortran to avoid int/float truncation)
        GEO(5) = imod  (+ 0.1 in Fortran)
    """
    id: int = 0
    fn_max: float = _EP15   # max normal force  (NFORCE)
    ft_max: float = _EP15   # max tangential force (TFORCE)
    d_max: float = _EP15    # max separation before damage starts (LENGTH)
    irot: int = 0           # 0=translation only, 1=translation+rotation
    imod: int = 1           # 1=rigid body (default), 2=old rigid link
    title: str = ""

    @property
    def fn2(self) -> float:
        """GEO(1): squared normal failure force (rivet1.F line 137)."""
        return self.fn_max ** 2 if self.fn_max != 0.0 else _EP15 ** 2

    @property
    def ft2(self) -> float:
        """GEO(2): squared tangential failure force (rivet1.F line 138)."""
        return self.ft_max ** 2 if self.ft_max != 0.0 else _EP15 ** 2

    @property
    def dmx2(self) -> float:
        """GEO(3): squared max separation distance (rivet1.F line 139)."""
        return self.d_max ** 2 if self.d_max != 0.0 else _EP15 ** 2


@dataclass
class Rivet:
    """A single two-node rivet connector element.

    Attributes
    ----------
    elem_id : int
        Element user ID (IXRT(4,I) in Fortran).
    n1 : int
        Index of node 1 into the global x/v/a arrays (0-based).
    n2 : int
        Index of node 2 into the global x/v/a arrays (0-based).
    prop : RivetProp
        Associated property data.
    off : float
        Damage counter: starts at 1.0, decremented by 0.1 each step the
        nodes are beyond d_max; set to 0.0 on failure (rivet1.F lines 93-111).
    failed : bool
        True once the rivet has fully failed (OFF <= 0).
    """
    elem_id: int
    n1: int
    n2: int
    prop: RivetProp
    off: float = 1.0
    failed: bool = False


# ---------------------------------------------------------------------------
# Engine step function
# ---------------------------------------------------------------------------


def rivet_step(
    rivets: List[Rivet],
    x: np.ndarray,
    v: np.ndarray,
    vr: np.ndarray,
    a: np.ndarray,
    ar: np.ndarray,
    mass: np.ndarray,
    inertia: np.ndarray,
    dt: float,
) -> List[int]:
    """Execute one explicit time step for all rivets, updating accelerations in-place.

    Implements RIVET1() from ``engine/source/elements/rivet/rivet1.F``.

    Parameters
    ----------
    rivets : list of Rivet
        All rivet elements to process.
    x : (N,3) ndarray
        Current nodal coordinates.
    v : (N,3) ndarray
        Nodal velocities at the half-step (updated in-place by the engine).
    vr : (N,3) ndarray
        Nodal rotational velocities (half-step).
    a : (N,3) ndarray
        Nodal accelerations — **updated in-place** (scattered forces / mass).
    ar : (N,3) ndarray
        Nodal rotational accelerations — updated in-place (IROT==1 path).
    mass : (N,) ndarray
        Nodal lumped masses MS(*).
    inertia : (N,) ndarray
        Nodal lumped rotational inertias IN(*).
    dt : float
        Half time step DT12 used in Fortran (= dt/2 in a central-difference
        scheme).  The caller passes the ACTUAL half-step DT12.

    Returns
    -------
    list of int
        Indices into *rivets* of elements that failed this step.
    """
    dt12m1 = 1.0 / dt if dt > 0.0 else 0.0
    newly_failed: List[int] = []

    for idx, rv in enumerate(rivets):
        if rv.failed:
            continue

        p = rv.prop
        n1, n2 = rv.n1, rv.n2
        off = rv.off

        # ------------------------------------------------------------------
        # Separation check & damage decrement
        # rivet1.F lines 100-112
        # ------------------------------------------------------------------
        dx = x[n2, 0] - x[n1, 0]
        dy = x[n2, 1] - x[n1, 1]
        dz = x[n2, 2] - x[n1, 2]
        dx2 = dx * dx + dy * dy + dz * dz

        if dx2 > p.dmx2:
            off -= _EM01
            if off <= 0.0:
                rv.off = 0.0
                rv.failed = True
                newly_failed.append(idx)
                logger.info("FAILURE OF RIVET %d", rv.elem_id)
                continue
            else:
                rv.off = off

        # ------------------------------------------------------------------
        # Select formulation
        # rivet1.F lines 114-119
        # ------------------------------------------------------------------
        if p.imod == 1:
            _rivet_rigid_body(rv, x, v, vr, a, ar, mass, inertia,
                              dt, dt12m1, dx, dy, dz, dx2, off)
        else:
            _rivet_rigid_link(rv, x, v, vr, a, ar, mass, inertia,
                              dt, dt12m1, dx, dy, dz, dx2, off)

    return newly_failed


# ---------------------------------------------------------------------------
# IMOD==1: Rigid Body formulation
# rivet1.F lines 127-339
# ---------------------------------------------------------------------------


def _rivet_rigid_body(
    rv: Rivet,
    x: np.ndarray, v: np.ndarray, vr: np.ndarray,
    a: np.ndarray, ar: np.ndarray,
    mass: np.ndarray, inertia: np.ndarray,
    dt: float, dt12m1: float,
    dx: float, dy: float, dz: float, dx2: float,
    off: float,
) -> None:
    """Rigid-body (IMOD=1) constraint for one rivet.

    Implements the IMOD==1 branch of RIVET1() (rivet1.F lines 143-338).

    Physics:
    --------
    Mass-weighted center-of-gravity velocity (rivet1.F lines 144-167):
        MASS  = m1 + m2
        XCDG  = (x1*m1 + x2*m2) / MASS
        VX_cm = (VX1*m1 + VX2*m2) / MASS    where  VXi = v_i + a_i*dt

    Force (acceleration) impulse on node 1 (rivet1.F lines 172-175):
        AX = (-a1x + (VX_cm - v1x)*dt12m1) * m1

    This is then projected into normal (AN) and tangential (AT) components,
    scaled by the failure surface ellipse (rivet1.F lines 177-202):
        ALP = OFF / max(sqrt(AN2/FN2 + AT2/FT2), 1.0)
        AX  = ALP * AX

    Acceleration scatter (rivet1.F lines 205-211):
        a[n1] += AX / m1   (action on node 1)
        a[n2] -= AX / m2   (reaction on node 2 — Newton III)

    IROT==1 adds the rotational CG constraint (rivet1.F lines 212-337).
    """
    p = rv.prop
    n1, n2 = rv.n1, rv.n2
    m1 = mass[n1]
    m2 = mass[n2]
    mtot = m1 + m2
    masm1 = 1.0 / mtot

    # CG position (rivet1.F lines 147-149) — used only when IROT==1
    xcdg = (x[n1, 0] * m1 + x[n2, 0] * m2) * masm1
    ycdg = (x[n1, 1] * m1 + x[n2, 1] * m2) * masm1
    zcdg = (x[n1, 2] * m1 + x[n2, 2] * m2) * masm1

    # Half-step velocities: Vi = v + a*dt  (rivet1.F lines 151-156)
    vx1 = v[n1, 0] + a[n1, 0] * dt
    vy1 = v[n1, 1] + a[n1, 1] * dt
    vz1 = v[n1, 2] + a[n1, 2] * dt
    vx2 = v[n2, 0] + a[n2, 0] * dt
    vy2 = v[n2, 1] + a[n2, 1] * dt
    vz2 = v[n2, 2] + a[n2, 2] * dt

    # Momentum-weighted: VMi = Vi * mi  (rivet1.F lines 158-163)
    vmx1 = vx1 * m1;  vmy1 = vy1 * m1;  vmz1 = vz1 * m1
    vmx2 = vx2 * m2;  vmy2 = vy2 * m2;  vmz2 = vz2 * m2

    # CG velocity  (rivet1.F lines 165-167)
    vcm_x = (vmx1 + vmx2) * masm1
    vcm_y = (vmy1 + vmy2) * masm1
    vcm_z = (vmz1 + vmz2) * masm1

    if p.irot == 0:
        # ------------------------------------------------------------------
        # Translation-only (IROT=0): rivet1.F lines 172-211
        # ------------------------------------------------------------------
        ax = (-a[n1, 0] + (vcm_x - v[n1, 0]) * dt12m1) * m1
        ay = (-a[n1, 1] + (vcm_y - v[n1, 1]) * dt12m1) * m1
        az = (-a[n1, 2] + (vcm_z - v[n1, 2]) * dt12m1) * m1

        ax, ay, az, _ = _apply_failure_surface(
            ax, ay, az, dx, dy, dz, dx2, p.fn2, p.ft2, off
        )

        # Scatter (rivet1.F lines 206-211)
        a[n1, 0] += ax / m1
        a[n1, 1] += ay / m1
        a[n1, 2] += az / m1
        a[n2, 0] -= ax / m2
        a[n2, 1] -= ay / m2
        a[n2, 2] -= az / m2

    else:
        # ------------------------------------------------------------------
        # Translation + Rotation (IROT=1): rivet1.F lines 213-338
        # ------------------------------------------------------------------
        # Arm vectors from CG (rivet1.F lines 216-221)
        xx1 = x[n1, 0] - xcdg;  yy1 = x[n1, 1] - ycdg;  zz1 = x[n1, 2] - zcdg
        xx2 = x[n2, 0] - xcdg;  yy2 = x[n2, 1] - ycdg;  zz2 = x[n2, 2] - zcdg

        # Rotational inertias about CG (rivet1.F lines 223-226)
        i1 = (xx1*xx1 + yy1*yy1 + zz1*zz1) * m1 + inertia[n1]
        i2 = (xx2*xx2 + yy2*yy2 + zz2*zz2) * m2 + inertia[n2]
        iner = i1 + i2
        inm1 = 1.0 / iner if iner > _EM15 else 0.0

        # Rotational half-step velocities (rivet1.F lines 228-233)
        vrx1 = vr[n1, 0] + ar[n1, 0] * dt;  vry1 = vr[n1, 1] + ar[n1, 1] * dt;  vrz1 = vr[n1, 2] + ar[n1, 2] * dt
        vrx2 = vr[n2, 0] + ar[n2, 0] * dt;  vry2 = vr[n2, 1] + ar[n2, 1] * dt;  vrz2 = vr[n2, 2] + ar[n2, 2] * dt

        # Angular velocity of CG system (rivet1.F lines 235-240)
        vxx = (vrx1 * inertia[n1] + yy1 * vmz1 - zz1 * vmy1
               + vrx2 * inertia[n2] + yy2 * vmz2 - zz2 * vmy2) * inm1
        vyy = (vry1 * inertia[n1] + zz1 * vmx1 - xx1 * vmz1
               + vry2 * inertia[n2] + zz2 * vmx2 - xx2 * vmz2) * inm1
        vzz = (vrz1 * inertia[n1] + xx1 * vmy1 - yy1 * vmx1
               + vrz2 * inertia[n2] + xx2 * vmy2 - yy2 * vmx2) * inm1

        # Tangential velocity from rigid-body rotation: vt = omega × r
        # (rivet1.F lines 244-249)
        vt1 = (zz1 * vyy - yy1 * vzz,
               xx1 * vzz - zz1 * vxx,
               yy1 * vxx - xx1 * vyy)
        vt2 = (zz2 * vyy - yy2 * vzz,
               xx2 * vzz - zz2 * vxx,
               yy2 * vxx - xx2 * vyy)

        # Force on n2 then overwritten by n1 (rivet1.F lines 251-257 — note
        # the Fortran computes n2 first, then re-assigns AX/AY/AZ with n1)
        ax = (-a[n1, 0] + (vcm_x + vt1[0] - v[n1, 0]) * dt12m1) * m1
        ay = (-a[n1, 1] + (vcm_y + vt1[1] - v[n1, 1]) * dt12m1) * m1
        az = (-a[n1, 2] + (vcm_z + vt1[2] - v[n1, 2]) * dt12m1) * m1

        ax, ay, az, alp = _apply_failure_surface(
            ax, ay, az, dx, dy, dz, dx2, p.fn2, p.ft2, off
        )

        # Centripetal correction terms (rivet1.F lines 289-294)
        half_dt2_dt12m1 = 0.5 * dt * dt * dt12m1  # HALF*DT2*DT12M1 where DT2=dt^2
        da1 = (half_dt2_dt12m1 * (vyy * vt1[2] - vzz * vt1[1]),
               half_dt2_dt12m1 * (vzz * vt1[0] - vxx * vt1[2]),
               half_dt2_dt12m1 * (vxx * vt1[0] - vyy * vt1[1]))
        da2 = (half_dt2_dt12m1 * (vyy * vt2[2] - vzz * vt2[1]),
               half_dt2_dt12m1 * (vzz * vt2[0] - vxx * vt2[2]),
               half_dt2_dt12m1 * (vxx * vt2[0] - vyy * vt2[1]))

        # Acceleration scatter (rivet1.F lines 296-301)
        a[n1, 0] += ax / m1 + da1[0]
        a[n1, 1] += ay / m1 + da1[1]
        a[n1, 2] += az / m1 + da1[2]
        a[n2, 0] -= ax / m2 - da2[0]
        a[n2, 1] -= ay / m2 - da2[1]
        a[n2, 2] -= az / m2 - da2[2]

        # Rotational accelerations (rivet1.F lines 311-326)
        amx = -ar[n1, 0] + (vxx - vr[n1, 0]) * dt12m1
        amy = -ar[n1, 1] + (vyy - vr[n1, 1]) * dt12m1
        amz = -ar[n1, 2] + (vzz - vr[n1, 2]) * dt12m1
        ar[n1, 0] += amx * alp
        ar[n1, 1] += amy * alp
        ar[n1, 2] += amz * alp

        amx = -ar[n2, 0] + (vxx - vr[n2, 0]) * dt12m1
        amy = -ar[n2, 1] + (vyy - vr[n2, 1]) * dt12m1
        amz = -ar[n2, 2] + (vzz - vr[n2, 2]) * dt12m1
        ar[n2, 0] += amx * alp
        ar[n2, 1] += amy * alp
        ar[n2, 2] += amz * alp


# ---------------------------------------------------------------------------
# IMOD==2: Old Rigid Link formulation
# rivet1.F lines 341-399
# ---------------------------------------------------------------------------


def _rivet_rigid_link(
    rv: Rivet,
    x: np.ndarray, v: np.ndarray, vr: np.ndarray,
    a: np.ndarray, ar: np.ndarray,
    mass: np.ndarray, inertia: np.ndarray,
    dt: float, dt12m1: float,
    dx: float, dy: float, dz: float, dx2: float,
    off: float,
) -> None:
    """Old rigid-link (IMOD=2) constraint for one rivet.

    Implements the IMOD==2 branch of RIVET1() (rivet1.F lines 341-399).

    Physics:
    --------
    Reduced mass (rivet1.F line 354):
        XM = m1 * m2 / (m1 + m2)

    Impulse to eliminate relative acceleration (rivet1.F lines 355-357):
        AMX = (a2x - a1x + (v2x - v1x) / dt) * XM

    This impulse is then projected and failure-limited:
        ALP = OFF / max(sqrt(AN2/FN2 + AT2/FT2), 1.0)
        AMX = ALP * AMX

    Scatter (rivet1.F lines 383-388):
        a[n1] += AMX / m1   (accelerate n1 toward n2)
        a[n2] -= AMX / m2   (reaction on n2)

    IROT==1: average rotational accelerations (rivet1.F lines 389-397):
        AR_avg = (AR1*I1 + AR2*I2) / (I1 + I2)
        AR1 = AR2 = AR_avg
    """
    p = rv.prop
    n1, n2 = rv.n1, rv.n2
    m1 = mass[n1]
    m2 = mass[n2]
    xm = m1 * m2 / (m1 + m2)  # rivet1.F line 354

    # Relative acceleration impulse (rivet1.F lines 355-357)
    amx = (a[n2, 0] - a[n1, 0] + (v[n2, 0] - v[n1, 0]) / dt) * xm
    amy = (a[n2, 1] - a[n1, 1] + (v[n2, 1] - v[n1, 1]) / dt) * xm
    amz = (a[n2, 2] - a[n1, 2] + (v[n2, 2] - v[n1, 2]) / dt) * xm

    amx, amy, amz, _ = _apply_failure_surface(
        amx, amy, amz, dx, dy, dz, dx2, p.fn2, p.ft2, off
    )

    # Scatter (rivet1.F lines 383-388)
    a[n1, 0] += amx / m1
    a[n1, 1] += amy / m1
    a[n1, 2] += amz / m1
    a[n2, 0] -= amx / m2
    a[n2, 1] -= amy / m2
    a[n2, 2] -= amz / m2

    # Rotational constraint (rivet1.F lines 389-397)
    if p.irot == 1:
        i1 = inertia[n1]
        i2 = inertia[n2]
        inm1 = 1.0 / (i1 + i2) if (i1 + i2) > _EM15 else 0.0
        for k in range(3):
            ar_avg = (ar[n1, k] * i1 + ar[n2, k] * i2) * inm1
            ar[n1, k] = ar_avg
            ar[n2, k] = ar_avg


# ---------------------------------------------------------------------------
# Shared helper: failure surface projection
# rivet1.F lines 177-202 (IROT=0) and 259-285 (IROT=1)
# ---------------------------------------------------------------------------


def _apply_failure_surface(
    ax: float, ay: float, az: float,
    dx: float, dy: float, dz: float, dx2: float,
    fn2: float, ft2: float, off: float,
) -> tuple:
    """Project the impulse force vector onto the elliptic failure surface.

    Returns the scaled (ax, ay, az, alp).

    When the nodes are coincident (dx2 <= EM15) only the normal criterion
    is applied (tangential contribution AT2=0).

    rivet1.F lines 177-202:
    -----------------------
    IF (DX2 > EM15) THEN
        S  = 1/sqrt(DX2)
        XN = dx*S  (unit normal)
        AN = AX*XN + AY*YN + AZ*ZN   (normal component)
        AN2 = AN**2
        ATi = Ai - AN*XNi            (tangential component)
        AT2 = ATX**2 + ATY**2 + ATZ**2
    ELSE
        AN2 = AX**2 + AY**2 + AZ**2
        AT2 = 0
    ENDIF

    ALP = sqrt(AN2/FN2 + AT2/FT2)
    ALP = OFF / MAX(ALP, 1.0)
    Ai  = ALP * Ai
    """
    if dx2 > _EM15:
        s = 1.0 / math.sqrt(dx2)
        xn = dx * s
        yn = dy * s
        zn = dz * s
        an = ax * xn + ay * yn + az * zn
        an2 = an * an
        atx = ax - an * xn
        aty = ay - an * yn
        atz = az - an * zn
        at2 = atx * atx + aty * aty + atz * atz
    else:
        # Coincident nodes: only normal criterion (rivet1.F lines 193-195)
        an2 = ax * ax + ay * ay + az * az
        at2 = 0.0

    # Elliptic failure surface (rivet1.F lines 198-202)
    alp_raw = math.sqrt(an2 / fn2 + at2 / ft2) if (fn2 > 0 and ft2 > 0) else 0.0
    alp = off / max(alp_raw, 1.0)
    return ax * alp, ay * alp, az * alp, alp
