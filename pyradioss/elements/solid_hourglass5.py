"""Task P2.5 — the 5-hourglass-mode control of the 8-node one-point solid.

Fortran origin (``$OR_SRC`` = the OpenRadioss tree)::

    engine/source/elements/solid/solide/s_hg5.F        the control itself
    engine/source/elements/solid/solide/shvis3.F       its viscous sibling
                                                        (the port's Isolid=1
                                                        default hourglass)
    engine/source/elements/solid/solidez/shour_ctl.F90 the distortion-control
                                                        coefficient ladder
    engine/source/elements/solid/solide/fderi3.F       the PX*H hourglass
                                                        projection
    starter/source/properties/solid/hm_read_prop14.F   the property read that
                                                        selects it (IHBE == 5
                                                        -> IINT = 3)

What selects it
---------------
``/PROP/SOLID Isolid=5``.  ``hm_read_prop14.F:233-235`` maps that formulation
flag to ``IINT = 3``, and ``sforc3.F:1232`` dispatches on it::

    IF (IINT==3) THEN
       CALL S_HG5(...)
    ELSE
       CALL SHVIS3(...)
    END IF

So ``Isolid=5`` is not "more of" the default viscous hourglass — it is a
*different* hourglass control, with its own coefficient ladder (``SFAC``,
``F_ET``, ``F_CL``), its own hourglass stress ``FHOUR`` and its own
trapezoidal energy booking.  :func:`enabled` is that dispatch, read off the
property the deck already carries, so a deck that does not ask for it is
untouched — which is also the default here, since nothing in the engine calls
this module yet.

The mode matrix is NOT recomputed here
-------------------------------------
``solid_hexa8`` already owns it: ``_hg_operators()`` returns ``gamma`` — the
Flanagan–Belytschko shape vectors ``_H`` orthogonalized against the linear
field, ``gamma[a,i] = _H[a,i] - (sum_j _H[a,j] x_j .) gradN_i``, which is the
port's ``PX*H`` hourglass matrix (``s_hg5.F`` lines 211-278 build the same
object column by column from the shape gradients).  This module *reads* it
through ``_hg_operators`` and never rebuilds it, per the P2.5 contract.

That matrix's rows are NOT in ``s_hg5.F``'s column order.  ``_H`` is the
Flanagan–Belytschko table; ``s_hg5`` numbers its columns by the sign product of
the coordinates the mode alternates in.  Reading the base patterns off the
Fortran G blocks (``s_hg5.F`` lines 212, 233, 253, 274) against the node sign
pattern ``_XI`` and picking ``_H``'s row with the same pattern (negated where
the two disagree) gives::

    Fortran column  base pattern     = sign product   _H row   sign
    1  (s_hg5.F:212)  1 -1 1 -1 1 -1 1 -1           x*y       2      +1
    2  (s_hg5.F:233)  1  1 -1 -1 -1 -1 1 1          y*z       0      +1
    3  (s_hg5.F:253)  1 -1 -1 1 -1 1 1 -1           x*z       1      +1
    4  (s_hg5.F:274)  1 -1 1 -1 -1 1 -1 1          -x*y*z    3      -1

(``_H[3]`` is ``+x*y*z``, hence the ``-1``; the column header names the sign
product of the *Fortran* pattern.)  So the public mode names are
``"xy" | "yz" | "zx" | "xyz"``, and :data:`_MODE_ROW` / :data:`_MODE_SIGN` are
that table.  Fortran mode ``n`` is also accepted as the integer ``n``.

One asymmetry, and it is upstream's
-----------------------------------
``s_hg5.F``'s fourth column (``HGX4`` and friends) carries **no** ``PX*H``
correction at all — it is the raw trilinear-warping pattern — while columns
1-3 carry one.  ``_H`` row 3 is that same raw pattern, so mode 4 uses it with
the sign that reproduces upstream's *base* column exactly, but the port's
``gamma`` carries its own orthogonalization on all four rows, so mode 4's
correction term is the port's where upstream has none.  Every other element of
the routine is common to all four modes.

The energy is the point
-----------------------
The M36 finding was a control that produced force but booked no work: the
force test passed and the global balance leaked.  ``s_hg5.F`` books it twice —
once against the hourglass stress *before* it is advanced and once *after*
(lines 297-305 and 371-379), each with the ``DT05 = dt/2`` of a trapezoid,
and multiplies ``FHOUR`` by ``OFF`` in between (line 307).  Both halves are
transcribed here, and the result is booked into BOTH ledgers the port keeps:
the per-element ``state["ehour"]`` that the energy-balance tests read, and
``state["evis"][8]`` — slot 8 being the hourglass row of the part energy
table (``PARTSAV(8,MX)``, ``s_hg5.F:393``).  A control that cannot book is a
control that leaks; :func:`apply` books unconditionally when it is enabled.

``rot``
------
``s_hg5.F`` rotates nothing: the port's ``Isolid=1`` solid carries its stress
in the global basis, so ``gamma`` (built from global gradients) and ``FHOUR``
are already global.  ``rot`` is accepted for signature symmetry with the other
solid kernels and is *rejected* when it is not the identity, rather than
silently inventing a rotated force upstream never computes.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20, EP30
from ..common.fastmath import scatter_add3
from . import solid_hexa8

ONE = 1.0
#: Fortran ``ZEP5`` (0.5), ``FOURTH`` (0.25), ``THIRD``, ``FOUR_OVER_3`` and the
#: ``SFAC`` prefactor of ``s_hg5.F:169``.
ZEP5 = 0.5
FOURTH = 0.25
THIRD = 1.0 / 3.0
FOUR_OVER_3 = 4.0 / 3.0
SFAC_COEF = 0.038
TWO_THIRD = 2.0 / 3.0
ONE_THIRD = 1.0 / 3.0

#: The hourglass mode names, in ``s_hg5.F`` column order.
MODES = ("xy", "yz", "zx", "xyz")

#: Fortran column -> row of ``solid_hexa8._H`` (see the module docstring).
_MODE_ROW = {"xy": 2, "yz": 0, "zx": 1, "xyz": 3}
_MODE_SIGN = {"xy": 1.0, "yz": 1.0, "zx": 1.0, "xyz": -1.0}

#: ``EVIS`` slot of the hourglass ledger (``PARTSAV(8,*)`` in ``s_hg5.F:393``).
EVIS_HOURGLASS = 8
#: Length of the ledger this module keeps in ``state["evis"]``.
EVIS_LEN = 16


# ----------------------------------------------------------------------------
# property / state helpers
# ----------------------------------------------------------------------------
def enabled(group) -> bool:
    """True when the group's /PROP/SOLID asks for this control (``Isolid=5``).

    ``hm_read_prop14.F:233-235`` turns the formulation flag 5 into ``IINT = 3``
    and ``sforc3.F:1232`` turns ``IINT == 3`` into ``S_HG5``.  Anything else —
    including every deck that never mentions ``Isolid`` — is off, and every
    other property leaves the default ``SHVIS3`` path alone.
    """
    for _sl, _mat, prop in group.state.get("slices", ()):
        if int(getattr(prop, "params", {}).get("isolid", 0) or 0) == 5:
            return True
    return False


def _pm(mat, index: int) -> float:
    """One isotropic entry of the Fortran ``PM`` row for a port material.

    ``PM(20)=E``, ``PM(21)=nu``, ``PM(22)=G``, ``PM(24)=E/(1-nu^2)`` (the
    plane-stress modulus the starter fills at ``hm_read_mat.F90:1489``) and
    ``PM(32)=K`` — the bulk modulus, which is what makes ``s_hg5``'s
    ``LAMG = C1 + 4*G0/3`` the P-wave modulus that ``LAMGT = c^2 rho`` also is.
    """
    e = float(getattr(mat, "E", 0.0))
    nu = float(getattr(mat, "nu", 0.0))
    if index == 20:
        return e
    if index == 21:
        return nu
    if index == 22:
        return float(getattr(mat, "G", 0.0))
    if index == 24:
        d = 1.0 - nu * nu
        return e / d if abs(d) > EM20 else EP30
    if index == 32:
        return float(getattr(mat, "K", 0.0))
    raise ValueError(f"solid_hourglass5: no isotropic PM({index}) mapping")


def _law(mat) -> int:
    """The material law number (``MTN``), normalising the string spellings."""
    law = getattr(mat, "law", 1)
    if isinstance(law, (int, np.integer)):
        return int(law)
    digits = "".join(ch for ch in str(law) if ch.isdigit())
    return int(digits) if digits else 0


def _state_arrays(group):
    """The two persistent arrays ``s_hg5.F`` keeps across cycles.

    ``FHOUR`` (the hourglass stress, ``INTENT(INOUT)`` at ``s_hg5.F:46``) is
    absent from the port's element state, and the hourglass ledger slot is
    absent too, so both are created on first use and kept thereafter.  Nothing
    else in the kernel is touched.
    """
    st = group.state
    n = group.n
    fhour = st.get("hg5_fhour")
    if fhour is None or len(fhour) != n:
        fhour = np.zeros((n, 3, 4))
        st["hg5_fhour"] = fhour
    evis = st.get("evis")
    if evis is None or len(evis) != EVIS_LEN:
        evis = np.zeros(EVIS_LEN)
        st["evis"] = evis
    return fhour, evis


def _kinematics(group, x, v):
    """``(x, v)`` defaults from the model ``init_group`` hung off the group."""
    model = getattr(group, "_model", None)
    if x is None:
        if model is None or getattr(model, "x", None) is None:
            raise ValueError("solid_hourglass5: no coordinates; pass x=")
        x = model.x
    if v is None:
        if model is None or getattr(model, "v", None) is None:
            raise ValueError("solid_hourglass5: no velocities; pass v=")
        v = model.v
    return np.asarray(x, dtype=float), np.asarray(v, dtype=float)


def _check_rot(rot, n: int) -> None:
    """``s_hg5.F`` has no element frame; reject a rotation it never applies."""
    if rot is None:
        return
    r = np.asarray(rot, dtype=float)
    if r.shape == (3, 3):
        r = np.broadcast_to(r, (n, 3, 3))
    if r.shape != (n, 3, 3) or not np.allclose(r, np.eye(3), rtol=0.0, atol=1e-12):
        raise ValueError(
            "solid_hourglass5: s_hg5.F rotates nothing — the hourglass mode "
            "matrix is built from global gradients, so a non-identity rot "
            "would produce a force upstream never computes"
        )


def _resolve_mode(hg_mode):
    """``hg_mode`` -> ``(rows, signs)``; ``None`` selects all four columns."""
    if hg_mode is None:
        return [0, 1, 2, 3], [1.0, 1.0, 1.0, -1.0]
    if isinstance(hg_mode, (int, np.integer)) and not isinstance(hg_mode, bool):
        if not 1 <= int(hg_mode) <= 4:
            raise ValueError(f"solid_hourglass5: Fortran hourglass mode {hg_mode} not in 1..4")
        hg_mode = MODES[int(hg_mode) - 1]
    if not isinstance(hg_mode, str):
        names = list(hg_mode)
        for name in names:
            if name not in _MODE_ROW:
                raise ValueError(
                    f"solid_hourglass5: unknown hourglass mode {name!r}; expected one of {MODES}")
        return [_MODE_ROW[m] for m in names], [_MODE_SIGN[m] for m in names]
    key = hg_mode.strip().lower()
    if key not in _MODE_ROW:
        raise ValueError(f"solid_hourglass5: unknown hourglass mode {hg_mode!r}; expected one of {MODES}")
    return [_MODE_ROW[key]], [_MODE_SIGN[key]]


# ----------------------------------------------------------------------------
# the coefficient ladder (s_hg5.F:128-209; shour_ctl.F90:132-203)
# ----------------------------------------------------------------------------
def _coefficients(group, dt, fhour, vol, rho):
    """``(fcl, sfac)`` for the whole group — ``s_hg5.F:128-209``.

    Transcribed branch for branch from the ``SELECT CASE (MTN)`` ladder, the
    material-specific stiffening blocks that follow it and the
    ``IF (ISCTL==0) F_ET=ONE`` reset.  Upstream reads ``MX = MAT(1)`` — one
    material per group — while the port slices a group by material, so the
    ladder is evaluated per slice, which is the same thing for the single-
    material groups the Fortran sees.
    """
    st = group.state
    n = group.n
    nu = np.zeros(n)
    g0 = np.zeros(n)
    c1 = np.zeros(n)
    e0 = np.zeros(n)
    e0_plane = np.zeros(n)
    law = np.zeros(n, dtype=np.int64)
    qh = np.zeros(n)
    isctl = np.zeros(n)
    cxx = np.zeros(n)

    for sl, mat, prop in st["slices"]:
        params = getattr(prop, "params", {})
        nu[sl] = _pm(mat, 21)
        g0[sl] = _pm(mat, 22)
        c1[sl] = _pm(mat, 32)
        e0[sl] = _pm(mat, 20)
        e0_plane[sl] = _pm(mat, 24)
        law[sl] = _law(mat)
        qh[sl] = float(params.get("h", 0.1))
        isctl[sl] = float(params.get("icontrol", 0) or 0)
        if hasattr(mat, "sound_speed_solid"):
            cxx[sl] = mat.sound_speed_solid()
        else:
            cxx[sl] = np.sqrt(max(_pm(mat, 32) + 4.0 * _pm(mat, 22) / 3.0, 0.0)
                              / max(float(getattr(mat, "rho0", 0.0)), EM20))

    # LAMGT = CXX**2 * RHO                                          (:136)
    lamgt = cxx * cxx * rho

    # IF (ISCTL>0 .AND. NU>0.48999) QH = ZEP5*QH                   (:135)
    qh = np.where((isctl > 0.0) & (nu > 0.48999), ZEP5 * qh, qh)

    f_et = np.ones(n)
    f_gt = np.ones(n)
    lamg = c1 + FOUR_OVER_3 * g0                                   # DEFAULT arm

    is70 = law == 70
    is4269 = np.isin(law, (42, 69))
    is62 = law == 62
    is88 = law == 88
    is90 = law == 90

    # CASE (70): E0 = PM(24)                                      (:140-141)
    lamg = np.where(is70, THIRD * (e0_plane / (1.0 - 2.0 * nu) + 2.0 * e0_plane / (1.0 + nu)), lamg)
    # CASE (42,69): C1 = E0/3/(1-2NU) ; G0 = G0/2                 (:142-147)
    c1_4269 = THIRD * e0 / (1.0 - 2.0 * nu)
    g0_4269 = ZEP5 * g0
    lamg = np.where(is4269, c1_4269 + FOUR_OVER_3 * g0_4269, lamg)
    f_gt = np.where(is4269, np.maximum(ONE, (lamgt - c1_4269) / np.maximum(g0_4269, EM20) / FOUR_OVER_3), f_gt)
    # CASE (1)                                                     (:148-149)
    # CASE (62): G0 = G0/2                                         (:150-153)
    lamg = np.where(is62, c1 + FOUR_OVER_3 * (ZEP5 * g0), lamg)
    # CASE (88): C1 = E0/3/(1-2NU)                                 (:154-157)
    lamg = np.where(is88, (THIRD * e0 / (1.0 - 2.0 * nu)) + FOUR_OVER_3 * g0, lamg)
    # CASE (90): IF (QH==ONE) QH=FOURTH                             (:158-161)
    qh = np.where(is90 & (qh == ONE), FOURTH, qh)
    lamg = np.where(is90, c1 + FOUR_OVER_3 * g0, lamg)

    # CASE (42,69)/(62)/(88) and the DEFAULT arm all set
    # F_ET = MAX(ONE, LAMGT/LAMG); CASE (90) uses the QH-weighted ratio;
    # CASE (70) and CASE (1) leave the ONE set at line 137 untouched.
    lam_safe = np.maximum(lamg, EM20)
    ratio = np.maximum(ONE, lamgt / lam_safe)
    other = ~np.isin(law, (70, 42, 69, 1, 62, 88, 90))
    f_et = np.where(is4269 | is62 | is88 | other, ratio, ONE)
    f_et = np.where(is90, np.maximum(ONE, qh * lamgt / lam_safe), f_et)

    f_sti = np.maximum(ONE, qh)                                     # (:168)
    sfac = SFAC_COEF * qh * lamg                                    # (:169)

    # special case for stiffing                                     (:170-186)
    sf1_62 = np.minimum(10.0, f_et)
    sf1_62 = np.where(sf1_62 > 2.0, 10.0, sf1_62)
    bump = np.where(is62, sf1_62, ONE)
    f_et = f_et * bump
    f_sti = f_sti * bump
    bump2 = np.where(is4269, np.where(f_et > ONE, np.minimum(4.0, f_gt), ONE), ONE)
    f_et = f_et * bump2
    f_sti = f_sti * bump2

    # law 1 / 62 raise the stiffness only after some hourglass strain (:187-202)
    sel = np.isin(law, (1, 62))
    if np.any(sel):
        s_max = np.max(np.abs(fhour), axis=(1, 2))
        fac1 = sfac * vol ** TWO_THIRD
        e_max = s_max / np.maximum(fac1, EM20)
        sf1 = np.where(sel, np.clip(2500.0 * e_max, ONE, 10.0), ONE)
        f_et = f_et * sf1
        f_sti = f_sti * sf1

    # IF (ISCTL==0) F_ET(1:NEL)=ONE                               (:204)
    f_et = np.where(isctl > 0.0, f_et, ONE)

    # CAQ = SFAC*DT1*OFF ; FCL = F_ET*CAQ*VOL**(1/3)               (:205-209)
    caq = sfac * dt * st["off"]
    fcl = f_et * caq * vol ** ONE_THIRD
    return fcl, sfac


# ----------------------------------------------------------------------------
# the control (s_hg5.F:205-384)
# ----------------------------------------------------------------------------
def apply(group, hg_mode=None, rot=None, *, x=None, v=None, dt=1.0, fint=None):
    """One ``S_HG5`` call: ``(f_hg, ehour)`` — the force and the energy booked.

    This is the single-cycle entry point; :func:`hg5_forces` and
    :func:`hg5_energy` are the two views of it the P2.5 contract asks for, and
    both book the ledger.  Call one of them per cycle — calling both books the
    work twice, exactly as calling ``S_HG5`` twice in one cycle would.

    Returns ``(f_hg, ehour)``: the ``(n, 8, 3)`` nodal force in the port's
    internal-force sign (addable straight into ``fint``, or to the ``fe`` of
    ``solid_hexa8._post``) and the total dissipation booked, which is ``>= 0``.
    When the group is not enabled, or is empty, both are zero and nothing is
    booked.
    """
    n = group.n
    if n == 0 or len(group.conn) == 0:
        return np.zeros((0, 8, 3)), 0.0
    if not enabled(group):
        return np.zeros((n, 8, 3)), 0.0

    rows, signs = _resolve_mode(hg_mode)
    x, v = _kinematics(group, x, v)
    _check_rot(rot, n)
    fhour, evis = _state_arrays(group)

    # the hourglass mode matrix solid_hexa8 already builds — read, never rebuilt
    conn, gamma, _gg, _k_hg, _k_stiff = solid_hexa8._hg_operators(group, x)
    xe = x[conn]
    _dndx, vol_raw = solid_hexa8._geometry(xe)
    vol = np.maximum(vol_raw, EM20)
    g = np.stack([gamma[:, a, :] * signs[k] for k, a in enumerate(rows)], axis=1)  # (n, nm, 8)

    rho = group.state["mass"] / vol
    fcl, _sfac = _coefficients(group, dt, fhour, vol, rho)

    # modal velocities HGX/HGY/HGZ                                (:211-278)
    hg = np.einsum("nmi,nid->nmd", g, v[conn]).transpose(0, 2, 1)   # (n, 3, nm)

    # EHOU, first (trapezoid) half, against FHOUR as it stands     (:297-305)
    dt05 = 0.5 * dt
    fh = fhour[:, :, rows]                                         # (n, 3, nm)
    ehour = dt05 * np.einsum("ndm,ndm->n", fh, hg)

    # FHOUR = FHOUR*OFF ; FHOUR += FCL*HG                          (:306-324)
    off = group.state["off"]
    fhour *= off[:, None, None]
    fh = fhour[:, :, rows] + fcl[:, None, None] * hg
    fhour[:, :, rows] = fh

    # EHOU, second half                                           (:371-379)
    ehour = ehour + dt05 * np.einsum("ndm,ndm->n", fh, hg)

    # nodal forces F1i = -sum_a G[a,i] * FHOUR[x,a]               (:342-368)
    f_hg = -np.einsum("nmi,ndm->nid", g, fh)

    # book: per-element (what the balance tests read) and the ledger slot
    group.state["ehour"] += ehour
    total = float(ehour.sum())
    evis[EVIS_HOURGLASS] += total

    if fint is not None:
        scatter_add3(fint, conn.reshape(-1), f_hg.reshape(-1, 3),
                     group.state.get("color_indices"), group.state.get("color_offsets"))
    return f_hg, total


def hg5_forces(group, hg_mode="zx", rot=None, *, x=None, v=None, dt=1.0, fint=None):
    """``hg5_forces(group, hg_mode, rot) -> np.ndarray`` (the P2.5 interface).

    The ``(n, 8, 3)`` hourglass force of ``S_HG5`` for the named mode (all four
    when ``hg_mode`` is ``None``), added into ``fint`` when one is passed.  A
    zero-energy (hourglass) velocity field produces a NONZERO force here — that
    is the control's whole job — and the work it does is booked to the
    hourglass ledger, not dropped.
    """
    return apply(group, hg_mode, rot, x=x, v=v, dt=dt, fint=fint)[0]


def hg5_energy(group, hg_mode=None, rot=None, *, x=None, v=None, dt=1.0) -> float:
    """``hg5_energy(group, ...) -> float`` — the work booked to ``EVIS(8)``.

    The trapezoidal booking of ``s_hg5.F:297-384``: half against the hourglass
    stress before it is advanced, half after.  It is ``>= 0`` for a viscous
    control and lands in ``state["evis"][8]`` as well as in the per-element
    ``state["ehour"]``.
    """
    return apply(group, hg_mode, rot, x=x, v=v, dt=dt)[1]