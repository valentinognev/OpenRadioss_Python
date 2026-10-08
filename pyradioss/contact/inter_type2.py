"""
/INTER/TYPE2 — tied contact (kinematic secondary-to-main gluing).

Fortran origin: ``engine/source/interfaces/int02/`` and ``engine/source/interfaces/interf/`` —

    i2for3.F   transfer of secondary nodal forces to the main segment (I2FOR3, I2FOMO3)
    i2vit3.F   kinematic update of secondary velocities from mains (I2VIT3, I2VIROT3, I2ROT3_27)
    i2curv.F   segment local co-rotating frame (t1, t2, n) for offset secondary nodes
    starter/source/interfaces/inter3d1/i2buc1.F, i2dst3.F, i2tid3.F
               (Starter) projection search: each secondary node onto its
               closest main segment, isoparametric weights + offset

Theory — the constrained (kinematic) tied formulation
-----------------------------------------------------
TYPE2 is not a force model: it is a *constraint*. Each secondary node s is
glued, for the whole run, to the material point of its main segment it was
projected on at time 0:

    x_s(t) = sum_k w_k x_k(t) + offset(t),         sum_k w_k = 1

with w_k the (constant) interpolation weights of the projection and
``offset`` the initial normal/tangential misfit expressed in the segment's
co-rotating local frame (so a spot-weld between two shell mid-surfaces,
which are half a thickness apart, rotates rigidly with the segment).

The explicit implementation is the classic lumped constraint method (the
same one the original OpenRadioss and LS-DYNA constrained tied contacts use):

1. **Force transfer** (i2for3.F): the secondary node's assembled force is
   distributed to the segment corners with the weights,
   f_k += w_k f_s, and removed from the secondary node.
   - Spotflag=0 (Standard translation): translational forces transferred
     according to projection weights w_k;
   - Spotflag=1 (Solid main - I2FOMO3): offset moment (dvec x F_s) plus
     secondary node moment (mint) transferred as an equilibrating force couple
     F_couple = A x r, where A = I^-1 M_tot and I is the segment pseudo-inertia;
   - Spotflag=2 (Shell main - I2MOM3): offset moment (dvec x F_s) plus
     secondary moment transferred directly into the rotational DOFs (mint)
     of the main shell nodes.
2. **Mass transfer** (Starter i2tid3.F): likewise, once, for the lumped masses:
   M_k += w_k m_s. Steps 1+2 make the main nodes carry the secondary's inertia
   and loading; total force and total mass are conserved by construction (sum w_k = 1).
3. **Kinematic update** (i2vit3.F): after the main nodes moved, the
   secondary node is *placed* — not integrated:
   x_s = sum w_k x_k + offset, and its velocity is set to the consistent
   v_s = (x_s^{n+1} - x_s^n) / dt.
   - Spotflag=1 (Solid main - I2VIROT3): angular velocity omega derived from
     main segment angular momentum L = sum r x v and assigned to vr_s, with
     rotational offset velocity omega x (x_new - x0) added to v_s;
   - Spotflag=2 (Shell main - I2ROT3_27): rotational DOFs vr_s interpolated
     from main shell nodes with weights w_k, with offset velocity
     omega x (x_new - x_interp) added to v_s.

Momentum is conserved *exactly*: d/dt(m_s v_s + sum M_k v_k) =
sum_k (M_k + w_k m_s) a_k = sum_k (f_k + w_k f_s) = total applied force.
And the tie does **no work by construction** — it books nothing into the
contact energy, and the global balance closes without a CE term
(asserted by the M4 and M489 tests).

Element deletion (M3<->M4): a tie whose main segment's parent element is
deleted is *released* (the crack must not keep carrying load through the
glue); its transferred mass is handed back so the released node resumes
free flight with its own inertia. A secondary node whose own elements all
died is released the same way.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple
import numpy as np

from ..common.constants import EM20
from ..model.model import Model
from . import tracking
from .inter_type7 import _closest_point_on_triangle
from .stiffness import _segment_areas


def _segment_frames(xs: np.ndarray):
    """Co-rotating orthonormal frame (t1, t2, n) of 3/4-node segments
    (i2curv.F lines 20-65). ``xs`` is (n, 4, 3); triangles repeat node 3.

    Built from the covariant mid-edge vectors r = (x2+x3)-(x1+x4) and
    s = (x3+x4)-(x1+x2): n = r x s (the average normal), t1 = r direction.
    This frame rotates rigidly with the segment, which is exactly what an
    offset tied node must follow.
    """
    if len(xs) == 0:
        return (np.zeros((0, 3), dtype=float),
                np.zeros((0, 3), dtype=float),
                np.zeros((0, 3), dtype=float))
    r = xs[:, 1] + xs[:, 2] - xs[:, 0] - xs[:, 3]
    s = xs[:, 2] + xs[:, 3] - xs[:, 0] - xs[:, 1]
    n = np.cross(r, s)
    norm_n = np.linalg.norm(n, axis=1)
    norm_n_clamped = np.where(norm_n > EM20, norm_n, 1.0)
    n = np.where((norm_n > EM20)[:, None], n / norm_n_clamped[:, None], np.array([0.0, 0.0, 1.0]))

    norm_r = np.linalg.norm(r, axis=1)
    norm_r_clamped = np.where(norm_r > EM20, norm_r, 1.0)
    t1 = np.where((norm_r > EM20)[:, None], r / norm_r_clamped[:, None], np.array([1.0, 0.0, 0.0]))

    t2 = np.cross(n, t1)
    norm_t2 = np.linalg.norm(t2, axis=1)
    deg_t2 = norm_t2 <= EM20
    if np.any(deg_t2):
        alt = np.where(np.abs(n[:, 0:1]) < 0.9, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        t1_cand = np.cross(n, alt)
        norm_t1_cand = np.maximum(np.linalg.norm(t1_cand, axis=1, keepdims=True), EM20)
        t1 = np.where(deg_t2[:, None], t1_cand / norm_t1_cand, t1)
        t2 = np.cross(n, t1)
    norm_t2_clamped = np.maximum(np.linalg.norm(t2, axis=1, keepdims=True), EM20)
    t2 = t2 / norm_t2_clamped
    return t1, t2, n


class ContactType2:
    """One /INTER/TYPE2 tied interface, engine-side."""

    def __init__(self, itf, model: Model, log):
        self.itf = itf
        self.model = model

        surf = model.surfaces.get(itf.surf_id)
        if surf is None:
            log.error(f"/INTER/TYPE2/{itf.id}: main surface {itf.surf_id} not found in model", "TIED INIT")
            self._init_empty()
            return

        segs = surf.segments if surf.segments is not None else np.zeros((0, 4), dtype=np.int64)
        segs = np.asarray(segs, dtype=np.int64)
        if segs.ndim == 2 and segs.shape[1] == 3:
            segs = np.column_stack([segs, segs[:, 2]])
        elif segs.ndim == 2 and segs.shape[1] == 4:
            segs = segs.copy()
            neg = segs[:, 3] < 0
            segs[neg, 3] = segs[neg, 2]

        grp = model.node_groups.get(itf.grnod_id)
        if grp is None or grp.node_idx is None:
            log.error(f"/INTER/TYPE2/{itf.id}: secondary node group {itf.grnod_id} not found in model", "TIED INIT")
            self._init_empty()
            return

        cand = np.asarray(grp.node_idx, dtype=np.int64)
        cand = np.unique(cand[(cand >= 0) & (cand < len(model.mass))])
        if len(cand) == 0 or len(segs) == 0:
            self._init_empty()
            return

        # a node that is itself a corner of the main surface cannot be
        # tied to it (it would be glued to itself); frozen (massless,
        # m = 1e30) nodes are structurally meaningless to tie
        surf_nodes = np.unique(segs)
        bad = np.isin(cand, surf_nodes) | (model.mass[cand] >= 1e29)
        if np.any(bad):
            log.warning(f"/INTER/TYPE2/{itf.id}: {int(bad.sum())} secondary "
                        f"node(s) excluded (main-surface corners or "
                        f"massless)", "TIED INIT")
        cand = cand[~bad]
        if len(cand) == 0:
            self._init_empty()
            return

        # ---- projection search (Starter i2buc1/i2dst3) --------------------
        # closest point of every candidate on every segment, chunked to
        # bound memory; keep the best segment within the search distance
        x0 = model.x0 if getattr(model, "x0", None) is not None and len(model.x0) > 0 else getattr(model, "x", np.zeros((0, 3)))
        area = _segment_areas(x0, segs)
        lc = float(np.sqrt(area.mean())) if len(area) else 1.0
        dsearch = itf.dsearch if itf.dsearch > 0 else lc

        nbest = len(cand)
        best_d = np.full(nbest, np.inf)
        best_seg = np.full(nbest, -1, dtype=np.int64)
        best_w = np.zeros((nbest, 4))
        chunk = max(1, 2 ** 22 // max(nbest, 1))     # ~4M pairs per chunk
        for s0 in range(0, len(segs), chunk):
            sc = segs[s0:s0 + chunk]
            # all (node, segment-of-chunk) pairs
            pi = np.repeat(np.arange(nbest), len(sc))
            sj = np.tile(np.arange(len(sc)), nbest)
            p = x0[cand[pi]]
            dmin = np.full(len(pi), np.inf)
            wq = np.zeros((len(pi), 4))
            for cols in ((0, 1, 2), (0, 2, 3)):
                a, b, c = (x0[sc[sj, cols[0]]], x0[sc[sj, cols[1]]],
                           x0[sc[sj, cols[2]]])
                pt, u, vv, w = _closest_point_on_triangle(p, a, b, c)
                d = np.linalg.norm(p - pt, axis=1)
                better = d < dmin
                dmin = np.where(better, d, dmin)
                wtmp = np.zeros((len(pi), 4))
                wtmp[:, cols[0]], wtmp[:, cols[1]], wtmp[:, cols[2]] = \
                    u, vv, w
                wq[better] = wtmp[better]
            # reduce over the chunk: best segment per node
            dmat = dmin.reshape(nbest, len(sc))
            jbest = dmat.argmin(axis=1)
            dbest = dmat[np.arange(nbest), jbest]
            upd = dbest < best_d
            best_d = np.where(upd, dbest, best_d)
            best_seg[upd] = s0 + jbest[upd]
            best_w[upd] = wq.reshape(nbest, len(sc), 4)[
                np.arange(nbest), jbest][upd]

        found = (best_d <= dsearch) & (best_seg >= 0)
        if np.any(~found):
            log.warning(f"/INTER/TYPE2/{itf.id}: {int((~found).sum())} "
                        f"secondary node(s) farther than the search "
                        f"distance {dsearch:.4g} — left free", "TIED INIT")

        self.snode = cand[found]                       # tied secondary nodes
        self.seg = segs[best_seg[found]]               # (nt, 4) node indices
        self.w = best_w[found]                         # (nt, 4) weights
        seg_rows = best_seg[found]
        self.seg_gtype = (surf.seg_gtype[seg_rows]
                          if surf.seg_gtype is not None
                          else np.zeros(len(seg_rows), dtype="<U8"))
        self.seg_elem = (surf.seg_elem[seg_rows]
                         if surf.seg_elem is not None
                         else np.full(len(seg_rows), -1, dtype=np.int64))

        # initial offset, expressed in the segment's local frame so it
        # co-rotates (see module docstring)
        xs = x0[self.seg]
        t1, t2, n = _segment_frames(xs)
        xc = np.einsum("nk,nkb->nb", self.w, xs)
        dvec = x0[self.snode] - xc
        self.off_loc = np.column_stack([
            np.einsum("nb,nb->n", dvec, t1),
            np.einsum("nb,nb->n", dvec, t2),
            np.einsum("nb,nb->n", dvec, n)])
        off_mag = np.linalg.norm(dvec, axis=1)
        if len(off_mag) and off_mag.max() > 0.5 * lc:
            log.warning(f"/INTER/TYPE2/{itf.id}: largest tie offset "
                        f"{off_mag.max():.4g} exceeds half the segment size "
                        f"— check the projection", "TIED INIT")
        log.info(f"     /INTER/TYPE2/{itf.id}: {len(self.snode)} NODE(S) "
                 f"TIED TO SURFACE {itf.surf_id}")

        # a tied node's motion is fully prescribed by the constraint: an
        # /IMPVEL on it would fight the tie (two kinematic conditions on
        # one DOF — the original Starter errors out on such clashes)
        for imp in model.impvel:
            g = model.node_groups.get(imp.grnod_id)
            if g is not None and g.node_idx is not None and \
                    np.isin(self.snode, g.node_idx).any():
                log.warning(f"/INTER/TYPE2/{itf.id}: /IMPVEL/{imp.id} "
                            f"drives tied secondary node(s) — the tie "
                            f"wins (kinematic condition clash)",
                            "TIED INIT")

        # tracked position of the tied nodes (see enforce) — from the
        # CURRENT coordinates, not x0: on a restart-chained run (M6) the
        # ties resume where the saved model left them (at a fresh start
        # model.x == x0, so nothing changes)
        self.x_prev = model.x[self.snode].copy()
        self.active = np.ones(len(self.snode), dtype=bool)

        # deletion bookkeeping (release, not force filtering)
        self.deletable = tracking.any_deletable(model, self.seg_gtype, sec_nodes=self.snode)
        if self.deletable:
            self.ref_total = tracking.node_reference_counts(
                model, alive_only=False)
            # ties already released by /FAIL deletion in a PREVIOUS run
            # (M6 restart): the deletion state lives in the model, so the
            # release set is reconstructed here instead of persisted
            seg_dead = ~tracking.alive_segment_mask(
                model, self.seg_gtype, self.seg_elem)
            node_dead = ~tracking.tracked_node_mask(
                model, self.ref_total)[self.snode]
            self.active &= ~(seg_dead | node_dead)

    def _init_empty(self) -> None:
        """Initialize empty tied interface."""
        self.snode = np.zeros(0, dtype=np.int64)
        self.seg = np.zeros((0, 4), dtype=np.int64)
        self.w = np.zeros((0, 4), dtype=float)
        self.seg_gtype = np.zeros(0, dtype="<U8")
        self.seg_elem = np.zeros(0, dtype=np.int64)
        self.off_loc = np.zeros((0, 3), dtype=float)
        self.x_prev = np.zeros((0, 3), dtype=float)
        self.active = np.zeros(0, dtype=bool)
        self.deletable = False

    # ------------------------------------------------------------------
    def augment_mass(self, mass_eff: np.ndarray) -> None:
        """Mass transfer M_k += w_k m_s (once, engine init per i2tid3.F).
        ``mass_eff`` is the Engine's EFFECTIVE mass used for accelerations only
        — the physical ``model.mass`` (energies, momentum) is untouched.
        Released ties (reconstructed at init after a chained restart)
        transfer nothing — their nodes fly with their own inertia."""
        act = self.active
        if not np.any(act):
            return
        m_s = self.model.mass[self.snode[act]]
        for k in range(4):
            np.add.at(mass_eff, self.seg[act, k], self.w[act, k] * m_s)

    # ------------------------------------------------------------------
    def _release(self, dead: np.ndarray, mass_eff: np.ndarray,
                 inv_mass_eff: np.ndarray) -> None:
        """Release ties (deletion): hand the transferred mass back and
        deactivate. ``dead`` is a mask over the tie arrays."""
        m_s = self.model.mass[self.snode[dead]]
        seg = self.seg[dead]
        w = self.w[dead]
        for k in range(4):
            np.add.at(mass_eff, seg[:, k], -w[:, k] * m_s)
        touched = np.unique(seg)
        touched = touched[touched >= 0]
        inv_mass_eff[touched] = np.where(mass_eff[touched] > 0.0, 1.0 / np.maximum(mass_eff[touched], 1e-30), 0.0)
        self.active[dead] = False

    # ------------------------------------------------------------------
    def transfer_forces(self, fint: np.ndarray, fext: np.ndarray,
                        fcont: np.ndarray, mint: np.ndarray, x: np.ndarray,
                        mass_eff: np.ndarray, inv_mass_eff: np.ndarray, cycle: int) -> None:
        """Per-cycle step 1 (i2for3.F / I2FOMO3): move the tied nodes' assembled
        forces (internal, external AND contact — a tied node can also be
        a penalty secondary) to their main segments. Also polls the
        deletion release (cheap, only for models that can actually delete
        elements)."""
        if self.deletable and cycle % 8 == 0:      # poll every few cycles
            seg_dead = ~tracking.alive_segment_mask(
                self.model, self.seg_gtype, self.seg_elem)
            node_dead = ~tracking.tracked_node_mask(
                self.model, self.ref_total)[self.snode]
            dead = self.active & (seg_dead | node_dead)
            if np.any(dead):
                self._release(dead, mass_eff, inv_mass_eff)

        act = self.active
        if not np.any(act):
            return
        sn = self.snode[act]
        seg = self.seg[act]
        w = self.w[act]

        sf = getattr(self.itf, "spotflag", 0)
        if sf in (1, 2):
            # M_offset = dvec x F_slave
            xc = np.einsum("nk,nkb->nb", w, x[seg])
            dvec = x[sn] - xc

        for arr in (fint, fext, fcont):
            F = arr[sn]
            if sf in (1, 2):
                # Total moment to transfer = M_slave (if any) + offset moment
                M_tot = np.cross(dvec, F)
                if arr is fint:
                    M_tot += mint[sn]
                    mint[sn] = 0.0

                if sf == 1:
                    # Spotflag 1 (Solid main): moment to force couple (I2FOMO3)
                    xs = x[seg]
                    is_tri = seg[:, 3] == seg[:, 2]
                    w_corner = np.ones((len(sn), 4, 1))
                    w_corner[is_tri, 3] = 0.0
                    counts = np.where(is_tri, 3.0, 4.0)[:, None]
                    x0 = np.sum(xs * w_corner, axis=1) / counts
                    r = xs - x0[:, None, :]
                    r_eff = r * np.sqrt(w_corner)

                    # Pseudo-inertia tensor I (unit mass at each vertex)
                    I_tensor = np.zeros((len(sn), 3, 3))
                    I_tensor[:, 0, 0] = np.sum(r_eff[:, :, 1]**2 + r_eff[:, :, 2]**2, axis=1)
                    I_tensor[:, 1, 1] = np.sum(r_eff[:, :, 0]**2 + r_eff[:, :, 2]**2, axis=1)
                    I_tensor[:, 2, 2] = np.sum(r_eff[:, :, 0]**2 + r_eff[:, :, 1]**2, axis=1)
                    I_tensor[:, 0, 1] = I_tensor[:, 1, 0] = -np.sum(r_eff[:, :, 0] * r_eff[:, :, 1], axis=1)
                    I_tensor[:, 0, 2] = I_tensor[:, 2, 0] = -np.sum(r_eff[:, :, 0] * r_eff[:, :, 2], axis=1)
                    I_tensor[:, 1, 2] = I_tensor[:, 2, 1] = -np.sum(r_eff[:, :, 1] * r_eff[:, :, 2], axis=1)

                    try:
                        I_inv = np.linalg.pinv(I_tensor, rcond=1e-8)
                        A = np.einsum("nij,nj->ni", I_inv, M_tot)
                        F_couple = np.cross(A[:, None, :], r)
                        if np.any(is_tri):
                            F_couple[is_tri, 3] = 0.0
                    except Exception:
                        F_couple = np.zeros_like(r)

                    for k in range(4):
                        np.add.at(arr, seg[:, k], w[:, k, None] * F + F_couple[:, k, :])
                else:
                    # Spotflag 2 (Shell main): moment directly to rotational DOFs (I2MOM3)
                    for k in range(4):
                        np.add.at(arr, seg[:, k], w[:, k, None] * F)
                        np.add.at(mint, seg[:, k], w[:, k, None] * M_tot)
            else:
                for k in range(4):
                    np.add.at(arr, seg[:, k], w[:, k, None] * F)

            arr[sn] = 0.0

    # ------------------------------------------------------------------
    def enforce(self, x: np.ndarray, v: np.ndarray, vr: np.ndarray, dt: float) -> None:
        """Per-cycle step 3 (i2vit3.F / I2VIROT3 / I2ROT3_27): place the tied
        nodes on their segments (weights + co-rotated offset) and set the
        consistent velocity. Called after the main nodes' position update."""
        act = self.active
        if not np.any(act):
            return
        sn = self.snode[act]
        xs = x[self.seg[act]]
        t1, t2, n = _segment_frames(xs)
        loc = self.off_loc[act]
        x_new = (np.einsum("nk,nkb->nb", self.w[act], xs)
                 + loc[:, 0:1] * t1 + loc[:, 1:2] * t2 + loc[:, 2:3] * n)

        sf = getattr(self.itf, "spotflag", 0)
        if dt > EM20:
            v[sn] = (x_new - self.x_prev[act]) / dt

            if sf == 1:
                # Spotflag 1 (Solid main): derive rotational velocity (I2VIROT3)
                is_tri = self.seg[act, 3] == self.seg[act, 2]
                w_corner = np.ones((len(sn), 4, 1))
                w_corner[is_tri, 3] = 0.0
                counts = np.where(is_tri, 3.0, 4.0)[:, None]
                x0 = np.sum(xs * w_corner, axis=1) / counts
                r = xs - x0[:, None, :]
                r_eff = r * np.sqrt(w_corner)
                vs = v[self.seg[act]]
                L = np.sum(np.cross(r, vs) * w_corner, axis=1)

                I_tensor = np.zeros((len(sn), 3, 3))
                I_tensor[:, 0, 0] = np.sum(r_eff[:, :, 1]**2 + r_eff[:, :, 2]**2, axis=1)
                I_tensor[:, 1, 1] = np.sum(r_eff[:, :, 0]**2 + r_eff[:, :, 2]**2, axis=1)
                I_tensor[:, 2, 2] = np.sum(r_eff[:, :, 0]**2 + r_eff[:, :, 1]**2, axis=1)
                I_tensor[:, 0, 1] = I_tensor[:, 1, 0] = -np.sum(r_eff[:, :, 0] * r_eff[:, :, 1], axis=1)
                I_tensor[:, 0, 2] = I_tensor[:, 2, 0] = -np.sum(r_eff[:, :, 0] * r_eff[:, :, 2], axis=1)
                I_tensor[:, 1, 2] = I_tensor[:, 2, 1] = -np.sum(r_eff[:, :, 1] * r_eff[:, :, 2], axis=1)

                try:
                    I_inv = np.linalg.pinv(I_tensor, rcond=1e-8)
                    omega_main = np.einsum("nij,nj->ni", I_inv, L)
                    vr[sn] = omega_main
                except Exception:
                    pass
            elif sf == 2:
                # Spotflag 2 (Shell main): interpolate rotational DOFs (I2ROT3_27)
                vrs = vr[self.seg[act]]
                omega_main = np.einsum("nk,nkb->nb", self.w[act], vrs)
                vr[sn] = omega_main

        x[sn] = x_new
        self.x_prev[act] = x_new

    def compute_thermal_conduction(
        self,
        temp: np.ndarray,
        dt: float,
        kthe: Optional[float] = None,
        theaccfact: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute thermal conduction across the tied contact interface.

        Ported from $OR_SRC/engine/source/interfaces/interf/i2therm.F

        Parameters
        ----------
        temp : np.ndarray
            Nodal temperatures.
        dt : float
            Time step dt.
        kthe : Optional[float]
            Thermal contact conductivity KTHE. If None, uses itf.kthe.
        theaccfact : float
            Thermal acceleration factor (default 1.0).

        Returns
        -------
        fthe : np.ndarray
            Nodal heat increments [J].
        condn : np.ndarray
            Nodal conductance [W/K].
        heat_transferred : float
            Total heat transferred across interface [J].
        """
        from .thermal_contact import thermal_contact_type2

        if kthe is None:
            kthe = getattr(self.itf, "kthe", 0.0) or getattr(self.itf, "cond", 0.0)

        x = getattr(self.model, "x", getattr(self.model, "x0", np.zeros((len(temp), 3))))
        return thermal_contact_type2(
            x=x,
            temp=temp,
            slave_nodes=self.snode,
            master_segs=self.seg,
            weights=self.w,
            kthe=kthe,
            dt=dt,
            theaccfact=theaccfact,
            active_mask=getattr(self, "alive", None),
        )


class LagmulType2:
    """One /INTER/LAGMUL/TYPE2 tied constraint, engine-side.

    Fortran origin: ``engine/source/tools/lagmul/lag_i2main.F`` (I2LAGM, LAG_I2MAIN).
    Enforces kinematic tying between secondary nodes and master surface segments
    (both 4-node quads and 3-node triangles) using the global sparse Lagrange
    multiplier solver.
    """

    def __init__(self, itf: Any, model: Model, log: Any = None):
        self.itf = itf
        self.model = model
        self.log = log if log is not None else getattr(model, "log", None)

        self._init_empty()
        self._resolve_entities()

        if len(self.snode) > 0 and len(self.master_segs) > 0:
            self._project_nodes(self.log)

    def _init_empty(self) -> None:
        """Initialize empty active arrays for safe fallback."""
        self.snode = np.zeros(0, dtype=np.int64)
        self.master_segs = np.zeros((0, 4), dtype=np.int64)
        self.active_snode = np.zeros(0, dtype=np.int64)
        self.active_segs = np.zeros((0, 4), dtype=np.int64)
        self.active_weights = np.zeros((0, 4), dtype=np.float64)
        self.active_dists = np.zeros(0, dtype=np.float64)
        self.active_offsets = np.zeros((0, 3), dtype=np.float64)

    def _resolve_entities(self) -> None:
        """Defensively resolve secondary nodes and master segments."""
        # 1. Secondary nodes (grnod_id or direct overrides)
        cand = None
        for attr in ("secondary_nodes", "snode", "nodes"):
            if hasattr(self.itf, attr) and getattr(self.itf, attr) is not None:
                cand = np.asarray(getattr(self.itf, attr), dtype=np.int64)
                break

        if cand is None:
            grnod_id = getattr(self.itf, "grnod_id", 0)
            grp = None
            if hasattr(self.model, "node_groups") and self.model.node_groups:
                grp = self.model.node_groups.get(grnod_id)
            if grp is not None:
                if getattr(grp, "node_idx", None) is not None:
                    cand = np.asarray(grp.node_idx, dtype=np.int64)
                elif getattr(grp, "node_ids", None) and hasattr(self.model, "node_id_to_idx"):
                    cand = np.array([
                        self.model.node_id_to_idx[nid]
                        for nid in grp.node_ids
                        if nid in self.model.node_id_to_idx
                    ], dtype=np.int64)
                elif getattr(grp, "nodes", None) is not None:
                    cand = np.asarray(grp.nodes, dtype=np.int64)

        if cand is None:
            cand = np.zeros(0, dtype=np.int64)
        else:
            cand = np.atleast_1d(cand).astype(np.int64)

        # 2. Master segments (surf_id or direct overrides)
        segs = None
        for attr in ("master_segments", "segments", "segs"):
            if hasattr(self.itf, attr) and getattr(self.itf, attr) is not None:
                raw_segs = np.asarray(getattr(self.itf, attr), dtype=np.int64)
                if raw_segs.ndim == 2 and raw_segs.shape[1] == 3:
                    segs = np.column_stack([raw_segs, raw_segs[:, 2]])
                elif raw_segs.ndim == 2 and raw_segs.shape[1] >= 4:
                    segs = raw_segs[:, :4]
                break

        if segs is None:
            surf_id = getattr(self.itf, "surf_id", 0)
            surf = None
            if hasattr(self.model, "surfaces") and self.model.surfaces:
                surf = self.model.surfaces.get(surf_id)
            if surf is not None:
                if getattr(surf, "segments", None) is not None:
                    raw_segs = np.asarray(surf.segments, dtype=np.int64)
                    if raw_segs.ndim == 2 and raw_segs.shape[1] == 3:
                        segs = np.column_stack([raw_segs, raw_segs[:, 2]])
                    elif raw_segs.ndim == 2 and raw_segs.shape[1] >= 4:
                        segs = raw_segs[:, :4]
                elif getattr(surf, "seg_nodes", None) is not None and len(surf.seg_nodes) > 0:
                    rows = []
                    node_map = getattr(self.model, "node_id_to_idx", {})
                    for sn in surf.seg_nodes:
                        idx_row = [node_map.get(nid, nid) for nid in sn]
                        if len(idx_row) == 3:
                            rows.append([idx_row[0], idx_row[1], idx_row[2], idx_row[2]])
                        elif len(idx_row) >= 4:
                            rows.append(idx_row[:4])
                    if rows:
                        segs = np.asarray(rows, dtype=np.int64)

        if segs is None:
            segs = np.zeros((0, 4), dtype=np.int64)
        else:
            segs = np.atleast_2d(segs).astype(np.int64)

        # Coordinate bound check
        n_coords = len(self.model.x0) if hasattr(self.model, "x0") and self.model.x0 is not None else 0
        if n_coords > 0:
            valid_cand = (cand >= 0) & (cand < n_coords)
            cand = cand[valid_cand]

            if len(segs) > 0:
                valid_segs = np.all((segs >= 0) & (segs < n_coords), axis=1)
                segs = segs[valid_segs]

        # Exclude secondary nodes that are corners of the master segments
        if len(cand) > 0 and len(segs) > 0:
            surf_nodes = np.unique(segs)
            bad = np.isin(cand, surf_nodes)
            if hasattr(self.model, "mass") and self.model.mass is not None and len(self.model.mass) >= n_coords:
                bad |= (self.model.mass[cand] >= 1e29)
            if np.any(bad) and self.log is not None:
                self.log.warning(
                    f"/INTER/LAGMUL/TYPE2/{getattr(self.itf, 'id', 0)}: {int(bad.sum())} "
                    f"secondary node(s) excluded (master corners or massless)",
                    "LAGMUL TYPE2 INIT"
                )
            cand = cand[~bad]

        self.snode = cand
        self.master_segs = segs

    def _project_nodes(self, log: Any = None) -> None:
        """Project candidate secondary nodes onto master segments within dsearch."""
        cand = self.snode
        segs = self.master_segs
        nbest = len(cand)
        if nbest == 0 or len(segs) == 0:
            return

        x0 = self.model.x0
        area = _segment_areas(x0, segs)
        lc = float(np.sqrt(area.mean())) if len(area) > 0 and area.mean() > 0 else 1.0
        dsearch = getattr(self.itf, "dsearch", 0.0)
        if dsearch <= 0.0:
            dsearch = lc

        best_d = np.full(nbest, np.inf)
        best_seg = np.full(nbest, -1, dtype=np.int64)
        best_w = np.zeros((nbest, 4), dtype=np.float64)

        chunk = max(1, 2 ** 22 // max(nbest, 1))
        for s0 in range(0, len(segs), chunk):
            sc = segs[s0:s0 + chunk]
            pi = np.repeat(np.arange(nbest), len(sc))
            sj = np.tile(np.arange(len(sc)), nbest)
            p = x0[cand[pi]]
            dmin = np.full(len(pi), np.inf)
            wq = np.zeros((len(pi), 4), dtype=np.float64)

            for cols in ((0, 1, 2), (0, 2, 3)):
                a = x0[sc[sj, cols[0]]]
                b = x0[sc[sj, cols[1]]]
                c = x0[sc[sj, cols[2]]]
                pt, u, vv, w = _closest_point_on_triangle(p, a, b, c)
                d = np.linalg.norm(p - pt, axis=1)
                better = d < dmin
                dmin = np.where(better, d, dmin)
                wtmp = np.zeros((len(pi), 4), dtype=np.float64)
                wtmp[:, cols[0]] = u
                wtmp[:, cols[1]] = vv
                wtmp[:, cols[2]] = w
                wq[better] = wtmp[better]

            dmat = dmin.reshape(nbest, len(sc))
            jbest = dmat.argmin(axis=1)
            dbest = dmat[np.arange(nbest), jbest]
            upd = dbest < best_d
            best_d = np.where(upd, dbest, best_d)
            best_seg[upd] = s0 + jbest[upd]
            best_w[upd] = wq.reshape(nbest, len(sc), 4)[np.arange(nbest), jbest][upd]

        found = (best_d <= dsearch) & (best_seg >= 0)
        if np.any(~found) and log is not None:
            log.warning(
                f"/INTER/LAGMUL/TYPE2/{getattr(self.itf, 'id', 0)}: {int((~found).sum())} "
                f"secondary node(s) farther than search distance {dsearch:.4g} — left free",
                "LAGMUL TYPE2 INIT"
            )

        if not np.any(found):
            return

        self.active_snode = cand[found]
        self.active_segs = segs[best_seg[found]]
        raw_w = best_w[found]
        # Normalize weights to enforce sum == 1.0 identically
        w_sums = np.sum(raw_w, axis=1, keepdims=True)
        w_sums = np.where(w_sums > EM20, w_sums, 1.0)
        self.active_weights = raw_w / w_sums
        self.active_dists = best_d[found]

        # Calculate initial physical offsets
        xc = np.einsum("nk,nkb->nb", self.active_weights, x0[self.active_segs])
        self.active_offsets = x0[self.active_snode] - xc

        if log is not None:
            log.info(
                f"     /INTER/LAGMUL/TYPE2/{getattr(self.itf, 'id', 0)}: {len(self.active_snode)} "
                f"NODE(S) TIED TO SURFACE {getattr(self.itf, 'surf_id', 0)}"
            )

    def generate_l_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Generate constraint rows for the global Lagrange multiplier solver.

        Returns (data, nodes, dofs, eq_ids, n_rows).
        For each active tied node, 3 equations (X, Y, Z) are generated.
        Master nodes receive +H_k (with sum H_k = 1.0), secondary node receives -1.0,
        guaranteeing exact linear momentum conservation (sum L_row = 0.0).
        """
        empty_ret = (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            0,
        )
        n = len(self.active_snode)
        if n == 0:
            return empty_ret

        spotflag = getattr(self.itf, "spotflag", 0)
        formulation = getattr(self.itf, "formulation", "isoparametric").lower()

        data = []
        nodes = []
        dofs = []
        eq_ids = []

        if formulation == "rigid" or spotflag == 1:
            # Fortran i2lagm.F rigid body formulation with pseudo-inertia rotation coupling
            x0 = self.model.x0
            n_rows = 0
            for i in range(n):
                snode = int(self.active_snode[i])
                seg = self.active_segs[i]
                nir = 3 if seg[3] == seg[2] else 4
                fact = 1.0 / float(nir)

                xs = x0[seg[:nir]]
                center = np.mean(xs, axis=0)
                r = xs - center
                r_sec = x0[snode] - center

                # Pseudo-inertia tensor components (I2LAGM lines 201-285)
                xx = float(np.sum(r[:, 0] ** 2))
                yy = float(np.sum(r[:, 1] ** 2))
                zz = float(np.sum(r[:, 2] ** 2))
                xy = float(np.sum(r[:, 0] * r[:, 1]))
                yz = float(np.sum(r[:, 1] * r[:, 2]))
                zx = float(np.sum(r[:, 2] * r[:, 0]))

                xxx = yy + zz
                yyy = zz + xx
                zzz = xx + yy
                xy2 = xy * xy
                yz2 = yz * yz
                zx2 = zx * zx

                det_val = (
                    xxx * yyy * zzz
                    - xxx * yz2
                    - yyy * zx2
                    - zzz * xy2
                    - 2.0 * xy * yz * zx
                )

                if det_val > 1e-30:
                    det_inv = 1.0 / det_val
                    b1 = zzz * yyy - yz2
                    b2 = xxx * zzz - zx2
                    b3 = yyy * xxx - xy2
                    c1 = xxx * yz + zx * xy
                    c2 = yyy * zx + xy * yz
                    c3 = zzz * xy + yz * zx

                    # Equation 1: Vx (ic = 1)
                    eq_x = n_rows
                    n_rows += 1
                    for jj in range(nir):
                        rx, ry, rz = r[jj]
                        # DOF 0 (X)
                        val_x = fact + det_inv * r_sec[2] * (b2 * rz - c1 * ry) - det_inv * r_sec[1] * (c1 * rz - b3 * ry)
                        data.append(val_x)
                        nodes.append(int(seg[jj]))
                        dofs.append(0)
                        eq_ids.append(eq_x)
                        # DOF 1 (Y)
                        val_y = det_inv * r_sec[2] * (c1 * rx - c3 * rz) - det_inv * r_sec[1] * (b3 * rx - c2 * rz)
                        data.append(val_y)
                        nodes.append(int(seg[jj]))
                        dofs.append(1)
                        eq_ids.append(eq_x)
                        # DOF 2 (Z)
                        val_z = det_inv * r_sec[2] * (c3 * ry - b2 * rx) - det_inv * r_sec[1] * (c2 * ry - c1 * rx)
                        data.append(val_z)
                        nodes.append(int(seg[jj]))
                        dofs.append(2)
                        eq_ids.append(eq_x)
                    # Secondary node (-1.0 on X)
                    data.append(-1.0)
                    nodes.append(snode)
                    dofs.append(0)
                    eq_ids.append(eq_x)

                    # Equation 2: Vy (ic = 2)
                    eq_y = n_rows
                    n_rows += 1
                    for jj in range(nir):
                        rx, ry, rz = r[jj]
                        # DOF 0 (X)
                        val_x = det_inv * r_sec[0] * (c1 * rz - b3 * ry) - det_inv * r_sec[2] * (c3 * rz - c2 * ry)
                        data.append(val_x)
                        nodes.append(int(seg[jj]))
                        dofs.append(0)
                        eq_ids.append(eq_y)
                        # DOF 1 (Y)
                        val_y = fact + det_inv * r_sec[0] * (b3 * rx - c2 * rz) - det_inv * r_sec[2] * (c2 * rx - b1 * rz)
                        data.append(val_y)
                        nodes.append(int(seg[jj]))
                        dofs.append(1)
                        eq_ids.append(eq_y)
                        # DOF 2 (Z)
                        val_z = det_inv * r_sec[0] * (c2 * ry - c1 * rx) - det_inv * r_sec[2] * (b1 * ry - c3 * rx)
                        data.append(val_z)
                        nodes.append(int(seg[jj]))
                        dofs.append(2)
                        eq_ids.append(eq_y)
                    # Secondary node (-1.0 on Y)
                    data.append(-1.0)
                    nodes.append(snode)
                    dofs.append(1)
                    eq_ids.append(eq_y)

                    # Equation 3: Vz (ic = 3)
                    eq_z = n_rows
                    n_rows += 1
                    for jj in range(nir):
                        rx, ry, rz = r[jj]
                        # DOF 0 (X)
                        val_x = det_inv * r_sec[1] * (c3 * rz - c2 * ry) - det_inv * r_sec[0] * (b2 * rz - c1 * ry)
                        data.append(val_x)
                        nodes.append(int(seg[jj]))
                        dofs.append(0)
                        eq_ids.append(eq_z)
                        # DOF 1 (Y)
                        val_y = det_inv * r_sec[1] * (c2 * rx - b1 * rz) - det_inv * r_sec[0] * (c1 * rx - c3 * rz)
                        data.append(val_y)
                        nodes.append(int(seg[jj]))
                        dofs.append(1)
                        eq_ids.append(eq_z)
                        # DOF 2 (Z)
                        val_z = fact + det_inv * r_sec[1] * (b1 * ry - c3 * rx) - det_inv * r_sec[0] * (c3 * ry - b2 * rx)
                        data.append(val_z)
                        nodes.append(int(seg[jj]))
                        dofs.append(2)
                        eq_ids.append(eq_z)
                    # Secondary node (-1.0 on Z)
                    data.append(-1.0)
                    nodes.append(snode)
                    dofs.append(2)
                    eq_ids.append(eq_z)
                else:
                    # Degenerate inertia fallback to isoparametric
                    w = self.active_weights[i]
                    if nir == 3:
                        seg_w = [float(w[0]), float(w[1]), float(w[2] + w[3])]
                    else:
                        seg_w = [float(w[k]) for k in range(4)]
                    for dof in range(3):
                        eq_id = n_rows
                        n_rows += 1
                        for k in range(nir):
                            data.append(seg_w[k])
                            nodes.append(int(seg[k]))
                            dofs.append(dof)
                            eq_ids.append(eq_id)
                        data.append(-1.0)
                        nodes.append(snode)
                        dofs.append(dof)
                        eq_ids.append(eq_id)
        else:
            # Standard Isoparametric Tied Formulation (sum_k H_k v_k - v_s = 0)
            n_rows = 0
            for i in range(n):
                snode = int(self.active_snode[i])
                seg = self.active_segs[i]
                w = self.active_weights[i]

                # If segment is triangle (seg[3] == seg[2]), combine weights on node 2
                if seg[3] == seg[2]:
                    seg_nodes = [int(seg[0]), int(seg[1]), int(seg[2])]
                    seg_w = [float(w[0]), float(w[1]), float(w[2] + w[3])]
                else:
                    seg_nodes = [int(seg[k]) for k in range(4)]
                    seg_w = [float(w[k]) for k in range(4)]

                for dof in range(3):
                    eq_id = n_rows
                    n_rows += 1

                    # Master nodes (+H_k)
                    for k in range(len(seg_nodes)):
                        wk = seg_w[k]
                        if abs(wk) > 1e-15:
                            data.append(wk)
                            nodes.append(seg_nodes[k])
                            dofs.append(dof)
                            eq_ids.append(eq_id)

                    # Secondary node (-1.0)
                    data.append(-1.0)
                    nodes.append(snode)
                    dofs.append(dof)
                    eq_ids.append(eq_id)

        if n_rows == 0:
            return empty_ret

        return (
            np.asarray(data, dtype=np.float64),
            np.asarray(nodes, dtype=np.int64),
            np.asarray(dofs, dtype=np.int64),
            np.asarray(eq_ids, dtype=np.int64),
            n_rows,
        )

    def compute_thermal_conduction(
        self,
        temp: np.ndarray,
        dt: float,
        kthe: Optional[float] = None,
        theaccfact: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute thermal conduction across the tied contact interface.

        Ported from $OR_SRC/engine/source/interfaces/interf/i2therm.F

        Parameters
        ----------
        temp : np.ndarray
            Nodal temperatures.
        dt : float
            Time step dt.
        kthe : Optional[float]
            Thermal contact conductivity KTHE. If None, uses itf.kthe.
        theaccfact : float
            Thermal acceleration factor (default 1.0).

        Returns
        -------
        fthe : np.ndarray
            Nodal heat increments [J].
        condn : np.ndarray
            Nodal conductance [W/K].
        heat_transferred : float
            Total heat transferred across interface [J].
        """
        from .thermal_contact import thermal_contact_type2

        if kthe is None:
            kthe = getattr(self.itf, "kthe", 0.0) or getattr(self.itf, "cond", 0.0)

        x = getattr(self.model, "x", getattr(self.model, "x0", np.zeros((len(temp), 3))))
        return thermal_contact_type2(
            x=x,
            temp=temp,
            slave_nodes=self.snode,
            master_segs=self.seg,
            weights=self.w,
            kthe=kthe,
            dt=dt,
            theaccfact=theaccfact,
            active_mask=self.alive,
        )


