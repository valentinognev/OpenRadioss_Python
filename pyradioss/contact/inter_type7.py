"""
/INTER/TYPE7 — penalty node-to-surface contact, full options (M4).

Fortran origin: ``engine/source/interfaces/`` — the cycle path is

    intsort/i7main_tri.F   candidate search bookkeeping (re-sort when needed)
    intsort/i7buce.F       bucket (voxel) search: nodes near segments
    int07/i7dst3.F         exact node-segment distance/projection
    int07/i7for3.F         penalty force + friction, scatter to nodes
    inter3d1/i7sti3.F      (Starter) per-segment/node stiffness and gap
                           — ported in pyradioss/contact/stiffness.py

Algorithm
---------
1. **Broad phase** (every ``refresh`` cycles) — a voxel (bucket) sort, the
   port of i7buce replacing M1's all-pairs bounding-box test:

   * every segment's bounding box, inflated by the gap plus a travel
     margin, is binned into a uniform grid whose cell size equals the
     LARGEST inflated box extent — so a box overlaps at most 2 cells per
     axis (8 cells total) and the binning stays fully vectorized;
   * every tracked secondary node is binned into its single cell;
   * candidate pairs = (node, segment) sharing a cell, deduplicated,
     minus pairs where the node is a corner of the segment (the classic
     self-exclusion, which is also what makes *self-impact* work: a
     surface's own nodes are tested against all its segments except the
     ones they belong to).

   The travel margin is sized from the current maximum nodal speed so no
   pair can penetrate undetected between refreshes — the same safety
   criterion as the original's bucket re-sort test.

2. **Narrow phase** (every cycle): each candidate quad segment is split
   into two triangles; the closest point on each triangle to the node is
   found exactly (barycentric clamping). Penetration is

       p = gap_pair - d,     d = |x_node - x_closest|

   where ``gap_pair`` is constant (Igap=0) or the sum of the two sides'
   physical gaps (Igap=1: half shell thicknesses — see stiffness.py).

3. **Penalty force** (i7for3): if p > 0, a spring pushes the node back
   along the closest-point direction, plus a small viscous damper on the
   normal relative velocity (stabilizes the bounce):

       Fn = K_pair * p + C * vn_rel,    C = 2 * visc * sqrt(K_pair * m_node)

   ``K_pair`` combines the main-segment and secondary-node element
   stiffness according to Istf (see stiffness.combine_stiffness). The
   reaction is distributed to the segment's corner nodes with the
   barycentric weights of the closest point (equal and opposite total
   force — momentum is conserved exactly).

4. **Coulomb friction**: the tangential relative velocity direction gets
   a force min(mu * Fn, ...) opposing it (kinetic friction, smoothly
   regularized near zero slip velocity). Since M15, ``Ifric > 0``
   replaces the constant mu with the MFROT friction MODELS mu(p, v) of
   i7for3.F — pressure p = Fn over the CURRENT main-segment area, slip
   speed v — and ``Ifiltr`` low-pass filters the tangential force with
   the per-pair IFQ exponential moving average (both live in
   ``contact/friction.py``, where the laws, the reader mapping and the
   one documented deviation are derived against the fetched source).
   The Ifric = 0 code path is untouched — bit-identical to M4 (asserted
   by the M15 tests). The mu(p, v) evaluation and the filter sit
   DOWNSTREAM of the numba-mirrored narrow phase (t7_narrow), in the
   NumPy force path both backends share — the M7 kernel parity contract
   holds with no mirror change (stated, not silent: the mirror boundary
   is the narrow phase, see the accel package docstring).
   The IFQ anchor store is engine-side state rebuilt EMPTY on a restart
   chain (like the broad-phase candidates); a chained IFQ run
   re-converges its filter within ~1/alpha cycles — the M6 restart
   bit-match contract is guaranteed for Ifiltr = 0 decks.

5. **Element deletion (M3<->M4)**: segments whose parent element has
   GBUF%OFF = 0 are masked out every cycle (crack faces stop pushing),
   and secondary nodes all of whose elements died stop being tracked at
   the next broad phase (see contact/tracking.py for the physics).

Time step: every penalty spring adds a frequency to the nodes it loads;
the stability bound of the node-on-spring system dt = sqrt(2 m / K) is
evaluated at init for the worst (mass, pair-stiffness) combination on
both sides and fed to the Engine's global minimum every cycle, exactly
like the original's interface time step.

Port simplifications vs the full TYPE7 (documented, roadmap M5+): no
Inacti initial-penetration treatments (small initial penetrations just
push back out), constant linear spring instead of the original's
stiffening K*p/(gap-p) near-crossing guard, no Tstart/Tstop or sensors,
Igap 2/3 (mesh-size-scaled gaps) not ported.
"""

from __future__ import annotations

import numpy as np

from ..accel import get as accel_get
from ..common.constants import EM20
from ..common.fastmath import cross3, norm3, scatter_add3
from ..model.model import Model
from . import friction, tracking
from .stiffness import (combine_stiffness, node_stiffness_gap,
                        segment_stiffness_gap, _segment_areas,
                        segment_mesh_gap, node_mesh_gap)

_VISC = 0.05  # normal damping ratio (Radioss VIS_S default 5%)


def _closest_point_on_triangle(p, a, b, c):
    """Vectorized exact closest point on triangles (Ericson, 'Real-Time
    Collision Detection' §5.1.5). All args (n,3). Returns (point, wa, wb, wc)
    barycentric weights."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("nk,nk->n", ab, ap)
    d2 = np.einsum("nk,nk->n", ac, ap)
    bp = p - b
    d3 = np.einsum("nk,nk->n", ab, bp)
    d4 = np.einsum("nk,nk->n", ac, bp)
    cp = p - c
    d5 = np.einsum("nk,nk->n", ab, cp)
    d6 = np.einsum("nk,nk->n", ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    # start from the inside-face solution and overwrite region by region
    denom = np.maximum(va + vb + vc, EM20)
    v = vb / denom
    w = vc / denom

    # edge AB region
    onAB = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / np.maximum(d1 - d3, EM20)
    v = np.where(onAB, t, v)
    w = np.where(onAB, 0.0, w)
    # edge AC region
    onAC = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / np.maximum(d2 - d6, EM20)
    v = np.where(onAC, 0.0, v)
    w = np.where(onAC, t, w)
    # edge BC region
    onBC = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
    t = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), EM20)
    v = np.where(onBC, 1.0 - t, v)
    w = np.where(onBC, t, w)
    # vertex regions
    atA = (d1 <= 0) & (d2 <= 0)
    v = np.where(atA, 0.0, v)
    w = np.where(atA, 0.0, w)
    atB = (d3 >= 0) & (d4 <= d3)
    v = np.where(atB, 1.0, v)
    w = np.where(atB, 0.0, w)
    atC = (d6 >= 0) & (d5 <= d6)
    v = np.where(atC, 0.0, v)
    w = np.where(atC, 1.0, w)

    u = 1.0 - v - w
    point = u[:, None] * a + v[:, None] * b + w[:, None] * c
    return point, u, v, w


def _narrow(x, ni, seg):
    """Narrow phase (i7dst3): exact closest point of each candidate node
    on its quad segment, the quad split into triangles (0,1,2) and
    (0,2,3). Returns (best_d, best_pt, best_w) with ``best_w`` the corner
    weights of the closest point. Mirrored by
    accel.jit_kernels.t7_narrow (M7 — this was the hottest single block
    of the self-impact examples; see the accel package docstring)."""
    best_d = np.full(len(ni), np.inf)
    best_pt = np.zeros((len(ni), 3))
    best_w = np.zeros((len(ni), 4))
    p = x[ni]
    for cols in ((0, 1, 2), (0, 2, 3)):
        if cols == (0, 2, 3):
            non_degen = seg[:, 2] != seg[:, 3]
            if not np.any(non_degen):
                continue
        a, b, c = (x[seg[:, cols[0]]], x[seg[:, cols[1]]],
                   x[seg[:, cols[2]]])
        pt, u, vv, w = _closest_point_on_triangle(p, a, b, c)
        d = norm3(p - pt)
        better = d < best_d
        if cols == (0, 2, 3):
            better = better & non_degen
        best_d = np.where(better, d, best_d)
        best_pt[better] = pt[better]
        wq = np.zeros((len(ni), 4))
        wq[:, cols[0]], wq[:, cols[1]], wq[:, cols[2]] = u, vv, w
        best_w[better] = wq[better]
    return best_d, best_pt, best_w


def _expand_matches(keys_a: np.ndarray, keys_b: np.ndarray):
    """All (i, j) index pairs with keys_a[i] == keys_b[j], fully
    vectorized (the join step of the voxel sort). Returns (ii, jj)."""
    order = np.argsort(keys_b, kind="stable")
    sorted_b = keys_b[order]
    left = np.searchsorted(sorted_b, keys_a, "left")
    right = np.searchsorted(sorted_b, keys_a, "right")
    counts = right - left
    total = int(counts.sum())
    if total == 0:
        return (np.zeros(0, dtype=np.int64),) * 2
    ii = np.repeat(np.arange(len(keys_a)), counts)
    # position of each match inside its own [left, right) run
    run_starts = np.cumsum(counts) - counts
    offs = np.arange(total) - np.repeat(run_starts, counts)
    jj = order[np.repeat(left, counts) + offs]
    return ii, jj


class ContactType7:
    """One /INTER/TYPE7 interface, engine-side."""

    def _init_empty(self):
        """Initialize empty/inactive state."""
        self.segs = np.zeros((0, 4), dtype=np.int64)
        self.seg_gtype = np.zeros(0, dtype="<U8")
        self.seg_elem = np.zeros(0, dtype=np.int64)
        self.nodes = np.zeros(0, dtype=np.int64)
        self.Km = np.zeros(0, dtype=float)
        self.Ks = np.zeros(0, dtype=float)
        self.gap_m = np.zeros(0, dtype=float)
        self.gap_s = np.zeros(0, dtype=float)
        self.gap_m_l = np.zeros(0, dtype=float)
        self.gap_s_l = np.zeros(0, dtype=float)
        self.gap_const = 0.0
        self.gap_bound = 0.0
        self.gap_min = 0.0
        self.gap_max = np.inf
        self.stmin = float(getattr(getattr(self, "itf", None), "stmin", 0.0) or 0.0)
        self.stmax = float(getattr(getattr(self, "itf", None), "stmax", 0.0) or 0.0)
        self.fric = float(getattr(self.itf, "fric", 0.0))
        self.mfrot = int(getattr(self.itf, "mfrot", 0))
        self.iform = int(getattr(self.itf, "iform", 0))
        self.ifq = int(getattr(self.itf, "ifq", 0))
        self.xfiltr = float(getattr(self.itf, "xfiltr", 0.0))
        self.fric_c = np.asarray(getattr(self.itf, "fric_c", (0.0,) * 6), dtype=float)
        self._filt_keys = np.zeros(0, dtype=np.int64)
        self._filt_vals = np.zeros((0, 3), dtype=float)
        self.dt_bound = np.inf
        self.idel = int(getattr(self.itf, "idel", 0))
        self.deletable = False
        self.seg_alive = np.zeros(0, dtype=bool)
        self.nodes_tracked = np.zeros(0, dtype=np.int64)
        self.pairs_node = np.zeros(0, dtype=np.int64)
        self.pairs_seg = np.zeros(0, dtype=np.int64)
        self._last_refresh = -10**9
        self.refresh = 20
        self.visc = float(getattr(self.itf, "viss", getattr(self.itf, "stiff_dc", 0.05)))
        if self.visc <= 0.0:
            self.visc = 0.05
        self.tstart = float(getattr(self.itf, "tstart", 0.0))
        self.tstop = float(getattr(self.itf, "tstop", 1.0e30))

    def __init__(self, itf, model: Model, log):
        self.itf = itf
        self.model = model

        self.visc = float(getattr(itf, "viss", getattr(itf, "stiff_dc", 0.05)))
        if self.visc <= 0.0:
            self.visc = 0.05
        self.tstart = float(getattr(itf, "tstart", 0.0))
        self.tstop = float(getattr(itf, "tstop", 1.0e30))

        surf = model.surfaces.get(itf.surf_id)
        if surf is None:
            log.warning(f"/INTER/TYPE7/{itf.id}: main surface {itf.surf_id} not found — "
                        f"interface inactive", "CONTACT INIT")
            self._init_empty()
            return

        segs = surf.segments
        if segs is None or len(segs) == 0:
            log.warning(f"/INTER/TYPE7/{itf.id}: main surface is empty — "
                        f"interface inactive", "CONTACT INIT")
            self._init_empty()
            return

        segs = np.asarray(segs, dtype=np.int64)
        if segs.ndim == 2 and segs.shape[1] == 3:
            segs = np.column_stack([segs, segs[:, 2]])
        elif segs.ndim == 2 and segs.shape[1] == 4:
            segs = segs.copy()
            neg = segs[:, 3] < 0
            segs[neg, 3] = segs[neg, 2]

        self.segs = segs
        self.seg_gtype = surf.seg_gtype
        self.seg_elem = surf.seg_elem

        # --- secondary nodes: group, or the surface's own nodes ----------
        # (grnod_ID = 0 -> self-impact, the Radioss single-surface input)
        if itf.grnod_id == 0:
            self.nodes = np.unique(self.segs)
            log.info(f"     /INTER/TYPE7/{itf.id}: SELF-IMPACT — "
                     f"{len(self.nodes)} SECONDARY NODES FROM THE MAIN "
                     f"SURFACE")
        else:
            g = model.node_groups.get(itf.grnod_id)
            if g is None or g.node_idx is None or len(g.node_idx) == 0:
                log.warning(f"/INTER/TYPE7/{itf.id}: secondary node group {itf.grnod_id} "
                            f"empty or not found — interface inactive", "CONTACT INIT")
                self._init_empty()
                return
            self.nodes = np.sort(np.asarray(g.node_idx, dtype=np.int64))

        if len(self.nodes) == 0:
            self._init_empty()
            return

        # --- penalty stiffness (i7sti3) -----------------------------------
        # Element-based stiffness on both sides; Istf picks the combination
        # (for Istf=1, stfac IS the stiffness and the element values are
        # only kept for the gap computation -> scale 1).
        scale = itf.stfac if itf.istf != 1 else 1.0
        fscale = getattr(itf, "fscale_gap", 1.0) if itf.igap in (2, 3) else 1.0
        Km, gm = segment_stiffness_gap(model, self.segs, self.seg_gtype,
                                       self.seg_elem, scale, fscale_gap=fscale)
        Ks_all, gs_all = node_stiffness_gap(model, scale, fscale_gap=fscale)
        self.Km = Km
        if len(self.nodes) and len(Ks_all):
            valid_k = self.nodes < len(Ks_all)
            self.Ks = np.zeros(len(self.nodes), dtype=float)
            self.Ks[valid_k] = Ks_all[self.nodes[valid_k]]
        else:
            self.Ks = np.zeros(len(self.nodes), dtype=float)

        if itf.igap == 3:
            pmesh = getattr(itf, "percent_mesh_size", 0.4)
            self.gap_m_l = segment_mesh_gap(model, self.segs, pmesh)
            self.gap_s_l = node_mesh_gap(model, self.segs, self.nodes, pmesh)

        # --- contact gap ---------------------------------------------------
        # lc = mean segment size, the reference length for the defaults
        area = _segment_areas(model.x0, self.segs)
        lc = float(np.sqrt(area.mean())) if len(area) else 1.0
        # the default MINIMUM gap: the physical half-thickness where the
        # main side is shells, a small fraction of the segment size
        # otherwise (a zero gap would make contact undetectable)
        gap_floor = itf.gap if itf.gap > 0 else (
            float(gm.mean()) if len(gm) and gm.max() > 0 else 0.02 * lc)
        if itf.igap in (1, 2, 3):
            # variable gap: g = g_s(node) + g_m(segment), clipped
            self.gap_m = gm
            if len(self.nodes) and len(gs_all):
                valid_g = self.nodes < len(gs_all)
                self.gap_s = np.zeros(len(self.nodes), dtype=float)
                self.gap_s[valid_g] = gs_all[self.nodes[valid_g]]
            else:
                self.gap_s = np.zeros(len(self.nodes), dtype=float)
            self.gap_min = gap_floor
            self.gap_max = itf.gap_max if itf.gap_max > 0 else np.inf
            gap_hi = (self.gap_s.max() if len(self.gap_s) else 0.0) + \
                (gm.max() if len(gm) else 0.0)
            self.gap_bound = float(np.clip(gap_hi, self.gap_min,
                                           self.gap_max))
        else:
            self.gap_const = gap_floor
            self.gap_bound = gap_floor
        self.fric = itf.fric
        self.stmin = float(getattr(itf, "stmin", 0.0) or 0.0)
        self.stmax = float(getattr(itf, "stmax", 0.0) or 0.0)

        # --- friction MODELS + IFQ filter state (M15) -----------------------
        # mfrot/ifq = 0 leaves every M4 path untouched (bit-identical).
        # The filter anchors are the CAND_F store of i7for3.F, keyed
        # node*nseg + segrow, sorted; rebuilt from the active pairs each
        # cycle (a separated pair restarts its history — the IFPEN scope).
        self.mfrot = int(getattr(itf, "mfrot", 0))
        self.iform = int(getattr(itf, "iform", 0))
        self.ifq = int(getattr(itf, "ifq", 0))
        self.xfiltr = float(getattr(itf, "xfiltr", 0.0))
        self.fric_c = np.asarray(getattr(itf, "fric_c",
                                         (0.0,) * 6), dtype=float)
        self._filt_keys = np.zeros(0, dtype=np.int64)
        self._filt_vals = np.zeros((0, 3))
        if self.mfrot > 0:
            log.info(f"     /INTER/TYPE7/{itf.id}: FRICTION MODEL "
                     f"MFROT={self.mfrot} (i7for3.F mu(p, v)), "
                     f"IFQ={self.ifq}, IFORM={self.iform}")

        # --- interface time step bound (see module docstring) --------------
        # Worst node-on-spring combination on each side, evaluated once
        # (masses and stiffness are constant; deletion only REMOVES
        # springs). The PHYSICAL pre-mass-scaling masses keep the bound
        # conservative and restart-invariant (M6).
        self.dt_bound = self._compute_dt_bound(
            getattr(model, "mass0", model.mass))

        # --- deletion bookkeeping (M3<->M4) --------------------------------
        # Fortran chkstfn3.F:1352 gates deletion on IDEL >= 1; Idel=0 (default)
        # means no deletion processing even if the material can fail.
        self.idel = int(getattr(itf, 'idel', 0))
        self.deletable = (self.idel >= 1 and
                          tracking.any_deletable(model, self.seg_gtype,
                                                 sec_nodes=self.nodes))
        if self.deletable:
            self.ref_total = tracking.node_reference_counts(
                model, alive_only=False)
        self.seg_alive = np.ones(len(self.segs), dtype=bool)
        self.nodes_tracked = self.nodes

        # broad-phase bookkeeping
        self.pairs_node = np.zeros(0, dtype=np.int64)  # global node index
        self.pairs_seg = np.zeros(0, dtype=np.int64)   # segment row
        self._last_refresh = -10**9
        self.refresh = 20                              # cycles

    # ------------------------------------------------------------------
    def _compute_dt_bound(self, mass) -> float:
        """dt <= sqrt(2 m / K) for every node a contact spring can load:
        each secondary node with the stiffest segment it could meet, each
        main corner node with its own segments against the stiffest
        secondary node."""
        if len(self.segs) == 0 or len(self.nodes) == 0:
            return np.inf
        itf = self.itf
        valid_nodes_mask = (self.nodes >= 0) & (self.nodes < len(mass))
        valid_nodes = self.nodes[valid_nodes_mask]
        if len(valid_nodes) == 0:
            return np.inf
        Km_val = self.Km.max() if len(self.Km) else 0.0
        Km_max = np.full(len(valid_nodes), Km_val)
        Ks_subset = self.Ks[valid_nodes_mask] if len(self.Ks) == len(self.nodes) else np.zeros(len(valid_nodes))
        K_sec = combine_stiffness(itf.istf, itf.stfac, Km_max, Ks_subset)
        m_sec = mass[valid_nodes]
        m_sec_pos = m_sec > 0.0
        dt_sec = np.sqrt(2.0 * m_sec[m_sec_pos] / np.maximum(K_sec[m_sec_pos], EM20)).min() if np.any(m_sec_pos) else np.inf

        valid_segs_mask = np.all((self.segs >= 0) & (self.segs < len(mass)), axis=1)
        if not np.any(valid_segs_mask):
            return float(dt_sec)
        valid_segs = self.segs[valid_segs_mask]
        Km_subset = self.Km[valid_segs_mask] if len(self.Km) == len(self.segs) else np.zeros(len(valid_segs))
        Ks_max = np.full(len(valid_segs),
                         self.Ks.max() if len(self.Ks) else 0.0)
        K_main = combine_stiffness(itf.istf, itf.stfac, Km_subset, Ks_max)
        m_corner = mass[valid_segs].min(axis=1)
        mc_pos = m_corner > 0.0
        dt_main = np.sqrt(2.0 * m_corner[mc_pos] / np.maximum(K_main[mc_pos], EM20)).min() if np.any(mc_pos) else np.inf
        return float(min(dt_sec, dt_main))

    # ------------------------------------------------------------------
    def _broad_phase(self, x: np.ndarray, v: np.ndarray, dt: float):
        """Voxel candidate search (i7buce port — see module docstring)."""
        segs = self.segs[self.seg_alive]
        seg_rows = np.where(self.seg_alive)[0]
        nodes = self.nodes_tracked
        if len(segs) == 0 or len(nodes) == 0:
            self.pairs_node = np.zeros(0, dtype=np.int64)
            self.pairs_seg = np.zeros(0, dtype=np.int64)
            return
        margin = self.gap_bound + 2.0 * self.refresh * dt * \
            (np.abs(v).max() if len(v) else 0.0)

        xs = x[segs]                                    # (nseg, 4, 3)
        lo = xs.min(axis=1) - margin
        hi = xs.max(axis=1) + margin
        # cell size = the largest inflated box extent -> every box spans
        # at most 2 cells per axis, so the box binning below (8 offsets)
        # is exhaustive
        h = max(float((hi - lo).max()), EM20)
        origin = np.minimum(lo.min(axis=0), x[nodes].min(axis=0))

        ilo = np.floor((lo - origin) / h).astype(np.int64)
        ihi = np.floor((hi - origin) / h).astype(np.int64)
        inode = np.floor((x[nodes] - origin) / h).astype(np.int64)

        # single-integer cell keys (grid dims from the node+box extents)
        dims = np.maximum(np.maximum(ihi.max(axis=0), inode.max(axis=0)),
                          0) + 2

        def key(ijk):
            return (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]

        # bin every segment box into its <= 8 covered cells
        seg_ids, seg_keys = [], []
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ijk = ilo + np.array([dx, dy, dz])
                    inside = np.all(ijk <= ihi, axis=1)
                    seg_ids.append(np.where(inside)[0])
                    seg_keys.append(key(ijk[inside]))
        seg_ids = np.concatenate(seg_ids)
        seg_keys = np.concatenate(seg_keys)

        ii, jj = _expand_matches(key(inode), seg_keys)
        ni = nodes[ii]
        sj = seg_rows[seg_ids[jj]]
        if len(ni):
            # deduplicate (a box can meet a node in several cells)
            pair_key = ni * len(self.segs) + sj
            _, first = np.unique(pair_key, return_index=True)
            ni, sj = ni[first], sj[first]
            # a secondary node that is a corner of the segment is never a
            # candidate against it (self-exclusion; enables self-impact)
            keep = np.ones(len(ni), dtype=bool)
            for k in range(4):
                keep &= self.segs[sj, k] != ni
            ni, sj = ni[keep], sj[keep]
        self.pairs_node = ni
        self.pairs_seg = sj

    # ------------------------------------------------------------------
    def forces(self, x, v, mass, dt, fcont, cycle=0, stifn=None, t=None, **kwargs):
        """Penalty forces for one cycle, scattered into ``fcont``.

        Returns (contact_work_increment, dt_interface). The work increment
        is the interface's own estimate at the pre-update velocities
        (-F.vrel dt); the Engine books the exact leapfrog-consistent value
        from the assembled ``fcont`` at the midstep velocity — see the
        'contact energy booking' block of engine.py for why.

        ``stifn`` (M6, /DT/NODA): when given, the NEAR-candidate spring
        stiffness accumulated per node (the same sums that feed this
        interface's own dt) is also added into this global nodal-stiffness
        array, so the nodal time step / mass scaling sees the contact
        springs exactly like the element stiffness.
        """
        if len(self.segs) == 0 or len(self.nodes) == 0 or dt <= 0.0:
            return 0.0, self.dt_bound

        # Time window gating
        if t is not None:
            if t < self.tstart or t > self.tstop:
                return 0.0, self.dt_bound

        # ---- deletion bookkeeping (cheap gathers; see tracking.py) -------
        if self.deletable:
            self.seg_alive = tracking.alive_segment_mask(
                self.model, self.seg_gtype, self.seg_elem)

        if cycle - self._last_refresh >= self.refresh:
            if self.deletable:
                mask = tracking.tracked_node_mask(self.model, self.ref_total)
                valid_m = (self.nodes >= 0) & (self.nodes < len(mask))
                tracked = np.zeros(len(self.nodes), dtype=bool)
                tracked[valid_m] = mask[self.nodes[valid_m]]
                self.nodes_tracked = self.nodes[tracked]
            self._broad_phase(x, v, dt)
            self._last_refresh = cycle
        if len(self.pairs_node) == 0:
            return 0.0, self.dt_bound

        # drop pairs whose segment died since the last refresh
        live = self.seg_alive[self.pairs_seg]
        ni = self.pairs_node[live]                       # global node idx
        srow = self.pairs_seg[live]
        if len(ni) == 0:
            return 0.0, self.dt_bound
        seg = self.segs[srow]                            # (np, 4)

        # ---- narrow phase: exact closest point (i7dst3) -------------------
        # (dispatched to the numba mirror when that backend is active)
        jit = accel_get("t7_narrow")
        if jit is not None:
            best_d, best_pt, best_w = jit(x, ni, seg)
        else:
            best_d, best_pt, best_w = _narrow(x, ni, seg)

        # ---- per-pair gap (Igap) ------------------------------------------
        loc = np.searchsorted(self.nodes, ni)    # nodes is sorted (init)
        loc = np.clip(loc, 0, max(len(self.nodes) - 1, 0))
        if self.itf.igap in (1, 2, 3):
            # gap_s is aligned with self.nodes; loc maps global -> local
            gap = self.gap_s[loc] + self.gap_m[srow]
            if self.itf.igap == 3:
                mesh_gap = self.gap_s_l[loc] + self.gap_m_l[srow]
                gap = np.minimum(gap, mesh_gap)
            gap = np.clip(gap, self.gap_min, self.gap_max)
        else:
            gap = np.full(len(ni), self.gap_const)

        # ---- interface time step: NEAR-pair stiffness accumulation --------
        # A spring that may close within the next cycles must already be
        # in the nodal bound dt = sqrt(2m/K) — and a node facing several
        # segments (a corner meets 3 faces, a self-impact node overhangs 4
        # segments) carries the SUM of their stiffnesses. Waiting for
        # actual penetration lets the first impact cycles run at an
        # unstable dt and pump energy in (found the hard way by the M4
        # /DT 0.9 tests); counting every box-overlap candidate is the
        # other extreme — it throttles dt while bodies are merely in the
        # same neighbourhood. 'Near' = within one extra gap (or one cycle
        # of closing speed) of touching; dt reacts every cycle, so that
        # horizon is enough. (The original's STIFN accumulation with its
        # sorted-candidate scope plays the same role.)
        vmax = float(np.abs(v).max()) if len(v) else 0.0
        near = best_d < gap + np.maximum(gap, 2.0 * vmax * dt)
        if not np.any(near):
            return 0.0, self.dt_bound
        ni = ni[near]
        seg = seg[near]
        srow = srow[near]
        loc = loc[near]
        gap = gap[near]
        best_d = best_d[near]
        best_pt = best_pt[near]
        best_w = best_w[near]

        K = combine_stiffness(self.itf.istf, self.itf.stfac,
                              self.Km[srow], self.Ks[loc])
        if self.stmin > 0.0:
            K = np.maximum(K, self.stmin)
        if self.stmax > 0.0:
            K = np.minimum(K, self.stmax)
        # per-node spring-stiffness sums (bincount = the fast add.at, M7):
        # full K on the secondary node, full K on each corner (weight <= 1)
        n_nod = len(fcont)
        valid_ni = (ni >= 0) & (ni < n_nod)
        Knode = np.bincount(ni[valid_ni], weights=K[valid_ni], minlength=n_nod)
        seg_flat = seg.reshape(-1)
        valid_sf = (seg_flat >= 0) & (seg_flat < n_nod)
        K_rep = np.repeat(K, 4)
        Knode += np.bincount(seg_flat[valid_sf], weights=K_rep[valid_sf],
                             minlength=n_nod)
        loaded = Knode > 0.0
        if np.any(loaded):
            m_loaded = np.maximum(mass[loaded], EM20)
            dt_int = min(self.dt_bound, float(
                np.sqrt(2.0 * m_loaded / Knode[loaded]).min()))
        else:
            dt_int = self.dt_bound
        if stifn is not None:                    # /DT/NODA accumulation
            stifn[loaded] += Knode[loaded]

        pen = gap - best_d
        active = pen > 0.0
        if not np.any(active):
            return 0.0, dt_int

        ni = ni[active]
        seg = seg[active]
        K = K[active]                            # per-pair stiffness (Istf)
        gap = gap[active]
        pen = pen[active]

        # oriented face normal
        d13 = x[seg[:, 2]] - x[seg[:, 0]]
        d24 = x[seg[:, 3]] - x[seg[:, 1]]
        n_seg_raw = cross3(d13, d24)
        n_seg_norm = norm3(n_seg_raw)
        n_seg = np.where(
            (n_seg_norm > EM20)[:, None],
            n_seg_raw / np.maximum(n_seg_norm, EM20)[:, None],
            np.array([0.0, 0.0, 1.0]),
        )

        dist = best_d[active]
        d = np.maximum(dist, EM20)
        nvec = (x[ni] - best_pt[active]) / d[:, None]    # push-out direction
        fallback = dist <= EM20
        nvec = np.where(fallback[:, None], n_seg, nvec)
        wseg = best_w[active]

        # relative velocity node vs interpolated segment point
        vseg = np.einsum("nk,nkb->nb", wseg, v[seg])
        vrel = v[ni] - vseg
        vn = np.einsum("nb,nb->n", vrel, nvec)

        # normal force: spring + damper (only damp approaching motion)
        C = self.visc * np.sqrt(2.0 * K * mass[ni])
        Fn = K * pen - C * np.minimum(vn, 0.0)
        Fvec = Fn[:, None] * nvec

        # Coulomb friction, regularized around zero slip. Ifric > 0 (M15)
        # swaps the constant mu for the MFROT mu(p, v) laws of i7for3.F
        # (contact/friction.py) and Ifiltr low-pass filters the tangential
        # force (the CAND_F exponential moving average). The mu = const,
        # no-filter path below is the M4 code verbatim (x + (-a) == x - a
        # exactly in IEEE — bit-identical, asserted by the M15 tests).
        if self.fric > 0.0 or self.mfrot > 0:
            gap_ref = float(np.mean(gap))
            vt = vrel - vn[:, None] * nvec
            vt_mag = norm3(vt)
            Fn_pos = np.maximum(Fn, 0.0)
            if self.mfrot > 0:
                # contact pressure p = Fn / (CURRENT main-segment area),
                # the i7for3 AREA = 1/2 |(x3-x1) x (x4-x2)| — its FNI by
                # this point includes the damper term, exactly like Fn
                d13 = x[seg[:, 2]] - x[seg[:, 0]]
                d24 = x[seg[:, 3]] - x[seg[:, 1]]
                area = 0.5 * norm3(cross3(d13, d24))
                pres = np.where(area > EM20, Fn_pos / np.maximum(area, EM20), 0.0)
                mu = friction.mu_kinetic(self.mfrot, self.fric,
                                         self.fric_c, pres, vt_mag)
            else:
                mu = self.fric

            if self.iform == 2 or self.ifq >= 10:
                alpha = friction.filter_alpha(self.ifq, self.xfiltr, dt)
                keys = ni * max(len(self.segs), 1) + srow[active]
                ftvec, self._filt_keys, self._filt_vals = friction.\
                    apply_incremental_stiffness(keys, K, vrel, dt, nvec, mu, Fn_pos,
                                                alpha, self._filt_keys,
                                                self._filt_vals)
                ftvec = -ftvec  # oppose sliding (i7for3.F:1511: FNCONT(JG) -= FXI)
            else:
                v_ref = np.maximum(1e-3 * gap_ref / max(dt, EM20), EM20)
                Ft = mu * Fn_pos * vt_mag / (vt_mag + v_ref)
                ftvec = -(Ft / np.maximum(vt_mag, EM20))[:, None] * vt
                if self.ifq > 0:
                    alpha = friction.filter_alpha(self.ifq, self.xfiltr, dt)
                    keys = ni * max(len(self.segs), 1) + srow[active]
                    ftvec, self._filt_keys, self._filt_vals = friction.\
                        apply_filter(keys, ftvec, alpha, self._filt_keys,
                                     self._filt_vals)
            Fvec += ftvec

        # scatter: action on the node, exact opposite reaction on the
        # segment corners (momentum conservation)
        valid_ni_act = (ni >= 0) & (ni < n_nod)
        if np.any(valid_ni_act):
            scatter_add3(fcont, ni[valid_ni_act], Fvec[valid_ni_act])

        w_F = (-wseg[:, :, None] * Fvec[:, None, :]).reshape(-1, 3)
        seg_flat_act = seg.reshape(-1)
        valid_sf_act = (seg_flat_act >= 0) & (seg_flat_act < n_nod)
        if np.any(valid_sf_act):
            scatter_add3(fcont, seg_flat_act[valid_sf_act], w_F[valid_sf_act])

        # contact work this cycle (stored elastic + dissipated), for the
        # energy balance
        wrk = float(np.einsum("nb,nb->", Fvec, vrel)) * dt
        return -wrk, dt_int

    def compute_thermal_conduction(
        self,
        temp: np.ndarray,
        dt: float,
        theaccfact: float = 1.0,
        kthe: Optional[float] = None,
        frad: Optional[float] = None,
        drad: Optional[float] = None,
        iform: Optional[int] = None,
        tint: Optional[float] = None,
        fheats: Optional[float] = None,
        fheatm: Optional[float] = None,
        efrict: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Compute thermal conduction, radiation, and friction heating for /INTER/TYPE7.

        Ported from $OR_SRC/engine/source/interfaces/int07/i7therm.F

        Parameters
        ----------
        temp : np.ndarray
            Nodal temperatures.
        dt : float
            Time step dt.
        theaccfact : float
            Thermal acceleration factor.
        kthe : Optional[float]
            Thermal interface conductance KTHE.
        frad : Optional[float]
            Radiation coefficient.
        drad : Optional[float]
            Radiation cutoff distance.
        iform : Optional[int]
            0 = ambient exchange, 1 = slave-master exchange.
        tint : Optional[float]
            Ambient temperature.
        fheats : Optional[float]
            Friction heating fraction to slave.
        fheatm : Optional[float]
            Friction heating fraction to master.
        efrict : Optional[np.ndarray]
            Frictional dissipation per pair over dt.

        Returns
        -------
        fthe : np.ndarray
            Nodal thermal energy increments [J].
        condint : np.ndarray
            Thermal conductance per pair [W/K].
        ledger : Dict[str, float]
            Conduction, radiation, and friction energy breakdown.
        """
        from .thermal_contact import thermal_contact_type7

        itf = self.itf
        if kthe is None:
            kthe = getattr(itf, "kthe", 0.0) or getattr(itf, "rstif", 0.0)
        if frad is None:
            frad = getattr(itf, "frad", 0.0)
        if drad is None:
            drad = getattr(itf, "drad", 0.0)
        if iform is None:
            iform = getattr(itf, "iform_th", getattr(itf, "iform", 1))
        if tint is None:
            tint = getattr(itf, "tint", 293.15)
        if fheats is None:
            fheats = getattr(itf, "fheats", 0.0)
        if fheatm is None:
            fheatm = getattr(itf, "fheatm", 0.0)

        # If pairs are active from broad/narrow phase
        if len(self.pairs_node) == 0:
            return np.zeros(len(temp), dtype=float), np.zeros(0, dtype=float), {"conduction": 0.0, "radiation": 0.0, "friction": 0.0}

        x = getattr(self.model, "x", getattr(self.model, "x0", np.zeros((len(temp), 3))))
        slave_nodes = self.pairs_node
        master_segs = self.segs[self.pairs_seg]
        weights = np.full((len(slave_nodes), 4), 0.25, dtype=float)

        return thermal_contact_type7(
            x=x,
            temp=temp,
            slave_nodes=slave_nodes,
            master_segs=master_segs,
            weights=weights,
            kthe=kthe,
            dt=dt,
            theaccfact=theaccfact,
            iform=iform,
            tint=tint,
            frad=frad,
            drad=drad,
            fheats=fheats,
            fheatm=fheatm,
            efrict=efrict,
        )


class LagmulType7:
    """One /INTER/LAGMUL/TYPE7 constraint, engine-side."""
    def __init__(self, itf, model: Model, log):
        self.itf = itf
        self.model = model
        
        # We reuse the initialization logic of ContactType7 to parse groups, segments, etc.
        # But we create a dummy ContactType7 to hold the state.
        self.penalty_handler = ContactType7(itf, model, log)
        
    def generate_l_matrix(self):
        """Yield (data, node_indices, dof_indices, eq_indices) arrays for the global L matrix.
        Only penetrating nodes approaching the surface (VN < 0) generate constraints.
        If a node hits multiple segments, only the most severe (minimum VN) is kept.
        """
        # Run broad/narrow phases using the penalty handler's logic
        handler = self.penalty_handler
        x = getattr(self.model, "x", self.model.x0) # current position
        v = self.model.v
        dt = self.model.dt
        
        if len(handler.segs) == 0 or len(handler.nodes) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), 0
            
        if handler.deletable:
            handler.seg_alive = tracking.alive_segment_mask(
                handler.model, handler.seg_gtype, handler.seg_elem)

        cycle = self.model.cycle
        if cycle - handler._last_refresh >= handler.refresh:
            if handler.deletable:
                mask = tracking.tracked_node_mask(handler.model, handler.ref_total)
                valid_m = (handler.nodes >= 0) & (handler.nodes < len(mask))
                tracked = np.zeros(len(handler.nodes), dtype=bool)
                tracked[valid_m] = mask[handler.nodes[valid_m]]
                handler.nodes_tracked = handler.nodes[tracked]
            handler._broad_phase(x, v, dt)
            handler._last_refresh = cycle
            
        if len(handler.pairs_node) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), 0
            
        live = handler.seg_alive[handler.pairs_seg]
        ni = handler.pairs_node[live]
        srow = handler.pairs_seg[live]
        if len(ni) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), 0
        seg = handler.segs[srow]

        jit = accel_get("t7_narrow")
        if jit is not None:
            best_d, best_pt, best_w = jit(x, ni, seg)
        else:
            best_d, best_pt, best_w = _narrow(x, ni, seg)
            
        loc = np.searchsorted(handler.nodes, ni)
        if handler.itf.igap in (1, 2, 3):
            gap = handler.gap_s[loc] + handler.gap_m[srow]
            if handler.itf.igap == 3:
                mesh_gap = handler.gap_s_l[loc] + handler.gap_m_l[srow]
                gap = np.minimum(gap, mesh_gap)
            gap = np.clip(gap, handler.gap_min, handler.gap_max)
        else:
            gap = np.full(len(ni), handler.gap_const)
            
        pen = gap - best_d
        active = pen > 0.0
        if not np.any(active):
            return np.array([]), np.array([]), np.array([]), np.array([]), 0
            
        ni = ni[active]
        seg = seg[active]
        pen = pen[active]
        dist_act = best_d[active]
        d = np.maximum(dist_act, EM20)
        nvec = (x[ni] - best_pt[active]) / d[:, None]
        fallback = dist_act <= EM20
        if np.any(fallback):
            d13 = x[seg[:, 2]] - x[seg[:, 0]]
            d24 = x[seg[:, 3]] - x[seg[:, 1]]
            n_seg = cross3(d13, d24)
            n_seg_norm = norm3(n_seg)
            n_seg_safe = np.where(n_seg_norm > EM20, n_seg_norm, 1.0)
            n_seg = np.where((n_seg_norm > EM20)[:, None], n_seg / n_seg_safe[:, None], np.array([0.0, 0.0, 1.0]))
            nvec = np.where(fallback[:, None], n_seg, nvec)
        wseg = best_w[active]
        
        # relative velocity node vs interpolated segment point
        vseg = np.einsum("nk,nkb->nb", wseg, v[seg])
        vrel = v[ni] - vseg
        vn = np.einsum("nb,nb->n", vrel, nvec)
        
        # Fortran I7LAGM filter: VN < XTAG (approaching). We only constraint if VN <= 0.
        approaching = vn <= 0.0
        if not np.any(approaching):
            return np.array([]), np.array([]), np.array([]), np.array([]), 0
            
        ni = ni[approaching]
        seg = seg[approaching]
        nvec = nvec[approaching]
        wseg = wseg[approaching]
        vn = vn[approaching]
        
        # If a node hits multiple segments, only keep the one with the most severe (minimum) VN
        # (This matches the XTAG(IG) logic in I7LAGM)
        sort_idx = np.argsort(vn)  # sorts from most negative (most severe) to least negative
        ni_sorted = ni[sort_idx]
        
        # Find the first occurrence of each node
        _, unique_idx = np.unique(ni_sorted, return_index=True)
        # Restore the selected rows from the sorted arrays
        final_idx = sort_idx[unique_idx]
        
        ni_f = ni[final_idx]
        seg_f = seg[final_idx]
        nvec_f = nvec[final_idx]
        wseg_f = wseg[final_idx]
        
        # Build the L matrix
        n = len(ni_f)
        data = []
        nodes = []
        dofs = []
        eq_ids = []
        
        for i in range(n):
            snode = ni_f[i]
            s_nodes = seg_f[i]
            nx, ny, nz = nvec_f[i]
            # wseg_f[i] contains H1, H2, H3, H4
            
            for dof, n_dof in enumerate((nx, ny, nz)):
                # Master nodes (+Hk * n)
                for k in range(4):
                    data.append(n_dof * wseg_f[i, k])
                    nodes.append(s_nodes[k])
                    dofs.append(dof)
                    eq_ids.append(i)
                        
                # Secondary node (-n)
                data.append(-n_dof)
                nodes.append(snode)
                dofs.append(dof)
                eq_ids.append(i)
                
        return np.array(data), np.array(nodes), np.array(dofs), np.array(eq_ids), n

