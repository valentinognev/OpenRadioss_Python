"""
/INTER/TYPE11 — penalty edge-to-edge contact.

Fortran origin: ``engine/source/interfaces/int11/`` —

    intsort/i11main_tri.F / i11buce.F   candidate edge pairs (bucket sort)
    int11/i11dst3.F                     exact edge-edge distance/projection
    int11/i11for3.F                     penalty force + friction
    inter3d1/i11sti3.F                  (Starter) edge stiffness and gap
                                        — ported in contact/stiffness.py

Why a separate interface exists at all: node-to-surface contact (TYPE7)
cannot see two edges crossing *between* their nodes — think of two shell
plates meeting like crossed swords, or a beam sliding along another
beam's flank. The closest points then lie in the interior of both edges
and no node penetrates any segment; only an edge-edge measure detects it.
The standard crash-model practice is a TYPE7 plus a TYPE11 on the same
parts, which the port supports (the two interfaces just add forces).

Algorithm — the same skeleton as the ported TYPE7:

1. **Broad phase** (every ``refresh`` cycles): voxel binning of the main
   edges' bounding boxes, inflated by gap + travel margin (cell size =
   largest box extent, so a box covers at most 8 cells); secondary edges
   are binned the same way; candidates = box pairs sharing a cell,
   deduplicated, minus pairs sharing a node.

2. **Narrow phase** (every cycle): exact closest points between the two
   segments (Ericson, 'Real-Time Collision Detection' §5.1.9 — the
   clamped two-parameter minimization), giving the points
   cS = a1 + s (a2-a1) on the secondary edge and cM = b1 + t (b2-b1) on
   the main edge. Penetration p = gap_pair - |cS - cM|.

3. **Penalty force** (i11for3): a spring along the connecting direction
   pushes the edges apart, plus the same normal damper and regularized
   Coulomb friction as TYPE7. The force is split onto the edge end nodes
   with the closest-point parameters — (1-s, s) on the secondary edge,
   -(1-t, t) on the main edge — which conserves BOTH linear and angular
   momentum (the two point forces are equal, opposite and collinear).

4. **Element deletion (M3<->M4)**: edges inherit their parent element
   from the /LINE (Starter provenance); edges of deleted elements are
   masked out every cycle, exactly like TYPE7 segments.

The interface time step is the node-on-spring bound dt = sqrt(2 m / K)
for the worst (mass, combined stiffness) on either side, like TYPE7.

Port simplifications: no Inacti / Tstart; the gap is constant
per pair (Igap=1 uses the two edges' half-thicknesses, no mesh-size
scaling variants); parallel-edge contact acts at the single closest-point
pair (the original distributes along the overlap; the resultant is the
same, the distribution slightly different).

Friction MODELS (M15, ``Ifric > 0``) — a documented PORT EXTENSION: the
ORIGINAL TYPE11 never evaluates the MFROT mu(p, v) laws (checked against
the source: ``i11mainf.F`` hardcodes MFROT = 0 and I11FOR3 receives no
FRIC_COEFS; the TYPE11 reader has no Ifric fields). The port extends the
TYPE7 evaluation to edge pairs deliberately so both contact types (and
both solvers) share one friction-model capability, with the edge-pair
contact pressure DEFINED as p = f_n / (L_main * gap_pair) — the current
main-edge length times the pair gap, the tributary strip of a line
contact (derivation and rationale in ``contact/friction.py``). Ifric = 0
keeps every M4 path bit-identical. The IFQ anchor store resets on a
restart chain exactly like TYPE7's (see inter_type7.py).
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from ..common.fastmath import norm3, scatter_add3
from ..model.model import Model
from . import friction, tracking
from .inter_type7 import _expand_matches
from .stiffness import combine_stiffness, edge_stiffness_gap

_VISC = 0.05  # normal damping ratio, as TYPE7


def _closest_points_on_segments(p1, q1, p2, q2):
    """Vectorized exact closest points between segments [p1,q1] and
    [p2,q2] (Ericson §5.1.9, Fortran origin: i11dst3.F). All args (n,3).
    Returns (s, t, cA, cB): parameters in [0,1] and the closest points
    on each segment."""
    if len(p1) == 0:
        return (np.zeros(0, dtype=float), np.zeros(0, dtype=float),
                np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float))
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = np.einsum("nk,nk->n", d1, d1)
    e = np.einsum("nk,nk->n", d2, d2)
    f = np.einsum("nk,nk->n", d2, r)
    c = np.einsum("nk,nk->n", d1, r)
    b = np.einsum("nk,nk->n", d1, d2)
    denom = a * e - b * b

    # general (non-parallel) case; parallel edges pick s = 0 and let the
    # t-clamp below find the right point on the other edge
    s = np.where(denom > EM20,
                 np.clip((b * f - c * e) / np.maximum(denom, EM20), 0., 1.),
                 0.0)
    t = (b * s + f) / np.maximum(e, EM20)

    # clamp t, then recompute s for the clamped t (still exact)
    tlo = t < 0.0
    thi = t > 1.0
    t = np.clip(t, 0.0, 1.0)
    s = np.where(tlo, np.clip(-c / np.maximum(a, EM20), 0.0, 1.0), s)
    s = np.where(thi, np.clip((b - c) / np.maximum(a, EM20), 0.0, 1.0), s)

    deg2 = e <= EM20
    s = np.where(deg2, np.clip(-c / np.maximum(a, EM20), 0.0, 1.0), s)
    t = np.where(deg2, 0.0, t)

    cA = p1 + s[:, None] * d1
    cB = p2 + t[:, None] * d2
    return s, t, cA, cB


class ContactType11:
    """One /INTER/TYPE11 interface, engine-side.

    Fortran origin: engine/source/interfaces/int11/ (i11main_tri.F, i11buce.F,
    i11dst3.F, i11for3.F) and starter/source/interfaces/inter3d1/i11sti3.F.
    """

    def _init_empty(self):
        """Initialize empty state for inactive or missing interfaces."""
        self.es = np.zeros((0, 2), dtype=np.int64)
        self.em = np.zeros((0, 2), dtype=np.int64)
        self.es_gtype = np.zeros(0, dtype="<U8")
        self.es_elem = np.zeros(0, dtype=np.int64)
        self.em_gtype = np.zeros(0, dtype="<U8")
        self.em_elem = np.zeros(0, dtype=np.int64)
        self.Ks = np.zeros(0, dtype=float)
        self.Km = np.zeros(0, dtype=float)
        self.gap_s = np.zeros(0, dtype=float)
        self.gap_m = np.zeros(0, dtype=float)
        self.gap_const = 0.0
        self.gap_min = 0.0
        self.gap_max = np.inf
        self.gap_bound = 0.0
        self.fric = float(getattr(self.itf, "fric", 0.0))
        self.mfrot = int(getattr(self.itf, "mfrot", 0))
        self.ifq = int(getattr(self.itf, "ifq", 0))
        self.xfiltr = float(getattr(self.itf, "xfiltr", 0.0))
        self.fric_c = np.asarray(getattr(self.itf, "fric_c", (0.0,) * 6), dtype=float)
        self._filt_keys = np.zeros(0, dtype=np.int64)
        self._filt_vals = np.zeros((0, 3))
        self.stmin = float(getattr(getattr(self, "itf", None), "stmin", 0.0) or 0.0)
        self.stmax = float(getattr(getattr(self, "itf", None), "stmax", 0.0) or 0.0)
        self.visc = float(getattr(getattr(self, "itf", None), "viss", getattr(getattr(self, "itf", None), "stiff_dc", 0.05)) or 0.05)
        if self.visc <= 0.0:
            self.visc = 0.05
        self.dt_bound = np.inf
        self.idel = int(getattr(self.itf, "idel", 0) or 0)
        self.deletable = False
        self.es_alive = np.zeros(0, dtype=bool)
        self.em_alive = np.zeros(0, dtype=bool)
        self.pairs_s = np.zeros(0, dtype=np.int64)
        self.pairs_m = np.zeros(0, dtype=np.int64)
        self._last_refresh = -10**9
        self.refresh = 20
        self.tstart = float(getattr(self.itf, "tstart", 0.0) or 0.0)
        self.tstop = float(getattr(self.itf, "tstop", np.inf) or np.inf)
        if self.tstop <= 0.0:
            self.tstop = np.inf

    def __init__(self, itf, model: Model, log):
        self.itf = itf
        self.model = model
        params = getattr(itf, "params", {}) or {}

        def _line(lid, side):
            ln = model.lines.get(lid) if hasattr(model, "lines") else None
            if ln is None:
                key = "line1" if side == "secondary" else "line2"
                if key in params and params[key] is not None:
                    segs = np.asarray(params[key], dtype=np.int64)
                    return (segs,
                            np.zeros(len(segs), dtype="<U8"),
                            np.full(len(segs), -1, dtype=np.int64))
                log.error(f"/INTER/TYPE11/{itf.id}: {side} line {lid} not found in model", "CONTACT INIT")
                return (np.zeros((0, 2), dtype=np.int64),
                        np.zeros(0, dtype="<U8"), np.zeros(0, dtype=np.int64))
            if ln.segments is None or len(ln.segments) == 0:
                log.warning(f"/INTER/TYPE11/{itf.id}: {side} line {lid} is "
                            f"empty — interface inactive", "CONTACT INIT")
                return (np.zeros((0, 2), dtype=np.int64),
                        np.zeros(0, dtype="<U8"), np.zeros(0, dtype=np.int64))
            seg_gtype = (ln.seg_gtype if ln.seg_gtype is not None
                         else np.zeros(len(ln.segments), dtype="<U8"))
            seg_elem = (ln.seg_elem if ln.seg_elem is not None
                         else np.full(len(ln.segments), -1, dtype=np.int64))
            return ln.segments, seg_gtype, seg_elem

        self.es, self.es_gtype, self.es_elem = _line(getattr(itf, "line_id1", 0),
                                                     "secondary")
        self.em, self.em_gtype, self.em_elem = _line(getattr(itf, "line_id2", 0), "main")

        if len(self.es) == 0 or len(self.em) == 0:
            self._init_empty()
            return

        # ---- per-edge stiffness and gap (i11sti3) --------------------------
        scale = itf.stfac if getattr(itf, "istf", 0) != 1 else 1.0
        self.Ks, gs = edge_stiffness_gap(model, self.es, self.es_gtype,
                                         self.es_elem, scale)
        self.Km, gm = edge_stiffness_gap(model, self.em, self.em_gtype,
                                         self.em_elem, scale)

        # ---- gap (same default policy as TYPE7) ----------------------------
        x_coords = model.x0 if getattr(model, "x0", None) is not None and len(model.x0) > 0 else getattr(model, "x", np.zeros((0, 3)))
        if len(self.em) and len(x_coords) > 0 and np.max(self.em) < len(x_coords):
            lc = float(np.linalg.norm(x_coords[self.em[:, 1]]
                                      - x_coords[self.em[:, 0]],
                                      axis=1).mean())
        else:
            lc = 1.0
        both = (gm.mean() if len(gm) else 0.0) + (gs.mean() if len(gs)
                                                  else 0.0)
        gap_param = float(params.get("gap", getattr(itf, "gap", 0.0)) or 0.0)
        gap_floor = gap_param if gap_param > 0 else (
            both if both > 0 else 0.02 * lc)
        if itf.igap == 1:
            self.gap_s = gs
            self.gap_m = gm
            self.gap_min = gap_floor
            self.gap_max = itf.gap_max if itf.gap_max > 0 else np.inf
            hi = (gs.max() if len(gs) else 0.0) + (gm.max() if len(gm)
                                                   else 0.0)
            self.gap_bound = float(np.clip(hi, self.gap_min, self.gap_max))
        else:
            self.gap_const = gap_floor
            self.gap_bound = gap_floor
        self.fric = itf.fric

        # ---- friction MODELS + IFQ filter state (M15, port extension —
        # module docstring). mfrot/ifq = 0: every M4 path bit-identical.
        self.mfrot = int(getattr(itf, "mfrot", 0))
        self.ifq = int(getattr(itf, "ifq", 0))
        self.xfiltr = float(getattr(itf, "xfiltr", 0.0))
        self.fric_c = np.asarray(getattr(itf, "fric_c",
                                         (0.0,) * 6), dtype=float)
        self._filt_keys = np.zeros(0, dtype=np.int64)
        self._filt_vals = np.zeros((0, 3))
        if self.mfrot > 0:
            log.info(f"     /INTER/TYPE11/{itf.id}: FRICTION MODEL "
                     f"MFROT={self.mfrot} (PORT EXTENSION — p = fn/(L*gap),"
                     f" see contact/friction.py), IFQ={self.ifq}")

        # ---- interface time step bound -------------------------------------
        # physical pre-mass-scaling masses: conservative and
        # restart-invariant (M6), like inter_type7
        m_phys = getattr(model, "mass0", None)
        if m_phys is None or len(m_phys) != len(model.mass):
            m_phys = model.mass
        self.dt_bound = self._compute_dt_bound(m_phys)

        # ---- deletion bookkeeping ------------------------------------------
        self.idel = int(getattr(itf, "idel", 0) or 0)
        self.deletable = (self.idel >= 1 and
                          (tracking.any_deletable(model, self.es_gtype)
                           or tracking.any_deletable(model, self.em_gtype)))
        self.es_alive = np.ones(len(self.es), dtype=bool)
        self.em_alive = np.ones(len(self.em), dtype=bool)

        self.pairs_s = np.zeros(0, dtype=np.int64)   # secondary edge rows
        self.pairs_m = np.zeros(0, dtype=np.int64)   # main edge rows
        self._last_refresh = -10**9
        self.refresh = 20
        self.tstart = float(getattr(itf, "tstart", 0.0) or 0.0)
        self.tstop = float(getattr(itf, "tstop", np.inf) or np.inf)
        self.stmin = float(getattr(itf, "stmin", 0.0) or 0.0)
        self.stmax = float(getattr(itf, "stmax", 0.0) or 0.0)
        self.visc = float(getattr(itf, "viss", getattr(itf, "stiff_dc", 0.05)) or 0.05)
        if self.visc <= 0.0:
            self.visc = 0.05
        if self.tstop <= 0.0:
            self.tstop = np.inf

    # ------------------------------------------------------------------
    def _compute_dt_bound(self, mass) -> float:
        """Worst node-on-spring bound over both edge sets (see TYPE7)."""
        if len(self.es) == 0 or len(self.em) == 0:
            return np.inf
        itf = self.itf
        K_s = combine_stiffness(itf.istf, itf.stfac,
                                np.full(len(self.es), self.Km.max() if len(self.Km) > 0 else 0.0),
                                self.Ks)
        m_s = mass[self.es].min(axis=1)
        m_s_pos = m_s > 0.0
        dt_s = np.sqrt(2.0 * m_s[m_s_pos] / np.maximum(K_s[m_s_pos], EM20)).min() if np.any(m_s_pos) else np.inf
        K_m = combine_stiffness(itf.istf, itf.stfac, self.Km,
                                np.full(len(self.em),
                                        self.Ks.max() if len(self.Ks)
                                        else 0.0))
        m_m = mass[self.em].min(axis=1)
        m_m_pos = m_m > 0.0
        dt_m = np.sqrt(2.0 * m_m[m_m_pos] / np.maximum(K_m[m_m_pos], EM20)).min() if np.any(m_m_pos) else np.inf
        return float(min(dt_s, dt_m))

    # ------------------------------------------------------------------
    def _broad_phase(self, x, v, dt):
        """Voxel candidate search over edge bounding boxes (i11buce)."""
        es = self.es[self.es_alive]
        em = self.em[self.em_alive]
        rows_s = np.where(self.es_alive)[0]
        rows_m = np.where(self.em_alive)[0]
        if len(es) == 0 or len(em) == 0:
            self.pairs_s = np.zeros(0, dtype=np.int64)
            self.pairs_m = np.zeros(0, dtype=np.int64)
            return
        margin = self.gap_bound + 2.0 * self.refresh * dt * \
            (np.abs(v).max() if len(v) else 0.0)

        def boxes(edges, infl):
            xe = x[edges]                       # (n, 2, 3)
            return xe.min(axis=1) - infl, xe.max(axis=1) + infl

        lo_m, hi_m = boxes(em, margin)          # inflate the main side only
        lo_s, hi_s = boxes(es, 0.0)
        h = max(float((hi_m - lo_m).max()), float((hi_s - lo_s).max()), EM20)
        origin = np.minimum(lo_m.min(axis=0), lo_s.min(axis=0))

        def bins(lo, hi):
            ilo = np.floor((lo - origin) / h).astype(np.int64)
            ihi = np.floor((hi - origin) / h).astype(np.int64)
            return ilo, ihi

        ilo_m, ihi_m = bins(lo_m, hi_m)
        ilo_s, ihi_s = bins(lo_s, hi_s)
        dims = np.maximum(ihi_m.max(axis=0), ihi_s.max(axis=0)) + 2

        def key(ijk):
            return (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]

        def bin_boxes(ilo, ihi):
            ids, keys = [], []
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        ijk = ilo + np.array([dx, dy, dz])
                        inside = np.all(ijk <= ihi, axis=1)
                        ids.append(np.where(inside)[0])
                        keys.append(key(ijk[inside]))
            return np.concatenate(ids), np.concatenate(keys)

        ids_s, keys_s = bin_boxes(ilo_s, ihi_s)
        ids_m, keys_m = bin_boxes(ilo_m, ihi_m)
        ii, jj = _expand_matches(keys_s, keys_m)
        ps = rows_s[ids_s[ii]]
        pm = rows_m[ids_m[jj]]
        if len(ps):
            pair_key = ps * len(self.em) + pm
            _, first = np.unique(pair_key, return_index=True)
            ps, pm = ps[first], pm[first]
            # edges sharing a node never contact each other (adjacent
            # edges of one shell mesh, and self-exclusion within one line)
            keep = np.ones(len(ps), dtype=bool)
            for a in range(2):
                for b in range(2):
                    keep &= self.es[ps, a] != self.em[pm, b]
            ps, pm = ps[keep], pm[keep]
        self.pairs_s = ps
        self.pairs_m = pm

    # ------------------------------------------------------------------
    def forces(self, x, v, mass, dt, fcont, cycle=0, stifn=None, t=None, **kwargs):
        """Penalty forces for one cycle, scattered into ``fcont``.
        Returns (contact_work_increment, dt_interface) — the same
        contract as ContactType7.forces (the Engine books the exact
        midstep contact energy from ``fcont``, see engine.py; ``stifn``
        is the /DT/NODA nodal-stiffness accumulation, M6)."""
        if len(self.es) == 0 or len(self.em) == 0:
            return 0.0, np.inf
        if dt <= 0.0:
            return 0.0, self.dt_bound if hasattr(self, "dt_bound") else np.inf

        if t is not None:
            if t < self.tstart or t > self.tstop:
                return 0.0, self.dt_bound if hasattr(self, "dt_bound") else np.inf

        if self.deletable:
            self.es_alive = tracking.alive_segment_mask(
                self.model, self.es_gtype, self.es_elem)
            self.em_alive = tracking.alive_segment_mask(
                self.model, self.em_gtype, self.em_elem)

        if cycle - self._last_refresh >= self.refresh:
            self._broad_phase(x, v, dt)
            self._last_refresh = cycle
        if len(self.pairs_s) == 0:
            return 0.0, self.dt_bound

        live = self.es_alive[self.pairs_s] & self.em_alive[self.pairs_m]
        ps = self.pairs_s[live]
        pm = self.pairs_m[live]
        if len(ps) == 0:
            return 0.0, self.dt_bound
        ea = self.es[ps]                                  # (np, 2)
        eb = self.em[pm]

        # ---- narrow phase: exact closest points (i11dst3) ------------------
        s, t, cA, cB = _closest_points_on_segments(
            x[ea[:, 0]], x[ea[:, 1]], x[eb[:, 0]], x[eb[:, 1]])
        dvec = cA - cB
        d = norm3(dvec)

        if self.itf.igap == 1:
            gap = np.clip(self.gap_s[ps] + self.gap_m[pm],
                          self.gap_min, self.gap_max)
        else:
            gap = np.full(len(ps), self.gap_const)

        # ---- interface dt: NEAR-pair stiffness accumulation ----------------
        # springs that may close within the next cycles enter the nodal
        # bound with their stiffnesses SUMMED on shared nodes — see the
        # matching block in inter_type7.py for the full reasoning
        vmax = float(np.abs(v).max()) if len(v) else 0.0
        near = d < gap + np.maximum(gap, 2.0 * vmax * dt)
        if not np.any(near):
            return 0.0, self.dt_bound
        ea, eb = ea[near], eb[near]
        s, t = s[near], t[near]
        d, gap, dvec = d[near], gap[near], dvec[near]
        ps, pm = ps[near], pm[near]

        K = combine_stiffness(self.itf.istf, self.itf.stfac,
                              self.Km[pm], self.Ks[ps])
        if self.stmin > 0.0:
            K = np.maximum(K, self.stmin)
        if self.stmax > 0.0:
            K = np.minimum(K, self.stmax)
        # per-node spring-stiffness sums (bincount = the fast add.at, M7)
        n_nod = len(fcont)
        ea_flat = ea.reshape(-1)
        w_a = np.repeat(K, 2)
        v_a = (ea_flat >= 0) & (ea_flat < n_nod)
        Knode = np.bincount(ea_flat[v_a], weights=w_a[v_a], minlength=n_nod)
        eb_flat = eb.reshape(-1)
        w_b = np.repeat(K, 2)
        v_b = (eb_flat >= 0) & (eb_flat < n_nod)
        Knode += np.bincount(eb_flat[v_b], weights=w_b[v_b], minlength=n_nod)
        loaded = Knode > 0.0
        if np.any(loaded):
            m_loaded = np.maximum(mass[loaded], EM20)
            dt_int = min(self.dt_bound, float(
                np.sqrt(2.0 * m_loaded / Knode[loaded]).min()))
        else:
            dt_int = self.dt_bound
        if stifn is not None:                    # /DT/NODA accumulation
            stifn[loaded] += Knode[loaded]

        pen = gap - d
        active = pen > 0.0
        if not np.any(active):
            return 0.0, dt_int

        ea, eb = ea[active], eb[active]
        K = K[active]                          # per-pair stiffness (Istf)
        s, t = s[active], t[active]
        gap = gap[active]
        pen = pen[active]
        norm_d = norm3(dvec[active])
        deg = norm_d <= EM20
        if np.any(deg):
            e1 = x[ea[:, 1]] - x[ea[:, 0]]
            e2 = x[eb[:, 1]] - x[eb[:, 0]]
            n_cross = np.cross(e1, e2)
            n_cross_norm = norm3(n_cross)
            valid_cross = n_cross_norm > EM20
            fallback = np.where(valid_cross[:, None],
                                n_cross / np.maximum(n_cross_norm, EM20)[:, None],
                                np.array([0.0, 0.0, 1.0]))
            d_safe = np.maximum(norm_d, EM20)
            nvec = np.where(deg[:, None], fallback, dvec[active] / d_safe[:, None])
        else:
            d_safe = np.maximum(d[active], EM20)
            nvec = dvec[active] / d_safe[:, None]

        # relative velocity of the two closest points
        vA = (1 - s)[:, None] * v[ea[:, 0]] + s[:, None] * v[ea[:, 1]]
        vB = (1 - t)[:, None] * v[eb[:, 0]] + t[:, None] * v[eb[:, 1]]
        vrel = vA - vB
        vn = np.einsum("nb,nb->n", vrel, nvec)

        # normal force: spring + damper (mass of the lighter secondary end
        # node sizes the damper, as the node mass does in TYPE7)
        m_ref = mass[ea].min(axis=1)
        C = self.visc * np.sqrt(2.0 * K * m_ref)
        Fn = K * pen - C * np.minimum(vn, 0.0)
        Fvec = Fn[:, None] * nvec

        # Coulomb friction, regularized around zero slip. Ifric > 0 (M15,
        # port extension — module docstring) swaps the constant mu for the
        # MFROT mu(p, v) laws with the edge-pair pressure definition
        # p = Fn/(L_main * gap); Ifiltr applies the IFQ filter. The
        # mu = const, no-filter path is the M4 code verbatim
        # (x + (-a) == x - a exactly in IEEE — bit-identical).
        if self.fric > 0.0 or self.mfrot > 0:
            Fn_pos = np.maximum(Fn, 0.0)
            gap_ref = float(np.mean(gap))
            vt = vrel - vn[:, None] * nvec
            vt_mag = norm3(vt)
            if self.mfrot > 0:
                # p = Fn / (current main-edge length x pair gap) — the
                # documented port DEFINITION of an edge pair's contact
                # pressure (contact/friction.py)
                lm = norm3(x[eb[:, 1]] - x[eb[:, 0]])
                pres = Fn_pos / np.maximum(lm * gap, EM20)
                mu = friction.mu_kinetic(self.mfrot, self.fric,
                                         self.fric_c, pres, vt_mag)
            else:
                mu = self.fric
            v_ref = np.maximum(1e-3 * gap_ref / max(dt, EM20), EM20)
            Ft = mu * Fn_pos * vt_mag / (vt_mag + v_ref)
            ftvec = -(Ft / np.maximum(vt_mag, EM20))[:, None] * vt
            if self.ifq > 0:
                alpha = friction.filter_alpha(self.ifq, self.xfiltr, dt)
                keys = ps[active] * max(len(self.em), 1) + pm[active]
                ftvec, self._filt_keys, self._filt_vals = friction.\
                    apply_filter(keys, ftvec, alpha, self._filt_keys,
                                 self._filt_vals)
            Fvec += ftvec

        # scatter with the closest-point parameters: +F on the secondary
        # edge ends, -F on the main edge ends (collinear equal/opposite
        # forces: linear AND angular momentum conserved)
        va = np.empty((len(s), 2, 3))
        va[:, 0, :] = (1 - s)[:, None] * Fvec
        va[:, 1, :] = s[:, None] * Fvec
        ea_act = ea.reshape(-1)
        va_act = va.reshape(-1, 3)
        v_act_a = (ea_act >= 0) & (ea_act < n_nod)
        scatter_add3(fcont, ea_act[v_act_a], va_act[v_act_a])
        vb = np.empty((len(t), 2, 3))
        vb[:, 0, :] = -(1 - t)[:, None] * Fvec
        vb[:, 1, :] = -t[:, None] * Fvec
        eb_act = eb.reshape(-1)
        vb_act = vb.reshape(-1, 3)
        v_act_b = (eb_act >= 0) & (eb_act < n_nod)
        scatter_add3(fcont, eb_act[v_act_b], vb_act[v_act_b])

        wrk = float(np.einsum("nb,nb->", Fvec, vrel)) * dt
        return -wrk, dt_int

    def compute_thermal_conduction(
        self,
        temp: np.ndarray,
        dt: float,
        kthe: Optional[float] = None,
        frad: Optional[float] = None,
        drad: Optional[float] = None,
        iform: Optional[int] = None,
        tint: Optional[float] = None,
        mat_cond: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute thermal conduction and radiation for /INTER/TYPE11 edge-to-edge contact.

        Ported from $OR_SRC/engine/source/interfaces/int11/i11therm.F

        Parameters
        ----------
        temp : np.ndarray
            Nodal temperatures.
        dt : float
            Time step dt.
        kthe : Optional[float]
            Thermal interface conductivity KTHE.
        frad : Optional[float]
            Radiation coefficient.
        drad : Optional[float]
            Radiation cutoff distance.
        iform : Optional[int]
            0 = ambient exchange, 1 = slave-master exchange.
        tint : Optional[float]
            Ambient temperature.
        mat_cond : Optional[np.ndarray]
            Material conductivity per pair.

        Returns
        -------
        fthe : np.ndarray
            Nodal thermal energy increments [J].
        condint : np.ndarray
            Thermal conductance per pair [W/K].
        heat_transferred : float
            Total thermal energy transferred across edges [J].
        """
        from .thermal_contact import thermal_contact_type11

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

        if len(self.pairs_s) == 0:
            return np.zeros(len(temp), dtype=float), np.zeros(0, dtype=float), 0.0

        live = self.es_alive[self.pairs_s] & self.em_alive[self.pairs_m]
        ps = self.pairs_s[live]
        pm = self.pairs_m[live]
        if len(ps) == 0:
            return np.zeros(len(temp), dtype=float), np.zeros(0, dtype=float), 0.0

        ea = self.es[ps]
        eb = self.em[pm]
        x = getattr(self.model, "x", getattr(self.model, "x0", np.zeros((len(temp), 3))))

        s, t, cA, cB = _closest_points_on_segments(
            x[ea[:, 0]], x[ea[:, 1]], x[eb[:, 0]], x[eb[:, 1]]
        )
        d = norm3(cA - cB)

        if self.itf.igap == 1:
            gap = self.gap_s[ps] + self.gap_m[pm]
            if self.gap_min > 0.0:
                gap = np.maximum(gap, self.gap_min)
            if self.gap_max < np.inf:
                gap = np.minimum(gap, self.gap_max)
        else:
            gap = np.full(len(ps), self.gap_const)

        penrad = d - gap
        hs = np.column_stack([1.0 - s, s])
        hm = np.column_stack([1.0 - t, t])

        # Tributary area: length of edge times gap
        l_ea = norm3(x[ea[:, 1]] - x[ea[:, 0]])
        l_eb = norm3(x[eb[:, 1]] - x[eb[:, 0]])
        areac = 0.5 * (l_ea + l_eb) * gap

        return thermal_contact_type11(
            temp=temp,
            slave_edge_nodes=ea,
            master_edge_nodes=eb,
            hs=hs,
            hm=hm,
            kthe=kthe,
            dt=dt,
            areac=areac,
            penrad=penrad,
            gapv=gap,
            frad=frad,
            drad=drad,
            iform=iform,
            tint=tint,
            mat_cond=mat_cond,
        )


class LagmulType11:
    """One /INTER/LAGMUL/TYPE11 edge-to-edge constraint, engine-side."""

    def __init__(self, itf, model: Model, log=None):
        self.itf = itf
        self.model = model
        self.log = log if log is not None else getattr(model, "log", None)
        self.penalty_handler = ContactType11(itf, model, self.log)
        if hasattr(itf, "params") and isinstance(itf.params, dict):
            if "line1" in itf.params and "line2" in itf.params:
                self.penalty_handler.es = np.asarray(itf.params["line1"], dtype=np.int64)
                self.penalty_handler.em = np.asarray(itf.params["line2"], dtype=np.int64)
                self.penalty_handler.es_alive = np.ones(len(self.penalty_handler.es), dtype=bool)
                self.penalty_handler.em_alive = np.ones(len(self.penalty_handler.em), dtype=bool)
                gap_val = float(itf.params.get("gap", 0.05))
                self.penalty_handler.gap_const = gap_val
                self.penalty_handler.gap_bound = gap_val
                n_s = len(self.penalty_handler.es)
                n_m = len(self.penalty_handler.em)
                ps, pm = np.meshgrid(np.arange(n_s, dtype=np.int64), np.arange(n_m, dtype=np.int64), indexing="ij")
                self.penalty_handler.pairs_s = ps.ravel()
                self.penalty_handler.pairs_m = pm.ravel()

    def generate_l_constraint_rows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate constraint rows (L_data, L_row, L_col) for edge-to-edge Lagrange multiplier contact.

        Constraint condition: normal gap rate (v_S - v_M) . n <= 0 for penetrating edge pairs.
        Returns:
            L_data: Non-zero values of the constraint matrix L.
            L_row:  Row indices (equation index).
            L_col:  Column indices (node_index * 3 + dof).
        """
        res = self.generate_l_matrix(1e-6)
        if len(res) == 3:
            return res
        data, nodes, dofs, eq_ids, n_rows = res
        if n_rows == 0:
            return (np.zeros(0, dtype=np.float64),
                    np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64))
        L_data = data
        L_row = eq_ids
        L_col = nodes * 3 + dofs
        return L_data, L_row, L_col

    def generate_l_matrix(self, dt=None):
        """Yield (data, node_indices, dof_indices, eq_indices, n_rows) for global LagmulSolver.
        If dt is provided, returns (L_data, L_row, L_col).

        Only penetrating edge pairs approaching each other (vn <= 0) generate constraints.
        """
        handler = self.penalty_handler
        x = getattr(self.model, "x", getattr(self.model, "x0", None))
        if x is None or len(x) == 0:
            if dt is not None:
                return (np.zeros(0, dtype=np.float64),
                        np.zeros(0, dtype=np.int64),
                        np.zeros(0, dtype=np.int64))
            return (np.zeros(0, dtype=float), np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0)

        v = getattr(self.model, "v", None)
        if v is None or len(v) != len(x):
            v = np.zeros_like(x)

        dt_val = float(dt) if (dt is not None and isinstance(dt, (int, float))) else getattr(self.model, "dt", 1e-6)

        if len(handler.es) == 0 or len(handler.em) == 0:
            if dt is not None:
                return (np.zeros(0, dtype=np.float64),
                        np.zeros(0, dtype=np.int64),
                        np.zeros(0, dtype=np.int64))
            return (np.zeros(0, dtype=float), np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0)

        if handler.deletable:
            handler.es_alive = tracking.alive_segment_mask(
                handler.model, handler.es_gtype, handler.es_elem)
            handler.em_alive = tracking.alive_segment_mask(
                handler.model, handler.em_gtype, handler.em_elem)

        cycle = getattr(self.model, "cycle", 0)
        if cycle - handler._last_refresh >= handler.refresh:
            handler._broad_phase(x, v, dt_val)
            handler._last_refresh = cycle

        if len(handler.pairs_s) == 0:
            if dt is not None:
                return (np.zeros(0, dtype=np.float64),
                        np.zeros(0, dtype=np.int64),
                        np.zeros(0, dtype=np.int64))
            return (np.zeros(0, dtype=float), np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0)

        live = handler.es_alive[handler.pairs_s] & handler.em_alive[handler.pairs_m]
        ps = handler.pairs_s[live]
        pm = handler.pairs_m[live]
        if len(ps) == 0:
            if dt is not None:
                return (np.zeros(0, dtype=np.float64),
                        np.zeros(0, dtype=np.int64),
                        np.zeros(0, dtype=np.int64))
            return (np.zeros(0, dtype=float), np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0)

        ea = handler.es[ps]
        eb = handler.em[pm]

        s, t, cA, cB = _closest_points_on_segments(
            x[ea[:, 0]], x[ea[:, 1]], x[eb[:, 0]], x[eb[:, 1]])
        dvec = cA - cB
        d = norm3(dvec)

        if handler.itf.igap == 1:
            gap = handler.gap_s[ps] + handler.gap_m[pm]
            if handler.gap_min > 0.0:
                gap = np.maximum(gap, handler.gap_min)
            if handler.gap_max < np.inf:
                gap = np.minimum(gap, handler.gap_max)
        else:
            gap = np.full(len(ps), handler.gap_const)

        pen = gap - d
        active = pen > 0.0
        if not np.any(active):
            if dt is not None:
                return (np.zeros(0, dtype=np.float64),
                        np.zeros(0, dtype=np.int64),
                        np.zeros(0, dtype=np.int64))
            return (np.zeros(0, dtype=float), np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0)

        ea = ea[active]
        eb = eb[active]
        s = s[active]
        t = t[active]
        d_act = d[active]
        dvec_act = dvec[active]

        norm_d = norm3(dvec_act)
        deg = norm_d <= EM20
        if np.any(deg):
            e1 = x[ea[:, 1]] - x[ea[:, 0]]
            e2 = x[eb[:, 1]] - x[eb[:, 0]]
            n_cross = np.cross(e1, e2)
            n_cross_norm = norm3(n_cross)
            valid_cross = n_cross_norm > EM20
            fallback = np.where(valid_cross[:, None],
                                n_cross / np.maximum(n_cross_norm, EM20)[:, None],
                                np.array([0.0, 0.0, 1.0]))
            d_safe = np.maximum(norm_d, EM20)
            nvec = np.where(deg[:, None], fallback, dvec_act / d_safe[:, None])
        else:
            d_safe = np.maximum(d_act, EM20)
            nvec = dvec_act / d_safe[:, None]

        # Relative velocity between closest points on secondary and master edges
        vA = (1.0 - s)[:, None] * v[ea[:, 0]] + s[:, None] * v[ea[:, 1]]
        vB = (1.0 - t)[:, None] * v[eb[:, 0]] + t[:, None] * v[eb[:, 1]]
        vrel = vA - vB
        vn = np.einsum("nb,nb->n", vrel, nvec)

        # Approaching edges condition: vn <= 0
        approaching = vn <= 0.0
        if not np.any(approaching):
            if dt is not None:
                return (np.zeros(0, dtype=np.float64),
                        np.zeros(0, dtype=np.int64),
                        np.zeros(0, dtype=np.int64))
            return (np.zeros(0, dtype=float), np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0)

        ea_app = ea[approaching]
        eb_app = eb[approaching]
        s_app = s[approaching]
        t_app = t[approaching]
        nvec_app = nvec[approaching]
        n_rows = len(ea_app)

        data: list[float] = []
        nodes: list[int] = []
        dofs: list[int] = []
        eq_ids: list[int] = []

        for i in range(n_rows):
            eq_id = i
            s1, s2 = int(ea_app[i, 0]), int(ea_app[i, 1])
            m1, m2 = int(eb_app[i, 0]), int(eb_app[i, 1])
            si = float(s_app[i])
            ti = float(t_app[i])
            nx, ny, nz = float(nvec_app[i, 0]), float(nvec_app[i, 1]), float(nvec_app[i, 2])

            for dof, n_dof in enumerate((nx, ny, nz)):
                # Secondary edge nodes (+v_S . n):
                # node s1 with weight (1 - s)
                data.append((1.0 - si) * n_dof)
                nodes.append(s1)
                dofs.append(dof)
                eq_ids.append(eq_id)

                # node s2 with weight s
                data.append(si * n_dof)
                nodes.append(s2)
                dofs.append(dof)
                eq_ids.append(eq_id)

                # Master edge nodes (-v_M . n):
                # node m1 with weight -(1 - t)
                data.append(-(1.0 - ti) * n_dof)
                nodes.append(m1)
                dofs.append(dof)
                eq_ids.append(eq_id)

                # node m2 with weight -t
                data.append(-ti * n_dof)
                nodes.append(m2)
                dofs.append(dof)
                eq_ids.append(eq_id)

        if dt is not None:
            L_data = np.asarray(data, dtype=np.float64)
            L_row = np.asarray(eq_ids, dtype=np.int64)
            L_col = np.asarray(nodes, dtype=np.int64) * 3 + np.asarray(dofs, dtype=np.int64)
            return L_data, L_row, L_col

        return (
            np.asarray(data, dtype=np.float64),
            np.asarray(nodes, dtype=np.int64),
            np.asarray(dofs, dtype=np.int64),
            np.asarray(eq_ids, dtype=np.int64),
            n_rows,
        )

