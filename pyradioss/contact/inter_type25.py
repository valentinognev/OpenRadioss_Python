"""
/INTER/TYPE25 — General segment-to-segment, edge-to-segment, and edge-to-edge contact
with cohesive adhesion (sigmaxadh) and Coulomb friction.

Fortran origin:
  OpenRadioss engine/source/interfaces/int25/:
    i25mainf.F      Interface dispatcher and candidate management
    i25for3.F       Segment-to-segment contact with cohesive adhesion
    i25for3_e2s.F   Edge-to-segment contact
    i25for3e.F      Edge-to-edge contact
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import numpy as np

from ..common.constants import EM20
from ..common.fastmath import norm3, scatter_add3
from ..model.model import Model
from . import tracking
from .inter_type11 import _closest_points_on_segments


def _closest_point_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Vectorized closest point on triangle (a, b, c) to point p.

    Returns (dist, closest_pt, weights (w_a, w_b, w_c)).
    """
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.einsum("nb,nb->n", ab, ap)
    d2 = np.einsum("nb,nb->n", ac, ap)

    # Vertex region outside A
    mask_a = (d1 <= 0.0) & (d2 <= 0.0)

    # Vertex region outside B
    bp = p - b
    d3 = np.einsum("nb,nb->n", ab, bp)
    d4 = np.einsum("nb,nb->n", ac, bp)
    mask_b = (d3 >= 0.0) & (d4 <= d3)

    # Edge region AB
    vc = d1 * d4 - d3 * d2
    mask_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)

    # Vertex region outside C
    cp = p - c
    d5 = np.einsum("nb,nb->n", ab, cp)
    d6 = np.einsum("nb,nb->n", ac, cp)
    mask_c = (d6 >= 0.0) & (d5 <= d6)

    # Edge region AC
    vb = d5 * d2 - d1 * d6
    mask_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)

    # Edge region BC
    va = d3 * d6 - d5 * d4
    mask_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)

    # Face region inside ABC
    denom = va + vb + vc
    denom_safe = np.where(denom > EM20, denom, 1.0)
    u = va / denom_safe
    v_coord = vb / denom_safe
    w_coord = 1.0 - u - v_coord

    closest = np.zeros_like(p)
    w = np.zeros((len(p), 3), dtype=float)

    # A
    if np.any(mask_a):
        closest[mask_a] = a[mask_a]
        w[mask_a] = [1.0, 0.0, 0.0]

    # B
    if np.any(mask_b):
        closest[mask_b] = b[mask_b]
        w[mask_b] = [0.0, 1.0, 0.0]

    # AB
    if np.any(mask_ab):
        t_ab = d1[mask_ab] / np.maximum(d1[mask_ab] - d3[mask_ab], EM20)
        closest[mask_ab] = a[mask_ab] + t_ab[:, None] * ab[mask_ab]
        w[mask_ab, 0] = 1.0 - t_ab
        w[mask_ab, 1] = t_ab

    # C
    if np.any(mask_c):
        closest[mask_c] = c[mask_c]
        w[mask_c] = [0.0, 0.0, 1.0]

    # AC
    if np.any(mask_ac):
        t_ac = d2[mask_ac] / np.maximum(d2[mask_ac] - d6[mask_ac], EM20)
        closest[mask_ac] = a[mask_ac] + t_ac[:, None] * ac[mask_ac]
        w[mask_ac, 0] = 1.0 - t_ac
        w[mask_ac, 2] = t_ac

    # BC
    if np.any(mask_bc):
        t_bc = (d4[mask_bc] - d3[mask_bc]) / np.maximum((d4[mask_bc] - d3[mask_bc]) + (d5[mask_bc] - d6[mask_bc]), EM20)
        closest[mask_bc] = b[mask_bc] + t_bc[:, None] * (c[mask_bc] - b[mask_bc])
        w[mask_bc, 1] = 1.0 - t_bc
        w[mask_bc, 2] = t_bc

    # Interior
    interior = ~(mask_a | mask_b | mask_ab | mask_c | mask_ac | mask_bc)
    if np.any(interior):
        closest[interior] = (w_coord[interior, None] * a[interior] +
                             u[interior, None] * b[interior] +
                             v_coord[interior, None] * c[interior])
        w[interior, 0] = w_coord[interior]
        w[interior, 1] = u[interior]
        w[interior, 2] = v_coord[interior]

    diff = p - closest
    dist = norm3(diff)
    return dist, closest, w


class ContactType25:
    """General contact interface /INTER/TYPE25 with cohesive adhesion (sigmaxadh).

    Supports:
    - Segment-to-segment / node-to-segment with cohesive adhesion spring laws
    - Edge-to-edge contact
    - Edge-to-segment contact
    """

    def __init__(self, itf: Any, model: Model, log: Any = None):
        self.itf = itf
        self.model = model
        self.log = log if log is not None else getattr(model, "log", None)

        params = getattr(itf, "params", {}) or {}
        self.stfac = float(params.get("stfac", getattr(itf, "stfac", 1.0)) or 1.0)
        self.fric = float(params.get("mu", params.get("fric", getattr(itf, "fric", 0.0))) or 0.0)
        self.gap = float(params.get("gap", getattr(itf, "gap", 0.0)) or 0.0)
        self.sigmaxadh = float(params.get("sigmaxadh", getattr(itf, "sigmaxadh", 0.0)) or 0.0)
        self.viscadhfact = float(params.get("viscadhfact", getattr(itf, "viscadhfact", 1.0)) or 1.0)
        self.visc = float(params.get("visc", getattr(itf, "visc", 0.05)) or 0.05)
        self.stmin = float(getattr(itf, "stmin", 0.0) or 0.0)
        self.stmax = float(getattr(itf, "stmax", np.inf) or np.inf)
        self.tstart = float(getattr(itf, "tstart", 0.0) or 0.0)
        self.tstop = float(getattr(itf, "tstop", np.inf) or np.inf)
        if self.tstop <= 0.0:
            self.tstop = np.inf
        self.dt_bound = np.inf

        self.secondary_nodes = np.zeros(0, dtype=np.int64)
        self.master_segments = np.zeros((0, 4), dtype=np.int64)
        self.secondary_edges = np.zeros((0, 2), dtype=np.int64)
        self.master_edges = np.zeros((0, 2), dtype=np.int64)

        self._init_entities()

    def _init_entities(self) -> None:
        """Resolve secondary and master entities."""
        params = getattr(self.itf, "params", {}) or {}
        # 1. Surface IDs
        surf_m = getattr(self.itf, "surf_id", getattr(self.itf, "main_id", 0))
        surf_s = getattr(self.itf, "grnod_id", getattr(self.itf, "grnd_id", 0))
        if not surf_s:
            for s_cand in ("surf_id2", "surf_id1", "sec_id"):
                val = getattr(self.itf, s_cand, 0)
                if val and val != surf_m:
                    surf_s = val
                    break

        if "secondary_nodes" in params and params["secondary_nodes"] is not None:
            self.secondary_nodes = np.asarray(params["secondary_nodes"], dtype=np.int64)
        elif hasattr(self.itf, "secondary_nodes") and self.itf.secondary_nodes is not None:
            self.secondary_nodes = np.asarray(self.itf.secondary_nodes, dtype=np.int64)
        elif surf_s > 0:
            if hasattr(self.model, "node_groups") and surf_s in self.model.node_groups:
                ng = self.model.node_groups[surf_s]
                if getattr(ng, "node_idx", None) is not None:
                    self.secondary_nodes = np.asarray(ng.node_idx, dtype=np.int64)
                elif getattr(ng, "nodes", None) is not None:
                    self.secondary_nodes = np.asarray(ng.nodes, dtype=np.int64)
            elif hasattr(self.model, "surfaces") and surf_s in self.model.surfaces:
                s = self.model.surfaces[surf_s]
                if getattr(s, "nodes", None) is not None:
                    self.secondary_nodes = np.asarray(s.nodes, dtype=np.int64)
                elif getattr(s, "segments", None) is not None and len(s.segments) > 0:
                    self.secondary_nodes = np.unique(s.segments.reshape(-1))

        # 2. Master segments
        if "master_segments" in params and params["master_segments"] is not None:
            self.master_segments = np.asarray(params["master_segments"], dtype=np.int64)
        elif hasattr(self.itf, "master_segments") and self.itf.master_segments is not None:
            self.master_segments = np.asarray(self.itf.master_segments, dtype=np.int64)
        elif surf_m > 0 and hasattr(self.model, "surfaces") and surf_m in self.model.surfaces:
            sm = self.model.surfaces[surf_m]
            if getattr(sm, "segments", None) is not None and len(sm.segments) > 0:
                segs = sm.segments
                if segs.shape[1] == 3:
                    segs = np.column_stack([segs, segs[:, 2]])
                self.master_segments = np.asarray(segs[:, :4], dtype=np.int64)

        if len(self.secondary_nodes) == 0 and len(self.master_segments) == 0 and getattr(self.itf, "surf_id", 0) > 0:
            # Self-contact fallback
            s_id = self.itf.surf_id
            if hasattr(self.model, "surfaces") and s_id in self.model.surfaces:
                sm = self.model.surfaces[s_id]
                if getattr(sm, "segments", None) is not None and len(sm.segments) > 0:
                    segs = sm.segments
                    if segs.shape[1] == 3:
                        segs = np.column_stack([segs, segs[:, 2]])
                    self.master_segments = np.asarray(segs[:, :4], dtype=np.int64)
                    self.secondary_nodes = np.unique(segs.reshape(-1))

        # 3. Master and secondary edges (for edge-to-edge / edge-to-segment)
        if hasattr(self.itf, "secondary_edges") and self.itf.secondary_edges is not None:
            self.secondary_edges = np.asarray(self.itf.secondary_edges, dtype=np.int64)
        elif len(self.master_segments) > 0 and len(self.secondary_edges) == 0:
            # Generate edges from segment boundaries
            edges = []
            for s in self.master_segments:
                edges.extend([[s[0], s[1]], [s[1], s[2]], [s[2], s[3]], [s[3], s[0]]])
            self.master_edges = np.unique(np.sort(np.asarray(edges, dtype=np.int64), axis=1), axis=0)

    def forces(self, x: np.ndarray, v: np.ndarray, mass: np.ndarray, dt: float,
               fcont: np.ndarray, cycle: int = 0, stifn: Optional[np.ndarray] = None,
               t: Optional[float] = None) -> Tuple[float, float]:
        """Compute penalty contact forces with cohesive adhesion.

        Fortran reference:
          engine/source/interfaces/int25/i25for3.F lines 347-365.
          engine/source/interfaces/int25/i25for3e.F.

        Returns (-work, dt_contact).
        """
        if dt <= 0.0:
            return 0.0, self.dt_bound

        if t is not None and (t < self.tstart or t > self.tstop):
            return 0.0, self.dt_bound

        n_coords = len(x)
        total_work = 0.0
        dt_min = self.dt_bound

        # ---------------------------------------------------------------------
        # Phase 1: Segment-to-segment / node-to-segment contact with cohesive adhesion
        # ---------------------------------------------------------------------
        if len(self.secondary_nodes) > 0 and len(self.master_segments) > 0:
            valid_sn = self.secondary_nodes[(self.secondary_nodes >= 0) & (self.secondary_nodes < n_coords)]
            valid_segs = np.all((self.master_segments >= 0) & (self.master_segments < n_coords), axis=1)
            msegs = self.master_segments[valid_segs]

            if len(valid_sn) > 0 and len(msegs) > 0:
                sn_x = x[valid_sn]
                m_x = x[msegs]  # (n_segs, 4, 3)

                # Broad phase AABB
                m_min = m_x.min(axis=1) - (self.gap + 1e-4)
                m_max = m_x.max(axis=1) + (self.gap + 1e-4)

                for i, pt in enumerate(sn_x):
                    snode = valid_sn[i]
                    inside = np.all((pt >= m_min) & (pt <= m_max), axis=1)
                    cand_segs = np.where(inside)[0]

                    for sj in cand_segs:
                        seg = msegs[sj]
                        # Split quad into two triangles (0, 1, 2) and (0, 2, 3)
                        p_arr = pt[None, :]
                        d1, c1, w1 = _closest_point_on_triangle(p_arr, x[seg[0:1]], x[seg[1:2]], x[seg[2:3]])
                        d2, c2, w2 = _closest_point_on_triangle(p_arr, x[seg[0:1]], x[seg[2:3]], x[seg[3:4]])

                        if d1[0] <= d2[0]:
                            dist = float(d1[0])
                            c_pt = c1[0]
                            H = np.array([w1[0, 0], w1[0, 1], w1[0, 2], 0.0])
                        else:
                            dist = float(d2[0])
                            c_pt = c2[0]
                            H = np.array([w2[0, 0], 0.0, w2[0, 1], w2[0, 2]])

                        # Master segment outward normal from diagonal cross product
                        d13 = x[seg[2]] - x[seg[0]]
                        d24 = x[seg[3]] - x[seg[1]]
                        n_surf = np.cross(d13, d24)
                        n_surf_norm = float(np.linalg.norm(n_surf))
                        if n_surf_norm > EM20:
                            n_surf /= n_surf_norm
                        else:
                            n_surf = np.array([0.0, 0.0, 1.0])

                        # Signed distance along outward normal (positive outside, negative penetrated)
                        dist_signed = float(np.dot(pt - c_pt, n_surf))
                        pen = self.gap - dist_signed

                        # Base tributary area
                        area_trib = float(n_surf_norm) * 0.5 * 0.25

                        m_node = float(mass[snode]) if len(mass) > snode else 1.0
                        k_base = 0.1 * m_node / (dt * dt) if dt > 0.0 else 1e6
                        K = self.stfac * k_base
                        if self.stmin > 0.0:
                            K = max(K, self.stmin)
                        if self.stmax < np.inf:
                            K = min(K, self.stmax)

                        # Cohesive Adhesion Model (Fortran i25for3.F lines 347-365)
                        fn_scalar = 0.0
                        if self.sigmaxadh > 0.0:
                            base_adh = max(self.gap, 1e-6)
                            stif_adh = (self.sigmaxadh * area_trib) / max(base_adh, 1e-30)

                            if pen < base_adh:
                                # Inside adhesion zone: attractive force pulling toward surface (-n_surf)
                                if pen > -base_adh:
                                    fn_scalar = -stif_adh * (base_adh - pen)
                            else:
                                # Inside compressive penetration zone: repulsive force pushing out (+n_surf)
                                fn_scalar = K * (pen - base_adh)
                        else:
                            # Standard contact repulsion: push out (+n_surf)
                            if pen > 0.0:
                                fn_scalar = K * pen

                        if abs(fn_scalar) <= 1e-20:
                            continue

                        # Damping
                        v_s = v[snode]
                        v_m = np.sum(H[:, None] * v[seg], axis=0)
                        v_rel = v_s - v_m
                        vn = float(np.dot(v_rel, n_surf))
                        c_damp = 2.0 * self.visc * np.sqrt(max(K * m_node, 1e-20))
                        if fn_scalar > 0.0:
                            fn_scalar = max(0.0, fn_scalar - c_damp * min(vn, 0.0))

                        f_total = fn_scalar * n_surf

                        # Coulomb friction
                        if self.fric > 0.0 and abs(fn_scalar) > 0.0:
                            vt = v_rel - vn * n_surf
                            vt_mag = np.linalg.norm(vt)
                            if vt_mag > 1e-12:
                                ft_mag = min(self.fric * abs(fn_scalar), 0.5 * m_node * vt_mag / dt)
                                f_total -= (ft_mag / vt_mag) * vt

                        # Distribute forces: +F on secondary node, -H_k * F on master segment nodes
                        fcont[snode] += f_total
                        for k in range(4):
                            fcont[seg[k]] -= H[k] * f_total

                        if stifn is not None:
                            stifn[snode] += K
                            for k in range(4):
                                stifn[seg[k]] += H[k] * K

                        dt_cand = np.sqrt(2.0 * m_node / max(K, 1e-20))
                        dt_min = min(dt_min, float(dt_cand))
                        total_work += float(np.dot(f_total, v_rel)) * dt

        # ---------------------------------------------------------------------
        # Phase 2: Edge-to-edge contact (i25for3e.F)
        # ---------------------------------------------------------------------
        if len(self.secondary_edges) > 0 and len(self.master_edges) > 0:
            es = self.secondary_edges
            em = self.master_edges
            valid_es = np.all((es >= 0) & (es < n_coords), axis=1)
            valid_em = np.all((em >= 0) & (em < n_coords), axis=1)
            es = es[valid_es]
            em = em[valid_em]

            if len(es) > 0 and len(em) > 0:
                for edge_s in es:
                    p1, q1 = x[edge_s[0]], x[edge_s[1]]
                    for edge_m in em:
                        p2, q2 = x[edge_m[0]], x[edge_m[1]]

                        s, t, cA, cB = _closest_points_on_segments(p1[None, :], q1[None, :], p2[None, :], q2[None, :])
                        diff = cA[0] - cB[0]
                        d = float(np.linalg.norm(diff))
                        pen = self.gap - d

                        if pen > 0.0:
                            n_vec = diff / d if d > EM20 else np.array([0.0, 0.0, 1.0])
                            m_eff = min(mass[edge_s[0]], mass[edge_m[0]]) if len(mass) > edge_s[0] else 1.0
                            K = self.stfac * 0.1 * m_eff / (dt * dt)

                            fn = K * pen
                            f_edge = fn * n_vec

                            si = float(s[0])
                            ti = float(t[0])

                            # Scatter: +(1-s) and +s on secondary edge, -(1-t) and -t on master edge
                            fcont[edge_s[0]] += (1.0 - si) * f_edge
                            fcont[edge_s[1]] += si * f_edge
                            fcont[edge_m[0]] -= (1.0 - ti) * f_edge
                            fcont[edge_m[1]] -= ti * f_edge

                            dt_cand = np.sqrt(2.0 * m_eff / max(K, 1e-20))
                            dt_min = min(dt_min, float(dt_cand))

        return -total_work, dt_min

    def compute_thermal_conduction(
        self,
        temp: np.ndarray,
        dt: float,
        kthe: Optional[float] = None,
        dcond: float = 0.0,
        fcond: Optional[Any] = None,
        frad: float = 0.0,
        drad: float = 0.0,
        iform: int = 1,
        tint: float = 293.15,
        cond_slave: Optional[np.ndarray] = None,
        cond_master: Optional[np.ndarray] = None,
        fheats: float = 0.0,
        fheatm: float = 0.0,
        efrict: Optional[np.ndarray] = None,
        theaccfact: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Compute general interface thermal contact conduction and radiation for /INTER/TYPE25.

        Ported from $OR_SRC/engine/source/interfaces/int25/i25therm.F

        Parameters
        ----------
        temp : np.ndarray
            Nodal temperatures.
        dt : float
            Time step dt.
        kthe : Optional[float]
            Thermal interface conductivity KTHE.
        dcond : float
            Distance decay threshold.
        fcond : Optional[Any]
            Decay curve.
        frad : float
            Radiation coefficient.
        drad : float
            Radiation distance cutoff.
        iform : int
            0 = ambient exchange, 1 = slave-master exchange.
        tint : float
            Ambient temperature.
        cond_slave : Optional[np.ndarray]
            Thermal conductivity of slave elements.
        cond_master : Optional[np.ndarray]
            Thermal conductivity of master elements.
        fheats : float
            Fraction of friction heat to slave.
        fheatm : float
            Fraction of friction heat to master.
        efrict : Optional[np.ndarray]
            Friction energy dissipation per pair over dt.
        theaccfact : float
            Thermal acceleration factor.

        Returns
        -------
        fthe : np.ndarray
            Nodal thermal energy increments [J].
        condint : np.ndarray
            Thermal conductance per pair [W/K].
        ledger : Dict[str, float]
            Conduction, radiation, and friction energy breakdown.
        """
        from .thermal_contact import thermal_contact_type25

        itf = self.itf
        if kthe is None:
            kthe = getattr(itf, "kthe", 0.0) or getattr(itf, "rstif", 0.0)

        x = getattr(self.model, "x", getattr(self.model, "x0", np.zeros((len(temp), 3))))

        segs_s = getattr(self, "segs_s", None)
        segs_m = getattr(self, "segs_m", getattr(self, "master_segments", np.zeros((0, 4), dtype=np.int64)))
        if segs_s is not None and len(segs_s) > 0:
            s_nodes = np.unique(segs_s.ravel())
        else:
            s_nodes = getattr(self, "secondary_nodes", np.zeros(0, dtype=np.int64))

        if len(s_nodes) == 0 or len(segs_m) == 0:
            return np.zeros(len(temp), dtype=float), np.zeros(0, dtype=float), {"conduction": 0.0, "radiation": 0.0, "friction": 0.0}

        n_pairs = len(s_nodes)
        if segs_m.shape[1] == 3:
            m_segs = np.column_stack([segs_m, segs_m[:, 2]])
        else:
            m_segs = segs_m

        m_segs_pair = np.tile(m_segs[0], (n_pairs, 1))
        weights = np.full((n_pairs, 4), 0.25, dtype=float)

        gap_val = getattr(self, "gap", getattr(itf, "gap", 0.01))

        return thermal_contact_type25(
            x=x,
            temp=temp,
            slave_nodes=s_nodes,
            master_segs=m_segs_pair,
            weights=weights,
            kthe=kthe,
            dt=dt,
            theaccfact=theaccfact,
            iform=iform,
            tint=tint,
            gapv=np.full(n_pairs, gap_val),
            dcond=dcond,
            fcond=fcond,
            frad=frad,
            drad=drad,
            cond_slave=cond_slave,
            cond_master=cond_master,
            fheats=fheats,
            fheatm=fheatm,
            efrict=efrict,
        )

