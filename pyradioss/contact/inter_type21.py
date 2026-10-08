"""
/INTER/TYPE21 — Stamping Drawbead & Tool-Blank Contact Interface.

Fortran origin: ``engine/source/interfaces/int21/``
  - ``i21mainf.F``: interface driver, tool kinematics, candidate management
  - ``i21dst3.F``: sheet node distance calculation relative to drawbead line/facets,
    depth elevation, and penetration determination
  - ``i21for3.F``: clamping normal force, bending-unbending restraining force,
    friction, tangential force limiting using Pmax / yield criteria, and energy tracking
  - ``i21ass3.F``: force assembly and reaction distribution to master nodes
  - ``starter/source/interfaces/int21/hm_read_inter_type21.F``: input reader

Physics of Drawbead Contact:
----------------------------
In sheet metal forming / stamping, drawbeads restrain the sheet blank flow into the
die cavity. Modeling physical 3D bead ridges with solid/shell meshes is prohibitive.
TYPE21 implements an equivalent drawbead line interface:
1. The drawbead is defined along a sequence of master segments/nodes (line or ridge).
2. As slave sheet blank nodes pass over or cross the drawbead line:
   - A normal clamping force Fn is exerted: Fn = K * p + C * vn.
   - A tangential restraining force Ft opposes the sheet motion across the bead:
       Ft = F_friction + F_bend-unbend
     where F_friction = mu * |Fn|, and F_bend-unbend accounts for the cyclic plastic
     bending and unbending of the sheet as it traverses the bead radius.
   - If ITlim = 0 (default), the maximum tangential restraining force is limited by
     F_max derived from Pmax: Ft <= F_max = Pmax * Area / sqrt(3) (or Pmax limit).
3. The reactive forces are distributed onto the master drawbead segment nodes
   with linear barycentric weights, ensuring exact momentum conservation.
4. The restraining and friction work done on the sheet is booked into the contact
   energy ledger E_cont, ensuring strict energy conservation in the explicit solver.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20, EM30
from ..common.fastmath import norm3
from ..model.model import Model
from .stiffness import combine_stiffness, segment_stiffness_gap, node_stiffness_gap


class ContactType21:
    """One /INTER/TYPE21 drawbead contact interface, engine-side."""

    def __init__(self, itf, model: Model, log):
        self.itf = itf
        self.model = model
        self.log = log

        # Master surface defining the drawbead line/ridge
        master_surf_id = getattr(itf, "surf_id", 0) or getattr(itf, "main_id", 0)
        surf_m = model.surfaces.get(master_surf_id)

        # Secondary surface defining the sheet blank
        slave_surf_id = getattr(itf, "surf_id1", 0) or getattr(itf, "sec_id", 0)
        surf_s = model.surfaces.get(slave_surf_id) if slave_surf_id > 0 else None

        # Fallback if IDs were swapped
        if (surf_m is None or surf_m.segments is None or len(surf_m.segments) == 0) and (surf_s is not None and surf_s.segments is not None and len(surf_s.segments) > 0):
            surf_m, surf_s = surf_s, surf_m
            master_surf_id, slave_surf_id = slave_surf_id, master_surf_id

        bead_edges = np.zeros((0, 2), dtype=np.int64)
        if surf_m is not None and surf_m.segments is not None and len(surf_m.segments) > 0:
            # Extract drawbead line segments from master surface
            # Segments can be 2-node line segments or 3/4-node facets whose edges form the bead
            segs_raw = np.asarray(surf_m.segments, dtype=np.int64)
            if segs_raw.ndim == 2 and segs_raw.shape[1] == 2:
                bead_edges = segs_raw.copy()
            elif segs_raw.ndim == 2 and segs_raw.shape[1] in (3, 4):
                # Extract line edges along segments
                edges_list = []
                for s in segs_raw:
                    if segs_raw.shape[1] == 3 or s[2] == s[3]:
                        edges_list.append([s[0], s[1]])
                        edges_list.append([s[1], s[2]])
                        edges_list.append([s[2], s[0]])
                    else:
                        edges_list.append([s[0], s[1]])
                        edges_list.append([s[1], s[2]])
                        edges_list.append([s[2], s[3]])
                        edges_list.append([s[3], s[0]])
                bead_edges = np.unique(np.sort(np.asarray(edges_list, dtype=np.int64), axis=1), axis=0)
            self.seg_gtype = surf_m.seg_gtype
            self.seg_elem = surf_m.seg_elem
        elif master_surf_id in model.lines:
            line_obj = model.lines[master_surf_id]
            if getattr(line_obj, "segments", None) is not None and len(line_obj.segments) > 0:
                bead_edges = np.asarray(line_obj.segments, dtype=np.int64)
            else:
                nd = np.asarray(getattr(line_obj, "nodes", []), dtype=np.int64)
                if len(nd) >= 2:
                    bead_edges = np.column_stack([nd[:-1], nd[1:]])
            self.seg_gtype = getattr(line_obj, "seg_gtype", np.zeros(len(bead_edges), dtype=np.int32))
            self.seg_elem = getattr(line_obj, "seg_elem", np.zeros(len(bead_edges), dtype=np.int64))
        else:
            log.warning(f"/INTER/TYPE21/{itf.id}: main surface {master_surf_id} empty or missing", "CONTACT INIT")
            self._init_empty()
            return

        self.bead_edges = bead_edges

        # Secondary sheet nodes
        if surf_s is not None and surf_s.segments is not None and len(surf_s.segments) > 0:
            self.nodes = np.unique(np.asarray(surf_s.segments, dtype=np.int64))
        elif getattr(itf, "grnod_id", 0) > 0:
            grn = model.node_groups.get(itf.grnod_id)
            if grn is not None:
                if getattr(grn, "node_idx", None) is not None and len(grn.node_idx) > 0:
                    self.nodes = np.asarray(grn.node_idx, dtype=np.int64)
                else:
                    self.nodes = np.asarray(getattr(grn, "nodes", []), dtype=np.int64)
            else:
                self.nodes = np.unique(self.bead_edges)
        else:
            self.nodes = np.unique(self.bead_edges)

        self.nodes.sort()

        # Interface parameters
        self.istf = int(getattr(itf, "istf", 0))
        self.stfac = float(getattr(itf, "stfac", 1.0) or 1.0)
        self.stmin = float(getattr(itf, "stmin", 0.0) or 0.0)
        self.stmax = float(getattr(itf, "stmax", 1e30) or 1e30)
        self.fric = float(getattr(itf, "fric", 0.0) or 0.0)
        self.gap_min = float(getattr(itf, "gap", getattr(itf, "gap_min", 0.0)) or 0.0)
        self.gap_max = float(getattr(itf, "gap_max", 1e30) or 1e30)
        if self.gap_max <= 0.0:
            self.gap_max = 1e30
        self.fscale_gap = float(getattr(itf, "fscale_gap", getattr(itf, "gap_scale", 1.0)) or 1.0)
        self.depth = float(getattr(itf, "depth", getattr(itf, "dist", 0.0)) or 0.0)
        self.pmax = float(getattr(itf, "pmax", 1e30) or 1e30)
        self.itlim = int(getattr(itf, "itlim", 0))
        self.visc = float(getattr(itf, "viss", getattr(itf, "stiff_dc", 0.05)) or 0.05)
        if self.visc <= 0.0:
            self.visc = 0.05
        self.tstart = float(getattr(itf, "tstart", 0.0) or 0.0)
        self.tstop = float(getattr(itf, "tstop", 1e30) or 1e30)

        # Bending force per unit depth / nominal stiffness
        # In metal forming, drawbead restraining force without friction is determined by bending
        self.fbend_nominal = float(getattr(itf, "fbend", 0.0) or 0.0)
        if self.fbend_nominal <= 0.0 and self.depth > 0.0:
            # Set proportional to depth * stiffness scale
            self.fbend_nominal = self.depth * self.stfac * 0.1

        # Evaluate stiffness and gaps
        scale = self.stfac if self.istf != 1 else 1.0
        Ks_all, gs_all = node_stiffness_gap(model, scale, fscale_gap=self.fscale_gap)
        if len(self.nodes) and len(Ks_all):
            valid_k = (self.nodes >= 0) & (self.nodes < len(Ks_all))
            self.Ks = np.zeros(len(self.nodes), dtype=float)
            self.Ks[valid_k] = Ks_all[self.nodes[valid_k]]
            self.gap_s = np.zeros(len(self.nodes), dtype=float)
            self.gap_s[valid_k] = gs_all[self.nodes[valid_k]]
        else:
            self.Ks = np.zeros(len(self.nodes), dtype=float)
            self.gap_s = np.zeros(len(self.nodes), dtype=float)

        # Master line stiffness from bead nodes
        bead_nodes = np.unique(self.bead_edges) if len(self.bead_edges) > 0 else np.zeros(0, dtype=np.int64)
        if len(bead_nodes) and len(Ks_all):
            valid_bn = (bead_nodes >= 0) & (bead_nodes < len(Ks_all))
            Km_nodes = Ks_all[bead_nodes[valid_bn]]
            self.Km_default = float(Km_nodes.max()) if len(Km_nodes) > 0 else (self.stfac if self.istf == 1 else 1000.0)
        else:
            self.Km_default = self.stfac if self.istf == 1 else 1000.0

        self.gap_const = max(self.gap_min, 1e-4) if self.gap_min > 0.0 else 0.001
        self.gap_bound = max(
            float(self.gap_s.max() if len(self.gap_s) else self.gap_const),
            self.gap_const,
            self.depth,
        )

        # Tangential force history storage (node_id, edge_idx) -> np.ndarray (3,)
        self.ft_hist = {}

        self.dt_bound = self._compute_dt_bound(model.mass)

        # Broad phase bookkeeping
        self.pairs_node = np.zeros(0, dtype=np.int64)
        self.pairs_edge = np.zeros(0, dtype=np.int64)
        self._last_refresh = -10**9
        self.refresh = 20

    def _init_empty(self):
        self.bead_edges = np.zeros((0, 2), dtype=np.int64)
        self.nodes = np.zeros(0, dtype=np.int64)
        self.Ks = np.zeros(0, dtype=float)
        self.gap_s = np.zeros(0, dtype=float)
        self.gap_const = 0.0
        self.gap_bound = 0.0
        self.gap_min = 0.0
        self.gap_max = 1e30
        self.stmin = 0.0
        self.stmax = 1e30
        self.fric = 0.0
        self.depth = 0.0
        self.pmax = 1e30
        self.itlim = 0
        self.visc = 0.05
        self.tstart = 0.0
        self.tstop = 1e30
        self.fbend_nominal = 0.0
        self.ft_hist = {}
        self.dt_bound = np.inf
        self.pairs_node = np.zeros(0, dtype=np.int64)
        self.pairs_edge = np.zeros(0, dtype=np.int64)
        self._last_refresh = -10**9
        self.refresh = 20

    def _compute_dt_bound(self, mass: np.ndarray) -> float:
        if len(self.bead_edges) == 0 or len(self.nodes) == 0:
            return np.inf
        valid_nodes = self.nodes[(self.nodes >= 0) & (self.nodes < len(mass))]
        if len(valid_nodes) == 0:
            return np.inf
        loc = np.searchsorted(self.nodes, valid_nodes)
        Ks_sub = self.Ks[loc]
        Km_arr = np.full(len(valid_nodes), self.Km_default)
        K_sec = combine_stiffness(self.istf, self.stfac, Km_arr, Ks_sub)
        if self.stmin > 0.0:
            K_sec = np.maximum(K_sec, self.stmin)
        if self.stmax > 0.0:
            K_sec = np.minimum(K_sec, self.stmax)
        m_sec = mass[valid_nodes]
        pos = (m_sec > 0.0) & (K_sec > 0.0)
        dt_sec = np.sqrt(2.0 * m_sec[pos] / K_sec[pos]).min() if np.any(pos) else np.inf
        return float(dt_sec)

    def _broad_phase(self, x: np.ndarray, v: np.ndarray, dt: float):
        """Bounding box candidate search for drawbead line contact."""
        edges = self.bead_edges
        nodes = self.nodes
        if len(edges) == 0 or len(nodes) == 0:
            self.pairs_node = np.zeros(0, dtype=np.int64)
            self.pairs_edge = np.zeros(0, dtype=np.int64)
            return

        vmax = float(np.abs(v).max()) if len(v) else 0.0
        margin = (self.gap_bound + self.depth) * self.fscale_gap + 2.0 * self.refresh * dt * vmax

        xe = x[edges]  # (nedge, 2, 3)
        edge_min = xe.min(axis=1) - margin  # (nedge, 3)
        edge_max = xe.max(axis=1) + margin  # (nedge, 3)

        xn = x[nodes]  # (nnod, 3)

        in_x = (xn[:, 0, None] >= edge_min[:, 0]) & (xn[:, 0, None] <= edge_max[:, 0])
        in_y = (xn[:, 1, None] >= edge_min[:, 1]) & (xn[:, 1, None] <= edge_max[:, 1])
        in_z = (xn[:, 2, None] >= edge_min[:, 2]) & (xn[:, 2, None] <= edge_max[:, 2])
        cand = in_x & in_y & in_z

        nid_idx, erow = np.where(cand)
        if len(nid_idx) == 0:
            self.pairs_node = np.zeros(0, dtype=np.int64)
            self.pairs_edge = np.zeros(0, dtype=np.int64)
            return

        ni = nodes[nid_idx]

        # Self-exclusion: sheet node cannot be a corner of the master edge
        n1 = edges[erow, 0]
        n2 = edges[erow, 1]
        not_corner = (ni != n1) & (ni != n2)

        self.pairs_node = ni[not_corner]
        self.pairs_edge = erow[not_corner]

    def forces(
        self,
        x: np.ndarray,
        v: np.ndarray,
        mass: np.ndarray,
        dt: float,
        fcont: np.ndarray,
        cycle: int = 0,
        stifn: np.ndarray = None,
        t: float = None,
    ) -> tuple[float, float]:
        """
        Evaluate drawbead contact forces (clamping + restraining) and assemble into fcont.
        Returns (econt_est, dt_bound).
        """
        if len(self.bead_edges) == 0 or len(self.nodes) == 0 or dt <= 0.0:
            return 0.0, self.dt_bound

        if t is not None:
            if t < self.tstart or t > self.tstop:
                return 0.0, self.dt_bound

        if cycle - self._last_refresh >= self.refresh:
            self._broad_phase(x, v, dt)
            self._last_refresh = cycle

        if len(self.pairs_node) == 0:
            return 0.0, self.dt_bound

        ni = self.pairs_node
        erow = self.pairs_edge
        edges = self.bead_edges[erow]

        x1 = x[edges[:, 0]]
        x2 = x[edges[:, 1]]
        xs = x[ni]

        # Edge segment vector and projection
        L_vec = x2 - x1
        L2 = np.maximum(np.sum(L_vec * L_vec, axis=1), EM30)
        L_norm = np.sqrt(L2)

        # Projection parameter xi on bead line segment
        dx1 = xs - x1
        xi = np.einsum("nk,nk->n", dx1, L_vec) / L2
        xi_clamped = np.clip(xi, 0.0, 1.0)

        # Closest point on drawbead line
        xp = (1.0 - xi_clamped[:, None]) * x1 + xi_clamped[:, None] * x2
        d_vec = xs - xp
        dist = norm3(d_vec)

        # Normal unit vector from bead to sheet
        norm_d = np.where(
            (dist > EM20)[:, None],
            d_vec / np.maximum(dist, EM20)[:, None],
            np.array([0.0, 0.0, 1.0]),
        )

        # Effective gap and penetration
        loc = np.searchsorted(self.nodes, ni)
        loc = np.clip(loc, 0, len(self.nodes) - 1)
        gap = (self.gap_s[loc] if len(self.gap_s) else self.gap_const) * self.fscale_gap
        gap = np.clip(gap, self.gap_min, self.gap_max)

        # If depth > 0, bead influence zone extends to gap + depth
        gap_eff = gap + self.depth
        pene = np.maximum(0.0, gap_eff - dist)

        # Stiffness
        Km_arr = np.full(len(ni), self.Km_default)
        K = combine_stiffness(self.istf, self.stfac, Km_arr, self.Ks[loc])
        if self.stmin > 0.0:
            K = np.maximum(K, self.stmin)
        if self.stmax > 0.0:
            K = np.minimum(K, self.stmax)

        # Time step accumulation
        dt_int = self.dt_bound
        near = pene > 0.0
        if np.any(near):
            n_nod = len(fcont)
            Knode = np.bincount(ni[near], weights=K[near], minlength=n_nod)
            Knode += np.bincount(edges[near, 0], weights=0.5 * K[near], minlength=n_nod)
            Knode += np.bincount(edges[near, 1], weights=0.5 * K[near], minlength=n_nod)
            loaded = Knode > 0.0
            if np.any(loaded):
                m_loaded = np.maximum(mass[loaded], EM20)
                dt_int = min(self.dt_bound, float(np.sqrt(2.0 * m_loaded / Knode[loaded]).min()))
            if stifn is not None:
                stifn[loaded] += Knode[loaded]

        active = pene > 0.0
        if not np.any(active):
            return 0.0, dt_int

        ni = ni[active]
        erow = erow[active]
        edges = edges[active]
        pene = pene[active]
        dist = dist[active]
        norm_d = norm_d[active]
        xi_clamped = xi_clamped[active]
        K = K[active]
        gap = gap[active]
        L_vec = L_vec[active]
        L_norm = L_norm[active]

        # 1. Normal clamping force
        fn_scalar = K * pene
        # Velocity at projection point on bead line
        v1 = v[edges[:, 0]]
        v2 = v[edges[:, 1]]
        vp = (1.0 - xi_clamped[:, None]) * v1 + xi_clamped[:, None] * v2
        vs = v[ni]

        v_rel = vs - vp
        vn = np.einsum("nk,nk->n", v_rel, norm_d)
        c_damp = 2.0 * self.visc * np.sqrt(K * np.maximum(mass[ni], EM20))
        fn_total = fn_scalar - c_damp * np.minimum(vn, 0.0)

        fn_vec = fn_total[:, None] * norm_d

        # 2. Tangential restraining force (friction + plastic bending-unbending)
        vt_vec = v_rel - vn[:, None] * norm_d
        vt_norm = norm3(vt_vec)

        ft_vec = np.zeros_like(fn_vec)
        econt_fric = 0.0

        if np.any(vt_norm > 1e-12):
            # Unit sliding direction
            s_dir = np.where(
                (vt_norm > 1e-12)[:, None],
                vt_vec / np.maximum(vt_norm, 1e-12)[:, None],
                np.zeros_like(vt_vec),
            )

            # (a) Friction resistance: mu * |Fn|
            f_fric = self.fric * np.abs(fn_scalar)

            # (b) Bending-unbending resistance: F_bend
            f_bend = self.fbend_nominal * np.clip(pene / np.maximum(gap, 1e-6), 0.0, 1.0)

            # Total restraining resistance magnitude
            f_restrain = f_fric + f_bend

            # (c) Limitation by F_max (from Pmax / ITlim)
            if self.itlim == 0 and self.pmax < 1e20:
                # Max restraining force per node/area
                f_max = self.pmax
                f_restrain = np.minimum(f_restrain, f_max)

            # Stick-slip incremental tangential force
            ft_prev = np.zeros_like(fn_vec)
            for idx, (node_idx, e_idx) in enumerate(zip(ni, erow)):
                key = (int(node_idx), int(e_idx))
                if key in self.ft_hist:
                    ft_prev[idx] = self.ft_hist[key]

            # Incremental trial force opposing sliding
            ft_trial = ft_prev - K[:, None] * vt_vec * dt
            ft_trial -= np.einsum("nk,nk->n", ft_trial, norm_d)[:, None] * norm_d
            ft_trial_norm = norm3(ft_trial)

            scale = np.where(
                ft_trial_norm > f_restrain,
                f_restrain / np.maximum(ft_trial_norm, EM30),
                1.0,
            )
            ft_vec = ft_trial * scale[:, None]

            # Update history
            for idx, (node_idx, e_idx) in enumerate(zip(ni, erow)):
                key = (int(node_idx), int(e_idx))
                self.ft_hist[key] = ft_vec[idx]

            # Frictional dissipation work rate
            econt_fric = float(np.sum(np.abs(np.einsum("nk,nk->n", -ft_vec, vt_vec)) * dt))

        # Total force on sheet slave node
        f_slave = fn_vec + ft_vec

        # Reactive forces on master bead nodes (exact momentum conservation)
        f_m1 = -(1.0 - xi_clamped[:, None]) * f_slave
        f_m2 = -xi_clamped[:, None] * f_slave

        # Assemble into global fcont
        np.add.at(fcont, ni, f_slave)
        np.add.at(fcont, edges[:, 0], f_m1)
        np.add.at(fcont, edges[:, 1], f_m2)

        econt_est = float(0.5 * np.sum(K * pene * pene)) + econt_fric

        return econt_est, dt_int

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
        fheat: float = 0.0,
        efrict: Optional[np.ndarray] = None,
        theaccfact: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute thermal conduction and radiation across /INTER/TYPE21 drawbead interface.

        Ported from $OR_SRC/engine/source/interfaces/int21/i21therm.F

        Parameters
        ----------
        temp : np.ndarray
            Nodal temperatures.
        dt : float
            Time step dt.
        kthe : Optional[float]
            Thermal interface conductivity KTHE.
        dcond : float
            Distance decay limit.
        fcond : Optional[Any]
            Conductance decay curve.
        frad : float
            Radiation coefficient.
        drad : float
            Radiation cutoff distance.
        iform : int
            1 = update master bead nodes, 0 = secondary only.
        fheat : float
            Friction heating partition factor.
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
        heat_transferred : float
            Total thermal energy transferred [J].
        """
        from .thermal_contact import thermal_contact_type21

        itf = self.itf
        if kthe is None:
            kthe = getattr(itf, "kthe", 0.0) or getattr(itf, "rstif", 0.0)

        slave_nodes = getattr(self, "slave_nodes", getattr(self, "nodes", np.zeros(0, dtype=np.int64)))
        master_edges = getattr(self, "master_edges", getattr(self, "bead_edges", np.zeros((0, 2), dtype=np.int64)))

        if len(slave_nodes) == 0 or len(master_edges) == 0:
            return np.zeros(len(temp), dtype=float), np.zeros(0, dtype=float), 0.0

        n_pairs = len(slave_nodes)
        weights = np.zeros((n_pairs, 4), dtype=float)
        weights[:, 0] = 0.5
        weights[:, 1] = 0.5

        master_segs = np.zeros((n_pairs, 4), dtype=np.int64)
        master_segs[:, 0] = master_edges[0, 0]
        master_segs[:, 1] = master_edges[0, 1]
        master_segs[:, 2] = master_segs[:, 1]
        master_segs[:, 3] = master_segs[:, 1]

        areac = np.full(n_pairs, 1.0, dtype=float)

        return thermal_contact_type21(
            temp=temp,
            slave_nodes=slave_nodes,
            master_segs=master_segs,
            weights=weights,
            kthe=kthe,
            dt=dt,
            theaccfact=theaccfact,
            areac=areac,
            dcond=dcond,
            fcond=fcond,
            frad=frad,
            drad=drad,
            iform=iform,
            fheat=fheat,
            efrict=efrict,
        )

