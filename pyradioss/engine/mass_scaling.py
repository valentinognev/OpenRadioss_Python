"""
/DT/NODA and /DT/NODA/CST — nodal time step and mass scaling (M6).

Fortran origin: ``engine/source/time_step/`` — the nodal time-step option
of the /DT engine card family:

    dtnoda.F    nodal time step from the assembled nodal stiffness STIFN
                and the nodal mass:  dt_i = sqrt(2 M_i / K_i)
    (the element routines accumulate their stiffness into STIFN, the
    interfaces add their penalty-spring stiffness — the same accumulation
    this port already performs for the interface dt, see the M4 lesson in
    contact/inter_type7.py)

Theory — the nodal time step
----------------------------
The element time step dt_e = 2/omega_max(e) bounds the step by the WORST
single element. The nodal view assembles, per node, an equivalent
stiffness from every element (and contact spring) loading it and bounds
the step by the worst node-on-spring system:

    K_i = sum_e k_i^e ,      dt_i = sqrt(2 M_i / K_i)

The port derives the element's nodal-stiffness contribution from the SAME
quantity the element kernels already compute — their critical time step:
an element whose lumped nodal mass share is m_i^e and whose critical step
is dt_e behaves, seen from one of its nodes, like a spring of stiffness

    k_i^e = 2 m_i^e / dt_e^2         (so that sqrt(2 m/k) = dt_e)

For a uniform mesh this makes the nodal dt EQUAL to the element dt (each
node's mass and stiffness scale together); where a small stiff element
borders heavy nodes the nodal dt is larger — the classic reason /DT/NODA
alone already buys a few percent of step. Deleted elements return
dt_e = 1e30, so their stiffness contribution vanishes automatically.

Mass scaling (/DT/NODA/CST)
---------------------------
With the CST ("constant") option the Engine no longer lets dt fall below
the target: whenever a node's dt_i drops under dt_min, MASS IS ADDED to
that node until it holds,

    M_i^needed = K_i * (dt_min / dt_sca)^2 / 2 ,   dm_i = M^needed - M_i

(mass is only ever added, never removed — the Radioss DMAS convention).
Added mass CHANGES THE PHYSICS: it increases inertia, momentum and
kinetic energy. The port keeps the balance honest:

* dm is added to the PHYSICAL nodal mass (the KE/momentum ledgers see
  it) and to the effective mass used for accelerations;
* the kinetic energy the addition creates at the node's current
  velocity, 0.5 dm |v|^2, is booked into a dedicated counter reported in
  the listing and INCLUDED in the energy balance (like external work);
* the total added mass and its fraction of the initial mass are printed
  as the run progresses and in the termination summary — a mass-scaled
  run with dM/M beyond a few percent is a different structure, and the
  listing must say so (the original prints its MAS.ERR column for the
  same reason).

Nodes whose motion is PRESCRIBED (rigid-body members, tied /INTER/TYPE2
secondaries, /RBE3 dependents, frozen massless placeholders) carry no
stability constraint of their own — their velocities are overwritten by
the constraint — so they are excluded both from the nodal-dt minimum and
from mass addition (adding mass there could not change dt anyway, but it
WOULD silently alter the constraint inertia the Starter assembled).

Rotational nodal time step (STIFR) — M40
----------------------------------------
``dtnoda.F`` bounds the step with a SECOND, rotational nodal dt whenever
the model carries rotational DOFs (``IRODDL /= 0``): for EVERY node with
``IN(N) > 0`` and an assembled rotational stiffness ``STIFR(N) > 0``,

    dt_i^rot = sqrt(2 IN_i / STIFR_i)          (dtnoda.F lines 452-471)

joins the same minimum as the translational one.  The element claims
mirror the translational ones — the shell dt routines set
``STIR = STI * (t^2 + A)/12`` (cndt3.F lines 209-218, the BATOZ/QEPH/DKT
family; chvis3/chsti3 use t^2/12 + A/9 for BT) and the assembly adds the
FULL element STIR to each of its nodes (cupdt3.F, pmcum3.F), exactly as
STI feeds STIFN.  The Starter lumps the nodal inertia with the MATCHING
factor (cinmas.F ~line 920: ``XI = m/4 (AREA/FAC + t^2/12)`` with
FAC = 12 for IHBE >= 11, 9 for BT) — so on an element-lumped free node
the rotational dt EQUALS the translational one by construction and the
claim never binds there.  It bites only where IN is decoupled from the
element lumping: above all the /RBODY master, whose transported
``STIFR(M) += Sum(STIFR(s) + DD*STIFN(s))`` (rgbodfp.F) grows with the
parallel-axis distance while ``IN(M)`` stays the body's minimum
principal moment — the RD-E-1000 rolling floor (M39 residual: the port's
translational-only transport floored c04 at 2.07e-2 vs the Fortran
1.64e-2 that includes the members' own STIFR).

The port derives the rotational claim exactly like the translational
one, from the element dt: a per-node lumped-inertia share ``I_i^e`` (the
SAME array the Starter assembled into ``model.inertia`` — kernels store
it as ``group.state['dt_iner']``) behaves like a torsion spring of

    kr_i^e = 2 I_i^e / dt_e^2       (so that sqrt(2 I/kr) = dt_e)

which reproduces upstream's ``STIR = STI * fac`` identically (both sides
carry the same factor, k_i^e * I/m) and keeps the free-node invariant
dt_rot == dt_tra to round-off.  With /DT/NODA/CST the rotational branch
scales INERTIA exactly as the translational one scales mass
(dtnoda.F lines 482-516: ``IN(N) = MAX(INER, IN(N))``, the DINERT
counter): only ever added, reported in the summary; no energy booking is
needed because the port's kinetic-energy ledger is translational (the
original books none for DINERT either).  Rotational SPRING stiffness
(torsional /PROP/TYPE8 etc.) is NOT claimed — those kernels currently
make no rotational dt claim at all; their translational claim flows
unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import math

import numpy as np

from ..common.constants import EM20, EP30
from ..model.model import Model


class NodalTimeStep:
    """Engine-side /DT/NODA[/CST] machinery (see module docstring)."""

    def __init__(self, model: Model, controls, log):
        self.model = model
        self.mode: str = getattr(controls, "dt_noda", "") if getattr(controls, "dt_noda", "") else "NODA"
        self.cst = self.mode == "CST" or self.mode.startswith("CST")
        self.dt_sca = getattr(controls, "dt_scale", 0.9)
        self.dt_min = getattr(controls, "dt_min", 0.0)
        n = model.numnod

        # per-node added mass tracking (M613)
        self.node_added_mass: np.ndarray = np.zeros(n, dtype=np.float64)
        self.initial_nodal_mass: np.ndarray = model.mass.copy()

        # critical entity tracking (M613)
        self.crit_node_id: int = 0
        self.crit_elem_type: str = "NODE"

        # /DT/NODA/STOP tracking (M613)
        self.stopped: bool = False
        self.stop_node: int = 0

        # /DT/NODA/SET acceleration scaling factors (M613)
        self._set_factors: Dict[int, float] = {}

        # per-group (node-index matrix, nodal mass-share matrix, nodal
        # inertia-share matrix or None): the same lumping the Starter used
        # to build the nodal mass (1/8 per brick corner, 1/4 per
        # shell/tetra corner, 1/3 per triangle corner, 1/2 per truss/
        # spring/beam end — the beam's 3rd node is orientation only and
        # carries nothing).  The inertia share ``dt_iner`` is the per-NODE
        # lumped rotational inertia the kernel's init_group assembled into
        # ``model.inertia`` — the rotational claim base (module docstring);
        # groups without rotational DOFs (solids, trusses, springs) leave
        # it unset and claim no STIFR, like the original's IRODDL gating.
        self._shares: List[Tuple[str, np.ndarray, np.ndarray,
                                 Optional[np.ndarray]]] = []
        for name, group in model.element_groups():
            conn = group.state.get("mass_conn", group.conn)
            share = group.state["mass"][:, None] / conn.shape[1]
            iner = group.state.get("dt_iner")
            if iner is not None:
                iner = np.broadcast_to(iner[:, None], conn.shape).copy()
            self._shares.append((name, conn, np.broadcast_to(
                share, conn.shape).copy(), iner))
        #: any rotational claims at all? (gates every STIFR op so a
        #: solids-only deck pays nothing — dtnoda.F's IRODDL == 0 branch)
        self._rot = any(s[3] is not None for s in self._shares)

        # nodes excluded from the nodal dt / mass addition (prescribed
        # motion — see module docstring); filled by ``set_prescribed``
        self.free = model.mass < 1e29
        self.stifn = np.zeros(n)         # assembled nodal stiffness
        self.stifr = np.zeros(n)         # assembled rotational stiffness
        # rigid bodies whose member stiffness is transported to the master
        # (rgbodfp.F + dtnoda.F — see ``add_rigid_body``/``_rigid_body_dt``):
        # (member node indices, DD=|x-x_master|^2, body mass M, min inertia)
        self._rbodies: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
        self.mass_added = 0.0            # cumulative added mass
        self.iner_added = 0.0            # cumulative added inertia (DINERT)
        self.e_madd = 0.0                # cumulative 0.5 dm v^2
        self.mom_added = np.zeros(3)     # cumulative dm * v
        self.mass0 = float(model.mass[model.mass < 1e29].sum())
        self._reported = 0.0             # last dM/M milestone printed
        #: SPMD ``WEIGHT`` array (``None`` = serial).  Under a domain
        #: decomposition every holder of a node adds the SAME mass (the
        #: frontier-summed STIFN is identical everywhere), so only the
        #: once-per-node COUNTERS (mass_added, iner_added, e_madd,
        #: mom_added) carry the weight — like dtnoda.F's DMAST sum which
        #: loops over the domain's own nodes; the global value is the sum
        #: over domains (see ``announce`` and engine.py).
        self.weight: Optional[np.ndarray] = None
        self.log = log
        if self.cst and self.dt_min <= 0.0:
            log.warning("/DT/NODA/CST: dT_min is zero — no mass will ever "
                        "be added (give a positive target step)",
                        "DT NODA")

    # ------------------------------------------------------------------
    def set_prescribed(self, idx: np.ndarray) -> None:
        """Mark nodes whose motion a constraint prescribes (rigid bodies,
        tied secondaries, RBE3 dependents): no dt claim, no mass add."""
        if len(idx):
            self.free[idx] = False

    # ------------------------------------------------------------------
    def add_rigid_body(self, nodes: np.ndarray, master: int, mass: float,
                       inertia: np.ndarray, x0: np.ndarray,
                       rb: Optional[Any] = None) -> None:
        """Register a rigid body so its member stiffness is TRANSPORTED to
        the master node and gives the body its own nodal time step — the
        thing that replaces the member nodes dropped by ``set_prescribed``.

        Fortran origin: ``rgbodfp.F`` (IFLAG=1 gather, called from
        ``rbyfor.F``) sums every slave's nodal stiffness onto the master —
        translational ``STIFN(M) += Sum STIFN(slave)`` and rotational
        ``STIFR(M) += Sum (STIFR(slave) + DD*STIFN(slave))`` with
        ``DD = |x_slave - x_master|^2`` the Huygens-Steiner parallel-axis
        transport of the slave spring to the master — then zeroes the
        slaves (rgbodfp.F 741-755).  ``dtnoda.F`` then bounds the step with
        the master's two nodal dts, ``sqrt(2 MS(M)/STIFN(M))`` (line 257)
        and ``sqrt(2 IN(M)/STIFR(M))`` (line 469), where the Starter set
        ``MS(M)`` = the body mass and ``IN(M) = MIN`` principal moment of
        the body inertia (``inirby.F`` line 838).

        Without this the port marks every member node prescribed and drops
        its stiffness, so a stiff shell welded into a rigid body never
        constrains dt (the M39 RD-E-1000 rolling bug: the port ran the
        clamped strip 2.6x too fast).

        ``DD`` is taken from the initial positions ``x0`` — a rigid body
        preserves its inter-node distances, so ``|x-x_master|`` is constant
        and equals its t=0 value (rgbodfp recomputes it from the current X
        each cycle only because the generic gather cannot assume rigidity).

        ``nodes`` includes the master itself (its ``DD`` is zero), so the
        sums pick up any element stiffness claimed AT the master — exactly
        rgbodfp's ``+=`` onto the master's pre-existing STIFN/STIFR.

        M40 completes the M39 transport with the members' own rotational
        stiffness: the port now assembles ``stifr`` (the shell bending/
        drilling and beam STIR claims, see the module docstring), and
        ``_rigid_body_dt`` sums the full rgbodfp gather
        ``F2 = STIFR(s) + DD*STIFN(s)`` (rgbodfp.F lines 116-118).  On the
        RD-E-1000 BATOZ roll this moves the c04 floor from the M39
        translational-only 2.07e-2 onto the Fortran 1.64e-2.
        """
        nodes = np.asarray(nodes)
        dd = ((x0[nodes] - x0[master]) ** 2).sum(axis=1)
        in_min = float(np.linalg.eigvalsh(inertia)[0])
        self._rbodies.append([nodes, dd, float(mass), in_min, int(master), rb])

    # ------------------------------------------------------------------
    def _rigid_body_dt(self) -> float:
        """The transported master nodal dt of every registered rigid body
        (see ``add_rigid_body``): min over bodies of the translational
        ``sqrt(2 M/K_tra)`` and rotational ``sqrt(2 IN/K_rot)`` steps, with
        ``K_tra = Sum stifn`` and ``K_rot = Sum (stifr + DD*stifn)`` over
        the member nodes (rgbodfp.F IFLAG=1: F1 = STIFN(s),
        F2 = STIFR(s) + DD*STIFN(s); the STIFR term is the M40
        completion).  Read from the accumulators BEFORE ``apply`` resets
        them."""
        dt = EP30
        for entry in self._rbodies:
            nodes, dd, mass, in_min = entry[0], entry[1], entry[2], entry[3]
            st = self.stifn[nodes]
            k_tra = float(st.sum())
            if k_tra > 0.0 and mass > 0.0:
                dt = min(dt, float(np.sqrt(2.0 * mass / k_tra)))
            k_rot = float((self.stifr[nodes] + dd * st).sum())
            if k_rot > 0.0 and in_min > 0.0:
                dt = min(dt, float(np.sqrt(2.0 * in_min / k_rot)))
        return dt

    # ------------------------------------------------------------------
    def assemble(self, dt_claims) -> None:
        """Assemble the nodal stiffness for this cycle.

        ``dt_claims`` — list of per-element critical-dt arrays, one per
        element group in ``model.element_groups()`` order (the arrays the
        kernels return each cycle). The contact contribution is
        accumulated directly into ``self.stifn`` by the interfaces (the
        Engine passes it to their ``forces``).  Groups with rotational
        DOFs also claim ``stifr`` from their lumped-inertia share
        (kr = 2 I / dt_e^2 — the module docstring; the STIR analogue of
        cupdt3.F/pmcum3.F's per-node accumulation)."""
        # (self.stifn already holds this cycle's contact springs)
        for (name, conn, share, iner), dt_e in zip(self._shares, dt_claims):
            k = 2.0 * share / np.maximum(dt_e, EM20)[:, None] ** 2
            # a deleted element's claim is 1e30 -> its k underflows to 0
            c_flat = conn.reshape(-1)
            k_flat = k.reshape(-1)
            valid = (c_flat >= 0) & (c_flat < len(self.stifn))
            np.add.at(self.stifn, c_flat[valid], k_flat[valid])
            if iner is not None:
                kr = 2.0 * iner / np.maximum(dt_e, EM20)[:, None] ** 2
                kr_flat = kr.reshape(-1)
                valid_r = (c_flat >= 0) & (c_flat < len(self.stifr))
                np.add.at(self.stifr, c_flat[valid_r], kr_flat[valid_r])

    # ------------------------------------------------------------------
    def apply(self, mass_eff: np.ndarray, inv_mass: np.ndarray,
              v: np.ndarray, t: float,
              inertia: Optional[np.ndarray] = None,
              inv_inertia: Optional[np.ndarray] = None,
              ams_nodes: Optional[np.ndarray] = None) -> float:
        """Mass scaling + nodal dt for this cycle. Must run BEFORE the
        acceleration update (the added mass stabilizes the very cycle
        that needed it). Returns the nodal critical time step.

        ``ams_nodes`` — boolean array of nodes active in AMS (which are 
        excluded from driving the explicit dt).
        
        ``inertia``/``inv_inertia`` — the physical nodal rotational
        inertia (``model.inertia``) and its inverse: enables the
        ROTATIONAL nodal dt ``sqrt(2 IN/STIFR)`` over every free node
        with IN > 0 (dtnoda.F lines 452-471 — ALL nodes, not only rigid
        bodies; see the module docstring for why it only ever binds
        through /RBODY-like constructs) and, with CST, the matching
        inertia scaling (dtnoda.F lines 482-516).

        The stiffness accumulators are consumed and reset here."""
        model = self.model

        # Accumulate rigid body slave node stiffness into master node (rgbodfp.F)
        for entry in self._rbodies:
            nodes, dd, mass_val, in_min = entry[0], entry[1], entry[2], entry[3]
            master = entry[4] if len(entry) > 4 else int(nodes[0])
            slave_mask = nodes != master
            if np.any(slave_mask):
                slaves = nodes[slave_mask]
                dd_sl = dd[slave_mask]
                self.stifn[master] += float(self.stifn[slaves].sum())
                if self._rot:
                    self.stifr[master] += float((self.stifr[slaves] + dd_sl * self.stifn[slaves]).sum())
                self.stifn[slaves] = 0.0
                if self._rot:
                    self.stifr[slaves] = 0.0

        if self.cst and self.dt_min > 0.0 and self.dt_sca > 0.0:
            for entry in self._rbodies:
                nodes, dd, mass_rb, in_min = entry[0], entry[1], entry[2], entry[3]
                master = entry[4] if len(entry) > 4 else int(nodes[0])
                rb = entry[5] if len(entry) > 5 else None

                # Translational mass scaling for rigid body (1.00001 factor, M613)
                k_tra = self.stifn[master]
                if k_tra > 0.0:
                    m_req = 1.00001 * k_tra * (self.dt_min / self.dt_sca) ** 2 / 2.0
                    dm = m_req - mass_rb
                    if dm > 0.0:
                        dm = float(dm)
                        model.mass[master] += dm
                        mass_eff[master] += dm
                        inv_mass[master] = 1.0 / mass_eff[master]
                        if self.weight is None:
                            self.mass_added += dm
                            self.e_madd += float(0.5 * dm * (v[master] ** 2).sum())
                            self.mom_added += dm * v[master]
                        else:
                            wm = float(self.weight[master])
                            self.mass_added += dm * wm
                            self.e_madd += float(0.5 * dm * (v[master] ** 2).sum()) * wm
                            self.mom_added += (dm * wm) * v[master]
                        self.node_added_mass[master] += dm
                        entry[2] += dm
                        if rb is not None:
                            rb.M += dm

                # Rotational inertia scaling for rigid body (1.00001 factor, M613)
                if self._rot and in_min > 0.0 and self.stifr[master] > 0.0:
                    k_rot = self.stifr[master]
                    i_req = 1.00001 * k_rot * (self.dt_min / self.dt_sca) ** 2 / 2.0
                    di = i_req - in_min
                    if di > 0.0:
                        di = float(di)
                        if inertia is not None:
                            inertia[master] += di
                            if inv_inertia is not None:
                                inv_inertia[master] = 1.0 / inertia[master]
                        self.iner_added += di if self.weight is None \
                            else di * float(self.weight[master])
                        entry[3] += di
                        if rb is not None:
                            rb.J0 += np.eye(3) * di

        rb_masters = [entry[4] for entry in self._rbodies if len(entry) > 4]
        loaded = (self.stifn > 0.0) & self.free & (model.mass > 0.0)
        if ams_nodes is not None:
            loaded &= ~ams_nodes
        if rb_masters:
            loaded[rb_masters] = False

        # rotational claims: dtnoda.F's IRODDL/IN(N)>0 gating.  stifr > 0
        # implies the node took a shell/beam claim, which also fed stifn,
        # so rot is a subset of loaded (contact springs feed only stifn).
        rot = None
        if self._rot and inertia is not None:
            rot = (self.stifr > 0.0) & self.free & (inertia > 0.0)
            if ams_nodes is not None:
                rot &= ~ams_nodes
            if rb_masters:
                rot[rb_masters] = False
            if not np.any(rot):
                rot = None

        if self.cst and self.dt_min > 0.0 and self.dt_sca > 0.0:
            if np.any(loaded):
                # mass needed so that dt_sca * sqrt(2 M / K) >= dt_min (1.00001 factor, M613)
                m_req = 1.00001 * self.stifn[loaded] * (self.dt_min / self.dt_sca) ** 2 \
                    / 2.0
                dm = m_req - mass_eff[loaded]
                add = dm > 0.0
                if np.any(add):
                    idx = np.where(loaded)[0][add]
                    dm_vals = dm[add]
                    # physical mass: the KE/momentum ledgers must see the new
                    # inertia (that is the honest part of mass scaling)
                    model.mass[idx] += dm_vals
                    mass_eff[idx] += dm_vals
                    inv_mass[idx] = 1.0 / mass_eff[idx]
                    self.node_added_mass[idx] += dm_vals
                    if self.weight is None:
                        self.mass_added += float(dm_vals.sum())
                        # the addition creates kinetic energy and momentum
                        # at the node's current velocity — booked and
                        # reported
                        self.e_madd += float(
                            0.5 * (dm_vals[:, None] * v[idx] ** 2).sum())
                        self.mom_added += (dm_vals[:, None] * v[idx]).sum(axis=0)
                    else:
                        # SPMD: once-per-node counters (WEIGHT, ecrit.F)
                        dmw = dm_vals * self.weight[idx]
                        self.mass_added += float(dmw.sum())
                        self.e_madd += float(
                            0.5 * (dmw[:, None] * v[idx] ** 2).sum())
                        self.mom_added += (dmw[:, None] * v[idx]).sum(axis=0)
            if rot is not None and np.any(rot):
                # rotational CST: inertia needed so that the rotational
                # nodal dt holds the target too (dtnoda.F 482-516,
                # IN(N) = MAX(INER, IN(N)) and the DINERT counter — only
                # ever added; no energy booking, the KE ledger is
                # translational, matching the original which books none)
                i_req = 1.00001 * self.stifr[rot] * (self.dt_min / self.dt_sca) ** 2 \
                    / 2.0
                di = i_req - inertia[rot]
                addr = di > 0.0
                if np.any(addr):
                    idx = np.where(rot)[0][addr]
                    di_vals = di[addr]
                    inertia[idx] += di_vals          # model.inertia (physical)
                    if inv_inertia is not None:
                        inv_inertia[idx] = 1.0 / inertia[idx]
                    if self.weight is None:
                        self.iner_added += float(di_vals.sum())
                    else:
                        self.iner_added += float(
                            (di_vals * self.weight[idx]).sum())
            if self.weight is None:
                frac = self.mass_added / max(self.mass0, EM20)
                if frac >= self._reported + 0.01:  # 1%-step announcements
                    self.log.info(
                        f" -- /DT/NODA/CST: ADDED MASS {self.mass_added:.5E}"
                        f" ({100.0 * frac:.2f}% OF THE INITIAL MASS)"
                        f" AT TIME {t:.5E}")
                    self._reported = frac
            # (SPMD: the local counter is a partial sum — the engine calls
            # ``announce`` with the global value from the dt packet)

        # /DT/NODA/SET acceleration scaling factors
        self._set_factors = {}
        if self.mode == "SET" and self.dt_min > 0.0 and self.dt_sca > 0.0:
            if np.any(loaded):
                m_req_set = self.stifn[loaded] * (self.dt_min / self.dt_sca) ** 2 / 2.0
                scale_mask = m_req_set > mass_eff[loaded]
                if np.any(scale_mask):
                    loaded_indices = np.where(loaded)[0]
                    scale_idx = loaded_indices[scale_mask]
                    facs = mass_eff[scale_idx] / m_req_set[scale_mask]
                    for s_i, fac in zip(scale_idx, facs):
                        self._set_factors[int(s_i)] = float(fac)

        dt_candidates = []
        if np.any(loaded):
            dt_i = np.sqrt(2.0 * mass_eff[loaded] / self.stifn[loaded])
            min_pos = int(np.argmin(dt_i))
            dt_candidates.append(float(dt_i[min_pos]))
            loaded_indices = np.where(loaded)[0]
            crit_idx = loaded_indices[min_pos]
            if hasattr(self.model, "node_ids") and self.model.node_ids is not None and len(self.model.node_ids) > crit_idx:
                self.crit_node_id = int(self.model.node_ids[crit_idx])
            else:
                self.crit_node_id = int(crit_idx + 1)
            self.crit_elem_type = "NODE"
        if rot is not None and np.any(rot):
            # the rotational nodal dt joins the same minimum (dtnoda.F
            # line 469: DTN = DTFAC*sqrt(2 IN/STIFR); the shared DTFAC —
            # the port's dt_scale — is applied by the caller)
            dt_r = np.sqrt(2.0 * inertia[rot] / self.stifr[rot])
            min_pos_r = int(np.argmin(dt_r))
            min_val_r = float(dt_r[min_pos_r])
            dt_candidates.append(min_val_r)
            if not dt_candidates or min_val_r < dt_candidates[0]:
                rot_indices = np.where(rot)[0]
                crit_idx = rot_indices[min_pos_r]
                if hasattr(self.model, "node_ids") and self.model.node_ids is not None and len(self.model.node_ids) > crit_idx:
                    self.crit_node_id = int(self.model.node_ids[crit_idx])
                else:
                    self.crit_node_id = int(crit_idx + 1)
                self.crit_elem_type = "NODE"
        # the rigid bodies' transported master dts join the free-node min
        # (rgbodfp.F/dtnoda.F — see add_rigid_body); a no-op with no /RBODY
        dt_rb = self._rigid_body_dt()
        if dt_rb < EP30:
            dt_candidates.append(dt_rb)
        dt = min(dt_candidates) if dt_candidates else EP30

        # Support /DT/NODA/STOP: if mode == 'STOP' and dt < dt_min, flag stop
        if self.mode == "STOP" and self.dt_min > 0.0:
            if dt < self.dt_min or (self.dt_sca > 0.0 and dt * self.dt_sca < self.dt_min):
                self.stopped = True
                self.stop_node = self.crit_node_id

        self.stifn[:] = 0.0
        if self._rot:
            self.stifr[:] = 0.0
        return dt

    # ------------------------------------------------------------------
    def announce(self, mass_added_glob: float, t: float) -> None:
        """SPMD form of the 1%-step /DT/NODA/CST added-mass announcement:
        ``mass_added_glob`` is the sum over domains of the weighted local
        counters (carried by a SUM slot of the spmd_glob_min5.F packet, see
        engine.py), ``mass0`` the global initial mass.  Identical on every
        domain, so the ``_reported`` milestone stays in step everywhere."""
        if not (self.cst and self.dt_min > 0.0 and self.dt_sca > 0.0):
            return
        frac = mass_added_glob / max(self.mass0, EM20)
        if frac >= self._reported + 0.01:  # 1%-step announcements
            self.log.info(
                f" -- /DT/NODA/CST: ADDED MASS {mass_added_glob:.5E}"
                f" ({100.0 * frac:.2f}% OF THE INITIAL MASS)"
                f" AT TIME {t:.5E}")
            self._reported = frac

    # ------------------------------------------------------------------
    def _compute_set_factors(self) -> Dict[int, float]:
        factors: Dict[int, float] = {}
        if self.dt_min > 0.0 and self.dt_sca > 0.0:
            loaded = (self.stifn > 0.0) & self.free & (self.model.mass > 0.0)
            if np.any(loaded):
                m_req_set = self.stifn[loaded] * (self.dt_min / self.dt_sca) ** 2 / 2.0
                scale_mask = m_req_set > self.model.mass[loaded]
                if np.any(scale_mask):
                    loaded_indices = np.where(loaded)[0]
                    scale_idx = loaded_indices[scale_mask]
                    facs = self.model.mass[scale_idx] / m_req_set[scale_mask]
                    for s_i, fac in zip(scale_idx, facs):
                        factors[int(s_i)] = float(fac)
        return factors

    def apply_set_acceleration(self, acc: np.ndarray) -> np.ndarray:
        """Scale acceleration directly for /DT/NODA/SET: acc[idx] *= (model.mass[idx] / m_req)."""
        factors = self._set_factors if self._set_factors else self._compute_set_factors()
        for idx, fac in factors.items():
            acc[idx] *= fac
        return acc

    # ------------------------------------------------------------------
    def apply_rayleigh_damping(
        self,
        dampr: Optional[Any] = None,
        node_groups: Optional[Any] = None,
        alpha: float = 0.0,
        beta: float = 0.0,
        t: float = 0.0,
        dt1: float = 0.0,
        dt_min: Optional[float] = None,
    ) -> None:
        """Apply Rayleigh damping nodal stiffness scaling matching dtnodarayl.F:

        Upstream Fortran reference:
            $OR_SRC/engine/source/time_step/dtnodarayl.F
            SUBROUTINE DTNODARAYL(MS, IN, STIFN, STIFR, DT2T, IGRNOD, DAMPR)

        When Rayleigh mass damping (alpha / DAMPAI) or stiffness damping (beta / DAMPBI)
        is active on node groups:
            For each node:
                dt_0 = sqrt(2 * M_i / K_i)
                bb = beta / dt_0 + 0.5 * alpha * dt_0
                fac = sqrt(bb**2 + 1.0) - bb
                coeff = 1.0 / (fac**2)
                K_i = K_i * coeff   (stiffness scaling effectively scales dt_i by fac <= 1.0)
            Rotational DOFs (if present):
                dt_rot0 = sqrt(2 * I_i / Kr_i)
                bb_r = beta / dt_rot0 + 0.5 * alpha * dt_rot0
                fac_r = sqrt(bb_r**2 + 1.0) - bb_r
                coeff_r = 1.0 / (fac_r**2)
                Kr_i = Kr_i * coeff_r

        Parameters:
            dampr: Rayleigh damping definitions. Can be:
                - None: uses global alpha and beta.
                - 2D numpy array DAMPR(NRDAMP, NDAMP) as in Fortran dtnodarayl.F.
                - List of dicts, e.g. [{"alpha": 0.0, "beta": 0.05, "group": 1, "tstart": 0.0, "tstop": 1e20}, ...]
                - Single dict, e.g. {"alpha": 0.0, "beta": 0.05, "group": 1}
                - Tuple/list of (alpha, beta).
            node_groups: Mapping or list of node groups (NodeGroup objects, index arrays, or lists of node IDs/indices).
                If None, uses self.model.node_groups if available, else applies to all loaded nodes.
            alpha: Fallback/global mass damping coefficient (alpha / DAMPA).
            beta: Fallback/global stiffness damping coefficient (beta / DAMPB).
            t: Current simulation time (for evaluating tstart <= t <= tstop).
            dt1: Current/previous time step (for DT0 upper bound if positive).
            dt_min: Target minimum time step for CST floor protection (defaults to self.dt_min).
        """
        records: List[Dict[str, Any]] = []
        if dampr is None:
            if alpha > 0.0 or beta > 0.0 or (alpha == 0.0 and beta == 0.0 and node_groups is None):
                records.append({
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "group": None,
                    "tstart": -float("inf"),
                    "tstop": float("inf"),
                })
        elif isinstance(dampr, (tuple, list)) and len(dampr) == 2 and isinstance(dampr[0], (int, float)) and isinstance(dampr[1], (int, float)):
            records.append({
                "alpha": float(dampr[0]),
                "beta": float(dampr[1]),
                "group": None,
                "tstart": -float("inf"),
                "tstop": float("inf"),
            })
        elif isinstance(dampr, np.ndarray) and dampr.ndim == 2:
            nrdamp, ndamp = dampr.shape
            for col in range(ndamp):
                if nrdamp > 18 and dampr[18, col] != 0:
                    continue
                itype = int(round(dampr[20, col])) if nrdamp > 20 else 0
                if itype == 3:  # FL_FREQ_RANGE
                    continue
                ts = float(dampr[16, col]) if nrdamp > 16 else -float("inf")
                te = float(dampr[17, col]) if nrdamp > 17 else float("inf")
                grp_id = int(round(dampr[1, col])) if nrdamp > 1 else None
                a_idx = [2, 4, 6, 8, 10, 12]
                dampa = max([float(dampr[k, col]) for k in a_idx if k < nrdamp] or [0.0])
                b_idx = [3, 5, 7, 9, 11, 13]
                dampb = max([float(dampr[k, col]) for k in b_idx if k < nrdamp] or [0.0])
                records.append({"alpha": dampa, "beta": dampb, "group": grp_id, "tstart": ts, "tstop": te})
        elif isinstance(dampr, dict):
            a = float(dampr.get("alpha", dampr.get("damp_a", alpha)))
            b = float(dampr.get("beta", dampr.get("damp_b", beta)))
            grp = dampr.get("group", dampr.get("node_group", dampr.get("group_id", None)))
            ts = float(dampr.get("tstart", -float("inf")))
            te = float(dampr.get("tstop", float("inf")))
            records.append({"alpha": a, "beta": b, "group": grp, "tstart": ts, "tstop": te})
        elif isinstance(dampr, (list, tuple)):
            for item in dampr:
                if isinstance(item, dict):
                    a = float(item.get("alpha", item.get("damp_a", 0.0)))
                    b = float(item.get("beta", item.get("damp_b", 0.0)))
                    grp = item.get("group", item.get("node_group", item.get("group_id", None)))
                    ts = float(item.get("tstart", -float("inf")))
                    te = float(item.get("tstop", float("inf")))
                    records.append({"alpha": a, "beta": b, "group": grp, "tstart": ts, "tstop": te})
                elif hasattr(item, "__dict__") or hasattr(item, "alpha") or hasattr(item, "damp_a"):
                    a = float(getattr(item, "alpha", getattr(item, "damp_a", 0.0)))
                    b = float(getattr(item, "beta", getattr(item, "damp_b", 0.0)))
                    grp = getattr(item, "group", getattr(item, "node_group", getattr(item, "group_id", None)))
                    ts = float(getattr(item, "tstart", -float("inf")))
                    te = float(getattr(item, "tstop", float("inf")))
                    records.append({"alpha": a, "beta": b, "group": grp, "tstart": ts, "tstop": te})
        elif hasattr(dampr, "__dict__") or hasattr(dampr, "alpha") or hasattr(dampr, "damp_a"):
            a = float(getattr(dampr, "alpha", getattr(dampr, "damp_a", alpha)))
            b = float(getattr(dampr, "beta", getattr(dampr, "damp_b", beta)))
            grp = getattr(dampr, "group", getattr(dampr, "node_group", getattr(dampr, "group_id", None)))
            ts = float(getattr(dampr, "tstart", -float("inf")))
            te = float(getattr(dampr, "tstop", float("inf")))
            records.append({"alpha": a, "beta": b, "group": grp, "tstart": ts, "tstop": te})

        if not records:
            return

        def _get_target_indices(grp_id: Any) -> Optional[np.ndarray]:
            if grp_id is None:
                return None
            target_grp = None
            if node_groups is not None:
                if isinstance(node_groups, dict):
                    target_grp = node_groups.get(grp_id)
                    if target_grp is None and isinstance(grp_id, int):
                        target_grp = node_groups.get(str(grp_id))
                elif isinstance(node_groups, (list, tuple)):
                    if isinstance(grp_id, int):
                        if 0 <= grp_id < len(node_groups):
                            target_grp = node_groups[grp_id]
                        elif 1 <= grp_id <= len(node_groups):
                            target_grp = node_groups[grp_id - 1]
            if target_grp is None and hasattr(self.model, "node_groups") and self.model.node_groups:
                if isinstance(grp_id, int) and grp_id in self.model.node_groups:
                    target_grp = self.model.node_groups[grp_id]
                elif isinstance(grp_id, str):
                    for gid, g in self.model.node_groups.items():
                        if getattr(g, "title", "") == grp_id or str(gid) == grp_id:
                            target_grp = g
                            break
            if target_grp is None and isinstance(grp_id, (list, tuple, np.ndarray)):
                target_grp = grp_id

            if target_grp is None:
                return None

            if hasattr(target_grp, "node_idx") and target_grp.node_idx is not None:
                return np.asarray(target_grp.node_idx, dtype=int)
            if hasattr(target_grp, "node_ids") and target_grp.node_ids is not None:
                raw_ids = target_grp.node_ids
            elif hasattr(target_grp, "entity") and target_grp.entity is not None:
                raw_ids = target_grp.entity
            elif hasattr(target_grp, "nodes") and target_grp.nodes is not None:
                raw_ids = target_grp.nodes
            else:
                raw_ids = target_grp

            raw_arr = np.asarray(raw_ids, dtype=int)
            if len(raw_arr) == 0:
                return np.zeros(0, dtype=int)
            numnod = self.model.numnod
            if hasattr(self.model, "node_ids") and self.model.node_ids is not None and len(self.model.node_ids) == numnod:
                if hasattr(self.model, "_id2idx") and self.model._id2idx:
                    mapped = [self.model._id2idx[nid] for nid in raw_arr if nid in self.model._id2idx]
                    if mapped:
                        return np.asarray(mapped, dtype=int)
                if np.all(raw_arr >= 0) and np.all(raw_arr < numnod):
                    return raw_arr
                mask = np.isin(self.model.node_ids, raw_arr)
                return np.where(mask)[0]
            valid = raw_arr[(raw_arr >= 0) & (raw_arr < numnod)]
            return valid

        dt_target = dt_min if dt_min is not None else self.dt_min
        dt_sca = self.dt_sca if self.dt_sca > 0.0 else 0.9
        dtmi2 = (dt_target / dt_sca) if (dt_target > 0.0 and dt_sca > 0.0) else 0.0

        for rec in records:
            if t < rec["tstart"] or t > rec["tstop"]:
                continue
            dampar_val = rec["alpha"]
            dampbi_val = rec["beta"]

            if dampar_val == 0.0 and dampbi_val == 0.0:
                continue

            # Minimum time step factor (dtnodarayl.F lines 130-135)
            faci = 0.0
            if dtmi2 > 0.0:
                bbi_0 = dampbi_val + 0.5 * dampar_val * (dtmi2 ** 2)
                dtn2 = math.sqrt(dtmi2 ** 2 + 2.0 * bbi_0 * dtmi2)
                bbi = (dampbi_val / dtn2) + 0.5 * dampar_val * dtn2
                faci = math.sqrt(bbi ** 2 + 1.0) - bbi

            target_idx = _get_target_indices(rec["group"])
            if target_idx is None:
                mask = (self.stifn > EM20) & (self.model.mass > 0.0)
                nodes_to_scale = np.where(mask)[0]
            else:
                if len(target_idx) == 0:
                    continue
                valid_mask = (self.stifn[target_idx] > EM20) & (self.model.mass[target_idx] > 0.0)
                nodes_to_scale = target_idx[valid_mask]

            if len(nodes_to_scale) > 0:
                dt0 = np.sqrt(2.0 * self.model.mass[nodes_to_scale] / self.stifn[nodes_to_scale])
                dt0 = np.minimum(1.0e4, dt0)
                bb = (dampbi_val / np.maximum(dt0, EM20)) + 0.5 * dampar_val * dt0
                fac = np.sqrt(bb ** 2 + 1.0) - bb
                if dtmi2 > 0.0:
                    fac = np.where(fac * dt0 < dtmi2, np.maximum(fac, faci), fac)
                coeff = 1.0 / np.maximum(fac ** 2, EM20)
                self.stifn[nodes_to_scale] *= coeff

            # Rotational DOFs (dtnodarayl.F lines 148-163)
            if self._rot and hasattr(self.model, "inertia") and self.model.inertia is not None:
                if target_idx is None:
                    rot_mask = (self.stifr > EM20) & (self.model.inertia > 0.0)
                    rot_nodes = np.where(rot_mask)[0]
                else:
                    valid_rot = (self.stifr[target_idx] > EM20) & (self.model.inertia[target_idx] > 0.0)
                    rot_nodes = target_idx[valid_rot]
                if len(rot_nodes) > 0:
                    dt0_r = np.sqrt(2.0 * self.model.inertia[rot_nodes] / self.stifr[rot_nodes])
                    dt0_r = np.minimum(1.0e4, dt0_r)
                    bb_r = (dampbi_val / np.maximum(dt0_r, EM20)) + 0.5 * dampar_val * dt0_r
                    fac_r = np.sqrt(bb_r ** 2 + 1.0) - bb_r
                    if dtmi2 > 0.0:
                        fac_r = np.where(fac_r * dt0_r < dtmi2, np.maximum(fac_r, faci), fac_r)
                    coeff_r = 1.0 / np.maximum(fac_r ** 2, EM20)
                    self.stifr[rot_nodes] *= coeff_r

    def apply_rayleigh_damping_stiffness(self, alpha: float, beta: float) -> None:
        """Apply Rayleigh damping stiffness scaling matching dtnodarayl.F:

        For each loaded node with dt_0 = sqrt(2 M / K):
            BB = beta / dt_0 + 0.5 * alpha * dt_0
            FAC = sqrt(BB^2 + 1) - BB
            COEFF = 1 / FAC^2
            K = K * COEFF
        """
        self.apply_rayleigh_damping(alpha=alpha, beta=beta)

    def step(
        self,
        mass_eff: np.ndarray,
        inv_mass: np.ndarray,
        v: np.ndarray,
        t: float,
        claims: Optional[Any] = None,
        dampr: Optional[Any] = None,
        node_groups: Optional[Any] = None,
        inertia: Optional[np.ndarray] = None,
        inv_inertia: Optional[np.ndarray] = None,
        ams_nodes: Optional[np.ndarray] = None,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> float:
        """Single-call interface: optionally assemble claims, apply Rayleigh damping, and execute apply()."""
        if claims is not None:
            self.assemble(claims)
        if dampr is not None or alpha > 0.0 or beta > 0.0:
            self.apply_rayleigh_damping(
                dampr=dampr,
                node_groups=node_groups,
                alpha=alpha,
                beta=beta,
                t=t,
            )
        return self.apply(
            mass_eff,
            inv_mass,
            v,
            t,
            inertia=inertia,
            inv_inertia=inv_inertia,
            ams_nodes=ams_nodes,
        )

    # ------------------------------------------------------------------
    def get_top_mass_nodes(self, n: int = 5) -> List[Tuple[int, float, float]]:
        """Return top n nodes by cumulative added mass: (node_id, delta_m, delta_m / m0)."""
        if not hasattr(self, "node_added_mass") or len(self.node_added_mass) == 0:
            return []
        active = np.where(self.node_added_mass > 0.0)[0]
        if len(active) == 0:
            return []
        order = active[np.argsort(-self.node_added_mass[active])]
        top_indices = order[:n]
        result = []
        for idx in top_indices:
            nid = int(self.model.node_ids[idx]) if (hasattr(self.model, "node_ids") and self.model.node_ids is not None and len(self.model.node_ids) > idx) else int(idx + 1)
            dm = float(self.node_added_mass[idx])
            m0 = float(self.initial_nodal_mass[idx]) if (hasattr(self, "initial_nodal_mass") and len(self.initial_nodal_mass) > idx) else 0.0
            rel = dm / max(m0, EM20) if m0 > 0.0 else (dm / max(self.mass0, EM20))
            result.append((nid, dm, rel))
        return result

    # ------------------------------------------------------------------
    def compute_target_dt(self, target_percent_addmass: float,
                          dt_scale: Optional[float] = None,
                          total_mass: Optional[float] = None) -> float:
        """Compute target time step for given added mass percentage matching find_dt_target.F."""
        dt_sca = dt_scale if dt_scale is not None else self.dt_sca
        tot_m = total_mass if total_mass is not None else self.mass0
        loaded = (self.stifn > EM20) & self.free & (self.model.mass > 0.0)
        return compute_target_dt(target_percent_addmass, dt_scale=dt_sca,
                                 total_mass=tot_m, ms=self.model.mass[loaded],
                                 stifn=self.stifn[loaded])

    # ------------------------------------------------------------------
    def summary(self, log) -> None:
        """Termination-page report (the honesty contract)."""
        if not self.cst:
            return
        frac = 100.0 * self.mass_added / max(self.mass0, EM20)
        log.info(f"     ADDED MASS (/DT/NODA/CST) : {self.mass_added:14.7E}"
                 f"  ({frac:.3f}% OF INITIAL MASS)")
        if self.iner_added > 0.0:
            # rotational analogue of DMAST: dtnoda.F's DINERT counter
            log.info(f"     ADDED INERTIA (DINERT). . : "
                     f"{self.iner_added:14.7E}")
        log.info(f"     ENERGY FROM ADDED MASS  . : {self.e_madd:14.7E}")
        log.info(f"     MOMENTUM FROM ADDED MASS  : "
                 f"{self.mom_added[0]:12.5E} {self.mom_added[1]:12.5E} "
                 f"{self.mom_added[2]:12.5E}")
        top_nodes = self.get_top_mass_nodes(n=5)
        if top_nodes:
            log.info("     TOP NODES BY ADDED MASS:")
            for nid, dm, rel in top_nodes:
                log.info(f"       NODE {nid:8d} : {dm:14.7E}  ({100.0 * rel:6.2f}%)")


# ----------------------------------------------------------------------
def compute_target_dt(
    target_percent_addmass: float,
    dt_scale: float = 0.9,
    total_mass: float = 0.0,
    ms: Optional[np.ndarray] = None,
    stifn: Optional[np.ndarray] = None,
) -> float:
    """Compute target time step matching Fortran find_dt_target.F.

    Calculates target dt given requested percentage of added mass.
    """
    if ms is None or stifn is None or len(ms) == 0:
        return 0.0

    threshold = target_percent_addmass if target_percent_addmass <= 1.0 else target_percent_addmass / 100.0
    if total_mass <= 0.0:
        total_mass = float(ms.sum())

    valid = (ms > 0.0) & (stifn > EM20)
    if not np.any(valid):
        return 0.0

    ms_v = ms[valid]
    stf_v = stifn[valid]

    # dt2_l = M / K (proportional to 0.5 * dt_i^2)
    dt2_l = ms_v / stf_v
    perm = np.argsort(dt2_l)
    dt_sorted = dt2_l[perm]
    ms_sorted = ms_v[perm]
    stf_sorted = stf_v[perm]

    nnod = len(dt_sorted)
    sumk = 0.0
    summ = 0.0
    sumk_old = 0.0
    summ_old = 0.0
    target_dt = 0.0

    for i in range(nnod):
        if i > 0:
            if dt_sorted[i] > dt_sorted[i - 1]:
                sumk_old = sumk
                summ_old = summ
        per_adm = (dt_sorted[i] * sumk_old - summ_old) / max(EM20, total_mass)
        if i > 0 and per_adm > threshold:
            target_dt = dt_scale * np.sqrt(2.0 * (total_mass * threshold + summ_old) / max(EM20, sumk_old))
            return float(target_dt)
        sumk += stf_sorted[i]
        summ += ms_sorted[i]
        if i == nnod - 1:
            target_dt = dt_scale * np.sqrt(2.0 * (total_mass * threshold + summ) / max(EM20, sumk))
            return float(target_dt)

    return float(target_dt)


# ----------------------------------------------------------------------
def apply_rayleigh_damping_nodal(
    stifn: np.ndarray,
    mass: np.ndarray,
    alpha: float = 0.0,
    beta: float = 0.0,
    stifr: Optional[np.ndarray] = None,
    inertia: Optional[np.ndarray] = None,
    node_indices: Optional[Sequence[int]] = None,
    dt_min: float = 0.0,
    dt_scale: float = 0.9,
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """Functional interface for Rayleigh damping stiffness reduction matching dtnodarayl.F:

    Upstream Fortran reference:
        $OR_SRC/engine/source/time_step/dtnodarayl.F
        SUBROUTINE DTNODARAYL

    Formulation:
        dt_0 = sqrt(2 * M_i / K_i)
        bb = beta / dt_0 + 0.5 * alpha * dt_0
        fac = sqrt(bb**2 + 1.0) - bb
        coeff = 1.0 / (fac**2)
        K_i = K_i * coeff

    Returns:
        (scaled_stifn, scaled_stifr, min_fac)
    """
    stifn_out = stifn.copy()
    stifr_out = stifr.copy() if stifr is not None else None
    min_fac = 1.0

    if alpha == 0.0 and beta == 0.0:
        return stifn_out, stifr_out, min_fac

    dtmi2 = (dt_min / dt_scale) if (dt_min > 0.0 and dt_scale > 0.0) else 0.0
    faci = 0.0
    if dtmi2 > 0.0:
        bbi_0 = beta + 0.5 * alpha * (dtmi2 ** 2)
        dtn2 = math.sqrt(dtmi2 ** 2 + 2.0 * bbi_0 * dtmi2)
        bbi = (beta / dtn2) + 0.5 * alpha * dtn2
        faci = math.sqrt(bbi ** 2 + 1.0) - bbi

    if node_indices is not None:
        idx = np.asarray(node_indices, dtype=int)
        mask = (stifn_out[idx] > EM20) & (mass[idx] > 0.0)
        target = idx[mask]
    else:
        mask = (stifn_out > EM20) & (mass > 0.0)
        target = np.where(mask)[0]

    if len(target) > 0:
        dt0 = np.sqrt(2.0 * mass[target] / stifn_out[target])
        dt0 = np.minimum(1.0e4, dt0)
        bb = (beta / np.maximum(dt0, EM20)) + 0.5 * alpha * dt0
        fac = np.sqrt(bb ** 2 + 1.0) - bb
        if dtmi2 > 0.0:
            fac = np.where(fac * dt0 < dtmi2, np.maximum(fac, faci), fac)
        min_fac = float(np.min(fac))
        coeff = 1.0 / np.maximum(fac ** 2, EM20)
        stifn_out[target] *= coeff

    if stifr_out is not None and inertia is not None:
        if node_indices is not None:
            r_mask = (stifr_out[idx] > EM20) & (inertia[idx] > 0.0)
            r_target = idx[r_mask]
        else:
            r_mask = (stifr_out > EM20) & (inertia > 0.0)
            r_target = np.where(r_mask)[0]
        if len(r_target) > 0:
            dt0_r = np.sqrt(2.0 * inertia[r_target] / stifr_out[r_target])
            dt0_r = np.minimum(1.0e4, dt0_r)
            bb_r = (beta / np.maximum(dt0_r, EM20)) + 0.5 * alpha * dt0_r
            fac_r = np.sqrt(bb_r ** 2 + 1.0) - bb_r
            if dtmi2 > 0.0:
                fac_r = np.where(fac_r * dt0_r < dtmi2, np.maximum(fac_r, faci), fac_r)
            coeff_r = 1.0 / np.maximum(fac_r ** 2, EM20)
            stifr_out[r_target] *= coeff_r

    return stifn_out, stifr_out, min_fac
