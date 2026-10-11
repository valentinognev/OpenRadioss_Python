"""
The Engine main program: the explicit time-integration loop.

Fortran origin: ``engine/source/engine/resol.F`` — the ~5000-line heart of
OpenRadioss. This module is deliberately the best-commented file of the
port: read it top to bottom to understand one complete explicit cycle.

The central-difference scheme
-----------------------------
Explicit dynamics staggers velocities and positions in time (leap-frog):

    known:  x^n  (positions at t_n),  v^{n-1/2}  (velocities at midstep)

    1.  f^n     = f_ext(t_n) - f_int(x^n, state)      forces at t_n
    2.  a^n     = f^n / m                              (lumped mass: trivial
                                                        'solve', the reason
                                                        explicit codes never
                                                        assemble a matrix)
    3.  v^{n+1/2} = v^{n-1/2} + a^n * dt               velocity update
    4.  kinematic conditions overwrite v^{n+1/2}       (BCS, /IMPVEL, walls)
    5.  x^{n+1}   = x^n + v^{n+1/2} * dt               position update
    6.  dt        = k * min(element/interface dt)      next stable step

The scheme is 2nd-order accurate and conditionally stable: dt must resolve
the fastest wave crossing the smallest element — the Courant condition.
Every cycle each element kernel returns its own critical dt and the global
step is the scaled minimum (dt_sca, default 0.9).

Energy balance
--------------
The classic quality metric of an explicit run (printed every /PRINT
cycles, exactly like the original's listing):

    error% = (IE + KE + HE + CE - EW - E0) / max(E0, EW, IE+KE) * 100

with IE internal, KE kinetic, HE hourglass, CE contact energy and EW the
external work (loads + kinematic conditions). A growing |error| signals
instability; the Engine stops beyond ``/STOP``'s threshold (default 15%),
like the original's energy-error check in resol.F.
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

import numpy as np

from .. import banner
from ..accel import auto_select_backend
from ..common.constants import EP30
from ..common.messages import MessageLog
from ..contact import build_contacts
from ..elements import KERNELS, shell_nonlocal
from ..input.deck_reader import read_deck
from ..input.engine_keywords import parse_engine_deck
from ..input.idsolid import solid_group_fragments
from ..input.mat_reader import refuse_inactive_materials
from ..input.prop_reader import refuse_inactive_properties
from ..model.model import EngineControls, Model
from ..output import TimeHistory, write_anim_state
from ..spmd.exchange import SpmdContext, serial_context
from ..starter.restart import read_restart, write_restart
from .airbag import update_airbag_thermodynamics, update_airbag_volume, apply_airbag_forces
from .airbag_fvm import update_fvmbag_volume, update_fvmbag_thermodynamics, apply_fvmbag_forces
from .damping import Dampers, DynamicRelaxation
from .kinematics import LoadsAndConstraints
from .lagmul import LagmulSolver
from .mass_scaling import NodalTimeStep
from .ams import AMSManager
from .mpc import build_mpc
from .rbe3 import build_rbe3
from .rigid_body import build_rigid_bodies
from .rigid_wall import RigidWalls
from .cyl_joint import build_cyl_joints
from .gjoint import build_gjoints
from .rlink import build_rlinks
from .kjoint import build_kjoints
from .sections import SectionForces
from .sensors import Sensors
from .element_erosion import compute_sdlenmax, check_solid_geometric_erosion, SdLenMaxArray  # noqa: F401



def run_name_from_input(path: str):
    """'MYRUN_0002.rad' -> ('MYRUN', 2) — the run number selects the
    restart to resume from (RunName_{nn-1}.rst) and names the outputs
    (RunName_{nn}.out, RunName T{nn} history, RunName_{nn}.rst)."""
    base = os.path.basename(path)
    m = re.match(r"(.+)_(\d{4})\.rad$", base, re.IGNORECASE)
    if m:
        return m.group(1), int(m.group(2))
    return os.path.splitext(base)[0], 1


class EngineState:
    """Everything the loop tracks besides the model itself."""

    def __init__(self):
        self.t = 0.0
        self.cycle = 0
        self.wext = 0.0        # accumulated external work
        self.econt = 0.0       # accumulated contact + rigid-wall energy
        self.ndel = 0          # deleted elements reported so far (/FAIL)
        self.epeak = 0.0       # running peak of IE+KE (error reference)
        self.e_num = 0.0       # numerical dissipation of the elements (M6)
        self.e_booked_prev = 0.0   # helper for the e_num ledger
        self.e_madd = 0.0      # kinetic energy from /DT/NODA/CST mass (M6)
        self.e_damp = 0.0      # /DAMP dissipation (M6)
        self.dt_prev = None    # previous cycle time step for leapfrog dt12
        self.stop_reason = ""
        self.crit_elem_id = 0  # critical element user ID (NELTST)
        self.crit_elem_type = ""  # critical element group name (ITYPTST)
        self.ek_ams = 0.0      # dual kinetic energy under AMS (sms_encin_2.F)
        self.e0 = None         # balance reference E0 (was _energies.e0)
        # SPMD (pyradioss/spmd): the domain context while a decomposed run
        # integrates (None in a serial run). Under SPMD the ledgers above
        # (wext, econt, e_num, e_madd, e_damp, ndel) are the LOCAL partial
        # sums of this domain; e0, epeak, t, cycle, dt_prev are global.
        self.spmd = None
        self.timet_glob = False  # /STOP/TIMET flag from the dt packet

    @property
    def e_ext(self) -> float:
        """Alias for wext (external work ledger)."""
        return self.wext

    @e_ext.setter
    def e_ext(self, val: float) -> None:
        self.wext = val


def _deleted_count(model: Model) -> int:
    """Total deleted elements (GBUF%OFF == 0) across all groups."""
    ndel = 0
    for _, group in model.element_groups():
        off = group.state.get("off")
        if off is not None:
            ndel += int((off == 0.0).sum())
    return ndel


def _matches_elem_dt_control(name: str, key: str) -> bool:
    """Check if element group name matches /DT/<elem> keyword specification (M580, M605)."""
    k = key.upper()
    if k in ("ELEM", "ALL"): return True
    if k in ("SOLID", "SOLI", "HEXA", "BRIC", "BRICK"):
        return any(s in name for s in solid_group_fragments())
    if k in ("PENTA", "PENTA6", "WEDGE") and ("penta" in name or "wedge" in name):
        return True
    if k in ("SHELL", "SHEL", "COQUE"):
        return ("shell" in name and "tshell" not in name) or "coque" in name
    if k in ("SH3N", "SH_3N", "TRI", "TRIA") and ("sh3n" in name or "tri" in name):
        return True
    if k in ("TSHELL", "TSH", "SHEL16") and ("tshell" in name or "shel16" in name):
        return True
    if k in ("TETRA", "TETRA10", "TETRA4") and "tetra" in name:
        return True
    if k in ("QUAD", "QUA") and "quad" in name:
        return True
    if k in ("BEAM",) and "beam" in name:
        return True
    if k in ("TRUSS",) and "truss" in name:
        return True
    if k in ("SPRING", "SPRIN") and "spring" in name:
        return True
    return False


def _element_energy_sum(model: Model) -> float:
    """Sum of the kernels' state-function bookings (eint + ehour) — the
    reference the numerical-dissipation ledger (see the cycle's step 6c)
    differences against."""
    e = 0.0
    for _, group in model.element_groups():
        e += float(group.state["eint"].sum())
        e += float(group.state["ehour"].sum())
    return e


# Startup conditioning for the energy-balance guards (M41).
#
# The percentage energy-error checks are only meaningful once the run
# carries a non-trivial energy scale.  During the first cycles of a
# slowly-loaded deck (RD-E-1000 Sf_0.1: /DT/NODA scale 0.1) the total
# reference energy is numerical dust — IE/KE/EW all ~1e-8 in the deck's
# units — and the leapfrog half-step lag between the accumulated
# external-work Riemann sum and the realized IE+KE reads as a huge
# percentage (ERR -84% at cycle 1, decaying to -17% at cycle 100, then
# to ~0 as the energy grows).  Judging %-error against that dust is
# ill-posed, so both guards are held INERT until the reference energy
# rises above this absolute floor.
#
# This mirrors the upstream engine, which conditions its ENERGY ERROR
# the same way (engine/source/output/ecrit.F):
#   * the error is computed only when the reference/expected total
#     energy clears an absolute floor —
#       IF(ABS(ENTOT1B) > EM20) THEN  ERR = ENTOTB/ENTOT1B - ONE
#       ELSE                          ERR = ZERO           (check inert)
#   * the reported percentage is clamped to ±99.9% (X99), never "huge";
#   * and the stop thresholds DEMXK/DEMXS default to EP20/EP30 (1e20 /
#     1e30) when the deck sets no /STOP-Emax or /KILL card (freform.F),
#     i.e. upstream does not abort on energy error by default at all —
#     which is why Fortran runs these Sf_0.1 decks to NORMAL.
# The port keeps a live 15% default limit as a divergence safety net
# (M37 — a young port wants the guard on), so the floor is sized to the
# numerical-dust band rather than to EM20's bare divide-by-zero role:
# ~6 orders below any physical energy these decks reach (IE climbs to
# ~1e0..1e6), so a genuine instability — which runs energy away
# super-linearly (the h=-1 teeth control reaches ~1e9) — crosses the
# floor within a handful of cycles and still trips.  The 15% / 30%
# limits themselves are UNCHANGED.
_ENERGY_START_FLOOR = 1.0e-6


def _energies(model: Model, state: EngineState, extra=None) -> dict:
    # SPMD (decomposed run): local partial sums + ONE all-reduce, see
    # _energies_spmd. ``extra`` rides along in that reduction (serial:
    # ignored).
    sp = getattr(state, "spmd", None)
    if sp is not None and sp.active:
        return _energies_spmd(model, state, sp, extra)
    # the balance reference lives on the state (the function attribute is
    # kept in step for callers that build an EngineState by hand)
    e0 = state.e0 if getattr(state, "e0", None) is not None else _energies.e0
    ie = he = 0.0
    for _, group in model.element_groups():
        if "eint" in group.state:
            ie += float(np.sum(group.state["eint"]))
        if "ehour" in group.state:
            he += float(np.sum(group.state["ehour"]))
    real = model.mass < 1e29
    ke = float(0.5 * (model.mass[real, None] * model.v[real] ** 2).sum())
    if getattr(model, "inertia", None) is not None and getattr(model, "vr", None) is not None:
        real_rot = model.inertia < 1e29
        ke += float(0.5 * (model.inertia[real_rot, None] * model.vr[real_rot] ** 2).sum())
    total = ie + ke + he + state.econt + state.e_num + state.e_damp
    # the error reference is the ENERGY SCALE OF THE RUN: the largest of
    # the initial energy, external work, current energies and the running
    # peak of IE+KE. The initial/peak terms matter for oscillating or
    # fully-arrested systems (a pendulum at its turning point, a block
    # brought to rest on a wall): all instantaneous energies pass through
    # zero there, and normalizing a fixed round-off residual by them
    # would scream divergence where there is none. (The original guards
    # its error the same way, with the initial/reference energy.)
    state.epeak = max(state.epeak, ie + ke)
    ref = max(abs(state.wext), ke, ie, state.epeak, abs(e0),
              1e-12)
    # e_madd: kinetic energy CREATED by /DT/NODA/CST mass additions at
    # moving nodes — an energy input like external work, reported by the
    # mass-scaling summary (the honesty contract of engine/mass_scaling)
    err = (total - state.wext - state.e_madd - e0) / ref * 100.0
    # ERRN: the numerical-dissipation ledger on the same % scale. A
    # diverging run drives EN hard NEGATIVE (the discrete elastic force
    # injects energy the state bookings cannot see) — and because EN is
    # part of the balance, |ERR| alone would stay blind to exactly that
    # divergence; the Engine stops on ERRN < -limit (step 7).
    return {"IE": ie, "KE": ke, "HE": he, "CE": state.econt,
            "EN": state.e_num, "DE": state.e_damp, "EW": state.wext,
            "ERR": err, "ERRN": state.e_num / ref * 100.0,
            # REF: the absolute energy scale the two percentages are
            # normalized by — the divergence guards gate on it so they
            # stay inert while it is numerical dust (see
            # _ENERGY_START_FLOOR and the M41 startup conditioning).
            "REF": ref}


def _energies_spmd(model: Model, state: EngineState, sp, extra=None) -> dict:
    """``_energies`` of a decomposed run (``ecrit.F`` under SPMD): every
    domain sums its own share — the internal/hourglass energies of its
    elements (partitioned: each element lives on exactly one domain) and
    the kinetic energy of its nodes times ``WEIGHT`` (``ecrit.F``
    ``MAS = MS(I)*WEIGHT_MD(I)``) — together with its partial ledgers
    (contact, numerical dissipation, damping, external work, added-mass
    energy); ONE all-reduce of that short vector (``spmd_glob_dsum``)
    gives the global values, from which every domain derives the SAME
    balance (so the divergence guards decide identically everywhere).
    ``extra`` values are summed in the same reduction and returned under
    the key ``"EXTRA"``."""
    w = sp.weight
    ie = he = 0.0
    for _, group in model.element_groups():
        if "eint" in group.state:
            ie += float(np.sum(group.state["eint"]))
        if "ehour" in group.state:
            he += float(np.sum(group.state["ehour"]))
    real = model.mass < 1e29
    ke = float(0.5 * ((w[real] * model.mass[real])[:, None]
                      * model.v[real] ** 2).sum())
    if getattr(model, "inertia", None) is not None and \
            getattr(model, "vr", None) is not None:
        real_rot = model.inertia < 1e29
        ke += float(0.5 * ((w[real_rot] * model.inertia[real_rot])[:, None]
                           * model.vr[real_rot] ** 2).sum())
    ext = [] if extra is None else [float(x) for x in extra]
    vec = np.array([ie, he, ke, state.econt, state.e_num, state.e_damp,
                    state.wext, state.e_madd] + ext, dtype=np.float64)
    g = sp.sum(vec)
    ie, he, ke, econt, e_num, e_damp, wext, e_madd = (float(x) for x in g[:8])
    e0 = state.e0 if state.e0 is not None else _energies.e0
    total = ie + ke + he + econt + e_num + e_damp
    state.epeak = max(state.epeak, ie + ke)      # global (identical everywhere)
    ref = max(abs(wext), ke, ie, state.epeak, abs(e0), 1e-12)
    err = (total - wext - e_madd - e0) / ref * 100.0
    return {"IE": ie, "KE": ke, "HE": he, "CE": econt,
            "EN": e_num, "DE": e_damp, "EW": wext,
            "ERR": err, "ERRN": e_num / ref * 100.0,
            "REF": ref, "EXTRA": g[8:]}


class _QuietLog(MessageLog):
    """The message log of a non-root SPMD domain: ``spmd_chkw.F`` — only
    domain 0 prints; the others keep counting their messages and write to
    their listing only when one is attached (PYRADIOSS_SPMD_LOG_ALL)."""

    def _emit(self, text: str) -> None:
        if self._listing is not None:
            self._listing.write(text + "\n")


def _refuse_spmd(controls, model) -> None:
    """Engine options the SPMD port does not decompose (contract: the
    Engine refuses them under NSPMD > 1 instead of running them wrong)."""
    bad = []
    if getattr(controls, "implicit", False):
        bad.append("/IMPL (implicit solver)")
    if getattr(controls, "impl_eigv", False) \
            or getattr(model, "eigen_modes", None):
        # (/EIG/OFF alone only deactivates modes — nothing to refuse)
        bad.append("/EIG (eigenvalue extraction)")
    if getattr(controls, "dt_ams", False):
        bad.append("/DT/AMS (advanced mass scaling)")
    if getattr(controls, "dyrel_active", False):
        bad.append("/DYREL (dynamic relaxation)")
    if getattr(controls, "kerel_active", False):
        bad.append("/KEREL (kinetic relaxation)")
    if getattr(controls, "adyrel_active", False):
        bad.append("/ADYREL (adaptive dynamic relaxation)")
    if bad:
        raise RuntimeError(
            "SPMD (NSPMD > 1): not supported by the domain-decomposed "
            "engine — run with -np 1: " + ", ".join(bad))


def run_engine(input_file: str, log: Optional[MessageLog] = None,
               comm=None) -> Model:
    """Run the Engine on ``RunName_NNNN.rad`` (+ RunName_{NNNN-1}.rst).

    Run number 1 starts fresh from the Starter restart; run numbers 2+
    RESUME from the previous engine run's restart (M6 chaining — see
    starter/restart.py for the contract).

    ``comm`` (SPMD, pyradioss/spmd): a communicator of size NSPMD > 1 —
    this call then runs domain ``comm.rank`` of a decomposed model
    (``radioss2.F``/``inipar.F``): it reads its domain restart
    ``RunName_{NNNN-1}_{rank+1:04d}.rst`` and exchanges with the other
    domains every cycle; domain 0 writes the listing and every output from
    the gathered global view and returns that GLOBAL model; the other
    domains return their local model.  ``comm=None`` (or size 1) is the
    unchanged serial run."""
    spmd_on = comm is not None and comm.active
    log = log or (MessageLog() if not spmd_on or comm.is_root
                  else _QuietLog())
    run_name, run_num = run_name_from_input(input_file)
    out_dir = os.path.dirname(os.path.abspath(input_file))

    listing_path = os.path.join(out_dir, f"{run_name}_{run_num:04d}.out")
    if spmd_on and not comm.is_root:
        # spmd_chkw.F: only P0 writes the listing (optional per-domain
        # listings for debugging)
        if os.environ.get("PYRADIOSS_SPMD_LOG_ALL", "") == "1":
            listing_path = os.path.join(
                out_dir, f"{run_name}_{run_num:04d}_{comm.rank + 1:04d}.out")
        else:
            listing_path = os.devnull
    with open(listing_path, "w") as listing:
        log.attach_listing(listing)
        log.info(banner())
        log.info(f" ENGINE INPUT FILE  . . . . . . . . . : {input_file}")

        # ---- read controls + restart (engine lectur.F + rdresb.F) --------
        controls = parse_engine_deck(read_deck(input_file), log)
        if spmd_on:
            # radioss2.F: domain ISPMD reads RunName_{run-1}_{ISPMD+1}.rst
            rst = os.path.join(
                out_dir, f"{run_name}_{run_num - 1:04d}_{comm.rank + 1:04d}.rst")
        else:
            rst = os.path.join(out_dir, f"{run_name}_{run_num - 1:04d}.rst")
        log.info(f" RESTART FILE . . . . . . . . . . . . : {rst}")
        model, saved = read_restart(rst)
        spmd = None
        gmodel = None
        if spmd_on:
            info = getattr(model, "spmd", None)
            if info is None or int(getattr(info, "nspmd", 1)) <= 1:
                raise RuntimeError(
                    f"SPMD: {rst} is not a domain restart (run the Starter "
                    f"with -np {comm.size})")
            _refuse_spmd(controls, model)
            spmd = SpmdContext(comm, info)
            log.info(f" SPMD DOMAINS (NSPMD) . . . . . . . . : {spmd.size}"
                     f"  ({comm.kind})")
            if spmd.is_root:
                # the rank-0 OUTPUT VIEW: the global model of the Starter
                # (spmd_collect.F gathers every domain's rows into it at
                # each output time). A resumed chain reads the gathered
                # global restart of the previous run instead.
                gpath = os.path.join(out_dir,
                                     f"{run_name}_{run_num - 1:04d}.rst")
                if not os.path.isfile(gpath):
                    gpath = info.global_rst
                gmodel, _ = read_restart(gpath)
                gmodel.spmd = None
        # M37: a model whose element groups reference a parsed-but-not-
        # implemented material (InactiveMaterial) is honestly not
        # simulatable — refuse loudly BEFORE any engine branch runs,
        # naming every offending law (raises InactiveMaterialError).
        refuse_inactive_materials(model, log)
        # M38: the same refusal for element groups that reference a
        # parsed-but-not-implemented PROPERTY (InactiveProperty — INJECT1,
        # TSHELL/TYPE20, composite stacks ...); raises
        # InactivePropertyError.  A property defined but referenced by no
        # elements (an /PROP/INJECT1 used only by a /MONVOL) does not block.
        refuse_inactive_properties(model, log)
        log.info(f" MODEL TITLE  . . . . . . . . . . . . : {model.title}")
        # M7/M40: which compute backend runs the kernels (accel package).
        # Resolved HERE — after the restart is read — because the M40 'auto'
        # default enables numba only once the model is known to be large
        # enough to amortise JIT warm-up (accel.auto_select_backend keys on
        # the element count; small decks stay NumPy).  PYRADIOSS_BACKEND /
        # -backend numpy|numba still PIN a backend, and a pinned-but-missing
        # numba surfaces its fallback warning here.  numba auto-enable is
        # gated to the explicit /RUN path (the M39 speed data's envelope);
        # this runs before the first kernel call either way.
        auto_select_backend(model, log, explicit=not controls.implicit)
        if saved is not None:
            log.info(f" RESUMING FROM TIME . . . . . . . . . : "
                     f"{saved['t']:12.5E} (CYCLE {saved['cycle']})")
        log.info(f" FINAL TIME (/RUN)  . . . . . . . . . : "
                 f"{controls.t_end:12.5E}")

        if controls.implicit:
            # M8: /IMPL switches to the implicit-static Newton driver — a
            # PARALLEL entry point that reuses the element force kernels for
            # the residual and adds the assembled tangent stiffness. The
            # explicit leap-frog loop below is left completely untouched.
            # M10: /IMPL/DYNA selects the implicit DYNAMIC (Newmark/HHT)
            # branch of that driver instead; a bare /IMPL stays static.
            if controls.impl_dyna:
                from ..implicit.dynamics import run_implicit_dynamic
                model = run_implicit_dynamic(model, controls, log, out_dir,
                                             run_name, run_num)
            else:
                from ..implicit.statics import run_implicit_static
                model = run_implicit_static(model, controls, log, out_dir,
                                            run_name, run_num)
        else:
            model = _integrate(model, controls, log, out_dir, run_name,
                               run_num, saved, spmd=spmd, gmodel=gmodel)
    return model


def _integrate(model: Model, controls: EngineControls, log: MessageLog,
               out_dir: str, run_name: str, run_num: int = 1,
               saved: Optional[dict] = None,
               spmd: Optional[SpmdContext] = None,
               gmodel: Optional[Model] = None) -> Model:
    state = EngineState()
    n = model.numnod
    # SPMD context (pyradioss/spmd/exchange.py): a no-op in a serial run —
    # every decomposition hook below is guarded by ``spmd.active`` so the
    # serial path is untouched. W is the domain's WEIGHT array (None serial).
    if spmd is None:
        spmd = serial_context()
    if spmd.active:
        state.spmd = spmd
    W = spmd.weight
    is_root = spmd.is_root
    resumed = saved is not None
    # reference physical masses (pre-/DT/NODA/CST) — older restarts and
    # hand-built models may not carry them
    if len(getattr(model, "mass0", ())) != n:
        model.mass0 = model.mass.copy()

    # ---- handle /DEL element deletion (M146) -----------------------------
    if controls.del_elements:
        for kind, eids in controls.del_elements.items():
            if not eids:
                continue
            eid_set = set(eids)
            k_upper = kind.upper()
            for gname, group in model.element_groups():
                match = (k_upper in ("ELEM", "ALL") or
                         (k_upper in ("BRICK", "BRIC", "SOLID") and any(s in gname for s in solid_group_fragments())) or
                         (k_upper in ("PENTA", "PENTA6", "WEDGE") and "penta" in gname) or
                         (k_upper in ("SHELL", "SHEL") and ("shel" in gname or "sh3n" in gname)) or
                         (k_upper in ("SH3N", "TRI", "TRIA") and "sh3n" in gname) or
                         (k_upper in ("TETRA", "TETRA10") and "tetra" in gname) or
                         (k_upper in ("QUAD",) and "quad" in gname) or
                         (k_upper in ("SPRING",) and "spring" in gname) or
                         (k_upper in ("TRUSS",) and "truss" in gname) or
                         (k_upper in ("BEAM",) and "beam" in gname))
                if match and hasattr(group, "ids") and "off" in group.state:
                    mask = np.isin(group.ids, list(eid_set))
                    if mask.any():
                        group.state["off"][mask] = 0.0
                        log.info(f" -- /DEL/{kind}: DELETED {int(mask.sum())} ELEMENT(S) IN {gname}")
        state.ndel = _deleted_count(model)

    # ---- engine-side setup (resol_init) ----------------------------------
    loads = LoadsAndConstraints(model, log, controls=controls)
    if spmd.active:
        loads.set_spmd_weight(W)     # gravit.F / fixvel.F WEIGHT
    walls = RigidWalls(model, log)
    # contact: penalty interfaces (TYPE7/TYPE11, force-based) and tied
    # interfaces (TYPE2, kinematic) hook into the cycle differently
    contacts, tied = build_contacts(model, log)
    if controls.inter_active:
        contacts = [ct for ct in contacts if controls.inter_active.get(ct.itf.id, True)]
        tied = [t for t in tied if controls.inter_active.get(t.itf.id, True)]
    # rigid bodies (/RBODY + /RBE2) and interpolation constraints (/RBE3):
    # both kinematic — see engine/rigid_body.py and engine/rbe3.py. The
    # rigid bodies also project the initial nodal velocities onto rigid
    # motion, so they run BEFORE the E0 reference below.
    rbodies = build_rigid_bodies(
        model, loads, log,
        saved_map=saved.get("rbodies") if resumed else None)
    if controls.rbody_active:
        rbodies = [rb for rb in rbodies if controls.rbody_active.get(rb.rb.id, True)]
    model.rigid_bodies = {rb.rb.id: rb for rb in rbodies}
    rbe3s = build_rbe3(model, log)
    mpc = build_mpc(model, loads, log)     # /MPC (M6)
    lagmul = LagmulSolver(model, loads, log)
    cyl_joints = build_cyl_joints(model, log)  # /CYL_JOINT (M586)
    rlinks = build_rlinks(model, log)          # /RLINK (M586)
    gjoints = build_gjoints(model, log)        # /GJOINT (M594)
    kjoints = build_kjoints(model, log)        # /PROP/TYPE33, /PROP/TYPE45 (M602)
    if spmd.active:
        # /SECT and /TH are evaluated by domain 0 on the gathered GLOBAL
        # view (spmd_collect.F); the other domains carry no output object
        sections = SectionForces(gmodel, log) if is_root else None
    else:
        sections = SectionForces(model, log)
    # /DT/NODA[/CST] (M6): nodal time step + mass scaling. Nodes whose
    # motion a constraint prescribes carry no stability constraint of
    # their own (see engine/mass_scaling.py).
    dampers = Dampers(model, log)          # /DAMP   (M6)
    dyn_relax = DynamicRelaxation(model, controls, log)  # /DYREL /KEREL
    sensors = Sensors(model, log)          # /SENSOR (M6)
    model.sensors_state = sensors
    for mat in model.materials.values():
        mat.sensors = sensors
    if resumed:                            # latched sensors stay latched
        sensors.fire_time.update(saved.get("sensors", {}))
        sensors.status.update(saved.get("sensors_status", {}))
        for sid in sensors.fire_time:
            sensors.status[sid] = True
    ams = AMSManager(model, controls) if getattr(controls, "dt_ams", False) else None
    noda = NodalTimeStep(model, controls, log) if controls.dt_noda else None
    nloc = shell_nonlocal.configure(model, controls, noda)  # NLOC_DMG /DT/NODA
    if noda is not None and spmd.active:
        # counters once per node (WEIGHT); the initial mass reference is
        # the GLOBAL physical mass (a resume overwrites it with the saved
        # global value below)
        noda.weight = W
        real_n = model.mass < 1e29
        noda.mass0 = float(spmd.sum(float(
            (W[real_n] * model.mass[real_n]).sum())))
    if noda is not None:
        for rb in rbodies:
            slave_mask = rb.nodes != rb.master
            noda.set_prescribed(rb.nodes[slave_mask])
            # ...but transport their member stiffness to the master so the
            # body still claims a nodal dt (rgbodfp.F/dtnoda.F — otherwise a
            # stiff shell welded into the body never constrains dt; the
            # RD-E-1000 rolling bug). No-op where no /RBODY exists.
            noda.add_rigid_body(rb.nodes, rb.master, rb.M, rb.J0, model.x0, rb=rb)
        for t2 in tied:
            noda.set_prescribed(t2.snode[t2.active])
        for r3 in rbe3s:
            noda.set_prescribed(np.array([r3.ref]))
        if resumed and saved.get("noda"):
            sv = saved["noda"]
            noda.mass_added = sv["mass_added"]
            # added inertia (M40 rotational CST): absent in pre-M40
            # snapshots — a fresh counter is the correct resume there
            noda.iner_added = sv.get("iner_added", 0.0)
            noda.e_madd = sv["e_madd"]
            noda.mom_added = sv["mom_added"].copy()
            noda.mass0 = sv["mass0"]
            noda._reported = sv["reported"]
    # each run of a chain writes its own T-file (T01, T02, ... — the
    # Radioss numbering convention)
    if spmd.active:
        # domain 0 writes the T-file from the gathered global view; which
        # data the output gathers must carry is decided once on domain 0
        # and broadcast (every domain must take part in the same gathers)
        if is_root:
            th = TimeHistory(
                os.path.join(out_dir, f"{run_name}T{run_num:02d}.csv"),
                gmodel, log)
            out_flags = (len(sections) > 0,
                         bool(th._part_req or th._other_req),
                         any(req[1] == "RBODY" for req in th._other_req))
        else:
            th = None
            out_flags = None
        sect_on, th_groups, th_rb = spmd.bcast(out_flags)
    else:
        th = TimeHistory(os.path.join(out_dir, f"{run_name}T{run_num:02d}.csv"),
                         model, log)

    fint = np.zeros((n, 3))    # -internal forces (see elements pkg doc)
    mint = np.zeros((n, 3))    # -internal moments (shell rotations)
    fext = np.zeros((n, 3))
    fcont = np.zeros((n, 3))   # contact forces, kept separate: the contact
    # energy must be booked with the leapfrog-consistent MIDSTEP velocity
    # (see step 5b below), which needs the isolated contact force vector
    # /INTER/TYPE2 mass transfer (i2 init): the EFFECTIVE mass — used for
    # accelerations only — of a tied node's main segment corners includes
    # the secondary mass, M_k += w_k m_s. Same for the /RBE3 dependent
    # node's mass on its masters. Physical masses (energies, momentum,
    # listing) stay in model.mass. See contact/inter_type2.py.
    mass_eff = model.mass.copy()
    for t2 in tied:
        t2.augment_mass(mass_eff)
    for r3 in rbe3s:
        r3.augment_mass(mass_eff)
    # a rigid body whose nodes carry tied secondaries must also carry
    # their inertia in its 6-DOF EOM (see rigid_body.finalize_mass)
    for rb in rbodies:
        rb.finalize_mass(mass_eff)
    inv_mass = np.where(mass_eff > 0.0, 1.0 / np.maximum(mass_eff, 1e-30), 0.0)
    has_inertia = model.inertia > 0.0
    inv_inertia = np.where(has_inertia, 1.0 / np.maximum(model.inertia, 1e-30),
                           0.0)
    # /MPC: project the initial velocities onto G v = 0 once — the
    # constraint force then does exactly zero work forever (mpc.py)
    if mpc is not None:
        mpc.enforce(model.v, model.vr, inv_mass, inv_inertia)
    if len(lagmul) > 0:
        lagmul.enforce(model.v, model.vr, inv_mass, inv_inertia)
    for rl in rlinks:
        rl.apply_velocity(model.v, mass_eff, getattr(model, "vr", None), getattr(model, "inertia", None))
    for cj in cyl_joints:
        cj.apply_velocity(model.v, model.x, mass_eff)
    for gj in gjoints:
        gj.apply_velocity(model.v, getattr(model, "vr", None), model.x, mass_eff, getattr(model, "inertia", None))
    for kj in kjoints:
        kj.apply_velocity(model.v, getattr(model, "vr", None), model.x, mass_eff, getattr(model, "inertia", None))

    real = model.mass < 1e29
    if resumed:
        # ---- restore the accumulated engine state (M6 chaining) ----------
        # the counters the ledgers depend on, the balance reference, and
        # the NEXT time step (already computed by the previous run —
        # taking anything else would fork from the unchained trajectory)
        for key in ("t", "cycle", "wext", "econt", "epeak", "ndel",
                    "e_num", "e_madd", "e_damp"):
            setattr(state, key, saved[key])
        state.dt_prev = saved.get("dt_prev", None)
        state.e0 = saved["e0"]
        _energies.e0 = state.e0
        if saved.get("dyn_relax"):
            dyn_relax.ke_prev = saved["dyn_relax"].get("ke_prev", 0.0)
        dt = saved["dt"]
        next_th = saved["next_th"]
        next_anim = saved["next_anim"]
        anim_no = saved["anim_no"]
        next_state = saved.get("next_state", controls.state_tstart) if controls.state_dt > 0 else EP30
        log.info(f" RESUMED TIME STEP  . . . . . . . . . : {dt:12.5E}\n")
    else:
        # initial energy = reference E0 of the balance: kinetic + any
        # initial internal energy (an /EOS with E0/P0 starts charged, M6)
        if spmd.active:
            # the GLOBAL reference: weighted nodal KE + this domain's
            # element energies, summed over the domains (ecrit.F)
            ke0 = float(0.5 * ((W[real] * model.mass[real])[:, None]
                               * model.v[real] ** 2).sum())
            if getattr(model, "inertia", None) is not None and getattr(model, "vr", None) is not None:
                real_rot = model.inertia < 1e29
                ke0 += float(0.5 * ((W[real_rot] * model.inertia[real_rot])[:, None]
                                    * model.vr[real_rot] ** 2).sum())
            state.e0 = float(spmd.sum(ke0 + _element_energy_sum(model)))
        else:
            ke0 = float(0.5 * (model.mass[real, None] * model.v[real] ** 2).sum())
            if getattr(model, "inertia", None) is not None and getattr(model, "vr", None) is not None:
                real_rot = model.inertia < 1e29
                ke0 += float(0.5 * (model.inertia[real_rot, None] * model.vr[real_rot] ** 2).sum())
            state.e0 = ke0 + _element_energy_sum(model)
        _energies.e0 = state.e0

        # ---- priming pass: dt=0 'cycle' just to collect the initial
        # critical time step from every kernel (no state advances at 0).
        dt = 0.0
        dt_next = EP30
        claims = []                # per-group dt_e arrays (for /DT/NODA)
        
        if hasattr(model, "nodal_vol_t"):
            model.nodal_vol_t[:] = 0.0
        for name, group in model.element_groups():
            if hasattr(KERNELS[name], "pre_forces"):
                KERNELS[name].pre_forces(group, model, model.x, 0.0)

        for name, group in model.element_groups():
            dt_e = KERNELS[name].forces(group, model.x, model.v, model.vr,
                                        0.0, fint, mint)
            claims.append(dt_e)
            dt_next = min(dt_next, float(dt_e.min()))
        if ams is not None:
            ams_nodes = ams.tag_nodes(claims)
            diag_added, M_offdiag, max_dmels = ams.build_ams_matrix(claims)
        else:
            ams_nodes = None
            M_offdiag = None
        if noda is not None:
            # nodal time step (and, with CST, the initial mass addition —
            # runs before the kinematics so we can reuse inv_mass). The
            # inertia arrays enable the ROTATIONAL nodal dt sqrt(2 IN/
            # STIFR) and its CST inertia scaling (dtnoda.F 452-516, M40)
            noda.assemble(claims)
            if spmd.active:
                # frontier sum of the nodal stiffness (spmd_exch_a.F STIFN/
                # STIFR slots) before the nodal dt / initial mass addition
                spmd.exch_forces(None, stifn=noda.stifn, stifr=noda.stifr)
            dt_next = noda.apply(mass_eff, inv_mass, model.v, 0.0,
                                 model.inertia, inv_inertia, ams_nodes)
            state.e_madd = noda.e_madd
        # penalty interfaces bound the step from cycle 0 (their stiffness
        # is static in this port), so an impact on the very first cycles
        # is safe (a /SENSOR-gated interface claims nothing until it fires
        # — its NEAR accumulation pulls dt down when it does, M4)
        if len(sensors):
            sensors.update(0.0, log)
        for ct in contacts:
            if sensors.active(ct.itf.sens_id):
                dt_next = min(dt_next, ct.dt_bound)
        if spmd.active:
            # global initial step (spmd_glob_min5.F)
            gm = spmd.glob_min(dt_next, "", 0, False,
                               noda.mass_added if noda is not None else 0.0)
            dt_next = gm.dt
            if noda is not None:
                noda.announce(gm.sums[0], 0.0)
        fint[:] = 0.0
        mint[:] = 0.0
        dt = controls.dt_scale * dt_next
        next_th = 0.0
        next_anim = 0.0
        anim_no = 0
        log.info(f" INITIAL TIME STEP  . . . . . . . . . : {dt:12.5E}\n")
    log.info("  CYCLE       TIME        TIME-STEP    ENERGY-IE    "
             "ENERGY-KE    ENERGY-HE    ENERGY-CE    ENERGY-EN    "
             "EXT-WORK   ERROR%")
    # reference for the numerical-dissipation ledger (step 6c): the state
    # bookings so far (the priming pass books nothing at dt = 0; on a
    # resume this equals the previous run's closing value by definition)
    state.e_booked_prev = _element_energy_sum(model)

    # /STATE/DT (M6): periodic restart snapshots (crash recovery / early
    # chaining points) — each write refreshes RunName_{nn}.rst
    if not resumed:
        next_state = controls.state_tstart if controls.state_dt > 0 else EP30

    def _engine_snapshot():
        """The accumulated-state dict of the restart contract (see
        starter/restart.py)."""
        return {
            "t": state.t, "cycle": state.cycle, "wext": state.wext,
            "econt": state.econt, "epeak": state.epeak, "ndel": state.ndel,
            "e_num": state.e_num, "e_madd": state.e_madd,
            "e_damp": state.e_damp, "e0": state.e0, "dt": dt,
            "dt_prev": state.dt_prev,
            "next_th": next_th, "next_anim": next_anim, "anim_no": anim_no,
            "next_state": next_state,
            "sensors": dict(sensors.fire_time),
            "sensors_status": dict(sensors.status),
            "rbodies": {rb.rb.id: {
                "R": rb.R.copy(), "L": rb.L.copy(),
                "v_ref": rb.v_ref.copy(), "w": rb.w.copy(),
                "x_ref": rb.x_ref.copy(), "xg": rb.xg.copy()}
                for rb in rbodies},
            "noda": None if noda is None else {
                "mass_added": noda.mass_added, "e_madd": noda.e_madd,
                "iner_added": noda.iner_added,
                "mom_added": noda.mom_added.copy(), "mass0": noda.mass0,
                "reported": noda._reported},
            "dyn_relax": {"ke_prev": dyn_relax.ke_prev},
        }

    rst_path = os.path.join(out_dir, f"{run_name}_{run_num:04d}.rst")
    if spmd.active:
        # wrrestp.F: domain ISPMD writes RunName_{run}_{ISPMD+1:04d}.rst;
        # domain 0 also writes the gathered GLOBAL RunName_{run}.rst
        rst_glob = rst_path
        rst_path = os.path.join(
            out_dir, f"{run_name}_{run_num:04d}_{spmd.rank + 1:04d}.rst")

    def _write_restarts():
        """The engine restart(s): serial — one file; SPMD — this domain's
        local restart + (collective) the rank-0 global one."""
        snap = _engine_snapshot()
        write_restart(model, rst_path, engine=snap)
        if not spmd.active:
            return
        gsnap = _global_snapshot(snap)
        spmd.gather_model_view(model, gmodel)
        if is_root:
            write_restart(gmodel, rst_glob, engine=gsnap)

    def _global_snapshot(snap):
        """The accumulated state of the GLOBAL restart: partial ledgers
        and /DT/NODA counters summed over the domains (rank order), rigid
        bodies merged (identical replicas — lowest domain wins). ONE
        all-gather."""
        vec = np.array([state.wext, state.econt, state.e_num, state.e_madd,
                        state.e_damp, float(_deleted_count(model))]
                       + ([noda.mass_added, noda.iner_added, noda.e_madd]
                          + list(noda.mom_added) if noda is not None
                          else [0.0] * 6), dtype=np.float64)
        parts = spmd.allgather((vec, snap["rbodies"]))
        tot = np.zeros_like(vec)
        rbm = {}
        for pv, prb in parts:
            tot += pv
            for k, val in prb.items():
                rbm.setdefault(k, val)
        g = dict(snap)
        (g["wext"], g["econt"], g["e_num"], g["e_madd"],
         g["e_damp"]) = (float(x) for x in tot[:5])
        g["ndel"] = int(round(tot[5]))
        g["rbodies"] = rbm
        if noda is not None and snap.get("noda") is not None:
            gn = dict(snap["noda"])
            gn["mass_added"], gn["iner_added"], gn["e_madd"] = \
                (float(x) for x in tot[6:9])
            gn["mom_added"] = tot[9:12].copy()
            g["noda"] = gn
        return g
    # dt-collapse guard reference: a run whose step implodes by many
    # orders of magnitude is dead whatever the ledgers say — without
    # this, a deck with dT_min = 0 can spin forever at dt ~ 1e-18
    dt_ref = dt
    t_wall0 = time.time()

    def _th_write_spmd(t_out):
        """SPMD time-history record (collective): global energies by
        reduction, then the gathered global view on domain 0 (nodal
        arrays, and element states / rigid bodies / moments only when the
        T-file asks for them) feeds the UNCHANGED TimeHistory/SectionForces."""
        e_th = _energies(model, state)
        ex = spmd.gather_model_view(
            model, gmodel, nodal=True, groups=th_groups,
            extra_nodal={"mint": mint} if sect_on else None,
            rigid_bodies=th_rb)
        if is_root:
            real_g = gmodel.mass < 1e29
            mom_g = (gmodel.mass[real_g, None] * gmodel.v[real_g]).sum(axis=0)
            svals_g = sections.compute(gmodel.x, gmodel.fint, ex["mint"]) \
                if sect_on else None
            th.write(t_out, e_th, float(gmodel.mass[real_g].sum()), mom_g,
                     svals_g)

    def _anim_write_spmd(path_a):
        """SPMD animation frame (collective gather, domain 0 writes)."""
        spmd.gather_model_view(model, gmodel)
        if is_root:
            write_anim_state(path_a, gmodel, state.t,
                             controls.anim_vect, controls.anim_elem,
                             cycle=state.cycle)

    # Write t = 0.0 initial state to TimeHistory (BUG-OUT-04)
    if not resumed and controls.th_dt > 0:
        model.fint = fint
        model.fext = fext
        if spmd.active:
            _th_write_spmd(0.0)
        else:
            e0 = _energies(model, state)
            mom0 = (model.mass[real, None] * model.v[real]).sum(axis=0)
            svals0 = sections.compute(model.x, fint, mint) if len(sections) else None
            th.write(0.0, e0, float(model.mass[real].sum()), mom0, svals0)
        while next_th <= state.t:
            next_th += controls.th_dt

    # ======================================================================
    #                       THE EXPLICIT TIME LOOP
    # ======================================================================
    # (the relative epsilon on the termination test avoids a degenerate
    # round-off-sized final cycle when the accumulated time lands within
    # one ulp of T_stop — which is exactly what happens at the junction
    # of a restart chain, M6)
    while state.t < controls.t_end * (1.0 - 1e-14):
        # Planned stops (/STOP/NSTEP, /STOP/TSTOP, /STOP/TIMET)
        if controls.stop_nstep > 0 and state.cycle >= controls.stop_nstep:
            state.stop_reason = f"/STOP/NSTEP REACHED (CYCLE {state.cycle})"
            break
        if controls.stop_tstop > 0 and state.t >= controls.stop_tstop * (1.0 - 1e-14):
            state.stop_reason = f"/STOP/TSTOP REACHED (TIME {state.t:.5E})"
            break
        if controls.stop_timet > 0 and (
                state.timet_glob if spmd.active else
                (time.time() - t_wall0) >= controls.stop_timet):
            # (SPMD: the wall clock differs per domain — the flag was
            # MAX-reduced in the previous cycle's dt packet, MSTOP2 slot)
            state.stop_reason = f"/STOP/TIMET REACHED (ELAPSED {time.time() - t_wall0:.1f}s)"
            break

        dt = min(dt, controls.t_end - state.t)  # land exactly on t_end
        if controls.stop_tstop > 0 and state.t + dt >= controls.stop_tstop:
            if controls.stop_tstop - state.t <= 1e-14 * dt_ref:
                state.stop_reason = f"/STOP/TSTOP REACHED (TIME {state.t:.5E})"
                break
            dt = max(controls.stop_tstop - state.t, 0.0)
        model.t = state.t

        # ---- 0. sensors (M6): poll and latch before anything acts --------
        if len(sensors):
            sensors.update(state.t, log)

        # ---- 0b. moving skews (M39, newskw.F) ----------------------------
        # /SKEW/MOV and /SKEW/MOV2 are rebuilt from the nodes' CURRENT
        # positions once per cycle, BEFORE the forces — exactly where
        # resol.F calls NEWSKW ('MOVING SKEW [MONO THREAD]', resol.F 2653).
        # Everything downstream (the TYPE8 spring frames, the skewed /BCS
        # projection, the skewed imposed motion) reads the rows it writes,
        # so a moving skew turns with its nodes. Free when nothing moves.
        model.skews.update(model.x)

        # ---- 0c. pre-forces pass (SFEM nodal volume scattering)
        if hasattr(model, "nodal_vol_t"):
            model.nodal_vol_t[:] = 0.0
        for name, group in model.element_groups():
            if hasattr(KERNELS[name], "pre_forces"):
                KERNELS[name].pre_forces(group, model, model.x, dt)

        # ---- 1. internal forces, element by element group ----------------
        fint[:] = 0.0
        mint[:] = 0.0
        dt_next = EP30
        claims = []
        try:
            from .element_erosion import check_solid_geometric_erosion
        except ImportError:
            def check_solid_geometric_erosion(group, x, ctrl):
                return np.zeros(len(group.conn) if hasattr(group, "conn") else 0, dtype=bool)

        for name, group in model.element_groups():
            dt_e = KERNELS[name].forces(group, model.x, model.v, model.vr,
                                        dt, fint, mint)
            # ---- M580/M605: /DT/<elem> element time-step & geometric controls (dtchk.F, sgeodel3.F) --
            if getattr(controls, "dt_controls", None):
                off = group.state.get("off") if hasattr(group, "state") else None
                if off is None:
                    off = np.ones(len(dt_e), dtype=np.float64)
                    if hasattr(group, "state"):
                        group.state["off"] = off
                for ctrl_key, ctrl in controls.dt_controls.items():
                    if _matches_elem_dt_control(name, ctrl_key):
                        action = ctrl.get("action", "")
                        if action == "DEL":
                            dt_min = ctrl.get("dt_min", 0.0)
                            del_mask_dt = (dt_e < dt_min) if dt_min > 0.0 else np.zeros(len(dt_e), dtype=bool)
                            del_mask_geom = check_solid_geometric_erosion(group, model.x, ctrl)
                            del_mask = (off > 0.0) & (del_mask_dt | del_mask_geom)
                            if np.any(del_mask):
                                off[del_mask] = 0.0
                                dt_e = dt_e.copy()
                                dt_e[del_mask] = EP30
                                state.ndel = getattr(state, "ndel", 0) + int(np.sum(del_mask))
                        elif action in ("STOP", ""):
                            dt_min = ctrl.get("dt_min", 0.0)
                            if dt_min > 0.0:
                                stop_mask = (off > 0.0) & (dt_e < dt_min)
                                if np.any(stop_mask):
                                    idx = np.where(stop_mask)[0][0]
                                    eid = group.ids[idx] if hasattr(group, "ids") else idx
                                    state.stop_reason = f"/DT/{ctrl_key}/STOP: ELEMENT {eid} TIME STEP BELOW MINIMUM {dt_min:.3E}"
                                    break
                        if ctrl.get("scale", 0.9) != 0.9 and ctrl.get("scale", 0.9) > 0.0:
                            scale_ratio = ctrl["scale"] / 0.9
                            dt_e = dt_e * scale_ratio
            claims.append(dt_e)
            if len(dt_e) > 0:
                dt_next_prev = dt_next
                dt_min_val = float(dt_e.min())
                dt_next = min(dt_next, dt_min_val)
                if dt_min_val < dt_next_prev:
                    min_idx = int(np.argmin(dt_e))
                    state.crit_elem_id = group.ids[min_idx] if hasattr(group, "ids") else min_idx
                    state.crit_elem_type = name
            if state.stop_reason:
                break
        if state.stop_reason and not spmd.active:
            break
        # (SPMD: a local /DT/<elem>/STOP is NOT a local break — every domain
        # must reach the same collectives; the flag rides the dt packet)

        nloc and nloc.cycle_step(dt)   # 1b. NLOCAL_ACC/VEL/INCR (resol.F)
        if hasattr(model, 'sph_cells') and model.sph_cells:  # /SPHCEL (M_SPH)
            from .sph_engine import sph_step
            sph_step(model, dt, state)

        # ---- 2. contact forces -------------------------------------------
        # (into their own array — see the fcont declaration and step 5b;
        # the work increment the interface returns is its own estimate at
        # the pre-update velocities, superseded by the exact booking below)
        # With /DT/NODA the interfaces accumulate their spring stiffness
        # into the nodal array instead of bounding dt with their own
        # scalar (mass scaling must be able to HOLD the step against
        # contact springs — the springs are in stifn, so the nodal
        # formula covers them; see engine/mass_scaling.py).
        fcont[:] = 0.0
        for ct in contacts:
            if not sensors.active(ct.itf.sens_id):
                continue                    # /SENSOR-gated, not fired yet
            _, dt_i = ct.forces(model.x, model.v, model.mass, dt,
                                fcont, state.cycle,
                                stifn=None if noda is None else noda.stifn,
                                t=state.t)
            if noda is None:
                dt_next = min(dt_next, dt_i)

        # ALE advection step (M_ALE)
        if getattr(controls, 'ale_on', False) or (hasattr(model, 'ale_bcs') and model.ale_bcs):
            from .ale_engine import ale_step
            ale_step(model, dt, state)

        # ---- 3. external loads (gravity, /CLOAD, /PLOAD) -------------------
        fext[:] = 0.0
        loads.external_forces(state.t, fext, model.x, sensors)

        for _, mv in model.monitored_volumes.items():
            if mv.vol_type == "AIRBAG1":
                # Matches monvol0.F -> airbaga / volpre
                update_airbag_volume(mv, model, model.x)
                update_airbag_thermodynamics(mv, model, dt, state.t)
                apply_airbag_forces(mv, model, model.x, fext)
            elif mv.vol_type == 'GAS':
                from .airbag import update_monvol_gas
                update_airbag_volume(mv, model, model.x)
                update_monvol_gas(mv, model, dt, state.t)
                apply_airbag_forces(mv, model, model.x, fext)
            elif mv.vol_type == 'PRES':
                from .airbag import update_monvol_pres
                update_airbag_volume(mv, model, model.x)
                update_monvol_pres(mv, model, dt, state.t)
                apply_airbag_forces(mv, model, model.x, fext)

        # Standalone /MONVOL/GAS or /MONVOL/PRES not in monitored_volumes
        if hasattr(model, "monvol_gases") and model.monvol_gases:
            for _, mg in model.monvol_gases.items():
                if mg.id not in model.monitored_volumes:
                    from .airbag import update_monvol_gas
                    update_airbag_volume(mg, model, model.x)
                    update_monvol_gas(mg, model, dt, state.t)
                    apply_airbag_forces(mg, model, model.x, fext)

        if hasattr(model, "monvol_pres") and model.monvol_pres:
            for _, mp in model.monvol_pres.items():
                if mp.id not in model.monitored_volumes:
                    from .airbag import update_monvol_pres
                    update_airbag_volume(mp, model, model.x)
                    update_monvol_pres(mp, model, dt, state.t)
                    apply_airbag_forces(mp, model, model.x, fext)

        # /MONVOL/FVMBAG1 and /MONVOL/FVMBAG2 (M108, M111, M583)
        if hasattr(model, "monvol_fvmbags") and model.monvol_fvmbags:
            for _, fv in model.monvol_fvmbags.items():
                update_fvmbag_volume(fv, model, model.x)
                update_fvmbag_thermodynamics(fv, model, dt, state.t)
                apply_fvmbag_forces(fv, model, model.x, fext)

        if hasattr(model, "monvol_fvmbag2s") and model.monvol_fvmbag2s:
            for _, fv in model.monvol_fvmbag2s.items():
                update_fvmbag_volume(fv, model, model.x)
                update_fvmbag_thermodynamics(fv, model, dt, state.t)
                apply_fvmbag_forces(fv, model, model.x, fext)

        # FSI coupling forces (M_FSI)
        if hasattr(model, 'inter_fsi') and model.inter_fsi:
            from .fsi_coupling import fsi_step
            fsi_step(model, dt, state, fext)

        # ---- 3b. tied interfaces (/INTER/TYPE2, i2for3): move the tied
        # nodes' internal + external forces onto their main segments (the
        # constraint carries them; the secondary rows are zeroed). Also
        # polls the deletion release. Does NO work by construction —
        # nothing is booked into econt. /RBE3 distributes its dependent
        # node's forces to the masters the same way (rbe3f).
        for t2 in tied:
            t2.transfer_forces(fint, fext, fcont, mint, model.x, mass_eff, inv_mass,
                               state.cycle)
        for r3 in rbe3s:
            r3.transfer_forces(fint, fext, fcont, mint, model.x)
        for cj in cyl_joints:
            cj.transfer_forces(fint, fext, fcont, mint, model.x)
        for rl in rlinks:
            rl.transfer_forces(fint, fext, fcont, mint, mass_eff)
        for gj in gjoints:
            gj.transfer_forces(fint, fext, fcont, mint, model.x, mass_eff, inv_mass, inv_inertia)
        for kj in kjoints:
            kj.transfer_forces(fint, fext, fcont, mint, model.x, mass_eff, inv_mass, inv_inertia)

        # ---- 3c. /DT/NODA[/CST]: nodal time step + mass scaling (M6) ------
        # Runs BEFORE the acceleration so the mass added for the target
        # step already stabilizes THIS cycle's update. The nodal dt
        # replaces the worst-element dt as the next-step bound.  The
        # inertia arrays enable the ROTATIONAL nodal dt and its CST
        # inertia scaling (dtnoda.F 452-516, M40 — see mass_scaling.py).
        if ams is not None:
            ams_nodes = ams.tag_nodes(claims)
            diag_added, M_offdiag, max_dmels = ams.build_ams_matrix(claims)
        else:
            ams_nodes = None
            M_offdiag = None

        if noda is not None:
            noda.assemble(claims)

        # ---- 3c'. SPMD frontier exchange (spmd_exch_a.F, resol.F 4740) --
        # Every partial-force producer of this domain has run (elements,
        # the owned contacts/monvols/pressure segments, the weighted
        # gravity/cload, the linear transfers of the replicated tied/RBE3/
        # joints) and the nodal stiffness is assembled: sum the frontier
        # rows over the domains so every holder of a node carries its
        # COMPLETE force and stiffness from here on (acceleration, mass
        # scaling, rigid bodies and walls then run identically on every
        # holder — no velocity exchange is needed).
        if spmd.active:
            spmd.exch_forces(fint, fext, fcont, mint,
                             noda.stifn if noda is not None else None,
                             noda.stifr if noda is not None else None)

        if noda is not None:
            if getattr(controls, "damp_alpha", 0.0) > 0.0 or getattr(controls, "damp_beta", 0.0) > 0.0:
                noda.apply_rayleigh_damping_stiffness(getattr(controls, "damp_alpha", 0.0),
                                                      getattr(controls, "damp_beta", 0.0))
            dt_next = noda.apply(mass_eff, inv_mass, model.v, state.t,
                                 model.inertia, inv_inertia, ams_nodes)
            state.e_madd = noda.e_madd
            if getattr(noda, "cst", False) and getattr(noda, "crit_node_id", 0) > 0:
                state.crit_elem_id = noda.crit_node_id
                state.crit_elem_type = "NODE"
            if getattr(noda, "stopped", False):
                state.stop_reason = f"/DT/NODA/STOP: NODE {noda.stop_node} TIME STEP BELOW MINIMUM {noda.dt_min:.3E}"
                if not spmd.active:
                    break

        # ---- 3c''. SPMD global time step (spmd_glob_min5.F, resol.F 6137)
        # dt min + the winning critical element, the stop flags (MSTOP1 =
        # any local stop, MSTOP2 = the /STOP/TIMET wall clock) and the
        # global /DT/NODA/CST added mass (a SUM slot).
        if spmd.active:
            gm = spmd.glob_min(
                dt_next, state.crit_elem_type, state.crit_elem_id,
                bool(state.stop_reason),
                noda.mass_added if noda is not None else 0.0,
                timet=(controls.stop_timet > 0 and
                       (time.time() - t_wall0) >= controls.stop_timet))
            dt_next = gm.dt
            state.crit_elem_type, state.crit_elem_id = gm.crit_type, gm.crit_id
            state.timet_glob = gm.timet
            if noda is not None:
                noda.announce(gm.sums[0], state.t)
            if gm.stop:
                # every domain stops at this same point; the reason text of
                # the lowest domain that raised one is reported everywhere
                reasons = spmd.allgather(state.stop_reason)
                state.stop_reason = next(r for r in reasons if r)
                break

        dt_prev = state.dt_prev if state.dt_prev is not None else dt
        dt = controls.dt_scale * dt_next
        dt = min(dt, controls.t_end - state.t)
        if controls.stop_tstop > 0 and state.t + dt >= controls.stop_tstop:
            dt = max(controls.stop_tstop - state.t, 0.0)
        dt12 = 0.5 * (dt_prev + dt)
        if dyn_relax.active and getattr(controls, "adyrel_active", False):
            dt = dyn_relax.compute_timestep_reduction(dt, dt12)
            dt12 = 0.5 * (dt_prev + dt)

        # ---- 3d. /MPC Lagrange forces (M6): the tiny coupled solve that
        # makes the ordinary update below satisfy G a = 0 (see mpc.py);
        # runs after mass scaling so it sees the final cycle masses
        if mpc is not None:
            mpc.transfer_forces(fint, fcont, fext, mint,
                                inv_mass, inv_inertia)
        if len(lagmul) > 0:
            lagmul.transfer_forces(fint, fcont, fext, mint,
                                   inv_mass, inv_inertia, dt)

        # ---- 4. acceleration + velocity update (leap-frog) ----------------
        v_old = model.v.copy()     # for wall energy + contact work booking
        vr_old = model.vr.copy()   # for the internal-work ledger (6c)
        f_total = fint + fcont + fext
        if M_offdiag is not None:
            M_diag = mass_eff + diag_added
            acc = f_total * inv_mass[:, None]
            rb_m_list = []
            rb_s_list = []
            for rb in rbodies:
                for s in rb.nodes:
                    if s != rb.master:
                        rb_m_list.append(rb.master)
                        rb_s_list.append(s)
            rb_masters_arr = np.array(rb_m_list, dtype=np.int64) if rb_m_list else None
            rb_slaves_arr = np.array(rb_s_list, dtype=np.int64) if rb_s_list else None
            fix_tra = getattr(loads, "fix_tra", None)
            iters, rel_res = ams.solve(acc, f_total, M_diag, M_offdiag,
                                       fix_tra=fix_tra,
                                       rb_masters=rb_masters_arr,
                                       rb_slaves=rb_slaves_arr)
            state.ams_iters = getattr(state, "ams_iters", 0) + iters
            state.ek_ams = ams.compute_kinetic_energy(model.v, acc, dt12, model.mass, M_diag, M_offdiag)
        else:
            acc = f_total * inv_mass[:, None]
        if noda is not None and getattr(controls, "dt_noda", "") == "SET":
            noda.apply_set_acceleration(acc)
        ar = mint * inv_inertia[:, None] if getattr(model, "vr", None) is not None else None
        for rl in rlinks:
            rl.apply_acceleration(acc, mass_eff, ar, getattr(model, "inertia", None))
        for cj in cyl_joints:
            cj.apply_acceleration(acc, model.x, mass_eff)
        for gj in gjoints:
            gj.apply_acceleration(acc, ar, model.x, mass_eff, getattr(model, "inertia", None))
        for kj in kjoints:
            kj.apply_acceleration(acc, ar, model.x, mass_eff, getattr(model, "inertia", None))
        # /IMPACC imposed acceleration (M595)
        state.wext += loads.apply_acceleration(
            state.t, dt12, acc, ar, mass_eff, getattr(model, "inertia", None),
            model.v, getattr(model, "vr", None), v_old, vr_old, sensors
        )
        model.impacc_reactions = loads.impacc_reactions
        model.fint = fint
        model.fext = fext
        model.a = acc
        model.v += acc * dt12
        if ar is not None:
            model.vr += ar * dt12
        else:
            model.vr += mint * inv_inertia[:, None] * dt12
        state.dt_prev = dt

        # ---- 4b. /DAMP mass damping (M6): exact integrating factor on the
        # freshly updated velocities; dissipation booked exactly from the
        # kinetic-energy identity (see engine/damping.py). Runs BEFORE the
        # kinematic conditions, which override it where they act.
        if len(dampers):
            v_star = model.v.copy()          # pre-damping velocities
            vr_star = model.vr.copy() if getattr(model, "vr", None) is not None else None
            state.e_damp += dampers.apply(state.t, dt, model.v, model.vr,
                                          model.mass, model.inertia,
                                          weight=W)
            # attribution correction on the damped nodes: the force-stage
            # work bookings (EW at the end-of-cycle velocity, the 6c
            # internal-work ledger at (v_old+v_new)/2) assume the update
            # velocities the damper then rescales — the mismatch is an
            # O(alpha dt) energy residual PER TRANSIT that the damper
            # rectifies into a net drift (measured +3.2% frozen on the
            # /MPC settle test before this correction). The exact
            # attribution of the force stage on a damped node is
            # (f_int+f_ext).(v_old+v_star)/2, so the residual booked by
            # the other ledgers is measured here and moved into EN:
            di = dampers.all_idx
            if W is None:
                resid = float(
                    np.einsum("nb,nb->", fext[di],
                              model.v[di] - 0.5 * (v_old[di] + v_star[di]))
                    + np.einsum("nb,nb->", fint[di],
                                0.5 * (model.v[di] - v_star[di]))) * dt
                if getattr(model, "vr", None) is not None and vr_star is not None:
                    resid += float(np.einsum("nb,nb->", mint[di], 0.5 * (model.vr[di] - vr_star[di]))) * dt
            else:
                wd = W[di]
                resid = float(
                    np.einsum("n,nb,nb->", wd, fext[di],
                              model.v[di] - 0.5 * (v_old[di] + v_star[di]))
                    + np.einsum("n,nb,nb->", wd, fint[di],
                                0.5 * (model.v[di] - v_star[di]))) * dt
                if getattr(model, "vr", None) is not None and vr_star is not None:
                    resid += float(np.einsum("n,nb,nb->", wd, mint[di], 0.5 * (model.vr[di] - vr_star[di]))) * dt
            state.e_num += resid

        # ---- 4c. dynamic relaxation (/DYREL /KEREL) -----------------------
        if len(dyn_relax):
            state.e_damp += dyn_relax.apply(state.t, dt12, model.v, model.vr,
                                            model.mass, model.inertia)

        # ---- 5. kinematic conditions overwrite velocities -----------------
        # 5a. rigid bodies (/RBODY, /RBE2 — rbyfor/rbycor): gather the
        # body forces, advance the 6-DOF EOM, scatter the rigid velocity
        # field. Runs FIRST so everything downstream (walls, the contact
        # work booking, the fext work) sees the velocities the body nodes
        # actually move with. Books only the master /IMPVEL drive work.
        for rb in rbodies:
            w_rb = rb.advance(fint, fext, fcont, mint, model.v,
                              model.vr, model.x, dt, state.t + dt)
            # SPMD: the body is replicated on every domain holding one of
            # its nodes; its drive work is booked once, by the master's
            # main domain (WEIGHT of the master, rbyvit.F/rbycor.F)
            state.wext += w_rb if W is None else w_rb * float(W[rb.master])
            w_sq = float(np.sum(rb.w ** 2))
            if dt ** 2 * w_sq > 1.0:
                log.warning(f"** WARNING: RIGID BODY {rb.rb.id} ROTATION ANGLE PER STEP EXCEEDS STABILITY LIMIT (DT^2*OMEGA^2 = {dt**2 * w_sq:.3f} > 1.0)")
        # (mass_eff: an /IMPVEL driving a tied main node reacts against
        # the secondary inertia it carries too. v_old is v^{n-1/2}, the
        # start-of-cycle velocity, so the constraint work is booked at the
        # leapfrog midstep — see LoadsAndConstraints.apply_kinematic; this
        # is what makes an IMPULSIVE imposed-velocity start (RD-V-0700)
        # balance instead of booking twice the work at cycle 1.)
        state.wext += loads.apply_kinematic(state.t + dt, model.v, model.vr,
                                            mass_eff, model.x, dt, v_old,
                                            model.inertia, vr_old,
                                            sensors=sensors)
        de_wall, dw_wall = walls.apply(model.x, model.v, v_old,
                                       model.mass, dt, weight=W)
        state.econt += de_wall
        state.wext += dw_wall

        # ---- 5b. contact energy booking (M4 lesson) ------------------------
        # In leap-frog, a force f^n changes the kinetic energy by exactly
        # f . (v^{n-1/2} + v^{n+1/2})/2 * dt — the MIDSTEP average, not
        # either endpoint. Booking contact work at v^{n-1/2} (as a contact
        # kernel alone could) leaves a positive-definite residual
        # f^2 dt^2 / 2m per cycle that reads as spurious energy CREATION
        # whenever penalty springs dominate the energy scale (light nodes,
        # short impacts). Booking from the assembled contact force with
        # the midstep velocity closes the balance to round-off; the
        # remaining CE drift is the real damper/friction dissipation.
        # (The original accumulates interface energies from the same
        # assembled forces in its FSAV blocks.)
        if W is None:
            if contacts:
                state.econt -= float(np.einsum(
                    "nb,nb->", fcont, 0.5 * (v_old + model.v))) * dt12

            # external work of the loads: force x actual displacement, booked with
            # midstep average velocity (resol.F:6289, force.F90:322)
            state.wext += float(np.einsum("nb,nb->", fext, 0.5 * (v_old + model.v))) * dt12
        else:
            # SPMD: fcont/fext are the frontier-summed COMPLETE forces on
            # every holder — booked once per node (WEIGHT, ecrit.F). The
            # contact booking runs on EVERY domain (a domain that owns no
            # interface still holds contact force on its frontier nodes).
            v_mid = 0.5 * (v_old + model.v)
            state.econt -= float(np.einsum("n,nb,nb->", W, fcont,
                                           v_mid)) * dt12
            state.wext += float(np.einsum("n,nb,nb->", W, fext,
                                          v_mid)) * dt12

        # /MPC and /LAGMUL velocity cleanup: enforce constraints on velocities
        # BEFORE position update to prevent geometric drift (BUG-ENG-05)
        if mpc is not None:
            mpc.enforce(model.v, model.vr, inv_mass, inv_inertia)
        if len(lagmul) > 0:
            lagmul.enforce(model.v, model.vr, inv_mass, inv_inertia)
        for rl in rlinks:
            rl.apply_velocity(model.v, mass_eff, getattr(model, "vr", None), getattr(model, "inertia", None))
        for cj in cyl_joints:
            cj.apply_velocity(model.v, model.x, mass_eff)
        for gj in gjoints:
            gj.apply_velocity(model.v, getattr(model, "vr", None), model.x, mass_eff, getattr(model, "inertia", None))
        for kj in kjoints:
            kj.apply_velocity(model.v, getattr(model, "vr", None), model.x, mass_eff, getattr(model, "inertia", None))

        # ---- 6. position update -------------------------------------------
        model.x += model.v * dt

        # ---- 6b. kinematic placements, dependency order: rigid bodies
        # first (their nodes may carry tied mains / RBE3 masters), then
        # tied interfaces (i2vit3), then RBE3 dependents.
        for rb in rbodies:
            rb.enforce(model.x, model.v, dt)
        for t2 in tied:
            t2.enforce(model.x, model.v, model.vr, dt)
        for r3 in rbe3s:
            r3.enforce(model.x, model.v, model.vr, dt)
        for rl in rlinks:
            rl.enforce(model.x, model.v, dt)
        for cj in cyl_joints:
            cj.enforce(model.x, model.v, dt)
        for gj in gjoints:
            gj.enforce(model.x, model.v, getattr(model, "vr", None), dt)
        for kj in kjoints:
            kj.enforce(model.x, model.v, getattr(model, "vr", None), dt)

        # ---- 6c. numerical-dissipation ledger (M6) --------------------------
        # The kernels book internal energy as a STATE FUNCTION
        # (sigma_mid : deps — exact for the constitutive path), but the
        # internal forces act on the nodes and take from the kinetic
        # energy exactly  -fint . (v_old + v_new)/2 * dt  per cycle (the
        # same leapfrog midstep identity as the M4 contact-work lesson).
        # For smooth motion the two agree to a BOUNDED O(dt^2) wiggle
        # (the elastic work telescopes onto the state function). But when
        # the bulk-viscosity / hourglass dampers act on BARELY-RESOLVED
        # content (omega*dt near 2 — the guide's 2x2x2 ringing
        # reproducer), the discrete elastic force becomes a genuine
        # energy sink the state function cannot see: the M1-era balance
        # drifted secularly (-35% over 2 ms of free flight) with the qb
        # linear damper as the visible culprit. The consistent treatment
        # is to MEASURE the exact internal work each cycle and book the
        # difference against the state bookings into this separate,
        # reported ledger (EN — 'numerical dissipation'):
        #
        #   EN += [- fint.v_avg dt - mint.vr_avg dt] - d(eint + ehour)
        #
        # * healthy runs: EN wiggles around zero and stays negligible
        #   (asserted by the M1-M6 validations);
        # * damped barely-resolved ringing: EN accumulates the real
        #   numerical dissipation and the balance closes honestly;
        # * a DIVERGING run drives EN hard NEGATIVE (the elastic force
        #   injects energy the state function does not book), so the
        #   |ERR| stop still fires — the metric stays alive. A material
        #   law that misbooks its own work now surfaces in EN instead of
        #   ERR; EN is printed in the listing and T01 so it cannot hide.
        e_booked = _element_energy_sum(model)
        if W is None:
            w_leave = -0.5 * dt12 * (
                float(np.einsum("nb,nb->", fint, v_old + model.v))
                + float(np.einsum("nb,nb->", mint, vr_old + model.vr)))
        else:
            # SPMD: complete nodal forces, once per node (WEIGHT); the
            # element bookings e_booked are this domain's own elements
            w_leave = -0.5 * dt12 * (
                float(np.einsum("n,nb,nb->", W, fint, v_old + model.v))
                + float(np.einsum("n,nb,nb->", W, mint, vr_old + model.vr)))
        state.e_num += w_leave - (e_booked - state.e_booked_prev)
        state.e_booked_prev = e_booked

        # ---- ADYREL adaptive dynamic relaxation frequency adaptation (static.F:312) ---
        if dyn_relax.adyrel_active:
            ke_curr = float(0.5 * np.sum(model.mass[real, None] * (model.v[real] ** 2)))
            if getattr(model, "vr", None) is not None and getattr(model, "inertia", None) is not None:
                has_in = (model.inertia[real] > 0.0)
                if np.any(has_in):
                    ir = real[has_in]
                    ke_curr += float(0.5 * np.sum(model.inertia[ir, None] * (model.vr[ir] ** 2)))
            dyn_relax.update_adaptive_frequency(state.t, dt, state.cycle, e_booked, ke_curr)

        state.t += dt
        state.cycle += 1

        # ---- 7. outputs ----------------------------------------------------
        # (SPMD: every output trigger below depends on t/cycle only, which
        # are identical on all domains — the collectives inside line up)
        if state.t >= next_th and controls.th_dt > 0:
            if spmd.active:
                _th_write_spmd(state.t)
            else:
                e = _energies(model, state)
                mom = (model.mass[real, None] * model.v[real]).sum(axis=0)
                # /SECT resultants from this cycle's assembled internal forces
                svals = sections.compute(model.x, fint, mint) if len(sections) \
                    else None
                th.write(state.t, e, float(model.mass[real].sum()), mom, svals)
            while next_th <= state.t:
                next_th += controls.th_dt
        if controls.anim_dt > 0 and state.t >= next_anim:
            path = os.path.join(out_dir, f"{run_name}A{anim_no:03d}.vtk")
            if spmd.active:
                _anim_write_spmd(path)
            else:
                write_anim_state(path, model, state.t,
                                 controls.anim_vect, controls.anim_elem,
                                 cycle=state.cycle)
            anim_no += 1
            while next_anim <= state.t:
                next_anim += controls.anim_dt
        if controls.state_dt > 0 and state.t >= next_state:
            # /STATE/DT snapshot: a full restart, resumable by the next
            # run of the chain (and the crash-recovery point)
            if spmd.active:
                _write_restarts()
            else:
                write_restart(model, rst_path, engine=_engine_snapshot())
            log.info(f" -- /STATE: RESTART SNAPSHOT WRITTEN AT TIME "
                     f"{state.t:12.5E}")
            while next_state <= state.t and controls.state_dt > 0:
                next_state += controls.state_dt
        if state.cycle % controls.print_cycles == 0 or \
                state.t >= controls.t_end:
            # element-deletion report (the original prints a "RUPTURE /
            # DELETE ELEMENT" message per element; the port reports the
            # running total at the listing frequency)
            if spmd.active:
                # the deletion counts and the added mass ride along in the
                # (single) energy reduction; state.ndel stays this domain's
                ndel = _deleted_count(model)
                e = _energies(model, state, extra=[
                    float(ndel), float(state.ndel),
                    noda.mass_added if noda is not None else 0.0])
                ndel_g, ndel_prev_g, madd_g = (float(x) for x in e["EXTRA"])
                if ndel_g > ndel_prev_g:
                    log.info(f" -- ELEMENT DELETION: "
                             f"{int(round(ndel_g - ndel_prev_g))} "
                             f"ELEMENT(S) DELETED (TOTAL {int(round(ndel_g))})")
                state.ndel = ndel
            else:
                ndel = _deleted_count(model)
                if ndel > state.ndel:
                    log.info(f" -- ELEMENT DELETION: {ndel - state.ndel} "
                             f"ELEMENT(S) DELETED (TOTAL {ndel})")
                    state.ndel = ndel
                e = _energies(model, state)
            log.info(f" {state.cycle:6d} {state.t:12.5E} {dt:12.5E} "
                     f"{e['IE']:12.5E} {e['KE']:12.5E} {e['HE']:12.5E} "
                     f"{e['CE']:12.5E} {e['EN']:12.5E} {e['EW']:12.5E} "
                     f"{e['ERR']:8.2f}")
            # ---- divergence guards (the original's energy-error and
            # minimum-time-step stops) --------------------------------
            # M41 startup conditioning: the two PERCENTAGE guards below
            # are inert until the reference energy clears the numerical-
            # dust floor (mirrors ecrit.F's ABS(ENTOT1B) > EM20 gate on
            # the ENERGY ERROR; see _ENERGY_START_FLOOR).  A slowly-
            # loaded deck's opening cycles carry only round-off-scale
            # energy, so a fixed leapfrog half-step lag reads as a huge
            # %-error on a dust denominator with no instability present
            # (RD-E-1000 Sf_0.1: ERR -17% at cycle 100, IE ~3.75e-8).
            # The NAN/INF check stays UNCONDITIONAL — a non-finite state
            # is always fatal.  The 15% / 30% limits are unchanged.
            if e["REF"] > _ENERGY_START_FLOOR:
                state.epeak = max(state.epeak, e["IE"] + e["KE"])
                if abs(e["ERR"]) > controls.energy_error_stop:
                    state.stop_reason = (
                        f"ENERGY ERROR {e['ERR']:.1f}% EXCEEDS "
                        f"LIMIT {controls.energy_error_stop}%")
                    break
                # a strongly NEGATIVE numerical-dissipation ledger is
                # energy INJECTION — the signature of instability that
                # the closed balance (which includes EN) cannot show in
                # ERR (M6). The threshold is 2x the ERR limit: EN
                # legitimately WOBBLES by a large fraction of the energy
                # scale when most of a small model's energy sits in a
                # barely-resolved mode (a single spring at omega*dt ~ 1.3
                # reads -35% with no instability at all), while true
                # injection runs away to -100% and beyond.
                if e["ERRN"] < -2.0 * controls.energy_error_stop:
                    state.stop_reason = (
                        f"NUMERICAL ENERGY INJECTION {e['ERRN']:.1f}% EXCEEDS "
                        f"LIMIT {2.0 * controls.energy_error_stop}% — "
                        f"RUN UNSTABLE")
                    break
            if controls.mass_error_stop > 0.0 and noda is not None:
                madd = madd_g if spmd.active else noda.mass_added
                frac_m = 100.0 * madd / max(noda.mass0, 1e-20)
                if frac_m > controls.mass_error_stop:
                    state.stop_reason = f"/STOP: MASS ERROR {frac_m:.2f}% EXCEEDED LIMIT {controls.mass_error_stop:.2f}%"
                    break
            if controls.nodal_mass_error_stop > 0.0 and noda is not None:
                if spmd.active:
                    top_nodes = _noda_top_nodes(noda, spmd, 1)
                else:
                    top_nodes = noda.get_top_mass_nodes(1)
                if top_nodes and top_nodes[0][2] * 100.0 > controls.nodal_mass_error_stop:
                    state.stop_reason = f"/STOP: NODAL MASS ERROR {top_nodes[0][2]*100.0:.2f}% ON NODE {top_nodes[0][0]} EXCEEDED LIMIT {controls.nodal_mass_error_stop:.2f}%"
                    break
            if dyn_relax.active and dyn_relax.check_convergence(e["KE"], e["IE"], max(state.epeak, e["REF"])):
                state.stop_reason = f"/STOP/STATIC CONVERGENCE REACHED AT TIME {state.t:.5E}"
                break
            if not all(np.isfinite(v) for v in (e["KE"], e["IE"], e["HE"], e["CE"], e["EW"], e["EN"], e["DE"], e["REF"])):
                state.stop_reason = "NAN/INF DETECTED — RUN DIVERGED"
                break

        # ---- 8. next time step ---------------------------------------------
        dt = controls.dt_scale * dt_next
        dt_ref = max(dt_ref, dt)
        # with /DT/NODA/CST the mass scaling guarantees dt >= dT_min by
        # construction (up to round-off) — the minimum-step stop is the
        # very thing the option replaces
        if controls.dt_min > 0 and dt < controls.dt_min and \
                controls.dt_noda != "CST":
            state.stop_reason = (f"TIME STEP {dt:.3E} BELOW MINIMUM "
                                 f"{controls.dt_min:.3E}")
            break
        if dt < 1e-9 * dt_ref:
            state.stop_reason = (f"TIME STEP COLLAPSED ({dt:.3E}, WAS "
                                 f"{dt_ref:.3E}) — RUN DEAD")
            break
        if not np.isfinite(dt) or dt <= 0.0:
            state.stop_reason = "NAN/INF DETECTED — RUN DIVERGED"
            break

    # ======================================================================
    # final state + termination summary (like the original's final page)
    if controls.anim_dt > 0:
        path = os.path.join(out_dir, f"{run_name}A{anim_no:03d}.vtk")
        if spmd.active:
            _anim_write_spmd(path)
        else:
            write_anim_state(path, model, state.t,
                             controls.anim_vect, controls.anim_elem,
                             cycle=state.cycle)
    if th is not None:
        th.close()
    # the ENGINE restart (M6 chaining contract): RunName_{nn+1}.rad
    # resumes from this file — written on ERROR stops too, so a diverged
    # run can be re-tried from its last state with different controls
    if spmd.active:
        _write_restarts()
        log.info(f" RESTART FILE WRITTEN . . . . . . . . : {rst_glob}")
    else:
        write_restart(model, rst_path, engine=_engine_snapshot())
        log.info(f" RESTART FILE WRITTEN . . . . . . . . : {rst_path}")
    # expose the exact accumulated state to callers/tests (not pickled —
    # attached after the restart write)
    model.engine_state = state

    if not state.stop_reason and not np.isfinite(state.t):
        state.stop_reason = "NAN/INF DETECTED — RUN DIVERGED"

    e = _energies(model, state)
    log.info("\n     ------------------------------------------------")
    if state.stop_reason:
        if state.stop_reason.startswith("/STOP/"):
            log.info(f"     ENGINE TERMINATION : NORMAL — {state.stop_reason}")
        else:
            log.info(f"     ENGINE TERMINATION : ERROR — {state.stop_reason}")
    else:
        log.info("     ENGINE TERMINATION : NORMAL")
    log.info(f"     FINAL TIME    . . . . . . : {state.t:14.7E}")
    log.info(f"     CYCLES        . . . . . . : {state.cycle}")
    log.info(f"     ENERGY ERROR  . . . . . . : {e['ERR']:8.2f} %")
    log.info(f"     INTERNAL ENERGY . . . . . : {e['IE']:14.7E}")
    log.info(f"     KINETIC ENERGY  . . . . . : {e['KE']:14.7E}")
    log.info(f"     HOURGLASS ENERGY  . . . . : {e['HE']:14.7E}")
    log.info(f"     CONTACT ENERGY  . . . . . : {e['CE']:14.7E}")
    log.info(f"     NUMERICAL DISSIPATION . . : {e['EN']:14.7E}")
    log.info(f"     DAMPING DISSIPATION . . . : {e['DE']:14.7E}")
    log.info(f"     EXTERNAL WORK . . . . . . : {e['EW']:14.7E}")
    if noda is not None:
        if spmd.active:
            _noda_global_view(noda, spmd).summary(log)
        else:
            noda.summary(log)
    log.info(f"     ELAPSED TIME  . . . . . . : "
             f"{time.time() - t_wall0:10.3f} s")
    log.info("     ------------------------------------------------")
    if spmd.active:
        # detach the context (no collective may run after the loop), and
        # hand domain 0's caller the GLOBAL view with the reduced ledgers
        red = spmd.reduce_state(state)
        state.spmd = None
        if is_root:
            gmodel.engine_state = red
            gmodel.spmd_local_model = model
            return gmodel
    return model


def _noda_top_nodes(noda, spmd, n: int = 5):
    """Global top-``n`` nodes by added mass (``sortie_error.F``) of a
    decomposed run: each domain proposes its own OWNED nodes (WEIGHT == 1,
    so a frontier node is counted once), one all-gather, merged by added
    mass (identical on every domain)."""
    w = spmd.weight
    nam = getattr(noda, "node_added_mass", None)
    cand = []
    if nam is not None and len(nam):
        act = np.where((nam > 0.0) & (w > 0.0))[0]
        order = act[np.argsort(-nam[act])][:n]
        m0 = getattr(noda, "initial_nodal_mass", None)
        for idx in order:
            nid = int(noda.model.node_ids[idx])
            dm = float(nam[idx])
            mi = float(m0[idx]) if m0 is not None and len(m0) > idx else 0.0
            rel = dm / max(mi, 1e-20) if mi > 0.0 else dm / max(noda.mass0, 1e-20)
            cand.append((nid, dm, rel))
    merged = [c for part in spmd.allgather(cand) for c in part]
    merged.sort(key=lambda c: (-c[1], c[0]))
    return merged[:n]


def _noda_global_view(noda, spmd):
    """A shallow copy of the /DT/NODA machinery carrying the GLOBAL
    counters (summed over the domains) and the global top-mass list, for
    the termination-page ``summary`` of a decomposed run (collective)."""
    import copy as _copy
    vec = np.array([noda.mass_added, noda.iner_added, noda.e_madd]
                   + list(noda.mom_added), dtype=np.float64)
    g = spmd.sum(vec)
    top = _noda_top_nodes(noda, spmd, 5)
    view = _copy.copy(noda)
    view.mass_added, view.iner_added, view.e_madd = (float(x) for x in g[:3])
    view.mom_added = np.array(g[3:6], dtype=np.float64)
    view.get_top_mass_nodes = lambda n=5: top[:n]
    return view
