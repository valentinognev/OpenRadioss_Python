# -*- coding: utf-8 -*-
"""
Material and equation-of-state readers - /KEYWORD blocks -> Model.

/MAT/** and /EOS/** - one reader per constitutive law, each calling
its ``hm_read_mat*.F`` counterpart.

Fortran origin: the ``hm_read_*.F`` routines under
``starter/source/**/reader/``, one routine per keyword, all driven from
``starter/source/starter/lectur.F``. Each reader's docstring names the
Fortran routine it mirrors; ``keywords/dispatch.py`` holds the keyword
-> parser table, and ``input/starter_keywords.py`` is a deprecated shim
over this package.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union
import os

import numpy as np

from ...common.messages import MessageLog
from ...common.tables import FunctTable
from ...model.entities import (
    AddedMass, BoundaryCondition, Box, ConcentratedLoad, Damping, Gravity,
    CentrifugalLoad, ImposedAcceleration, ImposedDisplacement, ImposedTemperature, ImposedVelocity, InitialVelocity, Interface, Line,
    Material, Mpc, NodeGroup, Part, PressureLoad, Property, Random, Rbe3, RigidBody,
    RigidWall, Section, MonitoredVolume, Sensor, Subdomain, Submodel, Surface, Table, THRequest, Xref,
    DetonatorPoint, DetonatorPlane,
    ConvectionLoad, InivolContainer, InitialVolume,
    RadiationLoad, ImposedFlux, InitialTemperature,
    InitialBrickState, InitialShellState,
    InitialTrussState, InitialBeamState, InitialSpringState,
    BcsNrf, BcsWall, RigidLink, CylJoint, GJoint, GeneralJoint,
    MergeNode, MergeRbody, IniCrack, IniCrackSegment, LaserLoad,
    PcylLoad, PfluidLoad, Preload, PreloadAxial, DampInter, DampRange,
    AnalyGlobal, UpwindGlobal, CaaControl,
    Gauge, Cluster, ExtLink, FxBody, IniGrav, IniMap1D, IniMap2D, IniStateFile,
    MonvolPres, MonvolGas, MonvolCommu1, MonvolLFluid, LeakMat,
    MonvolAirbag, MonvolAirbagJet, MonvolAirbagVent, MonvolCommu, MonvolPart,
    DampGlobal, DampPart, TransformProjection, TransformFrame,
    LoadGravity, LoadBody, LoadTherm, EulerBcs, HeatBcs,
    RwallBox, RwallCone, SectBox, SectCut,
    InivelPart, InivelSph, DetLine, DetCirc, IniMap3D, SetGeneric,
    AleGrid, AleLink, AleSolver, AleClose,
    Retractor, Slipring, UserWindow,
    Drape, IniBriEref, IncludeDyna, MonvolFvmBag1, MonvolFvmbag,
    FvmInjector, FvmVent, FvmOrifice, FvmPorousSurface, FvmChamber,
    GaugePoint, SphGlo, AnalyOptions, AleCfdSph,
    FailOrthBiquad, SlipringShell,
    Upbeam, RelaxSystem, MonvolComm,
)
from ...model.model import Model
from ...model.skew import SkewFrame
from .. import mat_reader
from ..card_layouts import LAYOUTS
from ..card_layouts import CARD_LAYOUTS
from ..card_layouts import split_fixed
from ..deck_reader import Card, KeywordBlock, _to_float, parse_fortran_float


# ----------------------------------------------------------------------------
# Small helpers shared by the parsers
# ----------------------------------------------------------------------------





def read_mat(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW<n>/mat_ID`` (aliases /MAT/ELAST, /MAT/PLAS_JOHNS,
    /MAT/PLAS_TAB, /MAT/PLAS_BRIT, /MAT/OGDEN).

    LAW1 (linear elastic) — Fortran starter/source/materials/mat/mat001::

        card 1:  mat_title
        card 2:  rho_0
        card 3:  E   nu

    LAW2 (Johnson–Cook) — Fortran .../mat002 (Iflag=0 classic input)::

        card 1:  mat_title
        card 2:  rho_0
        card 3:  E   nu
        card 4:  A   B   n   eps_p_max   sig_max
        card 5:  c   eps_dot_0   [ICC  Fsmooth  F_cut  — ignored]
        card 6:  m   T_melt   rho_Cp   [T_i]          (optional — M6)

      yield stress
      sigma_y = (A + B*eps_p^n) (1 + c*ln(eps_dot/eps_dot_0)) (1 - T*^m)
      capped at sig_max; the element is DELETED when the plastic strain
      reaches eps_p_max (since M3). Card 5 is optional (no rate effect if
      absent). Card 6 (M6) turns on the ADIABATIC thermal terms: the
      plastic work heats the material, dT = sigma_y d(eps_p) / rho_Cp
      (rho_Cp = specific heat per unit volume), and the homologous
      temperature T* = (T - T_i)/(T_melt - T_i) softens the yield stress
      (and feeds /FAIL/JOHNSON's D5 term). T_i defaults to 298 K; there
      is no heat conduction (adiabatic — the crash/impact regime).

    LAW27 (brittle, shells only) — Fortran .../mat027::

        card 1:  mat_title
        card 2:  rho_0
        card 3:  E   nu
        card 4:  A   B   n   eps_p_max   sig_max      (plasticity card 1)
        card 5:  c   eps_dot_0   [STRFLAG...]         (plasticity card 2)
        card 6:  eps_t1   eps_m1   dmax1   eps_f1     (crack direction 1)
        card 7:  eps_t2   eps_m2   dmax2   eps_f2     (optional, = card 6)

      tensile cracking: damage starts at strain eps_t, reaches dmax at
      eps_m, layer breaks at eps_f (see law27_brittle.py). The plastic
      block implements Johnson-Cook hardening with iterative exact
      plane-stress return (or radial projection).

    LAW36 (tabulated plasticity) — Fortran .../mat036. TWO dialects are
    accepted (dispatched on the card count — the real layout always has
    at least 6 data cards, the compact one at most 5):

      * the port's compact layout::

            card 1:  mat_title
            card 2:  rho_0
            card 3:  E   nu
            card 4:  N_funct   [eps_p_max]
            card 5:  fct_ID1 ... fct_ID_N     (hardening curves eps_p->sig_y)
            card 6:  rate_1 ... rate_N        (required when N_funct > 1,
                     strictly increasing strain rates, one per curve)

      * the REAL fixed-format layout (cfg ``matl36_plas_tab.cfg``
        radioss2017+ / ``hm_read_mat36.F``), as written by real decks::

            card 1:  mat_title
            card 2:  rho_0                                        (%20lg)
            card 3:  E   Nu   Eps_p_max   Eps_t   Eps_m           (5 %20lg)
            card 4:  N_funct  F_smooth  C_hard  F_cut  Eps_f  VP
                     (%10d %10d %20lg %20lg %20lg 10x %10d)
            card 5:  fct_IDp  Fscale  fct_IDE  EInf  CE
                     (%10d %20lg %10d %20lg %20lg)
            card 6+: fct_ID1...  (%10d, 5 per card), then
                     Fscale_1... (%20lg, 5 per card, 0 -> 1.0), then
                     Eps_dot_1... (%20lg, 5 per card)

        Fields the port does not implement (F_smooth/C_hard/F_cut/Eps_f/
        VP/Eps_t/Eps_m, the fct_IDp pressure function, the fct_IDE
        modulus evolution) are accepted and reported in ONE warning,
        mirroring the 'accepted, ignored' contract of the original
        Starter listing.  The per-curve Fscale_i IS applied (M40): it
        scales the hardening curve (value and slope, sigeps36.F YFAC)
        when the /FUNCT arrays are resolved.

      the yield stress follows the /FUNCT curves, linearly interpolated
      in strain rate; the element is deleted at eps_p_max (0 = no limit).

    LAW42 (Ogden hyperelastic, solids only) — Fortran .../mat042::

        card 1:  mat_title
        card 2:  rho_0
        card 3:  mu_1  mu_2  mu_3  mu_4  mu_5
        card 4:  alpha_1 ... alpha_5
        card 5:  nu                          (default 0.495, near-incompr.)

      W = sum mu_p/alpha_p (lb1^a + lb2^a + lb3^a - 3) + K/2 (J-1)^2;
      every used pair must satisfy mu_p*alpha_p > 0; the ground-state
      shear modulus is G0 = sum(mu_p*alpha_p)/2 and the derived E, K
      follow from nu (stored in params so the generic elastic machinery
      — time step, contact stiffness — works unchanged).
    """
    if block.key0 not in ("MAT",):
        lawname = block.key0.upper()
    elif len(block.parts) > 1:
        lawname = block.parts[1].upper()
    else:
        lawname = ""
    if lawname.startswith("MAT_"):
        lawname = lawname[4:]
    if lawname in ("PLAS_ZERIL", "PLAS_ZERI", "ZERIL", "ZERILLI"):
        read_mat_plas_zeril(block, model, log)
        return
    if lawname in ("PLAS_BODNE", "PLAS_BODN", "BODNE", "BODNER"):
        read_mat_plas_bodne(block, model, log)
        return
    if lawname in ("VISC_PLAS", "PLAS_VISC") or (lawname == "VISC" and len(block.parts) > 2 and block.parts[2].upper() in ("PLAS", "VISC_PLAS")):
        read_visc_plas(block, model, log)
        return
    if lawname == "MNF":
        read_mat_law100(block, model, log)
        return
    if lawname in ("VISC_HYP",) or (lawname == "VISC" and len(block.parts) > 2 and block.parts[2].upper().startswith("HYP")):
        is_law62 = False
        _, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        val_cards = [c for c in cards if not c.is_blank and not str(c.raw if hasattr(c, "raw") else c).strip().startswith("#")]
        if len(val_cards) > 1:
            tok = val_cards[1].tokens()[0] if not block.fixed else (val_cards[1].raw[:20].strip() if hasattr(val_cards[1], "raw") else str(val_cards[1])[:20].strip())
            try:
                fval = float(tok)
                if 0.0 < fval < 1.0:
                    is_law62 = True
            except (ValueError, TypeError):
                pass
        if is_law62:
            read_mat_law62(block, model, log)
        else:
            read_mat_law100(block, model, log)
        return
    if lawname in ("VISC_LPRONY", "LPRONY", "VISCO_LPRONY"):
        read_mat_visc_lprony(block, model, log)
        return
    if lawname in ("VISC_PRONY", "PRONY", "VISC"):
        read_mat_visc_prony(block, model, log)
        return
    if lawname in ("THERM_STRESS", "THERM", "THERMAL"):
        read_mat_therm_stress(block, model, log)
        return
    if lawname in ("LAW114", "SPR_SEATBELT", "SEATBELT_SPR", "SEATBELT_SPRING", "LAW114_SPR_SEATBELT"):
        read_mat_law114(block, model, log)
        return
    if lawname in ("LAW119", "SH_SEATBELT", "SEATBELT_SH", "SEATBELT_SHELL", "LAW119_SH_SEATBELT"):
        read_mat_law119(block, model, log)
        return
    if lawname in ("LAW120", "TAPO", "TAB_PONT_ORTH", "LAW120_TAPO"):
        read_mat_law120(block, model, log)
        return
    if lawname in ("LAW121", "PLAS_RATE", "PLAS_TAB_RATE", "LAW121_PLAS_RATE"):
        read_mat_law121(block, model, log)
        return
    if lawname in ("LAW124", "CDPM2", "LAW124_CDPM2"):
        read_mat_law124(block, model, log)
        return
    if lawname in ("LAW24", "CONC", "LAW24_CONC"):
        read_mat_conc(block, model, log)
        return
    if lawname in ("LAW83", "SPR_JOU", "LAW83_SPR_JOU"):
        read_mat_law83(block, model, log)
        return
    if lawname in ("LAW80", "TRANSFO", "LAW80_TRANSFO"):
        read_mat_law80(block, model, log)
        return
    if lawname in ("LAW117", "COH_MC", "COHESIVE", "LAW117_COH_MC"):
        read_mat_law117(block, model, log)
        return
    if lawname in ("LAW90", "TAB_FOAM", "TABULAR_FOAM", "LAW90_TAB_FOAM", "HYST_FOAM", "LAW90_HYST_FOAM"):
        read_mat_law90(block, model, log)
        return
    if lawname in ("LAW33", "FOAM_PLAS", "LAW33_FOAM_PLAS"):
        read_mat_law33(block, model, log)
        return
    if lawname in ("LAW66", "66", "FOAM_TAB", "LAW66_FOAM_TAB", "PLAS_TAB_COSSER", "PLAS_COSSER", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "LAW66_PLAS_TAB_COSSER", "LAW66_PLAS_COSSER"):
        read_mat_law66(block, model, log)
        return
    if lawname in ("LAW35", "FOAM_VISC", "LAW35_FOAM_VISC"):
        read_mat_law35(block, model, log)
        return
    if lawname in ("LAW62", "VISC_HYP", "LAW62_VISC_HYP"):
        read_mat_law62(block, model, log)
        return
    if lawname in ("LAW28", "HONEYCOMB", "HONEYCOMB_SOL", "HONEY_SOL", "LAW28_HONEYCOMB", "LAW28_HONEYCOMB_SOL"):
        read_mat_law28(block, model, log)
        return
    if lawname in ("LAW44", "COWPER_SYMONDS", "PLAS_COWPER", "LAW44_COWPER_SYMONDS"):
        read_mat_law44(block, model, log)
        return
    # M173/M565: MAT LAW88 (HYPER_ELAS, TABULATED_HYPERELASTIC, TAB_HYP)
    if lawname in ("LAW88", "HYPER_ELAS", "LAW88_HYPER_ELAS", "TABULATED_HYPERELASTIC", "TABULATED_HYP", "TAB_HYP", "MAT_LAW88", "MAT_HYPER_ELAS", "MAT_TABULATED_HYPERELASTIC", "MAT_TAB_HYP"):
        read_mat_law88(block, model, log)
        return
    if lawname in ("LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE", "LAW92_ARRUDA_BOYCE"):
        read_mat_law92(block, model, log)
        return
    if lawname in ("LAW94", "YEOH", "LAW94_YEOH", "MAT_LAW94", "MAT_YEOH"):
        read_mat_law94(block, model, log)
        return
    if lawname in ("LAW46", "HYD_VISC", "LES_FLUID", "LAW46_HYD_VISC"):
        read_mat_law46(block, model, log)
        return
    if lawname in ("LAW69", "HYP_EXT_COMP", "HYPER_EXT_COMP", "LAW69_HYP_EXT_COMP", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYP_ELAS"):
        read_mat_law69(block, model, log)
        return
    # M174: MAT LAW124 (CDPM2), LAW126 (JOHNSON_HOLMQUIST_CONCRETE), LAW125 (LAMINATED_COMPOSITE), LAW127 (ENHANCED_COMPOSITE), LAW130 (MODIFIED_HONEYCOMB)
    if lawname in ("LAW124", "CDPM2", "LAW124_CDPM2"):
        read_mat_law124(block, model, log)
        return
    if lawname in ("LAW126", "JOHNSON_HOLMQUIST_CONCRETE", "JH_CONC", "JHC", "LAW126_JOHNSON_HOLMQUIST_CONCRETE"):
        read_mat_law126(block, model, log)
        return
    if lawname in ("LAW125", "LAMINATED_COMPOSITE", "LAM_COMP", "LAW125_LAMINATED_COMPOSITE"):
        read_mat_law125(block, model, log)
        return
    if lawname in ("LAW127", "ENHANCED_COMPOSITE", "ENH_COMP", "LAW127_ENHANCED_COMPOSITE"):
        read_mat_law127(block, model, log)
        return
    if lawname in ("LAW130", "MODIFIED_HONEYCOMB", "MOD_HONEYCOMB", "LAW130_MODIFIED_HONEYCOMB"):
        read_mat_law130(block, model, log)
        return
    # M175: MAT LAW128 (HILL_VISC_PLAST), LAW129 (THERM_CREEP), LAW123 (DAIMLER_PINHO), LAW132 (DAIMLER_CAMANHO), LAW134 (VISCOUS_FOAM)
    if lawname in ("LAW128", "HILL_VISC_PLAST", "HILL_VISCO_PLASTIC", "LAW128_HILL_VISC_PLAST"):
        read_mat_law128(block, model, log)
        return
    if lawname in ("LAW129", "THERM_CREEP", "THERMAL_CREEP", "THERMO_ELASTO_VISCOPLASTIC_CREEP", "LAW129_THERM_CREEP"):
        read_mat_law129(block, model, log)
        return
    if lawname in ("LAW123", "DAIMLER_PINHO", "DAIMLER-PINHO", "LAMINATED_FRACTURE_DAIMLER_PINHO", "LAW123_DAIMLER_PINHO"):
        read_mat_law123(block, model, log)
        return
    if lawname in ("LAW132", "DAIMLER_CAMANHO", "DAIMLER-CAMANHO", "LAMINATED_FRACTURE_DAIMLER_CAMANHO", "LAW132_DAIMLER_CAMANHO"):
        read_mat_law132(block, model, log)
        return
    if lawname in ("LAW134", "VISCOUS_FOAM", "VISC_FOAM", "LAW134_VISCOUS_FOAM"):
        read_mat_law134(block, model, log)
        return
    # M176: MAT LAW104 (JOHNS_VOCE_DRUCKER), LAW105 (POWDER_BURN), LAW106 (JCOOK_ALM), LAW107 (PAPER_LIGHT), LAW110 (VEGTER), LAW115 (DESHPANDE_FLECK)
    if lawname in ("LAW104", "JOHNS_VOCE_DRUCKER", "PLAS_DRUCKER", "LAW104_JOHNS_VOCE_DRUCKER"):
        read_mat_law104(block, model, log)
        return
    if lawname in ("LAW105", "POWDER_BURN", "POWDERBURN", "LAW105_POWDER_BURN"):
        read_mat_law105(block, model, log)
        return
    if lawname in ("LAW106", "JCOOK_ALM", "JOHNS_COOK_ALM", "LAW106_JCOOK_ALM"):
        read_mat_law106(block, model, log)
        return
    if lawname in ("LAW107", "PAPER_LIGHT", "PLAS_PAPER_LIGHT", "LAW107_PAPER_LIGHT", "PFEIFFER", "MAT_PFEIFFER", "MAT_PAPER_LIGHT", "MAT_107"):
        read_mat_law107(block, model, log)
        return
    if lawname in ("LAW110", "110", "VEGTER", "PLAS_VEGTER", "LAW110_VEGTER", "MAT_LAW110", "MAT_VEGTER", "MAT_PLAS_VEGTER", "MLAW110", "MAT_110"):
        read_mat_law110(block, model, log)
        return
    if lawname in ("LAW115", "DESHPANDE_FLECK", "DESHFLECK", "FOAM_DESHPANDE", "LAW115_DESHPANDE_FLECK"):
        read_mat_law115(block, model, log)
        return
    if lawname in ("LAW109", "109", "TAB_PLAS", "ELASTO_PLAS_TAB", "LAW109_TAB_PLAS", "LAW109_LAW109", "MAT_LAW109", "MAT_TAB_PLAS", "MAT_ELASTO_PLAS_TAB", "MLAW109"):
        read_mat_law109(block, model, log)
        return
    if lawname in ("LAW111", "MARLOW", "LAW111_MARLOW"):
        read_mat_law111(block, model, log)
        return
    if lawname in ("LAW112", "PAPER", "PLAS_PAPER", "XIA", "LAW112_PAPER"):
        read_mat_law112(block, model, log)
        return
    if lawname in ("LAW116", "COH_HYST", "COHESIVE_HYSTERETIC", "LAW116_COH_HYST"):
        read_mat_law116(block, model, log)
        return
    if lawname in ("LAW122", "MODIFIED_LADEVEZE", "LADEVEZE_DELAM", "LADEVEZE", "LAW122_MODIFIED_LADEVEZE"):
        read_mat_law122(block, model, log)
        return
    if lawname in ("LAW158", "FABR_NL", "FABRIC_NL", "FABRIC_NONLIN", "LAW158_FABR_NL"):
        read_mat_law158(block, model, log)
        return
    # M179: MAT LAW113 (SPR_BEAM), LAW79 (JOHN_HOLM), VISC_LPRONY
    if lawname in ("LAW113", "SPR_BEAM", "SPRING_BEAM", "LAW113_SPR_BEAM"):
        read_mat_law113(block, model, log)
        return
    if lawname in ("79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "LAW79_JOHN_HOLM", "MAT_JOHN_HOLM", "MAT_JH2", "MAT_JOHNSON_HOLMQUIST", "MAT_LAW79"):
        read_mat_law79(block, model, log)
        return
    if lawname in ("VISC_LPRONY", "LPRONY", "VISCO_LPRONY"):
        read_mat_visc_lprony(block, model, log)
        return
    # M180: MAT LAW190 (FOAM_DUBOIS), LAW41 (LEE_T)
    if lawname in ("LAW190", "FOAM_DUBOIS", "DUBOIS", "LAW190_FOAM_DUBOIS"):
        read_mat_law190(block, model, log)
        return
    if lawname in ("LAW41", "LEE_T", "LEETARVER", "LEE_TARVER", "LAW41_LEE_T"):
        read_mat_law41(block, model, log)
        return
    # M182: MAT LAW114 (SPR_SEATBELT), LAW117 (COH_TAB), LAW119 (SH_SEATBELT), LAW120 (TAPO), LAW121 (PLAS_RATE)
    if lawname in ("LAW114", "SPR_SEATBELT", "SEATBELT_SPRING", "LAW114_SPR_SEATBELT"):
        read_mat_law114(block, model, log)
        return
    if lawname in ("LAW117", "COH_TAB", "COHESIVE_TABULATED", "LAW117_COH_TAB"):
        read_mat_law117(block, model, log)
        return
    if lawname in ("LAW119", "SH_SEATBELT", "SEATBELT_SHELL", "LAW119_SH_SEATBELT"):
        read_mat_law119(block, model, log)
        return
    if lawname in ("LAW120", "TAPO", "TAB_PONT_ORTH", "LAW120_TAPO"):
        read_mat_law120(block, model, log)
        return
    if lawname in ("LAW121", "PLAS_RATE", "PLAS_TAB_RATE", "LAW121_PLAS_RATE"):
        read_mat_law121(block, model, log)
        return
    # M183: LAW50 (VISC_HONEY), LAW57 (BARLAT3), LAW87 (BARLAT_YLD2000), LAW95 (BERGSTROM_BOYCE), LAW163 (CRUSHABLE_FOAM), LAW169 (ARUP_ADHESIVE)
    if lawname in ("50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_50", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM", "LAW50_VISC_HONEY", "LAW50_HYP_FOAM"):
        read_mat_law50(block, model, log)
        return
    if lawname in ("57", "LAW57", "BARLAT3", "MAT_BARLAT3", "LAW57_BARLAT3"):
        read_mat_law57(block, model, log)
        return
    if lawname in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_87", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT", "LAW87_BARLAT2000", "LAW87_BARLAT_2000"):
        read_mat_law87(block, model, log)
        return
    if lawname in ("95", "LAW95", "MAT_95", "MAT_LAW95", "BERGSTROM_BOYCE", "HYP_VISC_PLAS", "FOAM_TAB", "MAT_BERGSTROM_BOYCE", "LAW95_BERGSTROM_BOYCE"):
        read_mat_law95(block, model, log)
        return
    if lawname in ("163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM", "LAW163_CRUSHABLE_FOAM", "LAW163_CRUSH_FOAM"):
        read_mat_law163(block, model, log)
        return
    if lawname in ("LAW169", "ARUP_ADHESIVE", "COH_TAB_3D", "COH_3D", "MAT_ARUP_ADHESIVE", "MAT_COH_TAB_3D", "MAT_COH_3D", "LAW169_ARUP_ADHESIVE"):
        read_mat_law169(block, model, log)
        return
    if lawname in ("49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_STEINB", "MAT_STEINBERG", "MAT_STEINBERG_GUINAN", "LAW49_STEINB"):
        read_mat_law49(block, model, log)
        return
    if lawname in (
        "76", "LAW76", "SAMP", "SAMP-1", "SAMP_1",
        "PLAS_SAMP", "SAMP_PLAS", "MAT_SAMP", "MAT_SAMP-1",
        "MAT_PLAS_SAMP", "LAW76_SAMP", "LAW76_SAMP-1", "MAT_LAW76",
    ):
        read_mat_law76(block, model, log)
        return
    # M185 / M551: LAW60 (PLAS_T3, FABRIC), LAW63 (HANSEL), LAW48 (ZHAO), LAW26 (SESAM)
    if lawname in ("LAW60", "PLAS_T3", "PLAST_T3", "MAXWELL", "FABRIC", "MAT_PLAS_T3", "MAT_MAXWELL", "MAT_FABRIC", "LAW60_PLAS_T3", "LAW60_FABRIC"):
        read_mat_law60(block, model, log)
        return
    if lawname in ("LAW63", "HANSEL", "PLAS_HANSEL", "TRANSFO_PLAS", "MAT_HANSEL", "LAW63_HANSEL"):
        read_mat_law63(block, model, log)
        return
    if lawname in ("LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "MAT_PLAS_ZHAO", "LAW48_ZHAO"):
        read_mat_law48(block, model, log)
        return
    if lawname in ("LAW26", "SESAM", "SESAME", "MAT_SESAM", "LAW26_SESAM"):
        read_mat_law26(block, model, log)
        return
    # M186: LAW6 (VISC_FLUID), LAW11 (BOUND), LAW77 (FOAM_AIR), LAW151 (MULTIFLUID), LAW187 (BARLAT20003D)
    if lawname in ("LAW6", "VISC_FLUID", "HYDRO", "HYDRO_VISC", "HYD_VISC", "K-EPS", "MAT_VISC_FLUID", "MAT_HYDRO", "MAT_HYDRO_VISC", "MAT_HYD_VISC", "MAT_K-EPS", "LAW6_VISC_FLUID", "LAW6_HYDRO"):
        read_mat_law6(block, model, log)
        return
    if lawname in ("LAW11", "BOUND", "B-K-EPS", "MAT_BOUND", "MAT_B-K-EPS", "LAW11_BOUND", "LAW11_B-K-EPS"):
        read_mat_law11(block, model, log)
        return
    if lawname in ("LAW77", "FOAM_AIR", "FOAM_HYST", "MAT_FOAM_AIR", "MAT_FOAM_HYST", "LAW77_FOAM_AIR", "LAW77_FOAM_HYST"):
        read_mat_law77(block, model, log)
        return
    if lawname in ("LAW151", "MULTIFLUID", "MULTI_MAT", "MULTIFLUID_MAT", "MAT_MULTIFLUID", "MAT_MULTI_MAT", "MAT_MULTIFLUID_MAT", "LAW151_MULTIFLUID"):
        read_mat_law151(block, model, log)
        return
    if lawname in ("LAW187", "BARLAT20003D", "BARLAT_3D", "PLAS_BARLAT3D", "BARLAT3D", "MAT_BARLAT20003D", "MAT_BARLAT_3D", "MAT_PLAS_BARLAT3D", "LAW187_BARLAT20003D"):
        read_mat_law187(block, model, log)
        return
    # M187: LAW3 (PLAS_BOST), LAW4 (HYD_JCOOK), LAW5 (JCOOK_TAB), LAW10 (SOIL), LAW14 (CAM_CLAY), LAW21 (DUCKHUB), LAW32 (HILL_TAB), LAW37 (BIQUAD)
    if lawname in ("LAW3", "PLAS_BOST", "BOSTEELS", "MAT_PLAS_BOST", "MAT_BOSTEELS", "LAW3_PLAS_BOST"):
        read_mat_law3(block, model, log)
        return
    if lawname in ("LAW4", "HYD_JCOOK", "HYDRO_JCOOK", "MAT_HYD_JCOOK", "MAT_HYDRO_JCOOK", "LAW4_HYD_JCOOK"):
        read_mat_law4(block, model, log)
        return
    if lawname in ("LAW5", "JCOOK_TAB", "JOHN_COOK_TAB", "MAT_JCOOK_TAB", "MAT_JOHN_COOK_TAB", "LAW5_JCOOK_TAB"):
        read_mat_law5(block, model, log)
        return
    if lawname in ("LAW10", "SOIL", "SOIL_CONC", "MAT_SOIL", "MAT_SOIL_CONC", "LAW10_SOIL"):
        read_mat_law10(block, model, log)
        return
    if lawname in ("CAM_CLAY", "CAMCLAY", "MAT_CAM_CLAY", "MAT_CAMCLAY", "LAW14_CAM_CLAY"):
        read_mat_cam_clay(block, model, log)
        return
    if lawname in ("LAW21", "21", "DPRAG", "MAT_DPRAG", "LAW21_DPRAG", "DUCKHUB", "MAT_DUCKHUB", "LAW21_DUCKHUB"):
        read_mat_law21(block, model, log)
        return
    if lawname in ("LAW32", "HILL", "MAT_HILL", "LAW32_HILL", "32"):
        read_mat_law32(block, model, log)
        return
    if lawname in ("LAW37", "BIQUAD", "BANABIC", "MAT_BIQUAD", "MAT_BANABIC", "LAW37_BIQUAD",
                   "BIPHAS", "BIPHASIC", "MAT_BIPHAS", "MAT_BIPHASIC", "LAW37_BIPHAS"):
        read_mat_law37(block, model, log)
        return
    # M188: LAW12 (3PARBI), LAW13 (HONEYCOMB), LAW15 (CHANG), LAW18 (CONCR_DRA), LAW22 (TSAI_WU), LAW25 (COMP_PLAS), LAW28 (HONEYCOMB_SOL)
    if lawname in ("12", "LAW12", "3PARBI", "3D_COMP", "COMP_3D", "RAGAB", "MAT_3PARBI", "MAT_3D_COMP", "MAT_COMP_3D", "MAT_RAGAB", "LAW12_3PARBI", "LAW12_3D_COMP", "LAW12_COMP_3D"):
        read_mat_law12(block, model, log)
        return
    if lawname in ("LAW13", "RIGID", "MAT_RIGID", "LAW13_RIGID"):
        read_mat_law13(block, model, log)
        return
    if lawname in ("LAW15", "CHANG", "CHANG_CHANG", "MAT_CHANG", "MAT_CHANG_CHANG", "LAW15_CHANG",
                   "PLAS_ANISO", "COMP_CHANG", "MAT_PLAS_ANISO", "MAT_COMP_CHANG",
                   "LAW15_PLAS_ANISO", "LAW15_COMP_CHANG"):
        read_mat_law15(block, model, log)
        return
    if lawname in ("LAW18", "CONCR_DRA", "DRAGON", "THERM", "MAT_CONCR_DRA", "MAT_DRAGON", "MAT_THERM", "LAW18_CONCR_DRA", "LAW18_THERM"):
        read_mat_law18(block, model, log)
        return
    if lawname in ("LAW22", "TSAIWU", "DAMA", "MAT_TSAIWU", "MAT_DAMA", "LAW22_TSAI_WU", "LAW22_DAMA", "PLAS_DAMA", "MAT_PLAS_DAMA", "22"):
        read_mat_law22(block, model, log)
        return
    if lawname in ("TSAI_WU", "MAT_TSAI_WU"):
        _vc = [c for c in block.cards if not c.is_blank and not c.raw.strip().startswith("#")]
        if len(_vc) >= 7:
            read_mat_law25(block, model, log)
        else:
            read_mat_law22(block, model, log)
        return
    if lawname in ("LAW25", "COMP_PLAS", "COMPOSITE_PLAS", "COMPSH", "MAT_COMP_PLAS", "MAT_COMPOSITE_PLAS", "MAT_COMPSH", "LAW25_COMP_PLAS", "LAW25_COMPSH", "CRASURV", "MAT_CRASURV", "LAW25_CRASURV", "LAW25_TSAI_WU"):
        read_mat_law25(block, model, log)
        return
    if lawname in ("LAW28", "HONEYCOMB", "HONEYCOMB_SOL", "HONEY_SOL", "MAT_HONEYCOMB", "MAT_HONEYCOMB_SOL", "MAT_HONEY_SOL", "LAW28_HONEYCOMB", "LAW28_HONEYCOMB_SOL"):
        read_mat_law28(block, model, log)
        return
    # M189: LAW52 (GURSON), LAW16 (GRAY), LAW14 (COMPSO), LAW59 (CONNECT), LAW64 (TRANSFO_MART)
    if lawname in ("LAW52", "GURSON", "PLAS_GURS", "MAT_GURSON", "MAT_PLAS_GURS", "LAW52_GURSON"):
        read_mat_law52(block, model, log)
        return
    if lawname in ("LAW16", "GRAY", "CAST_IRON", "MAT_GRAY", "MAT_CAST_IRON", "LAW16_GRAY"):
        read_mat_law16(block, model, log)
        return
    if lawname in ("COMPSO", "COMP_SOL", "MAT_COMPSO", "MAT_COMP_SOL", "LAW14_COMPSO", "LAW14_COMP_SOL"):
        read_mat_compso(block, model, log)
        return
    if lawname in ("LAW14", "MAT_LAW14"):
        read_mat_law14(block, model, log)
        return
    if lawname in ("LAW59", "CONNECT", "CONNECTOR", "MAT_CONNECT", "MAT_CONNECTOR", "LAW59_CONNECT"):
        read_mat_law59(block, model, log)
        return
    if lawname in ("LAW64", "TRANSFO_MART", "MARTENSITE", "MAT_TRANSFO_MART", "MAT_MARTENSITE", "LAW64_TRANSFO_MART"):
        read_mat_law64(block, model, log)
        return
    # M190: LAW68, LAW72, LAW65, LAW58, LAW20, LAW38, LAW29, LAW34, LAW23, LAW78
    if lawname in ("LAW68", "COSSER", "COSSERAT", "MAT_COSSER", "MAT_COSSERAT", "LAW68_COSSER"):
        read_mat_law68(block, model, log)
        return
    if lawname in ("LAW72", "HILL_MMC", "MAT_HILL_MMC", "LAW72_HILL_MMC"):
        read_mat_law72(block, model, log)
        return
    if lawname in ("LAW65", "ELASTOMER", "MAT_ELASTOMER", "LAW65_ELASTOMER"):
        read_mat_law65(block, model, log)
        return
    if lawname in ("LAW58", "FABR_A", "MAT_FABR_A", "FABRIC_A", "MAT_FABRIC_A", "MAT_LAW58", "LAW58_FABR_A"):
        read_mat_law58(block, model, log)
        return
    if lawname in ("LAW20", "BIMAT", "MAT_BIMAT", "LAW20_BIMAT"):
        read_mat_law20(block, model, log)
        return
    if lawname in ("LAW38", "VISC_TAB", "MAT_VISC_TAB", "LAW38_VISC_TAB", "38"):
        read_mat_law38(block, model, log)
        return
    if lawname in ("LAW29", "FEM", "MAT29_FEM", "MAT_FEM", "LAW29_FEM"):
        read_mat_law29(block, model, log)
        return
    if lawname in ("LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW", "MAT_BOLTZMAN", "MAT_BOLTZMANN", "MAT_VISC_MAXW", "LAW34_BOLTZMAN", "LAW34_VISC_MAXW"):
        read_mat_law34(block, model, log)
        return

    if lawname in ("LAW23", "LAW23_PLAS_DAMA"):
        read_mat_law23(block, model, log)
        return
    if lawname in ("LAW78", "MAT_LAW78", "LAW78_78"):
        read_mat_law78(block, model, log)
        return
    # M191 / M570: LAW100 (VISC_HYP / MNF / SPOTWELD), LAW97 (EXPLOSIVE_JWLS), LAW71 (SUPER_ELAS), LAW73 (THERM_HILL), LAW84 (SWIFT_VOCE), LAW93 (ORTH_HILL), LAW133 (GRANULAR), LAW101 (PLAS_POLY), LAW43 (HILL_TAB)
    if lawname in ("100", "LAW100", "VISC_HYP", "MNF", "MAT_100", "MAT_LAW100", "MAT_VISC_HYP", "MAT_MNF", "LAW100_VISC_HYP", "LAW100_MNF", "SPOTWELD", "STRUCTURAL_ADHESIVE", "MAT_SPOTWELD", "MAT_STRUCTURAL_ADHESIVE", "LAW100_SPOTWELD"):
        read_mat_law100(block, model, log)
        return
    if lawname in ("LAW97", "EXPLOSIVE_JWLS", "JWLS", "MAT_EXPLOSIVE_JWLS", "MAT_JWLS", "LAW97_EXPLOSIVE_JWLS"):
        read_mat_law97(block, model, log)
        return
    if lawname in ("71", "LAW71", "SUPER_ELAS", "NITINOL", "MAT_71", "MAT_LAW71", "MAT_SUPER_ELAS", "MAT_NITINOL", "LAW71_SUPER_ELAS", "LAW71_NITINOL"):
        read_mat_law71(block, model, log)
        return
    if lawname in ("73", "LAW73", "LAW73_THERM_HILL", "LAW73_HILL_THERM", "MAT_LAW73", "MAT_73"):
        read_mat_law73(block, model, log)
        return
    if lawname in ("LAW84", "SWIFT_VOCE", "PLAS_SWIFT_VOCE", "MAT_SWIFT_VOCE", "MAT_PLAS_SWIFT_VOCE", "LAW84_SWIFT_VOCE"):
        read_mat_law84(block, model, log)
        return
    if lawname in ("LAW93", "ORTH_HILL", "MAT_ORTH_HILL", "LAW93_ORTH_HILL"):
        read_mat_law93(block, model, log)
        return
    if lawname in ("LAW133", "GRANULAR", "MAT_GRANULAR", "LAW133_GRANULAR"):
        read_mat_law133(block, model, log)
        return
    if lawname in ("101", "LAW101", "PP", "MAT_PP", "PLAS_POLY", "MAT_PLAS_POLY", "LAW101_PLAS_POLY", "LAW101_PP", "MAT_LAW101"):
        read_mat_law101(block, model, log)
        return
    if lawname in ("LAW43", "HILL_TAB", "MAT_HILL_TAB", "LAW43_HILL_TAB", "HILL_PLAS_TAB", "MAT_HILL_PLAS_TAB"):
        read_mat_law43(block, model, log)
        return
    # M193: LAW53 (TSAI_TAB), LAW54 (PREDIT), LAW74 (HILL_THERM), LAW82 (OGDEN)
    if lawname in ("LAW53", "TSAI_TAB", "MAT_TSAI_TAB", "LAW53_TSAI_TAB"):
        read_mat_law53(block, model, log)
        return
    if lawname in ("LAW54", "PREDIT", "MAT_PREDIT", "LAW54_PREDIT"):
        read_mat_law54(block, model, log)
        return
    if lawname in ("74", "LAW74", "HILL_3D", "ORTH_PLAS", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "LAW74_HILL_THERM", "LAW74_THERM_HILL", "LAW74_HILL_3D", "LAW74_ORTH_PLAS", "MAT_74"):
        read_mat_law74(block, model, log)
        return
    if lawname in ("HILL_THERM", "MAT_HILL_THERM", "THERM_HILL", "MAT_THERM_HILL"):
        # Disambiguate between LAW73 (shell Thermal Hill, Card 2 has 2 values) and LAW74 (3D solid Thermal Hill, Card 2 has 5 values)
        _t, _cards = _fixed_data(block) if block.fixed else _title_and_data(block)
        _vc = [c for c in _cards if not c.is_blank and not c.raw.strip().startswith("#")]
        if len(_vc) > 1:
            _toks = _vc[1].tokens() if not block.fixed else [_vc[1].raw[i*20:(i+1)*20].strip() for i in range(5) if _vc[1].raw[i*20:(i+1)*20].strip()]
            if len(_toks) > 2:
                read_mat_law74(block, model, log)
                return
        read_mat_law73(block, model, log)
        return
    if lawname in ("LAW82", "MAT_LAW82", "LAW82_OGDEN", "OGDEN_82", "MAT_OGDEN_82"):
        read_mat_law82(block, model, log)
        return
    if lawname in ("OGDEN", "MAT_OGDEN"):
        title, data_c = _fixed_data(block) if block.fixed else _title_and_data(block)
        data_valid = [c for c in data_c if not c.is_blank and not c.raw.strip().startswith("#")]
        is_law82 = False
        if len(data_valid) > 1:
            raw_field = (data_valid[1].raw[:10] if block.fixed else data_valid[1].raw).replace(",", " ").split()
            first_val = raw_field[0].strip() if raw_field else ""
            if first_val.isdigit() and 1 <= int(first_val) <= 10:
                is_law82 = True
        if is_law82:
            read_mat_law82(block, model, log)
            return
    # M194: LAW40 (CONCR_SUB), LAW102 (HILL_48), NLOCAL
    if lawname in ("LAW40", "CONCR_SUB", "MAT_CONCR_SUB", "LAW40_CONCR_SUB"):
        read_mat_law40(block, model, log)
        return
    if lawname in ("LAW102", "HILL_48", "MAT_HILL_48", "LAW102_HILL_48"):
        read_mat_law102(block, model, log)
        return
    if lawname in ("NLOCAL", "NONLOCAL_PLAS", "MAT_NLOCAL"):
        read_mat_nlocal(block, model, log)
        return
    # M195: LAW103 (HENSEL_SPITTEL), LAW108 (SPR_GENE), PLAS_PREDEF, DPRAG2
    if lawname in ("LAW103", "HENSEL_SPITTEL", "HENSEL-SPITTEL", "HEN", "MAT_HENSEL_SPITTEL", "LAW103_HENSEL_SPITTEL", "PLAS_HENS", "MAT_PLAS_HENS", "MAT_LAW103"):
        read_mat_law103(block, model, log)
        return
    # M574: LAW104 (JOHNS_VOCE_DRUCKER / DRUCKER / PLAS_DRUCK)
    if lawname in ("LAW104", "JOHNS_VOCE_DRUCKER", "JOHNS-VOCE-DRUCKER", "DRUCKER", "PLAS_DRUCK", "MAT_LAW104", "MAT_DRUCKER", "MAT_JOHNS_VOCE_DRUCKER", "MAT_PLAS_DRUCK", "LAW104_DRUCKER", "LAW104_JOHNS_VOCE_DRUCKER"):
        read_mat_law104(block, model, log)
        return
    if lawname in ("LAW108", "SPR_GENE", "MAT_SPR_GENE", "LAW108_SPR_GENE"):
        read_mat_law108(block, model, log)
        return
    if lawname in ("PLAS_PREDEF", "PREDEF", "PREDEF_PLAS", "MAT_PLAS_PREDEF"):
        read_mat_plas_predef(block, model, log)
        return
    if lawname in ("DPRAG2", "DRUCKER_PRAGER_2", "MAT_DPRAG2"):
        read_mat_dprag2(block, model, log)
        return
    # M196: LAW113 (SPR_BEAM), LAW95 (SEW), LAW48 (ZHAO), LAW49 (STEINB), LAW106 (P_FOAM), LAW77 (OGDEN_HYPO), LAW63 (SOIL_DISC), LAW92 (HILL_ORTH)
    if lawname in ("LAW113", "SPR_BEAM", "MAT_SPR_BEAM", "LAW113_SPR_BEAM"):
        read_mat_law113(block, model, log)
        return
    if lawname in ("LAW95", "SEW", "SEWING", "MAT_SEW", "MAT_SEWING", "LAW95_SEW"):
        read_mat_law95(block, model, log)
        return
    if lawname in ("LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "MAT_PLAS_ZHAO", "LAW48_ZHAO"):
        read_mat_law48(block, model, log)
        return
    if lawname in ("49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_STEINB", "MAT_STEINBERG", "MAT_STEINBERG_GUINAN", "LAW49_STEINB"):
        read_mat_law49(block, model, log)
        return
    if lawname in ("LAW106", "P_FOAM", "POLY_FOAM", "MAT_P_FOAM", "MAT_POLY_FOAM", "LAW106_P_FOAM"):
        read_mat_law106(block, model, log)
        return
    if lawname in ("LAW77", "OGDEN_HYPO", "HYPO_VISCO", "MAT_OGDEN_HYPO", "MAT_HYPO_VISCO", "LAW77_OGDEN_HYPO"):
        read_mat_law77(block, model, log)
        return
    if lawname in ("LAW63", "SOIL_DISC", "HANSEL", "MAT_SOIL_DISC", "MAT_HANSEL", "LAW63_SOIL_DISC"):
        read_mat_law63(block, model, log)
        return
    if lawname in ("LAW92", "HILL_ORTH", "MAT_HILL_ORTH", "LAW92_HILL_ORTH"):
        read_mat_law92(block, model, log)
        return
    if lawname in ("HEAT", "HEAT_TRANSFER"):
        read_mat_heat(block, model, log)
        return
    if lawname in ("NONLOCAL", "NON_LOCAL"):
        read_mat_nonlocal(block, model, log)
        return
    # M199: DRUCKER_PRAGER / BRITTLE
    subaction = block.parts[2].upper() if len(block.parts) > 2 else ""
    if (lawname in ("DRUCKER_PRAGER", "BRITTLE", "MAT_DRUCKER_PRAGER", "MAT_BRITTLE", "LAW51_DRUCKER_PRAGER", "LAW51_BRITTLE")
            or subaction in ("DRUCKER_PRAGER", "BRITTLE")):
        read_mat_law51(block, model, log)
        return
    # M591: LAW126 (Johnson-Holmquist Concrete / HJC) and LAW169 (Arup Structural Adhesive)
    if lawname in ("LAW126", "JOHNSON_HOLMQUIST_CONCRETE", "MAT_LAW126", "MAT_JOHNSON_HOLMQUIST_CONCRETE", "LAW126_JOHNSON_HOLMQUIST_CONCRETE", "HJC", "MAT_HJC"):
        read_mat_law126(block, model, log)
        return
    if lawname in ("LAW169", "ARUP_ADHESIVE", "ARUP", "ADHESIVE", "MAT_LAW169", "MAT_ARUP_ADHESIVE", "LAW169_ARUP_ADHESIVE"):
        read_mat_law169(block, model, log)
        return

    law_aliases = {"LAW1": 1, "ELAST": 1, "LAW2": 2, "PLAS_JOHNS": 2,
                   "LAW27": 27, "PLAS_BRIT": 27,
                   "LAW36": 36, "PLAS_TAB": 36,
                   "LAW42": 42, "OGDEN": 42}
    if lawname not in law_aliases:
        # M37: every OTHER law goes through the cfg-driven generic
        # reader (pyradioss/input/mat_reader.py) — full params + density
        # parse (mass init works); laws with a registered physics
        # constructor (MAT_PHYSICS_REGISTRY) become live materials,
        # everything else an InactiveMaterial the Engine refuses to run.
        mat_reader.read_generic_mat(block, model, log)
        return
    law = law_aliases[lawname]
    if block.fixed:
        # M37 fixed dialect: title card always present, blank cards kept
        # (they are real cards with every field default — the card
        # indices of the cfg layouts only line up with them in place)
        title, cards = _fixed_data(block)
    else:
        title, cards = _title_and_data(block)
    if len(cards) < 2:
        log.error(f"/MAT/{lawname}/{block.user_id}: needs at least rho and "
                  f"elasticity cards", block.source)
        return
    # density card: cfg "%20lg%20lg" RHO_I RHO_O — RHO_I is the density
    rho0 = _fval(cards[0].cut("MAT_RHO")[0]) if block.fixed \
        else cards[0].floats()[0]

    if law == 42:
        if block.fixed:
            # real matl42_Ogden.cfg (radioss140) card order: rho /
            # [nu sig_cut] / mu_1..5 / mu_6..10 / alpha_1..5 /
            # alpha_6..10 — nu sits BEFORE the moduli (blank -> 0.495,
            # hm_read_mat42.F line 148) and the port carries 5 pairs
            # (pairs 6..10 warned about when set).
            f = cards[1].cut("LAW42_NU")
            nu = _fval(f[0])
            mu = _cut_floats(cards[2], "F20X5") if len(cards) >= 3 \
                else [0.0] * 5
            al = _cut_floats(cards[4], "F20X5") if len(cards) >= 5 \
                else [0.0] * 5
            hi = []
            if len(cards) >= 4:
                hi += _cut_floats(cards[3], "F20X5")
            if len(cards) >= 6:
                hi += _cut_floats(cards[5], "F20X5")
            _warn_ignored(log, f"/MAT/LAW42/{block.user_id}", block.source,
                          [("sig_cut", f[1])]
                          + [("mu/alpha_6..10", v) for v in hi if v])
        else:
            # port compact layout: rho / mu / alpha / nu
            mu = _floats(cards[1], 5)
            al = _floats(cards[2], 5) if len(cards) >= 3 else [0.0] * 5
            nu = cards[3].floats()[0] if len(cards) >= 4 else 0.495
        nu = nu if nu > 0 else 0.495
        used = [(m, a) for m, a in zip(mu, al) if m != 0.0]
        if not used:
            log.error(f"/MAT/LAW42/{block.user_id}: all mu_p are zero",
                      block.source)
            return
        G0 = sum(m * a for m, a in used) / 2.0
        if G0 <= 0.0:
            log.error(f"/MAT/LAW42/{block.user_id}: sum(mu_p * alpha_p) must "
                      f"be > 0 (material stability)",
                      block.source)
            return
        params = {"E": 2.0 * G0 * (1.0 + nu), "nu": nu,
                  "mu": [m for m, _ in used], "alpha": [a for _, a in used]}
        model.materials[block.user_id] = Material(
            id=block.user_id, law=law, rho0=rho0, title=title, params=params)
        return

    law2_iflag = 0
    if block.fixed:
        # matl*.cfg elasticity card "%20lg%20lg%10d%10d" E Nu Iflag VP —
        # cut at columns (abutting values, blank -> 0). LAW2 Iflag = 1
        # selects the SIG_Y/UTS/EUTS ultimate-tensile yield input, converted
        # to a/b/n below (hm_read_mat02_jc.F90, the iflag==1 branch).
        e_nu = cards[1].cut("MAT_E_NU")
        E, nu = _fval(e_nu[0]), _fval(e_nu[1])
        if law == 2:
            law2_iflag = _ival(e_nu[2])
    else:
        E, nu = _floats(cards[1], 2)
        if law == 2:
            # free dialect: an optional Iflag rides the E/Nu card 3rd token
            etok = cards[1].tokens()
            if len(etok) > 2:
                law2_iflag = _ival(etok[2])
    params = {"E": E, "nu": nu}
    if law == 2:
        if len(cards) < 3:
            log.error(f"/MAT/LAW2/{block.user_id}: missing A,B,n card",
                      block.source)
            return
        # yield card "%20lg"*5: Iflag=0 -> a b n EPS_p_max SIG_max0;
        # Iflag=1 -> SIG_Y UTS EUTS EPS_p_max SIG_max0. Real decks pack
        # e.g. '0.51.00000000000000E+301.00000000000000E+30' with no
        # whitespace — cut at columns for the fixed dialect.
        av = _cut_floats(cards[2], "LAW2_A") if block.fixed \
            else _floats(cards[2], 5, defaults=[0, 0, 1.0, 1e30, 1e30])
        A, B, n, epsmax, sigmax = av[:5]
        if law2_iflag == 1:
            # convert the ultimate-tensile input (A=SIG_Y, B=UTS, n=EUTS)
            # to the Johnson-Cook hardening a/b/n (hm_read_mat02_jc.F90)
            A, B, n = _law2_iflag1_to_abn(A, B, n, E, block, log)
        # Radioss conventions: eps_p_max=0 means "no limit", sig_max=0 too.
        params.update(A=A, B=B, n=n if n > 0 else 1.0,
                      eps_p_max=epsmax if epsmax > 0 else 1e30,
                      sig_max=sigmax if sigmax > 0 else 1e30)
        if len(cards) >= 4:
            if block.fixed:
                # "%20lg%20lg%10d%10d%20lg%20lg" c EPS_DOT_0 ICC Fsmooth
                # F_cut Chard — c/eps_dot_0 ported; the trailing flags are
                # read to surface the un-ported ones (Chard below).
                cv = cards[3].cut("LAW2_C")
                c, eps0 = _fval(cv[0]), _fval(cv[1], 1.0)
                chard = _fval(cv[5]) if len(cv) > 5 else 0.0
            else:
                cvals = _floats(cards[3], 6,
                                defaults=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
                c, eps0, chard = cvals[0], cvals[1], cvals[5]
            params.update(c=c, eps_dot_0=eps0 if eps0 > 0 else 1.0)
            # Chard (Fisokin) = iso-kinematic hardening fraction: 0 = pure
            # ISOTROPIC (the only mode this radial return implements — no
            # back-stress state is carried), 1 = pure Prager KINEMATIC. A
            # non-zero Chard would be silently mis-simulated, so warn (as
            # LAW44 does for its own kinematic term). Monotonic loading is
            # unaffected — isotropic and kinematic coincide until reversal.
            if chard != 0.0:
                log.warning(f"/MAT/LAW2/{block.user_id}: Chard={chard:g} "
                            f"(kinematic hardening) is not ported — the "
                            f"radial return is purely isotropic (correct "
                            f"only for monotonic loading)", block.source)
        else:
            params.update(c=0.0, eps_dot_0=1.0)
        if len(cards) >= 5:                    # thermal card (M6)
            mT, tmelt, rho_cp, ti = \
                _cut_floats(cards[4], "LAW2_M") if block.fixed else \
                _floats(cards[4], 4, defaults=[0.0, 0.0, 0.0, 298.0])
            ti = ti if ti > 0 else 298.0
            if mT > 0 and (tmelt <= ti or rho_cp <= 0):
                log.error(f"/MAT/LAW2/{block.user_id}: thermal card needs "
                          f"T_melt > T_i and rho_Cp > 0", block.source)
            elif mT > 0:
                params.update(mT=mT, T_melt=tmelt, rho_cp=rho_cp, T_i=ti)
    elif law == 27:
        if block.fixed:
            if len(cards) < 5:
                log.error(f"/MAT/LAW27/{block.user_id}: missing damage "
                          f"card 'eps_t1 eps_m1 dmax1 eps_f1'",
                          block.source)
                return
            plast = _cut_floats(cards[2], "LAW2_A") \
                + _cut_floats(cards[3], "LAW2_A")[:2]
            a, b, n, epsmax, ymax, c, eps0 = plast
            params.update(A=a, B=b, n=n, sig_max=ymax, c=c, eps_dot_0=eps0)
            dmg = cards[4:6]
        else:
            if len(cards) < 3:
                log.error(f"/MAT/LAW27/{block.user_id}: missing damage "
                          f"card 'eps_t1 eps_m1 dmax1 eps_f1'",
                          block.source)
                return
            if len(cards) >= 5:
                plast = _floats(cards[2], 5, defaults=[0.0]*5)
                plast2 = _floats(cards[3], 3, defaults=[0.0]*3)
                a, b, n, epsmax, ymax = plast
                c, eps0, icc = plast2
                params.update(A=a, B=b, n=n, sig_max=ymax, c=c, eps_dot_0=eps0)
                dmg = cards[4:6]
            else:
                params.update(A=0.0, B=0.0, n=0.0, sig_max=1e30, c=0.0, eps_dot_0=1.0)
                dmg = cards[2:4]

        def _dmg_vals(card, defaults):
            return _cut_floats(card, "LAW27_DMG") if block.fixed \
                else _floats(card, 4, defaults=defaults)

        t1, m1, d1, f1 = _dmg_vals(dmg[0], [0.0, 0.0, 0.999, 1e30])
        if not (0.0 < t1 < m1):
            log.error(f"/MAT/LAW27/{block.user_id}: need 0 < eps_t1 < "
                      f"eps_m1", block.source)
            return
        d1 = min(d1 if d1 > 0 else 0.999, 1.0)
        f1 = f1 if f1 > 0 else 1e30
        if len(dmg) >= 2:
            t2, m2, d2, f2 = _dmg_vals(dmg[1], [t1, m1, d1, f1])
            t2, m2 = (t2 if t2 > 0 else t1), (m2 if m2 > 0 else m1)
            d2 = min(d2 if d2 > 0 else d1, 1.0)
            f2 = f2 if f2 > 0 else f1
        else:
            t2, m2, d2, f2 = t1, m1, d1, f1
        params.update(eps_t1=t1, eps_m1=m1, dmax1=d1, eps_f1=f1,
                      eps_t2=t2, eps_m2=m2, dmax2=d2, eps_f2=f2)
    elif law == 36:
        if len(cards) < 4:
            log.error(f"/MAT/LAW36/{block.user_id}: needs N_funct and "
                      f"function-ID cards", block.source)
            return
        if block.fixed or len(cards) >= 6:
            # ---- the REAL fixed-format layout (see docstring) -------------
            # A fixed-dialect deck ALWAYS uses this layout (its blank
            # cards were kept, so the card indices line up — e.g. a
            # blank fct_IDp/Fscale card no longer shifts the function-id
            # list, the tensile_LAW36 crash of M36); the >= 6 card-count
            # heuristic remains for real-layout snippets without /BEGIN.
            # rho / E-card already read; Eps_p_max sits ON the E-card here.
            v1 = _cut_floats(cards[1], "LAW36_E") if block.fixed \
                else _floats(cards[1], 5)
            params["eps_p_max"] = v1[2] if v1[2] > 0 else 1e30
            ign: List[str] = []          # accepted-but-not-ported fields
            if v1[3] != 0.0 or v1[4] != 0.0:
                ign.append(f"Eps_t={v1[3]:g} Eps_m={v1[4]:g}")
            # card 4: N_funct F_smooth C_hard F_cut Eps_f (10 blank) VP
            f2 = _fixed_vals(cards[2], [10, 10, 20, 20, 20, 10, 10])
            nfun = _ival(f2[0])
            if nfun <= 0:
                log.error(f"/MAT/LAW36/{block.user_id}: N_funct={nfun} "
                          f"(needs at least one hardening curve)",
                          block.source)
                return
            for name, s, key in (("F_smooth", f2[1], "f_smooth"), ("C_hard", f2[2], "c_hard"),
                                 ("F_cut", f2[3], "f_cut"), ("VP", f2[6], "vp")):
                if s and _to_float(s) != 0.0:
                    params[key] = _to_float(s)
            
            if f2[4] and _to_float(f2[4]) != 0.0:
                ign.append(f"Eps_f={f2[4]}")
            # card 5: fct_IDp Fscale fct_IDE EInf CE
            f3 = _fixed_vals(cards[3], [10, 20, 10, 20, 20])
            if _ival(f3[0]) != 0:
                ign.append(f"fct_IDp={f3[0]} (pressure-dependent yield)")
            if _ival(f3[2]) != 0:
                ign.append(f"fct_IDE={f3[2]} (modulus evolution EInf/CE)")
            # function-id cards (5 per card %10d), then Fscale_i, then
            # Eps_dot_i (5 per card %20lg) — column-cut for fixed decks
            idx, fids = 4, []
            while idx < len(cards) and len(fids) < nfun:
                if block.fixed:
                    fids.extend(int(s) for s in cards[idx].cut("IDS10")
                                if s)
                else:
                    fids.extend(cards[idx].ints())
                idx += 1
            if len(fids) < nfun:
                log.error(f"/MAT/LAW36/{block.user_id}: N_funct={nfun} but "
                          f"only {len(fids)} function ids given",
                          block.source)
                return
            params["funct_ids"] = fids[:nfun]

            def _list20(card):
                return [_fval(s) for s in card.cut("F20X5") if s] \
                    if block.fixed else card.floats()

            nlist = (nfun + 4) // 5
            yfac: List[float] = []
            for _ in range(nlist):
                if idx < len(cards):
                    yfac.extend(_list20(cards[idx]))
                    idx += 1
            # hm_read_mat36.F: YFAC == 0 -> 1.0 (default scale).  YFAC
            # multiplies BOTH the curve value and its slope at engine
            # evaluation time (sigeps36.F: Y1*YFAC, DYDX1*YFAC before the
            # rate interpolation) — the port applies it once, per curve,
            # when the /FUNCT arrays are resolved (resolve_material_curves),
            # which is algebraically identical.  M40: previously parsed but
            # only WARNED about ("curves used unscaled") — on the RD-V-0700
            # LAW36 decks (curves in MPa, Fscale_i = 1e-3, work unit GPa)
            # that made the yield 1000x too high: the material never
            # yielded, /FAIL/JOHNSON never accumulated damage, and the
            # solids deviated ~19 % on IE where LAW2 matched at 2.3 %.
            yfac = [y if y != 0.0 else 1.0 for y in yfac[:nfun]]
            yfac += [1.0] * (nfun - len(yfac))
            params["yfac"] = yfac
            rates: List[float] = []
            for _ in range(nlist):
                if idx < len(cards):
                    rates.extend(_list20(cards[idx]))
                    idx += 1
            if nfun > 1:
                if len(rates) < nfun:
                    log.error(f"/MAT/LAW36/{block.user_id}: N_funct={nfun} "
                              f"needs {nfun} strain rates (Eps_dot_i cards)",
                              block.source)
                    return
                rates = rates[:nfun]
                if any(b <= a for a, b in zip(rates, rates[1:])):
                    log.error(f"/MAT/LAW36/{block.user_id}: strain rates "
                              f"must be strictly increasing", block.source)
                    return
                params["rates"] = rates
            else:
                params["rates"] = [0.0]
            if ign:
                log.warning(f"/MAT/LAW36/{block.user_id}: real-format "
                            f"fields not ported — ignored: "
                            f"{'; '.join(ign)}", block.source)
        else:
            # ---- the port's compact layout ---------------------------------
            v = _floats(cards[2], 2, defaults=[1, 0.0])
            nfun = int(v[0]) if v[0] > 0 else 1
            params["eps_p_max"] = v[1] if v[1] > 0 else 1e30
            fids = cards[3].ints()
            if len(fids) < nfun:
                log.error(f"/MAT/LAW36/{block.user_id}: N_funct={nfun} but "
                          f"only {len(fids)} function ids given",
                          block.source)
                return
            params["funct_ids"] = fids[:nfun]
            params["yfac"] = [1.0] * nfun    # compact layout has no Fscale_i
            if nfun > 1:
                if len(cards) < 5:
                    log.error(f"/MAT/LAW36/{block.user_id}: N_funct>1 needs "
                              f"a strain-rate card", block.source)
                    return
                rates = _floats(cards[4], nfun)
                if any(b <= a for a, b in zip(rates, rates[1:])):
                    log.error(f"/MAT/LAW36/{block.user_id}: strain rates "
                              f"must be strictly increasing", block.source)
                    return
                params["rates"] = rates
            else:
                params["rates"] = [0.0]
    model.materials[block.user_id] = Material(
        id=block.user_id, law=law, rho0=rho0, title=title, params=params)




def read_eos(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EOS/POLYNOMIAL/mat_ID`` and ``/EOS/IDEAL-GAS/mat_ID`` (M6):
    attach an equation of state to a material (the trailing id IS the
    material id, like /FAIL; the EOS pressure then replaces the law's
    own pressure for solid elements — laws 1, 2 and 36).

    POLYNOMIAL — Fortran starter/source/materials/eos (polynomial)::

        card 1:  C0   C1   C2   C3   C4   C5
        card 2:  E0                       (initial energy per unit
                                           initial volume; optional, 0)

      p = C0 + C1*mu + C2*max(mu,0)^2 + C3*mu^3 + (C4 + C5*mu)*E with
      mu = rho/rho0 - 1 (C2 dropped in tension, Radioss convention).

    IDEAL-GAS::

        card 1:  gamma   P0

      the perfect gas p = (gamma-1)*(1+mu)*E, stored as the equivalent
      polynomial C4 = C5 = gamma-1 with E0 = P0/(gamma-1). P0 > 0 makes
      a pre-pressurized gas (it pushes from cycle 1 — confine it).

    REAL dialect (cfg MAT/mat_EOS.cfg, radioss2022; M37): a **title
    card** leads the block (its text — 'EOS AIR HIGHP', 'Conversion of
    Mat Law 6...' — crashed the float parse pre-M37), and the layouts
    are::

        IDEAL-GAS:   title / Gamma  P0  PSH  T0  RHO_0     (5 x %20lg)
        POLYNOMIAL:  title / C0 C1 C2 C3 / C4 C5 E0 Psh RHO_0

      PSH (pressure shift) and RHO_0/T0 are accepted + warned when set
      (the port's EOS has no pressure shift and takes the density from
      the material).  The port's compact one-card dialects (no title)
      are kept: a POLYNOMIAL data card with 6 populated columns is the
      compact 'C0..C5' card, the real card 1 has only 4.
    """
    from ...model.entities import EquationOfState
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    aliases = {
        "IDEAL_GAS": "IDEAL-GAS",
        "STIFF_GAS": "STIFF-GAS",
        "STIFFENED_GAS": "STIFF-GAS",
        "NOBLE_ABEL": "NOBLE-ABEL",
        "GRUN": "GRUNEISEN",
        "POLY": "POLYNOMIAL",
        "LINE": "LINEAR",
        "TILL": "TILLOTSON",
        "MURN": "MURNAGHAN",
        "OSBO": "OSBORNE",
        "TYPE5": "JWL",
        "POWDERBURN": "POWDER-BURN",
        "POWDER_BURN": "POWDER-BURN",
        "COMPACTION_TAB": "COMPACTION_TAB",
        "COMPACTION-TAB": "COMPACTION_TAB",
        "IDEAL_GAS_VT": "IDEAL-GAS-VT",
        "IDEAL-GAS_VT": "IDEAL-GAS-VT",
        "IDEALGAS_VT": "IDEAL-GAS-VT",
        "IDEAL-GAS-VE": "IDEAL-GAS",
        "NASG": "NASG",
        "NOBLE-ABEL-STIFFENED-GAS": "NASG",
        "NOBLE_ABEL_STIFFENED_GAS": "NASG",
        "NA": "NOBLE-ABEL",
        "SG": "STIFF-GAS",
    }
    kind = aliases.get(kind, kind)
    supported_eos = (
        "POLYNOMIAL", "IDEAL-GAS", "LINEAR", "STIFF-GAS",
        "GRUNEISEN", "PUFF", "TILLOTSON", "MURNAGHAN",
        "OSBORNE", "LSZK", "NOBLE-ABEL", "JWL", "NASG",
        "COMPACT", "COMPACTION", "COMPACTION2", "COMPACTION_TAB", "SESAME", "IGNITION_GROWTH",
        "POWDER-BURN", "EXPONENTIAL", "IDEAL-GAS-VT", "TABULATED",
    )
    if kind not in supported_eos:
        log.warning(f"/EOS/{kind} not ported — skipped", block.source)
        return
    mat_id = block.user_id
    # the real /EOS block starts with a title card; the port's compact
    # dialect has none — the numeric-card heuristic separates them
    _, cards = _title_and_data(block)
    if not cards:
        log.error(f"/EOS/{kind}/{mat_id}: missing data card", block.source)
        return
    # Legacy M166 compatibility: /EOS/COMPACTION2 used with COMPACTION card layout (card 2 has 3 tokens)
    if kind == "COMPACTION2" and len(cards) > 1 and len(cards[1].tokens()) == 3:
        kind = "COMPACTION"
    if kind == "POLYNOMIAL":
        f = cards[0].cut("EOS_POLY_1") if block.fixed else []
        if block.fixed and (not f[4] and not f[5]) and len(cards) > 1:
            # real 2-card layout: C0 C1 C2 C3 / C4 C5 E0 Psh RHO_0
            c0, c1, c2, c3 = [_fval(s) for s in f[:4]]
            g = cards[1].cut("EOS_POLY_2")
            c4, c5, e0 = _fval(g[0]), _fval(g[1]), _fval(g[2])
            _warn_ignored(log, f"/EOS/POLYNOMIAL/{mat_id}", block.source,
                          [("Psh", g[3]), ("RHO_0", g[4])])
        else:
            c0, c1, c2, c3, c4, c5 = \
                [_fval(s) for s in f[:6]] if block.fixed \
                else _floats(cards[0], 6)
            e0 = _fval(cards[1].cut("EOS_POLY_2")[0]) if (
                block.fixed and len(cards) > 1) else (
                cards[1].floats()[0] if len(cards) > 1 else 0.0)
        params = {"c0": c0, "c1": c1, "c2": c2, "c3": c3, "c4": c4,
                  "c5": c5, "e0": e0}
    elif kind == "IDEAL-GAS":
        if block.fixed:
            f = cards[0].cut("EOS_IDEAL_GAS")
            gamma, p0 = _fval(f[0], 1.4), _fval(f[1])
            gamma = gamma if gamma else 1.4
            _warn_ignored(log, f"/EOS/IDEAL-GAS/{mat_id}", block.source,
                          [("PSH", f[2]), ("T0", f[3])])
            # RHO_0 is kept: a /MAT/GAS host has no density card of its
            # own and picks it up from here (M37 pack 1, mat_gas.py);
            # for other hosts the material density still rules
            rho0_card = _fval(f[4])
        else:
            gamma, p0 = _floats(cards[0], 2, defaults=[1.4, 0.0])
            rho0_card = 0.0            # the compact card has no RHO_0
        if gamma <= 1.0:
            log.error(f"/EOS/IDEAL-GAS/{mat_id}: gamma must be > 1",
                      block.source)
            return
        params = {"c0": 0.0, "c1": 0.0, "c2": 0.0, "c3": 0.0,
                  "c4": gamma - 1.0, "c5": gamma - 1.0,
                  "e0": p0 / (gamma - 1.0), "gamma": gamma,
                  "rho0_card": rho0_card}
    elif kind == "LINEAR":
        if block.fixed:
            f = cards[0].cut("EOS_LINEAR")
            p0, bulk, psh, rho0_card = _fval(f[0]), _fval(f[1]), _fval(f[2]), _fval(f[3])
        else:
            p0, bulk, psh, rho0_card = _floats(cards[0], 4)
        params = {"c0": p0 - psh, "c1": bulk, "c2": 0.0, "c3": 0.0,
                  "c4": 0.0, "c5": 0.0, "e0": 0.0, "psh": psh,
                  "rho0_card": rho0_card}
    elif kind == "STIFF-GAS":
        if block.fixed:
            f = cards[0].cut("EOS_STIFF_1")
            gamma, p0, psh, p_star = _fval(f[0]), _fval(f[1]), _fval(f[2]), _fval(f[3])
            rho0_card = _fval(f[4]) if len(f) > 4 else 0.0
        else:
            gamma, p0, psh, p_star, rho0_card = _floats(cards[0], 5)
        
        if gamma is None or gamma <= 1.0:
            log.error(f"/EOS/STIFF-GAS/{mat_id}: gamma must be > 1.0", block.source)
            return
            
        params = {"gamma": gamma, "p0": p0, "psh": psh, "p_star": p_star, "rho0_card": rho0_card}
    elif kind == "GRUNEISEN":
        if block.fixed:
            c1 = cards[0].cut("EOS_GRUN_1")
            c, s1, s2, s3 = [_fval(x) for x in c1[:4]]
            c2 = cards[1].cut("EOS_GRUN_2") if len(cards) > 1 else []
            gamma0 = _fval(c2[0]) if len(c2) > 0 else 0.0
            a = _fval(c2[1]) if len(c2) > 1 else 0.0
            e0 = _fval(c2[2]) if len(c2) > 2 else 0.0
            rho0_card = _fval(c2[3]) if len(c2) > 3 else 0.0
        else:
            t1 = cards[0].tokens()
            c = float(t1[0]) if len(t1) > 0 else 0.0
            s1 = float(t1[1]) if len(t1) > 1 else 0.0
            s2 = float(t1[2]) if len(t1) > 2 else 0.0
            s3 = float(t1[3]) if len(t1) > 3 else 0.0
            t2 = cards[1].tokens() if len(cards) > 1 else []
            gamma0 = float(t2[0]) if len(t2) > 0 else 0.0
            a = float(t2[1]) if len(t2) > 1 else 0.0
            e0 = float(t2[2]) if len(t2) > 2 else 0.0
            rho0_card = float(t2[3]) if len(t2) > 3 else 0.0
        params = {"c": c, "s1": s1, "s2": s2, "s3": s3, "gamma0": gamma0, "a": a, "e0": e0, "rho0_card": rho0_card}
    elif kind == "PUFF":
        if block.fixed:
            c1 = cards[0].cut("EOS_PUFF_1")
            c1_val, c2_val, c3_val, gamma0 = [_fval(x) for x in c1[:4]]
            c2 = cards[1].cut("EOS_PUFF_2") if len(cards) > 1 else []
            t1 = _fval(c2[0]) if len(c2) > 0 else 0.0
            t2 = _fval(c2[1]) if len(c2) > 1 else 0.0
            es = _fval(c2[2]) if len(c2) > 2 else 0.0
            c3 = cards[2].cut("EOS_PUFF_3") if len(cards) > 2 else []
            h = _fval(c3[0]) if len(c3) > 0 else 0.0
            e0 = _fval(c3[1]) if len(c3) > 1 else 0.0
            rho0_card = _fval(c3[2]) if len(c3) > 2 else 0.0
        else:
            t1_tok = cards[0].tokens()
            c1_val = float(t1_tok[0]) if len(t1_tok) > 0 else 0.0
            c2_val = float(t1_tok[1]) if len(t1_tok) > 1 else 0.0
            c3_val = float(t1_tok[2]) if len(t1_tok) > 2 else 0.0
            gamma0 = float(t1_tok[3]) if len(t1_tok) > 3 else 0.0
            t2_tok = cards[1].tokens() if len(cards) > 1 else []
            t1 = float(t2_tok[0]) if len(t2_tok) > 0 else 0.0
            t2 = float(t2_tok[1]) if len(t2_tok) > 1 else 0.0
            es = float(t2_tok[2]) if len(t2_tok) > 2 else 0.0
            t3_tok = cards[2].tokens() if len(cards) > 2 else []
            h = float(t3_tok[0]) if len(t3_tok) > 0 else 0.0
            e0 = float(t3_tok[1]) if len(t3_tok) > 1 else 0.0
            rho0_card = float(t3_tok[2]) if len(t3_tok) > 2 else 0.0
        params = {"c1": c1_val, "c2": c2_val, "c3": c3_val, "gamma0": gamma0, "t1": t1, "t2": t2, "es": es, "h": h, "e0": e0, "rho0_card": rho0_card}
    elif kind == "TILLOTSON":
        if block.fixed:
            c1 = cards[0].cut("EOS_TILL_1")
            c1_val, c2_val, a, b = [_fval(x) for x in c1[:4]]
            c2 = cards[1].cut("EOS_TILL_2") if len(cards) > 1 else []
            er = _fval(c2[0]) if len(c2) > 0 else 0.0
            es = _fval(c2[1]) if len(c2) > 1 else 0.0
            vs = _fval(c2[2]) if len(c2) > 2 else 0.0
            e0 = _fval(c2[3]) if len(c2) > 3 else 0.0
            rho0_card = _fval(c2[4]) if len(c2) > 4 else 0.0
            c3 = cards[2].cut("EOS_TILL_3") if len(cards) > 2 else []
            alpha = _fval(c3[0]) if len(c3) > 0 else 0.0
            beta = _fval(c3[1]) if len(c3) > 1 else 0.0
        else:
            t1_tok = cards[0].tokens()
            c1_val = float(t1_tok[0]) if len(t1_tok) > 0 else 0.0
            c2_val = float(t1_tok[1]) if len(t1_tok) > 1 else 0.0
            a = float(t1_tok[2]) if len(t1_tok) > 2 else 0.0
            b = float(t1_tok[3]) if len(t1_tok) > 3 else 0.0
            t2_tok = cards[1].tokens() if len(cards) > 1 else []
            er = float(t2_tok[0]) if len(t2_tok) > 0 else 0.0
            es = float(t2_tok[1]) if len(t2_tok) > 1 else 0.0
            vs = float(t2_tok[2]) if len(t2_tok) > 2 else 0.0
            e0 = float(t2_tok[3]) if len(t2_tok) > 3 else 0.0
            rho0_card = float(t2_tok[4]) if len(t2_tok) > 4 else 0.0
            t3_tok = cards[2].tokens() if len(cards) > 2 else []
            alpha = float(t3_tok[0]) if len(t3_tok) > 0 else 0.0
            beta = float(t3_tok[1]) if len(t3_tok) > 1 else 0.0
        params = {"c1": c1_val, "c2": c2_val, "a": a, "b": b, "er": er, "es": es, "vs": vs, "e0": e0, "rho0_card": rho0_card, "alpha": alpha, "beta": beta}
    elif kind == "MURNAGHAN":
        if block.fixed:
            c1 = cards[0].cut("EOS_MURN_1")
            k0 = _fval(c1[0]) if len(c1) > 0 else 0.0
            k1 = _fval(c1[1]) if len(c1) > 1 else 0.0
            p0 = _fval(c1[2]) if len(c1) > 2 else 0.0
            psh = _fval(c1[3]) if len(c1) > 3 else 0.0
            rho0_card = _fval(c1[4]) if len(c1) > 4 else 0.0
        else:
            t1 = cards[0].tokens()
            k0 = float(t1[0]) if len(t1) > 0 else 0.0
            k1 = float(t1[1]) if len(t1) > 1 else 0.0
            p0 = float(t1[2]) if len(t1) > 2 else 0.0
            psh = float(t1[3]) if len(t1) > 3 else 0.0
            rho0_card = float(t1[4]) if len(t1) > 4 else 0.0
        params = {"k0": k0, "k1": k1, "p0": p0, "psh": psh, "rho0_card": rho0_card}
    elif kind == "OSBORNE":
        if block.fixed:
            c1 = cards[0].cut("EOS_OSBO_1")
            a1, a2, b0, b1, b2 = [_fval(x) for x in c1[:5]]
            c2 = cards[1].cut("EOS_OSBO_2") if len(cards) > 1 else []
            c0 = _fval(c2[0]) if len(c2) > 0 else 0.0
            c1_val = _fval(c2[1]) if len(c2) > 1 else 0.0
            d0 = _fval(c2[2]) if len(c2) > 2 else 0.0
            p0 = _fval(c2[3]) if len(c2) > 3 else 0.0
            c3 = cards[2].cut("F20X5") if len(cards) > 2 else []
            rho0_card = _fval(c3[0]) if len(c3) > 0 else 0.0
        else:
            t1 = cards[0].tokens()
            a1 = float(t1[0]) if len(t1) > 0 else 0.0
            a2 = float(t1[1]) if len(t1) > 1 else 0.0
            b0 = float(t1[2]) if len(t1) > 2 else 0.0
            b1 = float(t1[3]) if len(t1) > 3 else 0.0
            b2 = float(t1[4]) if len(t1) > 4 else 0.0
            t2 = cards[1].tokens() if len(cards) > 1 else []
            c0 = float(t2[0]) if len(t2) > 0 else 0.0
            c1_val = float(t2[1]) if len(t2) > 1 else 0.0
            d0 = float(t2[2]) if len(t2) > 2 else 0.0
            p0 = float(t2[3]) if len(t2) > 3 else 0.0
            t3 = cards[2].tokens() if len(cards) > 2 else []
            rho0_card = float(t3[0]) if len(t3) > 0 else 0.0
        params = {"a1": a1, "a2": a2, "b0": b0, "b1": b1, "b2": b2, "c0": c0, "c1": c1_val, "d0": d0, "p0": p0, "rho0_card": rho0_card}
    elif kind == "LSZK":
        if block.fixed:
            c1 = cards[0].cut("EOS_LSZK_1")
            gamma = _fval(c1[0]) if len(c1) > 0 else 0.0
            p0 = _fval(c1[1]) if len(c1) > 1 else 0.0
            psh = _fval(c1[2]) if len(c1) > 2 else 0.0
            a = _fval(c1[3]) if len(c1) > 3 else 0.0
            b = _fval(c1[4]) if len(c1) > 4 else 0.0
            c2 = cards[1].cut("F20X5") if len(cards) > 1 else []
            rho0_card = _fval(c2[0]) if len(c2) > 0 else 0.0
        else:
            t1 = cards[0].tokens()
            gamma = float(t1[0]) if len(t1) > 0 else 0.0
            p0 = float(t1[1]) if len(t1) > 1 else 0.0
            psh = float(t1[2]) if len(t1) > 2 else 0.0
            a = float(t1[3]) if len(t1) > 3 else 0.0
            b = float(t1[4]) if len(t1) > 4 else 0.0
            t2 = cards[1].tokens() if len(cards) > 1 else []
            rho0_card = float(t2[0]) if len(t2) > 0 else 0.0
        params = {"gamma": gamma, "p0": p0, "psh": psh, "a": a, "b": b, "rho0_card": rho0_card}
    elif kind == "NOBLE-ABEL":
        if block.fixed:
            c1 = cards[0].cut("EOS_NOBLE_1")
            b = _fval(c1[0]) if len(c1) > 0 else 0.0
            gamma = _fval(c1[1]) if len(c1) > 1 else 0.0
            e0 = _fval(c1[2]) if len(c1) > 2 else 0.0
            psh = _fval(c1[3]) if len(c1) > 3 else 0.0
            rho0_card = _fval(c1[4]) if len(c1) > 4 else 0.0
        else:
            t1 = cards[0].tokens()
            b = float(t1[0]) if len(t1) > 0 else 0.0
            gamma = float(t1[1]) if len(t1) > 1 else 0.0
            e0 = float(t1[2]) if len(t1) > 2 else 0.0
            psh = float(t1[3]) if len(t1) > 3 else 0.0
            rho0_card = float(t1[4]) if len(t1) > 4 else 0.0
        params = {"b": b, "gamma": gamma, "e0": e0, "psh": psh, "rho0_card": rho0_card}
    elif kind == "JWL":
        if block.fixed:
            c1 = cards[0].cut("EOS_JWL_1")
            a, b, r1, r2, omega = [_fval(x) for x in c1[:5]]
            c2 = cards[1].cut("EOS_JWL_2") if len(cards) > 1 else []
            e0 = _fval(c2[0]) if len(c2) > 0 else 0.0
            psh = _fval(c2[1]) if len(c2) > 1 else 0.0
            rho0_card = _fval(c2[2]) if len(c2) > 2 else 0.0
        else:
            t1 = cards[0].tokens()
            a = float(t1[0]) if len(t1) > 0 else 0.0
            b = float(t1[1]) if len(t1) > 1 else 0.0
            r1 = float(t1[2]) if len(t1) > 2 else 0.0
            r2 = float(t1[3]) if len(t1) > 3 else 0.0
            omega = float(t1[4]) if len(t1) > 4 else 0.0
            t2 = cards[1].tokens() if len(cards) > 1 else []
            e0 = float(t2[0]) if len(t2) > 0 else 0.0
            psh = float(t2[1]) if len(t2) > 1 else 0.0
            rho0_card = float(t2[2]) if len(t2) > 2 else 0.0
        params = {"a": a, "b": b, "r1": r1, "r2": r2, "omega": omega, "e0": e0, "psh": psh, "rho0_card": rho0_card}
    elif kind == "NASG":
        if block.fixed:
            c1 = cards[0].cut("EOS_NASG_1")
            b_cov, gamma, p_star, q = [_fval(x) for x in c1[:4]]
            c2 = cards[1].cut("EOS_NASG_2") if len(cards) > 1 else []
            psh = _fval(c2[0]) if len(c2) > 0 else 0.0
            p0 = _fval(c2[1]) if len(c2) > 1 else 0.0
            cv = _fval(c2[2]) if len(c2) > 2 else 0.0
            rho0_card = _fval(c2[3]) if len(c2) > 3 else 0.0
        else:
            t1 = cards[0].tokens()
            b_cov = float(t1[0]) if len(t1) > 0 else 0.0
            gamma = float(t1[1]) if len(t1) > 1 else 1.4
            p_star = float(t1[2]) if len(t1) > 2 else 0.0
            q = float(t1[3]) if len(t1) > 3 else 0.0
            t2 = cards[1].tokens() if len(cards) > 1 else []
            psh = float(t2[0]) if len(t2) > 0 else 0.0
            p0 = float(t2[1]) if len(t2) > 1 else 0.0
            cv = float(t2[2]) if len(t2) > 2 else 0.0
            rho0_card = float(t2[3]) if len(t2) > 3 else 0.0

        if gamma <= 1.0:
            log.error(f"/EOS/NASG/{mat_id}: gamma must be > 1.0", block.source)
            return

        rho0 = rho0_card if rho0_card > 0.0 else 1.0
        e0 = (p0 + gamma * p_star) * (1.0 - rho0 * b_cov) / (gamma - 1.0) + rho0 * q
        params = {"b": b_cov, "gamma": gamma, "p_star": p_star, "q": q,
                  "psh": psh, "p0": p0, "cv": cv, "rho0_card": rho0_card, "e0": e0}
    elif kind == "POWDER-BURN":
        if block.fixed:
            c1 = cards[0].cut("EOS_POWDER_1") if len(cards) > 0 else []
            bulk = _fval(c1[0]) if len(c1) > 0 else 0.0
            p0 = _fval(c1[1]) if len(c1) > 1 else 0.0
            psh = _fval(c1[2]) if len(c1) > 2 else 0.0

            c2 = cards[1].cut("EOS_POWDER_2") if len(cards) > 1 else []
            d = _fval(c2[0]) if len(c2) > 0 else 0.0
            eg = _fval(c2[1]) if len(c2) > 1 else 0.0

            c3 = cards[2].cut("EOS_POWDER_3") if len(cards) > 2 else []
            gr = _fval(c3[0]) if len(c3) > 0 else 0.0
            c_val = _fval(c3[1]) if len(c3) > 1 else 0.0
            alpha = _fval(c3[2]) if len(c3) > 2 else 0.0

            c4 = cards[3].cut("EOS_POWDER_4") if len(cards) > 3 else []
            c1_val = _fval(c4[0]) if len(c4) > 0 else 0.0
            c2_val = _fval(c4[1]) if len(c4) > 1 else 0.0

            c5 = cards[4].cut("EOS_POWDER_5") if len(cards) > 4 else []
            func_b = _ival(c5[0]) if len(c5) > 0 else 0
            scale_b = _fval(c5[1], 1.0) if len(c5) > 1 else 1.0
            scale_p = _fval(c5[2], 1.0) if len(c5) > 2 else 1.0

            c6 = cards[5].cut("EOS_POWDER_6") if len(cards) > 5 else []
            func_gam = _ival(c6[0]) if len(c6) > 0 else 0
            scale_gam = _fval(c6[1], 1.0) if len(c6) > 1 else 1.0
            scale_rho = _fval(c6[2], 1.0) if len(c6) > 2 else 1.0
        else:
            t1 = cards[0].tokens() if len(cards) > 0 else []
            bulk = float(t1[0]) if len(t1) > 0 else 0.0
            p0 = float(t1[1]) if len(t1) > 1 else 0.0
            psh = float(t1[2]) if len(t1) > 2 else 0.0

            t2 = cards[1].tokens() if len(cards) > 1 else []
            d = float(t2[0]) if len(t2) > 0 else 0.0
            eg = float(t2[1]) if len(t2) > 1 else 0.0

            t3 = cards[2].tokens() if len(cards) > 2 else []
            gr = float(t3[0]) if len(t3) > 0 else 0.0
            c_val = float(t3[1]) if len(t3) > 1 else 0.0
            alpha = float(t3[2]) if len(t3) > 2 else 0.0

            t4 = cards[3].tokens() if len(cards) > 3 else []
            c1_val = float(t4[0]) if len(t4) > 0 else 0.0
            c2_val = float(t4[1]) if len(t4) > 1 else 0.0

            t5 = cards[4].tokens() if len(cards) > 4 else []
            func_b = int(float(t5[0])) if len(t5) > 0 else 0
            scale_b = float(t5[1]) if len(t5) > 1 else 1.0
            scale_p = float(t5[2]) if len(t5) > 2 else 1.0

            t6 = cards[5].tokens() if len(cards) > 5 else []
            func_gam = int(float(t6[0])) if len(t6) > 0 else 0
            scale_gam = float(t6[1]) if len(t6) > 1 else 1.0
            scale_rho = float(t6[2]) if len(t6) > 2 else 1.0

        params = {
            "bulk": bulk, "p0": p0, "psh": psh, "d": d, "eg": eg,
            "gr": gr, "c": c_val, "alpha": alpha, "c1": c1_val, "c2": c2_val,
            "func_b": func_b, "scale_b": scale_b, "scale_p": scale_p,
            "func_gam": func_gam, "scale_gam": scale_gam, "scale_rho": scale_rho,
        }
    elif kind == "COMPACTION":
        if block.fixed:
            c1 = cards[0].cut("EOS_COMPACT_2") if len(cards) > 0 else []
            c0 = _fval(c1[0]) if len(c1) > 0 else 0.0
            c1_val = _fval(c1[1]) if len(c1) > 1 else 0.0
            c2_val = _fval(c1[2]) if len(c1) > 2 else 0.0
            c3_val = _fval(c1[3]) if len(c1) > 3 else 0.0

            c2 = cards[1].cut("EOS_COMPACT_3") if len(cards) > 1 else []
            mue_min = _fval(c2[0]) if len(c2) > 0 else 0.0
            mue_max = _fval(c2[1]) if len(c2) > 1 else 0.0
            b = _fval(c2[2]) if len(c2) > 2 else 0.0

            c3 = cards[2].cut("EOS_COMPACT_4") if len(cards) > 2 else []
            psh = _fval(c3[0]) if len(c3) > 0 else 0.0
            rho0_card = _fval(c3[1]) if len(c3) > 1 else 0.0
        else:
            t1 = cards[0].tokens() if len(cards) > 0 else []
            c0 = float(t1[0]) if len(t1) > 0 else 0.0
            c1_val = float(t1[1]) if len(t1) > 1 else 0.0
            c2_val = float(t1[2]) if len(t1) > 2 else 0.0
            c3_val = float(t1[3]) if len(t1) > 3 else 0.0

            t2 = cards[1].tokens() if len(cards) > 1 else []
            mue_min = float(t2[0]) if len(t2) > 0 else 0.0
            mue_max = float(t2[1]) if len(t2) > 1 else 0.0
            b = float(t2[2]) if len(t2) > 2 else 0.0

            t3 = cards[2].tokens() if len(cards) > 2 else []
            psh = float(t3[0]) if len(t3) > 0 else 0.0
            rho0_card = float(t3[1]) if len(t3) > 1 else 0.0

        params = {
            "c0": c0, "c1": c1_val, "c2": c2_val, "c3": c3_val,
            "mue_min": mue_min, "mue_max": mue_max, "b": b,
            "psh": psh, "rho0_card": rho0_card,
        }
    elif kind == "EXPONENTIAL":
        if block.fixed:
            c1 = cards[0].cut("EOS_EXPONENTIAL_1") if len(cards) > 0 else []
            p0 = _fval(c1[0]) if len(c1) > 0 else 0.0
            alpha = _fval(c1[1]) if len(c1) > 1 else 0.0
            psh = _fval(c1[2]) if len(c1) > 2 else 0.0
        else:
            t1 = cards[0].tokens() if len(cards) > 0 else []
            p0 = float(t1[0]) if len(t1) > 0 else 0.0
            alpha = float(t1[1]) if len(t1) > 1 else 0.0
            psh = float(t1[2]) if len(t1) > 2 else 0.0

        params = {"p0": p0, "alpha": alpha, "psh": psh}
    elif kind == "IDEAL-GAS-VT":
        if block.fixed:
            c1 = cards[0].cut("EOS_IDEAL_GAS_VT_1") if len(cards) > 0 else []
            r_gas = _fval(c1[0]) if len(c1) > 0 else 0.0
            p0 = _fval(c1[1]) if len(c1) > 1 else 0.0
            psh = _fval(c1[2]) if len(c1) > 2 else 0.0
            t0 = _fval(c1[3]) if len(c1) > 3 else 0.0
            rho0_card = _fval(c1[4]) if len(c1) > 4 else 0.0

            c2 = cards[1].cut("EOS_IDEAL_GAS_VT_2") if len(cards) > 1 else []
            a0 = _fval(c2[0]) if len(c2) > 0 else 0.0
            a1 = _fval(c2[1]) if len(c2) > 1 else 0.0
            a2 = _fval(c2[2]) if len(c2) > 2 else 0.0
            a3 = _fval(c2[3]) if len(c2) > 3 else 0.0
            a4 = _fval(c2[4]) if len(c2) > 4 else 0.0
        else:
            t1 = cards[0].tokens() if len(cards) > 0 else []
            r_gas = float(t1[0]) if len(t1) > 0 else 0.0
            p0 = float(t1[1]) if len(t1) > 1 else 0.0
            psh = float(t1[2]) if len(t1) > 2 else 0.0
            t0 = float(t1[3]) if len(t1) > 3 else 0.0
            rho0_card = float(t1[4]) if len(t1) > 4 else 0.0

            t2 = cards[1].tokens() if len(cards) > 1 else []
            a0 = float(t2[0]) if len(t2) > 0 else 0.0
            a1 = float(t2[1]) if len(t2) > 1 else 0.0
            a2 = float(t2[2]) if len(t2) > 2 else 0.0
            a3 = float(t2[3]) if len(t2) > 3 else 0.0
            a4 = float(t2[4]) if len(t2) > 4 else 0.0

        params = {
            "r_gas": r_gas, "p0": p0, "psh": psh, "t0": t0, "rho0_card": rho0_card,
            "a0": a0, "a1": a1, "a2": a2, "a3": a3, "a4": a4,
        }
    elif kind == "COMPACTION2":
        if block.fixed:
            c1 = cards[0].cut("F20X5") if len(cards) > 0 else []
            p_func = _ival(c1[0]) if len(c1) > 0 else 0
            fscale = _fval(c1[1], 1.0) if len(c1) > 1 and c1[1].strip() else 1.0
            xscale = _fval(c1[2], 1.0) if len(c1) > 2 and c1[2].strip() else 1.0
            iform = _ival(c1[3], 2) if len(c1) > 3 and c1[3].strip() else 2
            c2 = cards[1].cut("F20X5") if len(cards) > 1 else []
            mumin = _fval(c2[0]) if len(c2) > 0 else 0.0
            mumax = _fval(c2[1], 1e20) if len(c2) > 1 and c2[1].strip() else 1e20
            bmin = _fval(c2[2], 1.0) if len(c2) > 2 and c2[2].strip() else 1.0
            bmax = _fval(c2[3], bmin) if len(c2) > 3 and c2[3].strip() else bmin
            c3 = cards[2].cut("F20X5") if len(cards) > 2 else []
            psh = _fval(c3[0]) if len(c3) > 0 else 0.0
            rho0_card = _fval(c3[1]) if len(c3) > 1 else 0.0
        else:
            t1 = cards[0].tokens() if len(cards) > 0 else []
            p_func = int(float(t1[0])) if len(t1) > 0 else 0
            fscale = float(t1[1]) if len(t1) > 1 and float(t1[1]) != 0.0 else 1.0
            xscale = float(t1[2]) if len(t1) > 2 and float(t1[2]) != 0.0 else 1.0
            iform = int(float(t1[3])) if len(t1) > 3 and int(float(t1[3])) != 0 else 2
            t2 = cards[1].tokens() if len(cards) > 1 else []
            mumin = float(t2[0]) if len(t2) > 0 else 0.0
            mumax = float(t2[1]) if len(t2) > 1 and float(t2[1]) != 0.0 else 1e20
            bmin = float(t2[2]) if len(t2) > 2 and float(t2[2]) != 0.0 else 1.0
            bmax = float(t2[3]) if len(t2) > 3 and float(t2[3]) != 0.0 else bmin
            t3 = cards[2].tokens() if len(cards) > 2 else []
            psh = float(t3[0]) if len(t3) > 0 else 0.0
            rho0_card = float(t3[1]) if len(t3) > 1 else 0.0
        params = {
            "p_func": p_func, "fscale": fscale, "xscale": xscale, "iform": iform,
            "mumin": mumin, "mumax": mumax, "bmin": bmin, "bmax": bmax,
            "psh": psh, "rho0_card": rho0_card,
        }
    elif kind == "COMPACTION_TAB":
        if block.fixed:
            c1 = cards[0].cut("F20X5") if len(cards) > 0 else []
            rho_tmd = _fval(c1[0]) if len(c1) > 0 else 0.0
            iplas = _ival(c1[1], 0) if len(c1) > 1 else 0
            c2 = cards[1].cut("F20X5") if len(cards) > 1 else []
            p_func = _ival(c2[0]) if len(c2) > 0 else 0
            pscale = _fval(c2[1], 1.0) if len(c2) > 1 and c2[1].strip() else 1.0
            c3 = cards[2].cut("F20X5") if len(cards) > 2 else []
            c_func = _ival(c3[0]) if len(c3) > 0 else 0
            cscale = _fval(c3[1], 1.0) if len(c3) > 1 and c3[1].strip() else 1.0
            c4 = cards[3].cut("F20X5") if len(cards) > 3 else []
            g_func = _ival(c4[0]) if len(c4) > 0 else 0
            gscale = _fval(c4[1], 1.0) if len(c4) > 1 and c4[1].strip() else 1.0
        else:
            t1 = cards[0].tokens() if len(cards) > 0 else []
            rho_tmd = float(t1[0]) if len(t1) > 0 else 0.0
            iplas = int(float(t1[1])) if len(t1) > 1 else 0
            t2 = cards[1].tokens() if len(cards) > 1 else []
            p_func = int(float(t2[0])) if len(t2) > 0 else 0
            pscale = float(t2[1]) if len(t2) > 1 and float(t2[1]) != 0.0 else 1.0
            t3 = cards[2].tokens() if len(cards) > 2 else []
            c_func = int(float(t3[0])) if len(t3) > 0 else 0
            cscale = float(t3[1]) if len(t3) > 1 and float(t3[1]) != 0.0 else 1.0
            t4 = cards[3].tokens() if len(cards) > 3 else []
            g_func = int(float(t4[0])) if len(t4) > 0 else 0
            gscale = float(t4[1]) if len(t4) > 1 and float(t4[1]) != 0.0 else 1.0
        params = {
            "rho_tmd": rho_tmd, "iplas": iplas,
            "p_func": p_func, "pscale": pscale,
            "c_func": c_func, "cscale": cscale,
            "g_func": g_func, "gscale": gscale,
        }
    elif kind == "TABULATED":
        if block.fixed:
            c1 = cards[0].cut("F20X5") if len(cards) > 0 else []
            a_func = _ival(c1[0]) if len(c1) > 0 else 0
            xscale_a = _fval(c1[1], 1.0) if len(c1) > 1 and c1[1].strip() else 1.0
            fscale_a = _fval(c1[2], 1.0) if len(c1) > 2 and c1[2].strip() else 1.0
            c2 = cards[1].cut("F20X5") if len(cards) > 1 else []
            b_func = _ival(c2[0]) if len(c2) > 0 else 0
            xscale_b = _fval(c2[1], 1.0) if len(c2) > 1 and c2[1].strip() else 1.0
            fscale_b = _fval(c2[2], 1.0) if len(c2) > 2 and c2[2].strip() else 1.0
            c3 = cards[2].cut("F20X5") if len(cards) > 2 else []
            e0 = _fval(c3[0]) if len(c3) > 0 else 0.0
            psh = _fval(c3[1]) if len(c3) > 1 else 0.0
            rho0_card = _fval(c3[2]) if len(c3) > 2 else 0.0
        else:
            t1 = cards[0].tokens() if len(cards) > 0 else []
            a_func = int(float(t1[0])) if len(t1) > 0 else 0
            xscale_a = float(t1[1]) if len(t1) > 1 and float(t1[1]) != 0.0 else 1.0
            fscale_a = float(t1[2]) if len(t1) > 2 and float(t1[2]) != 0.0 else 1.0
            t2 = cards[1].tokens() if len(cards) > 1 else []
            b_func = int(float(t2[0])) if len(t2) > 0 else 0
            xscale_b = float(t2[1]) if len(t2) > 1 and float(t2[1]) != 0.0 else 1.0
            fscale_b = float(t2[2]) if len(t2) > 2 and float(t2[2]) != 0.0 else 1.0
            t3 = cards[2].tokens() if len(cards) > 2 else []
            e0 = float(t3[0]) if len(t3) > 0 else 0.0
            psh = float(t3[1]) if len(t3) > 1 else 0.0
            rho0_card = float(t3[2]) if len(t3) > 2 else 0.0
        params = {
            "func_a": a_func, "xscale_a": xscale_a, "fscale_a": fscale_a,
            "func_b": b_func, "xscale_b": xscale_b, "fscale_b": fscale_b,
            "e0": e0, "psh": psh, "rho0_card": rho0_card,
        }
    elif kind == "SESAME":
        if block.fixed:
            c1 = cards[0].cut("EOS_SESAME_1") if len(cards) > 0 else []
            e0 = _fval(c1[0]) if len(c1) > 0 else 0.0
            rho0_card = _fval(c1[1]) if len(c1) > 1 else 0.0
            filename = cards[1].raw.strip() if len(cards) > 1 else ""
        else:
            t1 = cards[0].tokens() if len(cards) > 0 else []
            e0 = float(t1[0]) if len(t1) > 0 else 0.0
            rho0_card = float(t1[1]) if len(t1) > 1 else 0.0
            filename = cards[1].raw.strip() if len(cards) > 1 else ""
        params = {"e0": e0, "rho0_card": rho0_card, "filename": filename}
    elif kind in ("COMPACT", "IGNITION_GROWTH"):
        if block.fixed:
            c1 = cards[0].cut("EOS_COMPACT_1")
            c1_vals = [_fval(x) for x in c1]
        else:
            t1 = cards[0].tokens()
            c1_vals = [float(x) for x in t1]
        params = {"values": c1_vals}

    model.raw_eos.append((mat_id, EquationOfState(kind=kind, params=params),
                          block.source))




def read_mat_plas_zeril(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/PLAS_ZERIL/mat_ID`` or ``/PLAS_ZERIL/mat_ID`` (M141): Zerilli-Armstrong plasticity modifier."""
    from ...model.entities import MaterialPlasZeril
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/MAT/PLAS_ZERIL/{mat_id}: missing data card", block.source)
        return
    c0, c1, c2, c3, c4 = 0.0, 0.0, 0.0, 0.0, 0.0
    c5, n, fcut = 0.0, 0.0, 0.0
    if block.fixed:
        f1 = cards[0].cut("MAT_PLAS_ZERIL_1")
        c0 = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        c1 = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
        c2 = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
        c3 = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
        c4 = _fval(f1[4], 0.0) if len(f1) > 4 else 0.0
        if len(cards) > 1:
            f2 = cards[1].cut("MAT_PLAS_ZERIL_2")
            c5 = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            n = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            fcut = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
    else:
        t1 = cards[0].tokens()
        c0 = float(t1[0]) if len(t1) > 0 else 0.0
        c1 = float(t1[1]) if len(t1) > 1 else 0.0
        c2 = float(t1[2]) if len(t1) > 2 else 0.0
        c3 = float(t1[3]) if len(t1) > 3 else 0.0
        c4 = float(t1[4]) if len(t1) > 4 else 0.0
        if len(cards) > 1:
            t2 = cards[1].tokens()
            c5 = float(t2[0]) if len(t2) > 0 else 0.0
            n = float(t2[1]) if len(t2) > 1 else 0.0
            fcut = float(t2[2]) if len(t2) > 2 else 0.0
    model.mat_plas_zerils[mat_id] = MaterialPlasZeril(
        mat_id=mat_id, title=title, c0=c0, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, n=n, fcut=fcut
    )




def read_mat_plas_bodne(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/PLAS_BODNE/mat_ID`` or ``/PLAS_BODNE/mat_ID`` (M141): Bodner-Partom viscoplasticity modifier."""
    from ...model.entities import MaterialPlasBodne
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/MAT/PLAS_BODNE/{mat_id}: missing data card", block.source)
        return
    z0, z1, m, n, d0 = 0.0, 0.0, 0.0, 0.0, 0.0
    a1, a2 = 0.0, 0.0
    if block.fixed:
        f1 = cards[0].cut("MAT_PLAS_BODNE_1")
        z0 = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        z1 = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
        m = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
        n = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
        d0 = _fval(f1[4], 0.0) if len(f1) > 4 else 0.0
        if len(cards) > 1:
            f2 = cards[1].cut("MAT_PLAS_BODNE_2")
            a1 = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            a2 = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
    else:
        t1 = cards[0].tokens()
        z0 = float(t1[0]) if len(t1) > 0 else 0.0
        z1 = float(t1[1]) if len(t1) > 1 else 0.0
        m = float(t1[2]) if len(t1) > 2 else 0.0
        n = float(t1[3]) if len(t1) > 3 else 0.0
        d0 = float(t1[4]) if len(t1) > 4 else 0.0
        if len(cards) > 1:
            t2 = cards[1].tokens()
            a1 = float(t2[0]) if len(t2) > 0 else 0.0
            a2 = float(t2[1]) if len(t2) > 1 else 0.0
    model.mat_plas_bodnes[mat_id] = MaterialPlasBodne(
        mat_id=mat_id, title=title, z0=z0, z1=z1, m=m, n=n, d0=d0, a1=a1, a2=a2
    )




def read_mat_visc_prony(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/VISC_PRONY/mat_ID`` or ``/VISC/LPRONY/mat_ID`` (M141): Prony relaxation series."""
    from ...model.entities import MaterialViscProny
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/VISC/PRONY/{mat_id}: missing data card", block.source)
        return
    order, form, flag_visc = 0, 0, 0
    if block.fixed:
        f0 = cards[0].cut("MAT_VISC_PRONY_1")
        order = _ival(f0[0]) if len(f0) > 0 else 0
        form = _ival(f0[1]) if len(f0) > 1 else 0
        flag_visc = _ival(f0[2]) if len(f0) > 2 else 0
    else:
        t0 = cards[0].tokens()
        order = int(float(t0[0])) if len(t0) > 0 else 0
        form = int(float(t0[1])) if len(t0) > 1 else 0
        flag_visc = int(float(t0[2])) if len(t0) > 2 else 0

    gammas, taus = [], []
    for c in cards[1:]:
        if c.is_blank:
            continue
        if block.fixed:
            f = c.cut("MAT_VISC_PRONY_2")
            g = _fval(f[0], 0.0) if len(f) > 0 else 0.0
            t = _fval(f[1], 0.0) if len(f) > 1 else 0.0
        else:
            toks = c.tokens()
            g = float(toks[0]) if len(toks) > 0 else 0.0
            t = float(toks[1]) if len(toks) > 1 else 0.0
        gammas.append(g)
        taus.append(t)
    model.mat_visc_pronys[mat_id] = MaterialViscProny(
        mat_id=mat_id, title=title, order=order, form=form, flag_visc=flag_visc,
        gammas=gammas, taus=taus
    )
    from ...model.entities import MatViscLprony
    model.mat_visc_lpronys[mat_id] = MatViscLprony(
        id=mat_id, m=order, form=form, flag_visc=flag_visc,
        gamai=gammas, taui=taus, title=title,
    )




def read_mat_therm_stress(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/THERM_STRESS/mat_ID`` or ``/THERM_STRESS/mat_ID`` (M141): Thermal stress expansion."""
    from ...model.entities import MaterialThermStress
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/THERM_STRESS/{mat_id}: missing data card", block.source)
        return
    if block.fixed:
        f = cards[0].cut("MAT_THERM_STRESS_1")
        alpha = _fval(f[0], 0.0) if len(f) > 0 else 0.0
        t0 = _fval(f[1], 293.15) if len(f) > 1 and _fval(f[1]) > 0.0 else 293.15
        ay = _fval(f[2], 0.0) if len(f) > 2 else 0.0
        az = _fval(f[3], 0.0) if len(f) > 3 else 0.0
    else:
        t = cards[0].tokens()
        alpha = float(t[0]) if len(t) > 0 else 0.0
        t0 = float(t[1]) if len(t) > 1 else 293.15
        ay = float(t[2]) if len(t) > 2 else 0.0
        az = float(t[3]) if len(t) > 3 else 0.0
    model.mat_therm_stresses[mat_id] = MaterialThermStress(
        mat_id=mat_id, title=title, alpha=alpha, t0=t0, alpha_y=ay, alpha_z=az
    )




def read_mat_spr_seatbelt(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW114/mat_ID`` or ``/MAT/SPR_SEATBELT/mat_ID`` (M161): Seatbelt spring material model."""
    from ...model.entities import MaterialSprSeatbelt
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/MAT/SPR_SEATBELT/{mat_id}: missing data card", block.source)
        return
    rho, lmin = 0.0, 0.0
    k, c = 0.0, 0.0
    fun_l, fun_ul, xscale, fscale = 0, 0, 1.0, 1.0
    e, i, j, fmax, mmax = 0.0, 0.0, 0.0, 0.0, 0.0
    as_, r = 0.0, 0.0
    if block.fixed:
        f1 = cards[0].cut("MAT_LAW114_1")
        rho = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        lmin = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MAT_LAW114_2")
            k = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            c = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("MAT_LAW114_3")
            fun_l = _ival(f3[0]) if len(f3) > 0 else 0
            fun_ul = _ival(f3[1]) if len(f3) > 1 else 0
            xscale = _fval(f3[2], 1.0) if len(f3) > 2 and _fval(f3[2]) > 0.0 else 1.0
            fscale = _fval(f3[3], 1.0) if len(f3) > 3 and _fval(f3[3]) > 0.0 else 1.0
        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("MAT_LAW114_4")
            e = _fval(f4[0], 0.0) if len(f4) > 0 else 0.0
            i = _fval(f4[1], 0.0) if len(f4) > 1 else 0.0
            j = _fval(f4[2], 0.0) if len(f4) > 2 else 0.0
            fmax = _fval(f4[3], 0.0) if len(f4) > 3 else 0.0
            mmax = _fval(f4[4], 0.0) if len(f4) > 4 else 0.0
        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("MAT_LAW114_5")
            as_ = _fval(f5[0], 0.0) if len(f5) > 0 else 0.0
            r = _fval(f5[1], 0.0) if len(f5) > 1 else 0.0
    else:
        t1 = cards[0].tokens()
        rho = float(t1[0]) if len(t1) > 0 else 0.0
        lmin = float(t1[1]) if len(t1) > 1 else 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            k = float(t2[0]) if len(t2) > 0 else 0.0
            c = float(t2[1]) if len(t2) > 1 else 0.0
        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            fun_l = int(float(t3[0])) if len(t3) > 0 else 0
            fun_ul = int(float(t3[1])) if len(t3) > 1 else 0
            xscale = float(t3[2]) if len(t3) > 2 else 1.0
            fscale = float(t3[3]) if len(t3) > 3 else 1.0
        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            e = float(t4[0]) if len(t4) > 0 else 0.0
            i = float(t4[1]) if len(t4) > 1 else 0.0
            j = float(t4[2]) if len(t4) > 2 else 0.0
            fmax = float(t4[3]) if len(t4) > 3 else 0.0
            mmax = float(t4[4]) if len(t4) > 4 else 0.0
        if len(cards) > 4 and not cards[4].is_blank:
            t5 = cards[4].tokens()
            as_ = float(t5[0]) if len(t5) > 0 else 0.0
            r = float(t5[1]) if len(t5) > 1 else 0.0
    mat = MaterialSprSeatbelt(
        id=mat_id, title=title, rho=rho, lmin=lmin, k=k, c=c,
        fun_l=fun_l, fun_ul=fun_ul, xscale=xscale, fscale=fscale,
        e=e, i=i, j=j, fmax=fmax, mmax=mmax, as_=as_, r=r
    )
    model.mat_spr_seatbelts[mat_id] = mat
    model.materials[mat_id] = mat




def read_mat_sh_seatbelt(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW119/mat_ID`` or ``/MAT/SH_SEATBELT/mat_ID`` (M161): 2D shell seatbelt fabric material model."""
    from ...model.entities import MaterialShSeatbelt
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/MAT/SH_SEATBELT/{mat_id}: missing data card", block.source)
        return
    rho, lmin = 0.0, 0.0
    k, c, re = 0.0, 0.0, 0.0
    fun_l, fun_ul, fscale1, fscale2, ireload = 0, 0, 1.0, 1.0, 0
    e22, nu12, g12, fscale22 = 0.0, 0.0, 0.0, 1.0
    ecoat, nucoat, tcoat = 0.0, 0.0, 0.0
    if block.fixed:
        f1 = cards[0].cut("MAT_LAW119_1")
        rho = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        lmin = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MAT_LAW119_2")
            k = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            c = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            re = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("MAT_LAW119_3")
            fun_l = _ival(f3[0]) if len(f3) > 0 else 0
            fun_ul = _ival(f3[1]) if len(f3) > 1 else 0
            fscale1 = _fval(f3[2], 1.0) if len(f3) > 2 and _fval(f3[2]) > 0.0 else 1.0
            fscale2 = _fval(f3[3], 1.0) if len(f3) > 3 and _fval(f3[3]) > 0.0 else 1.0
            ireload = _ival(f3[4]) if len(f3) > 4 else 0
        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("MAT_LAW119_4")
            e22 = _fval(f4[0], 0.0) if len(f4) > 0 else 0.0
            nu12 = _fval(f4[1], 0.0) if len(f4) > 1 else 0.0
            g12 = _fval(f4[2], 0.0) if len(f4) > 2 else 0.0
            fscale22 = _fval(f4[3], 1.0) if len(f4) > 3 and _fval(f4[3]) > 0.0 else 1.0
        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("MAT_LAW119_5")
            ecoat = _fval(f5[0], 0.0) if len(f5) > 0 else 0.0
            nucoat = _fval(f5[1], 0.0) if len(f5) > 1 else 0.0
            tcoat = _fval(f5[2], 0.0) if len(f5) > 2 else 0.0
    else:
        t1 = cards[0].tokens()
        rho = float(t1[0]) if len(t1) > 0 else 0.0
        lmin = float(t1[1]) if len(t1) > 1 else 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            k = float(t2[0]) if len(t2) > 0 else 0.0
            c = float(t2[1]) if len(t2) > 1 else 0.0
            re = float(t2[2]) if len(t2) > 2 else 0.0
        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            fun_l = int(float(t3[0])) if len(t3) > 0 else 0
            fun_ul = int(float(t3[1])) if len(t3) > 1 else 0
            fscale1 = float(t3[2]) if len(t3) > 2 else 1.0
            fscale2 = float(t3[3]) if len(t3) > 3 else 1.0
            ireload = int(float(t3[4])) if len(t3) > 4 else 0
        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            e22 = float(t4[0]) if len(t4) > 0 else 0.0
            nu12 = float(t4[1]) if len(t4) > 1 else 0.0
            g12 = float(t4[2]) if len(t4) > 2 else 0.0
            fscale22 = float(t4[3]) if len(t4) > 3 else 1.0
        if len(cards) > 4 and not cards[4].is_blank:
            t5 = cards[4].tokens()
            ecoat = float(t5[0]) if len(t5) > 0 else 0.0
            nucoat = float(t5[1]) if len(t5) > 1 else 0.0
            tcoat = float(t5[2]) if len(t5) > 2 else 0.0
    mat = MaterialShSeatbelt(
        id=mat_id, title=title, rho=rho, lmin=lmin, k=k, c=c, re=re,
        fun_l=fun_l, fun_ul=fun_ul, fscale1=fscale1, fscale2=fscale2, ireload=ireload,
        e22=e22, nu12=nu12, g12=g12, fscale22=fscale22, ecoat=ecoat, nucoat=nucoat, tcoat=tcoat
    )
    model.mat_sh_seatbelts[mat_id] = mat
    model.materials[mat_id] = mat




def read_mat_tapo(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW120/mat_ID`` or ``/MAT/TAPO/mat_ID`` (M161): Tape/woven fabric material model."""
    from ...model.entities import MaterialTapo
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/MAT/TAPO/{mat_id}: missing data card", block.source)
        return
    rho, refer_rho = 0.0, 0.0
    e, nu = 0.0, 0.0
    iform, itrx, idam = 1, 0, 0
    thick = 0.0
    tab_id, xscale, yscale = 0, 1.0, 1.0
    tau, q, beta, h = 0.0, 0.0, 1.0, 0.0
    af1, af2, ah1, ah2, as_ = 0.0, 0.0, 0.0, 0.0, 0.0
    cc, gam0, gamf = 1e21, 0.0, 0.0
    d1c, d2c, d1f, d2f = 0.0, 0.0, 0.0, 0.0
    d_trx, d_jc, exp_n = 0.0, 0.0, 0.0
    if block.fixed:
        f1 = cards[0].cut("MAT_LAW120_1")
        rho = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        refer_rho = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MAT_LAW120_2")
            e = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            nu = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            iform = _ival(f2[2], 1) if len(f2) > 2 else 1
            itrx = _ival(f2[3], 0) if len(f2) > 3 else 0
            idam = _ival(f2[4], 0) if len(f2) > 4 else 0
            thick = _fval(f2[6], 0.0) if len(f2) > 6 else 0.0
        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("MAT_LAW120_3")
            tab_id = _ival(f3[0], 0) if len(f3) > 0 else 0
            xscale = _fval(f3[1], 1.0) if len(f3) > 1 and _fval(f3[1]) > 0.0 else 1.0
            yscale = _fval(f3[2], 1.0) if len(f3) > 2 and _fval(f3[2]) > 0.0 else 1.0
        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("MAT_LAW120_4")
            tau = _fval(f4[0], 0.0) if len(f4) > 0 else 0.0
            q = _fval(f4[1], 0.0) if len(f4) > 0 else 0.0
            beta = _fval(f4[2], 1.0) if len(f4) > 2 and _fval(f4[2]) > 0.0 else 1.0
            h = _fval(f4[3], 0.0) if len(f4) > 3 else 0.0
        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("MAT_LAW120_5")
            af1 = _fval(f5[0], 0.0) if len(f5) > 0 else 0.0
            af2 = _fval(f5[1], 0.0) if len(f5) > 1 else 0.0
            ah1 = _fval(f5[2], 0.0) if len(f5) > 2 else 0.0
            ah2 = _fval(f5[3], 0.0) if len(f5) > 3 else 0.0
            as_ = _fval(f5[4], 0.0) if len(f5) > 4 else 0.0
        if len(cards) > 5 and not cards[5].is_blank:
            f6 = cards[5].cut("MAT_LAW120_6")
            cc = _fval(f6[0], 1e21) if len(f6) > 0 and _fval(f6[0]) > 0.0 else 1e21
            gam0 = _fval(f6[1], 0.0) if len(f6) > 1 else 0.0
            gamf = _fval(f6[2], 0.0) if len(f6) > 2 else 0.0
        if len(cards) > 6 and not cards[6].is_blank:
            f7 = cards[6].cut("MAT_LAW120_7")
            d1c = _fval(f7[0], 0.0) if len(f7) > 0 else 0.0
            d2c = _fval(f7[1], 0.0) if len(f7) > 1 else 0.0
            d1f = _fval(f7[2], 0.0) if len(f7) > 2 else 0.0
            d2f = _fval(f7[3], 0.0) if len(f7) > 3 else 0.0
        if len(cards) > 7 and not cards[7].is_blank:
            f8 = cards[7].cut("MAT_LAW120_8")
            d_trx = _fval(f8[0], 0.0) if len(f8) > 0 else 0.0
            d_jc = _fval(f8[1], 0.0) if len(f8) > 1 else 0.0
            exp_n = _fval(f8[2], 0.0) if len(f8) > 2 else 0.0
    else:
        t1 = cards[0].tokens()
        rho = float(t1[0]) if len(t1) > 0 else 0.0
        refer_rho = float(t1[1]) if len(t1) > 1 else 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            e = float(t2[0]) if len(t2) > 0 else 0.0
            nu = float(t2[1]) if len(t2) > 1 else 0.0
            iform = int(float(t2[2])) if len(t2) > 2 else 1
            itrx = int(float(t2[3])) if len(t2) > 3 else 0
            idam = int(float(t2[4])) if len(t2) > 4 else 0
            thick = float(t2[5]) if len(t2) > 5 else 0.0
        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            tab_id = int(float(t3[0])) if len(t3) > 0 else 0
            xscale = float(t3[1]) if len(t3) > 1 else 1.0
            yscale = float(t3[2]) if len(t3) > 2 else 1.0
        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            tau = float(t4[0]) if len(t4) > 0 else 0.0
            q = float(t4[1]) if len(t4) > 1 else 0.0
            beta = float(t4[2]) if len(t4) > 2 else 1.0
            h = float(t4[3]) if len(t4) > 3 else 0.0
        if len(cards) > 4 and not cards[4].is_blank:
            t5 = cards[4].tokens()
            af1 = float(t5[0]) if len(t5) > 0 else 0.0
            af2 = float(t5[1]) if len(t5) > 1 else 0.0
            ah1 = float(t5[2]) if len(t5) > 2 else 0.0
            ah2 = float(t5[3]) if len(t5) > 3 else 0.0
            as_ = float(t5[4]) if len(t5) > 4 else 0.0
        if len(cards) > 5 and not cards[5].is_blank:
            t6 = cards[5].tokens()
            cc = float(t6[0]) if len(t6) > 0 else 1e21
            gam0 = float(t6[1]) if len(t6) > 1 else 0.0
            gamf = float(t6[2]) if len(t6) > 2 else 0.0
        if len(cards) > 6 and not cards[6].is_blank:
            t7 = cards[6].tokens()
            d1c = float(t7[0]) if len(t7) > 0 else 0.0
            d2c = float(t7[1]) if len(t7) > 1 else 0.0
            d1f = float(t7[2]) if len(t7) > 2 else 0.0
            d2f = float(t7[3]) if len(t7) > 3 else 0.0
        if len(cards) > 7 and not cards[7].is_blank:
            t8 = cards[7].tokens()
            d_trx = float(t8[0]) if len(t8) > 0 else 0.0
            d_jc = float(t8[1]) if len(t8) > 1 else 0.0
            exp_n = float(t8[2]) if len(t8) > 2 else 0.0
    mat = MaterialTapo(
        id=mat_id, title=title, rho=rho, refer_rho=refer_rho,
        e=e, nu=nu, iform=iform, itrx=itrx, idam=idam, thick=thick,
        tab_id=tab_id, xscale=xscale, yscale=yscale, tau=tau, q=q,
        beta=beta, h=h, af1=af1, af2=af2, ah1=ah1, ah2=ah2, as_=as_,
        cc=cc, gam0=gam0, gamf=gamf, d1c=d1c, d2c=d2c, d1f=d1f,
        d2f=d2f, d_trx=d_trx, d_jc=d_jc, exp_n=exp_n
    )
    model.mat_tapos[mat_id] = mat
    model.materials[mat_id] = mat




def read_mat_plas_rate(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW121/mat_ID`` or ``/MAT/PLAS_RATE/mat_ID`` (M161): Strain-rate dependent elastoplastic material model."""
    from ...model.entities import MaterialPlasRate
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/MAT/PLAS_RATE/{mat_id}: missing data card", block.source)
        return
    rho = 0.0
    e, nu = 0.0, 0.0
    ires, ivisc = 0, 0
    fcut, tdel = 0.0, 0.0
    fct_sig0, xscale_sig0, yscale_sig0 = 0, 1.0, 1.0
    fct_youn, xscale_youn, yscale_youn = 0, 1.0, 1.0
    fct_tang, xscale_tang, tang = 0, 1.0, 0.0
    fct_fail, ifail, xscale_fail, yscale_fail = 0, 0, 1.0, 1.0
    if block.fixed:
        f1 = cards[0].cut("MAT_LAW121_1")
        rho = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MAT_LAW121_2")
            e = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            nu = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            ires = _ival(f2[2], 0) if len(f2) > 2 else 0
            ivisc = _ival(f2[3], 0) if len(f2) > 3 else 0
            fcut = _fval(f2[4], 0.0) if len(f2) > 4 else 0.0
            tdel = _fval(f2[5], 0.0) if len(f2) > 5 else 0.0
        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("MAT_LAW121_3")
            fct_sig0 = _ival(f3[0], 0) if len(f3) > 0 else 0
            xscale_sig0 = _fval(f3[2], 1.0) if len(f3) > 2 and _fval(f3[2]) > 0.0 else 1.0
            yscale_sig0 = _fval(f3[3], 1.0) if len(f3) > 3 and _fval(f3[3]) > 0.0 else 1.0
        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[2].cut("MAT_LAW121_4") if len(cards) == 4 else cards[3].cut("MAT_LAW121_4")
            fct_youn = _ival(f4[0], 0) if len(f4) > 0 else 0
            xscale_youn = _fval(f4[2], 1.0) if len(f4) > 2 and _fval(f4[2]) > 0.0 else 1.0
            yscale_youn = _fval(f4[3], 1.0) if len(f4) > 3 and _fval(f4[3]) > 0.0 else 1.0
        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("MAT_LAW121_5")
            fct_tang = _ival(f5[0], 0) if len(f5) > 0 else 0
            xscale_tang = _fval(f5[2], 1.0) if len(f5) > 2 and _fval(f5[2]) > 0.0 else 1.0
            tang = _fval(f5[3], 0.0) if len(f5) > 3 else 0.0
        if len(cards) > 5 and not cards[5].is_blank:
            f6 = cards[5].cut("MAT_LAW121_6")
            fct_fail = _ival(f6[0], 0) if len(f6) > 0 else 0
            ifail = _ival(f6[1], 0) if len(f6) > 1 else 0
            xscale_fail = _fval(f6[2], 1.0) if len(f6) > 2 and _fval(f6[2]) > 0.0 else 1.0
            yscale_fail = _fval(f6[3], 1.0) if len(f6) > 3 and _fval(f6[3]) > 0.0 else 1.0
    else:
        t1 = cards[0].tokens()
        rho = float(t1[0]) if len(t1) > 0 else 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            e = float(t2[0]) if len(t2) > 0 else 0.0
            nu = float(t2[1]) if len(t2) > 1 else 0.0
            ires = int(float(t2[2])) if len(t2) > 2 else 0
            ivisc = int(float(t2[3])) if len(t2) > 3 else 0
            fcut = float(t2[4]) if len(t2) > 4 else 0.0
            tdel = float(t2[5]) if len(t2) > 5 else 0.0
        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            fct_sig0 = int(float(t3[0])) if len(t3) > 0 else 0
            xscale_sig0 = float(t3[1]) if len(t3) > 1 else 1.0
            yscale_sig0 = float(t3[2]) if len(t3) > 2 else 1.0
        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            fct_youn = int(float(t4[0])) if len(t4) > 0 else 0
            xscale_youn = float(t4[1]) if len(t4) > 1 else 1.0
            yscale_youn = float(t4[2]) if len(t4) > 2 else 1.0
        if len(cards) > 4 and not cards[4].is_blank:
            t5 = cards[4].tokens()
            fct_tang = int(float(t5[0])) if len(t5) > 0 else 0
            xscale_tang = float(t5[1]) if len(t5) > 1 else 1.0
            tang = float(t5[2]) if len(t5) > 2 else 0.0
        if len(cards) > 5 and not cards[5].is_blank:
            t6 = cards[5].tokens()
            fct_fail = int(float(t6[0])) if len(t6) > 0 else 0
            ifail = int(float(t6[1])) if len(t6) > 1 else 0
            xscale_fail = float(t6[2]) if len(t6) > 2 else 1.0
            yscale_fail = float(t6[3]) if len(t6) > 3 else 1.0
    mat = MaterialPlasRate(
        id=mat_id, title=title, rho=rho, e=e, nu=nu, ires=ires, ivisc=ivisc,
        fcut=fcut, tdel=tdel, fct_sig0=fct_sig0, xscale_sig0=xscale_sig0, yscale_sig0=yscale_sig0,
        fct_youn=fct_youn, xscale_youn=xscale_youn, yscale_youn=yscale_youn,
        fct_tang=fct_tang, xscale_tang=xscale_tang, tang=tang,
        fct_fail=fct_fail, ifail=ifail, xscale_fail=xscale_fail, yscale_fail=yscale_fail
    )
    model.mat_plas_rates[mat_id] = mat
    model.materials[mat_id] = mat




def read_mat_cdpm2(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW124/mat_ID`` or ``/MAT/CDPM2/mat_ID`` (M161): Concrete damage plasticity model 2."""
    from ...model.entities import MaterialCdpm2
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/MAT/CDPM2/{mat_id}: missing data card", block.source)
        return
    rho = 0.0
    e, nu = 0.0, 0.0
    irate = 0
    fcut = 0.0
    idel = 1
    ecc, qh0, ft, fc, hp = 0.0, 0.0, 0.0, 0.0, 0.0
    ah, bh, ch, dh = 0.0, 0.0, 0.0, 0.0
    as_, bs, df = 0.0, 0.0, 0.0
    dflag, dtype, ireg = 0, 0, 0
    wf, wf1, ft1, efc = 0.0, 0.0, 0.0, 0.0
    if block.fixed:
        f1 = cards[0].cut("MAT_LAW124_1")
        rho = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MAT_LAW124_2")
            if len(f2) >= 7:
                e = _fval(f2[0], 0.0)
                nu = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
                idel = _ival(f2[3], 1) if len(f2) > 3 and f2[3].strip() else 1
                irate = _ival(f2[5], 0) if len(f2) > 5 else 0
                fcut = _fval(f2[6], 0.0) if len(f2) > 6 else 0.0
            elif len(f2) >= 5:
                e = _fval(f2[0], 0.0)
                nu = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
                irate = _ival(f2[3], 0) if len(f2) > 3 else 0
                fcut = _fval(f2[4], 0.0) if len(f2) > 4 else 0.0
            else:
                e = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
                nu = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("MAT_LAW124_3")
            ecc = _fval(f3[0], 0.0) if len(f3) > 0 else 0.0
            qh0 = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
            ft = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
            fc = _fval(f3[3], 0.0) if len(f3) > 3 else 0.0
            hp = _fval(f3[4], 0.0) if len(f3) > 4 else 0.0
        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("MAT_LAW124_4")
            ah = _fval(f4[0], 0.0) if len(f4) > 0 else 0.0
            bh = _fval(f4[1], 0.0) if len(f4) > 1 else 0.0
            ch = _fval(f4[2], 0.0) if len(f4) > 2 else 0.0
            dh = _fval(f4[3], 0.0) if len(f4) > 3 else 0.0
        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("MAT_LAW124_5")
            as_ = _fval(f5[0], 0.0) if len(f5) > 0 else 0.0
            bs = _fval(f5[1], 0.0) if len(f5) > 1 else 0.0
            df = _fval(f5[2], 0.0) if len(f5) > 2 else 0.0
            dflag = _ival(f5[4], 0) if len(f5) > 4 else 0
            dtype = _ival(f5[5], 0) if len(f5) > 5 else 0
            ireg = _ival(f5[6], 0) if len(f5) > 6 else 0
        if len(cards) > 5 and not cards[5].is_blank:
            f6 = cards[5].cut("MAT_LAW124_6")
            wf = _fval(f6[0], 0.0) if len(f6) > 0 else 0.0
            wf1 = _fval(f6[1], 0.0) if len(f6) > 1 else 0.0
            ft1 = _fval(f6[2], 0.0) if len(f6) > 2 else 0.0
            efc = _fval(f6[3], 0.0) if len(f6) > 3 else 0.0
    else:
        t1 = cards[0].tokens()
        rho = float(t1[0]) if len(t1) > 0 else 0.0
        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            e = float(t2[0]) if len(t2) > 0 else 0.0
            nu = float(t2[1]) if len(t2) > 1 else 0.0
            if len(t2) >= 5:
                idel = int(float(t2[2]))
                irate = int(float(t2[3]))
                fcut = float(t2[4])
            elif len(t2) >= 4:
                irate = int(float(t2[2]))
                fcut = float(t2[3])
            elif len(t2) >= 3:
                irate = int(float(t2[2]))
        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            ecc = float(t3[0]) if len(t3) > 0 else 0.0
            qh0 = float(t3[1]) if len(t3) > 1 else 0.0
            ft = float(t3[2]) if len(t3) > 2 else 0.0
            fc = float(t3[3]) if len(t3) > 3 else 0.0
            hp = float(t3[4]) if len(t3) > 4 else 0.0
        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            ah = float(t4[0]) if len(t4) > 0 else 0.0
            bh = float(t4[1]) if len(t4) > 1 else 0.0
            ch = float(t4[2]) if len(t4) > 2 else 0.0
            dh = float(t4[3]) if len(t4) > 3 else 0.0
        if len(cards) > 4 and not cards[4].is_blank:
            t5 = cards[4].tokens()
            as_ = float(t5[0]) if len(t5) > 0 else 0.0
            bs = float(t5[1]) if len(t5) > 1 else 0.0
            df = float(t5[2]) if len(t5) > 2 else 0.0
            dflag = int(float(t5[3])) if len(t5) > 3 else 0
            dtype = int(float(t5[4])) if len(t5) > 4 else 0
            ireg = int(float(t5[5])) if len(t5) > 5 else 0
        if len(cards) > 5 and not cards[5].is_blank:
            t6 = cards[5].tokens()
            wf = float(t6[0]) if len(t6) > 0 else 0.0
            wf1 = float(t6[1]) if len(t6) > 1 else 0.0
            ft1 = float(t6[2]) if len(t6) > 2 else 0.0
            efc = float(t6[3]) if len(t6) > 3 else 0.0
    mat = MaterialCdpm2(
        id=mat_id, title=title, rho=rho, e=e, nu=nu, irate=irate, fcut=fcut,
        ecc=ecc, qh0=qh0, ft=ft, fc=fc, hp=hp, ah=ah, bh=bh, ch=ch, dh=dh,
        as_=as_, bs=bs, df=df, dflag=dflag, dtype=dtype, ireg=ireg,
        wf=wf, wf1=wf1, ft1=ft1, efc=efc
    )
    model.mat_cdpm2s[mat_id] = mat
    from ...model.entities import Material
    from ..mat_reader import GenericMaterialRecord
    mat124 = Material(
        id=mat_id, law=124, rho0=rho, title=title,
        params={
            "E": e if e > 0.0 else 1.0, "nu": nu if 0.0 <= nu < 0.5 else 0.2,
            "MAT_E": e, "MAT_NU": nu, "IRATE": irate, "FCUT": fcut, "IDEL": idel,
            "MAT_ECC": ecc, "MAT_QH0": qh0, "MAT_FT": ft, "MAT_FC": fc, "MAT_HP": hp,
            "MAT_AH": ah, "MAT_BH": bh, "MAT_CH": ch, "MAT_DH": dh,
            "MAT_AS": as_, "MAT_BS": bs, "MAT_DF": df, "DFLAG": dflag, "DTYPE": dtype, "IREG": ireg,
            "MAT_WF": wf, "MAT_WF1": wf1, "MAT_FT1": ft1, "MAT_EFC": efc,
            "e": e, "nu": nu, "irate": irate, "fcut": fcut, "idel": idel,
            "ecc": ecc, "qh0": qh0, "ft": ft, "fc": fc, "hp": hp,
            "ah": ah, "bh": bh, "ch": ch, "dh": dh,
            "as": as_, "bs": bs, "df": df, "dflag": dflag, "dtype": dtype, "ireg": ireg,
            "wf": wf, "wf1": wf1, "ft1": ft1, "efc": efc,
        }
    )
    mat124.record = GenericMaterialRecord(
        law_name="CDPM2", law_number=124, id=mat_id, title=title,
        params=mat124.params, density=rho, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat124




def read_mat_conc(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW24`` or ``/MAT/CONC`` (M170): Concrete material model."""
    from ...model.entities import MaterialConc, Material
    mat_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MAT/LAW24/{mat_id}: missing data card", block.source)
        return

    rho0, refer_rho = 0.0, 0.0
    e_c, nu = 0.0, 0.0
    f_c, ft_on_fc, fb_on_fc, f2_on_fc, s0_on_fc = 0.0, 0.0, 0.0, 0.0, 0.0
    h_t, d_sup, eps_max = 0.0, 0.0, 0.0
    k_y, r_t, r_c, h_bp = 0.0, 0.0, 0.0, 0.0
    alpha_y, alpha_f, v_max = 0.0, 0.0, 0.0
    f_k, f0, h_v0 = 0.0, 0.0, 0.0
    e2, ssig, setan = 0.0, 0.0, 0.0
    alpha1, alpha2, alpha3 = 0.0, 0.0, 0.0

    if block.fixed:
        f1 = cards[0].cut("MAT_CONC_1")
        rho0 = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        refer_rho = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MAT_CONC_2")
            e_c = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            nu = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("MAT_CONC_3")
            f_c = _fval(f3[0], 0.0) if len(f3) > 0 else 0.0
            ft_on_fc = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
            fb_on_fc = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
            f2_on_fc = _fval(f3[3], 0.0) if len(f3) > 3 else 0.0
            s0_on_fc = _fval(f3[4], 0.0) if len(f3) > 4 else 0.0

        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("MAT_CONC_4")
            h_t = _fval(f4[0], 0.0) if len(f4) > 0 else 0.0
            d_sup = _fval(f4[1], 0.0) if len(f4) > 1 else 0.0
            eps_max = _fval(f4[2], 0.0) if len(f4) > 2 else 0.0

        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("MAT_CONC_5")
            k_y = _fval(f5[0], 0.0) if len(f5) > 0 else 0.0
            r_t = _fval(f5[1], 0.0) if len(f5) > 1 else 0.0
            r_c = _fval(f5[2], 0.0) if len(f5) > 2 else 0.0
            h_bp = _fval(f5[3], 0.0) if len(f5) > 3 else 0.0

        if len(cards) > 5 and not cards[5].is_blank:
            f6 = cards[5].cut("MAT_CONC_6")
            alpha_y = _fval(f6[0], 0.0) if len(f6) > 0 else 0.0
            alpha_f = _fval(f6[1], 0.0) if len(f6) > 1 else 0.0
            v_max = _fval(f6[2], 0.0) if len(f6) > 2 else 0.0

        if len(cards) > 6 and not cards[6].is_blank:
            f7 = cards[6].cut("MAT_CONC_7")
            f_k = _fval(f7[0], 0.0) if len(f7) > 0 else 0.0
            f0 = _fval(f7[1], 0.0) if len(f7) > 1 else 0.0
            h_v0 = _fval(f7[2], 0.0) if len(f7) > 2 else 0.0

        if len(cards) > 7 and not cards[7].is_blank:
            f8 = cards[7].cut("MAT_CONC_8")
            e2 = _fval(f8[0], 0.0) if len(f8) > 0 else 0.0
            ssig = _fval(f8[1], 0.0) if len(f8) > 1 else 0.0
            setan = _fval(f8[2], 0.0) if len(f8) > 2 else 0.0

        if len(cards) > 8 and not cards[8].is_blank:
            f9 = cards[8].cut("MAT_CONC_9")
            alpha1 = _fval(f9[0], 0.0) if len(f9) > 0 else 0.0
            alpha2 = _fval(f9[1], 0.0) if len(f9) > 1 else 0.0
            alpha3 = _fval(f9[2], 0.0) if len(f9) > 2 else 0.0
    else:
        t1 = cards[0].tokens()
        rho0 = float(t1[0]) if len(t1) > 0 else 0.0
        refer_rho = float(t1[1]) if len(t1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            e_c = float(t2[0]) if len(t2) > 0 else 0.0
            nu = float(t2[1]) if len(t2) > 1 else 0.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            f_c = float(t3[0]) if len(t3) > 0 else 0.0
            ft_on_fc = float(t3[1]) if len(t3) > 1 else 0.0
            fb_on_fc = float(t3[2]) if len(t3) > 2 else 0.0
            f2_on_fc = float(t3[3]) if len(t3) > 3 else 0.0
            s0_on_fc = float(t3[4]) if len(t3) > 4 else 0.0

        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            h_t = float(t4[0]) if len(t4) > 0 else 0.0
            d_sup = float(t4[1]) if len(t4) > 1 else 0.0
            eps_max = float(t4[2]) if len(t4) > 2 else 0.0

        if len(cards) > 4 and not cards[4].is_blank:
            t5 = cards[4].tokens()
            k_y = float(t5[0]) if len(t5) > 0 else 0.0
            r_t = float(t5[1]) if len(t5) > 1 else 0.0
            r_c = float(t5[2]) if len(t5) > 2 else 0.0
            h_bp = float(t5[3]) if len(t5) > 3 else 0.0

        if len(cards) > 5 and not cards[5].is_blank:
            t6 = cards[5].tokens()
            alpha_y = float(t6[0]) if len(t6) > 0 else 0.0
            alpha_f = float(t6[1]) if len(t6) > 1 else 0.0
            v_max = float(t6[2]) if len(t6) > 2 else 0.0

        if len(cards) > 6 and not cards[6].is_blank:
            t7 = cards[6].tokens()
            f_k = float(t7[0]) if len(t7) > 0 else 0.0
            f0 = float(t7[1]) if len(t7) > 1 else 0.0
            h_v0 = float(t7[2]) if len(t7) > 2 else 0.0

        if len(cards) > 7 and not cards[7].is_blank:
            t8 = cards[7].tokens()
            e2 = float(t8[0]) if len(t8) > 0 else 0.0
            ssig = float(t8[1]) if len(t8) > 1 else 0.0
            setan = float(t8[2]) if len(t8) > 2 else 0.0

        if len(cards) > 8 and not cards[8].is_blank:
            t9 = cards[8].tokens()
            alpha1 = float(t9[0]) if len(t9) > 0 else 0.0
            alpha2 = float(t9[1]) if len(t9) > 1 else 0.0
            alpha3 = float(t9[2]) if len(t9) > 2 else 0.0

    mc = MaterialConc(
        id=mat_id, title=title, rho0=rho0, refer_rho=refer_rho,
        e_c=e_c, nu=nu, f_c=f_c, ft_on_fc=ft_on_fc, fb_on_fc=fb_on_fc,
        f2_on_fc=f2_on_fc, s0_on_fc=s0_on_fc, h_t=h_t, d_sup=d_sup,
        eps_max=eps_max, k_y=k_y, r_t=r_t, r_c=r_c, h_bp=h_bp,
        alpha_y=alpha_y, alpha_f=alpha_f, v_max=v_max, f_k=f_k, f0=f0,
        h_v0=h_v0, e2=e2, ssig=ssig, setan=setan,
        alpha1=alpha1, alpha2=alpha2, alpha3=alpha3,
    )
    model.mat_concs[mat_id] = mc
    model.materials[mat_id] = Material(
        id=mat_id, law=24, rho0=rho0, title=title,
        params={
            "E": e_c, "nu": nu, "f_c": f_c, "ft_on_fc": ft_on_fc,
            "fb_on_fc": fb_on_fc, "f2_on_fc": f2_on_fc, "s0_on_fc": s0_on_fc,
            "h_t": h_t, "d_sup": d_sup, "eps_max": eps_max, "k_y": k_y,
            "r_t": r_t, "r_c": r_c, "h_bp": h_bp, "alpha_y": alpha_y,
            "alpha_f": alpha_f, "v_max": v_max, "f_k": f_k, "f0": f0,
            "h_v0": h_v0, "e2": e2, "ssig": ssig, "setan": setan,
            "alpha1": alpha1, "alpha2": alpha2, "alpha3": alpha3,
        }
    )




def read_mat_barlat(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW87`` or ``/MAT/BARLAT`` (M170): Barlat 2000 anisotropic plasticity model."""
    from ...model.entities import MaterialBarlat, Material
    mat_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MAT/LAW87/{mat_id}: missing data card", block.source)
        return

    rho0, refer_rho = 0.0, 0.0
    e, nu = 0.0, 0.0
    iflag, vflag = 0, 0
    strain1, exp1 = 0.0, 0.0
    ifit = 0
    alphas = [1.0] * 8
    sigma_00, sigma_45, sigma_90, sigma_b = 0.0, 0.0, 0.0, 0.0
    r_00, r_45, r_90, r_b = 0.0, 0.0, 0.0, 0.0
    a_exp = 6
    alpha_vol, n_hard = 1.0, 0.0
    fcut = 0.0
    fsmooth = 0
    a_swift, eps0, q_voce, beta, k0 = 0.0, 0.0, 0.0, 0.0, 0.0

    if block.fixed:
        f1 = cards[0].cut("MAT_BARLAT_1")
        rho0 = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        refer_rho = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MAT_BARLAT_2")
            e = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            nu = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            iflag = _ival(f2[2]) if len(f2) > 2 else 0
            vflag = _ival(f2[3]) if len(f2) > 3 else 0
            strain1 = _fval(f2[4], 0.0) if len(f2) > 4 else 0.0
            exp1 = _fval(f2[5], 0.0) if len(f2) > 5 else 0.0

        card2_raw = cards[2].raw if len(cards) > 2 else ""
        if len(card2_raw) >= 90 and card2_raw[80:90].strip():
            ifit = _ival(card2_raw[80:90])
        else:
            ifit = 0

        card_idx = 2
        if ifit == 0:
            if card_idx < len(cards) and not cards[card_idx].is_blank:
                fa1 = cards[card_idx].cut("MAT_BARLAT_ALPHA_1")
                alphas[0] = _fval(fa1[0], 1.0) if len(fa1) > 0 else 1.0
                alphas[1] = _fval(fa1[1], 1.0) if len(fa1) > 1 else 1.0
                alphas[2] = _fval(fa1[2], 1.0) if len(fa1) > 2 else 1.0
                alphas[3] = _fval(fa1[3], 1.0) if len(fa1) > 3 else 1.0
                card_idx += 1
            if card_idx < len(cards) and not cards[card_idx].is_blank:
                fa2 = cards[card_idx].cut("MAT_BARLAT_ALPHA_2")
                alphas[4] = _fval(fa2[0], 1.0) if len(fa2) > 0 else 1.0
                alphas[5] = _fval(fa2[1], 1.0) if len(fa2) > 1 else 1.0
                alphas[6] = _fval(fa2[2], 1.0) if len(fa2) > 2 else 1.0
                alphas[7] = _fval(fa2[3], 1.0) if len(fa2) > 3 else 1.0
                card_idx += 1
        else:
            if card_idx < len(cards) and not cards[card_idx].is_blank:
                ff1 = cards[card_idx].cut("MAT_BARLAT_FIT_1")
                sigma_00 = _fval(ff1[0], 0.0) if len(ff1) > 0 else 0.0
                sigma_45 = _fval(ff1[1], 0.0) if len(ff1) > 1 else 0.0
                sigma_90 = _fval(ff1[2], 0.0) if len(ff1) > 2 else 0.0
                sigma_b = _fval(ff1[3], 0.0) if len(ff1) > 3 else 0.0
                card_idx += 1
            if card_idx < len(cards) and not cards[card_idx].is_blank:
                ff2 = cards[card_idx].cut("MAT_BARLAT_FIT_2")
                r_00 = _fval(ff2[0], 0.0) if len(ff2) > 0 else 0.0
                r_45 = _fval(ff2[1], 0.0) if len(ff2) > 1 else 0.0
                r_90 = _fval(ff2[2], 0.0) if len(ff2) > 2 else 0.0
                r_b = _fval(ff2[3], 0.0) if len(ff2) > 3 else 0.0
                card_idx += 1

        if card_idx < len(cards) and not cards[card_idx].is_blank:
            card_idx += 1

        if card_idx < len(cards) and not cards[card_idx].is_blank:
            fh1 = cards[card_idx].cut("MAT_BARLAT_HARD_1")
            a_exp = _ival(fh1[0], 6) if len(fh1) > 0 else 6
            alpha_vol = _fval(fh1[1], 1.0) if len(fh1) > 1 else 1.0
            n_hard = _fval(fh1[2], 0.0) if len(fh1) > 2 else 0.0
            fcut = _fval(fh1[3], 0.0) if len(fh1) > 3 else 0.0
            fsmooth = _ival(fh1[4]) if len(fh1) > 4 else 0
            card_idx += 1

        if card_idx < len(cards) and not cards[card_idx].is_blank:
            fsw = cards[card_idx].cut("MAT_BARLAT_SWIFT")
            a_swift = _fval(fsw[0], 0.0) if len(fsw) > 0 else 0.0
            eps0 = _fval(fsw[1], 0.0) if len(fsw) > 1 else 0.0
            q_voce = _fval(fsw[2], 0.0) if len(fsw) > 2 else 0.0
            beta = _fval(fsw[3], 0.0) if len(fsw) > 3 else 0.0
            k0 = _fval(fsw[4], 0.0) if len(fsw) > 4 else 0.0
    else:
        t1 = cards[0].tokens()
        rho0 = float(t1[0]) if len(t1) > 0 else 0.0
        refer_rho = float(t1[1]) if len(t1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            e = float(t2[0]) if len(t2) > 0 else 0.0
            nu = float(t2[1]) if len(t2) > 1 else 0.0
            iflag = int(float(t2[2])) if len(t2) > 2 else 0
            vflag = int(float(t2[3])) if len(t2) > 3 else 0
            strain1 = float(t2[4]) if len(t2) > 4 else 0.0
            exp1 = float(t2[5]) if len(t2) > 5 else 0.0

        card_idx = 2
        if len(cards) > 2 and len(cards[2].tokens()) == 5:
            ifit = 1
            t_fit1 = cards[2].tokens()
            sigma_00 = float(t_fit1[0])
            sigma_45 = float(t_fit1[1])
            sigma_90 = float(t_fit1[2])
            sigma_b = float(t_fit1[3])
            card_idx = 3
            if card_idx < len(cards):
                t_fit2 = cards[card_idx].tokens()
                r_00 = float(t_fit2[0]) if len(t_fit2) > 0 else 0.0
                r_45 = float(t_fit2[1]) if len(t_fit2) > 1 else 0.0
                r_90 = float(t_fit2[2]) if len(t_fit2) > 2 else 0.0
                r_b = float(t_fit2[3]) if len(t_fit2) > 3 else 0.0
                card_idx += 1
        else:
            ifit = 0
            if card_idx < len(cards):
                ta1 = cards[card_idx].tokens()
                for i in range(min(4, len(ta1))):
                    alphas[i] = float(ta1[i])
                card_idx += 1
            if card_idx < len(cards):
                ta2 = cards[card_idx].tokens()
                for i in range(min(4, len(ta2))):
                    alphas[4 + i] = float(ta2[i])
                card_idx += 1

        if card_idx < len(cards) and len(cards[card_idx].tokens()) <= 1:
            card_idx += 1

        if card_idx < len(cards):
            th1 = cards[card_idx].tokens()
            a_exp = int(float(th1[0])) if len(th1) > 0 else 6
            alpha_vol = float(th1[1]) if len(th1) > 1 else 1.0
            n_hard = float(th1[2]) if len(th1) > 2 else 0.0
            fcut = float(th1[3]) if len(th1) > 3 else 0.0
            fsmooth = int(float(th1[4])) if len(th1) > 4 else 0
            card_idx += 1

        if card_idx < len(cards):
            tsw = cards[card_idx].tokens()
            a_swift = float(tsw[0]) if len(tsw) > 0 else 0.0
            eps0 = float(tsw[1]) if len(tsw) > 1 else 0.0
            q_voce = float(tsw[2]) if len(tsw) > 2 else 0.0
            beta = float(tsw[3]) if len(tsw) > 3 else 0.0
            k0 = float(tsw[4]) if len(tsw) > 4 else 0.0

    mb = MaterialBarlat(
        id=mat_id, title=title, rho0=rho0, refer_rho=refer_rho,
        e=e, nu=nu, iflag=iflag, vflag=vflag, strain1=strain1, exp1=exp1,
        ifit=ifit, alphas=alphas, sigma_00=sigma_00, sigma_45=sigma_45,
        sigma_90=sigma_90, sigma_b=sigma_b, r_00=r_00, r_45=r_45,
        r_90=r_90, r_b=r_b, a_exp=a_exp, alpha_vol=alpha_vol, n_hard=n_hard,
        fcut=fcut, fsmooth=fsmooth, a_swift=a_swift, eps0=eps0,
        q_voce=q_voce, beta=beta, k0=k0,
    )
    model.mat_barlats[mat_id] = mb
    model.materials[mat_id] = Material(
        id=mat_id, law=87, rho0=rho0, title=title,
        params={
            "E": e, "nu": nu, "iflag": iflag, "vflag": vflag,
            "strain1": strain1, "exp1": exp1, "ifit": ifit,
            "alphas": alphas, "sigma_00": sigma_00, "sigma_45": sigma_45,
            "sigma_90": sigma_90, "sigma_b": sigma_b, "r_00": r_00,
            "r_45": r_45, "r_90": r_90, "r_b": r_b, "a_exp": a_exp,
            "alpha_vol": alpha_vol, "n_hard": n_hard, "fcut": fcut,
            "fsmooth": fsmooth, "a_swift": a_swift, "eps0": eps0,
            "q_voce": q_voce, "beta": beta, "k0": k0,
        }
    )




def read_mat_law83(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW83`` or ``/MAT/SPR_JOU`` (M170): Non-linear spring/joint material model."""
    from ...model.entities import MaterialLaw83, Material
    mat_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MAT/LAW83/{mat_id}: missing data card", block.source)
        return

    rho0, refer_rho = 0.0, 0.0
    e, imass = 0.0, 0
    fun_a1 = 0
    fscale11, fscale22 = 1.0, 1.0
    alpha, beta = 0.0, 0.0
    rn, rs = 0.0, 0.0
    fsmooth = 0
    fcut = 0.0
    fun_a2, fun_a3 = 0, 0
    fscale33 = 1.0

    if block.fixed:
        f1 = cards[0].cut("MAT_LAW83_1")
        rho0 = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        refer_rho = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MAT_LAW83_2")
            e = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            imass = _ival(f2[2]) if len(f2) > 2 else 0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("MAT_LAW83_3")
            fun_a1 = _ival(f3[0]) if len(f3) > 0 else 0
            fscale11 = _fval(f3[2], 1.0) if len(f3) > 2 else 1.0
            fscale22 = _fval(f3[3], 1.0) if len(f3) > 3 else 1.0
            alpha = _fval(f3[4], 0.0) if len(f3) > 4 else 0.0
            beta = _fval(f3[5], 0.0) if len(f3) > 5 else 0.0

        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("MAT_LAW83_4")
            rn = _fval(f4[0], 0.0) if len(f4) > 0 else 0.0
            rs = _fval(f4[1], 0.0) if len(f4) > 1 else 0.0
            fsmooth = _ival(f4[2]) if len(f4) > 2 else 0
            fcut = _fval(f4[3], 0.0) if len(f4) > 3 else 0.0

        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("MAT_LAW83_5")
            fun_a2 = _ival(f5[0]) if len(f5) > 0 else 0
            fun_a3 = _ival(f5[1]) if len(f5) > 1 else 0
            fscale33 = _fval(f5[2], 1.0) if len(f5) > 2 else 1.0
    else:
        t1 = cards[0].tokens()
        rho0 = float(t1[0]) if len(t1) > 0 else 0.0
        refer_rho = float(t1[1]) if len(t1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            e = float(t2[0]) if len(t2) > 0 else 0.0
            imass = int(float(t2[1])) if len(t2) > 1 else 0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            fun_a1 = int(float(t3[0])) if len(t3) > 0 else 0
            fscale11 = float(t3[1]) if len(t3) > 1 else 1.0
            fscale22 = float(t3[2]) if len(t3) > 2 else 1.0
            alpha = float(t3[3]) if len(t3) > 3 else 0.0
            beta = float(t3[4]) if len(t3) > 4 else 0.0

        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            rn = float(t4[0]) if len(t4) > 0 else 0.0
            rs = float(t4[1]) if len(t4) > 1 else 0.0
            fsmooth = int(float(t4[2])) if len(t4) > 2 else 0
            fcut = float(t4[3]) if len(t4) > 3 else 0.0

        if len(cards) > 4 and not cards[4].is_blank:
            t5 = cards[4].tokens()
            fun_a2 = int(float(t5[0])) if len(t5) > 0 else 0
            fun_a3 = int(float(t5[1])) if len(t5) > 1 else 0
            fscale33 = float(t5[2]) if len(t5) > 2 else 1.0

    m83 = MaterialLaw83(
        id=mat_id, title=title, rho0=rho0, refer_rho=refer_rho,
        e=e, imass=imass, fun_a1=fun_a1, fscale11=fscale11,
        fscale22=fscale22, alpha=alpha, beta=beta, rn=rn, rs=rs,
        fsmooth=fsmooth, fcut=fcut, fun_a2=fun_a2, fun_a3=fun_a3,
        fscale33=fscale33,
    )
    model.mat_law83s[mat_id] = m83
    model.materials[mat_id] = Material(
        id=mat_id, law=83, rho0=rho0, title=title,
        params={
            "E": e, "nu": 0.3,  # nu not native to LAW83 but needed for dt
            "imass": imass, "fun_a1": fun_a1,
            "fscale11": fscale11, "fscale22": fscale22, "alpha": alpha,
            "beta": beta, "rn": rn, "rs": rs, "fsmooth": fsmooth,
            "fcut": fcut, "fun_a2": fun_a2, "fun_a3": fun_a3,
            "fscale33": fscale33,
        }
    )




def read_mat_law80(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW80`` or ``/MAT/TRANSFO`` (M170): Metallurgical phase transformation steel model."""
    from ...model.entities import MaterialLaw80, Material
    mat_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MAT/LAW80/{mat_id}: missing data card", block.source)
        return

    rho0, refer_rho = 0.0, 0.0
    e, nu = 0.0, 0.0
    fct_ide = 0
    scale_e = 1.0
    time_unit = 3600.0
    fsmooth = 0
    fcut = 0.0
    ceps, peps = 0.0, 0.0
    fun_a = [0] * 5
    fscale_y = [1.0] * 5
    scale_x = [1.0] * 5
    theta = [0.0] * 4
    alpha1, alpha2 = 0.0, 0.0

    if block.fixed:
        f1 = cards[0].cut("MAT_LAW80_1")
        rho0 = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        refer_rho = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MAT_LAW80_2")
            e = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            nu = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            fct_ide = _ival(f2[2]) if len(f2) > 2 else 0
            scale_e = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0
            time_unit = _fval(f2[4], 3600.0) if len(f2) > 4 else 3600.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("MAT_LAW80_3")
            fsmooth = _ival(f3[0]) if len(f3) > 0 else 0
            fcut = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
            ceps = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
            peps = _fval(f3[3], 0.0) if len(f3) > 3 else 0.0

        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("MAT_LAW80_4")
            for i in range(min(5, len(f4))):
                fun_a[i] = _ival(f4[i])

        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("MAT_LAW80_5")
            for i in range(min(5, len(f5))):
                fscale_y[i] = _fval(f5[i], 1.0)

        if len(cards) > 5 and not cards[5].is_blank:
            f6 = cards[5].cut("MAT_LAW80_6")
            for i in range(min(5, len(f6))):
                scale_x[i] = _fval(f6[i], 1.0)

        if len(cards) > 6 and not cards[6].is_blank:
            f7 = cards[6].cut("MAT_LAW80_7")
            for i in range(min(4, len(f7))):
                theta[i] = _fval(f7[i], 0.0)

        if len(cards) > 7 and not cards[7].is_blank:
            f8 = cards[7].cut("MAT_LAW80_8")
            alpha1 = _fval(f8[0], 0.0) if len(f8) > 0 else 0.0
            alpha2 = _fval(f8[1], 0.0) if len(f8) > 1 else 0.0
    else:
        t1 = cards[0].tokens()
        rho0 = float(t1[0]) if len(t1) > 0 else 0.0
        refer_rho = float(t1[1]) if len(t1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            e = float(t2[0]) if len(t2) > 0 else 0.0
            nu = float(t2[1]) if len(t2) > 1 else 0.0
            fct_ide = int(float(t2[2])) if len(t2) > 2 else 0
            scale_e = float(t2[3]) if len(t2) > 3 else 1.0
            time_unit = float(t2[4]) if len(t2) > 4 else 3600.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            fsmooth = int(float(t3[0])) if len(t3) > 0 else 0
            fcut = float(t3[1]) if len(t3) > 1 else 0.0
            ceps = float(t3[2]) if len(t3) > 2 else 0.0
            peps = float(t3[3]) if len(t3) > 3 else 0.0

        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            for i in range(min(5, len(t4))):
                fun_a[i] = int(float(t4[i]))

        if len(cards) > 4 and not cards[4].is_blank:
            t5 = cards[4].tokens()
            for i in range(min(5, len(t5))):
                fscale_y[i] = float(t5[i])

        if len(cards) > 5 and not cards[5].is_blank:
            t6 = cards[5].tokens()
            for i in range(min(5, len(t6))):
                scale_x[i] = float(t6[i])

        if len(cards) > 6 and not cards[6].is_blank:
            t7 = cards[6].tokens()
            for i in range(min(4, len(t7))):
                theta[i] = float(t7[i])

        if len(cards) > 7 and not cards[7].is_blank:
            t8 = cards[7].tokens()
            alpha1 = float(t8[0]) if len(t8) > 0 else 0.0
            alpha2 = float(t8[1]) if len(t8) > 1 else 0.0

    m80 = MaterialLaw80(
        id=mat_id, title=title, rho0=rho0, refer_rho=refer_rho,
        e=e, nu=nu, fct_ide=fct_ide, scale_e=scale_e, time_unit=time_unit,
        fsmooth=fsmooth, fcut=fcut, ceps=ceps, peps=peps, fun_a=fun_a,
        fscale_y=fscale_y, scale_x=scale_x, theta=theta, alpha1=alpha1,
        alpha2=alpha2,
    )
    model.mat_law80s[mat_id] = m80
    model.materials[mat_id] = Material(
        id=mat_id, law=80, rho0=rho0, title=title,
        params={
            "E": e, "nu": nu, "fct_ide": fct_ide, "scale_e": scale_e,
            "time_unit": time_unit, "fsmooth": fsmooth, "fcut": fcut,
            "ceps": ceps, "peps": peps, "fun_a": fun_a,
            "fscale_y": fscale_y, "scale_x": scale_x, "theta": theta,
            "alpha1": alpha1, "alpha2": alpha2,
        }
    )




def read_mat_law117(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW117`` or ``/MAT/COH_MC`` (M171): Cohesive element material model."""
    from ...model.entities import MaterialLaw117, Material
    mat_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MAT/LAW117/{mat_id}: missing data card", block.source)
        return

    rho0, refer_rho = 0.0, 0.0
    e_elas_n, e_elas_s = 0.0, 0.0
    imass, idel, irupt = 0, 0, 0
    fct_tn, fct_tt = 0, 0
    tmax_n, tmax_s = 0.0, 0.0
    fscale_x = 1.0
    gic, giic = 0.0, 0.0
    exp_g, exp_bk = 1.0, 1.0
    gamma = 0.0

    if block.fixed:
        f1 = cards[0].cut("MAT_LAW117_1")
        rho0 = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        refer_rho = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MAT_LAW117_2")
            e_elas_n = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            e_elas_s = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            imass = _ival(f2[2]) if len(f2) > 2 else 0
            idel = _ival(f2[3]) if len(f2) > 3 else 0
            irupt = _ival(f2[4]) if len(f2) > 4 else 0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("MAT_LAW117_3")
            fct_tn = _ival(f3[0]) if len(f3) > 0 else 0
            fct_tt = _ival(f3[1]) if len(f3) > 1 else 0
            tmax_n = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
            tmax_s = _fval(f3[3], 0.0) if len(f3) > 3 else 0.0
            fscale_x = _fval(f3[4], 1.0) if len(f3) > 4 else 1.0

        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[2].cut("MAT_LAW117_4") if len(cards) == 4 and cards[2].raw.count(" ") < 20 else cards[3].cut("MAT_LAW117_4")
            gic = _fval(f4[0], 0.0) if len(f4) > 0 else 0.0
            giic = _fval(f4[1], 0.0) if len(f4) > 1 else 0.0
            exp_g = _fval(f4[2], 1.0) if len(f4) > 2 else 1.0
            exp_bk = _fval(f4[3], 1.0) if len(f4) > 3 else 1.0
            gamma = _fval(f4[4], 0.0) if len(f4) > 4 else 0.0
    else:
        t1 = cards[0].tokens()
        rho0 = float(t1[0]) if len(t1) > 0 else 0.0
        refer_rho = float(t1[1]) if len(t1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            e_elas_n = float(t2[0]) if len(t2) > 0 else 0.0
            e_elas_s = float(t2[1]) if len(t2) > 1 else 0.0
            imass = int(float(t2[2])) if len(t2) > 2 else 0
            idel = int(float(t2[3])) if len(t2) > 3 else 0
            irupt = int(float(t2[4])) if len(t2) > 4 else 0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            fct_tn = int(float(t3[0])) if len(t3) > 0 else 0
            fct_tt = int(float(t3[1])) if len(t3) > 1 else 0
            tmax_n = float(t3[2]) if len(t3) > 2 else 0.0
            tmax_s = float(t3[3]) if len(t3) > 3 else 0.0
            fscale_x = float(t3[4]) if len(t3) > 4 else 1.0

        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            gic = float(t4[0]) if len(t4) > 0 else 0.0
            giic = float(t4[1]) if len(t4) > 1 else 0.0
            exp_g = float(t4[2]) if len(t4) > 2 else 1.0
            exp_bk = float(t4[3]) if len(t4) > 3 else 1.0
            gamma = float(t4[4]) if len(t4) > 4 else 0.0

    m117 = MaterialLaw117(
        id=mat_id, title=title, rho0=rho0, refer_rho=refer_rho,
        e_elas_n=e_elas_n, e_elas_s=e_elas_s, imass=imass, idel=idel,
        irupt=irupt, fct_tn=fct_tn, fct_tt=fct_tt, tmax_n=tmax_n,
        tmax_s=tmax_s, fscale_x=fscale_x, gic=gic, giic=giic,
        exp_g=exp_g, exp_bk=exp_bk, gamma=gamma,
    )
    model.mat_law117s[mat_id] = m117
    params_117 = {
        "E_n": e_elas_n, "e_elas_n": e_elas_n, "E_elas_n": e_elas_n, "EN": e_elas_n, "en": e_elas_n, "E": e_elas_n, "e": e_elas_n, "MAT_E_ELAS_N": e_elas_n,
        "E_t": e_elas_s, "e_elas_s": e_elas_s, "E_elas_s": e_elas_s, "ES": e_elas_s, "es": e_elas_s, "G": e_elas_s, "g": e_elas_s, "MAT_E_ELAS_S": e_elas_s,
        "sigma_max": tmax_n, "tmax_n": tmax_n, "TMAX_N": tmax_n, "TN": tmax_n, "tn": tmax_n, "MAT_TMAX_N": tmax_n,
        "tau_max": tmax_s, "tmax_s": tmax_s, "TMAX_S": tmax_s, "TS": tmax_s, "ts": tmax_s, "MAT_TMAX_S": tmax_s,
        "G_Ic": gic, "gic": gic, "GIC": gic, "MAT_GIC": gic,
        "G_IIc": giic, "giic": giic, "GIIC": giic, "MAT_GIIC": giic,
        "imass": imass, "idel": idel, "irupt": irupt,
        "fct_tn": fct_tn, "fct_tt": fct_tt, "fscale_x": fscale_x,
        "exp_g": exp_g, "exp_bk": exp_bk, "gamma": gamma,
        "rho0": rho0, "rho": rho0, "refer_rho": refer_rho, "MAT_RHO": rho0,
    }
    model.materials[mat_id] = Material(
        id=mat_id, law=117, rho0=rho0, title=title,
        params=params_117
    )
    from ..mat_reader import GenericMaterialRecord
    model.materials[mat_id].record = GenericMaterialRecord(
        law_name="LAW117", law_number=117, id=mat_id, title=title,
        params=params_117, density=rho0, unit_id=block.unit_id,
    )




def read_mat_law90(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW90`` or ``/MAT/HYST_FOAM`` (M171/M592): Tabular strain-rate hysteretic foam material model."""
    from ...model.entities import MaterialLaw90, Material
    from ...materials.law90_foam import Law90Params
    mat_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MAT/LAW90/{mat_id}: missing data card", block.source)
        return

    rho0, refer_rho = 0.0, 0.0
    e0, nu = 0.0, 0.0
    tflag = 1
    fail = 0
    econt = 0.0
    tcut = 1e20
    nl, ismooth = 0, 0
    fcut = 0.0
    shape, hys = 1.0, 1.0
    alpha = 1.0
    fct_ids: List[int] = []
    eps_dots: List[float] = []
    fscales: List[float] = []

    if block.fixed:
        f1 = cards[0].cut("MAT_LAW90_1")
        rho0 = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        refer_rho = _fval(f1[1], rho0) if len(f1) > 1 and f1[1].strip() else rho0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MAT_LAW90_2")
            e0 = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            nu = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            tflag = _ival(f2[2], 1) if len(f2) > 2 and f2[2].strip() else 1
            fail = _ival(f2[3], 0) if len(f2) > 3 and f2[3].strip() else 0
            econt = _fval(f2[4], e0) if len(f2) > 4 and f2[4].strip() else e0
            tcut = _fval(f2[5], 1e20) if len(f2) > 5 and f2[5].strip() else 1e20

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("MAT_LAW90_3")
            nl = _ival(f3[0]) if len(f3) > 0 else 0
            ismooth = _ival(f3[1]) if len(f3) > 1 and f3[1].strip() else 0
            fcut = _fval(f3[2], 0.0) if len(f3) > 2 and f3[2].strip() else 0.0
            shape = _fval(f3[3], 1.0) if len(f3) > 3 and f3[3].strip() else 1.0
            hys = _fval(f3[4], 1.0) if len(f3) > 4 and f3[4].strip() else 1.0
            alpha = _fval(f3[5], 1.0) if len(f3) > 5 and f3[5].strip() else 1.0

        for c in cards[3:3 + nl]:
            if c.is_blank:
                continue
            ff = c.cut("MAT_LAW90_FUNC")
            fct_ids.append(_ival(ff[0]) if len(ff) > 0 else 0)
            eps_dots.append(_fval(ff[1], 0.0) if len(ff) > 1 else 0.0)
            fscales.append(_fval(ff[2], 1.0) if len(ff) > 2 else 1.0)
    else:
        t1 = cards[0].tokens()
        rho0 = float(t1[0]) if len(t1) > 0 else 0.0
        refer_rho = float(t1[1]) if len(t1) > 1 else rho0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            e0 = float(t2[0]) if len(t2) > 0 else 0.0
            nu = float(t2[1]) if len(t2) > 1 else 0.0
            tflag = int(float(t2[2])) if len(t2) > 2 else 1
            fail = int(float(t2[3])) if len(t2) > 3 else 0
            econt = float(t2[4]) if len(t2) > 4 else e0
            tcut = float(t2[5]) if len(t2) > 5 else 1e20

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            nl = int(float(t3[0])) if len(t3) > 0 else 0
            ismooth = int(float(t3[1])) if len(t3) > 1 else 0
            fcut = float(t3[2]) if len(t3) > 2 else 0.0
            shape = float(t3[3]) if len(t3) > 3 else 1.0
            hys = float(t3[4]) if len(t3) > 4 else 1.0
            alpha = float(t3[5]) if len(t3) > 5 else 1.0

        for c in cards[3:3 + nl]:
            if c.is_blank:
                continue
            toks = c.tokens()
            fct_ids.append(int(float(toks[0])) if len(toks) > 0 else 0)
            eps_dots.append(float(toks[1]) if len(toks) > 1 else 0.0)
            fscales.append(float(toks[2]) if len(toks) > 2 else 1.0)

    if tcut <= 0.0:
        tcut = 1e20
    if shape <= 0.0:
        shape = 1.0
    if alpha <= 0.0:
        alpha = 1.0
    if hys <= 0.0:
        hys = 1.0

    m90 = MaterialLaw90(
        id=mat_id, title=title, rho0=rho0, refer_rho=refer_rho,
        e0=e0, nu=nu, nl=nl, ismooth=ismooth, fcut=fcut, shape=shape,
        hys=hys, fct_ids=fct_ids, eps_dots=eps_dots, fscales=fscales,
        alpha=alpha, gamma=alpha, tflag=tflag, fail=fail, econt=econt, tcut=tcut,
    )
    model.mat_law90s[mat_id] = m90
    mat = Material(
        id=mat_id, law=90, rho0=rho0, title=title,
        params={
            "E": e0, "E0": e0, "nu": nu, "NL": nl, "Ismooth": ismooth, "Fcut": fcut,
            "shape": shape, "hys": hys, "gamma": alpha, "alpha": alpha,
            "tcut": tcut, "tflag": tflag, "fail": fail, "econt": econt,
            "fct_ids": fct_ids, "eps_dots": eps_dots, "fscales": fscales,
            "curves": [], "E_MAX": max(e0, 100.0 * e0),
        }
    )
    mat.law90_params = Law90Params(
        rho0=rho0, refer_rho=refer_rho, E0=e0, nu=nu, shape=shape,
        hys=hys, gamma=alpha, alpha=alpha, tcut=tcut, tflag=tflag, fail=fail,
        econt=econt, ismooth=ismooth, fcut=fcut, nl=nl, fct_ids=fct_ids,
        eps_dots=eps_dots, fscales=fscales, E_MAX=max(e0, 100.0 * e0),
    )
    model.materials[mat_id] = mat




def read_mat_law33(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW33`` or ``/MAT/FOAM_PLAS`` (M171): Crushable foam plasticity material model."""
    from ...model.entities import MaterialLaw33, Material
    mat_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MAT/LAW33/{mat_id}: missing data card", block.source)
        return

    rho0, refer_rho = 0.0, 0.0
    e = 0.0
    itype = 0
    fun_a1 = 0
    ifscale = 1.0
    p0, phi, gama0 = 0.0, 0.0, 0.0
    a0, a1, a2 = 0.0, 0.0, 0.0
    e1, e2, etan, eta1, eta2 = 0.0, 0.0, 0.0, 0.0, 0.0

    if block.fixed:
        f1 = cards[0].cut("MAT_LAW33_1")
        rho0 = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        refer_rho = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MAT_LAW33_2")
            e = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            itype = _ival(f2[1]) if len(f2) > 1 else 0
            fun_a1 = _ival(f2[2]) if len(f2) > 2 else 0
            ifscale = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("MAT_LAW33_3")
            p0 = _fval(f3[0], 0.0) if len(f3) > 0 else 0.0
            phi = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
            gama0 = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0

        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("MAT_LAW33_4")
            a0 = _fval(f4[0], 0.0) if len(f4) > 0 else 0.0
            a1 = _fval(f4[1], 0.0) if len(f4) > 1 else 0.0
            a2 = _fval(f4[2], 0.0) if len(f4) > 2 else 0.0

        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("MAT_LAW33_5")
            e1 = _fval(f5[0], 0.0) if len(f5) > 0 else 0.0
            e2 = _fval(f5[1], 0.0) if len(f5) > 1 else 0.0
            etan = _fval(f5[2], 0.0) if len(f5) > 2 else 0.0
            eta1 = _fval(f5[3], 0.0) if len(f5) > 3 else 0.0
            eta2 = _fval(f5[4], 0.0) if len(f5) > 4 else 0.0
    else:
        t1 = cards[0].tokens()
        rho0 = float(t1[0]) if len(t1) > 0 else 0.0
        refer_rho = float(t1[1]) if len(t1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            e = float(t2[0]) if len(t2) > 0 else 0.0
            itype = int(float(t2[1])) if len(t2) > 1 else 0
            fun_a1 = int(float(t2[2])) if len(t2) > 2 else 0
            ifscale = float(t2[3]) if len(t2) > 3 else 1.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            p0 = float(t3[0]) if len(t3) > 0 else 0.0
            phi = float(t3[1]) if len(t3) > 1 else 0.0
            gama0 = float(t3[2]) if len(t3) > 2 else 0.0

        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            a0 = float(t4[0]) if len(t4) > 0 else 0.0
            a1 = float(t4[1]) if len(t4) > 1 else 0.0
            a2 = float(t4[2]) if len(t4) > 2 else 0.0

        if len(cards) > 4 and not cards[4].is_blank:
            t5 = cards[4].tokens()
            e1 = float(t5[0]) if len(t5) > 0 else 0.0
            e2 = float(t5[1]) if len(t5) > 1 else 0.0
            etan = float(t5[2]) if len(t5) > 2 else 0.0
            eta1 = float(t5[3]) if len(t5) > 3 else 0.0
            eta2 = float(t5[4]) if len(t5) > 4 else 0.0

    m33 = MaterialLaw33(
        id=mat_id, title=title, rho0=rho0, refer_rho=refer_rho,
        e=e, itype=itype, fun_a1=fun_a1, ifscale=ifscale,
        p0=p0, phi=phi, gama0=gama0, a0=a0, a1=a1, a2=a2,
        e1=e1, e2=e2, etan=etan, eta1=eta1, eta2=eta2,
    )
    model.mat_law33s[mat_id] = m33
    model.materials[mat_id] = Material(
        id=mat_id, law=33, rho0=rho0, title=title,
        params={
            "E": e, "Itype": itype, "fun_a1": fun_a1, "ifscale": ifscale,
            "p0": p0, "phi": phi, "gama0": gama0, "a0": a0, "a1": a1, "a2": a2,
            "e1": e1, "e2": e2, "etan": etan, "eta1": eta1, "eta2": eta2,
        }
    )




def read_mat_heat(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/HEAT`` or ``/HEAT/MAT`` (M171): Material thermal property modifier."""
    from ...model.entities import MatHeatModifier
    obj_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MAT/HEAT/{obj_id}: missing data card", block.source)
        return

    t0, rho0_cp, as_solid, bs_solid = 0.0, 0.0, 0.0, 0.0
    t1, al_liquid, bl_liquid, efrac = 1.0e30, 0.0, 0.0, 1.0

    if block.fixed:
        f1 = cards[0].cut("MAT_HEAT_MOD_1")
        t0 = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        rho0_cp = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
        as_solid = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
        bs_solid = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MAT_HEAT_MOD_2")
            t1 = _fval(f2[0], 1.0e30) if len(f2) > 0 else 1.0e30
            al_liquid = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            bl_liquid = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            efrac = _fval(f2[3], 1.0) if len(f2) > 3 else 1.0
    else:
        t1_tok = cards[0].tokens()
        t0 = float(t1_tok[0]) if len(t1_tok) > 0 else 0.0
        rho0_cp = float(t1_tok[1]) if len(t1_tok) > 1 else 0.0
        as_solid = float(t1_tok[2]) if len(t1_tok) > 2 else 0.0
        bs_solid = float(t1_tok[3]) if len(t1_tok) > 3 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2_tok = cards[1].tokens()
            t1 = float(t2_tok[0]) if len(t2_tok) > 0 else 1.0e30
            al_liquid = float(t2_tok[1]) if len(t2_tok) > 1 else 0.0
            bl_liquid = float(t2_tok[2]) if len(t2_tok) > 2 else 0.0
            efrac = float(t2_tok[3]) if len(t2_tok) > 3 else 1.0

    hm = MatHeatModifier(
        id=obj_id, mat_id=obj_id, t0=t0, rho0_cp=rho0_cp,
        as_solid=as_solid, bs_solid=bs_solid, t1=t1,
        al_liquid=al_liquid, bl_liquid=bl_liquid, efrac=efrac,
    )
    model.mat_heat_modifiers[obj_id] = hm
    if not hasattr(model, "raw_mat_notes"):
        model.raw_mat_notes = []
    model.raw_mat_notes.append((
        "HEAT/MAT", obj_id, {
            "HEAT_T0": t0, "HEAT_RHocp": rho0_cp, "HEAT_AS": as_solid,
            "HEAT_BS": bs_solid, "HEAT_T1": t1, "HEAT_AL": al_liquid,
            "HEAT_BL": bl_liquid, "HEAT_EFRAC": efrac,
        }, block.source
    ))




def read_mat_nonlocal(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/NONLOCAL`` or ``/NONLOCAL/MAT`` (M171): Non-local regularized damage modifier."""
    from ...model.entities import MatNonlocalModifier
    obj_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MAT/NONLOCAL/{obj_id}: missing data card", block.source)
        return

    length, le_max = 0.0, 0.0
    if block.fixed:
        f1 = cards[0].cut("MAT_NONLOCAL_MOD_1")
        length = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        le_max = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
    else:
        toks = cards[0].tokens()
        length = float(toks[0]) if len(toks) > 0 else 0.0
        le_max = float(toks[1]) if len(toks) > 1 else 0.0

    nl_mod = MatNonlocalModifier(id=obj_id, mat_id=obj_id, length=length, le_max=le_max)
    model.mat_nonlocal_modifiers[obj_id] = nl_mod




def read_mat_law66(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW66`` (``/MAT/PLAS_TAB_COSSER``, ``/MAT/PLAS_COSSER``, ``/MAT/FOAM_TAB``) (M172, M562):
    Tabulated tension-compression plastic material model.

    Fortran origin: ``starter/source/materials/mat/mat066/hm_read_mat66.F``.
    CFG reference: ``radioss2022/MAT/mat_law66.cfg``.
    """
    from ...model.entities import MatLaw66, Material
    mat_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW66/{mat_id}: missing data card", block.source)
        return

    rho0, refer_rho = 0.0, 0.0
    e, nu, c_hard, f_cut = 0.0, 0.0, 0.0, 0.0
    fsmooth, israte = 0, 1
    p_c, p_t, ec, rpct = 0.0, 0.0, 0.0, 1.0
    funct_idc, funct_idt = 0, 0
    fscalec, fscalet = 1.0, 1.0
    epsilon_0, c, sigma_y0 = 1.0, 1.0, 0.0
    vp = 0
    fnyrt_idc, fnyrt_idt = 0, 0
    yrate_fscalec, yrate_fscalet = 1.0, 1.0
    nfunc, tfunc = 0, 0
    abg_ipt: List[int] = []
    k_a1: List[float] = []
    fp1: List[float] = []
    abg_ipdel: List[int] = []
    k_b1: List[float] = []
    fp2: List[float] = []

    is_fixed = block.fixed
    if is_fixed:
        for vc in valid_cards[:3]:
            if "," in vc.raw or (len(vc.tokens()) > 1 and len(vc.raw[:20].split()) > 1):
                is_fixed = False
                break

    if is_fixed:
        # Card 1: RHO_I [Refer_Rho]
        f1 = valid_cards[0].cut("MAT_LAW66_1")
        rho0 = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        refer_rho = _fval(f1[1], 0.0) if len(f1) > 1 and f1[1].strip() else 0.0

        # Card 2: E, Nu, C_hard, F_cut, Fsmooth, Iyld_rate
        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("MAT_LAW66_2")
            e = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            nu = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            c_hard = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            f_cut = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
            fsmooth = _ival(f2[4]) if len(f2) > 4 else 0
            israte = _ival(f2[5]) if len(f2) > 5 else 1

        # Card 3: P_c, P_t, EC, RPCT
        if len(valid_cards) > 2:
            f3 = valid_cards[2].cut("MAT_LAW66_3")
            p_c = _fval(f3[0], 0.0) if len(f3) > 0 else 0.0
            p_t = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
            ec = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
            rpct = _fval(f3[3], 1.0) if len(f3) > 3 and f3[3].strip() else 1.0

        card_idx = 3
        if israte in (0, 1, 2, 3):
            if len(valid_cards) > card_idx:
                f4 = valid_cards[card_idx].cut("MAT_LAW66_4")
                funct_idc = _ival(f4[0]) if len(f4) > 0 else 0
                funct_idt = _ival(f4[1]) if len(f4) > 1 else 0
                fscalec = _fval(f4[2], 1.0) if len(f4) > 2 and f4[2].strip() else 1.0
                fscalet = _fval(f4[3], 1.0) if len(f4) > 3 and f4[3].strip() else 1.0
                card_idx += 1
            if israte in (0, 1, 2) and len(valid_cards) > card_idx:
                f5 = valid_cards[card_idx].cut("MAT_LAW66_5")
                epsilon_0 = _fval(f5[0], 1.0) if len(f5) > 0 and f5[0].strip() else 1.0
                c = _fval(f5[1], 1.0) if len(f5) > 1 and f5[1].strip() else 1.0
                sigma_y0 = _fval(f5[2], 0.0) if len(f5) > 2 else 0.0
                vp = _ival(f5[3]) if len(f5) > 3 else 0
                card_idx += 1
            elif israte == 3 and len(valid_cards) > card_idx:
                f5 = valid_cards[card_idx].cut("MAT_LAW66_ISRATE_3")
                fnyrt_idc = _ival(f5[0]) if len(f5) > 0 else 0
                fnyrt_idt = _ival(f5[1]) if len(f5) > 1 else 0
                yrate_fscalec = _fval(f5[2], 1.0) if len(f5) > 2 and f5[2].strip() else 1.0
                yrate_fscalet = _fval(f5[3], 1.0) if len(f5) > 3 and f5[3].strip() else 1.0
                card_idx += 1
        elif israte == 4:
            if len(valid_cards) > card_idx:
                f4 = valid_cards[card_idx].cut("MAT_LAW66_ISRATE_4")
                nfunc = _ival(f4[0]) if len(f4) > 0 else 0
                tfunc = _ival(f4[1]) if len(f4) > 1 else 0
                card_idx += 1
            for _ in range(nfunc):
                if card_idx < len(valid_cards):
                    fc = valid_cards[card_idx].cut("MAT_LAW66_CURVE")
                    abg_ipt.append(_ival(fc[0]) if len(fc) > 0 else 0)
                    k_a1.append(_fval(fc[2], 0.0) if len(fc) > 2 else 0.0)
                    fp1.append(_fval(fc[3], 1.0) if len(fc) > 3 and fc[3].strip() else 1.0)
                    card_idx += 1
            for _ in range(tfunc):
                if card_idx < len(valid_cards):
                    ft = valid_cards[card_idx].cut("MAT_LAW66_CURVE")
                    abg_ipdel.append(_ival(ft[0]) if len(ft) > 0 else 0)
                    k_b1.append(_fval(ft[2], 0.0) if len(ft) > 2 else 0.0)
                    fp2.append(_fval(ft[3], 1.0) if len(ft) > 3 and ft[3].strip() else 1.0)
                    card_idx += 1
    else:
        # Free format
        t1 = valid_cards[0].tokens()
        rho0 = float(t1[0]) if len(t1) > 0 else 0.0
        refer_rho = float(t1[1]) if len(t1) > 1 else 0.0

        if len(valid_cards) > 1:
            t2 = valid_cards[1].tokens()
            e = float(t2[0]) if len(t2) > 0 else 0.0
            nu = float(t2[1]) if len(t2) > 1 else 0.0
            c_hard = float(t2[2]) if len(t2) > 2 else 0.0
            f_cut = float(t2[3]) if len(t2) > 3 else 0.0
            fsmooth = int(float(t2[4])) if len(t2) > 4 else 0
            israte = int(float(t2[5])) if len(t2) > 5 else 1

        if len(valid_cards) > 2:
            t3 = valid_cards[2].tokens()
            p_c = float(t3[0]) if len(t3) > 0 else 0.0
            p_t = float(t3[1]) if len(t3) > 1 else 0.0
            ec = float(t3[2]) if len(t3) > 2 else 0.0
            rpct = float(t3[3]) if len(t3) > 3 else 1.0

        card_idx = 3
        if israte in (0, 1, 2, 3):
            if len(valid_cards) > card_idx:
                t4 = valid_cards[card_idx].tokens()
                funct_idc = int(float(t4[0])) if len(t4) > 0 else 0
                funct_idt = int(float(t4[1])) if len(t4) > 1 else 0
                fscalec = float(t4[2]) if len(t4) > 2 else 1.0
                fscalet = float(t4[3]) if len(t4) > 3 else 1.0
                card_idx += 1
            if israte in (0, 1, 2) and len(valid_cards) > card_idx:
                t5 = valid_cards[card_idx].tokens()
                epsilon_0 = float(t5[0]) if len(t5) > 0 else 1.0
                c = float(t5[1]) if len(t5) > 1 else 1.0
                sigma_y0 = float(t5[2]) if len(t5) > 2 else 0.0
                vp = int(float(t5[3])) if len(t5) > 3 else 0
                card_idx += 1
            elif israte == 3 and len(valid_cards) > card_idx:
                t5 = valid_cards[card_idx].tokens()
                fnyrt_idc = int(float(t5[0])) if len(t5) > 0 else 0
                fnyrt_idt = int(float(t5[1])) if len(t5) > 1 else 0
                yrate_fscalec = float(t5[2]) if len(t5) > 2 else 1.0
                yrate_fscalet = float(t5[3]) if len(t5) > 3 else 1.0
                card_idx += 1
        elif israte == 4:
            if len(valid_cards) > card_idx:
                t4 = valid_cards[card_idx].tokens()
                nfunc = int(float(t4[0])) if len(t4) > 0 else 0
                tfunc = int(float(t4[1])) if len(t4) > 1 else 0
                card_idx += 1
            for _ in range(nfunc):
                if card_idx < len(valid_cards):
                    tok = valid_cards[card_idx].tokens()
                    if len(tok) >= 4:
                        abg_ipt.append(int(float(tok[0])))
                        k_a1.append(float(tok[-2]))
                        fp1.append(float(tok[-1]))
                    else:
                        abg_ipt.append(int(float(tok[0])) if len(tok) > 0 else 0)
                        k_a1.append(float(tok[1]) if len(tok) > 1 else 0.0)
                        fp1.append(float(tok[2]) if len(tok) > 2 else 1.0)
                    card_idx += 1
            for _ in range(tfunc):
                if card_idx < len(valid_cards):
                    tok = valid_cards[card_idx].tokens()
                    if len(tok) >= 4:
                        abg_ipdel.append(int(float(tok[0])))
                        k_b1.append(float(tok[-2]))
                        fp2.append(float(tok[-1]))
                    else:
                        abg_ipdel.append(int(float(tok[0])) if len(tok) > 0 else 0)
                        k_b1.append(float(tok[1]) if len(tok) > 1 else 0.0)
                        fp2.append(float(tok[2]) if len(tok) > 2 else 1.0)
                    card_idx += 1

    # Fortran defaults alignment:
    if epsilon_0 == 0.0:
        epsilon_0 = 1.0
    if c == 0.0:
        c = 1.0
    if rpct == 0.0:
        rpct = 1.0

    actual_refer_rho = refer_rho if refer_rho != 0.0 else rho0

    m66 = MatLaw66(
        id=mat_id,
        rho=rho0,
        e=e,
        nu=nu,
        ec=ec,
        pc=p_c,
        pt=p_t,
        rpct=rpct,
        chard=c_hard,
        asrate=f_cut,
        fsmooth=fsmooth,
        israte=israte,
        fun_a1=funct_idc,
        fun_a2=funct_idt,
        fscale11=fscalec,
        fscale22=fscalet,
        epsp0=epsilon_0,
        cp=c,
        sigy=sigma_y0,
        vp=vp,
        fun_b1=fnyrt_idc,
        fun_b2=fnyrt_idt,
        fscale33=yrate_fscalec,
        fscale12=yrate_fscalet,
        nfunc=nfunc,
        tfunc=tfunc,
        abg_ipt=abg_ipt,
        k_a1=k_a1,
        fp1=fp1,
        abg_ipdel=abg_ipdel,
        k_b1=k_b1,
        fp2=fp2,
        title=title,
        law=66,
        law_name="LAW66",
        refer_rho=actual_refer_rho,
    )
    model.mat_law66s[mat_id] = m66
    model.materials[mat_id] = Material(
        id=mat_id,
        law=66,
        rho0=rho0,
        title=title,
        params={
            "rho": rho0,
            "rho0": rho0,
            "rhor": actual_refer_rho,
            "refer_rho": actual_refer_rho,
            "e": e,
            "E": e,
            "nu": nu,
            "Nu": nu,
            "C_hard": c_hard,
            "c_hard": c_hard,
            "chard": c_hard,
            "fisokin": c_hard,
            "F_cut": f_cut,
            "f_cut": f_cut,
            "asrate": f_cut,
            "Fsmooth": fsmooth,
            "fsmooth": fsmooth,
            "ISRATE": israte,
            "israte": israte,
            "P_c": p_c,
            "pc": p_c,
            "PC": p_c,
            "P_t": p_t,
            "pt": p_t,
            "PT": p_t,
            "EC": ec,
            "ec": ec,
            "RPCT": rpct,
            "rpct": rpct,
            "funct_IDc": funct_idc,
            "funct_idc": funct_idc,
            "fun_a1": funct_idc,
            "funct_IDt": funct_idt,
            "funct_idt": funct_idt,
            "fun_a2": funct_idt,
            "Fscalec": fscalec,
            "fscalec": fscalec,
            "fscale11": fscalec,
            "Fscalet": fscalet,
            "fscalet": fscalet,
            "fscale22": fscalet,
            "Epsilon_0": epsilon_0,
            "epsilon_0": epsilon_0,
            "eps_0": epsilon_0,
            "epsp0": epsilon_0,
            "c": c,
            "cp": c,
            "Sigma_Y0": sigma_y0,
            "sigma_y0": sigma_y0,
            "sigy": sigma_y0,
            "VP": vp,
            "vp": vp,
            "fnYrt_IDc": fnyrt_idc,
            "fnyrt_idc": fnyrt_idc,
            "fun_b1": fnyrt_idc,
            "fnYrt_IDt": fnyrt_idt,
            "fnyrt_idt": fnyrt_idt,
            "fun_b2": fnyrt_idt,
            "Yrate_Fscalec": yrate_fscalec,
            "yrate_fscalec": yrate_fscalec,
            "fscale33": yrate_fscalec,
            "Yrate_Fscalet": yrate_fscalet,
            "yrate_fscalet": yrate_fscalet,
            "fscale12": yrate_fscalet,
            "NFUNCC": nfunc,
            "NFUNCT": tfunc,
            "NFUNC": nfunc,
            "TFUNC": tfunc,
            "nfunc": nfunc,
            "tfunc": tfunc,
            "func_c_list": abg_ipt,
            "eps_c_list": k_a1,
            "fscale_c_list": fp1,
            "func_t_list": abg_ipdel,
            "eps_t_list": k_b1,
            "fscale_t_list": fp2,
            "ABG_IPt": abg_ipt,
            "K_A1": k_a1,
            "Fp1": fp1,
            "ABG_IPdel": abg_ipdel,
            "K_B1": k_b1,
            "Fp2": fp2,
            "abg_ipt": abg_ipt,
            "k_a1": k_a1,
            "fp1": fp1,
            "abg_ipdel": abg_ipdel,
            "k_b1": k_b1,
            "fp2": fp2,
            "G": m66.G,
            "bulk": m66.bulk,
            "K": m66.K,
            "sound_speed": m66.sound_speed,
            "sound_speed_solid": m66.sound_speed_solid,
            "sound_speed_shell": m66.sound_speed_shell,
        },
    )




def read_mat_law35(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW35`` or ``/MAT/FOAM_VISC`` (M172): Viscoelastic foam material model.

    Fortran origin: ``starter/source/materials/mat/mat035/hm_read_mat35.F``.
    """
    from ...model.entities import MaterialLaw35, Material
    mat_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MAT/LAW35/{mat_id}: missing data card", block.source)
        return

    rho0, refer_rho = 0.0, 0.0
    e, nu, e1, e2, n = 0.0, 0.0, 0.0, 0.0, 0.0
    c1, c2, c3, itype, pmin = 0.0, 0.0, 0.0, 0, 0.0
    func_idf, fscalepres = 0, 1.0
    fsmooth, fcut = 0, 0.0
    et, nu_t, eta_0, lamda = 0.0, 0.0, 0.0, 0.0
    p0, phi, gama0 = 0.0, 0.0, 0.0

    if block.fixed:
        f1 = cards[0].cut("MAT_LAW35_1")
        rho0 = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        refer_rho = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MAT_LAW35_2")
            e = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            nu = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            e1 = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            e2 = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
            n = _fval(f2[4], 0.0) if len(f2) > 4 else 0.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("MAT_LAW35_3")
            c1 = _fval(f3[0], 0.0) if len(f3) > 0 else 0.0
            c2 = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
            c3 = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
            itype = _ival(f3[4]) if len(f3) > 4 else 0
            pmin = _fval(f3[5], 0.0) if len(f3) > 5 else 0.0

        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("MAT_LAW35_4")
            func_idf = _ival(f4[0]) if len(f4) > 0 else 0
            fscalepres = _fval(f4[2], 1.0) if len(f4) > 2 else 1.0
            fsmooth = _ival(f4[4]) if len(f4) > 4 else 0
            fcut = _fval(f4[5], 0.0) if len(f4) > 5 else 0.0

        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("MAT_LAW35_5")
            et = _fval(f5[0], 0.0) if len(f5) > 0 else 0.0
            nu_t = _fval(f5[1], 0.0) if len(f5) > 1 else 0.0
            eta_0 = _fval(f5[2], 0.0) if len(f5) > 2 else 0.0
            lamda = _fval(f5[3], 0.0) if len(f5) > 3 else 0.0

        if len(cards) > 5 and not cards[5].is_blank:
            f6 = cards[5].cut("MAT_LAW35_6")
            p0 = _fval(f6[0], 0.0) if len(f6) > 0 else 0.0
            phi = _fval(f6[1], 0.0) if len(f6) > 1 else 0.0
            gama0 = _fval(f6[2], 0.0) if len(f6) > 2 else 0.0
    else:
        t1 = cards[0].tokens()
        rho0 = float(t1[0]) if len(t1) > 0 else 0.0
        refer_rho = float(t1[1]) if len(t1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            e = float(t2[0]) if len(t2) > 0 else 0.0
            nu = float(t2[1]) if len(t2) > 1 else 0.0
            e1 = float(t2[2]) if len(t2) > 2 else 0.0
            e2 = float(t2[3]) if len(t2) > 3 else 0.0
            n = float(t2[4]) if len(t2) > 4 else 0.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            c1 = float(t3[0]) if len(t3) > 0 else 0.0
            c2 = float(t3[1]) if len(t3) > 1 else 0.0
            c3 = float(t3[2]) if len(t3) > 2 else 0.0
            itype = int(float(t3[3])) if len(t3) > 3 else 0
            pmin = float(t3[4]) if len(t3) > 4 else 0.0

        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            func_idf = int(float(t4[0])) if len(t4) > 0 else 0
            fscalepres = float(t4[1]) if len(t4) > 1 else 1.0
            fsmooth = int(float(t4[2])) if len(t4) > 2 else 0
            fcut = float(t4[3]) if len(t4) > 3 else 0.0

        if len(cards) > 4 and not cards[4].is_blank:
            t5 = cards[4].tokens()
            et = float(t5[0]) if len(t5) > 0 else 0.0
            nu_t = float(t5[1]) if len(t5) > 1 else 0.0
            eta_0 = float(t5[2]) if len(t5) > 2 else 0.0
            lamda = float(t5[3]) if len(t5) > 3 else 0.0

        if len(cards) > 5 and not cards[5].is_blank:
            t6 = cards[5].tokens()
            p0 = float(t6[0]) if len(t6) > 0 else 0.0
            phi = float(t6[1]) if len(t6) > 1 else 0.0
            gama0 = float(t6[2]) if len(t6) > 2 else 0.0

    pmin_val = pmin
    if pmin_val == 0.0:
        pmin_val = -1e20
    elif pmin_val > 0.0:
        pmin_val = -pmin_val
    ismooth_val = fsmooth
    fcut_val = fcut
    if fcut_val != 0.0:
        ismooth_val = 1
    elif ismooth_val != 0:
        fcut_val = 10000.0
    fscale_val = fscalepres if fscalepres != 0.0 else 1.0

    m35 = MaterialLaw35(
        id=mat_id, title=title, rho0=rho0, ref_rho=refer_rho,
        e=e, nu=nu, e1=e1, e2=e2, n=n, c1=c1, c2=c2, c3=c3,
        itype=itype, pmin=pmin, func_idf=func_idf, fscalepres=fscalepres,
        fsmooth=fsmooth, fcut=fcut, et=et, nu_t=nu_t, eta_0=eta_0,
        lamda=lamda, p0=p0, phi=phi, gama0=gama0,
    )
    model.mat_law35s[mat_id] = m35
    model.materials[mat_id] = Material(
        id=mat_id, law=35, rho0=rho0, title=title,
        params={
            "E": e, "nu": nu,
            "E1": e1, "E2": e2, "N": n, "n": n,
            "Et": et, "nut": nu_t, "Nu_t": nu_t,
            "mu_visc": eta_0, "eta_0": eta_0,
            "lambda_visc": lamda, "Lamda": lamda,
            "C1": c1, "C2": c2, "C3": c3,
            "itype": itype, "Iflag": itype,
            "pmin": pmin_val, "Pmin": pmin,
            "P0": p0, "phi": phi, "Phi": phi, "gama0": gama0, "gamma_0": gama0,
            "fct_id": func_idf, "func_IDf": func_idf,
            "fscale": fscale_val, "Fscalepres": fscalepres,
            "ismooth": ismooth_val, "Fsmooth": fsmooth, "fcut": fcut_val, "Fcut": fcut,
        }
    )




def read_mat_law62(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW62`` or ``/MAT/VISC_HYP`` (M172): Viscoelastic hyperelastic Ogden material model.

    Fortran origin: ``starter/source/materials/mat/mat062/hm_read_mat62.F``.
    """
    from ...model.entities import MaterialLaw62, Material
    mat_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MAT/LAW62/{mat_id}: missing data card", block.source)
        return

    rho0, refer_rho = 0.0, 0.0
    nu = 0.495
    order_n, order_m = 0, 0
    mu_max = 0.0
    mu_arr: List[float] = []
    alpha_arr: List[float] = []
    gamma_arr: List[float] = []
    tau_arr: List[float] = []

    if block.fixed:
        f1 = cards[0].cut("MAT_LAW62_1")
        rho0 = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        refer_rho = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MAT_LAW62_2")
            nu = _fval(f2[0], 0.495) if len(f2) > 0 and f2[0].strip() else 0.495
            order_n = _ival(f2[1]) if len(f2) > 1 else 0
            order_m = _ival(f2[2]) if len(f2) > 2 else 0
            mu_max = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0

        card_idx = 2
        if order_n > 0:
            if card_idx < len(cards) and not cards[card_idx].is_blank:
                mu_arr = [_fval(x, 0.0) for x in cards[card_idx].cut("F20X5")[:order_n]]
                card_idx += 1
            if card_idx < len(cards) and not cards[card_idx].is_blank:
                alpha_arr = [_fval(x, 0.0) for x in cards[card_idx].cut("F20X5")[:order_n]]
                card_idx += 1
        if order_m > 0:
            if card_idx < len(cards) and not cards[card_idx].is_blank:
                gamma_arr = [_fval(x, 0.0) for x in cards[card_idx].cut("F20X5")[:order_m]]
                card_idx += 1
            if card_idx < len(cards) and not cards[card_idx].is_blank:
                tau_arr = [_fval(x, 0.0) for x in cards[card_idx].cut("F20X5")[:order_m]]
                card_idx += 1
    else:
        t1 = cards[0].tokens()
        rho0 = float(t1[0]) if len(t1) > 0 else 0.0
        refer_rho = float(t1[1]) if len(t1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            nu = float(t2[0]) if len(t2) > 0 else 0.495
            order_n = int(float(t2[1])) if len(t2) > 1 else 0
            order_m = int(float(t2[2])) if len(t2) > 2 else 0
            mu_max = float(t2[3]) if len(t2) > 3 else 0.0

        card_idx = 2
        if order_n > 0:
            if card_idx < len(cards) and not cards[card_idx].is_blank:
                mu_arr = [float(x) for x in cards[card_idx].tokens()[:order_n]]
                card_idx += 1
            if card_idx < len(cards) and not cards[card_idx].is_blank:
                alpha_arr = [float(x) for x in cards[card_idx].tokens()[:order_n]]
                card_idx += 1
        if order_m > 0:
            if card_idx < len(cards) and not cards[card_idx].is_blank:
                gamma_arr = [float(x) for x in cards[card_idx].tokens()[:order_m]]
                card_idx += 1
            if card_idx < len(cards) and not cards[card_idx].is_blank:
                tau_arr = [float(x) for x in cards[card_idx].tokens()[:order_m]]
                card_idx += 1

    m62 = MaterialLaw62(
        id=mat_id, title=title, rho0=rho0, ref_rho=refer_rho,
        nu=nu, order_n=order_n, order_m=order_m, mu_max=mu_max,
        mu_arr=mu_arr, alpha_arr=alpha_arr, gamma_arr=gamma_arr, tau_arr=tau_arr,
    )
    model.mat_law62s[mat_id] = m62
    model.materials[mat_id] = Material(
        id=mat_id, law=62, rho0=rho0, title=title,
        params={
            "nu": nu, "N": order_n, "M": order_m, "mu_max": mu_max,
            "mu_arr": mu_arr, "alpha_arr": alpha_arr,
            "gamma_arr": gamma_arr, "tau_arr": tau_arr,
        }
    )




def read_mat_law28(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW28`` or ``/MAT/HONEYCOMB`` (M172): Orthotropic honeycomb crushable material model.

    Fortran origin: ``starter/source/materials/mat/mat028/hm_read_mat28.F``.
    """
    from ...model.entities import MaterialLaw28, Material
    mat_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MAT/LAW28/{mat_id}: missing data card", block.source)
        return

    rho0, refer_rho = 0.0, 0.0
    e11, e22, e33 = 0.0, 0.0, 0.0
    g12, g23, g31 = 0.0, 0.0, 0.0
    fun_a1, fun_b1, fun_a2, gflag = 0, 0, 0, 0
    fscale11, fscale22, fscale33 = 1.0, 1.0, 1.0
    epsr1, epsr2, epsr3 = 0.0, 0.0, 0.0
    fun_a3, fun_b3, fun_a4, vflag = 0, 0, 0, 0
    fscale12, fscale23, fscale13 = 1.0, 1.0, 1.0
    epsr4, epsr5, epsr6 = 0.0, 0.0, 0.0

    if block.fixed:
        f1 = cards[0].cut("MAT_LAW28_1")
        rho0 = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        refer_rho = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MAT_LAW28_2")
            e11 = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            e22 = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            e33 = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("MAT_LAW28_3")
            g12 = _fval(f3[0], 0.0) if len(f3) > 0 else 0.0
            g23 = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
            g31 = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0

        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("MAT_LAW28_4")
            fun_a1 = _ival(f4[0]) if len(f4) > 0 else 0
            fun_b1 = _ival(f4[1]) if len(f4) > 1 else 0
            fun_a2 = _ival(f4[2]) if len(f4) > 2 else 0
            gflag = _ival(f4[3]) if len(f4) > 3 else 0
            fscale11 = _fval(f4[4], 1.0) if len(f4) > 4 and f4[4].strip() else 1.0
            fscale22 = _fval(f4[5], 1.0) if len(f4) > 5 and f4[5].strip() else 1.0
            fscale33 = _fval(f4[6], 1.0) if len(f4) > 6 and f4[6].strip() else 1.0

        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("MAT_LAW28_5")
            epsr1 = _fval(f5[0], 0.0) if len(f5) > 0 else 0.0
            epsr2 = _fval(f5[1], 0.0) if len(f5) > 1 else 0.0
            epsr3 = _fval(f5[2], 0.0) if len(f5) > 2 else 0.0

        if len(cards) > 5 and not cards[5].is_blank:
            f6 = cards[6].cut("MAT_LAW28_6") if len(cards) > 6 and cards[5].raw.count(" ") < 20 else cards[5].cut("MAT_LAW28_6")
            fun_a3 = _ival(f6[0]) if len(f6) > 0 else 0
            fun_b3 = _ival(f6[1]) if len(f6) > 1 else 0
            fun_a4 = _ival(f6[2]) if len(f6) > 2 else 0
            vflag = _ival(f6[3]) if len(f6) > 3 else 0
            fscale12 = _fval(f6[4], 1.0) if len(f6) > 4 and f6[4].strip() else 1.0
            fscale23 = _fval(f6[5], 1.0) if len(f6) > 5 and f6[5].strip() else 1.0
            fscale13 = _fval(f6[6], 1.0) if len(f6) > 6 and f6[6].strip() else 1.0

        if len(cards) > 6 and not cards[6].is_blank:
            f7 = cards[6].cut("MAT_LAW28_7")
            epsr4 = _fval(f7[0], 0.0) if len(f7) > 0 else 0.0
            epsr5 = _fval(f7[1], 0.0) if len(f7) > 1 else 0.0
            epsr6 = _fval(f7[2], 0.0) if len(f7) > 2 else 0.0
    else:
        t1 = cards[0].tokens()
        rho0 = float(t1[0]) if len(t1) > 0 else 0.0
        refer_rho = float(t1[1]) if len(t1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            e11 = float(t2[0]) if len(t2) > 0 else 0.0
            e22 = float(t2[1]) if len(t2) > 1 else 0.0
            e33 = float(t2[2]) if len(t2) > 2 else 0.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            g12 = float(t3[0]) if len(t3) > 0 else 0.0
            g23 = float(t3[1]) if len(t3) > 1 else 0.0
            g31 = float(t3[2]) if len(t3) > 2 else 0.0

        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            fun_a1 = int(float(t4[0])) if len(t4) > 0 else 0
            fun_b1 = int(float(t4[1])) if len(t4) > 1 else 0
            fun_a2 = int(float(t4[2])) if len(t4) > 2 else 0
            gflag = int(float(t4[3])) if len(t4) > 3 else 0
            fscale11 = float(t4[4]) if len(t4) > 4 else 1.0
            fscale22 = float(t4[5]) if len(t4) > 5 else 1.0
            fscale33 = float(t4[6]) if len(t4) > 6 else 1.0

        if len(cards) > 4 and not cards[4].is_blank:
            t5 = cards[4].tokens()
            epsr1 = float(t5[0]) if len(t5) > 0 else 0.0
            epsr2 = float(t5[1]) if len(t5) > 1 else 0.0
            epsr3 = float(t5[2]) if len(t5) > 2 else 0.0

        if len(cards) > 5 and not cards[5].is_blank:
            t6 = cards[5].tokens()
            fun_a3 = int(float(t6[0])) if len(t6) > 0 else 0
            fun_b3 = int(float(t6[1])) if len(t6) > 1 else 0
            fun_a4 = int(float(t6[2])) if len(t6) > 2 else 0
            vflag = int(float(t6[3])) if len(t6) > 3 else 0
            fscale12 = float(t6[4]) if len(t6) > 4 else 1.0
            fscale23 = float(t6[5]) if len(t6) > 5 else 1.0
            fscale13 = float(t6[6]) if len(t6) > 6 else 1.0

        if len(cards) > 6 and not cards[6].is_blank:
            t7 = cards[6].tokens()
            epsr4 = float(t7[0]) if len(t7) > 0 else 0.0
            epsr5 = float(t7[1]) if len(t7) > 1 else 0.0
            epsr6 = float(t7[2]) if len(t7) > 2 else 0.0

    m28 = MaterialLaw28(
        id=mat_id, title=title, rho0=rho0, ref_rho=refer_rho,
        e11=e11, e22=e22, e33=e33, g12=g12, g23=g23, g31=g31,
        fun_a1=fun_a1, fun_b1=fun_b1, fun_a2=fun_a2, gflag=gflag,
        fscale11=fscale11, fscale22=fscale22, fscale33=fscale33,
        epsr1=epsr1, epsr2=epsr2, epsr3=epsr3,
        fun_a3=fun_a3, fun_b3=fun_b3, fun_a4=fun_a4, vflag=vflag,
        fscale12=fscale12, fscale23=fscale23, fscale13=fscale13,
        epsr4=epsr4, epsr5=epsr5, epsr6=epsr6,
    )
    model.mat_law28s[mat_id] = m28
    from ...materials.law28_honeycomb import build_law28
    model.materials[mat_id] = build_law28(m28)




def read_mat_law44(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW44`` or ``/MAT/COWPER_SYMONDS`` (M172): Cowper-Symonds strain-rate elastoplastic material model.

    Fortran origin: ``starter/source/materials/mat/mat044/hm_read_mat44.F``.
    """
    from ...model.entities import MaterialLaw44, Material
    mat_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards or cards[0].is_blank:
        log.error(f"/MAT/LAW44/{mat_id}: missing data card", block.source)
        return

    rho0, refer_rho = 0.0, 0.0
    e, nu = 0.0, 0.0
    iflag = 0
    a, b, n = 0.0, 0.0, 0.0
    hard, sig_max = 0.0, 0.0
    src, sre = 0.0, 0.0
    strflag, fsmooth = 0, 0
    fcut = 0.0
    vflag = 0
    eps_max, eta1, eta2 = 0.0, 0.0, 0.0
    yld_func = 0
    yld_scale = 1.0

    if block.fixed:
        f1 = cards[0].cut("MAT_LAW44_1")
        rho0 = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
        refer_rho = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cards[1].cut("MAT_LAW44_2")
            e = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            nu = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            iflag = _ival(f2[2]) if len(f2) > 2 else 0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cards[2].cut("MAT_LAW44_3")
            a = _fval(f3[0], 0.0) if len(f3) > 0 else 0.0
            b = _fval(f3[1], 0.0) if len(f3) > 1 else 0.0
            n = _fval(f3[2], 0.0) if len(f3) > 2 else 0.0
            hard = _fval(f3[3], 0.0) if len(f3) > 3 else 0.0
            sig_max = _fval(f3[4], 0.0) if len(f3) > 4 else 0.0

        if len(cards) > 3 and not cards[3].is_blank:
            f4 = cards[3].cut("MAT_LAW44_4")
            src = _fval(f4[0], 0.0) if len(f4) > 0 else 0.0
            sre = _fval(f4[1], 0.0) if len(f4) > 1 else 0.0
            strflag = _ival(f4[2]) if len(f4) > 2 else 0
            fsmooth = _ival(f4[3]) if len(f4) > 3 else 0
            fcut = _fval(f4[4], 0.0) if len(f4) > 4 else 0.0
            vflag = _ival(f4[5]) if len(f4) > 5 else 0

        if len(cards) > 4 and not cards[4].is_blank:
            f5 = cards[4].cut("MAT_LAW44_5")
            eps_max = _fval(f5[0], 0.0) if len(f5) > 0 else 0.0
            eta1 = _fval(f5[1], 0.0) if len(f5) > 1 else 0.0
            eta2 = _fval(f5[2], 0.0) if len(f5) > 2 else 0.0

        if len(cards) > 5 and not cards[5].is_blank:
            f6 = cards[5].cut("MAT_LAW44_6")
            yld_func = _ival(f6[0]) if len(f6) > 0 else 0
            yld_scale = _fval(f6[1], 1.0) if len(f6) > 1 and f6[1].strip() else 1.0
    else:
        t1 = cards[0].tokens()
        rho0 = float(t1[0]) if len(t1) > 0 else 0.0
        refer_rho = float(t1[1]) if len(t1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            e = float(t2[0]) if len(t2) > 0 else 0.0
            nu = float(t2[1]) if len(t2) > 1 else 0.0
            iflag = int(float(t2[2])) if len(t2) > 2 else 0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            a = float(t3[0]) if len(t3) > 0 else 0.0
            b = float(t3[1]) if len(t3) > 1 else 0.0
            n = float(t3[2]) if len(t3) > 2 else 0.0
            hard = float(t3[3]) if len(t3) > 3 else 0.0
            sig_max = float(t3[4]) if len(t3) > 4 else 0.0

        if len(cards) > 3 and not cards[3].is_blank:
            t4 = cards[3].tokens()
            src = float(t4[0]) if len(t4) > 0 else 0.0
            sre = float(t4[1]) if len(t4) > 1 else 0.0
            strflag = int(float(t4[2])) if len(t4) > 2 else 0
            fsmooth = int(float(t4[3])) if len(t4) > 3 else 0
            fcut = float(t4[4]) if len(t4) > 4 else 0.0
            vflag = int(float(t4[5])) if len(t4) > 5 else 0

        if len(cards) > 4 and not cards[4].is_blank:
            t5 = cards[4].tokens()
            eps_max = float(t5[0]) if len(t5) > 0 else 0.0
            eta1 = float(t5[1]) if len(t5) > 1 else 0.0
            eta2 = float(t5[2]) if len(t5) > 2 else 0.0

        if len(cards) > 5 and not cards[5].is_blank:
            t6 = cards[5].tokens()
            yld_func = int(float(t6[0])) if len(t6) > 0 else 0
            yld_scale = float(t6[1]) if len(t6) > 1 else 1.0

    _INF = 1e30
    cb = b
    if n > 1.0:
        cb = 0.0
    cc = 1.0 / src if src != 0.0 else 0.0
    cp = 1.0 / (sre if sre != 0.0 else 1.0)
    icc = strflag if strflag != 0 else 1
    vflag_val = vflag if vflag != 0 else 2
    epsm = eps_max if eps_max != 0.0 else _INF
    epsr1_val = eta1 if eta1 != 0.0 else _INF
    epsr2_val = eta2 if eta2 != 0.0 else 2.0 * _INF
    sigm = sig_max if sig_max != 0.0 else _INF
    fisokin = hard
    fct = yld_func
    yscale = yld_scale if (yld_scale != 0.0 or fct == 0) else 1.0
    if fct == 0:
        yscale = 0.0
    ca = a
    if fct > 0 and ca != 0.0 and vflag_val != 1:
        ca = 0.0
    ismooth = fsmooth
    fcut_val = fcut
    if vflag_val == 1:
        ismooth, fcut_val = 1, fcut if fcut != 0.0 else 10000.0
    elif fcut_val != 0.0:
        ismooth = 1
    elif ismooth != 0:
        fcut_val = 10000.0
    epsgm = ((sigm - ca) / cb) ** (1.0 / n) if (n != 0.0 and cb != 0.0 and sigm < _INF) else _INF

    m44 = MaterialLaw44(
        id=mat_id, title=title, rho0=rho0, ref_rho=refer_rho,
        e=e, nu=nu, iflag=iflag, a=a, b=b, n=n, hard=hard,
        sig_max=sig_max, src=src, sre=sre, strflag=strflag,
        fsmooth=fsmooth, fcut=fcut, vflag=vflag,
        eps_max=eps_max, eta1=eta1, eta2=eta2,
        yld_func=yld_func, yld_scale=yld_scale,
    )
    model.mat_law44s[mat_id] = m44
    model.materials[mat_id] = Material(
        id=mat_id, law=44, rho0=rho0, title=title,
        params={
            "E": e, "nu": nu, "A": ca, "B": cb, "n": n,
            "cc": cc, "cp": cp, "icc": icc, "vflag": vflag_val,
            "sig_max": sigm, "epsr1": epsr1_val, "epsr2": epsr2_val,
            "epsgm": epsgm, "fisokin": fisokin,
            "yld_fct": fct, "yscale": yscale,
            "ismooth": ismooth, "fcut": fcut_val,
            "eps_p_max": epsm,
            "Iflag": iflag, "a": a, "b": b, "SIGY": a, "HARD": hard,
            "SRC": src, "SRE": sre, "STRFLAG": strflag, "Fsmooth": fsmooth,
            "Fcut": fcut, "Vflag": vflag, "EPS_max": eps_max,
            "ETA1": eta1, "ETA2": eta2, "YLD_FUNC": yld_func, "YLD_SCALE": yld_scale,
        }
    )




def read_mat_law88(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW88/id`` or ``/MAT/HYPER_ELAS/id`` (M173): Tabulated hyperelastic Ogden model."""
    from ...model.entities import MaterialLaw88, Material
    mat_id = block.user_id or 0
    # Initial format probe
    is_fixed = block.fixed
    title, cards = _fixed_data(block) if is_fixed else _title_and_data(block)
    # If block.fixed is True, check whether the actual data cards are free-format tokens
    if is_fixed and cards:
        first_raw = cards[0].raw.rstrip()
        toks0 = cards[0].tokens()
        if any("," in c.raw for c in cards):
            is_fixed = False
        elif len(toks0) > 1 and len(first_raw) < 20:
            is_fixed = False
        elif len(cards) > 1 and len(cards[1].tokens()) >= 3 and len(cards[1].raw.rstrip()) < 50:
            is_fixed = False
        if not is_fixed:
            title, cards = _title_and_data(block)
    elif not is_fixed and cards:
        raw_lines = [c.raw for c in cards if not c.is_blank]
        if any(len(line) >= 50 and "," not in line for line in raw_lines):
            is_fixed = True
            title, cards = _fixed_data(block)

    rho0 = 0.0
    refer_rho = 0.0
    nu = 0.495
    bulk = 0.0
    fcut = 0.0
    fsmooth = 0
    nl = 0
    ifunc_unload = 0
    fscale_unload = 1.0
    hys = 0.0
    shape = 1.0
    tension = 0
    rtype = 0
    func_load_list: list[int] = []
    fscale_load_list: list[float] = []
    rate_load_list: list[float] = []
    lamfit_list: list[float] = []
    sgl = 0.0
    sw = 0.0
    st = 0.0
    g = 0.0
    sigf = 0.0
    kfail = 0.0
    gam1 = 0.0
    gam2 = 0.0
    eh = 0.0
    failip = 0

    def _card_tokens(c: Card) -> list[str]:
        raw = c.raw.strip()
        if "," in raw:
            parts = [p.strip() for p in raw.replace(",", " ").split()]
        else:
            parts = c.tokens()
        return [p.rstrip(",") for p in parts if p]

    if is_fixed:
        if len(cards) > 0 and not cards[0].is_blank:
            f1 = cut(cards[0].raw, "MAT_LAW88_1")
            rho0 = _f(f1[0])
            refer_rho = _f(f1[1]) if len(f1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cut(cards[1].raw, "MAT_LAW88_2")
            nu = _f(f2[0])
            bulk = _f(f2[1]) if len(f2) > 1 else 0.0
            fcut = _f(f2[2]) if len(f2) > 2 else 0.0
            fsmooth = _i(f2[3]) if len(f2) > 3 else 0
            nl = _i(f2[4]) if len(f2) > 4 else 0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cut(cards[2].raw, "MAT_LAW88_3")
            w3 = cards[2].raw[:20].split()
            ifunc_unload = _i(w3[0]) if w3 else 0
            fscale_unload = _f(f3[2]) if len(f3) > 2 and f3[2].strip() else 1.0
            hys = _f(f3[3]) if len(f3) > 3 else 0.0
            shape = _f(f3[4]) if len(f3) > 4 and f3[4].strip() else 1.0
            tension = _i(f3[5]) if len(f3) > 5 else 0
            rtype = _i(f3[6]) if len(f3) > 6 else 0

        idx = 3
        for _ in range(nl):
            if idx < len(cards) and not cards[idx].is_blank:
                fr = cut(cards[idx].raw, "MAT_LAW88_4_ROW")
                wr = cards[idx].raw[:20].split()
                fid = _i(wr[0]) if wr else 0
                fsc = _f(fr[2]) if len(fr) > 2 and fr[2].strip() else 1.0
                frate = _f(fr[3]) if len(fr) > 3 else 0.0
                flam = _f(fr[4]) if len(fr) > 4 else 0.0
                func_load_list.append(fid)
                fscale_load_list.append(fsc)
                rate_load_list.append(frate)
                lamfit_list.append(flam)
            idx += 1

        if idx < len(cards) and not cards[idx].is_blank:
            f5 = cut(cards[idx].raw, "MAT_LAW88_5")
            sgl = _f(f5[0])
            sw = _f(f5[1]) if len(f5) > 1 else 0.0
            st = _f(f5[2]) if len(f5) > 2 else 0.0
            g = _f(f5[3]) if len(f5) > 3 else 0.0
            sigf = _f(f5[4]) if len(f5) > 4 else 0.0
            idx += 1

        if idx < len(cards) and not cards[idx].is_blank:
            f6 = cut(cards[idx].raw, "MAT_LAW88_6")
            kfail = _f(f6[0])
            gam1 = _f(f6[1]) if len(f6) > 1 else 0.0
            gam2 = _f(f6[2]) if len(f6) > 2 else 0.0
            eh = _f(f6[3]) if len(f6) > 3 else 0.0
            failip = _i(f6[5]) if len(f6) > 5 else 0
    else:
        if len(cards) > 0 and not cards[0].is_blank:
            t1 = _card_tokens(cards[0])
            rho0 = float(t1[0]) if len(t1) > 0 else 0.0
            refer_rho = float(t1[1]) if len(t1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = _card_tokens(cards[1])
            nu = float(t2[0]) if len(t2) > 0 else 0.495
            bulk = float(t2[1]) if len(t2) > 1 else 0.0
            fcut = float(t2[2]) if len(t2) > 2 else 0.0
            fsmooth = int(float(t2[3])) if len(t2) > 3 else 0
            nl = int(float(t2[4])) if len(t2) > 4 else 0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = _card_tokens(cards[2])
            if len(t3) >= 7:
                ifunc_unload = int(float(t3[0]))
                fscale_unload = float(t3[2])
                hys = float(t3[3])
                shape = float(t3[4])
                tension = int(float(t3[5]))
                rtype = int(float(t3[6]))
            else:
                ifunc_unload = int(float(t3[0])) if len(t3) > 0 else 0
                fscale_unload = float(t3[1]) if len(t3) > 1 else 1.0
                hys = float(t3[2]) if len(t3) > 2 else 0.0
                shape = float(t3[3]) if len(t3) > 3 else 1.0
                tension = int(float(t3[4])) if len(t3) > 4 else 0
                rtype = int(float(t3[5])) if len(t3) > 5 else 0

        idx = 3
        for _ in range(nl):
            if idx < len(cards) and not cards[idx].is_blank:
                tr = _card_tokens(cards[idx])
                if len(tr) >= 5:
                    fid = int(float(tr[0]))
                    fsc = float(tr[2])
                    frate = float(tr[3])
                    flam = float(tr[4])
                else:
                    fid = int(float(tr[0])) if len(tr) > 0 else 0
                    fsc = float(tr[1]) if len(tr) > 1 else 1.0
                    frate = float(tr[2]) if len(tr) > 2 else 0.0
                    flam = float(tr[3]) if len(tr) > 3 else 0.0
                func_load_list.append(fid)
                fscale_load_list.append(fsc)
                rate_load_list.append(frate)
                lamfit_list.append(flam)
            idx += 1

        if idx < len(cards) and not cards[idx].is_blank:
            t5 = _card_tokens(cards[idx])
            sgl = float(t5[0]) if len(t5) > 0 else 0.0
            sw = float(t5[1]) if len(t5) > 1 else 0.0
            st = float(t5[2]) if len(t5) > 2 else 0.0
            g = float(t5[3]) if len(t5) > 3 else 0.0
            sigf = float(t5[4]) if len(t5) > 4 else 0.0
            idx += 1

        if idx < len(cards) and not cards[idx].is_blank:
            t6 = _card_tokens(cards[idx])
            if len(t6) >= 6:
                kfail = float(t6[0])
                gam1 = float(t6[1])
                gam2 = float(t6[2])
                eh = float(t6[3])
                failip = int(float(t6[5]))
            else:
                kfail = float(t6[0]) if len(t6) > 0 else 0.0
                gam1 = float(t6[1]) if len(t6) > 1 else 0.0
                gam2 = float(t6[2]) if len(t6) > 2 else 0.0
                eh = float(t6[3]) if len(t6) > 3 else 0.0
                failip = int(float(t6[4])) if len(t6) > 4 else 0

    # Specimen dimensions scaling per hm_read_mat88.F90:546-558
    fscale_load_unscaled = list(fscale_load_list)
    if sw > 0.0 and st > 0.0:
        areafac = 1.0 / (sw * st)
        fscale_load_list = [sc * areafac for sc in fscale_load_list]

    beta = 0.0
    if nu <= 0.0:
        beta = abs(nu)
        nu = 0.495
    if shape == 0.0:
        shape = 1.0
    if hys == 0.0:
        hys = 1.0
    if fscale_unload == 0.0:
        fscale_unload = 1.0
    eh = max(min(eh, 1.0), 0.0)
    failip = max(failip, 0)
    if ifunc_unload > 0 and len(func_load_list) > 0 and ifunc_unload == func_load_list[0]:
        ifunc_unload = 0

    # Derive initial moduli per hm_read_mat88.F90:593-619
    e_equiv = 3.0 * bulk * (1.0 - 2.0 * nu) if bulk > 0.0 else 0.0
    if e_equiv <= 0.0 and g > 0.0:
        e_equiv = 2.0 * g * (1.0 + nu)
    if e_equiv <= 0.0:
        e_equiv = 1.0
    if bulk <= 0.0:
        bulk = e_equiv / (3.0 * (1.0 - 2.0 * nu))
    shear_mod = 3.0 * bulk * e_equiv / (9.0 * bulk - e_equiv) if (9.0 * bulk - e_equiv) != 0.0 else 0.0
    if shear_mod < 0.0 or g < 0.0:
        log.warning(f"Material {mat_id} ({title}): shear modulus is negative. [ANCMSG 3109]")
        bulk = 4.0 * (e_equiv / 9.0) * (1.0 + 1e-3)
        shear_mod = 3.0 * bulk * e_equiv / (9.0 * bulk - e_equiv) if (9.0 * bulk - e_equiv) != 0.0 else 0.0

    if g <= 0.0:
        g = shear_mod

    m88 = MaterialLaw88(
        id=mat_id, title=title, rho0=rho0, ref_rho=refer_rho,
        nu=nu, bulk=bulk, fcut=fcut, fsmooth=fsmooth, nl=nl,
        ifunc_unload=ifunc_unload, fscale_unload=fscale_unload,
        hys=hys, shape=shape, tension=tension, rtype=rtype,
        func_load_list=func_load_list, fscale_load_list=fscale_load_list,
        fscale_load_card=fscale_load_unscaled,
        rate_load_list=rate_load_list, lamfit_list=lamfit_list,
        sgl=sgl, sw=sw, st=st, g=g, sigf=sigf,
        kfail=kfail, gam1=gam1, gam2=gam2, eh=eh, failip=failip,
        beta=beta, young=e_equiv, shear=shear_mod,
    )
    model.mat_law88s[mat_id] = m88

    from ..mat_reader import GenericMaterialRecord
    mat88 = Material(
        id=mat_id, law=88, rho0=rho0, title=title,
        params={
            "E": e_equiv, "nu": nu, "young": e_equiv, "shear": shear_mod,
            "LAW88_Nu": nu, "LAW88_K": bulk, "LAW88_Fcut": fcut,
            "LAW88_Fsmooth": fsmooth, "LAW88_NL": nl,
            "LAW88_fct_IDunL": ifunc_unload, "LAW88_FscaleunL": fscale_unload,
            "LAW88_Hys": hys, "LAW88_Shape": shape, "LAW88_Tension": tension,
            "LAW88_RTYPE": rtype, "LAW88_arr1": func_load_list,
            "LAW88_arr2": fscale_load_list, "LAW88_arr3": rate_load_list,
            "LAW88_LAMFIT": lamfit_list,
            "LAW88_SGL": sgl, "LAW88_SW": sw, "LAW88_ST": st,
            "LAW88_G": g, "LAW88_SIGF": sigf, "LAW88_KFAIL": kfail,
            "LAW88_GAM1": gam1, "LAW88_GAM2": gam2, "LAW88_EH": eh,
            "LAW88_FAILIP": failip,
            "bulk": bulk, "K": bulk, "fcut": fcut, "fsmooth": fsmooth,
            "nl": nl, "ifunc_unload": ifunc_unload, "fscale_unload": fscale_unload,
            "hys": hys, "shape": shape, "tension": tension, "rtype": rtype,
            "func_load_list": func_load_list, "fscale_load_list": fscale_load_list,
            "fscale_load_card": fscale_load_unscaled,
            "fscale_load_list_scaled": fscale_load_list,
            "rate_load_list": rate_load_list, "lamfit_list": lamfit_list,
            "sgl": sgl, "sw": sw, "st": st, "g": g, "sigf": sigf,
            "kfail": kfail, "gam1": gam1, "gam2": gam2, "eh": eh, "failip": failip,
            "beta": beta,
        }
    )
    mat88.record = GenericMaterialRecord(
        law_name="LAW88", law_number=88, id=mat_id, title=title,
        params=mat88.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat88




def read_mat_law92(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW92/id`` or ``/MAT/ARRUDA_BOYCE/id`` (M173): Arruda-Boyce 8-chain model."""
    from ...model.entities import MaterialLaw92, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho0 = 0.0
    refer_rho = 0.0
    mu = 0.0
    d = 0.0
    lam = 7.0
    itype = 1
    fct_id = 0
    nu = 0.0
    fscale = 1.0

    if block.fixed:
        if len(cards) > 0 and not cards[0].is_blank:
            f1 = cut(cards[0].raw, "MAT_LAW92_1")
            rho0 = _f(f1[0])
            refer_rho = _f(f1[1]) if len(f1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cut(cards[1].raw, "MAT_LAW92_2")
            mu = _f(f2[0])
            d = _f(f2[1]) if len(f2) > 1 else 0.0
            lam = _f(f2[2]) if len(f2) > 2 else 7.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cut(cards[2].raw, "MAT_LAW92_3")
            itype = _i(f3[0])
            fct_id = _i(f3[1]) if len(f3) > 1 else 0
            nu = _f(f3[2]) if len(f3) > 2 else 0.0
            fscale = _f(f3[3]) if len(f3) > 3 and f3[3].strip() else 1.0
    else:
        if len(cards) > 0 and not cards[0].is_blank:
            t1 = cards[0].tokens()
            rho0 = float(t1[0]) if len(t1) > 0 else 0.0
            refer_rho = float(t1[1]) if len(t1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            mu = float(t2[0]) if len(t2) > 0 else 0.0
            d = float(t2[1]) if len(t2) > 1 else 0.0
            lam = float(t2[2]) if len(t2) > 2 else 7.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            itype = int(float(t3[0])) if len(t3) > 0 else 1
            fct_id = int(float(t3[1])) if len(t3) > 1 else 0
            nu = float(t3[2]) if len(t3) > 2 else 0.0
            fscale = float(t3[3]) if len(t3) > 3 else 1.0

    if itype == 0:
        itype = 1
    if lam == 0.0:
        lam = 7.0
    if fscale == 0.0:
        fscale = 1.0

    m92 = MaterialLaw92(
        id=mat_id, title=title, rho0=rho0, ref_rho=refer_rho,
        mu=mu, d=d, lam=lam, itype=itype, fct_id=fct_id, nu=nu, fscale=fscale,
    )
    model.mat_law92s[mat_id] = m92
    e_equiv = 2.0 * mu * (1.0 + nu) if mu > 0.0 else 0.0
    from ..mat_reader import GenericMaterialRecord
    mat92 = Material(
        id=mat_id, law=92, rho0=rho0, title=title,
        params={
            "rho0": rho0, "ref_rho": refer_rho, "rho_initial": rho0, "RHO0": rho0,
            "E": e_equiv if e_equiv > 0.0 else 1.0, "nu": nu if nu > 0.0 else 0.495,
            "MAT_MUE1": mu, "MAT_D": d, "MAT_Lamda": lam,
            "Itype": itype, "MAT_FCT_IDI": fct_id, "MAT_NU": nu,
            "MAT_FScale": fscale,
            "mu": mu, "d": d, "lam": lam, "itype": itype,
            "fct_id": fct_id, "fscale": fscale,
        }
    )
    mat92.mu = mu
    mat92.d = d
    mat92.lam = lam
    mat92.itype = itype
    mat92.fct_id = fct_id
    mat92.fscale = fscale
    mat92.record = GenericMaterialRecord(
        law_name="LAW92", law_number=92, id=mat_id, title=title,
        params=mat92.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat92




def read_mat_law94(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW94/id`` or ``/MAT/YEOH/id`` (M173): Yeoh hyperelastic model."""
    from ...model.entities import MaterialLaw94, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho0 = 0.0
    refer_rho = 0.0
    c10 = 0.0
    c20 = 0.0
    c30 = 0.0
    d1 = 0.0
    d2 = 0.0
    d3 = 0.0

    valid_cards = [c for c in cards if not c.is_blank]

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW94_1")
            rho0 = _f(f1[0])
            refer_rho = _f(f1[1]) if len(f1) > 1 else 0.0

        val_idx = 1
        if len(valid_cards) > val_idx:
            tokens = valid_cards[val_idx].tokens()
            if len(tokens) >= 3:
                f2 = cut(valid_cards[val_idx].raw, "MAT_LAW94_2")
                c10 = _f(f2[0])
                c20 = _f(f2[1]) if len(f2) > 1 else 0.0
                c30 = _f(f2[2]) if len(f2) > 2 else 0.0
                val_idx += 1
            elif len(valid_cards) > val_idx + 1:
                val_idx += 1
                f2 = cut(valid_cards[val_idx].raw, "MAT_LAW94_2")
                c10 = _f(f2[0])
                c20 = _f(f2[1]) if len(f2) > 1 else 0.0
                c30 = _f(f2[2]) if len(f2) > 2 else 0.0
                val_idx += 1

        if len(valid_cards) > val_idx:
            f3 = cut(valid_cards[val_idx].raw, "MAT_LAW94_3")
            d1 = _f(f3[0])
            d2 = _f(f3[1]) if len(f3) > 1 else 0.0
            d3 = _f(f3[2]) if len(f3) > 2 else 0.0
    else:
        if len(valid_cards) > 0:
            t1 = valid_cards[0].tokens()
            rho0 = float(t1[0]) if len(t1) > 0 else 0.0
            refer_rho = float(t1[1]) if len(t1) > 1 else 0.0

        val_idx = 1
        if len(valid_cards) > val_idx:
            t2 = valid_cards[val_idx].tokens()
            if len(t2) >= 3:
                c10 = float(t2[0]) if len(t2) > 0 else 0.0
                c20 = float(t2[1]) if len(t2) > 1 else 0.0
                c30 = float(t2[2]) if len(t2) > 2 else 0.0
                val_idx += 1
            elif len(valid_cards) > val_idx + 1:
                val_idx += 1
                t2 = valid_cards[val_idx].tokens()
                c10 = float(t2[0]) if len(t2) > 0 else 0.0
                c20 = float(t2[1]) if len(t2) > 1 else 0.0
                c30 = float(t2[2]) if len(t2) > 2 else 0.0
                val_idx += 1

        if len(valid_cards) > val_idx:
            t3 = valid_cards[val_idx].tokens()
            d1 = float(t3[0]) if len(t3) > 0 else 0.0
            d2 = float(t3[1]) if len(t3) > 1 else 0.0
            d3 = float(t3[2]) if len(t3) > 2 else 0.0

    g0 = 2.0 * c10 if c10 > 0.0 else 0.0
    if d1 == 0.0:
        nu_val = 0.495
        rbulk = (2.0 / 3.0) * g0 * (1.0 + nu_val) / max(1e-30, 1.0 - 2.0 * nu_val)
        e_equiv = 2.0 * g0 * (1.0 + nu_val)
    else:
        d1_inv = 1.0 / d1
        rbulk = 2.0 * d1_inv
        denom = 3.0 * rbulk + g0
        nu_val = (3.0 * rbulk - 2.0 * g0) / (2.0 * denom) if denom > 1e-30 else 0.495
        e_equiv = 9.0 * rbulk * g0 / denom if denom > 1e-30 else 2.0 * g0 * (1.0 + nu_val)

    m94 = MaterialLaw94(
        id=mat_id, title=title, rho0=rho0, ref_rho=refer_rho,
        c10=c10, c20=c20, c30=c30, d1=d1, d2=d2, d3=d3,
        nu=nu_val,
    )
    m94.g0 = g0
    m94.rbulk = rbulk
    m94.e = e_equiv

    model.mat_law94s[mat_id] = m94
    if hasattr(model, "mat_yeohs"):
        model.mat_yeohs[mat_id] = m94

    from ..mat_reader import GenericMaterialRecord
    mat94 = Material(
        id=mat_id, law=94, rho0=rho0, title=title,
        params={
            "rho0": rho0, "ref_rho": refer_rho, "rho_initial": rho0, "RHO0": rho0,
            "E": e_equiv if e_equiv > 0.0 else 1.0, "young": e_equiv, "nu": nu_val,
            "G": g0, "G0": g0, "K": rbulk, "bulk": rbulk, "rbulk": rbulk,
            "LAW94_C01": c10, "LAW94_C02": c20, "LAW94_C03": c30,
            "LAW94_D1": d1, "LAW94_D2": d2, "LAW94_D3": d3,
            "C10": c10, "C20": c20, "C30": c30, "D1": d1, "D2": d2, "D3": d3,
            "c10": c10, "c20": c20, "c30": c30, "d1": d1, "d2": d2, "d3": d3,
        }
    )
    mat94.c10 = c10
    mat94.c20 = c20
    mat94.c30 = c30
    mat94.d1 = d1
    mat94.d2 = d2
    mat94.d3 = d3
    mat94.g0 = g0
    mat94.rho0 = rho0
    mat94.ref_rho = refer_rho
    mat94.record = GenericMaterialRecord(
        law_name="LAW94", law_number=94, id=mat_id, title=title,
        params=mat94.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat94



def read_mat_law51(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW51`` or ``/MAT/MULTIMAT`` or ``/MAT/DRUCKER_PRAGER`` (M199): Multi-material / Drucker-Prager brittle model."""
    from ...model.entities import MatLaw51, Material
    mat_id = block.user_id or 1
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if not cards:
        log.error(f"/MAT/LAW51/{mat_id}: missing data card", block.source)
        return

    # Check if cards match Drucker-Prager brittle format (3 cards with 10, 10, 5 entries)
    # or Eulerian multi-material phase format
    is_dp = False
    c0 = cards[0]
    toks0 = c0.tokens() if not block.fixed else [x for x in cut(c0.raw, "MAT_LAW51_DP_1") if x.strip()]
    if len(toks0) >= 3:
        try:
            if float(toks0[0]) > 0.0:
                is_dp = True
        except ValueError:
            pass

    if is_dp:
        if block.fixed:
            f0 = cut(cards[0].raw, "MAT_LAW51_DP_1")
            rho = _fval(f0[0], 0.0) if len(f0) > 0 else 0.0
            e = _fval(f0[1], 0.0) if len(f0) > 1 else 0.0
            nu = _fval(f0[2], 0.0) if len(f0) > 2 else 0.0
            a0 = _fval(f0[3], 0.0) if len(f0) > 3 else 0.0
            a1 = _fval(f0[4], 0.0) if len(f0) > 4 else 0.0
            a2 = _fval(f0[5], 0.0) if len(f0) > 5 else 0.0
            fc = _fval(f0[6], 0.0) if len(f0) > 6 else 0.0
            ft = _fval(f0[7], 0.0) if len(f0) > 7 else 0.0
            fmax = _fval(f0[8], 0.0) if len(f0) > 8 else 0.0
            fres = _fval(f0[9], 0.0) if len(f0) > 9 else 0.0

            f1 = cut(cards[1].raw, "MAT_LAW51_DP_2") if len(cards) > 1 else []
            eps_c = _fval(f1[0], 0.0) if len(f1) > 0 else 0.0
            eps_t = _fval(f1[1], 0.0) if len(f1) > 1 else 0.0
            eps_res = _fval(f1[2], 0.0) if len(f1) > 2 else 0.0
            b = _fval(f1[3], 0.0) if len(f1) > 3 else 0.0
            iflag = _ival(f1[4], 0) if len(f1) > 4 else 0
            icomp = _ival(f1[5], 0) if len(f1) > 5 else 0
            itot = _ival(f1[6], 0) if len(f1) > 6 else 0
            pc = _fval(f1[7], 0.0) if len(f1) > 7 else 0.0
            gamma = _fval(f1[8], 0.0) if len(f1) > 8 else 0.0
            pt = _fval(f1[9], 0.0) if len(f1) > 9 else 0.0

            f2 = cut(cards[2].raw, "MAT_LAW51_DP_3") if len(cards) > 2 else []
            psi = _fval(f2[0], 0.0) if len(f2) > 0 else 0.0
            p0 = _fval(f2[1], 0.0) if len(f2) > 1 else 0.0
            beta = _fval(f2[2], 0.0) if len(f2) > 2 else 0.0
            epsp_max = _fval(f2[3], 0.0) if len(f2) > 3 else 0.0
            fac_e = _fval(f2[4], 0.0) if len(f2) > 4 else 0.0
        else:
            t0 = cards[0].tokens()
            rho = float(t0[0]) if len(t0) > 0 else 0.0
            e = float(t0[1]) if len(t0) > 1 else 0.0
            nu = float(t0[2]) if len(t0) > 2 else 0.0
            a0 = float(t0[3]) if len(t0) > 3 else 0.0
            a1 = float(t0[4]) if len(t0) > 4 else 0.0
            a2 = float(t0[5]) if len(t0) > 5 else 0.0
            fc = float(t0[6]) if len(t0) > 6 else 0.0
            ft = float(t0[7]) if len(t0) > 7 else 0.0
            fmax = float(t0[8]) if len(t0) > 8 else 0.0
            fres = float(t0[9]) if len(t0) > 9 else 0.0

            t1 = cards[1].tokens() if len(cards) > 1 else []
            eps_c = float(t1[0]) if len(t1) > 0 else 0.0
            eps_t = float(t1[1]) if len(t1) > 1 else 0.0
            eps_res = float(t1[2]) if len(t1) > 2 else 0.0
            b = float(t1[3]) if len(t1) > 3 else 0.0
            iflag = int(float(t1[4])) if len(t1) > 4 else 0
            icomp = int(float(t1[5])) if len(t1) > 5 else 0
            itot = int(float(t1[6])) if len(t1) > 6 else 0
            pc = float(t1[7]) if len(t1) > 7 else 0.0
            gamma = float(t1[8]) if len(t1) > 8 else 0.0
            pt = float(t1[9]) if len(t1) > 9 else 0.0

            t2 = cards[2].tokens() if len(cards) > 2 else []
            psi = float(t2[0]) if len(t2) > 0 else 0.0
            p0 = float(t2[1]) if len(t2) > 1 else 0.0
            beta = float(t2[2]) if len(t2) > 2 else 0.0
            epsp_max = float(t2[3]) if len(t2) > 3 else 0.0
            fac_e = float(t2[4]) if len(t2) > 4 else 0.0

        mat = MatLaw51(
            id=mat_id, title=title, rho0=rho, rhor=rho, rho=rho, e=e, nu=nu,
            a0=a0, a1=a1, a2=a2, fc=fc, ft=ft, fmax=fmax, fres=fres,
            eps_c=eps_c, eps_t=eps_t, eps_res=eps_res, b=b,
            iflag=iflag, icomp=icomp, itot=itot, pc=pc, gamma=gamma, pt=pt,
            psi=psi, p0=p0, beta=beta, epsp_max=epsp_max, fac_e=fac_e,
            params={
                "rho": rho, "E": e, "nu": nu, "a0": a0, "a1": a1, "a2": a2,
                "fc": fc, "ft": ft, "fmax": fmax, "fres": fres,
                "eps_c": eps_c, "eps_t": eps_t, "eps_res": eps_res, "b": b,
                "iflag": iflag, "icomp": icomp, "itot": itot, "pc": pc, "gamma": gamma, "pt": pt,
                "psi": psi, "p0": p0, "beta": beta, "epsp_max": epsp_max, "fac_e": fac_e
            }
        )
        model.mat_law51s[mat_id] = mat
        model.materials[mat_id] = Material(
            id=mat_id, law=51, rho0=rho, title=title,
            params={"E": e, "nu": nu, "law51": mat}
        )
        return

    # Eulerian multi-material formulation
    rho0 = 0.0
    refer_rho = 0.0
    iform = 0
    iopt = 0
    pext = 0.0
    nu = 0.0
    lamda = 0.0
    phases = []

    idx = 0
    c0 = cards[idx]
    idx += 1
    if block.fixed:
        f0 = cut(c0.raw, "MAT_LAW51_IFORM")
        val0 = _ival(f0[0])
        if len(c0.raw.rstrip()) <= 20 and val0 in (0, 1, 2, 3, 4, 5, 10):
            iform = val0
            iopt = _ival(f0[1]) if len(f0) > 1 else 0
        else:
            f_rho = cut(c0.raw, "MAT_LAW51_GEN")
            rho0 = _fval(f_rho[0], 0.0) if len(f_rho) > 0 else 0.0
            refer_rho = _fval(f_rho[1], 0.0) if len(f_rho) > 1 else 0.0
            if idx < len(cards):
                c1 = cards[idx]
                idx += 1
                f1 = cut(c1.raw, "MAT_LAW51_IFORM")
                iform = _ival(f1[0]) if len(f1) > 0 else 0
                iopt = _ival(f1[1]) if len(f1) > 1 else 0
    else:
        toks = c0.tokens()
        if len(toks) > 0:
            try:
                val = int(float(toks[0]))
                if len(toks) <= 2 and val in (0, 1, 2, 3, 4, 5, 10):
                    iform = val
                    iopt = int(float(toks[1])) if len(toks) > 1 else 0
                else:
                    rho0 = float(toks[0])
                    refer_rho = float(toks[1]) if len(toks) > 1 else 0.0
                    if idx < len(cards):
                        t1 = cards[idx].tokens()
                        idx += 1
                        iform = int(float(t1[0])) if len(t1) > 0 else 0
                        iopt = int(float(t1[1])) if len(t1) > 1 else 0
            except ValueError:
                pass

    if iform in (0, 1) and idx < len(cards):
        c_gen = cards[idx]
        idx += 1
        if block.fixed:
            fg = cut(c_gen.raw, "MAT_LAW51_GEN")
            pext = _fval(fg[0], 0.0) if len(fg) > 0 else 0.0
            nu = _fval(fg[1], 0.0) if len(fg) > 1 else 0.0
            lamda = _fval(fg[2], 0.0) if len(fg) > 2 else 0.0
        else:
            tg = c_gen.tokens()
            pext = float(tg[0]) if len(tg) > 0 else 0.0
            nu = float(tg[1]) if len(tg) > 1 else 0.0
            lamda = float(tg[2]) if len(tg) > 2 else 0.0

    while idx < len(cards):
        c_p1 = cards[idx]
        idx += 1
        if c_p1.is_blank:
            continue
        if block.fixed:
            fp1 = cut(c_p1.raw, "MAT_LAW51_PHASE_1")
            alpha0 = _fval(fp1[0], 0.0) if len(fp1) > 0 else 0.0
            rho_p = _fval(fp1[1], 0.0) if len(fp1) > 1 else 0.0
            e0 = _fval(fp1[2], 0.0) if len(fp1) > 2 else 0.0
            pmin = _fval(fp1[3], 0.0) if len(fp1) > 3 else 0.0
            c0_p = _fval(fp1[4], 0.0) if len(fp1) > 4 else 0.0
        else:
            tp1 = c_p1.tokens()
            alpha0 = float(tp1[0]) if len(tp1) > 0 else 0.0
            rho_p = float(tp1[1]) if len(tp1) > 1 else 0.0
            e0 = float(tp1[2]) if len(tp1) > 2 else 0.0
            pmin = float(tp1[3]) if len(tp1) > 3 else 0.0
            c0_p = float(tp1[4]) if len(tp1) > 4 else 0.0

        c_coeffs = []
        if iform in (0, 1) and idx < len(cards):
            c_p2 = cards[idx]
            idx += 1
            if block.fixed:
                fp2 = cut(c_p2.raw, "MAT_LAW51_PHASE_2")
                c_coeffs = [_fval(x, 0.0) for x in fp2]
            else:
                tp2 = c_p2.tokens()
                c_coeffs = [float(x) for x in tp2]

        g1, a, b, n = 0.0, 0.0, 0.0, 0.0
        if iform in (0, 1) and idx < len(cards):
            c_p3 = cards[idx]
            idx += 1
            if block.fixed:
                fp3 = cut(c_p3.raw, "MAT_LAW51_PHASE_3")
                g1 = _fval(fp3[0], 0.0) if len(fp3) > 0 else 0.0
                a = _fval(fp3[1], 0.0) if len(fp3) > 1 else 0.0
                b = _fval(fp3[2], 0.0) if len(fp3) > 2 else 0.0
                n = _fval(fp3[3], 0.0) if len(fp3) > 3 else 0.0
            else:
                tp3 = c_p3.tokens()
                g1 = float(tp3[0]) if len(tp3) > 0 else 0.0
                a = float(tp3[1]) if len(tp3) > 1 else 0.0
                b = float(tp3[2]) if len(tp3) > 2 else 0.0
                n = float(tp3[3]) if len(tp3) > 3 else 0.0

        phase_dict = {
            "alpha0": alpha0, "rho0": rho_p, "e0": e0, "pmin": pmin, "c0": c0_p,
            "c_coeffs": c_coeffs, "g1": g1, "a": a, "b": b, "n": n
        }
        phases.append(phase_dict)
        if len(phases) >= 3:
            break

    mat = MatLaw51(
        id=mat_id, title=title, rho0=rho0 if rho0 > 0.0 else (phases[0]["rho0"] if phases else 0.0),
        rhor=refer_rho, iform=iform, pext=pext, nu=nu, lamda=lamda,
        rho=rho0 if rho0 > 0.0 else (phases[0]["rho0"] if phases else 0.0),
        e=phases[0]["e0"] if phases else 0.0,
        params={"iform": iform, "iopt": iopt, "pext": pext, "nu": nu, "lamda": lamda, "phases": phases}
    )
    model.mat_law51s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=51, rho0=mat.rho0, title=title,
        params={"E": mat.e, "nu": mat.nu, "law51": mat, "iform": iform, "phases": phases}
    )





def read_mat_law46(block: KeywordBlock, model: Model, log: MessageLog) -> None:

    """``/MAT/LAW46/id`` or ``/MAT/HYD_VISC/id`` or ``/MAT/LES_FLUID/id`` (M173): Hydrodynamic viscous model."""
    from ...model.entities import MaterialLaw46, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho0 = 0.0
    refer_rho = 0.0
    c = 0.0
    nu = 0.0
    istf = 1
    smag = 1.0
    cps = 0.0

    if block.fixed:
        if len(cards) > 0 and not cards[0].is_blank:
            f1 = cut(cards[0].raw, "MAT_LAW46_1")
            rho0 = _f(f1[0])
            refer_rho = _f(f1[1]) if len(f1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            f2 = cut(cards[1].raw, "MAT_LAW46_2")
            c = _f(f2[0])
            nu = _f(f2[1]) if len(f2) > 1 else 0.0

        if len(cards) > 2 and not cards[2].is_blank:
            f3 = cut(cards[2].raw, "MAT_LAW46_3")
            istf = _i(f3[0])
            smag = _f(f3[1]) if len(f3) > 1 and f3[1].strip() else 1.0
            cps = _f(f3[2]) if len(f3) > 2 else 0.0
    else:
        if len(cards) > 0 and not cards[0].is_blank:
            t1 = cards[0].tokens()
            rho0 = float(t1[0]) if len(t1) > 0 else 0.0
            refer_rho = float(t1[1]) if len(t1) > 1 else 0.0

        if len(cards) > 1 and not cards[1].is_blank:
            t2 = cards[1].tokens()
            c = float(t2[0]) if len(t2) > 0 else 0.0
            nu = float(t2[1]) if len(t2) > 1 else 0.0

        if len(cards) > 2 and not cards[2].is_blank:
            t3 = cards[2].tokens()
            istf = int(float(t3[0])) if len(t3) > 0 else 1
            smag = float(t3[1]) if len(t3) > 1 else 1.0
            cps = float(t3[2]) if len(t3) > 2 else 0.0

    if istf == 0:
        istf = 1
    if smag == 0.0:
        smag = 1.0

    m46 = MaterialLaw46(
        id=mat_id, title=title, rho0=rho0, ref_rho=refer_rho,
        c=c, nu=nu, istf=istf, smag=smag, cps=cps,
    )
    model.mat_law46s[mat_id] = m46
    k_bulk = rho0 * (c ** 2) if (rho0 > 0.0 and c > 0.0) else 1.0
    e_equiv = 3.0 * k_bulk * (1.0 - 2.0 * 0.495)
    from ..mat_reader import GenericMaterialRecord
    mat46 = Material(
        id=mat_id, law=46, rho0=rho0, title=title,
        params={
            "E": e_equiv if e_equiv > 0.0 else 1.0, "nu": 0.495,
            "MAT_C": c, "MAT_NU": nu, "Istf": istf,
            "MAT_C5": smag, "MAT_CO1": cps,
            "c": c, "nu": nu, "istf": istf, "smag": smag, "cps": cps,
        }
    )
    mat46.record = GenericMaterialRecord(
        law_name="LAW46", law_number=46, id=mat_id, title=title,
        params=mat46.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat46




def read_mat_law69(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW69/id`` or ``/MAT/HYP_EXT_COMP/id`` (M173, M550): Hyperelastic extended to compression."""
    from ...model.entities import MaterialLaw69, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho0 = 0.0
    refer_rho = 0.0
    iflag = 1
    fct_id_bulk = 0
    nu = 0.495
    fscale = 1.0
    nip = 2
    icheck = -3
    fct_id_data = 0

    valid_cards = [c for c in cards if not c.is_blank]

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW69_1")
            rho0 = _f(f1[0])
            refer_rho = _f(f1[1]) if len(f1) > 1 and f1[1].strip() else 0.0

        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW69_2")
            iflag = _i(f2[0]) if len(f2) > 0 and f2[0].strip() else 1
            fct_id_bulk = _i(f2[1]) if len(f2) > 1 and f2[1].strip() else 0
            nu = _f(f2[2]) if len(f2) > 2 and f2[2].strip() else 0.495
            fscale = _f(f2[3]) if len(f2) > 3 and f2[3].strip() else 1.0
            nip = _i(f2[4]) if len(f2) > 4 and f2[4].strip() else 2
            if len(f2) > 5 and f2[5].strip():
                icheck = _i(f2[5])
            elif len(valid_cards[1].raw) > 70 and valid_cards[1].raw[70:80].strip():
                icheck = _i(valid_cards[1].raw[70:80].strip())
            else:
                icheck = -3

        if len(valid_cards) > 2:
            f3 = cut(valid_cards[2].raw, "MAT_LAW69_3")
            fct_id_data = _i(f3[0]) if len(f3) > 0 and f3[0].strip() else 0
    else:
        if len(valid_cards) > 0:
            t1 = valid_cards[0].tokens()
            rho0 = float(t1[0]) if len(t1) > 0 else 0.0
            refer_rho = float(t1[1]) if len(t1) > 1 else 0.0

        if len(valid_cards) > 1:
            t2 = valid_cards[1].tokens()
            iflag = int(float(t2[0])) if len(t2) > 0 else 1
            fct_id_bulk = int(float(t2[1])) if len(t2) > 1 else 0
            nu = float(t2[2]) if len(t2) > 2 else 0.495
            fscale = float(t2[3]) if len(t2) > 3 else 1.0
            nip = int(float(t2[4])) if len(t2) > 4 else 2
            icheck = int(float(t2[5])) if len(t2) > 5 else -3

        if len(valid_cards) > 2:
            t3 = valid_cards[2].tokens()
            fct_id_data = int(float(t3[0])) if len(t3) > 0 else 0

    if iflag == 0:
        iflag = 1
    if icheck == 0:
        icheck = -3
    if nip == 0:
        nip = 2
    if fscale == 0.0:
        fscale = 1.0
    if nu == 0.0:
        nu = 0.495

    m69 = MaterialLaw69(
        id=mat_id, title=title, rho0=rho0, ref_rho=refer_rho,
        iflag=iflag, fct_id_bulk=fct_id_bulk, nu=nu, fscale=fscale,
        nip=nip, icheck=icheck, fct_id_data=fct_id_data,
    )
    model.mat_law69s[mat_id] = m69
    from ..mat_reader import GenericMaterialRecord
    mat69 = Material(
        id=mat_id, law=69, rho0=rho0, title=title,
        params={
            "E": 1.0, "nu": nu,
            "MAT_Iflag": iflag, "FUN_A1": fct_id_bulk, "MAT_NU": nu,
            "MAT_FScale": fscale, "NIP": nip, "Gflag": icheck,
            "FUN_B1": fct_id_data,
            "iflag": iflag, "law_id": iflag,
            "fct_id_bulk": fct_id_bulk, "fct_id": fct_id_bulk,
            "nu": nu,
            "fscale": fscale,
            "nip": nip, "n_pair": nip,
            "gflag": icheck, "icheck": icheck,
            "fct_id_data": fct_id_data, "fct_id1": fct_id_data,
            "rho0": rho0, "rho": rho0,
            "rhor": refer_rho, "ref_rho": refer_rho,
        }
    )
    mat69.record = GenericMaterialRecord(
        law_name="LAW69", law_number=69, id=mat_id, title=title,
        params=mat69.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat69




def read_mat_law124(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW124/id`` or ``/MAT/CDPM2/id`` (M174): Concrete Damage Plastic Model 2."""
    from ...model.entities import MaterialLaw124, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho0 = 0.0
    e, nu = 0.0, 0.0
    irate = 0
    fcut = 0.0
    idel = 1
    ecc, qh0, ft, fc, hp = 0.0, 0.0, 0.0, 0.0, 0.0
    ah, bh, ch, dh = 0.0, 0.0, 0.0, 0.0
    as_, bs, df = 0.0, 0.0, 0.0
    dflag, dtype, ireg = 0, 0, 0
    wf, wf1, ft1, efc = 0.0, 0.0, 0.0, 0.0

    valid_cards = [c for c in cards if not c.is_blank]

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW124_1")
            rho0 = _f(f1[0])
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW124_2")
            if len(f2) >= 7:
                e = _f(f2[0])
                nu = _f(f2[1]) if len(f2) > 1 else 0.0
                idel = _i(f2[3]) if len(f2) > 3 and f2[3].strip() else 1
                irate = _i(f2[5]) if len(f2) > 5 else 0
                fcut = _f(f2[6]) if len(f2) > 6 else 0.0
            elif len(f2) >= 5:
                e = _f(f2[0])
                nu = _f(f2[1]) if len(f2) > 1 else 0.0
                irate = _i(f2[3]) if len(f2) > 3 else 0
                fcut = _f(f2[4]) if len(f2) > 4 else 0.0
            else:
                e = _f(f2[0]) if len(f2) > 0 else 0.0
                nu = _f(f2[1]) if len(f2) > 1 else 0.0
        if len(valid_cards) > 2:
            f3 = cut(valid_cards[2].raw, "MAT_LAW124_3")
            ecc = _f(f3[0])
            qh0 = _f(f3[1]) if len(f3) > 1 else 0.0
            ft = _f(f3[2]) if len(f3) > 2 else 0.0
            fc = _f(f3[3]) if len(f3) > 3 else 0.0
            hp = _f(f3[4]) if len(f3) > 4 else 0.0
        if len(valid_cards) > 3:
            f4 = cut(valid_cards[3].raw, "MAT_LAW124_4")
            ah = _f(f4[0])
            bh = _f(f4[1]) if len(f4) > 1 else 0.0
            ch = _f(f4[2]) if len(f4) > 2 else 0.0
            dh = _f(f4[3]) if len(f4) > 3 else 0.0
        if len(valid_cards) > 4:
            f5 = cut(valid_cards[4].raw, "MAT_LAW124_5")
            as_ = _f(f5[0])
            bs = _f(f5[1]) if len(f5) > 1 else 0.0
            df = _f(f5[2]) if len(f5) > 2 else 0.0
            dflag = _i(f5[4]) if len(f5) > 4 else 0
            dtype = _i(f5[5]) if len(f5) > 5 else 0
            ireg = _i(f5[6]) if len(f5) > 6 else 0
        if len(valid_cards) > 5:
            f6 = cut(valid_cards[5].raw, "MAT_LAW124_6")
            wf = _f(f6[0])
            wf1 = _f(f6[1]) if len(f6) > 1 else 0.0
            ft1 = _f(f6[2]) if len(f6) > 2 else 0.0
            efc = _f(f6[3]) if len(f6) > 3 else 0.0
    else:
        if len(valid_cards) > 0:
            t1 = valid_cards[0].tokens()
            rho0 = float(t1[0]) if len(t1) > 0 else 0.0
        if len(valid_cards) > 1:
            t2 = valid_cards[1].tokens()
            e = float(t2[0]) if len(t2) > 0 else 0.0
            nu = float(t2[1]) if len(t2) > 1 else 0.0
            if len(t2) >= 5:
                idel = int(float(t2[2]))
                irate = int(float(t2[3]))
                fcut = float(t2[4])
            elif len(t2) >= 4:
                irate = int(float(t2[2]))
                fcut = float(t2[3])
            elif len(t2) >= 3:
                irate = int(float(t2[2]))
        if len(valid_cards) > 2:
            t3 = valid_cards[2].tokens()
            ecc = float(t3[0]) if len(t3) > 0 else 0.0
            qh0 = float(t3[1]) if len(t3) > 1 else 0.0
            ft = float(t3[2]) if len(t3) > 2 else 0.0
            fc = float(t3[3]) if len(t3) > 3 else 0.0
            hp = float(t3[4]) if len(t3) > 4 else 0.0
        if len(valid_cards) > 3:
            t4 = valid_cards[3].tokens()
            ah = float(t4[0]) if len(t4) > 0 else 0.0
            bh = float(t4[1]) if len(t4) > 1 else 0.0
            ch = float(t4[2]) if len(t4) > 2 else 0.0
            dh = float(t4[3]) if len(t4) > 3 else 0.0
        if len(valid_cards) > 4:
            t5 = valid_cards[4].tokens()
            as_ = float(t5[0]) if len(t5) > 0 else 0.0
            bs = float(t5[1]) if len(t5) > 1 else 0.0
            df = float(t5[2]) if len(t5) > 2 else 0.0
            dflag = int(float(t5[3])) if len(t5) > 3 else 0
            dtype = int(float(t5[4])) if len(t5) > 4 else 0
            ireg = int(float(t5[5])) if len(t5) > 5 else 0
        if len(valid_cards) > 5:
            t6 = valid_cards[5].tokens()
            wf = float(t6[0]) if len(t6) > 0 else 0.0
            wf1 = float(t6[1]) if len(t6) > 1 else 0.0
            ft1 = float(t6[2]) if len(t6) > 2 else 0.0
            efc = float(t6[3]) if len(t6) > 3 else 0.0

    m124 = MaterialLaw124(
        id=mat_id, title=title, rho0=rho0, rho=rho0, e=e, nu=nu, irate=irate, fcut=fcut,
        ecc=ecc, qh0=qh0, ft=ft, fc=fc, hp=hp,
        ah=ah, bh=bh, ch=ch, dh=dh, as_=as_, bs=bs, df=df,
        dflag=dflag, dtype=dtype, ireg=ireg, wf=wf, wf1=wf1, ft1=ft1, efc=efc,
    )
    model.mat_law124s[mat_id] = m124
    model.mat_cdpm2s[mat_id] = m124
    from ..mat_reader import GenericMaterialRecord
    mat124 = Material(
        id=mat_id, law=124, rho0=rho0, title=title,
        params={
            "E": e if e > 0.0 else 1.0, "nu": nu if 0.0 <= nu < 0.5 else 0.2,
            "MAT_E": e, "MAT_NU": nu, "IRATE": irate, "FCUT": fcut, "IDEL": idel,
            "MAT_ECC": ecc, "MAT_QH0": qh0, "MAT_FT": ft, "MAT_FC": fc, "MAT_HP": hp,
            "MAT_AH": ah, "MAT_BH": bh, "MAT_CH": ch, "MAT_DH": dh,
            "MAT_AS": as_, "MAT_BS": bs, "MAT_DF": df, "DFLAG": dflag, "DTYPE": dtype, "IREG": ireg,
            "MAT_WF": wf, "MAT_WF1": wf1, "MAT_FT1": ft1, "MAT_EFC": efc,
            "e": e, "nu": nu, "irate": irate, "fcut": fcut, "idel": idel,
            "ecc": ecc, "qh0": qh0, "ft": ft, "fc": fc, "hp": hp,
            "ah": ah, "bh": bh, "ch": ch, "dh": dh,
            "as": as_, "bs": bs, "df": df, "dflag": dflag, "dtype": dtype, "ireg": ireg,
            "wf": wf, "wf1": wf1, "ft1": ft1, "efc": efc,
        }
    )
    mat124.record = GenericMaterialRecord(
        law_name="LAW124", law_number=124, id=mat_id, title=title,
        params=mat124.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat124




def read_mat_law126(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW126/id`` or ``/MAT/JOHNSON_HOLMQUIST_CONCRETE/id`` (M174): Johnson-Holmquist concrete damage model."""
    from ...model.entities import MaterialLaw126, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho0 = 0.0
    g = 0.0
    a, b, n, fc, t0 = 0.0, 0.0, 0.0, 0.0, 0.0
    c, eps0, fcut, sfmax, efmin = 0.0, 0.0, 0.0, 0.0, 0.0
    pc, muc, pl, mul = 0.0, 0.0, 0.0, 0.0
    k1, k2, k3 = 0.0, 0.0, 0.0
    d1, d2 = 0.0, 0.0
    idel = 0
    eps_max = 0.0
    ifailso = 0
    ct, powt, cc, powc = 0.0, 0.0, 0.0, 0.0

    valid_cards = [c for c in cards if not c.is_blank]

    parsed = False
    if block.fixed:
        try:
            if len(valid_cards) > 0:
                f1 = cut(valid_cards[0].raw, "MAT_LAW126_1")
                rho0 = _f(f1[0])
            if len(valid_cards) > 1:
                f2 = cut(valid_cards[1].raw, "MAT_LAW126_2")
                g = _f(f2[0])
            if len(valid_cards) > 2:
                f3 = cut(valid_cards[2].raw, "MAT_LAW126_3")
                a = _f(f3[0])
                b = _f(f3[1]) if len(f3) > 1 else 0.0
                n = _f(f3[2]) if len(f3) > 2 else 0.0
                fc = _f(f3[3]) if len(f3) > 3 else 0.0
                t0 = _f(f3[4]) if len(f3) > 4 else 0.0
            if len(valid_cards) > 3:
                f4 = cut(valid_cards[3].raw, "MAT_LAW126_4")
                c = _f(f4[0])
                eps0 = _f(f4[1]) if len(f4) > 1 else 0.0
                fcut = _f(f4[2]) if len(f4) > 2 else 0.0
                sfmax = _f(f4[3]) if len(f4) > 3 else 0.0
                efmin = _f(f4[4]) if len(f4) > 4 else 0.0
            if len(valid_cards) > 4:
                f5 = cut(valid_cards[4].raw, "MAT_LAW126_5")
                pc = _f(f5[0])
                muc = _f(f5[1]) if len(f5) > 1 else 0.0
                pl = _f(f5[2]) if len(f5) > 2 else 0.0
                mul = _f(f5[3]) if len(f5) > 3 else 0.0
            if len(valid_cards) > 5:
                f6 = cut(valid_cards[5].raw, "MAT_LAW126_6")
                k1 = _f(f6[0])
                k2 = _f(f6[1]) if len(f6) > 1 else 0.0
                k3 = _f(f6[2]) if len(f6) > 2 else 0.0
            if len(valid_cards) > 6:
                f7 = cut(valid_cards[6].raw, "MAT_LAW126_7")
                d1 = _f(f7[0])
                d2 = _f(f7[1]) if len(f7) > 1 else 0.0
                idel = _i(f7[3]) if len(f7) > 3 else 0
                eps_max = _f(f7[4]) if len(f7) > 4 else 0.0
                ifailso = _i(f7[6]) if len(f7) > 6 else 0
            if len(valid_cards) > 7:
                f8 = cut(valid_cards[7].raw, "MAT_LAW126_8")
                ct = _f(f8[0])
                powt = _f(f8[1]) if len(f8) > 1 else 0.0
                cc = _f(f8[2]) if len(f8) > 2 else 0.0
                powc = _f(f8[3]) if len(f8) > 3 else 0.0
            parsed = True
        except (ValueError, IndexError):
            parsed = False

    if not parsed:
        if len(valid_cards) > 0:
            t1 = valid_cards[0].tokens()
            rho0 = float(t1[0]) if len(t1) > 0 else 0.0
        offset = 0
        if len(valid_cards) > 1:
            t2 = valid_cards[1].tokens()
            if len(t2) >= 6:
                g = float(t2[0])
                a = float(t2[1])
                b = float(t2[2])
                n = float(t2[3])
                fc = float(t2[4])
                t0 = float(t2[5])
                offset = 1
            else:
                g = float(t2[0]) if len(t2) > 0 else 0.0
        if offset == 0 and len(valid_cards) > 2:
            t3 = valid_cards[2].tokens()
            a = float(t3[0]) if len(t3) > 0 else 0.0
            b = float(t3[1]) if len(t3) > 1 else 0.0
            n = float(t3[2]) if len(t3) > 2 else 0.0
            fc = float(t3[3]) if len(t3) > 3 else 0.0
            t0 = float(t3[4]) if len(t3) > 4 else 0.0

        c_idx = 3 - offset
        if len(valid_cards) > c_idx:
            t4 = valid_cards[c_idx].tokens()
            c = float(t4[0]) if len(t4) > 0 else 0.0
            eps0 = float(t4[1]) if len(t4) > 1 else 0.0
            fcut = float(t4[2]) if len(t4) > 2 else 0.0
            sfmax = float(t4[3]) if len(t4) > 3 else 0.0
            efmin = float(t4[4]) if len(t4) > 4 else 0.0
        c_idx += 1
        if len(valid_cards) > c_idx:
            t5 = valid_cards[c_idx].tokens()
            pc = float(t5[0]) if len(t5) > 0 else 0.0
            muc = float(t5[1]) if len(t5) > 1 else 0.0
            pl = float(t5[2]) if len(t5) > 2 else 0.0
            mul = float(t5[3]) if len(t5) > 3 else 0.0
        c_idx += 1
        if len(valid_cards) > c_idx:
            t6 = valid_cards[c_idx].tokens()
            k1 = float(t6[0]) if len(t6) > 0 else 0.0
            k2 = float(t6[1]) if len(t6) > 1 else 0.0
            k3 = float(t6[2]) if len(t6) > 2 else 0.0
        c_idx += 1
        if len(valid_cards) > c_idx:
            t7 = valid_cards[c_idx].tokens()
            d1 = float(t7[0]) if len(t7) > 0 else 0.0
            d2 = float(t7[1]) if len(t7) > 1 else 0.0
            idel = int(float(t7[2])) if len(t7) > 2 else 0
            eps_max = float(t7[3]) if len(t7) > 3 else 0.0
            ifailso = int(float(t7[4])) if len(t7) > 4 else 0
        c_idx += 1
        if len(valid_cards) > c_idx:
            t8 = valid_cards[c_idx].tokens()
            ct = float(t8[0]) if len(t8) > 0 else 0.0
            powt = float(t8[1]) if len(t8) > 1 else 0.0
            cc = float(t8[2]) if len(t8) > 2 else 0.0
            powc = float(t8[3]) if len(t8) > 3 else 0.0

    m126 = MaterialLaw126(
        id=mat_id, title=title, rho0=rho0, g=g, a=a, b=b, n=n, fc=fc, t0=t0,
        c=c, eps0=eps0, fcut=fcut, sfmax=sfmax, efmin=efmin,
        pc=pc, muc=muc, pl=pl, mul=mul, k1=k1, k2=k2, k3=k3,
        d1=d1, d2=d2, idel=idel, eps_max=eps_max, ifailso=ifailso,
        ct=ct, powt=powt, cc=cc, powc=powc,
    )
    model.mat_law126s[mat_id] = m126
    nu_est = (3.0 * k1 - 2.0 * g) / (2.0 * (3.0 * k1 + g)) if (k1 > 0.0 and g > 0.0) else 0.2
    if not (0.0 <= nu_est < 0.5):
        nu_est = 0.2
    e_equiv = 2.0 * g * (1.0 + nu_est) if g > 0.0 else (3.0 * k1 * (1.0 - 2.0 * nu_est) if k1 > 0.0 else 1.0)
    from ..mat_reader import GenericMaterialRecord
    mat126 = Material(
        id=mat_id, law=126, rho0=rho0, title=title,
        params={
            "E": e_equiv if e_equiv > 0.0 else 1.0, "nu": nu_est, "G": g,
            "MAT_G": g, "MAT_A": a, "MAT_B": b, "MAT_N": n, "MAT_FC": fc, "MAT_T0": t0,
            "MAT_C": c, "MAT_EPS0": eps0, "MAT_FCUT": fcut, "MAT_SFMAX": sfmax, "MAT_EFMIN": efmin,
            "MAT_PC": pc, "MAT_MUC": muc, "MAT_PL": pl, "MAT_MUL": mul,
            "MAT_K1": k1, "MAT_K2": k2, "MAT_K3": k3,
            "MAT_D1": d1, "MAT_D2": d2, "IDEL": idel, "MAT_EPSMAX": eps_max, "IFAILSO": ifailso,
            "MAT_CT": ct, "MAT_POWT": powt, "MAT_CC": cc, "MAT_POWC": powc,
            "g": g, "a": a, "b": b, "n": n, "fc": fc, "t0": t0,
            "c": c, "eps0": eps0, "fcut": fcut, "sfmax": sfmax, "efmin": efmin,
            "pc": pc, "muc": muc, "pl": pl, "mul": mul,
            "k1": k1, "k2": k2, "k3": k3,
            "d1": d1, "d2": d2, "idel": idel, "eps_max": eps_max, "ifailso": ifailso,
            "ct": ct, "powt": powt, "cc": cc, "powc": powc,
        }
    )
    from ...materials.law126_hjc import build_law126
    mat126 = build_law126(mat126)
    mat126.record = GenericMaterialRecord(
        law_name="LAW126", law_number=126, id=mat_id, title=title,
        params=mat126.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat126




def read_mat_law125(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW125/id`` or ``/MAT/LAMINATED_COMPOSITE/id`` (M174): Multi-layered laminated composite model."""
    from ...model.entities import MaterialLaw125, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho0 = 0.0
    ea, eb, ec = 0.0, 0.0, 0.0
    ifail = 0
    gab, gca, gbc = 0.0, 0.0, 0.0
    prba, prca, prcb = 0.0, 0.0, 0.0
    lce11t, e11t, fct_t11, t11, slimt11 = 0, 0.0, 0, 0.0, 0.0
    lce11c, e11c, fct_c11, c11, slimc11 = 0, 0.0, 0, 0.0, 0.0
    lce22t, e22t, fct_t22, t22, slimt22 = 0, 0.0, 0, 0.0, 0.0
    lce22c, e22c, fct_c22, c22, slimc22 = 0, 0.0, 0, 0.0, 0.0
    lce33t, e33t, fct_t33, t33, slimt33 = 0, 0.0, 0, 0.0, 0.0
    lce33c, e33c, fct_c33, c33, slimc33 = 0, 0.0, 0, 0.0, 0.0
    g12a, t12a, g12b, t12b, slims12 = 0.0, 0.0, 0.0, 0.0, 0.0
    fct_g12a, fct_t12a, fct_g12b, fct_t12b = 0, 0, 0, 0
    g31a, t31a, g31b, t31b, slims31 = 0.0, 0.0, 0.0, 0.0, 0.0
    fct_g31a, fct_t31a, fct_g31b, fct_t31b = 0, 0, 0, 0
    g23a, t23a, g23b, t23b, slims23 = 0.0, 0.0, 0.0, 0.0, 0.0
    fct_g23a, fct_t23a, fct_g23b, fct_t23b = 0, 0, 0, 0
    epsf, epsr, dmax = 0.0, 0.0, 0.0
    fct_fail, fail = 0, 0.0
    fcut = 0.0

    valid_cards = [c for c in cards if not c.is_blank]

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW125_1")
            rho0 = _f(f1[0])
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW125_2")
            ea = _f(f2[0])
            eb = _f(f2[1]) if len(f2) > 1 else 0.0
            ec = _f(f2[2]) if len(f2) > 2 else 0.0
            ifail = _i(f2[4]) if len(f2) > 4 else 0
        if len(valid_cards) > 2:
            f3 = cut(valid_cards[2].raw, "MAT_LAW125_3")
            gab = _f(f3[0])
            gca = _f(f3[1]) if len(f3) > 1 else 0.0
            gbc = _f(f3[2]) if len(f3) > 2 else 0.0
        if len(valid_cards) > 3:
            f4 = cut(valid_cards[3].raw, "MAT_LAW125_4")
            prba = _f(f4[0])
            prca = _f(f4[1]) if len(f4) > 1 else 0.0
            prcb = _f(f4[2]) if len(f4) > 2 else 0.0
        if len(valid_cards) > 4:
            f5 = cut(valid_cards[4].raw, "MAT_LAW125_5")
            lce11t = _i(f5[0])
            e11t = _f(f5[1]) if len(f5) > 1 else 0.0
            fct_t11 = _i(f5[2]) if len(f5) > 2 else 0
            t11 = _f(f5[3]) if len(f5) > 3 else 0.0
            slimt11 = _f(f5[4]) if len(f5) > 4 else 0.0
        if len(valid_cards) > 5:
            f6 = cut(valid_cards[5].raw, "MAT_LAW125_6")
            lce11c = _i(f6[0])
            e11c = _f(f6[1]) if len(f6) > 1 else 0.0
            fct_c11 = _i(f6[2]) if len(f6) > 2 else 0
            c11 = _f(f6[3]) if len(f6) > 3 else 0.0
            slimc11 = _f(f6[4]) if len(f6) > 4 else 0.0
        if len(valid_cards) > 6:
            f7 = cut(valid_cards[6].raw, "MAT_LAW125_7")
            lce22t = _i(f7[0])
            e22t = _f(f7[1]) if len(f7) > 1 else 0.0
            fct_t22 = _i(f7[2]) if len(f7) > 2 else 0
            t22 = _f(f7[3]) if len(f7) > 3 else 0.0
            slimt22 = _f(f7[4]) if len(f7) > 4 else 0.0
        if len(valid_cards) > 7:
            f8 = cut(valid_cards[7].raw, "MAT_LAW125_8")
            lce22c = _i(f8[0])
            e22c = _f(f8[1]) if len(f8) > 1 else 0.0
            fct_c22 = _i(f8[2]) if len(f8) > 2 else 0
            c22 = _f(f8[3]) if len(f8) > 3 else 0.0
            slimc22 = _f(f8[4]) if len(f8) > 4 else 0.0
        if len(valid_cards) > 8:
            f9 = cut(valid_cards[8].raw, "MAT_LAW125_9")
            lce33t = _i(f9[0])
            e33t = _f(f9[1]) if len(f9) > 1 else 0.0
            fct_t33 = _i(f9[2]) if len(f9) > 2 else 0
            t33 = _f(f9[3]) if len(f9) > 3 else 0.0
            slimt33 = _f(f9[4]) if len(f9) > 4 else 0.0
        if len(valid_cards) > 9:
            f10 = cut(valid_cards[9].raw, "MAT_LAW125_10")
            lce33c = _i(f10[0])
            e33c = _f(f10[1]) if len(f10) > 1 else 0.0
            fct_c33 = _i(f10[2]) if len(f10) > 2 else 0
            c33 = _f(f10[3]) if len(f10) > 3 else 0.0
            slimc33 = _f(f10[4]) if len(f10) > 4 else 0.0
        if len(valid_cards) > 10:
            f11 = cut(valid_cards[10].raw, "MAT_LAW125_11")
            g12a = _f(f11[0])
            t12a = _f(f11[1]) if len(f11) > 1 else 0.0
            g12b = _f(f11[2]) if len(f11) > 2 else 0.0
            t12b = _f(f11[3]) if len(f11) > 3 else 0.0
            slims12 = _f(f11[4]) if len(f11) > 4 else 0.0
        if len(valid_cards) > 11:
            f12 = cut(valid_cards[11].raw, "MAT_LAW125_12")
            fct_g12a = _i(f12[0])
            fct_t12a = _i(f12[1]) if len(f12) > 1 else 0
            fct_g12b = _i(f12[2]) if len(f12) > 2 else 0
            fct_t12b = _i(f12[3]) if len(f12) > 3 else 0
        if len(valid_cards) > 12:
            f13 = cut(valid_cards[12].raw, "MAT_LAW125_13")
            g31a = _f(f13[0])
            t31a = _f(f13[1]) if len(f13) > 1 else 0.0
            g31b = _f(f13[2]) if len(f13) > 2 else 0.0
            t31b = _f(f13[3]) if len(f13) > 3 else 0.0
            slims31 = _f(f13[4]) if len(f13) > 4 else 0.0
        if len(valid_cards) > 13:
            f14 = cut(valid_cards[13].raw, "MAT_LAW125_14")
            fct_g31a = _i(f14[0])
            fct_t31a = _i(f14[1]) if len(f14) > 1 else 0
            fct_g31b = _i(f14[2]) if len(f14) > 2 else 0
            fct_t31b = _i(f14[3]) if len(f14) > 3 else 0
        if len(valid_cards) > 14:
            f15 = cut(valid_cards[14].raw, "MAT_LAW125_15")
            g23a = _f(f15[0])
            t23a = _f(f15[1]) if len(f15) > 1 else 0.0
            g23b = _f(f15[2]) if len(f15) > 2 else 0.0
            t23b = _f(f15[3]) if len(f15) > 3 else 0.0
            slims23 = _f(f15[4]) if len(f15) > 4 else 0.0
        if len(valid_cards) > 15:
            f16 = cut(valid_cards[15].raw, "MAT_LAW125_16")
            fct_g23a = _i(f16[0])
            fct_t23a = _i(f16[1]) if len(f16) > 1 else 0
            fct_g23b = _i(f16[2]) if len(f16) > 2 else 0
            fct_t23b = _i(f16[3]) if len(f16) > 3 else 0
        if len(valid_cards) > 16:
            f17 = cut(valid_cards[16].raw, "MAT_LAW125_17")
            epsf = _f(f17[0])
            epsr = _f(f17[1]) if len(f17) > 1 else 0.0
            dmax = _f(f17[2]) if len(f17) > 2 else 0.0
        if len(valid_cards) > 17:
            f18 = cut(valid_cards[17].raw, "MAT_LAW125_18")
            fct_fail = _i(f18[0])
            fail = _f(f18[1]) if len(f18) > 1 else 0.0
        if len(valid_cards) > 18:
            f19 = cut(valid_cards[18].raw, "MAT_LAW125_19")
            fcut = _f(f19[0])
    else:
        if len(valid_cards) > 0:
            toks1 = valid_cards[0].tokens()
            rho0 = float(toks1[0]) if len(toks1) > 0 else 0.0
        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            ea = float(toks2[0]) if len(toks2) > 0 else 0.0
            eb = float(toks2[1]) if len(toks2) > 1 else 0.0
            ec = float(toks2[2]) if len(toks2) > 2 else 0.0
            ifail = int(float(toks2[3])) if len(toks2) > 3 else 0
        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            gab = float(toks3[0]) if len(toks3) > 0 else 0.0
            gca = float(toks3[1]) if len(toks3) > 1 else 0.0
            gbc = float(toks3[2]) if len(toks3) > 2 else 0.0
        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            prba = float(toks4[0]) if len(toks4) > 0 else 0.0
            prca = float(toks4[1]) if len(toks4) > 1 else 0.0
            prcb = float(toks4[2]) if len(toks4) > 2 else 0.0
        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            lce11t = int(float(toks5[0])) if len(toks5) > 0 else 0
            e11t = float(toks5[1]) if len(toks5) > 1 else 0.0
            fct_t11 = int(float(toks5[2])) if len(toks5) > 2 else 0
            t11 = float(toks5[3]) if len(toks5) > 3 else 0.0
            slimt11 = float(toks5[4]) if len(toks5) > 4 else 0.0
        if len(valid_cards) > 5:
            toks6 = valid_cards[5].tokens()
            lce11c = int(float(toks6[0])) if len(toks6) > 0 else 0
            e11c = float(toks6[1]) if len(toks6) > 1 else 0.0
            fct_c11 = int(float(toks6[2])) if len(toks6) > 2 else 0
            c11 = float(toks6[3]) if len(toks6) > 3 else 0.0
            slimc11 = float(toks6[4]) if len(toks6) > 4 else 0.0
        if len(valid_cards) > 6:
            toks7 = valid_cards[6].tokens()
            lce22t = int(float(toks7[0])) if len(toks7) > 0 else 0
            e22t = float(toks7[1]) if len(toks7) > 1 else 0.0
            fct_t22 = int(float(toks7[2])) if len(toks7) > 2 else 0
            t22 = float(toks7[3]) if len(toks7) > 3 else 0.0
            slimt22 = float(toks7[4]) if len(toks7) > 4 else 0.0
        if len(valid_cards) > 7:
            toks8 = valid_cards[7].tokens()
            lce22c = int(float(toks8[0])) if len(toks8) > 0 else 0
            e22c = float(toks8[1]) if len(toks8) > 1 else 0.0
            fct_c22 = int(float(toks8[2])) if len(toks8) > 2 else 0
            c22 = float(toks8[3]) if len(toks8) > 3 else 0.0
            slimc22 = float(toks8[4]) if len(toks8) > 4 else 0.0
        if len(valid_cards) > 8:
            toks9 = valid_cards[8].tokens()
            lce33t = int(float(toks9[0])) if len(toks9) > 0 else 0
            e33t = float(toks9[1]) if len(toks9) > 1 else 0.0
            fct_t33 = int(float(toks9[2])) if len(toks9) > 2 else 0
            t33 = float(toks9[3]) if len(toks9) > 3 else 0.0
            slimt33 = float(toks9[4]) if len(toks9) > 4 else 0.0
        if len(valid_cards) > 9:
            toks10 = valid_cards[9].tokens()
            lce33c = int(float(toks10[0])) if len(toks10) > 0 else 0
            e33c = float(toks10[1]) if len(toks10) > 1 else 0.0
            fct_c33 = int(float(toks10[2])) if len(toks10) > 2 else 0
            c33 = float(toks10[3]) if len(toks10) > 3 else 0.0
            slimc33 = float(toks10[4]) if len(toks10) > 4 else 0.0
        if len(valid_cards) > 10:
            toks11 = valid_cards[10].tokens()
            g12a = float(toks11[0]) if len(toks11) > 0 else 0.0
            t12a = float(toks11[1]) if len(toks11) > 1 else 0.0
            g12b = float(toks11[2]) if len(toks11) > 2 else 0.0
            t12b = float(toks11[3]) if len(toks11) > 3 else 0.0
            slims12 = float(toks11[4]) if len(toks11) > 4 else 0.0
        if len(valid_cards) > 11:
            toks12 = valid_cards[11].tokens()
            fct_g12a = int(float(toks12[0])) if len(toks12) > 0 else 0
            fct_t12a = int(float(toks12[1])) if len(toks12) > 1 else 0
            fct_g12b = int(float(toks12[2])) if len(toks12) > 2 else 0
            fct_t12b = int(float(toks12[3])) if len(toks12) > 3 else 0
        if len(valid_cards) > 12:
            toks13 = valid_cards[12].tokens()
            g31a = float(toks13[0]) if len(toks13) > 0 else 0.0
            t31a = float(toks13[1]) if len(toks13) > 1 else 0.0
            g31b = float(toks13[2]) if len(toks13) > 2 else 0.0
            t31b = float(toks13[3]) if len(toks13) > 3 else 0.0
            slims31 = float(toks13[4]) if len(toks13) > 4 else 0.0
        if len(valid_cards) > 13:
            toks14 = valid_cards[13].tokens()
            fct_g31a = int(float(toks14[0])) if len(toks14) > 0 else 0
            fct_t31a = int(float(toks14[1])) if len(toks14) > 1 else 0
            fct_g31b = int(float(toks14[2])) if len(toks14) > 2 else 0
            fct_t31b = int(float(toks14[3])) if len(toks14) > 3 else 0
        if len(valid_cards) > 14:
            toks15 = valid_cards[14].tokens()
            g23a = float(toks15[0]) if len(toks15) > 0 else 0.0
            t23a = float(toks15[1]) if len(toks15) > 1 else 0.0
            g23b = float(toks15[2]) if len(toks15) > 2 else 0.0
            t23b = float(toks15[3]) if len(toks15) > 3 else 0.0
            slims23 = float(toks15[4]) if len(toks15) > 4 else 0.0
        if len(valid_cards) > 15:
            toks16 = valid_cards[15].tokens()
            fct_g23a = int(float(toks16[0])) if len(toks16) > 0 else 0
            fct_t23a = int(float(toks16[1])) if len(toks16) > 1 else 0
            fct_g23b = int(float(toks16[2])) if len(toks16) > 2 else 0
            fct_t23b = int(float(toks16[3])) if len(toks16) > 3 else 0
        if len(valid_cards) > 16:
            toks17 = valid_cards[16].tokens()
            epsf = float(toks17[0]) if len(toks17) > 0 else 0.0
            epsr = float(toks17[1]) if len(toks17) > 1 else 0.0
            dmax = float(toks17[2]) if len(toks17) > 2 else 0.0
        if len(valid_cards) > 17:
            toks18 = valid_cards[17].tokens()
            fct_fail = int(float(toks18[0])) if len(toks18) > 0 else 0
            fail = float(toks18[1]) if len(toks18) > 1 else 0.0
        if len(valid_cards) > 18:
            toks19 = valid_cards[18].tokens()
            fcut = float(toks19[0]) if len(toks19) > 0 else 0.0

    m125 = MaterialLaw125(
        id=mat_id, title=title, rho0=rho0, ea=ea, eb=eb, ec=ec, ifail=ifail,
        gab=gab, gca=gca, gbc=gbc, prba=prba, prca=prca, prcb=prcb,
        lce11t=lce11t, e11t=e11t, fct_t11=fct_t11, t11=t11, slimt11=slimt11,
        lce11c=lce11c, e11c=e11c, fct_c11=fct_c11, c11=c11, slimc11=slimc11,
        lce22t=lce22t, e22t=e22t, fct_t22=fct_t22, t22=t22, slimt22=slimt22,
        lce22c=lce22c, e22c=e22c, fct_c22=fct_c22, c22=c22, slimc22=slimc22,
        lce33t=lce33t, e33t=e33t, fct_t33=fct_t33, t33=t33, slimt33=slimt33,
        lce33c=lce33c, e33c=e33c, fct_c33=fct_c33, c33=c33, slimc33=slimc33,
        g12a=g12a, t12a=t12a, g12b=g12b, t12b=t12b, slims12=slims12,
        fct_g12a=fct_g12a, fct_t12a=fct_t12a, fct_g12b=fct_g12b, fct_t12b=fct_t12b,
        g31a=g31a, t31a=t31a, g31b=g31b, t31b=t31b, slims31=slims31,
        fct_g31a=fct_g31a, fct_t31a=fct_t31a, fct_g31b=fct_g31b, fct_t31b=fct_t31b,
        g23a=g23a, t23a=t23a, g23b=g23b, t23b=t23b, slims23=slims23,
        fct_g23a=fct_g23a, fct_t23a=fct_t23a, fct_g23b=fct_g23b, fct_t23b=fct_t23b,
        epsf=epsf, epsr=epsr, dmax=dmax, fct_fail=fct_fail, fail=fail, fcut=fcut,
    )
    model.mat_law125s[mat_id] = m125
    e_eff = max(ea, eb, ec) if max(ea, eb, ec) > 0.0 else 1.0
    from ..mat_reader import GenericMaterialRecord
    mat125 = Material(
        id=mat_id, law=125, rho0=rho0, title=title,
        params={
            "E": e_eff, "nu": prba if 0.0 <= prba < 0.5 else 0.3,
            "LSD_MAT_EA": ea, "LSD_MAT_EB": eb, "LSD_MAT_EC": ec, "LSD_FS": ifail,
            "LSD_MAT_GAB": gab, "LSD_MAT_GCA": gca, "LSD_MAT_GBC": gbc,
            "LSD_MAT_PRBA": prba, "LSDYNA_PRCA": prca, "LSDYNA_PRCB": prcb,
            "ea": ea, "eb": eb, "ec": ec, "ifail": ifail,
            "gab": gab, "gca": gca, "gbc": gbc,
            "prba": prba, "prca": prca, "prcb": prcb,
            "lce11t": lce11t, "e11t": e11t, "fct_t11": fct_t11, "t11": t11, "slimt11": slimt11,
            "lce11c": lce11c, "e11c": e11c, "fct_c11": fct_c11, "c11": c11, "slimc11": slimc11,
            "lce22t": lce22t, "e22t": e22t, "fct_t22": fct_t22, "t22": t22, "slimt22": slimt22,
            "lce22c": lce22c, "e22c": e22c, "fct_c22": fct_c22, "c22": c22, "slimc22": slimc22,
            "lce33t": lce33t, "e33t": e33t, "fct_t33": fct_t33, "t33": t33, "slimt33": slimt33,
            "lce33c": lce33c, "e33c": e33c, "fct_c33": fct_c33, "c33": c33, "slimc33": slimc33,
            "g12a": g12a, "t12a": t12a, "g12b": g12b, "t12b": t12b, "slims12": slims12,
            "fct_g12a": fct_g12a, "fct_t12a": fct_t12a, "fct_g12b": fct_g12b, "fct_t12b": fct_t12b,
            "g31a": g31a, "t31a": t31a, "g31b": g31b, "t31b": t31b, "slims31": slims31,
            "fct_g31a": fct_g31a, "fct_t31a": fct_t31a, "fct_g31b": fct_g31b, "fct_t31b": fct_t31b,
            "g23a": g23a, "t23a": t23a, "g23b": g23b, "t23b": t23b, "slims23": slims23,
            "fct_g23a": fct_g23a, "fct_t23a": fct_t23a, "fct_g23b": fct_g23b, "fct_t23b": fct_t23b,
            "epsf": epsf, "epsr": epsr, "dmax": dmax, "fct_fail": fct_fail, "fail": fail, "fcut": fcut,
        }
    )
    mat125.record = GenericMaterialRecord(
        law_name="LAW125", law_number=125, id=mat_id, title=title,
        params=mat125.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat125




def read_mat_law127(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW127/id`` or ``/MAT/ENHANCED_COMPOSITE/id`` (M174): Enhanced orthotropic composite model."""
    from ...model.entities import MaterialLaw127, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho0 = 0.0
    ea, eb, ec = 0.0, 0.0, 0.0
    gab, gca, gbc = 0.0, 0.0, 0.0
    prba, prca, prcb = 0.0, 0.0, 0.0
    xt, slimt1, lcxt, scalcxt = 0.0, 0.0, 0, 1.0
    yt, slimt2, lcyt, scalcyt = 0.0, 0.0, 0, 1.0
    sc, slimsc, lcsc, scalcsc = 0.0, 0.0, 0, 1.0
    xc, slimc1, lcxc, scalcxc = 0.0, 0.0, 0, 1.0
    yc, slimc2, lcyc, scalcyc = 0.0, 0.0, 0, 1.0
    fcut = 0.0
    alph, beta = 0.0, 0.0
    two_way, ti = 0, 0
    dfailt, dfailc, dfails, dfailm, ratio = 0.0, 0.0, 0.0, 0.0, 0.0
    ncyred = 0
    tfail, fbrt, ycfac = 0.0, 0.0, 0.0
    efs, epsf, epsr, tsmd = 0.0, 0.0, 0.0, 0.0

    valid_cards = [c for c in cards if not c.is_blank]

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW127_1")
            rho0 = _f(f1[0])
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW127_2")
            ea = _f(f2[0])
            eb = _f(f2[1]) if len(f2) > 1 else 0.0
            ec = _f(f2[2]) if len(f2) > 2 else 0.0
        if len(valid_cards) > 2:
            f3 = cut(valid_cards[2].raw, "MAT_LAW127_3")
            gab = _f(f3[0])
            gca = _f(f3[1]) if len(f3) > 1 else 0.0
            gbc = _f(f3[2]) if len(f3) > 2 else 0.0
        if len(valid_cards) > 3:
            f4 = cut(valid_cards[3].raw, "MAT_LAW127_4")
            prba = _f(f4[0])
            prca = _f(f4[1]) if len(f4) > 1 else 0.0
            prcb = _f(f4[2]) if len(f4) > 2 else 0.0
        if len(valid_cards) > 4:
            f5 = cut(valid_cards[4].raw, "MAT_LAW127_5")
            xt = _f(f5[0])
            slimt1 = _f(f5[1]) if len(f5) > 1 else 0.0
            lcxt = _i(f5[3]) if len(f5) > 3 else 0
            scalcxt = _f(f5[4], 1.0) if len(f5) > 4 and f5[4].strip() else 1.0
        if len(valid_cards) > 5:
            f6 = cut(valid_cards[5].raw, "MAT_LAW127_6")
            yt = _f(f6[0])
            slimt2 = _f(f6[1]) if len(f6) > 1 else 0.0
            lcyt = _i(f6[3]) if len(f6) > 3 else 0
            scalcyt = _f(f6[4], 1.0) if len(f6) > 4 and f6[4].strip() else 1.0
        if len(valid_cards) > 6:
            f7 = cut(valid_cards[6].raw, "MAT_LAW127_7")
            sc = _f(f7[0])
            slimsc = _f(f7[1]) if len(f7) > 1 else 0.0
            lcsc = _i(f7[3]) if len(f7) > 3 else 0
            scalcsc = _f(f7[4], 1.0) if len(f7) > 4 and f7[4].strip() else 1.0
        if len(valid_cards) > 7:
            f8 = cut(valid_cards[7].raw, "MAT_LAW127_8")
            xc = _f(f8[0])
            slimc1 = _f(f8[1]) if len(f8) > 1 else 0.0
            lcxc = _i(f8[3]) if len(f8) > 3 else 0
            scalcxc = _f(f8[4], 1.0) if len(f8) > 4 and f8[4].strip() else 1.0
        if len(valid_cards) > 8:
            f9 = cut(valid_cards[8].raw, "MAT_LAW127_9")
            yc = _f(f9[0])
            slimc2 = _f(f9[1]) if len(f9) > 1 else 0.0
            lcyc = _i(f9[3]) if len(f9) > 3 else 0
            scalcyc = _f(f9[4], 1.0) if len(f9) > 4 and f9[4].strip() else 1.0
        if len(valid_cards) > 9:
            f10 = cut(valid_cards[9].raw, "MAT_LAW127_10")
            fcut = _f(f10[0])
        if len(valid_cards) > 10:
            f11 = cut(valid_cards[10].raw, "MAT_LAW127_11")
            alph = _f(f11[0])
            beta = _f(f11[1]) if len(f11) > 1 else 0.0
            two_way = _i(f11[2]) if len(f11) > 2 else 0
            ti = _i(f11[3]) if len(f11) > 3 else 0
        if len(valid_cards) > 11:
            f12 = cut(valid_cards[11].raw, "MAT_LAW127_12")
            dfailt = _f(f12[0])
            dfailc = _f(f12[1]) if len(f12) > 1 else 0.0
            dfails = _f(f12[2]) if len(f12) > 2 else 0.0
            dfailm = _f(f12[3]) if len(f12) > 3 else 0.0
            ratio = _f(f12[4]) if len(f12) > 4 else 0.0
        if len(valid_cards) > 12:
            f13 = cut(valid_cards[12].raw, "MAT_LAW127_13")
            ncyred = _i(f13[1]) if len(f13) > 1 else 0
            tfail = _f(f13[2]) if len(f13) > 2 else 0.0
            fbrt = _f(f13[3]) if len(f13) > 3 else 0.0
            ycfac = _f(f13[4]) if len(f13) > 4 else 0.0
        if len(valid_cards) > 13:
            f14 = cut(valid_cards[13].raw, "MAT_LAW127_14")
            efs = _f(f14[0])
            epsf = _f(f14[1]) if len(f14) > 1 else 0.0
            epsr = _f(f14[2]) if len(f14) > 2 else 0.0
            tsmd = _f(f14[3]) if len(f14) > 3 else 0.0
    else:
        if len(valid_cards) > 0:
            t1 = valid_cards[0].tokens()
            rho0 = float(t1[0]) if len(t1) > 0 else 0.0
        if len(valid_cards) > 1:
            t2 = valid_cards[1].tokens()
            ea = float(t2[0]) if len(t2) > 0 else 0.0
            eb = float(t2[1]) if len(t2) > 1 else 0.0
            ec = float(t2[2]) if len(t2) > 2 else 0.0
        if len(valid_cards) > 2:
            t3 = valid_cards[2].tokens()
            gab = float(t3[0]) if len(t3) > 0 else 0.0
            gca = float(t3[1]) if len(t3) > 1 else 0.0
            gbc = float(t3[2]) if len(t3) > 2 else 0.0
        if len(valid_cards) > 3:
            t4 = valid_cards[3].tokens()
            prba = float(t4[0]) if len(t4) > 0 else 0.0
            prca = float(t4[1]) if len(t4) > 1 else 0.0
            prcb = float(t4[2]) if len(t4) > 2 else 0.0
        if len(valid_cards) > 4:
            t5 = valid_cards[4].tokens()
            xt = float(t5[0]) if len(t5) > 0 else 0.0
            slimt1 = float(t5[1]) if len(t5) > 1 else 0.0
            lcxt = int(float(t5[2])) if len(t5) > 2 else 0
            scalcxt = float(t5[3]) if len(t5) > 3 else 1.0
        if len(valid_cards) > 5:
            t6 = valid_cards[5].tokens()
            yt = float(t6[0]) if len(t6) > 0 else 0.0
            slimt2 = float(t6[1]) if len(t6) > 1 else 0.0
            lcyt = int(float(t6[2])) if len(t6) > 2 else 0
            scalcyt = float(t6[3]) if len(t6) > 3 else 1.0
        if len(valid_cards) > 6:
            t7 = valid_cards[6].tokens()
            sc = float(t7[0]) if len(t7) > 0 else 0.0
            slimsc = float(t7[1]) if len(t7) > 1 else 0.0
            lcsc = int(float(t7[2])) if len(t7) > 2 else 0
            scalcsc = float(t7[3]) if len(t7) > 3 else 1.0
        if len(valid_cards) > 7:
            t8 = valid_cards[7].tokens()
            xc = float(t8[0]) if len(t8) > 0 else 0.0
            slimc1 = float(t8[1]) if len(t8) > 1 else 0.0
            lcxc = int(float(t8[2])) if len(t8) > 2 else 0
            scalcxc = float(t8[3]) if len(t8) > 3 else 1.0
        if len(valid_cards) > 8:
            t9 = valid_cards[8].tokens()
            yc = float(t9[0]) if len(t9) > 0 else 0.0
            slimc2 = float(t9[1]) if len(t9) > 1 else 0.0
            lcyc = int(float(t9[2])) if len(t9) > 2 else 0
            scalcyc = float(t9[3]) if len(t9) > 3 else 1.0
        if len(valid_cards) > 9:
            t10 = valid_cards[9].tokens()
            fcut = float(t10[0]) if len(t10) > 0 else 0.0
        if len(valid_cards) > 10:
            t11 = valid_cards[10].tokens()
            alph = float(t11[0]) if len(t11) > 0 else 0.0
            beta = float(t11[1]) if len(t11) > 1 else 0.0
            two_way = int(float(t11[2])) if len(t11) > 2 else 0
            ti = int(float(t11[3])) if len(t11) > 3 else 0
        if len(valid_cards) > 11:
            t12 = valid_cards[11].tokens()
            dfailt = float(t12[0]) if len(t12) > 0 else 0.0
            dfailc = float(t12[1]) if len(t12) > 1 else 0.0
            dfails = float(t12[2]) if len(t12) > 2 else 0.0
            dfailm = float(t12[3]) if len(t12) > 3 else 0.0
            ratio = float(t12[4]) if len(t12) > 4 else 0.0
        if len(valid_cards) > 12:
            t13 = valid_cards[12].tokens()
            ncyred = int(float(t13[0])) if len(t13) > 0 else 0
            tfail = float(t13[1]) if len(t13) > 1 else 0.0
            fbrt = float(t13[2]) if len(t13) > 2 else 0.0
            ycfac = float(t13[3]) if len(t13) > 3 else 0.0
        if len(valid_cards) > 13:
            t14 = valid_cards[13].tokens()
            efs = float(t14[0]) if len(t14) > 0 else 0.0
            epsf = float(t14[1]) if len(t14) > 1 else 0.0
            epsr = float(t14[2]) if len(t14) > 2 else 0.0
            tsmd = float(t14[3]) if len(t14) > 3 else 0.0

    m127 = MaterialLaw127(
        id=mat_id, title=title, rho0=rho0, ea=ea, eb=eb, ec=ec,
        gab=gab, gca=gca, gbc=gbc, prba=prba, prca=prca, prcb=prcb,
        xt=xt, slimt1=slimt1, lcxt=lcxt, scalcxt=scalcxt,
        yt=yt, slimt2=slimt2, lcyt=lcyt, scalcyt=scalcyt,
        sc=sc, slimsc=slimsc, lcsc=lcsc, scalcsc=scalcsc,
        xc=xc, slimc1=slimc1, lcxc=lcxc, scalcxc=scalcxc,
        yc=yc, slimc2=slimc2, lcyc=lcyc, scalcyc=scalcyc,
        fcut=fcut, alph=alph, beta=beta, two_way=two_way, ti=ti,
        dfailt=dfailt, dfailc=dfailc, dfails=dfails, dfailm=dfailm, ratio=ratio,
        ncyred=ncyred, tfail=tfail, fbrt=fbrt, ycfac=ycfac,
        efs=efs, epsf=epsf, epsr=epsr, tsmd=tsmd,
    )
    model.mat_law127s[mat_id] = m127
    e_eff = max(ea, eb, ec) if max(ea, eb, ec) > 0.0 else 1.0
    from ..mat_reader import GenericMaterialRecord
    mat127 = Material(
        id=mat_id, law=127, rho0=rho0, title=title,
        params={
            "E": e_eff, "nu": prba if 0.0 <= prba < 0.5 else 0.3,
            "LSDYNA_EA": ea, "LSDYNA_EB": eb, "LSDYNA_EC": ec,
            "LSDYNA_GAB": gab, "LSDYNA_GCA": gca, "LSDYNA_GBC": gbc,
            "LSDYNA_PRBA": prba, "LSDYNA_PRCA": prca, "LSDYNA_PRCB": prcb,
            "ea": ea, "eb": eb, "ec": ec,
            "gab": gab, "gca": gca, "gbc": gbc,
            "prba": prba, "prca": prca, "prcb": prcb,
            "xt": xt, "slimt1": slimt1, "lcxt": lcxt, "scalcxt": scalcxt,
            "yt": yt, "slimt2": slimt2, "lcyt": lcyt, "scalcyt": scalcyt,
            "sc": sc, "slimsc": slimsc, "lcsc": lcsc, "scalcsc": scalcsc,
            "xc": xc, "slimc1": slimc1, "lcxc": lcxc, "scalcxc": scalcxc,
            "yc": yc, "slimc2": slimc2, "lcyc": lcyc, "scalcyc": scalcyc,
            "fcut": fcut, "alph": alph, "beta": beta, "two_way": two_way, "ti": ti,
            "dfailt": dfailt, "dfailc": dfailc, "dfails": dfails, "dfailm": dfailm, "ratio": ratio,
            "ncyred": ncyred, "tfail": tfail, "fbrt": fbrt, "ycfac": ycfac,
            "efs": efs, "epsf": epsf, "epsr": epsr, "tsmd": tsmd,
        }
    )
    mat127.record = GenericMaterialRecord(
        law_name="LAW127", law_number=127, id=mat_id, title=title,
        params=mat127.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat127




def read_mat_law130(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW130/id`` or ``/MAT/MODIFIED_HONEYCOMB/id`` (M174): Modified crushable honeycomb material model."""
    from ...model.entities import MaterialLaw130, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho0 = 0.0
    e, nu = 0.0, 0.0
    sigy = 0.0
    vf = 0.0
    mu = 0.0
    iform, shdflg = 0, 0
    lca, lcb, lcc, lcs, lcab, lcbc, lcca, lcsr = 0, 0, 0, 0, 0, 0, 0, 0
    eaau, ebbu, eccu, gabu, gbcu, gcau = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    rfac = 0.0
    tsef, ssef = 0.0, 0.0
    pru = 0
    lcsra, lcsrb, lcsrc, lcsrab, lcsrbc, lcsrca = 0, 0, 0, 0, 0, 0
    pruab, pruac, prubc, pruba, pruca, prucb = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    valid_cards = [c for c in cards if not c.is_blank]

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW130_1")
            rho0 = _f(f1[0])
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW130_2")
            e = _f(f2[0])
            nu = _f(f2[1]) if len(f2) > 1 else 0.0
            sigy = _f(f2[2]) if len(f2) > 2 else 0.0
            vf = _f(f2[3]) if len(f2) > 3 else 0.0
            mu = _f(f2[4]) if len(f2) > 4 else 0.0
        if len(valid_cards) > 2:
            f3 = cut(valid_cards[2].raw, "MAT_LAW130_3")
            iform = _i(f3[0])
            shdflg = _i(f3[1]) if len(f3) > 1 else 0
            lca = _i(f3[2]) if len(f3) > 2 else 0
            lcb = _i(f3[3]) if len(f3) > 3 else 0
            lcc = _i(f3[4]) if len(f3) > 4 else 0
            lcs = _i(f3[5]) if len(f3) > 5 else 0
            lcab = _i(f3[6]) if len(f3) > 6 else 0
            lcbc = _i(f3[7]) if len(f3) > 7 else 0
            lcca = _i(f3[8]) if len(f3) > 8 else 0
            lcsr = _i(f3[9]) if len(f3) > 9 else 0
        if len(valid_cards) > 3:
            f4 = cut(valid_cards[3].raw, "MAT_LAW130_4")
            eaau = _f(f4[0])
            ebbu = _f(f4[1]) if len(f4) > 1 else 0.0
            eccu = _f(f4[2]) if len(f4) > 2 else 0.0
            gabu = _f(f4[3]) if len(f4) > 3 else 0.0
            gbcu = _f(f4[4]) if len(f4) > 4 else 0.0
        if len(valid_cards) > 4:
            f5 = cut(valid_cards[4].raw, "MAT_LAW130_5")
            gcau = _f(f5[0])
            rfac = _f(f5[1]) if len(f5) > 1 else 0.0
            tsef = _f(f5[2]) if len(f5) > 2 else 0.0
            ssef = _f(f5[3]) if len(f5) > 3 else 0.0
            pru = _i(f5[5]) if len(f5) > 5 else 0

        val_idx = 5
        if lcsr < 0 and len(valid_cards) > val_idx:
            f6 = cut(valid_cards[val_idx].raw, "MAT_LAW130_6")
            lcsra = _i(f6[0])
            lcsrb = _i(f6[1]) if len(f6) > 1 else 0
            lcsrc = _i(f6[2]) if len(f6) > 2 else 0
            lcsrab = _i(f6[3]) if len(f6) > 3 else 0
            lcsrbc = _i(f6[4]) if len(f6) > 4 else 0
            lcsrca = _i(f6[5]) if len(f6) > 5 else 0
            val_idx += 1

        if pru == 2 and len(valid_cards) > val_idx:
            f7 = cut(valid_cards[val_idx].raw, "MAT_LAW130_7")
            pruab = _f(f7[0])
            pruac = _f(f7[1]) if len(f7) > 1 else 0.0
            prubc = _f(f7[2]) if len(f7) > 2 else 0.0
            pruba = _f(f7[3]) if len(f7) > 3 else 0.0
            pruca = _f(f7[4]) if len(f7) > 4 else 0.0
            val_idx += 1
            if len(valid_cards) > val_idx:
                f8 = cut(valid_cards[val_idx].raw, "MAT_LAW130_8")
                prucb = _f(f8[0])
    else:
        if len(valid_cards) > 0:
            t1 = valid_cards[0].tokens()
            rho0 = float(t1[0]) if len(t1) > 0 else 0.0
        if len(valid_cards) > 1:
            t2 = valid_cards[1].tokens()
            e = float(t2[0]) if len(t2) > 0 else 0.0
            nu = float(t2[1]) if len(t2) > 1 else 0.0
            sigy = float(t2[2]) if len(t2) > 2 else 0.0
            vf = float(t2[3]) if len(t2) > 3 else 0.0
            mu = float(t2[4]) if len(t2) > 4 else 0.0
        if len(valid_cards) > 2:
            t3 = valid_cards[2].tokens()
            iform = int(float(t3[0])) if len(t3) > 0 else 0
            shdflg = int(float(t3[1])) if len(t3) > 1 else 0
            lca = int(float(t3[2])) if len(t3) > 2 else 0
            lcb = int(float(t3[3])) if len(t3) > 3 else 0
            lcc = int(float(t3[4])) if len(t3) > 4 else 0
            lcs = int(float(t3[5])) if len(t3) > 5 else 0
            lcab = int(float(t3[6])) if len(t3) > 6 else 0
            lcbc = int(float(t3[7])) if len(t3) > 7 else 0
            lcca = int(float(t3[8])) if len(t3) > 8 else 0
            lcsr = int(float(t3[9])) if len(t3) > 9 else 0
        if len(valid_cards) > 3:
            t4 = valid_cards[3].tokens()
            eaau = float(t4[0]) if len(t4) > 0 else 0.0
            ebbu = float(t4[1]) if len(t4) > 1 else 0.0
            eccu = float(t4[2]) if len(t4) > 2 else 0.0
            gabu = float(t4[3]) if len(t4) > 3 else 0.0
            gbcu = float(t4[4]) if len(t4) > 4 else 0.0
        if len(valid_cards) > 4:
            t5 = valid_cards[4].tokens()
            gcau = float(t5[0]) if len(t5) > 0 else 0.0
            rfac = float(t5[1]) if len(t5) > 1 else 0.0
            tsef = float(t5[2]) if len(t5) > 2 else 0.0
            ssef = float(t5[3]) if len(t5) > 3 else 0.0
            pru = int(float(t5[4])) if len(t5) > 4 else 0

        val_idx = 5
        if lcsr < 0 and len(valid_cards) > val_idx:
            t6 = valid_cards[val_idx].tokens()
            lcsra = int(float(t6[0])) if len(t6) > 0 else 0
            lcsrb = int(float(t6[1])) if len(t6) > 1 else 0
            lcsrc = int(float(t6[2])) if len(t6) > 2 else 0
            lcsrab = int(float(t6[3])) if len(t6) > 3 else 0
            lcsrbc = int(float(t6[4])) if len(t6) > 4 else 0
            lcsrca = int(float(t6[5])) if len(t6) > 5 else 0
            val_idx += 1

        if pru == 2 and len(valid_cards) > val_idx:
            t7 = valid_cards[val_idx].tokens()
            pruab = float(t7[0]) if len(t7) > 0 else 0.0
            pruac = float(t7[1]) if len(t7) > 1 else 0.0
            prubc = float(t7[2]) if len(t7) > 2 else 0.0
            pruba = float(t7[3]) if len(t7) > 3 else 0.0
            pruca = float(t7[4]) if len(t7) > 4 else 0.0
            val_idx += 1
            if len(valid_cards) > val_idx:
                t8 = valid_cards[val_idx].tokens()
                prucb = float(t8[0]) if len(t8) > 0 else 0.0

    m130 = MaterialLaw130(
        id=mat_id, title=title, rho0=rho0, e=e, nu=nu, sigy=sigy, vf=vf, mu=mu,
        iform=iform, shdflg=shdflg, lca=lca, lcb=lcb, lcc=lcc, lcs=lcs,
        lcab=lcab, lcbc=lcbc, lcca=lcca, lcsr=lcsr,
        eaau=eaau, ebbu=ebbu, eccu=eccu, gabu=gabu, gbcu=gbcu, gcau=gcau,
        rfac=rfac, tsef=tsef, ssef=ssef, pru=pru,
        lcsra=lcsra, lcsrb=lcsrb, lcsrc=lcsrc, lcsrab=lcsrab, lcsrbc=lcsrbc, lcsrca=lcsrca,
        pruab=pruab, pruac=pruac, prubc=prubc, pruba=pruba, pruca=pruca, prucb=prucb,
    )
    model.mat_law130s[mat_id] = m130
    e_eff = e if e > 0.0 else max(eaau, ebbu, eccu, 1.0)
    from ..mat_reader import GenericMaterialRecord
    mat130 = Material(
        id=mat_id, law=130, rho0=rho0, title=title,
        params={
            "E": e_eff, "nu": nu if 0.0 <= nu < 0.5 else 0.2,
            "MAT_E": e, "MAT_NU": nu, "LSDYNA_SIGY": sigy, "LSDYNA_VF": vf, "LSDYNA_MU": mu,
            "IFORM": iform, "MATL126_SHDFLG": shdflg,
            "LSD_LCID": lca, "LSD_LCID2": lcb, "LSD_LCID3": lcc, "LSD_LCID4": lcs,
            "LSD_LCID5": lcab, "LSD_LCID6": lcbc, "LSD_LCID7": lcca, "LSD_LCID8": lcsr,
            "LSDYNA_EAAU": eaau, "LSDYNA_EBBU": ebbu, "LSDYNA_ECCU": eccu,
            "LSDYNA_GABU": gabu, "LSDYNA_GBCU": gbcu, "LSDYNA_GCAU": gcau,
            "LSDYNA_RFAC": rfac, "LSDYNA_SIGF": tsef, "LSDYNA_EPSF": ssef, "LSDYNA_PRU": pru,
            "e": e, "nu": nu, "sigy": sigy, "vf": vf, "mu": mu,
            "iform": iform, "shdflg": shdflg,
            "lca": lca, "lcb": lcb, "lcc": lcc, "lcs": lcs,
            "lcab": lcab, "lcbc": lcbc, "lcca": lcca, "lcsr": lcsr,
            "eaau": eaau, "ebbu": ebbu, "eccu": eccu,
            "gabu": gabu, "gbcu": gbcu, "gcau": gcau,
            "rfac": rfac, "tsef": tsef, "ssef": ssef, "pru": pru,
            "lcsra": lcsra, "lcsrb": lcsrb, "lcsrc": lcsrc, "lcsrab": lcsrab, "lcsrbc": lcsrbc, "lcsrca": lcsrca,
            "pruab": pruab, "pruac": pruac, "prubc": prubc, "pruba": pruba, "pruca": pruca, "prucb": prucb,
        }
    )
    mat130.record = GenericMaterialRecord(
        law_name="LAW130", law_number=130, id=mat_id, title=title,
        params=mat130.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat130




def read_mat_law128(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW128/id`` or ``/MAT/HILL_VISC_PLAST/id`` (M175): Hill anisotropic viscoplastic model."""
    from ...model.entities import MaterialLaw128, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho0 = 0.0
    e, nu = 0.0, 0.0
    sigy, kin = 0.0, 0.0
    tab_id = 0
    facy, facx = 0.0, 0.0
    qr1, cr1, qr2, cr2 = 0.0, 0.0, 0.0, 0.0
    qx1, cx1, qx2, cx2 = 0.0, 0.0, 0.0, 0.0
    epsp0, cp = 0.0, 0.0
    r00, r45, r90 = 1.0, 1.0, 1.0
    f, g, h = 0.0, 0.0, 0.0
    l, m, n = 0.0, 0.0, 0.0

    valid_cards = [c for c in cards if not c.is_blank]

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW128_1")
            rho0 = _f(f1[0])
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW128_2")
            e = _f(f2[0])
            nu = _f(f2[1]) if len(f2) > 1 else 0.0
            sigy = _f(f2[2]) if len(f2) > 2 else 0.0
            kin = _f(f2[3]) if len(f2) > 3 else 0.0
        if len(valid_cards) > 2:
            f3 = cut(valid_cards[2].raw, "MAT_LAW128_3")
            tab_id = _i(f3[0])
            facy = _f(f3[2]) if len(f3) > 2 else 0.0
            facx = _f(f3[3]) if len(f3) > 3 else 0.0
        if len(valid_cards) > 3:
            f4 = cut(valid_cards[3].raw, "MAT_LAW128_4")
            qr1 = _f(f4[0])
            cr1 = _f(f4[1]) if len(f4) > 1 else 0.0
            qr2 = _f(f4[2]) if len(f4) > 2 else 0.0
            cr2 = _f(f4[3]) if len(f4) > 3 else 0.0
        if len(valid_cards) > 4:
            f5 = cut(valid_cards[4].raw, "MAT_LAW128_5")
            qx1 = _f(f5[0])
            cx1 = _f(f5[1]) if len(f5) > 1 else 0.0
            qx2 = _f(f5[2]) if len(f5) > 2 else 0.0
            cx2 = _f(f5[3]) if len(f5) > 3 else 0.0
        if len(valid_cards) > 5:
            f6 = cut(valid_cards[5].raw, "MAT_LAW128_6")
            epsp0 = _f(f6[0])
            cp = _f(f6[1]) if len(f6) > 1 else 0.0
        if len(valid_cards) > 6:
            f7 = cut(valid_cards[6].raw, "MAT_LAW128_7")
            r00 = _f(f7[0])
            r45 = _f(f7[1]) if len(f7) > 1 else 0.0
            r90 = _f(f7[2]) if len(f7) > 2 else 0.0
        if len(valid_cards) > 7:
            f8 = cut(valid_cards[7].raw, "MAT_LAW128_8")
            f = _f(f8[0])
            g = _f(f8[1]) if len(f8) > 1 else 0.0
            h = _f(f8[2]) if len(f8) > 2 else 0.0
        if len(valid_cards) > 8:
            f9 = cut(valid_cards[8].raw, "MAT_LAW128_9")
            l = _f(f9[0])
            m = _f(f9[1]) if len(f9) > 1 else 0.0
            n = _f(f9[2]) if len(f9) > 2 else 0.0
    else:
        if len(valid_cards) > 0:
            toks1 = valid_cards[0].tokens()
            rho0 = float(toks1[0]) if len(toks1) > 0 else 0.0
        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            e = float(toks2[0]) if len(toks2) > 0 else 0.0
            nu = float(toks2[1]) if len(toks2) > 1 else 0.0
            sigy = float(toks2[2]) if len(toks2) > 2 else 0.0
            kin = float(toks2[3]) if len(toks2) > 3 else 0.0
        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            tab_id = int(float(toks3[0])) if len(toks3) > 0 else 0
            facy = float(toks3[1]) if len(toks3) > 1 else 0.0
            facx = float(toks3[2]) if len(toks3) > 2 else 0.0
        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            qr1 = float(toks4[0]) if len(toks4) > 0 else 0.0
            cr1 = float(toks4[1]) if len(toks4) > 1 else 0.0
            qr2 = float(toks4[2]) if len(toks4) > 2 else 0.0
            cr2 = float(toks4[3]) if len(toks4) > 3 else 0.0
        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            qx1 = float(toks5[0]) if len(toks5) > 0 else 0.0
            cx1 = float(toks5[1]) if len(toks5) > 1 else 0.0
            qx2 = float(toks5[2]) if len(toks5) > 2 else 0.0
            cx2 = float(toks5[3]) if len(toks5) > 3 else 0.0
        if len(valid_cards) > 5:
            toks6 = valid_cards[5].tokens()
            epsp0 = float(toks6[0]) if len(toks6) > 0 else 0.0
            cp = float(toks6[1]) if len(toks6) > 1 else 0.0
        if len(valid_cards) > 6:
            toks7 = valid_cards[6].tokens()
            r00 = float(toks7[0]) if len(toks7) > 0 else 1.0
            r45 = float(toks7[1]) if len(toks7) > 1 else 1.0
            r90 = float(toks7[2]) if len(toks7) > 2 else 1.0
        if len(valid_cards) > 7:
            toks8 = valid_cards[7].tokens()
            f = float(toks8[0]) if len(toks8) > 0 else 0.0
            g = float(toks8[1]) if len(toks8) > 1 else 0.0
            h = float(toks8[2]) if len(toks8) > 2 else 0.0
        if len(valid_cards) > 8:
            toks9 = valid_cards[8].tokens()
            l = float(toks9[0]) if len(toks9) > 0 else 0.0
            m = float(toks9[1]) if len(toks9) > 1 else 0.0
            n = float(toks9[2]) if len(toks9) > 2 else 0.0

    m128 = MaterialLaw128(
        id=mat_id, title=title, rho0=rho0, e=e, nu=nu, sigy=sigy, kin=kin,
        tab_id=tab_id, facy=facy, facx=facx,
        qr1=qr1, cr1=cr1, qr2=qr2, cr2=cr2,
        qx1=qx1, cx1=cx1, qx2=qx2, cx2=cx2,
        epsp0=epsp0, cp=cp, r00=r00, r45=r45, r90=r90,
        f=f, g=g, h=h, l=l, m=m, n=n,
    )
    model.mat_law128s[mat_id] = m128
    from ..mat_reader import GenericMaterialRecord
    mat128 = Material(
        id=mat_id, law=128, rho0=rho0, title=title,
        params={
            "E": e if e > 0.0 else 1.0, "nu": nu if 0.0 <= nu < 0.5 else 0.3,
            "LAW128_E": e, "LAW128_NU": nu, "LAW128_SIGY": sigy, "LAW128_KIN": kin,
            "LAW128_TAB_ID": tab_id, "LAW128_FACY": facy, "LAW128_FACX": facx,
            "LAW128_QR1": qr1, "LAW128_CR1": cr1, "LAW128_QR2": qr2, "LAW128_CR2": cr2,
            "LAW128_QX1": qx1, "LAW128_CX1": cx1, "LAW128_QX2": qx2, "LAW128_CX2": cx2,
            "LAW128_EPSP0": epsp0, "LAW128_CP": cp,
            "LAW128_R00": r00, "LAW128_R45": r45, "LAW128_R90": r90,
            "LAW128_F": f, "LAW128_G": g, "LAW128_H": h,
            "LAW128_L": l, "LAW128_M": m, "LAW128_N": n,
            "e": e, "nu": nu, "sigy": sigy, "kin": kin,
            "tab_id": tab_id, "facy": facy, "facx": facx,
            "qr1": qr1, "cr1": cr1, "qr2": qr2, "cr2": cr2,
            "qx1": qx1, "cx1": cx1, "qx2": qx2, "cx2": cx2,
            "epsp0": epsp0, "cp": cp, "r00": r00, "r45": r45, "r90": r90,
            "f": f, "g": g, "h": h, "l": l, "m": m, "n": n,
        }
    )
    mat128.record = GenericMaterialRecord(
        law_name="LAW128", law_number=128, id=mat_id, title=title,
        params=mat128.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat128




def read_mat_law129(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW129/id`` or ``/MAT/THERM_CREEP/id`` (M175): Thermo-elasto-viscoplastic creep material model."""
    from ...model.entities import MaterialLaw129, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho0 = 0.0
    e, nu, sigy, alpha, tref = 0.0, 0.0, 0.0, 0.0, 0.0
    f_young, f_nu, f_yld, f_alpha, isensor = 0, 0, 0, 0, 0
    itab = 0
    facy = 0.0
    qr1, cr1, qr2, cr2 = 0.0, 0.0, 0.0, 0.0
    f_qr, f_cr = 0, 0
    qx1, cx1, qx2, cx2 = 0.0, 0.0, 0.0, 0.0
    f_qx, f_cx = 0, 0
    epsp0, cp = 0.0, 0.0
    f_cc, f_cp = 0, 0
    crpa, crpn, crpm = 0.0, 0.0, 0.0
    f_a, f_n, f_m, crp_law = 0, 0, 0, 0
    crsig, crt, crpq, eps0 = 0.0, 0.0, 0.0, 0.0
    f_q, f_sig = 0, 0

    valid_cards = [c for c in cards if not c.is_blank]

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW129_1")
            rho0 = _f(f1[0])
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW129_2")
            e = _f(f2[0])
            nu = _f(f2[1]) if len(f2) > 1 else 0.0
            sigy = _f(f2[2]) if len(f2) > 2 else 0.0
            alpha = _f(f2[3]) if len(f2) > 3 else 0.0
            tref = _f(f2[4]) if len(f2) > 4 else 0.0
        if len(valid_cards) > 2:
            f3 = cut(valid_cards[2].raw, "MAT_LAW129_3")
            f_young = _i(f3[0])
            f_nu = _i(f3[1]) if len(f3) > 1 else 0
            f_yld = _i(f3[2]) if len(f3) > 2 else 0
            f_alpha = _i(f3[3]) if len(f3) > 3 else 0
            isensor = _i(f3[5]) if len(f3) > 5 else 0
        if len(valid_cards) > 3:
            f4 = cut(valid_cards[3].raw, "MAT_LAW129_4")
            itab = _i(f4[0])
            facy = _f(f4[2]) if len(f4) > 2 else 0.0
        if len(valid_cards) > 4:
            f5 = cut(valid_cards[4].raw, "MAT_LAW129_5")
            qr1 = _f(f5[0])
            cr1 = _f(f5[1]) if len(f5) > 1 else 0.0
            qr2 = _f(f5[2]) if len(f5) > 2 else 0.0
            cr2 = _f(f5[3]) if len(f5) > 3 else 0.0
            f_qr = _i(f5[4]) if len(f5) > 4 else 0
            f_cr = _i(f5[5]) if len(f5) > 5 else 0
        if len(valid_cards) > 5:
            f6 = cut(valid_cards[5].raw, "MAT_LAW129_6")
            qx1 = _f(f6[0])
            cx1 = _f(f6[1]) if len(f6) > 1 else 0.0
            qx2 = _f(f6[2]) if len(f6) > 2 else 0.0
            cx2 = _f(f6[3]) if len(f6) > 3 else 0.0
            f_qx = _i(f6[4]) if len(f6) > 4 else 0
            f_cx = _i(f6[5]) if len(f6) > 5 else 0
        if len(valid_cards) > 6:
            f7 = cut(valid_cards[6].raw, "MAT_LAW129_7")
            epsp0 = _f(f7[0])
            cp = _f(f7[1]) if len(f7) > 1 else 0.0
            f_cc = _i(f7[2]) if len(f7) > 2 else 0
            f_cp = _i(f7[3]) if len(f7) > 3 else 0
        if len(valid_cards) > 7:
            f8 = cut(valid_cards[7].raw, "MAT_LAW129_8")
            crpa = _f(f8[0])
            crpn = _f(f8[1]) if len(f8) > 1 else 0.0
            crpm = _f(f8[2]) if len(f8) > 2 else 0.0
            f_a = _i(f8[3]) if len(f8) > 3 else 0
            f_n = _i(f8[4]) if len(f8) > 4 else 0
            f_m = _i(f8[5]) if len(f8) > 5 else 0
            crp_law = _i(f8[6]) if len(f8) > 6 else 0
        if len(valid_cards) > 8:
            f9 = cut(valid_cards[8].raw, "MAT_LAW129_9")
            crsig = _f(f9[0])
            crt = _f(f9[1]) if len(f9) > 1 else 0.0
            crpq = _f(f9[2]) if len(f9) > 2 else 0.0
            eps0 = _f(f9[3]) if len(f9) > 3 else 0.0
            f_q = _i(f9[4]) if len(f9) > 4 else 0
            f_sig = _i(f9[5]) if len(f9) > 5 else 0
    else:
        if len(valid_cards) > 0:
            toks1 = valid_cards[0].tokens()
            rho0 = float(toks1[0]) if len(toks1) > 0 else 0.0
        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            e = float(toks2[0]) if len(toks2) > 0 else 0.0
            nu = float(toks2[1]) if len(toks2) > 1 else 0.0
            sigy = float(toks2[2]) if len(toks2) > 2 else 0.0
            alpha = float(toks2[3]) if len(toks2) > 3 else 0.0
            tref = float(toks2[4]) if len(toks2) > 4 else 0.0
        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            f_young = int(float(toks3[0])) if len(toks3) > 0 else 0
            f_nu = int(float(toks3[1])) if len(toks3) > 1 else 0
            f_yld = int(float(toks3[2])) if len(toks3) > 2 else 0
            f_alpha = int(float(toks3[3])) if len(toks3) > 3 else 0
            isensor = int(float(toks3[4])) if len(toks3) > 4 else 0
        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            itab = int(float(toks4[0])) if len(toks4) > 0 else 0
            facy = float(toks4[1]) if len(toks4) > 1 else 0.0
        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            qr1 = float(toks5[0]) if len(toks5) > 0 else 0.0
            cr1 = float(toks5[1]) if len(toks5) > 1 else 0.0
            qr2 = float(toks5[2]) if len(toks5) > 2 else 0.0
            cr2 = float(toks5[3]) if len(toks5) > 3 else 0.0
            f_qr = int(float(toks5[4])) if len(toks5) > 4 else 0
            f_cr = int(float(toks5[5])) if len(toks5) > 5 else 0
        if len(valid_cards) > 5:
            toks6 = valid_cards[5].tokens()
            qx1 = float(toks6[0]) if len(toks6) > 0 else 0.0
            cx1 = float(toks6[1]) if len(toks6) > 1 else 0.0
            qx2 = float(toks6[2]) if len(toks6) > 2 else 0.0
            cx2 = float(toks6[3]) if len(toks6) > 3 else 0.0
            f_qx = int(float(toks6[4])) if len(toks6) > 4 else 0
            f_cx = int(float(toks6[5])) if len(toks6) > 5 else 0
        if len(valid_cards) > 6:
            toks7 = valid_cards[6].tokens()
            epsp0 = float(toks7[0]) if len(toks7) > 0 else 0.0
            cp = float(toks7[1]) if len(toks7) > 1 else 0.0
            f_cc = int(float(toks7[2])) if len(toks7) > 2 else 0
            f_cp = int(float(toks7[3])) if len(toks7) > 3 else 0
        if len(valid_cards) > 7:
            toks8 = valid_cards[7].tokens()
            crpa = float(toks8[0]) if len(toks8) > 0 else 0.0
            crpn = float(toks8[1]) if len(toks8) > 1 else 0.0
            crpm = float(toks8[2]) if len(toks8) > 2 else 0.0
            f_a = int(float(toks8[3])) if len(toks8) > 3 else 0
            f_n = int(float(toks8[4])) if len(toks8) > 4 else 0
            f_m = int(float(toks8[5])) if len(toks8) > 5 else 0
            crp_law = int(float(toks8[6])) if len(toks8) > 6 else 0
        if len(valid_cards) > 8:
            toks9 = valid_cards[8].tokens()
            crsig = float(toks9[0]) if len(toks9) > 0 else 0.0
            crt = float(toks9[1]) if len(toks9) > 1 else 0.0
            crpq = float(toks9[2]) if len(toks9) > 2 else 0.0
            eps0 = float(toks9[3]) if len(toks9) > 3 else 0.0
            f_q = int(float(toks9[4])) if len(toks9) > 4 else 0
            f_sig = int(float(toks9[5])) if len(toks9) > 5 else 0

    m129 = MaterialLaw129(
        id=mat_id, title=title, rho0=rho0, e=e, nu=nu, sigy=sigy, alpha=alpha, tref=tref,
        f_young=f_young, f_nu=f_nu, f_yld=f_yld, f_alpha=f_alpha, isensor=isensor,
        itab=itab, facy=facy,
        qr1=qr1, cr1=cr1, qr2=qr2, cr2=cr2, f_qr=f_qr, f_cr=f_cr,
        qx1=qx1, cx1=cx1, qx2=qx2, cx2=cx2, f_qx=f_qx, f_cx=f_cx,
        epsp0=epsp0, cp=cp, f_cc=f_cc, f_cp=f_cp,
        crpa=crpa, crpn=crpn, crpm=crpm, f_a=f_a, f_n=f_n, f_m=f_m, crp_law=crp_law,
        crsig=crsig, crt=crt, crpq=crpq, eps0=eps0, f_q=f_q, f_sig=f_sig,
    )
    model.mat_law129s[mat_id] = m129
    from ..mat_reader import GenericMaterialRecord
    mat129 = Material(
        id=mat_id, law=129, rho0=rho0, title=title,
        params={
            "E": e if e > 0.0 else 1.0, "nu": nu if 0.0 <= nu < 0.5 else 0.3,
            "MAT_E": e, "MAT_NU": nu, "MAT_SIGY": sigy, "MAT_ALPHA": alpha, "MAT_TREF": tref,
            "MAT_f_young": f_young, "MAT_f_nu": f_nu, "MAT_f_yld": f_yld, "MAT_f_alpha": f_alpha, "ISENSOR": isensor,
            "MAT_ITAB": itab, "MAT_FACY": facy,
            "MAT_QR1": qr1, "MAT_CR1": cr1, "MAT_QR2": qr2, "MAT_CR2": cr2, "MAT_f_qr": f_qr, "MAT_f_cr": f_cr,
            "MAT_QX1": qx1, "MAT_CX1": cx1, "MAT_QX2": qx2, "MAT_CX2": cx2, "MAT_f_qx": f_qx, "MAT_f_cx": f_cx,
            "MAT_EPSP0": epsp0, "MAT_CP": cp, "MAT_f_cc": f_cc, "MAT_f_cp": f_cp,
            "MAT_CRPA": crpa, "MAT_CRPN": crpn, "MAT_CRPM": crpm, "MAT_fa": f_a, "MAT_fn": f_n, "MAT_fm": f_m, "MAT_CRPL": crp_law,
            "MAT_CRSIG": crsig, "MAT_CRT": crt, "MAT_CRPQ": crpq, "MAT_EPS0": eps0, "MAT_fq": f_q, "MAT_fsig": f_sig,
            "e": e, "nu": nu, "sigy": sigy, "alpha": alpha, "tref": tref,
            "f_young": f_young, "f_nu": f_nu, "f_yld": f_yld, "f_alpha": f_alpha, "isensor": isensor,
            "itab": itab, "facy": facy,
            "qr1": qr1, "cr1": cr1, "qr2": qr2, "cr2": cr2, "f_qr": f_qr, "f_cr": f_cr,
            "qx1": qx1, "cx1": cx1, "qx2": qx2, "cx2": cx2, "f_qx": f_qx, "f_cx": f_cx,
            "epsp0": epsp0, "cp": cp, "f_cc": f_cc, "f_cp": f_cp,
            "crpa": crpa, "crpn": crpn, "crpm": crpm, "f_a": f_a, "f_n": f_n, "f_m": f_m, "crp_law": crp_law,
            "crsig": crsig, "crt": crt, "crpq": crpq, "eps0": eps0, "f_q": f_q, "f_sig": f_sig,
        }
    )
    mat129.record = GenericMaterialRecord(
        law_name="LAW129", law_number=129, id=mat_id, title=title,
        params=mat129.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat129




def read_mat_law123(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW123/id`` or ``/MAT/DAIMLER_PINHO/id`` (M175): Daimler-Pinho 3D composite damage model."""
    from ...model.entities import MaterialLaw123, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho0 = 0.0
    ea, eb, ec = 0.0, 0.0, 0.0
    gab, gca, gbc = 0.0, 0.0, 0.0
    prba, prca, prcb = 0.0, 0.0, 0.0
    enkink, ena, enb, ent, enl = 0.0, 0.0, 0.0, 0.0, 0.0
    xc, xt, yc, yt, sl = 0.0, 0.0, 0.0, 0.0, 0.0
    fio, sigy = 53.0, 0.0
    lcss = 0
    beta = 0.0
    efs, ratio, fcut = 0.0, 0.0, 0.0

    valid_cards = [c for c in cards if not c.is_blank]

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW123_1")
            rho0 = _f(f1[0])
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW123_2")
            ea = _f(f2[0])
            eb = _f(f2[1]) if len(f2) > 1 else 0.0
            ec = _f(f2[2]) if len(f2) > 2 else 0.0
        if len(valid_cards) > 2:
            f3 = cut(valid_cards[2].raw, "MAT_LAW123_3")
            gab = _f(f3[0])
            gca = _f(f3[1]) if len(f3) > 1 else 0.0
            gbc = _f(f3[2]) if len(f3) > 2 else 0.0
        if len(valid_cards) > 3:
            f4 = cut(valid_cards[3].raw, "MAT_LAW123_4")
            prba = _f(f4[0])
            prca = _f(f4[1]) if len(f4) > 1 else 0.0
            prcb = _f(f4[2]) if len(f4) > 2 else 0.0
        if len(valid_cards) > 4:
            f5 = cut(valid_cards[4].raw, "MAT_LAW123_5")
            enkink = _f(f5[0])
            ena = _f(f5[1]) if len(f5) > 1 else 0.0
            enb = _f(f5[2]) if len(f5) > 2 else 0.0
            ent = _f(f5[3]) if len(f5) > 3 else 0.0
            enl = _f(f5[4]) if len(f5) > 4 else 0.0
        if len(valid_cards) > 5:
            f6 = cut(valid_cards[5].raw, "MAT_LAW123_6")
            xc = _f(f6[0])
            xt = _f(f6[1]) if len(f6) > 1 else 0.0
            yc = _f(f6[2]) if len(f6) > 2 else 0.0
            yt = _f(f6[3]) if len(f6) > 3 else 0.0
            sl = _f(f6[4]) if len(f6) > 4 else 0.0
        if len(valid_cards) > 6:
            f7 = cut(valid_cards[6].raw, "MAT_LAW123_7")
            fio = _f(f7[0]) if f7[0].strip() else 53.0
            sigy = _f(f7[1]) if len(f7) > 1 else 0.0
            lcss = _i(f7[2]) if len(f7) > 2 else 0
            beta = _f(f7[4]) if len(f7) > 4 else 0.0
        if len(valid_cards) > 7:
            f8 = cut(valid_cards[7].raw, "MAT_LAW123_8")
            efs = _f(f8[0])
            ratio = _f(f8[1]) if len(f8) > 1 else 0.0
            fcut = _f(f8[2]) if len(f8) > 2 else 0.0
    else:
        if len(valid_cards) > 0:
            toks1 = valid_cards[0].tokens()
            rho0 = float(toks1[0]) if len(toks1) > 0 else 0.0
        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            ea = float(toks2[0]) if len(toks2) > 0 else 0.0
            eb = float(toks2[1]) if len(toks2) > 1 else 0.0
            ec = float(toks2[2]) if len(toks2) > 2 else 0.0
        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            gab = float(toks3[0]) if len(toks3) > 0 else 0.0
            gca = float(toks3[1]) if len(toks3) > 1 else 0.0
            gbc = float(toks3[2]) if len(toks3) > 2 else 0.0
        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            prba = float(toks4[0]) if len(toks4) > 0 else 0.0
            prca = float(toks4[1]) if len(toks4) > 1 else 0.0
            prcb = float(toks4[2]) if len(toks4) > 2 else 0.0
        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            enkink = float(toks5[0]) if len(toks5) > 0 else 0.0
            ena = float(toks5[1]) if len(toks5) > 1 else 0.0
            enb = float(toks5[2]) if len(toks5) > 2 else 0.0
            ent = float(toks5[3]) if len(toks5) > 3 else 0.0
            enl = float(toks5[4]) if len(toks5) > 4 else 0.0
        if len(valid_cards) > 5:
            toks6 = valid_cards[5].tokens()
            xc = float(toks6[0]) if len(toks6) > 0 else 0.0
            xt = float(toks6[1]) if len(toks6) > 1 else 0.0
            yc = float(toks6[2]) if len(toks6) > 2 else 0.0
            yt = float(toks6[3]) if len(toks6) > 3 else 0.0
            sl = float(toks6[4]) if len(toks6) > 4 else 0.0
        if len(valid_cards) > 6:
            toks7 = valid_cards[6].tokens()
            fio = float(toks7[0]) if len(toks7) > 0 else 53.0
            sigy = float(toks7[1]) if len(toks7) > 1 else 0.0
            lcss = int(float(toks7[2])) if len(toks7) > 2 else 0
            beta = float(toks7[3]) if len(toks7) > 3 else 0.0
        if len(valid_cards) > 7:
            toks8 = valid_cards[7].tokens()
            efs = float(toks8[0]) if len(toks8) > 0 else 0.0
            ratio = float(toks8[1]) if len(toks8) > 1 else 0.0
            fcut = float(toks8[2]) if len(toks8) > 2 else 0.0

    m123 = MaterialLaw123(
        id=mat_id, title=title, rho0=rho0, ea=ea, eb=eb, ec=ec,
        gab=gab, gca=gca, gbc=gbc, prba=prba, prca=prca, prcb=prcb,
        enkink=enkink, ena=ena, enb=enb, ent=ent, enl=enl,
        xc=xc, xt=xt, yc=yc, yt=yt, sl=sl,
        fio=fio, sigy=sigy, lcss=lcss, beta=beta,
        efs=efs, ratio=ratio, fcut=fcut,
    )
    model.mat_law123s[mat_id] = m123
    e_eff = max(ea, eb, ec) if max(ea, eb, ec) > 0.0 else 1.0
    from ..mat_reader import GenericMaterialRecord
    mat123 = Material(
        id=mat_id, law=123, rho0=rho0, title=title,
        params={
            "E": e_eff, "nu": prba if 0.0 <= prba < 0.5 else 0.3,
            "LSDYNA_EA": ea, "LSDYNA_EB": eb, "LSDYNA_EC": ec,
            "LSDYNA_GAB": gab, "LSDYNA_GCA": gca, "LSDYNA_GBC": gbc,
            "LSDYNA_PRBA": prba, "LSDYNA_PRCA": prca, "LSDYNA_PRCB": prcb,
            "LSD_ENKINK": enkink, "LSD_ENA": ena, "LSD_ENB": enb, "LSD_ENT": ent, "LSD_ENL": enl,
            "LSD_XC": xc, "LSD_MAT_XT": xt, "LSD_MAT_YC": yc, "LSD_MAT_YT": yt, "LSD_SL": sl,
            "LSD_FIO": fio, "LSDYNA_SIGY": sigy, "LSD_LCSS": lcss, "LSD_MAT_BETA": beta,
            "EFS": efs, "LRD_RATIO": ratio, "FCUT": fcut,
            "ea": ea, "eb": eb, "ec": ec,
            "gab": gab, "gca": gca, "gbc": gbc,
            "prba": prba, "prca": prca, "prcb": prcb,
            "enkink": enkink, "ena": ena, "enb": enb, "ent": ent, "enl": enl,
            "xc": xc, "xt": xt, "yc": yc, "yt": yt, "sl": sl,
            "fio": fio, "sigy": sigy, "lcss": lcss, "beta": beta,
            "efs": efs, "ratio": ratio, "fcut": fcut,
        }
    )
    mat123.record = GenericMaterialRecord(
        law_name="LAW123", law_number=123, id=mat_id, title=title,
        params=mat123.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat123




def read_mat_law132(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW132/id`` or ``/MAT/DAIMLER_CAMANHO/id`` (M175): Daimler-Camanho composite failure model."""
    from ...model.entities import MaterialLaw132, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho0 = 0.0
    ea, eb, ec = 0.0, 0.0, 0.0
    gab, gca, gbc = 0.0, 0.0, 0.0
    prba, prca, prcb = 0.0, 0.0, 0.0
    gxc, gxt, gyc, gyt, gsl = 0.0, 0.0, 0.0, 0.0, 0.0
    xc, xt, yc, yt, sl = 0.0, 0.0, 0.0, 0.0, 0.0
    gxc0, gxt0, xc0, xt0 = 0.0, 0.0, 0.0, 0.0
    fio, sigy, etan, beta = 53.0, 0.0, 0.0, 0.0
    lcss = 0
    epsf23, epsr23, tsmd23 = 0.0, 0.0, 0.0
    epsf31, epsr31, tsmd31 = 0.0, 0.0, 0.0
    ef11t, ef11c, ef22t, ef22c, ef12 = 0.0, 0.0, 0.0, 0.0, 0.0
    ef23, ef31, cf12, cf23, cf31 = 0.0, 0.0, 0.0, 0.0, 0.0
    ratio, fcut = 0.0, 0.0

    valid_cards = [c for c in cards if not c.is_blank]

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW132_1")
            rho0 = _f(f1[0])
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW132_2")
            ea = _f(f2[0])
            eb = _f(f2[1]) if len(f2) > 1 else 0.0
            ec = _f(f2[2]) if len(f2) > 2 else 0.0
        if len(valid_cards) > 2:
            f3 = cut(valid_cards[2].raw, "MAT_LAW132_3")
            gab = _f(f3[0])
            gca = _f(f3[1]) if len(f3) > 1 else 0.0
            gbc = _f(f3[2]) if len(f3) > 2 else 0.0
        if len(valid_cards) > 3:
            f4 = cut(valid_cards[3].raw, "MAT_LAW132_4")
            prba = _f(f4[0])
            prca = _f(f4[1]) if len(f4) > 1 else 0.0
            prcb = _f(f4[2]) if len(f4) > 2 else 0.0
        if len(valid_cards) > 4:
            f5 = cut(valid_cards[4].raw, "MAT_LAW132_5")
            gxc = _f(f5[0])
            gxt = _f(f5[1]) if len(f5) > 1 else 0.0
            gyc = _f(f5[2]) if len(f5) > 2 else 0.0
            gyt = _f(f5[3]) if len(f5) > 3 else 0.0
            gsl = _f(f5[4]) if len(f5) > 4 else 0.0
        if len(valid_cards) > 5:
            f6 = cut(valid_cards[5].raw, "MAT_LAW132_6")
            xc = _f(f6[0])
            xt = _f(f6[1]) if len(f6) > 1 else 0.0
            yc = _f(f6[2]) if len(f6) > 2 else 0.0
            yt = _f(f6[3]) if len(f6) > 3 else 0.0
            sl = _f(f6[4]) if len(f6) > 4 else 0.0
        if len(valid_cards) > 6:
            f7 = cut(valid_cards[6].raw, "MAT_LAW132_7")
            gxc0 = _f(f7[0])
            gxt0 = _f(f7[1]) if len(f7) > 1 else 0.0
            xc0 = _f(f7[2]) if len(f7) > 2 else 0.0
            xt0 = _f(f7[3]) if len(f7) > 3 else 0.0
        if len(valid_cards) > 7:
            f8 = cut(valid_cards[7].raw, "MAT_LAW132_8")
            fio = _f(f8[0]) if f8[0].strip() else 53.0
            sigy = _f(f8[1]) if len(f8) > 1 else 0.0
            etan = _f(f8[2]) if len(f8) > 2 else 0.0
            beta = _f(f8[3]) if len(f8) > 3 else 0.0
            lcss = _i(f8[4]) if len(f8) > 4 else 0
        if len(valid_cards) > 8:
            f9 = cut(valid_cards[8].raw, "MAT_LAW132_9")
            epsf23 = _f(f9[0])
            epsr23 = _f(f9[1]) if len(f9) > 1 else 0.0
            tsmd23 = _f(f9[2]) if len(f9) > 2 else 0.0
        if len(valid_cards) > 9:
            f10 = cut(valid_cards[9].raw, "MAT_LAW132_10")
            epsf31 = _f(f10[0])
            epsr31 = _f(f10[1]) if len(f10) > 1 else 0.0
            tsmd31 = _f(f10[2]) if len(f10) > 2 else 0.0
        if len(valid_cards) > 10:
            f11 = cut(valid_cards[10].raw, "MAT_LAW132_11")
            ef11t = _f(f11[0])
            ef11c = _f(f11[1]) if len(f11) > 1 else 0.0
            ef22t = _f(f11[2]) if len(f11) > 2 else 0.0
            ef22c = _f(f11[3]) if len(f11) > 3 else 0.0
            ef12 = _f(f11[4]) if len(f11) > 4 else 0.0
        if len(valid_cards) > 11:
            f12 = cut(valid_cards[11].raw, "MAT_LAW132_12")
            ef23 = _f(f12[0])
            ef31 = _f(f12[1]) if len(f12) > 1 else 0.0
            cf12 = _f(f12[2]) if len(f12) > 2 else 0.0
            cf23 = _f(f12[3]) if len(f12) > 3 else 0.0
            cf31 = _f(f12[4]) if len(f12) > 4 else 0.0
        if len(valid_cards) > 12:
            f13 = cut(valid_cards[12].raw, "MAT_LAW132_13")
            ratio = _f(f13[0])
            fcut = _f(f13[1]) if len(f13) > 1 else 0.0
    else:
        if len(valid_cards) > 0:
            toks1 = valid_cards[0].tokens()
            rho0 = float(toks1[0]) if len(toks1) > 0 else 0.0
        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            ea = float(toks2[0]) if len(toks2) > 0 else 0.0
            eb = float(toks2[1]) if len(toks2) > 1 else 0.0
            ec = float(toks2[2]) if len(toks2) > 2 else 0.0
        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            gab = float(toks3[0]) if len(toks3) > 0 else 0.0
            gca = float(toks3[1]) if len(toks3) > 1 else 0.0
            gbc = float(toks3[2]) if len(toks3) > 2 else 0.0
        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            prba = float(toks4[0]) if len(toks4) > 0 else 0.0
            prca = float(toks4[1]) if len(toks4) > 1 else 0.0
            prcb = float(toks4[2]) if len(toks4) > 2 else 0.0
        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            gxc = float(toks5[0]) if len(toks5) > 0 else 0.0
            gxt = float(toks5[1]) if len(toks5) > 1 else 0.0
            gyc = float(toks5[2]) if len(toks5) > 2 else 0.0
            gyt = float(toks5[3]) if len(toks5) > 3 else 0.0
            gsl = float(toks5[4]) if len(toks5) > 4 else 0.0
        if len(valid_cards) > 5:
            toks6 = valid_cards[5].tokens()
            xc = float(toks6[0]) if len(toks6) > 0 else 0.0
            xt = float(toks6[1]) if len(toks6) > 1 else 0.0
            yc = float(toks6[2]) if len(toks6) > 2 else 0.0
            yt = float(toks6[3]) if len(toks6) > 3 else 0.0
            sl = float(toks6[4]) if len(toks6) > 4 else 0.0
        if len(valid_cards) > 6:
            toks7 = valid_cards[6].tokens()
            gxc0 = float(toks7[0]) if len(toks7) > 0 else 0.0
            gxt0 = float(toks7[1]) if len(toks7) > 1 else 0.0
            xc0 = float(toks7[2]) if len(toks7) > 2 else 0.0
            xt0 = float(toks7[3]) if len(toks7) > 3 else 0.0
        if len(valid_cards) > 7:
            toks8 = valid_cards[7].tokens()
            fio = float(toks8[0]) if len(toks8) > 0 else 53.0
            sigy = float(toks8[1]) if len(toks8) > 1 else 0.0
            etan = float(toks8[2]) if len(toks8) > 2 else 0.0
            beta = float(toks8[3]) if len(toks8) > 3 else 0.0
            lcss = int(float(toks8[4])) if len(toks8) > 4 else 0
        if len(valid_cards) > 8:
            toks9 = valid_cards[8].tokens()
            epsf23 = float(toks9[0]) if len(toks9) > 0 else 0.0
            epsr23 = float(toks9[1]) if len(toks9) > 1 else 0.0
            tsmd23 = float(toks9[2]) if len(toks9) > 2 else 0.0
        if len(valid_cards) > 9:
            toks10 = valid_cards[9].tokens()
            epsf31 = float(toks10[0]) if len(toks10) > 0 else 0.0
            epsr31 = float(toks10[1]) if len(toks10) > 1 else 0.0
            tsmd31 = float(toks10[2]) if len(toks10) > 2 else 0.0
        if len(valid_cards) > 10:
            toks11 = valid_cards[10].tokens()
            ef11t = float(toks11[0]) if len(toks11) > 0 else 0.0
            ef11c = float(toks11[1]) if len(toks11) > 1 else 0.0
            ef22t = float(toks11[2]) if len(toks11) > 2 else 0.0
            ef22c = float(toks11[3]) if len(toks11) > 3 else 0.0
            ef12 = float(toks11[4]) if len(toks11) > 4 else 0.0
        if len(valid_cards) > 11:
            toks12 = valid_cards[11].tokens()
            ef23 = float(toks12[0]) if len(toks12) > 0 else 0.0
            ef31 = float(toks12[1]) if len(toks12) > 1 else 0.0
            cf12 = float(toks12[2]) if len(toks12) > 2 else 0.0
            cf23 = float(toks12[3]) if len(toks12) > 3 else 0.0
            cf31 = float(toks12[4]) if len(toks12) > 4 else 0.0
        if len(valid_cards) > 12:
            toks13 = valid_cards[12].tokens()
            ratio = float(toks13[0]) if len(toks13) > 0 else 0.0
            fcut = float(toks13[1]) if len(toks13) > 1 else 0.0

    m132 = MaterialLaw132(
        id=mat_id, title=title, rho0=rho0, ea=ea, eb=eb, ec=ec,
        gab=gab, gca=gca, gbc=gbc, prba=prba, prca=prca, prcb=prcb,
        gxc=gxc, gxt=gxt, gyc=gyc, gyt=gyt, gsl=gsl,
        xc=xc, xt=xt, yc=yc, yt=yt, sl=sl,
        gxc0=gxc0, gxt0=gxt0, xc0=xc0, xt0=xt0,
        fio=fio, sigy=sigy, etan=etan, beta=beta, lcss=lcss,
        epsf23=epsf23, epsr23=epsr23, tsmd23=tsmd23,
        epsf31=epsf31, epsr31=epsr31, tsmd31=tsmd31,
        ef11t=ef11t, ef11c=ef11c, ef22t=ef22t, ef22c=ef22c, ef12=ef12,
        ef23=ef23, ef31=ef31, cf12=cf12, cf23=cf23, cf31=cf31,
        ratio=ratio, fcut=fcut,
    )
    model.mat_law132s[mat_id] = m132
    e_eff = max(ea, eb, ec) if max(ea, eb, ec) > 0.0 else 1.0
    from ..mat_reader import GenericMaterialRecord
    mat132 = Material(
        id=mat_id, law=132, rho0=rho0, title=title,
        params={
            "E": e_eff, "nu": prba if 0.0 <= prba < 0.5 else 0.3,
            "LSDYNA_EA": ea, "LSDYNA_EB": eb, "LSDYNA_EC": ec,
            "LSDYNA_GAB": gab, "LSDYNA_GCA": gca, "LSDYNA_GBC": gbc,
            "LSDYNA_PRBA": prba, "LSDYNA_PRCA": prca, "LSDYNA_PRCB": prcb,
            "LSD_GXC": gxc, "LSD_GXT": gxt, "LSD_GYC": gyc, "LSD_GYT": gyt, "LSD_GSL": gsl,
            "LSD_MAT_XC": xc, "LSD_MAT_XT": xt, "LSD_MAT_YC": yc, "LSD_MAT_YT": yt, "LSD_MAT_SL": sl,
            "LSD_GXC0": gxc0, "LSD_GXT0": gxt0, "LSD_MAT_XC0": xc0, "LSD_MAT_XT0": xt0,
            "LSD_FIO": fio, "LSDYNA_SIGY": sigy, "LSDYNA_ETAN": etan, "LSD_MAT_BETA": beta, "LSD_LCSS": lcss,
            "LSD_MAT_EPSF23": epsf23, "LSD_MAT_EPSR23": epsr23, "LSD_MAT_TSMD23": tsmd23,
            "LSD_MAT_EPSF31": epsf31, "LSD_MAT_EPSR31": epsr31, "LSD_MAT_TSMD31": tsmd31,
            "LSD_MAT_EF11T": ef11t, "LSD_MAT_EF11C": ef11c, "LSD_MAT_EF22T": ef22t, "LSD_MAT_EF22C": ef22c, "LSD_MAT_EF12": ef12,
            "LSD_MAT_EF23": ef23, "LSD_MAT_EF31": ef31, "LSD_MAT_CF12": cf12, "LSD_MAT_CF23": cf23, "LSD_MAT_CF31": cf31,
            "LRD_RATIO": ratio, "FCUT": fcut,
            "ea": ea, "eb": eb, "ec": ec,
            "gab": gab, "gca": gca, "gbc": gbc,
            "prba": prba, "prca": prca, "prcb": prcb,
            "gxc": gxc, "gxt": gxt, "gyc": gyc, "gyt": gyt, "gsl": gsl,
            "xc": xc, "xt": xt, "yc": yc, "yt": yt, "sl": sl,
            "gxc0": gxc0, "gxt0": gxt0, "xc0": xc0, "xt0": xt0,
            "fio": fio, "sigy": sigy, "etan": etan, "beta": beta, "lcss": lcss,
            "epsf23": epsf23, "epsr23": epsr23, "tsmd23": tsmd23,
            "epsf31": epsf31, "epsr31": epsr31, "tsmd31": tsmd31,
            "ef11t": ef11t, "ef11c": ef11c, "ef22t": ef22t, "ef22c": ef22c, "ef12": ef12,
            "ef23": ef23, "ef31": ef31, "cf12": cf12, "cf23": cf23, "cf31": cf31,
            "ratio": ratio, "fcut": fcut,
        }
    )
    mat132.record = GenericMaterialRecord(
        law_name="LAW132", law_number=132, id=mat_id, title=title,
        params=mat132.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat132




def read_mat_law134(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW134/id`` or ``/MAT/VISCOUS_FOAM/id`` (M175): Viscous foam material model."""
    from ...model.entities import MaterialLaw134, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho0 = 0.0
    e1, n1, nu = 0.0, 0.0, 0.0
    e2, v2, n2 = 0.0, 0.0, 0.0

    valid_cards = [c for c in cards if not c.is_blank]

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW134_1")
            rho0 = _f(f1[0])
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW134_2")
            e1 = _f(f2[0])
            n1 = _f(f2[1]) if len(f2) > 1 else 0.0
            nu = _f(f2[2]) if len(f2) > 2 else 0.0
        if len(valid_cards) > 2:
            f3 = cut(valid_cards[2].raw, "MAT_LAW134_3")
            e2 = _f(f3[0])
            v2 = _f(f3[1]) if len(f3) > 1 else 0.0
            n2 = _f(f3[2]) if len(f3) > 2 else 0.0
    else:
        if len(valid_cards) > 0:
            toks1 = valid_cards[0].tokens()
            rho0 = float(toks1[0]) if len(toks1) > 0 else 0.0
        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            e1 = float(toks2[0]) if len(toks2) > 0 else 0.0
            n1 = float(toks2[1]) if len(toks2) > 1 else 0.0
            nu = float(toks2[2]) if len(toks2) > 2 else 0.0
        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            e2 = float(toks3[0]) if len(toks3) > 0 else 0.0
            v2 = float(toks3[1]) if len(toks3) > 1 else 0.0
            n2 = float(toks3[2]) if len(toks3) > 2 else 0.0

    m134 = MaterialLaw134(
        id=mat_id, title=title, rho0=rho0,
        e1=e1, n1=n1, nu=nu, e2=e2, v2=v2, n2=n2,
    )
    model.mat_law134s[mat_id] = m134
    young = e1 + e2
    from ..mat_reader import GenericMaterialRecord
    mat134 = Material(
        id=mat_id, law=134, rho0=rho0, title=title,
        params={
            "E": young if young > 0.0 else 1.0, "nu": nu if 0.0 <= nu < 0.5 else 0.3,
            "LSD_MAT_E1": e1, "LSD_MAT_N1": n1, "MAT_NU": nu,
            "LSD_MAT_E2": e2, "LSD_MAT_V2": v2, "LSD_MAT_N2": n2,
            "e1": e1, "n1": n1, "nu": nu,
            "e2": e2, "v2": v2, "n2": n2,
        }
    )
    mat134.record = GenericMaterialRecord(
        law_name="LAW134", law_number=134, id=mat_id, title=title,
        params=mat134.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat134




def read_mat_law104(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW104/id`` or ``/MAT/JOHNS_VOCE_DRUCKER/id`` (M176): Combined Drucker-Prager and Voce hardening."""
    from ...model.entities import MaterialLaw104, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho0 = 0.0
    young, nu, ires = 0.0, 0.0, 1
    sigma_r, h, qv, bv, cdr = 1e20, 0.0, 0.0, 0.0, 0.0
    cjc, epsp0, fcut = 0.0, 1.0, 1e4
    tss, tref, tini = 0.0, 20.0, 20.0
    eta, cp, eps_iso, eps_ad = 0.0, 0.0, 1e20, 2e20

    valid_cards = [c for c in cards if not c.is_blank]

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW104_1")
            rho0 = _f(f1[0])
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW104_2")
            young = _f(f2[0])
            nu = _f(f2[1]) if len(f2) > 1 else 0.0
            ires = _i(f2[2]) if len(f2) > 2 and f2[2].strip() else 1
        if len(valid_cards) > 2:
            f3 = cut(valid_cards[2].raw, "MAT_LAW104_3")
            sigma_r = _f(f3[0]) if len(f3) > 0 and f3[0].strip() else 1e20
            h = _f(f3[1]) if len(f3) > 1 else 0.0
            qv = _f(f3[2]) if len(f3) > 2 else 0.0
            bv = _f(f3[3]) if len(f3) > 3 else 0.0
            cdr = _f(f3[4]) if len(f3) > 4 else 0.0
        if len(valid_cards) > 3:
            f4 = cut(valid_cards[3].raw, "MAT_LAW104_4")
            cjc = _f(f4[0]) if len(f4) > 0 else 0.0
            epsp0 = _f(f4[1]) if len(f4) > 1 and f4[1].strip() else 1.0
            if len(f4) > 2 and f4[2].strip():
                fcut = _f(f4[2])
        if len(valid_cards) > 4:
            f5 = cut(valid_cards[4].raw, "MAT_LAW104_5")
            tss = _f(f5[0]) if len(f5) > 0 else 0.0
            tref = _f(f5[1]) if len(f5) > 1 and f5[1].strip() else 20.0
            tini = _f(f5[2]) if len(f5) > 2 and f5[2].strip() else 20.0
        if len(valid_cards) > 5:
            f6 = cut(valid_cards[5].raw, "MAT_LAW104_6")
            eta = _f(f6[0]) if len(f6) > 0 else 0.0
            cp = _f(f6[1]) if len(f6) > 1 else 0.0
            eps_iso = _f(f6[2]) if len(f6) > 2 and f6[2].strip() else 1e20
            eps_ad = _f(f6[3]) if len(f6) > 3 and f6[3].strip() else 2e20
    else:
        if len(valid_cards) > 0:
            toks1 = valid_cards[0].tokens()
            rho0 = float(toks1[0]) if len(toks1) > 0 else 0.0
        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            young = float(toks2[0]) if len(toks2) > 0 else 0.0
            nu = float(toks2[1]) if len(toks2) > 1 else 0.0
            ires = int(toks2[2]) if len(toks2) > 2 else 1
        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            sigma_r = float(toks3[0]) if len(toks3) > 0 else 1e20
            h = float(toks3[1]) if len(toks3) > 1 else 0.0
            qv = float(toks3[2]) if len(toks3) > 2 else 0.0
            bv = float(toks3[3]) if len(toks3) > 3 else 0.0
            cdr = float(toks3[4]) if len(toks3) > 4 else 0.0
        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            cjc = float(toks4[0]) if len(toks4) > 0 else 0.0
            epsp0 = float(toks4[1]) if len(toks4) > 1 else 1.0
            if len(toks4) > 2:
                fcut = float(toks4[2])
        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            tss = float(toks5[0]) if len(toks5) > 0 else 0.0
            tref = float(toks5[1]) if len(toks5) > 1 else 20.0
            tini = float(toks5[2]) if len(toks5) > 2 else 20.0
        if len(valid_cards) > 5:
            toks6 = valid_cards[5].tokens()
            eta = float(toks6[0]) if len(toks6) > 0 else 0.0
            cp = float(toks6[1]) if len(toks6) > 1 else 0.0
            eps_iso = float(toks6[2]) if len(toks6) > 2 else 1e20
            eps_ad = float(toks6[3]) if len(toks6) > 3 else 2e20

    m104 = MaterialLaw104(
        id=mat_id, title=title, rho0=rho0,
        young=young, nu=nu, ires=ires,
        sigma_r=sigma_r, h=h, qv=qv, bv=bv, cdr=cdr,
        cjc=cjc, epsp0=epsp0, fcut=fcut,
        tss=tss, tref=tref, tini=tini,
        eta=eta, cp=cp, eps_iso=eps_iso, eps_ad=eps_ad,
    )
    model.mat_law104s[mat_id] = m104
    from ..mat_reader import GenericMaterialRecord
    mat104 = Material(
        id=mat_id, law=104, rho0=rho0, title=title,
        params={
            "E": young if young > 0.0 else 1.0, "nu": nu if 0.0 <= nu < 0.5 else 0.3,
            "SIGMA_r": sigma_r, "MAT_RHO": rho0, "MAT_E": young, "MAT_NU": nu,
            "MAT104_Ires": ires, "MAT104_H": h, "MAT_PR": qv, "MAT104_Bv": bv, "MAT104_Cdr": cdr,
            "MAT104_Cjc": cjc, "MAT104_Eps0": epsp0, "MAT104_Fcut": fcut,
            "MAT104_Tss": tss, "MAT104_Tref": tref, "T_Initial": tini,
            "MAT_ETA": eta, "MAT_SPHEAT": cp, "MAT104_EpsIso": eps_iso, "MAT104_EpsAd": eps_ad,
        }
    )
    mat104.record = GenericMaterialRecord(
        law_name="LAW104", law_number=104, id=mat_id, title=title,
        params=mat104.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat104




def read_mat_law105(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW105/id`` or ``/MAT/POWDER_BURN/id`` (M176/M576): Powder burn propellant model."""
    from ...model.entities import MaterialLaw105, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho0 = 0.0
    rhor = 0.0
    bulk, p0, psh = 0.0, 0.0, 0.0
    gas_d, gas_eg = 0.0, 0.0
    gr, c, alpha = 0.0, 0.0, 0.0
    func_b, scale_b, scale_p = 0, 1.0, 1.0
    func_gam, scale_gam, scale_rho, c1, c2 = 0, 1.0, 1.0, 0.0, 0.0

    valid_cards = [c for c in cards if not c.is_blank]

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW105_1")
            rho0 = _f(f1[0])
            rhor = _f(f1[1]) if len(f1) > 1 and f1[1].strip() else rho0
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW105_2")
            bulk = _f(f2[0])
            p0 = _f(f2[1]) if len(f2) > 1 else 0.0
            psh = _f(f2[2]) if len(f2) > 2 else 0.0
        if len(valid_cards) > 2:
            f3 = cut(valid_cards[2].raw, "MAT_LAW105_3")
            gas_d = _f(f3[0]) if len(f3) > 0 else 0.0
            gas_eg = _f(f3[1]) if len(f3) > 1 else 0.0
        if len(valid_cards) > 3:
            f4 = cut(valid_cards[3].raw, "MAT_LAW105_4")
            gr = _f(f4[0]) if len(f4) > 0 else 0.0
            c = _f(f4[1]) if len(f4) > 1 else 0.0
            alpha = _f(f4[2]) if len(f4) > 2 else 0.0
        if len(valid_cards) > 4:
            f5 = cut(valid_cards[4].raw, "MAT_LAW105_5")
            func_b = _i(f5[0]) if len(f5) > 0 else 0
            scale_b = _f(f5[2]) if len(f5) > 2 and f5[2].strip() else 1.0
            scale_p = _f(f5[3]) if len(f5) > 3 and f5[3].strip() else 1.0
        if len(valid_cards) > 5:
            f6 = cut(valid_cards[5].raw, "MAT_LAW105_6")
            func_gam = _i(f6[0]) if len(f6) > 0 else 0
            scale_gam = _f(f6[2]) if len(f6) > 2 and f6[2].strip() else 1.0
            scale_rho = _f(f6[3]) if len(f6) > 3 and f6[3].strip() else 1.0
            c1 = _f(f6[4]) if len(f6) > 4 else 0.0
            c2 = _f(f6[5]) if len(f6) > 5 else 0.0
    else:
        if len(valid_cards) > 0:
            toks1 = valid_cards[0].tokens()
            rho0 = float(toks1[0]) if len(toks1) > 0 else 0.0
            rhor = float(toks1[1]) if len(toks1) > 1 else rho0
        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            bulk = float(toks2[0]) if len(toks2) > 0 else 0.0
            p0 = float(toks2[1]) if len(toks2) > 1 else 0.0
            psh = float(toks2[2]) if len(toks2) > 2 else 0.0
        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            gas_d = float(toks3[0]) if len(toks3) > 0 else 0.0
            gas_eg = float(toks3[1]) if len(toks3) > 1 else 0.0
        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            gr = float(toks4[0]) if len(toks4) > 0 else 0.0
            c = float(toks4[1]) if len(toks4) > 1 else 0.0
            alpha = float(toks4[2]) if len(toks4) > 2 else 0.0
    compac = 0.93
    if block.fixed:
        if len(valid_cards) > 6:
            toks7 = valid_cards[6].tokens()
            if len(toks7) > 0:
                compac = float(toks7[0])
    else:
        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            func_b = int(toks5[0]) if len(toks5) > 0 else 0
            if len(toks5) >= 4:
                scale_b = float(toks5[2]) if toks5[2] else 1.0
                scale_p = float(toks5[3]) if toks5[3] else 1.0
            else:
                scale_b = float(toks5[1]) if len(toks5) > 1 else 1.0
                scale_p = float(toks5[2]) if len(toks5) > 2 else 1.0
        if len(valid_cards) > 5:
            toks6 = valid_cards[5].tokens()
            func_gam = int(toks6[0]) if len(toks6) > 0 else 0
            if len(toks6) >= 6:
                scale_gam = float(toks6[2]) if toks6[2] else 1.0
                scale_rho = float(toks6[3]) if toks6[3] else 1.0
                c1 = float(toks6[4]) if len(toks6) > 4 else 0.0
                c2 = float(toks6[5]) if len(toks6) > 5 else 0.0
            else:
                scale_gam = float(toks6[1]) if len(toks6) > 1 else 1.0
                scale_rho = float(toks6[2]) if len(toks6) > 2 else 1.0
                c1 = float(toks6[3]) if len(toks6) > 3 else 0.0
                c2 = float(toks6[4]) if len(toks6) > 4 else 0.0
        if len(valid_cards) > 6:
            toks7 = valid_cards[6].tokens()
            if len(toks7) > 0:
                compac = float(toks7[0])

    if rhor <= 0.0:
        rhor = rho0

    m105 = MaterialLaw105(
        id=mat_id, title=title, rho0=rho0,
        bulk=bulk, p0=p0, psh=psh,
        gas_d=gas_d, gas_eg=gas_eg,
        gr=gr, c=c, alpha=alpha,
        func_b=func_b, scale_b=scale_b, scale_p=scale_p,
        func_gam=func_gam, scale_gam=scale_gam, scale_rho=scale_rho,
        c1=c1, c2=c2,
        refer_rho=rhor, rhor=rhor,
        compac=compac,
    )
    mat_law_name = block.parts[1].upper() if len(block.parts) > 1 else "LAW105"
    m105.law_name = mat_law_name
    model.mat_law105s[mat_id] = m105
    from ..mat_reader import GenericMaterialRecord
    mat105 = Material(
        id=mat_id, law=105, rho0=rho0, title=title, law_name=mat_law_name,
        params={
            "E": 3.0 * bulk * (1.0 - 2.0 * 0.3) if bulk > 0.0 else 1.0, "nu": 0.3,
            "MAT_RHO": rho0, "Refer_Rho": rhor, "POWDER_BULK": bulk, "POWDER_P0": p0, "MAT_PSH": psh,
            "GAS_D": gas_d, "GAS_EG": gas_eg,
            "POWDER_Gr": gr, "POWDER_C": c, "Alpha": alpha,
            "POWDER_B_FUNC": func_b, "POWDER_SCALE_B": scale_b, "POWDER_SCALE_P": scale_p,
            "POWDER_GAM_FUNC": func_gam, "POWDER_SCALE_GAM": scale_gam, "POWDER_SCALE_RHO": scale_rho,
            "MAT_C1": c1, "MAT_C2": c2,
            "compac": compac, "COMPAC": compac,
        }
    )
    m105.params = mat105.params
    mat105.record = GenericMaterialRecord(
        law_name=mat_law_name, law_number=105, id=mat_id, title=title,
        params=mat105.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat105




def read_mat_law106(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW106``, ``/MAT/JCOOK_ALM`` (M176/M575): Johnson-Cook additive manufacturing model.

    Fortran origin: ``starter/source/materials/mat/mat106/hm_read_mat106.F90``.
    """
    from ...model.entities import MaterialLaw106, Material
    from ...materials.law106_jcook_alm import (
        _DEFAULT_EPS_MAX, _DEFAULT_SIGMA_MAX, _DEFAULT_FCUT, _DEFAULT_TOL,
        _DEFAULT_TMELT, _DEFAULT_TREF
    )
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]

    rho0, rhor = 0.0, 0.0
    young, nu = 0.0, 0.0
    fct_id1, fct_id2, fct_id3 = 0, 0, 0
    sigy, beta, hard_n = 0.0, 0.0, 1.0
    ep_max, sig_max = _DEFAULT_EPS_MAX, _DEFAULT_SIGMA_MAX
    fcut = _DEFAULT_FCUT
    vp, nmax = 2, 3
    tol = _DEFAULT_TOL
    cjc, deps0 = 0.0, 1.0
    m, tmelt = 1.0, _DEFAULT_TMELT
    spheat, eta, t0, tr = 0.0, 1.0, _DEFAULT_TREF, _DEFAULT_TREF
    params: Dict[str, Any] = {}

    card_idx = 0
    if block.fixed:
        if card_idx < len(valid_cards):
            f1 = cut(valid_cards[card_idx].raw, "MAT_LAW106_1")
            rho0 = _safe_float(f1[0]) if len(f1) > 0 else 0.0
            rhor = _safe_float(f1[1]) if len(f1) > 1 and f1[1].strip() else rho0
            card_idx += 1
        if card_idx < len(valid_cards):
            f2 = cut(valid_cards[card_idx].raw, "MAT_LAW106_2")
            young = _safe_float(f2[0]) if len(f2) > 0 else 0.0
            nu = _safe_float(f2[1]) if len(f2) > 1 else 0.0
            fct_id1 = _safe_int(f2[2]) if len(f2) > 2 else 0
            fct_id2 = _safe_int(f2[3]) if len(f2) > 3 else 0
            fct_id3 = _safe_int(f2[4]) if len(f2) > 4 else 0
            card_idx += 1
        if card_idx < len(valid_cards):
            f3 = cut(valid_cards[card_idx].raw, "MAT_LAW106_3")
            sigy = _safe_float(f3[0]) if len(f3) > 0 else 0.0
            beta = _safe_float(f3[1]) if len(f3) > 1 else 0.0
            hard_n = _safe_float(f3[2]) if len(f3) > 2 and f3[2].strip() else 1.0
            ep_max = _safe_float(f3[3]) if len(f3) > 3 and f3[3].strip() else _DEFAULT_EPS_MAX
            sig_max = _safe_float(f3[4]) if len(f3) > 4 and f3[4].strip() else _DEFAULT_SIGMA_MAX
            card_idx += 1
        if card_idx < len(valid_cards):
            f4 = cut(valid_cards[card_idx].raw, "MAT_LAW106_4")
            fcut = _safe_float(f4[0]) if len(f4) > 0 and f4[0].strip() else _DEFAULT_FCUT
            vp = _safe_int(f4[1]) if len(f4) > 1 and f4[1].strip() else 2
            nmax = _safe_int(f4[2]) if len(f4) > 2 and f4[2].strip() else 3
            tol = _safe_float(f4[3]) if len(f4) > 3 and f4[3].strip() else _DEFAULT_TOL
            cjc = _safe_float(f4[4]) if len(f4) > 4 else 0.0
            deps0 = _safe_float(f4[5]) if len(f4) > 5 and f4[5].strip() else 1.0
            card_idx += 1
        if card_idx < len(valid_cards):
            f5 = cut(valid_cards[card_idx].raw, "MAT_LAW106_5")
            m = _safe_float(f5[1]) if len(f5) > 1 and f5[1].strip() else 1.0
            tmelt = _safe_float(f5[2]) if len(f5) > 2 and f5[2].strip() else _DEFAULT_TMELT
            card_idx += 1
        if card_idx < len(valid_cards):
            f6 = cut(valid_cards[card_idx].raw, "MAT_LAW106_6")
            spheat = _safe_float(f6[0]) if len(f6) > 0 else 0.0
            eta = _safe_float(f6[1]) if len(f6) > 1 and f6[1].strip() else 1.0
            t0 = _safe_float(f6[2]) if len(f6) > 2 and f6[2].strip() else _DEFAULT_TREF
            tr = _safe_float(f6[3]) if len(f6) > 3 and f6[3].strip() else _DEFAULT_TREF
            card_idx += 1
    else:
        if card_idx < len(valid_cards):
            c0 = valid_cards[card_idx].tokens()
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else rho0
            card_idx += 1
        if card_idx < len(valid_cards):
            c1 = valid_cards[card_idx].tokens()
            young = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            fct_id1 = _safe_int(c1[2]) if len(c1) > 2 else 0
            fct_id2 = _safe_int(c1[3]) if len(c1) > 3 else 0
            fct_id3 = _safe_int(c1[4]) if len(c1) > 4 else 0
            card_idx += 1
        if card_idx < len(valid_cards):
            c2 = valid_cards[card_idx].tokens()
            sigy = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            beta = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            hard_n = _safe_float(c2[2]) if len(c2) > 2 and c2[2].strip() else 1.0
            ep_max = _safe_float(c2[3]) if len(c2) > 3 and c2[3].strip() else _DEFAULT_EPS_MAX
            sig_max = _safe_float(c2[4]) if len(c2) > 4 and c2[4].strip() else _DEFAULT_SIGMA_MAX
            card_idx += 1
        if card_idx < len(valid_cards):
            c3 = valid_cards[card_idx].tokens()
            fcut = _safe_float(c3[0]) if len(c3) > 0 and c3[0].strip() else _DEFAULT_FCUT
            vp = _safe_int(c3[1]) if len(c3) > 1 and c3[1].strip() else 2
            nmax = _safe_int(c3[2]) if len(c3) > 2 and c3[2].strip() else 3
            tol = _safe_float(c3[3]) if len(c3) > 3 and c3[3].strip() else _DEFAULT_TOL
            cjc = _safe_float(c3[4]) if len(c3) > 4 else 0.0
            deps0 = _safe_float(c3[5]) if len(c3) > 5 and c3[5].strip() else 1.0
            card_idx += 1
        if card_idx < len(valid_cards):
            c4 = valid_cards[card_idx].tokens()
            if len(c4) == 2:
                m = _safe_float(c4[0]) if c4[0].strip() else 1.0
                tmelt = _safe_float(c4[1]) if c4[1].strip() else _DEFAULT_TMELT
            elif len(c4) >= 3:
                m = _safe_float(c4[-2]) if c4[-2].strip() else 1.0
                tmelt = _safe_float(c4[-1]) if c4[-1].strip() else _DEFAULT_TMELT
            card_idx += 1
        if card_idx < len(valid_cards):
            c5 = valid_cards[card_idx].tokens()
            spheat = _safe_float(c5[0]) if len(c5) > 0 else 0.0
            eta = _safe_float(c5[1]) if len(c5) > 1 and c5[1].strip() else 1.0
            t0 = _safe_float(c5[2]) if len(c5) > 2 and c5[2].strip() else _DEFAULT_TREF
            tr = _safe_float(c5[3]) if len(c5) > 3 and c5[3].strip() else _DEFAULT_TREF
            card_idx += 1

    # Apply defaults matching hm_read_mat106.F90
    if rhor == 0.0:
        rhor = rho0
    if ep_max == 0.0:
        ep_max = _DEFAULT_EPS_MAX
    if sig_max == 0.0:
        sig_max = _DEFAULT_SIGMA_MAX
    if deps0 == 0.0:
        deps0 = 1.0
    if tol == 0.0:
        tol = _DEFAULT_TOL
    if tmelt <= 0.0:
        tmelt = _DEFAULT_TMELT
    if tr <= 0.0:
        tr = _DEFAULT_TREF
    if t0 <= 0.0:
        t0 = tr
    if eta == 0.0:
        eta = 1.0
    vp = min(max(vp, 0), 3)
    if vp == 0:
        vp = 2
    if nmax == 0:
        nmax = 6 if vp == 1 else 3
    if vp == 1:
        fcut = 0.0
    elif fcut == 0.0:
        fcut = _DEFAULT_FCUT

    params.update({
        "rho": rho0, "rho0": rho0, "refer_rho": rhor, "rhor": rhor,
        "MAT_RHO": rho0, "Refer_Rho": rhor,
        "young": young, "e": young, "E": young, "nu": nu, "Nu": nu,
        "MAT_E": young, "MAT_NU": nu,
        "fct_id1": fct_id1, "fct_id2": fct_id2, "fct_id3": fct_id3,
        "MLAW106_FCT_ID1": fct_id1, "MLAW106_FCT_ID2": fct_id2, "MLAW106_FCT_ID3": fct_id3,
        "a": sigy, "b": beta, "n": hard_n,
        "sigy": sigy, "beta": beta, "hard_n": hard_n,
        "MAT_SIGY": sigy, "MAT_BETA": beta, "MAT_HARD": hard_n,
        "eps_max": ep_max, "sigma_max": sig_max,
        "MLAW106_EP_MAX": ep_max, "MLAW106_SIGMA_MAX": sig_max,
        "fcut": fcut, "vp": vp, "nmax": nmax, "tol": tol,
        "MLAW106_FCUT": fcut, "MLAW106_VP": vp, "MLAW106_NMAX": nmax, "MLAW106_TOL": tol,
        "cjc": cjc, "deps0": deps0,
        "MLAW106_CJC": cjc, "MLAW106_DEPS0": deps0,
        "m": m, "tmelt": tmelt,
        "MAT_M": m, "MAT_TMELT": tmelt,
        "cs": spheat, "spheat": spheat, "rhocp": spheat,
        "MAT_SPHEAT": spheat,
        "eta": eta, "MLAW106_ETA": eta,
        "t0": t0, "MLAW106_T0": t0,
        "tref": tr, "tr": tr, "MLAW106_TR": tr,
    })

    mat = MaterialLaw106(
        id=mat_id, title=title, rho0=rho0, rhor=rhor,
        young=young, nu=nu,
        fct_id1=fct_id1, fct_id2=fct_id2, fct_id3=fct_id3,
        sigy=sigy, beta=beta, hard_n=hard_n,
        ep_max=ep_max, sig_max=sig_max,
        fcut=fcut, vp=vp, nmax=nmax, tol=tol,
        cjc=cjc, deps0=deps0,
        m=m, tmelt=tmelt,
        spheat=spheat, eta=eta, t0=t0, tr=tr,
        params=params, law=106, law_name="LAW106"
    )
    model.mat_law106s[mat_id] = mat
    from ..mat_reader import GenericMaterialRecord
    mat_ent = Material(id=mat_id, law=106, rho0=rho0, title=title, params=params)
    mat_ent.record = GenericMaterialRecord(
        law_name="LAW106", law_number=106, id=mat_id, title=title,
        params=params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat_ent




read_mat_jcook_alm = read_mat_law106


read_mat_johns_cook_alm = read_mat_law106




def read_mat_law107(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW107/id`` or ``/MAT/PAPER_LIGHT/id`` (M176): Paper plasticity model."""
    from ...model.entities import MaterialLaw107, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho0, rhor = 0.0, 0.0
    e1, e2, e3 = 0.0, 0.0, 0.0
    ires, itab, ismooth = 0, 0, 0
    nu21, g12, g23, g13 = 0.0, 0.0, 0.0, 0.0
    xi1, xi2, g1c, d1, d2 = 0.0, 0.0, 0.0, 0.0, 0.0
    k1, k2, k3 = 0.0, 0.0, 0.0
    k4, k5, k6 = 0.0, 0.0, 0.0
    sigy1, cini1, s1 = 0.0, 0.0, 0.0
    sigy2, cini2, s2 = 0.0, 0.0, 0.0
    sigy1c, cini1c, s1c = 0.0, 0.0, 0.0
    sigy2c, cini2c, s2c = 0.0, 0.0, 0.0
    sigyt, cinit, st = 0.0, 0.0, 0.0
    tab_yld1, xscale1, yscale1 = 0, 1.0, 1.0
    tab_yld2, xscale2, yscale2 = 0, 1.0, 1.0
    tab_yld1c, xscale1c, yscale1c = 0, 1.0, 1.0
    tab_yld2c, xscale2c, yscale2c = 0, 1.0, 1.0
    tab_yldt, xscale_t, yscale_t = 0, 1.0, 1.0

    valid_cards = [c for c in cards if not c.is_blank]

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW107_1")
            rho0 = _f(f1[0])
            rhor = _f(f1[1]) if len(f1) > 1 and f1[1].strip() else rho0
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW107_2")
            e1 = _f(f2[0])
            e2 = _f(f2[1]) if len(f2) > 1 else 0.0
            e3 = _f(f2[2]) if len(f2) > 2 else 0.0
            ires = _i(f2[3]) if len(f2) > 3 else 0
            itab = _i(f2[4]) if len(f2) > 4 else 0
            ismooth = _i(f2[5]) if len(f2) > 5 else 0
        if len(valid_cards) > 2:
            f3 = cut(valid_cards[2].raw, "MAT_LAW107_3")
            nu21 = _f(f3[0]) if len(f3) > 0 else 0.0
            g12 = _f(f3[1]) if len(f3) > 1 else 0.0
            g23 = _f(f3[2]) if len(f3) > 2 else 0.0
            g13 = _f(f3[3]) if len(f3) > 3 else 0.0
        if len(valid_cards) > 3:
            f4 = cut(valid_cards[3].raw, "MAT_LAW107_4")
            xi1 = _f(f4[0]) if len(f4) > 0 else 0.0
            xi2 = _f(f4[1]) if len(f4) > 1 else 0.0
            g1c = _f(f4[2]) if len(f4) > 2 else 0.0
            d1 = _f(f4[3]) if len(f4) > 3 else 0.0
            d2 = _f(f4[4]) if len(f4) > 4 else 0.0
        if len(valid_cards) > 4:
            f5 = cut(valid_cards[4].raw, "MAT_LAW107_5")
            k1 = _f(f5[0]) if len(f5) > 0 else 0.0
            k2 = _f(f5[1]) if len(f5) > 1 else 0.0
            k3 = _f(f5[2]) if len(f5) > 2 else 0.0
        if len(valid_cards) > 5:
            f6 = cut(valid_cards[5].raw, "MAT_LAW107_6")
            k4 = _f(f6[0]) if len(f6) > 0 else 0.0
            k5 = _f(f6[1]) if len(f6) > 1 else 0.0
            k6 = _f(f6[2]) if len(f6) > 2 else 0.0
        if itab == 0:
            if len(valid_cards) > 6:
                f7 = cut(valid_cards[6].raw, "MAT_LAW107_7")
                sigy1 = _f(f7[0]) if len(f7) > 0 else 0.0
                cini1 = _f(f7[1]) if len(f7) > 1 else 0.0
                s1 = _f(f7[2]) if len(f7) > 2 else 0.0
            if len(valid_cards) > 7:
                f8 = cut(valid_cards[7].raw, "MAT_LAW107_8")
                sigy2 = _f(f8[0]) if len(f8) > 0 else 0.0
                cini2 = _f(f8[1]) if len(f8) > 1 else 0.0
                s2 = _f(f8[2]) if len(f8) > 2 else 0.0
            if len(valid_cards) > 8:
                f9 = cut(valid_cards[8].raw, "MAT_LAW107_9")
                sigy1c = _f(f9[0]) if len(f9) > 0 else 0.0
                cini1c = _f(f9[1]) if len(f9) > 1 else 0.0
                s1c = _f(f9[2]) if len(f9) > 2 else 0.0
            if len(valid_cards) > 9:
                f10 = cut(valid_cards[9].raw, "MAT_LAW107_10")
                sigy2c = _f(f10[0]) if len(f10) > 0 else 0.0
                cini2c = _f(f10[1]) if len(f10) > 1 else 0.0
                s2c = _f(f10[2]) if len(f10) > 2 else 0.0
            if len(valid_cards) > 10:
                f11 = cut(valid_cards[10].raw, "MAT_LAW107_11")
                sigyt = _f(f11[0]) if len(f11) > 0 else 0.0
                cinit = _f(f11[1]) if len(f11) > 1 else 0.0
                st = _f(f11[2]) if len(f11) > 2 else 0.0
        else:
            if len(valid_cards) > 6:
                f7 = cut(valid_cards[6].raw, "MAT_LAW107_TAB")
                tab_yld1 = _i(f7[1]) if len(f7) > 1 else 0
                xscale1 = _f(f7[2]) if len(f7) > 2 and f7[2].strip() else 1.0
                yscale1 = _f(f7[3]) if len(f7) > 3 and f7[3].strip() else 1.0
            if len(valid_cards) > 7:
                f8 = cut(valid_cards[7].raw, "MAT_LAW107_TAB")
                tab_yld2 = _i(f8[1]) if len(f8) > 1 else 0
                xscale2 = _f(f8[2]) if len(f8) > 2 and f8[2].strip() else 1.0
                yscale2 = _f(f8[3]) if len(f8) > 3 and f8[3].strip() else 1.0
            if len(valid_cards) > 8:
                f9 = cut(valid_cards[8].raw, "MAT_LAW107_TAB")
                tab_yld1c = _i(f9[1]) if len(f9) > 1 else 0
                xscale1c = _f(f9[2]) if len(f9) > 2 and f9[2].strip() else 1.0
                yscale1c = _f(f9[3]) if len(f9) > 3 and f9[3].strip() else 1.0
            if len(valid_cards) > 9:
                f10 = cut(valid_cards[9].raw, "MAT_LAW107_TAB")
                tab_yld2c = _i(f10[1]) if len(f10) > 1 else 0
                xscale2c = _f(f10[2]) if len(f10) > 2 and f10[2].strip() else 1.0
                yscale2c = _f(f10[3]) if len(f10) > 3 and f10[3].strip() else 1.0
            if len(valid_cards) > 10:
                f11 = cut(valid_cards[10].raw, "MAT_LAW107_TAB")
                tab_yldt = _i(f11[1]) if len(f11) > 1 else 0
                xscale_t = _f(f11[2]) if len(f11) > 2 and f11[2].strip() else 1.0
                yscale_t = _f(f11[3]) if len(f11) > 3 and f11[3].strip() else 1.0
    else:
        if len(valid_cards) > 0:
            toks1 = valid_cards[0].tokens()
            rho0 = float(toks1[0]) if len(toks1) > 0 else 0.0
            rhor = float(toks1[1]) if len(toks1) > 1 else rho0
        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            e1 = float(toks2[0]) if len(toks2) > 0 else 0.0
            e2 = float(toks2[1]) if len(toks2) > 1 else 0.0
            e3 = float(toks2[2]) if len(toks2) > 2 else 0.0
            ires = int(toks2[3]) if len(toks2) > 3 else 0
            itab = int(toks2[4]) if len(toks2) > 4 else 0
            ismooth = int(toks2[5]) if len(toks2) > 5 else 0
        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            nu21 = float(toks3[0]) if len(toks3) > 0 else 0.0
            g12 = float(toks3[1]) if len(toks3) > 1 else 0.0
            g23 = float(toks3[2]) if len(toks3) > 2 else 0.0
            g13 = float(toks3[3]) if len(toks3) > 3 else 0.0
        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            xi1 = float(toks4[0]) if len(toks4) > 0 else 0.0
            xi2 = float(toks4[1]) if len(toks4) > 1 else 0.0
            g1c = float(toks4[2]) if len(toks4) > 2 else 0.0
            d1 = float(toks4[3]) if len(toks4) > 3 else 0.0
            d2 = float(toks4[4]) if len(toks4) > 4 else 0.0
        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            k1 = float(toks5[0]) if len(toks5) > 0 else 0.0
            k2 = float(toks5[1]) if len(toks5) > 1 else 0.0
            k3 = float(toks5[2]) if len(toks5) > 2 else 0.0
        if len(valid_cards) > 5:
            toks6 = valid_cards[5].tokens()
            k4 = float(toks6[0]) if len(toks6) > 0 else 0.0
            k5 = float(toks6[1]) if len(toks6) > 1 else 0.0
            k6 = float(toks6[2]) if len(toks6) > 2 else 0.0
        if itab == 0:
            if len(valid_cards) > 6:
                toks7 = valid_cards[6].tokens()
                sigy1 = float(toks7[0]) if len(toks7) > 0 else 0.0
                cini1 = float(toks7[1]) if len(toks7) > 1 else 0.0
                s1 = float(toks7[2]) if len(toks7) > 2 else 0.0
            if len(valid_cards) > 7:
                toks8 = valid_cards[7].tokens()
                sigy2 = float(toks8[0]) if len(toks8) > 0 else 0.0
                cini2 = float(toks8[1]) if len(toks8) > 1 else 0.0
                s2 = float(toks8[2]) if len(toks8) > 2 else 0.0
            if len(valid_cards) > 8:
                toks9 = valid_cards[8].tokens()
                sigy1c = float(toks9[0]) if len(toks9) > 0 else 0.0
                cini1c = float(toks9[1]) if len(toks9) > 1 else 0.0
                s1c = float(toks9[2]) if len(toks9) > 2 else 0.0
            if len(valid_cards) > 9:
                toks10 = valid_cards[9].tokens()
                sigy2c = float(toks10[0]) if len(toks10) > 0 else 0.0
                cini2c = float(toks10[1]) if len(toks10) > 1 else 0.0
                s2c = float(toks10[2]) if len(toks10) > 2 else 0.0
            if len(valid_cards) > 10:
                toks11 = valid_cards[10].tokens()
                sigyt = float(toks11[0]) if len(toks11) > 0 else 0.0
                cinit = float(toks11[1]) if len(toks11) > 1 else 0.0
                st = float(toks11[2]) if len(toks11) > 2 else 0.0
        else:
            if len(valid_cards) > 6:
                toks7 = valid_cards[6].tokens()
                tab_yld1 = int(toks7[0]) if len(toks7) > 0 else 0
                xscale1 = float(toks7[1]) if len(toks7) > 1 else 1.0
                yscale1 = float(toks7[2]) if len(toks7) > 2 else 1.0
            if len(valid_cards) > 7:
                toks8 = valid_cards[7].tokens()
                tab_yld2 = int(toks8[0]) if len(toks8) > 0 else 0
                xscale2 = float(toks8[1]) if len(toks8) > 1 else 1.0
                yscale2 = float(toks8[2]) if len(toks8) > 2 else 1.0
            if len(valid_cards) > 8:
                toks9 = valid_cards[8].tokens()
                tab_yld1c = int(toks9[0]) if len(toks9) > 0 else 0
                xscale1c = float(toks9[1]) if len(toks9) > 1 else 1.0
                yscale1c = float(toks9[2]) if len(toks9) > 2 else 1.0
            if len(valid_cards) > 9:
                toks10 = valid_cards[9].tokens()
                tab_yld2c = int(toks10[0]) if len(toks10) > 0 else 0
                xscale2c = float(toks10[1]) if len(toks10) > 1 else 1.0
                yscale2c = float(toks10[2]) if len(toks10) > 2 else 1.0
            if len(valid_cards) > 10:
                toks11 = valid_cards[10].tokens()
                tab_yldt = int(toks11[0]) if len(toks11) > 0 else 0
                xscale_t = float(toks11[1]) if len(toks11) > 1 else 1.0
                yscale_t = float(toks11[2]) if len(toks11) > 2 else 1.0

    m107 = MaterialLaw107(
        id=mat_id, title=title, rho0=rho0, rhor=rhor,
        e1=e1, e2=e2, e3=e3,
        ires=ires, itab=itab, ismooth=ismooth,
        nu21=nu21, g12=g12, g23=g23, g13=g13,
        xi1=xi1, xi2=xi2, g1c=g1c, d1=d1, d2=d2,
        k1=k1, k2=k2, k3=k3, k4=k4, k5=k5, k6=k6,
        sigy1=sigy1, cini1=cini1, s1=s1,
        sigy2=sigy2, cini2=cini2, s2=s2,
        sigy1c=sigy1c, cini1c=cini1c, s1c=s1c,
        sigy2c=sigy2c, cini2c=cini2c, s2c=s2c,
        sigyt=sigyt, cinit=cinit, st=st,
        tab_yld1=tab_yld1, xscale1=xscale1, yscale1=yscale1,
        tab_yld2=tab_yld2, xscale2=xscale2, yscale2=yscale2,
        tab_yld1c=tab_yld1c, xscale1c=xscale1c, yscale1c=yscale1c,
        tab_yld2c=tab_yld2c, xscale2c=xscale2c, yscale2c=yscale2c,
        tab_yldt=tab_yldt, xscale_t=xscale_t, yscale_t=yscale_t,
    )
    model.mat_law107s[mat_id] = m107
    e_eff = max(e1, e2, e3) if max(e1, e2, e3) > 0.0 else 1.0
    from ..mat_reader import GenericMaterialRecord
    mat107 = Material(
        id=mat_id, law=107, rho0=rho0, title=title,
        params={
            "E": e_eff, "nu": nu21 if 0.0 <= nu21 < 0.5 else 0.3,
            "MAT_RHO": rho0, "Refer_Rho": rhor,
            "MAT_E1": e1, "MAT_E2": e2, "MAT_E3": e3,
            "MAT_IRES": ires, "MAT_ITAB": itab, "MAT_SMOOTH": ismooth,
            "MAT_NU21": nu21, "MAT_G12": g12, "MAT_G23": g23, "MAT_G13": g13,
            "MAT_XI1": xi1, "MAT_XI2": xi2, "MAT_G1C": g1c, "MAT_D1": d1, "MAT_D2": d2,
            "MAT_K1": k1, "MAT_K2": k2, "MAT_K3": k3, "MAT_K4": k4, "MAT_K5": k5, "MAT_K6": k6,
            "MAT_SIGY1": sigy1, "MAT_CINI1": cini1, "MAT_S1": s1,
            "MAT_SIGY2": sigy2, "MAT_CINI2": cini2, "MAT_S2": s2,
            "MAT_SIGY1C": sigy1c, "MAT_CINI1C": cini1c, "MAT_S1C": s1c,
            "MAT_SIGY2C": sigy2c, "MAT_CINI2C": cini2c, "MAT_S2C": s2c,
            "MAT_SIGYT": sigyt, "MAT_CINIT": cinit, "MAT_ST": st,
            "TAB_YLD1": tab_yld1, "MAT_Xscale1": xscale1, "MAT_Yscale1": yscale1,
            "TAB_YLD2": tab_yld2, "MAT_Xscale2": xscale2, "MAT_Yscale2": yscale2,
            "TAB_YLD1C": tab_yld1c, "MAT_Xscale1C": xscale1c, "MAT_Yscale1C": yscale1c,
            "TAB_YLD2C": tab_yld2c, "MAT_Xscale2C": xscale2c, "MAT_Yscale2C": yscale2c,
            "TAB_YLDT": tab_yldt, "MAT_XscaleT": xscale_t, "MAT_YscaleT": yscale_t,
        }
    )
    mat107.e1 = e1
    mat107.e2 = e2
    mat107.e3 = e3
    mat107.nu21 = nu21
    mat107.g12 = g12
    mat107.g23 = g23
    mat107.g13 = g13
    mat107.g31 = g13
    mat107.ires = ires
    mat107.itab = itab
    mat107.ismooth = ismooth
    mat107.xi1 = xi1
    mat107.xi2 = xi2
    mat107.g1c = g1c
    mat107.d1 = d1
    mat107.d2 = d2
    mat107.k1 = k1
    mat107.k2 = k2
    mat107.k3 = k3
    mat107.k4 = k4
    mat107.k5 = k5
    mat107.k6 = k6
    mat107.sigy1 = sigy1
    mat107.sigy2 = sigy2
    mat107.sigy1c = sigy1c
    mat107.sigy2c = sigy2c
    mat107.sigyt = sigyt
    mat107.record = GenericMaterialRecord(
        law_name="LAW107", law_number=107, id=mat_id, title=title,
        params=mat107.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat107




read_mat_paper_light = read_mat_law107


read_mat_plas_paper_light = read_mat_law107


read_mat_pfeiffer = read_mat_law107





def read_mat_law110(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW110/id`` or ``/MAT/VEGTER/id`` (M176): Vegter anisotropic yield locus model."""
    from ...model.entities import MaterialLaw110, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho0, rhor = 0.0, 0.0
    young, nu, ires = 0.0, 0.0, 0
    icrit, tab_yld = 1, 0
    xscale, yscale = 1.0, 1.0
    fbi, rhobi = 0.0, 0.0
    sigma_r, dsigm, beta, omega, hard_n = 0.0, 0.0, 0.0, 0.0, 0.0
    eps0, sigs, dg0, deps0, m = 0.0, 0.0, 0.0, 0.0, 0.0
    tini, chard, fcut, vp, ismooth, tab_temp = 0.0, 0.0, 0.0, 0, 0, 0
    rm_0, rm_45, rm_90 = 0.0, 0.0, 0.0
    ag_0, ag_45, ag_90 = 0.0, 0.0, 0.0
    r_0, r_45, r_90 = 1.0, 1.0, 1.0
    angles_data = []

    valid_cards = [c for c in cards if not c.is_blank]

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW110_1")
            rho0 = _f(f1[0])
            rhor = _f(f1[1]) if len(f1) > 1 and f1[1].strip() else rho0
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW110_2")
            young = _f(f2[0])
            nu = _f(f2[1]) if len(f2) > 1 else 0.0
            ires = _i(f2[2]) if len(f2) > 2 else 0
        if len(valid_cards) > 2:
            f3 = cut(valid_cards[2].raw, "MAT_LAW110_3")
            icrit = _i(f3[0]) if len(f3) > 0 and f3[0].strip() else 1
            tab_yld = _i(f3[1]) if len(f3) > 1 else 0
            xscale = _f(f3[2]) if len(f3) > 2 and f3[2].strip() else 1.0
            yscale = _f(f3[3]) if len(f3) > 3 and f3[3].strip() else 1.0
            fbi = _f(f3[4]) if len(f3) > 4 else 0.0
            rhobi = _f(f3[5]) if len(f3) > 5 else 0.0
        if len(valid_cards) > 3:
            f4 = cut(valid_cards[3].raw, "MAT_LAW110_4")
            sigma_r = _f(f4[0]) if len(f4) > 0 else 0.0
            dsigm = _f(f4[1]) if len(f4) > 1 else 0.0
            beta = _f(f4[2]) if len(f4) > 2 else 0.0
            omega = _f(f4[3]) if len(f4) > 3 else 0.0
            hard_n = _f(f4[4]) if len(f4) > 4 else 0.0
        if len(valid_cards) > 4:
            f5 = cut(valid_cards[4].raw, "MAT_LAW110_5")
            eps0 = _f(f5[0]) if len(f5) > 0 else 0.0
            sigs = _f(f5[1]) if len(f5) > 1 else 0.0
            dg0 = _f(f5[2]) if len(f5) > 2 else 0.0
            deps0 = _f(f5[3]) if len(f5) > 3 else 0.0
            m = _f(f5[4]) if len(f5) > 4 else 0.0
        if len(valid_cards) > 5:
            f6 = cut(valid_cards[5].raw, "MAT_LAW110_6")
            tini = _f(f6[0]) if len(f6) > 0 else 0.0
            chard = _f(f6[1]) if len(f6) > 1 else 0.0
            fcut = _f(f6[2]) if len(f6) > 2 else 0.0
            vp = _i(f6[3]) if len(f6) > 3 else 0
            ismooth = _i(f6[4]) if len(f6) > 4 else 0
            tab_temp = _i(f6[5]) if len(f6) > 5 else 0
        if icrit == 3:
            if len(valid_cards) > 6:
                f7 = cut(valid_cards[6].raw, "MAT_LAW110_7_3")
                rm_0 = _f(f7[0]) if len(f7) > 0 else 0.0
                rm_45 = _f(f7[1]) if len(f7) > 1 else 0.0
                rm_90 = _f(f7[2]) if len(f7) > 2 else 0.0
                ag_0 = _f(f7[3]) if len(f7) > 3 else 0.0
                ag_45 = _f(f7[4]) if len(f7) > 4 else 0.0
            if len(valid_cards) > 7:
                f8 = cut(valid_cards[7].raw, "MAT_LAW110_8_3")
                ag_90 = _f(f8[0]) if len(f8) > 0 else 0.0
                r_0 = _f(f8[1]) if len(f8) > 1 and f8[1].strip() else 1.0
                r_45 = _f(f8[2]) if len(f8) > 2 and f8[2].strip() else 1.0
                r_90 = _f(f8[3]) if len(f8) > 3 and f8[3].strip() else 1.0
        else:
            for c_idx in range(6, len(valid_cards)):
                if icrit == 4:
                    f_ang = cut(valid_cards[c_idx].raw, "MAT_LAW110_7_4")
                    angles_data.append([_f(x) for x in f_ang])
                else:
                    f_ang = cut(valid_cards[c_idx].raw, "MAT_LAW110_7_1")
                    angles_data.append([_f(x) for x in f_ang])
    else:
        def _toks(c):
            return [p.strip() for p in c.raw.replace(",", " ").split() if p.strip()]

        if len(valid_cards) > 0:
            toks1 = _toks(valid_cards[0])
            rho0 = float(toks1[0]) if len(toks1) > 0 else 0.0
            rhor = float(toks1[1]) if len(toks1) > 1 else rho0
        if len(valid_cards) > 1:
            toks2 = _toks(valid_cards[1])
            young = float(toks2[0]) if len(toks2) > 0 else 0.0
            nu = float(toks2[1]) if len(toks2) > 1 else 0.0
            ires = int(toks2[2]) if len(toks2) > 2 else 0
        if len(valid_cards) > 2:
            toks3 = _toks(valid_cards[2])
            icrit = int(toks3[0]) if len(toks3) > 0 else 1
            tab_yld = int(toks3[1]) if len(toks3) > 1 else 0
            xscale = float(toks3[2]) if len(toks3) > 2 else 1.0
            yscale = float(toks3[3]) if len(toks3) > 3 else 1.0
            fbi = float(toks3[4]) if len(toks3) > 4 else 0.0
            rhobi = float(toks3[5]) if len(toks3) > 5 else 0.0
        if len(valid_cards) > 3:
            toks4 = _toks(valid_cards[3])
            sigma_r = float(toks4[0]) if len(toks4) > 0 else 0.0
            dsigm = float(toks4[1]) if len(toks4) > 1 else 0.0
            beta = float(toks4[2]) if len(toks4) > 2 else 0.0
            omega = float(toks4[3]) if len(toks4) > 3 else 0.0
            hard_n = float(toks4[4]) if len(toks4) > 4 else 0.0
        if len(valid_cards) > 4:
            toks5 = _toks(valid_cards[4])
            eps0 = float(toks5[0]) if len(toks5) > 0 else 0.0
            sigs = float(toks5[1]) if len(toks5) > 1 else 0.0
            dg0 = float(toks5[2]) if len(toks5) > 2 else 0.0
            deps0 = float(toks5[3]) if len(toks5) > 3 else 0.0
            m = float(toks5[4]) if len(toks5) > 4 else 0.0
        if len(valid_cards) > 5:
            toks6 = _toks(valid_cards[5])
            tini = float(toks6[0]) if len(toks6) > 0 else 0.0
            chard = float(toks6[1]) if len(toks6) > 1 else 0.0
            fcut = float(toks6[2]) if len(toks6) > 2 else 0.0
            vp = int(toks6[3]) if len(toks6) > 3 else 0
            ismooth = int(toks6[4]) if len(toks6) > 4 else 0
            tab_temp = int(toks6[5]) if len(toks6) > 5 else 0
        if icrit == 3:
            if len(valid_cards) > 6:
                toks7 = _toks(valid_cards[6])
                rm_0 = float(toks7[0]) if len(toks7) > 0 else 0.0
                rm_45 = float(toks7[1]) if len(toks7) > 1 else 0.0
                rm_90 = float(toks7[2]) if len(toks7) > 2 else 0.0
                ag_0 = float(toks7[3]) if len(toks7) > 3 else 0.0
                ag_45 = float(toks7[4]) if len(toks7) > 4 else 0.0
            if len(valid_cards) > 7:
                toks8 = _toks(valid_cards[7])
                ag_90 = float(toks8[0]) if len(toks8) > 0 else 0.0
                r_0 = float(toks8[1]) if len(toks8) > 1 else 1.0
                r_45 = float(toks8[2]) if len(toks8) > 2 else 1.0
                r_90 = float(toks8[3]) if len(toks8) > 3 else 1.0
        else:
            for c_idx in range(6, len(valid_cards)):
                toks_ang = _toks(valid_cards[c_idx])
                angles_data.append([float(x) for x in toks_ang])

    m110 = MaterialLaw110(
        id=mat_id, title=title, rho0=rho0, rhor=rhor,
        young=young, nu=nu, ires=ires,
        icrit=icrit, tab_yld=tab_yld, xscale=xscale, yscale=yscale,
        fbi=fbi, rhobi=rhobi,
        sigma_r=sigma_r, dsigm=dsigm, beta=beta, omega=omega, hard_n=hard_n,
        eps0=eps0, sigs=sigs, dg0=dg0, deps0=deps0, m=m,
        tini=tini, chard=chard, fcut=fcut, vp=vp, ismooth=ismooth, tab_temp=tab_temp,
        rm_0=rm_0, rm_45=rm_45, rm_90=rm_90,
        ag_0=ag_0, ag_45=ag_45, ag_90=ag_90,
        r_0=r_0, r_45=r_45, r_90=r_90,
        angles_data=angles_data,
        unit_system=block.unit_id,
    )
    model.mat_law110s[mat_id] = m110
    from ..mat_reader import GenericMaterialRecord
    mat110 = Material(
        id=mat_id, law=110, rho0=rho0, title=title,
        params={
            "E": young if young > 0.0 else 1.0, "nu": nu if 0.0 <= nu < 0.5 else 0.3,
            "MAT_RHO": rho0, "Refer_Rho": rhor, "MAT_E": young, "MAT_NU": nu,
            "MAT_Ires": ires, "MAT_Icrit": icrit, "MAT_TAB_YLD": tab_yld,
            "MAT_Xscale": xscale, "MAT_Yscale": yscale, "MAT_fBI": fbi, "MAT_rhoBI": rhobi,
            "SIGMA_r": sigma_r, "MAT_DSIGM": dsigm, "MAT_BETA": beta, "Omega": omega, "MAT_HARD": hard_n,
            "Epsilon_0": eps0, "MAT_SIGS": sigs, "MAT_DG0": dg0, "MAT_Deps0": deps0, "MAT_StrainRate_m": m,
            "T_Initial": tini, "MAT_CHARD": chard, "Fcut": fcut, "Vflag": vp,
            "MAT_Ismooth": ismooth, "MAT_TAB_TEMP": tab_temp,
            "MAT_RM_0": rm_0, "MAT_RM_45": rm_45, "MAT_RM_90": rm_90,
            "MAT_AG_0": ag_0, "MAT_AG_45": ag_45, "MAT_AG_90": ag_90,
            "MAT_R_0": r_0, "MAT_R_45": r_45, "MAT_R_90": r_90,
        }
    )
    mat110.law_name = "VEGTER"
    mat110.m110 = m110
    mat110.angles_data = angles_data
    mat110.record = GenericMaterialRecord(
        law_name="LAW110", law_number=110, id=mat_id, title=title,
        params=mat110.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat110




read_mat_vegter = read_mat_law110


read_mat_plas_vegter = read_mat_law110


read_mat_law110_vegter = read_mat_law110




def read_mat_law115(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW115/id`` or ``/MAT/DESHPANDE_FLECK/id`` (M176): Deshpande-Fleck foam model."""
    from ...model.entities import MaterialLaw115, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho0 = 0.0
    young, nu, ires, istat = 0.0, 0.0, 2, 0
    alpha, cfail, pfail = 0.0, 0.0, 0.0
    sigp, gamma, epsd, alpha2, beta = 0.0, 0.0, 0.0, 0.0, 0.0
    rhof0 = 0.0
    sigp_c0, sigp_c1, sigp_n = 0.0, 0.0, 0.0
    alpha2_c0, alpha2_c1, alpha2_n = 0.0, 0.0, 0.0
    gamma_c0, gamma_c1, gamma_n = 0.0, 0.0, 0.0
    beta_c0, beta_c1, beta_n = 0.0, 0.0, 0.0

    valid_cards = [c for c in cards if not c.is_blank]

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW115_1")
            rho0 = _f(f1[0])
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW115_2")
            young = _f(f2[0])
            nu = _f(f2[1]) if len(f2) > 1 else 0.0
            ires = _i(f2[2]) if len(f2) > 2 and f2[2].strip() else 2
            istat = _i(f2[3]) if len(f2) > 3 and f2[3].strip() else 0
        if istat == 0:
            if len(valid_cards) > 2:
                f3 = cut(valid_cards[2].raw, "MAT_LAW115_3_0")
                alpha = _f(f3[0]) if len(f3) > 0 else 0.0
                cfail = _f(f3[1]) if len(f3) > 1 else 0.0
                pfail = _f(f3[2]) if len(f3) > 2 else 0.0
            if len(valid_cards) > 3:
                f4 = cut(valid_cards[3].raw, "MAT_LAW115_4_0")
                sigp = _f(f4[0]) if len(f4) > 0 else 0.0
                gamma = _f(f4[1]) if len(f4) > 1 else 0.0
                epsd = _f(f4[2]) if len(f4) > 2 else 0.0
                alpha2 = _f(f4[3]) if len(f4) > 3 else 0.0
                beta = _f(f4[4]) if len(f4) > 4 else 0.0
        else:
            if len(valid_cards) > 2:
                f3 = cut(valid_cards[2].raw, "MAT_LAW115_3_1")
                alpha = _f(f3[0]) if len(f3) > 0 else 0.0
                cfail = _f(f3[1]) if len(f3) > 1 else 0.0
                pfail = _f(f3[2]) if len(f3) > 2 else 0.0
                rhof0 = _f(f3[3]) if len(f3) > 3 else 0.0
            if len(valid_cards) > 3:
                f4 = cut(valid_cards[3].raw, "MAT_LAW115_4_1")
                sigp_c0 = _f(f4[0]) if len(f4) > 0 else 0.0
                sigp_c1 = _f(f4[1]) if len(f4) > 1 else 0.0
                sigp_n = _f(f4[2]) if len(f4) > 2 else 0.0
            if len(valid_cards) > 4:
                f5 = cut(valid_cards[4].raw, "MAT_LAW115_5_1")
                alpha2_c0 = _f(f5[0]) if len(f5) > 0 else 0.0
                alpha2_c1 = _f(f5[1]) if len(f5) > 1 else 0.0
                alpha2_n = _f(f5[2]) if len(f5) > 2 else 0.0
            if len(valid_cards) > 5:
                f6 = cut(valid_cards[5].raw, "MAT_LAW115_6_1")
                gamma_c0 = _f(f6[0]) if len(f6) > 0 else 0.0
                gamma_c1 = _f(f6[1]) if len(f6) > 1 else 0.0
                gamma_n = _f(f6[2]) if len(f6) > 2 else 0.0
            if len(valid_cards) > 6:
                f7 = cut(valid_cards[6].raw, "MAT_LAW115_7_1")
                beta_c0 = _f(f7[0]) if len(f7) > 0 else 0.0
                beta_c1 = _f(f7[1]) if len(f7) > 1 else 0.0
                beta_n = _f(f7[2]) if len(f7) > 2 else 0.0
    else:
        if len(valid_cards) > 0:
            toks1 = valid_cards[0].tokens()
            rho0 = float(toks1[0]) if len(toks1) > 0 else 0.0
        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            young = float(toks2[0]) if len(toks2) > 0 else 0.0
            nu = float(toks2[1]) if len(toks2) > 1 else 0.0
            ires = int(toks2[2]) if len(toks2) > 2 else 2
            istat = int(toks2[3]) if len(toks2) > 3 else 0
        if istat == 0:
            if len(valid_cards) > 2:
                toks3 = valid_cards[2].tokens()
                alpha = float(toks3[0]) if len(toks3) > 0 else 0.0
                cfail = float(toks3[1]) if len(toks3) > 1 else 0.0
                pfail = float(toks3[2]) if len(toks3) > 2 else 0.0
            if len(valid_cards) > 3:
                toks4 = valid_cards[3].tokens()
                sigp = float(toks4[0]) if len(toks4) > 0 else 0.0
                gamma = float(toks4[1]) if len(toks4) > 1 else 0.0
                epsd = float(toks4[2]) if len(toks4) > 2 else 0.0
                alpha2 = float(toks4[3]) if len(toks4) > 3 else 0.0
                beta = float(toks4[4]) if len(toks4) > 4 else 0.0
        else:
            if len(valid_cards) > 2:
                toks3 = valid_cards[2].tokens()
                alpha = float(toks3[0]) if len(toks3) > 0 else 0.0
                cfail = float(toks3[1]) if len(toks3) > 1 else 0.0
                pfail = float(toks3[2]) if len(toks3) > 2 else 0.0
                rhof0 = float(toks3[3]) if len(toks3) > 3 else 0.0
            if len(valid_cards) > 3:
                toks4 = valid_cards[3].tokens()
                sigp_c0 = float(toks4[0]) if len(toks4) > 0 else 0.0
                sigp_c1 = float(toks4[1]) if len(toks4) > 1 else 0.0
                sigp_n = float(toks4[2]) if len(toks4) > 2 else 0.0
            if len(valid_cards) > 4:
                toks5 = valid_cards[4].tokens()
                alpha2_c0 = float(toks5[0]) if len(toks5) > 0 else 0.0
                alpha2_c1 = float(toks5[1]) if len(toks5) > 1 else 0.0
                alpha2_n = float(toks5[2]) if len(toks5) > 2 else 0.0
            if len(valid_cards) > 5:
                toks6 = valid_cards[5].tokens()
                gamma_c0 = float(toks6[0]) if len(toks6) > 0 else 0.0
                gamma_c1 = float(toks6[1]) if len(toks6) > 1 else 0.0
                gamma_n = float(toks6[2]) if len(toks6) > 2 else 0.0
            if len(valid_cards) > 6:
                toks7 = valid_cards[6].tokens()
                beta_c0 = float(toks7[0]) if len(toks7) > 0 else 0.0
                beta_c1 = float(toks7[1]) if len(toks7) > 1 else 0.0
                beta_n = float(toks7[2]) if len(toks7) > 2 else 0.0

    m115 = MaterialLaw115(
        id=mat_id, title=title, rho0=rho0,
        young=young, nu=nu, ires=ires, istat=istat,
        alpha=alpha, cfail=cfail, pfail=pfail,
        sigp=sigp, gamma=gamma, epsd=epsd, alpha2=alpha2, beta=beta,
        rhof0=rhof0,
        sigp_c0=sigp_c0, sigp_c1=sigp_c1, sigp_n=sigp_n,
        alpha2_c0=alpha2_c0, alpha2_c1=alpha2_c1, alpha2_n=alpha2_n,
        gamma_c0=gamma_c0, gamma_c1=gamma_c1, gamma_n=gamma_n,
        beta_c0=beta_c0, beta_c1=beta_c1, beta_n=beta_n,
    )
    model.mat_law115s[mat_id] = m115
    from ..mat_reader import GenericMaterialRecord
    mat115 = Material(
        id=mat_id, law=115, rho0=rho0, title=title,
        params={
            "E": young if young > 0.0 else 1.0, "nu": nu if 0.0 <= nu < 0.5 else 0.3,
            "MAT_RHO": rho0, "MAT_E": young, "MAT_NU": nu,
            "MAT_IRES": ires, "MAT_ISTAT": istat,
            "MAT_ALPHA": alpha, "MAT_CFAIL": cfail, "MAT_PFAIL": pfail,
            "MAT_SIGP": sigp, "MAT_GAMMA": gamma, "MAT_EPSD": epsd,
            "MAT_ALPHA2": alpha2, "MAT_BETA": beta, "MAT_RHOF0": rhof0,
            "MAT_SIGP_C0": sigp_c0, "MAT_SIGP_C1": sigp_c1, "MAT_SIGP_N": sigp_n,
            "MAT_ALPHA2_C0": alpha2_c0, "MAT_ALPHA2_C1": alpha2_c1, "MAT_ALPHA2_N": alpha2_n,
            "MAT_GAMMA_C0": gamma_c0, "MAT_GAMMA_C1": gamma_c1, "MAT_GAMMA_N": gamma_n,
            "MAT_BETA_C0": beta_c0, "MAT_BETA_C1": beta_c1, "MAT_BETA_N": beta_n,
        }
    )
    mat115.record = GenericMaterialRecord(
        law_name="LAW115", law_number=115, id=mat_id, title=title,
        params=mat115.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat115




def read_mat_law109(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW109/id`` (M177): Thermo-viscoplastic with tabular yield, temperature, and Taylor-Quinney conversion."""
    from ...model.entities import MaterialLaw109
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/MAT/LAW109/{mat_id}: missing data cards", block.source)
        return

    rho0, refer_rho, young, nu = 0.0, 0.0, 0.0, 0.0
    cp, eta, tref, tini = 0.0, 1.0, 293.0, 293.0
    tab_yld, tab_temp, xscale_h, yscale_h, ismooth = 0, 0, 1.0, 1.0, 1
    tab_eta, xscale_eta = 0, 1.0

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW109_1")
            rho0 = _f(f1[0]) if len(f1) > 0 else 0.0
            refer_rho = _f(f1[1]) if len(f1) > 1 and f1[1] else 0.0
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW109_2")
            young = _f(f2[0]) if len(f2) > 0 else 0.0
            nu = _f(f2[1]) if len(f2) > 1 else 0.0
        if len(valid_cards) > 2:
            f3 = cut(valid_cards[2].raw, "MAT_LAW109_3")
            cp = _f(f3[0]) if len(f3) > 0 else 0.0
            eta = _f(f3[1]) if len(f3) > 1 else 1.0
            tref = _f(f3[2]) if len(f3) > 2 else 293.0
            tini = _f(f3[3]) if len(f3) > 3 else 293.0
        if len(valid_cards) > 3:
            f4 = cut(valid_cards[3].raw, "MAT_LAW109_4")
            tab_yld = _i(f4[0]) if len(f4) > 0 else 0
            tab_temp = _i(f4[1]) if len(f4) > 1 else 0
            xscale_h = _f(f4[2]) if len(f4) > 2 else 1.0
            yscale_h = _f(f4[3]) if len(f4) > 3 else 1.0
            ismooth = _i(f4[5]) if len(f4) > 5 and f4[5] else 1
        if len(valid_cards) > 4:
            f5 = cut(valid_cards[4].raw, "MAT_LAW109_5")
            tab_eta = _i(f5[0]) if len(f5) > 0 else 0
            xscale_eta = _f(f5[1]) if len(f5) > 1 else 1.0
    else:
        def _get_toks(c):
            return [t.strip().rstrip(",").strip() for t in c.raw.replace(",", " ").split()]

        if len(valid_cards) > 0:
            toks1 = _get_toks(valid_cards[0])
            rho0 = float(toks1[0]) if len(toks1) > 0 else 0.0
            refer_rho = float(toks1[1]) if len(toks1) > 1 else 0.0
        if len(valid_cards) > 1:
            toks2 = _get_toks(valid_cards[1])
            young = float(toks2[0]) if len(toks2) > 0 else 0.0
            nu = float(toks2[1]) if len(toks2) > 1 else 0.0
        if len(valid_cards) > 2:
            toks3 = _get_toks(valid_cards[2])
            cp = float(toks3[0]) if len(toks3) > 0 else 0.0
            eta = float(toks3[1]) if len(toks3) > 1 else 1.0
            tref = float(toks3[2]) if len(toks3) > 2 else 293.0
            tini = float(toks3[3]) if len(toks3) > 3 else 293.0
        if len(valid_cards) > 3:
            toks4 = _get_toks(valid_cards[3])
            tab_yld = int(toks4[0]) if len(toks4) > 0 else 0
            tab_temp = int(toks4[1]) if len(toks4) > 1 else 0
            xscale_h = float(toks4[2]) if len(toks4) > 2 else 1.0
            yscale_h = float(toks4[3]) if len(toks4) > 3 else 1.0
            if len(toks4) >= 6:
                try:
                    ismooth = int(toks4[5])
                except ValueError:
                    ismooth = int(toks4[4])
            elif len(toks4) >= 5:
                ismooth = int(toks4[4])
            else:
                ismooth = 1
        if len(valid_cards) > 4:
            toks5 = _get_toks(valid_cards[4])
            tab_eta = int(toks5[0]) if len(toks5) > 0 else 0
            xscale_eta = float(toks5[1]) if len(toks5) > 1 else 1.0

    m109 = MaterialLaw109(
        id=mat_id, title=title, rho0=rho0, refer_rho=refer_rho,
        young=young, nu=nu, cp=cp, eta=eta, tref=tref, tini=tini,
        tab_yld=tab_yld, tab_temp=tab_temp, xscale_h=xscale_h, yscale_h=yscale_h, ismooth=ismooth,
        tab_eta=tab_eta, xscale_eta=xscale_eta, unit_system=block.unit_id,
    )
    model.mat_law109s[mat_id] = m109
    from ..mat_reader import GenericMaterialRecord
    mat109 = Material(
        id=mat_id, law=109, rho0=rho0, title=title,
        params={
            "E": young if young > 0.0 else 1.0, "nu": nu if 0.0 <= nu < 0.5 else 0.3,
            "MAT_RHO": rho0, "Refer_Rho": refer_rho, "MAT_E": young, "MAT_NU": nu,
            "MAT_SPHEAT": cp, "MAT_ETA": eta, "WPREF": tref, "T_Initial": tini,
            "MAT_TAB_YLD": tab_yld, "MAT_TAB_TEMP": tab_temp,
            "MAT_Xscale": xscale_h, "MAT_Yscale": yscale_h, "MAT_Ismooth": ismooth,
            "TAB_ETA": tab_eta, "MAT_Xrate": xscale_eta,
        }
    )
    mat109.law_name = "TAB_PLAS"
    mat109.m109 = m109
    mat109.record = GenericMaterialRecord(
        law_name="LAW109", law_number=109, id=mat_id, title=title,
        params=mat109.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat109




def read_mat_law111(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW111/id`` or ``/MAT/MARLOW/id`` (M177): Marlow hyperelastic material model."""
    from ...model.entities import MaterialLaw111
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/MAT/LAW111/{mat_id}: missing data cards", block.source)
        return

    rho0, itype, fct_id, fscale, nu = 0.0, 1, 0, 1.0, 0.495

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW111_1")
            rho0 = _f(f1[0]) if len(f1) > 0 else 0.0
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW111_2")
            itype = _i(f2[0]) if len(f2) > 0 else 1
            fct_id = _i(f2[1]) if len(f2) > 1 else 0
            fscale = _f(f2[2]) if len(f2) > 2 else 1.0
            nu = _f(f2[3]) if len(f2) > 3 else 0.495
    else:
        if len(valid_cards) > 0:
            toks1 = valid_cards[0].tokens()
            rho0 = float(toks1[0]) if len(toks1) > 0 else 0.0
        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            itype = int(toks2[0]) if len(toks2) > 0 else 1
            fct_id = int(toks2[1]) if len(toks2) > 1 else 0
            fscale = float(toks2[2]) if len(toks2) > 2 else 1.0
            nu = float(toks2[3]) if len(toks2) > 3 else 0.495

    m111 = MaterialLaw111(
        id=mat_id, title=title, rho0=rho0,
        itype=itype, fct_id=fct_id, fscale=fscale, nu=nu,
    )
    model.mat_law111s[mat_id] = m111
    from ..mat_reader import GenericMaterialRecord
    mat111 = Material(
        id=mat_id, law=111, rho0=rho0, title=title,
        params={
            "E": 1.0, "nu": nu if 0.0 <= nu < 0.5 else 0.495,
            "MAT_RHO": rho0, "Itype": itype, "FUN_A1": fct_id, "MAT_FScale": fscale, "MAT_NU": nu,
        }
    )
    mat111.record = GenericMaterialRecord(
        law_name="LAW111", law_number=111, id=mat_id, title=title,
        params=mat111.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat111




def read_mat_law112(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW112/id`` or ``/MAT/PAPER/id`` / ``/MAT/PLAS_PAPER/id`` / ``/MAT/XIA/id`` (M177): Comprehensive 3D orthotropic paper plasticity."""
    from ...model.entities import MaterialLaw112
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/MAT/LAW112/{mat_id}: missing data cards", block.source)
        return

    rho0, rhor = 0.0, 0.0
    e1, e2, e3, ires, itab, ismooth = 0.0, 0.0, 0.0, 0, 0, 0
    nu21, g12, g23, g13 = 0.0, 0.0, 0.0, 0.0
    k, e3c, cc = 0.0, 0.0, 0.0
    nu1p, nu2p, nu4p, nu5p = 0.0, 0.0, 0.0, 0.0
    s01, a01, b01, c01 = 0.0, 0.0, 0.0, 0.0
    s02, a02, b02, c02 = 0.0, 0.0, 0.0, 0.0
    s03, a03, b03, c03 = 0.0, 0.0, 0.0, 0.0
    s04, a04, b04, c04 = 0.0, 0.0, 0.0, 0.0
    s05, a05, b05, c05 = 0.0, 0.0, 0.0, 0.0
    asig, bsig, csig = 0.0, 0.0, 0.0
    tau0, atau, btau = 0.0, 0.0, 0.0
    tab_yld1, xscale1, yscale1 = 0, 1.0, 1.0
    tab_yld2, xscale2, yscale2 = 0, 1.0, 1.0
    tab_yld3, xscale3, yscale3 = 0, 1.0, 1.0
    tab_yld4, xscale4, yscale4 = 0, 1.0, 1.0
    tab_yld5, xscale5, yscale5 = 0, 1.0, 1.0
    tab_yldc, xscalec, yscalec = 0, 1.0, 1.0
    tab_ylds, xscales, yscales = 0, 1.0, 1.0

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW112_1")
            rho0 = _f(f1[0]) if len(f1) > 0 else 0.0
            rhor = _f(f1[1]) if len(f1) > 1 else rho0
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW112_2")
            e1 = _f(f2[0]) if len(f2) > 0 else 0.0
            e2 = _f(f2[1]) if len(f2) > 1 else 0.0
            e3 = _f(f2[2]) if len(f2) > 2 else 0.0
            ires = _i(f2[3]) if len(f2) > 3 else 0
            itab = _i(f2[4]) if len(f2) > 4 else 0
            ismooth = _i(f2[5]) if len(f2) > 5 else 0
        if len(valid_cards) > 2:
            f3 = cut(valid_cards[2].raw, "MAT_LAW112_3")
            nu21 = _f(f3[0]) if len(f3) > 0 else 0.0
            g12 = _f(f3[1]) if len(f3) > 1 else 0.0
            g23 = _f(f3[2]) if len(f3) > 2 else 0.0
            g13 = _f(f3[3]) if len(f3) > 3 else 0.0
        if len(valid_cards) > 3:
            f4 = cut(valid_cards[3].raw, "MAT_LAW112_4")
            k = _f(f4[0]) if len(f4) > 0 else 0.0
            e3c = _f(f4[1]) if len(f4) > 1 else 0.0
            cc = _f(f4[2]) if len(f4) > 2 else 0.0
        if len(valid_cards) > 4:
            f5 = cut(valid_cards[4].raw, "MAT_LAW112_5")
            nu1p = _f(f5[0]) if len(f5) > 0 else 0.0
            nu2p = _f(f5[1]) if len(f5) > 1 else 0.0
            nu4p = _f(f5[2]) if len(f5) > 2 else 0.0
            nu5p = _f(f5[3]) if len(f5) > 3 else 0.0
        if itab == 0:
            if len(valid_cards) > 5:
                f6 = cut(valid_cards[5].raw, "MAT_LAW112_6_0")
                s01 = _f(f6[0]) if len(f6) > 0 else 0.0
                a01 = _f(f6[1]) if len(f6) > 1 else 0.0
                b01 = _f(f6[2]) if len(f6) > 2 else 0.0
                c01 = _f(f6[3]) if len(f6) > 3 else 0.0
            if len(valid_cards) > 6:
                f7 = cut(valid_cards[6].raw, "MAT_LAW112_7_0")
                s02 = _f(f7[0]) if len(f7) > 0 else 0.0
                a02 = _f(f7[1]) if len(f7) > 1 else 0.0
                b02 = _f(f7[2]) if len(f7) > 2 else 0.0
                c02 = _f(f7[3]) if len(f7) > 3 else 0.0
            if len(valid_cards) > 7:
                f8 = cut(valid_cards[7].raw, "MAT_LAW112_8_0")
                s03 = _f(f8[0]) if len(f8) > 0 else 0.0
                a03 = _f(f8[1]) if len(f8) > 1 else 0.0
                b03 = _f(f8[2]) if len(f8) > 2 else 0.0
                c03 = _f(f8[3]) if len(f8) > 3 else 0.0
            if len(valid_cards) > 8:
                f9 = cut(valid_cards[8].raw, "MAT_LAW112_9_0")
                s04 = _f(f9[0]) if len(f9) > 0 else 0.0
                a04 = _f(f9[1]) if len(f9) > 1 else 0.0
                b04 = _f(f9[2]) if len(f9) > 2 else 0.0
                c04 = _f(f9[3]) if len(f9) > 3 else 0.0
            if len(valid_cards) > 9:
                f10 = cut(valid_cards[9].raw, "MAT_LAW112_10_0")
                s05 = _f(f10[0]) if len(f10) > 0 else 0.0
                a05 = _f(f10[1]) if len(f10) > 1 else 0.0
                b05 = _f(f10[2]) if len(f10) > 2 else 0.0
                c05 = _f(f10[3]) if len(f10) > 3 else 0.0
            if len(valid_cards) > 10:
                f11 = cut(valid_cards[10].raw, "MAT_LAW112_11_0")
                asig = _f(f11[0]) if len(f11) > 0 else 0.0
                bsig = _f(f11[1]) if len(f11) > 1 else 0.0
                csig = _f(f11[2]) if len(f11) > 2 else 0.0
            if len(valid_cards) > 11:
                f12 = cut(valid_cards[11].raw, "MAT_LAW112_12_0")
                tau0 = _f(f12[0]) if len(f12) > 0 else 0.0
                atau = _f(f12[1]) if len(f12) > 1 else 0.0
                btau = _f(f12[2]) if len(f12) > 2 else 0.0
        else:
            if len(valid_cards) > 5:
                f6 = cut(valid_cards[5].raw, "MAT_LAW112_TAB")
                tab_yld1 = _i(f6[1]) if len(f6) > 1 else 0
                xscale1 = _f(f6[2]) if len(f6) > 2 else 1.0
                yscale1 = _f(f6[3]) if len(f6) > 3 else 1.0
            if len(valid_cards) > 6:
                f7 = cut(valid_cards[6].raw, "MAT_LAW112_TAB")
                tab_yld2 = _i(f7[1]) if len(f7) > 1 else 0
                xscale2 = _f(f7[2]) if len(f7) > 2 else 1.0
                yscale2 = _f(f7[3]) if len(f7) > 3 else 1.0
            if len(valid_cards) > 7:
                f8 = cut(valid_cards[7].raw, "MAT_LAW112_TAB")
                tab_yld3 = _i(f8[1]) if len(f8) > 1 else 0
                xscale3 = _f(f8[2]) if len(f8) > 2 else 1.0
                yscale3 = _f(f8[3]) if len(f8) > 3 else 1.0
            if len(valid_cards) > 8:
                f9 = cut(valid_cards[8].raw, "MAT_LAW112_TAB")
                tab_yld4 = _i(f9[1]) if len(f9) > 1 else 0
                xscale4 = _f(f9[2]) if len(f9) > 2 else 1.0
                yscale4 = _f(f9[3]) if len(f9) > 3 else 1.0
            if len(valid_cards) > 9:
                f10 = cut(valid_cards[9].raw, "MAT_LAW112_TAB")
                tab_yld5 = _i(f10[1]) if len(f10) > 1 else 0
                xscale5 = _f(f10[2]) if len(f10) > 2 else 1.0
                yscale5 = _f(f10[3]) if len(f10) > 3 else 1.0
            if len(valid_cards) > 10:
                f11 = cut(valid_cards[10].raw, "MAT_LAW112_TAB")
                tab_yldc = _i(f11[1]) if len(f11) > 1 else 0
                xscalec = _f(f11[2]) if len(f11) > 2 else 1.0
                yscalec = _f(f11[3]) if len(f11) > 3 else 1.0
            if len(valid_cards) > 11:
                f12 = cut(valid_cards[11].raw, "MAT_LAW112_TAB")
                tab_ylds = _i(f12[1]) if len(f12) > 1 else 0
                xscales = _f(f12[2]) if len(f12) > 2 else 1.0
                yscales = _f(f12[3]) if len(f12) > 3 else 1.0
    else:
        if len(valid_cards) > 0:
            toks1 = valid_cards[0].tokens()
            rho0 = float(toks1[0]) if len(toks1) > 0 else 0.0
            rhor = float(toks1[1]) if len(toks1) > 1 else rho0
        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            e1 = float(toks2[0]) if len(toks2) > 0 else 0.0
            e2 = float(toks2[1]) if len(toks2) > 1 else 0.0
            e3 = float(toks2[2]) if len(toks2) > 2 else 0.0
            ires = int(toks2[3]) if len(toks2) > 3 else 0
            itab = int(toks2[4]) if len(toks2) > 4 else 0
            ismooth = int(toks2[5]) if len(toks2) > 5 else 0
        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            nu21 = float(toks3[0]) if len(toks3) > 0 else 0.0
            g12 = float(toks3[1]) if len(toks3) > 1 else 0.0
            g23 = float(toks3[2]) if len(toks3) > 2 else 0.0
            g13 = float(toks3[3]) if len(toks3) > 3 else 0.0
        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            k = float(toks4[0]) if len(toks4) > 0 else 0.0
            e3c = float(toks4[1]) if len(toks4) > 1 else 0.0
            cc = float(toks4[2]) if len(toks4) > 2 else 0.0
        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            nu1p = float(toks5[0]) if len(toks5) > 0 else 0.0
            nu2p = float(toks5[1]) if len(toks5) > 1 else 0.0
            nu4p = float(toks5[2]) if len(toks5) > 2 else 0.0
            nu5p = float(toks5[3]) if len(toks5) > 3 else 0.0
        if itab == 0:
            if len(valid_cards) > 5:
                toks6 = valid_cards[5].tokens()
                s01 = float(toks6[0]) if len(toks6) > 0 else 0.0
                a01 = float(toks6[1]) if len(toks6) > 1 else 0.0
                b01 = float(toks6[2]) if len(toks6) > 2 else 0.0
                c01 = float(toks6[3]) if len(toks6) > 3 else 0.0
            if len(valid_cards) > 6:
                toks7 = valid_cards[6].tokens()
                s02 = float(toks7[0]) if len(toks7) > 0 else 0.0
                a02 = float(toks7[1]) if len(toks7) > 1 else 0.0
                b02 = float(toks7[2]) if len(toks7) > 2 else 0.0
                c02 = float(toks7[3]) if len(toks7) > 3 else 0.0
            if len(valid_cards) > 7:
                toks8 = valid_cards[7].tokens()
                s03 = float(toks8[0]) if len(toks8) > 0 else 0.0
                a03 = float(toks8[1]) if len(toks8) > 1 else 0.0
                b03 = float(toks8[2]) if len(toks8) > 2 else 0.0
                c03 = float(toks8[3]) if len(toks8) > 3 else 0.0
            if len(valid_cards) > 8:
                toks9 = valid_cards[8].tokens()
                s04 = float(toks9[0]) if len(toks9) > 0 else 0.0
                a04 = float(toks9[1]) if len(toks9) > 1 else 0.0
                b04 = float(toks9[2]) if len(toks9) > 2 else 0.0
                c04 = float(toks9[3]) if len(toks9) > 3 else 0.0
            if len(valid_cards) > 9:
                toks10 = valid_cards[9].tokens()
                s05 = float(toks10[0]) if len(toks10) > 0 else 0.0
                a05 = float(toks10[1]) if len(toks10) > 1 else 0.0
                b05 = float(toks10[2]) if len(toks10) > 2 else 0.0
                c05 = float(toks10[3]) if len(toks10) > 3 else 0.0
            if len(valid_cards) > 10:
                toks11 = valid_cards[10].tokens()
                asig = float(toks11[0]) if len(toks11) > 0 else 0.0
                bsig = float(toks11[1]) if len(toks11) > 1 else 0.0
                csig = float(toks11[2]) if len(toks11) > 2 else 0.0
            if len(valid_cards) > 11:
                toks12 = valid_cards[11].tokens()
                tau0 = float(toks12[0]) if len(toks12) > 0 else 0.0
                atau = float(toks12[1]) if len(toks12) > 1 else 0.0
                btau = float(toks12[2]) if len(toks12) > 2 else 0.0
        else:
            if len(valid_cards) > 5:
                toks6 = valid_cards[5].tokens()
                tab_yld1 = int(toks6[0]) if len(toks6) > 0 else 0
                xscale1 = float(toks6[1]) if len(toks6) > 1 else 1.0
                yscale1 = float(toks6[2]) if len(toks6) > 2 else 1.0
            if len(valid_cards) > 6:
                toks7 = valid_cards[6].tokens()
                tab_yld2 = int(toks7[0]) if len(toks7) > 0 else 0
                xscale2 = float(toks7[1]) if len(toks7) > 1 else 1.0
                yscale2 = float(toks7[2]) if len(toks7) > 2 else 1.0
            if len(valid_cards) > 7:
                toks8 = valid_cards[7].tokens()
                tab_yld3 = int(toks8[0]) if len(toks8) > 0 else 0
                xscale3 = float(toks8[1]) if len(toks8) > 1 else 1.0
                yscale3 = float(toks8[2]) if len(toks8) > 2 else 1.0
            if len(valid_cards) > 8:
                toks9 = valid_cards[8].tokens()
                tab_yld4 = int(toks9[0]) if len(toks9) > 0 else 0
                xscale4 = float(toks9[1]) if len(toks9) > 1 else 1.0
                yscale4 = float(toks9[2]) if len(toks9) > 2 else 1.0
            if len(valid_cards) > 9:
                toks10 = valid_cards[9].tokens()
                tab_yld5 = int(toks10[0]) if len(toks10) > 0 else 0
                xscale5 = float(toks10[1]) if len(toks10) > 1 else 1.0
                yscale5 = float(toks10[2]) if len(toks10) > 2 else 1.0
            if len(valid_cards) > 10:
                toks11 = valid_cards[10].tokens()
                tab_yldc = int(toks11[0]) if len(toks11) > 0 else 0
                xscalec = float(toks11[1]) if len(toks11) > 1 else 1.0
                yscalec = float(toks11[2]) if len(toks11) > 2 else 1.0
            if len(valid_cards) > 11:
                toks12 = valid_cards[11].tokens()
                tab_ylds = int(toks12[0]) if len(toks12) > 0 else 0
                xscales = float(toks12[1]) if len(toks12) > 1 else 1.0
                yscales = float(toks12[2]) if len(toks12) > 2 else 1.0

    m112 = MaterialLaw112(
        id=mat_id, title=title, rho0=rho0, rhor=rhor,
        e1=e1, e2=e2, e3=e3, ires=ires, itab=itab, ismooth=ismooth,
        nu21=nu21, g12=g12, g23=g23, g13=g13,
        k=k, e3c=e3c, cc=cc,
        nu1p=nu1p, nu2p=nu2p, nu4p=nu4p, nu5p=nu5p,
        s01=s01, a01=a01, b01=b01, c01=c01,
        s02=s02, a02=a02, b02=b02, c02=c02,
        s03=s03, a03=a03, b03=b03, c03=c03,
        s04=s04, a04=a04, b04=b04, c04=c04,
        s05=s05, a05=a05, b05=b05, c05=c05,
        asig=asig, bsig=bsig, csig=csig,
        tau0=tau0, atau=atau, btau=btau,
        tab_yld1=tab_yld1, xscale1=xscale1, yscale1=yscale1,
        tab_yld2=tab_yld2, xscale2=xscale2, yscale2=yscale2,
        tab_yld3=tab_yld3, xscale3=xscale3, yscale3=yscale3,
        tab_yld4=tab_yld4, xscale4=xscale4, yscale4=yscale4,
        tab_yld5=tab_yld5, xscale5=xscale5, yscale5=yscale5,
        tab_yldc=tab_yldc, xscalec=xscalec, yscalec=yscalec,
        tab_ylds=tab_ylds, xscales=xscales, yscales=yscales,
    )
    model.mat_law112s[mat_id] = m112
    from ..mat_reader import GenericMaterialRecord
    mat112 = Material(
        id=mat_id, law=112, rho0=rho0, title=title,
        params={
            "E": e1 if e1 > 0.0 else 1.0, "nu": nu21 if 0.0 <= nu21 < 0.5 else 0.3,
            "MAT_RHO": rho0, "Refer_Rho": rhor,
            "MAT_E1": e1, "MAT_E2": e2, "MAT_E3": e3, "MAT_IRES": ires, "MAT_ITAB": itab, "MAT_SMOOTH": ismooth,
            "MAT_NU21": nu21, "MAT_G12": g12, "MAT_G23": g23, "MAT_G13": g13,
            "MAT_K": k, "MAT_E3C": e3c, "MAT_CC": cc,
            "MAT_NU1P": nu1p, "MAT_NU2P": nu2p, "MAT_NU4P": nu4p, "MAT_NU5P": nu5p,
            "MAT_S01": s01, "MAT_A01": a01, "MAT_B01": b01, "MAT_C01": c01,
            "MAT_S02": s02, "MAT_A02": a02, "MAT_B02": b02, "MAT_C02": c02,
            "MAT_S03": s03, "MAT_A03": a03, "MAT_B03": b03, "MAT_C03": c03,
            "MAT_S04": s04, "MAT_A04": a04, "MAT_B04": b04, "MAT_C04": c04,
            "MAT_S05": s05, "MAT_A05": a05, "MAT_B05": b05, "MAT_C05": c05,
            "MAT_ASIG": asig, "MAT_BSIG": bsig, "MAT_CSIG": csig,
            "MAT_TAU0": tau0, "MAT_ATAU": atau, "MAT_BTAU": btau,
            "TAB_YLD1": tab_yld1, "MAT_Xscale1": xscale1, "MAT_Yscale1": yscale1,
            "TAB_YLD2": tab_yld2, "MAT_Xscale2": xscale2, "MAT_Yscale2": yscale2,
            "TAB_YLD3": tab_yld3, "MAT_Xscale3": xscale3, "MAT_Yscale3": yscale3,
            "TAB_YLD4": tab_yld4, "MAT_Xscale4": xscale4, "MAT_Yscale4": yscale4,
            "TAB_YLD5": tab_yld5, "MAT_Xscale5": xscale5, "MAT_Yscale5": yscale5,
            "TAB_YLDC": tab_yldc, "MAT_XscaleC": xscalec, "MAT_YscaleC": yscalec,
            "TAB_YLDS": tab_ylds, "MAT_XscaleS": xscales, "MAT_YscaleS": yscales,
        }
    )
    mat112.record = GenericMaterialRecord(
        law_name="LAW112", law_number=112, id=mat_id, title=title,
        params=mat112.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat112




def read_mat_law116(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW116/id`` or ``/MAT/COH_HYST/id`` / ``/MAT/COHESIVE_HYSTERETIC/id`` (M177): Cohesive zone hysteretic damage model."""
    from ...model.entities import MaterialLaw116
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/MAT/LAW116/{mat_id}: missing data cards", block.source)
        return

    rho0, young, g, thick = 0.0, 0.0, 0.0, 0.0
    imass, idel, icrit = 1, 1, 1
    gc1_ini, gc1_inf, sratg1, fg1 = 0.0, 0.0, 0.0, 0.0
    gc2_ini, gc2_inf, sratg2, fg2 = 0.0, 0.0, 0.0, 0.0
    siga1, sigb1, srate1, order1, fail1 = 0.0, 0.0, 0.0, 1, 1
    siga2, sigb2, srate2, order2, fail2 = 0.0, 0.0, 0.0, 1, 1

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW116_1")
            rho0 = _f(f1[0]) if len(f1) > 0 else 0.0
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW116_2")
            young = _f(f2[0]) if len(f2) > 0 else 0.0
            g = _f(f2[1]) if len(f2) > 1 else 0.0
            thick = _f(f2[2]) if len(f2) > 2 else 0.0
            imass = _i(f2[3]) if len(f2) > 3 else 1
            idel = _i(f2[4]) if len(f2) > 4 else 1
            icrit = _i(f2[5]) if len(f2) > 5 else 1
        if len(valid_cards) > 2:
            f3 = cut(valid_cards[2].raw, "MAT_LAW116_3")
            gc1_ini = _f(f3[0]) if len(f3) > 0 else 0.0
            gc1_inf = _f(f3[1]) if len(f3) > 1 else 0.0
            sratg1 = _f(f3[2]) if len(f3) > 2 else 0.0
            fg1 = _f(f3[3]) if len(f3) > 3 else 0.0
        if len(valid_cards) > 3:
            f4 = cut(valid_cards[3].raw, "MAT_LAW116_4")
            gc2_ini = _f(f4[0]) if len(f4) > 0 else 0.0
            gc2_inf = _f(f4[1]) if len(f4) > 1 else 0.0
            sratg2 = _f(f4[2]) if len(f4) > 2 else 0.0
            fg2 = _f(f4[3]) if len(f4) > 3 else 0.0
        if len(valid_cards) > 4:
            f5 = cut(valid_cards[4].raw, "MAT_LAW116_5")
            siga1 = _f(f5[0]) if len(f5) > 0 else 0.0
            sigb1 = _f(f5[1]) if len(f5) > 1 else 0.0
            srate1 = _f(f5[2]) if len(f5) > 2 else 0.0
            order1 = _i(f5[3]) if len(f5) > 3 else 1
            fail1 = _i(f5[4]) if len(f5) > 4 else 1
        if len(valid_cards) > 5:
            f6 = cut(valid_cards[5].raw, "MAT_LAW116_6")
            siga2 = _f(f6[0]) if len(f6) > 0 else 0.0
            sigb2 = _f(f6[1]) if len(f6) > 1 else 0.0
            srate2 = _f(f6[2]) if len(f6) > 2 else 0.0
            order2 = _i(f6[3]) if len(f6) > 3 else 1
            fail2 = _i(f6[4]) if len(f6) > 4 else 1
    else:
        if len(valid_cards) > 0:
            toks1 = valid_cards[0].tokens()
            rho0 = float(toks1[0]) if len(toks1) > 0 else 0.0
        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            young = float(toks2[0]) if len(toks2) > 0 else 0.0
            g = float(toks2[1]) if len(toks2) > 1 else 0.0
            thick = float(toks2[2]) if len(toks2) > 2 else 0.0
            imass = int(toks2[3]) if len(toks2) > 3 else 1
            idel = int(toks2[4]) if len(toks2) > 4 else 1
            icrit = int(toks2[5]) if len(toks2) > 5 else 1
        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            gc1_ini = float(toks3[0]) if len(toks3) > 0 else 0.0
            gc1_inf = float(toks3[1]) if len(toks3) > 1 else 0.0
            sratg1 = float(toks3[2]) if len(toks3) > 2 else 0.0
            fg1 = float(toks3[3]) if len(toks3) > 3 else 0.0
        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            gc2_ini = float(toks4[0]) if len(toks4) > 0 else 0.0
            gc2_inf = float(toks4[1]) if len(toks4) > 1 else 0.0
            sratg2 = float(toks4[2]) if len(toks4) > 2 else 0.0
            fg2 = float(toks4[3]) if len(toks4) > 3 else 0.0
        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            siga1 = float(toks5[0]) if len(toks5) > 0 else 0.0
            sigb1 = float(toks5[1]) if len(toks5) > 1 else 0.0
            srate1 = float(toks5[2]) if len(toks5) > 2 else 0.0
            order1 = int(toks5[3]) if len(toks5) > 3 else 1
            fail1 = int(toks5[4]) if len(toks5) > 4 else 1
        if len(valid_cards) > 5:
            toks6 = valid_cards[5].tokens()
            siga2 = float(toks6[0]) if len(toks6) > 0 else 0.0
            sigb2 = float(toks6[1]) if len(toks6) > 1 else 0.0
            srate2 = float(toks6[2]) if len(toks6) > 2 else 0.0
            order2 = int(toks6[3]) if len(toks6) > 3 else 1
            fail2 = int(toks6[4]) if len(toks6) > 4 else 1

    m116 = MaterialLaw116(
        id=mat_id, title=title, rho0=rho0,
        young=young, g=g, thick=thick, imass=imass, idel=idel, icrit=icrit,
        gc1_ini=gc1_ini, gc1_inf=gc1_inf, sratg1=sratg1, fg1=fg1,
        gc2_ini=gc2_ini, gc2_inf=gc2_inf, sratg2=sratg2, fg2=fg2,
        siga1=siga1, sigb1=sigb1, srate1=srate1, order1=order1, fail1=fail1,
        siga2=siga2, sigb2=sigb2, srate2=srate2, order2=order2, fail2=fail2,
    )
    model.mat_law116s[mat_id] = m116
    from ..mat_reader import GenericMaterialRecord
    mat116 = Material(
        id=mat_id, law=116, rho0=rho0, title=title,
        params={
            "E": young if young > 0.0 else 1.0, "nu": 0.3,
            "MAT_RHO": rho0, "MAT_E": young, "MAT_G": g, "MAT_THICK": thick,
            "MAT_IMASS": imass, "MAT_IDEL": idel, "MAT_ICRIT": icrit,
            "MAT_GC1_ini": gc1_ini, "MAT_GC1_inf": gc1_inf, "MAT_SRATG1": sratg1, "MAT_FG1": fg1,
            "MAT_GC2_ini": gc2_ini, "MAT_GC2_inf": gc2_inf, "MAT_SRATG2": sratg2, "MAT_FG2": fg2,
            "MAT_SIGA1": siga1, "MAT_SIGB1": sigb1, "MAT_SRATE1": srate1, "MAT_ORDER1": order1, "MAT_FAIL1": fail1,
            "MAT_SIGA2": siga2, "MAT_SIGB2": sigb2, "MAT_SRATE2": srate2, "MAT_ORDER2": order2, "MAT_FAIL2": fail2,
        }
    )
    mat116.record = GenericMaterialRecord(
        law_name="LAW116", law_number=116, id=mat_id, title=title,
        params=mat116.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat116




def read_mat_law122(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW122/id`` or ``/MAT/MODIFIED_LADEVEZE/id`` / ``/MAT/LADEVEZE_DELAM/id`` (M177): Modified Ladevèze Delamination & Composite Damage Model."""
    from ...model.entities import MaterialLaw122
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/MAT/LAW122/{mat_id}: missing data cards", block.source)
        return

    rho0 = 0.0
    e1, e2, e3, g12, g23 = 0.0, 0.0, 0.0, 0.0, 0.0
    g31, nu12, nu23, nu31 = 0.0, 0.0, 0.0, 0.0
    e1c, gamma, ish, itr, ires = 0.0, 0.0, 0, 0, 0
    sigy0, beta, hard_m, hard_a = 0.0, 0.0, 0.0, 0.0
    eps_fti, eps_ftu, dftu = 0.0, 0.0, 0.0
    eps_fci, eps_fcu, dfcu, ibuck = 0.0, 0.0, 0.0, 0
    ifuncd1, dsat1, y0, yc, b = 0, 0.0, 0.0, 0.0, 0.0
    dmax, yr, ysp = 0.0, 0.0, 0.0
    ifuncd2, dsat2, y0p, ycp = 0, 0.0, 0.0, 0.0
    ifuncd2c, dsat2c, y0pc, ycpc = 0, 0.0, 0.0, 0.0
    epsd11, d11, n11, d11u, n11u = 0.0, 0.0, 0.0, 0.0, 0.0
    epsd12, d22, n22, d12, n12 = 0.0, 0.0, 0.0, 0.0, 0.0
    epsdr0, dr0, nr0, ltype11, ltype12, ltyper0 = 0.0, 0.0, 0.0, 0, 0, 0
    fcut = 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW122_1")
            rho0 = _f(f1[0]) if len(f1) > 0 else 0.0
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW122_2")
            e1 = _f(f2[0]) if len(f2) > 0 else 0.0
            e2 = _f(f2[1]) if len(f2) > 1 else 0.0
            e3 = _f(f2[2]) if len(f2) > 2 else 0.0
            g12 = _f(f2[3]) if len(f2) > 3 else 0.0
            g23 = _f(f2[4]) if len(f2) > 4 else 0.0
        if len(valid_cards) > 2:
            f3 = cut(valid_cards[2].raw, "MAT_LAW122_3")
            g31 = _f(f3[0]) if len(f3) > 0 else 0.0
            nu12 = _f(f3[1]) if len(f3) > 1 else 0.0
            nu23 = _f(f3[2]) if len(f3) > 2 else 0.0
            nu31 = _f(f3[3]) if len(f3) > 3 else 0.0
        if len(valid_cards) > 3:
            f4 = cut(valid_cards[3].raw, "MAT_LAW122_4")
            e1c = _f(f4[0]) if len(f4) > 0 else 0.0
            gamma = _f(f4[1]) if len(f4) > 1 else 0.0
            ish = _i(f4[3]) if len(f4) > 3 else 0
            itr = _i(f4[5]) if len(f4) > 5 else 0
            ires = _i(f4[7]) if len(f4) > 7 else 0
        if len(valid_cards) > 4:
            f5 = cut(valid_cards[4].raw, "MAT_LAW122_5")
            sigy0 = _f(f5[0]) if len(f5) > 0 else 0.0
            beta = _f(f5[1]) if len(f5) > 1 else 0.0
            hard_m = _f(f5[2]) if len(f5) > 2 else 0.0
            hard_a = _f(f5[3]) if len(f5) > 3 else 0.0
        if len(valid_cards) > 5:
            f6 = cut(valid_cards[5].raw, "MAT_LAW122_6")
            eps_fti = _f(f6[0]) if len(f6) > 0 else 0.0
            eps_ftu = _f(f6[1]) if len(f6) > 1 else 0.0
            dftu = _f(f6[2]) if len(f6) > 2 else 0.0
        if len(valid_cards) > 6:
            f7 = cut(valid_cards[6].raw, "MAT_LAW122_7")
            eps_fci = _f(f7[0]) if len(f7) > 0 else 0.0
            eps_fcu = _f(f7[1]) if len(f7) > 1 else 0.0
            dfcu = _f(f7[2]) if len(f7) > 2 else 0.0
            ibuck = _i(f7[4]) if len(f7) > 4 else 0
        if len(valid_cards) > 7:
            f8 = cut(valid_cards[7].raw, "MAT_LAW122_8")
            ifuncd1 = _i(f8[1]) if len(f8) > 1 else 0
            dsat1 = _f(f8[2]) if len(f8) > 2 else 0.0
            y0 = _f(f8[3]) if len(f8) > 3 else 0.0
            yc = _f(f8[4]) if len(f8) > 4 else 0.0
            b = _f(f8[5]) if len(f8) > 5 else 0.0
        if len(valid_cards) > 8:
            f9 = cut(valid_cards[8].raw, "MAT_LAW122_9")
            dmax = _f(f9[0]) if len(f9) > 0 else 0.0
            yr = _f(f9[1]) if len(f9) > 1 else 0.0
            ysp = _f(f9[2]) if len(f9) > 2 else 0.0
        if len(valid_cards) > 9:
            f10 = cut(valid_cards[9].raw, "MAT_LAW122_10")
            ifuncd2 = _i(f10[1]) if len(f10) > 1 else 0
            dsat2 = _f(f10[2]) if len(f10) > 2 else 0.0
            y0p = _f(f10[3]) if len(f10) > 3 else 0.0
            ycp = _f(f10[4]) if len(f10) > 4 else 0.0
        if len(valid_cards) > 10:
            f11 = cut(valid_cards[10].raw, "MAT_LAW122_11")
            ifuncd2c = _i(f11[1]) if len(f11) > 1 else 0
            dsat2c = _f(f11[2]) if len(f11) > 2 else 0.0
            y0pc = _f(f11[3]) if len(f11) > 3 else 0.0
            ycpc = _f(f11[4]) if len(f11) > 4 else 0.0
        if len(valid_cards) > 11:
            f12 = cut(valid_cards[11].raw, "MAT_LAW122_12")
            epsd11 = _f(f12[0]) if len(f12) > 0 else 0.0
            d11 = _f(f12[1]) if len(f12) > 1 else 0.0
            n11 = _f(f12[2]) if len(f12) > 2 else 0.0
            d11u = _f(f12[3]) if len(f12) > 3 else 0.0
            n11u = _f(f12[4]) if len(f12) > 4 else 0.0
        if len(valid_cards) > 12:
            f13 = cut(valid_cards[12].raw, "MAT_LAW122_13")
            epsd12 = _f(f13[0]) if len(f13) > 0 else 0.0
            d22 = _f(f13[1]) if len(f13) > 1 else 0.0
            n22 = _f(f13[2]) if len(f13) > 2 else 0.0
            d12 = _f(f13[3]) if len(f13) > 3 else 0.0
            n12 = _f(f13[4]) if len(f13) > 4 else 0.0
        if len(valid_cards) > 13:
            f14 = cut(valid_cards[13].raw, "MAT_LAW122_14")
            epsdr0 = _f(f14[0]) if len(f14) > 0 else 0.0
            dr0 = _f(f14[1]) if len(f14) > 1 else 0.0
            nr0 = _f(f14[2]) if len(f14) > 2 else 0.0
            ltype11 = _i(f14[4]) if len(f14) > 4 else 0
            ltype12 = _i(f14[5]) if len(f14) > 5 else 0
            ltyper0 = _i(f14[6]) if len(f14) > 6 else 0
        if len(valid_cards) > 14:
            f15 = cut(valid_cards[14].raw, "MAT_LAW122_15")
            fcut = _f(f15[0]) if len(f15) > 0 else 0.0
    else:
        if len(valid_cards) > 0:
            toks1 = valid_cards[0].tokens()
            rho0 = float(toks1[0]) if len(toks1) > 0 else 0.0
        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            e1 = float(toks2[0]) if len(toks2) > 0 else 0.0
            e2 = float(toks2[1]) if len(toks2) > 1 else 0.0
            e3 = float(toks2[2]) if len(toks2) > 2 else 0.0
            g12 = float(toks2[3]) if len(toks2) > 3 else 0.0
            g23 = float(toks2[4]) if len(toks2) > 4 else 0.0
        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            g31 = float(toks3[0]) if len(toks3) > 0 else 0.0
            nu12 = float(toks3[1]) if len(toks3) > 1 else 0.0
            nu23 = float(toks3[2]) if len(toks3) > 2 else 0.0
            nu31 = float(toks3[3]) if len(toks3) > 3 else 0.0
        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            e1c = float(toks4[0]) if len(toks4) > 0 else 0.0
            gamma = float(toks4[1]) if len(toks4) > 1 else 0.0
            ish = int(toks4[2]) if len(toks4) > 2 else 0
            itr = int(toks4[3]) if len(toks4) > 3 else 0
            ires = int(toks4[4]) if len(toks4) > 4 else 0
        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            sigy0 = float(toks5[0]) if len(toks5) > 0 else 0.0
            beta = float(toks5[1]) if len(toks5) > 1 else 0.0
            hard_m = float(toks5[2]) if len(toks5) > 2 else 0.0
            hard_a = float(toks5[3]) if len(toks5) > 3 else 0.0
        if len(valid_cards) > 5:
            toks6 = valid_cards[5].tokens()
            eps_fti = float(toks6[0]) if len(toks6) > 0 else 0.0
            eps_ftu = float(toks6[1]) if len(toks6) > 1 else 0.0
            dftu = float(toks6[2]) if len(toks6) > 2 else 0.0
        if len(valid_cards) > 6:
            toks7 = valid_cards[6].tokens()
            eps_fci = float(toks7[0]) if len(toks7) > 0 else 0.0
            eps_fcu = float(toks7[1]) if len(toks7) > 1 else 0.0
            dfcu = float(toks7[2]) if len(toks7) > 2 else 0.0
            ibuck = int(toks7[3]) if len(toks7) > 3 else 0
        if len(valid_cards) > 7:
            toks8 = valid_cards[7].tokens()
            ifuncd1 = int(toks8[0]) if len(toks8) > 0 else 0
            dsat1 = float(toks8[1]) if len(toks8) > 1 else 0.0
            y0 = float(toks8[2]) if len(toks8) > 2 else 0.0
            yc = float(toks8[3]) if len(toks8) > 3 else 0.0
            b = float(toks8[4]) if len(toks8) > 4 else 0.0
        if len(valid_cards) > 8:
            toks9 = valid_cards[8].tokens()
            dmax = float(toks9[0]) if len(toks9) > 0 else 0.0
            yr = float(toks9[1]) if len(toks9) > 1 else 0.0
            ysp = float(toks9[2]) if len(toks9) > 2 else 0.0
        if len(valid_cards) > 9:
            toks10 = valid_cards[9].tokens()
            ifuncd2 = int(toks10[0]) if len(toks10) > 0 else 0
            dsat2 = float(toks10[1]) if len(toks10) > 1 else 0.0
            y0p = float(toks10[2]) if len(toks10) > 2 else 0.0
            ycp = float(toks10[3]) if len(toks10) > 3 else 0.0
        if len(valid_cards) > 10:
            toks11 = valid_cards[10].tokens()
            ifuncd2c = int(toks11[0]) if len(toks11) > 0 else 0
            dsat2c = float(toks11[1]) if len(toks11) > 1 else 0.0
            y0pc = float(toks11[2]) if len(toks11) > 2 else 0.0
            ycpc = float(toks11[3]) if len(toks11) > 3 else 0.0
        if len(valid_cards) > 11:
            toks12 = valid_cards[11].tokens()
            epsd11 = float(toks12[0]) if len(toks12) > 0 else 0.0
            d11 = float(toks12[1]) if len(toks12) > 1 else 0.0
            n11 = float(toks12[2]) if len(toks12) > 2 else 0.0
            d11u = float(toks12[3]) if len(toks12) > 3 else 0.0
            n11u = float(toks12[4]) if len(toks12) > 4 else 0.0
        if len(valid_cards) > 12:
            toks13 = valid_cards[12].tokens()
            epsd12 = float(toks13[0]) if len(toks13) > 0 else 0.0
            d22 = float(toks13[1]) if len(toks13) > 1 else 0.0
            n22 = float(toks13[2]) if len(toks13) > 2 else 0.0
            d12 = float(toks13[3]) if len(toks13) > 3 else 0.0
            n12 = float(toks13[4]) if len(toks13) > 4 else 0.0
        if len(valid_cards) > 13:
            toks14 = valid_cards[13].tokens()
            epsdr0 = float(toks14[0]) if len(toks14) > 0 else 0.0
            dr0 = float(toks14[1]) if len(toks14) > 1 else 0.0
            nr0 = float(toks14[2]) if len(toks14) > 2 else 0.0
            ltype11 = int(toks14[3]) if len(toks14) > 3 else 0
            ltype12 = int(toks14[4]) if len(toks14) > 4 else 0
            ltyper0 = int(toks14[5]) if len(toks14) > 5 else 0
        if len(valid_cards) > 14:
            toks15 = valid_cards[14].tokens()
            fcut = float(toks15[0]) if len(toks15) > 0 else 0.0

    m122 = MaterialLaw122(
        id=mat_id, title=title, rho0=rho0,
        e1=e1, e2=e2, e3=e3, g12=g12, g23=g23, g31=g31,
        nu12=nu12, nu23=nu23, nu31=nu31,
        e1c=e1c, gamma=gamma, ish=ish, itr=itr, ires=ires,
        sigy0=sigy0, beta=beta, hard_m=hard_m, hard_a=hard_a,
        eps_fti=eps_fti, eps_ftu=eps_ftu, dftu=dftu,
        eps_fci=eps_fci, eps_fcu=eps_fcu, dfcu=dfcu, ibuck=ibuck,
        ifuncd1=ifuncd1, dsat1=dsat1, y0=y0, yc=yc, b=b,
        dmax=dmax, yr=yr, ysp=yssp if 'yssp' in locals() else ysp,
        ifuncd2=ifuncd2, dsat2=dsat2, y0p=y0p, ycp=ycp,
        ifuncd2c=ifuncd2c, dsat2c=dsat2c, y0pc=y0pc, ycpc=ycpc,
        epsd11=epsd11, d11=d11, n11=n11, d11u=d11u, n11u=n11u,
        epsd12=epsd12, d22=d22, n22=n22, d12=d12, n12=n12,
        epsdr0=epsdr0, dr0=dr0, nr0=nr0,
        ltype11=ltype11, ltype12=ltype12, ltyper0=ltyper0,
        fcut=fcut,
    )
    model.mat_law122s[mat_id] = m122
    from ..mat_reader import GenericMaterialRecord
    mat122 = Material(
        id=mat_id, law=122, rho0=rho0, title=title,
        params={
            "E": e1 if e1 > 0.0 else 1.0, "nu": nu12 if 0.0 <= nu12 < 0.5 else 0.3,
            "MAT_RHO": rho0, "MAT_E1": e1, "MAT_E2": e2, "MAT_E3": e3, "MAT_G12": g12, "MAT_G23": g23, "MAT_G31": g31,
            "MAT_NU12": nu12, "MAT_NU23": nu23, "MAT_NU31": nu31,
            "MAT_E1C": e1c, "MAT_GAMMA": gamma, "ISH": ish, "ITR": itr, "IRES": ires,
            "MAT_SIGY0": sigy0, "MAT_BETA": beta, "MAT_M": hard_m, "MAT_A": hard_a,
            "MAT_EFTI": eps_fti, "MAT_EFTU": eps_ftu, "MAT_DFTU": dftu,
            "MAT_EFCI": eps_fci, "MAT_EFCU": eps_fcu, "MAT_DFCU": dfcu, "IBUCK": ibuck,
            "IFUNCD1": ifuncd1, "MAT_DSAT1": dsat1, "MAT_Y0": y0, "MAT_YC": yc, "MAT_B": b,
            "MAT_DMAX": dmax, "MAT_YR": yr, "MAT_YSP": ysp,
            "IFUNCD2": ifuncd2, "MAT_DSAT2": dsat2, "MAT_Y0P": y0p, "MAT_YCP": ycp,
            "IFUNCD2C": ifuncd2c, "MAT_DSAT2C": dsat2c, "MAT_Y0PC": y0pc, "MAT_YCPC": ycpc,
            "MAT_EPSD11": epsd11, "MAT_D11": d11, "MAT_N11": n11, "MAT_D11U": d11u, "MAT_N11U": n11u,
            "MAT_EPSD12": epsd12, "MAT_D22": d22, "MAT_N22": n22, "MAT_D12": d12, "MAT_N12": n12,
            "MAT_EPSDR0": epsdr0, "MAT_DR0": dr0, "MAT_NR0": nr0,
            "LTYPE11": ltype11, "LTYPE12": ltype12, "LTYPER0": ltyper0,
            "FCUT": fcut,
        }
    )
    mat122.record = GenericMaterialRecord(
        law_name="LAW122", law_number=122, id=mat_id, title=title,
        params=mat122.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat122




def read_mat_law158(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW158/id`` or ``/MAT/FABR_NL/id`` / ``/MAT/FABRIC_NL/id`` (M177): Nonlinear Anisotropic Fabric Material."""
    from ...model.entities import MaterialLaw158
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/MAT/LAW158/{mat_id}: missing data cards", block.source)
        return

    rho0 = 0.0
    s1, s2, flex, flex1, flex2 = 0.1, 0.1, 0.0, 0.0, 0.0
    zerostress, sensor_id = 0.0, 0
    fun_a1, c1 = 0, 1.0
    fun_a2, c2 = 0, 1.0
    fun_a3, c3 = 0, 1.0
    fun_a4, fun_a5 = 0, 0

    if block.fixed:
        if len(valid_cards) > 0:
            f1 = cut(valid_cards[0].raw, "MAT_LAW158_1")
            rho0 = _f(f1[0]) if len(f1) > 0 else 0.0
        if len(valid_cards) > 1:
            f2 = cut(valid_cards[1].raw, "MAT_LAW158_2")
            s1 = _f(f2[0]) if len(f2) > 0 else 0.1
            s2 = _f(f2[1]) if len(f2) > 1 else 0.1
            flex = _f(f2[2]) if len(f2) > 2 else 0.0
            flex1 = _f(f2[3]) if len(f2) > 3 else 0.0
            flex2 = _f(f2[4]) if len(f2) > 4 else 0.0
        if len(valid_cards) > 2:
            f3 = cut(valid_cards[2].raw, "MAT_LAW158_3")
            zerostress = _f(f3[0]) if len(f3) > 0 else 0.0
            sensor_id = _i(f3[2]) if len(f3) > 2 else 0
        if len(valid_cards) > 3:
            f4 = cut(valid_cards[3].raw, "MAT_LAW158_4")
            fun_a1 = _i(f4[0]) if len(f4) > 0 else 0
            c1 = _f(f4[2]) if len(f4) > 2 else 1.0
        if len(valid_cards) > 4:
            f5 = cut(valid_cards[4].raw, "MAT_LAW158_4")
            fun_a2 = _i(f5[0]) if len(f5) > 0 else 0
            c2 = _f(f5[2]) if len(f5) > 2 else 1.0
        if len(valid_cards) > 5:
            f6 = cut(valid_cards[5].raw, "MAT_LAW158_4")
            fun_a3 = _i(f6[0]) if len(f6) > 0 else 0
            c3 = _f(f6[2]) if len(f6) > 2 else 1.0
        if len(valid_cards) > 6:
            f7 = cut(valid_cards[6].raw, "MAT_LAW158_5")
            fun_a4 = _i(f7[0]) if len(f7) > 0 else 0
            fun_a5 = _i(f7[1]) if len(f7) > 1 else 0
    else:
        if len(valid_cards) > 0:
            toks1 = valid_cards[0].tokens()
            rho0 = float(toks1[0]) if len(toks1) > 0 else 0.0
        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            s1 = float(toks2[0]) if len(toks2) > 0 else 0.1
            s2 = float(toks2[1]) if len(toks2) > 1 else 0.1
            flex = float(toks2[2]) if len(toks2) > 2 else 0.0
            flex1 = float(toks2[3]) if len(toks2) > 3 else 0.0
            flex2 = float(toks2[4]) if len(toks2) > 4 else 0.0
        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            zerostress = float(toks3[0]) if len(toks3) > 0 else 0.0
            sensor_id = int(toks3[1]) if len(toks3) > 1 else 0
        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            fun_a1 = int(toks4[0]) if len(toks4) > 0 else 0
            c1 = float(toks4[1]) if len(toks4) > 1 else 1.0
        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            fun_a2 = int(toks5[0]) if len(toks5) > 0 else 0
            c2 = float(toks5[1]) if len(toks5) > 1 else 1.0
        if len(valid_cards) > 5:
            toks6 = valid_cards[5].tokens()
            fun_a3 = int(toks6[0]) if len(toks6) > 0 else 0
            c3 = float(toks6[1]) if len(toks6) > 1 else 1.0
        if len(valid_cards) > 6:
            toks7 = valid_cards[6].tokens()
            fun_a4 = int(toks7[0]) if len(toks7) > 0 else 0
            fun_a5 = int(toks7[1]) if len(toks7) > 1 else 0

    m158 = MaterialLaw158(
        id=mat_id, title=title, rho0=rho0,
        s1=s1, s2=s2, flex=flex, flex1=flex1, flex2=flex2,
        zerostress=zerostress, sensor_id=sensor_id,
        fun_a1=fun_a1, c1=c1, fun_a2=fun_a2, c2=c2, fun_a3=fun_a3, c3=c3,
        fun_a4=fun_a4, fun_a5=fun_a5,
    )
    model.mat_law158s[mat_id] = m158
    from ..mat_reader import GenericMaterialRecord
    mat158 = Material(
        id=mat_id, law=158, rho0=rho0, title=title,
        params={
            "E": 1000.0, "nu": 0.3,
            "MAT_RHO": rho0, "S1": s1, "S2": s2, "MAT_FLEX": flex, "MAT_FLX1": flex1, "MAT_FLX2": flex2,
            "Zerostress": zerostress, "ISENSOR": sensor_id,
            "FUN_A1": fun_a1, "MAT_C1": c1,
            "FUN_A2": fun_a2, "MAT_C2": c2,
            "FUN_A3": fun_a3, "MAT_C3": c3,
            "FUN_A4": fun_a4, "FUN_A5": fun_a5,
        }
    )
    mat158.record = GenericMaterialRecord(
        law_name="LAW158", law_number=158, id=mat_id, title=title,
        params=mat158.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat158




def read_mat_law169(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW169/id`` or ``/MAT/ARUP_ADHESIVE/id`` (M591): Arup structural adhesive cohesive model."""
    from ...model.entities import MaterialLaw169, Material
    from ..mat_reader import GenericMaterialRecord
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/MAT/LAW169/{mat_id}: missing data cards", block.source)
        return

    rho0 = 0.0
    young, nu = 0.0, 0.0
    sht_sl, tenmax, gcten = 0.0, 1e20, 1e20
    shrmax, gcshr = 1e20, 1e20
    pwrt, pwrs = 2, 2
    shrp = 0.0

    parsed = False
    if block.fixed:
        try:
            if len(valid_cards) > 0:
                f1 = cut(valid_cards[0].raw, "MAT_LAW169_1")
                rho0 = _f(f1[0]) if len(f1) > 0 else 0.0
            if len(valid_cards) > 1:
                f2 = cut(valid_cards[1].raw, "MAT_LAW169_2")
                young = _f(f2[0]) if len(f2) > 0 else 0.0
                nu = _f(f2[1]) if len(f2) > 1 else 0.0
                sht_sl = _f(f2[2]) if len(f2) > 2 else 0.0
                tenmax = _f(f2[3]) if len(f2) > 3 and f2[3].strip() else 1e20
                gcten = _f(f2[4]) if len(f2) > 4 and f2[4].strip() else 1e20
            if len(valid_cards) > 2:
                f3 = cut(valid_cards[2].raw, "MAT_LAW169_3")
                shrmax = _f(f3[0]) if len(f3) > 0 and f3[0].strip() else 1e20
                gcshr = _f(f3[1]) if len(f3) > 1 and f3[1].strip() else 1e20
                pwrt = _i(f3[2]) if len(f3) > 2 and f3[2].strip() else 2
                pwrs = _i(f3[3]) if len(f3) > 3 and f3[3].strip() else 2
                shrp = _f(f3[4]) if len(f3) > 4 else 0.0
            parsed = True
        except (ValueError, IndexError):
            parsed = False

    if not parsed:
        if len(valid_cards) > 0:
            t1 = valid_cards[0].tokens()
            rho0 = float(t1[0]) if len(t1) > 0 else 0.0
        if len(valid_cards) > 1:
            t2 = valid_cards[1].tokens()
            young = float(t2[0]) if len(t2) > 0 else 0.0
            nu = float(t2[1]) if len(t2) > 1 else 0.0
            sht_sl = float(t2[2]) if len(t2) > 2 else 0.0
            tenmax = float(t2[3]) if len(t2) > 3 else 1e20
            gcten = float(t2[4]) if len(t2) > 4 else 1e20
        if len(valid_cards) > 2:
            t3 = valid_cards[2].tokens()
            shrmax = float(t3[0]) if len(t3) > 0 else 1e20
            gcshr = float(t3[1]) if len(t3) > 1 else 1e20
            pwrt = int(float(t3[2])) if len(t3) > 2 else 2
            pwrs = int(float(t3[3])) if len(t3) > 3 else 2
            shrp = float(t3[4]) if len(t3) > 4 else 0.0

    m169 = MaterialLaw169(
        id=mat_id, title=title, rho0=rho0,
        young=young, nu=nu, sht_sl=sht_sl, tenmax=tenmax, gcten=gcten,
        shrmax=shrmax, gcshr=gcshr, pwrt=pwrt, pwrs=pwrs, shrp=shrp,
    )
    model.mat_law169s[mat_id] = m169
    mat169 = Material(
        id=mat_id, law=169, rho0=rho0, title=title,
        params={
            "E": young, "nu": nu, "Rho": rho0, "MAT_RHO": rho0,
            "SHT_SL": sht_sl, "MAT169_SHT_SL": sht_sl, "sht_sl": sht_sl,
            "TENMAX": tenmax, "MAT169_TENMAX": tenmax, "tenmax": tenmax,
            "GCTEN": gcten, "MAT169_GCTEN": gcten, "gcten": gcten,
            "SHRMAX": shrmax, "MAT169_SHRMAX": shrmax, "shrmax": shrmax,
            "GCSHR": gcshr, "MAT169_GCSHR": gcshr, "gcshr": gcshr,
            "PWRT": pwrt, "MAT169_PWRT": pwrt, "pwrt": pwrt,
            "PWRS": pwrs, "MAT169_PWRS": pwrs, "pwrs": pwrs,
            "SHRP": shrp, "MAT169_SHRP": shrp, "shrp": shrp,
        }
    )
    from ...materials.law169_arup import build_law169
    mat169 = build_law169(mat169)
    mat169.record = GenericMaterialRecord(
        law_name="LAW169", law_number=169, id=mat_id, title=title,
        params=mat169.params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat169




def read_mat_law113(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW113/id`` or ``/MAT/SPR_BEAM/id`` (M179): Nonlinear spring-beam material."""
    from ...model.entities import MatLaw113, MatLaw113Dof
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/MAT/LAW113/{mat_id}: missing data cards", block.source)
        return

    rho, ifail, ileng, ifail2 = 0.0, 0, 0, 0
    if block.fixed:
        f1 = cut(valid_cards[0].raw, "MAT_LAW113_1")
        rho = _f(f1[0]) if len(f1) > 0 else 0.0
        ifail = _i(f1[1]) if len(f1) > 1 else 0
        ileng = _i(f1[2]) if len(f1) > 2 else 0
        ifail2 = _i(f1[3]) if len(f1) > 3 else 0
    else:
        toks1 = valid_cards[0].tokens()
        rho = float(toks1[0]) if len(toks1) > 0 else 0.0
        ifail = int(float(toks1[1])) if len(toks1) > 1 else 0
        ileng = int(float(toks1[2])) if len(toks1) > 2 else 0
        ifail2 = int(float(toks1[3])) if len(toks1) > 3 else 0

    card_idx = 1
    dofs: List[MatLaw113Dof] = []
    for d in range(6):
        dof = MatLaw113Dof()
        if card_idx < len(valid_cards):
            if block.fixed:
                fa = cut(valid_cards[card_idx].raw, "MAT_LAW113_DOF_1")
                dof.stiff = _f(fa[0]) if len(fa) > 0 else 0.0
                dof.damp = _f(fa[1]) if len(fa) > 1 else 0.0
                dof.acoeft = _f(fa[2], 1.0) if len(fa) > 2 and fa[2].strip() else 1.0
                dof.bcoeft = _f(fa[3], 1.0) if len(fa) > 3 and fa[3].strip() else 1.0
                dof.dcoeft = _f(fa[4], 1.0) if len(fa) > 4 and fa[4].strip() else 1.0
            else:
                ta = valid_cards[card_idx].tokens()
                dof.stiff = float(ta[0]) if len(ta) > 0 else 0.0
                dof.damp = float(ta[1]) if len(ta) > 1 else 0.0
                dof.acoeft = float(ta[2]) if len(ta) > 2 else 1.0
                dof.bcoeft = float(ta[3]) if len(ta) > 3 else 1.0
                dof.dcoeft = float(ta[4]) if len(ta) > 4 else 1.0
            card_idx += 1

        if card_idx < len(valid_cards):
            if block.fixed:
                fb = cut(valid_cards[card_idx].raw, "MAT_LAW113_DOF_2")
                dof.fun_a = _i(fb[0]) if len(fb) > 0 else 0
                dof.hflag = _i(fb[1]) if len(fb) > 1 else 0
                dof.fun_b = _i(fb[2]) if len(fb) > 2 else 0
                dof.fun_c = _i(fb[3]) if len(fb) > 3 else 0
                dof.fun_d = _i(fb[4]) if len(fb) > 4 else 0
            else:
                tb = valid_cards[card_idx].tokens()
                dof.fun_a = int(float(tb[0])) if len(tb) > 0 else 0
                dof.hflag = int(float(tb[1])) if len(tb) > 1 else 0
                dof.fun_b = int(float(tb[2])) if len(tb) > 2 else 0
                dof.fun_c = int(float(tb[3])) if len(tb) > 3 else 0
                dof.fun_d = int(float(tb[4])) if len(tb) > 4 else 0
            card_idx += 1

        if card_idx < len(valid_cards):
            if block.fixed:
                fc = cut(valid_cards[card_idx].raw, "MAT_LAW113_DOF_3")
                dof.min_rup = _f(fc[0], -1e30) if len(fc) > 0 and fc[0].strip() else -1e30
                dof.max_rup = _f(fc[1], 1e30) if len(fc) > 1 and fc[1].strip() else 1e30
                dof.prop_f = _f(fc[2]) if len(fc) > 2 else 0.0
                dof.prop_e = _f(fc[3]) if len(fc) > 3 else 0.0
                dof.scale = _f(fc[4], 1.0) if len(fc) > 4 and fc[4].strip() else 1.0
            else:
                tc = valid_cards[card_idx].tokens()
                dof.min_rup = float(tc[0]) if len(tc) > 0 else -1e30
                dof.max_rup = float(tc[1]) if len(tc) > 1 else 1e30
                dof.prop_f = float(tc[2]) if len(tc) > 2 else 0.0
                dof.prop_e = float(tc[3]) if len(tc) > 3 else 0.0
                dof.scale = float(tc[4]) if len(tc) > 4 else 1.0
            card_idx += 1

        if card_idx < len(valid_cards):
            if block.fixed:
                fd = cut(valid_cards[card_idx].raw, "MAT_LAW113_DOF_4")
                dof.prop_h = _f(fd[0], 1.0) if len(fd) > 0 and fd[0].strip() else 1.0
                dof.fun_k = _i(fd[1]) if len(fd) > 1 else 0
            else:
                td = valid_cards[card_idx].tokens()
                dof.prop_h = float(td[0]) if len(td) > 0 else 1.0
                dof.fun_k = int(float(td[1])) if len(td) > 1 else 0
            card_idx += 1

        dofs.append(dof)

    trans_vel0, rot_vel0, asrate, israte = 1.0, 1.0, 1.0e30, 0
    if card_idx < len(valid_cards):
        if block.fixed:
            fr = cut(valid_cards[card_idx].raw, "MAT_LAW113_RATE")
            trans_vel0 = _f(fr[0], 1.0) if len(fr) > 0 and fr[0].strip() else 1.0
            rot_vel0 = _f(fr[1], 1.0) if len(fr) > 1 and fr[1].strip() else 1.0
            asrate = _f(fr[2], 1.0e30) if len(fr) > 2 and fr[2].strip() else 1.0e30
            israte = _i(fr[3]) if len(fr) > 3 else 0
        else:
            tr = valid_cards[card_idx].tokens()
            trans_vel0 = float(tr[0]) if len(tr) > 0 else 1.0
            rot_vel0 = float(tr[1]) if len(tr) > 1 else 1.0
            asrate = float(tr[2]) if len(tr) > 2 else 1.0e30
            israte = int(float(tr[3])) if len(tr) > 3 else 0
        card_idx += 1

    dir_fails = []
    while card_idx < len(valid_cards):
        if block.fixed:
            ff = cut(valid_cards[card_idx].raw, "MAT_LAW113_DIRFAIL")
            c_val = _f(ff[0]) if len(ff) > 0 else 0.0
            rel_exp = _f(ff[1]) if len(ff) > 1 else 0.0
            alpha = _f(ff[2], 1.0) if len(ff) > 2 and ff[2].strip() else 1.0
            beta = _f(ff[3], 2.0) if len(ff) > 3 and ff[3].strip() else 2.0
        else:
            tf = valid_cards[card_idx].tokens()
            c_val = float(tf[0]) if len(tf) > 0 else 0.0
            rel_exp = float(tf[1]) if len(tf) > 1 else 0.0
            alpha = float(tf[2]) if len(tf) > 2 else 1.0
            beta = float(tf[3]) if len(tf) > 3 else 2.0
        dir_fails.append([c_val, rel_exp, alpha, beta])
        card_idx += 1

    m113 = MatLaw113(
        id=mat_id, rho=rho, ifail=ifail, ileng=ileng, ifail2=ifail2,
        dofs=dofs, trans_vel0=trans_vel0, rot_vel0=rot_vel0, asrate=asrate, israte=israte,
        dir_fails=dir_fails, title=title,
    )
    model.mat_law113s[mat_id] = m113
    from ..mat_reader import GenericMaterialRecord
    stiff0 = dofs[0].stiff if dofs else 1000.0
    mat113 = Material(
        id=mat_id, law=113, rho0=rho, title=title,
        params={"E": stiff0, "nu": 0.3, "MAT_RHO": rho, "Ifail": ifail, "Ileng": ileng, "Ifail2": ifail2}
    )
    mat113.record = GenericMaterialRecord(
        law_name="LAW113", law_number=113, id=mat_id, title=title,
        params=mat113.params, density=rho, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat113




def read_mat_law79(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW79/id`` or ``/MAT/JOHN_HOLM/id`` (M179/M558): Johnson-Holmquist ceramic material model."""
    from ...model.entities import MatLaw79, Material
    mat_id = block.user_id or 0

    def _card_tokens(c: Any) -> list[str]:
        raw = getattr(c, "raw", str(c)).strip()
        for ch in ("#", "$"):
            if ch in raw:
                raw = raw.split(ch)[0].strip()
        if not raw:
            return []
        if "," in raw:
            parts = [p.strip() for p in raw.split(",")]
            while parts and parts[-1] == "":
                parts.pop()
            return parts
        return raw.split()

    def _fval_safe(v: Any, default: float = 0.0) -> float:
        if v is None:
            return default
        s = str(v).strip().replace("D", "E").replace("d", "e")
        if not s:
            return default
        try:
            return float(s)
        except (ValueError, TypeError):
            return default

    has_comma = any("," in getattr(c, "raw", str(c)) for c in block.cards)
    is_fixed = block.fixed and not has_comma
    if is_fixed:
        for c in block.cards:
            c_raw = getattr(c, "raw", str(c))
            raw_s = c_raw.strip()
            if not raw_s or raw_s.startswith(("#", "$")):
                continue
            if "\t" in c_raw:
                is_fixed = False
                break
            toks = raw_s.split()
            if len(toks) > 1 and len(c_raw[:20].split()) > 1:
                is_fixed = False
                break
            if len(toks) > 1 and len(c_raw.rstrip()) <= 20:
                is_fixed = False
                break

    title, cards = _fixed_data(block) if is_fixed else _title_and_data(block)
    if is_fixed:
        valid_cards = [c for c in cards if not getattr(c, "raw", str(c)).strip().startswith(("#", "$"))]
    else:
        valid_cards = [c for c in cards if not getattr(c, "is_blank", False) and not getattr(c, "raw", str(c)).strip().startswith(("#", "$"))]

    if not valid_cards:
        log.error(f"/MAT/LAW79/{mat_id}: missing data cards", block.source)
        return

    rho, refer_rho = 0.0, 0.0
    tau_shear = 0.0
    a, b, m, n = 0.0, 0.0, 0.0, 0.0
    c_val, eps0, sigfmax, fcut = 0.0, 1.0, 1.0e20, 0.0
    t, hel, phel = 0.0, 0.0, 0.0
    d1, d2, idel, epsmax = 0.0, 0.0, 0, 0.0
    k1, k2, k3, beta = 0.0, 0.0, 0.0, 0.0

    if is_fixed:
        # Card 1: RHO, [Refer_Rho] (MAT_LAW79_1: [20, 20])
        if len(valid_cards) > 0:
            f1 = valid_cards[0].cut("MAT_LAW79_1")
            rho = _fval_safe(f1[0]) if len(f1) > 0 else 0.0
            refer_rho = _fval_safe(f1[1]) if len(f1) > 1 else 0.0
        # Card 2: tau_shear (MAT_LAW79_2: [20])
        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("MAT_LAW79_2")
            tau_shear = _fval_safe(f2[0]) if len(f2) > 0 else 0.0
        # Card 3: a, b, m, n (MAT_LAW79_3: [20, 20, 20, 20])
        if len(valid_cards) > 2:
            f3 = valid_cards[2].cut("MAT_LAW79_3")
            a = _fval_safe(f3[0]) if len(f3) > 0 else 0.0
            b = _fval_safe(f3[1]) if len(f3) > 1 else 0.0
            m = _fval_safe(f3[2]) if len(f3) > 2 else 0.0
            n = _fval_safe(f3[3]) if len(f3) > 3 else 0.0
        # Card 4: c, eps0, sigfmax, fcut (MAT_LAW79_4: [20, 20, 20, 20])
        if len(valid_cards) > 3:
            f4 = valid_cards[3].cut("MAT_LAW79_4")
            c_val = _fval_safe(f4[0]) if len(f4) > 0 else 0.0
            eps0 = _fval_safe(f4[1], 1.0) if len(f4) > 1 and f4[1].strip() else 1.0
            sigfmax = _fval_safe(f4[2], 1.0e20) if len(f4) > 2 and f4[2].strip() else 1.0e20
            fcut = _fval_safe(f4[3]) if len(f4) > 3 else 0.0
        # Card 5: t, hel, phel (MAT_LAW79_5: [20, 20, 20])
        if len(valid_cards) > 4:
            f5 = valid_cards[4].cut("MAT_LAW79_5")
            t = _fval_safe(f5[0]) if len(f5) > 0 else 0.0
            hel = _fval_safe(f5[1]) if len(f5) > 1 else 0.0
            phel = _fval_safe(f5[2]) if len(f5) > 2 else 0.0
        # Card 6: d1, d2, [blank], idel, epsmax (MAT_LAW79_6: [20, 20, 10, 10, 20])
        if len(valid_cards) > 5:
            f6 = valid_cards[5].cut("MAT_LAW79_6")
            d1 = _fval_safe(f6[0]) if len(f6) > 0 else 0.0
            d2 = _fval_safe(f6[1]) if len(f6) > 1 else 0.0
            idel = int(_fval_safe(f6[3])) if len(f6) > 3 else 0
            epsmax = _fval_safe(f6[4], 1.0e20) if len(f6) > 4 and f6[4].strip() else 1.0e20
        # Card 7: k1, k2, k3, beta (MAT_LAW79_7: [20, 20, 20, 20])
        if len(valid_cards) > 6:
            f7 = valid_cards[6].cut("MAT_LAW79_7")
            k1 = _fval_safe(f7[0]) if len(f7) > 0 else 0.0
            k2 = _fval_safe(f7[1]) if len(f7) > 1 else 0.0
            k3 = _fval_safe(f7[2]) if len(f7) > 2 else 0.0
            beta = _fval_safe(f7[3]) if len(f7) > 3 else 0.0
    else:
        # Free format
        if len(valid_cards) > 0:
            t1 = _card_tokens(valid_cards[0])
            rho = _fval_safe(t1[0]) if len(t1) > 0 else 0.0
            refer_rho = _fval_safe(t1[1]) if len(t1) > 1 else 0.0
        if len(valid_cards) > 1:
            t2 = _card_tokens(valid_cards[1])
            tau_shear = _fval_safe(t2[0]) if len(t2) > 0 else 0.0
        if len(valid_cards) > 2:
            t3 = _card_tokens(valid_cards[2])
            a = _fval_safe(t3[0]) if len(t3) > 0 else 0.0
            b = _fval_safe(t3[1]) if len(t3) > 1 else 0.0
            m = _fval_safe(t3[2]) if len(t3) > 2 else 0.0
            n = _fval_safe(t3[3]) if len(t3) > 3 else 0.0
        if len(valid_cards) > 3:
            t4 = _card_tokens(valid_cards[3])
            c_val = _fval_safe(t4[0]) if len(t4) > 0 else 0.0
            eps0 = _fval_safe(t4[1], 1.0) if len(t4) > 1 and t4[1].strip() else 1.0
            sigfmax = _fval_safe(t4[2], 1.0e20) if len(t4) > 2 and t4[2].strip() else 1.0e20
            fcut = _fval_safe(t4[3]) if len(t4) > 3 else 0.0
        if len(valid_cards) > 4:
            t5 = _card_tokens(valid_cards[4])
            t = _fval_safe(t5[0]) if len(t5) > 0 else 0.0
            hel = _fval_safe(t5[1]) if len(t5) > 1 else 0.0
            phel = _fval_safe(t5[2]) if len(t5) > 2 else 0.0
        if len(valid_cards) > 5:
            t6 = _card_tokens(valid_cards[5])
            d1 = _fval_safe(t6[0]) if len(t6) > 0 else 0.0
            d2 = _fval_safe(t6[1]) if len(t6) > 1 else 0.0
            if len(t6) >= 5:
                if t6[2] == "":
                    idel = int(_fval_safe(t6[3]))
                    epsmax = _fval_safe(t6[4], 1.0e20) if t6[4].strip() else 1.0e20
                else:
                    try:
                        idel = int(_fval_safe(t6[3]))
                        epsmax = _fval_safe(t6[4], 1.0e20) if t6[4].strip() else 1.0e20
                    except (ValueError, IndexError):
                        idel = int(_fval_safe(t6[2]))
                        epsmax = _fval_safe(t6[3], 1.0e20) if t6[3].strip() else 1.0e20
            elif len(t6) >= 4:
                idel = int(_fval_safe(t6[2]))
                epsmax = _fval_safe(t6[3], 1.0e20) if t6[3].strip() else 1.0e20
            elif len(t6) == 3:
                idel = int(_fval_safe(t6[2]))
        if len(valid_cards) > 6:
            t7 = _card_tokens(valid_cards[6])
            k1 = _fval_safe(t7[0]) if len(t7) > 0 else 0.0
            k2 = _fval_safe(t7[1]) if len(t7) > 1 else 0.0
            k3 = _fval_safe(t7[2]) if len(t7) > 2 else 0.0
            beta = _fval_safe(t7[3]) if len(t7) > 3 else 0.0

    # Defaults per hm_read_mat79.F
    if refer_rho == 0.0:
        refer_rho = rho
    if c_val == 0.0 and eps0 == 0.0:
        eps0 = 1.0
    elif eps0 == 0.0:
        eps0 = 1.0
    if sigfmax == 0.0:
        sigfmax = 1.0e20
    if epsmax == 0.0:
        epsmax = 1.0e20
    idel = min(max(0, idel), 3)

    m79 = MatLaw79(
        id=mat_id, rho=rho, refer_rho=refer_rho, tau_shear=tau_shear,
        a=a, b=b, m=m, n=n, c=c_val, eps0=eps0, sigfmax=sigfmax, fcut=fcut,
        t=t, hel=hel, phel=phel, d1=d1, d2=d2, idel=idel, epsmax=epsmax,
        k1=k1, k2=k2, k3=k3, beta=beta, title=title,
    )
    model.mat_law79s[mat_id] = m79
    if hasattr(model, "mat_john_holms"):
        model.mat_john_holms[mat_id] = m79
    if hasattr(model, "mat_jh2s"):
        model.mat_jh2s[mat_id] = m79

    from ..mat_reader import GenericMaterialRecord
    denom_e = 3.0 * k1 + tau_shear
    e_val = (9.0 * k1 * tau_shear) / denom_e if denom_e > 0 else 2.0 * tau_shear * 1.3
    denom_nu = 6.0 * k1 + 2.0 * tau_shear
    nu_val = (3.0 * k1 - 2.0 * tau_shear) / denom_nu if denom_nu > 0 else 0.3
    mat79 = Material(
        id=mat_id, law=79, rho0=rho, title=title,
        params={
            "E": e_val, "nu": nu_val, "G": tau_shear, "shear": tau_shear, "MAT_RHO": rho,
            "tau_shear": tau_shear, "rho": rho, "refer_rho": refer_rho,
            "K1": k1, "K2": k2, "K3": k3, "k1": k1, "k2": k2, "k3": k3,
            "MAT_A": a, "MAT_B": b, "MAT_M": m, "MAT_N": n, "MAT_C": c_val,
            "a": a, "b": b, "m": m, "n": n, "c": c_val,
            "MAT_Epsilon_F": eps0, "eps0": eps0,
            "MAT_SIG1max_t": sigfmax, "sigfmax": sigfmax, "sigma_fmax": sigfmax,
            "MAT_FCUT": fcut, "fcut": fcut,
            "MAT_T0": t, "t": t, "t0": t,
            "MAT_E": hel, "hel": hel,
            "MAT_EPS": phel, "phel": phel,
            "shel": 1.5 * (hel - phel),
            "tstar": (t / phel) if phel != 0.0 else 0.0,
            "D1": d1, "d1": d1, "D2": d2, "d2": d2,
            "IDEL": idel, "idel": idel,
            "EPSMAX": epsmax, "epsmax": epsmax, "eps_max": epsmax,
            "MAT_Beta": beta, "beta": beta,
        }
    )
    mat79.record = GenericMaterialRecord(
        law_name="LAW79", law_number=79, id=mat_id, title=title,
        params=mat79.params, density=rho, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat79




def read_mat_visc_lprony(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/VISC_LPRONY/id`` or ``/VISC/LPRONY/id`` (M179): Viscoelastic Large Prony series."""
    from ...model.entities import MatViscLprony
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/MAT/VISC_LPRONY/{mat_id}: missing data cards", block.source)
        return

    m_order, form, flag_visc = 0, 0, 0
    if block.fixed:
        f1 = valid_cards[0].cut("MAT_VISC_LPRONY_1")
        m_order = _ival(f1[0]) if len(f1) > 0 else 0
        form = _ival(f1[1]) if len(f1) > 1 else 0
        flag_visc = _ival(f1[2]) if len(f1) > 2 else 0
    else:
        toks1 = valid_cards[0].tokens()
        m_order = int(float(toks1[0])) if len(toks1) > 0 else 0
        form = int(float(toks1[1])) if len(toks1) > 1 else 0
        flag_visc = int(float(toks1[2])) if len(toks1) > 2 else 0

    gamai = []
    taui = []
    for c in valid_cards[1: 1 + m_order]:
        if block.fixed:
            fc = c.cut("MAT_VISC_LPRONY_ITEM")
            g = _fval(fc[0]) if len(fc) > 0 else 0.0
            t = _fval(fc[1]) if len(fc) > 1 else 0.0
        else:
            tc = c.tokens()
            g = float(tc[0]) if len(tc) > 0 else 0.0
            t = float(tc[1]) if len(tc) > 1 else 0.0
        gamai.append(g)
        taui.append(t)

    model.mat_visc_lpronys[mat_id] = MatViscLprony(
        id=mat_id, m=m_order, form=form, flag_visc=flag_visc,
        gamai=gamai, taui=taui, title=title,
    )
    from ...model.entities import MaterialViscProny
    model.mat_visc_pronys[mat_id] = MaterialViscProny(
        mat_id=mat_id, title=title, order=m_order, form=form, flag_visc=flag_visc,
        gammas=gamai, taus=taui
    )




# M180: MAT_LAW190, MAT_LAW41, PROP_TYPE20, PROP_TYPE21, PROP_TYPE22
def read_mat_law190(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW190/id`` or ``/MAT/FOAM_DUBOIS/id`` (M180): Du Bois foam model with 3D table."""
    from ...model.entities import MatLaw190
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/MAT/LAW190/{mat_id}: missing data cards", block.source)
        return

    rho, e0, nu = 0.0, 0.0, 0.0
    hu, shape = 0.0, 1.0
    fun_1, xscale_1, scale_1 = 0, 1.0, 1.0
    tcut, fail = 1.0e20, 0

    if block.fixed:
        f1 = valid_cards[0].cut("MAT_LAW190_1")
        rho = _fval(f1[0]) if len(f1) > 0 else 0.0

        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("MAT_LAW190_2")
            e0 = _fval(f2[0]) if len(f2) > 0 else 0.0
            nu = _fval(f2[1]) if len(f2) > 1 else 0.0

        if len(valid_cards) > 2:
            f3 = valid_cards[2].cut("MAT_LAW190_3")
            hu = _fval(f3[0]) if len(f3) > 0 else 0.0
            shape = _fval(f3[1], 1.0) if len(f3) > 1 and f3[1].strip() else 1.0

        if len(valid_cards) > 3:
            f4 = valid_cards[3].cut("MAT_LAW190_4")
            fun_1 = _ival(f4[0]) if len(f4) > 0 else 0
            xscale_1 = _fval(f4[1], 1.0) if len(f4) > 1 and f4[1].strip() else 1.0
            scale_1 = _fval(f4[2], 1.0) if len(f4) > 2 and f4[2].strip() else 1.0

        if len(valid_cards) > 4:
            f5 = valid_cards[4].cut("MAT_LAW190_5")
            tcut = _fval(f5[0], 1.0e20) if len(f5) > 0 and f5[0].strip() else 1.0e20
            fail = _ival(f5[1], 0) if len(f5) > 1 and f5[1].strip() else 0
    else:
        toks1 = valid_cards[0].tokens()
        rho = float(toks1[0]) if len(toks1) > 0 else 0.0

        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            e0 = float(toks2[0]) if len(toks2) > 0 else 0.0
            nu = float(toks2[1]) if len(toks2) > 1 else 0.0

        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            hu = float(toks3[0]) if len(toks3) > 0 else 0.0
            shape = float(toks3[1]) if len(toks3) > 1 else 1.0

        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            fun_1 = int(float(toks4[0])) if len(toks4) > 0 else 0
            xscale_1 = float(toks4[1]) if len(toks4) > 1 else 1.0
            scale_1 = float(toks4[2]) if len(toks4) > 2 else 1.0

        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            tcut = float(toks5[0]) if len(toks5) > 0 else 1.0e20
            fail = int(float(toks5[1])) if len(toks5) > 1 else 0

    m190 = MatLaw190(
        id=mat_id, rho=rho, e0=e0, nu=nu, hu=hu, shape=shape,
        fun_1=fun_1, xscale_1=xscale_1, scale_1=scale_1,
        tcut=tcut, fail=fail, title=title,
    )
    model.mat_law190s[mat_id] = m190

    params = {
        "E": e0, "E0": e0, "e0": e0, "MAT_E": e0,
        "nu": nu, "Nu": nu, "MAT_NU": nu,
        "MAT_RHO": rho, "rho0": rho, "rho": rho,
        "MAT_HU": hu, "hu": hu, "hys": hu, "HU": hu,
        "MAT_SHAPE": shape, "shape": shape, "SHAPE": shape,
        "FUN_1": fun_1, "fun_1": fun_1, "table_id": fun_1,
        "XSCALE_1": xscale_1, "xscale_1": xscale_1, "xscale": xscale_1,
        "SCALE_1": scale_1, "scale_1": scale_1, "scale": scale_1,
        "tcut": tcut, "TCUT": tcut, "fail": fail, "FAIL": fail,
    }
    from ..mat_reader import GenericMaterialRecord
    mat190 = Material(
        id=mat_id, law=190, rho0=rho, title=title,
        params=params,
    )
    mat190.record = GenericMaterialRecord(
        law_name="LAW190", law_number=190, id=mat_id, title=title,
        params=mat190.params, density=rho, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat190




def read_mat_law41(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW41/id`` or ``/MAT/LEE_T/id`` (M180): Lee-Tarver explosive reaction kinetics and JWL EOS."""
    from ...model.entities import MatLaw41
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/MAT/LAW41/{mat_id}: missing data cards", block.source)
        return

    rho, refer_rho = 0.0, 0.0
    ireac = 0
    a_r, b_r, r_1r, r_2r, r_3r = 0.0, 0.0, 0.0, 0.0, 0.0
    a_p, b_p, r_1p, r_2p, r_3p = 0.0, 0.0, 0.0, 0.0, 0.0
    c_vr, c_vp, enq = 0.0, 0.0, 0.0
    nitrs, epsilon_0, ftol = 0, 0.0, 0.0
    i_coeff, b_coeff, x_coeff = 0.0, 0.0, 0.0
    g1, d_coeff, y_coeff, c_coeff = 0.0, 0.0, 0.0, 0.0
    kn, chi, tol = 0.0, 0.0, 0.0
    g2, e_coeff, g_coeff, z_coeff = 0.0, 0.0, 0.0, 0.0
    ccrit, figmax, fg1max, fg2min = 0.0, 0.0, 0.0, 0.0
    g0, t_initial = 0.0, 293.15

    if block.fixed:
        f1 = valid_cards[0].cut("MAT_LAW41_1")
        rho = _fval(f1[0]) if len(f1) > 0 else 0.0
        refer_rho = _fval(f1[1]) if len(f1) > 1 else 0.0

        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("MAT_LAW41_2")
            ireac = _ival(f2[0]) if len(f2) > 0 else 0
            a_r = _fval(f2[1]) if len(f2) > 1 else 0.0
            b_r = _fval(f2[2]) if len(f2) > 2 else 0.0
            r_1r = _fval(f2[3]) if len(f2) > 3 else 0.0
            r_2r = _fval(f2[4]) if len(f2) > 4 else 0.0
            r_3r = _fval(f2[5]) if len(f2) > 5 else 0.0

        if len(valid_cards) > 2:
            f3 = valid_cards[2].cut("MAT_LAW41_3")
            a_p = _fval(f3[0]) if len(f3) > 0 else 0.0
            b_p = _fval(f3[1]) if len(f3) > 1 else 0.0
            r_1p = _fval(f3[2]) if len(f3) > 2 else 0.0
            r_2p = _fval(f3[3]) if len(f3) > 3 else 0.0
            r_3p = _fval(f3[4]) if len(f3) > 4 else 0.0

        if len(valid_cards) > 3:
            f4 = valid_cards[3].cut("MAT_LAW41_4")
            c_vr = _fval(f4[0]) if len(f4) > 0 else 0.0
            c_vp = _fval(f4[1]) if len(f4) > 1 else 0.0
            enq = _fval(f4[2]) if len(f4) > 2 else 0.0

        if len(valid_cards) > 4:
            f5 = valid_cards[4].cut("MAT_LAW41_5")
            nitrs = _ival(f5[0]) if len(f5) > 0 else 0
            epsilon_0 = _fval(f5[1]) if len(f5) > 1 else 0.0
            ftol = _fval(f5[2]) if len(f5) > 2 else 0.0

        if len(valid_cards) > 5:
            f6 = valid_cards[5].cut("MAT_LAW41_6")
            i_coeff = _fval(f6[0]) if len(f6) > 0 else 0.0
            b_coeff = _fval(f6[1]) if len(f6) > 1 else 0.0
            x_coeff = _fval(f6[2]) if len(f6) > 2 else 0.0

        if len(valid_cards) > 6:
            f7 = valid_cards[6].cut("MAT_LAW41_7")
            g1 = _fval(f7[0]) if len(f7) > 0 else 0.0
            d_coeff = _fval(f7[1]) if len(f7) > 1 else 0.0
            y_coeff = _fval(f7[2]) if len(f7) > 2 else 0.0
            c_coeff = _fval(f7[3]) if len(f7) > 3 else 0.0

        if len(valid_cards) > 7:
            f8 = valid_cards[7].cut("MAT_LAW41_8")
            kn = _fval(f8[0]) if len(f8) > 0 else 0.0
            chi = _fval(f8[1]) if len(f8) > 1 else 0.0
            tol = _fval(f8[2]) if len(f8) > 2 else 0.0

        if len(valid_cards) > 8:
            f9 = valid_cards[8].cut("MAT_LAW41_9")
            g2 = _fval(f9[0]) if len(f9) > 0 else 0.0
            e_coeff = _fval(f9[1]) if len(f9) > 1 else 0.0
            g_coeff = _fval(f9[2]) if len(f9) > 2 else 0.0
            z_coeff = _fval(f9[3]) if len(f9) > 3 else 0.0

        if len(valid_cards) > 9:
            f10 = valid_cards[9].cut("MAT_LAW41_10")
            ccrit = _fval(f10[0]) if len(f10) > 0 else 0.0
            figmax = _fval(f10[1]) if len(f10) > 1 else 0.0
            fg1max = _fval(f10[2]) if len(f10) > 2 else 0.0
            fg2min = _fval(f10[3]) if len(f10) > 3 else 0.0

        if len(valid_cards) > 10:
            f11 = valid_cards[10].cut("MAT_LAW41_11")
            g0 = _fval(f11[0]) if len(f11) > 0 else 0.0
            t_initial = _fval(f11[1], 293.15) if len(f11) > 1 and f11[1].strip() else 293.15
    else:
        toks1 = valid_cards[0].tokens()
        rho = float(toks1[0]) if len(toks1) > 0 else 0.0
        refer_rho = float(toks1[1]) if len(toks1) > 1 else 0.0

        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            ireac = int(float(toks2[0])) if len(toks2) > 0 else 0
            a_r = float(toks2[1]) if len(toks2) > 1 else 0.0
            b_r = float(toks2[2]) if len(toks2) > 2 else 0.0
            r_1r = float(toks2[3]) if len(toks2) > 3 else 0.0
            r_2r = float(toks2[4]) if len(toks2) > 4 else 0.0
            r_3r = float(toks2[5]) if len(toks2) > 5 else 0.0

        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            a_p = float(toks3[0]) if len(toks3) > 0 else 0.0
            b_p = float(toks3[1]) if len(toks3) > 1 else 0.0
            r_1p = float(toks3[2]) if len(toks3) > 2 else 0.0
            r_2p = float(toks3[3]) if len(toks3) > 3 else 0.0
            r_3p = float(toks3[4]) if len(toks3) > 4 else 0.0

        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            c_vr = float(toks4[0]) if len(toks4) > 0 else 0.0
            c_vp = float(toks4[1]) if len(toks4) > 1 else 0.0
            enq = float(toks4[2]) if len(toks4) > 2 else 0.0

        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            nitrs = int(float(toks5[0])) if len(toks5) > 0 else 0
            epsilon_0 = float(toks5[1]) if len(toks5) > 1 else 0.0
            ftol = float(toks5[2]) if len(toks5) > 2 else 0.0

        if len(valid_cards) > 5:
            toks6 = valid_cards[5].tokens()
            i_coeff = float(toks6[0]) if len(toks6) > 0 else 0.0
            b_coeff = float(toks6[1]) if len(toks6) > 1 else 0.0
            x_coeff = float(toks6[2]) if len(toks6) > 2 else 0.0

        if len(valid_cards) > 6:
            toks7 = valid_cards[6].tokens()
            g1 = float(toks7[0]) if len(toks7) > 0 else 0.0
            d_coeff = float(toks7[1]) if len(toks7) > 1 else 0.0
            y_coeff = float(toks7[2]) if len(toks7) > 2 else 0.0
            c_coeff = float(toks7[3]) if len(toks7) > 3 else 0.0

        if len(valid_cards) > 7:
            toks8 = valid_cards[7].tokens()
            kn = float(toks8[0]) if len(toks8) > 0 else 0.0
            chi = float(toks8[1]) if len(toks8) > 1 else 0.0
            tol = float(toks8[2]) if len(toks8) > 2 else 0.0

        if len(valid_cards) > 8:
            toks9 = valid_cards[8].tokens()
            g2 = float(toks9[0]) if len(toks9) > 0 else 0.0
            e_coeff = float(toks9[1]) if len(toks9) > 1 else 0.0
            g_coeff = float(toks9[2]) if len(toks9) > 2 else 0.0
            z_coeff = float(toks9[3]) if len(toks9) > 3 else 0.0

        if len(valid_cards) > 9:
            toks10 = valid_cards[9].tokens()
            ccrit = float(toks10[0]) if len(toks10) > 0 else 0.0
            figmax = float(toks10[1]) if len(toks10) > 1 else 0.0
            fg1max = float(toks10[2]) if len(toks10) > 2 else 0.0
            fg2min = float(toks10[3]) if len(toks10) > 3 else 0.0

        if len(valid_cards) > 10:
            toks11 = valid_cards[10].tokens()
            g0 = float(toks11[0]) if len(toks11) > 0 else 0.0
            t_initial = float(toks11[1]) if len(toks11) > 1 else 293.15

    m41 = MatLaw41(
        id=mat_id, rho=rho, refer_rho=refer_rho, ireac=ireac,
        a_r=a_r, b_r=b_r, r_1r=r_1r, r_2r=r_2r, r_3r=r_3r,
        a_p=a_p, b_p=b_p, r_1p=r_1p, r_2p=r_2p, r_3p=r_3p,
        c_vr=c_vr, c_vp=c_vp, enq=enq,
        nitrs=nitrs, epsilon_0=epsilon_0, ftol=ftol,
        i_coeff=i_coeff, b_coeff=b_coeff, x_coeff=x_coeff,
        g1=g1, d_coeff=d_coeff, y_coeff=y_coeff, c_coeff=c_coeff,
        kn=kn, chi=chi, tol=tol,
        g2=g2, e_coeff=e_coeff, g_coeff=g_coeff, z_coeff=z_coeff,
        ccrit=ccrit, figmax=figmax, fg1max=fg1max, fg2min=fg2min,
        g0=g0, t_initial=t_initial, title=title,
    )
    model.mat_law41s[mat_id] = m41
    from ..mat_reader import GenericMaterialRecord
    e_est = 2.0 * g0 * 1.3 if g0 > 0 else 1.0e5
    mat41 = Material(
        id=mat_id, law=41, rho0=rho, title=title,
        params={"E": e_est, "nu": 0.3, "G0": g0, "MAT_RHO": rho, "Ireac": ireac, "enq": enq}
    )
    mat41.record = GenericMaterialRecord(
        law_name="LAW41", law_number=41, id=mat_id, title=title,
        params=mat41.params, density=rho, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat41





# M182: MAT_LAW114, MAT_LAW117, MAT_LAW119, MAT_LAW120, MAT_LAW121, PROP_TYPE26, PROP_TYPE27
def read_mat_law114(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW114/id`` or ``/MAT/SPR_SEATBELT/id`` (M182): 1D Seatbelt spring material."""
    from ...model.entities import MatLaw114
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/MAT/LAW114/{mat_id}: missing data cards", block.source)
        return

    rho, lmin = 0.0, 0.0
    stiff1, damp1 = 0.0, 0.0
    fun_l, fun_ul = 0, 0
    xcoeft1, fcoeft1 = 1.0, 1.0
    young, ibend, itors, fmax, mmax = 0.0, 0.0, 0.0, 0.0, 0.0
    shear_area, rfac = 0.0, 0.0

    if block.fixed:
        f1 = valid_cards[0].cut("MAT_LAW114_1")
        rho = _fval(f1[0]) if len(f1) > 0 else 0.0
        lmin = _fval(f1[1]) if len(f1) > 1 else 0.0

        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("MAT_LAW114_2")
            stiff1 = _fval(f2[0]) if len(f2) > 0 else 0.0
            damp1 = _fval(f2[1]) if len(f2) > 1 else 0.0

        if len(valid_cards) > 2:
            f3 = valid_cards[2].cut("MAT_LAW114_3")
            fun_l = _ival(f3[0]) if len(f3) > 0 else 0
            fun_ul = _ival(f3[1]) if len(f3) > 1 else 0
            xcoeft1 = _fval(f3[2], 1.0) if len(f3) > 2 and f3[2].strip() else 1.0
            fcoeft1 = _fval(f3[3], 1.0) if len(f3) > 3 and f3[3].strip() else 1.0

        if len(valid_cards) > 3:
            f4 = valid_cards[3].cut("MAT_LAW114_4")
            young = _fval(f4[0]) if len(f4) > 0 else 0.0
            ibend = _fval(f4[1]) if len(f4) > 1 else 0.0
            itors = _fval(f4[2]) if len(f4) > 2 else 0.0
            fmax = _fval(f4[3]) if len(f4) > 3 else 0.0
            mmax = _fval(f4[4]) if len(f4) > 4 else 0.0

        if len(valid_cards) > 4:
            f5 = valid_cards[4].cut("MAT_LAW114_5")
            shear_area = _fval(f5[0]) if len(f5) > 0 else 0.0
            rfac = _fval(f5[1]) if len(f5) > 1 else 0.0
    else:
        toks1 = valid_cards[0].tokens()
        rho = float(toks1[0]) if len(toks1) > 0 else 0.0
        lmin = float(toks1[1]) if len(toks1) > 1 else 0.0

        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            stiff1 = float(toks2[0]) if len(toks2) > 0 else 0.0
            damp1 = float(toks2[1]) if len(toks2) > 1 else 0.0

        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            fun_l = int(float(toks3[0])) if len(toks3) > 0 else 0
            fun_ul = int(float(toks3[1])) if len(toks3) > 1 else 0
            xcoeft1 = float(toks3[2]) if len(toks3) > 2 else 1.0
            fcoeft1 = float(toks3[3]) if len(toks3) > 3 else 1.0

        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            young = float(toks4[0]) if len(toks4) > 0 else 0.0
            ibend = float(toks4[1]) if len(toks4) > 1 else 0.0
            itors = float(toks4[2]) if len(toks4) > 2 else 0.0
            fmax = float(toks4[3]) if len(toks4) > 3 else 0.0
            mmax = float(toks4[4]) if len(toks4) > 4 else 0.0

        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            shear_area = float(toks5[0]) if len(toks5) > 0 else 0.0
            rfac = float(toks5[1]) if len(toks5) > 1 else 0.0

    m114 = MatLaw114(
        id=mat_id, rho=rho, lmin=lmin, stiff1=stiff1, damp1=damp1,
        fun_l=fun_l, fun_ul=fun_ul, xcoeft1=xcoeft1, fcoeft1=fcoeft1,
        young=young, ibend=ibend, itors=itors, fmax=fmax, mmax=mmax,
        shear_area=shear_area, rfac=rfac, title=title,
    )
    model.mat_law114s[mat_id] = m114
    model.mat_spr_seatbelts[mat_id] = m114
    from ..mat_reader import GenericMaterialRecord
    params_114 = {
        "MAT_RHO": rho, "rho": rho, "rho0": rho,
        "LMIN": lmin, "lmin": lmin,
        "STIFF1": stiff1, "k": stiff1, "stiff1": stiff1,
        "DAMP1": damp1, "c": damp1, "damp1": damp1,
        "FUN_L": fun_l, "fun_l": fun_l,
        "FUN_UL": fun_ul, "fun_ul": fun_ul,
        "E": young, "e": young, "young": young,
        "xscale": xcoeft1, "Xscale": xcoeft1, "xcoeft1": xcoeft1,
        "fscale": fcoeft1, "Fscale": fcoeft1, "fcoeft1": fcoeft1,
        "i": ibend, "j": itors, "fmax": fmax, "mmax": mmax,
        "as": shear_area, "as_": shear_area, "r": rfac,
    }
    mat114 = Material(
        id=mat_id, law=114, rho0=rho, title=title,
        params=params_114
    )
    mat114.record = GenericMaterialRecord(
        law_name="LAW114", law_number=114, id=mat_id, title=title,
        params=mat114.params, density=rho, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat114




def read_mat_law117_m182(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW117/id`` or ``/MAT/COH_TAB/id`` (M182): Tabulated cohesive zone material."""
    from ...model.entities import MatLaw117
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/MAT/LAW117/{mat_id}: missing data cards", block.source)
        return

    rho, refer_rho = 0.0, 0.0
    en, es = 0.0, 0.0
    imass, idel, irupt = 0, 0, 0
    fct_tn, fct_tt = 0, 0
    tn, ts, fscale_x = 0.0, 0.0, 1.0
    gic, giic, exp_g, exp_bk, gamma = 0.0, 0.0, 0.0, 0.0, 0.0

    if block.fixed:
        f1 = valid_cards[0].cut("MAT_LAW117_1")
        rho = _fval(f1[0]) if len(f1) > 0 else 0.0
        refer_rho = _fval(f1[1]) if len(f1) > 1 and f1[1].strip() else rho

        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("MAT_LAW117_2")
            en = _fval(f2[0]) if len(f2) > 0 else 0.0
            es = _fval(f2[1]) if len(f2) > 1 else 0.0
            imass = _ival(f2[2]) if len(f2) > 2 else 0
            idel = _ival(f2[3]) if len(f2) > 3 else 0
            irupt = _ival(f2[4]) if len(f2) > 4 else 0

        if len(valid_cards) > 2:
            f3 = valid_cards[2].cut("MAT_LAW117_3")
            fct_tn = _ival(f3[0]) if len(f3) > 0 else 0
            fct_tt = _ival(f3[1]) if len(f3) > 1 else 0
            tn = _fval(f3[2]) if len(f3) > 2 else 0.0
            ts = _fval(f3[3]) if len(f3) > 3 else 0.0
            fscale_x = _fval(f3[4], 1.0) if len(f3) > 4 and f3[4].strip() else 1.0

        if len(valid_cards) > 3:
            f4 = valid_cards[3].cut("MAT_LAW117_4")
            gic = _fval(f4[0]) if len(f4) > 0 else 0.0
            giic = _fval(f4[1]) if len(f4) > 1 else 0.0
            exp_g = _fval(f4[2]) if len(f4) > 2 else 0.0
            exp_bk = _fval(f4[3]) if len(f4) > 3 else 0.0
            gamma = _fval(f4[4]) if len(f4) > 4 else 0.0
    else:
        toks1 = valid_cards[0].tokens()
        rho = float(toks1[0]) if len(toks1) > 0 else 0.0
        refer_rho = float(toks1[1]) if len(toks1) > 1 else rho

        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            en = float(toks2[0]) if len(toks2) > 0 else 0.0
            es = float(toks2[1]) if len(toks2) > 1 else 0.0
            imass = int(float(toks2[2])) if len(toks2) > 2 else 0
            idel = int(float(toks2[3])) if len(toks2) > 3 else 0
            irupt = int(float(toks2[4])) if len(toks2) > 4 else 0

        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            fct_tn = int(float(toks3[0])) if len(toks3) > 0 else 0
            fct_tt = int(float(toks3[1])) if len(toks3) > 1 else 0
            tn = float(toks3[2]) if len(toks3) > 2 else 0.0
            ts = float(toks3[3]) if len(toks3) > 3 else 0.0
            fscale_x = float(toks3[4]) if len(toks3) > 4 else 1.0

        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            gic = float(toks4[0]) if len(toks4) > 0 else 0.0
            giic = float(toks4[1]) if len(toks4) > 1 else 0.0
            exp_g = float(toks4[2]) if len(toks4) > 2 else 0.0
            exp_bk = float(toks4[3]) if len(toks4) > 3 else 0.0
            gamma = float(toks4[4]) if len(toks4) > 4 else 0.0

    m117 = MatLaw117(
        id=mat_id, rho=rho, refer_rho=refer_rho, en=en, es=es, imass=imass, idel=idel, irupt=irupt,
        fct_tn=fct_tn, fct_tt=fct_tt, tn=tn, ts=ts, fscale_x=fscale_x,
        gic=gic, giic=giic, exp_g=exp_g, exp_bk=exp_bk, gamma=gamma, title=title,
    )
    model.mat_law117s[mat_id] = m117
    from ..mat_reader import GenericMaterialRecord
    params_117 = {
        "MAT_RHO": rho, "rho": rho, "rho0": rho, "refer_rho": refer_rho,
        "EN": en, "en": en, "e_elas_n": en, "E_elas_n": en, "E": en, "e": en,
        "ES": es, "es": es, "e_elas_s": es, "E_elas_s": es,
        "TN": tn, "tn": tn, "TS": ts, "ts": ts, "tmax_n": tn, "tmax_s": ts,
        "GIC": gic, "gic": gic, "GIIC": giic, "giic": giic,
        "imass": imass, "idel": idel, "irupt": irupt,
        "fct_tn": fct_tn, "fct_tt": fct_tt, "fscale_x": fscale_x,
        "exp_g": exp_g, "exp_bk": exp_bk, "gamma": gamma,
    }
    mat117 = Material(
        id=mat_id, law=117, rho0=rho, title=title,
        params=params_117
    )
    mat117.record = GenericMaterialRecord(
        law_name="LAW117", law_number=117, id=mat_id, title=title,
        params=mat117.params, density=rho, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat117




def read_mat_law119(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW119/id`` or ``/MAT/SH_SEATBELT/id`` (M182): 2D shell seatbelt material."""
    from ...model.entities import MatLaw119
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/MAT/LAW119/{mat_id}: missing data cards", block.source)
        return

    rho, lmin = 0.0, 0.0
    stiff1, damp1, re = 0.0, 0.0, 0.0
    fun_l, fun_ul = 0, 0
    fcoeft1, fcoeft2 = 1.0, 1.0
    ireload = 0
    e22, nu12, g12, fcoeft22 = 0.0, 0.0, 0.0, 1.0
    ecoat, nucoat, tcoat = 0.0, 0.0, 0.0

    if block.fixed:
        f1 = valid_cards[0].cut("MAT_LAW119_1")
        rho = _fval(f1[0]) if len(f1) > 0 else 0.0
        lmin = _fval(f1[1]) if len(f1) > 1 else 0.0

        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("MAT_LAW119_2")
            stiff1 = _fval(f2[0]) if len(f2) > 0 else 0.0
            damp1 = _fval(f2[1]) if len(f2) > 1 else 0.0
            re = _fval(f2[2]) if len(f2) > 2 else 0.0

        if len(valid_cards) > 2:
            f3 = valid_cards[2].cut("MAT_LAW119_3")
            fun_l = _ival(f3[0]) if len(f3) > 0 else 0
            fun_ul = _ival(f3[1]) if len(f3) > 1 else 0
            fcoeft1 = _fval(f3[2], 1.0) if len(f3) > 2 and f3[2].strip() else 1.0
            fcoeft2 = _fval(f3[3], 1.0) if len(f3) > 3 and f3[3].strip() else 1.0
            ireload = _ival(f3[4]) if len(f3) > 4 else 0

        if len(valid_cards) > 3:
            f4 = valid_cards[3].cut("MAT_LAW119_4")
            e22 = _fval(f4[0]) if len(f4) > 0 else 0.0
            nu12 = _fval(f4[1]) if len(f4) > 1 else 0.0
            g12 = _fval(f4[2]) if len(f4) > 2 else 0.0
            fcoeft22 = _fval(f4[3], 1.0) if len(f4) > 3 and f4[3].strip() else 1.0

        if len(valid_cards) > 4:
            f5 = valid_cards[4].cut("MAT_LAW119_5")
            ecoat = _fval(f5[0]) if len(f5) > 0 else 0.0
            nucoat = _fval(f5[1]) if len(f5) > 1 else 0.0
            tcoat = _fval(f5[2]) if len(f5) > 2 else 0.0
    else:
        toks1 = valid_cards[0].tokens()
        rho = float(toks1[0]) if len(toks1) > 0 else 0.0
        lmin = float(toks1[1]) if len(toks1) > 1 else 0.0

        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            stiff1 = float(toks2[0]) if len(toks2) > 0 else 0.0
            damp1 = float(toks2[1]) if len(toks2) > 1 else 0.0
            re = float(toks2[2]) if len(toks2) > 2 else 0.0

        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            fun_l = int(float(toks3[0])) if len(toks3) > 0 else 0
            fun_ul = int(float(toks3[1])) if len(toks3) > 1 else 0
            fcoeft1 = float(toks3[2]) if len(toks3) > 2 else 1.0
            fcoeft2 = float(toks3[3]) if len(toks3) > 3 else 1.0
            ireload = int(float(toks3[4])) if len(toks3) > 4 else 0

        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            e22 = float(toks4[0]) if len(toks4) > 0 else 0.0
            nu12 = float(toks4[1]) if len(toks4) > 1 else 0.0
            g12 = float(toks4[2]) if len(toks4) > 2 else 0.0
            fcoeft22 = float(toks4[3]) if len(toks4) > 3 else 1.0

        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            ecoat = float(toks5[0]) if len(toks5) > 0 else 0.0
            nucoat = float(toks5[1]) if len(toks5) > 1 else 0.0
            tcoat = float(toks5[2]) if len(toks5) > 2 else 0.0

    m119 = MatLaw119(
        id=mat_id, rho=rho, lmin=lmin, stiff1=stiff1, damp1=damp1, re=re,
        fun_l=fun_l, fun_ul=fun_ul, fcoeft1=fcoeft1, fcoeft2=fcoeft2, ireload=ireload,
        e22=e22, nu12=nu12, g12=g12, fcoeft22=fcoeft22,
        ecoat=ecoat, nucoat=nucoat, tcoat=tcoat, title=title,
    )
    model.mat_law119s[mat_id] = m119
    model.mat_sh_seatbelts[mat_id] = m119
    from ..mat_reader import GenericMaterialRecord
    params_119 = {
        "MAT_RHO": rho, "rho": rho, "rho0": rho,
        "LMIN": lmin, "lmin": lmin,
        "STIFF1": stiff1, "k": stiff1, "stiff1": stiff1,
        "DAMP1": damp1, "c": damp1, "damp1": damp1,
        "re": re, "FUN_L": fun_l, "fun_l": fun_l, "FUN_UL": fun_ul, "fun_ul": fun_ul,
        "E22": e22, "e22": e22, "NU12": nu12, "nu12": nu12, "G12": g12, "g12": g12,
        "fscale1": fcoeft1, "fscale2": fcoeft2, "fscale22": fcoeft22,
        "ireload": ireload, "ecoat": ecoat, "nucoat": nucoat, "tcoat": tcoat,
    }
    mat119 = Material(
        id=mat_id, law=119, rho0=rho, title=title,
        params=params_119
    )
    mat119.record = GenericMaterialRecord(
        law_name="LAW119", law_number=119, id=mat_id, title=title,
        params=mat119.params, density=rho, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat119




def read_mat_law120(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW120/id`` or ``/MAT/TAPO/id`` (M182): Tabulated orthotropic Pont-Pack material."""
    from ...model.entities import MatLaw120
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/MAT/LAW120/{mat_id}: missing data cards", block.source)
        return

    rho, refer_rho = 0.0, 0.0
    e, nu = 0.0, 0.0
    iform, itrx, idam = 0, 0, 0
    thick = 0.0
    tab_id, xscale, yscale = 0, 1.0, 1.0
    tau0, q, beta, h = 0.0, 0.0, 0.0, 0.0
    af1, af2, ah1, ah2, as_ = 0.0, 0.0, 0.0, 0.0, 0.0
    cc, gam0, gamf = 0.0, 0.0, 0.0
    d1c, d2c, d1f, d2f = 0.0, 0.0, 0.0, 0.0
    dtrx, djc, exp_n = 0.0, 0.0, 0.0

    if block.fixed:
        f1 = valid_cards[0].cut("MAT_LAW120_1")
        rho = _fval(f1[0]) if len(f1) > 0 else 0.0
        refer_rho = _fval(f1[1]) if len(f1) > 1 else 0.0

        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("MAT_LAW120_2")
            e = _fval(f2[0]) if len(f2) > 0 else 0.0
            nu = _fval(f2[1]) if len(f2) > 1 else 0.0
            iform = _ival(f2[2]) if len(f2) > 2 else 0
            itrx = _ival(f2[3]) if len(f2) > 3 else 0
            idam = _ival(f2[4]) if len(f2) > 4 else 0
            thick = _fval(f2[6]) if len(f2) > 6 else 0.0

        if len(valid_cards) > 2:
            f3 = valid_cards[2].cut("MAT_LAW120_3")
            tab_id = _ival(f3[0]) if len(f3) > 0 else 0
            xscale = _fval(f3[1], 1.0) if len(f3) > 1 and f3[1].strip() else 1.0
            yscale = _fval(f3[2], 1.0) if len(f3) > 2 and f3[2].strip() else 1.0

        if len(valid_cards) > 3:
            f4 = valid_cards[3].cut("MAT_LAW120_4")
            tau0 = _fval(f4[0]) if len(f4) > 0 else 0.0
            q = _fval(f4[1]) if len(f4) > 1 else 0.0
            beta = _fval(f4[2]) if len(f4) > 2 else 0.0
            h = _fval(f4[3]) if len(f4) > 3 else 0.0

        if len(valid_cards) > 4:
            f5 = valid_cards[4].cut("MAT_LAW120_5")
            af1 = _fval(f5[0]) if len(f5) > 0 else 0.0
            af2 = _fval(f5[1]) if len(f5) > 0 else 0.0
            ah1 = _fval(f5[2]) if len(f5) > 2 else 0.0
            ah2 = _fval(f5[3]) if len(f5) > 3 else 0.0
            as_ = _fval(f5[4]) if len(f5) > 4 else 0.0

        if len(valid_cards) > 5:
            f6 = valid_cards[5].cut("MAT_LAW120_6")
            cc = _fval(f6[0]) if len(f6) > 0 else 0.0
            gam0 = _fval(f6[1]) if len(f6) > 1 else 0.0
            gamf = _fval(f6[2]) if len(f6) > 2 else 0.0

        if len(valid_cards) > 6:
            f7 = valid_cards[6].cut("MAT_LAW120_7")
            d1c = _fval(f7[0]) if len(f7) > 0 else 0.0
            d2c = _fval(f7[1]) if len(f7) > 1 else 0.0
            d1f = _fval(f7[2]) if len(f7) > 2 else 0.0
            d2f = _fval(f7[3]) if len(f7) > 3 else 0.0

        if len(valid_cards) > 7:
            f8 = valid_cards[7].cut("MAT_LAW120_8")
            dtrx = _fval(f8[0]) if len(f8) > 0 else 0.0
            djc = _fval(f8[1]) if len(f8) > 1 else 0.0
            exp_n = _fval(f8[2]) if len(f8) > 2 else 0.0
    else:
        toks1 = valid_cards[0].tokens()
        rho = float(toks1[0]) if len(toks1) > 0 else 0.0
        refer_rho = float(toks1[1]) if len(toks1) > 1 else 0.0

        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            e = float(toks2[0]) if len(toks2) > 0 else 0.0
            nu = float(toks2[1]) if len(toks2) > 1 else 0.0
            iform = int(float(toks2[2])) if len(toks2) > 2 else 0
            itrx = int(float(toks2[3])) if len(toks2) > 3 else 0
            idam = int(float(toks2[4])) if len(toks2) > 4 else 0
            thick = float(toks2[5]) if len(toks2) > 5 else 0.0

        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            tab_id = int(float(toks3[0])) if len(toks3) > 0 else 0
            xscale = float(toks3[1]) if len(toks3) > 1 else 1.0
            yscale = float(toks3[2]) if len(toks3) > 2 else 1.0

        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            tau0 = float(toks4[0]) if len(toks4) > 0 else 0.0
            q = float(toks4[1]) if len(toks4) > 1 else 0.0
            beta = float(toks4[2]) if len(toks4) > 2 else 0.0
            h = float(toks4[3]) if len(toks4) > 3 else 0.0

        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            af1 = float(toks5[0]) if len(toks5) > 0 else 0.0
            af2 = float(toks5[1]) if len(toks5) > 0 else 0.0
            ah1 = float(toks5[2]) if len(toks5) > 2 else 0.0
            ah2 = float(toks5[3]) if len(toks5) > 3 else 0.0
            as_ = float(toks5[4]) if len(toks5) > 4 else 0.0

        if len(valid_cards) > 5:
            toks6 = valid_cards[5].tokens()
            cc = float(toks6[0]) if len(toks6) > 0 else 0.0
            gam0 = float(toks6[1]) if len(toks6) > 1 else 0.0
            gamf = float(toks6[2]) if len(toks6) > 2 else 0.0

        if len(valid_cards) > 6:
            toks7 = valid_cards[6].tokens()
            d1c = float(toks7[0]) if len(toks7) > 0 else 0.0
            d2c = float(toks7[1]) if len(toks7) > 1 else 0.0
            d1f = float(toks7[2]) if len(toks7) > 2 else 0.0
            d2f = float(toks7[3]) if len(toks7) > 3 else 0.0

        if len(valid_cards) > 7:
            toks8 = valid_cards[7].tokens()
            dtrx = float(toks8[0]) if len(toks8) > 0 else 0.0
            djc = float(toks8[1]) if len(toks8) > 1 else 0.0
            exp_n = float(toks8[2]) if len(toks8) > 2 else 0.0

    m120 = MatLaw120(
        id=mat_id, rho=rho, refer_rho=refer_rho, e=e, nu=nu,
        iform=iform, itrx=itrx, idam=idam, thick=thick, tab_id=tab_id,
        xscale=xscale, yscale=yscale, tau0=tau0, q=q, beta=beta, h=h,
        af1=af1, af2=af2, ah1=ah1, ah2=ah2, as_=as_, cc=cc,
        gam0=gam0, gamf=gamf, d1c=d1c, d2c=d2c, d1f=d1f, d2f=d2f,
        dtrx=dtrx, djc=djc, exp_n=exp_n, title=title,
    )
    model.mat_law120s[mat_id] = m120
    model.mat_tapos[mat_id] = m120
    from ..mat_reader import GenericMaterialRecord
    params_120 = {
        "MAT_RHO": rho, "rho": rho, "rho0": rho, "refer_rho": refer_rho,
        "E": e, "e": e, "NU": nu, "nu": nu,
        "TAB_ID": tab_id, "tab_id": tab_id, "THICK": thick, "thick": thick,
        "TAU0": tau0, "tau0": tau0, "tau": tau0, "iform": iform, "itrx": itrx, "idam": idam,
        "xscale": xscale, "yscale": yscale, "q": q, "beta": beta, "h": h,
        "af1": af1, "af2": af2, "ah1": ah1, "ah2": ah2,
        "as": as_, "as_": as_, "cc": cc, "gam0": gam0, "gamf": gamf,
        "d1c": d1c, "d2c": d2c, "d1f": d1f, "d2f": d2f,
        "dtrx": dtrx, "djc": djc, "exp_n": exp_n,
    }
    mat120 = Material(
        id=mat_id, law=120, rho0=rho, title=title,
        params=params_120
    )
    mat120.record = GenericMaterialRecord(
        law_name="LAW120", law_number=120, id=mat_id, title=title,
        params=mat120.params, density=rho, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat120




def read_mat_law121(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW121/id`` or ``/MAT/PLAS_RATE/id`` (M182): Tabulated rate-dependent elastoplastic material."""
    from ...model.entities import MatLaw121
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/MAT/LAW121/{mat_id}: missing data cards", block.source)
        return

    rho = 0.0
    e, nu = 0.0, 0.0
    ires, ivisc = 2, 0
    fcut, dtmin = 0.0, 0.0
    fct_sig0, xscale_sig0, yscale_sig0 = 0, 1.0, 1.0
    fct_youn, xscale_youn, yscale_youn = 0, 1.0, 1.0
    fct_tang, xscale_tang, tang = 0, 1.0, 0.0
    fct_fail, ifail, xscale_fail, yscale_fail = 0, 0, 1.0, 1.0

    if block.fixed:
        f1 = valid_cards[0].cut("MAT_LAW121_1")
        rho = _fval(f1[0]) if len(f1) > 0 else 0.0

        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("MAT_LAW121_2")
            e = _fval(f2[0]) if len(f2) > 0 else 0.0
            nu = _fval(f2[1]) if len(f2) > 1 else 0.0
            ires = _ival(f2[2], 2) if len(f2) > 2 and f2[2].strip() else 2
            ivisc = _ival(f2[3]) if len(f2) > 3 else 0
            fcut = _fval(f2[4]) if len(f2) > 4 else 0.0
            dtmin = _fval(f2[5]) if len(f2) > 5 else 0.0

        if len(valid_cards) > 2:
            f3 = valid_cards[2].cut("MAT_LAW121_3")
            fct_sig0 = _ival(f3[0]) if len(f3) > 0 else 0
            xscale_sig0 = _fval(f3[2], 1.0) if len(f3) > 2 and f3[2].strip() else 1.0
            yscale_sig0 = _fval(f3[3], 1.0) if len(f3) > 3 and f3[3].strip() else 1.0

        if len(valid_cards) > 3:
            f4 = valid_cards[3].cut("MAT_LAW121_4")
            fct_youn = _ival(f4[0]) if len(f4) > 0 else 0
            xscale_youn = _fval(f4[2], 1.0) if len(f4) > 2 and f4[2].strip() else 1.0
            yscale_youn = _fval(f4[3], 1.0) if len(f4) > 3 and f4[3].strip() else 1.0

        if len(valid_cards) > 4:
            f5 = valid_cards[4].cut("MAT_LAW121_5")
            fct_tang = _ival(f5[0]) if len(f5) > 0 else 0
            xscale_tang = _fval(f5[2], 1.0) if len(f5) > 2 and f5[2].strip() else 1.0
            tang = _fval(f5[3]) if len(f5) > 3 else 0.0

        if len(valid_cards) > 5:
            f6 = valid_cards[5].cut("MAT_LAW121_6")
            fct_fail = _ival(f6[0]) if len(f6) > 0 else 0
            ifail = _ival(f6[1]) if len(f6) > 1 else 0
            xscale_fail = _fval(f6[2], 1.0) if len(f6) > 2 and f6[2].strip() else 1.0
            yscale_fail = _fval(f6[3], 1.0) if len(f6) > 3 and f6[3].strip() else 1.0
    else:
        toks1 = valid_cards[0].tokens()
        rho = float(toks1[0]) if len(toks1) > 0 else 0.0

        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            e = float(toks2[0]) if len(toks2) > 0 else 0.0
            nu = float(toks2[1]) if len(toks2) > 1 else 0.0
            ires = int(float(toks2[2])) if len(toks2) > 2 else 2
            ivisc = int(float(toks2[3])) if len(toks2) > 3 else 0
            fcut = float(toks2[4]) if len(toks2) > 4 else 0.0
            dtmin = float(toks2[5]) if len(toks2) > 5 else 0.0

        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            fct_sig0 = int(float(toks3[0])) if len(toks3) > 0 else 0
            xscale_sig0 = float(toks3[1]) if len(toks3) > 1 else 1.0
            yscale_sig0 = float(toks3[2]) if len(toks3) > 2 else 1.0

        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            fct_youn = int(float(toks4[0])) if len(toks4) > 0 else 0
            xscale_youn = float(toks4[1]) if len(toks4) > 1 else 1.0
            yscale_youn = float(toks4[2]) if len(toks4) > 2 else 1.0

        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            fct_tang = int(float(toks5[0])) if len(toks5) > 0 else 0
            xscale_tang = float(toks5[1]) if len(toks5) > 1 else 1.0
            tang = float(toks5[2]) if len(toks5) > 2 else 0.0

        if len(valid_cards) > 5:
            toks6 = valid_cards[5].tokens()
            fct_fail = int(float(toks6[0])) if len(toks6) > 0 else 0
            ifail = int(float(toks6[1])) if len(toks6) > 1 else 0
            xscale_fail = float(toks6[2]) if len(toks6) > 2 else 1.0
            yscale_fail = float(toks6[3]) if len(toks6) > 3 else 1.0

    m121 = MatLaw121(
        id=mat_id, rho=rho, e=e, nu=nu, ires=ires, ivisc=ivisc, fcut=fcut, dtmin=dtmin,
        fct_sig0=fct_sig0, xscale_sig0=xscale_sig0, yscale_sig0=yscale_sig0,
        fct_youn=fct_youn, xscale_youn=xscale_youn, yscale_youn=yscale_youn,
        fct_tang=fct_tang, xscale_tang=xscale_tang, tang=tang,
        fct_fail=fct_fail, ifail=ifail, xscale_fail=xscale_fail, yscale_fail=yscale_fail,
        title=title,
    )
    model.mat_law121s[mat_id] = m121
    model.mat_plas_rates[mat_id] = m121
    from ..mat_reader import GenericMaterialRecord
    params_121 = {
        "MAT_RHO": rho, "rho": rho, "rho0": rho,
        "E": e, "e": e, "NU": nu, "nu": nu,
        "FCT_SIG0": fct_sig0, "fct_sig0": fct_sig0,
        "TANG": tang, "tang": tang, "ires": ires, "ivisc": ivisc,
        "fcut": fcut, "dtmin": dtmin, "tdel": dtmin,
        "xscale_sig0": xscale_sig0, "yscale_sig0": yscale_sig0,
        "fct_youn": fct_youn, "xscale_youn": xscale_youn, "yscale_youn": yscale_youn,
        "fct_tang": fct_tang, "xscale_tang": xscale_tang,
        "fct_fail": fct_fail, "ifail": ifail, "xscale_fail": xscale_fail, "yscale_fail": yscale_fail,
    }
    mat121 = Material(
        id=mat_id, law=121, rho0=rho, title=title,
        params=params_121
    )
    mat121.record = GenericMaterialRecord(
        law_name="LAW121", law_number=121, id=mat_id, title=title,
        params=mat121.params, density=rho, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat121




# ============================================================================
# M183 Materials: LAW50, LAW57, LAW87, LAW95, LAW163, LAW169
# ============================================================================

def read_mat_law50(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW50/id`` or ``/MAT/VISC_HONEY/id`` or ``/MAT/HYP_FOAM/id`` (M183, M559): Rate-dependent honeycomb."""
    from ...model.entities import MatLaw50, Material
    mat_id = block.user_id or 0

    def _card_tokens(c: Any) -> list[str]:
        raw = getattr(c, "raw", str(c)).strip()
        for ch in ("#", "$"):
            if ch in raw:
                raw = raw.split(ch)[0].strip()
        if not raw:
            return []
        if "," in raw:
            parts = [p.strip() for p in raw.split(",")]
            while parts and parts[-1] == "":
                parts.pop()
            return parts
        return raw.split()

    def _fval_safe(v: Any, default: float = 0.0) -> float:
        if v is None:
            return default
        s = str(v).strip().replace("D", "E").replace("d", "e")
        if not s:
            return default
        try:
            return float(s)
        except (ValueError, TypeError):
            return default

    def _ival_safe(v: Any, default: int = 0) -> int:
        if v is None:
            return default
        s = str(v).strip()
        if not s:
            return default
        try:
            return int(float(s))
        except (ValueError, TypeError):
            return default

    has_comma = any("," in getattr(c, "raw", str(c)) for c in block.cards)
    is_fixed = block.fixed and not has_comma
    if is_fixed:
        for c in block.cards:
            c_raw = getattr(c, "raw", str(c))
            raw_s = c_raw.strip()
            if not raw_s or raw_s.startswith(("#", "$")):
                continue
            if "\t" in c_raw:
                is_fixed = False
                break
            toks = raw_s.split()
            if len(toks) > 1 and len(c_raw[:20].split()) > 1:
                is_fixed = False
                break
            if len(toks) > 1 and len(c_raw.rstrip()) <= 20:
                is_fixed = False
                break

    title, cards = _fixed_data(block) if is_fixed else _title_and_data(block)
    if is_fixed:
        valid_cards = [c for c in cards if not getattr(c, "raw", str(c)).strip().startswith(("#", "$"))]
    else:
        valid_cards = [c for c in cards if not getattr(c, "is_blank", False) and not getattr(c, "raw", str(c)).strip().startswith(("#", "$"))]

    if not valid_cards:
        log.error(f"/MAT/LAW50/{mat_id}: missing data cards", block.source)
        return

    rho, refer_rho = 0.0, 0.0
    ea, eb, ec = 0.0, 0.0, 0.0
    gab, gbc, gca = 0.0, 0.0, 0.0
    asrate = 0.0
    irate = 2
    gflag = 0
    eps_max11, eps_max22, eps_max33 = 1.0e30, 1.0e30, 1.0e30
    yfun11, sfac11, eps11 = [], [1.0, 1.0, 1.0, 1.0, 1.0], []
    yfun22, sfac22, eps22 = [], [1.0, 1.0, 1.0, 1.0, 1.0], []
    yfun33, sfac33, eps33 = [], [1.0, 1.0, 1.0, 1.0, 1.0], []
    vflag = 0
    eps_max12, eps_max23, eps_max31 = 1.0e30, 1.0e30, 1.0e30
    yfun12, sfac12, eps12 = [], [1.0, 1.0, 1.0, 1.0, 1.0], []
    yfun23, sfac23, eps23 = [], [1.0, 1.0, 1.0, 1.0, 1.0], []
    yfun31, sfac31, eps31 = [], [1.0, 1.0, 1.0, 1.0, 1.0], []
    ecomp, pr, sigy, et, vcomp = 0.0, 0.0, 0.0, 0.0, 0.0

    def _pad_sfac(sfac_list: list[float]) -> list[float]:
        res = list(sfac_list)
        while len(res) < 5:
            res.append(1.0)
        return res[:5]

    if is_fixed:
        f1 = valid_cards[0].cut("MAT_LAW50_1")
        rho = _fval_safe(f1[0]) if len(f1) > 0 else 0.0
        refer_rho = _fval_safe(f1[1], rho) if len(f1) > 1 and f1[1].strip() else rho
        if refer_rho == 0.0:
            refer_rho = rho

        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("MAT_LAW50_2")
            ea = _fval_safe(f2[0]) if len(f2) > 0 else 0.0
            eb = _fval_safe(f2[1]) if len(f2) > 1 else 0.0
            ec = _fval_safe(f2[2]) if len(f2) > 2 else 0.0

        if len(valid_cards) > 2:
            f3 = valid_cards[2].cut("MAT_LAW50_3")
            gab = _fval_safe(f3[0]) if len(f3) > 0 else 0.0
            gbc = _fval_safe(f3[1]) if len(f3) > 1 else 0.0
            gca = _fval_safe(f3[2]) if len(f3) > 2 else 0.0

        if len(valid_cards) > 3:
            f4 = valid_cards[3].cut("MAT_LAW50_4")
            asrate = _fval_safe(f4[0]) if len(f4) > 0 else 0.0
            irate = _ival_safe(f4[1], default=2) if len(f4) > 1 and f4[1].strip() else 2
            if irate == 0:
                irate = 2

        if len(valid_cards) > 4:
            f5 = valid_cards[4].cut("MAT_LAW50_5")
            gflag = _ival_safe(f5[0]) if len(f5) > 0 else 0
            eps_max11 = _fval_safe(f5[1], 1.0e30) if len(f5) > 1 and f5[1].strip() else 1.0e30
            eps_max22 = _fval_safe(f5[2], 1.0e30) if len(f5) > 2 and f5[2].strip() else 1.0e30
            eps_max33 = _fval_safe(f5[3], 1.0e30) if len(f5) > 3 and f5[3].strip() else 1.0e30
            if eps_max11 <= 0.0: eps_max11 = 1.0e30
            if eps_max22 <= 0.0: eps_max22 = 1.0e30
            if eps_max33 <= 0.0: eps_max33 = 1.0e30

        if len(valid_cards) > 5:
            f6 = valid_cards[5].cut("MAT_LAW50_6")
            yfun11 = [_ival_safe(x) for x in f6 if x.strip()]
        if len(valid_cards) > 6:
            f7 = valid_cards[6].cut("MAT_LAW50_7")
            v7 = [_fval_safe(x, 1.0) for x in f7 if x.strip()]
            if v7:
                sfac11 = v7
        if len(valid_cards) > 7:
            f8 = valid_cards[7].cut("MAT_LAW50_8")
            eps11 = [_fval_safe(x) for x in f8 if x.strip()]

        if len(valid_cards) > 8:
            f9 = valid_cards[8].cut("MAT_LAW50_9")
            yfun22 = [_ival_safe(x) for x in f9 if x.strip()]
        if len(valid_cards) > 9:
            f10 = valid_cards[9].cut("MAT_LAW50_10")
            v10 = [_fval_safe(x, 1.0) for x in f10 if x.strip()]
            if v10:
                sfac22 = v10
        if len(valid_cards) > 10:
            f11 = valid_cards[10].cut("MAT_LAW50_11")
            eps22 = [_fval_safe(x) for x in f11 if x.strip()]

        if len(valid_cards) > 11:
            f12 = valid_cards[11].cut("MAT_LAW50_12")
            yfun33 = [_ival_safe(x) for x in f12 if x.strip()]
        if len(valid_cards) > 12:
            f13 = valid_cards[12].cut("MAT_LAW50_13")
            v13 = [_fval_safe(x, 1.0) for x in f13 if x.strip()]
            if v13:
                sfac33 = v13
        if len(valid_cards) > 13:
            f14 = valid_cards[13].cut("MAT_LAW50_14")
            eps33 = [_fval_safe(x) for x in f14 if x.strip()]

        if len(valid_cards) > 14:
            f15 = valid_cards[14].cut("MAT_LAW50_15")
            vflag = _ival_safe(f15[0]) if len(f15) > 0 else 0
            eps_max12 = _fval_safe(f15[1], 1.0e30) if len(f15) > 1 and f15[1].strip() else 1.0e30
            eps_max23 = _fval_safe(f15[2], 1.0e30) if len(f15) > 2 and f15[2].strip() else 1.0e30
            eps_max31 = _fval_safe(f15[3], 1.0e30) if len(f15) > 3 and f15[3].strip() else 1.0e30
            if eps_max12 <= 0.0: eps_max12 = 1.0e30
            if eps_max23 <= 0.0: eps_max23 = 1.0e30
            if eps_max31 <= 0.0: eps_max31 = 1.0e30

        if len(valid_cards) > 15:
            f16 = valid_cards[15].cut("MAT_LAW50_16")
            yfun12 = [_ival_safe(x) for x in f16 if x.strip()]
        if len(valid_cards) > 16:
            f17 = valid_cards[16].cut("MAT_LAW50_17")
            v17 = [_fval_safe(x, 1.0) for x in f17 if x.strip()]
            if v17:
                sfac12 = v17
        if len(valid_cards) > 17:
            f18 = valid_cards[17].cut("MAT_LAW50_18")
            eps12 = [_fval_safe(x) for x in f18 if x.strip()]

        if len(valid_cards) > 18:
            f19 = valid_cards[18].cut("MAT_LAW50_19")
            yfun23 = [_ival_safe(x) for x in f19 if x.strip()]
        if len(valid_cards) > 19:
            f20 = valid_cards[19].cut("MAT_LAW50_20")
            v20 = [_fval_safe(x, 1.0) for x in f20 if x.strip()]
            if v20:
                sfac23 = v20
        if len(valid_cards) > 20:
            f21 = valid_cards[20].cut("MAT_LAW50_21")
            eps23 = [_fval_safe(x) for x in f21 if x.strip()]

        if len(valid_cards) > 21:
            f22 = valid_cards[21].cut("MAT_LAW50_22")
            yfun31 = [_ival_safe(x) for x in f22 if x.strip()]
        if len(valid_cards) > 22:
            f23 = valid_cards[22].cut("MAT_LAW50_23")
            v23 = [_fval_safe(x, 1.0) for x in f23 if x.strip()]
            if v23:
                sfac31 = v23
        if len(valid_cards) > 23:
            f24 = valid_cards[23].cut("MAT_LAW50_24")
            eps31 = [_fval_safe(x) for x in f24 if x.strip()]

        if len(valid_cards) > 24:
            f25 = valid_cards[24].cut("MAT_LAW50_25")
            ecomp = _fval_safe(f25[0]) if len(f25) > 0 else 0.0
            pr = _fval_safe(f25[1]) if len(f25) > 1 else 0.0
            sigy = _fval_safe(f25[2]) if len(f25) > 2 else 0.0
            et = _fval_safe(f25[3]) if len(f25) > 3 else 0.0
            vcomp = _fval_safe(f25[4]) if len(f25) > 4 else 0.0
        elif len(valid_cards) == 5:
            f5_comp = valid_cards[4].cut("MAT_LAW50_25")
            if len(f5_comp) >= 5 and any(_fval_safe(x) != 0.0 for x in f5_comp):
                ecomp = _fval_safe(f5_comp[0]) if len(f5_comp) > 0 else 0.0
                pr = _fval_safe(f5_comp[1]) if len(f5_comp) > 1 else 0.0
                sigy = _fval_safe(f5_comp[2]) if len(f5_comp) > 2 else 0.0
                et = _fval_safe(f5_comp[3]) if len(f5_comp) > 3 else 0.0
                vcomp = _fval_safe(f5_comp[4]) if len(f5_comp) > 4 else 0.0
    else:
        toks1 = _card_tokens(valid_cards[0])
        rho = _fval_safe(toks1[0]) if len(toks1) > 0 else 0.0
        refer_rho = _fval_safe(toks1[1], rho) if len(toks1) > 1 and toks1[1].strip() else rho
        if refer_rho == 0.0:
            refer_rho = rho

        if len(valid_cards) > 1:
            toks2 = _card_tokens(valid_cards[1])
            ea = _fval_safe(toks2[0]) if len(toks2) > 0 else 0.0
            eb = _fval_safe(toks2[1]) if len(toks2) > 1 else 0.0
            ec = _fval_safe(toks2[2]) if len(toks2) > 2 else 0.0

        if len(valid_cards) > 2:
            toks3 = _card_tokens(valid_cards[2])
            gab = _fval_safe(toks3[0]) if len(toks3) > 0 else 0.0
            gbc = _fval_safe(toks3[1]) if len(toks3) > 1 else 0.0
            gca = _fval_safe(toks3[2]) if len(toks3) > 2 else 0.0

        if len(valid_cards) > 3:
            toks4 = _card_tokens(valid_cards[3])
            asrate = _fval_safe(toks4[0]) if len(toks4) > 0 else 0.0
            irate = _ival_safe(toks4[1], default=2) if len(toks4) > 1 and toks4[1].strip() else 2
            if irate == 0:
                irate = 2

        if len(valid_cards) > 4:
            toks5 = _card_tokens(valid_cards[4])
            gflag = _ival_safe(toks5[0]) if len(toks5) > 0 else 0
            eps_max11 = _fval_safe(toks5[1], 1.0e30) if len(toks5) > 1 and toks5[1].strip() else 1.0e30
            eps_max22 = _fval_safe(toks5[2], 1.0e30) if len(toks5) > 2 and toks5[2].strip() else 1.0e30
            eps_max33 = _fval_safe(toks5[3], 1.0e30) if len(toks5) > 3 and toks5[3].strip() else 1.0e30
            if eps_max11 <= 0.0: eps_max11 = 1.0e30
            if eps_max22 <= 0.0: eps_max22 = 1.0e30
            if eps_max33 <= 0.0: eps_max33 = 1.0e30

        if len(valid_cards) > 5:
            yfun11 = [_ival_safe(x) for x in _card_tokens(valid_cards[5])]
        if len(valid_cards) > 6:
            v6 = [_fval_safe(x, 1.0) for x in _card_tokens(valid_cards[6])]
            if v6:
                sfac11 = v6
        if len(valid_cards) > 7:
            eps11 = [_fval_safe(x) for x in _card_tokens(valid_cards[7])]

        if len(valid_cards) > 8:
            yfun22 = [_ival_safe(x) for x in _card_tokens(valid_cards[8])]
        if len(valid_cards) > 9:
            v9 = [_fval_safe(x, 1.0) for x in _card_tokens(valid_cards[9])]
            if v9:
                sfac22 = v9
        if len(valid_cards) > 10:
            eps22 = [_fval_safe(x) for x in _card_tokens(valid_cards[10])]

        if len(valid_cards) > 11:
            yfun33 = [_ival_safe(x) for x in _card_tokens(valid_cards[11])]
        if len(valid_cards) > 12:
            v12 = [_fval_safe(x, 1.0) for x in _card_tokens(valid_cards[12])]
            if v12:
                sfac33 = v12
        if len(valid_cards) > 13:
            eps33 = [_fval_safe(x) for x in _card_tokens(valid_cards[13])]

        if len(valid_cards) > 14:
            toks15 = _card_tokens(valid_cards[14])
            vflag = _ival_safe(toks15[0]) if len(toks15) > 0 else 0
            eps_max12 = _fval_safe(toks15[1], 1.0e30) if len(toks15) > 1 and toks15[1].strip() else 1.0e30
            eps_max23 = _fval_safe(toks15[2], 1.0e30) if len(toks15) > 2 and toks15[2].strip() else 1.0e30
            eps_max31 = _fval_safe(toks15[3], 1.0e30) if len(toks15) > 3 and toks15[3].strip() else 1.0e30
            if eps_max12 <= 0.0: eps_max12 = 1.0e30
            if eps_max23 <= 0.0: eps_max23 = 1.0e30
            if eps_max31 <= 0.0: eps_max31 = 1.0e30

        if len(valid_cards) > 15:
            yfun12 = [_ival_safe(x) for x in _card_tokens(valid_cards[15])]
        if len(valid_cards) > 16:
            v16 = [_fval_safe(x, 1.0) for x in _card_tokens(valid_cards[16])]
            if v16:
                sfac12 = v16
        if len(valid_cards) > 17:
            eps12 = [_fval_safe(x) for x in _card_tokens(valid_cards[17])]

        if len(valid_cards) > 18:
            yfun23 = [_ival_safe(x) for x in _card_tokens(valid_cards[18])]
        if len(valid_cards) > 19:
            v19 = [_fval_safe(x, 1.0) for x in _card_tokens(valid_cards[19])]
            if v19:
                sfac23 = v19
        if len(valid_cards) > 20:
            eps23 = [_fval_safe(x) for x in _card_tokens(valid_cards[20])]

        if len(valid_cards) > 21:
            yfun31 = [_ival_safe(x) for x in _card_tokens(valid_cards[21])]
        if len(valid_cards) > 22:
            v22 = [_fval_safe(x, 1.0) for x in _card_tokens(valid_cards[22])]
            if v22:
                sfac31 = v22
        if len(valid_cards) > 23:
            eps31 = [_fval_safe(x) for x in _card_tokens(valid_cards[23])]

        if len(valid_cards) > 24:
            toks25 = _card_tokens(valid_cards[24])
            ecomp = _fval_safe(toks25[0]) if len(toks25) > 0 else 0.0
            pr = _fval_safe(toks25[1]) if len(toks25) > 1 else 0.0
            sigy = _fval_safe(toks25[2]) if len(toks25) > 2 else 0.0
            et = _fval_safe(toks25[3]) if len(toks25) > 3 else 0.0
            vcomp = _fval_safe(toks25[4]) if len(toks25) > 4 else 0.0
        elif len(valid_cards) == 5:
            toks5_comp = _card_tokens(valid_cards[4])
            if len(toks5_comp) >= 5:
                ecomp = _fval_safe(toks5_comp[0]) if len(toks5_comp) > 0 else 0.0
                pr = _fval_safe(toks5_comp[1]) if len(toks5_comp) > 1 else 0.0
                sigy = _fval_safe(toks5_comp[2]) if len(toks5_comp) > 2 else 0.0
                et = _fval_safe(toks5_comp[3]) if len(toks5_comp) > 3 else 0.0
                vcomp = _fval_safe(toks5_comp[4]) if len(toks5_comp) > 4 else 0.0

    icompact = 1 if (ecomp * sigy * vcomp > 0.0) else 0
    nu_eff = min(pr, 0.495)
    gcomp = ecomp / (1.0 + nu_eff) if (icompact == 1 and ecomp > 0.0) else 0.0
    bulk = ecomp / (3.0 * (1.0 - 2.0 * nu_eff)) if (icompact == 1 and ecomp > 0.0) else max(ea, eb, ec, gab, gbc, gca)

    raw_law = block.parts[1].upper() if len(block.parts) > 1 else "LAW50"
    if raw_law in ("50", "LAW50", "MAT_LAW50"):
        law_name = "LAW50"
    elif raw_law in ("VISC_HONEY", "MAT_VISC_HONEY", "LAW50_VISC_HONEY"):
        law_name = "VISC_HONEY"
    elif raw_law in ("HYP_FOAM", "MAT_HYP_FOAM", "LAW50_HYP_FOAM"):
        law_name = "HYP_FOAM"
    else:
        law_name = raw_law

    unit_id = getattr(block, "unit_id", None)

    params = {
        "rho": rho, "rho0": rho, "refer_rho": refer_rho, "rhor": refer_rho,
        "ea": ea, "eb": eb, "ec": ec, "e11": ea, "e22": eb, "e33": ec,
        "E": max(ea, eb, ec), "E11": ea, "E22": eb, "E33": ec,
        "gab": gab, "gbc": gbc, "gca": gca, "g12": gab, "g23": gbc, "g31": gca,
        "G": max(gab, gbc, gca), "G12": gab, "G23": gbc, "G31": gca,
        "asrate": asrate, "fcut": asrate, "irate": irate, "gflag": gflag, "vflag": vflag,
        "eps_max11": eps_max11, "eps_max22": eps_max22, "eps_max33": eps_max33,
        "eps_max12": eps_max12, "eps_max23": eps_max23, "eps_max31": eps_max31,
        "yfun11": yfun11, "sfac11": sfac11, "eps11": eps11,
        "yfun22": yfun22, "sfac22": sfac22, "eps22": eps22,
        "yfun33": yfun33, "sfac33": sfac33, "eps33": eps33,
        "yfun12": yfun12, "sfac12": sfac12, "eps12": eps12,
        "yfun23": yfun23, "sfac23": sfac23, "eps23": eps23,
        "yfun31": yfun31, "sfac31": sfac31, "eps31": eps31,
        "ecomp": ecomp, "pr": pr, "nu": pr, "sigy": sigy, "et": et, "hcomp": et, "vcomp": vcomp,
        "icompact": icompact, "icomp": icompact, "gcomp": gcomp, "bulk": bulk, "K": bulk,
    }

    m50 = MatLaw50(
        id=mat_id, rho=rho, refer_rho=refer_rho, ea=ea, eb=eb, ec=ec,
        gab=gab, gbc=gbc, gca=gca, asrate=asrate, irate=irate, gflag=gflag,
        eps_max11=eps_max11, eps_max22=eps_max22, eps_max33=eps_max33,
        yfun11=yfun11, sfac11=sfac11, eps11=eps11,
        yfun22=yfun22, sfac22=sfac22, eps22=eps22,
        yfun33=yfun33, sfac33=sfac33, eps33=eps33,
        vflag=vflag, eps_max12=eps_max12, eps_max23=eps_max23, eps_max31=eps_max31,
        yfun12=yfun12, sfac12=sfac12, eps12=eps12,
        yfun23=yfun23, sfac23=sfac23, eps23=eps23,
        yfun31=yfun31, sfac31=sfac31, eps31=eps31,
        ecomp=ecomp, pr=pr, sigy=sigy, et=et, vcomp=vcomp,
        title=title, law=50, law_name=law_name, unit_id=unit_id, params=params,
    )
    model.mat_law50s[mat_id] = m50
    if hasattr(model, "mat_visc_honeys"):
        model.mat_visc_honeys[mat_id] = m50
    if hasattr(model, "mat_hyp_foams"):
        model.mat_hyp_foams[mat_id] = m50

    model.materials[mat_id] = Material(
        id=mat_id, law=50, rho0=rho, title=title,
        law_name=law_name,
        params=params,
    )





def read_mat_law57(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW57/id`` or ``/MAT/BARLAT3/id`` (M183, M555): Barlat 3-parameter anisotropic plasticity.

    Cites hm_read_mat57.F90 and matl57_BARLAT3.cfg:
      Card 1: RHO, Refer_Rho (%20lg%20lg)
      Card 2: E, NU (%20lg%20lg)
      Card 3: FUNCT_IDE, EINF, CE (%10d          %20lg%20lg)
      Card 4: r00, r45, r90, C_hard, m (%20lg%20lg%20lg%20lg%20lg)
      Card 5: EPSP_max, EPS_t1, EPS_t2, Fcut, Fsmooth, VP (%20lg%20lg%20lg%20lg%10d%10d)
      Curves: funct_ID, Fscale_i, EPS_i (%10d          %20lg%20lg)

    Also supports legacy 4-card format (radioss90 / M183):
      Card 1: RHO, Refer_Rho
      Card 2: E, NU
      Card 3: r00, r45, r90, C_hard, m
      Card 4: EPSP_max, EPS_t1, EPS_t2
      Curves: funct_ID, Fscale_i, EPS_i
    """
    from ...model.entities import MatLaw57, MatLaw57Curve, Material
    from ..card_layouts import split_fixed
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/MAT/LAW57/{mat_id}: missing data cards", block.source)
        return

    rho = 0.0
    refer_rho = 0.0
    e = 0.0
    nu = 0.0
    ifunce = 0
    einf = 0.0
    ce = 0.0
    r00 = 1.0
    r45 = 1.0
    r90 = 1.0
    chard = 0.0
    m = 6.0
    eps_max = 1.0e30
    eps_t1 = 1.0e30
    eps_t2 = 2.0e30
    fcut = 1.0e30
    fsmooth = 0
    vp = 0
    curves: List[MatLaw57Curve] = []

    def _card_tokens(c) -> List[str]:
        raw = c.raw if hasattr(c, "raw") else str(c)
        return [t.strip() for t in raw.replace(",", " ").split() if t.strip()]

    is_fixed = getattr(block, "fixed", False)
    if is_fixed and valid_cards:
        for c in valid_cards[:5]:
            c_raw = c.raw if hasattr(c, "raw") else str(c)
            if "," in c_raw:
                is_fixed = False
                break

    # Detect 5-card vs 4-card format
    is_5_card = False
    if is_fixed:
        if len(valid_cards) >= 3:
            c3_raw = valid_cards[2].raw if hasattr(valid_cards[2], "raw") else str(valid_cards[2])
            if len(c3_raw) > 60 and c3_raw[60:].strip():
                is_5_card = False
            elif len(valid_cards) >= 4:
                c4_raw = valid_cards[3].raw if hasattr(valid_cards[3], "raw") else str(valid_cards[3])
                if len(c4_raw) > 60 and c4_raw[60:].strip():
                    is_5_card = True
                elif len(valid_cards) >= 5:
                    is_5_card = True
    else:
        if len(valid_cards) >= 3:
            toks3 = _card_tokens(valid_cards[2])
            if len(toks3) >= 4:
                is_5_card = False
            elif len(valid_cards) >= 4:
                toks4 = _card_tokens(valid_cards[3])
                if len(toks4) >= 4 or len(valid_cards) >= 5:
                    is_5_card = True

    if is_fixed:
        f1 = valid_cards[0].cut("MAT_LAW57_1")
        rho = _fval(f1[0]) if len(f1) > 0 and f1[0].strip() else 0.0
        refer_rho = _fval(f1[1]) if len(f1) > 1 and f1[1].strip() else 0.0

        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("MAT_LAW57_2")
            e = _fval(f2[0]) if len(f2) > 0 and f2[0].strip() else 0.0
            nu = _fval(f2[1]) if len(f2) > 1 and f2[1].strip() else 0.0

        curve_start = 5 if is_5_card else 4

        if is_5_card:
            if len(valid_cards) > 2:
                f3 = valid_cards[2].cut("MAT_LAW57_3")
                ifunce = _ival(f3[0]) if len(f3) > 0 and f3[0].strip() else 0
                if len(f3) == 4:
                    einf = _fval(f3[2]) if f3[2].strip() else 0.0
                    ce = _fval(f3[3]) if f3[3].strip() else 0.0
                elif len(f3) == 3:
                    einf = _fval(f3[1]) if f3[1].strip() else 0.0
                    ce = _fval(f3[2]) if f3[2].strip() else 0.0
            if len(valid_cards) > 3:
                f4 = valid_cards[3].cut("MAT_LAW57_4")
                r00 = _fval(f4[0], 1.0) if len(f4) > 0 and f4[0].strip() else 1.0
                r45 = _fval(f4[1], 1.0) if len(f4) > 1 and f4[1].strip() else 1.0
                r90 = _fval(f4[2], 1.0) if len(f4) > 2 and f4[2].strip() else 1.0
                chard = _fval(f4[3]) if len(f4) > 3 and f4[3].strip() else 0.0
                m = _fval(f4[4], 6.0) if len(f4) > 4 and f4[4].strip() else 6.0
            if len(valid_cards) > 4:
                f5 = valid_cards[4].cut("MAT_LAW57_5")
                eps_max = _fval(f5[0], 1.0e30) if len(f5) > 0 and f5[0].strip() else 1.0e30
                eps_t1 = _fval(f5[1], 1.0e30) if len(f5) > 1 and f5[1].strip() else 1.0e30
                eps_t2 = _fval(f5[2], 2.0e30) if len(f5) > 2 and f5[2].strip() else 2.0e30
                fcut = _fval(f5[3], 1.0e30) if len(f5) > 3 and f5[3].strip() else 1.0e30
                fsmooth = _ival(f5[4]) if len(f5) > 4 and f5[4].strip() else 0
                vp = _ival(f5[5]) if len(f5) > 5 and f5[5].strip() else 0
        else:
            if len(valid_cards) > 2:
                f3 = split_fixed(valid_cards[2].raw, [20, 20, 20, 20, 20])
                r00 = _fval(f3[0], 1.0) if len(f3) > 0 and f3[0].strip() else 1.0
                r45 = _fval(f3[1], 1.0) if len(f3) > 1 and f3[1].strip() else 1.0
                r90 = _fval(f3[2], 1.0) if len(f3) > 2 and f3[2].strip() else 1.0
                chard = _fval(f3[3]) if len(f3) > 3 and f3[3].strip() else 0.0
                m = _fval(f3[4], 6.0) if len(f3) > 4 and f3[4].strip() else 6.0
            if len(valid_cards) > 3:
                f4 = split_fixed(valid_cards[3].raw, [20, 20, 20])
                eps_max = _fval(f4[0], 1.0e30) if len(f4) > 0 and f4[0].strip() else 1.0e30
                eps_t1 = _fval(f4[1], 1.0e30) if len(f4) > 1 and f4[1].strip() else 1.0e30
                eps_t2 = _fval(f4[2], 2.0e30) if len(f4) > 2 and f4[2].strip() else 2.0e30

        for c in valid_cards[curve_start:]:
            fc = c.cut("MAT_LAW57_CURVE")
            if fc and fc[0].strip():
                fid = _ival(fc[0])
                if len(fc) == 4:
                    if fc[1].strip():
                        fsc = _fval(fc[1], 1.0)
                        eps = _fval(fc[2]) if len(fc) > 2 and fc[2].strip() else 0.0
                    else:
                        fsc = _fval(fc[2], 1.0) if len(fc) > 2 and fc[2].strip() else 1.0
                        eps = _fval(fc[3]) if len(fc) > 3 and fc[3].strip() else 0.0
                elif len(fc) >= 3:
                    fsc = _fval(fc[1], 1.0) if fc[1].strip() else 1.0
                    eps = _fval(fc[2]) if len(fc) > 2 and fc[2].strip() else 0.0
                else:
                    fsc = 1.0
                    eps = 0.0
                curves.append(MatLaw57Curve(fct_id=fid, fscale=fsc, eps=eps))
    else:
        toks1 = _card_tokens(valid_cards[0])
        rho = float(toks1[0]) if len(toks1) > 0 else 0.0
        refer_rho = float(toks1[1]) if len(toks1) > 1 else 0.0

        if len(valid_cards) > 1:
            toks2 = _card_tokens(valid_cards[1])
            e = float(toks2[0]) if len(toks2) > 0 else 0.0
            nu = float(toks2[1]) if len(toks2) > 1 else 0.0

        curve_start = 5 if is_5_card else 4

        if is_5_card:
            if len(valid_cards) > 2:
                toks3 = _card_tokens(valid_cards[2])
                ifunce = int(float(toks3[0])) if len(toks3) > 0 else 0
                einf = float(toks3[1]) if len(toks3) > 1 else 0.0
                ce = float(toks3[2]) if len(toks3) > 2 else 0.0
            if len(valid_cards) > 3:
                toks4 = _card_tokens(valid_cards[3])
                r00 = float(toks4[0]) if len(toks4) > 0 else 1.0
                r45 = float(toks4[1]) if len(toks4) > 1 else 1.0
                r90 = float(toks4[2]) if len(toks4) > 2 else 1.0
                chard = float(toks4[3]) if len(toks4) > 3 else 0.0
                m = float(toks4[4]) if len(toks4) > 4 else 6.0
            if len(valid_cards) > 4:
                toks5 = _card_tokens(valid_cards[4])
                eps_max = float(toks5[0]) if len(toks5) > 0 else 1.0e30
                eps_t1 = float(toks5[1]) if len(toks5) > 1 else 1.0e30
                eps_t2 = float(toks5[2]) if len(toks5) > 2 else 2.0e30
                fcut = float(toks5[3]) if len(toks5) > 3 else 1.0e30
                fsmooth = int(float(toks5[4])) if len(toks5) > 4 else 0
                vp = int(float(toks5[5])) if len(toks5) > 5 else 0
        else:
            if len(valid_cards) > 2:
                toks3 = _card_tokens(valid_cards[2])
                r00 = float(toks3[0]) if len(toks3) > 0 else 1.0
                r45 = float(toks3[1]) if len(toks3) > 1 else 1.0
                r90 = float(toks3[2]) if len(toks3) > 2 else 1.0
                chard = float(toks3[3]) if len(toks3) > 3 else 0.0
                m = float(toks3[4]) if len(toks3) > 4 else 6.0
            if len(valid_cards) > 3:
                toks4 = _card_tokens(valid_cards[3])
                eps_max = float(toks4[0]) if len(toks4) > 0 else 1.0e30
                eps_t1 = float(toks4[1]) if len(toks4) > 1 else 1.0e30
                eps_t2 = float(toks4[2]) if len(toks4) > 2 else 2.0e30

        for c in valid_cards[curve_start:]:
            toksc = _card_tokens(c)
            if toksc:
                fid = int(float(toksc[0]))
                fsc = float(toksc[1]) if len(toksc) > 1 else 1.0
                eps = float(toksc[2]) if len(toksc) > 2 else 0.0
                curves.append(MatLaw57Curve(fct_id=fid, fscale=fsc, eps=eps))

    # Apply defaults per hm_read_mat57.F90
    if refer_rho == 0.0:
        refer_rho = rho
    if r00 == 0.0:
        r00 = 1.0
    if r45 == 0.0:
        r45 = 1.0
    if r90 == 0.0:
        r90 = 1.0
    if m == 0.0:
        m = 6.0
    if eps_max == 0.0:
        eps_max = 1.0e30
    if eps_t1 == 0.0:
        eps_t1 = 1.0e30
    if eps_t2 == 0.0:
        eps_t2 = 2.0e30
    if fcut <= 0.0:
        fcut = 1.0e30
    vp = min(max(vp, 0), 1)

    m57 = MatLaw57(
        id=mat_id, rho=rho, refer_rho=refer_rho, e=e, nu=nu,
        ifunce=ifunce, einf=einf, ce=ce,
        r00=r00, r45=r45, r90=r90, chard=chard, m=m,
        eps_max=eps_max, eps_t1=eps_t1, eps_t2=eps_t2,
        fcut=fcut, fsmooth=fsmooth, vp=vp,
        curves=curves, title=title,
    )
    model.mat_law57s[mat_id] = m57
    model.materials[mat_id] = Material(
        id=mat_id, law=57, rho0=rho, title=title,
        params=m57.params,
    )




def read_mat_law87(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW87/id`` or ``/MAT/BARLAT_YLD2000/id`` (M183): Barlat Yld2000 anisotropic plasticity."""
    from ...model.entities import MatLaw87, MatLaw87Curve
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/MAT/LAW87/{mat_id}: missing data cards", block.source)
        return

    rho, refer_rho = 0.0, 0.0
    e, nu = 0.0, 0.0
    iflag, vp = 0, 0
    strain1, exp1 = 0.0, 0.0
    ifit = 0
    alpha = [1.0] * 8
    sigma_00, sigma_45, sigma_90, sigma_b = 0.0, 0.0, 0.0, 0.0
    r_00, r_45, r_90, r_b = 1.0, 1.0, 1.0, 1.0
    chard = 0.0
    ikin = 0
    exp_a, fcut = 6.0, 0.0
    fsmooth, nrate = 0, 0
    curves: List[MatLaw87Curve] = []
    aswift, eps0, qvoce, beta, k0 = 0.0, 0.0, 0.0, 0.0, 0.0
    tab_id0, fscale0, epsd0 = 0, 1.0, 0.0
    tab_id45, fscale45, epsd45 = 0, 1.0, 0.0
    tab_id90, fscale90, epsd90 = 0, 1.0, 0.0

    ckh = [0.0, 0.0, 0.0, 0.0]
    akh = [0.0, 0.0, 0.0, 0.0]
    alpha_vol = 1.0
    n_hard = 0.0
    card_idx = 0
    def _card_tokens(c: Card) -> List[str]:
        raw = c.raw.strip()
        if "," in raw:
            parts = [p.strip() for p in raw.replace(",", " ").split()]
        else:
            parts = c.tokens()
        return [p.rstrip(",") for p in parts if p]

    # Card 1: RHO, RHOR
    toks1 = _card_tokens(valid_cards[0])
    rho = _fval(toks1[0]) if len(toks1) > 0 else 0.0
    refer_rho = _fval(toks1[1]) if len(toks1) > 1 else 0.0
    card_idx = 1

    # Card 2: E, Nu, Iflag, VP, c, P
    if card_idx < len(valid_cards):
        toks2 = _card_tokens(valid_cards[card_idx])
        e = _fval(toks2[0]) if len(toks2) > 0 else 0.0
        nu = _fval(toks2[1]) if len(toks2) > 1 else 0.0
        iflag = _ival(toks2[2]) if len(toks2) > 2 else 0
        vp = _ival(toks2[3]) if len(toks2) > 3 else 0
        strain1 = _fval(toks2[4]) if len(toks2) > 4 else 0.0
        exp1 = _fval(toks2[5]) if len(toks2) > 5 else 0.0
        card_idx += 1

    # Cards 3 & 4: (flag_fit, alphas or yield stresses/Lankford)
    if card_idx < len(valid_cards):
        raw3 = valid_cards[card_idx].raw
        toks3 = _card_tokens(valid_cards[card_idx])
        if len(toks3) >= 5 and toks3[0] in ("0", "1", "0.0", "1.0"):
            # Format 1: flag_fit is the first token
            ifit = _ival(toks3[0])
            v1, v2, v3, v4 = _fval(toks3[1]), _fval(toks3[2]), _fval(toks3[3]), _fval(toks3[4])
        elif len(toks3) >= 5 and toks3[4] in ("0", "1", "0.0", "1.0"):
            # Format 2: ifit is in col 81-90 (5th token)
            ifit = _ival(toks3[4])
            v1, v2, v3, v4 = _fval(toks3[0]), _fval(toks3[1]), _fval(toks3[2]), _fval(toks3[3])
        elif len(toks3) >= 4:
            ifit = 0
            v1, v2, v3, v4 = _fval(toks3[0]), _fval(toks3[1]), _fval(toks3[2]), _fval(toks3[3])
        else:
            ifit = 0
            v1 = _fval(toks3[0]) if len(toks3) > 0 else 0.0
            v2 = _fval(toks3[1]) if len(toks3) > 1 else 0.0
            v3 = _fval(toks3[2]) if len(toks3) > 2 else 0.0
            v4 = _fval(toks3[3]) if len(toks3) > 3 else 0.0
        card_idx += 1

        v5, v6, v7, v8 = 1.0, 1.0, 1.0, 1.0
        if card_idx < len(valid_cards):
            toks4 = _card_tokens(valid_cards[card_idx])
            v5 = _fval(toks4[0], 1.0) if len(toks4) > 0 else 1.0
            v6 = _fval(toks4[1], 1.0) if len(toks4) > 1 else 1.0
            v7 = _fval(toks4[2], 1.0) if len(toks4) > 2 else 1.0
            v8 = _fval(toks4[3], 1.0) if len(toks4) > 3 else 1.0
            card_idx += 1

        if ifit == 1:
            sigma_00, sigma_45, sigma_90, sigma_b = v1, v2, v3, v4
            r_00, r_45, r_90, r_b = v5, v6, v7, v8
        else:
            alpha[0], alpha[1], alpha[2], alpha[3] = v1, v2, v3, v4
            alpha[4], alpha[5], alpha[6], alpha[7] = v5, v6, v7, v8

    # Card 5 and subsequent cards
    if card_idx < len(valid_cards):
        toks5 = _card_tokens(valid_cards[card_idx])
        if len(toks5) >= 3:
            chard = _fval(toks5[0])
            ikin = _ival(toks5[1])
            exp_a = _fval(toks5[2], 6.0)
            fcut = _fval(toks5[3]) if len(toks5) > 3 else 0.0
            fsmooth = _ival(toks5[4]) if len(toks5) > 4 else 0
            card_idx += 1
            if iflag == 0:
                if card_idx < len(valid_cards):
                    t_next = _card_tokens(valid_cards[card_idx])
                    if ikin == 1 and chard > 0.0 and len(t_next) >= 4:
                        ckh[0] = _fval(t_next[0])
                        ckh[1] = _fval(t_next[1])
                        ckh[2] = _fval(t_next[2])
                        ckh[3] = _fval(t_next[3])
                        card_idx += 1
                        if card_idx < len(valid_cards):
                            t_akh = _card_tokens(valid_cards[card_idx])
                            akh[0] = _fval(t_akh[0])
                            akh[1] = _fval(t_akh[1])
                            akh[2] = _fval(t_akh[2])
                            akh[3] = _fval(t_akh[3])
                            card_idx += 1
                if card_idx < len(valid_cards):
                    t_rate = _card_tokens(valid_cards[card_idx])
                    if len(t_rate) == 1:
                        nrate = _ival(t_rate[0])
                        card_idx += 1
                while card_idx < len(valid_cards):
                    tc = _card_tokens(valid_cards[card_idx])
                    if tc and len(curves) < (nrate if nrate > 0 else 9999):
                        if len(tc) >= 4:
                            fid = _ival(tc[1])
                            ep = _fval(tc[2])
                            fsc = _fval(tc[3], 1.0)
                        elif len(tc) == 3:
                            fid = _ival(tc[0])
                            fsc = _fval(tc[1], 1.0)
                            ep = _fval(tc[2])
                        elif len(tc) == 2:
                            fid = _ival(tc[0])
                            fsc = _fval(tc[1], 1.0)
                            ep = 0.0
                        else:
                            fid = _ival(tc[0])
                            fsc = 1.0
                            ep = 0.0
                        curves.append(MatLaw87Curve(fct_id=fid, fscale=fsc, epsp=ep))
                        card_idx += 1
                    else:
                        break
            elif iflag == 1:
                if card_idx < len(valid_cards):
                    toks6 = _card_tokens(valid_cards[card_idx])
                    nrate = _ival(toks6[0]) if len(toks6) > 0 else 0
                    aswift = _fval(toks6[1]) if len(toks6) > 1 else 0.0
                    n_hard = _fval(toks6[2]) if len(toks6) > 2 else 0.0
                    alpha_vol = _fval(toks6[3], 1.0) if len(toks6) > 3 else 1.0
                    eps0 = _fval(toks6[4]) if len(toks6) > 4 else 0.0
                    card_idx += 1
                if card_idx < len(valid_cards):
                    toks7 = _card_tokens(valid_cards[card_idx])
                    qvoce = _fval(toks7[0]) if len(toks7) > 0 else 0.0
                    beta = _fval(toks7[1]) if len(toks7) > 1 else 0.0
                    k0 = _fval(toks7[2]) if len(toks7) > 2 else 0.0
                    card_idx += 1
                if card_idx < len(valid_cards) and ikin == 1 and chard > 0.0:
                    toks_k1 = _card_tokens(valid_cards[card_idx])
                    if len(toks_k1) >= 4:
                        ckh[0] = _fval(toks_k1[0])
                        ckh[1] = _fval(toks_k1[1])
                        ckh[2] = _fval(toks_k1[2])
                        ckh[3] = _fval(toks_k1[3])
                        card_idx += 1
                        if card_idx < len(valid_cards):
                            toks_k2 = _card_tokens(valid_cards[card_idx])
                            akh[0] = _fval(toks_k2[0])
                            akh[1] = _fval(toks_k2[1])
                            akh[2] = _fval(toks_k2[2])
                            akh[3] = _fval(toks_k2[3])
                            card_idx += 1
        else:
            chard = _fval(toks5[0]) if len(toks5) > 0 else 0.0
            ikin = _ival(toks5[1]) if len(toks5) > 1 else 0
            card_idx += 1
            if iflag == 0:
                if card_idx < len(valid_cards):
                    card6_raw = valid_cards[card_idx].raw
                    toks6 = _card_tokens(valid_cards[card_idx])
                    is_fixed_c6 = (
                        len(card6_raw) >= 60
                        and not card6_raw[20:60].strip()
                        and len(card6_raw[:20].split()) <= 1
                    )
                    if is_fixed_c6:
                        f6 = split_fixed(card6_raw, [20, 20, 20, 20, 10, 10])
                        exp_a = _fval(f6[0], 6.0)
                        alpha_vol = _fval(f6[1], 1.0)
                        n_hard = _fval(f6[2], 0.0)
                        fcut = _fval(f6[3], 0.0)
                        fsmooth = _ival(f6[4], 0)
                        nrate = _ival(f6[5], 0)
                    elif len(toks6) == 4:
                        exp_a = _fval(toks6[0], 6.0)
                        fcut = _fval(toks6[1], 0.0)
                        fsmooth = _ival(toks6[2], 0)
                        nrate = _ival(toks6[3], 0)
                    elif len(toks6) >= 6:
                        exp_a = _fval(toks6[0], 6.0)
                        alpha_vol = _fval(toks6[1], 1.0)
                        n_hard = _fval(toks6[2], 0.0)
                        fcut = _fval(toks6[3], 0.0)
                        fsmooth = _ival(toks6[4], 0)
                        nrate = _ival(toks6[5], 0)
                    else:
                        exp_a = _fval(toks6[0], 6.0) if len(toks6) > 0 else 6.0
                        fcut = _fval(toks6[1], 0.0) if len(toks6) > 1 else 0.0
                        fsmooth = _ival(toks6[2], 0) if len(toks6) > 2 else 0
                        nrate = _ival(toks6[3], 0) if len(toks6) > 3 else 0
                    card_idx += 1
                while card_idx < len(valid_cards):
                    tc = _card_tokens(valid_cards[card_idx])
                    if tc and len(curves) < (nrate if nrate > 0 else 9999):
                        if len(tc) >= 4:
                            fid = _ival(tc[1])
                            ep = _fval(tc[2])
                            fsc = _fval(tc[3], 1.0)
                        elif len(tc) == 3:
                            fid = _ival(tc[0])
                            fsc = _fval(tc[1], 1.0)
                            ep = _fval(tc[2])
                        elif len(tc) == 2:
                            fid = _ival(tc[0])
                            fsc = _fval(tc[1], 1.0)
                            ep = 0.0
                        else:
                            fid = _ival(tc[0])
                            fsc = 1.0
                            ep = 0.0
                        curves.append(MatLaw87Curve(fct_id=fid, fscale=fsc, epsp=ep))
                        card_idx += 1
                    else:
                        break
            elif iflag == 1:
                if card_idx < len(valid_cards):
                    toks6 = _card_tokens(valid_cards[card_idx])
                    exp_a = _fval(toks6[0], 6.0) if len(toks6) > 0 else 6.0
                    alpha_vol = _fval(toks6[1], 1.0) if len(toks6) > 1 else 1.0
                    n_hard = _fval(toks6[2]) if len(toks6) > 2 else 0.0
                    fcut = _fval(toks6[3]) if len(toks6) > 3 else 0.0
                    fsmooth = _ival(toks6[4]) if len(toks6) > 4 else 0
                    nrate = _ival(toks6[5]) if len(toks6) > 5 else 0
                    card_idx += 1
                if card_idx < len(valid_cards):
                    toks7 = _card_tokens(valid_cards[card_idx])
                    aswift = _fval(toks7[0]) if len(toks7) > 0 else 0.0
                    eps0 = _fval(toks7[1]) if len(toks7) > 1 else 0.0
                    qvoce = _fval(toks7[2]) if len(toks7) > 2 else 0.0
                    beta = _fval(toks7[3]) if len(toks7) > 3 else 0.0
                    k0 = _fval(toks7[4]) if len(toks7) > 4 else 0.0
                    card_idx += 1
            elif iflag == 3:
                if card_idx < len(valid_cards):
                    toks6 = _card_tokens(valid_cards[card_idx])
                    exp_a = _fval(toks6[0], 6.0) if len(toks6) > 0 else 6.0
                    fcut = _fval(toks6[1]) if len(toks6) > 1 else 0.0
                    fsmooth = _ival(toks6[2]) if len(toks6) > 2 else 0
                    card_idx += 1
                if card_idx < len(valid_cards):
                    toks_tab0 = _card_tokens(valid_cards[card_idx])
                    idx_t = 1 if len(toks_tab0) > 3 else 0
                    tab_id0 = _ival(toks_tab0[idx_t]) if len(toks_tab0) > idx_t else 0
                    fscale0 = _fval(toks_tab0[idx_t + 1], 1.0) if len(toks_tab0) > idx_t + 1 else 1.0
                    epsd0 = _fval(toks_tab0[idx_t + 2]) if len(toks_tab0) > idx_t + 2 else 0.0
                    card_idx += 1
                if card_idx < len(valid_cards):
                    toks_tab45 = _card_tokens(valid_cards[card_idx])
                    idx_t = 1 if len(toks_tab45) > 3 else 0
                    tab_id45 = _ival(toks_tab45[idx_t]) if len(toks_tab45) > idx_t else 0
                    fscale45 = _fval(toks_tab45[idx_t + 1], 1.0) if len(toks_tab45) > idx_t + 1 else 1.0
                    epsd45 = _fval(toks_tab45[idx_t + 2]) if len(toks_tab45) > idx_t + 2 else 0.0
                    card_idx += 1
                if card_idx < len(valid_cards):
                    toks_tab90 = _card_tokens(valid_cards[card_idx])
                    idx_t = 1 if len(toks_tab90) > 3 else 0
                    tab_id90 = _ival(toks_tab90[idx_t]) if len(toks_tab90) > idx_t else 0
                    fscale90 = _fval(toks_tab90[idx_t + 1], 1.0) if len(toks_tab90) > idx_t + 1 else 1.0
                    epsd90 = _fval(toks_tab90[idx_t + 2]) if len(toks_tab90) > idx_t + 2 else 0.0
                    card_idx += 1

            if card_idx < len(valid_cards) and ikin == 1 and chard > 0.0:
                toks_k1 = _card_tokens(valid_cards[card_idx])
                if len(toks_k1) >= 4:
                    ckh[0] = _fval(toks_k1[0])
                    ckh[1] = _fval(toks_k1[1])
                    ckh[2] = _fval(toks_k1[2])
                    ckh[3] = _fval(toks_k1[3])
                    card_idx += 1
                    if card_idx < len(valid_cards):
                        toks_k2 = _card_tokens(valid_cards[card_idx])
                        akh[0] = _fval(toks_k2[0])
                        akh[1] = _fval(toks_k2[1])
                        akh[2] = _fval(toks_k2[2])
                        akh[3] = _fval(toks_k2[3])
                        card_idx += 1


    m87 = MatLaw87(
        id=mat_id, rho=rho, refer_rho=refer_rho, e=e, nu=nu,
        iflag=iflag, vp=vp, strain1=strain1, exp1=exp1,
        ifit=ifit, alpha=alpha, sigma_00=sigma_00, sigma_45=sigma_45,
        sigma_90=sigma_90, sigma_b=sigma_b, r_00=r_00, r_45=r_45,
        r_90=r_90, r_b=r_b, chard=chard, ikin=ikin,
        exp_a=exp_a, alpha_vol=alpha_vol, n_hard=n_hard, fcut=fcut, fsmooth=fsmooth, nrate=nrate,
        curves=curves, aswift=aswift, eps0=eps0, qvoce=qvoce, beta=beta, k0=k0,
        tab_id0=tab_id0, fscale0=fscale0, epsd0=epsd0,
        tab_id45=tab_id45, fscale45=fscale45, epsd45=epsd45,
        tab_id90=tab_id90, fscale90=fscale90, epsd90=epsd90,
        ckh=tuple(ckh), akh=tuple(akh),
        title=title,
    )
    model.mat_law87s[mat_id] = m87
    model.mat_barlat2000s[mat_id] = m87
    model.mat_barlats[mat_id] = m87
    model.materials[mat_id] = Material(
        id=mat_id, law=87, rho0=rho, title=title,
        params={
            "rho": rho, "rho0": rho, "refer_rho": refer_rho,
            "e": e, "nu": nu, "E": e, "Nu": nu,
            "iflag": iflag, "vp": vp, "vflag": vp, "strain1": strain1, "exp1": exp1,
            "ifit": ifit, "alpha": alpha, "alphas": alpha,
            "sigma_00": sigma_00, "sigma_45": sigma_45, "sigma_90": sigma_90, "sigma_b": sigma_b,
            "r_00": r_00, "r_45": r_45, "r_90": r_90, "r_b": r_b,
            "chard": chard, "ikin": ikin, "exp_a": exp_a, "a_exp": int(exp_a),
            "alpha_vol": alpha_vol, "n_hard": n_hard, "fcut": fcut, "fsmooth": fsmooth,
            "aswift": aswift, "a_swift": aswift, "eps0": eps0, "qvoce": qvoce, "q_voce": qvoce, "beta": beta, "k0": k0,
            "ckh": tuple(ckh), "akh": tuple(akh),
            "curves": [{"fct_id": c.fct_id, "fscale": c.fscale, "epsp": c.epsp} for c in curves],
        }
    )




def read_mat_law95(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW95/id`` or ``/MAT/BERGSTROM_BOYCE/id`` (M569): Bergstrom-Boyce visco-hyperelastic polymer."""
    from ...model.entities import MatLaw95, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not str(c).strip().startswith("#") and not str(c).strip().startswith("$")]
    if not valid_cards:
        log.error(f"/MAT/LAW95/{mat_id}: missing data cards", block.source)
        return

    rho = 0.0
    c10, c01, c20, c11, c02 = 0.0, 0.0, 0.0, 0.0, 0.0
    c30, c21, c12, c03, sb = 0.0, 0.0, 0.0, 0.0, 0.0
    d1, d2, d3 = 0.0, 0.0, 0.0
    nu = 0.0
    iform = 1
    a, c, m, ksi, tau_ref = 0.0, -0.7, 1.0, 0.01, 1.0

    if block.fixed:
        f1 = valid_cards[0].cut("MAT_LAW95_1")
        rho = _fval(f1[0]) if len(f1) > 0 else 0.0

        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("MAT_LAW95_2")
            c10 = _fval(f2[0]) if len(f2) > 0 else 0.0
            c01 = _fval(f2[1]) if len(f2) > 1 else 0.0
            c20 = _fval(f2[2]) if len(f2) > 2 else 0.0
            c11 = _fval(f2[3]) if len(f2) > 3 else 0.0
            c02 = _fval(f2[4]) if len(f2) > 4 else 0.0

        if len(valid_cards) > 2:
            f3 = valid_cards[2].cut("MAT_LAW95_3")
            c30 = _fval(f3[0]) if len(f3) > 0 else 0.0
            c21 = _fval(f3[1]) if len(f3) > 1 else 0.0
            c12 = _fval(f3[2]) if len(f3) > 2 else 0.0
            c03 = _fval(f3[3]) if len(f3) > 3 else 0.0
            sb = _fval(f3[4]) if len(f3) > 4 and f3[4].strip() else 0.0

        if len(valid_cards) > 3:
            raw_c4 = valid_cards[3].raw if hasattr(valid_cards[3], "raw") else str(valid_cards[3])
            layout4 = "MAT_LAW95_4_2020" if len(raw_c4.rstrip()) <= 60 else "MAT_LAW95_4"
            f4 = valid_cards[3].cut(layout4)
            d1 = _fval(f4[0]) if len(f4) > 0 else 0.0
            d2 = _fval(f4[1]) if len(f4) > 1 else 0.0
            d3 = _fval(f4[2]) if len(f4) > 2 else 0.0
            if len(f4) > 3 and f4[3].strip():
                nu = _fval(f4[3])
            if len(f4) > 4 and f4[4].strip():
                iform = _ival(f4[4])

        if len(valid_cards) > 4:
            raw_c5 = valid_cards[4].raw if hasattr(valid_cards[4], "raw") else str(valid_cards[4])
            layout5 = "MAT_LAW95_5_2018" if len(raw_c5.rstrip()) <= 80 else "MAT_LAW95_5"
            f5 = valid_cards[4].cut(layout5)
            a = _fval(f5[0]) if len(f5) > 0 else 0.0
            c = _fval(f5[1], -0.7) if len(f5) > 1 and f5[1].strip() else -0.7
            m = _fval(f5[2], 1.0) if len(f5) > 2 and f5[2].strip() else 1.0
            ksi = _fval(f5[3], 0.01) if len(f5) > 3 and f5[3].strip() else 0.01
            tau_ref = _fval(f5[4], 1.0) if len(f5) > 4 and f5[4].strip() else 1.0
    else:
        toks1 = valid_cards[0].tokens()
        rho = float(toks1[0]) if len(toks1) > 0 else 0.0

        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            c10 = float(toks2[0]) if len(toks2) > 0 else 0.0
            c01 = float(toks2[1]) if len(toks2) > 1 else 0.0
            c20 = float(toks2[2]) if len(toks2) > 2 else 0.0
            c11 = float(toks2[3]) if len(toks2) > 3 else 0.0
            c02 = float(toks2[4]) if len(toks2) > 4 else 0.0

        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            c30 = float(toks3[0]) if len(toks3) > 0 else 0.0
            c21 = float(toks3[1]) if len(toks3) > 1 else 0.0
            c12 = float(toks3[2]) if len(toks3) > 2 else 0.0
            c03 = float(toks3[3]) if len(toks3) > 3 else 0.0
            sb = float(toks3[4]) if len(toks3) > 4 else 0.0

        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            d1 = float(toks4[0]) if len(toks4) > 0 else 0.0
            d2 = float(toks4[1]) if len(toks4) > 1 else 0.0
            d3 = float(toks4[2]) if len(toks4) > 2 else 0.0
            nu = float(toks4[3]) if len(toks4) > 3 else 0.0
            iform = int(float(toks4[4])) if len(toks4) > 4 else 1

        if len(valid_cards) > 4:
            toks5 = valid_cards[4].tokens()
            a = float(toks5[0]) if len(toks5) > 0 else 0.0
            c = float(toks5[1]) if len(toks5) > 1 else -0.7
            m = float(toks5[2]) if len(toks5) > 2 else 1.0
            ksi = float(toks5[3]) if len(toks5) > 3 else 0.01
            tau_ref = float(toks5[4]) if len(toks5) > 4 else 1.0

    # Fortran defaults (hm_read_mat95.F:168-173, 200-202)
    if iform == 0:
        iform = 1
    if tau_ref == 0.0:
        tau_ref = 1.0
    if m == 0.0:
        m = 1.0
    if c == 0.0:
        c = -0.7
    if ksi == 0.0:
        ksi = 0.01

    # Linear elasticity derivation (hm_read_mat95.F:176-195)
    g0 = 2.0 * (c10 + c01) * (sb + 1.0)
    if d1 != 0.0:
        d1_inv = 1.0 / d1
        rbulk = 2.0 * d1_inv * (1.0 + sb)
        denom = 3.0 * rbulk + g0
        nu_calc = (3.0 * rbulk - 2.0 * g0) / (2.0 * denom) if denom != 0.0 else 0.495
        e_calc = 9.0 * rbulk * g0 / denom if denom != 0.0 else 2.0 * g0 * (1.0 + nu_calc)
    elif nu != 0.0:
        d2 = 0.0
        d3 = 0.0
        nu_calc = nu
        e_calc = 2.0 * g0 * (1.0 + nu)
        rbulk = (2.0 / 3.0) * g0 * (1.0 + nu) / (1.0 - 2.0 * nu) if (1.0 - 2.0 * nu) != 0.0 else 100.0 * g0
    else:
        d2 = 0.0
        d3 = 0.0
        nu_calc = 0.495
        rbulk = (2.0 / 3.0) * g0 * (1.0 + nu_calc) / (1.0 - 2.0 * nu_calc)
        e_calc = 2.0 * g0 * (1.0 + nu_calc)

    m95 = MatLaw95(
        id=mat_id, rho0=rho, rhor=rho, c10=c10, c01=c01, c20=c20, c11=c11, c02=c02,
        c30=c30, c21=c21, c12=c12, c03=c03, sb=sb, d1=d1, d2=d2, d3=d3,
        nu_val=nu if nu != 0.0 else nu_calc, iform=iform, a=a, expc=c, expm=m, ksi=ksi, tauref=tau_ref,
        title=title,
    )
    if not hasattr(model, "mat_law95s") or model.mat_law95s is None:
        model.mat_law95s = {}
    model.mat_law95s[mat_id] = m95
    if hasattr(model, "mat_bergstrom_boyces"):
        model.mat_bergstrom_boyces[mat_id] = m95

    model.materials[mat_id] = Material(
        id=mat_id, law=95, rho0=rho, title=title,
        params={
            "rho": rho, "rho0": rho,
            "c10": c10, "c01": c01, "c20": c20, "c11": c11, "c02": c02,
            "c30": c30, "c21": c21, "c12": c12, "c03": c03, "sb": sb,
            "d1": d1, "d2": d2, "d3": d3, "nu": nu if nu != 0.0 else nu_calc, "nu_calc": nu_calc, "iform": iform,
            "a": a, "c": c, "m": m, "ksi": ksi, "tau_ref": tau_ref,
            "g0": g0, "G": g0, "rbulk": rbulk, "bulk": rbulk, "K": rbulk,
            "E": e_calc,
        }
    )




def read_mat_law163(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW163/id`` or ``/MAT/CRUSHABLE_FOAM/id`` or ``/MAT/CRUSH_FOAM/id`` (M183, M560): Crushable foam."""
    from ...model.entities import MatLaw163, Material
    mat_id = block.user_id or 0

    def _card_tokens(c: Any) -> list[str]:
        raw = getattr(c, "raw", str(c)).strip()
        for ch in ("#", "$"):
            if ch in raw:
                raw = raw.split(ch)[0].strip()
        if not raw:
            return []
        if "," in raw:
            parts = [p.strip() for p in raw.split(",")]
            while parts and parts[-1] == "":
                parts.pop()
            return parts
        return raw.split()

    def _fval_safe(v: Any, default: float = 0.0) -> float:
        if v is None:
            return default
        s = str(v).strip().replace("D", "E").replace("d", "e")
        if not s:
            return default
        try:
            return float(s)
        except (ValueError, TypeError):
            return default

    def _ival_safe(v: Any, default: int = 0) -> int:
        if v is None:
            return default
        s = str(v).strip()
        if not s:
            return default
        try:
            return int(float(s))
        except (ValueError, TypeError):
            return default

    has_comma = any("," in getattr(c, "raw", str(c)) for c in block.cards)
    is_fixed = block.fixed and not has_comma
    if is_fixed:
        for c in block.cards:
            c_raw = getattr(c, "raw", str(c))
            raw_s = c_raw.strip()
            if not raw_s or raw_s.startswith(("#", "$")):
                continue
            if "\t" in c_raw:
                is_fixed = False
                break
            toks = raw_s.split()
            if len(toks) > 1 and len(c_raw[:20].split()) > 1:
                is_fixed = False
                break
            if len(toks) > 1 and len(c_raw.rstrip()) <= 20:
                is_fixed = False
                break

    title, cards = _fixed_data(block) if is_fixed else _title_and_data(block)
    if is_fixed:
        valid_cards = [c for c in cards if not getattr(c, "raw", str(c)).strip().startswith(("#", "$"))]
    else:
        valid_cards = [c for c in cards if not getattr(c, "is_blank", False) and not getattr(c, "raw", str(c)).strip().startswith(("#", "$"))]

    if not valid_cards:
        log.error(f"/MAT/LAW163/{mat_id}: missing data cards", block.source)
        return

    rho = 0.0
    e, nu, tsc = 0.0, 0.0, 0.0
    damp = 0.10
    ncycle = 12
    tab_id = 0
    epsd_ref = 0.0
    fscale = 1.0
    srclmt = 1.0e20
    nrs = 0

    if is_fixed:
        # Card 1: RHO (%20lg)
        if len(valid_cards) > 0:
            f1 = valid_cards[0].cut("MAT_LAW163_1")
            rho = _fval_safe(f1[0]) if len(f1) > 0 else 0.0

        # Card 2: E, NU, TSC, DAMP, [blank], NCYCLE (%20lg%20lg%20lg%20lg%10s%10d)
        if len(valid_cards) > 1:
            f2 = valid_cards[1].cut("MAT_LAW163_2")
            e = _fval_safe(f2[0]) if len(f2) > 0 else 0.0
            nu = _fval_safe(f2[1]) if len(f2) > 1 else 0.0
            tsc = _fval_safe(f2[2]) if len(f2) > 2 else 0.0
            damp_s = f2[3].strip() if len(f2) > 3 else ""
            damp = _fval_safe(damp_s, 0.10) if damp_s else 0.10
            damp = damp if damp != 0.0 else 0.10
            ncycle_s = f2[5].strip() if len(f2) > 5 else ""
            ncycle_val = _ival_safe(ncycle_s, 12) if ncycle_s else 12
            ncycle = ncycle_val if ncycle_val > 0 else 12

        # Card 3: [blank], TAB_ID, EPSD_REF, FSCALE, SRCLMT, [blank], NRS (%10s%10d%20lg%20lg%20lg%10s%10d)
        if len(valid_cards) > 2:
            f3 = valid_cards[2].cut("MAT_LAW163_3")
            tab_id = _ival_safe(f3[1]) if len(f3) > 1 else 0
            epsd_ref = _fval_safe(f3[2]) if len(f3) > 2 else 0.0
            fscale_s = f3[3].strip() if len(f3) > 3 else ""
            fscale_val = _fval_safe(fscale_s, 1.0) if fscale_s else 1.0
            fscale = fscale_val if fscale_val != 0.0 else 1.0
            srclmt_s = f3[4].strip() if len(f3) > 4 else ""
            srclmt_val = _fval_safe(srclmt_s, 1.0e20) if srclmt_s else 1.0e20
            srclmt = srclmt_val if srclmt_val > 0.0 else 1.0e20
            nrs = _ival_safe(f3[6]) if len(f3) > 6 else 0
    else:
        # Card 1: rho
        if len(valid_cards) > 0:
            toks1 = _card_tokens(valid_cards[0])
            rho = _fval_safe(toks1[0]) if len(toks1) > 0 else 0.0

        # Card 2: e, nu, tsc, damp, ncycle
        if len(valid_cards) > 1:
            toks2 = _card_tokens(valid_cards[1])
            e = _fval_safe(toks2[0]) if len(toks2) > 0 else 0.0
            nu = _fval_safe(toks2[1]) if len(toks2) > 1 else 0.0
            tsc = _fval_safe(toks2[2]) if len(toks2) > 2 else 0.0
            damp_s = toks2[3].strip() if len(toks2) > 3 else ""
            damp = _fval_safe(damp_s, 0.10) if damp_s else 0.10
            damp = damp if damp != 0.0 else 0.10
            if len(toks2) >= 6:
                ncycle_s = toks2[5].strip()
            elif len(toks2) >= 5:
                ncycle_s = toks2[4].strip()
            else:
                ncycle_s = ""
            ncycle_val = _ival_safe(ncycle_s, 12) if ncycle_s else 12
            ncycle = ncycle_val if ncycle_val > 0 else 12

        # Card 3: tab_id, epsd_ref, fscale, srclmt, nrs (or with blanks: 7 tokens)
        if len(valid_cards) > 2:
            toks3 = _card_tokens(valid_cards[2])
            if len(toks3) >= 7:
                tab_id = _ival_safe(toks3[1])
                epsd_ref = _fval_safe(toks3[2])
                fscale_s = toks3[3].strip() if len(toks3) > 3 else ""
                fscale_val = _fval_safe(fscale_s, 1.0) if fscale_s else 1.0
                srclmt_s = toks3[4].strip() if len(toks3) > 4 else ""
                srclmt_val = _fval_safe(srclmt_s, 1.0e20) if srclmt_s else 1.0e20
                nrs = _ival_safe(toks3[6]) if len(toks3) > 6 else 0
            else:
                tab_id = _ival_safe(toks3[0]) if len(toks3) > 0 else 0
                epsd_ref = _fval_safe(toks3[1]) if len(toks3) > 1 else 0.0
                fscale_s = toks3[2].strip() if len(toks3) > 2 else ""
                fscale_val = _fval_safe(fscale_s, 1.0) if fscale_s else 1.0
                srclmt_s = toks3[3].strip() if len(toks3) > 3 else ""
                srclmt_val = _fval_safe(srclmt_s, 1.0e20) if srclmt_s else 1.0e20
                nrs = _ival_safe(toks3[4]) if len(toks3) > 4 else 0
            fscale = fscale_val if fscale_val != 0.0 else 1.0
            srclmt = srclmt_val if srclmt_val > 0.0 else 1.0e20

    nrs = max(min(int(nrs), 1), 0)

    params = {
        "rho": rho, "rho0": rho, "e": e, "nu": nu, "E": e, "Nu": nu,
        "tsc": tsc, "damp": damp, "ncycle": ncycle,
        "tab_id": tab_id, "epsd_ref": epsd_ref, "fscale": fscale,
        "srclmt": srclmt, "nrs": nrs,
    }
    m163 = MatLaw163(
        id=mat_id, rho=rho, e=e, nu=nu, tsc=tsc, damp=damp, ncycle=ncycle,
        tab_id=tab_id, epsd_ref=epsd_ref, fscale=fscale, srclmt=srclmt,
        nrs=nrs, title=title, params=params,
    )
    model.mat_law163s[mat_id] = m163
    model.materials[mat_id] = Material(
        id=mat_id, law=163, rho0=rho, title=title,
        params=params,
    )




# read_mat_law169 is defined under M591 above


# =========================================================================
# M184: Steinberg-Guinan Plasticity, SAMP Plasticity, Sandwich Shell,
#       Fabric Shell, Composite Stack & Crushing Spring Suite
# =========================================================================

def read_mat_law49(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW49/id`` or ``/MAT/STEINB/id`` (M184/M557): Steinberg-Guinan high-pressure plasticity model."""
    from ...model.entities import MatLaw49, Material
    mat_id = block.user_id or 0

    def _card_tokens(c: Card) -> list[str]:
        raw = c.raw.strip()
        for ch in ("#", "$"):
            if ch in raw:
                raw = raw.split(ch)[0].strip()
        if not raw:
            return []
        if "," in raw:
            parts = [p.strip() for p in raw.split(",")]
            while parts and parts[-1] == "":
                parts.pop()
            return parts
        return raw.split()

    def _fval_safe(v: Any, default: float = 0.0) -> float:
        if v is None:
            return default
        s = str(v).strip().replace("D", "E").replace("d", "e")
        if not s:
            return default
        try:
            return float(s)
        except (ValueError, TypeError):
            return default

    has_comma = any("," in getattr(c, "raw", str(c)) for c in block.cards)
    is_fixed = block.fixed and not has_comma
    if is_fixed:
        for c in block.cards:
            c_raw = getattr(c, "raw", str(c))
            raw_s = c_raw.strip()
            if not raw_s or raw_s.startswith(("#", "$")):
                continue
            if "\t" in c_raw:
                is_fixed = False
                break
            toks = raw_s.split()
            if len(toks) > 1 and len(c_raw[:20].split()) > 1:
                is_fixed = False
                break
            if len(toks) > 1 and len(c_raw.rstrip()) <= 20:
                is_fixed = False
                break

    title, cards = _fixed_data(block) if is_fixed else _title_and_data(block)
    if is_fixed:
        valid_cards = [c for c in cards if not c.raw.strip().startswith(("#", "$"))]
    else:
        valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith(("#", "$"))]

    if not valid_cards:
        log.error(f"/MAT/LAW49/{mat_id}: missing data cards", block.source)
        return

    rho, refer_rho = 0.0, 0.0
    e0, nu = 0.0, 0.0
    sigy, beta, n, eps_max, sigma_max = 0.0, 0.0, 0.0, 0.0, 0.0
    t0, tmelt, rhoc_p, pmin = 0.0, 0.0, 0.0, 0.0
    b1, b2, h, f = 0.0, 0.0, 0.0, 0.0

    try:
        if is_fixed:
            # Card 1: RHO_I, [RHO_O] (MAT_LAW49_1: [20, 20])
            if len(valid_cards) > 0:
                f1 = valid_cards[0].cut("MAT_LAW49_1")
                rho = _fval_safe(f1[0]) if len(f1) > 0 else 0.0
                refer_rho = _fval_safe(f1[1]) if len(f1) > 1 else 0.0

            # Card 2: E0, nu (MAT_LAW49_2: [20, 20])
            if len(valid_cards) > 1:
                f2 = valid_cards[1].cut("MAT_LAW49_2")
                e0 = _fval_safe(f2[0]) if len(f2) > 0 else 0.0
                nu = _fval_safe(f2[1]) if len(f2) > 1 else 0.0

            # Card 3: sigma_0, beta, n, EPS_max, SIGMA_max (MAT_LAW49_3: [20, 20, 20, 20, 20])
            if len(valid_cards) > 2:
                f3 = valid_cards[2].cut("MAT_LAW49_3")
                sigy = _fval_safe(f3[0]) if len(f3) > 0 else 0.0
                beta = _fval_safe(f3[1]) if len(f3) > 1 else 0.0
                n = _fval_safe(f3[2]) if len(f3) > 2 else 0.0
                eps_max = _fval_safe(f3[3]) if len(f3) > 3 else 0.0
                sigma_max = _fval_safe(f3[4]) if len(f3) > 4 else 0.0

            # Card 4: T_0, Tmelt, rhoC_p, Pmin (MAT_LAW49_4: [20, 20, 20, 20])
            if len(valid_cards) > 3:
                f4 = valid_cards[3].cut("MAT_LAW49_4")
                t0 = _fval_safe(f4[0]) if len(f4) > 0 else 0.0
                tmelt = _fval_safe(f4[1]) if len(f4) > 1 else 0.0
                rhoc_p = _fval_safe(f4[2]) if len(f4) > 2 else 0.0
                pmin = _fval_safe(f4[3]) if len(f4) > 3 else 0.0

            # Card 5: b1, b2, h, f (MAT_LAW49_5: [20, 20, 20, 20])
            if len(valid_cards) > 4:
                f5 = valid_cards[4].cut("MAT_LAW49_5")
                b1 = _fval_safe(f5[0]) if len(f5) > 0 else 0.0
                b2 = _fval_safe(f5[1]) if len(f5) > 1 else 0.0
                h = _fval_safe(f5[2]) if len(f5) > 2 else 0.0
                f = _fval_safe(f5[3]) if len(f5) > 3 else 0.0
        else:
            # Free format (comma or space delimited)
            # Card 1: rho, refer_rho
            if len(valid_cards) > 0:
                toks1 = _card_tokens(valid_cards[0])
                rho = _fval_safe(toks1[0]) if len(toks1) > 0 else 0.0
                refer_rho = _fval_safe(toks1[1]) if len(toks1) > 1 else 0.0

            # Card 2: e0, nu
            if len(valid_cards) > 1:
                toks2 = _card_tokens(valid_cards[1])
                e0 = _fval_safe(toks2[0]) if len(toks2) > 0 else 0.0
                nu = _fval_safe(toks2[1]) if len(toks2) > 1 else 0.0

            # Card 3: sigy, beta, n, eps_max, sigma_max
            if len(valid_cards) > 2:
                toks3 = _card_tokens(valid_cards[2])
                sigy = _fval_safe(toks3[0]) if len(toks3) > 0 else 0.0
                beta = _fval_safe(toks3[1]) if len(toks3) > 1 else 0.0
                n = _fval_safe(toks3[2]) if len(toks3) > 2 else 0.0
                eps_max = _fval_safe(toks3[3]) if len(toks3) > 3 else 0.0
                sigma_max = _fval_safe(toks3[4]) if len(toks3) > 4 else 0.0

            # Card 4: t0, tmelt, rhoc_p, pmin
            if len(valid_cards) > 3:
                toks4 = _card_tokens(valid_cards[3])
                t0 = _fval_safe(toks4[0]) if len(toks4) > 0 else 0.0
                tmelt = _fval_safe(toks4[1]) if len(toks4) > 1 else 0.0
                rhoc_p = _fval_safe(toks4[2]) if len(toks4) > 2 else 0.0
                pmin = _fval_safe(toks4[3]) if len(toks4) > 3 else 0.0

            # Card 5: b1, b2, h, f
            if len(valid_cards) > 4:
                toks5 = _card_tokens(valid_cards[4])
                b1 = _fval_safe(toks5[0]) if len(toks5) > 0 else 0.0
                b2 = _fval_safe(toks5[1]) if len(toks5) > 1 else 0.0
                h = _fval_safe(toks5[2]) if len(toks5) > 2 else 0.0
                f = _fval_safe(toks5[3]) if len(toks5) > 3 else 0.0

        # Apply defaults per hm_read_mat49.F
        if refer_rho == 0.0:
            refer_rho = rho
        if eps_max == 0.0:
            eps_max = 1.0e20
        if sigma_max == 0.0:
            sigma_max = 1.0e20
        if t0 == 0.0:
            t0 = 300.0
        if tmelt == 0.0:
            tmelt = 1.0e20
        if pmin == 0.0:
            pmin = -1.0e20

        unit_id = block.unit_id if hasattr(block, "unit_id") else None
        law_name = block.parts[1].upper() if len(block.parts) > 1 else "LAW49"
        if law_name in ("49", "LAW49"):
            law_name = "LAW49"

        m49 = MatLaw49(
            id=mat_id, rho=rho, refer_rho=refer_rho, e0=e0, nu=nu,
            sigy=sigy, beta=beta, n=n, eps_max=eps_max, sigma_max=sigma_max,
            t0=t0, tmelt=tmelt, rhoc_p=rhoc_p, pmin=pmin,
            b1=b1, b2=b2, h=h, f=f, title=title,
            law=49, law_name=law_name, unit_id=unit_id,
        )
        model.mat_law49s[mat_id] = m49
        if hasattr(model, "mat_steinbs"):
            model.mat_steinbs[mat_id] = m49
        if hasattr(model, "mat_steinbergs"):
            model.mat_steinbergs[mat_id] = m49
        if hasattr(model, "mat_steinberg_guinans"):
            model.mat_steinberg_guinans[mat_id] = m49

        params: Dict[str, Any] = {
            "rho": rho, "rho0": rho, "refer_rho": refer_rho, "rhor": refer_rho,
            "e0": e0, "e": e0, "E": e0, "nu": nu, "Nu": nu, "pr": nu,
            "sigy": sigy, "sigma_0": sigy, "sig0": sigy,
            "beta": beta, "n": n, "hard": n,
            "eps_max": eps_max, "sigma_max": sigma_max,
            "t0": t0, "tmelt": tmelt, "rhoc_p": rhoc_p, "pmin": pmin,
            "b1": b1, "b2": b2, "h": h, "f": f,
            "G": m49.G, "G0": m49.G0, "g0": m49.G0, "bulk": m49.bulk, "K": m49.bulk, "C1": m49.C1,
        }

        model.materials[mat_id] = Material(
            id=mat_id, law=49, rho0=rho, title=title,
            law_name=law_name,
            params=params,
        )
    except ValueError as err:
        log.error(f"/MAT/LAW49/{mat_id}: malformed numeric input ({err})", block.source)




def read_mat_law76(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW76/id`` or ``/MAT/SAMP/id`` (M184): Semi-Analytical Model for Plastics (SAMP)."""
    from ...model.entities import MatLaw76
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank]
    if not valid_cards:
        log.error(f"/MAT/LAW76/{mat_id}: missing data cards", block.source)
        return

    rho, refer_rho = 0.0, 0.0
    e, nu = 0.0, 0.0
    fun_d1, fun_d2, fun_d3, fun_d4 = 0, 0, 0, 0
    fscale11, fscale22, fscale33, fscale12 = 1.0, 1.0, 1.0, 1.0
    facx, mat_nut, fun_b5, mat_pscale = 1.0, 0.0, 0, 1.0
    israte, asrate = 0, 0.0
    epsilon_f, epsilon_0, dc = 0.0, 0.0, 0.0
    fun_a1, fun_a2, fun_a3, scale = 0, 0, 0, 1.0
    iform, iflag, gflag = 0, 0, 0

    if block.fixed:
        if len(valid_cards) >= 8:
            f1 = valid_cards[0].cut("MAT_LAW76_1")
            rho = _fval(f1[0]) if len(f1) > 0 else 0.0
            refer_rho = _fval(f1[1]) if len(f1) > 1 else 0.0

            f2 = valid_cards[1].cut("MAT_LAW76_2")
            e = _fval(f2[0]) if len(f2) > 0 else 0.0
            nu = _fval(f2[1]) if len(f2) > 1 else 0.0

            f3 = valid_cards[2].cut([10, 10, 10, 10])
            fun_d1 = _ival(f3[0]) if len(f3) > 0 else 0
            fun_d2 = _ival(f3[1]) if len(f3) > 1 else 0
            fun_d3 = _ival(f3[2]) if len(f3) > 2 else 0
            fun_d4 = _ival(f3[3]) if len(f3) > 3 else 0

            f4 = valid_cards[3].cut([20, 20, 20, 20, 20])
            fscale11 = _fval(f4[0], 1.0) if len(f4) > 0 and f4[0].strip() else 1.0
            fscale22 = _fval(f4[1], 1.0) if len(f4) > 1 and f4[1].strip() else 1.0
            fscale33 = _fval(f4[2], 1.0) if len(f4) > 2 and f4[2].strip() else 1.0
            fscale12 = _fval(f4[3], 1.0) if len(f4) > 3 and f4[3].strip() else 1.0
            facx = _fval(f4[4], 1.0) if len(f4) > 4 and f4[4].strip() else 1.0

            f5 = valid_cards[4].cut([20, 10, 20, 10, 20])
            mat_nut = _fval(f5[0]) if len(f5) > 0 else 0.0
            fun_b5 = _ival(f5[1]) if len(f5) > 1 else 0
            mat_pscale = _fval(f5[2], 1.0) if len(f5) > 2 and f5[2].strip() else 1.0
            israte = _ival(f5[3]) if len(f5) > 3 else 0
            asrate = _fval(f5[4]) if len(f5) > 4 else 0.0

            f6 = valid_cards[5].cut([20, 20, 20])
            epsilon_f = _fval(f6[0]) if len(f6) > 0 else 0.0
            epsilon_0 = _fval(f6[1]) if len(f6) > 1 else 0.0
            dc = _fval(f6[2]) if len(f6) > 2 else 0.0

            f7 = valid_cards[6].cut([10, 10, 10, 20])
            fun_a1 = _ival(f7[0]) if len(f7) > 0 else 0
            fun_a2 = _ival(f7[1]) if len(f7) > 1 else 0
            fun_a3 = _ival(f7[2]) if len(f7) > 2 else 0
            scale = _fval(f7[3], 1.0) if len(f7) > 3 and f7[3].strip() else 1.0

            f8 = valid_cards[7].cut([10, 10, 10])
            iform = _ival(f8[0]) if len(f8) > 0 else 0
            iflag = _ival(f8[1]) if len(f8) > 1 else 0
            gflag = _ival(f8[2]) if len(f8) > 2 else 0
        else:
            f1 = valid_cards[0].cut("MAT_LAW76_1")
            rho = _fval(f1[0]) if len(f1) > 0 else 0.0
            refer_rho = _fval(f1[1]) if len(f1) > 1 else 0.0

            if len(valid_cards) > 1:
                f2 = valid_cards[1].cut("MAT_LAW76_2")
                e = _fval(f2[0]) if len(f2) > 0 else 0.0
                nu = _fval(f2[1]) if len(f2) > 1 else 0.0

            if len(valid_cards) > 2:
                f3 = valid_cards[2].cut("MAT_LAW76_3")
                fun_d1 = _ival(f3[0]) if len(f3) > 0 else 0
                fun_d2 = _ival(f3[1]) if len(f3) > 1 else 0
                fun_d3 = _ival(f3[2]) if len(f3) > 2 else 0
                fun_d4 = _ival(f3[3]) if len(f3) > 3 else 0
                fscale11 = _fval(f3[4], 1.0) if len(f3) > 4 and f3[4].strip() else 1.0
                fscale22 = _fval(f3[5], 1.0) if len(f3) > 5 and f3[5].strip() else 1.0
                fscale33 = _fval(f3[6], 1.0) if len(f3) > 6 and f3[6].strip() else 1.0

            if len(valid_cards) > 3:
                f4 = valid_cards[3].cut("MAT_LAW76_4")
                fscale12 = _fval(f4[0], 1.0) if len(f4) > 0 and f4[0].strip() else 1.0
                facx = _fval(f4[1], 1.0) if len(f4) > 1 and f4[1].strip() else 1.0
                mat_nut = _fval(f4[2]) if len(f4) > 2 else 0.0
                fun_b5 = _ival(f4[3]) if len(f4) > 3 else 0
                mat_pscale = _fval(f4[4], 1.0) if len(f4) > 4 and f4[4].strip() else 1.0
                israte = _ival(f4[5]) if len(f4) > 5 else 0
                asrate = _fval(f4[6]) if len(f4) > 6 else 0.0

            if len(valid_cards) > 4:
                f5 = valid_cards[4].cut("MAT_LAW76_5")
                epsilon_f = _fval(f5[0]) if len(f5) > 0 else 0.0
                epsilon_0 = _fval(f5[1]) if len(f5) > 1 else 0.0
                dc = _fval(f5[2]) if len(f5) > 2 else 0.0
                fun_a1 = _ival(f5[3]) if len(f5) > 3 else 0
                fun_a2 = _ival(f5[4]) if len(f5) > 4 else 0
                fun_a3 = _ival(f5[5]) if len(f5) > 5 else 0
                scale = _fval(f5[6], 1.0) if len(f5) > 6 and f5[6].strip() else 1.0

            if len(valid_cards) > 5:
                f6 = valid_cards[5].cut("MAT_LAW76_6")
                iform = _ival(f6[0]) if len(f6) > 0 else 0
                iflag = _ival(f6[1]) if len(f6) > 1 else 0
                gflag = _ival(f6[2]) if len(f6) > 2 else 0
    else:
        if len(valid_cards) >= 8:
            toks1 = valid_cards[0].tokens()
            rho = float(toks1[0]) if len(toks1) > 0 else 0.0
            refer_rho = float(toks1[1]) if len(toks1) > 1 else 0.0

            toks2 = valid_cards[1].tokens()
            e = float(toks2[0]) if len(toks2) > 0 else 0.0
            nu = float(toks2[1]) if len(toks2) > 1 else 0.0

            toks3 = valid_cards[2].tokens()
            fun_d1 = int(float(toks3[0])) if len(toks3) > 0 else 0
            fun_d2 = int(float(toks3[1])) if len(toks3) > 1 else 0
            fun_d3 = int(float(toks3[2])) if len(toks3) > 2 else 0
            fun_d4 = int(float(toks3[3])) if len(toks3) > 3 else 0

            toks4 = valid_cards[3].tokens()
            fscale11 = float(toks4[0]) if len(toks4) > 0 else 1.0
            fscale22 = float(toks4[1]) if len(toks4) > 1 else 1.0
            fscale33 = float(toks4[2]) if len(toks4) > 2 else 1.0
            fscale12 = float(toks4[3]) if len(toks4) > 3 else 1.0
            facx = float(toks4[4]) if len(toks4) > 4 else 1.0

            toks5 = valid_cards[4].tokens()
            mat_nut = float(toks5[0]) if len(toks5) > 0 else 0.0
            fun_b5 = int(float(toks5[1])) if len(toks5) > 1 else 0
            mat_pscale = float(toks5[2]) if len(toks5) > 2 else 1.0
            israte = int(float(toks5[3])) if len(toks5) > 3 else 0
            asrate = float(toks5[4]) if len(toks5) > 4 else 0.0

            toks6 = valid_cards[5].tokens()
            epsilon_f = float(toks6[0]) if len(toks6) > 0 else 0.0
            epsilon_0 = float(toks6[1]) if len(toks6) > 1 else 0.0
            dc = float(toks6[2]) if len(toks6) > 2 else 0.0

            toks7 = valid_cards[6].tokens()
            fun_a1 = int(float(toks7[0])) if len(toks7) > 0 else 0
            fun_a2 = int(float(toks7[1])) if len(toks7) > 1 else 0
            fun_a3 = int(float(toks7[2])) if len(toks7) > 2 else 0
            scale = float(toks7[3]) if len(toks7) > 3 else 1.0

            toks8 = valid_cards[7].tokens()
            iform = int(float(toks8[0])) if len(toks8) > 0 else 0
            iflag = int(float(toks8[1])) if len(toks8) > 1 else 0
            gflag = int(float(toks8[2])) if len(toks8) > 2 else 0
        else:
            toks1 = valid_cards[0].tokens()
            rho = float(toks1[0]) if len(toks1) > 0 else 0.0
            refer_rho = float(toks1[1]) if len(toks1) > 1 else 0.0

            if len(valid_cards) > 1:
                toks2 = valid_cards[1].tokens()
                e = float(toks2[0]) if len(toks2) > 0 else 0.0
                nu = float(toks2[1]) if len(toks2) > 1 else 0.0

            if len(valid_cards) > 2:
                toks3 = valid_cards[2].tokens()
                fun_d1 = int(float(toks3[0])) if len(toks3) > 0 else 0
                fun_d2 = int(float(toks3[1])) if len(toks3) > 1 else 0
                fun_d3 = int(float(toks3[2])) if len(toks3) > 2 else 0
                fun_d4 = int(float(toks3[3])) if len(toks3) > 3 else 0
                fscale11 = float(toks3[4]) if len(toks3) > 4 else 1.0
                fscale22 = float(toks3[5]) if len(toks3) > 5 else 1.0
                fscale33 = float(toks3[6]) if len(toks3) > 6 else 1.0

            if len(valid_cards) > 3:
                toks4 = valid_cards[3].tokens()
                fscale12 = float(toks4[0]) if len(toks4) > 0 else 1.0
                facx = float(toks4[1]) if len(toks4) > 1 else 1.0
                mat_nut = float(toks4[2]) if len(toks4) > 2 else 0.0
                fun_b5 = int(float(toks4[3])) if len(toks4) > 3 else 0
                mat_pscale = float(toks4[4]) if len(toks4) > 4 else 1.0
                israte = int(float(toks4[5])) if len(toks4) > 5 else 0
                asrate = float(toks4[6]) if len(toks4) > 6 else 0.0

            if len(valid_cards) > 4:
                toks5 = valid_cards[4].tokens()
                epsilon_f = float(toks5[0]) if len(toks5) > 0 else 0.0
                epsilon_0 = float(toks5[1]) if len(toks5) > 1 else 0.0
                dc = float(toks5[2]) if len(toks5) > 2 else 0.0
                fun_a1 = int(float(toks5[3])) if len(toks5) > 3 else 0
                fun_a2 = int(float(toks5[4])) if len(toks5) > 4 else 0
                fun_a3 = int(float(toks5[5])) if len(toks5) > 5 else 0
                scale = float(toks5[6]) if len(toks5) > 6 else 1.0

            if len(valid_cards) > 5:
                toks6 = valid_cards[5].tokens()
                iform = int(float(toks6[0])) if len(toks6) > 0 else 0
                iflag = int(float(toks6[1])) if len(toks6) > 1 else 0
                gflag = int(float(toks6[2])) if len(toks6) > 2 else 0

    m76 = MatLaw76(
        id=mat_id, rho=rho, refer_rho=refer_rho, e=e, nu=nu,
        fun_d1=fun_d1, fun_d2=fun_d2, fun_d3=fun_d3, fun_d4=fun_d4,
        fscale11=fscale11, fscale22=fscale22, fscale33=fscale33, fscale12=fscale12,
        facx=facx, mat_nut=mat_nut, fun_b5=fun_b5, mat_pscale=mat_pscale,
        israte=israte, asrate=asrate, epsilon_f=epsilon_f, epsilon_0=epsilon_0,
        dc=dc, fun_a1=fun_a1, fun_a2=fun_a2, fun_a3=fun_a3, scale=scale,
        iform=iform, iflag=iflag, gflag=gflag, title=title,
    )
    model.mat_law76s[mat_id] = m76
    model.materials[mat_id] = Material(
        id=mat_id, law=76, law_name="LAW76", rho0=rho, title=title,
        params={
            "rho": rho, "rho0": rho, "refer_rho": refer_rho,
            "e": e, "E": e, "young": e, "nu": nu, "pr": nu, "poisson": nu,
            "fun_d1": fun_d1, "fun_d2": fun_d2, "fun_d3": fun_d3, "fun_d4": fun_d4,
            "fct_id_t": fun_d1, "fct_id_c": fun_d2, "fct_id_s": fun_d3, "fct_id_b": fun_d4,
            "tab_id1": fun_d1, "tab_id2": fun_d2, "tab_id3": fun_d3,
            "fscale11": fscale11, "fscale22": fscale22, "fscale33": fscale33, "fscale12": fscale12,
            "fscale_t": fscale11, "fscale_c": fscale22, "fscale_s": fscale33, "fscale_b": fscale12,
            "facx": facx, "mat_nut": mat_nut, "nu_p": mat_nut, "nup": mat_nut,
            "fun_b5": fun_b5, "fct_id_nu": fun_b5, "fct_idpr": fun_b5,
            "mat_pscale": mat_pscale, "fscale_pr": mat_pscale,
            "israte": israte, "asrate": asrate, "fcut": asrate,
            "epsilon_f": epsilon_f, "eps_f": epsilon_f, "epsilon_0": epsilon_0, "eps_r": epsilon_0, "dc": dc,
            "fun_a1": fun_a1, "fun_a2": fun_a2, "fun_a3": fun_a3,
            "fct_id_dmg": fun_a1, "scale": scale, "scale_dmg": scale,
            "iform": iform, "iflag": iflag, "iquad": iflag, "gflag": gflag, "iconv": gflag,
            "law": 76, "law_name": "LAW76",
        }
    )




def read_mat_law60(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW60``, ``/MAT/PLAS_T3``, or ``/MAT/FABRIC`` (M185, M551): Tabulated temperature/rate plasticity model."""
    from ...model.entities import MatLaw60, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho = 0.0
    ref_rho = 0.0
    e = 0.0
    nu = 0.0
    eps_p_max = 1.0e30
    eps_t1 = 1.0e30
    eps_t2 = 2.0e30
    nfunc = 5
    fsmooth = 0
    mat_hard = 0.0
    fcut = 1.0e30
    xr_fun = 0
    ifunce = 0
    mat_fscale = 1.0
    einf = 0.0
    ce = 0.0
    funcs: list[int] = []
    fscales: list[float] = []
    rates: list[float] = []

    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]

    def _card_tokens(card) -> list[str]:
        raw = card.raw.strip()
        if "," in raw:
            return [t.strip() for t in raw.split(",") if t.strip()]
        return card.tokens()

    is_fixed = getattr(block, "fixed", False)
    if is_fixed and valid_cards:
        raw0 = valid_cards[0].raw.rstrip()
        raw1 = valid_cards[1].raw.rstrip() if len(valid_cards) > 1 else ""
        if "," in raw0 or "," in raw1:
            is_fixed = False
        elif len(valid_cards) > 1 and len(valid_cards[1].tokens()) >= 3 and len(raw1) < 41:
            is_fixed = False

    if is_fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW60_1")
            rho = _safe_float(c0[0]) if len(c0) > 0 and c0[0].strip() else 0.0
            ref_rho = _safe_float(c0[1]) if len(c0) > 1 and c0[1].strip() else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW60_2")
            e = _safe_float(c1[0]) if len(c1) > 0 and c1[0].strip() else 0.0
            nu = _safe_float(c1[1]) if len(c1) > 1 and c1[1].strip() else 0.0
            eps_p_max = _safe_float(c1[2]) if len(c1) > 2 and c1[2].strip() else 1.0e30
            eps_t1 = _safe_float(c1[3]) if len(c1) > 3 and c1[3].strip() else 1.0e30
            eps_t2 = _safe_float(c1[4]) if len(c1) > 4 and c1[4].strip() else 2.0e30
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW60_3")
            nfunc = _safe_int(c2[0]) if len(c2) > 0 and c2[0].strip() else 5
            fsmooth = _safe_int(c2[1]) if len(c2) > 1 and c2[1].strip() else 0
            mat_hard = _safe_float(c2[2]) if len(c2) > 2 and c2[2].strip() else 0.0
            fcut = _safe_float(c2[3]) if len(c2) > 3 and c2[3].strip() else 1.0e30
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW60_4")
            xr_fun = _safe_int(c3[0]) if len(c3) > 0 and c3[0].strip() else 0
            mat_fscale = _safe_float(c3[1]) if len(c3) > 1 and c3[1].strip() else 1.0
            ifunce = _safe_int(c3[2]) if len(c3) > 2 and c3[2].strip() else 0
            einf = _safe_float(c3[3]) if len(c3) > 3 and c3[3].strip() else 0.0
            ce = _safe_float(c3[4]) if len(c3) > 4 and c3[4].strip() else 0.0

        card_idx = 4
        # Functions Card 5 (1..5)
        if len(valid_cards) > card_idx:
            c_fun1 = valid_cards[card_idx].cut("MAT_LAW60_5")
            for val in c_fun1:
                fid = _safe_int(val) if val.strip() else 0
                if fid != 0:
                    funcs.append(fid)
            card_idx += 1
        # Functions Card 6 (6..10) if nfunc > 5
        if nfunc > 5 and len(valid_cards) > card_idx:
            c_fun2 = valid_cards[card_idx].cut("MAT_LAW60_6")
            for val in c_fun2:
                fid = _safe_int(val) if val.strip() else 0
                if fid != 0:
                    funcs.append(fid)
            card_idx += 1

        # Scale factors Card 7 (1..5)
        if len(valid_cards) > card_idx:
            c_sc1 = valid_cards[card_idx].cut("MAT_LAW60_7")
            for val in c_sc1:
                if val.strip():
                    fscales.append(_safe_float(val))
            card_idx += 1
        # Scale factors Card 8 (6..10) if nfunc > 5
        if nfunc > 5 and len(valid_cards) > card_idx:
            c_sc2 = valid_cards[card_idx].cut("MAT_LAW60_8")
            for val in c_sc2:
                if val.strip():
                    fscales.append(_safe_float(val))
            card_idx += 1

        # Strain rates Card 9 (1..5)
        if len(valid_cards) > card_idx:
            c_sr1 = valid_cards[card_idx].cut("MAT_LAW60_9")
            for val in c_sr1:
                if val.strip():
                    rates.append(_safe_float(val))
            card_idx += 1
        # Strain rates Card 10 (6..10) if nfunc > 5
        if nfunc > 5 and len(valid_cards) > card_idx:
            c_sr2 = valid_cards[card_idx].cut("MAT_LAW60_10")
            for val in c_sr2:
                if val.strip():
                    rates.append(_safe_float(val))
            card_idx += 1
    else:
        if len(valid_cards) > 0:
            toks = _card_tokens(valid_cards[0])
            rho = _safe_float(toks[0]) if len(toks) > 0 and toks[0] else 0.0
            ref_rho = _safe_float(toks[1]) if len(toks) > 1 and toks[1] else 0.0
        if len(valid_cards) > 1:
            toks = _card_tokens(valid_cards[1])
            e = _safe_float(toks[0]) if len(toks) > 0 and toks[0] else 0.0
            nu = _safe_float(toks[1]) if len(toks) > 1 and toks[1] else 0.0
            eps_p_max = _safe_float(toks[2]) if len(toks) > 2 and toks[2] else 1.0e30
            eps_t1 = _safe_float(toks[3]) if len(toks) > 3 and toks[3] else 1.0e30
            eps_t2 = _safe_float(toks[4]) if len(toks) > 4 and toks[4] else 2.0e30
        if len(valid_cards) > 2:
            toks = _card_tokens(valid_cards[2])
            nfunc = _safe_int(toks[0]) if len(toks) > 0 and toks[0] else 5
            fsmooth = _safe_int(toks[1]) if len(toks) > 1 and toks[1] else 0
            mat_hard = _safe_float(toks[2]) if len(toks) > 2 and toks[2] else 0.0
            fcut = _safe_float(toks[3]) if len(toks) > 3 and toks[3] else 1.0e30
        if len(valid_cards) > 3:
            toks = _card_tokens(valid_cards[3])
            xr_fun = _safe_int(toks[0]) if len(toks) > 0 and toks[0] else 0
            mat_fscale = _safe_float(toks[1]) if len(toks) > 1 and toks[1] else 1.0
            ifunce = _safe_int(toks[2]) if len(toks) > 2 and toks[2] else 0
            einf = _safe_float(toks[3]) if len(toks) > 3 and toks[3] else 0.0
            ce = _safe_float(toks[4]) if len(toks) > 4 and toks[4] else 0.0

        card_idx = 4
        # Functions Card(s)
        if len(valid_cards) > card_idx:
            toks = _card_tokens(valid_cards[card_idx])
            for t in toks:
                fid = _safe_int(t)
                if fid != 0:
                    funcs.append(fid)
            card_idx += 1
        if nfunc > 5 and len(funcs) < nfunc and len(valid_cards) > card_idx:
            toks = _card_tokens(valid_cards[card_idx])
            for t in toks:
                fid = _safe_int(t)
                if fid != 0:
                    funcs.append(fid)
            card_idx += 1

        # Scale factors Card(s)
        if len(valid_cards) > card_idx:
            toks = _card_tokens(valid_cards[card_idx])
            for t in toks:
                fscales.append(_safe_float(t))
            card_idx += 1
        if nfunc > 5 and len(fscales) < nfunc and len(valid_cards) > card_idx:
            toks = _card_tokens(valid_cards[card_idx])
            for t in toks:
                fscales.append(_safe_float(t))
            card_idx += 1

        # Strain rates Card(s)
        if len(valid_cards) > card_idx:
            toks = _card_tokens(valid_cards[card_idx])
            for t in toks:
                rates.append(_safe_float(t))
            card_idx += 1
        if nfunc > 5 and len(rates) < nfunc and len(valid_cards) > card_idx:
            toks = _card_tokens(valid_cards[card_idx])
            for t in toks:
                rates.append(_safe_float(t))
            card_idx += 1

    params = {
        "rho": rho,
        "ref_rho": ref_rho,
        "refer_rho": ref_rho,
        "E": e,
        "e": e,
        "nu": nu,
        "eps_p_max": eps_p_max,
        "eps_t1": eps_t1,
        "eps_t2": eps_t2,
        "nfunc": nfunc,
        "fsmooth": fsmooth,
        "mat_hard": mat_hard,
        "chard": mat_hard,
        "fcut": fcut,
        "xr_fun": xr_fun,
        "ifunce": ifunce,
        "mat_fscale": mat_fscale,
        "fpscale": mat_fscale,
        "einf": einf,
        "ce": ce,
        "funcs": funcs,
        "fun_ids": funcs,
        "fscales": fscales,
        "rates": rates,
        "eps_rates": rates,
    }

    m60 = MatLaw60(
        id=mat_id,
        rho=rho,
        ref_rho=ref_rho,
        e=e,
        nu=nu,
        eps_p_max=eps_p_max,
        eps_t1=eps_t1,
        eps_t2=eps_t2,
        nfunc=nfunc,
        fsmooth=fsmooth,
        mat_hard=mat_hard,
        fcut=fcut,
        xr_fun=xr_fun,
        ifunce=ifunce,
        mat_fscale=mat_fscale,
        einf=einf,
        ce=ce,
        funcs=funcs,
        fscales=fscales,
        rates=rates,
        title=title,
    )
    model.mat_law60s[mat_id] = m60
    model.materials[mat_id] = Material(
        id=mat_id,
        law=60,
        rho0=rho,
        title=title,
        law_name="LAW60",
        params=params,
    )




def read_mat_law63(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW63`` or ``/MAT/HANSEL`` (M185): Hänsel transformation plasticity model."""
    from ...model.entities import MatLaw63, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho = 0.0
    refer_rho = 0.0
    e = 0.0
    nu = 0.0
    cp = 0.0
    a = 0.0
    b = 0.0
    q = 0.0
    c = 0.0
    d = 0.0
    p = 0.0
    ahs = 0.0
    bhs = 0.0
    m = 0.0
    n = 0.0
    k1 = 0.0
    k2 = 0.0
    dh = 0.0
    vm0 = 0.0
    eps0 = 0.0
    t0 = 0.0

    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW63_1")
            rho = _safe_float(c0[0])
            refer_rho = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW63_2")
            e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            cp = _safe_float(c1[2]) if len(c1) > 2 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW63_3")
            a = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            b = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            q = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            c = _safe_float(c2[3]) if len(c2) > 3 else 0.0
            d = _safe_float(c2[4]) if len(c2) > 4 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW63_4")
            p = _safe_float(c3[0]) if len(c3) > 0 else 0.0
            ahs = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            bhs = _safe_float(c3[2]) if len(c3) > 2 else 0.0
            m = _safe_float(c3[3]) if len(c3) > 3 else 0.0
            n = _safe_float(c3[4]) if len(c3) > 4 else 0.0
        if len(valid_cards) > 4:
            c4 = valid_cards[4].cut("MAT_LAW63_5")
            k1 = _safe_float(c4[0]) if len(c4) > 0 else 0.0
            k2 = _safe_float(c4[1]) if len(c4) > 1 else 0.0
            dh = _safe_float(c4[2]) if len(c4) > 2 else 0.0
            vm0 = _safe_float(c4[3]) if len(c4) > 3 else 0.0
            eps0 = _safe_float(c4[4]) if len(c4) > 4 else 0.0
        if len(valid_cards) > 5:
            c5 = valid_cards[5].cut("MAT_LAW63_6")
            t0 = _safe_float(c5[0]) if len(c5) > 0 else 0.0
    else:
        if len(valid_cards) > 0:
            toks = valid_cards[0].tokens()
            rho = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            refer_rho = _safe_float(toks[1]) if len(toks) > 1 else 0.0
        if len(valid_cards) > 1:
            toks = valid_cards[1].tokens()
            e = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            nu = _safe_float(toks[1]) if len(toks) > 1 else 0.0
            cp = _safe_float(toks[2]) if len(toks) > 2 else 0.0
        if len(valid_cards) > 2:
            toks = valid_cards[2].tokens()
            a = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            b = _safe_float(toks[1]) if len(toks) > 1 else 0.0
            q = _safe_float(toks[2]) if len(toks) > 2 else 0.0
            c = _safe_float(toks[3]) if len(toks) > 3 else 0.0
            d = _safe_float(toks[4]) if len(toks) > 4 else 0.0
        if len(valid_cards) > 3:
            toks = valid_cards[3].tokens()
            p = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            ahs = _safe_float(toks[1]) if len(toks) > 1 else 0.0
            bhs = _safe_float(toks[2]) if len(toks) > 2 else 0.0
            m = _safe_float(toks[3]) if len(toks) > 3 else 0.0
            n = _safe_float(toks[4]) if len(toks) > 4 else 0.0
        if len(valid_cards) > 4:
            toks = valid_cards[4].tokens()
            k1 = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            k2 = _safe_float(toks[1]) if len(toks) > 1 else 0.0
            dh = _safe_float(toks[2]) if len(toks) > 2 else 0.0
            vm0 = _safe_float(toks[3]) if len(toks) > 3 else 0.0
            eps0 = _safe_float(toks[4]) if len(toks) > 4 else 0.0
        if len(valid_cards) > 5:
            toks = valid_cards[5].tokens()
            t0 = _safe_float(toks[0]) if len(toks) > 0 else 0.0

    m63 = MatLaw63(
        id=mat_id, rho=rho, refer_rho=refer_rho, e=e, nu=nu, cp=cp,
        a=a, b=b, q=q, c=c, d=d, p=p, ahs=ahs, bhs=bhs, m=m, n=n,
        k1=k1, k2=k2, dh=dh, vm0=vm0, eps0=eps0, t0=t0,
        title=title,
    )
    model.mat_law63s[mat_id] = m63
    model.materials[mat_id] = Material(
        id=mat_id, law=63, rho0=rho, title=title,
        params={
            "refer_rho": refer_rho, "E": e, "nu": nu, "cp": cp,
            "a": a, "b": b, "q": q, "c": c, "d": d,
            "p": p, "ahs": ahs, "bhs": bhs, "m": m, "n": n,
            "k1": k1, "k2": k2, "dh": dh, "vm0": vm0, "eps0": eps0, "t0": t0,
        }
    )




def read_mat_law48(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW48`` or ``/MAT/ZHAO`` (M185): Zhao strain-rate hardening plasticity model."""
    from ...model.entities import MatLaw48, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho = 0.0
    refer_rho = 0.0
    e = 0.0
    nu = 0.0
    a = 0.0
    b = 0.0
    n = 0.0
    chard = 0.0
    sig_max = 0.0
    c = 0.0
    d = 0.0
    m = 0.0
    e1 = 0.0
    k = 0.0
    eps_rate_0 = 0.0
    fcut = 0.0
    eps_max = 0.0
    eps_t1 = 0.0
    eps_t2 = 0.0

    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]

    def _card_tokens(card) -> list[str]:
        raw = card.raw.strip()
        if "," in raw:
            return [t.strip() for t in raw.split(",")]
        return card.tokens()

    is_fixed = getattr(block, "fixed", False)
    if is_fixed and valid_cards:
        raw0 = valid_cards[0].raw.rstrip()
        raw1 = valid_cards[1].raw.rstrip() if len(valid_cards) > 1 else ""
        raw2 = valid_cards[2].raw.rstrip() if len(valid_cards) > 2 else ""
        if "," in raw0 or "," in raw1 or "," in raw2:
            is_fixed = False
        elif len(valid_cards) > 1 and len(valid_cards[1].tokens()) >= 2 and len(raw1) < 35:
            is_fixed = False
        elif len(valid_cards) > 2 and len(valid_cards[2].tokens()) >= 3 and len(raw2) < 55:
            is_fixed = False

    if is_fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW48_1")
            rho = _safe_float(c0[0]) if len(c0) > 0 and c0[0].strip() else 0.0
            refer_rho = _safe_float(c0[1]) if len(c0) > 1 and c0[1].strip() else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW48_2")
            e = _safe_float(c1[0]) if len(c1) > 0 and c1[0].strip() else 0.0
            nu = _safe_float(c1[1]) if len(c1) > 1 and c1[1].strip() else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW48_3")
            a = _safe_float(c2[0]) if len(c2) > 0 and c2[0].strip() else 0.0
            b = _safe_float(c2[1]) if len(c2) > 1 and c2[1].strip() else 0.0
            n = _safe_float(c2[2]) if len(c2) > 2 and c2[2].strip() else 0.0
            chard = _safe_float(c2[3]) if len(c2) > 3 and c2[3].strip() else 0.0
            sig_max = _safe_float(c2[4]) if len(c2) > 4 and c2[4].strip() else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW48_4")
            c = _safe_float(c3[0]) if len(c3) > 0 and c3[0].strip() else 0.0
            d = _safe_float(c3[1]) if len(c3) > 1 and c3[1].strip() else 0.0
            m = _safe_float(c3[2]) if len(c3) > 2 and c3[2].strip() else 0.0
            e1 = _safe_float(c3[3]) if len(c3) > 3 and c3[3].strip() else 0.0
            k = _safe_float(c3[4]) if len(c3) > 4 and c3[4].strip() else 0.0
        if len(valid_cards) > 4:
            c4 = valid_cards[4].cut("MAT_LAW48_5")
            eps_rate_0 = _safe_float(c4[0]) if len(c4) > 0 and c4[0].strip() else 0.0
            fcut = _safe_float(c4[1]) if len(c4) > 1 and c4[1].strip() else 0.0
        if len(valid_cards) > 5:
            c5 = valid_cards[5].cut("MAT_LAW48_6")
            eps_max = _safe_float(c5[0]) if len(c5) > 0 and c5[0].strip() else 0.0
            eps_t1 = _safe_float(c5[1]) if len(c5) > 1 and c5[1].strip() else 0.0
            eps_t2 = _safe_float(c5[2]) if len(c5) > 2 and c5[2].strip() else 0.0
    else:
        if len(valid_cards) > 0:
            toks = _card_tokens(valid_cards[0])
            rho = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            refer_rho = _safe_float(toks[1]) if len(toks) > 1 else 0.0
        if len(valid_cards) > 1:
            toks = _card_tokens(valid_cards[1])
            e = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            nu = _safe_float(toks[1]) if len(toks) > 1 else 0.0
        if len(valid_cards) > 2:
            toks = _card_tokens(valid_cards[2])
            a = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            b = _safe_float(toks[1]) if len(toks) > 1 else 0.0
            n = _safe_float(toks[2]) if len(toks) > 2 else 0.0
            chard = _safe_float(toks[3]) if len(toks) > 3 else 0.0
            sig_max = _safe_float(toks[4]) if len(toks) > 4 else 0.0
        if len(valid_cards) > 3:
            toks = _card_tokens(valid_cards[3])
            c = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            d = _safe_float(toks[1]) if len(toks) > 1 else 0.0
            m = _safe_float(toks[2]) if len(toks) > 2 else 0.0
            e1 = _safe_float(toks[3]) if len(toks) > 3 else 0.0
            k = _safe_float(toks[4]) if len(toks) > 4 else 0.0
        if len(valid_cards) > 4:
            toks = _card_tokens(valid_cards[4])
            eps_rate_0 = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            fcut = _safe_float(toks[1]) if len(toks) > 1 else 0.0
        if len(valid_cards) > 5:
            toks = _card_tokens(valid_cards[5])
            eps_max = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            eps_t1 = _safe_float(toks[1]) if len(toks) > 1 else 0.0
            eps_t2 = _safe_float(toks[2]) if len(toks) > 2 else 0.0
    # Default values matching hm_read_mat48.F:
    if n == 0.0 or n == 1.0:
        n = 1.0001
    if m == 0.0 or m == 1.0:
        m = 1.0001
    if k == 0.0:
        k = 1.0
    if eps_t1 == 0.0:
        eps_t1 = 1.0e30
    if eps_t2 == 0.0:
        eps_t2 = 2.0e30
    if eps_max == 0.0:
        eps_max = 1.0e30
    if sig_max == 0.0:
        sig_max = 1.0e30
    if c == 0.0:
        eps_rate_0 = 1.0
    if fcut <= 0.0:
        fcut = 1.0e30

    m48 = MatLaw48(
        id=mat_id, rho=rho, refer_rho=refer_rho, e=e, nu=nu,
        a=a, b=b, n=n, chard=chard, sig_max=sig_max,
        c=c, d=d, m=m, e1=e1, k=k,
        eps_rate_0=eps_rate_0, fcut=fcut,
        eps_max=eps_max, eps_t1=eps_t1, eps_t2=eps_t2,
        title=title,
    )
    model.mat_law48s[mat_id] = m48
    ref_rho = refer_rho if refer_rho != 0.0 else rho
    model.materials[mat_id] = Material(
        id=mat_id, law=48, rho0=rho, title=title,
        params={
            "refer_rho": ref_rho, "rho": rho, "rho0": rho, "E": e, "nu": nu, "sigy": a, "a": a, "b": b, "n": n,
            "chard": chard, "sig_max": sig_max,
            "c": c, "d": d, "m": m, "e1": e1, "k": k,
            "eps_rate_0": eps_rate_0, "fcut": fcut,
            "eps_max": eps_max, "eps_t1": eps_t1, "eps_t2": eps_t2,
        }
    )




def read_mat_law26(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW26`` or ``/MAT/SESAM`` (M185): SESAME equation of state & hydrodynamic material model."""
    from ...model.entities import MatLaw26, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)

    rho = 0.0
    refer_rho = 0.0
    e = 0.0
    nu = 0.0
    a = 0.0
    b = 0.0
    n = 0.0
    eps_max = 0.0
    sig_max = 0.0
    e0 = 0.0
    sesam301 = ""
    c = 0.0
    eps0 = 0.0
    m = 0.0
    tmelt = 0.0
    tmax = 0.0

    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW26_1")
            rho = _safe_float(c0[0])
            refer_rho = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW26_2")
            e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW26_3")
            a = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            b = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            n = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            eps_max = _safe_float(c2[3]) if len(c2) > 3 else 0.0
            sig_max = _safe_float(c2[4]) if len(c2) > 4 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW26_4")
            e0 = _safe_float(c3[0]) if len(c3) > 0 else 0.0
        if len(valid_cards) > 4:
            c4 = valid_cards[4].cut("MAT_LAW26_5")
            sesam301 = c4[0].strip() if len(c4) > 0 else ""
        if len(valid_cards) > 5:
            c5 = valid_cards[5].cut("MAT_LAW26_6")
            c = _safe_float(c5[0]) if len(c5) > 0 else 0.0
            eps0 = _safe_float(c5[1]) if len(c5) > 1 else 0.0
            m = _safe_float(c5[2]) if len(c5) > 2 else 0.0
            tmelt = _safe_float(c5[3]) if len(c5) > 3 else 0.0
            tmax = _safe_float(c5[4]) if len(c5) > 4 else 0.0
    else:
        if len(valid_cards) > 0:
            toks = valid_cards[0].tokens()
            rho = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            refer_rho = _safe_float(toks[1]) if len(toks) > 1 else 0.0
        if len(valid_cards) > 1:
            toks = valid_cards[1].tokens()
            e = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            nu = _safe_float(toks[1]) if len(toks) > 1 else 0.0
        if len(valid_cards) > 2:
            toks = valid_cards[2].tokens()
            a = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            b = _safe_float(toks[1]) if len(toks) > 1 else 0.0
            n = _safe_float(toks[2]) if len(toks) > 2 else 0.0
            eps_max = _safe_float(toks[3]) if len(toks) > 3 else 0.0
            sig_max = _safe_float(toks[4]) if len(toks) > 4 else 0.0
        if len(valid_cards) > 3:
            toks = valid_cards[3].tokens()
            e0 = _safe_float(toks[0]) if len(toks) > 0 else 0.0
        if len(valid_cards) > 4:
            sesam301 = valid_cards[4].raw.strip()
        if len(valid_cards) > 5:
            toks = valid_cards[5].tokens()
            c = _safe_float(toks[0]) if len(toks) > 0 else 0.0
            eps0 = _safe_float(toks[1]) if len(toks) > 1 else 0.0
            m = _safe_float(toks[2]) if len(toks) > 2 else 0.0
            tmelt = _safe_float(toks[3]) if len(toks) > 3 else 0.0
            tmax = _safe_float(toks[4]) if len(toks) > 4 else 0.0

    m26 = MatLaw26(
        id=mat_id, rho=rho, refer_rho=refer_rho, e=e, nu=nu,
        a=a, b=b, n=n, eps_max=eps_max, sig_max=sig_max,
        e0=e0, sesam301=sesam301,
        c=c, eps0=eps0, m=m, tmelt=tmelt, tmax=tmax,
        title=title,
    )
    model.mat_law26s[mat_id] = m26
    model.materials[mat_id] = Material(
        id=mat_id, law=26, rho0=rho, title=title,
        params={
            "refer_rho": refer_rho, "E": e, "nu": nu, "sigy": a, "a": a, "b": b, "n": n,
            "eps_max": eps_max, "sig_max": sig_max, "e0": e0,
            "sesam301": sesam301, "c": c, "eps0": eps0, "m": m,
            "tmelt": tmelt, "tmax": tmax,
        }
    )




def read_mat_law6(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW6`` or ``/MAT/VISC_FLUID`` / ``/MAT/HYDRO`` / ``/MAT/K-EPS`` (M186): Hydrodynamic fluid with EOS & turbulence."""
    from ...model.entities import MatLaw6, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW6/{mat_id}: missing data card", block.source)
        return

    rho, rho_ref, nu = 0.0, 0.0, 0.0
    c0, c1, c2, c3 = 0.0, 0.0, 0.0, 0.0
    pmin, psh = 0.0, 0.0
    c4, c5, e0 = 0.0, 0.0, 0.0
    r0k0, ssl, c_mu, sig_k, sig_eps, bulk_ratio = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    c1_e, c2_e, c3_e = 0.0, 0.0, 0.0
    kappa, e_wall, alpha, gsi_t = 0.0, 0.0, 0.0, 0.0

    if block.fixed:
        c0_cut = valid_cards[0].cut("MAT_LAW6_1") if len(valid_cards) > 0 else []
        rho = _safe_float(c0_cut[0]) if len(c0_cut) > 0 else 0.0
        rho_ref = _safe_float(c0_cut[1]) if len(c0_cut) > 1 else rho
        if len(valid_cards) > 1:
            c1_cut = valid_cards[1].cut("MAT_LAW6_2")
            nu = _safe_float(c1_cut[0]) if len(c1_cut) > 0 else 0.0
        if len(valid_cards) > 2:
            c2_cut = valid_cards[2].cut("MAT_LAW6_3")
            c0 = _safe_float(c2_cut[0]) if len(c2_cut) > 0 else 0.0
            c1 = _safe_float(c2_cut[1]) if len(c2_cut) > 1 else 0.0
            c2 = _safe_float(c2_cut[2]) if len(c2_cut) > 2 else 0.0
            c3 = _safe_float(c2_cut[3]) if len(c2_cut) > 3 else 0.0
        if len(valid_cards) > 3:
            c3_cut = valid_cards[3].cut("MAT_LAW6_4")
            pmin = _safe_float(c3_cut[0]) if len(c3_cut) > 0 else 0.0
            psh = _safe_float(c3_cut[1]) if len(c3_cut) > 1 else 0.0
        if len(valid_cards) > 4:
            c4_cut = valid_cards[4].cut("MAT_LAW6_5")
            c4 = _safe_float(c4_cut[0]) if len(c4_cut) > 0 else 0.0
            c5 = _safe_float(c4_cut[1]) if len(c4_cut) > 1 else 0.0
            e0 = _safe_float(c4_cut[2]) if len(c4_cut) > 2 else 0.0
        if len(valid_cards) > 5:
            c5_cut = valid_cards[5].cut("MAT_LAW6_6")
            r0k0 = _safe_float(c5_cut[0]) if len(c5_cut) > 0 else 0.0
            ssl = _safe_float(c5_cut[1]) if len(c5_cut) > 1 else 0.0
        if len(valid_cards) > 6:
            c6_cut = valid_cards[6].cut("MAT_LAW6_7")
            c_mu = _safe_float(c6_cut[0]) if len(c6_cut) > 0 else 0.0
            sig_k = _safe_float(c6_cut[1]) if len(c6_cut) > 1 else 0.0
            sig_eps = _safe_float(c6_cut[2]) if len(c6_cut) > 2 else 0.0
            bulk_ratio = _safe_float(c6_cut[3]) if len(c6_cut) > 3 else 0.0
        if len(valid_cards) > 7:
            c7_cut = valid_cards[7].cut("MAT_LAW6_8")
            c1_e = _safe_float(c7_cut[0]) if len(c7_cut) > 0 else 0.0
            c2_e = _safe_float(c7_cut[1]) if len(c7_cut) > 1 else 0.0
            c3_e = _safe_float(c7_cut[2]) if len(c7_cut) > 2 else 0.0
        if len(valid_cards) > 8:
            c8_cut = valid_cards[8].cut("MAT_LAW6_9")
            kappa = _safe_float(c8_cut[0]) if len(c8_cut) > 0 else 0.0
            e_wall = _safe_float(c8_cut[1]) if len(c8_cut) > 1 else 0.0
            alpha = _safe_float(c8_cut[2]) if len(c8_cut) > 2 else 0.0
            gsi_t = _safe_float(c8_cut[3]) if len(c8_cut) > 3 else 0.0
    else:
        toks0 = valid_cards[0].tokens()
        rho = _safe_float(toks0[0]) if len(toks0) > 0 else 0.0
        rho_ref = _safe_float(toks0[1]) if len(toks0) > 1 else rho
        if len(valid_cards) > 1:
            toks1 = valid_cards[1].tokens()
            nu = _safe_float(toks1[0]) if len(toks1) > 0 else 0.0
        if len(valid_cards) > 2:
            toks2 = valid_cards[2].tokens()
            c0 = _safe_float(toks2[0]) if len(toks2) > 0 else 0.0
            c1 = _safe_float(toks2[1]) if len(toks2) > 1 else 0.0
            c2 = _safe_float(toks2[2]) if len(toks2) > 2 else 0.0
            c3 = _safe_float(toks2[3]) if len(toks2) > 3 else 0.0
        if len(valid_cards) > 3:
            toks3 = valid_cards[3].tokens()
            pmin = _safe_float(toks3[0]) if len(toks3) > 0 else 0.0
            psh = _safe_float(toks3[1]) if len(toks3) > 1 else 0.0
        if len(valid_cards) > 4:
            toks4 = valid_cards[4].tokens()
            c4 = _safe_float(toks4[0]) if len(toks4) > 0 else 0.0
            c5 = _safe_float(toks4[1]) if len(toks4) > 1 else 0.0
            e0 = _safe_float(toks4[2]) if len(toks4) > 2 else 0.0
        if len(valid_cards) > 5:
            toks5 = valid_cards[5].tokens()
            r0k0 = _safe_float(toks5[0]) if len(toks5) > 0 else 0.0
            ssl = _safe_float(toks5[1]) if len(toks5) > 1 else 0.0
        if len(valid_cards) > 6:
            toks6 = valid_cards[6].tokens()
            c_mu = _safe_float(toks6[0]) if len(toks6) > 0 else 0.0
            sig_k = _safe_float(toks6[1]) if len(toks6) > 1 else 0.0
            sig_eps = _safe_float(toks6[2]) if len(toks6) > 2 else 0.0
            bulk_ratio = _safe_float(toks6[3]) if len(toks6) > 3 else 0.0
        if len(valid_cards) > 7:
            toks7 = valid_cards[7].tokens()
            c1_e = _safe_float(toks7[0]) if len(toks7) > 0 else 0.0
            c2_e = _safe_float(toks7[1]) if len(toks7) > 1 else 0.0
            c3_e = _safe_float(toks7[2]) if len(toks7) > 2 else 0.0
        if len(valid_cards) > 8:
            toks8 = valid_cards[8].tokens()
            kappa = _safe_float(toks8[0]) if len(toks8) > 0 else 0.0
            e_wall = _safe_float(toks8[1]) if len(toks8) > 1 else 0.0
            alpha = _safe_float(toks8[2]) if len(toks8) > 2 else 0.0
            gsi_t = _safe_float(toks8[3]) if len(toks8) > 3 else 0.0

    mat = MatLaw6(
        id=mat_id, rho=rho, rho_ref=rho_ref, nu=nu,
        c0=c0, c1=c1, c2=c2, c3=c3, pmin=pmin, psh=psh,
        c4=c4, c5=c5, e0=e0, r0k0=r0k0, ssl=ssl,
        c_mu=c_mu, sig_k=sig_k, sig_eps=sig_eps, bulk_ratio=bulk_ratio,
        c1_e=c1_e, c2_e=c2_e, c3_e=c3_e, kappa=kappa, e_wall=e_wall,
        alpha=alpha, gsi_t=gsi_t, title=title,
    )
    model.mat_law6s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=6, rho0=rho, title=title,
        params={
            "rho": rho, "rho_ref": rho_ref, "nu": nu, "damp1": nu,
            "c0": c0, "c1": c1, "c2": c2, "c3": c3, "pmin": pmin, "pc": pmin, "psh": psh,
            "c4": c4, "c5": c5, "e0": e0, "ea": e0, "r0k0": r0k0, "ssl": ssl,
            "c_mu": c_mu, "sig_k": sig_k, "sig_eps": sig_eps, "bulk_ratio": bulk_ratio,
            "c1_e": c1_e, "c2_e": c2_e, "c3_e": c3_e, "kappa": kappa, "e_wall": e_wall,
            "alpha": alpha, "gsi_t": gsi_t,
        }
    )




def read_mat_law11(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW11`` or ``/MAT/BOUND`` / ``/MAT/B-K-EPS`` (M186): Boundary fluid & k-epsilon turbulence model."""
    from ...model.entities import MatLaw11, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW11/{mat_id}: missing data card", block.source)
        return

    rho, rho_ref = 0.0, 0.0
    itype = 1
    psh, scale = 0.0, 1.0
    node1 = 0
    gamma = 1.4
    k_cdi, h, c1 = 0.0, 0.0, 0.0
    fun_a1, fun_a2, pscale, fun_a6, e0 = 0, 0, 1.0, 0, 0.0
    xt_fun, yt_fun = 0, 0

    if block.fixed:
        c0_cut = valid_cards[0].cut("MAT_LAW11_1") if len(valid_cards) > 0 else []
        rho = _safe_float(c0_cut[0]) if len(c0_cut) > 0 else 0.0
        rho_ref = _safe_float(c0_cut[1]) if len(c0_cut) > 1 else rho
        if len(valid_cards) > 1:
            c1_cut = valid_cards[1].cut("MAT_LAW11_2")
            itype = _safe_int(c1_cut[0]) if len(c1_cut) > 0 else 1
            psh = _safe_float(c1_cut[1]) if len(c1_cut) > 1 else 0.0
            scale = _safe_float(c1_cut[2]) if len(c1_cut) > 2 and c1_cut[2].strip() else 1.0
        if len(valid_cards) > 2:
            c2_cut = valid_cards[2].cut("MAT_LAW11_3")
            node1 = _safe_int(c2_cut[0]) if len(c2_cut) > 0 else 0
            gamma = _safe_float(c2_cut[1]) if len(c2_cut) > 1 and c2_cut[1].strip() else 1.4
            k_cdi = _safe_float(c2_cut[2]) if len(c2_cut) > 2 else 0.0
    else:
        toks0 = valid_cards[0].tokens()
        rho = _safe_float(toks0[0]) if len(toks0) > 0 else 0.0
        rho_ref = _safe_float(toks0[1]) if len(toks0) > 1 else rho
        if len(valid_cards) > 1:
            toks1 = valid_cards[1].tokens()
            itype = _safe_int(toks1[0]) if len(toks1) > 0 else 1
            psh = _safe_float(toks1[1]) if len(toks1) > 1 else 0.0
            scale = _safe_float(toks1[2]) if len(toks1) > 2 else 1.0
        if len(valid_cards) > 2:
            toks2 = valid_cards[2].tokens()
            node1 = _safe_int(toks2[0]) if len(toks2) > 0 else 0
            gamma = _safe_float(toks2[1]) if len(toks2) > 1 else 1.4
            k_cdi = _safe_float(toks2[2]) if len(toks2) > 2 else 0.0

    mat = MatLaw11(
        id=mat_id, rho=rho, rho_ref=rho_ref, itype=itype, psh=psh, scale=scale,
        node1=node1, gamma=gamma, k_cdi=k_cdi, h=h, c1=c1,
        fun_a1=fun_a1, fun_a2=fun_a2, pscale=pscale, fun_a6=fun_a6, e0=e0,
        xt_fun=xt_fun, yt_fun=yt_fun, title=title,
    )
    model.mat_law11s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=11, rho0=rho, title=title,
        params={
            "rho": rho, "rho_ref": rho_ref, "itype": itype, "psh": psh, "scale": scale,
            "node1": node1, "gamma": gamma, "k_cdi": k_cdi, "h": h, "c1": c1,
            "fun_a1": fun_a1, "fun_a2": fun_a2, "pscale": pscale, "fun_a6": fun_a6, "e0": e0,
            "xt_fun": xt_fun, "yt_fun": yt_fun,
        }
    )




def read_mat_law77(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW77`` or ``/MAT/FOAM_AIR`` / ``/MAT/FOAM_HYST`` (M186): Foam with gas cavity and hysteresis unloading."""
    from ...model.entities import MatLaw77, MatLaw77Curve, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW77/{mat_id}: missing data card", block.source)
        return

    rho, rho_ref = 0.0, 0.0
    e0, nu, emax, epsmax = 0.0, 0.0, 0.0, 0.0
    fcut, fsmooth, nload, nunload, iflag, shape, hyst = 0.0, 0, 0, 0, 0, 0.0, 0.0
    load_curves: list[MatLaw77Curve] = []
    unload_curves: list[MatLaw77Curve] = []
    rho_gas, p0, gamma, poros = 0.0, 0.0, 1.4, 1.0
    rho_ext, pext, iclos, inc_gas = 0.0, 0.0, 0, 0

    idx = 0
    if block.fixed:
        if idx < len(valid_cards):
            c0 = valid_cards[idx].cut("MAT_LAW77_1")
            rho = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rho_ref = _safe_float(c0[1]) if len(c0) > 1 else rho
            idx += 1
        if idx < len(valid_cards):
            c1 = valid_cards[idx].cut("MAT_LAW77_2")
            e0 = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            emax = _safe_float(c1[2]) if len(c1) > 2 else 0.0
            epsmax = _safe_float(c1[3]) if len(c1) > 3 else 0.0
            idx += 1
        if idx < len(valid_cards):
            c2 = valid_cards[idx].cut("MAT_LAW77_3")
            fcut = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            fsmooth = _safe_int(c2[1]) if len(c2) > 1 else 0
            nload = _safe_int(c2[2]) if len(c2) > 2 else 0
            nunload = _safe_int(c2[3]) if len(c2) > 3 else 0
            iflag = _safe_int(c2[4]) if len(c2) > 4 else 0
            shape = _safe_float(c2[5]) if len(c2) > 5 else 0.0
            hyst = _safe_float(c2[6]) if len(c2) > 6 else 0.0
            idx += 1
        for _ in range(nload):
            if idx < len(valid_cards):
                cl = valid_cards[idx].cut("MAT_LAW77_LOAD")
                fid = _safe_int(cl[0]) if len(cl) > 0 else 0
                srate = _safe_float(cl[1]) if len(cl) > 1 else 0.0
                sc = _safe_float(cl[2]) if len(cl) > 2 and cl[2].strip() else 1.0
                load_curves.append(MatLaw77Curve(fct_id=fid, strain_rate=srate, scale=sc))
                idx += 1
        for _ in range(nunload):
            if idx < len(valid_cards):
                cu = valid_cards[idx].cut("MAT_LAW77_UNLOAD")
                fid = _safe_int(cu[0]) if len(cu) > 0 else 0
                srate = _safe_float(cu[1]) if len(cu) > 1 else 0.0
                sc = _safe_float(cu[2]) if len(cu) > 2 and cu[2].strip() else 1.0
                unload_curves.append(MatLaw77Curve(fct_id=fid, strain_rate=srate, scale=sc))
                idx += 1
        if idx < len(valid_cards):
            c4 = valid_cards[idx].cut("MAT_LAW77_4")
            rho_gas = _safe_float(c4[0]) if len(c4) > 0 else 0.0
            p0 = _safe_float(c4[1]) if len(c4) > 1 else 0.0
            gamma = _safe_float(c4[2]) if len(c4) > 2 and c4[2].strip() else 1.4
            poros = _safe_float(c4[4]) if len(c4) > 4 and c4[4].strip() else 1.0
            idx += 1
        if idx < len(valid_cards):
            c5 = valid_cards[idx].cut("MAT_LAW77_5")
            rho_ext = _safe_float(c5[0]) if len(c5) > 0 else 0.0
            pext = _safe_float(c5[1]) if len(c5) > 1 else 0.0
            iclos = _safe_int(c5[2]) if len(c5) > 2 else 0
            inc_gas = _safe_int(c5[3]) if len(c5) > 3 else 0
            idx += 1
    else:
        if idx < len(valid_cards):
            t0 = valid_cards[idx].tokens()
            rho = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rho_ref = _safe_float(t0[1]) if len(t0) > 1 else rho
            idx += 1
        if idx < len(valid_cards):
            t1 = valid_cards[idx].tokens()
            e0 = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            emax = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            epsmax = _safe_float(t1[3]) if len(t1) > 3 else 0.0
            idx += 1
        if idx < len(valid_cards):
            t2 = valid_cards[idx].tokens()
            fcut = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            fsmooth = _safe_int(t2[1]) if len(t2) > 1 else 0
            nload = _safe_int(t2[2]) if len(t2) > 2 else 0
            nunload = _safe_int(t2[3]) if len(t2) > 3 else 0
            iflag = _safe_int(t2[4]) if len(t2) > 4 else 0
            shape = _safe_float(t2[5]) if len(t2) > 5 else 0.0
            hyst = _safe_float(t2[6]) if len(t2) > 6 else 0.0
            idx += 1
        for _ in range(nload):
            if idx < len(valid_cards):
                tl = valid_cards[idx].tokens()
                fid = _safe_int(tl[0]) if len(tl) > 0 else 0
                srate = _safe_float(tl[1]) if len(tl) > 1 else 0.0
                sc = _safe_float(tl[2]) if len(tl) > 2 else 1.0
                load_curves.append(MatLaw77Curve(fct_id=fid, strain_rate=srate, scale=sc))
                idx += 1
        for _ in range(nunload):
            if idx < len(valid_cards):
                tu = valid_cards[idx].tokens()
                fid = _safe_int(tu[0]) if len(tu) > 0 else 0
                srate = _safe_float(tu[1]) if len(tu) > 1 else 0.0
                sc = _safe_float(tu[2]) if len(tu) > 2 else 1.0
                unload_curves.append(MatLaw77Curve(fct_id=fid, strain_rate=srate, scale=sc))
                idx += 1
        if idx < len(valid_cards):
            t4 = valid_cards[idx].tokens()
            rho_gas = _safe_float(t4[0]) if len(t4) > 0 else 0.0
            p0 = _safe_float(t4[1]) if len(t4) > 1 else 0.0
            gamma = _safe_float(t4[2]) if len(t4) > 2 else 1.4
            poros = _safe_float(t4[3]) if len(t4) > 3 else 1.0
            idx += 1
        if idx < len(valid_cards):
            t5 = valid_cards[idx].tokens()
            rho_ext = _safe_float(t5[0]) if len(t5) > 0 else 0.0
            pext = _safe_float(t5[1]) if len(t5) > 1 else 0.0
            iclos = _safe_int(t5[2]) if len(t5) > 2 else 0
            inc_gas = _safe_int(t5[3]) if len(t5) > 3 else 0
            idx += 1

    mat = MatLaw77(
        id=mat_id, rho=rho, rho_ref=rho_ref, e0=e0, nu=nu, emax=emax, epsmax=epsmax,
        fcut=fcut, fsmooth=fsmooth, nload=nload, nunload=nunload, iflag=iflag,
        shape=shape, hyst=hyst, load_curves=load_curves, unload_curves=unload_curves,
        rho_gas=rho_gas, p0=p0, gamma=gamma, poros=poros,
        rho_ext=rho_ext, pext=pext, iclos=iclos, inc_gas=inc_gas,
        title=title,
    )
    model.mat_law77s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=77, rho0=rho, title=title,
        params={
            "rho": rho, "rho_ref": rho_ref, "e": e0, "e0": e0, "nu": nu,
            "emax": emax, "epsmax": epsmax, "fcut": fcut, "fsmooth": fsmooth,
            "nload": nload, "nunload": nunload, "iflag": iflag, "shape": shape, "hyst": hyst,
            "load_curves": load_curves, "unload_curves": unload_curves,
            "rho_gas": rho_gas, "p0": p0, "gamma": gamma, "poros": poros, "r": poros,
            "rho_ext": rho_ext, "pext": pext, "iclos": iclos, "inc_gas": inc_gas,
        }
    )




def read_mat_law151(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW151`` or ``/MAT/MULTIFLUID`` / ``/MAT/MULTI_MAT`` (M186): Multi-material mixture law."""
    from ...model.entities import MatLaw151, MatMultiFluidFraction, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW151/{mat_id}: missing data card", block.source)
        return

    fractions: list[MatMultiFluidFraction] = []
    for line in valid_cards:
        if block.fixed:
            cf = line.cut("MAT_LAW151_ENTRY")
            mid = _safe_int(cf[0]) if len(cf) > 0 else 0
            vfrac = _safe_float(cf[1]) if len(cf) > 1 else 0.0
        else:
            tf = line.tokens()
            mid = _safe_int(tf[0]) if len(tf) > 0 else 0
            vfrac = _safe_float(tf[1]) if len(tf) > 1 else 0.0
        if mid != 0 or vfrac != 0.0:
            fractions.append(MatMultiFluidFraction(mat_id=mid, vol_frac=vfrac))

    mat = MatLaw151(id=mat_id, fractions=fractions, title=title)
    model.mat_law151s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=151, rho0=0.0, title=title,
        params={
            "fractions": fractions,
            "nip": len(fractions),
            "NIP": len(fractions),
            "MAT_ID_ARRAY": [f.mat_id for f in fractions],
            "VOL_FRAC_ARRAY": [f.vol_frac for f in fractions],
        }
    )




def read_mat_law187(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW187`` or ``/MAT/BARLAT20003D`` / ``/MAT/BARLAT_3D`` (M186): Barlat 2000 3D anisotropic plasticity."""
    from ...model.entities import MatLaw187, MatLaw187Rate, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW187/{mat_id}: missing data card", block.source)
        return

    rho, rho_ref = 0.0, 0.0
    e, nu = 0.0, 0.0
    iflag, vp = 0, 0
    c, p_exp = 0.0, 0.0
    alphas = [1.0] * 12
    a_exp = 8
    alpha_xy = 1.0
    n_exp = 0.0
    fcut = 0.0
    fsmooth = 0
    nrate = 0
    a_hard, eps0, q, b_hard, k0 = 0.0, 0.0, 0.0, 0.0, 0.0
    rates: list[MatLaw187Rate] = []

    idx = 0
    if block.fixed:
        if idx < len(valid_cards):
            c0 = valid_cards[idx].cut("MAT_LAW187_1")
            rho = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rho_ref = _safe_float(c0[1]) if len(c0) > 1 else rho
            idx += 1
        if idx < len(valid_cards):
            c1 = valid_cards[idx].cut("MAT_LAW187_2")
            e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            iflag = _safe_int(c1[2]) if len(c1) > 2 else 0
            vp = _safe_int(c1[3]) if len(c1) > 3 else 0
            c = _safe_float(c1[4]) if len(c1) > 4 else 0.0
            p_exp = _safe_float(c1[5]) if len(c1) > 5 else 0.0
            idx += 1
        if idx < len(valid_cards):
            c2 = valid_cards[idx].cut("MAT_LAW187_3")
            for j in range(4):
                if j < len(c2) and c2[j].strip():
                    alphas[j] = _safe_float(c2[j])
            idx += 1
        if idx < len(valid_cards):
            c3 = valid_cards[idx].cut("MAT_LAW187_4")
            for j in range(4):
                if j < len(c3) and c3[j].strip():
                    alphas[4 + j] = _safe_float(c3[j])
            idx += 1
        if idx < len(valid_cards):
            c4 = valid_cards[idx].cut("MAT_LAW187_5")
            for j in range(4):
                if j < len(c4) and c4[j].strip():
                    alphas[8 + j] = _safe_float(c4[j])
            idx += 1
        if idx < len(valid_cards):
            c5 = valid_cards[idx].cut("MAT_LAW187_6")
            a_exp = _safe_int(c5[1]) if len(c5) > 1 and c5[1].strip() else 8
            alpha_xy = _safe_float(c5[2]) if len(c5) > 2 and c5[2].strip() else 1.0
            n_exp = _safe_float(c5[3]) if len(c5) > 3 else 0.0
            fcut = _safe_float(c5[4]) if len(c5) > 4 else 0.0
            fsmooth = _safe_int(c5[5]) if len(c5) > 5 else 0
            nrate = _safe_int(c5[6]) if len(c5) > 6 else 0
            idx += 1
        if idx < len(valid_cards):
            c6 = valid_cards[idx].cut("MAT_LAW187_7")
            a_hard = _safe_float(c6[0]) if len(c6) > 0 else 0.0
            eps0 = _safe_float(c6[1]) if len(c6) > 1 else 0.0
            q = _safe_float(c6[2]) if len(c6) > 2 else 0.0
            b_hard = _safe_float(c6[3]) if len(c6) > 3 else 0.0
            k0 = _safe_float(c6[4]) if len(c6) > 4 else 0.0
            idx += 1
        for _ in range(nrate):
            if idx < len(valid_cards):
                cr = valid_cards[idx].cut("MAT_LAW187_RATE")
                fid = _safe_int(cr[0]) if len(cr) > 0 else 0
                sc = _safe_float(cr[2]) if len(cr) > 2 and cr[2].strip() else 1.0
                srate = _safe_float(cr[3]) if len(cr) > 3 else 0.0
                rates.append(MatLaw187Rate(fct_id=fid, scale=sc, strain_rate=srate))
                idx += 1
    else:
        if idx < len(valid_cards):
            t0 = valid_cards[idx].tokens()
            rho = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rho_ref = _safe_float(t0[1]) if len(t0) > 1 else rho
            idx += 1
        if idx < len(valid_cards):
            t1 = valid_cards[idx].tokens()
            e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            iflag = _safe_int(t1[2]) if len(t1) > 2 else 0
            vp = _safe_int(t1[3]) if len(t1) > 3 else 0
            c = _safe_float(t1[4]) if len(t1) > 4 else 0.0
            p_exp = _safe_float(t1[5]) if len(t1) > 5 else 0.0
            idx += 1
        if idx < len(valid_cards):
            t2 = valid_cards[idx].tokens()
            for j in range(min(4, len(t2))):
                alphas[j] = _safe_float(t2[j])
            idx += 1
        if idx < len(valid_cards):
            t3 = valid_cards[idx].tokens()
            for j in range(min(4, len(t3))):
                alphas[4 + j] = _safe_float(t3[j])
            idx += 1
        if idx < len(valid_cards):
            t4 = valid_cards[idx].tokens()
            for j in range(min(4, len(t4))):
                alphas[8 + j] = _safe_float(t4[j])
            idx += 1
        if idx < len(valid_cards):
            t5 = valid_cards[idx].tokens()
            a_exp = _safe_int(t5[0]) if len(t5) > 0 else 8
            alpha_xy = _safe_float(t5[1]) if len(t5) > 1 else 1.0
            n_exp = _safe_float(t5[2]) if len(t5) > 2 else 0.0
            fcut = _safe_float(t5[3]) if len(t5) > 3 else 0.0
            fsmooth = _safe_int(t5[4]) if len(t5) > 4 else 0
            nrate = _safe_int(t5[5]) if len(t5) > 5 else 0
            idx += 1
        if idx < len(valid_cards):
            t6 = valid_cards[idx].tokens()
            a_hard = _safe_float(t6[0]) if len(t6) > 0 else 0.0
            eps0 = _safe_float(t6[1]) if len(t6) > 1 else 0.0
            q = _safe_float(t6[2]) if len(t6) > 2 else 0.0
            b_hard = _safe_float(t6[3]) if len(t6) > 3 else 0.0
            k0 = _safe_float(t6[4]) if len(t6) > 4 else 0.0
            idx += 1
        for _ in range(nrate):
            if idx < len(valid_cards):
                tr = valid_cards[idx].tokens()
                fid = _safe_int(tr[0]) if len(tr) > 0 else 0
                sc = _safe_float(tr[1]) if len(tr) > 1 else 1.0
                srate = _safe_float(tr[2]) if len(tr) > 2 else 0.0
                rates.append(MatLaw187Rate(fct_id=fid, scale=sc, strain_rate=srate))
                idx += 1

    mat = MatLaw187(
        id=mat_id, rho=rho, rho_ref=rho_ref, e=e, nu=nu,
        iflag=iflag, vp=vp, c=c, p_exp=p_exp,
        alpha1=alphas[0], alpha2=alphas[1], alpha3=alphas[2], alpha4=alphas[3],
        alpha5=alphas[4], alpha6=alphas[5], alpha7=alphas[6], alpha8=alphas[7],
        alpha9=alphas[8], alpha10=alphas[9], alpha11=alphas[10], alpha12=alphas[11],
        a_exp=a_exp, alpha_xy=alpha_xy, n_exp=n_exp,
        fcut=fcut, fsmooth=fsmooth, nrate=nrate,
        a_hard=a_hard, eps0=eps0, q=q, b_hard=b_hard, k0=k0,
        rates=rates, title=title,
    )
    model.mat_law187s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=187, rho0=rho, title=title,
        params={
            "rho": rho, "rho_ref": rho_ref, "e": e, "nu": nu,
            "iflag": iflag, "vp": vp, "c": c, "p_exp": p_exp,
            "alpha1": alphas[0], "alpha2": alphas[1], "alpha3": alphas[2], "alpha4": alphas[3],
            "alpha5": alphas[4], "alpha6": alphas[5], "alpha7": alphas[6], "alpha8": alphas[7],
            "alpha9": alphas[8], "alpha10": alphas[9], "alpha11": alphas[10], "alpha12": alphas[11],
            "a_exp": a_exp, "alpha_xy": alpha_xy, "n_exp": n_exp,
            "fcut": fcut, "fsmooth": fsmooth, "nrate": nrate,
            "a_hard": a_hard, "eps0": eps0, "q": q, "b_hard": b_hard, "k0": k0,
            "rates": rates,
        }
    )




# M187: Geotechnical, Hydrodynamic, Tabulated Plasticity & Advanced Joint/Interface Suite

def read_mat_law3(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW3`` or ``/MAT/PLAS_BOST`` (M187): Elastoplastic material with Cowper-Symonds rate hardening."""
    from ...model.entities import MatLaw3
    from ..mat_reader import InactiveMaterial
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW3/{mat_id}: missing data card", block.source)
        return

    rho0, e, nu = 0.0, 0.0, 0.0
    sig_y, e_t, c, p = 0.0, 0.0, 0.0, 0.0

    # Check for legacy 3-card format (Card 0: rho0, Card 1: E nu, Card 2: sig_y e_t c p)
    if len(valid_cards) >= 3 and len(valid_cards[0].tokens()) <= 2:
        t0 = valid_cards[0].tokens()
        rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
        t1 = valid_cards[1].tokens()
        e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
        nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
        t2 = valid_cards[2].tokens()
        sig_y = _safe_float(t2[0]) if len(t2) > 0 else 0.0
        e_t = _safe_float(t2[1]) if len(t2) > 1 else 0.0
        c = _safe_float(t2[2]) if len(t2) > 2 else 0.0
        p = _safe_float(t2[3]) if len(t2) > 3 else 0.0
    elif block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW3_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            e = _safe_float(c0[1]) if len(c0) > 1 else 0.0
            nu = _safe_float(c0[2]) if len(c0) > 2 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW3_2")
            sig_y = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            e_t = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            c = _safe_float(c1[2]) if len(c1) > 2 else 0.0
            p = _safe_float(c1[3]) if len(c1) > 3 else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            e = _safe_float(t0[1]) if len(t0) > 1 else 0.0
            nu = _safe_float(t0[2]) if len(t0) > 2 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            sig_y = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            e_t = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            c = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            p = _safe_float(t1[3]) if len(t1) > 3 else 0.0

    mat = MatLaw3(id=mat_id, rho0=rho0, e=e, nu=nu, sig_y=sig_y, e_t=e_t, c=c, p=p, title=title)
    model.mat_law3s[mat_id] = mat
    model.materials[mat_id] = InactiveMaterial(
        id=mat_id, law=3, rho0=rho0, title=title, law_name="LAW3",
        params={
            "E": e if e > 0 else 200e9, "MAT_E": e if e > 0 else 200e9,
            "nu": nu if 0.0 <= nu < 0.5 else 0.3, "MAT_NU": nu if 0.0 <= nu < 0.5 else 0.3,
            "sig_y": sig_y, "SIG_Y": sig_y, "MAT_SIGY": sig_y,
            "e_t": e_t, "E_T": e_t, "MAT_BETA": e_t,
            "c": c, "C": c, "p": p, "P": p, "rho": rho0, "RHO": rho0, "MAT_RHO": rho0
        }
    )




def read_mat_law4(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW4`` or ``/MAT/HYD_JCOOK`` (M187): Hydrodynamic Johnson-Cook elastoplasticity with EOS."""
    from ...model.entities import MatLaw4
    from ..mat_reader import InactiveMaterial
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW4/{mat_id}: missing data card", block.source)
        return

    # Check if this is the RD-E-4601 style (Card 0: RHO_O, Card 1: E nu, Card 2: A B n epsmax sigmax, ...)
    if len(valid_cards) >= 5 and len(valid_cards[0].tokens()) <= 2:
        t0 = valid_cards[0].tokens()
        rho_i = _safe_float(t0[0]) if len(t0) > 0 else 0.0
        t1 = valid_cards[1].tokens()
        e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
        nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
        t2 = valid_cards[2].tokens()
        a = _safe_float(t2[0]) if len(t2) > 0 else 0.0
        b = _safe_float(t2[1]) if len(t2) > 1 else 0.0
        n = _safe_float(t2[2]) if len(t2) > 2 else 0.0
        eps_max = _safe_float(t2[3]) if len(t2) > 3 else 0.0
        sig_max = _safe_float(t2[4]) if len(t2) > 4 else 0.0
        t3 = valid_cards[3].tokens()
        pmin = _safe_float(t3[0]) if len(t3) > 0 else 0.0
        t4 = valid_cards[4].tokens()
        c = _safe_float(t4[0]) if len(t4) > 0 else 0.0
        eps_dot_0 = _safe_float(t4[1]) if len(t4) > 1 else 0.0
        m = _safe_float(t4[2]) if len(t4) > 2 else 0.0
        tmelt = _safe_float(t4[3]) if len(t4) > 3 else 0.0
        tmax = _safe_float(t4[4]) if len(t4) > 4 else 0.0
        rhocp, tr = 0.0, 0.0
        if len(valid_cards) > 5:
            t5 = valid_cards[5].tokens()
            rhocp = _safe_float(t5[0]) if len(t5) > 0 else 0.0
            tr = _safe_float(t5[1]) if len(t5) > 1 else 0.0

        mat = MatLaw4(
            id=mat_id, rho_i=rho_i, c0_eos=0.0, s_eos=0.0, gamma0=0.0, a_eos=0.0,
            a=a, b=b, n=n, c=c, eps_max=eps_max, sig_max=sig_max,
            t0=tr, tm=tmelt, m=m, cp=rhocp, pmin=pmin,
            c0=0.0, c1=0.0, c2=0.0, c3=0.0, c4=0.0, c5=0.0, title=title
        )
        model.mat_law4s[mat_id] = mat
        e_val = e if e > 0 else 200e9
        nu_val = nu if 0.0 <= nu < 0.5 else 0.3
        params = {
            "E": e_val, "MAT_E": e_val, "nu": nu_val, "MAT_NU": nu_val,
            "A": a, "MAT_SIGY": a, "B": b, "MAT_BETA": b, "n": n, "N": n, "MAT_HARD": n,
            "c": c, "C": c, "MAT_SRC": c, "eps0": eps_dot_0, "MAT_SRP": eps_dot_0,
            "Pmin": pmin, "pmin": pmin, "MAT_PC": pmin,
            "epsmax": eps_max, "eps_max": eps_max, "MAT_EPS": eps_max,
            "sigmax": sig_max, "sig_max": sig_max, "MAT_SIG": sig_max,
            "M": m, "m": m, "MAT_M": m,
            "Tmelt": tmelt, "MAT_TMELT": tmelt, "Tmax": tmax, "MAT_TMAX": tmax,
            "RHOCP": rhocp, "rho_cp": rhocp, "MAT_SPHEAT": rhocp,
            "Tr": tr, "T0": tr, "MAT_T0": tr, "rho": rho_i, "MAT_RHO": rho_i
        }
        from ... import materials  # noqa: F401 (ensure registry loaded)
        from ..mat_reader import MAT_PHYSICS_REGISTRY, GenericMaterialRecord
        builder = MAT_PHYSICS_REGISTRY.get("LAW4") or MAT_PHYSICS_REGISTRY.get("HYD_JCOOK")
        if builder is not None:
            try:
                rec = GenericMaterialRecord(
                    law_name="LAW4", law_number=4, id=mat_id, title=title,
                    params=params, density=rho_i,
                )
                live_mat = builder(rec)
                if live_mat is not None:
                    model.materials[mat_id] = live_mat
                    return
            except Exception:
                pass
        model.materials[mat_id] = InactiveMaterial(
            id=mat_id, law=4, rho0=rho_i, title=title, law_name="LAW4",
            params=params
        )
        return

    # Standard 4-card format
    rho_i, c0_eos, s_eos, gamma0, a_eos = 0.0, 0.0, 0.0, 0.0, 0.0
    a, b, n, c, eps_max, sig_max = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    t0, tm, m, cp, pmin = 0.0, 0.0, 0.0, 0.0, 0.0
    c0, c1, c2, c3, c4, c5 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            f0 = valid_cards[0].cut("MAT_LAW4_1")
            rho_i = _safe_float(f0[0]) if len(f0) > 0 else 0.0
            c0_eos = _safe_float(f0[1]) if len(f0) > 1 else 0.0
            s_eos = _safe_float(f0[2]) if len(f0) > 2 else 0.0
            gamma0 = _safe_float(f0[3]) if len(f0) > 3 else 0.0
            a_eos = _safe_float(f0[4]) if len(f0) > 4 else 0.0
        if len(valid_cards) > 1:
            f1 = valid_cards[1].cut("MAT_LAW4_2")
            a = _safe_float(f1[0]) if len(f1) > 0 else 0.0
            b = _safe_float(f1[1]) if len(f1) > 1 else 0.0
            n = _safe_float(f1[2]) if len(f1) > 2 else 0.0
            c = _safe_float(f1[3]) if len(f1) > 3 else 0.0
            eps_max = _safe_float(f1[4]) if len(f1) > 4 else 0.0
            sig_max = _safe_float(f1[5]) if len(f1) > 5 else 0.0
        if len(valid_cards) > 2:
            f2 = valid_cards[2].cut("MAT_LAW4_3")
            t0 = _safe_float(f2[0]) if len(f2) > 0 else 0.0
            tm = _safe_float(f2[1]) if len(f2) > 1 else 0.0
            m = _safe_float(f2[2]) if len(f2) > 2 else 0.0
            cp = _safe_float(f2[3]) if len(f2) > 3 else 0.0
            pmin = _safe_float(f2[4]) if len(f2) > 4 else 0.0
        if len(valid_cards) > 3:
            f3 = valid_cards[3].cut("MAT_LAW4_4")
            c0 = _safe_float(f3[0]) if len(f3) > 0 else 0.0
            c1 = _safe_float(f3[1]) if len(f3) > 1 else 0.0
            c2 = _safe_float(f3[2]) if len(f3) > 2 else 0.0
            c3 = _safe_float(f3[3]) if len(f3) > 3 else 0.0
            c4 = _safe_float(f3[4]) if len(f3) > 4 else 0.0
            c5 = _safe_float(f3[5]) if len(f3) > 5 else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho_i = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            c0_eos = _safe_float(t0[1]) if len(t0) > 1 else 0.0
            s_eos = _safe_float(t0[2]) if len(t0) > 2 else 0.0
            gamma0 = _safe_float(t0[3]) if len(t0) > 3 else 0.0
            a_eos = _safe_float(t0[4]) if len(t0) > 4 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            a = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            b = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            n = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            c = _safe_float(t1[3]) if len(t1) > 3 else 0.0
            eps_max = _safe_float(t1[4]) if len(t1) > 4 else 0.0
            sig_max = _safe_float(t1[5]) if len(t1) > 5 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            t0 = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            tm = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            m = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            cp = _safe_float(t2[3]) if len(t2) > 3 else 0.0
            pmin = _safe_float(t2[4]) if len(t2) > 4 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            c0 = _safe_float(t3[0]) if len(t3) > 0 else 0.0
            c1 = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            c2 = _safe_float(t3[2]) if len(t3) > 2 else 0.0
            c3 = _safe_float(t3[3]) if len(t3) > 3 else 0.0
            c4 = _safe_float(t3[4]) if len(t3) > 4 else 0.0
            c5 = _safe_float(t3[5]) if len(t3) > 5 else 0.0

    mat = MatLaw4(
        id=mat_id, rho_i=rho_i, c0_eos=c0_eos, s_eos=s_eos, gamma0=gamma0, a_eos=a_eos,
        a=a, b=b, n=n, c=c, eps_max=eps_max, sig_max=sig_max,
        t0=t0, tm=tm, m=m, cp=cp, pmin=pmin,
        c0=c0, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, title=title
    )
    model.mat_law4s[mat_id] = mat
    e_val = (c0_eos * c0_eos * rho_i) if (c0_eos * c0_eos * rho_i > 0) else 200e9
    params = {
        "rho": rho_i, "c0_eos": c0_eos, "s_eos": s_eos, "gamma0": gamma0, "a_eos": a_eos,
        "a": a, "b": b, "n": n, "c": c, "eps_max": eps_max, "sig_max": sig_max,
        "t0": t0, "tm": tm, "m": m, "cp": cp, "pmin": pmin,
        "c0": c0, "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5,
        "E": e_val, "MAT_E": e_val, "nu": 0.3, "MAT_NU": 0.3,
        "MAT_SIGY": a if a > 0 else (c0 if c0 > 0 else 200e6), "SIG_Y": a,
    }
    from ... import materials  # noqa: F401 (ensure registry loaded)
    from ..mat_reader import MAT_PHYSICS_REGISTRY, GenericMaterialRecord
    builder = MAT_PHYSICS_REGISTRY.get("LAW4") or MAT_PHYSICS_REGISTRY.get("HYD_JCOOK")
    if builder is not None:
        try:
            rec = GenericMaterialRecord(
                law_name="LAW4", law_number=4, id=mat_id, title=title,
                params=params, density=rho_i,
            )
            live_mat = builder(rec)
            if live_mat is not None:
                model.materials[mat_id] = live_mat
                return
        except Exception:
            pass
    model.materials[mat_id] = InactiveMaterial(
        id=mat_id, law=4, rho0=rho_i, title=title, law_name="LAW4",
        params=params
    )




def read_mat_law5(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW5`` or ``/MAT/JCOOK_TAB`` (M187): Tabulated Johnson-Cook plasticity."""
    from ...model.entities import MatLaw5
    from ..mat_reader import InactiveMaterial
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW5/{mat_id}: missing data card", block.source)
        return

    if len(valid_cards) >= 4:
        from ..mat_reader import read_generic_mat
        read_generic_mat(block, model, log)
        return

    rho0, e, nu = 0.0, 0.0, 0.0
    a, b, n, c, sig_max = 0.0, 0.0, 0.0, 0.0, 0.0
    fct_id1, fct_id2, fct_id3, fct_id4, fct_id5 = 0, 0, 0, 0, 0

    if block.fixed:
        if len(valid_cards) > 0:
            f0 = valid_cards[0].cut("MAT_JCOOK_TAB_1")
            rho0 = _safe_float(f0[0]) if len(f0) > 0 else 0.0
            e = _safe_float(f0[1]) if len(f0) > 1 else 0.0
            nu = _safe_float(f0[2]) if len(f0) > 2 else 0.0
        if len(valid_cards) > 1:
            f1 = valid_cards[1].cut("MAT_JCOOK_TAB_2")
            a = _safe_float(f1[0]) if len(f1) > 0 else 0.0
            b = _safe_float(f1[1]) if len(f1) > 1 else 0.0
            n = _safe_float(f1[2]) if len(f1) > 2 else 0.0
            c = _safe_float(f1[3]) if len(f1) > 3 else 0.0
            sig_max = _safe_float(f1[4]) if len(f1) > 4 else 0.0
        if len(valid_cards) > 2:
            f2 = valid_cards[2].cut("MAT_JCOOK_TAB_3")
            fct_id1 = _safe_int(f2[0]) if len(f2) > 0 else 0
            fct_id2 = _safe_int(f2[1]) if len(f2) > 1 else 0
            fct_id3 = _safe_int(f2[2]) if len(f2) > 2 else 0
            fct_id4 = _safe_int(f2[3]) if len(f2) > 3 else 0
            fct_id5 = _safe_int(f2[4]) if len(f2) > 4 else 0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            e = _safe_float(t0[1]) if len(t0) > 1 else 0.0
            nu = _safe_float(t0[2]) if len(t0) > 2 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            a = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            b = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            n = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            c = _safe_float(t1[3]) if len(t1) > 3 else 0.0
            sig_max = _safe_float(t1[4]) if len(t1) > 4 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            fct_id1 = _safe_int(t2[0]) if len(t2) > 0 else 0
            fct_id2 = _safe_int(t2[1]) if len(t2) > 1 else 0
            fct_id3 = _safe_int(t2[2]) if len(t2) > 2 else 0
            fct_id4 = _safe_int(t2[3]) if len(t2) > 3 else 0
            fct_id5 = _safe_int(t2[4]) if len(t2) > 4 else 0

    mat = MatLaw5(
        id=mat_id, rho0=rho0, e=e, nu=nu, a=a, b=b, n=n, c=c, sig_max=sig_max,
        fct_id1=fct_id1, fct_id2=fct_id2, fct_id3=fct_id3, fct_id4=fct_id4, fct_id5=fct_id5,
        title=title
    )
    model.mat_law5s[mat_id] = mat
    model.materials[mat_id] = InactiveMaterial(
        id=mat_id, law=5, rho0=rho0, title=title, law_name="LAW5",
        params={
            "E": e if e > 0 else 200e9, "nu": nu if 0.0 <= nu < 0.5 else 0.3, "a": a, "b": b, "n": n, "c": c, "sig_max": sig_max,
            "fct_id1": fct_id1, "fct_id2": fct_id2, "fct_id3": fct_id3, "fct_id4": fct_id4, "fct_id5": fct_id5,
            "rho": rho0
        }
    )




def read_mat_law10(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW10`` or ``/MAT/SOIL`` / ``/MAT/SOIL_CONC`` (M187): Soil and crushable concrete model."""
    from ...model.entities import MatLaw10
    from ..mat_reader import InactiveMaterial
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW10/{mat_id}: missing data card", block.source)
        return

    if len(valid_cards) >= 5:
        from ..mat_reader import read_generic_mat
        read_generic_mat(block, model, log)
        return

    rho0, g, k, a0, a1, a2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    p_cut, p_min, fct_id_p = 0.0, 0.0, 0

    if block.fixed:
        if len(valid_cards) > 0:
            f0 = valid_cards[0].cut("MAT_LAW10_1")
            rho0 = _safe_float(f0[0]) if len(f0) > 0 else 0.0
            g = _safe_float(f0[1]) if len(f0) > 1 else 0.0
            k = _safe_float(f0[2]) if len(f0) > 2 else 0.0
            a0 = _safe_float(f0[3]) if len(f0) > 3 else 0.0
            a1 = _safe_float(f0[4]) if len(f0) > 4 else 0.0
            a2 = _safe_float(f0[5]) if len(f0) > 5 else 0.0
        if len(valid_cards) > 1:
            f1 = valid_cards[1].cut("MAT_LAW10_2")
            p_cut = _safe_float(f1[0]) if len(f1) > 0 else 0.0
            p_min = _safe_float(f1[1]) if len(f1) > 1 else 0.0
            fct_id_p = _safe_int(f1[2]) if len(f1) > 2 else 0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            g = _safe_float(t0[1]) if len(t0) > 1 else 0.0
            k = _safe_float(t0[2]) if len(t0) > 2 else 0.0
            a0 = _safe_float(t0[3]) if len(t0) > 3 else 0.0
            a1 = _safe_float(t0[4]) if len(t0) > 4 else 0.0
            a2 = _safe_float(t0[5]) if len(t0) > 5 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            p_cut = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            p_min = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            fct_id_p = _safe_int(t1[2]) if len(t1) > 2 else 0

    mat = MatLaw10(
        id=mat_id, rho0=rho0, g=g, k=k, a0=a0, a1=a1, a2=a2,
        p_cut=p_cut, p_min=p_min, fct_id_p=fct_id_p, title=title
    )
    model.mat_law10s[mat_id] = mat
    nu_est = (3.0 * k - 2.0 * g) / (6.0 * k + 2.0 * g) if (6.0 * k + 2.0 * g) > 0 else 0.3
    if not (0.0 <= nu_est < 0.5):
        nu_est = 0.3
    e_est = 2.0 * g * (1.0 + nu_est) if g > 0 else 200e9
    model.materials[mat_id] = InactiveMaterial(
        id=mat_id, law=10, rho0=rho0, title=title, law_name="LAW10",
        params={
            "G": g, "K": k, "a0": a0, "a1": a1, "a2": a2,
            "p_cut": p_cut, "p_min": p_min, "fct_id_p": fct_id_p,
            "E": e_est, "nu": nu_est, "rho": rho0
        }
    )




def read_mat_cam_clay(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW14`` or ``/MAT/CAM_CLAY`` (M187): Modified Cam-Clay critical state geotechnical model."""
    from ...model.entities import MatLaw14
    from ..mat_reader import InactiveMaterial
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW14/{mat_id}: missing data card", block.source)
        return

    rho0, g, nu = 0.0, 0.0, 0.0
    m, lamda, kappa, e0, pc0 = 0.0, 0.0, 0.0, 0.0, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            f0 = valid_cards[0].cut("MAT_CAM_CLAY_1")
            rho0 = _safe_float(f0[0]) if len(f0) > 0 else 0.0
            g = _safe_float(f0[1]) if len(f0) > 1 else 0.0
            nu = _safe_float(f0[2]) if len(f0) > 2 else 0.0
        if len(valid_cards) > 1:
            f1 = valid_cards[1].cut("MAT_CAM_CLAY_2")
            m = _safe_float(f1[0]) if len(f1) > 0 else 0.0
            lamda = _safe_float(f1[1]) if len(f1) > 1 else 0.0
            kappa = _safe_float(f1[2]) if len(f1) > 2 else 0.0
            e0 = _safe_float(f1[3]) if len(f1) > 3 else 0.0
            pc0 = _safe_float(f1[4]) if len(f1) > 4 else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            g = _safe_float(t0[1]) if len(t0) > 1 else 0.0
            nu = _safe_float(t0[2]) if len(t0) > 2 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            m = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            lamda = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            kappa = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            e0 = _safe_float(t1[3]) if len(t1) > 3 else 0.0
            pc0 = _safe_float(t1[4]) if len(t1) > 4 else 0.0

    mat = MatLaw14(
        id=mat_id, rho0=rho0, g=g, nu=nu, m=m, lamda=lamda, lambda_=lamda, kappa=kappa, e0=e0, pc0=pc0, title=title
    )
    model.mat_law14s[mat_id] = mat
    nu_val = nu if 0.0 <= nu < 0.5 else 0.3
    e_est = 2.0 * g * (1.0 + nu_val) if g > 0 else 200e9
    model.materials[mat_id] = InactiveMaterial(
        id=mat_id, law=14, rho0=rho0, title=title, law_name="LAW14",
        params={
            "G": g, "nu": nu_val, "E": e_est, "m": m, "lamda": lamda, "kappa": kappa, "e0": e0, "pc0": pc0, "rho": rho0
        }
    )




def read_mat_law21(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW21`` or ``/MAT/DPRAG`` (M556): Drucker-Prager geological / soil / concrete model.

    Upstream Fortran reference:
      hm_read_mat21.F and matl21_dprag.cfg (radioss110).
    """
    from ...model.entities import MatLaw21, Material
    mat_id = block.user_id or 0

    def _card_tokens(c: Card) -> list[str]:
        raw = c.raw.strip()
        for ch in ("#", "$"):
            if ch in raw:
                raw = raw.split(ch)[0].strip()
        if not raw:
            return []
        if "," in raw:
            parts = [p.strip() for p in raw.split(",")]
            while parts and parts[-1] == "":
                parts.pop()
            toks = []
            for p in parts:
                toks.append(p)
            return toks
        return raw.split()

    has_comma = any("," in c.raw for c in block.cards)
    is_fixed = block.fixed and not has_comma
    if is_fixed:
        for c in block.cards:
            raw_s = c.raw.strip()
            if not raw_s or raw_s.startswith(("#", "$")):
                continue
            if "\t" in c.raw:
                is_fixed = False
                break
            toks = raw_s.split()
            if len(toks) > 1 and len(c.raw[:20].split()) > 1:
                is_fixed = False
                break
            if len(toks) > 1 and len(c.raw.rstrip()) <= 20:
                is_fixed = False
                break

    title, cards = _fixed_data(block) if is_fixed else _title_and_data(block)
    if is_fixed:
        valid_cards = [c for c in cards if not c.raw.strip().startswith(("#", "$"))]
    else:
        valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith(("#", "$"))]

    if not valid_cards:
        log.error(f"/MAT/LAW21/{mat_id}: missing data card", block.source)
        return

    rho = 0.0
    refer_rho = 0.0
    e = 0.0
    nu = 0.0
    a0 = 0.0
    a1 = 0.0
    a2 = 0.0
    amax = 1.0e20
    ifunc = 0
    c1 = 0.0
    pfscale = 1.0
    pmin = -1.0e30
    pext = 0.0
    bunl = 0.0
    mumax = 1.0e20
    w = 0.0
    d = 0.0
    x0 = 0.0

    try:
        if is_fixed:
            # Check for legacy 2-card format (M187: Card 1 = [rho, e, nu], Card 2 = [a0, a1, a2, w, d, x0])
            raw0 = valid_cards[0].raw if len(valid_cards) > 0 else ""
            raw1 = valid_cards[1].raw if len(valid_cards) > 1 else ""
            if len(valid_cards) <= 2 and ((len(raw0) > 40 and raw0[40:60].strip()) or (len(raw1) > 60 and raw1[60:].strip())):
                # Legacy 2-card format
                if len(valid_cards) > 0:
                    rho = _safe_float(raw0[:20]) if len(raw0) >= 20 else _safe_float(raw0)
                    refer_rho = rho
                    if len(raw0) > 20:
                        e = _safe_float(raw0[20:40])
                    if len(raw0) > 40:
                        nu = _safe_float(raw0[40:60])
                if len(valid_cards) > 1:
                    if len(raw1) >= 20:
                        a0 = _safe_float(raw1[:20])
                    if len(raw1) > 20:
                        a1 = _safe_float(raw1[20:40])
                    if len(raw1) > 40:
                        a2 = _safe_float(raw1[40:60])
                    if len(raw1) > 60:
                        w = _safe_float(raw1[60:80])
                    if len(raw1) > 80:
                        d = _safe_float(raw1[80:100])
                    if len(raw1) > 100:
                        x0 = _safe_float(raw1[100:120])
            else:
                # Canonical 6-card format (matl21_dprag.cfg)
                # Card 1: rho, refer_rho (MAT_LAW21_1: [20, 20])
                if len(valid_cards) > 0:
                    f0 = valid_cards[0].cut("MAT_LAW21_1")
                    rho = _safe_float(f0[0]) if len(f0) > 0 and f0[0].strip() else 0.0
                    refer_rho = _safe_float(f0[1]) if len(f0) > 1 and f0[1].strip() else 0.0

                # Card 2: e, nu (MAT_LAW21_2: [20, 20])
                if len(valid_cards) > 1:
                    f1 = valid_cards[1].cut("MAT_LAW21_2")
                    e = _safe_float(f1[0]) if len(f1) > 0 and f1[0].strip() else 0.0
                    nu = _safe_float(f1[1]) if len(f1) > 1 and f1[1].strip() else 0.0

                # Card 3: a0, a1, a2, amax (MAT_LAW21_3: [20, 20, 20, 20])
                if len(valid_cards) > 2:
                    f2 = valid_cards[2].cut("MAT_LAW21_3")
                    a0 = _safe_float(f2[0]) if len(f2) > 0 and f2[0].strip() else 0.0
                    a1 = _safe_float(f2[1]) if len(f2) > 1 and f2[1].strip() else 0.0
                    a2 = _safe_float(f2[2]) if len(f2) > 2 and f2[2].strip() else 0.0
                    amax = _safe_float(f2[3]) if len(f2) > 3 and f2[3].strip() else 0.0

                # Card 4: ifunc (10 col), blank (10 col), c1 (20 col), pfscale (20 col)
                if len(valid_cards) > 3:
                    f3 = valid_cards[3].cut("MAT_LAW21_4")
                    ifunc = _safe_int(f3[0]) if len(f3) > 0 and f3[0].strip() else 0
                    c1 = _safe_float(f3[2]) if len(f3) > 2 and f3[2].strip() else 0.0
                    pfscale = _safe_float(f3[3]) if len(f3) > 3 and f3[3].strip() else 0.0

                # Card 5: pmin (MAT_LAW21_5: [20]), optional pext in cols 20-40 (radioss130)
                if len(valid_cards) > 4:
                    f4 = valid_cards[4].cut("MAT_LAW21_5")
                    pmin = _safe_float(f4[0]) if len(f4) > 0 and f4[0].strip() else 0.0
                    raw4 = valid_cards[4].raw
                    if len(raw4) > 20:
                        pext_str = raw4[20:40].strip()
                        if pext_str:
                            pext = _safe_float(pext_str)

                # Card 6: bunl, mumax (MAT_LAW21_6: [20, 20])
                if len(valid_cards) > 5:
                    f5 = valid_cards[5].cut("MAT_LAW21_6")
                    bunl = _safe_float(f5[0]) if len(f5) > 0 and f5[0].strip() else 0.0
                    mumax = _safe_float(f5[1]) if len(f5) > 1 and f5[1].strip() else 0.0
        else:
            # Free format (comma or space delimited)
            t0 = _card_tokens(valid_cards[0]) if len(valid_cards) > 0 else []
            t1 = _card_tokens(valid_cards[1]) if len(valid_cards) > 1 else []
            if len(valid_cards) <= 2 and (len(t0) >= 3 or len(t1) >= 5):
                # Legacy 2-card format (M187)
                if len(valid_cards) > 0:
                    rho = _safe_float(t0[0]) if len(t0) > 0 and t0[0] else 0.0
                    refer_rho = rho
                    e = _safe_float(t0[1]) if len(t0) > 1 and t0[1] else 0.0
                    nu = _safe_float(t0[2]) if len(t0) > 2 and t0[2] else 0.0
                if len(valid_cards) > 1:
                    a0 = _safe_float(t1[0]) if len(t1) > 0 and t1[0] else 0.0
                    a1 = _safe_float(t1[1]) if len(t1) > 1 and t1[1] else 0.0
                    a2 = _safe_float(t1[2]) if len(t1) > 2 and t1[2] else 0.0
                    w = _safe_float(t1[3]) if len(t1) > 3 and t1[3] else 0.0
                    d = _safe_float(t1[4]) if len(t1) > 4 and t1[4] else 0.0
                    x0 = _safe_float(t1[5]) if len(t1) > 5 and t1[5] else 0.0
            else:
                # Canonical free format
                # Card 1: rho, refer_rho
                if len(valid_cards) > 0:
                    rho = _safe_float(t0[0]) if len(t0) > 0 and t0[0] else 0.0
                    refer_rho = _safe_float(t0[1]) if len(t0) > 1 and t0[1] else 0.0

                # Card 2: e, nu
                if len(valid_cards) > 1:
                    e = _safe_float(t1[0]) if len(t1) > 0 and t1[0] else 0.0
                    nu = _safe_float(t1[1]) if len(t1) > 1 and t1[1] else 0.0

                # Card 3: a0, a1, a2, amax
                if len(valid_cards) > 2:
                    t2 = _card_tokens(valid_cards[2])
                    a0 = _safe_float(t2[0]) if len(t2) > 0 and t2[0] else 0.0
                    a1 = _safe_float(t2[1]) if len(t2) > 1 and t2[1] else 0.0
                    a2 = _safe_float(t2[2]) if len(t2) > 2 and t2[2] else 0.0
                    amax = _safe_float(t2[3]) if len(t2) > 3 and t2[3] else 0.0

                # Card 4: ifunc, [blank], c1, pfscale
                if len(valid_cards) > 3:
                    t3 = _card_tokens(valid_cards[3])
                    if len(t3) >= 4:
                        ifunc = _safe_int(t3[0]) if t3[0] else 0
                        # t3[1] is blank / dummy
                        c1 = _safe_float(t3[2]) if t3[2] else 0.0
                        pfscale = _safe_float(t3[3]) if t3[3] else 0.0
                    elif len(t3) == 3:
                        ifunc = _safe_int(t3[0]) if t3[0] else 0
                        c1 = _safe_float(t3[1]) if t3[1] else 0.0
                        pfscale = _safe_float(t3[2]) if t3[2] else 0.0
                    elif len(t3) == 2:
                        ifunc = _safe_int(t3[0]) if t3[0] else 0
                        c1 = _safe_float(t3[1]) if t3[1] else 0.0
                    elif len(t3) == 1 and t3[0]:
                        try:
                            c1 = float(t3[0])
                        except ValueError:
                            ifunc = _safe_int(t3[0])

                # Card 5: pmin, pext
                if len(valid_cards) > 4:
                    t4 = _card_tokens(valid_cards[4])
                    pmin = _safe_float(t4[0]) if len(t4) > 0 and t4[0] else 0.0
                    if len(t4) > 1 and t4[1]:
                        pext = _safe_float(t4[1])

                # Card 6: bunl, mumax
                if len(valid_cards) > 5:
                    t5 = _card_tokens(valid_cards[5])
                    bunl = _safe_float(t5[0]) if len(t5) > 0 and t5[0] else 0.0
                    mumax = _safe_float(t5[1]) if len(t5) > 1 and t5[1] else 0.0

        # Default values per hm_read_mat21.F:
        if refer_rho == 0.0:
            refer_rho = rho
        if pmin == 0.0:
            pmin = -1.0e30
        if bunl == 0.0:
            bunl = c1
        if amax == 0.0:
            amax = 1.0e20
        if mumax == 0.0:
            mumax = 1.0e20
        if pfscale == 0.0:
            pfscale = 1.0

        unit_id = block.unit_id if hasattr(block, "unit_id") else None

        mat = MatLaw21(
            id=mat_id,
            rho=rho,
            refer_rho=refer_rho,
            e=e,
            nu=nu,
            a0=a0,
            a1=a1,
            a2=a2,
            amax=amax,
            ifunc=ifunc,
            c1=c1,
            pfscale=pfscale,
            pmin=pmin,
            pext=pext,
            bunl=bunl,
            mumax=mumax,
            w=w,
            d=d,
            x0=x0,
            title=title,
            law=21,
            law_name="LAW21",
            unit_id=unit_id,
        )
        model.mat_law21s[mat_id] = mat

        params: Dict[str, Any] = {
            "rho": rho, "rho0": rho, "refer_rho": refer_rho, "rhor": refer_rho,
            "e": e, "E": e, "young": e,
            "nu": nu, "NU": nu,
            "a0": a0, "A0": a0,
            "a1": a1, "A1": a1,
            "a2": a2, "A2": a2,
            "amax": amax, "AMAX": amax, "amx": amax,
            "ifunc": ifunc, "IFUNC": ifunc, "fun_a1": ifunc, "FUN_A1": ifunc,
            "c1": c1, "C1": c1, "bulk": c1, "BULK": c1, "mat_bulk": c1, "MAT_BULK": c1,
            "pfscale": pfscale, "PFSCALE": pfscale, "fac_y": pfscale,
            "pmin": pmin, "PMIN": pmin, "mat_pc": pmin, "MAT_PC": pmin,
            "pext": pext, "PEXT": pext,
            "bunl": bunl, "BUNL": bunl, "k_unload": bunl, "mat_k_unload": bunl,
            "mumax": mumax, "MUMAX": mumax, "xmumx": mumax, "mat_sig": mumax,
            "w": w, "W": w,
            "d": d, "D": d,
            "x0": x0, "X0": x0,
            "G": mat.G,
        }

        model.materials[mat_id] = Material(
            id=mat_id,
            law=21,
            rho0=rho,
            title=title,
            law_name="LAW21",
            params=params,
        )
    except ValueError as err:
        log.error(f"/MAT/LAW21/{mat_id}: malformed numeric input ({err})", block.source)





def read_mat_law32(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW32`` or ``/MAT/HILL``: Hill (1948) anisotropic plasticity model for shells (M542)."""
    from ...model.entities import MatLaw32, Material
    from ..mat_reader import InactiveMaterial
    mat_id = block.user_id or 0

    def _card_tokens(c: Card) -> list[str]:
        raw = c.raw.strip()
        for ch in ("#", "$"):
            if ch in raw:
                raw = raw.split(ch)[0].strip()
        if not raw:
            return []
        if "," in raw:
            parts = [p.strip() for p in raw.split(",")]
            while parts and parts[-1] == "":
                parts.pop()
            toks = []
            for p in parts:
                if not p:
                    toks.append("")
                else:
                    toks.extend(p.split())
            return toks
        return raw.split()

    has_comma = any("," in c.raw for c in block.cards)
    is_fixed = block.fixed and not has_comma
    if is_fixed:
        for c in block.cards:
            raw_s = c.raw.strip()
            if not raw_s or raw_s.startswith(("#", "$")):
                continue
            if "\t" in c.raw:
                is_fixed = False
                break
            toks = raw_s.split()
            if len(toks) > 1 and len(c.raw[:20].split()) > 1:
                is_fixed = False
                break
            if len(toks) > 1 and len(c.raw.rstrip()) <= 20:
                is_fixed = False
                break

    title, cards = _fixed_data(block) if is_fixed else _title_and_data(block)
    if is_fixed:
        valid_cards = [c for c in cards if not c.raw.strip().startswith(("#", "$"))]
    else:
        valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith(("#", "$"))]
    if not valid_cards:
        log.error(f"/MAT/LAW32/{mat_id}: missing data card", block.source)
        return

    try:
        # Check for legacy M187 3-card tabulated format (Card 0 has >= 6 tokens)
        t0_peek = _card_tokens(valid_cards[0])
        if len(t0_peek) >= 6:
            rho0 = _safe_float(t0_peek[0]) if len(t0_peek) > 0 and t0_peek[0] else 0.0
            e1 = _safe_float(t0_peek[1]) if len(t0_peek) > 1 and t0_peek[1] else 0.0
            e2 = _safe_float(t0_peek[2]) if len(t0_peek) > 2 and t0_peek[2] else 0.0
            e3 = _safe_float(t0_peek[3]) if len(t0_peek) > 3 and t0_peek[3] else 0.0
            nu12 = _safe_float(t0_peek[4]) if len(t0_peek) > 4 and t0_peek[4] else 0.0
            nu23 = _safe_float(t0_peek[5]) if len(t0_peek) > 5 and t0_peek[5] else 0.0
            nu31 = _safe_float(t0_peek[6]) if len(t0_peek) > 6 and t0_peek[6] else 0.0
            g12, g23, g31 = 0.0, 0.0, 0.0
            if len(valid_cards) > 1:
                t1 = _card_tokens(valid_cards[1])
                g12 = _safe_float(t1[0]) if len(t1) > 0 and t1[0] else 0.0
                g23 = _safe_float(t1[1]) if len(t1) > 1 and t1[1] else 0.0
                g31 = _safe_float(t1[2]) if len(t1) > 2 and t1[2] else 0.0
            fct_id11, fct_id22, fct_id33, fct_id12, fct_id23, fct_id31 = 0, 0, 0, 0, 0, 0
            if len(valid_cards) > 2:
                t2 = _card_tokens(valid_cards[2])
                fct_id11 = _safe_int(t2[0]) if len(t2) > 0 and t2[0] else 0
                fct_id22 = _safe_int(t2[1]) if len(t2) > 1 and t2[1] else 0
                fct_id33 = _safe_int(t2[2]) if len(t2) > 2 and t2[2] else 0
                fct_id12 = _safe_int(t2[3]) if len(t2) > 3 and t2[3] else 0
                fct_id23 = _safe_int(t2[4]) if len(t2) > 4 and t2[4] else 0
                fct_id31 = _safe_int(t2[5]) if len(t2) > 5 and t2[5] else 0

            mat = MatLaw32(id=mat_id, rho0=rho0, title=title)
            mat.e1, mat.e2, mat.e3 = e1, e2, e3
            mat.nu12, mat.nu23, mat.nu31 = nu12, nu23, nu31
            mat.g12, mat.g23, mat.g31 = g12, g23, g31
            mat.fct_id11, mat.fct_id22, mat.fct_id33 = fct_id11, fct_id22, fct_id33
            mat.fct_id12, mat.fct_id23, mat.fct_id31 = fct_id12, fct_id23, fct_id31
            model.mat_law32s[mat_id] = mat
            e_val = e1 if e1 > 0 else 200e9
            nu_val = nu12 if 0.0 <= nu12 < 0.5 else 0.3
            model.materials[mat_id] = InactiveMaterial(
                id=mat_id, law=32, rho0=rho0, title=title, law_name="LAW32",
                params={
                    "E1": e1, "E2": e2, "E3": e3, "nu12": nu12, "nu23": nu23, "nu31": nu31,
                    "G12": g12, "G23": g23, "G31": g31, "E": e_val, "nu": nu_val,
                    "fct_id11": fct_id11, "fct_id22": fct_id22, "fct_id33": fct_id33,
                    "fct_id12": fct_id12, "fct_id23": fct_id23, "fct_id31": fct_id31, "rho": rho0
                }
            )
            return

        # Check for legacy RD-E-2500 tabulated format
        if len(valid_cards) >= 4 and len(_card_tokens(valid_cards[2])) == 3 and len(_card_tokens(valid_cards[3])) == 4:
            t0 = _card_tokens(valid_cards[0])
            rho0 = _safe_float(t0[0]) if len(t0) > 0 and t0[0] else 0.0
            t1 = _card_tokens(valid_cards[1])
            e = _safe_float(t1[0]) if len(t1) > 0 and t1[0] else 0.0
            nu = _safe_float(t1[1]) if len(t1) > 1 and t1[1] else 0.0
            t2 = _card_tokens(valid_cards[2])
            funct_ide = _safe_int(t2[0]) if len(t2) > 0 and t2[0] else 0
            einf = _safe_float(t2[1]) if len(t2) > 1 and t2[1] else 0.0
            ce = _safe_float(t2[2]) if len(t2) > 2 and t2[2] else 0.0
            t3 = _card_tokens(valid_cards[3])
            r00 = _safe_float(t3[0]) if len(t3) > 0 and t3[0] else 0.0
            r45 = _safe_float(t3[1]) if len(t3) > 1 and t3[1] else 0.0
            r90 = _safe_float(t3[2]) if len(t3) > 2 and t3[2] else 0.0
            c_hard = _safe_float(t3[3]) if len(t3) > 3 and t3[3] else 0.0

            mat = MatLaw32(id=mat_id, rho0=rho0, title=title)
            mat.e1, mat.e2, mat.e3 = e, e, e
            mat.nu12, mat.nu23, mat.nu31 = nu, nu, nu
            mat.fct_id11 = funct_ide
            model.mat_law32s[mat_id] = mat
            model.materials[mat_id] = InactiveMaterial(
                id=mat_id, law=32, rho0=rho0, title=title, law_name="LAW32",
                params={
                    "E": e if e > 0 else 200e9, "NU": nu if 0.0 <= nu < 0.5 else 0.3, "nu": nu if 0.0 <= nu < 0.5 else 0.3, "rho": rho0,
                    "FUNCT_IDE": funct_ide, "EINF": einf, "CE": ce,
                    "r00": r00, "r45": r45, "r90": r90, "C_hard": c_hard
                }
            )
            return

        # Standard OpenRadioss /MAT/LAW32 (/MAT/HILL) 5-card format per matl32_hill.cfg & hm_read_mat32.F
        rho0 = 0.0
        rhor = 0.0
        e = 0.0
        nu = 0.0
        sigy = 1.0e30
        beta = 0.0
        hard = 1.0
        eps = 1.0e30
        sig = 1.0e30
        srp = 1.0
        src = 0.0
        r00 = 1.0
        r45 = 1.0
        r90 = 1.0

        if is_fixed:
            # Card 1: MAT_RHO, [Refer_Rho]
            if len(valid_cards) > 0:
                c0 = valid_cards[0].cut("MAT_LAW32_1")
                rho0 = _safe_float(c0[0]) if len(c0) > 0 and c0[0] else 0.0
                rhor = _safe_float(c0[1]) if len(c0) > 1 and c0[1] else rho0
            # Card 2: MAT_E, MAT_NU
            if len(valid_cards) > 1:
                c1 = valid_cards[1].cut("MAT_LAW32_2")
                e = _safe_float(c1[0]) if len(c1) > 0 and c1[0] else 0.0
                nu = _safe_float(c1[1]) if len(c1) > 1 and c1[1] else 0.0
            # Card 3: MAT_SIGY, MAT_BETA, MAT_HARD, MAT_EPS, MAT_SIG
            if len(valid_cards) > 2:
                c2 = valid_cards[2].cut("MAT_LAW32_3")
                sigy = _safe_float(c2[0]) if len(c2) > 0 and c2[0] else 1.0e30
                beta = _safe_float(c2[1]) if len(c2) > 1 and c2[1] else 0.0
                hard = _safe_float(c2[2]) if len(c2) > 2 and c2[2] else 1.0
                eps = _safe_float(c2[3]) if len(c2) > 3 and c2[3] else 1.0e30
                sig = _safe_float(c2[4]) if len(c2) > 4 and c2[4] else 1.0e30
            # Card 4: MAT_SRP, MAT_SRC
            if len(valid_cards) > 3:
                c3 = valid_cards[3].cut("MAT_LAW32_4")
                srp = _safe_float(c3[0]) if len(c3) > 0 and c3[0] else 1.0
                src = _safe_float(c3[1]) if len(c3) > 1 and c3[1] else 0.0
            # Card 5: MAT_R00, MAT_R45, MAT_R90
            if len(valid_cards) > 4:
                c4 = valid_cards[4].cut("MAT_LAW32_5")
                r00 = _safe_float(c4[0]) if len(c4) > 0 and c4[0] else 1.0
                r45 = _safe_float(c4[1]) if len(c4) > 1 and c4[1] else 1.0
                r90 = _safe_float(c4[2]) if len(c4) > 2 and c4[2] else 1.0
        else:
            # Free-format
            if len(valid_cards) > 0:
                t0 = _card_tokens(valid_cards[0])
                rho0 = _safe_float(t0[0]) if len(t0) > 0 and t0[0] else 0.0
                rhor = _safe_float(t0[1]) if len(t0) > 1 and t0[1] else rho0
            if len(valid_cards) > 1:
                t1 = _card_tokens(valid_cards[1])
                e = _safe_float(t1[0]) if len(t1) > 0 and t1[0] else 0.0
                nu = _safe_float(t1[1]) if len(t1) > 1 and t1[1] else 0.0
            if len(valid_cards) > 2:
                t2 = _card_tokens(valid_cards[2])
                sigy = _safe_float(t2[0]) if len(t2) > 0 and t2[0] else 1.0e30
                beta = _safe_float(t2[1]) if len(t2) > 1 and t2[1] else 0.0
                hard = _safe_float(t2[2]) if len(t2) > 2 and t2[2] else 1.0
                eps = _safe_float(t2[3]) if len(t2) > 3 and t2[3] else 1.0e30
                sig = _safe_float(t2[4]) if len(t2) > 4 and t2[4] else 1.0e30
            if len(valid_cards) > 3:
                t3 = _card_tokens(valid_cards[3])
                srp = _safe_float(t3[0]) if len(t3) > 0 and t3[0] else 1.0
                src = _safe_float(t3[1]) if len(t3) > 1 and t3[1] else 0.0
            if len(valid_cards) > 4:
                t4 = _card_tokens(valid_cards[4])
                r00 = _safe_float(t4[0]) if len(t4) > 0 and t4[0] else 1.0
                r45 = _safe_float(t4[1]) if len(t4) > 1 and t4[1] else 1.0
                r90 = _safe_float(t4[2]) if len(t4) > 2 and t4[2] else 1.0

        # Default injections per hm_read_mat32.F lines 135-147:
        if rhor == 0.0:
            rhor = rho0
        if nu == 0.5:
            nu = 0.499999
        if r00 == 0.0:
            r00 = 1.0
        if r45 == 0.0:
            r45 = 1.0
        if r90 == 0.0:
            r90 = 1.0
        if sigy == 0.0:
            sigy = 1.0e30
        if hard == 0.0:
            hard = 1.0
        if eps == 0.0:
            eps = 1.0e30
        if sig == 0.0:
            sig = 1.0e30
        if src == 0.0:
            srp = 1.0

        # Anisotropic Hill coefficients (hm_read_mat32.F lines 157-168)
        r = 0.25 * (r00 + 2.0 * r45 + r90)
        h = r / (1.0 + r) if (1.0 + r) != 0.0 else 0.5
        a11 = h * (1.0 + 1.0 / r00) if r00 != 0.0 else 1.0
        a22 = h * (1.0 + 1.0 / r90) if r90 != 0.0 else 1.0
        a1122 = 2.0 * h
        a12 = 2.0 * h * (r45 + 0.5) * (1.0 / r00 + 1.0 / r90) if (r00 != 0.0 and r90 != 0.0) else 1.0

        # Elastic plane stress moduli
        a1_mod = e / (1.0 - nu ** 2) if (1.0 - nu ** 2) != 0.0 else e
        a2_mod = nu * a1_mod
        g_mod = e / (2.0 * (1.0 + nu)) if (1.0 + nu) != 0.0 else e / 2.0
        k_mod = e / (3.0 * (1.0 - 2.0 * nu)) if (1.0 - 2.0 * nu) != 0.0 else e / 3.0
        c_sound = (a1_mod / max(rho0, 1.0e-20)) ** 0.5 if rho0 > 0.0 else max(a1_mod, 0.0) ** 0.5

        mat = MatLaw32(
            id=mat_id,
            rho0=rho0,
            rhor=rhor,
            e=e,
            nu=nu,
            sigy=sigy,
            beta=beta,
            hard=hard,
            eps=eps,
            sig=sig,
            srp=srp,
            src=src,
            r00=r00,
            r45=r45,
            r90=r90,
            title=title,
        )
        model.mat_law32s[mat_id] = mat

        params: Dict[str, Any] = {
            "rho": rho0, "rho0": rho0, "rhor": rhor, "Refer_Rho": rhor,
            "E": e, "MAT_E": e, "young": e,
            "nu": nu, "NU": nu, "MAT_NU": nu,
            "G": g_mod, "K": k_mod, "A1": a1_mod, "A2": a2_mod,
            "A": sigy, "a": sigy, "sigy": sigy, "SIGY": sigy, "MAT_SIGY": sigy,
            "B": beta, "b": beta, "beta": beta, "BETA": beta, "epsilon_0": beta, "EPSILON_0": beta, "MAT_BETA": beta,
            "n": hard, "N": hard, "hard": hard, "HARD": hard, "MAT_HARD": hard,
            "eps": eps, "EPS": eps, "eps_max": eps, "EPS_max": eps, "eps_p_max": eps, "MAT_EPS": eps,
            "sig": sig, "SIG": sig, "sig_max": sig, "SIGMA_max": sig, "MAT_SIG": sig,
            "srp": srp, "SRP": srp, "eps0": srp, "eps_dot_0": srp, "EPS_DOT_0": srp, "MAT_SRP": srp,
            "src": src, "SRC": src, "m": src, "M": src, "MAT_SRC": src,
            "r00": r00, "R00": r00, "r_00": r00, "MAT_R00": r00,
            "r45": r45, "R45": r45, "r_45": r45, "MAT_R45": r45,
            "r90": r90, "R90": r90, "r_90": r90, "MAT_R90": r90,
            "A11": a11, "A22": a22, "A1122": a1122, "A12": a12,
            "c": c_sound,
        }

        model.materials[mat_id] = Material(
            id=mat_id,
            law=32,
            rho0=rho0,
            title=title,
            law_name="LAW32",
            params=params,
        )
    except ValueError as err:
        log.error(f"/MAT/LAW32/{mat_id}: malformed numeric input ({err})", block.source)




read_mat_hill = read_mat_law32




def read_mat_law37(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW37`` or ``/MAT/BIQUAD`` / ``/MAT/BIPHAS`` (M187): Biquadratic anisotropic yield criterion / Two-phase liquid-gas fluid model."""
    from ...model.entities import MatLaw37
    from ..mat_reader import InactiveMaterial
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW37/{mat_id}: missing data card", block.source)
        return

    first_raw = valid_cards[0].raw
    if hasattr(block, "action") and block.action:
        lawname = block.action.upper()
    elif len(block.parts) > 1 and block.parts[0].upper() == "MAT":
        lawname = block.parts[1].upper()
    elif block.parts:
        lawname = block.parts[0].upper()
    else:
        lawname = ""
    if lawname.startswith("MAT_"):
        lawname = lawname[4:]
    is_biphas = lawname in ("BIPHAS", "BIPHASIC", "LAW37_BIPHAS")
    is_biquad = lawname in ("BIQUAD", "BANABIC", "LAW37_BIQUAD")

    has_density_card = False
    if not is_biquad and not is_biphas:
        # Generic /MAT/LAW37: disambiguate between BIQUAD (anisotropic metal, 6/8 tokens)
        # and BIPHAS (two-phase fluid, 1-3 tokens card 0, 5 tokens card 1 & 2)
        t0 = valid_cards[0].tokens()
        t1 = valid_cards[1].tokens() if len(valid_cards) > 1 else []
        if len(t1) >= 6 or (len(t0) >= 6 and len(t1) != 5):
            is_biquad = True
        else:
            is_biphas = True
            if len(valid_cards) >= 3 and len(t0) <= 3:
                has_density_card = True
            else:
                has_density_card = False
    elif is_biphas:
        if len(valid_cards) >= 3 and len(valid_cards[0].tokens()) <= 3:
            has_density_card = True
        else:
            has_density_card = False

    if is_biphas:
        # Two-phase fluid format (BIPHAS / matl37_biphas.cfg)
        isolver_val = 1
        if has_density_card:
            card0_raw = valid_cards[0].raw
            tokens0 = valid_cards[0].tokens()
            if len(card0_raw) >= 60:
                f_rho = [card0_raw[:20].strip(), card0_raw[20:40].strip(), card0_raw[40:60].strip()]
            elif len(card0_raw) >= 40:
                f_rho = [card0_raw[:20].strip(), card0_raw[20:40].strip()]
            elif block.fixed:
                f_rho = valid_cards[0].cut("MAT_LAW37_1")
            else:
                f_rho = tokens0

            rho_init = _safe_float(f_rho[0]) if len(f_rho) > 0 and f_rho[0] else 0.0
            rhor_psh = _safe_float(f_rho[1]) if len(f_rho) > 1 and f_rho[1] else 0.0
            if len(f_rho) > 2 and f_rho[2]:
                try:
                    isolver_val = int(float(f_rho[2]))
                except ValueError:
                    isolver_val = 1
            elif len(tokens0) > 2:
                try:
                    isolver_val = int(float(tokens0[2]))
                except ValueError:
                    isolver_val = 1

            card_l = valid_cards[1]
            card_g = valid_cards[2]
        else:
            rho_init = 0.0
            rhor_psh = 0.0
            isolver_val = 1
            card_l = valid_cards[0]
            card_g = valid_cards[1]

        if len(card_l.raw) >= 40:
            f0 = [card_l.raw[i:i+20].strip() for i in range(0, min(len(card_l.raw), 100), 20)]
        else:
            f0 = card_l.tokens()

        if len(card_g.raw) >= 40:
            f1 = [card_g.raw[i:i+20].strip() for i in range(0, min(len(card_g.raw), 100), 20)]
        else:
            f1 = card_g.tokens()

        rho_l0 = _safe_float(f0[0]) if len(f0) > 0 else 0.0
        c_l = _safe_float(f0[1]) if len(f0) > 1 else 0.0
        alpha_l = _safe_float(f0[2]) if len(f0) > 2 else 0.0
        nu_l = _safe_float(f0[3]) if len(f0) > 3 else 0.0
        nu_vol_l = _safe_float(f0[4]) if len(f0) > 4 else 0.0

        rho_g0 = _safe_float(f1[0]) if len(f1) > 0 else 0.0
        gamma = _safe_float(f1[1]) if len(f1) > 1 else 0.0
        p0 = _safe_float(f1[2]) if len(f1) > 2 else 0.0
        nu_g = _safe_float(f1[3]) if len(f1) > 3 else 0.0
        nu_vol_g = _safe_float(f1[4]) if len(f1) > 4 else 0.0

        if rho_init > 0.0:
            rho0 = rho_init
        else:
            rho0 = rho_l0 * alpha_l + (1.0 - alpha_l) * rho_g0 if (rho_l0 > 0 or rho_g0 > 0) else 0.0

        mat = MatLaw37(
            id=mat_id, rho0=rho0, e=c_l, nu=nu_l, a=0.0, b=0.0, n=0.0,
            c1=0.0, c2=0.0, c3=0.0, c4=0.0, c5=0.0, c6=0.0, c7=0.0, c8=0.0, p=0.0, q=0.0,
            title=title
        )
        model.mat_law37s[mat_id] = mat
        model.materials[mat_id] = Material(
            id=mat_id, law=37, rho0=rho0, title=title, law_name="LAW37",
            params={
                "rho_l0": rho_l0, "Lqud_Rho_l": rho_l0, "RHO_l0": rho_l0,
                "c_l": c_l, "C_l": c_l, "bulk_l": c_l,
                "alpha1": alpha_l, "ALPHA1": alpha_l, "Alpha_l": alpha_l, "ALPHA_l": alpha_l,
                "nu_l": nu_l, "Nu_l": nu_l, "NU_l": nu_l,
                "nu_vol_l": nu_vol_l, "Bulk_Ratio_l": nu_vol_l, "Nu_vol_l": nu_vol_l, "NU_VOL_l": nu_vol_l,
                "rho_g0": rho_g0, "Lqud_Rho_g": rho_g0, "RHO_G0": rho_g0,
                "gamma_g": gamma, "gamma": gamma, "Lqud_Gamma_bulk": gamma, "GAMMA": gamma,
                "p0_g": p0, "p0": p0, "Lqud_P0": p0, "P0": p0,
                "nu_g": nu_g, "Nu_g": nu_g, "NU_g": nu_g,
                "nu_vol_g": nu_vol_g, "Bulk_Ratio_g": nu_vol_g, "Nu_vol_g": nu_vol_g, "NU_VOL_g": nu_vol_g,
                "rho": rho0, "rho0": rho0, "rhor": rhor_psh, "pshift": rhor_psh, "psh": rhor_psh,
                "isolver": isolver_val,
                "E": c_l if c_l > 0 else 200e9, "nu": nu_l if 0.0 <= nu_l < 0.5 else 0.3,
            }
        )
        return

    # Standard BIQUAD format
    rho0, e, nu, a, b, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    c1, c2, c3, c4, c5, c6, c7, c8 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    p, q = 0.0, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            f0 = valid_cards[0].cut("MAT_BIQUAD_1")
            rho0 = _safe_float(f0[0]) if len(f0) > 0 else 0.0
            e = _safe_float(f0[1]) if len(f0) > 1 else 0.0
            nu = _safe_float(f0[2]) if len(f0) > 2 else 0.0
            a = _safe_float(f0[3]) if len(f0) > 3 else 0.0
            b = _safe_float(f0[4]) if len(f0) > 4 else 0.0
            n = _safe_float(f0[5]) if len(f0) > 5 else 0.0
        if len(valid_cards) > 1:
            f1 = valid_cards[1].cut("MAT_BIQUAD_2")
            c1 = _safe_float(f1[0]) if len(f1) > 0 else 0.0
            c2 = _safe_float(f1[1]) if len(f1) > 1 else 0.0
            c3 = _safe_float(f1[2]) if len(f1) > 2 else 0.0
            c4 = _safe_float(f1[3]) if len(f1) > 3 else 0.0
            c5 = _safe_float(f1[4]) if len(f1) > 4 else 0.0
            c6 = _safe_float(f1[5]) if len(f1) > 5 else 0.0
            c7 = _safe_float(f1[6]) if len(f1) > 6 else 0.0
            c8 = _safe_float(f1[7]) if len(f1) > 7 else 0.0
        if len(valid_cards) > 2:
            f2 = valid_cards[2].cut("MAT_BIQUAD_3")
            p = _safe_float(f2[0]) if len(f2) > 0 else 0.0
            q = _safe_float(f2[1]) if len(f2) > 1 else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            e = _safe_float(t0[1]) if len(t0) > 1 else 0.0
            nu = _safe_float(t0[2]) if len(t0) > 2 else 0.0
            a = _safe_float(t0[3]) if len(t0) > 3 else 0.0
            b = _safe_float(t0[4]) if len(t0) > 4 else 0.0
            n = _safe_float(t0[5]) if len(t0) > 5 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            c1 = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            c2 = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            c3 = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            c4 = _safe_float(t1[3]) if len(t1) > 3 else 0.0
            c5 = _safe_float(t1[4]) if len(t1) > 4 else 0.0
            c6 = _safe_float(t1[5]) if len(t1) > 5 else 0.0
            c7 = _safe_float(t1[6]) if len(t1) > 6 else 0.0
            c8 = _safe_float(t1[7]) if len(t1) > 7 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            p = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            q = _safe_float(t2[1]) if len(t2) > 1 else 0.0

    mat = MatLaw37(
        id=mat_id, rho0=rho0, e=e, nu=nu, a=a, b=b, n=n,
        c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6, c7=c7, c8=c8, p=p, q=q, title=title
    )
    model.mat_law37s[mat_id] = mat
    model.materials[mat_id] = InactiveMaterial(
        id=mat_id, law=37, rho0=rho0, title=title, law_name="LAW37",
        params={
            "E": e if e > 0 else 200e9, "nu": nu if 0.0 <= nu < 0.5 else 0.3, "a": a, "b": b, "n": n,
            "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5, "c6": c6, "c7": c7, "c8": c8,
            "p": p, "q": q, "rho": rho0
        }
    )




# ----------------------------------------------------------------------------
# M188: Composite, Honeycomb, Concrete Damage & Advanced Shell/Solid Props
# ----------------------------------------------------------------------------

def read_mat_law12(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW12`` or ``/MAT/3PARBI`` / ``/MAT/3D_COMP`` / ``/MAT/RAGAB`` (M188): 3D composite Drucker-Prager."""
    from ...model.entities import MatLaw12
    from ..mat_reader import InactiveMaterial
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW12/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    e11, e22, e33 = 0.0, 0.0, 0.0
    nu12, nu23, nu31 = 0.0, 0.0, 0.0
    g12, g23, g31 = 0.0, 0.0, 0.0
    sig_t1, sig_t2, sig_t3, delta = 0.0, 0.0, 0.0, 0.05
    b, n, fmax, wplaref = 0.0, 1.0, 1.0e10, 1.0
    sig_1yt, sig_2yt, sig_1yc, sig_2yc = 0.0, 0.0, 0.0, 0.0
    sig_12yt, sig_12yc, sig_23yt, sig_23yc = 0.0, 0.0, 0.0, 0.0
    sig_3yt, sig_3yc, sig_13yt, sig_13yc = 0.0, 0.0, 0.0, 0.0
    alpha, efib, c, eps0 = 0.0, 0.0, 0.0, 0.0
    icc = 1

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW12_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW12_2")
            e11 = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            e22 = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            e33 = _safe_float(c1[2]) if len(c1) > 2 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW12_3")
            nu12 = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            nu23 = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            nu31 = _safe_float(c2[2]) if len(c2) > 2 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW12_4")
            g12 = _safe_float(c3[0]) if len(c3) > 0 else 0.0
            g23 = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            g31 = _safe_float(c3[2]) if len(c3) > 2 else 0.0
        if len(valid_cards) > 4:
            c4 = valid_cards[4].cut("MAT_LAW12_5")
            sig_t1 = _safe_float(c4[0]) if len(c4) > 0 else 0.0
            sig_t2 = _safe_float(c4[1]) if len(c4) > 1 else 0.0
            sig_t3 = _safe_float(c4[2]) if len(c4) > 2 else 0.0
            delta = _safe_float(c4[3]) if len(c4) > 3 else 0.0
        if len(valid_cards) > 5:
            c5 = valid_cards[5].cut("MAT_LAW12_6")
            b = _safe_float(c5[0]) if len(c5) > 0 else 0.0
            n = _safe_float(c5[1]) if len(c5) > 1 else 0.0
            fmax = _safe_float(c5[2]) if len(c5) > 2 else 0.0
            wplaref = _safe_float(c5[3]) if len(c5) > 3 and c5[3].strip() else 1.0
        if len(valid_cards) > 6:
            c6 = valid_cards[6].cut("MAT_LAW12_7")
            sig_1yt = _safe_float(c6[0]) if len(c6) > 0 else 0.0
            sig_2yt = _safe_float(c6[1]) if len(c6) > 1 else 0.0
            sig_1yc = _safe_float(c6[2]) if len(c6) > 2 else 0.0
            sig_2yc = _safe_float(c6[3]) if len(c6) > 3 else 0.0
        if len(valid_cards) > 7:
            c7 = valid_cards[7].cut("MAT_LAW12_8")
            sig_12yt = _safe_float(c7[0]) if len(c7) > 0 else 0.0
            sig_12yc = _safe_float(c7[1]) if len(c7) > 1 else 0.0
            sig_23yt = _safe_float(c7[2]) if len(c7) > 2 else 0.0
            sig_23yc = _safe_float(c7[3]) if len(c7) > 3 else 0.0
        if len(valid_cards) > 8:
            c8 = valid_cards[8].cut("MAT_LAW12_9")
            sig_3yt = _safe_float(c8[0]) if len(c8) > 0 else 0.0
            sig_3yc = _safe_float(c8[1]) if len(c8) > 1 else 0.0
            sig_13yt = _safe_float(c8[2]) if len(c8) > 2 else 0.0
            sig_13yc = _safe_float(c8[3]) if len(c8) > 3 else 0.0
        if len(valid_cards) > 9:
            c9 = valid_cards[9].cut("MAT_LAW12_10")
            alpha = _safe_float(c9[0]) if len(c9) > 0 else 0.0
            efib = _safe_float(c9[1]) if len(c9) > 1 else 0.0
            c = _safe_float(c9[2]) if len(c9) > 2 else 0.0
            eps0 = _safe_float(c9[3]) if len(c9) > 3 else 0.0
            icc = _safe_int(c9[4]) if len(c9) > 4 else 0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            e11 = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            e22 = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            e33 = _safe_float(t1[2]) if len(t1) > 2 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            nu12 = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            nu23 = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            nu31 = _safe_float(t2[2]) if len(t2) > 2 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            g12 = _safe_float(t3[0]) if len(t3) > 0 else 0.0
            g23 = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            g31 = _safe_float(t3[2]) if len(t3) > 2 else 0.0
        if len(valid_cards) > 4:
            t4 = valid_cards[4].tokens()
            sig_t1 = _safe_float(t4[0]) if len(t4) > 0 else 0.0
            sig_t2 = _safe_float(t4[1]) if len(t4) > 1 else 0.0
            sig_t3 = _safe_float(t4[2]) if len(t4) > 2 else 0.0
            delta = _safe_float(t4[3]) if len(t4) > 3 else 0.0
        if len(valid_cards) > 5:
            t5 = valid_cards[5].tokens()
            b = _safe_float(t5[0]) if len(t5) > 0 else 0.0
            n = _safe_float(t5[1]) if len(t5) > 1 else 0.0
            fmax = _safe_float(t5[2]) if len(t5) > 2 else 0.0
            wplaref = _safe_float(t5[3]) if len(t5) > 3 and t5[3].strip() else 1.0
        if len(valid_cards) > 6:
            t6 = valid_cards[6].tokens()
            sig_1yt = _safe_float(t6[0]) if len(t6) > 0 else 0.0
            sig_2yt = _safe_float(t6[1]) if len(t6) > 1 else 0.0
            sig_1yc = _safe_float(t6[2]) if len(t6) > 2 else 0.0
            sig_2yc = _safe_float(t6[3]) if len(t6) > 3 else 0.0
        if len(valid_cards) > 7:
            t7 = valid_cards[7].tokens()
            sig_12yt = _safe_float(t7[0]) if len(t7) > 0 else 0.0
            sig_12yc = _safe_float(t7[1]) if len(t7) > 1 else 0.0
            sig_23yt = _safe_float(t7[2]) if len(t7) > 2 else 0.0
            sig_23yc = _safe_float(t7[3]) if len(t7) > 3 else 0.0
        if len(valid_cards) > 8:
            t8 = valid_cards[8].tokens()
            sig_3yt = _safe_float(t8[0]) if len(t8) > 0 else 0.0
            sig_3yc = _safe_float(t8[1]) if len(t8) > 1 else 0.0
            sig_13yt = _safe_float(t8[2]) if len(t8) > 2 else 0.0
            sig_13yc = _safe_float(t8[3]) if len(t8) > 3 else 0.0
        if len(valid_cards) > 9:
            t9 = valid_cards[9].tokens()
            alpha = _safe_float(t9[0]) if len(t9) > 0 else 0.0
            efib = _safe_float(t9[1]) if len(t9) > 1 else 0.0
            c = _safe_float(t9[2]) if len(t9) > 2 else 0.0
            eps0 = _safe_float(t9[3]) if len(t9) > 3 else 0.0
            icc = _safe_int(t9[4]) if len(t9) > 4 else 0

    # Fortran defaults (hm_read_mat12.F:154, 178-200)
    if rhor == 0.0:
        rhor = rho0
    if delta == 0.0:
        delta = 0.05
    if n == 0.0:
        n = 1.0
    if fmax == 0.0:
        fmax = 1.0e10
    if wplaref == 0.0:
        wplaref = 1.0
    if c == 0.0 and eps0 == 0.0:
        eps0 = 1.0
    if icc == 0:
        icc = 1

    law_name = "LAW12"
    if block.keyword:
        parts = block.keyword.split("/")
        if len(parts) > 2 and parts[2].strip():
            law_name = parts[2].strip()

    mat = MatLaw12(
        id=mat_id, rho0=rho0, rhor=rhor, e11=e11, e22=e22, e33=e33,
        nu12=nu12, nu23=nu23, nu31=nu31, g12=g12, g23=g23, g31=g31,
        sig_t1=sig_t1, sig_t2=sig_t2, sig_t3=sig_t3, delta=delta,
        b=b, n=n, fmax=fmax, wplaref=wplaref, sig_1yt=sig_1yt, sig_2yt=sig_2yt,
        sig_1yc=sig_1yc, sig_2yc=sig_2yc, sig_12yt=sig_12yt, sig_12yc=sig_12yc,
        sig_23yt=sig_23yt, sig_23yc=sig_23yc, sig_3yt=sig_3yt, sig_3yc=sig_3yc,
        sig_13yt=sig_13yt, sig_13yc=sig_13yc, alpha=alpha, efib=efib, c=c, eps0=eps0,
        icc=icc, title=title, law_name=law_name
    )
    model.mat_law12s[mat_id] = mat
    try:
        from ...materials.law12_comp3d import build_law12
        model.materials[mat_id] = build_law12(mat)
    except Exception:
        e_val = max(e11, e22, e33, 200e9) if max(e11, e22, e33) > 0 else 200e9
        model.materials[mat_id] = InactiveMaterial(
            id=mat_id, law=12, rho0=rho0, title=title, law_name=law_name,
            params={
                "E": e_val, "MAT_E": e_val, "nu": nu12 if 0.0 <= nu12 < 0.5 else 0.3, "MAT_NU": nu12 if 0.0 <= nu12 < 0.5 else 0.3,
                "MAT_SIGY": sig_1yt if sig_1yt > 0 else 200e6, "rho": rho0, "MAT_RHO": rho0,
                "E11": e11, "E22": e22, "E33": e33, "NU12": nu12, "NU23": nu23, "NU31": nu31,
                "G12": g12, "G23": g23, "G31": g31,
            }
        )



read_mat_3d_comp = read_mat_law12


read_mat_comp_3d = read_mat_law12


read_mat_3parbi = read_mat_law12


read_mat_ragab = read_mat_law12




def read_mat_law13(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW13`` or ``/MAT/HONEYCOMB`` / ``/MAT/RIGID`` (M188): Honeycomb crush / rigid material."""
    from ...model.entities import MatLaw13
    from ..mat_reader import InactiveMaterial
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW13/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    e, nu = 0.0, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW13_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW13_2")
            e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0

    mat = MatLaw13(id=mat_id, rho0=rho0, rhor=rhor, e=e, nu=nu, title=title)
    model.mat_law13s[mat_id] = mat
    e_val = e if e > 0 else 200e9
    model.materials[mat_id] = InactiveMaterial(
        id=mat_id, law=13, rho0=rho0, title=title, law_name="LAW13",
        params={
            "E": e_val, "MAT_E": e_val, "nu": nu if 0.0 <= nu < 0.5 else 0.3, "MAT_NU": nu if 0.0 <= nu < 0.5 else 0.3,
            "rho": rho0, "MAT_RHO": rho0
        }
    )




def read_mat_law15(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW15`` or ``/MAT/CHANG`` / ``/MAT/CHANG_CHANG`` (M188): Chang-Chang composite failure model."""
    from ...model.entities import MatLaw15
    from ..mat_reader import InactiveMaterial
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW15/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    e11, e22, nu12 = 0.0, 0.0, 0.0
    g12, g23, g31 = 0.0, 0.0, 0.0
    b, n, fmax = 0.0, 0.0, 0.0
    wpmax, wpref = 0.0, 0.0
    ioff = 0
    sig_1yt, sig_2yt, sig_1yc, sig_2yc, alpha = 0.0, 0.0, 0.0, 0.0, 0.0
    sig_12yc, sig_12yt, c, eps_dot_0 = 0.0, 0.0, 0.0, 0.0
    icc = 0
    beta, tmax, s1, s2, s12 = 0.0, 0.0, 0.0, 0.0, 0.0
    fsmooth = 0
    fcut, c1, c2 = 0.0, 0.0, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            cut0 = valid_cards[0].cut("MAT_LAW15_1")
            rho0 = _safe_float(cut0[0]) if len(cut0) > 0 else 0.0
            rhor = _safe_float(cut0[1]) if len(cut0) > 1 else 0.0
        if len(valid_cards) > 1:
            cut1 = valid_cards[1].cut("MAT_LAW15_2")
            e11 = _safe_float(cut1[0]) if len(cut1) > 0 else 0.0
            e22 = _safe_float(cut1[1]) if len(cut1) > 1 else 0.0
            nu12 = _safe_float(cut1[2]) if len(cut1) > 2 else 0.0
        if len(valid_cards) > 2:
            cut2 = valid_cards[2].cut("MAT_LAW15_3")
            g12 = _safe_float(cut2[0]) if len(cut2) > 0 else 0.0
            g23 = _safe_float(cut2[1]) if len(cut2) > 1 else 0.0
            g31 = _safe_float(cut2[2]) if len(cut2) > 2 else 0.0
        if len(valid_cards) > 3:
            cut3 = valid_cards[3].cut("MAT_LAW15_4")
            b = _safe_float(cut3[0]) if len(cut3) > 0 else 0.0
            n = _safe_float(cut3[1]) if len(cut3) > 1 else 0.0
            fmax = _safe_float(cut3[2]) if len(cut3) > 2 else 0.0
        if len(valid_cards) > 4:
            cut4 = valid_cards[4].cut("MAT_LAW15_5")
            wpmax = _safe_float(cut4[0]) if len(cut4) > 0 else 0.0
            wpref = _safe_float(cut4[1]) if len(cut4) > 1 else 0.0
            ioff = _safe_int(cut4[2]) if len(cut4) > 2 else 0
        if len(valid_cards) > 5:
            cut5 = valid_cards[5].cut("MAT_LAW15_6")
            sig_1yt = _safe_float(cut5[0]) if len(cut5) > 0 else 0.0
            sig_2yt = _safe_float(cut5[1]) if len(cut5) > 1 else 0.0
            sig_1yc = _safe_float(cut5[2]) if len(cut5) > 2 else 0.0
            sig_2yc = _safe_float(cut5[3]) if len(cut5) > 3 else 0.0
            alpha = _safe_float(cut5[4]) if len(cut5) > 4 else 0.0
        if len(valid_cards) > 6:
            cut6 = valid_cards[6].cut("MAT_LAW15_7")
            sig_12yc = _safe_float(cut6[0]) if len(cut6) > 0 else 0.0
            sig_12yt = _safe_float(cut6[1]) if len(cut6) > 1 else 0.0
            c = _safe_float(cut6[2]) if len(cut6) > 2 else 0.0
            eps_dot_0 = _safe_float(cut6[3]) if len(cut6) > 3 else 0.0
            icc = _safe_int(cut6[4]) if len(cut6) > 4 else 0
        if len(valid_cards) > 7:
            cut7 = valid_cards[7].cut("MAT_LAW15_8")
            beta = _safe_float(cut7[0]) if len(cut7) > 0 else 0.0
            tmax = _safe_float(cut7[1]) if len(cut7) > 1 else 0.0
            s1 = _safe_float(cut7[2]) if len(cut7) > 2 else 0.0
            s2 = _safe_float(cut7[3]) if len(cut7) > 3 else 0.0
            s12 = _safe_float(cut7[4]) if len(cut7) > 4 else 0.0
        if len(valid_cards) > 8:
            cut8 = valid_cards[8].cut("MAT_LAW15_9")
            fsmooth = _safe_int(cut8[0]) if len(cut8) > 0 else 0
            fcut = _safe_float(cut8[1]) if len(cut8) > 1 else 0.0
            c1 = _safe_float(cut8[2]) if len(cut8) > 2 else 0.0
            c2 = _safe_float(cut8[3]) if len(cut8) > 3 else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            e11 = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            e22 = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            nu12 = _safe_float(t1[2]) if len(t1) > 2 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            g12 = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            g23 = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            g31 = _safe_float(t2[2]) if len(t2) > 2 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            b = _safe_float(t3[0]) if len(t3) > 0 else 0.0
            n = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            fmax = _safe_float(t3[2]) if len(t3) > 2 else 0.0
        if len(valid_cards) > 4:
            t4 = valid_cards[4].tokens()
            wpmax = _safe_float(t4[0]) if len(t4) > 0 else 0.0
            wpref = _safe_float(t4[1]) if len(t4) > 1 else 0.0
            ioff = _safe_int(t4[2]) if len(t4) > 2 else 0
        if len(valid_cards) > 5:
            t5 = valid_cards[5].tokens()
            sig_1yt = _safe_float(t5[0]) if len(t5) > 0 else 0.0
            sig_2yt = _safe_float(t5[1]) if len(t5) > 1 else 0.0
            sig_1yc = _safe_float(t5[2]) if len(t5) > 2 else 0.0
            sig_2yc = _safe_float(t5[3]) if len(t5) > 3 else 0.0
            alpha = _safe_float(t5[4]) if len(t5) > 4 else 0.0
        if len(valid_cards) > 6:
            t6 = valid_cards[6].tokens()
            sig_12yc = _safe_float(t6[0]) if len(t6) > 0 else 0.0
            sig_12yt = _safe_float(t6[1]) if len(t6) > 1 else 0.0
            c = _safe_float(t6[2]) if len(t6) > 2 else 0.0
            eps_dot_0 = _safe_float(t6[3]) if len(t6) > 3 else 0.0
            icc = _safe_int(t6[4]) if len(t6) > 4 else 0
        if len(valid_cards) > 7:
            t7 = valid_cards[7].tokens()
            beta = _safe_float(t7[0]) if len(t7) > 0 else 0.0
            tmax = _safe_float(t7[1]) if len(t7) > 1 else 0.0
            s1 = _safe_float(t7[2]) if len(t7) > 2 else 0.0
            s2 = _safe_float(t7[3]) if len(t7) > 3 else 0.0
            s12 = _safe_float(t7[4]) if len(t7) > 4 else 0.0
        if len(valid_cards) > 8:
            t8 = valid_cards[8].tokens()
            fsmooth = _safe_int(t8[0]) if len(t8) > 0 else 0
            fcut = _safe_float(t8[1]) if len(t8) > 1 else 0.0
            c1 = _safe_float(t8[2]) if len(t8) > 2 else 0.0
            c2 = _safe_float(t8[3]) if len(t8) > 3 else 0.0

    law_name = "LAW15"
    if len(block.parts) > 1 and block.parts[1].upper() in ("LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG", "CHANG_CHANG"):
        law_name = block.parts[1].upper()

    mat = MatLaw15(
        id=mat_id, rho0=rho0, rhor=rhor, e11=e11, e22=e22, nu12=nu12,
        g12=g12, g23=g23, g31=g31, b=b, n=n, fmax=fmax,
        wpmax=wpmax, wpref=wpref, ioff=ioff,
        sig_1yt=sig_1yt, sig_2yt=sig_2yt, sig_1yc=sig_1yc, sig_2yc=sig_2yc, alpha=alpha,
        sig_12yc=sig_12yc, sig_12yt=sig_12yt, c=c, eps_dot_0=eps_dot_0, icc=icc,
        beta=beta, tmax=tmax, s1=s1, s2=s2, s12=s12, fsmooth=fsmooth, fcut=fcut,
        c1=c1, c2=c2, title=title, law_name=law_name
    )
    model.mat_law15s[mat_id] = mat
    try:
        from ...materials.law15_chang import build_law15
        model.materials[mat_id] = build_law15(mat)
    except Exception:
        e_val = max(e11, e22, 200e9) if max(e11, e22) > 0 else 200e9
        model.materials[mat_id] = InactiveMaterial(
            id=mat_id, law=15, rho0=rho0, title=title, law_name=law_name,
            params={
                "E": e_val, "MAT_E": e_val, "nu": nu12 if 0.0 <= nu12 < 0.5 else 0.3, "MAT_NU": nu12 if 0.0 <= nu12 < 0.5 else 0.3,
                "MAT_SIGY": sig_1yt if sig_1yt > 0 else 200e6, "rho": rho0, "MAT_RHO": rho0,
                "E11": e11, "E22": e22, "NU12": nu12, "G12": g12, "G23": g23, "G31": g31,
            }
        )




read_mat_chang = read_mat_law15


read_mat_plas_aniso = read_mat_law15


read_mat_comp_chang = read_mat_law15




def read_mat_law18(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW18`` or ``/MAT/CONCR_DRA`` / ``/MAT/DRAGON`` / ``/MAT/THERM`` (M188): Concrete damage / thermal model."""
    from ...model.entities import MatLaw18
    from ..mat_reader import InactiveMaterial
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW18/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    spheat, a, b = 0.0, 0.0, 0.0
    fct_idt, t0, scale = 0, 0.0, 0.0
    fct_idsph, fct_idas = 0, 0
    fscalesph, fscalee, fscalek = 0.0, 0.0, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW18_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW18_2")
            spheat = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            a = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            b = _safe_float(c1[2]) if len(c1) > 2 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW18_3")
            fct_idt = _safe_int(c2[0]) if len(c2) > 0 else 0
            t0 = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            scale = _safe_float(c2[2]) if len(c2) > 2 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW18_4")
            fct_idsph = _safe_int(c3[0]) if len(c3) > 0 else 0
            fct_idas = _safe_int(c3[1]) if len(c3) > 1 else 0
            fscalesph = _safe_float(c3[2]) if len(c3) > 2 else 0.0
            fscalee = _safe_float(c3[3]) if len(c3) > 3 else 0.0
            fscalek = _safe_float(c3[4]) if len(c3) > 4 else 0.0
    else:
        if len(valid_cards) > 0:
            t0_tok = valid_cards[0].tokens()
            rho0 = _safe_float(t0_tok[0]) if len(t0_tok) > 0 else 0.0
            rhor = _safe_float(t0_tok[1]) if len(t0_tok) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            spheat = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            a = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            b = _safe_float(t1[2]) if len(t1) > 2 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            fct_idt = _safe_int(t2[0]) if len(t2) > 0 else 0
            t0 = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            scale = _safe_float(t2[2]) if len(t2) > 2 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            fct_idsph = _safe_int(t3[0]) if len(t3) > 0 else 0
            fct_idas = _safe_int(t3[1]) if len(t3) > 1 else 0
            fscalesph = _safe_float(t3[2]) if len(t3) > 2 else 0.0
            fscalee = _safe_float(t3[3]) if len(t3) > 3 else 0.0
            fscalek = _safe_float(t3[4]) if len(t3) > 4 else 0.0

    mat = MatLaw18(
        id=mat_id, rho0=rho0, rhor=rhor, spheat=spheat, a=a, b=b,
        fct_idt=fct_idt, t0=t0, scale=scale, fct_idsph=fct_idsph,
        fct_idas=fct_idas, fscalesph=fscalesph, fscalee=fscalee, fscalek=fscalek,
        title=title
    )
    model.mat_law18s[mat_id] = mat
    model.materials[mat_id] = InactiveMaterial(
        id=mat_id, law=18, rho0=rho0, title=title, law_name="LAW18",
        params={
            "E": 200e9, "MAT_E": 200e9, "nu": 0.3, "MAT_NU": 0.3,
            "rho": rho0, "MAT_RHO": rho0, "SPHEAT": spheat, "A": a, "B": b,
        }
    )




def read_mat_law22(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW22`` or ``/MAT/DAMA`` / ``/MAT/PLAS_DAMA`` (M545): Damaged elasto-plastic material model."""
    from ...model.entities import MatLaw22
    from ..mat_reader import InactiveMaterial
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#") and not c.raw.strip().startswith("$")]
    if not valid_cards:
        log.error(f"/MAT/LAW22/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    e, nu = 0.0, 0.0
    a, b, n = 0.0, 0.0, 1.0
    eps_max, sig_max = 1.0e30, 1.0e30
    c, eps_dot_0 = 0.0, 0.0
    icc = 1
    eps_dam, e_tan = 0.15, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW22_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW22_2")
            e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW22_3")
            a = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            b = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            n = _safe_float(c2[2]) if len(c2) > 2 and c2[2] != "" else 1.0
            eps_max = _safe_float(c2[3]) if len(c2) > 3 and c2[3] != "" else 1.0e30
            sig_max = _safe_float(c2[4]) if len(c2) > 4 and c2[4] != "" else 1.0e30
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW22_4")
            c = _safe_float(c3[0]) if len(c3) > 0 else 0.0
            eps_dot_0 = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            icc = _safe_int(c3[2]) if len(c3) > 2 and c3[2] != "" else 1
        if len(valid_cards) > 4:
            c4 = valid_cards[4].cut("MAT_LAW22_5")
            eps_dam = _safe_float(c4[0]) if len(c4) > 0 and c4[0] != "" else 0.15
            e_tan = _safe_float(c4[1]) if len(c4) > 1 and c4[1] != "" else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            a = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            b = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            n = _safe_float(t2[2]) if len(t2) > 2 else 1.0
            eps_max = _safe_float(t2[3]) if len(t2) > 3 else 1.0e30
            sig_max = _safe_float(t2[4]) if len(t2) > 4 else 1.0e30
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            c = _safe_float(t3[0]) if len(t3) > 0 else 0.0
            eps_dot_0 = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            icc = _safe_int(t3[2]) if len(t3) > 2 else 1
        if len(valid_cards) > 4:
            t4 = valid_cards[4].tokens()
            eps_dam = _safe_float(t4[0]) if len(t4) > 0 else 0.15
            e_tan = _safe_float(t4[1]) if len(t4) > 1 else 0.0

    law_name = "LAW22"
    if len(block.parts) > 1 and block.parts[1].upper() in ("LAW22", "DAMA", "PLAS_DAMA", "MAT_LAW22", "MAT_DAMA", "MAT_PLAS_DAMA"):
        law_name = block.parts[1].upper()

    mat = MatLaw22(
        id=mat_id, rho0=rho0, rhor=rhor, e=e, nu=nu, a=a, b=b,
        n=n, eps_max=eps_max, sig_max=sig_max, c=c, eps_dot_0=eps_dot_0,
        icc=icc, eps_dam=eps_dam, e_tan=e_tan, title=title, law_name=law_name
    )
    model.mat_law22s[mat_id] = mat
    if law_name in ("PLAS_DAMA", "MAT_PLAS_DAMA"):
        model.mat_law23s[mat_id] = mat
    from ...materials.law22_dama import build_law22
    try:
        model.materials[mat_id] = build_law22(mat)
    except Exception as ex:
        log.warning(f"/MAT/LAW22/{mat_id}: build_law22 fallback ({ex})", block.source)
        e_val = e if e > 0 else 200e9
        model.materials[mat_id] = InactiveMaterial(
            id=mat_id, law=22, rho0=rho0, title=title, law_name=law_name,
            params={
                "E": e_val, "MAT_E": e_val, "nu": nu if 0.0 <= nu < 0.5 else 0.3, "MAT_NU": nu if 0.0 <= nu < 0.5 else 0.3,
                "MAT_SIGY": a if a > 0 else 200e6, "SIG_Y": a, "rho": rho0, "MAT_RHO": rho0,
                "A": a, "B": b, "n": n, "EPS_MAX": eps_max, "SIG_MAX": sig_max,
                "C": c, "EPS_DOT_0": eps_dot_0, "ICC": icc, "EPS_DAM": eps_dam, "E_TAN": e_tan,
            }
        )



read_mat_dama = read_mat_law22


read_mat_plas_dama = read_mat_law22




def read_mat_law25(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW25`` or ``/MAT/COMP_PLAS`` / ``/MAT/COMPSH`` / ``/MAT/CRASURV``: Composite plasticity model."""
    from ...model.entities import MatLaw25
    from ..mat_reader import InactiveMaterial
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    if block.fixed:
        valid_cards = [c for c in cards if not c.raw.strip().startswith("#") and not c.raw.strip().startswith("$")]
    else:
        valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#") and not c.raw.strip().startswith("$")]
    if not valid_cards:
        log.error(f"/MAT/LAW25/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    e11, e22, nu12, iform, e33 = 0.0, 0.0, 0.0, 0, 0.0
    g12, g23, g31, eps_f1, eps_f2 = 0.0, 0.0, 0.0, 0.0, 0.0
    eps_t1, eps_m1, eps_t2, eps_m2, dmax = 0.0, 0.0, 0.0, 0.0, 0.0
    wpmax, wpref = 0.0, 0.0
    ioff = 0
    b, n, fmax = 0.0, 0.0, 0.0
    sig_1yt, sig_2yt, sig_1yc, sig_2yc, alpha = 0.0, 0.0, 0.0, 0.0, 0.0
    sig_12yc, sig_12yt, c, eps_rate_0 = 0.0, 0.0, 0.0, 0.0
    icc = 0

    iflawp = 0
    b_1t, n_1t, sig_1maxt, c_1t = 0.0, 1.0, 0.0, 0.0
    eps_1t1, eps_2t1, sig_rst1, wpmax_t1 = 0.0, 0.0, 0.0, 0.0
    b_2t, n_2t, sig_2maxt, c_2t = 0.0, 1.0, 0.0, 0.0
    eps_1t2, eps_2t2, sig_rst2, wpmax_t2 = 0.0, 0.0, 0.0, 0.0
    b_1c, n_1c, sig_1maxc, c_1c = 0.0, 1.0, 0.0, 0.0
    eps_1c1, eps_2c1, sig_rsc1, wpmax_c1 = 0.0, 0.0, 0.0, 0.0
    b_2c, n_2c, sig_2maxc, c_2c = 0.0, 1.0, 0.0, 0.0
    eps_1c2, eps_2c2, sig_rsc2, wpmax_c2 = 0.0, 0.0, 0.0, 0.0
    b_12t, n_12t, sig_12maxt, c_12t = 0.0, 1.0, 0.0, 0.0
    eps_1t12, eps_2t12, sig_rst12, wpmax_t12 = 0.0, 0.0, 0.0, 0.0

    gamma_ini, gamma_max, d3max = 0.0, 0.0, 0.0
    fsmooth, fcut = 0, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW25_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW25_2")
            e11 = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            e22 = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            nu12 = _safe_float(c1[2]) if len(c1) > 2 else 0.0
            iform = _safe_int(c1[3]) if len(c1) > 3 else 0
            e33 = _safe_float(c1[5]) if len(c1) > 5 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW25_3")
            g12 = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            g23 = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            g31 = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            eps_f1 = _safe_float(c2[3]) if len(c2) > 3 else 0.0
            eps_f2 = _safe_float(c2[4]) if len(c2) > 4 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW25_4")
            eps_t1 = _safe_float(c3[0]) if len(c3) > 0 else 0.0
            eps_m1 = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            eps_t2 = _safe_float(c3[2]) if len(c3) > 2 else 0.0
            eps_m2 = _safe_float(c3[3]) if len(c3) > 3 else 0.0
            dmax = _safe_float(c3[4]) if len(c3) > 4 else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            e11 = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            e22 = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            nu12 = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            iform = _safe_int(t1[3]) if len(t1) > 3 else 0
            e33 = _safe_float(t1[4]) if len(t1) > 4 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            g12 = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            g23 = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            g31 = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            eps_f1 = _safe_float(t2[3]) if len(t2) > 3 else 0.0
            eps_f2 = _safe_float(t2[4]) if len(t2) > 4 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            eps_t1 = _safe_float(t3[0]) if len(t3) > 0 else 0.0
            eps_m1 = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            eps_t2 = _safe_float(t3[2]) if len(t3) > 2 else 0.0
            eps_m2 = _safe_float(t3[3]) if len(t3) > 3 else 0.0
            dmax = _safe_float(t3[4]) if len(t3) > 4 else 0.0

    parts_upper = [p.upper() for p in getattr(block, "parts", [])]
    kw_upper = (getattr(block, "keyword", "") or "").upper()
    is_crasurv = (
        any(p in ("CRASURV", "MAT_CRASURV", "LAW25_CRASURV") for p in parts_upper)
        or ("CRASURV" in kw_upper)
        or (iform == 1 and len(valid_cards) >= 12)
    )

    if not is_crasurv:
        # Tsai-Wu formulation (cards 4..9)
        if block.fixed:
            if len(valid_cards) > 4:
                c4 = valid_cards[4].cut("MAT_LAW25_5")
                wpmax = _safe_float(c4[0]) if len(c4) > 0 else 0.0
                wpref = _safe_float(c4[1]) if len(c4) > 1 else 0.0
                ioff = _safe_int(c4[2]) if len(c4) > 2 else 0
            if len(valid_cards) > 5:
                c5 = valid_cards[5].cut("MAT_LAW25_6")
                b = _safe_float(c5[0]) if len(c5) > 0 else 0.0
                n = _safe_float(c5[1]) if len(c5) > 1 else 0.0
                fmax = _safe_float(c5[2]) if len(c5) > 2 else 0.0
            if len(valid_cards) > 6:
                c6 = valid_cards[6].cut("MAT_LAW25_7")
                sig_1yt = _safe_float(c6[0]) if len(c6) > 0 else 0.0
                sig_2yt = _safe_float(c6[1]) if len(c6) > 1 else 0.0
                sig_1yc = _safe_float(c6[2]) if len(c6) > 2 else 0.0
                sig_2yc = _safe_float(c6[3]) if len(c6) > 3 else 0.0
                alpha = _safe_float(c6[4]) if len(c6) > 4 else 0.0
            if len(valid_cards) > 7:
                c7 = valid_cards[7].cut("MAT_LAW25_8")
                sig_12yc = _safe_float(c7[0]) if len(c7) > 0 else 0.0
                sig_12yt = _safe_float(c7[1]) if len(c7) > 1 else 0.0
                c = _safe_float(c7[2]) if len(c7) > 2 else 0.0
                eps_rate_0 = _safe_float(c7[3]) if len(c7) > 3 else 0.0
                icc = _safe_int(c7[4]) if len(c7) > 4 else 0
            if len(valid_cards) > 8:
                c8 = valid_cards[8].cut("MAT_LAW25_9")
                gamma_ini = _safe_float(c8[0]) if len(c8) > 0 else 0.0
                gamma_max = _safe_float(c8[1]) if len(c8) > 1 else 0.0
                d3max = _safe_float(c8[2]) if len(c8) > 2 else 0.0
            if len(valid_cards) > 9:
                c9 = valid_cards[9].cut("MAT_LAW25_10")
                fsmooth = _safe_int(c9[0]) if len(c9) > 0 else 0
                fcut = _safe_float(c9[1]) if len(c9) > 1 else 0.0
        else:
            if len(valid_cards) > 4:
                t4 = valid_cards[4].tokens()
                wpmax = _safe_float(t4[0]) if len(t4) > 0 else 0.0
                wpref = _safe_float(t4[1]) if len(t4) > 1 else 0.0
                ioff = _safe_int(t4[2]) if len(t4) > 2 else 0
            if len(valid_cards) > 5:
                t5 = valid_cards[5].tokens()
                b = _safe_float(t5[0]) if len(t5) > 0 else 0.0
                n = _safe_float(t5[1]) if len(t5) > 1 else 0.0
                fmax = _safe_float(t5[2]) if len(t5) > 2 else 0.0
            if len(valid_cards) > 6:
                t6 = valid_cards[6].tokens()
                sig_1yt = _safe_float(t6[0]) if len(t6) > 0 else 0.0
                sig_2yt = _safe_float(t6[1]) if len(t6) > 1 else 0.0
                sig_1yc = _safe_float(t6[2]) if len(t6) > 2 else 0.0
                sig_2yc = _safe_float(t6[3]) if len(t6) > 3 else 0.0
                alpha = _safe_float(t6[4]) if len(t6) > 4 else 0.0
            if len(valid_cards) > 7:
                t7 = valid_cards[7].tokens()
                sig_12yc = _safe_float(t7[0]) if len(t7) > 0 else 0.0
                sig_12yt = _safe_float(t7[1]) if len(t7) > 1 else 0.0
                c = _safe_float(t7[2]) if len(t7) > 2 else 0.0
                eps_rate_0 = _safe_float(t7[3]) if len(t7) > 3 else 0.0
                icc = _safe_int(t7[4]) if len(t7) > 4 else 0
            if len(valid_cards) > 8:
                t8 = valid_cards[8].tokens()
                gamma_ini = _safe_float(t8[0]) if len(t8) > 0 else 0.0
                gamma_max = _safe_float(t8[1]) if len(t8) > 1 else 0.0
                d3max = _safe_float(t8[2]) if len(t8) > 2 else 0.0
            if len(valid_cards) > 9:
                t9 = valid_cards[9].tokens()
                fsmooth = _safe_int(t9[0]) if len(t9) > 0 else 0
                fcut = _safe_float(t9[1]) if len(t9) > 1 else 0.0
    else:
        # CRASURV formulation (iform=1, cards 4..17)
        iform = 1
        if block.fixed:
            if len(valid_cards) > 4:
                c4 = valid_cards[4].cut("MAT_CRASURV_5")
                wpmax = _safe_float(c4[0]) if len(c4) > 0 else 0.0
                wpref = _safe_float(c4[1]) if len(c4) > 1 else 0.0
                ioff = _safe_int(c4[2]) if len(c4) > 2 else 0
                iflawp = _safe_int(c4[3]) if len(c4) > 3 else 0
            if len(valid_cards) > 5:
                c5 = valid_cards[5].cut("MAT_CRASURV_6")
                c = _safe_float(c5[0]) if len(c5) > 0 else 0.0
                eps_rate_0 = _safe_float(c5[1]) if len(c5) > 1 else 0.0
                alpha = _safe_float(c5[2]) if len(c5) > 2 else 0.0
                icc = _safe_int(c5[4]) if len(c5) > 4 else 0
            if len(valid_cards) > 6:
                c6 = valid_cards[6].cut("MAT_CRASURV_7")
                sig_1yt = _safe_float(c6[0]) if len(c6) > 0 else 0.0
                b_1t = _safe_float(c6[1]) if len(c6) > 1 else 0.0
                n_1t = _safe_float(c6[2]) if len(c6) > 2 else 1.0
                sig_1maxt = _safe_float(c6[3]) if len(c6) > 3 else 0.0
                c_1t = _safe_float(c6[4]) if len(c6) > 4 else 0.0
            if len(valid_cards) > 7:
                c7 = valid_cards[7].cut("MAT_CRASURV_8")
                eps_1t1 = _safe_float(c7[0]) if len(c7) > 0 else 0.0
                eps_2t1 = _safe_float(c7[1]) if len(c7) > 1 else 0.0
                sig_rst1 = _safe_float(c7[2]) if len(c7) > 2 else 0.0
                wpmax_t1 = _safe_float(c7[3]) if len(c7) > 3 else 0.0
            if len(valid_cards) > 8:
                c8 = valid_cards[8].cut("MAT_CRASURV_9")
                sig_2yt = _safe_float(c8[0]) if len(c8) > 0 else 0.0
                b_2t = _safe_float(c8[1]) if len(c8) > 1 else 0.0
                n_2t = _safe_float(c8[2]) if len(c8) > 2 else 1.0
                sig_2maxt = _safe_float(c8[3]) if len(c8) > 3 else 0.0
                c_2t = _safe_float(c8[4]) if len(c8) > 4 else 0.0
            if len(valid_cards) > 9:
                c9 = valid_cards[9].cut("MAT_CRASURV_10")
                eps_1t2 = _safe_float(c9[0]) if len(c9) > 0 else 0.0
                eps_2t2 = _safe_float(c9[1]) if len(c9) > 1 else 0.0
                sig_rst2 = _safe_float(c9[2]) if len(c9) > 2 else 0.0
                wpmax_t2 = _safe_float(c9[3]) if len(c9) > 3 else 0.0
            if len(valid_cards) > 10:
                c10 = valid_cards[10].cut("MAT_CRASURV_11")
                sig_1yc = _safe_float(c10[0]) if len(c10) > 0 else 0.0
                b_1c = _safe_float(c10[1]) if len(c10) > 1 else 0.0
                n_1c = _safe_float(c10[2]) if len(c10) > 2 else 1.0
                sig_1maxc = _safe_float(c10[3]) if len(c10) > 3 else 0.0
                c_1c = _safe_float(c10[4]) if len(c10) > 4 else 0.0
            if len(valid_cards) > 11:
                c11 = valid_cards[11].cut("MAT_CRASURV_12")
                eps_1c1 = _safe_float(c11[0]) if len(c11) > 0 else 0.0
                eps_2c1 = _safe_float(c11[1]) if len(c11) > 1 else 0.0
                sig_rsc1 = _safe_float(c11[2]) if len(c11) > 2 else 0.0
                wpmax_c1 = _safe_float(c11[3]) if len(c11) > 3 else 0.0
            if len(valid_cards) > 12:
                c12 = valid_cards[12].cut("MAT_CRASURV_13")
                sig_2yc = _safe_float(c12[0]) if len(c12) > 0 else 0.0
                b_2c = _safe_float(c12[1]) if len(c12) > 1 else 0.0
                n_2c = _safe_float(c12[2]) if len(c12) > 2 else 1.0
                sig_2maxc = _safe_float(c12[3]) if len(c12) > 3 else 0.0
                c_2c = _safe_float(c12[4]) if len(c12) > 4 else 0.0
            if len(valid_cards) > 13:
                c13 = valid_cards[13].cut("MAT_CRASURV_14")
                eps_1c2 = _safe_float(c13[0]) if len(c13) > 0 else 0.0
                eps_2c2 = _safe_float(c13[1]) if len(c13) > 1 else 0.0
                sig_rsc2 = _safe_float(c13[2]) if len(c13) > 2 else 0.0
                wpmax_c2 = _safe_float(c13[3]) if len(c13) > 3 else 0.0
            if len(valid_cards) > 14:
                c14 = valid_cards[14].cut("MAT_CRASURV_15")
                sig_12yt = _safe_float(c14[0]) if len(c14) > 0 else 0.0
                b_12t = _safe_float(c14[1]) if len(c14) > 1 else 0.0
                n_12t = _safe_float(c14[2]) if len(c14) > 2 else 1.0
                sig_12maxt = _safe_float(c14[3]) if len(c14) > 3 else 0.0
                c_12t = _safe_float(c14[4]) if len(c14) > 4 else 0.0
            if len(valid_cards) > 15:
                c15 = valid_cards[15].cut("MAT_CRASURV_16")
                eps_1t12 = _safe_float(c15[0]) if len(c15) > 0 else 0.0
                eps_2t12 = _safe_float(c15[1]) if len(c15) > 1 else 0.0
                sig_rst12 = _safe_float(c15[2]) if len(c15) > 2 else 0.0
                wpmax_t12 = _safe_float(c15[3]) if len(c15) > 3 else 0.0
            if len(valid_cards) > 16:
                c16 = valid_cards[16].cut("MAT_CRASURV_17")
                gamma_ini = _safe_float(c16[0]) if len(c16) > 0 else 0.0
                gamma_max = _safe_float(c16[1]) if len(c16) > 1 else 0.0
                d3max = _safe_float(c16[2]) if len(c16) > 2 else 0.0
            if len(valid_cards) > 17:
                c17 = valid_cards[17].cut("MAT_CRASURV_18")
                fsmooth = _safe_int(c17[0]) if len(c17) > 0 else 0
                fcut = _safe_float(c17[1]) if len(c17) > 1 else 0.0
        else:
            if len(valid_cards) > 4:
                t4 = valid_cards[4].tokens()
                wpmax = _safe_float(t4[0]) if len(t4) > 0 else 0.0
                wpref = _safe_float(t4[1]) if len(t4) > 1 else 0.0
                ioff = _safe_int(t4[2]) if len(t4) > 2 else 0
                iflawp = _safe_int(t4[3]) if len(t4) > 3 else 0
            if len(valid_cards) > 5:
                t5 = valid_cards[5].tokens()
                c = _safe_float(t5[0]) if len(t5) > 0 else 0.0
                eps_rate_0 = _safe_float(t5[1]) if len(t5) > 1 else 0.0
                alpha = _safe_float(t5[2]) if len(t5) > 2 else 0.0
                if len(t5) > 3:
                    icc = _safe_int(t5[3])
            if len(valid_cards) > 6:
                t6 = valid_cards[6].tokens()
                sig_1yt = _safe_float(t6[0]) if len(t6) > 0 else 0.0
                b_1t = _safe_float(t6[1]) if len(t6) > 1 else 0.0
                n_1t = _safe_float(t6[2]) if len(t6) > 2 else 1.0
                sig_1maxt = _safe_float(t6[3]) if len(t6) > 3 else 0.0
                c_1t = _safe_float(t6[4]) if len(t6) > 4 else 0.0
            if len(valid_cards) > 7:
                t7 = valid_cards[7].tokens()
                eps_1t1 = _safe_float(t7[0]) if len(t7) > 0 else 0.0
                eps_2t1 = _safe_float(t7[1]) if len(t7) > 1 else 0.0
                sig_rst1 = _safe_float(t7[2]) if len(t7) > 2 else 0.0
                wpmax_t1 = _safe_float(t7[3]) if len(t7) > 3 else 0.0
            if len(valid_cards) > 8:
                t8 = valid_cards[8].tokens()
                sig_2yt = _safe_float(t8[0]) if len(t8) > 0 else 0.0
                b_2t = _safe_float(t8[1]) if len(t8) > 1 else 0.0
                n_2t = _safe_float(t8[2]) if len(t8) > 2 else 1.0
                sig_2maxt = _safe_float(t8[3]) if len(t8) > 3 else 0.0
                c_2t = _safe_float(t8[4]) if len(t8) > 4 else 0.0
            if len(valid_cards) > 9:
                t9 = valid_cards[9].tokens()
                eps_1t2 = _safe_float(t9[0]) if len(t9) > 0 else 0.0
                eps_2t2 = _safe_float(t9[1]) if len(t9) > 1 else 0.0
                sig_rst2 = _safe_float(t9[2]) if len(t9) > 2 else 0.0
                wpmax_t2 = _safe_float(t9[3]) if len(t9) > 3 else 0.0
            if len(valid_cards) > 10:
                t10 = valid_cards[10].tokens()
                sig_1yc = _safe_float(t10[0]) if len(t10) > 0 else 0.0
                b_1c = _safe_float(t10[1]) if len(t10) > 1 else 0.0
                n_1c = _safe_float(t10[2]) if len(t10) > 2 else 1.0
                sig_1maxc = _safe_float(t10[3]) if len(t10) > 3 else 0.0
                c_1c = _safe_float(t10[4]) if len(t10) > 4 else 0.0
            if len(valid_cards) > 11:
                t11 = valid_cards[11].tokens()
                eps_1c1 = _safe_float(t11[0]) if len(t11) > 0 else 0.0
                eps_2c1 = _safe_float(t11[1]) if len(t11) > 1 else 0.0
                sig_rsc1 = _safe_float(t11[2]) if len(t11) > 2 else 0.0
                wpmax_c1 = _safe_float(t11[3]) if len(t11) > 3 else 0.0
            if len(valid_cards) > 12:
                t12 = valid_cards[12].tokens()
                sig_2yc = _safe_float(t12[0]) if len(t12) > 0 else 0.0
                b_2c = _safe_float(t12[1]) if len(t12) > 1 else 0.0
                n_2c = _safe_float(t12[2]) if len(t12) > 2 else 1.0
                sig_2maxc = _safe_float(t12[3]) if len(t12) > 3 else 0.0
                c_2c = _safe_float(t12[4]) if len(t12) > 4 else 0.0
            if len(valid_cards) > 13:
                t13 = valid_cards[13].tokens()
                eps_1c2 = _safe_float(t13[0]) if len(t13) > 0 else 0.0
                eps_2c2 = _safe_float(t13[1]) if len(t13) > 1 else 0.0
                sig_rsc2 = _safe_float(t13[2]) if len(t13) > 2 else 0.0
                wpmax_c2 = _safe_float(t13[3]) if len(t13) > 3 else 0.0
            if len(valid_cards) > 14:
                t14 = valid_cards[14].tokens()
                sig_12yt = _safe_float(t14[0]) if len(t14) > 0 else 0.0
                b_12t = _safe_float(t14[1]) if len(t14) > 1 else 0.0
                n_12t = _safe_float(t14[2]) if len(t14) > 2 else 1.0
                sig_12maxt = _safe_float(t14[3]) if len(t14) > 3 else 0.0
                c_12t = _safe_float(t14[4]) if len(t14) > 4 else 0.0
            if len(valid_cards) > 15:
                t15 = valid_cards[15].tokens()
                eps_1t12 = _safe_float(t15[0]) if len(t15) > 0 else 0.0
                eps_2t12 = _safe_float(t15[1]) if len(t15) > 1 else 0.0
                sig_rst12 = _safe_float(t15[2]) if len(t15) > 2 else 0.0
                wpmax_t12 = _safe_float(t15[3]) if len(t15) > 3 else 0.0
            if len(valid_cards) > 16:
                t16 = valid_cards[16].tokens()
                gamma_ini = _safe_float(t16[0]) if len(t16) > 0 else 0.0
                gamma_max = _safe_float(t16[1]) if len(t16) > 1 else 0.0
                d3max = _safe_float(t16[2]) if len(t16) > 2 else 0.0
            if len(valid_cards) > 17:
                t17 = valid_cards[17].tokens()
                fsmooth = _safe_int(t17[0]) if len(t17) > 0 else 0
                fcut = _safe_float(t17[1]) if len(t17) > 1 else 0.0

    mat = MatLaw25(
        id=mat_id, rho0=rho0, rhor=rhor, e11=e11, e22=e22, nu12=nu12,
        iform=iform, e33=e33, g12=g12, g23=g23, g31=g31,
        eps_f1=eps_f1, eps_f2=eps_f2, eps_t1=eps_t1, eps_m1=eps_m1,
        eps_t2=eps_t2, eps_m2=eps_m2, dmax=dmax, wpmax=wpmax, wpref=wpref,
        ioff=ioff, b=b, n=n, fmax=fmax, sig_1yt=sig_1yt, sig_2yt=sig_2yt,
        sig_1yc=sig_1yc, sig_2yc=sig_2yc, alpha=alpha, sig_12yc=sig_12yc,
        sig_12yt=sig_12yt, c=c, eps_rate_0=eps_rate_0, icc=icc, title=title,
        iflawp=iflawp,
        b_1t=b_1t, n_1t=n_1t, sig_1maxt=sig_1maxt, c_1t=c_1t,
        eps_1t1=eps_1t1, eps_2t1=eps_2t1, sig_rst1=sig_rst1, wpmax_t1=wpmax_t1,
        b_2t=b_2t, n_2t=n_2t, sig_2maxt=sig_2maxt, c_2t=c_2t,
        eps_1t2=eps_1t2, eps_2t2=eps_2t2, sig_rst2=sig_rst2, wpmax_t2=wpmax_t2,
        b_1c=b_1c, n_1c=n_1c, sig_1maxc=sig_1maxc, c_1c=c_1c,
        eps_1c1=eps_1c1, eps_2c1=eps_2c1, sig_rsc1=sig_rsc1, wpmax_c1=wpmax_c1,
        b_2c=b_2c, n_2c=n_2c, sig_2maxc=sig_2maxc, c_2c=c_2c,
        eps_1c2=eps_1c2, eps_2c2=eps_2c2, sig_rsc2=sig_rsc2, wpmax_c2=wpmax_c2,
        b_12t=b_12t, n_12t=n_12t, sig_12maxt=sig_12maxt, c_12t=c_12t,
        eps_1t12=eps_1t12, eps_2t12=eps_2t12, sig_rst12=sig_rst12, wpmax_t12=wpmax_t12,
        gamma_ini=gamma_ini, gamma_max=gamma_max, d3max=d3max,
        fsmooth=fsmooth, fcut=fcut,
    )
    model.mat_law25s[mat_id] = mat
    try:
        from ...materials.law25_composite import build_law25
        model.materials[mat_id] = build_law25(mat)
    except Exception:
        e_val = max(e11, e22, e33, 200e9) if max(e11, e22, e33) > 0 else 200e9
        model.materials[mat_id] = InactiveMaterial(
            id=mat_id, law=25, rho0=rho0, title=title, law_name="LAW25",
            params={
                "E": e_val, "MAT_E": e_val, "nu": nu12 if 0.0 <= nu12 < 0.5 else 0.3, "MAT_NU": nu12 if 0.0 <= nu12 < 0.5 else 0.3,
                "MAT_SIGY": sig_1yt if sig_1yt > 0 else 200e6, "rho": rho0, "MAT_RHO": rho0,
                "E11": e11, "E22": e22, "E33": e33, "NU12": nu12, "G12": g12, "G23": g23, "G31": g31,
                "IFORM": iform, "MAT_IFLAG": iform,
            }
        )




read_mat_comp_plas = read_mat_law25


read_mat_compsh = read_mat_law25


read_mat_tsai_wu = read_mat_law25


read_mat_crasurv = read_mat_law25


read_mat_composite_plas = read_mat_law25




def read_mat_law28_m188(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW28`` or ``/MAT/HONEYCOMB_SOL`` (M188): Solid honeycomb crush model."""
    from ...model.entities import MatLaw28
    from ..mat_reader import InactiveMaterial
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW28/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    e11, e22, e33 = 0.0, 0.0, 0.0
    g12, g23, g31 = 0.0, 0.0, 0.0
    fun_id11, fun_id22, fun_id33, iflag1 = 0, 0, 0, 0
    fscale11, fscale22, fscale33 = 0.0, 0.0, 0.0
    eps_max11, eps_max22, eps_max33 = 0.0, 0.0, 0.0
    fun_id12, fun_id23, fun_id31, iflag2 = 0, 0, 0, 0
    fscale12, fscale23, fscale31 = 0.0, 0.0, 0.0
    eps_max12, eps_max23, eps_max31 = 0.0, 0.0, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW28_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW28_2")
            e11 = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            e22 = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            e33 = _safe_float(c1[2]) if len(c1) > 2 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW28_3")
            g12 = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            g23 = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            g31 = _safe_float(c2[2]) if len(c2) > 2 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW28_4")
            fun_id11 = _safe_int(c3[0]) if len(c3) > 0 else 0
            fun_id22 = _safe_int(c3[1]) if len(c3) > 1 else 0
            fun_id33 = _safe_int(c3[2]) if len(c3) > 2 else 0
            iflag1 = _safe_int(c3[3]) if len(c3) > 3 else 0
            fscale11 = _safe_float(c3[4]) if len(c3) > 4 else 0.0
            fscale22 = _safe_float(c3[5]) if len(c3) > 5 else 0.0
            fscale33 = _safe_float(c3[6]) if len(c3) > 6 else 0.0
        if len(valid_cards) > 4:
            c4 = valid_cards[4].cut("MAT_LAW28_5")
            eps_max11 = _safe_float(c4[0]) if len(c4) > 0 else 0.0
            eps_max22 = _safe_float(c4[1]) if len(c4) > 1 else 0.0
            eps_max33 = _safe_float(c4[2]) if len(c4) > 2 else 0.0
        if len(valid_cards) > 5:
            c5 = valid_cards[5].cut("MAT_LAW28_6")
            fun_id12 = _safe_int(c5[0]) if len(c5) > 0 else 0
            fun_id23 = _safe_int(c5[1]) if len(c5) > 1 else 0
            fun_id31 = _safe_int(c5[2]) if len(c5) > 2 else 0
            iflag2 = _safe_int(c5[3]) if len(c5) > 3 else 0
            fscale12 = _safe_float(c5[4]) if len(c5) > 4 else 0.0
            fscale23 = _safe_float(c5[5]) if len(c5) > 5 else 0.0
            fscale31 = _safe_float(c5[6]) if len(c5) > 6 else 0.0
        if len(valid_cards) > 6:
            c6 = valid_cards[6].cut("MAT_LAW28_7")
            eps_max12 = _safe_float(c6[0]) if len(c6) > 0 else 0.0
            eps_max23 = _safe_float(c6[1]) if len(c6) > 1 else 0.0
            eps_max31 = _safe_float(c6[2]) if len(c6) > 2 else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            e11 = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            e22 = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            e33 = _safe_float(t1[2]) if len(t1) > 2 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            g12 = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            g23 = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            g31 = _safe_float(t2[2]) if len(t2) > 2 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            fun_id11 = _safe_int(t3[0]) if len(t3) > 0 else 0
            fun_id22 = _safe_int(t3[1]) if len(t3) > 1 else 0
            fun_id33 = _safe_int(t3[2]) if len(t3) > 2 else 0
            iflag1 = _safe_int(t3[3]) if len(t3) > 3 else 0
            fscale11 = _safe_float(t3[4]) if len(t3) > 4 else 0.0
            fscale22 = _safe_float(t3[5]) if len(t3) > 5 else 0.0
            fscale33 = _safe_float(t3[6]) if len(t3) > 6 else 0.0
        if len(valid_cards) > 4:
            t4 = valid_cards[4].tokens()
            eps_max11 = _safe_float(t4[0]) if len(t4) > 0 else 0.0
            eps_max22 = _safe_float(t4[1]) if len(t4) > 1 else 0.0
            eps_max33 = _safe_float(t4[2]) if len(t4) > 2 else 0.0
        if len(valid_cards) > 5:
            t5 = valid_cards[5].tokens()
            fun_id12 = _safe_int(t5[0]) if len(t5) > 0 else 0
            fun_id23 = _safe_int(t5[1]) if len(t5) > 1 else 0
            fun_id31 = _safe_int(t5[2]) if len(t5) > 2 else 0
            iflag2 = _safe_int(t5[3]) if len(t5) > 3 else 0
            fscale12 = _safe_float(t5[4]) if len(t5) > 4 else 0.0
            fscale23 = _safe_float(t5[5]) if len(t5) > 5 else 0.0
            fscale31 = _safe_float(t5[6]) if len(t5) > 6 else 0.0
        if len(valid_cards) > 6:
            t6 = valid_cards[6].tokens()
            eps_max12 = _safe_float(t6[0]) if len(t6) > 0 else 0.0
            eps_max23 = _safe_float(t6[1]) if len(t6) > 1 else 0.0
            eps_max31 = _safe_float(t6[2]) if len(t6) > 2 else 0.0

    mat = MatLaw28(
        id=mat_id, rho0=rho0, rhor=rhor, e11=e11, e22=e22, e33=e33,
        g12=g12, g23=g23, g31=g31, fun_id11=fun_id11, fun_id22=fun_id22,
        fun_id33=fun_id33, iflag1=iflag1, fscale11=fscale11, fscale22=fscale22,
        fscale33=fscale33, eps_max11=eps_max11, eps_max22=eps_max22,
        eps_max33=eps_max33, fun_id12=fun_id12, fun_id23=fun_id23,
        fun_id31=fun_id31, iflag2=iflag2, fscale12=fscale12, fscale23=fscale23,
        fscale31=fscale31, eps_max12=eps_max12, eps_max23=eps_max23,
        eps_max31=eps_max31, title=title
    )
    model.mat_law28s[mat_id] = mat
    from ...materials.law28_honeycomb import build_law28
    model.materials[mat_id] = build_law28(mat)








# ----------------------------------------------------------------------
# M189: Gurson, Gray Cast Iron, Composite Solid, Connector, Martensite Materials,
# Advanced Failure Criteria & Generalized Spring/Solid Properties
# ----------------------------------------------------------------------


def read_mat_law52(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW52`` or ``/MAT/GURSON`` / ``/MAT/PLAS_GURS`` (M189/M554): Gurson porous metal plasticity."""
    from ...model.entities import MatLaw52, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#") and not c.raw.strip().startswith("$")]
    if not valid_cards:
        log.error(f"/MAT/LAW52/{mat_id}: missing data card", block.source)
        return

    rho, refer_rho = 0.0, 0.0
    e, nu = 0.0, 0.0
    iflag, fsmooth = 0, 0
    fcut = 0.0
    iyield = 0
    a, b, n, c, pc = 0.0, 0.0, 0.0, 0.0, 0.0
    q1, q2, q3, s_n, eps_n = 0.0, 0.0, 0.0, 0.0, 0.0
    f_i, f_n, f_c, f_f = 0.0, 0.0, 0.0, 0.0
    itable = 0
    xfac, yfac = 0.0, 0.0

    is_fixed = getattr(block, "fixed", False)
    if is_fixed and valid_cards:
        raw0 = valid_cards[0].raw.rstrip()
        raw1 = valid_cards[1].raw.rstrip() if len(valid_cards) > 1 else ""
        raw2 = valid_cards[2].raw.rstrip() if len(valid_cards) > 2 else ""
        if "," in raw0 or "," in raw1 or "," in raw2:
            is_fixed = False
        elif len(valid_cards) > 1 and len(valid_cards[1].tokens()) >= 2 and len(raw1) < 80:
            is_fixed = False
        elif len(valid_cards) > 2 and len(valid_cards[2].tokens()) >= 2 and len(raw2) < 80:
            is_fixed = False

    def _split_tokens(card: Card) -> List[str]:
        return card.raw.replace(",", " ").split()

    if is_fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW52_1")
            rho = _safe_float(c0[0]) if len(c0) > 0 and c0[0] else 0.0
            refer_rho = _safe_float(c0[1]) if len(c0) > 1 and c0[1] else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW52_2")
            e = _safe_float(c1[0]) if len(c1) > 0 and c1[0] else 0.0
            nu = _safe_float(c1[1]) if len(c1) > 1 and c1[1] else 0.0
            iflag = _safe_int(c1[2]) if len(c1) > 2 and c1[2] else 0
            fsmooth = _safe_int(c1[3]) if len(c1) > 3 and c1[3] else 0
            fcut = _safe_float(c1[4]) if len(c1) > 4 and c1[4] else 0.0
            iyield = _safe_int(c1[5]) if len(c1) > 5 and c1[5] else 0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW52_3")
            a = _safe_float(c2[0]) if len(c2) > 0 and c2[0] else 0.0
            b = _safe_float(c2[1]) if len(c2) > 1 and c2[1] else 0.0
            n = _safe_float(c2[2]) if len(c2) > 2 and c2[2] else 0.0
            c = _safe_float(c2[3]) if len(c2) > 3 and c2[3] else 0.0
            pc = _safe_float(c2[4]) if len(c2) > 4 and c2[4] else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW52_4")
            q1 = _safe_float(c3[0]) if len(c3) > 0 and c3[0] else 0.0
            q2 = _safe_float(c3[1]) if len(c3) > 1 and c3[1] else 0.0
            q3 = _safe_float(c3[2]) if len(c3) > 2 and c3[2] else 0.0
            s_n = _safe_float(c3[3]) if len(c3) > 3 and c3[3] else 0.0
            eps_n = _safe_float(c3[4]) if len(c3) > 4 and c3[4] else 0.0
        if len(valid_cards) > 4:
            c4 = valid_cards[4].cut("MAT_LAW52_5")
            f_i = _safe_float(c4[0]) if len(c4) > 0 and c4[0] else 0.0
            f_n = _safe_float(c4[1]) if len(c4) > 1 and c4[1] else 0.0
            f_c = _safe_float(c4[2]) if len(c4) > 2 and c4[2] else 0.0
            f_f = _safe_float(c4[3]) if len(c4) > 3 and c4[3] else 0.0
        if len(valid_cards) > 5:
            c5 = valid_cards[5].cut("MAT_LAW52_6")
            itable = _safe_int(c5[0]) if len(c5) > 0 and c5[0] else 0
            xfac = _safe_float(c5[2]) if len(c5) > 2 and c5[2] else 0.0
            yfac = _safe_float(c5[3]) if len(c5) > 3 and c5[3] else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = _split_tokens(valid_cards[0])
            rho = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            refer_rho = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = _split_tokens(valid_cards[1])
            e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            iflag = _safe_int(t1[2]) if len(t1) > 2 else 0
            fsmooth = _safe_int(t1[3]) if len(t1) > 3 else 0
            fcut = _safe_float(t1[4]) if len(t1) > 4 else 0.0
            iyield = _safe_int(t1[5]) if len(t1) > 5 else 0
        if len(valid_cards) > 2:
            t2 = _split_tokens(valid_cards[2])
            a = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            b = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            n = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            c = _safe_float(t2[3]) if len(t2) > 3 else 0.0
            pc = _safe_float(t2[4]) if len(t2) > 4 else 0.0
        if len(valid_cards) > 3:
            t3 = _split_tokens(valid_cards[3])
            q1 = _safe_float(t3[0]) if len(t3) > 0 else 0.0
            q2 = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            q3 = _safe_float(t3[2]) if len(t3) > 2 else 0.0
            s_n = _safe_float(t3[3]) if len(t3) > 3 else 0.0
            eps_n = _safe_float(t3[4]) if len(t3) > 4 else 0.0
        if len(valid_cards) > 4:
            t4 = _split_tokens(valid_cards[4])
            f_i = _safe_float(t4[0]) if len(t4) > 0 else 0.0
            f_n = _safe_float(t4[1]) if len(t4) > 1 else 0.0
            f_c = _safe_float(t4[2]) if len(t4) > 2 else 0.0
            f_f = _safe_float(t4[3]) if len(t4) > 3 else 0.0
        if len(valid_cards) > 5:
            t5 = _split_tokens(valid_cards[5])
            itable = _safe_int(t5[0]) if len(t5) > 0 else 0
            if len(t5) >= 3:
                xfac = _safe_float(t5[1])
                yfac = _safe_float(t5[2])
            elif len(t5) == 2:
                xfac = _safe_float(t5[1])

    raw_fcut = fcut
    raw_c = c
    raw_pc = pc

    # Default values matching hm_read_mat52.F:
    if refer_rho == 0.0:
        refer_rho = rho
    if c > 0.0 and pc > 0.0 and fcut == 0.0:
        log.warning(f"/MAT/LAW52/{mat_id}: strain rate filtering is recommended when C > 0 and P > 0 with Fcut == 0 (ANCMSG 1220)", block.source)
    if c == 0.0:
        c = 1.0e30
    if pc == 0.0:
        pc = 1.0
    if fcut <= 0.0:
        fcut = 1.0e30
    if q1 == 0.0:
        q1 = 1.0e-20
    fu = 1.0 / q1
    if xfac == 0.0:
        xfac = 1.0
    if yfac == 0.0:
        yfac = 1.0

    if f_f < f_c or f_f < f_i or f_c < f_i:
        log.error(f"/MAT/LAW52/{mat_id}: void volume fractions must satisfy f_I <= f_C <= f_F (got f_I={f_i:g}, f_C={f_c:g}, f_F={f_f:g}) (ANCMSG 1745)", block.source)

    mat = MatLaw52(
        id=mat_id, rho=rho, refer_rho=refer_rho, e=e, nu=nu,
        a=a, b=b, n=n, c=c, pc=pc, q1=q1, q2=q2, q3=q3,
        s_n=s_n, eps_n=eps_n, f_i=f_i, f_n=f_n, f_c=f_c, f_f=f_f,
        iflag=iflag, fsmooth=fsmooth, fcut=fcut,
        itable=itable, xfac=xfac, yfac=yfac,
        title=title, raw_fcut=raw_fcut, raw_c=raw_c, raw_pc=raw_pc,
    )
    model.mat_law52s[mat_id] = mat
    if hasattr(model, "mat_gursons"):
        model.mat_gursons[mat_id] = mat
    if hasattr(model, "mat_plas_gurs"):
        model.mat_plas_gurs[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=52, rho0=rho, title=title,
        params={
            "rho": rho, "rho_i": rho, "refer_rho": refer_rho, "rho_o": refer_rho,
            "e": e, "E": e, "nu": nu, "NU": nu, "iflag": iflag, "fsmooth": fsmooth, "fcut": fcut,
            "raw_fcut": raw_fcut, "raw_c": raw_c, "raw_pc": raw_pc,
            "a": a, "A": a, "yield": a, "b": b, "B": b, "n": n, "N": n, "c": c, "C": c, "pc": pc, "P": pc,
            "q1": q1, "q2": q2, "q3": q3, "s_n": s_n, "sn": s_n, "eps_n": eps_n, "epsn": eps_n,
            "f_i": f_i, "f0": f_i, "f_0": f_i, "fi": f_i, "f_n": f_n, "fn": f_n, "f_c": f_c, "fc": f_c, "f_f": f_f, "ff": f_f, "fu": fu,
            "itable": itable, "xfac": xfac, "yfac": yfac,
        }
    )




def read_mat_law16(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW16`` or ``/MAT/GRAY`` (M189): Gray cast iron EOS and asymmetric plasticity model."""
    from ...model.entities import MatLaw16, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW16/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    p0, c, s, gamma0, a = 0.0, 0.0, 0.0, 0.0, 0.0
    e0, v0, e, nu, sig_y = 0.0, 0.0, 0.0, 0.0, 0.0
    beta, hard, sig_max, eps_max = 0.0, 0.0, 0.0, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW16_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW16_2")
            p0 = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            c = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            s = _safe_float(c1[2]) if len(c1) > 2 else 0.0
            gamma0 = _safe_float(c1[3]) if len(c1) > 3 else 0.0
            a = _safe_float(c1[4]) if len(c1) > 4 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW16_3")
            e0 = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            v0 = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            e = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            nu = _safe_float(c2[3]) if len(c2) > 3 else 0.0
            sig_y = _safe_float(c2[4]) if len(c2) > 4 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW16_4")
            beta = _safe_float(c3[0]) if len(c3) > 0 else 0.0
            hard = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            sig_max = _safe_float(c3[2]) if len(c3) > 2 else 0.0
            eps_max = _safe_float(c3[3]) if len(c3) > 3 else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            p0 = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            c = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            s = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            gamma0 = _safe_float(t1[3]) if len(t1) > 3 else 0.0
            a = _safe_float(t1[4]) if len(t1) > 4 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            e0 = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            v0 = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            e = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            nu = _safe_float(t2[3]) if len(t2) > 3 else 0.0
            sig_y = _safe_float(t2[4]) if len(t2) > 4 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            beta = _safe_float(t3[0]) if len(t3) > 0 else 0.0
            hard = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            sig_max = _safe_float(t3[2]) if len(t3) > 2 else 0.0
            eps_max = _safe_float(t3[3]) if len(t3) > 3 else 0.0

    mat = MatLaw16(
        id=mat_id, rho0=rho0, rhor=rhor, p0=p0, c=c, s=s, gamma0=gamma0, a=a,
        e0=e0, v0=v0, e=e, nu=nu, sig_y=sig_y, beta=beta, hard=hard,
        sig_max=sig_max, eps_max=eps_max, title=title
    )
    model.mat_law16s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=16, rho0=rho0, title=title,
        params={
            "rho_i": rho0, "rho_o": rhor, "p0": p0, "c": c, "s": s, "gamma0": gamma0, "a": a,
            "e0": e0, "v0": v0, "e": e, "nu": nu, "sig_y": sig_y, "beta": beta,
            "hard": hard, "sig_max": sig_max, "eps_max": eps_max
        }
    )




def read_mat_compso(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/COMPSO`` or ``/MAT/COMP_SOL`` (M189/M547): Composite solid orthotropic material model."""
    from ...model.entities import MatLaw14
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/COMPSO/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    ea, eb, ec = 0.0, 0.0, 0.0
    prab, prbc, prca = 0.0, 0.0, 0.0
    gab, gbc, gca = 0.0, 0.0, 0.0
    sigt1, sigt2, sigt3, delta = 0.0, 0.0, 0.0, 0.05
    cb, cn, fmax, wplaref = 0.0, 1.0, 1.0e10, 1.0
    sigyt1, sigyt2, sigyc1, sigyc2 = 0.0, 0.0, 0.0, 0.0
    sigt12, sigc12, sigt23, sigc23 = 0.0, 0.0, 0.0, 0.0
    alpha, efib, cc, eps0 = 0.0, 0.0, 0.0, 0.0
    strflag = 1

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_COMPSO_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_COMPSO_2")
            ea = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            eb = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            ec = _safe_float(c1[2]) if len(c1) > 2 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_COMPSO_3")
            prab = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            prbc = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            prca = _safe_float(c2[2]) if len(c2) > 2 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_COMPSO_4")
            gab = _safe_float(c3[0]) if len(c3) > 0 else 0.0
            gbc = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            gca = _safe_float(c3[2]) if len(c3) > 2 else 0.0
        if len(valid_cards) > 4:
            c4 = valid_cards[4].cut("MAT_COMPSO_5")
            sigt1 = _safe_float(c4[0]) if len(c4) > 0 else 0.0
            sigt2 = _safe_float(c4[1]) if len(c4) > 1 else 0.0
            sigt3 = _safe_float(c4[2]) if len(c4) > 2 else 0.0
            delta = _safe_float(c4[3]) if len(c4) > 3 else 0.0
        if len(valid_cards) > 5:
            c5 = valid_cards[5].cut("MAT_COMPSO_6")
            cb = _safe_float(c5[0]) if len(c5) > 0 else 0.0
            cn = _safe_float(c5[1]) if len(c5) > 1 else 0.0
            fmax = _safe_float(c5[2]) if len(c5) > 2 else 0.0
            wplaref = _safe_float(c5[3]) if len(c5) > 3 and c5[3].strip() else 1.0
        if len(valid_cards) > 6:
            c6 = valid_cards[6].cut("MAT_COMPSO_7")
            sigyt1 = _safe_float(c6[0]) if len(c6) > 0 else 0.0
            sigyt2 = _safe_float(c6[1]) if len(c6) > 1 else 0.0
            sigyc1 = _safe_float(c6[2]) if len(c6) > 2 else 0.0
            sigyc2 = _safe_float(c6[3]) if len(c6) > 3 else 0.0
        if len(valid_cards) > 7:
            c7 = valid_cards[7].cut("MAT_COMPSO_8")
            sigt12 = _safe_float(c7[0]) if len(c7) > 0 else 0.0
            sigc12 = _safe_float(c7[1]) if len(c7) > 1 else 0.0
            sigt23 = _safe_float(c7[2]) if len(c7) > 2 else 0.0
            sigc23 = _safe_float(c7[3]) if len(c7) > 3 else 0.0
        if len(valid_cards) > 8:
            c8 = valid_cards[8].cut("MAT_COMPSO_9")
            alpha = _safe_float(c8[0]) if len(c8) > 0 else 0.0
            efib = _safe_float(c8[1]) if len(c8) > 1 else 0.0
            cc = _safe_float(c8[2]) if len(c8) > 2 else 0.0
            eps0 = _safe_float(c8[3]) if len(c8) > 3 else 0.0
            strflag = _safe_int(c8[4]) if len(c8) > 4 else 1
    else:
        # Check if legacy 6-card free format (Card 1 has >= 5 tokens) or standard 9-card
        is_legacy_6card = (len(valid_cards) <= 6 and len(valid_cards) > 1 and len(valid_cards[1].tokens()) >= 5)
        if is_legacy_6card:
            if len(valid_cards) > 0:
                t0 = valid_cards[0].tokens()
                rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
                rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
            if len(valid_cards) > 1:
                t1 = valid_cards[1].tokens()
                ea = _safe_float(t1[0]) if len(t1) > 0 else 0.0
                eb = _safe_float(t1[1]) if len(t1) > 1 else 0.0
                ec = _safe_float(t1[2]) if len(t1) > 2 else 0.0
                prab = _safe_float(t1[3]) if len(t1) > 3 else 0.0
                prbc = _safe_float(t1[4]) if len(t1) > 4 else 0.0
            if len(valid_cards) > 2:
                t2 = valid_cards[2].tokens()
                prca = _safe_float(t2[0]) if len(t2) > 0 else 0.0
                gab = _safe_float(t2[1]) if len(t2) > 1 else 0.0
                gbc = _safe_float(t2[2]) if len(t2) > 2 else 0.0
                gca = _safe_float(t2[3]) if len(t2) > 3 else 0.0
            if len(valid_cards) > 3:
                t3 = valid_cards[3].tokens()
                sigt1 = _safe_float(t3[0]) if len(t3) > 0 else 0.0
                sigt2 = _safe_float(t3[1]) if len(t3) > 1 else 0.0
                sigt3 = _safe_float(t3[2]) if len(t3) > 2 else 0.0
                delta = _safe_float(t3[3]) if len(t3) > 3 else 0.0
                cb = _safe_float(t3[4]) if len(t3) > 4 else 0.0
            if len(valid_cards) > 4:
                t4 = valid_cards[4].tokens()
                cn = _safe_float(t4[0]) if len(t4) > 0 else 0.0
                fmax = _safe_float(t4[1]) if len(t4) > 1 else 0.0
                sigyt1 = _safe_float(t4[2]) if len(t4) > 2 else 0.0
                sigyt2 = _safe_float(t4[3]) if len(t4) > 3 else 0.0
                sigyc1 = _safe_float(t4[4]) if len(t4) > 4 else 0.0
            if len(valid_cards) > 5:
                t5 = valid_cards[5].tokens()
                sigyc2 = _safe_float(t5[0]) if len(t5) > 0 else 0.0
                sigt12 = _safe_float(t5[1]) if len(t5) > 1 else 0.0
                sigt23 = _safe_float(t5[2]) if len(t5) > 2 else 0.0
                sigc12 = _safe_float(t5[3]) if len(t5) > 3 else 0.0
                sigc23 = _safe_float(t5[4]) if len(t5) > 4 else 0.0
                alpha = _safe_float(t5[5]) if len(t5) > 5 else 0.0
                efib = _safe_float(t5[6]) if len(t5) > 6 else 0.0
                cc = _safe_float(t5[7]) if len(t5) > 7 else 0.0
                strflag = _safe_int(t5[8]) if len(t5) > 8 else 1
        else:
            if len(valid_cards) > 0:
                t0 = valid_cards[0].tokens()
                rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
                rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
            if len(valid_cards) > 1:
                t1 = valid_cards[1].tokens()
                ea = _safe_float(t1[0]) if len(t1) > 0 else 0.0
                eb = _safe_float(t1[1]) if len(t1) > 1 else 0.0
                ec = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            if len(valid_cards) > 2:
                t2 = valid_cards[2].tokens()
                prab = _safe_float(t2[0]) if len(t2) > 0 else 0.0
                prbc = _safe_float(t2[1]) if len(t2) > 1 else 0.0
                prca = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            if len(valid_cards) > 3:
                t3 = valid_cards[3].tokens()
                gab = _safe_float(t3[0]) if len(t3) > 0 else 0.0
                gbc = _safe_float(t3[1]) if len(t3) > 1 else 0.0
                gca = _safe_float(t3[2]) if len(t3) > 2 else 0.0
            if len(valid_cards) > 4:
                t4 = valid_cards[4].tokens()
                sigt1 = _safe_float(t4[0]) if len(t4) > 0 else 0.0
                sigt2 = _safe_float(t4[1]) if len(t4) > 1 else 0.0
                sigt3 = _safe_float(t4[2]) if len(t4) > 2 else 0.0
                delta = _safe_float(t4[3]) if len(t4) > 3 else 0.0
            if len(valid_cards) > 5:
                t5 = valid_cards[5].tokens()
                cb = _safe_float(t5[0]) if len(t5) > 0 else 0.0
                cn = _safe_float(t5[1]) if len(t5) > 1 else 0.0
                fmax = _safe_float(t5[2]) if len(t5) > 2 else 0.0
                wplaref = _safe_float(t5[3]) if len(t5) > 3 and t5[3].strip() else 1.0
            if len(valid_cards) > 6:
                t6 = valid_cards[6].tokens()
                sigyt1 = _safe_float(t6[0]) if len(t6) > 0 else 0.0
                sigyt2 = _safe_float(t6[1]) if len(t6) > 1 else 0.0
                sigyc1 = _safe_float(t6[2]) if len(t6) > 2 else 0.0
                sigyc2 = _safe_float(t6[3]) if len(t6) > 3 else 0.0
            if len(valid_cards) > 7:
                t7 = valid_cards[7].tokens()
                sigt12 = _safe_float(t7[0]) if len(t7) > 0 else 0.0
                sigc12 = _safe_float(t7[1]) if len(t7) > 1 else 0.0
                sigt23 = _safe_float(t7[2]) if len(t7) > 2 else 0.0
                sigc23 = _safe_float(t7[3]) if len(t7) > 3 else 0.0
            if len(valid_cards) > 8:
                t8 = valid_cards[8].tokens()
                alpha = _safe_float(t8[0]) if len(t8) > 0 else 0.0
                efib = _safe_float(t8[1]) if len(t8) > 1 else 0.0
                cc = _safe_float(t8[2]) if len(t8) > 2 else 0.0
                eps0 = _safe_float(t8[3]) if len(t8) > 3 else 0.0
                strflag = _safe_int(t8[4]) if len(t8) > 4 else 1

    # Fortran defaults (hm_read_mat14.F:161-174)
    if rhor == 0.0:
        rhor = rho0
    if delta == 0.0:
        delta = 0.05
    if sigt1 == 0.0:
        sigt1 = 1.0e30
    if sigt2 == 0.0:
        sigt2 = sigt1
    if sigt3 == 0.0:
        sigt3 = sigt1
    if cn == 0.0:
        cn = 1.0
    if fmax == 0.0:
        fmax = 1.0e10
    if wplaref == 0.0:
        wplaref = 1.0
    if cc == 0.0 and eps0 == 0.0:
        eps0 = 1.0
    if strflag == 0:
        strflag = 1

    law_name = "LAW14"
    if block.keyword:
        parts = block.keyword.split("/")
        if len(parts) > 2 and parts[2].strip():
            law_name = parts[2].strip()

    mat = MatLaw14(
        id=mat_id, rho0=rho0, rhor=rhor, ea=ea, eb=eb, ec=ec,
        prab=prab, prbc=prbc, prca=prca, gab=gab, gbc=gbc, gca=gca,
        sigt1=sigt1, sigt2=sigt2, sigt3=sigt3, damage=delta, delta=delta, beta=cb, cb=cb,
        hard=cn, cn=cn, sig_max=fmax, fmax=fmax, wpref=wplaref, wplaref=wplaref,
        sigyt1=sigyt1, sigyt2=sigyt2, sigyc1=sigyc1, sigyc2=sigyc2,
        sigt12=sigt12, sigc12=sigc12, sigt23=sigt23, sigc23=sigc23,
        sigyt12=sigt12, sigyc12=sigc12, sigyt23=sigt23, sigyc23=sigc23,
        alpha_fib=alpha, alpha=alpha, e_fib=efib, efib=efib, src=cc, cc=cc,
        srp=eps0, eps0=eps0, strflag=strflag, icc=strflag,
        title=title, law_name=law_name, law=14
    )
    model.mat_law14s[mat_id] = mat
    try:
        from ...materials.law14_compso import build_law14
        model.materials[mat_id] = build_law14(mat)
    except Exception:
        from ..mat_reader import InactiveMaterial
        e_val = max(ea, eb, ec, 200e9) if max(ea, eb, ec) > 0 else 200e9
        model.materials[mat_id] = InactiveMaterial(
            id=mat_id, law=14, rho0=rho0, title=title, law_name=law_name,
            params={
                "rho_i": rho0, "rho_o": rhor, "ea": ea, "eb": eb, "ec": ec,
                "prab": prab, "prbc": prbc, "prca": prca, "gab": gab, "gbc": gbc, "gca": gca,
                "sigt1": sigt1, "sigt2": sigt2, "sigt3": sigt3, "damage": delta, "delta": delta,
                "beta": cb, "cb": cb, "hard": cn, "cn": cn, "sig_max": fmax, "fmax": fmax,
                "wpref": wplaref, "wplaref": wplaref,
                "sigyt1": sigyt1, "sigyt2": sigyt2, "sigyc1": sigyc1, "sigyc2": sigyc2,
                "sigt12": sigt12, "sigc12": sigc12, "sigt23": sigt23, "sigc23": sigc23,
                "alpha_fib": alpha, "alpha": alpha, "e_fib": efib, "efib": efib,
                "src": cc, "cc": cc, "srp": eps0, "eps0": eps0, "strflag": strflag, "icc": strflag,
                "E": e_val, "MAT_E": e_val, "nu": prab if 0.0 <= prab < 0.5 else 0.3,
                "MAT_NU": prab if 0.0 <= prab < 0.5 else 0.3, "MAT_SIGY": sigyt1 if sigyt1 > 0 else 200e6,
                "rho": rho0, "MAT_RHO": rho0
            }
        )



read_mat_comp_sol = read_mat_compso




def read_mat_law14(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW14``: dispatches to Cam-Clay (M187, <= 2 cards) or Composite Solid (M189/M547, > 2 cards)."""
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if len(valid_cards) <= 2:
        read_mat_cam_clay(block, model, log)
    else:
        read_mat_compso(block, model, log)




def read_mat_law59(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW59`` or ``/MAT/CONNECT`` (M189): Connector / fastener material model."""
    from ...model.entities import MatLaw59, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW59/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    e, g0 = 0.0, 0.0
    fsmooth = 0
    fcut = 0.0
    iflag = 0
    functions = []

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW59_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW59_2")
            e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            g0 = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            fsmooth = _safe_int(c1[2]) if len(c1) > 2 else 0
            fcut = _safe_float(c1[3]) if len(c1) > 3 else 0.0
            iflag = _safe_int(c1[4]) if len(c1) > 4 else 0
        for card in valid_cards[2:]:
            cf = card.cut("MAT_LAW59_3")
            ipt = _safe_int(cf[0]) if len(cf) > 0 else 0
            ipdel = _safe_int(cf[1]) if len(cf) > 1 else 0
            fp1 = _safe_float(cf[2]) if len(cf) > 2 else 0.0
            fp2 = _safe_float(cf[3]) if len(cf) > 3 else 0.0
            functions.append({"ipt": ipt, "ipdel": ipdel, "fp1": fp1, "fp2": fp2})
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            g0 = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            fsmooth = _safe_int(t1[2]) if len(t1) > 2 else 0
            fcut = _safe_float(t1[3]) if len(t1) > 3 else 0.0
            iflag = _safe_int(t1[4]) if len(t1) > 4 else 0
        for card in valid_cards[2:]:
            tf = card.tokens()
            ipt = _safe_int(tf[0]) if len(tf) > 0 else 0
            ipdel = _safe_int(tf[1]) if len(tf) > 1 else 0
            fp1 = _safe_float(tf[2]) if len(tf) > 2 else 0.0
            fp2 = _safe_float(tf[3]) if len(tf) > 3 else 0.0
            functions.append({"ipt": ipt, "ipdel": ipdel, "fp1": fp1, "fp2": fp2})

    mat = MatLaw59(
        id=mat_id, rho0=rho0, rhor=rhor, e=e, g0=g0, fsmooth=fsmooth, fcut=fcut,
        iflag=iflag, functions=functions, title=title
    )
    model.mat_law59s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=59, rho0=rho0, title=title,
        params={
            "rho_i": rho0, "rho_o": rhor, "e": e, "g0": g0, "fsmooth": fsmooth,
            "fcut": fcut, "iflag": iflag, "functions": functions
        }
    )




def read_mat_law64(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW64`` or ``/MAT/TRANSFO_MART`` (M189): Martensitic transformation plasticity model."""
    from ...model.entities import MatLaw64, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW64/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    e, nu, cp, d, n = 0.0, 0.0, 0.0, 0.0, 0.0
    md, v0, vmc, t_ini = 0.0, 0.0, 0.0, 0.0
    funct_id_0, funct_id_1 = 0, 0
    scale_0, scale_1 = 1.0, 1.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW64_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW64_2")
            e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            cp = _safe_float(c1[2]) if len(c1) > 2 else 0.0
            d = _safe_float(c1[3]) if len(c1) > 3 else 0.0
            n = _safe_float(c1[4]) if len(c1) > 4 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW64_3")
            md = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            v0 = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            vmc = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            t_ini = _safe_float(c2[3]) if len(c2) > 3 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW64_4")
            funct_id_0 = _safe_int(c3[0]) if len(c3) > 0 else 0
            funct_id_1 = _safe_int(c3[1]) if len(c3) > 1 else 0
            scale_0 = _safe_float(c3[2], 1.0) if len(c3) > 2 and c3[2].strip() else 1.0
            scale_1 = _safe_float(c3[3], 1.0) if len(c3) > 3 and c3[3].strip() else 1.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            cp = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            d = _safe_float(t1[3]) if len(t1) > 3 else 0.0
            n = _safe_float(t1[4]) if len(t1) > 4 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            md = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            v0 = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            vmc = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            t_ini = _safe_float(t2[3]) if len(t2) > 3 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            funct_id_0 = _safe_int(t3[0]) if len(t3) > 0 else 0
            funct_id_1 = _safe_int(t3[1]) if len(t3) > 1 else 0
            scale_0 = _safe_float(t3[2], 1.0) if len(t3) > 2 else 1.0
            scale_1 = _safe_float(t3[3], 1.0) if len(t3) > 3 else 1.0

    mat = MatLaw64(
        id=mat_id, rho0=rho0, rhor=rhor, e=e, nu=nu, cp=cp, d=d, n=n,
        md=md, v0=v0, vmc=vmc, funct_id_0=funct_id_0, funct_id_1=funct_id_1,
        scale_0=scale_0, scale_1=scale_1, t_ini=t_ini, title=title
    )
    model.mat_law64s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=64, rho0=rho0, title=title,
        params={
            "rho_i": rho0, "rho_o": rhor, "e": e, "nu": nu, "cp": cp, "d": d, "n": n,
            "md": md, "v0": v0, "vmc": vmc, "funct_id_0": funct_id_0, "funct_id_1": funct_id_1,
            "scale_0": scale_0, "scale_1": scale_1, "t_ini": t_ini
        }
    )




# --- M190: Cosserat, Hill-MMC, Elastomer, Fabric, Bi-Material, Tabulated Viscoelastic, FEM User Material, Boltzmann, Lemaitre Plastic Damage, Law 78 Materials, Hashin/Tensile Strain/Energy/User Failure Criteria, and SPH/User Properties ---

def read_mat_law68(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW68`` or ``/MAT/COSSER`` (M190): 3D Cosserat continuum material model."""
    from ...model.entities import MatLaw68, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW68/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    e_11, e_22, e_33 = 0.0, 0.0, 0.0
    g_12, g_23, g_31 = 0.0, 0.0, 0.0
    fun_id11i, fun_id22i, fun_id33i, iflag1 = 0, 0, 0, 0
    fscale11i, fscale22i, fscale33i = 1.0, 1.0, 1.0
    eps_max11i, eps_max22i, eps_max33i = 0.0, 0.0, 0.0
    fun_id12i, fun_id23i, fun_id31i, iflag2 = 0, 0, 0, 0
    fscale12i, fscale23i, fscale31i = 1.0, 1.0, 1.0
    eps_max12i, eps_max23i, eps_max31i = 0.0, 0.0, 0.0
    fun_id21i, fun_id32i, fun_id13i = 0, 0, 0
    fscale21i, fscale32i, fscale13i = 1.0, 1.0, 1.0
    fun_id11r, fun_id22r, fun_id33r = 0, 0, 0
    fscale11r, fscale22r, fscale33r = 1.0, 1.0, 1.0
    eps_trans11r, eps_trans22r, eps_trans33r = 0.0, 0.0, 0.0
    fun_id12r, fun_id23r, fun_id31r = 0, 0, 0
    fscale12r, fscale23r, fscale31r = 1.0, 1.0, 1.0
    eps_trans12r, eps_trans23r, eps_trans31r = 0.0, 0.0, 0.0
    fun_id21r, fun_id32r, fun_id13r = 0, 0, 0
    fscale21r, fscale32r, fscale13r = 1.0, 1.0, 1.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW68_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW68_2")
            e_11 = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            e_22 = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            e_33 = _safe_float(c1[2]) if len(c1) > 2 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW68_3")
            g_12 = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            g_23 = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            g_31 = _safe_float(c2[2]) if len(c2) > 2 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW68_4")
            fun_id11i = _safe_int(c3[0]) if len(c3) > 0 else 0
            fun_id22i = _safe_int(c3[1]) if len(c3) > 1 else 0
            fun_id33i = _safe_int(c3[2]) if len(c3) > 2 else 0
            iflag1 = _safe_int(c3[3]) if len(c3) > 3 else 0
            fscale11i = _safe_float(c3[4], 1.0) if len(c3) > 4 and c3[4].strip() else 1.0
            fscale22i = _safe_float(c3[5], 1.0) if len(c3) > 5 and c3[5].strip() else 1.0
            fscale33i = _safe_float(c3[6], 1.0) if len(c3) > 6 and c3[6].strip() else 1.0
        if len(valid_cards) > 4:
            c4 = valid_cards[4].cut("MAT_LAW68_5")
            eps_max11i = _safe_float(c4[0]) if len(c4) > 0 else 0.0
            eps_max22i = _safe_float(c4[1]) if len(c4) > 1 else 0.0
            eps_max33i = _safe_float(c4[2]) if len(c4) > 2 else 0.0
        if len(valid_cards) > 5:
            c5 = valid_cards[5].cut("MAT_LAW68_6")
            fun_id12i = _safe_int(c5[0]) if len(c5) > 0 else 0
            fun_id23i = _safe_int(c5[1]) if len(c5) > 1 else 0
            fun_id31i = _safe_int(c5[2]) if len(c5) > 2 else 0
            iflag2 = _safe_int(c5[3]) if len(c5) > 3 else 0
            fscale12i = _safe_float(c5[4], 1.0) if len(c5) > 4 and c5[4].strip() else 1.0
            fscale23i = _safe_float(c5[5], 1.0) if len(c5) > 5 and c5[5].strip() else 1.0
            fscale31i = _safe_float(c5[6], 1.0) if len(c5) > 6 and c5[6].strip() else 1.0
        if len(valid_cards) > 6:
            c6 = valid_cards[6].cut("MAT_LAW68_7")
            eps_max12i = _safe_float(c6[0]) if len(c6) > 0 else 0.0
            eps_max23i = _safe_float(c6[1]) if len(c6) > 1 else 0.0
            eps_max31i = _safe_float(c6[2]) if len(c6) > 2 else 0.0
        if len(valid_cards) > 7:
            c7 = valid_cards[7].cut("MAT_LAW68_8")
            fun_id21i = _safe_int(c7[0]) if len(c7) > 0 else 0
            fun_id32i = _safe_int(c7[1]) if len(c7) > 1 else 0
            fun_id13i = _safe_int(c7[2]) if len(c7) > 2 else 0
            fscale21i = _safe_float(c7[3], 1.0) if len(c7) > 3 and c7[3].strip() else 1.0
            fscale32i = _safe_float(c7[4], 1.0) if len(c7) > 4 and c7[4].strip() else 1.0
            fscale13i = _safe_float(c7[5], 1.0) if len(c7) > 5 and c7[5].strip() else 1.0
        if len(valid_cards) > 8:
            c8 = valid_cards[8].cut("MAT_LAW68_9")
            fun_id11r = _safe_int(c8[0]) if len(c8) > 0 else 0
            fun_id22r = _safe_int(c8[1]) if len(c8) > 1 else 0
            fun_id33r = _safe_int(c8[2]) if len(c8) > 2 else 0
            fscale11r = _safe_float(c8[3], 1.0) if len(c8) > 3 and c8[3].strip() else 1.0
            fscale22r = _safe_float(c8[4], 1.0) if len(c8) > 4 and c8[4].strip() else 1.0
            fscale33r = _safe_float(c8[5], 1.0) if len(c8) > 5 and c8[5].strip() else 1.0
        if len(valid_cards) > 9:
            c9 = valid_cards[9].cut("MAT_LAW68_10")
            eps_trans11r = _safe_float(c9[0]) if len(c9) > 0 else 0.0
            eps_trans22r = _safe_float(c9[1]) if len(c9) > 1 else 0.0
            eps_trans33r = _safe_float(c9[2]) if len(c9) > 2 else 0.0
        if len(valid_cards) > 10:
            c10 = valid_cards[10].cut("MAT_LAW68_11")
            fun_id12r = _safe_int(c10[0]) if len(c10) > 0 else 0
            fun_id23r = _safe_int(c10[1]) if len(c10) > 1 else 0
            fun_id31r = _safe_int(c10[2]) if len(c10) > 2 else 0
            fscale12r = _safe_float(c10[3], 1.0) if len(c10) > 3 and c10[3].strip() else 1.0
            fscale23r = _safe_float(c10[4], 1.0) if len(c10) > 4 and c10[4].strip() else 1.0
            fscale31r = _safe_float(c10[5], 1.0) if len(c10) > 5 and c10[5].strip() else 1.0
        if len(valid_cards) > 11:
            c11 = valid_cards[11].cut("MAT_LAW68_12")
            eps_trans12r = _safe_float(c11[0]) if len(c11) > 0 else 0.0
            eps_trans23r = _safe_float(c11[1]) if len(c11) > 1 else 0.0
            eps_trans31r = _safe_float(c11[2]) if len(c11) > 2 else 0.0
        if len(valid_cards) > 12:
            c12 = valid_cards[12].cut("MAT_LAW68_13")
            fun_id21r = _safe_int(c12[0]) if len(c12) > 0 else 0
            fun_id32r = _safe_int(c12[1]) if len(c12) > 1 else 0
            fun_id13r = _safe_int(c12[2]) if len(c12) > 2 else 0
            fscale21r = _safe_float(c12[3], 1.0) if len(c12) > 3 and c12[3].strip() else 1.0
            fscale32r = _safe_float(c12[4], 1.0) if len(c12) > 4 and c12[4].strip() else 1.0
            fscale13r = _safe_float(c12[5], 1.0) if len(c12) > 5 and c12[5].strip() else 1.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            e_11 = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            e_22 = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            e_33 = _safe_float(t1[2]) if len(t1) > 2 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            g_12 = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            g_23 = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            g_31 = _safe_float(t2[2]) if len(t2) > 2 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            fun_id11i = _safe_int(t3[0]) if len(t3) > 0 else 0
            fun_id22i = _safe_int(t3[1]) if len(t3) > 1 else 0
            fun_id33i = _safe_int(t3[2]) if len(t3) > 2 else 0
            iflag1 = _safe_int(t3[3]) if len(t3) > 3 else 0
            fscale11i = _safe_float(t3[4], 1.0) if len(t3) > 4 else 1.0
            fscale22i = _safe_float(t3[5], 1.0) if len(t3) > 5 else 1.0
            fscale33i = _safe_float(t3[6], 1.0) if len(t3) > 6 else 1.0
        if len(valid_cards) > 4:
            t4 = valid_cards[4].tokens()
            eps_max11i = _safe_float(t4[0]) if len(t4) > 0 else 0.0
            eps_max22i = _safe_float(t4[1]) if len(t4) > 1 else 0.0
            eps_max33i = _safe_float(t4[2]) if len(t4) > 2 else 0.0
        if len(valid_cards) > 5:
            t5 = valid_cards[5].tokens()
            fun_id12i = _safe_int(t5[0]) if len(t5) > 0 else 0
            fun_id23i = _safe_int(t5[1]) if len(t5) > 1 else 0
            fun_id31i = _safe_int(t5[2]) if len(t5) > 2 else 0
            iflag2 = _safe_int(t5[3]) if len(t5) > 3 else 0
            fscale12i = _safe_float(t5[4], 1.0) if len(t5) > 4 else 1.0
            fscale23i = _safe_float(t5[5], 1.0) if len(t5) > 5 else 1.0
            fscale31i = _safe_float(t5[6], 1.0) if len(t5) > 6 else 1.0
        if len(valid_cards) > 6:
            t6 = valid_cards[6].tokens()
            eps_max12i = _safe_float(t6[0]) if len(t6) > 0 else 0.0
            eps_max23i = _safe_float(t6[1]) if len(t6) > 1 else 0.0
            eps_max31i = _safe_float(t6[2]) if len(t6) > 2 else 0.0
        if len(valid_cards) > 7:
            t7 = valid_cards[7].tokens()
            fun_id21i = _safe_int(t7[0]) if len(t7) > 0 else 0
            fun_id32i = _safe_int(t7[1]) if len(t7) > 1 else 0
            fun_id13i = _safe_int(t7[2]) if len(t7) > 2 else 0
            fscale21i = _safe_float(t7[3], 1.0) if len(t7) > 3 else 1.0
            fscale32i = _safe_float(t7[4], 1.0) if len(t7) > 4 else 1.0
            fscale13i = _safe_float(t7[5], 1.0) if len(t7) > 5 else 1.0
        if len(valid_cards) > 8:
            t8 = valid_cards[8].tokens()
            fun_id11r = _safe_int(t8[0]) if len(t8) > 0 else 0
            fun_id22r = _safe_int(t8[1]) if len(t8) > 1 else 0
            fun_id33r = _safe_int(t8[2]) if len(t8) > 2 else 0
            fscale11r = _safe_float(t8[3], 1.0) if len(t8) > 3 else 1.0
            fscale22r = _safe_float(t8[4], 1.0) if len(t8) > 4 else 1.0
            fscale33r = _safe_float(t8[5], 1.0) if len(t8) > 5 else 1.0
        if len(valid_cards) > 9:
            t9 = valid_cards[9].tokens()
            eps_trans11r = _safe_float(t9[0]) if len(t9) > 0 else 0.0
            eps_trans22r = _safe_float(t9[1]) if len(t9) > 1 else 0.0
            eps_trans33r = _safe_float(t9[2]) if len(t9) > 2 else 0.0
        if len(valid_cards) > 10:
            t10 = valid_cards[10].tokens()
            fun_id12r = _safe_int(t10[0]) if len(t10) > 0 else 0
            fun_id23r = _safe_int(t10[1]) if len(t10) > 1 else 0
            fun_id31r = _safe_int(t10[2]) if len(t10) > 2 else 0
            fscale12r = _safe_float(t10[3], 1.0) if len(t10) > 3 else 1.0
            fscale23r = _safe_float(t10[4], 1.0) if len(t10) > 4 else 1.0
            fscale31r = _safe_float(t10[5], 1.0) if len(t10) > 5 else 1.0
        if len(valid_cards) > 11:
            t11 = valid_cards[11].tokens()
            eps_trans12r = _safe_float(t11[0]) if len(t11) > 0 else 0.0
            eps_trans23r = _safe_float(t11[1]) if len(t11) > 1 else 0.0
            eps_trans31r = _safe_float(t11[2]) if len(t11) > 2 else 0.0
        if len(valid_cards) > 12:
            t12 = valid_cards[12].tokens()
            fun_id21r = _safe_int(t12[0]) if len(t12) > 0 else 0
            fun_id32r = _safe_int(t12[1]) if len(t12) > 1 else 0
            fun_id13r = _safe_int(t12[2]) if len(t12) > 2 else 0
            fscale21r = _safe_float(t12[3], 1.0) if len(t12) > 3 else 1.0
            fscale32r = _safe_float(t12[4], 1.0) if len(t12) > 4 else 1.0
            fscale13r = _safe_float(t12[5], 1.0) if len(t12) > 5 else 1.0

    mat = MatLaw68(
        id=mat_id, rho0=rho0, rhor=rhor,
        e_11=e_11, e_22=e_22, e_33=e_33,
        g_12=g_12, g_23=g_23, g_31=g_31,
        fun_id11i=fun_id11i, fun_id22i=fun_id22i, fun_id33i=fun_id33i, iflag1=iflag1,
        fscale11i=fscale11i, fscale22i=fscale22i, fscale33i=fscale33i,
        eps_max11i=eps_max11i, eps_max22i=eps_max22i, eps_max33i=eps_max33i,
        fun_id12i=fun_id12i, fun_id23i=fun_id23i, fun_id31i=fun_id31i, iflag2=iflag2,
        fscale12i=fscale12i, fscale23i=fscale23i, fscale31i=fscale31i,
        eps_max12i=eps_max12i, eps_max23i=eps_max23i, eps_max31i=eps_max31i,
        fun_id21i=fun_id21i, fun_id32i=fun_id32i, fun_id13i=fun_id13i,
        fscale21i=fscale21i, fscale32i=fscale32i, fscale13i=fscale13i,
        fun_id11r=fun_id11r, fun_id22r=fun_id22r, fun_id33r=fun_id33r,
        fscale11r=fscale11r, fscale22r=fscale22r, fscale33r=fscale33r,
        eps_trans11r=eps_trans11r, eps_trans22r=eps_trans22r, eps_trans33r=eps_trans33r,
        fun_id12r=fun_id12r, fun_id23r=fun_id23r, fun_id31r=fun_id31r,
        fscale12r=fscale12r, fscale23r=fscale23r, fscale31r=fscale31r,
        eps_trans12r=eps_trans12r, eps_trans23r=eps_trans23r, eps_trans31r=eps_trans31r,
        fun_id21r=fun_id21r, fun_id32r=fun_id32r, fun_id13r=fun_id13r,
        fscale21r=fscale21r, fscale32r=fscale32r, fscale13r=fscale13r,
        title=title
    )
    model.mat_law68s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=68, rho0=rho0, title=title,
        params={
            "rho": rho0, "rho_i": rho0, "rho_o": rhor,
            "e_11": e_11, "e_22": e_22, "e_33": e_33,
            "g_12": g_12, "g_23": g_23, "g_31": g_31,
            "fun_id11i": fun_id11i, "fun_id22i": fun_id22i, "fun_id33i": fun_id33i, "iflag1": iflag1,
            "eps_max11i": eps_max11i, "eps_max22i": eps_max22i, "eps_max33i": eps_max33i,
            "fun_id12i": fun_id12i, "fun_id23i": fun_id23i, "fun_id31i": fun_id31i, "iflag2": iflag2,
            "eps_max12i": eps_max12i, "eps_max23i": eps_max23i, "eps_max31i": eps_max31i,
        }
    )




def read_mat_law72(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW72`` or ``/MAT/HILL_MMC`` (M190): Hill orthotropic plasticity with Modified Mohr-Coulomb ductile fracture."""
    from ...model.entities import MatLaw72, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW72/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    e, nu = 0.0, 0.0
    sig0, eps0, n, f, g = 0.0, 0.0, 0.0, 0.0, 0.0
    h, big_n, l, m = 0.0, 0.0, 0.0, 0.0
    c1, c2, c3, mmc_m, dc = 0.0, 0.0, 0.0, 0.0, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW72_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1_card = valid_cards[1].cut("MAT_LAW72_2")
            e = _safe_float(c1_card[0]) if len(c1_card) > 0 else 0.0
            nu = _safe_float(c1_card[1]) if len(c1_card) > 1 else 0.0
        if len(valid_cards) > 2:
            c2_card = valid_cards[2].cut("MAT_LAW72_3")
            sig0 = _safe_float(c2_card[0]) if len(c2_card) > 0 else 0.0
            eps0 = _safe_float(c2_card[1]) if len(c2_card) > 1 else 0.0
            n = _safe_float(c2_card[2]) if len(c2_card) > 2 else 0.0
            f = _safe_float(c2_card[3]) if len(c2_card) > 3 else 0.0
            g = _safe_float(c2_card[4]) if len(c2_card) > 4 else 0.0
        if len(valid_cards) > 3:
            c3_card = valid_cards[3].cut("MAT_LAW72_4")
            h = _safe_float(c3_card[0]) if len(c3_card) > 0 else 0.0
            big_n = _safe_float(c3_card[1]) if len(c3_card) > 1 else 0.0
            l = _safe_float(c3_card[2]) if len(c3_card) > 2 else 0.0
            m = _safe_float(c3_card[3]) if len(c3_card) > 3 else 0.0
        if len(valid_cards) > 4:
            c4_card = valid_cards[4].cut("MAT_LAW72_5")
            c1 = _safe_float(c4_card[0]) if len(c4_card) > 0 else 0.0
            c2 = _safe_float(c4_card[1]) if len(c4_card) > 1 else 0.0
            c3 = _safe_float(c4_card[2]) if len(c4_card) > 2 else 0.0
            mmc_m = _safe_float(c4_card[3]) if len(c4_card) > 3 else 0.0
            dc = _safe_float(c4_card[4]) if len(c4_card) > 4 else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            sig0 = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            eps0 = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            n = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            f = _safe_float(t2[3]) if len(t2) > 3 else 0.0
            g = _safe_float(t2[4]) if len(t2) > 4 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            h = _safe_float(t3[0]) if len(t3) > 0 else 0.0
            big_n = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            l = _safe_float(t3[2]) if len(t3) > 2 else 0.0
            m = _safe_float(t3[3]) if len(t3) > 3 else 0.0
        if len(valid_cards) > 4:
            t4 = valid_cards[4].tokens()
            c1 = _safe_float(t4[0]) if len(t4) > 0 else 0.0
            c2 = _safe_float(t4[1]) if len(t4) > 1 else 0.0
            c3 = _safe_float(t4[2]) if len(t4) > 2 else 0.0
            mmc_m = _safe_float(t4[3]) if len(t4) > 3 else 0.0
            dc = _safe_float(t4[4]) if len(t4) > 4 else 0.0

    mat = MatLaw72(
        id=mat_id, rho0=rho0, rhor=rhor,
        e=e, nu=nu, sig0=sig0, eps0=eps0, n=n, f=f, g=g,
        h=h, big_n=big_n, l=l, m=m,
        c1=c1, c2=c2, c3=c3, mmc_m=mmc_m, dc=dc,
        title=title
    )
    model.mat_law72s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=72, rho0=rho0, title=title,
        params={
            "rho": rho0, "e": e, "nu": nu,
            "sig0": sig0, "eps0": eps0, "n": n, "f": f, "g": g,
            "h": h, "big_n": big_n, "l": l, "m": m,
            "c1": c1, "c2": c2, "c3": c3, "mmc_m": mmc_m, "dc": dc,
        }
    )




def read_mat_law65(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW65`` or ``/MAT/ELASTOMER`` (M190): 3D Elastomer hyperelastic model."""
    from ...model.entities import MatLaw65, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW65/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    e0, nu, eps_max = 0.0, 0.0, 0.0
    nrate, fsmooth = 0, 0
    fcut = 0.0
    rates = []

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW65_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW65_2")
            e0 = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            eps_max = _safe_float(c1[2]) if len(c1) > 2 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW65_3")
            nrate = _safe_int(c2[0]) if len(c2) > 0 else 0
            fsmooth = _safe_int(c2[1]) if len(c2) > 1 else 0
            fcut = _safe_float(c2[2]) if len(c2) > 2 else 0.0
        for rc in valid_cards[3:]:
            cr = rc.cut("MAT_LAW65_RATE")
            func_idld = _safe_int(cr[0]) if len(cr) > 0 else 0
            func_idul = _safe_int(cr[1]) if len(cr) > 1 else 0
            fscalestress = _safe_float(cr[2], 1.0) if len(cr) > 2 and cr[2].strip() else 1.0
            eps = _safe_float(cr[3]) if len(cr) > 3 else 0.0
            rates.append({"func_idld": func_idld, "func_idul": func_idul, "fscalestress": fscalestress, "eps": eps})
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            e0 = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            eps_max = _safe_float(t1[2]) if len(t1) > 2 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            nrate = _safe_int(t2[0]) if len(t2) > 0 else 0
            fsmooth = _safe_int(t2[1]) if len(t2) > 1 else 0
            fcut = _safe_float(t2[2]) if len(t2) > 2 else 0.0
        for rc in valid_cards[3:]:
            tr = rc.tokens()
            func_idld = _safe_int(tr[0]) if len(tr) > 0 else 0
            func_idul = _safe_int(tr[1]) if len(tr) > 1 else 0
            fscalestress = _safe_float(tr[2], 1.0) if len(tr) > 2 else 1.0
            eps = _safe_float(tr[3]) if len(tr) > 3 else 0.0
            rates.append({"func_idld": func_idld, "func_idul": func_idul, "fscalestress": fscalestress, "eps": eps})

    mat = MatLaw65(
        id=mat_id, rho0=rho0, rhor=rhor,
        e0=e0, nu=nu, eps_max=eps_max,
        nrate=nrate, fsmooth=fsmooth, fcut=fcut,
        rates=rates, title=title
    )
    model.mat_law65s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=65, rho0=rho0, title=title,
        params={
            "rho": rho0, "e0": e0, "nu": nu, "eps_max": eps_max,
            "nrate": nrate, "fsmooth": fsmooth, "fcut": fcut, "rates": rates,
        }
    )




def read_mat_law58(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW58``, ``/MAT/FABR_A``, or ``/MAT/FABRIC_A`` (M190/M553): Anisotropic fabric material model.

    Fortran origin: ``starter/source/materials/mat/mat058/hm_read_mat58.F`` and
    ``radioss2017/MAT/matl58_fabr_a.cfg``.
    """
    from ...model.entities import MatLaw58, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#") and not c.raw.strip().startswith("$")]
    if not valid_cards:
        log.error(f"/MAT/LAW58/{mat_id}: missing data card", block.source)
        return

    rho, refer_rho = 0.0, 0.0
    e1, b1, e2, b2, f = 0.0, 0.0, 0.0, 0.0, 0.01
    g0, gi, alpha, g5, isensor = 0.0, 0.0, 0.0, 0.0, 0
    df, ds, friction_phi, m58_zerostress = 0.05, 0.0, 0.0, 0.0
    n1_warp, n2_weft, s1, s2, c4, c5 = 1, 1, 0.1, 0.1, 0.0, 0.0
    fun_a1, c1 = 0, 1.0
    fun_a2, c2 = 0, 1.0
    fun_a3, c3 = 0, 1.0
    fun_a4, scale4 = 0, 1.0
    fun_a5, scale5 = 0, 1.0
    fun_a6, scale6 = 0, 1.0

    is_fixed = getattr(block, "fixed", False)
    if is_fixed and valid_cards:
        raw0 = valid_cards[0].raw.rstrip()
        raw1 = valid_cards[1].raw.rstrip() if len(valid_cards) > 1 else ""
        raw2 = valid_cards[2].raw.rstrip() if len(valid_cards) > 2 else ""
        if "," in raw0 or "," in raw1 or "," in raw2:
            is_fixed = False
        elif len(valid_cards) > 1 and len(valid_cards[1].tokens()) >= 2 and len(raw1) < 80:
            is_fixed = False
        elif len(valid_cards) > 2 and len(valid_cards[2].tokens()) >= 2 and len(raw2) < 80:
            is_fixed = False

    if is_fixed:
        # Card 1: RHO, Refer_Rho (MAT_LAW58_1: [20, 20])
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW58_1")
            rho = _safe_float(c0[0]) if len(c0) > 0 and c0[0] else 0.0
            refer_rho = _safe_float(c0[1]) if len(c0) > 1 and c0[1] else 0.0
        # Card 2: E1, B1, E2, B2, Flex (MAT_LAW58_2: [20, 20, 20, 20, 20])
        if len(valid_cards) > 1:
            c1_cut = valid_cards[1].cut("MAT_LAW58_2")
            e1 = _safe_float(c1_cut[0]) if len(c1_cut) > 0 and c1_cut[0] else 0.0
            b1 = _safe_float(c1_cut[1]) if len(c1_cut) > 1 and c1_cut[1] else 0.0
            e2 = _safe_float(c1_cut[2]) if len(c1_cut) > 2 and c1_cut[2] else 0.0
            b2 = _safe_float(c1_cut[3]) if len(c1_cut) > 3 and c1_cut[3] else 0.0
            f = _safe_float(c1_cut[4]) if len(c1_cut) > 4 and c1_cut[4] else 0.0
        # Card 3: G0, GT, AlphaT, Gsh, blank, sensor_ID (MAT_LAW58_3: [20, 20, 20, 20, 10, 10])
        if len(valid_cards) > 2:
            c2_cut = valid_cards[2].cut("MAT_LAW58_3")
            g0 = _safe_float(c2_cut[0]) if len(c2_cut) > 0 and c2_cut[0] else 0.0
            gi = _safe_float(c2_cut[1]) if len(c2_cut) > 1 and c2_cut[1] else 0.0
            alpha = _safe_float(c2_cut[2]) if len(c2_cut) > 2 and c2_cut[2] else 0.0
            g5 = _safe_float(c2_cut[3]) if len(c2_cut) > 3 and c2_cut[3] else 0.0
            isensor = _safe_int(c2_cut[5]) if len(c2_cut) > 5 and c2_cut[5] else (_safe_int(c2_cut[4]) if len(c2_cut) > 4 and c2_cut[4] else 0)
        # Card 4: Df, Ds, Friction_phi, blank, ZERO_STRESS (MAT_LAW58_4: [20, 20, 20, 20, 20])
        if len(valid_cards) > 3:
            c3_cut = valid_cards[3].cut("MAT_LAW58_4")
            df = _safe_float(c3_cut[0]) if len(c3_cut) > 0 and c3_cut[0] else 0.0
            ds = _safe_float(c3_cut[1]) if len(c3_cut) > 1 and c3_cut[1] else 0.0
            friction_phi = _safe_float(c3_cut[2]) if len(c3_cut) > 2 and c3_cut[2] else 0.0
            m58_zerostress = _safe_float(c3_cut[4]) if len(c3_cut) > 4 and c3_cut[4] else (_safe_float(c3_cut[3]) if len(c3_cut) > 3 and c3_cut[3] else 0.0)
        # Card 5: N1_warp, N2_weft, S1, S2, C4, C5 (MAT_LAW58_5: [10, 10, 20, 20, 20, 20])
        if len(valid_cards) > 4:
            c4_cut = valid_cards[4].cut("MAT_LAW58_5")
            n1_warp = _safe_int(c4_cut[0]) if len(c4_cut) > 0 and c4_cut[0] else 0
            n2_weft = _safe_int(c4_cut[1]) if len(c4_cut) > 1 and c4_cut[1] else 0
            s1 = _safe_float(c4_cut[2]) if len(c4_cut) > 2 and c4_cut[2] else 0.0
            s2 = _safe_float(c4_cut[3]) if len(c4_cut) > 3 and c4_cut[3] else 0.0
            c4 = _safe_float(c4_cut[4]) if len(c4_cut) > 4 and c4_cut[4] else 0.0
            c5 = _safe_float(c4_cut[5]) if len(c4_cut) > 5 and c4_cut[5] else 0.0
        # Optional tabulated loading curves:
        # Card 6: FUN_A1, blank, MAT_C1 (MAT_LAW58_6: [10, 10, 20])
        if len(valid_cards) > 5:
            c5_toks = valid_cards[5].tokens()
            if len(valid_cards) == 6 and len(c5_toks) >= 4:
                c_unl = valid_cards[5].cut("MAT_LAW58_7")
                fun_a4 = _safe_int(c_unl[0]) if len(c_unl) > 0 and c_unl[0] else 0
                fun_a5 = _safe_int(c_unl[1]) if len(c_unl) > 1 and c_unl[1] else 0
                scale4 = _safe_float(c_unl[2]) if len(c_unl) > 2 and c_unl[2] else 1.0
                scale5 = _safe_float(c_unl[3]) if len(c_unl) > 3 and c_unl[3] else 1.0
                fun_a6 = _safe_int(c_unl[4]) if len(c_unl) > 4 and c_unl[4] else 0
                scale6 = _safe_float(c_unl[5]) if len(c_unl) > 5 and c_unl[5] else 1.0
            else:
                c5_cut = valid_cards[5].cut("MAT_LAW58_6")
                fun_a1 = _safe_int(c5_cut[0]) if len(c5_cut) > 0 and c5_cut[0] else 0
                c1 = _safe_float(c5_cut[2]) if len(c5_cut) > 2 and c5_cut[2] else (_safe_float(c5_cut[1]) if len(c5_cut) > 1 and c5_cut[1] else 1.0)
        # Card 7: FUN_A2, blank, MAT_C2
        if len(valid_cards) > 6:
            c6_cut = valid_cards[6].cut("MAT_LAW58_6")
            fun_a2 = _safe_int(c6_cut[0]) if len(c6_cut) > 0 and c6_cut[0] else 0
            c2 = _safe_float(c6_cut[2]) if len(c6_cut) > 2 and c6_cut[2] else (_safe_float(c6_cut[1]) if len(c6_cut) > 1 and c6_cut[1] else 1.0)
        # Card 8: FUN_A3, blank, MAT_C3
        if len(valid_cards) > 7:
            c7_cut = valid_cards[7].cut("MAT_LAW58_6")
            fun_a3 = _safe_int(c7_cut[0]) if len(c7_cut) > 0 and c7_cut[0] else 0
            c3 = _safe_float(c7_cut[2]) if len(c7_cut) > 2 and c7_cut[2] else (_safe_float(c7_cut[1]) if len(c7_cut) > 1 and c7_cut[1] else 1.0)
        # Card 9: FUN_A4, FUN_A5, scale4, scale5, FUN_A6, scale6 (MAT_LAW58_7: [10, 10, 20, 20, 10, 20])
        if len(valid_cards) > 8:
            c8_cut = valid_cards[8].cut("MAT_LAW58_7")
            fun_a4 = _safe_int(c8_cut[0]) if len(c8_cut) > 0 and c8_cut[0] else 0
            fun_a5 = _safe_int(c8_cut[1]) if len(c8_cut) > 1 and c8_cut[1] else 0
            scale4 = _safe_float(c8_cut[2]) if len(c8_cut) > 2 and c8_cut[2] else 1.0
            scale5 = _safe_float(c8_cut[3]) if len(c8_cut) > 3 and c8_cut[3] else 1.0
            fun_a6 = _safe_int(c8_cut[4]) if len(c8_cut) > 4 and c8_cut[4] else 0
            scale6 = _safe_float(c8_cut[5]) if len(c8_cut) > 5 and c8_cut[5] else 1.0
    else:
        def _get_tokens(card: Card) -> List[str]:
            raw = card.raw.strip()
            if "," in raw:
                return [t.strip() for t in raw.split(",") if t.strip()]
            return card.tokens()

        if len(valid_cards) > 0:
            t0 = _get_tokens(valid_cards[0])
            rho = _safe_float(t0[0]) if len(t0) > 0 and t0[0] else 0.0
            refer_rho = _safe_float(t0[1]) if len(t0) > 1 and t0[1] else 0.0
        if len(valid_cards) > 1:
            t1 = _get_tokens(valid_cards[1])
            e1 = _safe_float(t1[0]) if len(t1) > 0 and t1[0] else 0.0
            b1 = _safe_float(t1[1]) if len(t1) > 1 and t1[1] else 0.0
            e2 = _safe_float(t1[2]) if len(t1) > 2 and t1[2] else 0.0
            b2 = _safe_float(t1[3]) if len(t1) > 3 and t1[3] else 0.0
            f = _safe_float(t1[4]) if len(t1) > 4 and t1[4] else 0.0
        if len(valid_cards) > 2:
            t2 = _get_tokens(valid_cards[2])
            g0 = _safe_float(t2[0]) if len(t2) > 0 and t2[0] else 0.0
            gi = _safe_float(t2[1]) if len(t2) > 1 and t2[1] else 0.0
            alpha = _safe_float(t2[2]) if len(t2) > 2 and t2[2] else 0.0
            if len(t2) == 4:
                if "." in t2[3] or "e" in t2[3].lower():
                    g5 = _safe_float(t2[3])
                else:
                    isensor = _safe_int(t2[3])
            elif len(t2) >= 5:
                g5 = _safe_float(t2[3])
                isensor = _safe_int(t2[4])
        if len(valid_cards) > 3:
            t3 = _get_tokens(valid_cards[3])
            df = _safe_float(t3[0]) if len(t3) > 0 and t3[0] else 0.0
            ds = _safe_float(t3[1]) if len(t3) > 1 and t3[1] else 0.0
            friction_phi = _safe_float(t3[2]) if len(t3) > 2 and t3[2] else 0.0
            if len(t3) == 4:
                m58_zerostress = _safe_float(t3[3])
            elif len(t3) >= 5:
                m58_zerostress = _safe_float(t3[4]) if t3[4] else _safe_float(t3[3])
        if len(valid_cards) > 4:
            t4 = _get_tokens(valid_cards[4])
            n1_warp = _safe_int(t4[0]) if len(t4) > 0 and t4[0] else 0
            n2_weft = _safe_int(t4[1]) if len(t4) > 1 and t4[1] else 0
            s1 = _safe_float(t4[2]) if len(t4) > 2 and t4[2] else 0.0
            s2 = _safe_float(t4[3]) if len(t4) > 3 and t4[3] else 0.0
            c4 = _safe_float(t4[4]) if len(t4) > 4 and t4[4] else 0.0
            c5 = _safe_float(t4[5]) if len(t4) > 5 and t4[5] else 0.0
        if len(valid_cards) > 5:
            t5 = _get_tokens(valid_cards[5])
            if len(valid_cards) == 6 and len(t5) >= 4:
                fun_a4 = _safe_int(t5[0]) if len(t5) > 0 and t5[0] else 0
                fun_a5 = _safe_int(t5[1]) if len(t5) > 1 and t5[1] else 0
                scale4 = _safe_float(t5[2]) if len(t5) > 2 and t5[2] else 1.0
                scale5 = _safe_float(t5[3]) if len(t5) > 3 and t5[3] else 1.0
                fun_a6 = _safe_int(t5[4]) if len(t5) > 4 and t5[4] else 0
                scale6 = _safe_float(t5[5]) if len(t5) > 5 and t5[5] else 1.0
            else:
                fun_a1 = _safe_int(t5[0]) if len(t5) > 0 and t5[0] else 0
                c1 = _safe_float(t5[1]) if len(t5) > 1 and t5[1] else 1.0
        if len(valid_cards) > 6:
            t6 = _get_tokens(valid_cards[6])
            fun_a2 = _safe_int(t6[0]) if len(t6) > 0 and t6[0] else 0
            c2 = _safe_float(t6[1]) if len(t6) > 1 and t6[1] else 1.0
        if len(valid_cards) > 7:
            t7 = _get_tokens(valid_cards[7])
            fun_a3 = _safe_int(t7[0]) if len(t7) > 0 and t7[0] else 0
            c3 = _safe_float(t7[1]) if len(t7) > 1 and t7[1] else 1.0
        if len(valid_cards) > 8:
            t8 = _get_tokens(valid_cards[8])
            fun_a4 = _safe_int(t8[0]) if len(t8) > 0 and t8[0] else 0
            fun_a5 = _safe_int(t8[1]) if len(t8) > 1 and t8[1] else 0
            scale4 = _safe_float(t8[2]) if len(t8) > 2 and t8[2] else 1.0
            scale5 = _safe_float(t8[3]) if len(t8) > 3 and t8[3] else 1.0
            fun_a6 = _safe_int(t8[4]) if len(t8) > 4 and t8[4] else 0
            scale6 = _safe_float(t8[5]) if len(t8) > 5 and t8[5] else 1.0

    # Apply exact Fortran defaults from hm_read_mat58.F
    if c1 == 0.0: c1 = 1.0
    if c2 == 0.0: c2 = 1.0
    if c3 == 0.0: c3 = 1.0
    if scale4 == 0.0: scale4 = 1.0
    if scale5 == 0.0: scale5 = 1.0
    if scale6 == 0.0: scale6 = 1.0

    if n1_warp == 0: n1_warp = 1
    if n2_weft == 0: n2_weft = 1
    if s1 == 0.0: s1 = 0.1
    if s2 == 0.0: s2 = 0.1
    if f == 0.0: f = 0.01

    if c4 == 0.0 and c5 == 0.0:
        c4 = f
        c5 = f
    elif c4 == 0.0 and c5 != 0.0:
        c4 = c5
    elif c5 == 0.0 and c4 != 0.0:
        c5 = c4

    if df == 0.0:
        df = 0.05

    # Check consistency of tabulated input data (loading and unloading)
    unloading_active = (fun_a4 != 0 or fun_a5 != 0 or fun_a6 != 0)
    if unloading_active:
        if fun_a4 == 0:
            fun_a4 = fun_a1
            scale4 = c1
        if fun_a5 == 0:
            fun_a5 = fun_a2
            scale5 = c2
        if fun_a6 == 0:
            fun_a6 = fun_a3
            scale6 = c3

        if fun_a1 == 0:
            log.error(f"/MAT/LAW58/{mat_id}: loading stress function ID in warp direction (FUN_A1) must be defined when unloading is active (ANCMSG 1578)", block.source)
        if fun_a2 == 0:
            log.error(f"/MAT/LAW58/{mat_id}: loading stress function ID in weft direction (FUN_A2) must be defined when unloading is active (ANCMSG 1579)", block.source)
        if fun_a3 == 0:
            log.error(f"/MAT/LAW58/{mat_id}: loading stress function ID in shear direction (FUN_A3) must be defined when unloading is active (ANCMSG 1580)", block.source)

    if gi == 0.0 and (e1 != 0.0 or e2 != 0.0):
        gi = 0.25 * (e1 + e2)

    mat = MatLaw58(
        id=mat_id,
        title=title,
        rho=rho,
        refer_rho=refer_rho,
        e1=e1,
        b1=b1,
        e2=e2,
        b2=b2,
        f=f,
        g0=g0,
        gi=gi,
        alpha=alpha,
        g5=g5,
        isensor=isensor,
        df=df,
        ds=ds,
        friction_phi=friction_phi,
        m58_zerostress=m58_zerostress,
        n1_warp=n1_warp,
        n2_weft=n2_weft,
        s1=s1,
        s2=s2,
        c4=c4,
        c5=c5,
        fun_a1=fun_a1,
        c1=c1,
        fun_a2=fun_a2,
        c2=c2,
        fun_a3=fun_a3,
        c3=c3,
        fun_a4=fun_a4,
        scale4=scale4,
        fun_a5=fun_a5,
        scale5=scale5,
        fun_a6=fun_a6,
        scale6=scale6,
    )
    model.mat_law58s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id,
        law=58,
        rho0=rho,
        title=title,
        params={
            "rho": rho, "refer_rho": refer_rho,
            "e1": e1, "b1": b1, "e2": e2, "b2": b2, "f": f,
            "E1": e1, "E2": e2, "B1": b1, "B2": b2, "RHO": rho, "RHO0": rho,
            "g0": g0, "gi": gi, "alpha": alpha, "g5": g5, "isensor": isensor,
            "df": df, "ds": ds, "friction_phi": friction_phi, "m58_zerostress": m58_zerostress,
            "n1_warp": n1_warp, "n2_weft": n2_weft, "s1": s1, "s2": s2, "c4": c4, "c5": c5,
            "fun_a1": fun_a1, "c1": c1, "fun_a2": fun_a2, "c2": c2, "fun_a3": fun_a3, "c3": c3,
            "fun_a4": fun_a4, "scale4": scale4, "fun_a5": fun_a5, "scale5": scale5, "fun_a6": fun_a6, "scale6": scale6,
            "flex": f, "gt": gi, "alphat": alpha, "sensor_id": isensor, "gfrot": friction_phi,
            "zero_stress": m58_zerostress, "n1": n1_warp, "n2": n2_weft,
            "phi_lock": alpha, "mu_frot": friction_phi, "arel": m58_zerostress, "a_rel": m58_zerostress,
            "c6": scale6, "flex1": c4, "flex2": c5, "gsh": g5,
        }
    )




def read_mat_law20(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW20`` or ``/MAT/BIMAT`` (M190): Bi-material mixture / layered material model."""
    from ...model.entities import MatLaw20, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW20/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    mat_id1, mat_id2 = 0, 0
    alpha1, alpha2 = 0.0, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW20_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW20_2")
            mat_id1 = _safe_int(c1[0]) if len(c1) > 0 else 0
            mat_id2 = _safe_int(c1[1]) if len(c1) > 1 else 0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW20_3")
            alpha1 = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            alpha2 = _safe_float(c2[1]) if len(c2) > 1 else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            mat_id1 = _safe_int(t1[0]) if len(t1) > 0 else 0
            mat_id2 = _safe_int(t1[1]) if len(t1) > 1 else 0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            alpha1 = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            alpha2 = _safe_float(t2[1]) if len(t2) > 1 else 0.0

    mat = MatLaw20(
        id=mat_id, rho0=rho0, rhor=rhor,
        mat_id1=mat_id1, mat_id2=mat_id2,
        alpha1=alpha1, alpha2=alpha2,
        title=title
    )
    model.mat_law20s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=20, rho0=rho0, title=title,
        params={
            "rho": rho0, "mat_id1": mat_id1, "mat_id2": mat_id2,
            "alpha1": alpha1, "alpha2": alpha2,
        }
    )




def read_mat_law38(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW38`` or ``/MAT/VISC_TAB``: Tabulated viscoelastic polymer/foam model (M190/M541)."""
    from ...model.entities import MatLaw38, Material
    mat_id = block.user_id or 0
    has_comma = any("," in c.raw for c in block.cards)
    is_fixed = block.fixed and not has_comma
    title, cards = _fixed_data(block) if is_fixed else _title_and_data(block)
    if is_fixed:
        valid_cards = [c for c in cards if not c.raw.strip().startswith("#")]
    else:
        valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW38/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    e, nu_t, nu_c, rv = 0.0, 0.0, 0.0, 0.0
    iflag, itotal = 0, 0
    beta, h, r_d = 0.0, 0.0, 0.0
    k_r, k_d = 0, 0
    instant_mod_upd = 0.0
    kair, np_val = 0, 0
    pscale = 0.0
    p0, rp, pmax, phi = 0.0, 0.0, 0.0, 0.0
    ful = 0
    alpha_unload, eps_unload, a, b = 0.0, 0.0, 0.0, 0.0
    m_func = 0
    cutoff = 0.0
    iinsta = 0
    e_final, epsi_final, lamb, visc, tol = 0.0, 0.0, 0.0, 0.0, 0.0
    fscale_i: list[float] = []
    epsilon_i: list[float] = []
    funct_id_load: list[int] = []
    funct_id_unload: list[int] = []

    if is_fixed:
        # Card 1: MAT_RHO, [Refer_Rho]
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW38_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 and c0[0] else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 and c0[1] else rho0
        # Card 2: MAT_E, MAT_NU, MAT_NUt, MAT_RV, MAT_IFLAG, ITOTAL
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW38_2")
            e = _safe_float(c1[0]) if len(c1) > 0 and c1[0] else 0.0
            nu_t = _safe_float(c1[1]) if len(c1) > 1 and c1[1] else 0.0
            nu_c = _safe_float(c1[2]) if len(c1) > 2 and c1[2] else 0.0
            rv = _safe_float(c1[3]) if len(c1) > 3 and c1[3] else 0.0
            iflag = _safe_int(c1[4]) if len(c1) > 4 and c1[4] else 0
            itotal = _safe_int(c1[5]) if len(c1) > 5 and c1[5] else 0
        # Card 3: MAT_RELX, MAT_HYST, DAMP1, Gflag, Vflag, MAT_Theta
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW38_3")
            beta = _safe_float(c2[0]) if len(c2) > 0 and c2[0] else 0.0
            h = _safe_float(c2[1]) if len(c2) > 1 and c2[1] else 0.0
            r_d = _safe_float(c2[2]) if len(c2) > 2 and c2[2] else 0.0
            k_r = _safe_int(c2[3]) if len(c2) > 3 and c2[3] else 0
            k_d = _safe_int(c2[4]) if len(c2) > 4 and c2[4] else 0
            instant_mod_upd = _safe_float(c2[5]) if len(c2) > 5 and c2[5] else 0.0
        # Card 4: MAT_Kair, FUN_A4, MAT_PScale
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW38_4")
            kair = _safe_int(c3[0]) if len(c3) > 0 and c3[0] else 0
            np_val = _safe_int(c3[1]) if len(c3) > 1 and c3[1] else 0
            pscale = _safe_float(c3[2]) if len(c3) > 2 and c3[2] else 0.0
        # Card 5: MAT_P0, MAT_PR, MAT_PMAX, MAT_POROS
        if len(valid_cards) > 4:
            c4 = valid_cards[4].cut("MAT_LAW38_5")
            p0 = _safe_float(c4[0]) if len(c4) > 0 and c4[0] else 0.0
            rp = _safe_float(c4[1]) if len(c4) > 1 and c4[1] else 0.0
            pmax = _safe_float(c4[2]) if len(c4) > 2 and c4[2] else 0.0
            phi = _safe_float(c4[3]) if len(c4) > 3 and c4[3] else 0.0
        # Card 6: FUN_B4, blank, MAT_ALPHA6, MAT_EPSF2, MAT_EXP1, MAT_EXP2
        if len(valid_cards) > 5:
            c5 = valid_cards[5].cut("MAT_LAW38_6")
            ful = _safe_int(c5[0]) if len(c5) > 0 and c5[0] else 0
            alpha_unload = _safe_float(c5[2]) if len(c5) > 2 and c5[2] else 0.0
            eps_unload = _safe_float(c5[3]) if len(c5) > 3 and c5[3] else 0.0
            a = _safe_float(c5[4]) if len(c5) > 4 and c5[4] else 0.0
            b = _safe_float(c5[5]) if len(c5) > 5 and c5[5] else 0.0
        # Card 7: NFUNC, blank, MAT_CUTOFF, MAT_Iinsta
        if len(valid_cards) > 6:
            c6 = valid_cards[6].cut("MAT_LAW38_7")
            m_func = _safe_int(c6[0]) if len(c6) > 0 and c6[0] else 0
            cutoff = _safe_float(c6[2]) if len(c6) > 2 and c6[2] else 0.0
            iinsta = _safe_int(c6[3]) if len(c6) > 3 and c6[3] else 0
        # Card 8: MAT_Efinal, MAT_Epsfinal, MAT_Lamda, MAT_MaxVisc, MAT_Tol
        if len(valid_cards) > 7:
            c7 = valid_cards[7].cut("MAT_LAW38_8")
            e_final = _safe_float(c7[0]) if len(c7) > 0 and c7[0] else 0.0
            epsi_final = _safe_float(c7[1]) if len(c7) > 1 and c7[1] else 0.0
            lamb = _safe_float(c7[2]) if len(c7) > 2 and c7[2] else 0.0
            visc = _safe_float(c7[3]) if len(c7) > 3 and c7[3] else 0.0
            tol = _safe_float(c7[4]) if len(c7) > 4 and c7[4] else 0.0

        if m_func == 0 and len(valid_cards) > 8:
            t8 = valid_cards[8].tokens()
            if t8:
                m_func = min(len(t8), 5)
        n = min(m_func, 5)
        if n > 0:
            if len(valid_cards) > 8:
                c8 = valid_cards[8].cut("MAT_LAW38_9")
                fscale_i = [_safe_float(c8[i]) if i < len(c8) and c8[i] else 1.0 for i in range(n)]
            if len(valid_cards) > 9:
                c9 = valid_cards[9].cut("MAT_LAW38_10")
                epsilon_i = [_safe_float(c9[i]) if i < len(c9) and c9[i] else 0.0 for i in range(n)]
            if len(valid_cards) > 10:
                c10 = valid_cards[10].cut("MAT_LAW38_11")
                funct_id_load = [_safe_int(c10[i]) if i < len(c10) and c10[i] else 0 for i in range(n)]
            if len(valid_cards) > 11:
                c11 = valid_cards[11].cut("MAT_LAW38_12")
                funct_id_unload = [_safe_int(c11[i]) if i < len(c11) and c11[i] else 0 for i in range(n)]
    else:
        # Free-format (supports comma-separated, whitespace-separated, and empty fields)
        def _card_tokens(c: Card) -> list[str]:
            raw = c.raw.strip()
            if not raw:
                return []
            if "," in raw:
                parts = [p.strip() for p in raw.split(",")]
                while parts and parts[-1] == "":
                    parts.pop()
                toks = []
                for p in parts:
                    if not p:
                        toks.append("")
                    else:
                        toks.extend(p.split())
                return toks
            return c.tokens()

        if len(valid_cards) > 0:
            t0 = _card_tokens(valid_cards[0])
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else rho0
        if len(valid_cards) > 1:
            t1 = _card_tokens(valid_cards[1])
            e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            nu_t = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            nu_c = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            rv = _safe_float(t1[3]) if len(t1) > 3 else 0.0
            iflag = _safe_int(t1[4]) if len(t1) > 4 else 0
            itotal = _safe_int(t1[5]) if len(t1) > 5 else 0
        if len(valid_cards) > 2:
            t2 = _card_tokens(valid_cards[2])
            beta = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            h = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            r_d = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            k_r = _safe_int(t2[3]) if len(t2) > 3 else 0
            k_d = _safe_int(t2[4]) if len(t2) > 4 else 0
            instant_mod_upd = _safe_float(t2[5]) if len(t2) > 5 else 0.0
        if len(valid_cards) > 3:
            t3 = _card_tokens(valid_cards[3])
            kair = _safe_int(t3[0]) if len(t3) > 0 else 0
            np_val = _safe_int(t3[1]) if len(t3) > 1 else 0
            pscale = _safe_float(t3[2]) if len(t3) > 2 else 0.0
        if len(valid_cards) > 4:
            t4 = _card_tokens(valid_cards[4])
            p0 = _safe_float(t4[0]) if len(t4) > 0 else 0.0
            rp = _safe_float(t4[1]) if len(t4) > 1 else 0.0
            pmax = _safe_float(t4[2]) if len(t4) > 2 else 0.0
            phi = _safe_float(t4[3]) if len(t4) > 3 else 0.0
        if len(valid_cards) > 5:
            t5 = _card_tokens(valid_cards[5])
            if len(t5) >= 6:
                ful = _safe_int(t5[0])
                alpha_unload = _safe_float(t5[2])
                eps_unload = _safe_float(t5[3])
                a = _safe_float(t5[4])
                b = _safe_float(t5[5])
            else:
                ful = _safe_int(t5[0]) if len(t5) > 0 else 0
                alpha_unload = _safe_float(t5[1]) if len(t5) > 1 else 0.0
                eps_unload = _safe_float(t5[2]) if len(t5) > 2 else 0.0
                a = _safe_float(t5[3]) if len(t5) > 3 else 0.0
                b = _safe_float(t5[4]) if len(t5) > 4 else 0.0
        if len(valid_cards) > 6:
            t6 = _card_tokens(valid_cards[6])
            if len(t6) >= 4:
                m_func = _safe_int(t6[0])
                cutoff = _safe_float(t6[2])
                iinsta = _safe_int(t6[3])
            else:
                m_func = _safe_int(t6[0]) if len(t6) > 0 else 0
                cutoff = _safe_float(t6[1]) if len(t6) > 1 else 0.0
                iinsta = _safe_int(t6[2]) if len(t6) > 2 else 0
        if len(valid_cards) > 7:
            t7 = _card_tokens(valid_cards[7])
            e_final = _safe_float(t7[0]) if len(t7) > 0 else 0.0
            epsi_final = _safe_float(t7[1]) if len(t7) > 1 else 0.0
            lamb = _safe_float(t7[2]) if len(t7) > 2 else 0.0
            visc = _safe_float(t7[3]) if len(t7) > 3 else 0.0
            tol = _safe_float(t7[4]) if len(t7) > 4 else 0.0

        if m_func == 0 and len(valid_cards) > 8:
            t8 = _card_tokens(valid_cards[8])
            if t8:
                m_func = min(len(t8), 5)
        n = min(m_func, 5)
        if n > 0:
            if len(valid_cards) > 8:
                t8 = _card_tokens(valid_cards[8])
                fscale_i = [_safe_float(t8[i]) if i < len(t8) else 1.0 for i in range(n)]
            if len(valid_cards) > 9:
                t9 = _card_tokens(valid_cards[9])
                epsilon_i = [_safe_float(t9[i]) if i < len(t9) else 0.0 for i in range(n)]
            if len(valid_cards) > 10:
                t10 = _card_tokens(valid_cards[10])
                funct_id_load = [_safe_int(t10[i]) if i < len(t10) else 0 for i in range(n)]
            if len(valid_cards) > 11:
                t11 = _card_tokens(valid_cards[11])
                funct_id_unload = [_safe_int(t11[i]) if i < len(t11) else 0 for i in range(n)]

    # Defaults and post-processing (citing hm_read_mat38.F lines 221-258)
    if rhor == 0.0:
        rhor = rho0
    if pscale == 0.0:
        pscale = 1.0
    if h == 0.0:
        h = 1.0
    if r_d == 0.0:
        r_d = 0.5
    if instant_mod_upd == 0.0:
        instant_mod_upd = 0.67
    if alpha_unload <= 0.0:
        alpha_unload = 1.0
    if a == 0.0:
        a = 1.0
    if b == 0.0:
        b = 1.0
    if epsi_final == 0.0:
        epsi_final = 1.0
    if lamb == 0.0:
        lamb = 1.0
    if tol == 0.0:
        tol = 1.0
    if e_final <= e and e_final == 0.0:
        e_final = e

    # Scale factors default <= 0 -> 1.0 (hm_read_mat38.F line 323)
    fscale_i = [1.0 if x <= 0.0 else x for x in fscale_i]
    # Unload functions default <= 0 -> funct_id_load[0] (hm_read_mat38.F line 343)
    for j in range(len(funct_id_unload)):
        if funct_id_unload[j] <= 0 and funct_id_load:
            funct_id_unload[j] = funct_id_load[0]

    mat = MatLaw38(
        id=mat_id, rho0=rho0, rhor=rhor,
        e=e, nu_t=nu_t, nu_c=nu_c, rv=rv, iflag=iflag, itotal=itotal,
        beta=beta, h=h, r_d=r_d, k_r=k_r, k_d=k_d, instant_mod_upd=instant_mod_upd,
        kair=kair, np=np_val, pscale=pscale,
        p0=p0, rp=rp, pmax=pmax, phi=phi,
        ful=ful, alpha_unload=alpha_unload, eps_unload=eps_unload, a=a, b=b,
        m_func=m_func, cutoff=cutoff, iinsta=iinsta,
        e_final=e_final, epsi_final=epsi_final, lamb=lamb, visc=visc, tol=tol,
        fscale_i=fscale_i, epsilon_i=epsilon_i,
        funct_id_load=funct_id_load, funct_id_unload=funct_id_unload,
        title=title
    )
    model.mat_law38s[mat_id] = mat

    nu_effective = max(nu_c, nu_t)
    model.materials[mat_id] = Material(
        id=mat_id, law=38, rho0=rho0, title=title, law_name="LAW38",
        params={
            "rho": rho0, "rho0": rho0, "rhor": rhor, "MAT_RHO": rho0, "Refer_Rho": rhor,
            "e": e, "MAT_E": e, "E": e,
            "nu": nu_effective, "nu_t": nu_t, "nu_c": nu_c, "MAT_NU": nu_t, "MAT_NUt": nu_c, "rv": rv, "MAT_RV": rv,
            "iflag": iflag, "MAT_IFLAG": iflag, "itotal": itotal, "ITOTAL": itotal,
            "beta": beta, "MAT_RELX": beta, "h": h, "MAT_HYST": h, "r_d": r_d, "damp1": r_d, "DAMP1": r_d,
            "k_r": k_r, "gflag": k_r, "Gflag": k_r, "k_d": k_d, "vflag": k_d, "Vflag": k_d,
            "instant_mod_upd": instant_mod_upd, "theta": instant_mod_upd, "MAT_Theta": instant_mod_upd,
            "kair": kair, "MAT_Kair": kair, "np": np_val, "fun_a4": np_val, "FUN_A4": np_val,
            "pscale": pscale, "MAT_PScale": pscale,
            "p0": p0, "MAT_P0": p0, "rp": rp, "pr": rp, "MAT_PR": rp,
            "pmax": pmax, "MAT_PMAX": pmax, "phi": phi, "poros": phi, "MAT_POROS": phi,
            "ful": ful, "fun_b4": ful, "FUN_B4": ful,
            "alpha_unload": alpha_unload, "alpha6": alpha_unload, "MAT_ALPHA6": alpha_unload,
            "eps_unload": eps_unload, "epsf2": eps_unload, "MAT_EPSF2": eps_unload,
            "a": a, "exp1": a, "MAT_EXP1": a, "b": b, "exp2": b, "MAT_EXP2": b,
            "m_func": m_func, "nfunc": m_func, "NFUNC": m_func,
            "cutoff": cutoff, "MAT_CUTOFF": cutoff, "iinsta": iinsta, "MAT_Iinsta": iinsta,
            "e_final": e_final, "efinal": e_final, "MAT_Efinal": e_final,
            "epsi_final": epsi_final, "epsfinal": epsi_final, "MAT_Epsfinal": epsi_final,
            "lamb": lamb, "lamda": lamb, "MAT_Lamda": lamb,
            "visc": visc, "maxvisc": visc, "MAT_MaxVisc": visc,
            "tol": tol, "MAT_Tol": tol,
            "fscale_i": fscale_i, "epsilon_i": epsilon_i,
            "funct_id_load": funct_id_load, "funct_id_unload": funct_id_unload,
            "fscale": fscale_i, "epsilon": epsilon_i,
        }
    )




read_mat_visc_tab = read_mat_law38




def read_mat_law29(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW29`` or ``/MAT/FEM`` (M190): User/FEM material model interface."""
    from ...model.entities import MatLaw29, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW29/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    version = ""
    frelim, dtmin, nf, velsc, rstrat, rtemp, encrypt, param_1 = 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, 0
    el_young, el_poiss, el_bulkm, el_shear, el_ortho, el_shrco, param_2, param_3 = 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0, 0
    pl_harde, pl_ortho, pl_iskin, pl_asymm, pl_waist, pl_biaxf, pl_compr, pl_damag = 0, 0, 0, 0, 0, 0, 0, 0
    nf_curve, nf_ortho, nf_depen, sf_curve, sf_param, sf_postc, param_4, param_5 = 0, 0, 0, 0, 0, 0, 0, 0
    cr_harde, cr_ortho, cr_iskin, cr_postc, cr_param, cr_check, param_6, mf_init = 0, 0, 0, 0, 0, 0, 0, 0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW29_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            version = valid_cards[1].raw.strip()
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW29_3")
            frelim = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            dtmin = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            nf = _safe_int(c2[2]) if len(c2) > 2 else 0
            velsc = _safe_float(c2[3]) if len(c2) > 3 else 0.0
            rstrat = _safe_float(c2[4]) if len(c2) > 4 else 0.0
            rtemp = _safe_float(c2[5]) if len(c2) > 5 else 0.0
            encrypt = _safe_int(c2[6]) if len(c2) > 6 else 0
            param_1 = _safe_int(c2[7]) if len(c2) > 7 else 0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW29_4")
            el_young = _safe_float(c3[0]) if len(c3) > 0 else 0.0
            el_poiss = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            el_bulkm = _safe_float(c3[2]) if len(c3) > 2 else 0.0
            el_shear = _safe_float(c3[3]) if len(c3) > 3 else 0.0
            el_ortho = _safe_int(c3[4]) if len(c3) > 4 else 0
            el_shrco = _safe_float(c3[5]) if len(c3) > 5 else 0.0
            param_2 = _safe_int(c3[6]) if len(c3) > 6 else 0
            param_3 = _safe_int(c3[7]) if len(c3) > 7 else 0
        if len(valid_cards) > 4:
            c4 = valid_cards[4].cut("MAT_LAW29_5")
            pl_harde = _safe_int(c4[0]) if len(c4) > 0 else 0
            pl_ortho = _safe_int(c4[1]) if len(c4) > 1 else 0
            pl_iskin = _safe_int(c4[2]) if len(c4) > 2 else 0
            pl_asymm = _safe_int(c4[3]) if len(c4) > 3 else 0
            pl_waist = _safe_int(c4[4]) if len(c4) > 4 else 0
            pl_biaxf = _safe_int(c4[5]) if len(c4) > 5 else 0
            pl_compr = _safe_int(c4[6]) if len(c4) > 6 else 0
            pl_damag = _safe_int(c4[7]) if len(c4) > 7 else 0
        if len(valid_cards) > 5:
            c5 = valid_cards[5].cut("MAT_LAW29_6")
            nf_curve = _safe_int(c5[0]) if len(c5) > 0 else 0
            nf_ortho = _safe_int(c5[1]) if len(c5) > 1 else 0
            nf_depen = _safe_int(c5[2]) if len(c5) > 2 else 0
            sf_curve = _safe_int(c5[3]) if len(c5) > 3 else 0
            sf_param = _safe_int(c5[4]) if len(c5) > 4 else 0
            sf_postc = _safe_int(c5[5]) if len(c5) > 5 else 0
            param_4 = _safe_int(c5[6]) if len(c5) > 6 else 0
            param_5 = _safe_int(c5[7]) if len(c5) > 7 else 0
        if len(valid_cards) > 6:
            c6 = valid_cards[6].cut("MAT_LAW29_7")
            cr_harde = _safe_int(c6[0]) if len(c6) > 0 else 0
            cr_ortho = _safe_int(c6[1]) if len(c6) > 1 else 0
            cr_iskin = _safe_int(c6[2]) if len(c6) > 2 else 0
            cr_postc = _safe_int(c6[3]) if len(c6) > 3 else 0
            cr_param = _safe_int(c6[4]) if len(c6) > 4 else 0
            cr_check = _safe_int(c6[5]) if len(c6) > 5 else 0
            param_6 = _safe_int(c6[6]) if len(c6) > 6 else 0
            mf_init = _safe_int(c6[7]) if len(c6) > 7 else 0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            version = valid_cards[1].raw.strip()
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            frelim = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            dtmin = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            nf = _safe_int(t2[2]) if len(t2) > 2 else 0
            velsc = _safe_float(t2[3]) if len(t2) > 3 else 0.0
            rstrat = _safe_float(t2[4]) if len(t2) > 4 else 0.0
            rtemp = _safe_float(t2[5]) if len(t2) > 5 else 0.0
            encrypt = _safe_int(t2[6]) if len(t2) > 6 else 0
            param_1 = _safe_int(t2[7]) if len(t2) > 7 else 0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            el_young = _safe_float(t3[0]) if len(t3) > 0 else 0.0
            el_poiss = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            el_bulkm = _safe_float(t3[2]) if len(t3) > 2 else 0.0
            el_shear = _safe_float(t3[3]) if len(t3) > 3 else 0.0
            el_ortho = _safe_int(t3[4]) if len(t3) > 4 else 0
            el_shrco = _safe_float(t3[5]) if len(t3) > 5 else 0.0
            param_2 = _safe_int(t3[6]) if len(t3) > 6 else 0
            param_3 = _safe_int(t3[7]) if len(t3) > 7 else 0
        if len(valid_cards) > 4:
            t4 = valid_cards[4].tokens()
            pl_harde = _safe_int(t4[0]) if len(t4) > 0 else 0
            pl_ortho = _safe_int(t4[1]) if len(t4) > 1 else 0
            pl_iskin = _safe_int(t4[2]) if len(t4) > 2 else 0
            pl_asymm = _safe_int(t4[3]) if len(t4) > 3 else 0
            pl_waist = _safe_int(t4[4]) if len(t4) > 4 else 0
            pl_biaxf = _safe_int(t4[5]) if len(t4) > 5 else 0
            pl_compr = _safe_int(t4[6]) if len(t4) > 6 else 0
            pl_damag = _safe_int(t4[7]) if len(t4) > 7 else 0
        if len(valid_cards) > 5:
            t5 = valid_cards[5].tokens()
            nf_curve = _safe_int(t5[0]) if len(t5) > 0 else 0
            nf_ortho = _safe_int(t5[1]) if len(t5) > 1 else 0
            nf_depen = _safe_int(t5[2]) if len(t5) > 2 else 0
            sf_curve = _safe_int(t5[3]) if len(t5) > 3 else 0
            sf_param = _safe_int(t5[4]) if len(t5) > 4 else 0
            sf_postc = _safe_int(t5[5]) if len(t5) > 5 else 0
            param_4 = _safe_int(t5[6]) if len(t5) > 6 else 0
            param_5 = _safe_int(t5[7]) if len(t5) > 7 else 0
        if len(valid_cards) > 6:
            t6 = valid_cards[6].tokens()
            cr_harde = _safe_int(t6[0]) if len(t6) > 0 else 0
            cr_ortho = _safe_int(t6[1]) if len(t6) > 1 else 0
            cr_iskin = _safe_int(t6[2]) if len(t6) > 2 else 0
            cr_postc = _safe_int(t6[3]) if len(t6) > 3 else 0
            cr_param = _safe_int(t6[4]) if len(t6) > 4 else 0
            cr_check = _safe_int(t6[5]) if len(t6) > 5 else 0
            param_6 = _safe_int(t6[6]) if len(t6) > 6 else 0
            mf_init = _safe_int(t6[7]) if len(t6) > 7 else 0

    mat = MatLaw29(
        id=mat_id, rho0=rho0, rhor=rhor, version=version,
        frelim=frelim, dtmin=dtmin, nf=nf, velsc=velsc, rstrat=rstrat, rtemp=rtemp, encrypt=encrypt, param_1=param_1,
        el_young=el_young, el_poiss=el_poiss, el_bulkm=el_bulkm, el_shear=el_shear, el_ortho=el_ortho, el_shrco=el_shrco, param_2=param_2, param_3=param_3,
        pl_harde=pl_harde, pl_ortho=pl_ortho, pl_iskin=pl_iskin, pl_asymm=pl_asymm, pl_waist=pl_waist, pl_biaxf=pl_biaxf, pl_compr=pl_compr, pl_damag=pl_damag,
        nf_curve=nf_curve, nf_ortho=nf_ortho, nf_depen=nf_depen, sf_curve=sf_curve, sf_param=sf_param, sf_postc=sf_postc, param_4=param_4, param_5=param_5,
        cr_harde=cr_harde, cr_ortho=cr_ortho, cr_iskin=cr_iskin, cr_postc=cr_postc, cr_param=cr_param, cr_check=cr_check, param_6=param_6, mf_init=mf_init,
        title=title
    )
    model.mat_law29s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=29, rho0=rho0, title=title,
        params={
            "rho": rho0, "version": version,
            "el_young": el_young, "el_poiss": el_poiss,
        }
    )




def read_mat_law34(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW34`` or ``/MAT/BOLTZMAN`` (M190): Boltzmann linear viscoelastic relaxation model."""
    from ...model.entities import MatLaw34, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW34/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    k = 0.0
    g0, gl, beta = 0.0, 0.0, 0.0
    p0, phi, gamma0 = 0.0, 0.0, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW34_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW34_2")
            k = _safe_float(c1[0]) if len(c1) > 0 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW34_3")
            g0 = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            gl = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            beta = _safe_float(c2[2]) if len(c2) > 2 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW34_4")
            p0 = _safe_float(c3[0]) if len(c3) > 0 else 0.0
            phi = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            gamma0 = _safe_float(c3[2]) if len(c3) > 2 else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            k = _safe_float(t1[0]) if len(t1) > 0 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            g0 = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            gl = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            beta = _safe_float(t2[2]) if len(t2) > 2 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            p0 = _safe_float(t3[0]) if len(t3) > 0 else 0.0
            phi = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            gamma0 = _safe_float(t3[2]) if len(t3) > 2 else 0.0

    mat = MatLaw34(
        id=mat_id, rho0=rho0, rhor=rhor,
        k=k, g0=g0, gl=gl, beta=beta,
        p0=p0, phi=phi, gamma0=gamma0,
        title=title
    )
    model.mat_law34s[mat_id] = mat
    if 3.0 * k + g0 > 0.0:
        young = (9.0 * k * g0) / (3.0 * k + g0)
        nu = (3.0 * k - 2.0 * g0) / (2.0 * (3.0 * k + g0))
    else:
        young = 0.0
        nu = 0.0

    model.materials[mat_id] = Material(
        id=mat_id, law=34, rho0=rho0, title=title,
        params={
            "rho": rho0, "rhor": rhor, "k": k, "bulk": k,
            "g0": g0, "gl": gl, "gi": gl, "beta": beta, "decay": beta,
            "p0": p0, "phi": phi, "gamma0": gamma0, "gama0": gamma0,
            "E": young, "nu": nu,

        }
    )





def read_mat_law23(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW23`` or ``/MAT/PLAS_DAMA`` (M190): Lemaitre ductile damage elastoplastic model."""
    from ...model.entities import MatLaw23, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW23/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    e, nu = 0.0, 0.0
    a, b, n, eps_max, sig_max = 0.0, 0.0, 0.0, 0.0, 0.0
    c, eps_0 = 0.0, 0.0
    icc = 0
    eps_dam, e_t = 0.0, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW23_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW23_2")
            e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW23_3")
            a = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            b = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            n = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            eps_max = _safe_float(c2[3]) if len(c2) > 3 else 0.0
            sig_max = _safe_float(c2[4]) if len(c2) > 4 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW23_4")
            c = _safe_float(c3[0]) if len(c3) > 0 else 0.0
            eps_0 = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            icc = _safe_int(c3[2]) if len(c3) > 2 else 0
        if len(valid_cards) > 4:
            c4 = valid_cards[4].cut("MAT_LAW23_5")
            eps_dam = _safe_float(c4[0]) if len(c4) > 0 else 0.0
            e_t = _safe_float(c4[1]) if len(c4) > 1 else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            a = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            b = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            n = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            eps_max = _safe_float(t2[3]) if len(t2) > 3 else 0.0
            sig_max = _safe_float(t2[4]) if len(t2) > 4 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            c = _safe_float(t3[0]) if len(t3) > 0 else 0.0
            eps_0 = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            icc = _safe_int(t3[2]) if len(t3) > 2 else 0
        if len(valid_cards) > 4:
            t4 = valid_cards[4].tokens()
            eps_dam = _safe_float(t4[0]) if len(t4) > 0 else 0.0
            e_t = _safe_float(t4[1]) if len(t4) > 1 else 0.0

    mat = MatLaw23(
        id=mat_id, rho0=rho0, rhor=rhor,
        e=e, nu=nu, a=a, b=b, n=n, eps_max=eps_max, sig_max=sig_max,
        c=c, eps_0=eps_0, icc=icc,
        eps_dam=eps_dam, e_t=e_t,
        title=title
    )
    model.mat_law23s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=23, rho0=rho0, title=title,
        params={
            "rho": rho0, "e": e, "nu": nu, "a": a, "b": b, "n": n,
            "eps_max": eps_max, "sig_max": sig_max, "c": c, "eps_0": eps_0, "icc": icc,
            "eps_dam": eps_dam, "e_t": e_t,
        }
    )




def read_mat_law78(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW78`` (M190): Rate-dependent elastoplastic constitutive law 78."""
    from ...model.entities import MatLaw78, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/MAT/LAW78/{mat_id}: missing data card", block.source)
        return

    rho0, rhor = 0.0, 0.0
    e, nu, eps_max, sig_max = 0.0, 0.0, 0.0, 0.0
    y, b, c, h, b0 = 0.0, 0.0, 0.0, 0.0, 0.0
    m, rsat, ea, ce = 0.0, 0.0, 0.0, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW78_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW78_2")
            e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            eps_max = _safe_float(c1[2]) if len(c1) > 2 else 0.0
            sig_max = _safe_float(c1[3]) if len(c1) > 3 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW78_3")
            y = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            b = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            c = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            h = _safe_float(c2[3]) if len(c2) > 3 else 0.0
            b0 = _safe_float(c2[4]) if len(c2) > 4 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW78_4")
            m = _safe_float(c3[0]) if len(c3) > 0 else 0.0
            rsat = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            ea = _safe_float(c3[2]) if len(c3) > 2 else 0.0
            ce = _safe_float(c3[3]) if len(c3) > 3 else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            eps_max = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            sig_max = _safe_float(t1[3]) if len(t1) > 3 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            y = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            b = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            c = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            h = _safe_float(t2[3]) if len(t2) > 3 else 0.0
            b0 = _safe_float(t2[4]) if len(t2) > 4 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            m = _safe_float(t3[0]) if len(t3) > 0 else 0.0
            rsat = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            ea = _safe_float(t3[2]) if len(t3) > 2 else 0.0
            ce = _safe_float(t3[3]) if len(t3) > 3 else 0.0

    mat = MatLaw78(
        id=mat_id, rho0=rho0, rhor=rhor,
        e=e, nu=nu, eps_max=eps_max, sig_max=sig_max,
        y=y, b=b, c=c, h=h, b0=b0,
        m=m, rsat=rsat, ea=ea, ce=ce,
        title=title
    )
    model.mat_law78s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=78, rho0=rho0, title=title,
        params={
            "rho": rho0, "e": e, "nu": nu, "eps_max": eps_max, "sig_max": sig_max,
            "y": y, "b": b, "c": c, "h": h, "b0": b0,
            "m": m, "rsat": rsat, "ea": ea, "ce": ce,
        }
    )




# ==============================================================================
# Milestone M191: Advanced Materials, Multi-axial & Visual Failures, Preloads & BCs
# ==============================================================================

def read_mat_law100(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW100`` or ``/MAT/VISC_HYP`` / ``/MAT/MNF`` (M191/M570): Multi-Network Visco-Hyperelastic polymer model."""
    from ...model.entities import MatLaw100, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]

    rho0, rhor = 0.0, 0.0
    n_net = 0
    flag_he, flag_cr = 1, 0
    c10, c01, c20, c11, c02 = 0.0, 0.0, 0.0, 0.0, 0.0
    c30, c21, c12, c03, d1 = 0.0, 0.0, 0.0, 0.0, 0.0
    d2, d3, mue1, d, lambda_m = 0.0, 0.0, 0.0, 0.0, 7.0
    itype, fct_id_ab, nu, fscale_ab = 1, 0, 0.0, 1.0
    fct_id_sm, fct_id_bm, fscale_sm, fscale_bm = 0, 0, 1.0, 1.0
    a_pl, sigma_pl, f_pl = 1.0, 1.0, 1.0
    epsilon_f, n_pl = 1.0, 1
    networks: list[dict] = []

    if len(valid_cards) == 0:
        log.error(f"/MAT/LAW100/{mat_id}: missing data cards", block.source)
        return

    # Card 0: Density
    if block.fixed:
        c0 = valid_cards[0].cut("MAT_LAW100_1")
        rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
        rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
    else:
        t0 = valid_cards[0].tokens()
        rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
        rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0

    card_idx = 1
    if len(valid_cards) > 1:
        if block.fixed:
            c1 = valid_cards[1].cut("MAT_LAW100_2")
            if len(c1) >= 3 and c1[2].strip():
                n_net = _safe_int(c1[0])
                flag_he = _safe_int(c1[1], 1)
                flag_cr = _safe_int(c1[2], 0)
            elif len(c1) == 2:
                # Legacy / flat format fallback
                flag_he = _safe_int(c1[0], 1)
                flag_cr = _safe_int(c1[1], 0)
            else:
                flag_he = _safe_int(c1[0], 1) if len(c1) > 0 else 1
        else:
            t1 = valid_cards[1].tokens()
            if len(t1) >= 3:
                n_net = _safe_int(t1[0])
                flag_he = _safe_int(t1[1], 1)
                flag_cr = _safe_int(t1[2], 0)
            elif len(t1) == 2:
                flag_he = _safe_int(t1[0], 1)
                flag_cr = _safe_int(t1[1], 0)
            else:
                flag_he = _safe_int(t1[0], 1) if len(t1) > 0 else 1
        card_idx = 2

    # Check for legacy flat 8-card format (e.g. len(valid_cards) >= 8 and card 2 has 5 tokens)
    is_legacy_flat = False
    if len(valid_cards) >= 8 and card_idx == 2:
        t2 = valid_cards[2].tokens()
        t3 = valid_cards[3].tokens()
        t4 = valid_cards[4].tokens()
        if len(t2) == 5 and len(t3) == 5 and len(t4) == 5:
            is_legacy_flat = True

    if is_legacy_flat:
        if block.fixed:
            c2 = valid_cards[2].cut("MAT_LAW100_HE1_1")
            c10 = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            c01 = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            c20 = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            c11 = _safe_float(c2[3]) if len(c2) > 3 else 0.0
            c02 = _safe_float(c2[4]) if len(c2) > 4 else 0.0
            c3 = valid_cards[3].cut("MAT_LAW100_HE1_2")
            c30 = _safe_float(c3[0]) if len(c3) > 0 else 0.0
            c21 = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            c12 = _safe_float(c3[2]) if len(c3) > 2 else 0.0
            c03 = _safe_float(c3[3]) if len(c3) > 3 else 0.0
            d1 = _safe_float(c3[4]) if len(c3) > 4 else 0.0
            c4 = valid_cards[4].cut("MAT_LAW100_HE1_3")
            d2 = _safe_float(c4[0]) if len(c4) > 0 else 0.0
            d3 = _safe_float(c4[1]) if len(c4) > 1 else 0.0
            mue1 = _safe_float(c4[2]) if len(c4) > 2 else 0.0
            d = _safe_float(c4[3]) if len(c4) > 3 else 0.0
            lambda_m = _safe_float(c4[4], 7.0) if len(c4) > 4 else 7.0
            c5 = valid_cards[5].cut("MAT_LAW100_HE2_2")
            itype = _safe_int(c5[0], 1) if len(c5) > 0 else 1
            fct_id_ab = _safe_int(c5[1]) if len(c5) > 1 else 0
            nu = _safe_float(c5[2]) if len(c5) > 2 else 0.0
            fct_id_sm = _safe_int(c5[3]) if len(c5) > 3 else 0
            fct_id_bm = _safe_int(c5[4]) if len(c5) > 4 else 0
            c6 = valid_cards[6].cut("MAT_LAW100_PL")
            fscale_sm = _safe_float(c6[0], 1.0) if len(c6) > 0 else 1.0
            fscale_bm = _safe_float(c6[1], 1.0) if len(c6) > 1 else 1.0
            a_pl = _safe_float(c6[2], 1.0) if len(c6) > 2 else 1.0
            sigma_pl = _safe_float(c6[3], 1.0) if len(c6) > 3 else 1.0
            f_pl = _safe_float(c6[4], 1.0) if len(c6) > 4 else 1.0
            c7 = valid_cards[7].cut("MAT_LAW100_PL")
            epsilon_f = _safe_float(c7[0], 1.0) if len(c7) > 0 else 1.0
            n_pl = _safe_int(c7[1], 1) if len(c7) > 1 else 1
        else:
            t2 = valid_cards[2].tokens()
            c10 = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            c01 = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            c20 = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            c11 = _safe_float(t2[3]) if len(t2) > 3 else 0.0
            c02 = _safe_float(t2[4]) if len(t2) > 4 else 0.0
            t3 = valid_cards[3].tokens()
            c30 = _safe_float(t3[0]) if len(t3) > 0 else 0.0
            c21 = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            c12 = _safe_float(t3[2]) if len(t3) > 2 else 0.0
            c03 = _safe_float(t3[3]) if len(t3) > 3 else 0.0
            d1 = _safe_float(t3[4]) if len(t3) > 4 else 0.0
            t4 = valid_cards[4].tokens()
            d2 = _safe_float(t4[0]) if len(t4) > 0 else 0.0
            d3 = _safe_float(t4[1]) if len(t4) > 1 else 0.0
            mue1 = _safe_float(t4[2]) if len(t4) > 2 else 0.0
            d = _safe_float(t4[3]) if len(t4) > 3 else 0.0
            lambda_m = _safe_float(t4[4], 7.0) if len(t4) > 4 else 7.0
            t5 = valid_cards[5].tokens()
            itype = _safe_int(t5[0], 1) if len(t5) > 0 else 1
            fct_id_ab = _safe_int(t5[1]) if len(t5) > 1 else 0
            nu = _safe_float(t5[2]) if len(t5) > 2 else 0.0
            fct_id_sm = _safe_int(t5[3]) if len(t5) > 3 else 0
            fct_id_bm = _safe_int(t5[4]) if len(t5) > 4 else 0
            t6 = valid_cards[6].tokens()
            fscale_sm = _safe_float(t6[0], 1.0) if len(t6) > 0 else 1.0
            fscale_bm = _safe_float(t6[1], 1.0) if len(t6) > 1 else 1.0
            a_pl = _safe_float(t6[2], 1.0) if len(t6) > 2 else 1.0
            sigma_pl = _safe_float(t6[3], 1.0) if len(t6) > 3 else 1.0
            f_pl = _safe_float(t6[4], 1.0) if len(t6) > 4 else 1.0
            t7 = valid_cards[7].tokens()
            epsilon_f = _safe_float(t7[0], 1.0) if len(t7) > 0 else 1.0
            n_pl = _safe_int(t7[1], 1) if len(t7) > 1 else 1
        card_idx = 8
    else:
        # Standard dynamic OpenRadioss card reading (LAW100.cfg)
        if flag_he == 1:
            if card_idx < len(valid_cards):
                c = valid_cards[card_idx].cut("MAT_LAW100_HE1_1") if block.fixed else valid_cards[card_idx].tokens()
                c10 = _safe_float(c[0]) if len(c) > 0 else 0.0
                c01 = _safe_float(c[1]) if len(c) > 1 else 0.0
                c20 = _safe_float(c[2]) if len(c) > 2 else 0.0
                c11 = _safe_float(c[3]) if len(c) > 3 else 0.0
                c02 = _safe_float(c[4]) if len(c) > 4 else 0.0
                card_idx += 1
            if card_idx < len(valid_cards):
                c = valid_cards[card_idx].cut("MAT_LAW100_HE1_2") if block.fixed else valid_cards[card_idx].tokens()
                c30 = _safe_float(c[0]) if len(c) > 0 else 0.0
                c21 = _safe_float(c[1]) if len(c) > 1 else 0.0
                c12 = _safe_float(c[2]) if len(c) > 2 else 0.0
                c03 = _safe_float(c[3]) if len(c) > 3 else 0.0
                card_idx += 1
            if card_idx < len(valid_cards):
                c = valid_cards[card_idx].cut("MAT_LAW100_HE1_3") if block.fixed else valid_cards[card_idx].tokens()
                d1 = _safe_float(c[0]) if len(c) > 0 else 0.0
                d2 = _safe_float(c[1]) if len(c) > 1 else 0.0
                d3 = _safe_float(c[2]) if len(c) > 2 else 0.0
                card_idx += 1
        elif flag_he == 2:
            if card_idx < len(valid_cards):
                c = valid_cards[card_idx].cut("MAT_LAW100_HE2_1") if block.fixed else valid_cards[card_idx].tokens()
                mue1 = _safe_float(c[0]) if len(c) > 0 else 0.0
                d = _safe_float(c[1]) if len(c) > 1 else 0.0
                lambda_m = _safe_float(c[2], 7.0) if len(c) > 2 else 7.0
                card_idx += 1
            if card_idx < len(valid_cards):
                c = valid_cards[card_idx].cut("MAT_LAW100_HE2_2") if block.fixed else valid_cards[card_idx].tokens()
                itype = _safe_int(c[0], 1) if len(c) > 0 else 1
                fct_id_ab = _safe_int(c[1]) if len(c) > 1 else 0
                nu = _safe_float(c[2]) if len(c) > 2 else 0.0
                fscale_ab = _safe_float(c[3], 1.0) if len(c) > 3 else 1.0
                card_idx += 1
        elif flag_he == 3:
            if card_idx < len(valid_cards):
                c = valid_cards[card_idx].cut("MAT_LAW100_HE3_1") if block.fixed else valid_cards[card_idx].tokens()
                c10 = _safe_float(c[0]) if len(c) > 0 else 0.0
                d1 = _safe_float(c[1]) if len(c) > 1 else 0.0
                card_idx += 1
        elif flag_he == 4:
            if card_idx < len(valid_cards):
                c = valid_cards[card_idx].cut("MAT_LAW100_HE4_1") if block.fixed else valid_cards[card_idx].tokens()
                c10 = _safe_float(c[0]) if len(c) > 0 else 0.0
                c01 = _safe_float(c[1]) if len(c) > 1 else 0.0
                d1 = _safe_float(c[2]) if len(c) > 2 else 0.0
                card_idx += 1
        elif flag_he == 5:
            if card_idx < len(valid_cards):
                c = valid_cards[card_idx].cut("MAT_LAW100_HE5_1") if block.fixed else valid_cards[card_idx].tokens()
                c10 = _safe_float(c[0]) if len(c) > 0 else 0.0
                c20 = _safe_float(c[1]) if len(c) > 1 else 0.0
                c30 = _safe_float(c[2]) if len(c) > 2 else 0.0
                d1 = _safe_float(c[3]) if len(c) > 3 else 0.0
                card_idx += 1
        elif flag_he == 13:
            if card_idx < len(valid_cards):
                c = valid_cards[card_idx].cut("MAT_LAW100_HE13_1") if block.fixed else valid_cards[card_idx].tokens()
                fct_id_sm = _safe_int(c[0]) if len(c) > 0 else 0
                fct_id_bm = _safe_int(c[1]) if len(c) > 1 else 0
                fscale_sm = _safe_float(c[2], 1.0) if len(c) > 2 else 1.0
                fscale_bm = _safe_float(c[3], 1.0) if len(c) > 3 else 1.0
                card_idx += 1

        # Creep / Plasticity in equilibrium network
        if flag_cr == 1 and card_idx < len(valid_cards):
            c = valid_cards[card_idx].cut("MAT_LAW100_PL") if block.fixed else valid_cards[card_idx].tokens()
            a_pl = _safe_float(c[0], 1.0) if len(c) > 0 else 1.0
            sigma_pl = _safe_float(c[1], 1.0) if len(c) > 1 else 1.0
            f_pl = _safe_float(c[2], 1.0) if len(c) > 2 else 1.0
            epsilon_f = _safe_float(c[3], 1.0) if len(c) > 3 else 1.0
            n_pl = _safe_int(c[4], 1) if len(c) > 4 else 1
            card_idx += 1

        # Secondary Networks (N_net >= 1)
        for net_i in range(n_net):
            if card_idx >= len(valid_cards):
                break
            # Network header line
            chdr = valid_cards[card_idx].cut("MAT_LAW100_NET_HDR") if block.fixed else valid_cards[card_idx].tokens()
            card_idx += 1
            fvisc = _safe_int(chdr[1], 1) if len(chdr) > 1 else 1
            stiffness = _safe_float(chdr[2], 1.0) if len(chdr) > 2 else 1.0

            net_dict = {
                "network_id": net_i + 1,
                "flag_visc": fvisc,
                "stiffness": stiffness,
            }
            if card_idx < len(valid_cards):
                cdata = valid_cards[card_idx].cut("MAT_LAW100_NET_V1") if block.fixed else valid_cards[card_idx].tokens()
                card_idx += 1
                if fvisc == 1:
                    net_dict["a"] = _safe_float(cdata[0]) if len(cdata) > 0 else 0.0
                    net_dict["c"] = _safe_float(cdata[1], -0.7) if len(cdata) > 1 else -0.7
                    net_dict["expc"] = net_dict["c"]
                    net_dict["m"] = _safe_float(cdata[2], 1.0) if len(cdata) > 2 else 1.0
                    net_dict["expm"] = net_dict["m"]
                    net_dict["ksi"] = _safe_float(cdata[3], 0.01) if len(cdata) > 3 else 0.01
                    net_dict["tauref"] = _safe_float(cdata[4], 1.0) if len(cdata) > 4 else 1.0
                    net_dict["tau_ref"] = net_dict["tauref"]
                elif fvisc == 2:
                    net_dict["a"] = _safe_float(cdata[0]) if len(cdata) > 0 else 0.0
                    b_val = _safe_float(cdata[1]) if len(cdata) > 1 else 0.0
                    net_dict["b0"] = b_val
                    net_dict["b"] = b_val
                    n_val = _safe_float(cdata[2], 1.0) if len(cdata) > 2 else 1.0
                    net_dict["expn"] = n_val
                    net_dict["n"] = n_val
                elif fvisc == 3:
                    net_dict["a"] = _safe_float(cdata[0]) if len(cdata) > 0 else 0.0
                    n_val = _safe_float(cdata[1], 1.0) if len(cdata) > 1 else 1.0
                    net_dict["expn"] = n_val
                    net_dict["n"] = n_val
                    m_val = _safe_float(cdata[2], 1.0) if len(cdata) > 2 else 1.0
                    net_dict["expm"] = m_val
                    net_dict["m"] = m_val
            networks.append(net_dict)

    mat = MatLaw100(
        id=mat_id, rho0=rho0, rhor=rhor, n_net=len(networks) if networks else n_net,
        flag_he=flag_he, flag_cr=flag_cr,
        c10=c10, c01=c01, c20=c20, c11=c11, c02=c02,
        c30=c30, c21=c21, c12=c12, c03=c03, d1=d1,
        d2=d2, d3=d3, mue1=mue1, d=d, lambda_m=lambda_m,
        itype=itype, fct_id_ab=fct_id_ab, nu_val=nu, fscale_ab=fscale_ab,
        fct_id_sm=fct_id_sm, fct_id_bm=fct_id_bm,
        fscale_sm=fscale_sm, fscale_bm=fscale_bm,
        a_pl=a_pl, sigma_pl=sigma_pl, f_pl=f_pl,
        epsilon_f=epsilon_f, n_pl=n_pl, title=title,
        networks=networks,
    )
    mat.params = {
        "rho": rho0, "rhor": rhor, "n_net": mat.n_net, "flag_he": flag_he, "flag_cr": flag_cr,
        "c10": c10, "c01": c01, "c20": c20, "c11": c11, "c02": c02,
        "c30": c30, "c21": c21, "c12": c12, "c03": c03, "d1": d1,
        "d2": d2, "d3": d3, "mu": mue1, "mue1": mue1, "d": d, "lambda_m": lambda_m,
        "itype": itype, "fct_id_ab": fct_id_ab, "nu": nu, "fscale_ab": fscale_ab,
        "fct_id_sm": fct_id_sm, "fct_id_bm": fct_id_bm,
        "fscale_sm": fscale_sm, "fscale_bm": fscale_bm,
        "a_pl": a_pl, "sigma_pl": sigma_pl, "f_pl": f_pl,
        "epsilon_f": epsilon_f, "n_pl": n_pl,
        "networks": networks,
        "G": mat.G, "K": mat.K, "E": mat.E, "sb": mat.sb,
    }
    model.mat_law100s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=100, rho0=rho0, title=title,
        params=mat.params,
    )





def read_mat_law97(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW97`` or ``/MAT/EXPLOSIVE_JWLS`` (M191): High-explosive detonation EOS."""
    from ...model.entities import MatLaw97, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    rho0, rhor = 0.0, 0.0
    p0, psh = 0.0, 0.0
    ibfrac = 0
    d, pcj, e0, omega, c = 0.0, 0.0, 0.0, 0.0, 0.0
    a1, a2, a3, a4, a5 = 0.0, 0.0, 0.0, 0.0, 0.0
    r1, r2, r3, r4, r5 = 0.0, 0.0, 0.0, 0.0, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW97_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW97_2")
            p0 = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            psh = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            ibfrac = _safe_int(c1[2]) if len(c1) > 2 else 0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW97_3")
            d = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            pcj = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            e0 = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            omega = _safe_float(c2[3]) if len(c2) > 3 else 0.0
            c = _safe_float(c2[4]) if len(c2) > 4 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW97_4")
            a1 = _safe_float(c3[0]) if len(c3) > 0 else 0.0
            a2 = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            a3 = _safe_float(c3[2]) if len(c3) > 2 else 0.0
            a4 = _safe_float(c3[3]) if len(c3) > 3 else 0.0
            a5 = _safe_float(c3[4]) if len(c3) > 4 else 0.0
        if len(valid_cards) > 4:
            c4 = valid_cards[4].cut("MAT_LAW97_5")
            r1 = _safe_float(c4[0]) if len(c4) > 0 else 0.0
            r2 = _safe_float(c4[1]) if len(c4) > 1 else 0.0
            r3 = _safe_float(c4[2]) if len(c4) > 2 else 0.0
            r4 = _safe_float(c4[3]) if len(c4) > 3 else 0.0
            r5 = _safe_float(c4[4]) if len(c4) > 4 else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            p0 = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            psh = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            ibfrac = _safe_int(t1[2]) if len(t1) > 2 else 0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            d = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            pcj = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            e0 = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            omega = _safe_float(t2[3]) if len(t2) > 3 else 0.0
            c = _safe_float(t2[4]) if len(t2) > 4 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            a1 = _safe_float(t3[0]) if len(t3) > 0 else 0.0
            a2 = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            a3 = _safe_float(t3[2]) if len(t3) > 2 else 0.0
            a4 = _safe_float(t3[3]) if len(t3) > 3 else 0.0
            a5 = _safe_float(t3[4]) if len(t3) > 4 else 0.0
        if len(valid_cards) > 4:
            t4 = valid_cards[4].tokens()
            r1 = _safe_float(t4[0]) if len(t4) > 0 else 0.0
            r2 = _safe_float(t4[1]) if len(t4) > 1 else 0.0
            r3 = _safe_float(t4[2]) if len(t4) > 2 else 0.0
            r4 = _safe_float(t4[3]) if len(t4) > 3 else 0.0
            r5 = _safe_float(t4[4]) if len(t4) > 4 else 0.0

    mat = MatLaw97(
        id=mat_id, rho0=rho0, rhor=rhor, p0=p0, psh=psh, ibfrac=ibfrac,
        d=d, pcj=pcj, e0=e0, omega=omega, c=c,
        a1=a1, a2=a2, a3=a3, a4=a4, a5=a5,
        r1=r1, r2=r2, r3=r3, r4=r4, r5=r5, title=title
    )
    model.mat_law97s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=97, rho0=rho0, title=title,
        params={
            "rho": rho0, "rhor": rhor, "p0": p0, "psh": psh, "ibfrac": ibfrac,
            "d": d, "pcj": pcj, "e0": e0, "omega": omega, "c": c,
            "a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5,
            "r1": r1, "r2": r2, "r3": r3, "r4": r4, "r5": r5,
        }
    )




def read_mat_law71(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW71`` or ``/MAT/SUPER_ELAS`` (M191, M582): Nitinol / superelastic shape memory alloy."""
    from ...model.entities import MatLaw71, Material
    from ...materials.law71_nitinol import build_law71
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    rho0, rhor = 0.0, 0.0
    e, nu, e_mart = 0.0, 0.0, 0.0
    sig_sas, sig_fas, sig_ssa, sig_fsa, alpha = 0.0, 0.0, 0.0, 0.0, 0.0
    epsl, cas, csa, tsas, tfas = 0.0, 0.0, 0.0, 0.0, 0.0
    tssa, tfsa, cp, tini = 0.0, 0.0, 0.0, 0.0

    if block.fixed:
        # Card 1: RHO_I, Refer_Rho
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW71_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        is_5field = False
        if len(valid_cards) > 1:
            tokens1 = valid_cards[1].tokens()
            if len(tokens1) >= 5 or len(valid_cards[1].raw.rstrip()) > 60:
                is_5field = True
        if is_5field:
            if len(valid_cards) > 1:
                c1 = valid_cards[1].cut([20, 20, 20, 20, 20])
                e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
                nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
                e_mart = _safe_float(c1[2]) if len(c1) > 2 else 0.0
                sig_sas = _safe_float(c1[3]) if len(c1) > 3 else 0.0
                sig_fas = _safe_float(c1[4]) if len(c1) > 4 else 0.0
            if len(valid_cards) > 2:
                c2 = valid_cards[2].cut([20, 20, 20, 20, 20])
                sig_ssa = _safe_float(c2[0]) if len(c2) > 0 else 0.0
                sig_fsa = _safe_float(c2[1]) if len(c2) > 1 else 0.0
                alpha = _safe_float(c2[2]) if len(c2) > 2 else 0.0
                epsl = _safe_float(c2[3]) if len(c2) > 3 else 0.0
                cas = _safe_float(c2[4]) if len(c2) > 4 else 0.0
            if len(valid_cards) > 3:
                c3 = valid_cards[3].cut([20, 20, 20, 20, 20])
                csa = _safe_float(c3[0]) if len(c3) > 0 else 0.0
                tsas = _safe_float(c3[1]) if len(c3) > 1 else 0.0
                tfas = _safe_float(c3[2]) if len(c3) > 2 else 0.0
                tssa = _safe_float(c3[3]) if len(c3) > 3 else 0.0
                tfsa = _safe_float(c3[4]) if len(c3) > 4 else 0.0
            if len(valid_cards) > 4:
                c4 = valid_cards[4].cut([20, 20, 20, 20, 20])
                cp = _safe_float(c4[0]) if len(c4) > 0 else 0.0
                tini = _safe_float(c4[1]) if len(c4) > 1 else 0.0
        else:
            # Card 2: E, nu, E_mart
            if len(valid_cards) > 1:
                c1 = valid_cards[1].cut("MAT_LAW71_2")
                e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
                nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
                e_mart = _safe_float(c1[2]) if len(c1) > 2 else 0.0
            # Card 3: Sig_sas, Sig_fas, Sig_ssa, Sig_fsa, Alpha
            if len(valid_cards) > 2:
                c2 = valid_cards[2].cut("MAT_LAW71_3")
                sig_sas = _safe_float(c2[0]) if len(c2) > 0 else 0.0
                sig_fas = _safe_float(c2[1]) if len(c2) > 1 else 0.0
                sig_ssa = _safe_float(c2[2]) if len(c2) > 2 else 0.0
                sig_fsa = _safe_float(c2[3]) if len(c2) > 3 else 0.0
                alpha = _safe_float(c2[4]) if len(c2) > 4 else 0.0
            # Card 4: EpsL, CAS, CSA, TSAS, TFAS
            if len(valid_cards) > 3:
                c3 = valid_cards[3].cut("MAT_LAW71_4")
                epsl = _safe_float(c3[0]) if len(c3) > 0 else 0.0
                cas = _safe_float(c3[1]) if len(c3) > 1 else 0.0
                csa = _safe_float(c3[2]) if len(c3) > 2 else 0.0
                tsas = _safe_float(c3[3]) if len(c3) > 3 else 0.0
                tfas = _safe_float(c3[4]) if len(c3) > 4 else 0.0
            # Card 5: TSSA, TFSA, CP, TINI
            if len(valid_cards) > 4:
                c4 = valid_cards[4].cut("MAT_LAW71_5")
                tssa = _safe_float(c4[0]) if len(c4) > 0 else 0.0
                tfsa = _safe_float(c4[1]) if len(c4) > 1 else 0.0
                cp = _safe_float(c4[2]) if len(c4) > 2 else 0.0
                tini = _safe_float(c4[3]) if len(c4) > 3 else 0.0
    else:
        # Card 1: RHO_I, Refer_Rho
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        is_5field = False
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            if len(t1) >= 5:
                is_5field = True
        if is_5field:
            if len(valid_cards) > 1:
                t1 = valid_cards[1].tokens()
                e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
                nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
                e_mart = _safe_float(t1[2]) if len(t1) > 2 else 0.0
                sig_sas = _safe_float(t1[3]) if len(t1) > 3 else 0.0
                sig_fas = _safe_float(t1[4]) if len(t1) > 4 else 0.0
            if len(valid_cards) > 2:
                t2 = valid_cards[2].tokens()
                sig_ssa = _safe_float(t2[0]) if len(t2) > 0 else 0.0
                sig_fsa = _safe_float(t2[1]) if len(t2) > 1 else 0.0
                alpha = _safe_float(t2[2]) if len(t2) > 2 else 0.0
                epsl = _safe_float(t2[3]) if len(t2) > 3 else 0.0
                cas = _safe_float(t2[4]) if len(t2) > 4 else 0.0
            if len(valid_cards) > 3:
                t3 = valid_cards[3].tokens()
                csa = _safe_float(t3[0]) if len(t3) > 0 else 0.0
                tsas = _safe_float(t3[1]) if len(t3) > 1 else 0.0
                tfas = _safe_float(t3[2]) if len(t3) > 2 else 0.0
                tssa = _safe_float(t3[3]) if len(t3) > 3 else 0.0
                tfsa = _safe_float(t3[4]) if len(t3) > 4 else 0.0
            if len(valid_cards) > 4:
                t4 = valid_cards[4].tokens()
                cp = _safe_float(t4[0]) if len(t4) > 0 else 0.0
                tini = _safe_float(t4[1]) if len(t4) > 1 else 0.0
        else:
            # Card 2: E, nu, E_mart
            if len(valid_cards) > 1:
                t1 = valid_cards[1].tokens()
                e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
                nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
                e_mart = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            # Card 3: Sig_sas, Sig_fas, Sig_ssa, Sig_fsa, Alpha
            if len(valid_cards) > 2:
                t2 = valid_cards[2].tokens()
                sig_sas = _safe_float(t2[0]) if len(t2) > 0 else 0.0
                sig_fas = _safe_float(t2[1]) if len(t2) > 1 else 0.0
                sig_ssa = _safe_float(t2[2]) if len(t2) > 2 else 0.0
                sig_fsa = _safe_float(t2[3]) if len(t2) > 3 else 0.0
                alpha = _safe_float(t2[4]) if len(t2) > 4 else 0.0
            # Card 4: EpsL, CAS, CSA, TSAS, TFAS
            if len(valid_cards) > 3:
                t3 = valid_cards[3].tokens()
                epsl = _safe_float(t3[0]) if len(t3) > 0 else 0.0
                cas = _safe_float(t3[1]) if len(t3) > 1 else 0.0
                csa = _safe_float(t3[2]) if len(t3) > 2 else 0.0
                tsas = _safe_float(t3[3]) if len(t3) > 3 else 0.0
                tfas = _safe_float(t3[4]) if len(t3) > 4 else 0.0
            # Card 5: TSSA, TFSA, CP, TINI
            if len(valid_cards) > 4:
                t4 = valid_cards[4].tokens()
                tssa = _safe_float(t4[0]) if len(t4) > 0 else 0.0
                tfsa = _safe_float(t4[1]) if len(t4) > 1 else 0.0
                cp = _safe_float(t4[2]) if len(t4) > 2 else 0.0
                tini = _safe_float(t4[3]) if len(t4) > 3 else 0.0

    if rhor == 0.0:
        rhor = rho0
    if tssa == 0.0:
        tssa = 298.0
    if tfsa == 0.0:
        tfsa = 298.0
    if tsas == 0.0:
        tsas = 298.0
    if tfas == 0.0:
        tfas = 298.0
    if cp == 0.0:
        cp = 1.0e20
    if tini == 0.0:
        tini = 360.0

    mat = MatLaw71(
        id=mat_id, rho0=rho0, rhor=rhor, e=e, nu=nu, e_mart=e_mart,
        sig_sas=sig_sas, sig_fas=sig_fas, sig_ssa=sig_ssa, sig_fsa=sig_fsa,
        alpha=alpha, epsl=epsl, cas=cas, csa=csa,
        tsas=tsas, tfas=tfas, tssa=tssa, tfsa=tfsa, cp=cp, tini=tini, title=title
    )
    model.mat_law71s[mat_id] = mat
    model.materials[mat_id] = build_law71(mat)




def read_mat_law73(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW73``, ``/MAT/BARLAT2000``, or ``/MAT/HILL_THERM`` (M191, M561):
    Thermal Hill orthotropic material model.
    Upstream Fortran reference: hm_read_mat73.F and CFG radioss140/MAT/matl73_73.cfg.
    """
    from ...model.entities import MatLaw73, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]

    rho0, rhor = 0.0, 0.0
    e, nu = 0.0, 0.0
    ifunce, einf, ce = 0, 0.0, 0.0
    r00, r45, r90 = 1.0, 1.0, 1.0
    chard, iyield = 0.0, 0
    eps_max, epsr1, epsr2 = 1.0e30, 1.0e30, 2.0e30
    table_id, fscale, pscale = 0, 1.0, 1.0
    t0, rhocp = 293.0, 0.0

    is_fixed = block.fixed
    if is_fixed:
        for vc in valid_cards[:4]:
            if "," in vc.raw or (len(vc.tokens()) > 1 and len(vc.raw[:20].split()) > 1):
                is_fixed = False
                break

    # Distinguish between legacy 5-card layout and standard 7-card layout
    if len(valid_cards) <= 5:
        # Legacy 5-card layout (M191)
        if is_fixed:
            if len(valid_cards) > 0:
                c0 = valid_cards[0].cut([20, 20])
                rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
                rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
            if len(valid_cards) > 1:
                c1 = valid_cards[1].cut([20, 20, 20, 20, 20])
                e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
                nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
                r00 = _safe_float(c1[2], 1.0) if len(c1) > 2 else 1.0
                r45 = _safe_float(c1[3], 1.0) if len(c1) > 3 else 1.0
                r90 = _safe_float(c1[4], 1.0) if len(c1) > 4 else 1.0
            if len(valid_cards) > 2:
                c2 = valid_cards[2].cut([20, 20, 20, 20, 10])
                chard = _safe_float(c2[0]) if len(c2) > 0 else 0.0
                eps_max = _safe_float(c2[1], 1.0e30) if len(c2) > 1 and c2[1].strip() else 1.0e30
                epsr1 = _safe_float(c2[2], 1.0e30) if len(c2) > 2 and c2[2].strip() else 1.0e30
                epsr2 = _safe_float(c2[3], 2.0e30) if len(c2) > 3 and c2[3].strip() else 2.0e30
                table_id = _safe_int(c2[4]) if len(c2) > 4 else 0
            if len(valid_cards) > 3:
                c3 = valid_cards[3].cut([20, 20, 20, 20, 10])
                fscale = _safe_float(c3[0], 1.0) if len(c3) > 0 and c3[0].strip() else 1.0
                pscale = _safe_float(c3[1], 1.0) if len(c3) > 1 and c3[1].strip() else 1.0
                t0 = _safe_float(c3[2], 293.0) if len(c3) > 2 and c3[2].strip() else 293.0
                rhocp = _safe_float(c3[3]) if len(c3) > 3 else 0.0
                iyield = _safe_int(c3[4]) if len(c3) > 4 else 0
            if len(valid_cards) > 4:
                c4 = valid_cards[4].cut([10, 20, 20])
                ifunce = _safe_int(c4[0]) if len(c4) > 0 else 0
                einf = _safe_float(c4[1]) if len(c4) > 1 else 0.0
                ce = _safe_float(c4[2]) if len(c4) > 2 else 0.0
        else:
            if len(valid_cards) > 0:
                tok0 = valid_cards[0].tokens()
                rho0 = _safe_float(tok0[0]) if len(tok0) > 0 else 0.0
                rhor = _safe_float(tok0[1]) if len(tok0) > 1 else 0.0
            if len(valid_cards) > 1:
                t1 = valid_cards[1].tokens()
                e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
                nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
                r00 = _safe_float(t1[2], 1.0) if len(t1) > 2 else 1.0
                r45 = _safe_float(t1[3], 1.0) if len(t1) > 3 else 1.0
                r90 = _safe_float(t1[4], 1.0) if len(t1) > 4 else 1.0
            if len(valid_cards) > 2:
                t2 = valid_cards[2].tokens()
                chard = _safe_float(t2[0]) if len(t2) > 0 else 0.0
                eps_max = _safe_float(t2[1], 1.0e30) if len(t2) > 1 else 1.0e30
                epsr1 = _safe_float(t2[2], 1.0e30) if len(t2) > 2 else 1.0e30
                epsr2 = _safe_float(t2[3], 2.0e30) if len(t2) > 3 else 2.0e30
                table_id = _safe_int(t2[4]) if len(t2) > 4 else 0
            if len(valid_cards) > 3:
                t3 = valid_cards[3].tokens()
                fscale = _safe_float(t3[0], 1.0) if len(t3) > 0 else 1.0
                pscale = _safe_float(t3[1], 1.0) if len(t3) > 1 else 1.0
                t0 = _safe_float(t3[2], 293.0) if len(t3) > 2 else 293.0
                rhocp = _safe_float(t3[3]) if len(t3) > 3 else 0.0
                iyield = _safe_int(t3[4]) if len(t3) > 4 else 0
            if len(valid_cards) > 4:
                t4 = valid_cards[4].tokens()
                ifunce = _safe_int(t4[0]) if len(t4) > 0 else 0
                einf = _safe_float(t4[1]) if len(t4) > 1 else 0.0
                ce = _safe_float(t4[2]) if len(t4) > 2 else 0.0
    else:
        # Standard 7-card layout (M561 / CFG matl73_73.cfg / hm_read_mat73.F)
        if is_fixed:
            # Card 1: RHO, [REFER_RHO]
            if len(valid_cards) > 0:
                c0 = valid_cards[0].cut("MAT_LAW73_2")
                rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
                rhor = _safe_float(c0[1]) if len(c0) > 1 and c0[1].strip() else 0.0
            # Card 2: E, NU
            if len(valid_cards) > 1:
                c1 = valid_cards[1].cut("MAT_LAW73_2")
                e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
                nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            # Card 3: IFUNCE (10), blank (10), EINF (20), CE (20)
            if len(valid_cards) > 2:
                c2 = valid_cards[2].cut("MAT_LAW73_3")
                if len(c2) > 0 and c2[0].strip():
                    ifunce = _safe_int(c2[0])
                elif len(c2) > 1 and c2[1].strip():
                    ifunce = _safe_int(c2[1])
                einf = _safe_float(c2[2]) if len(c2) > 2 else 0.0
                ce = _safe_float(c2[3]) if len(c2) > 3 else 0.0
            # Card 4: R00 (20), R45 (20), R90 (20), CHARD (20), [blank (10)], IYIELD (10)
            if len(valid_cards) > 3:
                c3 = valid_cards[3].cut("MAT_LAW73_4")
                r00 = _safe_float(c3[0], 1.0) if len(c3) > 0 and c3[0].strip() else 1.0
                r45 = _safe_float(c3[1], 1.0) if len(c3) > 1 and c3[1].strip() else 1.0
                r90 = _safe_float(c3[2], 1.0) if len(c3) > 2 and c3[2].strip() else 1.0
                chard = _safe_float(c3[3]) if len(c3) > 3 else 0.0
                if len(c3) > 4 and c3[4].strip():
                    iyield = _safe_int(c3[4])
                elif len(c3) > 5 and c3[5].strip():
                    iyield = _safe_int(c3[5])
            # Card 5: EPS_MAX (20), EPSR1 (20), EPSR2 (20)
            if len(valid_cards) > 4:
                c4 = valid_cards[4].cut("MAT_LAW73_5")
                eps_max = _safe_float(c4[0], 1.0e30) if len(c4) > 0 and c4[0].strip() else 1.0e30
                epsr1 = _safe_float(c4[1], 1.0e30) if len(c4) > 1 and c4[1].strip() else 1.0e30
                epsr2 = _safe_float(c4[2], 2.0e30) if len(c4) > 2 and c4[2].strip() else 2.0e30
            # Card 6: [blank (10)], TABLE_ID (10), FSCALE (20), PSCALE (20)
            if len(valid_cards) > 5:
                c5 = valid_cards[5].cut("MAT_LAW73_6")
                if len(c5) > 0 and c5[0].strip():
                    table_id = _safe_int(c5[0])
                elif len(c5) > 1 and c5[1].strip():
                    table_id = _safe_int(c5[1])
                fscale = _safe_float(c5[2], 1.0) if len(c5) > 2 and c5[2].strip() else 1.0
                pscale = _safe_float(c5[3], 1.0) if len(c5) > 3 and c5[3].strip() else 1.0
            # Card 7: T0 (20), RHOCP (20)
            if len(valid_cards) > 6:
                c6 = valid_cards[6].cut("MAT_LAW73_7")
                t0 = _safe_float(c6[0], 293.0) if len(c6) > 0 and c6[0].strip() else 293.0
                rhocp = _safe_float(c6[1]) if len(c6) > 1 else 0.0
        else:
            # Free format 7 cards
            if len(valid_cards) > 0:
                tok0 = valid_cards[0].tokens()
                rho0 = _safe_float(tok0[0]) if len(tok0) > 0 else 0.0
                rhor = _safe_float(tok0[1]) if len(tok0) > 1 else 0.0
            if len(valid_cards) > 1:
                t1 = valid_cards[1].tokens()
                e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
                nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            if len(valid_cards) > 2:
                t2 = valid_cards[2].tokens()
                if len(t2) == 2:
                    einf = _safe_float(t2[0])
                    ce = _safe_float(t2[1])
                elif len(t2) == 3:
                    ifunce = _safe_int(t2[0])
                    einf = _safe_float(t2[1])
                    ce = _safe_float(t2[2])
                elif len(t2) >= 4:
                    ifunce = _safe_int(t2[0]) if _safe_int(t2[0]) != 0 else _safe_int(t2[1])
                    einf = _safe_float(t2[2])
                    ce = _safe_float(t2[3])
            if len(valid_cards) > 3:
                t3 = valid_cards[3].tokens()
                r00 = _safe_float(t3[0], 1.0) if len(t3) > 0 else 1.0
                r45 = _safe_float(t3[1], 1.0) if len(t3) > 1 else 1.0
                r90 = _safe_float(t3[2], 1.0) if len(t3) > 2 else 1.0
                chard = _safe_float(t3[3]) if len(t3) > 3 else 0.0
                if len(t3) > 4 and t3[4].strip():
                    iyield = _safe_int(t3[4])
                if len(t3) > 5 and t3[5].strip():
                    iyield = _safe_int(t3[5])
            if len(valid_cards) > 4:
                t4 = valid_cards[4].tokens()
                eps_max = _safe_float(t4[0], 1.0e30) if len(t4) > 0 else 1.0e30
                epsr1 = _safe_float(t4[1], 1.0e30) if len(t4) > 1 else 1.0e30
                epsr2 = _safe_float(t4[2], 2.0e30) if len(t4) > 2 else 2.0e30
            if len(valid_cards) > 5:
                t5 = valid_cards[5].tokens()
                if len(t5) == 1:
                    table_id = _safe_int(t5[0])
                elif len(t5) == 2:
                    table_id = _safe_int(t5[0])
                    fscale = _safe_float(t5[1], 1.0)
                elif len(t5) == 3:
                    table_id = _safe_int(t5[0])
                    fscale = _safe_float(t5[1], 1.0)
                    pscale = _safe_float(t5[2], 1.0)
                elif len(t5) >= 4:
                    table_id = _safe_int(t5[0]) if _safe_int(t5[0]) != 0 else _safe_int(t5[1])
                    fscale = _safe_float(t5[2], 1.0)
                    pscale = _safe_float(t5[3], 1.0)
            if len(valid_cards) > 6:
                t6 = valid_cards[6].tokens()
                t0 = _safe_float(t6[0], 293.0) if len(t6) > 0 else 293.0
                rhocp = _safe_float(t6[1]) if len(t6) > 1 else 0.0

    # Upstream defaults from hm_read_mat73.F
    if r00 == 0.0:
        r00 = 1.0
    if r45 == 0.0:
        r45 = 1.0
    if r90 == 0.0:
        r90 = 1.0
    if eps_max == 0.0:
        eps_max = 1.0e30
    if epsr1 == 0.0:
        epsr1 = 1.0e30
    if epsr2 == 0.0:
        epsr2 = 2.0e30
    if fscale == 0.0:
        fscale = 1.0
    if pscale == 0.0:
        pscale = 1.0
    if t0 == 0.0:
        t0 = 293.0
    if rhor == 0.0:
        rhor = rho0

    mat = MatLaw73(
        id=mat_id, rho=rho0, e=e, nu=nu, ifunce=ifunce, einf=einf, ce=ce,
        r00=r00, r45=r45, r90=r90, chard=chard, iyield=iyield,
        eps_max=eps_max, epsr1=epsr1, epsr2=epsr2, table_id=table_id,
        fscale=fscale, pscale=pscale, t0=t0, rhocp=rhocp,
        title=title, refer_rho=rhor
    )
    model.mat_law73s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=73, rho0=rho0, title=title,
        params=mat.params
    )




def read_mat_law84(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW84`` or ``/MAT/SWIFT_VOCE`` (M191): Swift-Voce hardening plastic material."""
    from ...model.entities import MatLaw84, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    rho0, rhor = 0.0, 0.0
    e, nu, fcut, cap_end, pc = 0.0, 0.0, 0.0, 0.0, 0.0
    pr, t0, c2_t, a2, c1_c = 0.0, 0.0, 0.0, 0.0, 0.0
    vol, nut, fscale11, fscale22, fscale33 = 0.0, 0.0, 1.0, 1.0, 1.0
    fscale12, fscale23, scale1, scale2, scale3 = 1.0, 1.0, 0.0, 0.0, 0.0
    scale4, scale5 = 0.0, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW84_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW84_2")
            e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            fcut = _safe_float(c1[2]) if len(c1) > 2 else 0.0
            cap_end = _safe_float(c1[3]) if len(c1) > 3 else 0.0
            pc = _safe_float(c1[4]) if len(c1) > 4 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW84_3")
            pr = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            t0 = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            c2_t = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            a2 = _safe_float(c2[3]) if len(c2) > 3 else 0.0
            c1_c = _safe_float(c2[4]) if len(c2) > 4 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW84_4")
            vol = _safe_float(c3[0]) if len(c3) > 0 else 0.0
            nut = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            fscale11 = _safe_float(c3[2], 1.0) if len(c3) > 2 else 1.0
            fscale22 = _safe_float(c3[3], 1.0) if len(c3) > 3 else 1.0
            fscale33 = _safe_float(c3[4], 1.0) if len(c3) > 4 else 1.0
        if len(valid_cards) > 4:
            c4 = valid_cards[4].cut("MAT_LAW84_5")
            fscale12 = _safe_float(c4[0], 1.0) if len(c4) > 0 else 1.0
            fscale23 = _safe_float(c4[1], 1.0) if len(c4) > 1 else 1.0
            scale1 = _safe_float(c4[2]) if len(c4) > 2 else 0.0
            scale2 = _safe_float(c4[3]) if len(c4) > 3 else 0.0
            scale3 = _safe_float(c4[4]) if len(c4) > 4 else 0.0
        if len(valid_cards) > 5:
            c5 = valid_cards[5].cut("MAT_LAW84_6")
            scale4 = _safe_float(c5[0]) if len(c5) > 0 else 0.0
            scale5 = _safe_float(c5[1]) if len(c5) > 1 else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            fcut = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            cap_end = _safe_float(t1[3]) if len(t1) > 3 else 0.0
            pc = _safe_float(t1[4]) if len(t1) > 4 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            pr = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            t0 = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            c2_t = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            a2 = _safe_float(t2[3]) if len(t2) > 3 else 0.0
            c1_c = _safe_float(t2[4]) if len(t2) > 4 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            vol = _safe_float(t3[0]) if len(t3) > 0 else 0.0
            nut = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            fscale11 = _safe_float(t3[2], 1.0) if len(t3) > 2 else 1.0
            fscale22 = _safe_float(t3[3], 1.0) if len(t3) > 3 else 1.0
            fscale33 = _safe_float(t3[4], 1.0) if len(t3) > 4 else 1.0
        if len(valid_cards) > 4:
            t4 = valid_cards[4].tokens()
            fscale12 = _safe_float(t4[0], 1.0) if len(t4) > 0 else 1.0
            fscale23 = _safe_float(t4[1], 1.0) if len(t4) > 1 else 1.0
            scale1 = _safe_float(t4[2]) if len(t4) > 2 else 0.0
            scale2 = _safe_float(t4[3]) if len(t4) > 3 else 0.0
            scale3 = _safe_float(t4[4]) if len(t4) > 4 else 0.0
        if len(valid_cards) > 5:
            t5 = valid_cards[5].tokens()
            scale4 = _safe_float(t5[0]) if len(t5) > 0 else 0.0
            scale5 = _safe_float(t5[1]) if len(t5) > 1 else 0.0

    mat = MatLaw84(
        id=mat_id, rho0=rho0, rhor=rhor, e=e, nu=nu, fcut=fcut, cap_end=cap_end, pc=pc,
        pr=pr, t0=t0, c2_t=c2_t, a2=a2, c1_c=c1_c, vol=vol, nut=nut,
        fscale11=fscale11, fscale22=fscale22, fscale33=fscale33, fscale12=fscale12, fscale23=fscale23,
        scale1=scale1, scale2=scale2, scale3=scale3, scale4=scale4, scale5=scale5, title=title
    )
    model.mat_law84s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=84, rho0=rho0, title=title,
        params={
            "rho": rho0, "rhor": rhor, "e": e, "nu": nu, "fcut": fcut, "cap_end": cap_end, "pc": pc,
            "pr": pr, "t0": t0, "c2_t": c2_t, "a2": a2, "c1_c": c1_c, "vol": vol, "nut": nut,
            "fscale11": fscale11, "fscale22": fscale22, "fscale33": fscale33, "fscale12": fscale12, "fscale23": fscale23,
            "scale1": scale1, "scale2": scale2, "scale3": scale3, "scale4": scale4, "scale5": scale5,
        }
    )




def read_mat_law93(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW93`` or ``/MAT/ORTH_HILL`` (M191/M568): Orthotropic Hill 1948 plasticity model."""
    from ...model.entities import MatLaw93, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    rho0, rhor = 0.0, 0.0
    e11, e22, e33, g12, nu12 = 0.0, 0.0, 0.0, 0.0, 0.0
    g13, g23, nu13, nu23, nl = 0.0, 0.0, 0.0, 0.0, 0
    sigma_y, qr1, cr1, qr2, cr2 = 0.0, 0.0, 0.0, 0.0, 0.0
    r11, r22, r12, r33, r13, r23 = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    fcut = 0.0
    vp = 0
    curves = []

    # Format detection: standard 8-card format vs legacy 6-card format
    # In standard format (matl93_ORTH_HILL.cfg):
    # Card 0: rho0, rhor
    # Card 1: e11, e22, e33, g12, nu12
    # Card 2: g13, g23, nu13, nu23
    # Card 3: nl, vp, fcut
    # If nl > 0: next nl cards are curves (fct_id, fscale, eps_dot)
    # Next card: sigma_y, qr1, cr1, qr2, cr2
    # Next card: r11, r22, r12
    # Next card: r33, r13, r23

    # Card 0
    if len(valid_cards) > 0:
        c0 = valid_cards[0].cut("MAT_LAW93_1") if block.fixed else valid_cards[0].tokens()
        rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
        rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0

    # Card 1
    if len(valid_cards) > 1:
        c1 = valid_cards[1].cut("MAT_LAW93_2") if block.fixed else valid_cards[1].tokens()
        e11 = _safe_float(c1[0]) if len(c1) > 0 else 0.0
        e22 = _safe_float(c1[1]) if len(c1) > 1 else 0.0
        e33 = _safe_float(c1[2]) if len(c1) > 2 else 0.0
        g12 = _safe_float(c1[3]) if len(c1) > 3 else 0.0
        nu12 = _safe_float(c1[4]) if len(c1) > 4 else 0.0

    # Card 2 & check format
    is_legacy = False
    if len(valid_cards) > 2:
        c2_toks = valid_cards[2].tokens()
        if len(c2_toks) >= 5 and _safe_int(c2_toks[4]) > 0 and len(valid_cards) > 3 and len(valid_cards[3].tokens()) >= 4:
            is_legacy = True

    if is_legacy:
        # Legacy 6-card format
        if block.fixed:
            c2 = valid_cards[2].cut("MAT_LAW93_3")
            g13 = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            g23 = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            nu13 = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            nu23 = _safe_float(c2[3]) if len(c2) > 3 else 0.0
            nl = _safe_int(c2[4]) if len(c2) > 4 else 0
            if len(valid_cards) > 3:
                c3 = valid_cards[3].cut("MAT_LAW93_4")
                sigma_y = _safe_float(c3[0]) if len(c3) > 0 else 0.0
                qr1 = _safe_float(c3[1]) if len(c3) > 1 else 0.0
                cr1 = _safe_float(c3[2]) if len(c3) > 2 else 0.0
                qr2 = _safe_float(c3[3]) if len(c3) > 3 else 0.0
                cr2 = _safe_float(c3[4]) if len(c3) > 4 else 0.0
            if len(valid_cards) > 4:
                c4 = valid_cards[4].cut("MAT_LAW93_5")
                r11 = _safe_float(c4[0], 1.0) if len(c4) > 0 else 1.0
                r22 = _safe_float(c4[1], 1.0) if len(c4) > 1 else 1.0
                r12 = _safe_float(c4[2], 1.0) if len(c4) > 2 else 1.0
                r33 = _safe_float(c4[3], 1.0) if len(c4) > 3 else 1.0
                r13 = _safe_float(c4[4], 1.0) if len(c4) > 4 else 1.0
            if len(valid_cards) > 5:
                c5 = valid_cards[5].cut("MAT_LAW93_6")
                r23 = _safe_float(c5[0], 1.0) if len(c5) > 0 else 1.0
                fcut = _safe_float(c5[1]) if len(c5) > 1 else 0.0
                vp = _safe_int(c5[2]) if len(c5) > 2 else 0
            for i in range(nl):
                c_idx = 6 + i
                if c_idx < len(valid_cards):
                    cc = valid_cards[c_idx].cut("MAT_LAW93_CURVE")
                    fct_id = _safe_int(cc[0]) if len(cc) > 0 else 0
                    fscale = _safe_float(cc[1], 1.0) if len(cc) > 1 else 1.0
                    eps_dot = _safe_float(cc[2]) if len(cc) > 2 else 0.0
                    curves.append({"fct_id": fct_id, "fscale": fscale, "eps_dot": eps_dot})
        else:
            t2 = valid_cards[2].tokens()
            g13 = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            g23 = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            nu13 = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            nu23 = _safe_float(t2[3]) if len(t2) > 3 else 0.0
            nl = _safe_int(t2[4]) if len(t2) > 4 else 0
            if len(valid_cards) > 3:
                t3 = valid_cards[3].tokens()
                sigma_y = _safe_float(t3[0]) if len(t3) > 0 else 0.0
                qr1 = _safe_float(t3[1]) if len(t3) > 1 else 0.0
                cr1 = _safe_float(t3[2]) if len(t3) > 2 else 0.0
                qr2 = _safe_float(t3[3]) if len(t3) > 3 else 0.0
                cr2 = _safe_float(t3[4]) if len(t3) > 4 else 0.0
            if len(valid_cards) > 4:
                t4 = valid_cards[4].tokens()
                r11 = _safe_float(t4[0], 1.0) if len(t4) > 0 else 1.0
                r22 = _safe_float(t4[1], 1.0) if len(t4) > 1 else 1.0
                r12 = _safe_float(t4[2], 1.0) if len(t4) > 2 else 1.0
                r33 = _safe_float(t4[3], 1.0) if len(t4) > 3 else 1.0
                r13 = _safe_float(t4[4], 1.0) if len(t4) > 4 else 1.0
            if len(valid_cards) > 5:
                t5 = valid_cards[5].tokens()
                r23 = _safe_float(t5[0], 1.0) if len(t5) > 0 else 1.0
                fcut = _safe_float(t5[1]) if len(t5) > 1 else 0.0
                vp = _safe_int(t5[2]) if len(t5) > 2 else 0
            for i in range(nl):
                c_idx = 6 + i
                if c_idx < len(valid_cards):
                    tt = valid_cards[c_idx].tokens()
                    fct_id = _safe_int(tt[0]) if len(tt) > 0 else 0
                    fscale = _safe_float(tt[1], 1.0) if len(tt) > 1 else 1.0
                    eps_dot = _safe_float(tt[2]) if len(tt) > 2 else 0.0
                    curves.append({"fct_id": fct_id, "fscale": fscale, "eps_dot": eps_dot})
    else:
        # Standard 8-card format (matl93_ORTH_HILL.cfg)
        # Card 2: G13, G23, Nu13, Nu23 (%20lg%20lg%20lg%20lg)
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW93_3_STD") if block.fixed else valid_cards[2].tokens()
            g13 = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            g23 = _safe_float(c2[1]) if len(c2) > 1 else 0.0
            nu13 = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            nu23 = _safe_float(c2[3]) if len(c2) > 3 else 0.0

        # Card 3: NL, VP, FCUT (%10d%10d%20lg)
        cur_idx = 3
        if len(valid_cards) > cur_idx:
            c3 = valid_cards[cur_idx].cut("MAT_LAW93_4_STD") if block.fixed else valid_cards[cur_idx].tokens()
            nl = _safe_int(c3[0]) if len(c3) > 0 else 0
            vp = _safe_int(c3[1]) if len(c3) > 1 else 0
            fcut = _safe_float(c3[2]) if len(c3) > 2 else 0.0
            cur_idx += 1

        # Curve cards (Card 4 .. 4+nl-1)
        for _ in range(nl):
            if cur_idx < len(valid_cards):
                if block.fixed:
                    cc = valid_cards[cur_idx].cut("MAT_LAW93_CURVE_STD")
                    fct_id = _safe_int(cc[0]) if len(cc) > 0 else 0
                    fscale = _safe_float(cc[2], 1.0) if len(cc) > 2 else 1.0
                    eps_dot = _safe_float(cc[3], 0.0) if len(cc) > 3 else 0.0
                else:
                    tt = valid_cards[cur_idx].tokens()
                    fct_id = _safe_int(tt[0]) if len(tt) > 0 else 0
                    fscale = _safe_float(tt[1], 1.0) if len(tt) > 1 else 1.0
                    eps_dot = _safe_float(tt[2], 0.0) if len(tt) > 2 else 0.0
                curves.append({"fct_id": fct_id, "fscale": fscale, "eps_dot": eps_dot})
                cur_idx += 1

        # Continuous hardening Card 5: Sigma_y, QR1, CR1, QR2, CR2 (%20lg%20lg%20lg%20lg%20lg)
        if cur_idx < len(valid_cards):
            c_voce = valid_cards[cur_idx].cut("MAT_LAW93_6_STD") if block.fixed else valid_cards[cur_idx].tokens()
            sigma_y = _safe_float(c_voce[0]) if len(c_voce) > 0 else 0.0
            qr1 = _safe_float(c_voce[1]) if len(c_voce) > 1 else 0.0
            cr1 = _safe_float(c_voce[2]) if len(c_voce) > 2 else 0.0
            qr2 = _safe_float(c_voce[3]) if len(c_voce) > 3 else 0.0
            cr2 = _safe_float(c_voce[4]) if len(c_voce) > 4 else 0.0
            cur_idx += 1

        # Hill Card 6: R11, R22, R12 (%20lg%20lg%20lg)
        if cur_idx < len(valid_cards):
            c_hill1 = valid_cards[cur_idx].cut("MAT_LAW93_7_STD") if block.fixed else valid_cards[cur_idx].tokens()
            r11 = _safe_float(c_hill1[0], 1.0) if len(c_hill1) > 0 else 1.0
            r22 = _safe_float(c_hill1[1], 1.0) if len(c_hill1) > 1 else 1.0
            r12 = _safe_float(c_hill1[2], 1.0) if len(c_hill1) > 2 else 1.0
            cur_idx += 1

        # Hill Card 7: R33, R13, R23 (%20lg%20lg%20lg)
        if cur_idx < len(valid_cards):
            c_hill2 = valid_cards[cur_idx].cut("MAT_LAW93_8_STD") if block.fixed else valid_cards[cur_idx].tokens()
            r33 = _safe_float(c_hill2[0], 1.0) if len(c_hill2) > 0 else 1.0
            r13 = _safe_float(c_hill2[1], 1.0) if len(c_hill2) > 1 else 1.0
            r23 = _safe_float(c_hill2[2], 1.0) if len(c_hill2) > 2 else 1.0
            cur_idx += 1

    # Apply OpenRadioss defaults
    if r11 == 0.0: r11 = 1.0
    if r22 == 0.0: r22 = 1.0
    if r33 == 0.0: r33 = 1.0
    if r12 == 0.0: r12 = 1.0
    if r13 == 0.0: r13 = 1.0
    if r23 == 0.0: r23 = 1.0

    mat = MatLaw93(
        id=mat_id, rho0=rho0, rhor=rhor, e11=e11, e22=e22, e33=e33, g12=g12, nu12=nu12,
        g13=g13, g23=g23, nu13=nu13, nu23=nu23, nl=nl, sigma_y=sigma_y,
        qr1=qr1, cr1=cr1, qr2=qr2, cr2=cr2, r11=r11, r22=r22, r12=r12, r33=r33, r13=r13,
        r23=r23, fcut=fcut, vp=vp, curves=curves, title=title
    )
    e_eff = max(e11, e22, e33)
    nu_eff = max(nu12, nu13, nu23)
    g_eff = g12 if g12 > 0.0 else e_eff / (2.0 * (1.0 + nu_eff))
    k_eff = e_eff / max(3.0 * (1.0 - 2.0 * nu_eff), 1.0e-12)
    mat_params = {
        "rho": rho0, "rho0": rho0, "rhor": rhor, "e11": e11, "e22": e22, "e33": e33, "g12": g12, "nu12": nu12,
        "g13": g13, "g23": g23, "nu13": nu13, "nu23": nu23, "nl": nl, "sigma_y": sigma_y,
        "qr1": qr1, "cr1": cr1, "qr2": qr2, "cr2": cr2, "r11": r11, "r22": r22, "r12": r12, "r33": r33, "r13": r13,
        "r23": r23, "fcut": fcut, "vp": vp, "curves": curves,
        "E": e_eff, "e": e_eff, "young": e_eff, "nu": nu_eff, "poisson": nu_eff,
        "G": g_eff, "g": g_eff, "K": k_eff, "bulk": k_eff,
    }
    mat.params = mat_params
    model.mat_law93s[mat_id] = mat
    if hasattr(model, "mat_orth_hills"):
        model.mat_orth_hills[mat_id] = mat
    gen_mat = Material(
        id=mat_id, law=93, rho0=rho0, title=title, params=mat_params
    )
    for k, v in mat_params.items():
        if not hasattr(Material, k):
            setattr(gen_mat, k, v)
    model.materials[mat_id] = gen_mat




def read_mat_law133(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW133`` or ``/MAT/GRANULAR`` (M191): Granular material model."""
    from ...model.entities import MatLaw133, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    rho0, rhor = 0.0, 0.0
    nu, pmin = 0.0, 0.0
    fct_id_g, fscale_g = 0, 1.0
    fct_id_y, fscale_y = 0, 1.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW133_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW133_2")
            nu = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            pmin = _safe_float(c1[1]) if len(c1) > 1 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW133_3")
            fct_id_g = _safe_int(c2[0]) if len(c2) > 0 else 0
            fscale_g = _safe_float(c2[2], 1.0) if len(c2) > 2 else 1.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW133_4")
            fct_id_y = _safe_int(c3[0]) if len(c3) > 0 else 0
            fscale_y = _safe_float(c3[2], 1.0) if len(c3) > 2 else 1.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            nu = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            pmin = _safe_float(t1[1]) if len(t1) > 1 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            fct_id_g = _safe_int(t2[0]) if len(t2) > 0 else 0
            fscale_g = _safe_float(t2[2] if len(t2) > 2 else t2[1], 1.0) if len(t2) > 1 else 1.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            fct_id_y = _safe_int(t3[0]) if len(t3) > 0 else 0
            fscale_y = _safe_float(t3[2] if len(t3) > 2 else t3[1], 1.0) if len(t3) > 1 else 1.0

    mat = MatLaw133(
        id=mat_id, rho0=rho0, rhor=rhor, nu=nu, pmin=pmin,
        fct_id_g=fct_id_g, fscale_g=fscale_g, fct_id_y=fct_id_y, fscale_y=fscale_y, title=title
    )
    model.mat_law133s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=133, rho0=rho0, title=title,
        params={
            "rho": rho0, "rhor": rhor, "nu": nu, "pmin": pmin,
            "fct_id_g": fct_id_g, "fscale_g": fscale_g, "fct_id_y": fct_id_y, "fscale_y": fscale_y,
        }
    )




def read_mat_law101(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW101`` or ``/MAT/PP`` / ``/MAT/PLAS_POLY`` (M571): Bouvard polymer viscoplasticity model.

    Citing starter/source/materials/mat/mat101/hm_read_mat101.F and radioss2021/MAT/mat_l101.cfg:
      Card 1: RHO_I (%20lg or %20lg%20lg)
      Card 2: EREF, E1, Nu, VE1 (%20lg%20lg%20lg%20lg)
      Card 3: VE2, EDOT_REF, GAMA_DOT_REF, ALPHAP (%20lg%20lg%20lg%20lg)
      Card 4: delta_H, V, m, C3 (%20lg%20lg%20lg%20lg)
      Card 5: C4, ALPHAK1, ALPHAK2, H0 (%20lg%20lg%20lg%20lg)
      Card 6: ZETA1_i, C5, C6, C7 (%20lg%20lg%20lg%20lg)
      Card 7: C8, C9, C10, h1 (%20lg%20lg%20lg%20lg)
      Card 8: ZETA2_i, C11, C12, C13 (%20lg%20lg%20lg%20lg)
      Card 9: C14, C1, C2, LAMBDA_L (%20lg%20lg%20lg%20lg)
      Card 10: RHO_theta_0, CV_theta_0, THETA0, ALPHA_TH (%20lg%20lg%20lg%20lg)
      Card 11: THETA_GLASS, TEMP_FACTOR, THETA_FLAG, THETAi (%20lg%20lg%20lg%20lg)
    """
    from ...model.entities import MatLaw101, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]

    rho0, rhor = 0.0, 0.0
    e, alpha1, nu, ve1 = 0.0, 0.0, 0.0, 0.0
    ve2, epsilonref, gamma0, alpha_p = 0.0, 0.0, 0.0, 0.0
    deltah, vol, m, c3 = 0.0, 0.0, 1.0, 0.0
    c4, alphak1, alphak2, hard = 0.0, 0.0, 0.0, 0.0
    zeta1i, c5, c6, c7 = 0.0, 0.0, 0.0, 0.0
    c8, c9, c10, hard1 = 0.0, 0.0, 0.0, 0.0
    zeta2i, c11, c12, c13 = 0.0, 0.0, 0.0, 0.0
    c14, c1, c2, lambdal = 0.0, 0.0, 0.0, 1.0
    rho_ref, cv_ref, tref, alpha_th = 0.0, 0.0, 293.15, 0.0
    theta_glass, omega, theta_flag, heat_t0 = 250.0, 0.0, 0.0, 293.15

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW101_1")
            rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            rhor = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1_cut = valid_cards[1].cut("MAT_LAW101_2")
            e = _safe_float(c1_cut[0]) if len(c1_cut) > 0 else 0.0
            alpha1 = _safe_float(c1_cut[1]) if len(c1_cut) > 1 else 0.0
            nu = _safe_float(c1_cut[2]) if len(c1_cut) > 2 else 0.0
            ve1 = _safe_float(c1_cut[3]) if len(c1_cut) > 3 else 0.0
        if len(valid_cards) > 2:
            c2_cut = valid_cards[2].cut("MAT_LAW101_3")
            ve2 = _safe_float(c2_cut[0]) if len(c2_cut) > 0 else 0.0
            epsilonref = _safe_float(c2_cut[1]) if len(c2_cut) > 1 else 0.0
            gamma0 = _safe_float(c2_cut[2]) if len(c2_cut) > 2 else 0.0
            alpha_p = _safe_float(c2_cut[3]) if len(c2_cut) > 3 else 0.0
        if len(valid_cards) > 3:
            c3_cut = valid_cards[3].cut("MAT_LAW101_4")
            deltah = _safe_float(c3_cut[0]) if len(c3_cut) > 0 else 0.0
            vol = _safe_float(c3_cut[1]) if len(c3_cut) > 1 else 0.0
            m = _safe_float(c3_cut[2]) if len(c3_cut) > 2 else 1.0
            c3 = _safe_float(c3_cut[3]) if len(c3_cut) > 3 else 0.0
        if len(valid_cards) > 4:
            c4_cut = valid_cards[4].cut("MAT_LAW101_5")
            c4 = _safe_float(c4_cut[0]) if len(c4_cut) > 0 else 0.0
            alphak1 = _safe_float(c4_cut[1]) if len(c4_cut) > 1 else 0.0
            alphak2 = _safe_float(c4_cut[2]) if len(c4_cut) > 2 else 0.0
            hard = _safe_float(c4_cut[3]) if len(c4_cut) > 3 else 0.0
        if len(valid_cards) > 5:
            c5_cut = valid_cards[5].cut("MAT_LAW101_6")
            zeta1i = _safe_float(c5_cut[0]) if len(c5_cut) > 0 else 0.0
            c5 = _safe_float(c5_cut[1]) if len(c5_cut) > 1 else 0.0
            c6 = _safe_float(c5_cut[2]) if len(c5_cut) > 2 else 0.0
            c7 = _safe_float(c5_cut[3]) if len(c5_cut) > 3 else 0.0
        if len(valid_cards) > 6:
            c6_cut = valid_cards[6].cut("MAT_LAW101_7")
            c8 = _safe_float(c6_cut[0]) if len(c6_cut) > 0 else 0.0
            c9 = _safe_float(c6_cut[1]) if len(c6_cut) > 1 else 0.0
            c10 = _safe_float(c6_cut[2]) if len(c6_cut) > 2 else 0.0
            hard1 = _safe_float(c6_cut[3]) if len(c6_cut) > 3 else 0.0
        if len(valid_cards) > 7:
            c7_cut = valid_cards[7].cut("MAT_LAW101_8")
            zeta2i = _safe_float(c7_cut[0]) if len(c7_cut) > 0 else 0.0
            c11 = _safe_float(c7_cut[1]) if len(c7_cut) > 1 else 0.0
            c12 = _safe_float(c7_cut[2]) if len(c7_cut) > 2 else 0.0
            c13 = _safe_float(c7_cut[3]) if len(c7_cut) > 3 else 0.0
        if len(valid_cards) > 8:
            c8_cut = valid_cards[8].cut("MAT_LAW101_9")
            c14 = _safe_float(c8_cut[0]) if len(c8_cut) > 0 else 0.0
            c1 = _safe_float(c8_cut[1]) if len(c8_cut) > 1 else 0.0
            c2 = _safe_float(c8_cut[2]) if len(c8_cut) > 2 else 0.0
            lambdal = _safe_float(c8_cut[3]) if len(c8_cut) > 3 else 1.0
        if len(valid_cards) > 9:
            c9_cut = valid_cards[9].cut("MAT_LAW101_10")
            rho_ref = _safe_float(c9_cut[0]) if len(c9_cut) > 0 else 0.0
            cv_ref = _safe_float(c9_cut[1]) if len(c9_cut) > 1 else 0.0
            tref = _safe_float(c9_cut[2]) if len(c9_cut) > 2 else 293.15
            alpha_th = _safe_float(c9_cut[3]) if len(c9_cut) > 3 else 0.0
        if len(valid_cards) > 10:
            c10_cut = valid_cards[10].cut("MAT_LAW101_11")
            theta_glass = _safe_float(c10_cut[0]) if len(c10_cut) > 0 else 250.0
            omega = _safe_float(c10_cut[1]) if len(c10_cut) > 1 else 0.0
            theta_flag = _safe_float(c10_cut[2]) if len(c10_cut) > 2 else 0.0
            heat_t0 = _safe_float(c10_cut[3]) if len(c10_cut) > 3 else 293.15
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            alpha1 = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            nu = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            ve1 = _safe_float(t1[3]) if len(t1) > 3 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            ve2 = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            epsilonref = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            gamma0 = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            alpha_p = _safe_float(t2[3]) if len(t2) > 3 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            deltah = _safe_float(t3[0]) if len(t3) > 0 else 0.0
            vol = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            m = _safe_float(t3[2]) if len(t3) > 2 else 1.0
            c3 = _safe_float(t3[3]) if len(t3) > 3 else 0.0
        if len(valid_cards) > 4:
            t4 = valid_cards[4].tokens()
            c4 = _safe_float(t4[0]) if len(t4) > 0 else 0.0
            alphak1 = _safe_float(t4[1]) if len(t4) > 1 else 0.0
            alphak2 = _safe_float(t4[2]) if len(t4) > 2 else 0.0
            hard = _safe_float(t4[3]) if len(t4) > 3 else 0.0
        if len(valid_cards) > 5:
            t5 = valid_cards[5].tokens()
            zeta1i = _safe_float(t5[0]) if len(t5) > 0 else 0.0
            c5 = _safe_float(t5[1]) if len(t5) > 1 else 0.0
            c6 = _safe_float(t5[2]) if len(t5) > 2 else 0.0
            c7 = _safe_float(t5[3]) if len(t5) > 3 else 0.0
        if len(valid_cards) > 6:
            t6 = valid_cards[6].tokens()
            c8 = _safe_float(t6[0]) if len(t6) > 0 else 0.0
            c9 = _safe_float(t6[1]) if len(t6) > 1 else 0.0
            c10 = _safe_float(t6[2]) if len(t6) > 2 else 0.0
            hard1 = _safe_float(t6[3]) if len(t6) > 3 else 0.0
        if len(valid_cards) > 7:
            t7 = valid_cards[7].tokens()
            zeta2i = _safe_float(t7[0]) if len(t7) > 0 else 0.0
            c11 = _safe_float(t7[1]) if len(t7) > 1 else 0.0
            c12 = _safe_float(t7[2]) if len(t7) > 2 else 0.0
            c13 = _safe_float(t7[3]) if len(t7) > 3 else 0.0
        if len(valid_cards) > 8:
            t8 = valid_cards[8].tokens()
            c14 = _safe_float(t8[0]) if len(t8) > 0 else 0.0
            c1 = _safe_float(t8[1]) if len(t8) > 1 else 0.0
            c2 = _safe_float(t8[2]) if len(t8) > 2 else 0.0
            lambdal = _safe_float(t8[3]) if len(t8) > 3 else 1.0
        if len(valid_cards) > 9:
            t9 = valid_cards[9].tokens()
            rho_ref = _safe_float(t9[0]) if len(t9) > 0 else 0.0
            cv_ref = _safe_float(t9[1]) if len(t9) > 1 else 0.0
            tref = _safe_float(t9[2]) if len(t9) > 2 else 293.15
            alpha_th = _safe_float(t9[3]) if len(t9) > 3 else 0.0
        if len(valid_cards) > 10:
            t10 = valid_cards[10].tokens()
            theta_glass = _safe_float(t10[0]) if len(t10) > 0 else 250.0
            omega = _safe_float(t10[1]) if len(t10) > 1 else 0.0
            theta_flag = _safe_float(t10[2]) if len(t10) > 2 else 0.0
            heat_t0 = _safe_float(t10[3]) if len(t10) > 3 else 293.15

    mat = MatLaw101(
        id=mat_id, rho0=rho0, rhor=rhor, e=e, alpha1=alpha1, nu=nu, ve1=ve1,
        ve2=ve2, epsilonref=epsilonref, gamma0=gamma0, alpha_p=alpha_p,
        deltah=deltah, vol=vol, m=m, c3=c3, c4=c4, alphak1=alphak1, alphak2=alphak2, hard=hard,
        zeta1i=zeta1i, c5=c5, c6=c6, c7=c7, c8=c8, c9=c9, c10=c10,
        hard1=hard1, zeta2i=zeta2i, c11=c11, c12=c12, c13=c13, c14=c14,
        c1=c1, c2=c2, lambdal=lambdal, rho_ref=rho_ref, cv_ref=cv_ref,
        tref=tref, alpha_th=alpha_th, theta_glass=theta_glass, omega=omega,
        theta_flag=theta_flag, heat_t0=heat_t0, title=title
    )
    model.mat_law101s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=101, rho0=rho0, title=title,
        params={
            "rho": rho0, "rhor": rhor, "e": e, "alpha1": alpha1, "nu": nu, "ve1": ve1,
            "ve2": ve2, "epsilonref": epsilonref, "gamma0": gamma0, "alpha_p": alpha_p,
            "deltah": deltah, "vol": vol, "m": m, "c3": c3, "c4": c4, "alphak1": alphak1, "alphak2": alphak2, "hard": hard,
            "zeta1i": zeta1i, "c5": c5, "c6": c6, "c7": c7, "c8": c8, "c9": c9, "c10": c10,
            "hard1": hard1, "zeta2i": zeta2i, "c11": c11, "c12": c12, "c13": c13, "c14": c14,
            "c1": c1, "c2": c2, "lambdal": lambdal, "rho_ref": rho_ref, "cv_ref": cv_ref,
            "tref": tref, "alpha_th": alpha_th, "theta_glass": theta_glass, "omega": omega,
            "theta_flag": theta_flag, "heat_t0": heat_t0,
        }
    )




def read_mat_law43(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW43`` or ``/MAT/HILL_TAB`` (M548): Tabulated Hill orthotropic material model.

    Citing starter/source/materials/mat/mat043/hm_read_mat43.F and radioss140/MAT/matl43_HILL_TAB.cfg:
      Card 1: RHO, Refer_Rho (%20lg%20lg)
      Card 2: E, nu (%20lg%20lg)
      Card 3: Yr_fun, MAT_EFIB, MAT_C (%10d[10s]%20lg%20lg)
      Card 4: MAT_R00, MAT_R45, MAT_R90, MAT_CHARD, MAT_Iyield (%20lg%20lg%20lg%20lg%10d)
      Card 5: MAT_EPS, MAT_EPST1, MAT_EPST2, Fcut, Fsmooth (%20lg%20lg%20lg%20lg%10d)
      Curve cards (repeated): FunctionIds, [10s], ABG_cpa, ABG_cpb (%10d[10s]%20lg%20lg)
    """
    from ...model.entities import MatLaw43, Material
    from ..card_layouts import split_fixed
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if valid_cards:
        t0_tokens = valid_cards[0].raw.split()
        if len(t0_tokens) >= 6 and len(valid_cards) <= 3:
            read_mat_law32(block, model, log)
            return

    rho0, rhor = 0.0, 0.0
    e, nu = 0.0, 0.0
    ifunce, einf, ce = 0, 0.0, 0.0
    r00, r45, r90, chard, iyield = 1.0, 1.0, 1.0, 0.0, 0
    eps_max, epst1, epst2, fcut, fsmooth = 0.0, 0.0, 0.0, 0.0, 0
    curves = []

    law_name = "LAW43"
    if block.keyword:
        parts = block.keyword.split("/")
        if len(parts) > 2 and parts[2].strip():
            law_name = parts[2].strip()

    if block.fixed:
        is_4card_fixed = False
        if len(valid_cards) > 1 and len(valid_cards[1].raw.rstrip()) > 40 and valid_cards[1].raw[40:].strip():
            is_4card_fixed = True

        if is_4card_fixed:
            # 4-card legacy format
            # Card 1: RHO, Refer_Rho
            if len(valid_cards) > 0:
                c0 = valid_cards[0].cut("MAT_LAW43_1")
                rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
                rhor = _safe_float(c0[1]) if len(c0) > 1 and c0[1].strip() else rho0
            # Card 2: E, nu, ifunce, einf, ce ([20, 20, 10, 20, 20])
            if len(valid_cards) > 1:
                c1 = split_fixed(valid_cards[1].raw, [20, 20, 10, 20, 20])
                e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
                nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
                ifunce = _safe_int(c1[2]) if len(c1) > 2 else 0
                einf = _safe_float(c1[3]) if len(c1) > 3 else 0.0
                ce = _safe_float(c1[4]) if len(c1) > 4 else 0.0
            # Card 3: R00, R45, R90, chard, iyield ([20, 20, 20, 20, 10])
            if len(valid_cards) > 2:
                c2 = split_fixed(valid_cards[2].raw, [20, 20, 20, 20, 10])
                r00 = _safe_float(c2[0], 1.0) if len(c2) > 0 and c2[0].strip() else 1.0
                r45 = _safe_float(c2[1], 1.0) if len(c2) > 1 and c2[1].strip() else 1.0
                r90 = _safe_float(c2[2], 1.0) if len(c2) > 2 and c2[2].strip() else 1.0
                chard = _safe_float(c2[3]) if len(c2) > 3 else 0.0
                iyield = _safe_int(c2[4]) if len(c2) > 4 else 0
            # Card 4: eps, epst1, epst2, num_curves, fsmooth, fcut ([20, 20, 20, 10, 10, 20])
            if len(valid_cards) > 3:
                c3 = split_fixed(valid_cards[3].raw, [20, 20, 20, 10, 10, 20])
                eps_max = _safe_float(c3[0]) if len(c3) > 0 else 0.0
                epst1 = _safe_float(c3[1]) if len(c3) > 1 else 0.0
                epst2 = _safe_float(c3[2]) if len(c3) > 2 else 0.0
                if len(c3) >= 6 and c3[5].strip():
                    fcut = _safe_float(c3[5])
                    fsmooth = _safe_int(c3[4])
                elif len(c3) > 4 and c3[4].strip():
                    fsmooth = _safe_int(c3[4])
            # Curve cards: from card 4 onwards
            for c_idx in range(4, len(valid_cards)):
                raw_c = valid_cards[c_idx].raw
                if len(raw_c.rstrip()) > 50:
                    cc = split_fixed(raw_c, [10, 10, 20, 20])
                    fct_id = _safe_int(cc[0]) if len(cc) > 0 else 0
                    fscale = _safe_float(cc[2], 1.0) if len(cc) > 2 and cc[2].strip() else 1.0
                    eps_dot = _safe_float(cc[3]) if len(cc) > 3 else 0.0
                else:
                    cc = valid_cards[c_idx].cut("MAT_LAW43_CURVE")
                    fct_id = _safe_int(cc[0]) if len(cc) > 0 else 0
                    fscale = _safe_float(cc[1], 1.0) if len(cc) > 1 and cc[1].strip() else 1.0
                    eps_dot = _safe_float(cc[2]) if len(cc) > 2 else 0.0
                if fscale == 0.0:
                    fscale = 1.0
                curves.append({"fct_id": fct_id, "funct_id": fct_id, "fscale": fscale, "eps_dot": eps_dot})
        else:
            # Card 1: RHO, Refer_Rho (MAT_LAW43_1: [20, 20])
            if len(valid_cards) > 0:
                c0 = valid_cards[0].cut("MAT_LAW43_1")
                rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
                rhor = _safe_float(c0[1]) if len(c0) > 1 and c0[1].strip() else rho0

            # Card 2: E, nu (MAT_LAW43_2: [20, 20])
            if len(valid_cards) > 1:
                c1 = valid_cards[1].cut("MAT_LAW43_2")
                e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
                nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0

            # Card 3: Yr_fun, Einf, C (MAT_LAW43_3: [10, 20, 20] or CFG format with 10 blank spaces [10, 10, 20, 20])
            if len(valid_cards) > 2:
                raw3 = valid_cards[2].raw
                if len(raw3.rstrip()) > 50 or (len(raw3) >= 60 and raw3[10:20].strip() == ""):
                    c2 = split_fixed(raw3, [10, 10, 20, 20])
                    ifunce = _safe_int(c2[0]) if len(c2) > 0 else 0
                    einf = _safe_float(c2[2]) if len(c2) > 2 else 0.0
                    ce = _safe_float(c2[3]) if len(c2) > 3 else 0.0
                else:
                    c2 = valid_cards[2].cut("MAT_LAW43_3")
                    ifunce = _safe_int(c2[0]) if len(c2) > 0 else 0
                    einf = _safe_float(c2[1]) if len(c2) > 1 else 0.0
                    ce = _safe_float(c2[2]) if len(c2) > 2 else 0.0

            # Card 4: R00, R45, R90, CHard, Iyield (MAT_LAW43_4: [20, 20, 20, 20, 10])
            if len(valid_cards) > 3:
                c3 = valid_cards[3].cut("MAT_LAW43_4")
                r00 = _safe_float(c3[0], 1.0) if len(c3) > 0 and c3[0].strip() else 1.0
                r45 = _safe_float(c3[1], 1.0) if len(c3) > 1 and c3[1].strip() else 1.0
                r90 = _safe_float(c3[2], 1.0) if len(c3) > 2 and c3[2].strip() else 1.0
                chard = _safe_float(c3[3]) if len(c3) > 3 else 0.0
                iyield = _safe_int(c3[4]) if len(c3) > 4 else 0

            # Card 5: EPS_max, EPST1, EPST2, Fcut, Fsmooth (MAT_LAW43_5: [20, 20, 20, 20, 10])
            if len(valid_cards) > 4:
                c4 = valid_cards[4].cut("MAT_LAW43_5")
                eps_max = _safe_float(c4[0]) if len(c4) > 0 else 0.0
                epst1 = _safe_float(c4[1]) if len(c4) > 1 else 0.0
                epst2 = _safe_float(c4[2]) if len(c4) > 2 else 0.0
                fcut = _safe_float(c4[3]) if len(c4) > 3 else 0.0
                fsmooth = _safe_int(c4[4]) if len(c4) > 4 else 0

            # Curve cards: cards 5 onwards (MAT_LAW43_CURVE: [10, 20, 20])
            for c_idx in range(5, len(valid_cards)):
                raw_c = valid_cards[c_idx].raw
                if len(raw_c.rstrip()) > 50 or (len(raw_c) >= 60 and raw_c[10:20].strip() == ""):
                    cc = split_fixed(raw_c, [10, 10, 20, 20])
                    fct_id = _safe_int(cc[0]) if len(cc) > 0 else 0
                    fscale = _safe_float(cc[2], 1.0) if len(cc) > 2 and cc[2].strip() else 1.0
                    eps_dot = _safe_float(cc[3]) if len(cc) > 3 else 0.0
                else:
                    cc = valid_cards[c_idx].cut("MAT_LAW43_CURVE")
                    fct_id = _safe_int(cc[0]) if len(cc) > 0 else 0
                    fscale = _safe_float(cc[1], 1.0) if len(cc) > 1 and cc[1].strip() else 1.0
                    eps_dot = _safe_float(cc[2]) if len(cc) > 2 else 0.0
                if fscale == 0.0:
                    fscale = 1.0
                curves.append({"fct_id": fct_id, "funct_id": fct_id, "fscale": fscale, "eps_dot": eps_dot})
    else:
        # Free format
        # Card 1: RHO, Refer_Rho
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho0 = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            rhor = _safe_float(t0[1]) if len(t0) > 1 else rho0

        # Check if legacy 4-card format where Card 2 has >= 3 tokens: [E, NU, YR_FUN, EFIB, C]
        if len(valid_cards) > 1 and len(valid_cards[1].tokens()) >= 3:
            t1 = valid_cards[1].tokens()
            e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            ifunce = _safe_int(t1[2]) if len(t1) > 2 else 0
            einf = _safe_float(t1[3]) if len(t1) > 3 else 0.0
            ce = _safe_float(t1[4]) if len(t1) > 4 else 0.0

            if len(valid_cards) > 2:
                t2 = valid_cards[2].tokens()
                r00 = _safe_float(t2[0], 1.0) if len(t2) > 0 else 1.0
                r45 = _safe_float(t2[1], 1.0) if len(t2) > 1 else 1.0
                r90 = _safe_float(t2[2], 1.0) if len(t2) > 2 else 1.0
                chard = _safe_float(t2[3]) if len(t2) > 3 else 0.0
                iyield = _safe_int(t2[4]) if len(t2) > 4 else 0

            if len(valid_cards) > 3:
                t3 = valid_cards[3].tokens()
                eps_max = _safe_float(t3[0]) if len(t3) > 0 else 0.0
                epst1 = _safe_float(t3[1]) if len(t3) > 1 else 0.0
                epst2 = _safe_float(t3[2]) if len(t3) > 2 else 0.0
                if len(t3) >= 6:
                    fsmooth = _safe_int(t3[4])
                    fcut = _safe_float(t3[5])
                else:
                    fcut = _safe_float(t3[3]) if len(t3) > 3 else 0.0
                    fsmooth = _safe_int(t3[4]) if len(t3) > 4 else 0

            for c_idx in range(4, len(valid_cards)):
                tt = valid_cards[c_idx].tokens()
                fct_id = _safe_int(tt[0]) if len(tt) > 0 else 0
                fscale = _safe_float(tt[1], 1.0) if len(tt) > 1 else 1.0
                eps_dot = _safe_float(tt[2]) if len(tt) > 2 else 0.0
                if fscale == 0.0:
                    fscale = 1.0
                curves.append({"fct_id": fct_id, "funct_id": fct_id, "fscale": fscale, "eps_dot": eps_dot})
        else:
            # Card 2: E, nu
            if len(valid_cards) > 1:
                t1 = valid_cards[1].tokens()
                e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
                nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0

            # Card 3: Yr_fun, Einf, C
            if len(valid_cards) > 2:
                t2 = valid_cards[2].tokens()
                if len(t2) == 4 and t2[1] == "":
                    ifunce = _safe_int(t2[0])
                    einf = _safe_float(t2[2])
                    ce = _safe_float(t2[3])
                else:
                    ifunce = _safe_int(t2[0]) if len(t2) > 0 else 0
                    einf = _safe_float(t2[1]) if len(t2) > 1 else 0.0
                    ce = _safe_float(t2[2]) if len(t2) > 2 else 0.0

            # Card 4: R00, R45, R90, CHard, Iyield
            if len(valid_cards) > 3:
                t3 = valid_cards[3].tokens()
                r00 = _safe_float(t3[0], 1.0) if len(t3) > 0 else 1.0
                r45 = _safe_float(t3[1], 1.0) if len(t3) > 1 else 1.0
                r90 = _safe_float(t3[2], 1.0) if len(t3) > 2 else 1.0
                chard = _safe_float(t3[3]) if len(t3) > 3 else 0.0
                iyield = _safe_int(t3[4]) if len(t3) > 4 else 0

            # Card 5: EPS_max, EPST1, EPST2, Fcut, Fsmooth
            if len(valid_cards) > 4:
                t4 = valid_cards[4].tokens()
                eps_max = _safe_float(t4[0]) if len(t4) > 0 else 0.0
                epst1 = _safe_float(t4[1]) if len(t4) > 1 else 0.0
                epst2 = _safe_float(t4[2]) if len(t4) > 2 else 0.0
                fcut = _safe_float(t4[3]) if len(t4) > 3 else 0.0
                fsmooth = _safe_int(t4[4]) if len(t4) > 4 else 0

            # Curve cards: cards 5 onwards
            for c_idx in range(5, len(valid_cards)):
                tt = valid_cards[c_idx].tokens()
                if len(tt) == 4 and tt[1] == "":
                    fct_id = _safe_int(tt[0])
                    fscale = _safe_float(tt[2], 1.0)
                    eps_dot = _safe_float(tt[3])
                else:
                    fct_id = _safe_int(tt[0]) if len(tt) > 0 else 0
                    fscale = _safe_float(tt[1], 1.0) if len(tt) > 1 else 1.0
                    eps_dot = _safe_float(tt[2]) if len(tt) > 2 else 0.0
                if fscale == 0.0:
                    fscale = 1.0
                curves.append({"fct_id": fct_id, "funct_id": fct_id, "fscale": fscale, "eps_dot": eps_dot})

    # Upstream defaults from hm_read_mat43.F:
    if rhor == 0.0:
        rhor = rho0
    epsr1 = epst1 if epst1 != 0.0 else 1.0e30
    epsr2 = epst2 if epst2 != 0.0 else 2.0e30
    fisokin = chard

    asrate = fcut
    israte = fsmooth
    if asrate != 0.0:
        israte = 1
    elif israte != 0:
        asrate = 10000.0
    else:
        asrate = 0.0

    mat = MatLaw43(
        id=mat_id, rho0=rho0, rhor=rhor, e=e, nu=nu, ifunce=ifunce, einf=einf, ce=ce,
        r00=r00, r45=r45, r90=r90, chard=chard, fisokin=fisokin, iyield=iyield,
        eps_max=eps_max, epsr1=epsr1, epst1=epst1, epsr2=epsr2, epst2=epst2,
        fcut=fcut, asrate=asrate, fsmooth=fsmooth, israte=israte,
        curves=curves, title=title, law=43, law_name=law_name
    )
    model.mat_law43s[mat_id] = mat

    try:
        from ...materials.law43_hill_tab import build_law43
        model.materials[mat_id] = build_law43(mat)
    except Exception:
        try:
            from ...materials import build_law43
            model.materials[mat_id] = build_law43(mat)
        except Exception:
            from ..mat_reader import InactiveMaterial
            model.materials[mat_id] = InactiveMaterial(
                id=mat_id, law=43, rho0=rho0, title=title, law_name=law_name,
                params={
                    "rho": rho0, "rho0": rho0, "rhor": rhor, "e": e, "nu": nu,
                    "ifunce": ifunce, "yr_fun": ifunce, "einf": einf, "efib": einf, "ce": ce, "c": ce,
                    "r00": r00, "r45": r45, "r90": r90, "chard": chard, "fisokin": fisokin, "iyield": iyield,
                    "eps_max": eps_max, "eps": eps_max, "epst1": epst1, "epsr1": epsr1,
                    "epst2": epst2, "epsr2": epsr2, "fcut": fcut, "asrate": asrate,
                    "fsmooth": fsmooth, "israte": israte, "curves": curves,
                    "MAT_RHO": rho0, "MAT_E": e, "MAT_NU": nu,
                }
            )




def read_mat_law53(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW53`` or ``/MAT/TSAI_TAB`` (M193): Tsai-Wu tabulated orthotropic plasticity model."""
    from ...model.entities import MatLaw53, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    rho, refer_rho = 0.0, 0.0
    e1, e2, gab, gbc = 0.0, 0.0, 0.0, 0.0
    fun_a1, fun_b1, fun_a3, fun_a5, fun_a6 = 0, 0, 0, 0, 0
    sfac11, sfac22, sfac12, sfac23, sfac45 = 1.0, 1.0, 1.0, 1.0, 1.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW53_1")
            rho = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            refer_rho = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW53_2")
            e1 = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            e2 = _safe_float(c1[1]) if len(c1) > 1 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW53_3")
            gab = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            gbc = _safe_float(c2[1]) if len(c2) > 1 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW53_4")
            fun_a1 = _safe_int(c3[0]) if len(c3) > 0 else 0
            fun_b1 = _safe_int(c3[1]) if len(c3) > 1 else 0
            fun_a3 = _safe_int(c3[2]) if len(c3) > 2 else 0
            fun_a5 = _safe_int(c3[3]) if len(c3) > 3 else 0
            fun_a6 = _safe_int(c3[4]) if len(c3) > 4 else 0
        if len(valid_cards) > 4:
            c4 = valid_cards[4].cut("MAT_LAW53_5")
            sfac11 = _safe_float(c4[0], 1.0) if len(c4) > 0 and c4[0].strip() else 1.0
            sfac22 = _safe_float(c4[1], 1.0) if len(c4) > 1 and c4[1].strip() else 1.0
            sfac12 = _safe_float(c4[2], 1.0) if len(c4) > 2 and c4[2].strip() else 1.0
            sfac23 = _safe_float(c4[3], 1.0) if len(c4) > 3 and c4[3].strip() else 1.0
            sfac45 = _safe_float(c4[4], 1.0) if len(c4) > 4 and c4[4].strip() else 1.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            refer_rho = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            e1 = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            e2 = _safe_float(t1[1]) if len(t1) > 1 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            gab = _safe_float(t2[0]) if len(t2) > 0 else 0.0
            gbc = _safe_float(t2[1]) if len(t2) > 1 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            fun_a1 = _safe_int(t3[0]) if len(t3) > 0 else 0
            fun_b1 = _safe_int(t3[1]) if len(t3) > 1 else 0
            fun_a3 = _safe_int(t3[2]) if len(t3) > 2 else 0
            fun_a5 = _safe_int(t3[3]) if len(t3) > 3 else 0
            fun_a6 = _safe_int(t3[4]) if len(t3) > 4 else 0
        if len(valid_cards) > 4:
            t4 = valid_cards[4].tokens()
            sfac11 = _safe_float(t4[0], 1.0) if len(t4) > 0 else 1.0
            sfac22 = _safe_float(t4[1], 1.0) if len(t4) > 1 else 1.0
            sfac12 = _safe_float(t4[2], 1.0) if len(t4) > 2 else 1.0
            sfac23 = _safe_float(t4[3], 1.0) if len(t4) > 3 else 1.0
            sfac45 = _safe_float(t4[4], 1.0) if len(t4) > 4 else 1.0

    mat = MatLaw53(
        id=mat_id, rho=rho, ref_rho=refer_rho, e1=e1, e2=e2, gab=gab, gbc=gbc,
        fun_a1=fun_a1, fun_b1=fun_b1, fun_a3=fun_a3, fun_a5=fun_a5, fun_a6=fun_a6,
        sfac11=sfac11, sfac22=sfac22, sfac12=sfac12, sfac23=sfac23, sfac45=sfac45,
        title=title,
    )
    model.mat_law53s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=53, rho0=rho, title=title,
        params={
            "MAT_RHO": rho, "rho": rho, "rho0": rho, "refer_rho": refer_rho,
            "MAT_E1": e1, "e1": e1, "MAT_E2": e2, "e2": e2,
            "MAT_GAB": gab, "gab": gab, "MAT_GBC": gbc, "gbc": gbc,
            "FUN_A1": fun_a1, "FUN_B1": fun_b1, "FUN_A3": fun_a3, "FUN_A5": fun_a5, "FUN_A6": fun_a6,
            "MAT_SFAC11": sfac11, "MAT_SFAC22": sfac22, "MAT_SFAC12": sfac12, "MAT_SFAC23": sfac23, "MAT_SFAC45": sfac45,
        }
    )




def read_mat_law54(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW54`` or ``/MAT/PREDIT`` (M193): Specialized progressive damage plasticity model."""
    from ...model.entities import MatLaw54, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    rho, refer_rho = 0.0, 0.0
    e, nu = 0.0, 0.0
    ifunc = 0
    a, b, n, sfac = 0.0, 0.0, 0.0, 1.0
    ay, az, by, bz, cx = 0.0, 0.0, 0.0, 0.0, 0.0
    dc, rc, eps_max = 0.0, 0.0, 0.0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW54_1")
            rho = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            refer_rho = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW54_2")
            e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW54_3")
            ifunc = _safe_int(c2[1]) if len(c2) > 1 else 0
            a = _safe_float(c2[2]) if len(c2) > 2 else 0.0
            b = _safe_float(c2[3]) if len(c2) > 3 else 0.0
            n = _safe_float(c2[4]) if len(c2) > 4 else 0.0
            sfac = _safe_float(c2[5], 1.0) if len(c2) > 5 and c2[5].strip() else 1.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW54_4")
            ay = _safe_float(c3[0]) if len(c3) > 0 else 0.0
            az = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            by = _safe_float(c3[2]) if len(c3) > 2 else 0.0
            bz = _safe_float(c3[3]) if len(c3) > 3 else 0.0
            cx = _safe_float(c3[4]) if len(c3) > 4 else 0.0
        if len(valid_cards) > 4:
            c4 = valid_cards[4].cut("MAT_LAW54_5")
            dc = _safe_float(c4[0]) if len(c4) > 0 else 0.0
            rc = _safe_float(c4[1]) if len(c4) > 1 else 0.0
            eps_max = _safe_float(c4[2]) if len(c4) > 2 else 0.0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            rho = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            refer_rho = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            ifunc = _safe_int(t2[0]) if len(t2) > 0 else 0
            a = _safe_float(t2[1]) if len(t2) > 1 else 0.0
            b = _safe_float(t2[2]) if len(t2) > 2 else 0.0
            n = _safe_float(t2[3]) if len(t2) > 3 else 0.0
            sfac = _safe_float(t2[4], 1.0) if len(t2) > 4 else 1.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            ay = _safe_float(t3[0]) if len(t3) > 0 else 0.0
            az = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            by = _safe_float(t3[2]) if len(t3) > 2 else 0.0
            bz = _safe_float(t3[3]) if len(t3) > 3 else 0.0
            cx = _safe_float(t3[4]) if len(t3) > 4 else 0.0
        if len(valid_cards) > 4:
            t4 = valid_cards[4].tokens()
            dc = _safe_float(t4[0]) if len(t4) > 0 else 0.0
            rc = _safe_float(t4[1]) if len(t4) > 1 else 0.0
            eps_max = _safe_float(t4[2]) if len(t4) > 2 else 0.0

    mat = MatLaw54(
        id=mat_id, rho=rho, ref_rho=refer_rho, e=e, nu=nu, ifunc=ifunc,
        a=a, b=b, n=n, sfac=sfac, ay=ay, az=az, by=by, bz=bz, cx=cx,
        dc=dc, rc=rc, eps_max=eps_max, title=title,
    )
    model.mat_law54s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=54, rho0=rho, title=title,
        params={
            "MAT_RHO": rho, "rho": rho, "rho0": rho, "refer_rho": refer_rho,
            "MAT_E": e, "e": e, "MAT_NU": nu, "nu": nu,
            "FUNC": ifunc, "MAT_A": a, "MAT_B": b, "MAT_N": n, "MAT_Sfac_Yield": sfac,
            "MAT_Ay": ay, "MAT_Az": az, "MAT_By": by, "MAT_Bz": bz, "MAT_Cx": cx,
            "MAT_Dc": dc, "MAT_Rc": rc, "MAT_EPS": eps_max,
        }
    )




def read_mat_law74(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW74``, ``/MAT/HILL_3D``, ``/MAT/ORTH_PLAS``, or ``/MAT/THERM_HILL`` (M193, M563):
    Tabulated Hill orthotropic plasticity for solids.
    Upstream reference: hm_read_mat74.F and CFG radioss120/MAT/matl74_74.cfg.
    """
    from ...model.entities import MatLaw74, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]

    rho, refer_rho = 0.0, 0.0
    e, nu = 0.0, 0.0
    eps_max, epsr1, epsr2 = 1.0e30, 1.0e30, 2.0e30
    ifunce, einf, ce = 0, 0.0, 0.0
    fsmooth = 0
    chard, fcut = 0.0, 0.0
    s11y, s22y, s33y = 1.0, 1.0, 1.0
    s12y, s23y, s31y = 1.0, 1.0, 1.0
    table_id = 0
    fscale, pscale = 1.0, 1.0
    t0, rhocp = 293.0, 0.0

    is_fixed = block.fixed
    if is_fixed:
        for vc in valid_cards[:4]:
            if "," in vc.raw or (len(vc.tokens()) > 1 and len(vc.raw[:20].split()) > 1):
                is_fixed = False
                break

    def _card_tokens(card: Any) -> List[str]:
        return [t.strip().rstrip(",") for t in card.raw.replace(",", " ").split() if t.strip().rstrip(",")]

    is_legacy_7card = False
    if len(valid_cards) <= 6:
        is_legacy_7card = True
    elif len(valid_cards) == 7:
        if is_fixed:
            c5 = valid_cards[5].cut("MAT_LAW74_6")
            has_3_stresses_at_5 = (
                len(c5) >= 3 and bool(c5[0].strip()) and bool(c5[1].strip()) and bool(c5[2].strip())
            )
            if not has_3_stresses_at_5:
                is_legacy_7card = True
        else:
            t6 = _card_tokens(valid_cards[6])
            t5 = _card_tokens(valid_cards[5])
            if len(t6) <= 2 and len(t5) == 3:
                try:
                    ti_cand = float(t6[0])
                    tab_cand = int(float(t5[0]))
                    if ti_cand > 100.0 and tab_cand < 10000:
                        is_legacy_7card = True
                except (ValueError, IndexError):
                    pass

    if is_legacy_7card:
        # Legacy 7-card layout (radioss110 without Yr_fun / Einf / Ce)
        if is_fixed:
            if len(valid_cards) > 0:
                c0 = valid_cards[0].cut("MAT_LAW74_1")
                rho = _safe_float(c0[0]) if len(c0) > 0 else 0.0
                refer_rho = _safe_float(c0[1]) if len(c0) > 1 and c0[1].strip() else 0.0
            if len(valid_cards) > 1:
                c1 = valid_cards[1].cut("MAT_LAW74_2")
                e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
                nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
                eps_max = _safe_float(c1[2], 1.0e30) if len(c1) > 2 and c1[2].strip() else 1.0e30
                epsr1 = _safe_float(c1[3], 1.0e30) if len(c1) > 3 and c1[3].strip() else 1.0e30
                epsr2 = _safe_float(c1[4], 2.0e30) if len(c1) > 4 and c1[4].strip() else 2.0e30
            if len(valid_cards) > 2:
                c2 = valid_cards[2].cut("MAT_LAW74_4")
                if len(c2) > 1 and c2[1].strip():
                    fsmooth = _safe_int(c2[1])
                elif len(c2) > 0 and c2[0].strip():
                    fsmooth = _safe_int(c2[0])
                chard = _safe_float(c2[2]) if len(c2) > 2 else 0.0
                fcut = _safe_float(c2[3]) if len(c2) > 3 and c2[3].strip() else 0.0
            if len(valid_cards) > 3:
                c3 = valid_cards[3].cut("MAT_LAW74_5")
                s11y = _safe_float(c3[0], 1.0) if len(c3) > 0 and c3[0].strip() else 1.0
                s22y = _safe_float(c3[1], 1.0) if len(c3) > 1 and c3[1].strip() else 1.0
                s33y = _safe_float(c3[2], 1.0) if len(c3) > 2 and c3[2].strip() else 1.0
            if len(valid_cards) > 4:
                c4 = valid_cards[4].cut("MAT_LAW74_6")
                s12y = _safe_float(c4[0], 1.0) if len(c4) > 0 and c4[0].strip() else 1.0
                s23y = _safe_float(c4[1], 1.0) if len(c4) > 1 and c4[1].strip() else 1.0
                s31y = _safe_float(c4[2], 1.0) if len(c4) > 2 and c4[2].strip() else 1.0
            if len(valid_cards) > 5:
                c5 = valid_cards[5].cut("MAT_LAW74_7")
                if len(c5) > 0 and c5[0].strip():
                    table_id = _safe_int(c5[0])
                elif len(c5) > 1 and c5[1].strip():
                    table_id = _safe_int(c5[1])
                fscale = _safe_float(c5[2], 1.0) if len(c5) > 2 and c5[2].strip() else 1.0
                pscale = _safe_float(c5[3], 1.0) if len(c5) > 3 and c5[3].strip() else 1.0
            if len(valid_cards) > 6:
                c6 = valid_cards[6].cut("MAT_LAW74_8")
                t0 = _safe_float(c6[0], 293.0) if len(c6) > 0 and c6[0].strip() else 293.0
                rhocp = _safe_float(c6[1]) if len(c6) > 1 else 0.0
        else:
            if len(valid_cards) > 0:
                t0_tok = _card_tokens(valid_cards[0])
                rho = _safe_float(t0_tok[0]) if len(t0_tok) > 0 else 0.0
                refer_rho = _safe_float(t0_tok[1]) if len(t0_tok) > 1 else 0.0
            if len(valid_cards) > 1:
                t1 = _card_tokens(valid_cards[1])
                e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
                nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
                eps_max = _safe_float(t1[2], 1.0e30) if len(t1) > 2 else 1.0e30
                epsr1 = _safe_float(t1[3], 1.0e30) if len(t1) > 3 else 1.0e30
                epsr2 = _safe_float(t1[4], 2.0e30) if len(t1) > 4 else 2.0e30
            if len(valid_cards) > 2:
                t2 = _card_tokens(valid_cards[2])
                if len(t2) == 1:
                    fsmooth = _safe_int(t2[0])
                elif len(t2) == 2:
                    fsmooth = _safe_int(t2[0])
                    chard = _safe_float(t2[1])
                elif len(t2) == 3:
                    fsmooth = _safe_int(t2[0])
                    chard = _safe_float(t2[1])
                    fcut = _safe_float(t2[2])
                elif len(t2) >= 4:
                    fsmooth = _safe_int(t2[0]) if _safe_int(t2[0]) != 0 else _safe_int(t2[1])
                    chard = _safe_float(t2[2])
                    fcut = _safe_float(t2[3])
            if len(valid_cards) > 3:
                t3 = _card_tokens(valid_cards[3])
                s11y = _safe_float(t3[0], 1.0) if len(t3) > 0 else 1.0
                s22y = _safe_float(t3[1], 1.0) if len(t3) > 1 else 1.0
                s33y = _safe_float(t3[2], 1.0) if len(t3) > 2 else 1.0
            if len(valid_cards) > 4:
                t4 = _card_tokens(valid_cards[4])
                s12y = _safe_float(t4[0], 1.0) if len(t4) > 0 else 1.0
                s23y = _safe_float(t4[1], 1.0) if len(t4) > 1 else 1.0
                s31y = _safe_float(t4[2], 1.0) if len(t4) > 2 else 1.0
            if len(valid_cards) > 5:
                t5 = _card_tokens(valid_cards[5])
                if len(t5) == 1:
                    table_id = _safe_int(t5[0])
                elif len(t5) == 2:
                    table_id = _safe_int(t5[0])
                    fscale = _safe_float(t5[1], 1.0)
                elif len(t5) == 3:
                    table_id = _safe_int(t5[0])
                    fscale = _safe_float(t5[1], 1.0)
                    pscale = _safe_float(t5[2], 1.0)
                elif len(t5) >= 4:
                    table_id = _safe_int(t5[0]) if _safe_int(t5[0]) != 0 else _safe_int(t5[1])
                    fscale = _safe_float(t5[2], 1.0)
                    pscale = _safe_float(t5[3], 1.0)
            if len(valid_cards) > 6:
                t6 = _card_tokens(valid_cards[6])
                t0 = _safe_float(t6[0], 293.0) if len(t6) > 0 else 293.0
                rhocp = _safe_float(t6[1]) if len(t6) > 1 else 0.0
    else:
        # Standard 8-card layout (radioss120 / CFG matl74_74.cfg / hm_read_mat74.F)
        if is_fixed:
            # Card 1: RHO, [Refer_Rho] (MAT_LAW74_1: [20, 20])
            if len(valid_cards) > 0:
                c0 = valid_cards[0].cut("MAT_LAW74_1")
                rho = _safe_float(c0[0]) if len(c0) > 0 else 0.0
                refer_rho = _safe_float(c0[1]) if len(c0) > 1 and c0[1].strip() else 0.0
            # Card 2: E, NU, EPS_MAX, EPSR1, EPSR2 (MAT_LAW74_2: [20, 20, 20, 20, 20])
            if len(valid_cards) > 1:
                c1 = valid_cards[1].cut("MAT_LAW74_2")
                e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
                nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
                eps_max = _safe_float(c1[2], 1.0e30) if len(c1) > 2 and c1[2].strip() else 1.0e30
                epsr1 = _safe_float(c1[3], 1.0e30) if len(c1) > 3 and c1[3].strip() else 1.0e30
                epsr2 = _safe_float(c1[4], 2.0e30) if len(c1) > 4 and c1[4].strip() else 2.0e30
            # Card 3: Yr_fun, blank(10), EINF, CE (MAT_LAW74_3: [10, 10, 20, 20])
            if len(valid_cards) > 2:
                c2 = valid_cards[2].cut("MAT_LAW74_3")
                if len(c2) > 0 and c2[0].strip():
                    ifunce = _safe_int(c2[0])
                elif len(c2) > 1 and c2[1].strip():
                    ifunce = _safe_int(c2[1])
                einf = _safe_float(c2[2]) if len(c2) > 2 else 0.0
                ce = _safe_float(c2[3]) if len(c2) > 3 else 0.0
            # Card 4: blank(10), Fsmooth, C_HARD, FCUT (MAT_LAW74_4: [10, 10, 20, 20])
            if len(valid_cards) > 3:
                c3 = valid_cards[3].cut("MAT_LAW74_4")
                if len(c3) > 1 and c3[1].strip():
                    fsmooth = _safe_int(c3[1])
                elif len(c3) > 0 and c3[0].strip():
                    fsmooth = _safe_int(c3[0])
                chard = _safe_float(c3[2]) if len(c3) > 2 else 0.0
                fcut = _safe_float(c3[3]) if len(c3) > 3 and c3[3].strip() else 0.0
            # Card 5: S11Y, S22Y, S33Y (MAT_LAW74_5: [20, 20, 20])
            if len(valid_cards) > 4:
                c4 = valid_cards[4].cut("MAT_LAW74_5")
                s11y = _safe_float(c4[0], 1.0) if len(c4) > 0 and c4[0].strip() else 1.0
                s22y = _safe_float(c4[1], 1.0) if len(c4) > 1 and c4[1].strip() else 1.0
                s33y = _safe_float(c4[2], 1.0) if len(c4) > 2 and c4[2].strip() else 1.0
            # Card 6: S12Y, S23Y, S31Y (MAT_LAW74_6: [20, 20, 20])
            if len(valid_cards) > 5:
                c5 = valid_cards[5].cut("MAT_LAW74_6")
                s12y = _safe_float(c5[0], 1.0) if len(c5) > 0 and c5[0].strip() else 1.0
                s23y = _safe_float(c5[1], 1.0) if len(c5) > 1 and c5[1].strip() else 1.0
                s31y = _safe_float(c5[2], 1.0) if len(c5) > 2 and c5[2].strip() else 1.0
            # Card 7: FUN_A1, blank(10), FSCALE, PSCALE (MAT_LAW74_7: [10, 10, 20, 20])
            if len(valid_cards) > 6:
                c6 = valid_cards[6].cut("MAT_LAW74_7")
                if len(c6) > 0 and c6[0].strip():
                    table_id = _safe_int(c6[0])
                elif len(c6) > 1 and c6[1].strip():
                    table_id = _safe_int(c6[1])
                fscale = _safe_float(c6[2], 1.0) if len(c6) > 2 and c6[2].strip() else 1.0
                pscale = _safe_float(c6[3], 1.0) if len(c6) > 3 and c6[3].strip() else 1.0
            # Card 8: T0, RHOCP (MAT_LAW74_8: [20, 20])
            if len(valid_cards) > 7:
                c7 = valid_cards[7].cut("MAT_LAW74_8")
                t0 = _safe_float(c7[0], 293.0) if len(c7) > 0 and c7[0].strip() else 293.0
                rhocp = _safe_float(c7[1]) if len(c7) > 1 else 0.0
        else:
            # Free format 8 cards
            if len(valid_cards) > 0:
                t0_tok = _card_tokens(valid_cards[0])
                rho = _safe_float(t0_tok[0]) if len(t0_tok) > 0 else 0.0
                refer_rho = _safe_float(t0_tok[1]) if len(t0_tok) > 1 else 0.0
            if len(valid_cards) > 1:
                t1 = _card_tokens(valid_cards[1])
                e = _safe_float(t1[0]) if len(t1) > 0 else 0.0
                nu = _safe_float(t1[1]) if len(t1) > 1 else 0.0
                eps_max = _safe_float(t1[2], 1.0e30) if len(t1) > 2 else 1.0e30
                epsr1 = _safe_float(t1[3], 1.0e30) if len(t1) > 3 else 1.0e30
                epsr2 = _safe_float(t1[4], 2.0e30) if len(t1) > 4 else 2.0e30
            if len(valid_cards) > 2:
                t2 = _card_tokens(valid_cards[2])
                if len(t2) == 2:
                    einf = _safe_float(t2[0])
                    ce = _safe_float(t2[1])
                elif len(t2) == 3:
                    ifunce = _safe_int(t2[0])
                    einf = _safe_float(t2[1])
                    ce = _safe_float(t2[2])
                elif len(t2) >= 4:
                    ifunce = _safe_int(t2[0]) if _safe_int(t2[0]) != 0 else _safe_int(t2[1])
                    einf = _safe_float(t2[2])
                    ce = _safe_float(t2[3])
            if len(valid_cards) > 3:
                t3 = _card_tokens(valid_cards[3])
                if len(t3) == 1:
                    fsmooth = _safe_int(t3[0])
                elif len(t3) == 2:
                    fsmooth = _safe_int(t3[0])
                    chard = _safe_float(t3[1])
                elif len(t3) == 3:
                    fsmooth = _safe_int(t3[0])
                    chard = _safe_float(t3[1])
                    fcut = _safe_float(t3[2])
                elif len(t3) >= 4:
                    fsmooth = _safe_int(t3[0]) if _safe_int(t3[0]) != 0 else _safe_int(t3[1])
                    chard = _safe_float(t3[2])
                    fcut = _safe_float(t3[3])
            if len(valid_cards) > 4:
                t4 = _card_tokens(valid_cards[4])
                s11y = _safe_float(t4[0], 1.0) if len(t4) > 0 else 1.0
                s22y = _safe_float(t4[1], 1.0) if len(t4) > 1 else 1.0
                s33y = _safe_float(t4[2], 1.0) if len(t4) > 2 else 1.0
            if len(valid_cards) > 5:
                t5 = _card_tokens(valid_cards[5])
                s12y = _safe_float(t5[0], 1.0) if len(t5) > 0 else 1.0
                s23y = _safe_float(t5[1], 1.0) if len(t5) > 1 else 1.0
                s31y = _safe_float(t5[2], 1.0) if len(t5) > 2 else 1.0
            if len(valid_cards) > 6:
                t6 = _card_tokens(valid_cards[6])
                if len(t6) == 1:
                    table_id = _safe_int(t6[0])
                elif len(t6) == 2:
                    table_id = _safe_int(t6[0])
                    fscale = _safe_float(t6[1], 1.0)
                elif len(t6) == 3:
                    table_id = _safe_int(t6[0])
                    fscale = _safe_float(t6[1], 1.0)
                    pscale = _safe_float(t6[2], 1.0)
                elif len(t6) >= 4:
                    table_id = _safe_int(t6[0]) if _safe_int(t6[0]) != 0 else _safe_int(t6[1])
                    fscale = _safe_float(t6[2], 1.0)
                    pscale = _safe_float(t6[3], 1.0)
            if len(valid_cards) > 7:
                t7 = _card_tokens(valid_cards[7])
                t0 = _safe_float(t7[0], 293.0) if len(t7) > 0 else 293.0
                rhocp = _safe_float(t7[1]) if len(t7) > 1 else 0.0

    # Upstream defaults from hm_read_mat74.F & matl74_74.cfg
    if eps_max == 0.0:
        eps_max = 1.0e30
    if epsr1 == 0.0:
        epsr1 = 1.0e30
    if epsr2 == 0.0:
        epsr2 = 2.0e30
    if fscale == 0.0:
        fscale = 1.0
    if pscale == 0.0:
        pscale = 1.0
    if t0 == 0.0:
        t0 = 293.0
    if refer_rho == 0.0:
        refer_rho = rho
    if s11y == 0.0:
        s11y = 1.0
    if s22y == 0.0:
        s22y = 1.0
    if s33y == 0.0:
        s33y = 1.0
    if s12y == 0.0:
        s12y = 1.0
    if s23y == 0.0:
        s23y = 1.0
    if s31y == 0.0:
        s31y = 1.0

    mat = MatLaw74(
        id=mat_id, rho=rho, refer_rho=refer_rho, e=e, nu=nu,
        eps_max=eps_max, epsr1=epsr1, epsr2=epsr2,
        ifunce=ifunce, einf=einf, ce=ce,
        fsmooth=fsmooth, chard=chard, fcut=fcut,
        s11y=s11y, s22y=s22y, s33y=s33y,
        s12y=s12y, s23y=s23y, s31y=s31y,
        table_id=table_id, fscale=fscale, pscale=pscale,
        t0=t0, rhocp=rhocp,
        title=title, law=74, law_name="LAW74",
    )
    model.mat_law74s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id, law=74, rho0=rho, title=title,
        params=mat.params,
    )




def read_mat_law82(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW82`` or ``/MAT/OGDEN``: Ogden hyperelastic material model."""
    from ...model.entities import MatLaw82, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    rho, refer_rho = 0.0, 0.0
    order = 0
    nu = 0.475
    mu_arr: list[float] = []
    alpha_arr: list[float] = []
    gamma_arr: list[float] = []

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW82_1")
            rho = _safe_float(c0[0]) if len(c0) > 0 else 0.0
            refer_rho = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW82_2")
            order = _safe_int(c1[0]) if len(c1) > 0 else 0
            nu = _safe_float(c1[2], 0.475) if len(c1) > 2 and c1[2].strip() else 0.475

        data_cards = valid_cards[2:]
        is_per_term = False
        raw_comments = [c.raw.upper() for c in block.cards if c.raw.strip().startswith("#")]
        has_array_comments = any("ALPHA" in c or "MU" in c or "GAMMA" in c or "D_I" in c for c in raw_comments)
        if not has_array_comments and order == 1 and len(data_cards) == 1:
            first_cuts = [data_cards[0].raw[i*20:(i+1)*20].strip() for i in range(3)]
            non_empty = [x for x in first_cuts if x]
            if len(non_empty) >= 2:
                is_per_term = True

        if is_per_term:
            for idx in range(min(order, len(data_cards))):
                c = data_cards[idx]
                cuts = [c.raw[i*20:(i+1)*20].strip() for i in range(3)]
                mu_i = _safe_float(cuts[0]) if len(cuts) > 0 and cuts[0] else 0.0
                al_i = _safe_float(cuts[1]) if len(cuts) > 1 and cuts[1] else 0.0
                d_i = _safe_float(cuts[2]) if len(cuts) > 2 and cuts[2] else 0.0
                mu_arr.append(mu_i)
                alpha_arr.append(al_i)
                gamma_arr.append(d_i)
        else:
            card_idx = 2
            # Mu list
            remaining = order
            while remaining > 0 and card_idx < len(valid_cards):
                c = valid_cards[card_idx]
                n_in_card = min(5, remaining)
                cuts = [c.raw[i*20:(i+1)*20].strip() for i in range(n_in_card)]
                for val in cuts:
                    if val:
                        mu_arr.append(_safe_float(val))
                remaining -= n_in_card
                card_idx += 1
            # Alpha list
            remaining = order
            while remaining > 0 and card_idx < len(valid_cards):
                c = valid_cards[card_idx]
                n_in_card = min(5, remaining)
                cuts = [c.raw[i*20:(i+1)*20].strip() for i in range(n_in_card)]
                for val in cuts:
                    if val:
                        alpha_arr.append(_safe_float(val))
                remaining -= n_in_card
                card_idx += 1
            # Gamma (D_i) list
            remaining = order
            while remaining > 0 and card_idx < len(valid_cards):
                c = valid_cards[card_idx]
                n_in_card = min(5, remaining)
                cuts = [c.raw[i*20:(i+1)*20].strip() for i in range(n_in_card)]
                for val in cuts:
                    if val:
                        gamma_arr.append(_safe_float(val))
                remaining -= n_in_card
                card_idx += 1
    else:
        def _card_tokens(c: Card) -> list[str]:
            raw = c.raw.strip()
            if "," in raw:
                return [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
            return c.tokens()

        if len(valid_cards) > 0:
            t0 = _card_tokens(valid_cards[0])
            rho = _safe_float(t0[0]) if len(t0) > 0 else 0.0
            refer_rho = _safe_float(t0[1]) if len(t0) > 1 else 0.0
        if len(valid_cards) > 1:
            t1 = _card_tokens(valid_cards[1])
            order = _safe_int(t1[0]) if len(t1) > 0 else 0
            nu = _safe_float(t1[1], 0.475) if len(t1) > 1 else 0.475

        data_cards = valid_cards[2:]
        is_per_term = False
        raw_comments = [c.raw.upper() for c in block.cards if c.raw.strip().startswith("#")]
        has_array_comments = any("ALPHA" in c or "MU" in c or "GAMMA" in c or "D_I" in c for c in raw_comments)
        if not has_array_comments and order == 1 and len(data_cards) == 1:
            toks0 = _card_tokens(data_cards[0])
            if len(toks0) >= 2:
                is_per_term = True

        if is_per_term:
            for idx in range(min(order, len(data_cards))):
                toks = _card_tokens(data_cards[idx])
                mu_i = _safe_float(toks[0]) if len(toks) > 0 else 0.0
                al_i = _safe_float(toks[1]) if len(toks) > 1 else 0.0
                d_i = _safe_float(toks[2]) if len(toks) > 2 else 0.0
                mu_arr.append(mu_i)
                alpha_arr.append(al_i)
                gamma_arr.append(d_i)
        else:
            card_idx = 2
            # Mu list
            while len(mu_arr) < order and card_idx < len(valid_cards):
                toks = _card_tokens(valid_cards[card_idx])
                for t in toks:
                    if len(mu_arr) < order:
                        mu_arr.append(_safe_float(t))
                card_idx += 1
            # Alpha list
            while len(alpha_arr) < order and card_idx < len(valid_cards):
                toks = _card_tokens(valid_cards[card_idx])
                for t in toks:
                    if len(alpha_arr) < order:
                        alpha_arr.append(_safe_float(t))
                card_idx += 1
            # Gamma list
            while len(gamma_arr) < order and card_idx < len(valid_cards):
                toks = _card_tokens(valid_cards[card_idx])
                for t in toks:
                    if len(gamma_arr) < order:
                        gamma_arr.append(_safe_float(t))
                card_idx += 1

    while len(alpha_arr) < len(mu_arr):
        alpha_arr.append(0.0)
    while len(gamma_arr) < len(mu_arr):
        gamma_arr.append(0.0)

    g0 = float(sum(mu_arr)) if mu_arr else 0.0
    if len(gamma_arr) > 0 and float(gamma_arr[0]) > 0.0:
        k0 = 2.0 / float(gamma_arr[0])
    else:
        denom = 3.0 * (1.0 - 2.0 * nu) if abs(1.0 - 2.0 * nu) > 1e-12 else 1e-6
        k0 = 2.0 * g0 * (1.0 + nu) / denom
    e0 = 2.0 * g0 * (1.0 + nu)

    mat = MatLaw82(
        id=mat_id,
        rho0=rho,
        rhor=refer_rho,
        nu=nu,
        nordre=order,
        mu=mu_arr,
        alpha=alpha_arr,
        d=gamma_arr,
        title=title,
    )
    model.mat_law82s[mat_id] = mat
    model.materials[mat_id] = Material(
        id=mat_id,
        law=82,
        rho0=rho,
        title=title,
        law_name="LAW82",
        params={
            "MAT_RHO": rho, "rho": rho, "rho0": rho, "refer_rho": refer_rho, "rhor": refer_rho,
            "ORDER": order, "nordre": order, "order": order,
            "MAT_NU": nu, "nu": nu,
            "Mu_arr": mu_arr, "mu": mu_arr,
            "Alpha_arr": alpha_arr, "alpha": alpha_arr,
            "Gamma_arr": gamma_arr, "d": gamma_arr,
            "G": g0, "K": k0, "E": e0,
        },
    )




def read_mat_law40(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW40`` or ``/MAT/CONCR_SUB`` (M194): Concrete subgrade model."""
    from ...model.entities import MatLaw40, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    rho, e, nu = 0.0, 0.0, 0.0
    params = {}
    if len(valid_cards) > 0:
        c0 = valid_cards[0].tokens()
        rho = _safe_float(c0[0]) if len(c0) > 0 else 0.0
        e = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        nu = _safe_float(c0[2]) if len(c0) > 2 else 0.0
    if len(valid_cards) > 1:
        c1 = valid_cards[1].tokens()
        for idx, val in enumerate(c1):
            params[f"param_{idx}"] = _safe_float(val)
    mat = MatLaw40(id=mat_id, title=title, rho=rho, e=e, nu=nu, params=params)
    model.mat_law40s[mat_id] = mat
    model.materials[mat_id] = Material(id=mat_id, law=40, rho0=rho, title=title, params={"rho": rho, "e": e, "nu": nu, **params})






def read_mat_law102(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW102`` or ``/MAT/DPRAG2`` (M194, M195, M572): Extended Drucker-Prager material model.

    Citing starter/source/materials/mat/mat102/hm_read_mat102.F and mat102_DPRAG2.cfg:
      Card 1: RHO_I (%20lg)
      Card 2: IFORM (%10d)
      Card 3: E, NU (%20lg%20lg)
      Card 4: C, PHI, AMAX (%20lg%20lg%20lg)
      Card 5: PMIN (%20lg)
    """
    from ...model.entities import MatLaw102, Material
    from ...materials.law102_dprag2 import compute_dprag2_constants
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]

    if not valid_cards:
        mat = MatLaw102(id=mat_id, title=title)
        model.mat_law102s[mat_id] = mat
        model.materials[mat_id] = Material(id=mat_id, law=102, rho0=0.0, title=title, params={})
        return

    # Check for legacy M194 2-card test layout:
    # Card 1 has >= 3 tokens (RHO, E, NU) and Card 2 has >= 4 tokens (F, G, H, L, M, N)
    c0_tokens = valid_cards[0].tokens()
    if len(valid_cards) == 2 and len(c0_tokens) >= 3 and len(valid_cards[1].tokens()) >= 4:
        rho = _safe_float(c0_tokens[0])
        e = _safe_float(c0_tokens[1])
        nu = _safe_float(c0_tokens[2])
        params = {}
        for idx, val in enumerate(valid_cards[1].tokens()):
            params[f"param_{idx}"] = _safe_float(val)
        mat = MatLaw102(id=mat_id, title=title, rho=rho, e=e, nu=nu, params=params)
        model.mat_law102s[mat_id] = mat
        model.materials[mat_id] = Material(id=mat_id, law=102, rho0=rho, title=title, params={"rho": rho, "e": e, "nu": nu, **params})
        return

    rho = 0.0
    iform = 2
    e, nu = 0.0, 0.0
    a0, a1, b0, b1 = 0.0, 0.0, 0.0, 0.0
    icrit = 1
    c, phi, amax = 0.0, 0.0, 1.0e30
    pmin = -1.0e30

    # Extract card tokens
    c0_tok = valid_cards[0].tokens() if len(valid_cards) > 0 else []
    c1_tok = valid_cards[1].tokens() if len(valid_cards) > 1 else []
    c2_tok = valid_cards[2].tokens() if len(valid_cards) > 2 else []
    c3_tok = valid_cards[3].tokens() if len(valid_cards) > 3 else []
    c4_tok = valid_cards[4].tokens() if len(valid_cards) > 4 else []

    if len(c0_tok) >= 3 and len(c1_tok) >= 4:
        # M195 test deck format: Card 0 (RHO, E, NU), Card 1 (A0, A1, B0, B1, ICRIT), Card 2 (E, NU), Card 3 (C, PHI, AMAX), Card 4 (PMIN)
        rho = _safe_float(c0_tok[0])
        e = _safe_float(c0_tok[1])
        nu = _safe_float(c0_tok[2])
        a0 = _safe_float(c1_tok[0])
        a1 = _safe_float(c1_tok[1])
        b0 = _safe_float(c1_tok[2]) if len(c1_tok) > 2 else 0.0
        b1 = _safe_float(c1_tok[3]) if len(c1_tok) > 3 else 0.0
        icrit = _safe_int(c1_tok[4], 1) if len(c1_tok) > 4 else 1
        iform = icrit
        if len(c2_tok) >= 2:
            e = _safe_float(c2_tok[0], e)
            nu = _safe_float(c2_tok[1], nu)
        if len(c3_tok) >= 2:
            c = _safe_float(c3_tok[0])
            phi = _safe_float(c3_tok[1])
            if len(c3_tok) > 2:
                amax = _safe_float(c3_tok[2], 1.0e30)
        if len(c4_tok) >= 1:
            pmin = _safe_float(c4_tok[0], -1.0e30)
    elif block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("MAT_LAW102_1")
            rho = _safe_float(c0[0]) if len(c0) > 0 else 0.0
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("MAT_LAW102_2")
            iform = _safe_int(c1[0], 2) if len(c1) > 0 else 2
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("MAT_LAW102_3")
            e = _safe_float(c2[0]) if len(c2) > 0 else 0.0
            nu = _safe_float(c2[1]) if len(c2) > 1 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("MAT_LAW102_4")
            c = _safe_float(c3[0]) if len(c3) > 0 else 0.0
            phi = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            if len(c3) > 2 and c3[2].strip():
                amax = _safe_float(c3[2], 1.0e30)
        if len(valid_cards) > 4:
            if len(c4_tok) >= 2 and amax == 1.0e30:
                amax = _safe_float(c4_tok[0], 1.0e30)
                pmin = _safe_float(c4_tok[1], -1.0e30)
            else:
                c4 = valid_cards[4].cut("MAT_LAW102_5")
                pmin = _safe_float(c4[0], -1.0e30) if len(c4) > 0 else -1.0e30
    else:
        # Free-format reading
        if len(valid_cards) > 0:
            rho = _safe_float(c0_tok[0]) if len(c0_tok) > 0 else 0.0
            if len(c0_tok) > 1 and e == 0.0:
                e = _safe_float(c0_tok[1])
            if len(c0_tok) > 2 and nu == 0.0:
                nu = _safe_float(c0_tok[2])
        if len(valid_cards) > 1:
            if len(c1_tok) == 1:
                iform = _safe_int(c1_tok[0], 2)
            elif len(c1_tok) >= 2 and e == 0.0:
                e = _safe_float(c1_tok[0])
                nu = _safe_float(c1_tok[1])
        if len(valid_cards) > 2 and (e == 0.0 and nu == 0.0):
            e = _safe_float(c2_tok[0]) if len(c2_tok) > 0 else 0.0
            nu = _safe_float(c2_tok[1]) if len(c2_tok) > 1 else 0.0
        if len(valid_cards) > 3:
            c = _safe_float(c3_tok[0]) if len(c3_tok) > 0 else 0.0
            phi = _safe_float(c3_tok[1]) if len(c3_tok) > 1 else 0.0
            if len(c3_tok) > 2:
                amax = _safe_float(c3_tok[2], 1.0e30)
        if len(valid_cards) > 4:
            if len(c4_tok) >= 2 and amax == 1.0e30:
                amax = _safe_float(c4_tok[0], 1.0e30)
                pmin = _safe_float(c4_tok[1], -1.0e30)
            elif len(c4_tok) > 0:
                pmin = _safe_float(c4_tok[0], -1.0e30)

    g, bulk, phi_rad, k_yield, alpha, a0, a1, a2, pstar, iform_sanitized = compute_dprag2_constants(
        e=e, nu=nu, c=c, phi_deg=phi, iform=iform, amax=amax, pmin=pmin
    )

    params = {
        "rho": rho,
        "iform": iform_sanitized,
        "e": e,
        "nu": nu,
        "c": c,
        "phi": phi,
        "amax": amax,
        "pmin": pmin,
        "a0": a0,
        "a1": a1,
        "a2": a2,
        "pstar": pstar,
        "g": g,
        "bulk": bulk,
    }
    mat = MatLaw102(
        id=mat_id,
        title=title,
        rho=rho,
        iform=iform_sanitized,
        e=e,
        nu=nu,
        c=c,
        phi=phi,
        amax=amax,
        pmin=pmin,
        a0=a0,
        a1=a1,
        a2=a2,
        params=params,
    )
    model.mat_law102s[mat_id] = mat
    model.materials[mat_id] = Material(id=mat_id, law=102, rho0=rho, title=title, params=params)





def read_mat_nlocal(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/NLOCAL`` (M194): Nonlocal plastic strain regularisation."""
    from ...model.entities import MatNLocal, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    rho, e, nu = 0.0, 0.0, 0.0
    params = {}
    if len(valid_cards) > 0:
        c0 = valid_cards[0].tokens()
        rho = _safe_float(c0[0]) if len(c0) > 0 else 0.0
        e = _safe_float(c0[1]) if len(c0) > 1 else 0.0
        nu = _safe_float(c0[2]) if len(c0) > 2 else 0.0
    if len(valid_cards) > 1:
        c1 = valid_cards[1].tokens()
        for idx, val in enumerate(c1):
            params[f"param_{idx}"] = _safe_float(val)
    mat = MatNLocal(id=mat_id, title=title, rho=rho, e=e, nu=nu, params=params)
    model.mat_nlocals[mat_id] = mat
    model.materials[mat_id] = Material(id=mat_id, law=0, rho0=rho, title=title, params={"rho": rho, "e": e, "nu": nu, **params})




def read_mat_law103(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW103`` or ``/MAT/HENSEL_SPITTEL`` (M195): Hensel-Spittel hot-forming material law."""
    from ...model.entities import MatLaw103, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    rho, refer_rho, e, nu = 0.0, 0.0, 0.0, 0.0
    a0, m1, m2, m3, m4 = 0.0, 0.0, 0.0, 0.0, 0.0
    m5, m7 = 0.0, 0.0
    fsmooth, fcut, eps_0, pmin = 0, 0.0, 0.0, -1.0e30
    rhocp, t0, eta = 0.0, 0.0, 0.0
    params = {}

    if len(valid_cards) > 0:
        c0 = valid_cards[0].tokens()
        rho = _safe_float(c0[0]) if len(c0) > 0 else 0.0
        refer_rho = _safe_float(c0[1]) if len(c0) > 1 else rho
    if len(valid_cards) > 1:
        c1 = valid_cards[1].tokens()
        e = _safe_float(c1[0]) if len(c1) > 0 else 0.0
        nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
    if len(valid_cards) > 2:
        c2 = valid_cards[2].tokens()
        a0 = _safe_float(c2[0]) if len(c2) > 0 else 0.0
        m1 = _safe_float(c2[1]) if len(c2) > 1 else 0.0
        m2 = _safe_float(c2[2]) if len(c2) > 2 else 0.0
        m3 = _safe_float(c2[3]) if len(c2) > 3 else 0.0
        m4 = _safe_float(c2[4]) if len(c2) > 4 else 0.0
    if len(valid_cards) > 3:
        c3 = valid_cards[3].tokens()
        m5 = _safe_float(c3[0]) if len(c3) > 0 else 0.0
        m7 = _safe_float(c3[1]) if len(c3) > 1 else 0.0
    if len(valid_cards) > 4:
        c4 = valid_cards[4].tokens()
        if len(c4) >= 5:
            fsmooth = _safe_int(c4[1])
            fcut = _safe_float(c4[2])
            eps_0 = _safe_float(c4[3])
            pmin = _safe_float(c4[4])
        else:
            fsmooth = _safe_int(c4[0]) if len(c4) > 0 else 0
            fcut = _safe_float(c4[1]) if len(c4) > 1 else 0.0
            eps_0 = _safe_float(c4[2]) if len(c4) > 2 else 0.0
            pmin = _safe_float(c4[3]) if len(c4) > 3 else -1.0e30
    if len(valid_cards) > 5:
        c5 = valid_cards[5].tokens()
        rhocp = _safe_float(c5[0]) if len(c5) > 0 else 0.0
        t0 = _safe_float(c5[1]) if len(c5) > 1 else 0.0
        eta = _safe_float(c5[2]) if len(c5) > 2 else 0.0

    # Defaults matching hm_read_mat103.F
    if pmin == 0.0:
        pmin = -1.0e30
    if rhocp == 0.0:
        rhocp = 1.0e30
    if refer_rho == 0.0:
        refer_rho = rho
    eta = min(1.0, eta)

    params.update({
        "rho": rho, "refer_rho": refer_rho, "rhor": refer_rho, "e": e, "nu": nu,
        "a0": a0, "m1": m1, "m2": m2, "m3": m3, "m4": m4, "m5": m5, "m7": m7,
        "fsmooth": fsmooth, "fcut": fcut, "eps_0": eps_0, "eps0": eps_0, "pmin": pmin,
        "rhocp": rhocp, "rcp": rhocp, "t0": t0, "eta": eta
    })
    mat = MatLaw103(
        id=mat_id, title=title, rho=rho, refer_rho=refer_rho, e=e, nu=nu,
        a0=a0, m1=m1, m2=m2, m3=m3, m4=m4, m5=m5, m7=m7,
        fsmooth=fsmooth, fcut=fcut, eps_0=eps_0, pmin=pmin,
        rhocp=rhocp, t0=t0, eta=eta, params=params
    )
    model.mat_law103s[mat_id] = mat
    model.materials[mat_id] = Material(id=mat_id, law=103, rho0=rho, title=title, params=params)




read_mat_hensel_spittel = read_mat_law103


read_mat_plas_hens = read_mat_law103




def read_mat_law104(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW104``, ``/MAT/JOHNS_VOCE_DRUCKER``, ``/MAT/DRUCKER``, ``/MAT/PLAS_DRUCK`` (M176/M574).

    Combined Drucker yield criterion, Voce hardening, Johnson-Cook rate sensitivity,
    and Taylor-Quinney self-heating.
    Fortran origin: ``starter/source/materials/mat/mat104/hm_read_mat104.F``.
    """
    from ...model.entities import MaterialLaw104, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]

    rho0, refer_rho = 0.0, 0.0
    young, nu = 0.0, 0.0
    ires = 1
    sigma0_yld, h, q_voce, b_voce, c_dr = 1.0e30, 0.0, 0.0, 0.0, 0.0
    c_jc, eps0, fcut = 0.0, 1.0, 10000.0
    tss, tref, tini = 0.0, 0.0, 0.0
    eta, cp, eps_iso, eps_ad = 0.0, 0.0, 1.0e30, 2.0e30
    params = {}

    card_idx = 0
    # Card 1: MAT_RHO (and optional refer_rho)
    if card_idx < len(valid_cards):
        c0 = valid_cards[card_idx].tokens()
        rho0 = _safe_float(c0[0]) if len(c0) > 0 else 0.0
        refer_rho = _safe_float(c0[1]) if len(c0) > 1 else rho0
        card_idx += 1

    # Card 2: MAT_E, MAT_NU, MAT104_Ires
    if card_idx < len(valid_cards):
        c1 = valid_cards[card_idx].tokens()
        young = _safe_float(c1[0]) if len(c1) > 0 else 0.0
        nu = _safe_float(c1[1]) if len(c1) > 1 else 0.0
        ires = _safe_int(c1[2]) if len(c1) > 2 else 1
        card_idx += 1

    # Card 3: SIGMA_r, MAT104_H, MAT_PR, MAT104_Bv, MAT104_Cdr
    if card_idx < len(valid_cards):
        c2 = valid_cards[card_idx].tokens()
        sigma0_yld = _safe_float(c2[0]) if len(c2) > 0 else 1.0e30
        h = _safe_float(c2[1]) if len(c2) > 1 else 0.0
        q_voce = _safe_float(c2[2]) if len(c2) > 2 else 0.0
        b_voce = _safe_float(c2[3]) if len(c2) > 3 else 0.0
        c_dr = _safe_float(c2[4]) if len(c2) > 4 else 0.0
        card_idx += 1

    # Card 4: MAT104_Cjc, MAT104_Eps0, MAT104_Fcut
    if card_idx < len(valid_cards):
        c3 = valid_cards[card_idx].tokens()
        c_jc = _safe_float(c3[0]) if len(c3) > 0 else 0.0
        eps0 = _safe_float(c3[1]) if len(c3) > 1 else 1.0
        fcut = _safe_float(c3[2]) if len(c3) > 2 else 10000.0
        card_idx += 1

    # Card 5: MAT104_Tss, MAT104_Tref, T_Initial
    if card_idx < len(valid_cards):
        c4 = valid_cards[card_idx].tokens()
        tss = _safe_float(c4[0]) if len(c4) > 0 else 0.0
        tref = _safe_float(c4[1]) if len(c4) > 1 else 0.0
        tini = _safe_float(c4[2]) if len(c4) > 2 else 0.0
        card_idx += 1

    # Card 6: MAT_ETA, MAT_SPHEAT, MAT104_EpsIso, MAT104_EpsAd
    if card_idx < len(valid_cards):
        c5 = valid_cards[card_idx].tokens()
        eta = _safe_float(c5[0]) if len(c5) > 0 else 0.0
        cp = _safe_float(c5[1]) if len(c5) > 1 else 0.0
        eps_iso = _safe_float(c5[2]) if len(c5) > 2 else 1.0e30
        eps_ad = _safe_float(c5[3]) if len(c5) > 3 else 2.0e30
        card_idx += 1

    # Apply defaults matching hm_read_mat104.F
    if sigma0_yld == 0.0:
        sigma0_yld = 1.0e30
    if fcut == 0.0:
        fcut = 10000.0
    if eps0 == 0.0:
        eps0 = 1.0
        c_jc = 0.0
    if eps_iso == 0.0:
        eps_iso = 1.0e30
    if eps_ad == 0.0:
        eps_ad = 2.0e30
    if ires == 0:
        ires = 1
    if refer_rho == 0.0:
        refer_rho = rho0

    params.update({
        "rho": rho0, "rho0": rho0, "refer_rho": refer_rho, "rhor": refer_rho,
        "young": young, "e": young, "nu": nu, "ires": ires,
        "sigma0_yld": sigma0_yld, "yld0": sigma0_yld, "sigy": sigma0_yld,
        "h": h, "hp": h,
        "q_voce": q_voce, "qvoce": q_voce, "qv": q_voce,
        "b_voce": b_voce, "bvoce": b_voce, "bv": b_voce,
        "c_dr": c_dr, "cdr": c_dr,
        "c_jc": c_jc, "cjc": c_jc,
        "eps0": eps0, "epsp0": eps0,
        "fcut": fcut,
        "tss": tss, "mu": tss, "mtemp": tss,
        "tref": tref,
        "tini": tini, "t0": tini,
        "eta": eta,
        "cp": cp, "rhocp": rho0 * cp if cp > 0.0 else 0.0,
        "eps_iso": eps_iso, "dpis": eps_iso,
        "eps_ad": eps_ad, "dpad": eps_ad,
    })

    mat = MaterialLaw104(
        id=mat_id, title=title, rho0=rho0, refer_rho=refer_rho,
        young=young, nu=nu, ires=ires,
        sigma_r=sigma0_yld, h=h, qv=q_voce, bv=b_voce, cdr=c_dr,
        cjc=c_jc, epsp0=eps0, fcut=fcut,
        tss=tss, tref=tref, tini=tini,
        eta=eta, cp=cp, eps_iso=eps_iso, eps_ad=eps_ad,
        params=params, law=104, law_name="LAW104"
    )
    model.mat_law104s[mat_id] = mat
    from ..mat_reader import GenericMaterialRecord
    mat_entity = Material(id=mat_id, law=104, rho0=rho0, title=title, params=params)
    params["MAT_PR"] = q_voce
    mat_entity.record = GenericMaterialRecord(
        law_name="LAW104", law_number=104, id=mat_id, title=title,
        params=params, density=rho0, unit_id=block.unit_id,
    )
    model.materials[mat_id] = mat_entity




read_mat_drucker = read_mat_law104


read_mat_johns_voce_drucker = read_mat_law104


read_mat_plas_druck = read_mat_law104




def read_mat_law108(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW108`` or ``/MAT/SPR_GENE`` (M195): Generalized 6-DOF nonlinear spring material."""
    from ...model.entities import MatLaw108, Material
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    rho = 0.0
    ifail, iequil, ifail2 = 0, 0, 0
    k = [0.0] * 6
    c = [0.0] * 6
    a = [0.0] * 6
    b = [0.0] * 6
    d = [0.0] * 6
    fct_id1 = [0] * 6
    h = [0.0] * 6
    fct_id2 = [0] * 6
    fct_id3 = [0] * 6
    fct_id4 = [0] * 6
    delta_min = [0.0] * 6
    delta_max = [0.0] * 6
    f_val = [0.0] * 6
    e_val = [0.0] * 6
    ascale = [1.0] * 6
    hscale = [1.0] * 6
    fsmooth = 0
    fcut = 0.0
    params = {}

    card_idx = 0
    if card_idx < len(valid_cards):
        c0 = valid_cards[card_idx].tokens()
        rho = _safe_float(c0[0]) if len(c0) > 0 else 0.0
        card_idx += 1
    if card_idx < len(valid_cards):
        c1 = valid_cards[card_idx].tokens()
        ifail = _safe_int(c1[0]) if len(c1) > 0 else 0
        iequil = _safe_int(c1[1]) if len(c1) > 1 else 0
        ifail2 = _safe_int(c1[2]) if len(c1) > 2 else 0
        card_idx += 1

    for dof in range(6):
        if card_idx < len(valid_cards):
            cd1 = valid_cards[card_idx].tokens()
            k[dof] = _safe_float(cd1[0]) if len(cd1) > 0 else 0.0
            c[dof] = _safe_float(cd1[1]) if len(cd1) > 1 else 0.0
            a[dof] = _safe_float(cd1[2]) if len(cd1) > 2 else 0.0
            b[dof] = _safe_float(cd1[3]) if len(cd1) > 3 else 0.0
            d[dof] = _safe_float(cd1[4]) if len(cd1) > 4 else 0.0
            card_idx += 1
        if card_idx < len(valid_cards):
            cd2 = valid_cards[card_idx].tokens()
            fct_id1[dof] = _safe_int(cd2[0]) if len(cd2) > 0 else 0
            h[dof] = _safe_float(cd2[1]) if len(cd2) > 1 else 0.0
            fct_id2[dof] = _safe_int(cd2[2]) if len(cd2) > 2 else 0
            fct_id3[dof] = _safe_int(cd2[3]) if len(cd2) > 3 else 0
            fct_id4[dof] = _safe_int(cd2[4]) if len(cd2) > 4 else 0
            delta_min[dof] = _safe_float(cd2[5]) if len(cd2) > 5 else 0.0
            delta_max[dof] = _safe_float(cd2[6]) if len(cd2) > 6 else 0.0
            card_idx += 1
        if card_idx < len(valid_cards):
            cd3 = valid_cards[card_idx].tokens()
            f_val[dof] = _safe_float(cd3[0]) if len(cd3) > 0 else 0.0
            e_val[dof] = _safe_float(cd3[1]) if len(cd3) > 1 else 0.0
            ascale[dof] = _safe_float(cd3[2]) if len(cd3) > 2 else 1.0
            hscale[dof] = _safe_float(cd3[3]) if len(cd3) > 3 else 1.0
            card_idx += 1

    if card_idx < len(valid_cards):
        clast = valid_cards[card_idx].tokens()
        fsmooth = _safe_int(clast[0]) if len(clast) > 0 else 0
        fcut = _safe_float(clast[1]) if len(clast) > 1 else 0.0

    params.update({
        "rho": rho, "ifail": ifail, "iequil": iequil, "ifail2": ifail2,
        "k": k, "c": c, "a": a, "b": b, "d": d,
        "fct_id1": fct_id1, "h": h, "fct_id2": fct_id2, "fct_id3": fct_id3, "fct_id4": fct_id4,
        "delta_min": delta_min, "delta_max": delta_max,
        "f_val": f_val, "e_val": e_val, "ascale": ascale, "hscale": hscale,
        "fsmooth": fsmooth, "fcut": fcut
    })
    mat = MatLaw108(
        id=mat_id, title=title, rho=rho, ifail=ifail, iequil=iequil, ifail2=ifail2,
        k=k, c=c, a=a, b=b, d=d,
        fct_id1=fct_id1, h=h, fct_id2=fct_id2, fct_id3=fct_id3, fct_id4=fct_id4,
        delta_min=delta_min, delta_max=delta_max, f_val=f_val, e_val=e_val, ascale=ascale, hscale=hscale,
        fsmooth=fsmooth, fcut=fcut, params=params
    )
    model.mat_law108s[mat_id] = mat
    model.materials[mat_id] = Material(id=mat_id, law=108, rho0=rho, title=title, params=params)




def read_mat_plas_predef(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/PLAS_PREDEF`` (M195): Predefined plasticity material model."""
    from ...model.entities import MatPlasPredef
    from ..mat_reader import InactiveMaterial
    mat_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    mat_name = ""
    rho, e, nu, sigy, uts, e_uts = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    epsp_f, vp, c_val, p_val, n_val = 0.0, 0.0, 0.0, 0.0, 0
    params = {}

    PREDEFINED_MATS = {
        "STEEL": (7.8e-6, 210.0, 0.3, 0.160, 0.380, 0.24),
        "HSS": (7.8e-6, 210.0, 0.3, 0.300, 0.510, 0.23),
        "UHSS": (7.8e-6, 210.0, 0.3, 0.500, 1.500, 0.045),
        "AA5182": (2.7e-6, 70.0, 0.33, 0.150, 0.300, 0.25),
        "AA6082-T6": (2.7e-6, 70.0, 0.33, 0.300, 0.360, 0.08),
        "PA6GF30": (1.3e-6, 7.0, 0.35, 0.050, 0.100, 0.02),
        "PPT40": (1.2e-6, 4.0, 0.3, 0.020, 0.030, 0.06),
    }

    if len(valid_cards) > 0:
        c0 = valid_cards[0].tokens()
        if len(c0) > 0:
            token0 = c0[0].strip().upper()
            if token0 in PREDEFINED_MATS:
                mat_name = token0
                rho, e, nu, sigy, uts, e_uts = PREDEFINED_MATS[token0]
            else:
                rho = _safe_float(c0[0])
                if len(c0) > 1:
                    e = _safe_float(c0[1])
                if len(c0) > 2:
                    nu = _safe_float(c0[2])

    if len(valid_cards) > 1 and not mat_name:
        c1 = valid_cards[1].tokens()
        sigy = _safe_float(c1[0]) if len(c1) > 0 else 0.0
        uts = _safe_float(c1[1]) if len(c1) > 1 else 0.0
        e_uts = _safe_float(c1[2]) if len(c1) > 2 else 0.0
        epsp_f = _safe_float(c1[3]) if len(c1) > 3 else 0.0
        vp = _safe_float(c1[4]) if len(c1) > 4 else 0.0
        c_val = _safe_float(c1[5]) if len(c1) > 5 else 0.0
        p_val = _safe_float(c1[6]) if len(c1) > 6 else 0.0
        n_val = _safe_int(c1[7]) if len(c1) > 7 else 0

    params.update({
        "mat_name": mat_name, "Material_Name_Str": mat_name, "rho": rho, "e": e, "nu": nu,
        "E": e, "Nu": nu,
        "sigy": sigy, "uts": uts, "e_uts": e_uts, "epsp_f": epsp_f,
        "vp": vp, "c": c_val, "p": p_val, "n": n_val
    })
    mat = MatPlasPredef(
        id=mat_id, title=title, mat_name=mat_name, rho=rho, e=e, nu=nu,
        sigy=sigy, uts=uts, e_uts=e_uts, epsp_f=epsp_f, vp=vp, c=c_val, p=p_val, n=n_val, params=params
    )
    model.mat_plas_predefs[mat_id] = mat
    model.materials[mat_id] = InactiveMaterial(
        id=mat_id, law=3, rho0=rho, title=title,
        law_name="PLAS_PREDEF",
        params=params,
    )




read_mat_dprag2 = read_mat_law102
