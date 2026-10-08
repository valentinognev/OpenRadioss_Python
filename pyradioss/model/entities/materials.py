"""Constitutive material laws, equations of state and failure models.

Split out of the former single-module ``pyradioss/model/entities.py`` (task
P1.4).  The blocks below are byte-identical to their originals, with one
mechanical exception: relative ``from . import`` statements gained a dot,
because code that was one module deep now sits one package level deeper.
Nothing else about any block changed.

This module is self-contained: the split was checked and leaves no cross-file
reference of any kind, so no sibling submodule is imported here.  A future
cross-file reference should be an ``if TYPE_CHECKING:`` import (annotations
are lazy here, via ``from __future__ import annotations``) unless the
reference is actually evaluated at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import math

import numpy as np
class CallableFloat(float):
    """Float that is also callable returning itself (compatible with both property and method access)."""
    def __call__(self) -> float:
        return float(self)


@dataclass
class FailureModel:
    """One /FAIL option, attached to a material (the keyword carries the
    material id: ``/FAIL/JOHNSON/mat_ID``).

    Fortran origin: the FAIL_PARAM structures of ``fail_param_mod.F``
    filled by ``starter/source/materials/fail/hm_read_fail*.F``.

    Attributes
    ----------
    type     : 'JOHNSON' | 'BIQUAD' (see pyradioss.failure)
    params   : criterion constants (D1..D4 / c1..c5 + derived fits)
    ifail_sh : shell deletion rule — 1 = delete when ONE integration
               layer is broken (default), 2 = when ALL layers are broken
    """

    type: str
    params: Dict[str, float] = field(default_factory=dict)
    ifail_sh: int = 1


@dataclass
class EquationOfState:
    """One /EOS option, attached to a material like /FAIL is (the keyword
    carries the material id: ``/EOS/POLYNOMIAL/mat_ID``).

    Fortran origin: the EOS_PARAM structures filled by
    ``starter/source/materials/eos/hm_read_eos.F``. Ported kinds:

    * ``POLYNOMIAL`` — p = C0 + C1 mu + C2 mubar^2 + C3 mu^3
      + (C4 + C5 mu) E  (params c0..c5, e0 = initial energy per V0);
    * ``IDEAL-GAS``   — stored as the equivalent polynomial
      (C4 = C5 = gamma - 1, e0 = P0/(gamma - 1)).

    ``rho0`` is copied from the host material at resolve time (the sound
    speed needs it). See pyradioss/materials/eos.py for the theory.
    """

    kind: str
    params: Dict[str, float] = field(default_factory=dict)
    rho0: float = 0.0


@dataclass
class Material:
    """One /MAT law. Only the fields common to all laws live here; law
    parameters are in ``params``, interpreted by the material kernel.

    Attributes
    ----------
    id, title : user id and title
    law       : integer law number (1 = elastic, 2 = Johnson-Cook,
                27 = brittle, 36 = tabulated, 42 = Ogden, ...)
    rho0      : initial density (PM(1) 'RHO0' in the Fortran)
    params    : law-specific constants, e.g. E, nu, A, B, n, c, eps0...
                (for LAW42 the parse stores the DERIVED E from
                G0 = sum(mu_p*alpha_p)/2 and nu, so the generic elastic
                properties below work for every law)
    fail      : optional /FAIL criterion attached to this material
    eos       : optional /EOS attached to this material (M6): the EOS
                pressure replaces the law's own for solid elements
    """

    id: int
    law: int
    rho0: float = 0.0
    title: str = ""
    params: Dict[str, float] = field(default_factory=dict)
    fail: Optional[FailureModel] = None
    eos: Optional["EquationOfState"] = None
    fail_models: List[Any] = field(default_factory=list)
    fm_type: Optional[str] = None
    law_name: Optional[str] = None

    # Convenience elastic constants (every implemented law defines these;
    # they drive the sound speed / time step and contact stiffness).
    @property
    def E(self) -> float:
        if self.law in (50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM") or getattr(self, "law_name", None) in ("50", "LAW50", "VISC_HONEY", "HYP_FOAM"):
            ea = float(self.params.get("ea", self.params.get("MAT_EA", self.params.get("e11", self.params.get("E11", 0.0)))))
            eb = float(self.params.get("eb", self.params.get("MAT_EB", self.params.get("e22", self.params.get("E22", 0.0)))))
            ec = float(self.params.get("ec", self.params.get("MAT_EC", self.params.get("e33", self.params.get("E33", 0.0)))))
            return max(ea, eb, ec)
        if "E" in self.params:
            return self.params["E"]
        if "Young" in self.params:
            return self.params["Young"]
        if "E0" in self.params:
            return self.params["E0"]
        if "e" in self.params:
            return float(self.params["e"])
        if "e0" in self.params:
            return float(self.params["e0"])
        if "MAT_E" in self.params:
            return float(self.params["MAT_E"])
        if "e1" in self.params:
            return float(self.params["e1"])
        if "MAT_E1" in self.params:
            return float(self.params["MAT_E1"])
        g = self.G
        if g > 0.0:
            return 2.0 * g * (1.0 + self.nu)
        return 0.0

    @property
    def nu(self) -> float:
        if "nu" in self.params:
            return self.params["nu"]
        if "Nu" in self.params:
            return self.params["Nu"]
        if "MAT_NU" in self.params:
            return self.params["MAT_NU"]
        if "nu_t" in self.params:
            return float(self.params["nu_t"])
        if "nu_c" in self.params:
            return float(self.params["nu_c"])
        return 0.3

    @property
    def G(self) -> float:
        """Shear modulus G = E / 2(1+nu)."""
        if self.law in (50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM") or getattr(self, "law_name", None) in ("50", "LAW50", "VISC_HONEY", "HYP_FOAM"):
            gab = float(self.params.get("gab", self.params.get("MAT_GAB", self.params.get("g12", self.params.get("G12", 0.0)))))
            gbc = float(self.params.get("gbc", self.params.get("MAT_GBC", self.params.get("g23", self.params.get("G23", 0.0)))))
            gca = float(self.params.get("gca", self.params.get("MAT_GCA", self.params.get("g31", self.params.get("G31", 0.0)))))
            return max(gab, gbc, gca)
        if "G" in self.params:
            return float(self.params["G"])
        if "mu" in self.params:
            mu = self.params["mu"]
            if isinstance(mu, (int, float)):
                return float(mu)
            # LAW42 Ogden: mu is a list of Ogden coefficients —
            # net shear modulus G0 = sum(mu_p * alpha_p) / 2
            if isinstance(mu, (list, tuple)) and "alpha" in self.params:
                return float(sum(m * a for m, a in
                                 zip(mu, self.params["alpha"])) / 2.0)
            # fall through to E-based computation
        if "Mu_arr" in self.params:
            mu_arr = self.params["Mu_arr"]
            if isinstance(mu_arr, (list, tuple)) and len(mu_arr) > 0:
                return float(sum(mu_arr))
        if "Mu" in self.params:
            return float(self.params["Mu"])
        if "c10" in self.params:
            return 2.0 * float(self.params["c10"])
        if "E" in self.params:
            return float(self.params["E"]) / (2.0 * (1.0 + self.nu))
        if "e" in self.params:
            return float(self.params["e"]) / (2.0 * (1.0 + self.nu))
        if "e0" in self.params:
            return float(self.params["e0"]) / (2.0 * (1.0 + self.nu))
        if "MAT_E" in self.params:
            return float(self.params["MAT_E"]) / (2.0 * (1.0 + self.nu))
        if "g5" in self.params and float(self.params["g5"]) > 0.0:
            return float(self.params["g5"])
        if "MAT_G5" in self.params and float(self.params["MAT_G5"]) > 0.0:
            return float(self.params["MAT_G5"])
        if "g0" in self.params and float(self.params["g0"]) > 0.0:
            return float(self.params["g0"])
        if "MAT_G0" in self.params and float(self.params["MAT_G0"]) > 0.0:
            return float(self.params["MAT_G0"])
        return 0.0

    @property
    def K(self) -> float:
        """Bulk modulus K = E / 3(1-2nu)."""
        if self.law in (82, "82", "LAW82", "OGDEN", "MAT_LAW82", "MAT_OGDEN"):
            d_arr = self.params.get("Gamma_arr", self.params.get("D_arr", []))
            if d_arr and len(d_arr) > 0 and float(d_arr[0]) > 0:
                return 2.0 / float(d_arr[0])
            gs = self.G
            nu = self.nu
            denom = 3.0 * (1.0 - 2.0 * nu)
            if abs(denom) > 1e-12:
                return 2.0 * gs * (1.0 + nu) / denom
        if self.law in (5, "5", "LAW5", "JWL"):
            return float(self.params.get("c1", self.params.get("bulk", 0.0)))
        if self.law in (38, "38", "LAW38", "VISC_TAB"):
            nu_max = min(0.499, max(self.params.get("nu_t", 0.0), self.params.get("nu_c", 0.0), 0.0))
            return self.E / (3.0 * (1.0 - 2.0 * nu_max))
        if "K" in self.params:
            return self.params["K"]
        if "bulk" in self.params:
            return self.params["bulk"]
        if "Bulk" in self.params:
            return self.params["Bulk"]
        if "d" in self.params and self.params["d"] > 0 and self.law not in (5, "5", "LAW5", "JWL"):
            return 2.0 / self.params["d"]
        denom = 3.0 * (1.0 - 2.0 * self.nu)
        if abs(denom) < 1e-12:
            denom = 1e-6
        return self.E / denom

    def sound_speed_solid(self) -> float:
        """3-D dilatational wave speed c = sqrt((K + 4G/3)/rho).

        This is the speed of the fastest (P-) wave in the continuum and is
        what the Courant stability condition of an explicit solid element
        must resolve (Fortran: computed in each material's ini routine,
        e.g. starter/source/materials/mat/mat001/ini... -> PM(27) 'SSP').
        """
        if self.law in (5, "5", "LAW5", "JWL"):
            try:
                from ...materials import law05_jwl
                return float(law05_jwl.sound_speed(self, rho=self.rho0))
            except Exception:
                return float(self.params.get("d", self.params.get("vdet", 0.0)))
        if self.law in (28, "28", "LAW28", "HONEYCOMB", "HONEYCOMB_SOL"):
            try:
                from ...materials import law28_honeycomb
                return float(law28_honeycomb.sound_speed(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS"):
            try:
                from ...materials import law25_composite
                return float(law25_composite.sound_speed(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (38, "38", "LAW38", "VISC_TAB"):
            try:
                from ...materials import law38_visc_tab
                return float(law38_visc_tab.sound_speed(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (12, "12", "LAW12", "3D_COMP", "COMP_3D", "3PARBI", "RAGAB"):
            if "ssp" in self.params:
                return float(self.params["ssp"])
            if "SSP" in self.params:
                return float(self.params["SSP"])
            try:
                from ...materials import law12_comp3d
                return float(law12_comp3d.sound_speed(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (14, "14", "LAW14", "COMPSO", "COMP_SOL"):
            if "ssp" in self.params:
                return float(self.params["ssp"])
            if "SSP" in self.params:
                return float(self.params["SSP"])
            try:
                from ...materials import law14_compso
                return float(law14_compso.sound_speed(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (82, "82", "LAW82", "OGDEN", "MAT_LAW82", "MAT_OGDEN"):
            try:
                from ...materials import law82_ogden
                return float(law82_ogden.sound_speed(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "MAT_LAW69", "MAT_HYP_ELAS", "MAT_HYPERELASTIC", "LAW69_HYPERELASTIC"):
            try:
                from ...materials import law69_hyperelastic
                return float(law69_hyperelastic.sound_speed(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_LAW48", "MAT_ZHAO", "LAW48_ZHAO") or getattr(self, "law_name", None) in ("48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_LAW48", "MAT_ZHAO", "LAW48_ZHAO"):
            try:
                from ...materials import law48_zhao
                return float(law48_zhao.sound_speed_solid_law48(self, rho0=self.rho0))
            except Exception:
                pass
        if self.law in (52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS") or getattr(self, "law_name", None) in ("52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS"):
            try:
                from ...materials import law52_gurson
                return float(law52_gurson.sound_speed_solid_law52(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (21, "21", "LAW21", "MAT_LAW21", "MAT_DPRAG", "DPRAG") or getattr(self, "law_name", None) in ("21", "LAW21", "MAT_LAW21", "MAT_DPRAG", "DPRAG"):
            try:
                from ...materials import law21_dprag
                return float(law21_dprag.sound_speed(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (49, "49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB") or getattr(self, "law_name", None) in ("49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB"):
            try:
                from ...materials import law49_steinb
                return float(law49_steinb.sound_speed_solid(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (79, "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM") or getattr(self, "law_name", None) in ("79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM"):
            try:
                from ...materials import law79_john_holm
                return float(law79_john_holm.sound_speed(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM") or getattr(self, "law_name", None) in ("50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM"):
            try:
                from ...materials import law50_visc_honey
                return float(law50_visc_honey.sound_speed_solid(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB") or getattr(self, "law_name", None) in ("66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB"):
            try:
                from ...materials import law66_plas_tab
                return float(law66_plas_tab.sound_speed_solid(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (93, "93", "LAW93", "ORTH_HILL", "MAT_LAW93", "MAT_ORTH_HILL", "LAW93_ORTH_HILL") or getattr(self, "law_name", None) in ("93", "LAW93", "ORTH_HILL", "MAT_LAW93", "MAT_ORTH_HILL", "LAW93_ORTH_HILL"):
            try:
                from ...materials import law93_orth_hill
                return float(law93_orth_hill.sound_speed(self, self.rho0))
            except Exception:
                pass
        if self.law in (95, "95", "LAW95", "BERGSTROM_BOYCE", "BERGSTROM-BOYCE", "MAT_LAW95", "MAT_BERGSTROM_BOYCE", "LAW95_BERGSTROM_BOYCE") or getattr(self, "law_name", None) in ("95", "LAW95", "BERGSTROM_BOYCE", "BERGSTROM-BOYCE", "MAT_LAW95", "MAT_BERGSTROM_BOYCE", "LAW95_BERGSTROM_BOYCE"):
            try:
                from ...materials import law95_bergstrom_boyce
                return float(law95_bergstrom_boyce.sound_speed(self, self.rho0))
            except Exception:
                pass
        if self.law in (100, "100", "LAW100", "VISC_HYP", "MNF") or getattr(self, "law_name", None) in ("100", "LAW100", "VISC_HYP", "MNF", "MAT_LAW100", "MAT_VISC_HYP", "MAT_MNF", "LAW100_VISC_HYP", "LAW100_MNF"):
            try:
                from ...materials import law100_multi_network
                return float(law100_multi_network.sound_speed(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (101, "101", "LAW101", "PP", "PLAS_POLY") or getattr(self, "law_name", None) in ("101", "LAW101", "PP", "MAT_PP", "PLAS_POLY", "MAT_PLAS_POLY", "MAT_LAW101", "LAW101_PP", "LAW101_PLAS_POLY"):
            try:
                from ...materials import law101_plas_poly
                return float(law101_plas_poly.sound_speed(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (102, "102", "LAW102", "DPRAG2") or getattr(self, "law_name", None) in ("102", "LAW102", "DPRAG2", "MAT_DPRAG2", "MAT_LAW102", "LAW102_DPRAG2", "DRUCKER_PRAGER_2"):
            try:
                from ...materials import law102_dprag2
                return float(law102_dprag2.sound_speed(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (103, "103", "LAW103", "HENSEL_SPITTEL", "HENSEL-SPITTEL", "PLAS_HENS") or getattr(self, "law_name", None) in ("103", "LAW103", "HENSEL_SPITTEL", "HENSEL-SPITTEL", "PLAS_HENS", "MAT_HENSEL_SPITTEL", "MAT_PLAS_HENS", "MAT_103", "MAT_LAW103", "HEN"):
            try:
                from ...materials import law103_hensel_spittel
                return float(law103_hensel_spittel.sound_speed(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (104, "104", "LAW104", "DRUCKER", "JOHNS_VOCE_DRUCKER", "PLAS_DRUCK") or getattr(self, "law_name", None) in ("104", "LAW104", "DRUCKER", "JOHNS_VOCE_DRUCKER", "JOHNS-VOCE-DRUCKER", "PLAS_DRUCK", "MAT_DRUCKER", "MAT_JOHNS_VOCE_DRUCKER", "MAT_PLAS_DRUCK", "MAT_104", "MAT_LAW104", "LAW104_DRUCKER"):
            try:
                from ...materials import law104_drucker
                return float(law104_drucker.sound_speed_solid(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (105, "105", "LAW105", "POWDER_BURN", "POWDERBURN") or getattr(self, "law_name", None) in ("105", "LAW105", "POWDER_BURN", "POWDERBURN", "MAT_POWDER_BURN", "MAT_POWDERBURN", "MAT_105", "MAT_LAW105", "LAW105_POWDER_BURN"):
            try:
                from ...materials import law105_powder_burn
                return float(law105_powder_burn.sound_speed(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (106, "106", "LAW106", "JCOOK_ALM", "JOHNS_COOK_ALM") or getattr(self, "law_name", None) in ("106", "LAW106", "JCOOK_ALM", "JOHNS_COOK_ALM", "MAT_JCOOK_ALM", "MAT_106", "MAT_LAW106", "LAW106_JCOOK_ALM"):
            try:
                from ...materials import law106_jcook_alm
                return float(law106_jcook_alm.sound_speed_solid(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (107, "107", "LAW107", "PAPER_LIGHT", "PLAS_PAPER_LIGHT") or getattr(self, "law_name", None) in ("107", "LAW107", "PAPER_LIGHT", "PLAS_PAPER_LIGHT", "LAW107_PAPER_LIGHT", "PFEIFFER", "MAT_PFEIFFER", "MAT_PAPER_LIGHT", "MAT_PLAS_PAPER_LIGHT", "MAT_107", "MAT_LAW107"):
            try:
                from ...materials import law107_paper_light
                return float(law107_paper_light.sound_speed_solid(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (109, "109", "LAW109", "TAB_PLAS", "ELASTO_PLAS_TAB") or getattr(self, "law_name", None) in ("109", "LAW109", "TAB_PLAS", "ELASTO_PLAS_TAB", "LAW109_TAB_PLAS", "MLAW109", "MAT_LAW109", "MAT_TAB_PLAS", "MAT_ELASTO_PLAS_TAB", "MAT_109"):
            try:
                from ...materials import law109_tab_plas
                return float(law109_tab_plas.sound_speed(self, rho=self.rho0, is_shell=False))
            except Exception:
                pass
        if self.rho0 <= 0.0:
            return 0.0
        return float(np.sqrt(max(0.0, (self.K + 4.0 * self.G / 3.0) / self.rho0)))

    def sound_speed_shell(self) -> float:
        """Plane-stress wave speed c = sqrt(E / (rho (1 - nu^2))).

        Shells use the plane-stress modulus because the through-thickness
        stress is zero, which softens the response relative to 3-D.
        """
        if self.law in (15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG"):
            try:
                from ...materials import law15_chang
                return float(law15_chang.sound_speed(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS"):
            try:
                from ...materials import law25_composite
                return float(law25_composite.sound_speed(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (82, "82", "LAW82", "OGDEN", "MAT_LAW82", "MAT_OGDEN"):
            try:
                from ...materials import law82_ogden
                return float(law82_ogden.sound_speed_shell(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "MAT_LAW69", "MAT_HYP_ELAS", "MAT_HYPERELASTIC", "LAW69_HYPERELASTIC"):
            try:
                from ...materials import law69_hyperelastic
                return float(law69_hyperelastic.sound_speed_shell(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_LAW48", "MAT_ZHAO", "LAW48_ZHAO") or getattr(self, "law_name", None) in ("48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_LAW48", "MAT_ZHAO", "LAW48_ZHAO"):
            try:
                from ...materials import law48_zhao
                return float(law48_zhao.sound_speed_shell_law48(self, rho0=self.rho0))
            except Exception:
                pass
        if self.law in (52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS") or getattr(self, "law_name", None) in ("52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS"):
            try:
                from ...materials import law52_gurson
                return float(law52_gurson.sound_speed_shell_law52(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (58, "58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A") or getattr(self, "law_name", None) in ("58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A"):
            try:
                from ...materials import law58_fabr_a
                return float(law58_fabr_a.sound_speed_shell_law58(self, rho0=self.rho0))
            except Exception:
                pass
        if self.law in (57, "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3") or getattr(self, "law_name", None) in ("57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3"):
            try:
                from ...materials import law57_barlat
                return float(law57_barlat.sound_speed_shell_law57(self, rho0=self.rho0))
            except Exception:
                pass
        if self.law in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL") or getattr(self, "law_name", None) in ("73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL"):
            try:
                from ...materials import law73_hill_therm
                return float(law73_hill_therm.sound_speed(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000") or getattr(self, "law_name", None) in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000"):
            E = float(self.params.get("E", self.params.get("e", 0.0))) if isinstance(self.params, dict) else 0.0
            nu = float(self.params.get("Nu", self.params.get("nu", 0.0))) if isinstance(self.params, dict) else 0.0
            rho = float(self.rho0) if self.rho0 > 0.0 else (float(self.params.get("rho", 0.0)) if isinstance(self.params, dict) else 0.0)
            if rho > 0.0 and E > 0.0 and (1.0 - nu**2) > 0.0:
                import math
                return float(math.sqrt(E / (rho * (1.0 - nu**2))))
        if self.law in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB") or getattr(self, "law_name", None) in ("66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB"):
            try:
                from ...materials import law66_plas_tab
                return float(law66_plas_tab.sound_speed_shell(self, rho=self.rho0))
            except Exception:
                pass
            rho0 = self.rho0 or (self.params.get("rho", 0.0) if hasattr(self, "params") and isinstance(self.params, dict) else 0.0)
            if rho0 > 0.0:
                p = getattr(self, "params", {}) or {}
                nc = max(p.get("n1", p.get("n1_warp", 1)) or 1, 1) if isinstance(p, dict) else 1
                nt = max(p.get("n2", p.get("n2_weft", 1)) or 1, 1) if isinstance(p, dict) else 1
                e1 = p.get("e1", 0.0) or 0.0 if isinstance(p, dict) else 0.0
                e2 = p.get("e2", 0.0) or 0.0 if isinstance(p, dict) else 0.0
                g0 = p.get("g0", 0.0) or 0.0 if isinstance(p, dict) else 0.0
                kc = e1 / nc
                kt = e2 / nt
                young = max(kc, kt, g0)
                if young <= 0.0:
                    young = max(e1, e2)
                if young > 0.0:
                    return float(np.sqrt(young / rho0))
        if self.law in (93, "93", "LAW93", "ORTH_HILL", "MAT_LAW93", "MAT_ORTH_HILL", "LAW93_ORTH_HILL") or getattr(self, "law_name", None) in ("93", "LAW93", "ORTH_HILL", "MAT_LAW93", "MAT_ORTH_HILL", "LAW93_ORTH_HILL"):
            try:
                from ...materials import law93_orth_hill
                return float(law93_orth_hill.sound_speed_shell(self, self.rho0))
            except Exception:
                pass
        if self.law in (104, "104", "LAW104", "DRUCKER", "JOHNS_VOCE_DRUCKER", "PLAS_DRUCK") or getattr(self, "law_name", None) in ("104", "LAW104", "DRUCKER", "JOHNS_VOCE_DRUCKER", "JOHNS-VOCE-DRUCKER", "PLAS_DRUCK", "MAT_DRUCKER", "MAT_JOHNS_VOCE_DRUCKER", "MAT_PLAS_DRUCK", "MAT_104", "MAT_LAW104", "LAW104_DRUCKER"):
            try:
                from ...materials import law104_drucker
                return float(law104_drucker.sound_speed_shell(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (106, "106", "LAW106", "JCOOK_ALM", "JOHNS_COOK_ALM") or getattr(self, "law_name", None) in ("106", "LAW106", "JCOOK_ALM", "JOHNS_COOK_ALM", "MAT_JCOOK_ALM", "MAT_106", "MAT_LAW106", "LAW106_JCOOK_ALM"):
            try:
                from ...materials import law106_jcook_alm
                return float(law106_jcook_alm.sound_speed_shell(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (107, "107", "LAW107", "PAPER_LIGHT", "PLAS_PAPER_LIGHT") or getattr(self, "law_name", None) in ("107", "LAW107", "PAPER_LIGHT", "PLAS_PAPER_LIGHT", "LAW107_PAPER_LIGHT", "PFEIFFER", "MAT_PFEIFFER", "MAT_PAPER_LIGHT", "MAT_PLAS_PAPER_LIGHT", "MAT_107", "MAT_LAW107"):
            try:
                from ...materials import law107_paper_light
                return float(law107_paper_light.sound_speed_shell(self, rho=self.rho0))
            except Exception:
                pass
        if self.law in (109, "109", "LAW109", "TAB_PLAS", "ELASTO_PLAS_TAB") or getattr(self, "law_name", None) in ("109", "LAW109", "TAB_PLAS", "ELASTO_PLAS_TAB", "LAW109_TAB_PLAS", "MLAW109", "MAT_LAW109", "MAT_TAB_PLAS", "MAT_ELASTO_PLAS_TAB", "MAT_109"):
            try:
                from ...materials import law109_tab_plas
                return float(law109_tab_plas.sound_speed(self, rho=self.rho0, is_shell=True))
            except Exception:
                pass
        if self.law in (110, "110", "LAW110", "VEGTER", "PLAS_VEGTER") or getattr(self, "law_name", None) in ("110", "LAW110", "VEGTER", "PLAS_VEGTER", "LAW110_VEGTER", "MLAW110", "MAT_LAW110", "MAT_VEGTER", "MAT_PLAS_VEGTER", "MAT_110"):
            try:
                from ...materials import law110_vegter
                return float(law110_vegter.sound_speed(self, rho=self.rho0, is_shell=True))
            except Exception:
                pass
        if self.rho0 <= 0.0 or (1.0 - self.nu ** 2) <= 0.0:
            return 0.0
        return float(np.sqrt(max(0.0, self.E / (self.rho0 * (1.0 - self.nu ** 2)))))

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        if "sound_speed_solid" in state and callable(state["sound_speed_solid"]):
            del state["sound_speed_solid"]
        if "sound_speed_shell" in state and callable(state["sound_speed_shell"]):
            del state["sound_speed_shell"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)


@dataclass
class FailurePerturbation:
    """/PERTURB/FAIL/BIQUAD/perturb_ID (M101): Failure parameter perturbation.

    Fortran origin: ``starter/source/general_controls/computation/hm_read_perturb_fail.F``.
    """
    id: int
    title: str = ""
    fail_id: int = 0
    parameter: str = "C3"
    fail_type: str = "BIQUAD"
    f_mean: float = 0.0
    deviation: float = 0.0
    min_cut: float = 0.0
    max_cut: float = 0.0
    seed: int = 0
    idistri: int = 2


@dataclass
class FailComposite:
    """/FAIL/COMPOSITE (M114): 3D anisotropic composite failure model.

    Fortran origin: ``starter/source/materials/failure/fail_composite.F``.
    """
    mat_id: int
    sig_1t: float = 0.0
    sig_1c: float = 0.0
    sig_2t: float = 0.0
    sig_2c: float = 0.0
    sig_12: float = 0.0
    sig_3t: float = 0.0
    sig_3c: float = 0.0
    sig_23: float = 0.0
    sig_31: float = 0.0
    beta: float = 0.0
    tau_max: float = 0.0
    expn: float = 0.0
    ifail_sh: int = 0
    ifail_so: int = 0
    fail_id: int = 0


@dataclass
class FailFractal:
    """/FAIL/FRACTAL_DMG or /FAIL/FRACTAL (M118): Fractal damage failure model."""
    mat_id: int
    grsh4n_1: int = 0
    grsh3n_1: int = 0
    grsh4n_2: int = 0
    grsh3n_2: int = 0
    damage: float = 0.0
    probability: float = 0.0
    seed: int = 0
    num_walk: int = 0
    printout: int = 0
    fail_id: int = 0


@dataclass
class FailOrthBiquad:
    """/FAIL/ORTHBIQUAD (M123): Orthotropic Biquadratic failure model for shells."""
    id: int
    mat_id: int = 0
    p_thickfail: float = 1.0
    m_flag: int = 0
    s_flag: int = 0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    inst_start: float = 0.0
    eps_dot0: float = 0.0
    c_jc: float = 0.0
    fct_id_rate: int = 0
    fct_id_el: int = 0
    ei_ref: float = 0.0
    r1: float = 1.0
    r2: float = 1.0
    r4: float = 1.0
    r5: float = 1.0
    title: str = ""



@dataclass
class FailRtcl:
    """/FAIL/RTCL: RTCL ductile failure model.

    Fortran origin: ``starter/source/materials/fail/fail_rtcl.F`` / CFG ``fail_rtcl.cfg``.
    """
    mat_id: int = 0
    title: str = ""
    epscal: float = 0.0
    inst: int = 0
    n: float = 0.0
    fail_id: int = 0
    ifail_sh: int = 1            # Shell element deletion flag

    @property
    def n_exp(self) -> float:
        """Alias: hardening exponent N (M259 name)."""
        return self.n

    @n_exp.setter
    def n_exp(self, value: float) -> None:
        self.n = value


@dataclass
class FailGurson:
    """/FAIL/GURSON: Gurson-Tvergaard-Needleman porous plasticity failure model.

    Fortran origin: ``starter/source/materials/fail/fail_gurson.F`` / CFG ``fail_gurson.cfg``.
    """
    mat_id: int = 0
    q1: float = 0.0
    q2: float = 0.0
    iloc: int = 1
    eps_n: float = 0.0
    a_s: float = 0.0
    k_w: float = 0.0
    f_c: float = 0.0
    f_r: float = 0.0
    f_0: float = 0.0
    r_len: float = 0.0
    h_chi: float = 0.0
    le_max: float = 0.0
    fail_id: int = 0
    f_u: float = 0.0
    s_n: float = 0.0
    f_n: float = 0.0
    ifail_sh: int = 1
    title: str = ""

    @property
    def i_loc(self) -> int:
        """Alias: damage formulation flag (M263 name)."""
        return self.iloc

    @i_loc.setter
    def i_loc(self, value: int) -> None:
        self.iloc = value


@dataclass
class FailPuck:
    """/FAIL/PUCK (M126/M189): Puck composite failure model.

    Fortran origin: ``starter/source/materials/fail/fail_puck.F`` / CFG ``fail_puck.cfg``.
    """
    id: int = 0
    mat_id: int = 0
    sigma_1t: float = 0.0
    sigma_2t: float = 0.0
    sigma_12: float = 0.0
    sigma_1c: float = 0.0
    sigma_2c: float = 0.0
    p12_pos: float = 0.0
    p12_neg: float = 0.0
    p22_neg: float = 0.0
    tau_max: float = 0.0
    ifail_sh: int = 1
    ifail_so: int = 1
    fcut: float = 0.0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailSahraei:
    """/FAIL/SAHRAEI: Sahraei battery cell and separator failure criterion.

    Fortran origin: ``starter/source/materials/fail/fail_sahraei.F`` / CFG ``fail_sahraei.cfg``.
    """
    mat_id: int = 0
    title: str = ""
    fct_ratio: int = 0
    num: int = 1                 # numerator strain component flag
    den: int = 1                 # denominator strain component flag
    ordi: int = 1                # failure ordinate component flag
    vol_strain: float = 0.0
    fct_elsize: int = 0
    el_ref: float = 0.0
    comp_dir: int = 0
    idel: int = 0
    max_comp_strain: float = 1e30
    ratio: float = 1.0
    fail_id: int = 0             # M126 backward-compat field
    ifail_sh: int = 1            # shell element deletion flag


@dataclass
class FailSyazwan:
    """/FAIL/SYAZWAN: Syazwan fracture and damage failure model.

    Fortran origin: ``starter/source/materials/fail/fail_syazwan.F`` / CFG ``fail_syazwan.cfg``.
    """
    mat_id: int = 0
    title: str = ""
    icard: int = 0
    epfmin: float = 0.0
    coeffs: List[float] = field(default_factory=list)
    fail_id: int = 0
    id: int = 0
    failip: int = 0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    c6: float = 0.0
    epf_comp: float = 0.0
    epf_shear: float = 0.0
    epf_tens: float = 0.0
    epf_plstrn: float = 0.0
    epf_biax: float = 0.0
    dinit: int = 0
    dam_sf: float = 0.0
    max_dam: float = 1.0
    inst: int = 0
    iform: int = 0
    n_val: float = 0.0
    softexp: float = 0.0
    reg_func: int = 0
    ref_len: float = 0.0
    reg_scale: float = 1.0
    ifail_sh: int = 1


@dataclass
class FailTab2:
    """/FAIL/TAB2 (M126): Tabulated failure model Version 2.

    Fortran origin: ``starter/source/materials/fail/fail_tab2.F`` / CFG ``fail_tab2.cfg``.
    """
    mat_id: int
    epsf_id: int = 0
    fcrit: float = 0.0
    failip: int = 0
    pthk: float = 0.0
    n: float = 0.0
    dcrit: float = 0.0
    inst_id: int = 0
    ecrit: float = 0.0
    fct_exp: int = 0
    exp_ref: float = 0.0
    exp: float = 0.0
    fail_id: int = 0


@dataclass
class FailGene1:
    """/FAIL/GENE1 (M126/M193): General multi-criteria failure model.

    Fortran origin: ``starter/source/materials/fail/fail_gene1.F`` / CFG ``fail_gene1.cfg``.
    """
    mat_id: int = 0
    pmin: float = 0.0
    pmax: float = 0.0
    sigp1_max: float = 0.0
    tmax: float = 0.0
    time_max: float = 0.0
    dtmin: float = 0.0
    fct_idsm: int = 0
    eps_dot_sm: float = 0.0
    sig_max: float = 0.0
    sigr: float = 0.0
    kf: float = 0.0
    k: float = 0.0
    fct_idps: int = 0
    eps_dot_ps: float = 0.0
    eps_max: float = 0.0
    eps_eff: float = 0.0
    eps_vol: float = 0.0
    eps_min: float = 0.0
    eps_sh: float = 0.0
    fct_idg12: int = 0
    fct_idg13: int = 0
    fct_ide1c: int = 0
    tab_idfld: int = 0
    itab: int = 0
    eps_dot_fld: float = 0.0
    nstep: int = 0
    ismooth: int = 0
    istrain: int = 0
    thinning: float = 0.0
    volfrac: float = 0.0
    pthk: float = 0.0
    ncs: int = 0
    temp_max: float = 0.0
    failip: int = 0
    fct_idel: int = 0
    fscale_el: float = 1.0
    el_ref: float = 0.0
    fail_id: int = 0
    title: str = ""


@dataclass
class MaterialPlasZeril:
    """/MAT/PLAS_ZERIL (M141): Zerilli-Armstrong plasticity modifier.

    Fortran origin: ``starter/source/materials/mat/matl2_plas_zeril.F``.
    """
    mat_id: int
    title: str = ""
    c0: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    n: float = 0.0
    fcut: float = 0.0


@dataclass
class MaterialPlasBodne:
    """/MAT/PLAS_BODNE (M141): Bodner-Partom viscoplasticity modifier.

    Fortran origin: ``starter/source/materials/mat/matl2_plas_bodne.F``.
    """
    mat_id: int
    title: str = ""
    z0: float = 0.0
    z1: float = 0.0
    m: float = 0.0
    n: float = 0.0
    d0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0


@dataclass
class MaterialViscProny:
    """/MAT/VISC_PRONY or /VISC/LPRONY (M141): Viscoelastic Prony relaxation series.

    Fortran origin: ``starter/source/materials/mat/mat_VISC_LPRONY.F``.
    """
    mat_id: int
    title: str = ""
    order: int = 0
    form: int = 0
    flag_visc: int = 0
    gammas: List[float] = field(default_factory=list)
    taus: List[float] = field(default_factory=list)


@dataclass
class MaterialThermStress:
    """/MAT/THERM_STRESS (M141): Thermal stress expansion modifier.

    Fortran origin: ``starter/source/materials/mat/mat_therm_stress.F``.
    """
    mat_id: int
    title: str = ""
    alpha: float = 0.0
    t0: float = 293.15
    alpha_y: float = 0.0
    alpha_z: float = 0.0


@dataclass
class MaterialViscPlas:
    """/MAT/VISC_PLAS or /VISC/PLAS (M147): Frequency independent damping model.

    Fortran origin: ``starter/source/materials/visc/hm_read_visc_plas.F90``.
    """
    mat_id: int
    title: str = ""
    lsd_g: float = 0.0
    lsdyna_sigf: float = 0.0


@dataclass
class FailNxt:
    """/FAIL/NXT (M159): Strain-rate dependent failure model.

    Fortran origin: ``starter/source/materials/fail/hm_read_fail_nxt.F`` / CFG ``fail_nxt.cfg``.
    """
    mat_id: int
    fct_id1: int = 0
    fct_id2: int = 0
    ifail_sh: int = 1
    fail_id: int = 0


@dataclass
class FailLadDama:
    """/FAIL/LAD_DAMA (M159/M189): Ladevèze damage failure model.

    Fortran origin: ``starter/source/materials/fail/hm_read_fail_lad_dama.F`` / CFG ``fail_lad_dama.cfg``.
    """
    id: int = 0
    mat_id: int = 0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    gamma1: float = 0.0
    gamma2: float = 0.0
    y0: float = 0.0
    yc: float = 0.0
    k: float = 0.0
    k_lad: float = 0.0
    a: float = 0.0
    a_dama: float = 0.0
    tau_max: float = 0.0
    ifail_sh: int = 1
    ifail_so: int = 1
    fail_id: int = 0
    title: str = ""


FailLadeveze = FailLadDama


@dataclass
class FailInievo:
    """/FAIL/INIEVO (M159): Multi-criterion damage initiation & evolution model.

    Fortran origin: ``starter/source/materials/fail/hm_read_fail_inievo.F`` / CFG ``fail_inievo.cfg``.
    """
    mat_id: int
    ninievo: int = 1
    ishear: int = 0
    ilen: int = 0
    failip: int = 0
    pthk: float = 0.0
    models: List[Dict] = field(default_factory=list)
    fail_id: int = 0


@dataclass
class MaterialSprSeatbelt:
    """/MAT/LAW114 or /MAT/SPR_SEATBELT (M161): Seatbelt spring material model.

    Fortran origin: ``starter/source/materials/mat/hm_read_mat114.F`` / CFG ``mat114_spr_seatbelt.cfg``.
    """
    id: int
    title: str = ""
    rho: float = 0.0
    lmin: float = 0.0
    k: float = 0.0
    c: float = 0.0
    fun_l: int = 0
    fun_ul: int = 0
    xscale: float = 1.0
    fscale: float = 1.0
    e: float = 0.0
    i: float = 0.0
    j: float = 0.0
    fmax: float = 0.0
    mmax: float = 0.0
    as_: float = 0.0
    r: float = 0.0
    law: int = 114
    rho0: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.rho0 and self.rho:
            self.rho0 = self.rho
        elif not self.rho and self.rho0:
            self.rho = self.rho0
        self.params = {
            "rho": self.rho, "rho0": self.rho0, "lmin": self.lmin, "k": self.k, "stiff1": self.k,
            "c": self.c, "damp1": self.c, "fun_l": self.fun_l, "fun_ul": self.fun_ul,
            "xscale": self.xscale, "fscale": self.fscale, "e": self.e, "E": self.e,
            "i": self.i, "j": self.j, "fmax": self.fmax, "mmax": self.mmax,
            "as": self.as_, "r": self.r
        }


@dataclass
class MaterialShSeatbelt:
    """/MAT/LAW119 or /MAT/SH_SEATBELT (M161): 2D shell seatbelt fabric material model.

    Fortran origin: ``starter/source/materials/mat/hm_read_mat119.F`` / CFG ``mat119_sh_seatbelt.cfg``.
    """
    id: int
    title: str = ""
    rho: float = 0.0
    lmin: float = 0.0
    k: float = 0.0
    c: float = 0.0
    re: float = 0.0
    fun_l: int = 0
    fun_ul: int = 0
    fscale1: float = 1.0
    fscale2: float = 1.0
    ireload: int = 0
    e22: float = 0.0
    nu12: float = 0.0
    g12: float = 0.0
    fscale22: float = 1.0
    ecoat: float = 0.0
    nucoat: float = 0.0
    tcoat: float = 0.0
    law: int = 119
    rho0: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.rho0 and self.rho:
            self.rho0 = self.rho
        elif not self.rho and self.rho0:
            self.rho = self.rho0
        self.params = {
            "rho": self.rho, "rho0": self.rho0, "lmin": self.lmin, "k": self.k, "stiff1": self.k,
            "c": self.c, "damp1": self.c, "re": self.re, "fun_l": self.fun_l, "fun_ul": self.fun_ul,
            "fscale1": self.fscale1, "fscale2": self.fscale2, "ireload": self.ireload,
            "e22": self.e22, "nu12": self.nu12, "g12": self.g12, "fscale22": self.fscale22,
            "ecoat": self.ecoat, "nucoat": self.nucoat, "tcoat": self.tcoat
        }


@dataclass
class MaterialTapo:
    """/MAT/LAW120 or /MAT/TAPO (M161): Tape/woven fabric material model with plasticity and damage.

    Fortran origin: ``starter/source/materials/mat/hm_read_mat120.F`` / CFG ``mat120_tapo.cfg``.
    """
    id: int
    title: str = ""
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    iform: int = 1
    itrx: int = 0
    idam: int = 0
    thick: float = 0.0
    tab_id: int = 0
    xscale: float = 1.0
    yscale: float = 1.0
    tau: float = 0.0
    q: float = 0.0
    beta: float = 1.0
    h: float = 0.0
    af1: float = 0.0
    af2: float = 0.0
    ah1: float = 0.0
    ah2: float = 0.0
    as_: float = 0.0
    cc: float = 1e21
    gam0: float = 0.0
    gamf: float = 0.0
    d1c: float = 0.0
    d2c: float = 0.0
    d1f: float = 0.0
    d2f: float = 0.0
    d_trx: float = 0.0
    d_jc: float = 0.0
    exp_n: float = 0.0
    law: int = 120
    rho0: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.rho0 and self.rho:
            self.rho0 = self.rho
        elif not self.rho and self.rho0:
            self.rho = self.rho0
        self.params = {
            "rho": self.rho, "rho0": self.rho0, "refer_rho": self.refer_rho, "e": self.e, "E": self.e,
            "nu": self.nu, "iform": self.iform, "itrx": self.itrx, "idam": self.idam,
            "thick": self.thick, "tab_id": self.tab_id, "xscale": self.xscale, "yscale": self.yscale,
            "tau": self.tau, "q": self.q, "beta": self.beta, "h": self.h,
            "af1": self.af1, "af2": self.af2, "ah1": self.ah1, "ah2": self.ah2, "as": self.as_,
            "cc": self.cc, "gam0": self.gam0, "gamf": self.gamf,
            "d1c": self.d1c, "d2c": self.d2c, "d1f": self.d1f, "d2f": self.d2f,
            "d_trx": self.d_trx, "d_jc": self.d_jc, "exp_n": self.exp_n
        }


@dataclass
class MaterialPlasRate:
    """/MAT/LAW121 or /MAT/PLAS_RATE (M161): Strain-rate dependent elastoplastic material model.

    Fortran origin: ``starter/source/materials/mat/hm_read_mat121.F`` / CFG ``matl121_plasrate.cfg``.
    """
    id: int
    title: str = ""
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    ires: int = 0
    ivisc: int = 0
    fcut: float = 0.0
    tdel: float = 0.0
    fct_sig0: int = 0
    xscale_sig0: float = 1.0
    yscale_sig0: float = 1.0
    fct_youn: int = 0
    xscale_youn: float = 1.0
    yscale_youn: float = 1.0
    fct_tang: int = 0
    xscale_tang: float = 1.0
    tang: float = 0.0
    fct_fail: int = 0
    ifail: int = 0
    xscale_fail: float = 1.0
    yscale_fail: float = 1.0
    law: int = 121
    rho0: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.rho0 and self.rho:
            self.rho0 = self.rho
        elif not self.rho and self.rho0:
            self.rho = self.rho0
        self.params = {
            "rho": self.rho, "rho0": self.rho0, "e": self.e, "E": self.e, "nu": self.nu,
            "ires": self.ires, "ivisc": self.ivisc, "fcut": self.fcut, "tdel": self.tdel,
            "fct_sig0": self.fct_sig0, "xscale_sig0": self.xscale_sig0, "yscale_sig0": self.yscale_sig0,
            "fct_youn": self.fct_youn, "xscale_youn": self.xscale_youn, "yscale_youn": self.yscale_youn,
            "fct_tang": self.fct_tang, "xscale_tang": self.xscale_tang, "tang": self.tang,
            "fct_fail": self.fct_fail, "ifail": self.ifail, "xscale_fail": self.xscale_fail, "yscale_fail": self.yscale_fail
        }


@dataclass
class MaterialCdpm2:
    """/MAT/LAW124 or /MAT/CDPM2 (M161): Concrete damage plasticity model 2.

    Fortran origin: ``starter/source/materials/mat/hm_read_mat124.F`` / CFG ``matl124_cdpm2.cfg``.
    """
    id: int
    title: str = ""
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    irate: int = 0
    fcut: float = 0.0
    ecc: float = 0.0
    qh0: float = 0.0
    ft: float = 0.0
    fc: float = 0.0
    hp: float = 0.0
    ah: float = 0.0
    bh: float = 0.0
    ch: float = 0.0
    dh: float = 0.0
    as_: float = 0.0
    bs: float = 0.0
    df: float = 0.0
    dflag: int = 0
    dtype: int = 0
    ireg: int = 0
    wf: float = 0.0
    wf1: float = 0.0
    ft1: float = 0.0
    efc: float = 0.0
    law: int = 124
    rho0: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.rho0 and self.rho:
            self.rho0 = self.rho
        elif not self.rho and self.rho0:
            self.rho = self.rho0
        self.params = {
            "rho": self.rho, "rho0": self.rho0, "e": self.e, "E": self.e, "nu": self.nu,
            "irate": self.irate, "fcut": self.fcut, "ecc": self.ecc, "qh0": self.qh0,
            "ft": self.ft, "fc": self.fc, "hp": self.hp, "ah": self.ah, "bh": self.bh,
            "ch": self.ch, "dh": self.dh, "as": self.as_, "bs": self.bs, "df": self.df,
            "dflag": self.dflag, "dtype": self.dtype, "ireg": self.ireg,
            "wf": self.wf, "wf1": self.wf1, "ft1": self.ft1, "efc": self.efc
        }


@dataclass
class FailHcDsse:
    """/FAIL/HC_DSSE (M162): Hosford-Coulomb & DSSE ductile failure and fracture model.

    Fortran origin: ``starter/source/materials/fail/hc_dsse/hm_read_fail_hc_dsse.F``.
    """
    mat_id: int
    ifail_sh: int = 1
    pthkf: float = 0.0
    iflag: int = 0
    a_hc_dsse: float = 0.0
    b_hc_dsse: float = 0.0
    c_hc_dsse: float = 0.0
    d_hc_dsse: float = 0.0
    n_f: float = 1.0
    fail_id: int = 0


@dataclass
class FailMullins:
    """/FAIL/MULLINS_OR or /FAIL/MULLINS (M162): Mullins effect hyperelastic damage model.

    Fortran origin: ``starter/source/materials/fail/mullins_or/hm_read_fail_mullins_or.F``.
    """
    mat_id: int
    coefr: float = 1.0
    beta: float = 0.0
    coefm: float = 0.0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailSnconnect:
    """/FAIL/SNCONNECT (M162): S-N curve based connector fatigue failure model.

    Fortran origin: ``starter/source/materials/fail/snconnect/hm_read_fail_snconnect.F``.
    """
    mat_id: int
    alpha_0: float = 0.0
    beta_0: float = 0.0
    alpha_f: float = 0.0
    beta_f: float = 0.0
    ifail_so: int = 0
    isym: int = 0
    fct_idon: int = 0
    fct_idos: int = 0
    fct_idfn: int = 0
    fct_idfs: int = 0
    xscale_0: float = 1.0
    xscale_f: float = 1.0
    area_scale: float = 1.0
    fail_id: int = 0


@dataclass
class FailSpalling:
    """/FAIL/SPALLING (M162/M189): Spalling / hydrodynamic tensile cutoff failure model.

    Fortran origin: ``starter/source/materials/fail/spalling/hm_read_fail_spalling.F90``.
    """
    id: int = 0
    mat_id: int = 0
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0
    d4: float = 0.0
    d5: float = 0.0
    eps_dot_0: float = 1.0e-20
    epsilon_dot_0: float = 1.0e-20
    p_min: float = -1.0e20
    ifail_so: int = 1
    fail_id: int = 0
    title: str = ""


FailSpall = FailSpalling


@dataclass
class MaterialConc:
    """/MAT/LAW24 or /MAT/CONC (M170): Concrete material model.

    Fortran origin: ``starter/source/materials/mat24/hm_read_mat24.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    e_c: float = 0.0
    nu: float = 0.0
    f_c: float = 0.0
    ft_on_fc: float = 0.0
    fb_on_fc: float = 0.0
    f2_on_fc: float = 0.0
    s0_on_fc: float = 0.0
    h_t: float = 0.0
    d_sup: float = 0.0
    eps_max: float = 0.0
    k_y: float = 0.0
    r_t: float = 0.0
    r_c: float = 0.0
    h_bp: float = 0.0
    alpha_y: float = 0.0
    alpha_f: float = 0.0
    v_max: float = 0.0
    f_k: float = 0.0
    f0: float = 0.0
    h_v0: float = 0.0
    e2: float = 0.0
    ssig: float = 0.0
    setan: float = 0.0
    alpha1: float = 0.0
    alpha2: float = 0.0
    alpha3: float = 0.0


@dataclass
class MaterialBarlat:
    """/MAT/LAW87 or /MAT/BARLAT (M170): Barlat 2000 anisotropic plasticity model.

    Fortran origin: ``starter/source/materials/mat87/hm_read_mat87.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    iflag: int = 0
    vflag: int = 0
    strain1: float = 0.0
    exp1: float = 0.0
    ifit: int = 0
    alphas: List[float] = field(default_factory=lambda: [1.0] * 8)
    sigma_00: float = 0.0
    sigma_45: float = 0.0
    sigma_90: float = 0.0
    sigma_b: float = 0.0
    r_00: float = 0.0
    r_45: float = 0.0
    r_90: float = 0.0
    r_b: float = 0.0
    a_exp: int = 6
    alpha_vol: float = 1.0
    n_hard: float = 0.0
    fcut: float = 0.0
    fsmooth: int = 0
    a_swift: float = 0.0
    eps0: float = 0.0
    q_voce: float = 0.0
    beta: float = 0.0
    k0: float = 0.0


@dataclass
class MaterialLaw83:
    """/MAT/LAW83 or /MAT/SPR_JOU (M170): Non-linear spring/joint material model.

    Fortran origin: ``starter/source/materials/mat83/hm_read_mat83.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    imass: int = 0
    fun_a1: int = 0
    fscale11: float = 1.0
    fscale22: float = 1.0
    alpha: float = 0.0
    beta: float = 0.0
    rn: float = 0.0
    rs: float = 0.0
    fsmooth: int = 0
    fcut: float = 0.0
    fun_a2: int = 0
    fun_a3: int = 0
    fscale33: float = 1.0


@dataclass
class MaterialLaw80:
    """/MAT/LAW80 or /MAT/TRANSFO (M170): Metallurgical phase transformation steel model.

    Fortran origin: ``starter/source/materials/mat80/hm_read_mat80.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    fct_ide: int = 0
    scale_e: float = 1.0
    time_unit: float = 3600.0
    fsmooth: int = 0
    fcut: float = 0.0
    ceps: float = 0.0
    peps: float = 0.0
    fun_a: List[int] = field(default_factory=lambda: [0] * 5)
    fscale_y: List[float] = field(default_factory=lambda: [1.0] * 5)
    scale_x: List[float] = field(default_factory=lambda: [1.0] * 5)
    theta: List[float] = field(default_factory=lambda: [0.0] * 4)
    alpha1: float = 0.0
    alpha2: float = 0.0


@dataclass
class MaterialLaw117:
    """/MAT/LAW117 or /MAT/COH_MC (M171): Cohesive element material model.

    Fortran origin: ``starter/source/materials/mat117/hm_read_mat117.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    e_elas_n: float = 0.0
    e_elas_s: float = 0.0
    imass: int = 0
    idel: int = 0
    irupt: int = 0
    fct_tn: int = 0
    fct_tt: int = 0
    tmax_n: float = 0.0
    tmax_s: float = 0.0
    fscale_x: float = 1.0
    gic: float = 0.0
    giic: float = 0.0
    exp_g: float = 1.0
    exp_bk: float = 1.0
    gamma: float = 0.0

    @property
    def rho(self) -> float:
        return self.rho0

    @property
    def en(self) -> float:
        return self.e_elas_n

    @property
    def es(self) -> float:
        return self.e_elas_s

    @property
    def tn(self) -> float:
        return self.tmax_n

    @property
    def ts(self) -> float:
        return self.tmax_s


@dataclass
class MaterialLaw90:
    """/MAT/LAW90 or /MAT/PLAS_TAB (M171): Strain-rate dependent tabular foam/plasticity material model.

    Fortran origin: ``starter/source/materials/mat90/hm_read_mat90.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    e0: float = 0.0
    nu: float = 0.0
    nl: int = 0
    ismooth: int = 0
    fcut: float = 0.0
    shape: float = 0.0
    hys: float = 0.0
    fct_ids: List[int] = field(default_factory=list)
    eps_dots: List[float] = field(default_factory=list)
    fscales: List[float] = field(default_factory=list)
    alpha: float = 1.0
    gamma: float = 1.0
    tflag: int = 1
    fail: int = 0
    econt: float = 0.0
    tcut: float = 1e20


@dataclass
class MaterialLaw33:
    """/MAT/LAW33 or /MAT/FOAM_PLAS (M171): Crushable foam plasticity material model.

    Fortran origin: ``starter/source/materials/mat33/hm_read_mat33.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    itype: int = 0
    fun_a1: int = 0
    ifscale: float = 1.0
    p0: float = 0.0
    phi: float = 0.0
    gama0: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    e1: float = 0.0
    e2: float = 0.0
    etan: float = 0.0
    eta1: float = 0.0
    eta2: float = 0.0


@dataclass
class MatHeatModifier:
    """/MAT/HEAT or /HEAT/MAT (M171): Material thermal property modifier.

    Fortran origin: ``starter/source/materials/heat/hm_read_heat.F``.
    """
    id: int
    mat_id: int = 0
    t0: float = 0.0
    rho0_cp: float = 0.0
    as_solid: float = 0.0
    bs_solid: float = 0.0
    t1: float = 1.0e30
    al_liquid: float = 0.0
    bl_liquid: float = 0.0
    efrac: float = 1.0


@dataclass
class MatNonlocalModifier:
    """/MAT/NONLOCAL or /NONLOCAL/MAT (M171): Non-local regularized damage material modifier.

    Fortran origin: ``starter/source/materials/nonlocal/hm_read_nonlocal.F``.
    """
    id: int
    mat_id: int = 0
    length: float = 0.0
    le_max: float = 0.0


# ----------------------------------------------------------------------------
# Advanced Tabular Foam, Viscoelastic Foam, Visco-Hyperelastic, Honeycomb & Cowper-Symonds Material Models (M172)
# ----------------------------------------------------------------------------

# MaterialLaw66 is defined as MatLaw66 (M562) below.


@dataclass
class MaterialLaw35:
    """/MAT/LAW35 or /MAT/FOAM_VISC (M172): Viscoelastic foam material model.

    Fortran origin: ``starter/source/materials/mat/mat035/hm_read_mat35.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ref_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    e1: float = 0.0
    e2: float = 0.0
    n: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    itype: int = 0
    pmin: float = 0.0
    func_idf: int = 0
    fscalepres: float = 1.0
    fsmooth: int = 0
    fcut: float = 0.0
    et: float = 0.0
    nu_t: float = 0.0
    eta_0: float = 0.0
    lamda: float = 0.0
    p0: float = 0.0
    phi: float = 0.0
    gama0: float = 0.0


@dataclass
class MaterialLaw62:
    """/MAT/LAW62 or /MAT/VISC_HYP (M172): Viscoelastic hyperelastic Ogden material model.

    Fortran origin: ``starter/source/materials/mat/mat062/hm_read_mat62.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ref_rho: float = 0.0
    nu: float = 0.0
    order_n: int = 0
    order_m: int = 0
    mu_max: float = 0.0
    mu_arr: List[float] = field(default_factory=list)
    alpha_arr: List[float] = field(default_factory=list)
    gamma_arr: List[float] = field(default_factory=list)
    tau_arr: List[float] = field(default_factory=list)


@dataclass
class MaterialLaw28:
    """/MAT/LAW28 or /MAT/HONEYCOMB (M172): Orthotropic honeycomb crushable material model.

    Fortran origin: ``starter/source/materials/mat/mat028/hm_read_mat28.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ref_rho: float = 0.0
    e11: float = 0.0
    e22: float = 0.0
    e33: float = 0.0
    g12: float = 0.0
    g23: float = 0.0
    g31: float = 0.0
    fun_a1: int = 0
    fun_b1: int = 0
    fun_a2: int = 0
    gflag: int = 0
    fscale11: float = 1.0
    fscale22: float = 1.0
    fscale33: float = 1.0
    epsr1: float = 0.0
    epsr2: float = 0.0
    epsr3: float = 0.0
    fun_a3: int = 0
    fun_b3: int = 0
    fun_a4: int = 0
    vflag: int = 0
    fscale12: float = 1.0
    fscale23: float = 1.0
    fscale13: float = 1.0
    epsr4: float = 0.0
    epsr5: float = 0.0
    epsr6: float = 0.0

    @property
    def fun_id11(self) -> int:
        return self.fun_a1

    @property
    def fun_id22(self) -> int:
        return self.fun_b1

    @property
    def fun_id33(self) -> int:
        return self.fun_a2

    @property
    def eps_max11(self) -> float:
        return self.epsr1

    @property
    def eps_max22(self) -> float:
        return self.epsr2

    @property
    def eps_max33(self) -> float:
        return self.epsr3

    @property
    def fun_id12(self) -> int:
        return self.fun_a3

    @property
    def fun_id23(self) -> int:
        return self.fun_b3

    @property
    def fun_id31(self) -> int:
        return self.fun_a4

    @property
    def eps_max12(self) -> float:
        return self.epsr4

    @property
    def eps_max23(self) -> float:
        return self.epsr5

    @property
    def eps_max31(self) -> float:
        return self.epsr6


@dataclass
class MaterialLaw44:
    """/MAT/LAW44 or /MAT/COWPER_SYMONDS (M172): Cowper-Symonds strain-rate dependent elastoplastic material model.

    Fortran origin: ``starter/source/materials/mat/mat044/hm_read_mat44.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ref_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    iflag: int = 0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    hard: float = 0.0
    sig_max: float = 0.0
    src: float = 0.0
    sre: float = 0.0
    strflag: int = 0
    fsmooth: int = 0
    fcut: float = 0.0
    vflag: int = 0
    eps_max: float = 0.0
    eta1: float = 0.0
    eta2: float = 0.0
    yld_func: int = 0
    yld_scale: float = 1.0


@dataclass
class MaterialLaw88:
    """/MAT/LAW88 or /MAT/HYPER_ELAS or /MAT/TABULATED_HYPERELASTIC (M173/M565):
    Tabulated hyperelastic Ogden material model with strain-rate unloading and damage.

    Fortran origin: ``starter/source/materials/mat/mat088/hm_read_mat88.F90``,
                    ``engine/source/materials/mat/mat088/sigeps88.F90``.
    """
    id: int = 0
    title: str = ""
    rho0: float = 0.0
    ref_rho: float = 0.0
    nu: float = 0.495
    bulk: float = 0.0
    fcut: float = 0.0
    fsmooth: int = 0
    nl: int = 0
    ifunc_unload: int = 0
    fscale_unload: float = 1.0
    hys: float = 0.0
    shape: float = 1.0
    tension: int = 0
    rtype: int = 0
    func_load_list: list = field(default_factory=list)
    fscale_load_list: list = field(default_factory=list)
    fscale_load_card: list = field(default_factory=list)
    rate_load_list: list = field(default_factory=list)
    lamfit_list: list = field(default_factory=list)
    sgl: float = 0.0
    sw: float = 0.0
    st: float = 0.0
    g: float = 0.0
    sigf: float = 0.0
    kfail: float = 0.0
    gam1: float = 0.0
    gam2: float = 0.0
    eh: float = 0.0
    failip: int = 0
    beta: float = 0.0
    young: float = 0.0
    shear: float = 0.0
    law: int = 88
    law_name: str = "LAW88"
    params: dict = field(default_factory=dict)

    @property
    def rho(self) -> float:
        return self.rho0 if self.rho0 > 0.0 else self.ref_rho

    @rho.setter
    def rho(self, val: float) -> None:
        self.rho0 = float(val)

    @property
    def E(self) -> float:
        if self.young > 0.0:
            return self.young
        if self.bulk > 0.0 and self.nu < 0.5:
            return 3.0 * self.bulk * (1.0 - 2.0 * self.nu)
        return 0.0

    @E.setter
    def E(self, val: float) -> None:
        self.young = float(val)

    @property
    def G(self) -> float:
        if self.shear > 0.0:
            return self.shear
        if self.g > 0.0:
            return self.g
        if self.bulk > 0.0 and self.nu < 0.5:
            return (3.0 * self.bulk * (1.0 - 2.0 * self.nu)) / (2.0 * (1.0 + self.nu))
        return 0.0

    @G.setter
    def G(self, val: float) -> None:
        self.shear = float(val)
        self.g = float(val)

    @property
    def K(self) -> float:
        if self.bulk > 0.0:
            return self.bulk
        if self.young > 0.0 and self.nu < 0.5:
            return self.young / (3.0 * (1.0 - 2.0 * self.nu))
        return 0.0

    @K.setter
    def K(self, val: float) -> None:
        self.bulk = float(val)

    @property
    def sound_speed(self) -> CallableFloat:
        import math
        r = self.rho
        if r > 0.0:
            k_mod = self.K
            g_mod = self.G
            c2 = (k_mod + (4.0 / 3.0) * g_mod) / r
            if c2 > 0.0:
                return CallableFloat(math.sqrt(c2))
        return CallableFloat(0.0)

    @property
    def sound_speed_solid(self) -> CallableFloat:
        return self.sound_speed

    @property
    def sound_speed_shell(self) -> CallableFloat:
        import math
        r = self.rho
        if r > 0.0:
            e_mod = self.E
            nu_val = self.nu
            denom = (1.0 - nu_val * nu_val) * r
            if denom > 0.0 and e_mod > 0.0:
                return CallableFloat(math.sqrt(e_mod / denom))
        return self.sound_speed

    @property
    def curves(self) -> list:
        return self.func_load_list

    # Mapping protocol
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if isinstance(self.params, dict) and key in self.params:
            return self.params[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            if not isinstance(self.params, dict):
                self.params = {}
            self.params[key] = value

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or (isinstance(self.params, dict) and key in self.params)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> List[str]:
        base_keys = [
            "id", "title", "rho0", "ref_rho", "nu", "bulk", "fcut", "fsmooth", "nl",
            "ifunc_unload", "fscale_unload", "hys", "shape", "tension", "rtype",
            "func_load_list", "fscale_load_list", "rate_load_list", "lamfit_list",
            "sgl", "sw", "st", "g", "sigf", "kfail", "gam1", "gam2", "eh", "failip",
            "beta", "young", "shear", "law", "law_name", "rho", "E", "G", "K",
            "sound_speed", "sound_speed_solid", "sound_speed_shell", "curves",
        ]
        if isinstance(self.params, dict):
            for k in self.params:
                if k not in base_keys:
                    base_keys.append(k)
        return base_keys

    def values(self) -> List[Any]:
        return [self[k] for k in self.keys()]

    def items(self) -> List[tuple]:
        return [(k, self[k]) for k in self.keys()]

    def __len__(self) -> int:
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())


MatLaw88 = MaterialLaw88
MatTabulatedHyperelastic = MaterialLaw88
MatHyperElas = MaterialLaw88
MatTabHyp = MaterialLaw88


@dataclass
class MaterialLaw92:
    """/MAT/LAW92 or /MAT/ARRUDA_BOYCE (M173, M566): Arruda-Boyce 8-chain hyperelastic polymer model.

    Fortran origin: ``starter/source/materials/mat/mat092/hm_read_mat92.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ref_rho: float = 0.0
    mu: float = 0.0
    d: float = 0.0
    lam: float = 7.0
    itype: int = 1
    fct_id: int = 0
    nu: float = 0.0
    fscale: float = 1.0
    law: int = 92
    law_name: str = "LAW92"
    params: dict = field(default_factory=dict)

    @property
    def rho(self) -> float:
        return self.ref_rho if self.ref_rho > 0.0 else self.rho0

    @property
    def G0(self) -> float:
        mu_val = self.mu
        if mu_val <= 0.0:
            return 0.0
        lam_val = self.lam if self.lam > 0.0 else 7.0
        beta = 1.0 / (lam_val * lam_val)
        return mu_val * (
            1.0
            + 0.6 * beta
            + (99.0 / 175.0) * (beta ** 2)
            + (513.0 / 875.0) * (beta ** 3)
            + (42039.0 / 67375.0) * (beta ** 4)
        )

    @property
    def G(self) -> float:
        return self.G0

    @property
    def K(self) -> float:
        if self.d > 0.0:
            return 2.0 / self.d
        nu_val = self.nu if self.nu > 0.0 else 0.495
        if nu_val < 0.5:
            g0 = self.G0
            return (2.0 / 3.0) * (1.0 + nu_val) * g0 / (1.0 - 2.0 * nu_val)
        return 0.0

    @property
    def bulk(self) -> float:
        return self.K

    @property
    def E(self) -> float:
        k_val = self.K
        g_val = self.G0
        denom = 3.0 * k_val + g_val
        if denom > 1e-20:
            return (9.0 * k_val * g_val) / denom
        nu_val = self.nu if self.nu > 0.0 else 0.495
        return 2.0 * g_val * (1.0 + nu_val)

    @property
    def young(self) -> float:
        return self.E

    @property
    def sound_speed(self) -> CallableFloat:
        import math
        r = self.rho
        if r > 0.0:
            c2 = (self.K + (4.0 / 3.0) * self.G0) / r
            if c2 > 0.0:
                return CallableFloat(math.sqrt(c2))
        return CallableFloat(0.0)

    @property
    def sound_speed_solid(self) -> CallableFloat:
        return self.sound_speed

    @property
    def sound_speed_shell(self) -> CallableFloat:
        import math
        r = self.rho
        if r > 0.0:
            nu_val = self.nu if self.nu > 0.0 else 0.495
            denom = (1.0 - nu_val * nu_val) * r
            e_val = self.E
            if denom > 0.0 and e_val > 0.0:
                return CallableFloat(math.sqrt(e_val / denom))
        return self.sound_speed

    # Mapping protocol
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if isinstance(self.params, dict) and key in self.params:
            return self.params[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            if not isinstance(self.params, dict):
                self.params = {}
            self.params[key] = value

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or (isinstance(self.params, dict) and key in self.params)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> List[str]:
        base_keys = [
            "id", "title", "rho0", "ref_rho", "mu", "d", "lam", "itype",
            "fct_id", "nu", "fscale", "law", "law_name", "rho", "G0", "G",
            "K", "bulk", "E", "young", "sound_speed", "sound_speed_solid",
            "sound_speed_shell",
        ]
        if isinstance(self.params, dict):
            for k in self.params:
                if k not in base_keys:
                    base_keys.append(k)
        return base_keys

    def values(self) -> List[Any]:
        return [self[k] for k in self.keys()]

    def items(self) -> List[tuple]:
        return [(k, self[k]) for k in self.keys()]

    def __len__(self) -> int:
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())


MatLaw92 = MaterialLaw92
MatArrudaBoyce = MaterialLaw92
MatArruda = MaterialLaw92


@dataclass
class MaterialLaw94:
    """/MAT/LAW94 or /MAT/YEOH (M173, M567): Yeoh 3rd-order polynomial hyperelastic model.

    Fortran origin: ``starter/source/materials/mat/mat094/hm_read_mat94.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ref_rho: float = 0.0
    c10: float = 0.0
    c20: float = 0.0
    c30: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0
    nu: float = 0.495
    law: int = 94
    law_name: str = "LAW94"
    params: dict = field(default_factory=dict)

    @property
    def rho(self) -> float:
        return self.ref_rho if self.ref_rho > 0.0 else self.rho0

    @property
    def G0(self) -> float:
        return 2.0 * self.c10

    @property
    def G(self) -> float:
        return self.G0

    @property
    def K(self) -> float:
        if self.d1 > 0.0:
            return 2.0 / self.d1
        g0 = self.G0
        nu_val = self.nu if (0.0 <= self.nu < 0.5) else 0.495
        if nu_val < 0.5:
            return (2.0 / 3.0) * (1.0 + nu_val) * g0 / max(1e-30, 1.0 - 2.0 * nu_val)
        return 0.0

    @property
    def bulk(self) -> float:
        return self.K

    @property
    def E(self) -> float:
        k_val = self.K
        g_val = self.G0
        denom = 3.0 * k_val + g_val
        if denom > 1e-20:
            return (9.0 * k_val * g_val) / denom
        nu_val = self.nu if (0.0 <= self.nu < 0.5) else 0.495
        return 2.0 * g_val * (1.0 + nu_val)

    @property
    def nu_eff(self) -> float:
        if self.d1 > 0.0:
            k_val = self.K
            g_val = self.G0
            denom = 3.0 * k_val + g_val
            if denom > 1e-20:
                return (3.0 * k_val - 2.0 * g_val) / (2.0 * denom)
        return self.nu if (0.0 <= self.nu < 0.5) else 0.495

    @property
    def young(self) -> float:
        return self.E

    @property
    def sound_speed(self) -> CallableFloat:
        import math
        r = self.rho
        if r > 0.0:
            c2 = (self.K + (4.0 / 3.0) * self.G0) / r
            if c2 > 0.0:
                return CallableFloat(math.sqrt(c2))
        return CallableFloat(0.0)

    @property
    def sound_speed_solid(self) -> CallableFloat:
        return self.sound_speed

    @property
    def sound_speed_shell(self) -> CallableFloat:
        import math
        r = self.rho
        if r > 0.0:
            nu_val = self.nu_eff
            denom = (1.0 - nu_val * nu_val) * r
            e_val = self.E
            if denom > 0.0 and e_val > 0.0:
                return CallableFloat(math.sqrt(e_val / denom))
        return self.sound_speed

    # Mapping protocol
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if isinstance(self.params, dict) and key in self.params:
            return self.params[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            if not isinstance(self.params, dict):
                self.params = {}
            self.params[key] = value

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or (isinstance(self.params, dict) and key in self.params)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> List[str]:
        base_keys = [
            "id", "title", "rho0", "ref_rho", "c10", "c20", "c30",
            "d1", "d2", "d3", "law", "law_name", "rho", "G0", "G",
            "K", "bulk", "nu", "E", "young", "sound_speed",
            "sound_speed_solid", "sound_speed_shell",
        ]
        if isinstance(self.params, dict):
            for k in self.params:
                if k not in base_keys:
                    base_keys.append(k)
        return base_keys

    def values(self) -> List[Any]:
        return [self[k] for k in self.keys()]

    def items(self) -> List[tuple]:
        return [(k, self[k]) for k in self.keys()]

    def __len__(self) -> int:
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())


MatLaw94 = MaterialLaw94
MatYeoh = MaterialLaw94



@dataclass
class MaterialLaw46:
    """/MAT/LAW46 or /MAT/HYD_VISC or /MAT/LES_FLUID (M173): Hydrodynamic viscous fluid model with Smagorinsky turbulence.

    Fortran origin: ``starter/source/materials/mat/mat046/hm_read_mat46.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ref_rho: float = 0.0
    c: float = 0.0
    nu: float = 0.0
    istf: int = 1
    smag: float = 1.0
    cps: float = 0.0


@dataclass
class MaterialLaw69:
    """/MAT/LAW69 or /MAT/HYP_EXT_COMP (M173, M550): Hyperelastic material model extended to compression.

    Fortran origin: ``starter/source/materials/mat/mat069/hm_read_mat69.F``.
    """
    id: int = 0
    title: str = ""
    rho0: float = 0.0
    ref_rho: float = 0.0
    iflag: int = 1
    fct_id_bulk: int = 0
    nu: float = 0.495
    fscale: float = 1.0
    nip: int = 2
    icheck: int = -3
    fct_id_data: int = 0
    mu: Any = None
    alpha: Any = None

    def __init__(
        self,
        id: int = 0,
        title: str = "",
        rho0: float = 0.0,
        ref_rho: float = 0.0,
        iflag: int = 1,
        fct_id_bulk: int = 0,
        nu: float = 0.495,
        fscale: float = 1.0,
        nip: int = 2,
        icheck: int = -3,
        fct_id_data: int = 0,
        mu: Any = None,
        alpha: Any = None,
        **kwargs,
    ):
        self.id = id
        self.title = kwargs.get("title", title)
        self.rho0 = kwargs.get("rho", rho0)
        self.ref_rho = kwargs.get("rhor", kwargs.get("refer_rho", ref_rho))
        self.iflag = kwargs.get("law_id", iflag)
        self.fct_id_bulk = kwargs.get("fct_id", fct_id_bulk)
        self.nu = kwargs.get("nu", nu)
        self.fscale = kwargs.get("fscale", fscale)
        self.nip = kwargs.get("n_pair", nip)
        self.icheck = kwargs.get("gflag", icheck)
        self.fct_id_data = kwargs.get("fct_id1", fct_id_data)
        self.mu = kwargs.get("mu", mu)
        self.alpha = kwargs.get("alpha", alpha)

    @property
    def rho(self) -> float:
        return self.rho0

    @rho.setter
    def rho(self, val: float) -> None:
        self.rho0 = float(val)

    @property
    def rhor(self) -> float:
        return self.ref_rho

    @rhor.setter
    def rhor(self, val: float) -> None:
        self.ref_rho = float(val)

    @property
    def law_id(self) -> int:
        return self.iflag

    @law_id.setter
    def law_id(self, val: int) -> None:
        self.iflag = int(val)

    @property
    def fct_id(self) -> int:
        return self.fct_id_bulk

    @fct_id.setter
    def fct_id(self, val: int) -> None:
        self.fct_id_bulk = int(val)

    @property
    def n_pair(self) -> int:
        return self.nip

    @n_pair.setter
    def n_pair(self, val: int) -> None:
        self.nip = int(val)

    @property
    def fct_id1(self) -> int:
        return self.fct_id_data

    @fct_id1.setter
    def fct_id1(self, val: int) -> None:
        self.fct_id_data = int(val)


MatLaw69 = MaterialLaw69
MatHypElas = MaterialLaw69
MatHyperelastic = MaterialLaw69


@dataclass
class MaterialLaw124:
    """/MAT/LAW124 or /MAT/CDPM2 (M174): Concrete Damage Plastic Model 2 (CDPM2).

    Fortran origin: ``starter/source/materials/mat/mat124/hm_read_mat124.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    irate: int = 0
    fcut: float = 0.0
    ecc: float = 0.0
    qh0: float = 0.0
    ft: float = 0.0
    fc: float = 0.0
    hp: float = 0.0
    ah: float = 0.0
    bh: float = 0.0
    ch: float = 0.0
    dh: float = 0.0
    as_: float = 0.0
    bs: float = 0.0
    df: float = 0.0
    dflag: int = 0
    dtype: int = 0
    ireg: int = 0
    wf: float = 0.0
    wf1: float = 0.0
    ft1: float = 0.0
    efc: float = 0.0


@dataclass
class MaterialLaw126:
    """/MAT/LAW126 or /MAT/JOHNSON_HOLMQUIST_CONCRETE (M174): Johnson-Holmquist concrete damage model.

    Fortran origin: ``starter/source/materials/mat/mat126/hm_read_mat126.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    g: float = 0.0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    fc: float = 0.0
    t0: float = 0.0
    c: float = 0.0
    eps0: float = 0.0
    fcut: float = 0.0
    sfmax: float = 0.0
    efmin: float = 0.0
    pc: float = 0.0
    muc: float = 0.0
    pl: float = 0.0
    mul: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    idel: int = 0
    eps_max: float = 0.0
    ifailso: int = 0
    ct: float = 0.0
    powt: float = 0.0
    cc: float = 0.0
    powc: float = 0.0


@dataclass
class MaterialLaw169:
    """/MAT/LAW169 or /MAT/ARUP_ADHESIVE (M591): Arup structural adhesive cohesive model.

    Fortran origin: ``starter/source/materials/mat/mat169/hm_read_mat169.F90``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    young: float = 0.0
    nu: float = 0.0
    sht_sl: float = 0.0
    tenmax: float = 1e20
    gcten: float = 1e20
    shrmax: float = 1e20
    gcshr: float = 1e20
    pwrt: int = 2
    pwrs: int = 2
    shrp: float = 0.0

    @property
    def e(self) -> float:
        return self.young

    @e.setter
    def e(self, val: float) -> None:
        self.young = val

    @property
    def rho(self) -> float:
        return self.rho0

    @rho.setter
    def rho(self, val: float) -> None:
        self.rho0 = val

    @property
    def pr(self) -> float:
        return self.nu

    @pr.setter
    def pr(self, val: float) -> None:
        self.nu = val


MatLaw169 = MaterialLaw169
MatArupAdhesive = MaterialLaw169


@dataclass
class MaterialLaw125:
    """/MAT/LAW125 or /MAT/LAMINATED_COMPOSITE (M174): Multi-layered laminated composite model.

    Fortran origin: ``starter/source/materials/mat/mat125/hm_read_mat125.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ea: float = 0.0
    eb: float = 0.0
    ec: float = 0.0
    ifail: int = 0
    gab: float = 0.0
    gca: float = 0.0
    gbc: float = 0.0
    prba: float = 0.0
    prca: float = 0.0
    prcb: float = 0.0
    lce11t: int = 0
    e11t: float = 0.0
    fct_t11: int = 0
    t11: float = 0.0
    slimt11: float = 0.0
    lce11c: int = 0
    e11c: float = 0.0
    fct_c11: int = 0
    c11: float = 0.0
    slimc11: float = 0.0
    lce22t: int = 0
    e22t: float = 0.0
    fct_t22: int = 0
    t22: float = 0.0
    slimt22: float = 0.0
    lce22c: int = 0
    e22c: float = 0.0
    fct_c22: int = 0
    c22: float = 0.0
    slimc22: float = 0.0
    lce33t: int = 0
    e33t: float = 0.0
    fct_t33: int = 0
    t33: float = 0.0
    slimt33: float = 0.0
    lce33c: int = 0
    e33c: float = 0.0
    fct_c33: int = 0
    c33: float = 0.0
    slimc33: float = 0.0
    g12a: float = 0.0
    t12a: float = 0.0
    g12b: float = 0.0
    t12b: float = 0.0
    slims12: float = 0.0
    fct_g12a: int = 0
    fct_t12a: int = 0
    fct_g12b: int = 0
    fct_t12b: int = 0
    g31a: float = 0.0
    t31a: float = 0.0
    g31b: float = 0.0
    t31b: float = 0.0
    slims31: float = 0.0
    fct_g31a: int = 0
    fct_t31a: int = 0
    fct_g31b: int = 0
    fct_t31b: int = 0
    g23a: float = 0.0
    t23a: float = 0.0
    g23b: float = 0.0
    t23b: float = 0.0
    slims23: float = 0.0
    fct_g23a: int = 0
    fct_t23a: int = 0
    fct_g23b: int = 0
    fct_t23b: int = 0
    epsf: float = 0.0
    epsr: float = 0.0
    dmax: float = 0.0
    fct_fail: int = 0
    fail: float = 0.0
    fcut: float = 0.0


@dataclass
class MaterialLaw127:
    """/MAT/LAW127 or /MAT/ENHANCED_COMPOSITE (M174): Enhanced orthotropic composite model.

    Fortran origin: ``starter/source/materials/mat/mat127/hm_read_mat127.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ea: float = 0.0
    eb: float = 0.0
    ec: float = 0.0
    gab: float = 0.0
    gca: float = 0.0
    gbc: float = 0.0
    prba: float = 0.0
    prca: float = 0.0
    prcb: float = 0.0
    xt: float = 0.0
    slimt1: float = 0.0
    lcxt: int = 0
    scalcxt: float = 1.0
    yt: float = 0.0
    slimt2: float = 0.0
    lcyt: int = 0
    scalcyt: float = 1.0
    sc: float = 0.0
    slimsc: float = 0.0
    lcsc: int = 0
    scalcsc: float = 1.0
    xc: float = 0.0
    slimc1: float = 0.0
    lcxc: int = 0
    scalcxc: float = 1.0
    yc: float = 0.0
    slimc2: float = 0.0
    lcyc: int = 0
    scalcyc: float = 1.0
    fcut: float = 0.0
    alph: float = 0.0
    beta: float = 0.0
    two_way: int = 0
    ti: int = 0
    dfailt: float = 0.0
    dfailc: float = 0.0
    dfails: float = 0.0
    dfailm: float = 0.0
    ratio: float = 0.0
    ncyred: int = 0
    tfail: float = 0.0
    fbrt: float = 0.0
    ycfac: float = 0.0
    efs: float = 0.0
    epsf: float = 0.0
    epsr: float = 0.0
    tsmd: float = 0.0


@dataclass
class MaterialLaw130:
    """/MAT/LAW130 or /MAT/MODIFIED_HONEYCOMB (M174): Modified crushable honeycomb material model.

    Fortran origin: ``starter/source/materials/mat/mat130/hm_read_mat130.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    sigy: float = 0.0
    vf: float = 0.0
    mu: float = 0.0
    iform: int = 0
    shdflg: int = 0
    lca: int = 0
    lcb: int = 0
    lcc: int = 0
    lcs: int = 0
    lcab: int = 0
    lcbc: int = 0
    lcca: int = 0
    lcsr: int = 0
    eaau: float = 0.0
    ebbu: float = 0.0
    eccu: float = 0.0
    gabu: float = 0.0
    gbcu: float = 0.0
    gcau: float = 0.0
    rfac: float = 0.0
    tsef: float = 0.0
    ssef: float = 0.0
    pru: int = 0
    lcsra: int = 0
    lcsrb: int = 0
    lcsrc: int = 0
    lcsrab: int = 0
    lcsrbc: int = 0
    lcsrca: int = 0
    pruab: float = 0.0
    pruac: float = 0.0
    prubc: float = 0.0
    pruba: float = 0.0
    pruca: float = 0.0
    prucb: float = 0.0


@dataclass
class MaterialLaw128:
    """/MAT/LAW128 or /MAT/HILL_VISC_PLAST (M175): Hill anisotropic viscoplastic material model.

    Fortran origin: ``starter/source/materials/mat/mat128/hm_read_mat128.F90`` / CFG ``Law128_hill_visc_plast.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    sigy: float = 0.0
    kin: float = 0.0
    tab_id: int = 0
    facy: float = 0.0
    facx: float = 0.0
    qr1: float = 0.0
    cr1: float = 0.0
    qr2: float = 0.0
    cr2: float = 0.0
    qx1: float = 0.0
    cx1: float = 0.0
    qx2: float = 0.0
    cx2: float = 0.0
    epsp0: float = 0.0
    cp: float = 0.0
    r00: float = 1.0
    r45: float = 1.0
    r90: float = 1.0
    f: float = 0.0
    g: float = 0.0
    h: float = 0.0
    l: float = 0.0
    m: float = 0.0
    n: float = 0.0


@dataclass
class MaterialLaw129:
    """/MAT/LAW129 or /MAT/THERM_CREEP (M175): Thermo-elasto-viscoplastic creep material model.

    Fortran origin: ``starter/source/materials/mat/mat129/hm_read_mat129.F90`` / CFG ``Law129_therm_creep.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    sigy: float = 0.0
    alpha: float = 0.0
    tref: float = 0.0
    f_young: int = 0
    f_nu: int = 0
    f_yld: int = 0
    f_alpha: int = 0
    isensor: int = 0
    itab: int = 0
    facy: float = 0.0
    qr1: float = 0.0
    cr1: float = 0.0
    qr2: float = 0.0
    cr2: float = 0.0
    f_qr: int = 0
    f_cr: int = 0
    qx1: float = 0.0
    cx1: float = 0.0
    qx2: float = 0.0
    cx2: float = 0.0
    f_qx: int = 0
    f_cx: int = 0
    epsp0: float = 0.0
    cp: float = 0.0
    f_cc: int = 0
    f_cp: int = 0
    crpa: float = 0.0
    crpn: float = 0.0
    crpm: float = 0.0
    f_a: int = 0
    f_n: int = 0
    f_m: int = 0
    crp_law: int = 0
    crsig: float = 0.0
    crt: float = 0.0
    crpq: float = 0.0
    eps0: float = 0.0
    f_q: int = 0
    f_sig: int = 0


@dataclass
class MaterialLaw123:
    """/MAT/LAW123 or /MAT/DAIMLER_PINHO (M175): Daimler-Pinho 3D composite damage model.

    Fortran origin: ``starter/source/materials/mat/mat123/hm_read_mat123.F90`` / CFG ``matl123_daimler_pinho.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ea: float = 0.0
    eb: float = 0.0
    ec: float = 0.0
    gab: float = 0.0
    gca: float = 0.0
    gbc: float = 0.0
    prba: float = 0.0
    prca: float = 0.0
    prcb: float = 0.0
    enkink: float = 0.0
    ena: float = 0.0
    enb: float = 0.0
    ent: float = 0.0
    enl: float = 0.0
    xc: float = 0.0
    xt: float = 0.0
    yc: float = 0.0
    yt: float = 0.0
    sl: float = 0.0
    fio: float = 53.0
    sigy: float = 0.0
    lcss: int = 0
    beta: float = 0.0
    efs: float = 0.0
    ratio: float = 0.0
    fcut: float = 0.0


@dataclass
class MaterialLaw132:
    """/MAT/LAW132 or /MAT/DAIMLER_CAMANHO (M175): Daimler-Camanho composite failure model.

    Fortran origin: ``starter/source/materials/mat/mat132/hm_read_mat132.F90`` / CFG ``matl132_daimler_camanho.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ea: float = 0.0
    eb: float = 0.0
    ec: float = 0.0
    gab: float = 0.0
    gca: float = 0.0
    gbc: float = 0.0
    prba: float = 0.0
    prca: float = 0.0
    prcb: float = 0.0
    gxc: float = 0.0
    gxt: float = 0.0
    gyc: float = 0.0
    gyt: float = 0.0
    gsl: float = 0.0
    xc: float = 0.0
    xt: float = 0.0
    yc: float = 0.0
    yt: float = 0.0
    sl: float = 0.0
    gxc0: float = 0.0
    gxt0: float = 0.0
    xc0: float = 0.0
    xt0: float = 0.0
    fio: float = 53.0
    sigy: float = 0.0
    etan: float = 0.0
    beta: float = 0.0
    lcss: int = 0
    epsf23: float = 0.0
    epsr23: float = 0.0
    tsmd23: float = 0.0
    epsf31: float = 0.0
    epsr31: float = 0.0
    tsmd31: float = 0.0
    ef11t: float = 0.0
    ef11c: float = 0.0
    ef22t: float = 0.0
    ef22c: float = 0.0
    ef12: float = 0.0
    ef23: float = 0.0
    ef31: float = 0.0
    cf12: float = 0.0
    cf23: float = 0.0
    cf31: float = 0.0
    ratio: float = 0.0
    fcut: float = 0.0


@dataclass
class MaterialLaw134:
    """/MAT/LAW134 or /MAT/VISCOUS_FOAM (M175): Viscous foam material model.

    Fortran origin: ``starter/source/materials/mat/mat134/hm_read_mat134.F90`` / CFG ``matl134_viscous_foam.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    e1: float = 0.0
    n1: float = 0.0
    nu: float = 0.0
    e2: float = 0.0
    v2: float = 0.0
    n2: float = 0.0


@dataclass
class MaterialLaw104:
    """/MAT/LAW104 or /MAT/JOHNS_VOCE_DRUCKER (M176/M574): Combined Drucker-Prager and Voce hardening model.

    Fortran origin: ``starter/source/materials/mat/mat104/hm_read_mat104.F`` / CFG ``matl104_drucker.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    young: float = 0.0
    nu: float = 0.0
    ires: int = 1
    sigma_r: float = 1e20
    h: float = 0.0
    qv: float = 0.0
    bv: float = 0.0
    cdr: float = 0.0
    cjc: float = 0.0
    epsp0: float = 1.0
    fcut: float = 1e4
    tss: float = 0.0
    tref: float = 20.0
    tini: float = 20.0
    eta: float = 0.0
    cp: float = 0.0
    eps_iso: float = 1e20
    eps_ad: float = 2e20
    params: dict = field(default_factory=dict)
    law: int = 104
    law_name: str = "LAW104"

    @property
    def rho(self) -> float:
        return self.rho0

    @property
    def rhor(self) -> float:
        return self.refer_rho if self.refer_rho > 0.0 else self.rho0

    @property
    def e(self) -> float:
        return self.young

    @property
    def E(self) -> float:
        return self.young

    @property
    def G(self) -> float:
        return self.young / (2.0 * (1.0 + self.nu)) if (1.0 + self.nu) != 0.0 else 0.0

    @property
    def K(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.nu)) if (1.0 - 2.0 * self.nu) != 0.0 else 0.0

    @property
    def bulk(self) -> float:
        return self.K

    @property
    def sound_speed(self) -> float:
        r = self.refer_rho if self.refer_rho > 0.0 else (self.rho0 if self.rho0 > 0.0 else 1.0)
        return math.sqrt(max(0.0, self.K + (4.0 / 3.0) * self.G) / r)

    @property
    def sound_speed_shell(self) -> float:
        r = self.refer_rho if self.refer_rho > 0.0 else (self.rho0 if self.rho0 > 0.0 else 1.0)
        denom = 1.0 - self.nu * self.nu
        mod = self.young / denom if denom > 0.0 else self.young
        return math.sqrt(max(0.0, mod) / r)

    @property
    def sigma0_yld(self) -> float:
        return self.sigma_r

    @property
    def sigy(self) -> float:
        return self.sigma_r

    @property
    def q_voce(self) -> float:
        return self.qv

    @property
    def b_voce(self) -> float:
        return self.bv

    @property
    def c_dr(self) -> float:
        return self.cdr

    @property
    def c_jc(self) -> float:
        return self.cjc

    @property
    def eps0(self) -> float:
        return self.epsp0

    @property
    def mu(self) -> float:
        return self.tss


MatLaw104 = MaterialLaw104
MatDrucker = MaterialLaw104
MatJohnsVoceDrucker = MaterialLaw104
MatPlasDruck = MaterialLaw104


@dataclass
class MaterialLaw105:
    """/MAT/LAW105 or /MAT/POWDER_BURN (M176): Powder burn propellant material model.

    Fortran origin: ``starter/source/materials/mat/mat105/hm_read_mat105.F90`` / CFG ``matl105.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    bulk: float = 0.0
    p0: float = 0.0
    psh: float = 0.0
    gas_d: float = 0.0
    gas_eg: float = 0.0
    gr: float = 0.0
    c: float = 0.0
    alpha: float = 0.0
    func_b: int = 0
    scale_b: float = 1.0
    scale_p: float = 1.0
    func_gam: int = 0
    scale_gam: float = 1.0
    scale_rho: float = 1.0
    c1: float = 0.0
    c2: float = 0.0
    refer_rho: float = 0.0
    rhor: float = 0.0
    compac: float = 0.93
    params: dict = field(default_factory=dict)
    law: int = 105
    law_name: str = "LAW105"

    @property
    def rho(self) -> float:
        return self.rho0

    @property
    def K(self) -> float:
        return self.bulk

    @property
    def young(self) -> float:
        return 1.2 * self.bulk

    @property
    def nu(self) -> float:
        return 0.3

    @property
    def G(self) -> float:
        return (6.0 / 13.0) * self.bulk

    @property
    def sound_speed(self) -> float:
        r = self.refer_rho if self.refer_rho > 0.0 else (self.rhor if self.rhor > 0.0 else (self.rho0 if self.rho0 > 0.0 else 1.0))
        return math.sqrt(max(0.0, self.bulk / r))

    @property
    def sound_speed_solid(self) -> float:
        return self.sound_speed


MatLaw105 = MaterialLaw105
MatPowderBurn = MaterialLaw105
MaterialPowderBurn = MaterialLaw105


@dataclass
class MaterialLaw106:
    """/MAT/LAW106 or /MAT/JCOOK_ALM (M176): Johnson-Cook additive manufacturing phase transformation model.

    Fortran origin: ``starter/source/materials/mat/mat106/hm_read_mat106.F90`` / CFG ``mat_law106.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    rhor: float = 0.0
    young: float = 0.0
    nu: float = 0.0
    fct_id1: int = 0
    fct_id2: int = 0
    fct_id3: int = 0
    sigy: float = 0.0
    beta: float = 0.0
    hard_n: float = 1.0
    ep_max: float = 1e30
    sig_max: float = 1e30
    fcut: float = 0.0
    vp: int = 2
    nmax: int = 3
    tol: float = 1e-7
    cjc: float = 0.0
    deps0: float = 0.0
    m: float = 1.0
    tmelt: float = 1e30
    spheat: float = 0.0
    eta: float = 0.0
    t0: float = 300.0
    tr: float = 300.0
    params: dict = field(default_factory=dict)
    law: int = 106
    law_name: str = "LAW106"

    @property
    def rho(self) -> float:
        return self.rho0

    @property
    def refer_rho(self) -> float:
        return self.rhor if self.rhor > 0.0 else self.rho0

    @property
    def e(self) -> float:
        return self.young

    @property
    def E(self) -> float:
        return self.young

    @property
    def G(self) -> float:
        return self.young / (2.0 * (1.0 + self.nu)) if (1.0 + self.nu) != 0.0 else 0.0

    @property
    def K(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.nu)) if (1.0 - 2.0 * self.nu) != 0.0 else 0.0

    @property
    def bulk(self) -> float:
        return self.K

    @property
    def sound_speed(self) -> float:
        r = self.refer_rho if self.refer_rho > 0.0 else (self.rho0 if self.rho0 > 0.0 else 1.0)
        return math.sqrt(max(0.0, self.K + (4.0 / 3.0) * self.G) / r)

    @property
    def sound_speed_shell(self) -> float:
        r = self.refer_rho if self.refer_rho > 0.0 else (self.rho0 if self.rho0 > 0.0 else 1.0)
        denom = 1.0 - self.nu * self.nu
        mod = self.young / denom if denom > 0.0 else self.young
        return math.sqrt(max(0.0, mod) / r)

    @property
    def a(self) -> float:
        return self.sigy

    @property
    def b(self) -> float:
        return self.beta

    @property
    def n(self) -> float:
        return self.hard_n

    @property
    def eps_max(self) -> float:
        return self.ep_max

    @property
    def sigma_max(self) -> float:
        return self.sig_max

    @property
    def cs(self) -> float:
        return self.spheat

    @property
    def tref(self) -> float:
        return self.tr

    @property
    def c(self) -> float:
        return self.cjc


MatLaw106 = MaterialLaw106
MatJCookAlm = MaterialLaw106
MatJohnsCookAlm = MaterialLaw106


@dataclass
class MaterialLaw107:
    """/MAT/LAW107 or /MAT/PAPER_LIGHT (M176): Orthotropic elastoplastic paper model.

    Fortran origin: ``starter/source/materials/mat/mat107/hm_read_mat107.F`` / CFG ``matl107_paper_light.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    rhor: float = 0.0
    e1: float = 0.0
    e2: float = 0.0
    e3: float = 0.0
    ires: int = 0
    itab: int = 0
    ismooth: int = 0
    nu21: float = 0.0
    g12: float = 0.0
    g23: float = 0.0
    g13: float = 0.0
    xi1: float = 0.0
    xi2: float = 0.0
    g1c: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    sigy1: float = 0.0
    cini1: float = 0.0
    s1: float = 0.0
    sigy2: float = 0.0
    cini2: float = 0.0
    s2: float = 0.0
    sigy1c: float = 0.0
    cini1c: float = 0.0
    s1c: float = 0.0
    sigy2c: float = 0.0
    cini2c: float = 0.0
    s2c: float = 0.0
    sigyt: float = 0.0
    cinit: float = 0.0
    st: float = 0.0
    tab_yld1: int = 0
    xscale1: float = 1.0
    yscale1: float = 1.0
    tab_yld2: int = 0
    xscale2: float = 1.0
    yscale2: float = 1.0
    tab_yld1c: int = 0
    xscale1c: float = 1.0
    yscale1c: float = 1.0
    tab_yld2c: int = 0
    xscale2c: float = 1.0
    yscale2c: float = 1.0
    tab_yldt: int = 0
    xscale_t: float = 1.0
    yscale_t: float = 1.0

    def __init__(
        self,
        id: int = 0,
        title: str = "",
        rho0: float = 0.0,
        rhor: float = 0.0,
        e1: float = 0.0,
        e2: float = 0.0,
        e3: float = 0.0,
        ires: int = 2,
        itab: int = 0,
        ismooth: int = 1,
        nu21: float = 0.0,
        g12: float = 0.0,
        g23: float = 0.0,
        g13: float = 0.0,
        xi1: float = 0.0,
        xi2: float = 0.0,
        g1c: float = 0.0,
        d1: float = 0.0,
        d2: float = 0.0,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        k4: float = 0.0,
        k5: float = 0.0,
        k6: float = 0.0,
        sigy1: float = 1.0e30,
        cini1: float = 1.0e30,
        s1: float = 0.0,
        sigy2: float = 1.0e30,
        cini2: float = 1.0e30,
        s2: float = 0.0,
        sigy1c: float = 1.0e30,
        cini1c: float = 1.0e30,
        s1c: float = 0.0,
        sigy2c: float = 1.0e30,
        cini2c: float = 1.0e30,
        s2c: float = 0.0,
        sigyt: float = 1.0e30,
        cinit: float = 1.0e30,
        st: float = 0.0,
        tab_yld1: int = 0,
        xscale1: float = 1.0,
        yscale1: float = 1.0,
        tab_yld2: int = 0,
        xscale2: float = 1.0,
        yscale2: float = 1.0,
        tab_yld1c: int = 0,
        xscale1c: float = 1.0,
        yscale1c: float = 1.0,
        tab_yld2c: int = 0,
        xscale2c: float = 1.0,
        yscale2c: float = 1.0,
        tab_yldt: int = 0,
        xscale_t: float = 1.0,
        yscale_t: float = 1.0,
        **kwargs: Any,
    ):
        self.id = int(kwargs.pop("mat_id", kwargs.pop("id", id)))
        self.title = str(kwargs.pop("mat_name", kwargs.pop("title", title)))
        self.rho0 = float(kwargs.pop("rho", rho0))
        self.rhor = float(kwargs.pop("refer_rho", rhor))
        self.e1 = float(kwargs.pop("young1", kwargs.pop("E1", kwargs.pop("Young1", e1))))
        self.e2 = float(kwargs.pop("young2", kwargs.pop("E2", kwargs.pop("Young2", e2))))
        self.e3 = float(kwargs.pop("young3", kwargs.pop("E3", kwargs.pop("Young3", e3))))
        self.ires = ires
        self.itab = itab
        self.ismooth = ismooth
        if "nu12" in kwargs:
            nu12_in = float(kwargs.pop("nu12"))
            self.nu21 = (nu12_in * self.e2 / self.e1) if abs(self.e1) > 1.0e-20 else nu12_in
        else:
            self.nu21 = float(kwargs.pop("nu21", nu21))
        self.g12 = g12
        self.g23 = g23
        self.g13 = float(kwargs.pop("g31", kwargs.pop("G31", g13)))
        self.xi1 = xi1
        self.xi2 = xi2
        self.g1c = g1c
        self.d1 = d1
        self.d2 = d2
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6
        self.sigy1 = sigy1
        self.cini1 = cini1
        self.s1 = s1
        self.sigy2 = sigy2
        self.cini2 = cini2
        self.s2 = s2
        self.sigy1c = sigy1c
        self.cini1c = cini1c
        self.s1c = s1c
        self.sigy2c = sigy2c
        self.cini2c = cini2c
        self.s2c = s2c
        self.sigyt = sigyt
        self.cinit = cinit
        self.st = st
        self.tab_yld1 = tab_yld1
        self.xscale1 = xscale1
        self.yscale1 = yscale1
        self.tab_yld2 = tab_yld2
        self.xscale2 = xscale2
        self.yscale2 = yscale2
        self.tab_yld1c = tab_yld1c
        self.xscale1c = xscale1c
        self.yscale1c = yscale1c
        self.tab_yld2c = tab_yld2c
        self.xscale2c = xscale2c
        self.yscale2c = yscale2c
        self.tab_yldt = tab_yldt
        self.xscale_t = xscale_t
        self.yscale_t = yscale_t
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def rho(self) -> float:
        return self.rho0

    @rho.setter
    def rho(self, value: float) -> None:
        self.rho0 = value

    @property
    def g31(self) -> float:
        return self.g13

    @g31.setter
    def g31(self, value: float) -> None:
        self.g13 = value

    @property
    def E1(self) -> float:
        return self.e1

    @E1.setter
    def E1(self, value: float) -> None:
        self.e1 = value

    @property
    def E2(self) -> float:
        return self.e2

    @E2.setter
    def E2(self, value: float) -> None:
        self.e2 = value

    @property
    def E3(self) -> float:
        return self.e3

    @E3.setter
    def E3(self, value: float) -> None:
        self.e3 = value

    @property
    def Young1(self) -> float:
        return self.e1

    @Young1.setter
    def Young1(self, value: float) -> None:
        self.e1 = value

    @property
    def Young2(self) -> float:
        return self.e2

    @Young2.setter
    def Young2(self, value: float) -> None:
        self.e2 = value

    @property
    def Young3(self) -> float:
        return self.e3

    @Young3.setter
    def Young3(self, value: float) -> None:
        self.e3 = value

    @property
    def nu12(self) -> float:
        return (self.nu21 * self.e1 / self.e2) if abs(self.e2) > 1.0e-20 else 0.0

    @nu12.setter
    def nu12(self, value: float) -> None:
        self.nu21 = (value * self.e2 / self.e1) if abs(self.e1) > 1.0e-20 else value

    @property
    def sound_speed(self) -> float:
        import math
        nu12 = self.nu12
        denom = 1.0 - nu12 * self.nu21
        if abs(denom) < 1.0e-20:
            denom = 1.0e-6
        a11 = self.e1 / denom
        a12 = nu12 * self.e2 / denom
        a21 = self.nu21 * self.e1 / denom
        a22 = self.e2 / denom
        rho = max(self.rho0, 1.0e-20)
        max_mod = max(a11, a12, a21, a22, self.e3, self.g12, self.g23, self.g13)
        return math.sqrt(max_mod / rho)

    @property
    def sound_speed_shell(self) -> float:
        import math
        nu12 = self.nu12
        denom = 1.0 - nu12 * self.nu21
        if abs(denom) < 1.0e-20:
            denom = 1.0e-6
        a11 = self.e1 / denom
        a12 = nu12 * self.e2 / denom
        a21 = self.nu21 * self.e1 / denom
        a22 = self.e2 / denom
        rho = max(self.rho0, 1.0e-20)
        max_mod = max(a11, a12, a21, a22, self.g12, self.g23, self.g13)
        return math.sqrt(max_mod / rho)


MatLaw107 = MaterialLaw107
MatPaperLight = MaterialLaw107
MatPlasPaperLight = MaterialLaw107
MatPfeiffer = MaterialLaw107
MaterialPaperLight = MaterialLaw107
MaterialPlasPaperLight = MaterialLaw107
MaterialPfeiffer = MaterialLaw107


@dataclass(init=False)
class MaterialLaw109:
    """/MAT/LAW109 or /MAT/TAB_PLAS (M578): Tabulated elasto-plastic material model.

    Fortran origin: ``starter/source/materials/mat/mat109/hm_read_mat109.F`` / CFG ``mat109.cfg``.
    """
    id: int
    title: str = ""
    rho_i: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    c_p: float = 0.0
    eta: float = 1.0
    t_ref: float = 293.0
    t_ini: float = 293.0
    tab_id_h: int = 0
    tab_id_t: int = 0
    xscale_h: float = 1.0
    yscale_h: float = 1.0
    i_smooth: int = 1
    tab_eta: int = 0
    xscale_eta: float = 1.0
    fcut: float = 10000.0

    comments: list[str] = field(default_factory=list)
    unit_system: Optional[str] = None
    law: int = 109
    law_name: str = "TAB_PLAS"

    # Resolved references
    yield_table: Any = None
    temp_table: Any = None
    eta_table: Any = None

    def __init__(
        self,
        id: int = 0,
        title: str = "",
        rho_i: float = 0.0,
        refer_rho: float = 0.0,
        e: float = 0.0,
        nu: float = 0.0,
        c_p: float = 0.0,
        eta: float = 1.0,
        t_ref: float = 293.0,
        t_ini: float = 293.0,
        tab_id_h: int = 0,
        tab_id_t: int = 0,
        xscale_h: float = 1.0,
        yscale_h: float = 1.0,
        i_smooth: int = 1,
        tab_eta: int = 0,
        xscale_eta: float = 1.0,
        fcut: float = 10000.0,
        comments: Optional[list] = None,
        unit_system: Optional[str] = None,
        law: int = 109,
        law_name: str = "TAB_PLAS",
        yield_table: Any = None,
        temp_table: Any = None,
        eta_table: Any = None,
        **kwargs: Any,
    ):
        self.id = int(id)
        self.title = str(title)
        self.rho_i = float(kwargs.get("rho0", kwargs.get("rho", rho_i)))
        self.refer_rho = float(kwargs.get("rhor", refer_rho))
        self.e = float(kwargs.get("young", kwargs.get("E", e)))
        self.nu = float(kwargs.get("Nu", nu))
        self.c_p = float(kwargs.get("cp", c_p))
        self.eta = float(kwargs.get("ETA", eta))
        self.t_ref = float(kwargs.get("tref", kwargs.get("T_ref", t_ref)))
        self.t_ini = float(kwargs.get("tini", kwargs.get("T_ini", t_ini)))
        self.tab_id_h = int(kwargs.get("tab_yld", kwargs.get("tab_h", tab_id_h)))
        self.tab_id_t = int(kwargs.get("tab_temp", kwargs.get("tab_t", tab_id_t)))
        self.xscale_h = float(kwargs.get("xscale_h", kwargs.get("xscale", kwargs.get("Xscale_h", xscale_h))))
        self.yscale_h = float(kwargs.get("yscale_h", kwargs.get("yscale", kwargs.get("Yscale_h", yscale_h))))
        self.i_smooth = int(kwargs.get("ismooth", kwargs.get("I_smooth", i_smooth)))
        self.tab_eta = int(kwargs.get("TAB_ETA", tab_eta))
        self.xscale_eta = float(kwargs.get("xscale_eta", kwargs.get("xrate", kwargs.get("Xscale_ETA", xscale_eta))))
        self.fcut = float(kwargs.get("FCUT", fcut))
        self.comments = list(comments) if comments is not None else []
        self.unit_system = unit_system
        self.law = int(law)
        self.law_name = str(law_name)
        self.yield_table = yield_table
        self.temp_table = temp_table
        self.eta_table = eta_table

    @property
    def rho0(self) -> float:
        return self.rho_i

    @rho0.setter
    def rho0(self, val: float) -> None:
        self.rho_i = val

    @property
    def rho(self) -> float:
        return self.rho_i

    @property
    def young(self) -> float:
        return self.e

    @young.setter
    def young(self, val: float) -> None:
        self.e = val

    @property
    def E(self) -> float:
        return self.e

    @property
    def Nu(self) -> float:
        return self.nu

    @property
    def cp(self) -> float:
        return self.c_p

    @cp.setter
    def cp(self, val: float) -> None:
        self.c_p = val

    @property
    def tref(self) -> float:
        return self.t_ref

    @property
    def tini(self) -> float:
        return self.t_ini

    @property
    def tab_yld(self) -> int:
        return self.tab_id_h

    @property
    def tab_temp(self) -> int:
        return self.tab_id_t

    @property
    def xscale(self) -> float:
        return self.xscale_h

    @property
    def yscale(self) -> float:
        return self.yscale_h

    @property
    def ismooth(self) -> int:
        return self.i_smooth

    @property
    def xrate(self) -> float:
        return self.xscale_eta

    @property
    def G(self) -> float:
        return self.e / (2.0 * (1.0 + self.nu)) if self.e > 0.0 and self.nu > -1.0 else 0.0

    @property
    def bulk(self) -> float:
        return self.e / (3.0 * (1.0 - 2.0 * self.nu)) if self.e > 0.0 and self.nu < 0.5 else 0.0

    def sound_speed_solid(self) -> float:
        import math
        rho = self.rho_i if self.rho_i > 0.0 else (self.refer_rho if self.refer_rho > 0.0 else 1.0)
        g = self.G
        k = self.bulk
        return math.sqrt(max((k + 4.0 / 3.0 * g) / rho, 0.0))

    def sound_speed_shell(self) -> float:
        import math
        rho = self.rho_i if self.rho_i > 0.0 else (self.refer_rho if self.refer_rho > 0.0 else 1.0)
        denom = 1.0 - self.nu**2
        if abs(denom) < 1.0e-20:
            denom = 1.0e-6
        a11 = self.e / denom
        return math.sqrt(max(a11 / rho, 0.0))

    def sound_speed(self) -> float:
        return self.sound_speed_solid()


MatLaw109 = MaterialLaw109
MatTabPlas = MaterialLaw109
MaterialTabPlas = MaterialLaw109
MatElastoPlasTab = MaterialLaw109
MaterialElastoPlasTab = MaterialLaw109
MatLaw109TabPlas = MaterialLaw109
MaterialLaw109TabPlas = MaterialLaw109


@dataclass
class MaterialLaw110:
    """/MAT/LAW110 or /MAT/VEGTER (M176 / M579): Vegter anisotropic yield locus material model for 2D shells.

    Fortran origin: ``starter/source/materials/mat/mat110/hm_read_mat110.F`` / CFG ``matl110_vegter.cfg``.
    Engine origin: ``engine/source/materials/mat/mat110/sigeps110c.F`` (plane stress shells).
    """
    id: int = 0
    title: str = ""
    rho0: float = 0.0
    rhor: float = 0.0
    young: float = 0.0
    nu: float = 0.0
    ires: int = 2
    icrit: int = 1
    tab_yld: int = 0
    xscale: float = 1.0
    yscale: float = 1.0
    fbi: float = 1.0
    rhobi: float = 1.0
    sigma_r: float = 0.0
    dsigm: float = 0.0
    beta: float = 0.0
    omega: float = 0.0
    hard_n: float = 0.0
    eps0: float = 0.0
    sigs: float = 0.0
    dg0: float = 0.0
    deps0: float = 0.0
    m: float = 0.0
    tini: float = 293.0
    chard: float = 0.0
    fcut: float = 1.0e20
    vp: int = 2
    ismooth: int = 1
    tab_temp: int = 0
    rm_0: float = 0.0
    rm_45: float = 0.0
    rm_90: float = 0.0
    ag_0: float = 0.0
    ag_45: float = 0.0
    ag_90: float = 0.0
    r_0: float = 1.0
    r_45: float = 1.0
    r_90: float = 1.0
    angles_data: list = field(default_factory=list)
    unit_system: Optional[int] = None
    comments: list[str] = field(default_factory=list)
    law: int = 110
    law_name: str = "VEGTER"
    curve_yld: Any = None
    curve_temp: Any = None

    def __init__(
        self,
        id: int = 0,
        title: str = "",
        rho0: float = 0.0,
        rhor: float = 0.0,
        young: float = 0.0,
        nu: float = 0.0,
        ires: int = 2,
        icrit: int = 1,
        tab_yld: int = 0,
        xscale: float = 1.0,
        yscale: float = 1.0,
        fbi: float = 1.0,
        rhobi: float = 1.0,
        sigma_r: float = 0.0,
        dsigm: float = 0.0,
        beta: float = 0.0,
        omega: float = 0.0,
        hard_n: float = 0.0,
        eps0: float = 0.0,
        sigs: float = 0.0,
        dg0: float = 0.0,
        deps0: float = 0.0,
        m: float = 0.0,
        tini: float = 293.0,
        chard: float = 0.0,
        fcut: float = 1.0e20,
        vp: int = 2,
        ismooth: int = 1,
        tab_temp: int = 0,
        rm_0: float = 0.0,
        rm_45: float = 0.0,
        rm_90: float = 0.0,
        ag_0: float = 0.0,
        ag_45: float = 0.0,
        ag_90: float = 0.0,
        r_0: float = 1.0,
        r_45: float = 1.0,
        r_90: float = 1.0,
        angles_data: Optional[list] = None,
        unit_system: Optional[int] = None,
        comments: Optional[list] = None,
        law: int = 110,
        law_name: str = "VEGTER",
        **kwargs: Any,
    ) -> None:
        self.id = int(kwargs.get("mid", id))
        self.title = str(kwargs.get("title", title))
        self.rho0 = float(kwargs.get("rho_i", kwargs.get("rho", rho0)))
        self.rhor = float(kwargs.get("refer_rho", rhor))
        self.young = float(kwargs.get("e", kwargs.get("E", young)))
        self.nu = float(kwargs.get("Nu", nu))
        self.ires = int(kwargs.get("ires", ires))
        self.icrit = int(kwargs.get("icrit", icrit))
        self.tab_yld = int(kwargs.get("tab_yld", tab_yld))
        self.xscale = float(kwargs.get("xscale", xscale))
        self.yscale = float(kwargs.get("yscale", yscale))
        self.fbi = float(kwargs.get("fbi", fbi))
        self.rhobi = float(kwargs.get("rhobi", rhobi))
        self.sigma_r = float(kwargs.get("sigma_r", kwargs.get("sig0", sigma_r)))
        self.dsigm = float(kwargs.get("dsigm", dsigm))
        self.beta = float(kwargs.get("beta", beta))
        self.omega = float(kwargs.get("omega", omega))
        self.hard_n = float(kwargs.get("hard_n", kwargs.get("n", hard_n)))
        self.eps0 = float(kwargs.get("eps0", eps0))
        self.sigs = float(kwargs.get("sigs", sigs))
        self.dg0 = float(kwargs.get("dg0", dg0))
        self.deps0 = float(kwargs.get("deps0", deps0))
        self.m = float(kwargs.get("m", m))
        self.tini = float(kwargs.get("tini", tini))
        self.chard = float(kwargs.get("chard", chard))
        self.fcut = float(kwargs.get("fcut", fcut))
        self.vp = int(kwargs.get("vp", vp))
        self.ismooth = int(kwargs.get("ismooth", ismooth))
        self.tab_temp = int(kwargs.get("tab_temp", tab_temp))
        self.rm_0 = float(kwargs.get("rm_0", rm_0))
        self.rm_45 = float(kwargs.get("rm_45", rm_45))
        self.rm_90 = float(kwargs.get("rm_90", rm_90))
        self.ag_0 = float(kwargs.get("ag_0", ag_0))
        self.ag_45 = float(kwargs.get("ag_45", ag_45))
        self.ag_90 = float(kwargs.get("ag_90", ag_90))
        self.r_0 = float(kwargs.get("r_0", r_0))
        self.r_45 = float(kwargs.get("r_45", r_45))
        self.r_90 = float(kwargs.get("r_90", r_90))
        self.angles_data = list(angles_data) if angles_data is not None else list(kwargs.get("angles_data", []))
        self.unit_system = unit_system
        self.comments = list(comments) if comments is not None else []
        self.law = int(law)
        self.law_name = str(law_name)
        self.curve_yld = kwargs.get("curve_yld", None)
        self.curve_temp = kwargs.get("curve_temp", None)

    @property
    def mid(self) -> int:
        return self.id

    @mid.setter
    def mid(self, value: int) -> None:
        self.id = value

    @property
    def rho_i(self) -> float:
        return self.rho0

    @rho_i.setter
    def rho_i(self, value: float) -> None:
        self.rho0 = value

    @property
    def refer_rho(self) -> float:
        return self.rhor

    @refer_rho.setter
    def refer_rho(self, value: float) -> None:
        self.rhor = value

    @property
    def e(self) -> float:
        return self.young

    @e.setter
    def e(self, value: float) -> None:
        self.young = value

    @property
    def nangle(self) -> int:
        return len(self.angles_data)

    @nangle.setter
    def nangle(self, value: int) -> None:
        pass


MatLaw110 = MaterialLaw110
MatVegter = MaterialLaw110
MaterialVegter = MaterialLaw110
MatPlasVegter = MaterialLaw110
MaterialPlasVegter = MaterialLaw110
MatLaw110Vegter = MaterialLaw110
MaterialLaw110Vegter = MaterialLaw110


@dataclass
class MaterialLaw115:
    """/MAT/LAW115 or /MAT/DESHPANDE_FLECK (M176): Deshpande-Fleck metallic foam model.

    Fortran origin: ``starter/source/materials/mat/mat115/hm_read_mat115.F`` / CFG ``matl115_deshfleck.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    young: float = 0.0
    nu: float = 0.0
    ires: int = 2
    istat: int = 0
    alpha: float = 0.0
    cfail: float = 0.0
    pfail: float = 0.0
    sigp: float = 0.0
    gamma: float = 0.0
    epsd: float = 0.0
    alpha2: float = 0.0
    beta: float = 0.0
    rhof0: float = 0.0
    sigp_c0: float = 0.0
    sigp_c1: float = 0.0
    sigp_n: float = 0.0
    alpha2_c0: float = 0.0
    alpha2_c1: float = 0.0
    alpha2_n: float = 0.0
    gamma_c0: float = 0.0
    gamma_c1: float = 0.0
    gamma_n: float = 0.0
    beta_c0: float = 0.0
    beta_c1: float = 0.0
    beta_n: float = 0.0


# M177 Material Dataclasses: LAW111, LAW112, LAW116, LAW122, LAW158


@dataclass
class MaterialLaw111:
    """/MAT/LAW111 & /MAT/MARLOW: Marlow hyperelastic model constructed from test data."""
    id: int
    title: str = ""
    rho0: float = 0.0
    itype: int = 1
    fct_id: int = 0
    fscale: float = 1.0
    nu: float = 0.495


@dataclass
class MaterialLaw112:
    """/MAT/LAW112 & /MAT/PAPER / /MAT/PLAS_PAPER: Comprehensive 3D orthotropic paper plasticity model."""
    id: int
    title: str = ""
    rho0: float = 0.0
    rhor: float = 0.0
    e1: float = 0.0
    e2: float = 0.0
    e3: float = 0.0
    ires: int = 0
    itab: int = 0
    ismooth: int = 0
    nu21: float = 0.0
    g12: float = 0.0
    g23: float = 0.0
    g13: float = 0.0
    k: float = 0.0
    e3c: float = 0.0
    cc: float = 0.0
    nu1p: float = 0.0
    nu2p: float = 0.0
    nu4p: float = 0.0
    nu5p: float = 0.0
    # Analytic (itab = 0)
    s01: float = 0.0
    a01: float = 0.0
    b01: float = 0.0
    c01: float = 0.0
    s02: float = 0.0
    a02: float = 0.0
    b02: float = 0.0
    c02: float = 0.0
    s03: float = 0.0
    a03: float = 0.0
    b03: float = 0.0
    c03: float = 0.0
    s04: float = 0.0
    a04: float = 0.0
    b04: float = 0.0
    c04: float = 0.0
    s05: float = 0.0
    a05: float = 0.0
    b05: float = 0.0
    c05: float = 0.0
    asig: float = 0.0
    bsig: float = 0.0
    csig: float = 0.0
    tau0: float = 0.0
    atau: float = 0.0
    btau: float = 0.0
    # Tabulated (itab = 1)
    tab_yld1: int = 0
    xscale1: float = 1.0
    yscale1: float = 1.0
    tab_yld2: int = 0
    xscale2: float = 1.0
    yscale2: float = 1.0
    tab_yld3: int = 0
    xscale3: float = 1.0
    yscale3: float = 1.0
    tab_yld4: int = 0
    xscale4: float = 1.0
    yscale4: float = 1.0
    tab_yld5: int = 0
    xscale5: float = 1.0
    yscale5: float = 1.0
    tab_yldc: int = 0
    xscalec: float = 1.0
    yscalec: float = 1.0
    tab_ylds: int = 0
    xscales: float = 1.0
    yscales: float = 1.0


@dataclass
class MaterialLaw116:
    """/MAT/LAW116 & /MAT/COH_HYST / /MAT/COHESIVE_HYSTERETIC: Cohesive zone hysteretic damage model."""
    id: int
    title: str = ""
    rho0: float = 0.0
    young: float = 0.0
    g: float = 0.0
    thick: float = 0.0
    imass: int = 1
    idel: int = 1
    icrit: int = 1
    gc1_ini: float = 0.0
    gc1_inf: float = 0.0
    sratg1: float = 0.0
    fg1: float = 0.0
    gc2_ini: float = 0.0
    gc2_inf: float = 0.0
    sratg2: float = 0.0
    fg2: float = 0.0
    siga1: float = 0.0
    sigb1: float = 0.0
    srate1: float = 0.0
    order1: int = 1
    fail1: int = 1
    siga2: float = 0.0
    sigb2: float = 0.0
    srate2: float = 0.0
    order2: int = 1
    fail2: int = 1


@dataclass
class MaterialLaw122:
    """/MAT/LAW122 & /MAT/MODIFIED_LADEVEZE / /MAT/LADEVEZE_DELAM: Modified Ladevèze Delamination & Composite Damage Model."""
    id: int
    title: str = ""
    rho0: float = 0.0
    e1: float = 0.0
    e2: float = 0.0
    e3: float = 0.0
    g12: float = 0.0
    g23: float = 0.0
    g31: float = 0.0
    nu12: float = 0.0
    nu23: float = 0.0
    nu31: float = 0.0
    e1c: float = 0.0
    gamma: float = 0.0
    ish: int = 0
    itr: int = 0
    ires: int = 0
    sigy0: float = 0.0
    beta: float = 0.0
    hard_m: float = 0.0
    hard_a: float = 0.0
    eps_fti: float = 0.0
    eps_ftu: float = 0.0
    dftu: float = 0.0
    eps_fci: float = 0.0
    eps_fcu: float = 0.0
    dfcu: float = 0.0
    ibuck: int = 0
    ifuncd1: int = 0
    dsat1: float = 0.0
    y0: float = 0.0
    yc: float = 0.0
    b: float = 0.0
    dmax: float = 0.0
    yr: float = 0.0
    ysp: float = 0.0
    ifuncd2: int = 0
    dsat2: float = 0.0
    y0p: float = 0.0
    ycp: float = 0.0
    ifuncd2c: int = 0
    dsat2c: float = 0.0
    y0pc: float = 0.0
    ycpc: float = 0.0
    epsd11: float = 0.0
    d11: float = 0.0
    n11: float = 0.0
    d11u: float = 0.0
    n11u: float = 0.0
    epsd12: float = 0.0
    d22: float = 0.0
    n22: float = 0.0
    d12: float = 0.0
    n12: float = 0.0
    epsdr0: float = 0.0
    dr0: float = 0.0
    nr0: float = 0.0
    ltype11: int = 0
    ltype12: int = 0
    ltyper0: int = 0
    fcut: float = 0.0


@dataclass
class MaterialLaw158:
    """/MAT/LAW158 & /MAT/FABR_NL / /MAT/FABRIC_NL: Nonlinear Anisotropic Fabric Material."""
    id: int
    title: str = ""
    rho0: float = 0.0
    s1: float = 0.1
    s2: float = 0.1
    flex: float = 0.0
    flex1: float = 0.0
    flex2: float = 0.0
    zerostress: float = 0.0
    sensor_id: int = 0
    fun_a1: int = 0
    c1: float = 1.0
    fun_a2: int = 0
    c2: float = 1.0
    fun_a3: int = 0
    c3: float = 1.0
    fun_a4: int = 0
    fun_a5: int = 0

@dataclass
class MatLaw113Dof:
    """DOF parameters for /MAT/LAW113 (M179)."""
    stiff: float = 0.0
    damp: float = 0.0
    acoeft: float = 1.0
    bcoeft: float = 1.0
    dcoeft: float = 1.0
    fun_a: int = 0
    hflag: int = 0
    fun_b: int = 0
    fun_c: int = 0
    fun_d: int = 0
    min_rup: float = -1.0e30
    max_rup: float = 1.0e30
    prop_f: float = 0.0
    prop_e: float = 0.0
    scale: float = 1.0
    prop_h: float = 1.0
    fun_k: int = 0


@dataclass
class MatLaw113:
    """/MAT/LAW113 / /MAT/SPR_BEAM (M179): Nonlinear spring-beam material model."""
    id: int
    rho: float = 0.0
    ifail: int = 0
    ileng: int = 0
    ifail2: int = 0
    dofs: List[MatLaw113Dof] = field(default_factory=list)
    trans_vel0: float = 1.0
    rot_vel0: float = 1.0
    asrate: float = 1.0e30
    israte: int = 0
    dir_fails: List[List[float]] = field(default_factory=list)
    title: str = ""


@dataclass
class MatLaw79:
    """/MAT/LAW79 / /MAT/JOHN_HOLM (M179/M558): Johnson-Holmquist JH-2 ceramic material model.

    Upstream Fortran reference:
      hm_read_mat79.F, sigeps79.F, and matl79_79.cfg (radioss2023).

    Yield surfaces:
      - Intact strength:
          sigma*_i = a * (P* + T*)^n * (1 + c * ln(eps_dot*))
      - Fractured strength:
          sigma*_f = b * (P*)^m * (1 + c * ln(eps_dot*)) <= sigfmax
      - Current strength:
          sigma* = sigma*_i - D * (sigma*_i - sigma*_f)
      where sigma* = sigma_eq / HEL_stress, P* = P / P_HEL, T* = T / P_HEL.
    """
    id: int = 0
    rho: float = 0.0
    refer_rho: float = 0.0
    tau_shear: float = 0.0
    a: float = 0.0
    b: float = 0.0
    m: float = 0.0
    n: float = 0.0
    c: float = 0.0
    eps0: float = 0.0
    sigfmax: float = 0.0
    fcut: float = 0.0
    t: float = 0.0
    hel: float = 0.0
    phel: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    idel: int = 0
    epsmax: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    beta: float = 0.0
    title: str = ""
    law: int = 79
    law_name: str = "LAW79"
    unit_id: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho
        if self.eps0 == 0.0:
            self.eps0 = 1.0
        if self.sigfmax == 0.0:
            self.sigfmax = 1.0e20
        if self.epsmax == 0.0:
            self.epsmax = 1.0e20
        self.idel = min(max(0, self.idel), 3)

        if not isinstance(self.params, dict):
            self.params = {}
        core: Dict[str, Any] = {
            "id": self.id,
            "rho": self.rho,
            "rho0": self.rho,
            "refer_rho": self.refer_rho,
            "rhor": self.refer_rho,
            "tau_shear": self.tau_shear,
            "G": self.tau_shear,
            "shear": self.tau_shear,
            "g": self.tau_shear,
            "a": self.a,
            "b": self.b,
            "m": self.m,
            "n": self.n,
            "c": self.c,
            "eps0": self.eps0,
            "sigfmax": self.sigfmax,
            "sigma_fmax": self.sigfmax,
            "fcut": self.fcut,
            "t": self.t,
            "t0": self.t,
            "hel": self.hel,
            "phel": self.phel,
            "shel": self.shel,
            "tstar": self.tstar,
            "d1": self.d1,
            "d2": self.d2,
            "idel": self.idel,
            "epsmax": self.epsmax,
            "eps_max": self.epsmax,
            "k1": self.k1,
            "k2": self.k2,
            "k3": self.k3,
            "K": self.k1,
            "bulk": self.k1,
            "beta": self.beta,
            "title": self.title,
            "young": self.young,
            "E": self.young,
            "nu": self.nu,
            "sound_speed": self.sound_speed,
        }
        for k, v in core.items():
            if k not in self.params:
                self.params[k] = v

    @property
    def rho0(self) -> float:
        return self.rho

    @rho0.setter
    def rho0(self, val: float) -> None:
        self.rho = float(val)

    @property
    def rhor(self) -> float:
        return self.refer_rho if self.refer_rho != 0.0 else self.rho

    @rhor.setter
    def rhor(self, val: float) -> None:
        self.refer_rho = float(val)

    @property
    def G(self) -> float:
        return self.tau_shear

    @G.setter
    def G(self, val: float) -> None:
        self.tau_shear = float(val)

    @property
    def shear(self) -> float:
        return self.tau_shear

    @shear.setter
    def shear(self, val: float) -> None:
        self.tau_shear = float(val)

    @property
    def g(self) -> float:
        return self.tau_shear

    @g.setter
    def g(self, val: float) -> None:
        self.tau_shear = float(val)

    @property
    def t0(self) -> float:
        return self.t

    @t0.setter
    def t0(self, val: float) -> None:
        self.t = float(val)

    @property
    def sigma_fmax(self) -> float:
        return self.sigfmax

    @sigma_fmax.setter
    def sigma_fmax(self, val: float) -> None:
        self.sigfmax = float(val)

    @property
    def eps_max(self) -> float:
        return self.epsmax

    @eps_max.setter
    def eps_max(self, val: float) -> None:
        self.epsmax = float(val)

    @property
    def shel(self) -> float:
        """Equivalent strength at HEL: shel = 1.5 * (HEL - PHEL) (UPARAM(12) in hm_read_mat79.F)."""
        return 1.5 * (self.hel - self.phel)

    @property
    def tstar(self) -> float:
        """Normalized tensile strength: tstar = T / PHEL (UPARAM(10) in hm_read_mat79.F)."""
        return self.t / self.phel if self.phel != 0.0 else 0.0

    @property
    def bulk(self) -> float:
        return self.k1

    @bulk.setter
    def bulk(self, val: float) -> None:
        self.k1 = float(val)

    @property
    def K(self) -> float:
        return self.k1

    @K.setter
    def K(self, val: float) -> None:
        self.k1 = float(val)

    @property
    def young(self) -> float:
        """Young's modulus derived from bulk modulus K1 and shear modulus G:
        YOUNG = 9 * K1 * G / (3 * K1 + G) per hm_read_mat79.F:225.
        """
        denom = 3.0 * self.k1 + self.tau_shear
        return (9.0 * self.k1 * self.tau_shear) / denom if denom > 0.0 else 0.0

    @property
    def E(self) -> float:
        return self.young

    @property
    def nu(self) -> float:
        """Poisson's ratio derived from bulk modulus K1 and shear modulus G:
        NU = (3 * K1 - 2 * G) / (6 * K1 + 2 * G) per hm_read_mat79.F:224.
        """
        denom = 6.0 * self.k1 + 2.0 * self.tau_shear
        return (3.0 * self.k1 - 2.0 * self.tau_shear) / denom if denom > 0.0 else 0.0

    @property
    def sound_speed(self) -> float:
        """Acoustic sound speed in solid: c = sqrt((K1 + 4/3 * G) / rho0) per sigeps79.F:292."""
        import math
        rho_val = self.rho0 if self.rho0 > 0.0 else self.refer_rho
        if rho_val > 0.0:
            c2 = (self.k1 + (4.0 / 3.0) * self.tau_shear) / rho_val
            if c2 > 0.0:
                return math.sqrt(c2)
        return 0.0

    @property
    def sound_speed_solid(self) -> float:
        return self.sound_speed

    # Mapping protocol
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if isinstance(self.params, dict) and key in self.params:
            return self.params[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or (isinstance(self.params, dict) and key in self.params)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> List[str]:
        base_keys = [
            "id", "rho", "refer_rho", "tau_shear", "a", "b", "m", "n",
            "c", "eps0", "sigfmax", "fcut", "t", "hel", "phel",
            "d1", "d2", "idel", "epsmax", "k1", "k2", "k3", "beta",
            "title", "law", "law_name", "G", "shear", "bulk", "K",
            "shel", "tstar", "young", "E", "nu", "sound_speed",
        ]
        if isinstance(self.params, dict):
            for k in self.params:
                if k not in base_keys:
                    base_keys.append(k)
        return base_keys


MatJohnHolm = MatLaw79
MatJohnsonHolmquist = MatLaw79
MatJH2 = MatLaw79


@dataclass
class MatViscLprony:
    """/MAT/VISC_LPRONY / /VISC/LPRONY (M179): Viscoelastic large Prony series model."""
    id: int
    m: int = 0
    form: int = 0
    flag_visc: int = 0
    gamai: List[float] = field(default_factory=list)
    taui: List[float] = field(default_factory=list)
    title: str = ""


# M180: MAT_LAW190, MAT_LAW41, FAIL_CHANG, PROP_TYPE20, PROP_TYPE21, PROP_TYPE22
@dataclass
class MatLaw190:
    """/MAT/LAW190 / /MAT/FOAM_DUBOIS (M180): Du Bois foam model with 3D table."""
    id: int
    rho: float = 0.0
    e0: float = 0.0
    nu: float = 0.0
    hu: float = 0.0
    shape: float = 1.0
    fun_1: int = 0
    xscale_1: float = 1.0
    scale_1: float = 1.0
    tcut: float = 1.0e20
    fail: int = 0
    table: Any = None
    title: str = ""


@dataclass
class MatLaw41:
    """/MAT/LAW41 / /MAT/LEE_T (M180): Lee-Tarver explosive reaction kinetics and JWL EOS."""
    id: int
    rho: float = 0.0
    refer_rho: float = 0.0
    ireac: int = 0
    a_r: float = 0.0
    b_r: float = 0.0
    r_1r: float = 0.0
    r_2r: float = 0.0
    r_3r: float = 0.0
    a_p: float = 0.0
    b_p: float = 0.0
    r_1p: float = 0.0
    r_2p: float = 0.0
    r_3p: float = 0.0
    c_vr: float = 0.0
    c_vp: float = 0.0
    enq: float = 0.0
    nitrs: int = 0
    epsilon_0: float = 0.0
    ftol: float = 0.0
    i_coeff: float = 0.0
    b_coeff: float = 0.0
    x_coeff: float = 0.0
    g1: float = 0.0
    d_coeff: float = 0.0
    y_coeff: float = 0.0
    c_coeff: float = 0.0
    kn: float = 0.0
    chi: float = 0.0
    tol: float = 0.0
    g2: float = 0.0
    e_coeff: float = 0.0
    g_coeff: float = 0.0
    z_coeff: float = 0.0
    ccrit: float = 0.0
    figmax: float = 0.0
    fg1max: float = 0.0
    fg2min: float = 0.0
    g0: float = 0.0
    t_initial: float = 293.15
    title: str = ""


@dataclass
class FailChang:
    """/FAIL/CHANG (M180): Chang-Chang composite failure model."""
    id: int = 0
    mat_id: int = 0
    sigma_1t: float = 0.0
    sigma_2t: float = 0.0
    sigma_12: float = 0.0
    sigma_1c: float = 0.0
    sigma_2c: float = 0.0
    beta: float = 0.0
    tau_max: float = 0.0
    ifail_sh: int = 1
    failip: int = 0
    fail_id: int = 0


@dataclass
class FailFabric:
    """/FAIL/FABRIC or /FAIL/FABR (M181): Fabric failure criterion."""
    id: int = 0
    mat_id: int = 0
    epsilon_f1: float = 0.0
    epsilon_r1: float = 0.0
    epsilon_f2: float = 0.0
    epsilon_r2: float = 0.0
    ndir: int = 0
    fct_id: int = 0
    fail_id: int = 0


@dataclass
class FailHoffman:
    """/FAIL/HOFFMAN (M181): Hoffman 3D orthotropic failure criterion."""
    id: int = 0
    mat_id: int = 0
    sigma_1t: float = 0.0
    sigma_2t: float = 0.0
    sigma_1c: float = 0.0
    sigma_2c: float = 0.0
    sigma_12: float = 0.0
    tau_max: float = 0.0
    fcut: float = 0.0
    ifail_sh: int = 1
    ifail_so: int = 0
    fail_id: int = 0


@dataclass
class FailMaxStrain:
    """/FAIL/MAX_STRAIN or /FAIL/MAXSTRAIN (M181): Maximum strain failure model."""
    id: int = 0
    mat_id: int = 0
    eps1_max: float = 0.0
    eps2_max: float = 0.0
    gam12_max: float = 0.0
    tau_max: float = 0.0
    fcut: float = 0.0
    ifail_sh: int = 1
    ifail_so: int = 0
    fail_id: int = 0


@dataclass
class FailTsaiHill:
    """/FAIL/TSAI_HILL or /FAIL/TSAIHILL (M181): Tsai-Hill composite failure criterion."""
    id: int = 0
    mat_id: int = 0
    x11: float = 0.0
    x22: float = 0.0
    s12: float = 0.0
    tau_max: float = 0.0
    fcut: float = 0.0
    ifail_sh: int = 1
    ifail_so: int = 0
    fail_id: int = 0


@dataclass
class FailTsaiWu:
    """/FAIL/TSAI_WU or /FAIL/TSAIWU (M181): Tsai-Wu quadratic composite failure criterion."""
    id: int = 0
    mat_id: int = 0
    sigma_1t: float = 0.0
    sigma_2t: float = 0.0
    sigma_1c: float = 0.0
    sigma_2c: float = 0.0
    sigma_12: float = 0.0
    alpha: float = 0.0
    tau_max: float = 0.0
    fcut: float = 0.0
    ifail_sh: int = 1
    ifail_so: int = 0
    fail_id: int = 0


# M182: MAT_LAW114, MAT_LAW117, MAT_LAW119, MAT_LAW120, MAT_LAW121, PROP_TYPE26, PROP_TYPE27

@dataclass
class MatLaw114:
    """/MAT/LAW114 or /MAT/SPR_SEATBELT (M182): 1D seatbelt spring material."""
    id: int
    rho: float = 0.0
    lmin: float = 0.0
    stiff1: float = 0.0
    damp1: float = 0.0
    fun_l: int = 0
    fun_ul: int = 0
    xcoeft1: float = 1.0
    fcoeft1: float = 1.0
    young: float = 0.0
    ibend: float = 0.0
    itors: float = 0.0
    fmax: float = 0.0
    mmax: float = 0.0
    shear_area: float = 0.0
    rfac: float = 0.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def k(self) -> float:
        return self.stiff1

    @property
    def c(self) -> float:
        return self.damp1

    @property
    def xscale(self) -> float:
        return self.xcoeft1

    @property
    def fscale(self) -> float:
        return self.fcoeft1

    @property
    def e(self) -> float:
        return self.young

    @property
    def i(self) -> float:
        return self.ibend

    @property
    def j(self) -> float:
        return self.itors

    @property
    def as_(self) -> float:
        return self.shear_area

    @property
    def r(self) -> float:
        return self.rfac


MatSprSeatbelt = MatLaw114


@dataclass
class MatLaw117:
    """/MAT/LAW117 or /MAT/COH_TAB (M182): Tabulated cohesive zone material."""
    id: int
    rho: float = 0.0
    refer_rho: float = 0.0
    en: float = 0.0
    es: float = 0.0
    imass: int = 0
    idel: int = 0
    irupt: int = 0
    fct_tn: int = 0
    fct_tt: int = 0
    tn: float = 0.0
    ts: float = 0.0
    fscale_x: float = 1.0
    gic: float = 0.0
    giic: float = 0.0
    exp_g: float = 0.0
    exp_bk: float = 0.0
    gamma: float = 0.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def e_elas_n(self) -> float:
        return self.en

    @property
    def e_elas_s(self) -> float:
        return self.es

    @property
    def e(self) -> float:
        return self.en

    @property
    def tmax_n(self) -> float:
        return self.tn

    @property
    def tmax_s(self) -> float:
        return self.ts


MatCohTab = MatLaw117


@dataclass
class MatLaw119:
    """/MAT/LAW119 or /MAT/SH_SEATBELT (M182): 2D shell seatbelt material."""
    id: int
    rho: float = 0.0
    lmin: float = 0.0
    stiff1: float = 0.0
    damp1: float = 0.0
    re: float = 0.0
    fun_l: int = 0
    fun_ul: int = 0
    fcoeft1: float = 1.0
    fcoeft2: float = 1.0
    ireload: int = 0
    e22: float = 0.0
    nu12: float = 0.0
    g12: float = 0.0
    fcoeft22: float = 1.0
    ecoat: float = 0.0
    nucoat: float = 0.0
    tcoat: float = 0.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def k(self) -> float:
        return self.stiff1

    @property
    def c(self) -> float:
        return self.damp1

    @property
    def fscale1(self) -> float:
        return self.fcoeft1

    @property
    def fscale2(self) -> float:
        return self.fcoeft2

    @property
    def fscale22(self) -> float:
        return self.fcoeft22


MatShSeatbelt = MatLaw119


@dataclass
class MatLaw120:
    """/MAT/LAW120 or /MAT/TAPO (M182): Tabulated orthotropic Pont-Pack material."""
    id: int
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    iform: int = 0
    itrx: int = 0
    idam: int = 0
    thick: float = 0.0
    tab_id: int = 0
    xscale: float = 1.0
    yscale: float = 1.0
    tau0: float = 0.0
    q: float = 0.0
    beta: float = 0.0
    h: float = 0.0
    af1: float = 0.0
    af2: float = 0.0
    ah1: float = 0.0
    ah2: float = 0.0
    as_: float = 0.0
    cc: float = 0.0
    gam0: float = 0.0
    gamf: float = 0.0
    d1c: float = 0.0
    d2c: float = 0.0
    d1f: float = 0.0
    d2f: float = 0.0
    dtrx: float = 0.0
    djc: float = 0.0
    exp_n: float = 0.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def tau(self) -> float:
        return self.tau0

    @property
    def d_trx(self) -> float:
        return self.dtrx

    @property
    def d_jc(self) -> float:
        return self.djc


MatTapo = MatLaw120


@dataclass
class MatLaw121:
    """/MAT/LAW121 or /MAT/PLAS_RATE (M182): Tabulated rate-dependent elastoplastic material."""
    id: int
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    ires: int = 2
    ivisc: int = 0
    fcut: float = 0.0
    dtmin: float = 0.0
    fct_sig0: int = 0
    xscale_sig0: float = 1.0
    yscale_sig0: float = 1.0
    fct_youn: int = 0
    xscale_youn: float = 1.0
    yscale_youn: float = 1.0
    fct_tang: int = 0
    xscale_tang: float = 1.0
    tang: float = 0.0
    fct_fail: int = 0
    ifail: int = 0
    xscale_fail: float = 1.0
    yscale_fail: float = 1.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def tdel(self) -> float:
        return self.dtmin


MatPlasRate = MatLaw121


# ============================================================================
# M183 Materials: LAW50, LAW57, LAW87, LAW95, LAW163, LAW169
# ============================================================================

try:
    CallableFloat  # type: ignore[name-defined]
except NameError:
    class CallableFloat(float):
        """Float that is also callable returning itself (compatible with both property and method access)."""
        def __call__(self) -> float:
            return float(self)


@dataclass
class MatLaw50:
    """/MAT/LAW50 or /MAT/VISC_HONEY or /MAT/HYP_FOAM (M183, M559): Rate-dependent honeycomb material."""
    id: int = 0
    rho: float = 0.0
    refer_rho: float = 0.0
    ea: float = 0.0
    eb: float = 0.0
    ec: float = 0.0
    gab: float = 0.0
    gbc: float = 0.0
    gca: float = 0.0
    asrate: float = 0.0
    irate: int = 2
    gflag: int = 0
    eps_max11: float = 0.0
    eps_max22: float = 0.0
    eps_max33: float = 0.0
    yfun11: list[int] = field(default_factory=list)
    sfac11: list[float] = field(default_factory=list)
    eps11: list[float] = field(default_factory=list)
    yfun22: list[int] = field(default_factory=list)
    sfac22: list[float] = field(default_factory=list)
    eps22: list[float] = field(default_factory=list)
    yfun33: list[int] = field(default_factory=list)
    sfac33: list[float] = field(default_factory=list)
    eps33: list[float] = field(default_factory=list)
    vflag: int = 0
    eps_max12: float = 0.0
    eps_max23: float = 0.0
    eps_max31: float = 0.0
    yfun12: list[int] = field(default_factory=list)
    sfac12: list[float] = field(default_factory=list)
    eps12: list[float] = field(default_factory=list)
    yfun23: list[int] = field(default_factory=list)
    sfac23: list[float] = field(default_factory=list)
    eps23: list[float] = field(default_factory=list)
    yfun31: list[int] = field(default_factory=list)
    sfac31: list[float] = field(default_factory=list)
    eps31: list[float] = field(default_factory=list)
    # Compaction fields (Card 25)
    ecomp: float = 0.0
    pr: float = 0.0
    sigy: float = 0.0
    et: float = 0.0
    vcomp: float = 0.0
    title: str = ""
    law: int = 50
    law_name: str = "LAW50"
    unit_id: Optional[int] = None
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho
        if not self.sfac11:
            self.sfac11 = [1.0, 1.0, 1.0, 1.0, 1.0]
        if not self.sfac22:
            self.sfac22 = [1.0, 1.0, 1.0, 1.0, 1.0]
        if not self.sfac33:
            self.sfac33 = [1.0, 1.0, 1.0, 1.0, 1.0]
        if not self.sfac12:
            self.sfac12 = [1.0, 1.0, 1.0, 1.0, 1.0]
        if not self.sfac23:
            self.sfac23 = [1.0, 1.0, 1.0, 1.0, 1.0]
        if not self.sfac31:
            self.sfac31 = [1.0, 1.0, 1.0, 1.0, 1.0]
        if self.irate == 0:
            self.irate = 2
        if not isinstance(self.params, dict):
            self.params = {}

    @property
    def rho0(self) -> float:
        return self.rho


    @property
    def rhor(self) -> float:
        return self.refer_rho

    @property
    def e11(self) -> float:
        return self.ea

    @property
    def e22(self) -> float:
        return self.eb

    @property
    def e33(self) -> float:
        return self.ec

    @property
    def g12(self) -> float:
        return self.gab

    @property
    def g23(self) -> float:
        return self.gbc

    @property
    def g31(self) -> float:
        return self.gca

    @property
    def nu(self) -> float:
        return self.pr

    @nu.setter
    def nu(self, value: float) -> None:
        self.pr = value

    @property
    def hcomp(self) -> float:
        return self.et

    @hcomp.setter
    def hcomp(self, value: float) -> None:
        self.et = value

    @property
    def gcomp(self) -> float:
        if self.ecomp > 0.0:
            return self.ecomp / (1.0 + min(self.pr, 0.495))
        return 0.0

    @property
    def bulk(self) -> float:
        if self.ecomp > 0.0:
            return self.ecomp / (3.0 * (1.0 - 2.0 * min(self.pr, 0.495)))
        return max(self.ea, self.eb, self.ec, self.gab, self.gbc, self.gca)

    @property
    def K(self) -> float:
        return self.bulk

    @property
    def E(self) -> float:
        return max(self.ea, self.eb, self.ec)

    @property
    def G(self) -> float:
        return max(self.gab, self.gbc, self.gca)

    @property
    def sound_speed(self) -> CallableFloat:
        import math
        rho_val = self.rho if self.rho > 0.0 else self.refer_rho
        if rho_val > 0.0:
            c = math.sqrt(max(self.ea, self.eb, self.ec, self.gab, self.gbc, self.gca) / rho_val)
            return CallableFloat(c)
        return CallableFloat(0.0)

    @property
    def sound_speed_solid(self) -> CallableFloat:
        return self.sound_speed

    @property
    def fcut(self) -> float:
        return self.asrate

    @fcut.setter
    def fcut(self, val: float) -> None:
        self.asrate = val

    @property
    def icomp(self) -> int:
        return 1 if (self.ecomp * self.sigy * self.vcomp > 0.0) else 0

    @property
    def icompact(self) -> int:
        return self.icomp


    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.params:
            return self.params[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        self.params[key] = value

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or (key in self.params)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.params.get(key, default)


MatViscHoney = MatLaw50
MatHypFoam = MatLaw50


try:
    CallableFloat  # type: ignore[name-defined]
except NameError:
    class CallableFloat(float):
        """Float that is also callable returning itself (compatible with both property and method access)."""
        def __call__(self) -> float:
            return float(self)


@dataclass
class MatLaw57Curve:
    """A single strain-rate plasticity curve for /MAT/LAW57 (/MAT/BARLAT3)."""
    fct_id: int = 0
    fscale: float = 1.0
    eps: float = 0.0

    @property
    def func_id(self) -> int:
        return self.fct_id

    @property
    def scale(self) -> float:
        return self.fscale

    @property
    def rate(self) -> float:
        return self.eps

    def __getitem__(self, item: str) -> Any:
        if item in ("fct_id", "func_id", "fid"):
            return self.fct_id
        if item in ("fscale", "scale"):
            return self.fscale
        if item in ("eps", "rate"):
            return self.eps
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        try:
            return self[item]
        except KeyError:
            return default


@dataclass
class MatLaw57:
    """/MAT/LAW57 or /MAT/BARLAT3 (M183, M555): Barlat 3-parameter anisotropic plasticity."""
    id: int = 0
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    ifunce: int = 0
    einf: float = 0.0
    ce: float = 0.0
    r00: float = 1.0
    r45: float = 1.0
    r90: float = 1.0
    chard: float = 0.0
    m: float = 6.0
    eps_max: float = 1.0e30
    eps_t1: float = 1.0e30
    eps_t2: float = 2.0e30
    fcut: float = 1.0e30
    fsmooth: int = 0
    vp: int = 0
    curves: list[MatLaw57Curve] = field(default_factory=list)
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def rhor(self) -> float:
        return self.refer_rho if self.refer_rho > 0.0 else self.rho

    @property
    def E(self) -> float:
        return self.e

    @E.setter
    def E(self, value: float) -> None:
        self.e = value

    @property
    def G(self) -> float:
        return 0.5 * self.e / (1.0 + self.nu) if (1.0 + self.nu) > 0.0 else 0.0

    @property
    def Nu(self) -> float:
        return self.nu

    @Nu.setter
    def Nu(self, value: float) -> None:
        self.nu = value

    @property
    def epsp_max(self) -> float:
        return self.eps_max

    @epsp_max.setter
    def epsp_max(self, value: float) -> None:
        self.eps_max = value

    @property
    def sound_speed_shell(self) -> CallableFloat:
        """Shell plane-stress sound speed: c = sqrt(E / (rho0 * (1 - nu^2)))."""
        rho_val = self.rho if self.rho > 0.0 else self.refer_rho
        if rho_val > 0.0 and self.e > 0.0 and (1.0 - self.nu**2) > 0.0:
            import math
            c = math.sqrt(self.e / (rho_val * (1.0 - self.nu**2)))
            return CallableFloat(c)
        return CallableFloat(0.0)

    @property
    def sound_speed(self) -> CallableFloat:
        return self.sound_speed_shell

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "rho": self.rho,
            "rho0": self.rho0,
            "refer_rho": self.refer_rho,
            "rhor": self.rhor,
            "e": self.e,
            "E": self.e,
            "nu": self.nu,
            "Nu": self.nu,
            "nu0": self.nu,
            "ifunce": self.ifunce,
            "einf": self.einf,
            "ce": self.ce,
            "r00": self.r00,
            "r45": self.r45,
            "r90": self.r90,
            "chard": self.chard,
            "m": self.m,
            "eps_max": self.eps_max,
            "epsp_max": self.eps_max,
            "eps_t1": self.eps_t1,
            "eps_t2": self.eps_t2,
            "fcut": self.fcut,
            "fsmooth": self.fsmooth,
            "vp": self.vp,
            "curves": self.curves,
        }

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        p = self.params
        if key in p:
            return p[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Cannot set unknown attribute {key!r} on MatLaw57")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or key in self.params

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


MatBarlat3 = MatLaw57


@dataclass
class MatLaw87Curve:
    fct_id: int = 0
    fscale: float = 1.0
    epsp: float = 0.0


class AlphaFloat(float):
    """Float representing scalar Swift-Voce mixing parameter that also
    supports indexing and iteration to access the 8 Barlat anisotropy alphas."""
    _owner: Any = None

    def __new__(cls, val: float, owner: Any = None):
        obj = super().__new__(cls, val)
        obj._owner = owner
        return obj

    def __getitem__(self, idx: Any) -> Any:
        if self._owner is not None and hasattr(self._owner, "alphas"):
            return self._owner.alphas[idx]
        raise IndexError("alpha index out of range")

    def __len__(self) -> int:
        if self._owner is not None and hasattr(self._owner, "alphas"):
            return len(self._owner.alphas)
        return 0

    def __iter__(self):
        if self._owner is not None and hasattr(self._owner, "alphas"):
            return iter(self._owner.alphas)
        return iter([])


@dataclass
class MatLaw87:
    """/MAT/LAW87 or /MAT/BARLAT2000 / /MAT/BARLAT_2000 / /MAT/BARLAT2000_2D (M183, M564):
    Barlat 2000 (Yld2000-2d) plane-stress anisotropic plasticity material model.

    Upstream Fortran:
      starter/source/materials/mat/mat087/hm_read_mat87.F90
      radioss140/MAT/matl87_barlat.cfg
      radioss2025/MAT/matl87_barlat.cfg
    """
    id: int = 0
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    iflag: int = 0
    iflagsr: int = 0
    invc: float = 0.0
    invp: float = 0.0
    flag_fit: int = 0
    al1: float = 1.0
    al2: float = 1.0
    al3: float = 1.0
    al4: float = 1.0
    al5: float = 1.0
    al6: float = 1.0
    al7: float = 1.0
    al8: float = 1.0
    fisokin: float = 0.0
    ikin: int = 1
    expa: float = 2.0
    fcut: float = 0.0
    fsmooth: int = 0
    nrate: int = 0
    aswift: float = 0.0
    nexp: float = 0.0
    alpha: float = 0.0
    epso: float = 0.0
    qvoce: float = 0.0
    beta: float = 0.0
    ko: float = 0.0
    ckh: tuple = (0.0, 0.0, 0.0, 0.0)
    akh: tuple = (0.0, 0.0, 0.0, 0.0)
    title: str = ""
    law: int = 87
    law_name: str = "LAW87"
    refer_rho: float = 0.0
    fail: Optional[Any] = None
    eos: Optional[Any] = None
    params: Optional[dict] = None
    # Extended / fitting / tabulated parameters:
    sigma_00: float = 0.0
    sigma_45: float = 0.0
    sigma_90: float = 0.0
    sigma_b: float = 0.0
    r_00: float = 1.0
    r_45: float = 1.0
    r_90: float = 1.0
    r_b: float = 1.0
    curves: list[MatLaw87Curve] = field(default_factory=list)
    tab_id0: int = 0
    fscale0: float = 1.0
    epsd0: float = 0.0
    tab_id45: int = 0
    fscale45: float = 1.0
    epsd45: float = 0.0
    tab_id90: int = 0
    fscale90: float = 1.0
    epsd90: float = 0.0

    def __init__(
        self,
        id: int = 0,
        rho: float = 0.0,
        e: float = 0.0,
        nu: float = 0.0,
        iflag: int = 0,
        iflagsr: int = 0,
        invc: float = 0.0,
        invp: float = 0.0,
        flag_fit: int = 0,
        al1: float = 1.0,
        al2: float = 1.0,
        al3: float = 1.0,
        al4: float = 1.0,
        al5: float = 1.0,
        al6: float = 1.0,
        al7: float = 1.0,
        al8: float = 1.0,
        fisokin: float = 0.0,
        ikin: int = 1,
        expa: float = 2.0,
        fcut: float = 0.0,
        fsmooth: int = 0,
        nrate: int = 0,
        aswift: float = 0.0,
        nexp: float = 0.0,
        alpha: float = 0.0,
        epso: float = 0.0,
        qvoce: float = 0.0,
        beta: float = 0.0,
        ko: float = 0.0,
        ckh: tuple = (0.0, 0.0, 0.0, 0.0),
        akh: tuple = (0.0, 0.0, 0.0, 0.0),
        title: str = "",
        law: int = 87,
        law_name: str = "LAW87",
        refer_rho: float = 0.0,
        fail: Optional[Any] = None,
        eos: Optional[Any] = None,
        params: Optional[dict] = None,
        **kwargs: Any,
    ):
        self.id = id
        self.rho = kwargs.get("rho0", rho)
        self.refer_rho = kwargs.get("rhor", refer_rho)
        self.e = kwargs.get("E", e)
        self.nu = kwargs.get("Nu", nu)
        self.iflag = iflag
        self.iflagsr = kwargs.get("vflag", kwargs.get("vp", iflagsr))
        self.invc = kwargs.get("strain1", kwargs.get("c", invc))
        self.invp = kwargs.get("exp1", kwargs.get("p", invp))
        self.flag_fit = kwargs.get("ifit", flag_fit)

        if "alphas" in kwargs and isinstance(kwargs["alphas"], (list, tuple)) and len(kwargs["alphas"]) >= 8:
            self.al1 = float(kwargs["alphas"][0])
            self.al2 = float(kwargs["alphas"][1])
            self.al3 = float(kwargs["alphas"][2])
            self.al4 = float(kwargs["alphas"][3])
            self.al5 = float(kwargs["alphas"][4])
            self.al6 = float(kwargs["alphas"][5])
            self.al7 = float(kwargs["alphas"][6])
            self.al8 = float(kwargs["alphas"][7])
        elif isinstance(alpha, (list, tuple)) and len(alpha) >= 8:
            self.al1 = float(alpha[0])
            self.al2 = float(alpha[1])
            self.al3 = float(alpha[2])
            self.al4 = float(alpha[3])
            self.al5 = float(alpha[4])
            self.al6 = float(alpha[5])
            self.al7 = float(alpha[6])
            self.al8 = float(alpha[7])
        else:
            self.al1 = kwargs.get("a1", kwargs.get("alpha1", al1))
            self.al2 = kwargs.get("a2", kwargs.get("alpha2", al2))
            self.al3 = kwargs.get("a3", kwargs.get("alpha3", al3))
            self.al4 = kwargs.get("a4", kwargs.get("alpha4", al4))
            self.al5 = kwargs.get("a5", kwargs.get("alpha5", al5))
            self.al6 = kwargs.get("a6", kwargs.get("alpha6", al6))
            self.al7 = kwargs.get("a7", kwargs.get("alpha7", al7))
            self.al8 = kwargs.get("a8", kwargs.get("alpha8", al8))

        self.fisokin = kwargs.get("chard", fisokin)
        self.ikin = ikin
        self.expa = kwargs.get("exp_a", kwargs.get("a_exp", kwargs.get("a", expa)))
        self.fcut = kwargs.get("f_cut", fcut)
        self.fsmooth = kwargs.get("f_smooth", fsmooth)
        self.nrate = nrate
        self.aswift = kwargs.get("a_swift", aswift)
        self.nexp = kwargs.get("n_hard", kwargs.get("n", nexp))
        if isinstance(alpha, (int, float)):
            self.alpha = AlphaFloat(kwargs.get("alpha_vol", float(alpha)), self)
        else:
            self.alpha = AlphaFloat(kwargs.get("alpha_vol", 0.0), self)
        self.epso = kwargs.get("eps0", epso)
        self.qvoce = kwargs.get("q_voce", qvoce)
        self.beta = beta
        self.ko = kwargs.get("k0", ko)
        self.ckh = ckh
        self.akh = akh
        self.title = title
        self.law = law
        self.law_name = law_name
        self.fail = fail
        self.eos = eos

        self.sigma_00 = kwargs.get("sigma_00", 0.0)
        self.sigma_45 = kwargs.get("sigma_45", 0.0)
        self.sigma_90 = kwargs.get("sigma_90", 0.0)
        self.sigma_b = kwargs.get("sigma_b", 0.0)
        self.r_00 = kwargs.get("r_00", 1.0)
        self.r_45 = kwargs.get("r_45", 1.0)
        self.r_90 = kwargs.get("r_90", 1.0)
        self.r_b = kwargs.get("r_b", 1.0)

        self.curves = list(kwargs.get("curves", []))
        self.tab_id0 = kwargs.get("tab_id0", 0)
        self.fscale0 = kwargs.get("fscale0", 1.0)
        self.epsd0 = kwargs.get("epsd0", 0.0)
        self.tab_id45 = kwargs.get("tab_id45", 0)
        self.fscale45 = kwargs.get("fscale45", 1.0)
        self.epsd45 = kwargs.get("epsd45", 0.0)
        self.tab_id90 = kwargs.get("tab_id90", 0)
        self.fscale90 = kwargs.get("fscale90", 1.0)
        self.epsd90 = kwargs.get("epsd90", 0.0)

        self._extra_params: dict[str, Any] = dict(params) if isinstance(params, dict) else {}
        for k, v in kwargs.items():
            if not hasattr(self, k):
                self._extra_params[k] = v

    @property
    def rho0(self) -> float:
        return self.rho

    @rho0.setter
    def rho0(self, val: float) -> None:
        self.rho = val

    @property
    def rhor(self) -> float:
        return self.refer_rho if self.refer_rho > 0.0 else self.rho

    @rhor.setter
    def rhor(self, val: float) -> None:
        self.refer_rho = val

    @property
    def E(self) -> float:
        return self.e

    @E.setter
    def E(self, val: float) -> None:
        self.e = val

    @property
    def Nu(self) -> float:
        return self.nu

    @Nu.setter
    def Nu(self, val: float) -> None:
        self.nu = val

    @property
    def G(self) -> float:
        return 0.5 * self.e / (1.0 + self.nu) if (1.0 + self.nu) > 0.0 else 0.0

    @property
    def bulk(self) -> float:
        denom = 3.0 * (1.0 - 2.0 * self.nu)
        return self.e / denom if abs(denom) > 1e-12 else 0.0

    @property
    def K(self) -> float:
        return self.bulk

    @property
    def sound_speed_shell(self) -> CallableFloat:
        """Shell plane-stress sound speed: c = sqrt(E / (rho * (1 - nu^2)))."""
        rho_val = self.rhor if self.refer_rho > 0.0 else self.rho
        denom = rho_val * (1.0 - self.nu ** 2)
        if denom > 0.0 and self.e > 0.0:
            import math
            return CallableFloat(math.sqrt(self.e / denom))
        return CallableFloat(0.0)

    @property
    def sound_speed(self) -> CallableFloat:
        return self.sound_speed_shell

    @property
    def Lp(self) -> np.ndarray:
        """First linear transformation matrix (3x3) for Barlat 2000 (Fortran hm_read_mat87.F90 lines 412-417)."""
        lp = np.zeros((3, 3), dtype=float)
        lp[0, 0] = 2.0 * self.al1 / 3.0
        lp[0, 1] = -self.al1 / 3.0
        lp[1, 0] = -self.al2 / 3.0
        lp[1, 1] = 2.0 * self.al2 / 3.0
        lp[2, 2] = self.al7
        return lp

    @property
    def Lpp(self) -> np.ndarray:
        """Second linear transformation matrix (3x3) for Barlat 2000 (Fortran hm_read_mat87.F90 lines 428-433)."""
        lpp = np.zeros((3, 3), dtype=float)
        lpp[0, 0] = (-2.0 * self.al3 + 2.0 * self.al4 + 8.0 * self.al5 - 2.0 * self.al6) / 9.0
        lpp[0, 1] = (self.al3 - 4.0 * self.al4 - 4.0 * self.al5 + 4.0 * self.al6) / 9.0
        lpp[1, 0] = (4.0 * self.al3 - 4.0 * self.al4 - 4.0 * self.al5 + self.al6) / 9.0
        lpp[1, 1] = (-2.0 * self.al3 + 8.0 * self.al4 + 2.0 * self.al5 - 2.0 * self.al6) / 9.0
        lpp[2, 2] = self.al8
        return lpp

    @property
    def alphas(self) -> list[float]:
        return [self.al1, self.al2, self.al3, self.al4, self.al5, self.al6, self.al7, self.al8]

    @property
    def a_exp(self) -> int:
        return int(round(self.expa))

    @property
    def a_swift(self) -> float:
        return self.aswift

    @property
    def q_voce(self) -> float:
        return self.qvoce

    @property
    def k0(self) -> float:
        return self.ko

    @property
    def eps0(self) -> float:
        return self.epso

    @property
    def ifit(self) -> int:
        return self.flag_fit

    @property
    def vp(self) -> int:
        return self.iflagsr

    @property
    def vflag(self) -> int:
        return self.iflagsr

    @property
    def strain1(self) -> float:
        return self.invc

    @property
    def c(self) -> float:
        return self.invc

    @property
    def exp1(self) -> float:
        return self.invp

    @property
    def p(self) -> float:
        return self.invp

    @property
    def chard(self) -> float:
        return self.fisokin

    @property
    def exp_a(self) -> float:
        return self.expa

    @property
    def alpha_vol(self) -> float:
        return float(self.alpha)

    @property
    def n_hard(self) -> float:
        return self.nexp

    @property
    def funct_ids(self) -> list[int]:
        return [getattr(c, "fct_id", getattr(c, "fid", 0)) for c in self.curves]

    @property
    def rates(self) -> list[float]:
        return [getattr(c, "epsp", getattr(c, "rate", 0.0)) for c in self.curves]

    @property
    def yfac(self) -> list[float]:
        return [getattr(c, "fscale", getattr(c, "scale", 1.0)) for c in self.curves]

    @property
    def params(self) -> Dict[str, Any]:
        p = {
            "rho": self.rho,
            "rho0": self.rho,
            "refer_rho": self.refer_rho,
            "rhor": self.rhor,
            "e": self.e,
            "E": self.e,
            "nu": self.nu,
            "Nu": self.nu,
            "iflag": self.iflag,
            "iflagsr": self.iflagsr,
            "vp": self.iflagsr,
            "vflag": self.iflagsr,
            "invc": self.invc,
            "strain1": self.invc,
            "c": self.invc,
            "invp": self.invp,
            "exp1": self.invp,
            "p": self.invp,
            "flag_fit": self.flag_fit,
            "ifit": self.flag_fit,
            "al1": self.al1,
            "al2": self.al2,
            "al3": self.al3,
            "al4": self.al4,
            "al5": self.al5,
            "al6": self.al6,
            "al7": self.al7,
            "al8": self.al8,
            "alphas": self.alphas,
            "fisokin": self.fisokin,
            "chard": self.fisokin,
            "ikin": self.ikin,
            "expa": self.expa,
            "exp_a": self.expa,
            "a_exp": self.a_exp,
            "fcut": self.fcut,
            "fsmooth": self.fsmooth,
            "nrate": self.nrate,
            "aswift": self.aswift,
            "a_swift": self.aswift,
            "nexp": self.nexp,
            "n_hard": self.nexp,
            "alpha": self.alpha,
            "alpha_vol": self.alpha,
            "epso": self.epso,
            "eps0": self.epso,
            "qvoce": self.qvoce,
            "q_voce": self.qvoce,
            "beta": self.beta,
            "ko": self.ko,
            "k0": self.ko,
            "ckh": self.ckh,
            "akh": self.akh,
            "curves": self.curves,
            "sigma_00": self.sigma_00,
            "sigma_45": self.sigma_45,
            "sigma_90": self.sigma_90,
            "sigma_b": self.sigma_b,
            "r_00": self.r_00,
            "r_45": self.r_45,
            "r_90": self.r_90,
            "r_b": self.r_b,
            "G": self.G,
            "bulk": self.bulk,
            "K": self.K,
            "sound_speed": self.sound_speed,
            "sound_speed_shell": self.sound_speed_shell,
            "title": self.title,
            "law": self.law,
            "law_name": self.law_name,
        }
        if hasattr(self, "_extra_params") and self._extra_params:
            p.update(self._extra_params)
        return p

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        p = self.params
        if key in p:
            return p[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            if not hasattr(self, "_extra_params"):
                self._extra_params = {}
            self._extra_params[key] = value

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or key in self.params

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        import dataclasses
        k = [f.name for f in dataclasses.fields(self)]
        k.extend([
            "rho0", "rhor", "E", "Nu", "G", "bulk", "K", "sound_speed", "sound_speed_shell",
            "Lp", "Lpp", "alphas", "a_exp", "a_swift", "q_voce", "k0", "eps0",
            "ifit", "vp", "vflag", "strain1", "exp1", "chard", "exp_a", "alpha_vol", "n_hard"
        ])
        if hasattr(self, "_extra_params") and self._extra_params:
            k.extend(self._extra_params.keys())
        return list(dict.fromkeys(k))

    def values(self) -> list[Any]:
        return [self[k] for k in self.keys()]

    def items(self) -> list[tuple[str, Any]]:
        return [(k, self[k]) for k in self.keys()]

    def __len__(self) -> int:
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())


MatBarlatYld2000 = MatLaw87
MatBarlat2000 = MatLaw87
MatBarlat20002D = MatLaw87
MaterialLaw87 = MatLaw87


# M569: /MAT/LAW95 (/MAT/BERGSTROM_BOYCE) defined below (see MatLaw95)


@dataclass
class MatLaw163:
    """/MAT/LAW163 or /MAT/CRUSHABLE_FOAM (M183, M560): Crushable foam material model."""
    id: int = 0
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    tsc: float = 0.0
    damp: float = 0.10
    ncycle: int = 12
    tab_id: int = 0
    epsd_ref: float = 0.0
    fscale: float = 1.0
    srclmt: float = 1.0e20
    nrs: int = 0
    title: str = ""
    law: int = 163
    law_name: str = "LAW163"
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.params, dict):
            self.params = {}
        if self.damp == 0.0:
            self.damp = 0.10
        if self.ncycle == 0:
            self.ncycle = 12
        if self.srclmt == 0.0:
            self.srclmt = 1.0e20
        if self.fscale == 0.0:
            self.fscale = 1.0
        self.nrs = max(min(int(self.nrs), 1), 0)

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def E(self) -> float:
        return self.e

    @property
    def G(self) -> float:
        nu_clamped = min(max(self.nu, 0.0), 0.499)
        denom = 2.0 * (1.0 + nu_clamped)
        return self.e / denom if abs(denom) > 1e-15 else 0.0

    @property
    def bulk(self) -> float:
        nu_clamped = min(max(self.nu, 0.0), 0.499)
        denom = 3.0 * (1.0 - 2.0 * nu_clamped)
        return self.e / denom if abs(denom) > 1e-15 else 0.0

    @property
    def K(self) -> float:
        return self.bulk

    @property
    def cii(self) -> float:
        nu_clamped = min(max(self.nu, 0.0), 0.499)
        denom = (1.0 + nu_clamped) * (1.0 - 2.0 * nu_clamped)
        lam = self.e * nu_clamped / denom if abs(denom) > 1e-15 else 0.0
        return lam + 2.0 * self.G

    @property
    def cij(self) -> float:
        nu_clamped = min(max(self.nu, 0.0), 0.499)
        denom = (1.0 + nu_clamped) * (1.0 - 2.0 * nu_clamped)
        return self.e * nu_clamped / denom if abs(denom) > 1e-15 else 0.0

    @property
    def sound_speed(self) -> CallableFloat:
        import math
        nu_clamped = min(max(self.nu, 0.0), 0.499)
        denom = self.rho * (1.0 - nu_clamped * nu_clamped)
        if denom > 0.0 and self.e > 0.0:
            return CallableFloat(math.sqrt(self.e / denom))
        return CallableFloat(0.0)

    @property
    def sound_speed_solid(self) -> CallableFloat:
        import math
        if self.rho > 0.0 and self.cii > 0.0:
            return CallableFloat(math.sqrt(self.cii / self.rho))
        return CallableFloat(0.0)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.params:
            return self.params[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        self.params[key] = value

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or (key in self.params)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.params.get(key, default)

    def keys(self) -> list[str]:
        base_keys = [
            "id", "rho", "e", "nu", "tsc", "damp", "ncycle",
            "tab_id", "epsd_ref", "fscale", "srclmt", "nrs", "title",
            "law", "law_name", "rho0", "E", "G", "bulk", "K", "cii", "cij",
            "sound_speed", "sound_speed_solid",
        ]
        for k in self.params.keys():
            if k not in base_keys:
                base_keys.append(k)
        return base_keys

    def values(self) -> list[Any]:
        return [self.get(k) for k in self.keys()]

    def items(self) -> list[tuple[str, Any]]:
        return [(k, self.get(k)) for k in self.keys()]


MatCrushableFoam = MatLaw163
MatCrushFoam = MatLaw163


@dataclass
class MatLaw169:
    """/MAT/LAW169 or /MAT/ARUP_ADHESIVE (M183): 3D cohesive adhesive material model."""
    id: int
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    sht_sl: float = 0.0
    tenmax: float = 0.0
    gcten: float = 0.0
    shrmax: float = 0.0
    gcshr: float = 0.0
    pwrt: int = 1
    pwrs: int = 1
    shrp: float = 0.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def pr(self) -> float:
        return self.nu


MatArupAdhesive = MatLaw169
MatCohTab3D = MatLaw169
MatCoh3D = MatLaw169


# =========================================================================
# M184: Steinberg-Guinan Plasticity, SAMP Plasticity, Sandwich Shell,
#       Fabric Shell, Composite Stack & Crushing Spring Suite
# =========================================================================

@dataclass
class MatLaw49:
    """/MAT/LAW49 or /MAT/STEINB (M184/M557): Steinberg-Guinan high-pressure plasticity model."""
    id: int = 0
    rho: float = 0.0
    refer_rho: float = 0.0
    e0: float = 0.0
    nu: float = 0.0
    sigy: float = 0.0
    beta: float = 0.0
    n: float = 0.0
    eps_max: float = 0.0
    sigma_max: float = 0.0
    t0: float = 0.0
    tmelt: float = 0.0
    rhoc_p: float = 0.0
    pmin: float = 0.0
    b1: float = 0.0
    b2: float = 0.0
    h: float = 0.0
    f: float = 0.0
    title: str = ""
    law: int = 49
    law_name: str = "LAW49"
    unit_id: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho
        if self.eps_max == 0.0:
            self.eps_max = 1.0e20
        if self.sigma_max == 0.0:
            self.sigma_max = 1.0e20
        if self.t0 == 0.0:
            self.t0 = 300.0
        if self.tmelt == 0.0:
            self.tmelt = 1.0e20
        if self.pmin == 0.0:
            self.pmin = -1.0e20

        core: Dict[str, Any] = {
            "rho": self.rho,
            "rho0": self.rho0,
            "refer_rho": self.refer_rho,
            "rhor": self.rhor,
            "e0": self.e0,
            "e": self.e,
            "E": self.E,
            "nu": self.nu,
            "Nu": self.Nu,
            "sigy": self.sigy,
            "sigma_0": self.sigma_0,
            "sig0": self.sigma_0,
            "beta": self.beta,
            "n": self.n,
            "hard": self.hard,
            "eps_max": self.eps_max,
            "sigma_max": self.sigma_max,
            "t0": self.t0,
            "tmelt": self.tmelt,
            "rhoc_p": self.rhoc_p,
            "pmin": self.pmin,
            "b1": self.b1,
            "b2": self.b2,
            "h": self.h,
            "f": self.f,
            "G": self.G,
            "G0": self.G0,
            "g0": self.G0,
            "bulk": self.bulk,
            "K": self.bulk,
            "C1": self.C1,
        }
        for k, v in core.items():
            if k not in self.params:
                self.params[k] = v

    @property
    def rho0(self) -> float:
        return self.rho

    @rho0.setter
    def rho0(self, val: float) -> None:
        self.rho = float(val)

    @property
    def rhor(self) -> float:
        return self.refer_rho if self.refer_rho != 0.0 else self.rho

    @rhor.setter
    def rhor(self, val: float) -> None:
        self.refer_rho = float(val)

    @property
    def e(self) -> float:
        return self.e0

    @e.setter
    def e(self, val: float) -> None:
        self.e0 = float(val)

    @property
    def E(self) -> float:
        return self.e0

    @E.setter
    def E(self, val: float) -> None:
        self.e0 = float(val)

    @property
    def Nu(self) -> float:
        return self.nu

    @Nu.setter
    def Nu(self, val: float) -> None:
        self.nu = float(val)

    @property
    def sigma_0(self) -> float:
        return self.sigy

    @sigma_0.setter
    def sigma_0(self, val: float) -> None:
        self.sigy = float(val)

    @property
    def hard(self) -> float:
        return self.n

    @hard.setter
    def hard(self, val: float) -> None:
        self.n = float(val)

    @property
    def G(self) -> float:
        """Elastic shear modulus: G0 = E / (2 * (1 + nu))."""
        return self.e0 / (2.0 * (1.0 + self.nu)) if (1.0 + self.nu) != 0.0 else 0.0

    @property
    def G0(self) -> float:
        return self.G

    @property
    def g0(self) -> float:
        return self.G

    @property
    def bulk(self) -> float:
        """Elastic bulk modulus: K = E / (3 * (1 - 2 * nu))."""
        denom = 3.0 * (1.0 - 2.0 * self.nu)
        return self.e0 / denom if denom != 0.0 else 0.0

    @property
    def K(self) -> float:
        return self.bulk

    @property
    def C1(self) -> float:
        return self.bulk

    @property
    def sound_speed(self) -> CallableFloat:
        """1D acoustic sound speed: c = sqrt(E / rho0)."""
        rho_val = self.rho0
        c = (self.e0 / rho_val)**0.5 if rho_val > 0.0 and self.e0 > 0.0 else 0.0
        return CallableFloat(c)

    @property
    def sound_speed_solid(self) -> CallableFloat:
        """Solid sound speed: c = sqrt((C1 + 4/3 G) / rho0) per matl49_steinb.cfg DRAWABLES."""
        rho_val = self.rho0
        c = 0.0
        if rho_val > 0.0:
            c2 = (self.bulk + (4.0 / 3.0) * self.G) / rho_val
            if c2 > 0.0:
                c = c2**0.5
        return CallableFloat(c)

    # Mapping protocol
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.params:
            return self.params[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.params[key] = value

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or (key in self.params)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> List[str]:
        import dataclasses
        k = [f.name for f in dataclasses.fields(self)]
        k.extend(["rho0", "rhor", "e", "E", "Nu", "sigma_0", "hard", "G", "G0", "bulk", "C1", "sound_speed", "sound_speed_solid"])
        k.extend(list(self.params.keys()))
        return list(dict.fromkeys(k))


MatSteinb = MatLaw49
MatSteinberg = MatLaw49
MatSteinbergGuinan = MatLaw49


@dataclass
class MatLaw76:
    """/MAT/LAW76 or /MAT/SAMP (M184): Semi-Analytical Model for Plastics (SAMP)."""
    id: int
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    fun_d1: int = 0
    fun_d2: int = 0
    fun_d3: int = 0
    fun_d4: int = 0
    fscale11: float = 1.0
    fscale22: float = 1.0
    fscale33: float = 1.0
    fscale12: float = 1.0
    facx: float = 1.0
    mat_nut: float = 0.0
    fun_b5: int = 0
    mat_pscale: float = 1.0
    israte: int = 0
    asrate: float = 0.0
    epsilon_f: float = 0.0
    epsilon_0: float = 0.0
    dc: float = 0.0
    fun_a1: int = 0
    fun_a2: int = 0
    fun_a3: int = 0
    scale: float = 1.0
    iform: int = 0
    iflag: int = 0
    gflag: int = 0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def nu_p(self) -> float:
        return self.mat_nut


MatSamp = MatLaw76
MatPlasSamp = MatLaw76
MatSampPlas = MatLaw76


# ============================================================================
# M185: LAW60 (PLAS_T3), LAW63 (HANSEL), LAW48 (ZHAO), LAW26 (SESAM),
#       PROP TYPE12 (SPR_PUL), PROP TYPE15 (POROUS), PROP TYPE28 (NSTRAND)
# ============================================================================

@dataclass(init=False)
class MatLaw60:
    """``/MAT/LAW60``, ``/MAT/PLAS_T3``, or ``/MAT/FABRIC``: Tabulated temperature/rate plasticity."""
    id: int
    rho: float
    ref_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    eps_p_max: float = 1.0e30
    eps_t1: float = 1.0e30
    eps_t2: float = 2.0e30
    nfunc: int = 5
    fsmooth: int = 0
    mat_hard: float = 0.0
    fcut: float = 1.0e30
    xr_fun: int = 0
    ifunce: int = 0
    mat_fscale: float = 1.0
    einf: float = 0.0
    ce: float = 0.0
    funcs: list[int] = field(default_factory=list)
    fscales: list[float] = field(default_factory=list)
    rates: list[float] = field(default_factory=list)
    title: str = ""

    def __init__(
        self,
        id: int = 0,
        rho: float = 0.0,
        ref_rho: float = 0.0,
        e: float = 0.0,
        nu: float = 0.0,
        eps_p_max: float = 1.0e30,
        eps_t1: float = 1.0e30,
        eps_t2: float = 2.0e30,
        nfunc: int = 5,
        fsmooth: int = 0,
        mat_hard: float = 0.0,
        fcut: float = 1.0e30,
        xr_fun: int = 0,
        ifunce: int = 0,
        mat_fscale: float = 1.0,
        einf: float = 0.0,
        ce: float = 0.0,
        funcs: Optional[list[int]] = None,
        fscales: Optional[list[float]] = None,
        rates: Optional[list[float]] = None,
        title: str = "",
        **kwargs,
    ):
        self.id = id
        self.rho = rho
        if "refer_rho" in kwargs and ref_rho == 0.0:
            ref_rho = kwargs["refer_rho"]
        self.ref_rho = ref_rho
        self.e = e
        self.nu = nu
        self.eps_p_max = eps_p_max
        self.eps_t1 = eps_t1
        self.eps_t2 = eps_t2
        self.nfunc = nfunc
        self.fsmooth = fsmooth
        if "chard" in kwargs and mat_hard == 0.0:
            mat_hard = kwargs["chard"]
        self.mat_hard = mat_hard
        self.fcut = fcut
        self.xr_fun = xr_fun
        self.ifunce = ifunce
        if "fpscale" in kwargs and mat_fscale == 1.0:
            mat_fscale = kwargs["fpscale"]
        self.mat_fscale = mat_fscale
        self.einf = einf
        self.ce = ce
        if funcs is None:
            funcs = kwargs.get("fun_ids", [])
        self.funcs = list(funcs)
        self.fscales = list(fscales) if fscales is not None else []
        if rates is None:
            rates = kwargs.get("eps_rates", [])
        self.rates = list(rates)
        self.title = title

    @property
    def refer_rho(self) -> float:
        return self.ref_rho

    @refer_rho.setter
    def refer_rho(self, v: float) -> None:
        self.ref_rho = v

    @property
    def rho0(self) -> float:
        return self.ref_rho if self.ref_rho != 0.0 else self.rho

    @property
    def E(self) -> float:
        return self.e

    @E.setter
    def E(self, v: float) -> None:
        self.e = v

    @property
    def hard(self) -> float:
        return self.mat_hard

    @property
    def chard(self) -> float:
        return self.mat_hard

    @chard.setter
    def chard(self, v: float) -> None:
        self.mat_hard = v

    @property
    def fpscale(self) -> float:
        return self.mat_fscale

    @fpscale.setter
    def fpscale(self, v: float) -> None:
        self.mat_fscale = v

    @property
    def fun_ids(self) -> list[int]:
        return self.funcs

    @fun_ids.setter
    def fun_ids(self, v: list[int]) -> None:
        self.funcs = v

    @property
    def eps_rates(self) -> list[float]:
        return self.rates

    @eps_rates.setter
    def eps_rates(self, v: list[float]) -> None:
        self.rates = v


MatPlasT3 = MatLaw60
MatPlastT3 = MatLaw60
MatMaxwell = MatLaw60
MatFabric = MatLaw60


@dataclass
class MatLaw63:
    """``/MAT/LAW63`` or ``/MAT/HANSEL``: Hänsel transformation plasticity."""
    id: int = 0
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    cp: float = 0.0
    a: float = 0.0
    b: float = 0.0
    q: float = 0.0
    c: float = 0.0
    d: float = 0.0
    p: float = 0.0
    ahs: float = 0.0
    bhs: float = 0.0
    m: float = 0.0
    n: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    dh: float = 0.0
    vm0: float = 0.0
    eps0: float = 0.0
    t0: float = 0.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.refer_rho if self.refer_rho != 0.0 else self.rho

    @property
    def mat_t0(self) -> float:
        return self.t0


MatHansel = MatLaw63
MatPlasHansel = MatLaw63
MatTransfoPlas = MatLaw63


@dataclass
class MatLaw48:
    """``/MAT/LAW48`` or ``/MAT/ZHAO``: Zhao strain-rate hardening plasticity."""
    id: int = 0
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    chard: float = 0.0
    sig_max: float = 0.0
    c: float = 0.0
    d: float = 0.0
    m: float = 0.0
    e1: float = 0.0
    k: float = 0.0
    eps_rate_0: float = 0.0
    fcut: float = 0.0
    eps_max: float = 0.0
    eps_t1: float = 0.0
    eps_t2: float = 0.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.refer_rho if self.refer_rho != 0.0 else self.rho

    @property
    def ref_rho(self) -> float:
        return self.refer_rho

    @ref_rho.setter
    def ref_rho(self, val: float) -> None:
        self.refer_rho = val

    @property
    def sigy(self) -> float:
        return self.a

    @sigy.setter
    def sigy(self, val: float) -> None:
        self.a = val

    @property
    def ca(self) -> float:
        return self.a

    @ca.setter
    def ca(self, val: float) -> None:
        self.a = val

    @property
    def cb(self) -> float:
        return self.b

    @cb.setter
    def cb(self, val: float) -> None:
        self.b = val

    @property
    def cn(self) -> float:
        return self.n

    @cn.setter
    def cn(self, val: float) -> None:
        self.n = val

    @property
    def cc(self) -> float:
        return self.c

    @cc.setter
    def cc(self, val: float) -> None:
        self.c = val

    @property
    def cd(self) -> float:
        return self.d

    @cd.setter
    def cd(self, val: float) -> None:
        self.d = val

    @property
    def cm(self) -> float:
        return self.m

    @cm.setter
    def cm(self, val: float) -> None:
        self.m = val

    @property
    def ce(self) -> float:
        return self.e1

    @ce.setter
    def ce(self, val: float) -> None:
        self.e1 = val

    @property
    def ck(self) -> float:
        return self.k

    @ck.setter
    def ck(self, val: float) -> None:
        self.k = val


    @property
    def hard(self) -> float:
        return self.n

    @hard.setter
    def hard(self, val: float) -> None:
        self.n = val

    @property
    def mat_hard(self) -> float:
        return self.chard

    @mat_hard.setter
    def mat_hard(self, val: float) -> None:
        self.chard = val

    @property
    def fisokin(self) -> float:
        return self.chard

    @fisokin.setter
    def fisokin(self, val: float) -> None:
        self.chard = val

    @property
    def E(self) -> float:
        return self.e

    @E.setter
    def E(self, val: float) -> None:
        self.e = val

    @property
    def eps0(self) -> float:
        return self.eps_rate_0

    @eps0.setter
    def eps0(self, val: float) -> None:
        self.eps_rate_0 = val

    @property
    def scale(self) -> float:
        return self.fcut

    @scale.setter
    def scale(self, val: float) -> None:
        self.fcut = val

    @property
    def sigma_max(self) -> float:
        return self.sig_max

    @sigma_max.setter
    def sigma_max(self, val: float) -> None:
        self.sig_max = val

    @property
    def eta1(self) -> float:
        return self.eps_t1

    @eta1.setter
    def eta1(self, val: float) -> None:
        self.eps_t1 = val

    @property
    def eta2(self) -> float:
        return self.eps_t2

    @eta2.setter
    def eta2(self, val: float) -> None:
        self.eps_t2 = val

    @property
    def G(self) -> float:
        return self.e / (2.0 * (1.0 + self.nu)) if (1.0 + self.nu) != 0.0 else 0.0

    @property
    def K(self) -> float:
        denom = 3.0 * (1.0 - 2.0 * self.nu)
        return self.e / denom if denom != 0.0 else 0.0

    @property
    def sound_speed(self) -> float:
        return (self.e / self.rho0)**0.5 if self.rho0 > 0.0 and self.e > 0.0 else 0.0

    def sound_speed_solid(self) -> float:
        if self.rho0 > 0.0 and self.e > 0.0:
            c1 = self.K
            g = self.G
            return ((c1 + 4.0 * g / 3.0) / self.rho0)**0.5
        return 0.0

    def sound_speed_shell(self) -> float:
        if self.rho0 > 0.0 and self.e > 0.0 and (1.0 - self.nu**2) > 0.0:
            return (self.e / (self.rho0 * (1.0 - self.nu**2)))**0.5
        return 0.0


MatZhao = MatLaw48
MatPlasZhao = MatLaw48


@dataclass
class MatLaw26:
    """``/MAT/LAW26`` or ``/MAT/SESAM``: SESAME equation of state & hydrodynamic constitutive model."""
    id: int = 0
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    eps_max: float = 0.0
    sig_max: float = 0.0
    e0: float = 0.0
    sesam301: str = ""
    c: float = 0.0
    eps0: float = 0.0
    m: float = 0.0
    tmelt: float = 0.0
    tmax: float = 0.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.refer_rho if self.refer_rho != 0.0 else self.rho

    @property
    def sigy(self) -> float:
        return self.a

    @property
    def hard(self) -> float:
        return self.n


MatSesam = MatLaw26
MatSesame = MatLaw26


# --- M186: Hydrodynamic Fluid, Boundary Layer, Foam-Air, Multi-Material, Barlat 3D, KJoint, Muscle, Stitch ---

@dataclass
class MatLaw6:
    """``/MAT/LAW6`` or ``/MAT/VISC_FLUID`` / ``/MAT/HYDRO`` / ``/MAT/K-EPS``: Hydrodynamic fluid with EOS & turbulence."""
    id: int = 0
    rho: float = 0.0
    rho_ref: float = 0.0
    nu: float = 0.0
    c0: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    pmin: float = 0.0
    psh: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    e0: float = 0.0
    r0k0: float = 0.0
    ssl: float = 0.0
    c_mu: float = 0.0
    sig_k: float = 0.0
    sig_eps: float = 0.0
    bulk_ratio: float = 0.0
    c1_e: float = 0.0
    c2_e: float = 0.0
    c3_e: float = 0.0
    kappa: float = 0.0
    e_wall: float = 0.0
    alpha: float = 0.0
    gsi_t: float = 0.0
    title: str = ""


MatViscFluid = MatLaw6
MatHydro = MatLaw6
MatHydroVisc = MatLaw6


@dataclass
class MatLaw11:
    """``/MAT/LAW11`` or ``/MAT/BOUND`` / ``/MAT/B-K-EPS``: Boundary fluid & k-epsilon turbulence model."""
    id: int = 0
    rho: float = 0.0
    rho_ref: float = 0.0
    itype: int = 1
    psh: float = 0.0
    scale: float = 1.0
    node1: int = 0
    gamma: float = 1.4
    k_cdi: float = 0.0
    h: float = 0.0
    c1: float = 0.0
    fun_a1: int = 0
    fun_a2: int = 0
    pscale: float = 1.0
    fun_a6: int = 0
    e0: float = 0.0
    xt_fun: int = 0
    yt_fun: int = 0
    title: str = ""


MatBound = MatLaw11
MatBkEps = MatLaw11


@dataclass
class MatLaw77Curve:
    """Loading/unloading curve entry for /MAT/LAW77 (FOAM_AIR)."""
    fct_id: int = 0
    strain_rate: float = 0.0
    scale: float = 1.0


@dataclass
class MatLaw77:
    """``/MAT/LAW77`` or ``/MAT/FOAM_AIR`` / ``/MAT/FOAM_HYST``: Foam with gas cavity and hysteresis unloading."""
    id: int = 0
    rho: float = 0.0
    rho_ref: float = 0.0
    e0: float = 0.0
    nu: float = 0.0
    emax: float = 0.0
    epsmax: float = 0.0
    fcut: float = 0.0
    fsmooth: int = 0
    nload: int = 0
    nunload: int = 0
    iflag: int = 0
    shape: float = 0.0
    hyst: float = 0.0
    load_curves: list[MatLaw77Curve] = field(default_factory=list)
    unload_curves: list[MatLaw77Curve] = field(default_factory=list)
    rho_gas: float = 0.0
    p0: float = 0.0
    gamma: float = 1.4
    poros: float = 1.0
    rho_ext: float = 0.0
    pext: float = 0.0
    iclos: int = 0
    inc_gas: int = 0
    title: str = ""


MatFoamAir = MatLaw77
MatFoamHyst = MatLaw77


@dataclass
class MatMultiFluidFraction:
    """Sub-material fraction entry for /MAT/LAW151 (MULTIFLUID)."""
    mat_id: int = 0
    vol_frac: float = 0.0


@dataclass
class MatLaw151:
    """``/MAT/LAW151`` or ``/MAT/MULTIFLUID`` / ``/MAT/MULTI_MAT``: Multi-material mixture law."""
    id: int = 0
    fractions: list[MatMultiFluidFraction] = field(default_factory=list)
    title: str = ""


MatMultiMat = MatLaw151
MatMultifluidMat = MatLaw151


@dataclass
class MatLaw187Rate:
    """Rate-dependent yield function entry for /MAT/LAW187 (BARLAT20003D)."""
    fct_id: int = 0
    scale: float = 1.0
    strain_rate: float = 0.0


@dataclass
class MatLaw187:
    """``/MAT/LAW187`` or ``/MAT/BARLAT20003D`` / ``/MAT/BARLAT_3D``: Barlat 2000 3D anisotropic plasticity."""
    id: int = 0
    rho: float = 0.0
    rho_ref: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    iflag: int = 0
    vp: int = 0
    c: float = 0.0
    p_exp: float = 0.0
    alpha1: float = 1.0
    alpha2: float = 1.0
    alpha3: float = 1.0
    alpha4: float = 1.0
    alpha5: float = 1.0
    alpha6: float = 1.0
    alpha7: float = 1.0
    alpha8: float = 1.0
    alpha9: float = 1.0
    alpha10: float = 1.0
    alpha11: float = 1.0
    alpha12: float = 1.0
    a_exp: int = 8
    alpha_xy: float = 1.0
    n_exp: float = 0.0
    fcut: float = 0.0
    fsmooth: int = 0
    nrate: int = 0
    a_hard: float = 0.0
    eps0: float = 0.0
    q: float = 0.0
    b_hard: float = 0.0
    k0: float = 0.0
    rates: list[MatLaw187Rate] = field(default_factory=list)
    title: str = ""


MatBarlat20003D = MatLaw187
MatBarlat3D = MatLaw187
MatPlasBarlat3D = MatLaw187


# M187: Geotechnical, Hydrodynamic, Tabulated Plasticity & Advanced Joint/Interface Suite

@dataclass
class MatLaw3:
    """``/MAT/LAW3`` or ``/MAT/PLAS_BOST``: Elastoplastic material with Cowper-Symonds rate hardening."""
    id: int = 0
    rho0: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    sig_y: float = 0.0
    e_t: float = 0.0
    c: float = 0.0
    p: float = 0.0
    title: str = ""


MatPlasBost = MatLaw3


@dataclass
class MatLaw4:
    """``/MAT/LAW4`` or ``/MAT/HYD_JCOOK``: Hydrodynamic Johnson-Cook elastoplasticity with EOS."""
    id: int = 0
    rho_i: float = 0.0
    c0_eos: float = 0.0
    s_eos: float = 0.0
    gamma0: float = 0.0
    a_eos: float = 0.0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    c: float = 0.0
    eps_max: float = 0.0
    sig_max: float = 0.0
    t0: float = 0.0
    tm: float = 0.0
    m: float = 0.0
    cp: float = 0.0
    pmin: float = 0.0
    c0: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    title: str = ""


MatHydJcook = MatLaw4


@dataclass
class MatLaw5:
    """``/MAT/LAW5`` or ``/MAT/JCOOK_TAB``: Tabulated Johnson-Cook plasticity with scale functions."""
    id: int = 0
    rho0: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    c: float = 0.0
    sig_max: float = 0.0
    fct_id1: int = 0
    fct_id2: int = 0
    fct_id3: int = 0
    fct_id4: int = 0
    fct_id5: int = 0
    title: str = ""


MatJcookTab = MatLaw5


@dataclass
class MatLaw10:
    """``/MAT/LAW10`` or ``/MAT/SOIL`` / ``/MAT/SOIL_CONC``: Soil and crushable concrete model."""
    id: int = 0
    rho0: float = 0.0
    g: float = 0.0
    k: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    p_cut: float = 0.0
    p_min: float = 0.0
    fct_id_p: int = 0
    title: str = ""


MatSoil = MatLaw10
MatSoilConc = MatLaw10


@dataclass
class MatLaw14:
    """``/MAT/LAW14``, ``/MAT/CAM_CLAY`` (M187), or ``/MAT/COMPSO`` (M189/M547)."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    # Cam-Clay fields (M187)
    g: float = 0.0
    nu: float = 0.0
    m: float = 0.0
    lamda: float = 0.0
    lambda_: float = 0.0
    kappa: float = 0.0
    e0: float = 0.0
    pc0: float = 0.0
    # Composite Solid fields (M189/M547)
    ea: float = 0.0
    eb: float = 0.0
    ec: float = 0.0
    prab: float = 0.0
    prbc: float = 0.0
    prca: float = 0.0
    gab: float = 0.0
    gbc: float = 0.0
    gca: float = 0.0
    sigt1: float = 0.0
    sigt2: float = 0.0
    sigt3: float = 0.0
    damage: float = 0.0
    delta: float = 0.0
    beta: float = 0.0
    cb: float = 0.0
    hard: float = 0.0
    cn: float = 0.0
    sig_max: float = 0.0
    fmax: float = 0.0
    wpref: float = 0.0
    wplaref: float = 0.0
    sigyt1: float = 0.0
    sigyt2: float = 0.0
    sigyc1: float = 0.0
    sigyc2: float = 0.0
    sigt12: float = 0.0
    sigc12: float = 0.0
    sigt23: float = 0.0
    sigc23: float = 0.0
    sigyt12: float = 0.0
    sigyc12: float = 0.0
    sigyt23: float = 0.0
    sigyc23: float = 0.0
    alpha_fib: float = 0.0
    alpha: float = 0.0
    e_fib: float = 0.0
    efib: float = 0.0
    src: float = 0.0
    cc: float = 0.0
    srp: float = 0.0
    eps0: float = 0.0
    strflag: int = 0
    icc: int = 0
    title: str = ""
    law: int = 14
    law_name: str = "LAW14"

    def __post_init__(self):
        if not self.lambda_ and self.lamda:
            self.lambda_ = self.lamda
        elif not self.lamda and self.lambda_:
            self.lamda = self.lambda_
        if not self.delta and self.damage:
            self.delta = self.damage
        elif not self.damage and self.delta:
            self.damage = self.delta
        if not self.cb and self.beta:
            self.cb = self.beta
        elif not self.beta and self.cb:
            self.beta = self.cb
        if not self.cn and self.hard:
            self.cn = self.hard
        elif not self.hard and self.cn:
            self.hard = self.cn
        if not self.fmax and self.sig_max:
            self.fmax = self.sig_max
        elif not self.sig_max and self.fmax:
            self.sig_max = self.fmax
        if not self.wplaref and self.wpref:
            self.wplaref = self.wpref
        elif not self.wpref and self.wplaref:
            self.wpref = self.wplaref
        if not self.alpha and self.alpha_fib:
            self.alpha = self.alpha_fib
        elif not self.alpha_fib and self.alpha:
            self.alpha_fib = self.alpha
        if not self.efib and self.e_fib:
            self.efib = self.e_fib
        elif not self.e_fib and self.efib:
            self.e_fib = self.efib
        if not self.cc and self.src:
            self.cc = self.src
        elif not self.src and self.cc:
            self.src = self.cc
        if not self.eps0 and self.srp:
            self.eps0 = self.srp
        elif not self.srp and self.eps0:
            self.srp = self.eps0
        if not self.icc and self.strflag:
            self.icc = self.strflag
        elif not self.strflag and self.icc:
            self.strflag = self.icc
        if not self.sigyt12 and self.sigt12:
            self.sigyt12 = self.sigt12
        elif not self.sigt12 and self.sigyt12:
            self.sigt12 = self.sigyt12
        if not self.sigyc12 and self.sigc12:
            self.sigyc12 = self.sigc12
        elif not self.sigc12 and self.sigyc12:
            self.sigc12 = self.sigyc12
        if not self.sigyt23 and self.sigt23:
            self.sigyt23 = self.sigt23
        elif not self.sigt23 and self.sigyt23:
            self.sigt23 = self.sigyt23
        if not self.sigyc23 and self.sigc23:
            self.sigyc23 = self.sigc23
        elif not self.sigc23 and self.sigyc23:
            self.sigc23 = self.sigyc23

    @property
    def lam(self) -> float:
        return self.lamda

    @lam.setter
    def lam(self, val: float) -> None:
        self.lamda = val
        self.lambda_ = val

    @property
    def rho(self) -> float:
        return self.rho0

    @rho.setter
    def rho(self, val: float) -> None:
        self.rho0 = val

    @property
    def refer_rho(self) -> float:
        return self.rhor

    @refer_rho.setter
    def refer_rho(self, val: float) -> None:
        self.rhor = val

    @property
    def b(self) -> float:
        return self.cb

    @b.setter
    def b(self, val: float) -> None:
        self.cb = val
        self.beta = val

    @property
    def n(self) -> float:
        return self.cn

    @n.setter
    def n(self, val: float) -> None:
        self.cn = val
        self.hard = val

    @property
    def c(self) -> float:
        return self.cc

    @c.setter
    def c(self, val: float) -> None:
        self.cc = val
        self.src = val

    @property
    def sig_t1(self) -> float:
        return self.sigt1

    @sig_t1.setter
    def sig_t1(self, val: float) -> None:
        self.sigt1 = val

    @property
    def sig_t2(self) -> float:
        return self.sigt2

    @sig_t2.setter
    def sig_t2(self, val: float) -> None:
        self.sigt2 = val

    @property
    def sig_t3(self) -> float:
        return self.sigt3

    @sig_t3.setter
    def sig_t3(self, val: float) -> None:
        self.sigt3 = val

    @property
    def e11(self) -> float:
        return self.ea

    @e11.setter
    def e11(self, val: float) -> None:
        self.ea = val

    @property
    def e22(self) -> float:
        return self.eb

    @e22.setter
    def e22(self, val: float) -> None:
        self.eb = val

    @property
    def e33(self) -> float:
        return self.ec

    @e33.setter
    def e33(self, val: float) -> None:
        self.ec = val

    @property
    def nu12(self) -> float:
        return self.prab

    @nu12.setter
    def nu12(self, val: float) -> None:
        self.prab = val

    @property
    def nu23(self) -> float:
        return self.prbc

    @nu23.setter
    def nu23(self, val: float) -> None:
        self.prbc = val

    @property
    def nu31(self) -> float:
        return self.prca

    @nu31.setter
    def nu31(self, val: float) -> None:
        self.prca = val

    @property
    def g12(self) -> float:
        return self.gab

    @g12.setter
    def g12(self, val: float) -> None:
        self.gab = val

    @property
    def g23(self) -> float:
        return self.gbc

    @g23.setter
    def g23(self, val: float) -> None:
        self.gbc = val

    @property
    def g31(self) -> float:
        return self.gca

    @g31.setter
    def g31(self, val: float) -> None:
        self.gca = val

    @property
    def sig_1yt(self) -> float:
        return self.sigyt1

    @sig_1yt.setter
    def sig_1yt(self, val: float) -> None:
        self.sigyt1 = val

    @property
    def sig_2yt(self) -> float:
        return self.sigyt2

    @sig_2yt.setter
    def sig_2yt(self, val: float) -> None:
        self.sigyt2 = val

    @property
    def sig_1yc(self) -> float:
        return self.sigyc1

    @sig_1yc.setter
    def sig_1yc(self, val: float) -> None:
        self.sigyc1 = val

    @property
    def sig_2yc(self) -> float:
        return self.sigyc2

    @sig_2yc.setter
    def sig_2yc(self, val: float) -> None:
        self.sigyc2 = val

    @property
    def sig_12yt(self) -> float:
        return self.sigt12

    @sig_12yt.setter
    def sig_12yt(self, val: float) -> None:
        self.sigt12 = val
        self.sigyt12 = val

    @property
    def sig_12yc(self) -> float:
        return self.sigc12

    @sig_12yc.setter
    def sig_12yc(self, val: float) -> None:
        self.sigc12 = val
        self.sigyc12 = val

    @property
    def sig_23yt(self) -> float:
        return self.sigt23

    @sig_23yt.setter
    def sig_23yt(self, val: float) -> None:
        self.sigt23 = val
        self.sigyt23 = val

    @property
    def sig_23yc(self) -> float:
        return self.sigc23

    @sig_23yc.setter
    def sig_23yc(self, val: float) -> None:
        self.sigc23 = val
        self.sigyc23 = val



MatCamClay = MatLaw14
MatCamclay = MatLaw14
MatCompso = MatLaw14
MatCompSol = MatLaw14


@dataclass
class MatLaw21:
    """``/MAT/LAW21`` or ``/MAT/DPRAG``: Drucker-Prager geological / soil / concrete material model.

    Upstream Fortran reference:
      hm_read_mat21.F and matl21_dprag.cfg (radioss110).

    Yield surface:
      F(p, J2) = sqrt(2 * J2) - (a0 + a1 * p + a2 * p^2) <= 0
      with amax limit on sqrt(2 * J2).
    """
    id: int = 0
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    amax: float = 1.0e20
    ifunc: int = 0
    c1: float = 0.0
    pfscale: float = 1.0
    pmin: float = -1.0e30
    pext: float = 0.0
    bunl: float = 0.0
    mumax: float = 1.0e20
    w: float = 0.0
    d: float = 0.0
    x0: float = 0.0
    title: str = ""
    law: int = 21
    law_name: str = "LAW21"
    unit_id: Optional[int] = None
    comments: List[str] = field(default_factory=list)
    fail: Optional[Any] = None
    eos: Optional[Any] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.params, dict):
            self.params = {}
        core = {
            "rho": self.rho,
            "rho0": self.rho0,
            "refer_rho": self.refer_rho,
            "rhor": self.rhor,
            "e": self.e,
            "E": self.e,
            "nu": self.nu,
            "Nu": self.nu,
            "a0": self.a0,
            "a1": self.a1,
            "a2": self.a2,
            "amax": self.amax,
            "ifunc": self.ifunc,
            "c1": self.c1,
            "pfscale": self.pfscale,
            "pmin": self.pmin,
            "pext": self.pext,
            "bunl": self.bunl,
            "mumax": self.mumax,
            "w": self.w,
            "d": self.d,
            "x0": self.x0,
            "G": self.G,
        }
        for k, v in core.items():
            if k not in self.params:
                self.params[k] = v

    # Property helpers
    @property
    def rho0(self) -> float:
        return self.rho

    @rho0.setter
    def rho0(self, val: float) -> None:
        self.rho = val

    @property
    def rhor(self) -> float:
        return self.refer_rho if self.refer_rho != 0.0 else self.rho

    @rhor.setter
    def rhor(self, val: float) -> None:
        self.refer_rho = val

    @property
    def G(self) -> float:
        return self.e / (2.0 * (1.0 + self.nu)) if (1.0 + self.nu) != 0.0 else 0.0

    @property
    def E(self) -> float:
        return self.e

    @E.setter
    def E(self, val: float) -> None:
        self.e = val

    @property
    def Nu(self) -> float:
        return self.nu

    @Nu.setter
    def Nu(self, val: float) -> None:
        self.nu = val

    @property
    def sound_speed(self) -> CallableFloat:
        rho_val = self.rho0
        c = (self.e / rho_val)**0.5 if rho_val > 0.0 and self.e > 0.0 else 0.0
        return CallableFloat(c)


    @property
    def sound_speed_solid(self) -> CallableFloat:
        """Solid sound speed: c = sqrt((C1 + 4/3 G) / rho0) per matl21_dprag.cfg DRAWABLES."""
        rho_val = self.rho0
        c = 0.0
        if rho_val > 0.0:
            bulk = self.c1
            if bulk <= 0.0 and self.e > 0.0 and (1.0 - 2.0 * self.nu) > 0.0:
                bulk = self.e / (3.0 * (1.0 - 2.0 * self.nu))
            c2 = (bulk + 4.0 * self.G / 3.0) / rho_val
            if c2 > 0.0:
                c = c2**0.5
        return CallableFloat(c)

    # Mapping protocol
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.params:
            return self.params[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.params[key] = value

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or (key in self.params)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> List[str]:
        import dataclasses
        k = [f.name for f in dataclasses.fields(self)]
        k.extend(["rho0", "rhor", "G", "sound_speed", "sound_speed_solid"])
        k.extend(list(self.params.keys()))
        return list(dict.fromkeys(k))


MatDprag = MatLaw21
MatDuckhub = MatLaw21


@dataclass
class MatLaw32:
    """``/MAT/LAW32`` or ``/MAT/HILL``: Hill (1948) anisotropic plasticity model (M542)."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    sigy: float = 1.0e30
    beta: float = 0.0
    hard: float = 1.0
    eps: float = 1.0e30
    sig: float = 1.0e30
    srp: float = 1.0
    src: float = 0.0
    r00: float = 1.0
    r45: float = 1.0
    r90: float = 1.0
    title: str = ""
    law: int = 32
    law_name: str = "LAW32"
    fail: Optional[Any] = None
    eos: Optional[Any] = None

    # Mathematical aliases matching upstream hm_read_mat32.F
    @property
    def a(self) -> float:
        return self.sigy

    @property
    def b(self) -> float:
        return self.beta

    @property
    def n(self) -> float:
        return self.hard

    @property
    def eps_max(self) -> float:
        return self.eps

    @property
    def sig_max(self) -> float:
        return self.sig

    @property
    def eps0(self) -> float:
        return self.srp

    @property
    def eps_dot_0(self) -> float:
        return self.srp

    @property
    def m(self) -> float:
        return self.src


MatHill = MatLaw32


@dataclass
class MatLaw37:
    """``/MAT/LAW37`` or ``/MAT/BIQUAD``: Biquadratic anisotropic yield criterion."""
    id: int = 0
    rho0: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    c6: float = 0.0
    c7: float = 0.0
    c8: float = 0.0
    p: float = 0.0
    q: float = 0.0
    title: str = ""


MatBiquad = MatLaw37


# ----------------------------------------------------------------------------
# M188: Composite, Honeycomb, Concrete Damage & Advanced Shell/Solid Props
# ----------------------------------------------------------------------------

@dataclass
class MatLaw12:
    """``/MAT/LAW12`` or ``/MAT/3PARBI`` / ``/MAT/3D_COMP`` / ``/MAT/COMP_3D`` / ``/MAT/RAGAB``: 3D composite elasto-plastic with cracking damage."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e11: float = 0.0
    e22: float = 0.0
    e33: float = 0.0
    nu12: float = 0.0
    nu23: float = 0.0
    nu31: float = 0.0
    g12: float = 0.0
    g23: float = 0.0
    g31: float = 0.0
    sig_t1: float = 0.0
    sig_t2: float = 0.0
    sig_t3: float = 0.0
    delta: float = 0.05
    b: float = 0.0
    n: float = 1.0
    fmax: float = 1.0e10
    wplaref: float = 1.0
    sig_1yt: float = 0.0
    sig_2yt: float = 0.0
    sig_1yc: float = 0.0
    sig_2yc: float = 0.0
    sig_12yt: float = 0.0
    sig_12yc: float = 0.0
    sig_23yt: float = 0.0
    sig_23yc: float = 0.0
    sig_3yt: float = 0.0
    sig_3yc: float = 0.0
    sig_13yt: float = 0.0
    sig_13yc: float = 0.0
    alpha: float = 0.0
    efib: float = 0.0
    c: float = 0.0
    eps0: float = 0.0
    icc: int = 1
    title: str = ""
    law_name: str = "LAW12"

    @property
    def cb(self) -> float:
        return self.b

    @cb.setter
    def cb(self, val: float) -> None:
        self.b = val

    @property
    def cn(self) -> float:
        return self.n

    @cn.setter
    def cn(self, val: float) -> None:
        self.n = val

    @property
    def wpref(self) -> float:
        return self.wplaref

    @wpref.setter
    def wpref(self, val: float) -> None:
        self.wplaref = val

    @property
    def strflag(self) -> int:
        return self.icc

    @strflag.setter
    def strflag(self, val: int) -> None:
        self.icc = val

    @property
    def rho(self) -> float:
        return self.rho0

    @rho.setter
    def rho(self, val: float) -> None:
        self.rho0 = val

    @property
    def refer_rho(self) -> float:
        return self.rhor

    @refer_rho.setter
    def refer_rho(self, val: float) -> None:
        self.rhor = val

    @property
    def ea(self) -> float:
        return self.e11

    @ea.setter
    def ea(self, val: float) -> None:
        self.e11 = val

    @property
    def eb(self) -> float:
        return self.e22

    @eb.setter
    def eb(self, val: float) -> None:
        self.e22 = val

    @property
    def ec(self) -> float:
        return self.e33

    @ec.setter
    def ec(self, val: float) -> None:
        self.e33 = val

    @property
    def prab(self) -> float:
        return self.nu12

    @prab.setter
    def prab(self, val: float) -> None:
        self.nu12 = val

    @property
    def prbc(self) -> float:
        return self.nu23

    @prbc.setter
    def prbc(self, val: float) -> None:
        self.nu23 = val

    @property
    def prca(self) -> float:
        return self.nu31

    @prca.setter
    def prca(self, val: float) -> None:
        self.nu31 = val

    @property
    def gab(self) -> float:
        return self.g12

    @gab.setter
    def gab(self, val: float) -> None:
        self.g12 = val

    @property
    def gbc(self) -> float:
        return self.g23

    @gbc.setter
    def gbc(self, val: float) -> None:
        self.g23 = val

    @property
    def gca(self) -> float:
        return self.g31

    @gca.setter
    def gca(self, val: float) -> None:
        self.g31 = val

    @property
    def sigt1(self) -> float:
        return self.sig_t1

    @sigt1.setter
    def sigt1(self, val: float) -> None:
        self.sig_t1 = val

    @property
    def sigt2(self) -> float:
        return self.sig_t2

    @sigt2.setter
    def sigt2(self, val: float) -> None:
        self.sig_t2 = val

    @property
    def sigt3(self) -> float:
        return self.sig_t3

    @sigt3.setter
    def sigt3(self, val: float) -> None:
        self.sig_t3 = val

    @property
    def sigyt1(self) -> float:
        return self.sig_1yt

    @sigyt1.setter
    def sigyt1(self, val: float) -> None:
        self.sig_1yt = val

    @property
    def sigyt2(self) -> float:
        return self.sig_2yt

    @sigyt2.setter
    def sigyt2(self, val: float) -> None:
        self.sig_2yt = val

    @property
    def sigyc1(self) -> float:
        return self.sig_1yc

    @sigyc1.setter
    def sigyc1(self, val: float) -> None:
        self.sig_1yc = val

    @property
    def sigyc2(self) -> float:
        return self.sig_2yc

    @sigyc2.setter
    def sigyc2(self, val: float) -> None:
        self.sig_2yc = val

    @property
    def sigyt12(self) -> float:
        return self.sig_12yt

    @sigyt12.setter
    def sigyt12(self, val: float) -> None:
        self.sig_12yt = val

    @property
    def sigyc12(self) -> float:
        return self.sig_12yc

    @sigyc12.setter
    def sigyc12(self, val: float) -> None:
        self.sig_12yc = val

    @property
    def sigyt23(self) -> float:
        return self.sig_23yt

    @sigyt23.setter
    def sigyt23(self, val: float) -> None:
        self.sig_23yt = val

    @property
    def sigyc23(self) -> float:
        return self.sig_23yc

    @sigyc23.setter
    def sigyc23(self, val: float) -> None:
        self.sig_23yc = val

    @property
    def sigyt3(self) -> float:
        return self.sig_3yt

    @sigyt3.setter
    def sigyt3(self, val: float) -> None:
        self.sig_3yt = val

    @property
    def sigyc3(self) -> float:
        return self.sig_3yc

    @sigyc3.setter
    def sigyc3(self, val: float) -> None:
        self.sig_3yc = val

    @property
    def sigyt13(self) -> float:
        return self.sig_13yt

    @sigyt13.setter
    def sigyt13(self, val: float) -> None:
        self.sig_13yt = val

    @property
    def sigyc13(self) -> float:
        return self.sig_13yc

    @sigyc13.setter
    def sigyc13(self, val: float) -> None:
        self.sig_13yc = val

    @property
    def cc(self) -> float:
        return self.c

    @cc.setter
    def cc(self, val: float) -> None:
        self.c = val


Mat3parbi = MatLaw12
Mat3dComp = MatLaw12
MatComp3d = MatLaw12
MatRagab = MatLaw12


@dataclass
class MatLaw13:
    """``/MAT/LAW13`` or ``/MAT/HONEYCOMB`` / ``/MAT/RIGID``: Honeycomb crush / rigid material model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    title: str = ""


MatHoneycomb = MatLaw13


@dataclass
class MatLaw15:
    """``/MAT/LAW15`` or ``/MAT/CHANG`` / ``/MAT/CHANG_CHANG``: Chang-Chang composite failure model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e11: float = 0.0
    e22: float = 0.0
    nu12: float = 0.0
    g12: float = 0.0
    g23: float = 0.0
    g31: float = 0.0
    b: float = 0.0
    n: float = 0.0
    fmax: float = 0.0
    wpmax: float = 0.0
    wpref: float = 0.0
    ioff: int = 0
    sig_1yt: float = 0.0
    sig_2yt: float = 0.0
    sig_1yc: float = 0.0
    sig_2yc: float = 0.0
    alpha: float = 0.0
    sig_12yc: float = 0.0
    sig_12yt: float = 0.0
    c: float = 0.0
    eps_dot_0: float = 0.0
    icc: int = 0
    beta: float = 0.0
    tmax: float = 0.0
    s1: float = 0.0
    s2: float = 0.0
    s12: float = 0.0
    fsmooth: int = 0
    fcut: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    title: str = ""
    law: int = 15
    law_name: str = "LAW15"


MatChang = MatLaw15
MatChangChang = MatLaw15
MatPlasAniso = MatLaw15
MatCompChang = MatLaw15


@dataclass
class MatLaw18:
    """``/MAT/LAW18`` or ``/MAT/CONCR_DRA`` / ``/MAT/DRAGON`` / ``/MAT/THERM``: Concrete damage / thermal model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    spheat: float = 0.0
    a: float = 0.0
    b: float = 0.0
    fct_idt: int = 0
    t0: float = 0.0
    scale: float = 0.0
    fct_idsph: int = 0
    fct_idas: int = 0
    fscalesph: float = 0.0
    fscalee: float = 0.0
    fscalek: float = 0.0
    title: str = ""


MatConcrDra = MatLaw18
MatDragon = MatLaw18
MatTherm = MatLaw18


@dataclass
class MatLaw22:
    """``/MAT/LAW22`` (/MAT/DAMA, /MAT/PLAS_DAMA): Elastoplastic material law with progressive damage & softening slope."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    eps_max: float = 0.0
    sig_max: float = 0.0
    c: float = 0.0
    eps_dot_0: float = 0.0
    icc: int = 0
    eps_dam: float = 0.0
    e_tan: float = 0.0
    title: str = ""
    law: int = 22
    law_name: str = "LAW22"

    def __init__(
        self,
        id: int = 0,
        rho0: float = 0.0,
        rhor: float = 0.0,
        e: float = 0.0,
        nu: float = 0.0,
        a: float = 0.0,
        b: float = 0.0,
        n: float = 0.0,
        eps_max: float = 0.0,
        sig_max: float = 0.0,
        c: float = 0.0,
        eps_dot_0: float = 0.0,
        icc: int = 0,
        eps_dam: float = 0.0,
        e_tan: float = 0.0,
        title: str = "",
        law: int = 22,
        law_name: str = "LAW22",
        sigy: Optional[float] = None,
        beta: Optional[float] = None,
        e_t: Optional[float] = None,
        **kwargs,
    ):
        self.id = id
        self.rho0 = rho0
        self.rhor = rhor
        self.e = e
        self.nu = nu
        self.a = sigy if sigy is not None else a
        self.b = beta if beta is not None else b
        self.n = n
        self.eps_max = eps_max
        self.sig_max = sig_max
        self.c = c
        self.eps_dot_0 = eps_dot_0
        self.icc = icc
        self.eps_dam = eps_dam
        self.e_tan = e_t if e_t is not None else e_tan
        self.title = title
        self.law = law
        self.law_name = law_name

    @property
    def sigy(self) -> float:
        return self.a

    @sigy.setter
    def sigy(self, val: float) -> None:
        self.a = val

    @property
    def beta(self) -> float:
        return self.b

    @beta.setter
    def beta(self, val: float) -> None:
        self.b = val

    @property
    def e_t(self) -> float:
        return self.e_tan

    @e_t.setter
    def e_t(self, val: float) -> None:
        self.e_tan = val

    @property
    def eps_0(self) -> float:
        return self.eps_dot_0

    @eps_0.setter
    def eps_0(self, val: float) -> None:
        self.eps_dot_0 = val


MatTsaiWu = MatLaw22
MatDama = MatLaw22
MatPlasDama = MatLaw22


@dataclass
class MatLaw25:
    """``/MAT/LAW25`` or ``/MAT/COMP_PLAS`` / ``/MAT/COMPSH`` / ``/MAT/TSAI_WU`` / ``/MAT/CRASURV``:
    Composite anisotropic plasticity model (Tsai-Wu or CRASURV formulation)."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e11: float = 0.0
    e22: float = 0.0
    nu12: float = 0.0
    iform: int = 0
    e33: float = 0.0
    g12: float = 0.0
    g23: float = 0.0
    g31: float = 0.0
    eps_f1: float = 0.0
    eps_f2: float = 0.0
    eps_t1: float = 0.0
    eps_m1: float = 0.0
    eps_t2: float = 0.0
    eps_m2: float = 0.0
    dmax: float = 0.0
    wpmax: float = 0.0
    wpref: float = 0.0
    ioff: int = 0
    b: float = 0.0
    n: float = 0.0
    fmax: float = 0.0
    sig_1yt: float = 0.0
    sig_2yt: float = 0.0
    sig_1yc: float = 0.0
    sig_2yc: float = 0.0
    alpha: float = 0.0
    sig_12yc: float = 0.0
    sig_12yt: float = 0.0
    c: float = 0.0
    eps_rate_0: float = 0.0
    icc: int = 0
    title: str = ""
    law: int = 25
    law_name: str = "LAW25"
    fail: Optional[Any] = None
    eos: Optional[Any] = None

    # CRASURV formulation fields (iform=1)
    iflawp: int = 0
    # Dir 1 tension
    b_1t: float = 0.0
    n_1t: float = 1.0
    sig_1maxt: float = 0.0
    c_1t: float = 0.0
    eps_1t1: float = 0.0
    eps_2t1: float = 0.0
    sig_rst1: float = 0.0
    wpmax_t1: float = 0.0
    # Dir 2 tension
    b_2t: float = 0.0
    n_2t: float = 1.0
    sig_2maxt: float = 0.0
    c_2t: float = 0.0
    eps_1t2: float = 0.0
    eps_2t2: float = 0.0
    sig_rst2: float = 0.0
    wpmax_t2: float = 0.0
    # Dir 1 compression
    b_1c: float = 0.0
    n_1c: float = 1.0
    sig_1maxc: float = 0.0
    c_1c: float = 0.0
    eps_1c1: float = 0.0
    eps_2c1: float = 0.0
    sig_rsc1: float = 0.0
    wpmax_c1: float = 0.0
    # Dir 2 compression
    b_2c: float = 0.0
    n_2c: float = 1.0
    sig_2maxc: float = 0.0
    c_2c: float = 0.0
    eps_1c2: float = 0.0
    eps_2c2: float = 0.0
    sig_rsc2: float = 0.0
    wpmax_c2: float = 0.0
    # Dir 12 shear
    b_12t: float = 0.0
    n_12t: float = 1.0
    sig_12maxt: float = 0.0
    c_12t: float = 0.0
    eps_1t12: float = 0.0
    eps_2t12: float = 0.0
    sig_rst12: float = 0.0
    wpmax_t12: float = 0.0

    # Delamination and rate filtering
    gamma_ini: float = 0.0
    gamma_max: float = 0.0
    d3max: float = 0.0
    fsmooth: int = 0
    fcut: float = 0.0


MatCompPlas = MatLaw25
MatCompsh = MatLaw25
MatTsaiWu = MatLaw25
MatCrasurv = MatLaw25
MatCompositePlas = MatLaw25


@dataclass
class MatLaw28:
    """``/MAT/LAW28`` or ``/MAT/HONEYCOMB_SOL``: Solid honeycomb crush material model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e11: float = 0.0
    e22: float = 0.0
    e33: float = 0.0
    g12: float = 0.0
    g23: float = 0.0
    g31: float = 0.0
    fun_id11: int = 0
    fun_id22: int = 0
    fun_id33: int = 0
    iflag1: int = 0
    fscale11: float = 0.0
    fscale22: float = 0.0
    fscale33: float = 0.0
    eps_max11: float = 0.0
    eps_max22: float = 0.0
    eps_max33: float = 0.0
    fun_id12: int = 0
    fun_id23: int = 0
    fun_id31: int = 0
    iflag2: int = 0
    fscale12: float = 0.0
    fscale23: float = 0.0
    fscale31: float = 0.0
    eps_max12: float = 0.0
    eps_max23: float = 0.0
    eps_max31: float = 0.0
    title: str = ""

    @property
    def fun_a1(self) -> int:
        return self.fun_id11

    @property
    def fun_b1(self) -> int:
        return self.fun_id22

    @property
    def fun_a2(self) -> int:
        return self.fun_id33

    @property
    def gflag(self) -> int:
        return self.iflag1

    @property
    def epsr1(self) -> float:
        return self.eps_max11

    @property
    def epsr2(self) -> float:
        return self.eps_max22

    @property
    def epsr3(self) -> float:
        return self.eps_max33

    @property
    def fun_a3(self) -> int:
        return self.fun_id12

    @property
    def fun_b3(self) -> int:
        return self.fun_id23

    @property
    def fun_a4(self) -> int:
        return self.fun_id31

    @property
    def vflag(self) -> int:
        return self.iflag2

    @property
    def fscale13(self) -> float:
        return self.fscale31

    @property
    def epsr4(self) -> float:
        return self.eps_max12

    @property
    def epsr5(self) -> float:
        return self.eps_max23

    @property
    def epsr6(self) -> float:
        return self.eps_max31


MatHoneycombSol = MatLaw28





# -------------------------------------------------------------------------
# M189: Gurson, Gray Cast Iron, Composite Solid, Connector, Martensite Materials,
# Advanced Failure Criteria & Generalized Spring/Solid Properties
# -------------------------------------------------------------------------


class CallableFloat(float):
    """Float that is also callable returning itself (compatible with both property and method access)."""
    def __call__(self) -> float:
        return float(self)


@dataclass
class MatLaw52:
    """``/MAT/LAW52`` or ``/MAT/GURSON`` / ``/MAT/PLAS_GURS`` (M189/M554): Gurson porous metal plasticity.

    Fortran origin: ``starter/source/materials/mat/mat052/hm_read_mat52.F`` and
    ``radioss110/MAT/matl52_gurson.cfg`` / ``radioss130/MAT/matl52_gurson.cfg``.
    """
    id: int = 0
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    c: float = 0.0
    pc: float = 0.0
    q1: float = 0.0
    q2: float = 0.0
    q3: float = 0.0
    s_n: float = 0.0
    eps_n: float = 0.0
    f_i: float = 0.0
    f_n: float = 0.0
    f_c: float = 0.0
    f_f: float = 0.0
    iflag: int = 0
    fsmooth: int = 0
    fcut: float = 0.0
    itable: int = 0
    xfac: float = 1.0
    yfac: float = 1.0
    title: str = ""
    raw_fcut: float = 0.0
    raw_c: float = 0.0
    raw_pc: float = 0.0
    law: int = 52
    law_name: str = "LAW52"
    fail: Optional[Any] = None
    eos: Optional[Any] = None

    def __init__(
        self,
        id: int = 0,
        rho: float = 0.0,
        refer_rho: float = 0.0,
        e: float = 0.0,
        nu: float = 0.0,
        a: float = 0.0,
        b: float = 0.0,
        n: float = 0.0,
        c: float = 0.0,
        pc: float = 0.0,
        q1: float = 0.0,
        q2: float = 0.0,
        q3: float = 0.0,
        s_n: float = 0.0,
        eps_n: float = 0.0,
        f_i: float = 0.0,
        f_n: float = 0.0,
        f_c: float = 0.0,
        f_f: float = 0.0,
        iflag: int = 0,
        fsmooth: int = 0,
        fcut: float = 0.0,
        itable: int = 0,
        xfac: float = 1.0,
        yfac: float = 1.0,
        title: str = "",
        raw_fcut: float = 0.0,
        raw_c: float = 0.0,
        raw_pc: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.id = id
        if rho == 0.0 and "rho0" in kwargs:
            rho = kwargs["rho0"]
        if refer_rho == 0.0 and "rhor" in kwargs:
            refer_rho = kwargs["rhor"]
        self.rho = float(rho)
        self.refer_rho = float(refer_rho)
        self.e = float(e)
        self.nu = float(nu)
        self.a = float(a)
        self.b = float(b)
        self.n = float(n)
        self.c = float(c)
        self.pc = float(pc)
        self.q1 = float(q1)
        self.q2 = float(q2)
        self.q3 = float(q3)
        self.s_n = float(s_n)
        self.eps_n = float(eps_n)
        self.f_i = float(f_i)
        self.f_n = float(f_n)
        self.f_c = float(f_c)
        self.f_f = float(f_f)
        self.iflag = int(iflag)
        self.fsmooth = int(fsmooth)
        self.fcut = float(fcut)
        self.itable = int(itable)
        self.xfac = float(xfac)
        self.yfac = float(yfac)
        self.title = str(title)
        self.raw_fcut = float(raw_fcut)
        self.raw_c = float(raw_c)
        self.raw_pc = float(raw_pc)

    @property
    def rho0(self) -> float:
        return self.rho

    @rho0.setter
    def rho0(self, value: float) -> None:
        self.rho = value

    @property
    def rhor(self) -> float:
        return self.refer_rho

    @rhor.setter
    def rhor(self, value: float) -> None:
        self.refer_rho = value

    @property
    def yield_stress(self) -> float:
        return self.a

    @yield_stress.setter
    def yield_stress(self, value: float) -> None:
        self.a = value

    @property
    def hardening_b(self) -> float:
        return self.b

    @hardening_b.setter
    def hardening_b(self, value: float) -> None:
        self.b = value

    @property
    def hardening_n(self) -> float:
        return self.n

    @hardening_n.setter
    def hardening_n(self, value: float) -> None:
        self.n = value

    @property
    def fu(self) -> float:
        q1_val = self.q1 if self.q1 != 0.0 else 1.0e-20
        return 1.0 / q1_val

    @property
    def sound_speed(self) -> CallableFloat:
        rho_val = self.rho if self.rho > 0.0 else self.refer_rho
        if rho_val > 0.0 and self.e > 0.0:
            import math
            return CallableFloat(math.sqrt(self.e / rho_val))
        return CallableFloat(0.0)

    @property
    def sound_speed_solid(self) -> CallableFloat:
        """Solid sound speed: c = sqrt((K + 4/3 G) / rho0)."""
        rho_val = self.rho if self.rho > 0.0 else self.refer_rho
        if rho_val > 0.0 and self.e > 0.0 and (1.0 - 2.0 * self.nu) > 0.0 and (1.0 + self.nu) > 0.0:
            import math
            g = 0.5 * self.e / (1.0 + self.nu)
            c1 = self.e / (3.0 * (1.0 - 2.0 * self.nu))
            c = math.sqrt((c1 + 4.0 * g / 3.0) / rho_val)
            return CallableFloat(c)
        return CallableFloat(0.0)

    @property
    def sound_speed_shell(self) -> CallableFloat:
        """Shell plane-stress sound speed: c = sqrt(E / (rho0 * (1 - nu^2)))."""
        rho_val = self.rho if self.rho > 0.0 else self.refer_rho
        if rho_val > 0.0 and self.e > 0.0 and (1.0 - self.nu**2) > 0.0:
            import math
            c = math.sqrt(self.e / (rho_val * (1.0 - self.nu**2)))
            return CallableFloat(c)
        return CallableFloat(0.0)

    @property
    def E(self) -> float:
        return self.e

    @E.setter
    def E(self, value: float) -> None:
        self.e = value

    @property
    def G(self) -> float:
        return 0.5 * self.e / (1.0 + self.nu) if (1.0 + self.nu) > 0.0 else 0.0

    @property
    def K(self) -> float:
        denom = 3.0 * (1.0 - 2.0 * self.nu)
        return self.e / denom if abs(denom) > 1e-12 else self.e

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "E": self.e,
            "nu": self.nu,
            "a": self.a,
            "b": self.b,
            "n": self.n,
            "c": self.c,
            "pc": self.pc,
            "q1": self.q1,
            "q2": self.q2,
            "q3": self.q3,
            "s_n": self.s_n,
            "eps_n": self.eps_n,
            "f_i": self.f_i,
            "f_n": self.f_n,
            "f_c": self.f_c,
            "f_f": self.f_f,
            "iflag": self.iflag,
            "fsmooth": self.fsmooth,
            "fcut": self.fcut,
            "itable": self.itable,
            "xfac": self.xfac,
            "yfac": self.yfac,
            "yield_a": self.a,
            "hard_b": self.b,
            "hard_n": self.n,
            "rho": self.rho,
            "rho0": self.rho0,
            "f0": self.f_i,
            "f_0": self.f_i,
            "fi": self.f_i,
            "fn": self.f_n,
            "fc": self.f_c,
            "ff": self.f_f,
            "sn": self.s_n,
            "epsn": self.eps_n,
            "A": self.a,
            "B": self.b,
            "C": self.c,
            "P": self.pc,
            "fu": self.fu,
        }

    @property
    def f_0(self) -> float:
        return self.f_i

    @f_0.setter
    def f_0(self, value: float) -> None:
        self.f_i = value

    @property
    def f0(self) -> float:
        return self.f_i

    @f0.setter
    def f0(self, value: float) -> None:
        self.f_i = value

    @property
    def fi(self) -> float:
        return self.f_i

    @fi.setter
    def fi(self, value: float) -> None:
        self.f_i = value

    @property
    def fn(self) -> float:
        return self.f_n

    @fn.setter
    def fn(self, value: float) -> None:
        self.f_n = value

    @property
    def fc(self) -> float:
        return self.f_c

    @fc.setter
    def fc(self, value: float) -> None:
        self.f_c = value

    @property
    def ff(self) -> float:
        return self.f_f

    @ff.setter
    def ff(self, value: float) -> None:
        self.f_f = value

    @property
    def sn(self) -> float:
        return self.s_n

    @sn.setter
    def sn(self, value: float) -> None:
        self.s_n = value

    @property
    def epsn(self) -> float:
        return self.eps_n

    @epsn.setter
    def epsn(self, value: float) -> None:
        self.eps_n = value

    @property
    def A(self) -> float:
        return self.a

    @A.setter
    def A(self, value: float) -> None:
        self.a = value

    @property
    def B(self) -> float:
        return self.b

    @B.setter
    def B(self, value: float) -> None:
        self.b = value

    @property
    def C(self) -> float:
        return self.c

    @C.setter
    def C(self, value: float) -> None:
        self.c = value

    @property
    def P(self) -> float:
        return self.pc

    @P.setter
    def P(self, value: float) -> None:
        self.pc = value

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        p = self.params
        if key in p:
            return p[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Cannot set unknown attribute {key!r} on MatLaw52")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or key in self.params

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


MatGurson = MatLaw52
MatPlasGurs = MatLaw52


@dataclass
class MatLaw16:
    """``/MAT/LAW16`` or ``/MAT/GRAY``: Gray cast iron EOS and asymmetric plasticity model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    p0: float = 0.0
    c: float = 0.0
    s: float = 0.0
    gamma0: float = 0.0
    a: float = 0.0
    e0: float = 0.0
    v0: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    sig_y: float = 0.0
    beta: float = 0.0
    hard: float = 0.0
    sig_max: float = 0.0
    eps_max: float = 0.0
    title: str = ""


MatGray = MatLaw16
MatCastIron = MatLaw16


@dataclass
class MatLaw59:
    """``/MAT/LAW59`` or ``/MAT/CONNECT``: Connector / fastener material model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    g0: float = 0.0
    fsmooth: int = 0
    fcut: float = 0.0
    iflag: int = 0
    functions: List[Dict[str, Any]] = field(default_factory=list)
    title: str = ""


MatConnect = MatLaw59
MatConnector = MatLaw59


@dataclass
class MatLaw64:
    """``/MAT/LAW64``: Martensitic transformation plasticity model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    cp: float = 0.0
    d: float = 0.0
    n: float = 0.0
    md: float = 0.0
    v0: float = 0.0
    vmc: float = 0.0
    funct_id_0: int = 0
    funct_id_1: int = 0
    scale_0: float = 1.0
    scale_1: float = 1.0
    t_ini: float = 0.0
    title: str = ""


MatTransfoMart = MatLaw64
MatMartensite = MatLaw64


@dataclass
class FailWierzbicki:
    """``/FAIL/WIERZBICKI`` or ``/FAIL/MMC``: Modified Mohr-Coulomb ductile fracture model."""
    id: int = 0
    mat_id: int = 0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    m: float = 0.0
    n: float = 0.0
    ifail_sh: int = 0
    ifail_so: int = 0
    imoy: int = 0
    title: str = ""


FailMmc = FailWierzbicki


@dataclass
class FailWilkins:
    """``/FAIL/WILKINS``: Wilkins cumulative damage fracture model."""
    id: int = 0
    mat_id: int = 0
    alpha: float = 0.0
    beta: float = 0.0
    plim: float = 0.0
    df: float = 0.0
    ifail_sh: int = 0
    ifail_so: int = 0
    title: str = ""


@dataclass
class FailTbutcher:
    """``/FAIL/TBUTCHER``: Tuler-Butcher dynamic spall fracture model."""
    id: int = 0
    mat_id: int = 0
    lam: float = 1.0
    k: float = 1.0e30
    sigr: float = 0.0
    ifail_sh: int = 1
    ifail_so: int = 1
    title: str = ""


@dataclass
class MatLaw68:
    """``/MAT/LAW68`` or ``/MAT/COSSER``: 3D Cosserat continuum material model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e_11: float = 0.0
    e_22: float = 0.0
    e_33: float = 0.0
    g_12: float = 0.0
    g_23: float = 0.0
    g_31: float = 0.0
    fun_id11i: int = 0
    fun_id22i: int = 0
    fun_id33i: int = 0
    iflag1: int = 0
    fscale11i: float = 1.0
    fscale22i: float = 1.0
    fscale33i: float = 1.0
    eps_max11i: float = 0.0
    eps_max22i: float = 0.0
    eps_max33i: float = 0.0
    fun_id12i: int = 0
    fun_id23i: int = 0
    fun_id31i: int = 0
    iflag2: int = 0
    fscale12i: float = 1.0
    fscale23i: float = 1.0
    fscale31i: float = 1.0
    eps_max12i: float = 0.0
    eps_max23i: float = 0.0
    eps_max31i: float = 0.0
    fun_id21i: int = 0
    fun_id32i: int = 0
    fun_id13i: int = 0
    fscale21i: float = 1.0
    fscale32i: float = 1.0
    fscale13i: float = 1.0
    fun_id11r: int = 0
    fun_id22r: int = 0
    fun_id33r: int = 0
    fscale11r: float = 1.0
    fscale22r: float = 1.0
    fscale33r: float = 1.0
    eps_trans11r: float = 0.0
    eps_trans22r: float = 0.0
    eps_trans33r: float = 0.0
    fun_id12r: int = 0
    fun_id23r: int = 0
    fun_id31r: int = 0
    fscale12r: float = 1.0
    fscale23r: float = 1.0
    fscale31r: float = 1.0
    eps_trans12r: float = 0.0
    eps_trans23r: float = 0.0
    eps_trans31r: float = 0.0
    fun_id21r: int = 0
    fun_id32r: int = 0
    fun_id13r: int = 0
    fscale21r: float = 1.0
    fscale32r: float = 1.0
    fscale13r: float = 1.0
    title: str = ""


MatCosser = MatLaw68
MatCosserat = MatLaw68


@dataclass
class MatLaw72:
    """``/MAT/LAW72`` or ``/MAT/HILL_MMC``: Hill orthotropic plasticity with MMC ductile fracture."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    sig0: float = 0.0
    eps0: float = 0.0
    n: float = 0.0
    f: float = 0.0
    g: float = 0.0
    h: float = 0.0
    big_n: float = 0.0
    l: float = 0.0
    m: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    mmc_m: float = 0.0
    dc: float = 0.0
    title: str = ""


MatHillMmc = MatLaw72


@dataclass
class MatLaw65:
    """``/MAT/LAW65`` or ``/MAT/ELASTOMER``: 3D Elastomer hyperelastic material model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e0: float = 0.0
    nu: float = 0.0
    eps_max: float = 0.0
    nrate: int = 0
    fsmooth: int = 0
    fcut: float = 0.0
    rates: List[Dict[str, Any]] = field(default_factory=list)
    title: str = ""


MatElastomer = MatLaw65


class CallableFloat(float):
    """Float that is also callable returning itself (compatible with both property and method access)."""
    def __call__(self) -> float:
        return float(self)


@dataclass
class MatLaw58:
    """``/MAT/LAW58`` or ``/MAT/FABR_A`` (M553): Anisotropic fabric material model.

    Fortran origin: ``starter/source/materials/mat/mat058/hm_read_mat58.F``,
    ``engine/source/materials/mat/mat058/sig_mat58.F``, and
    ``radioss2017/MAT/matl58_fabr_a.cfg``.
    """
    id: int = 0
    title: str = ""
    rho: float = 0.0
    refer_rho: float = 0.0
    e1: float = 0.0
    b1: float = 0.0
    e2: float = 0.0
    b2: float = 0.0
    f: float = 0.01
    g0: float = 0.0
    gi: float = 0.0
    alpha: float = 0.0
    g5: float = 0.0
    isensor: int = 0
    df: float = 0.05
    ds: float = 0.0
    friction_phi: float = 0.0
    m58_zerostress: float = 0.0
    n1_warp: int = 1
    n2_weft: int = 1
    s1: float = 0.1
    s2: float = 0.1
    c4: float = 0.0
    c5: float = 0.0
    fun_a1: int = 0
    c1: float = 1.0
    fun_a2: int = 0
    c2: float = 1.0
    fun_a3: int = 0
    c3: float = 1.0
    fun_a4: int = 0
    scale4: float = 1.0
    fun_a5: int = 0
    scale5: float = 1.0
    fun_a6: int = 0
    scale6: float = 1.0

    @property
    def rho0(self) -> float:
        return self.rho

    @rho0.setter
    def rho0(self, value: float) -> None:
        self.rho = value

    @property
    def sound_speed_shell(self) -> CallableFloat:
        """Shell/membrane sound speed: c = sqrt(E / rho0) with E = max(E1/N1, E2/N2)."""
        import math
        rho_val = self.rho if self.rho > 0.0 else self.refer_rho
        if rho_val <= 0.0:
            return CallableFloat(0.0)
        nc = max(self.n1_warp, 1) if self.n1_warp != 0 else 1
        nt = max(self.n2_weft, 1) if self.n2_weft != 0 else 1
        kc = self.e1 / nc
        kt = self.e2 / nt
        young = max(kc, kt)
        if young <= 0.0:
            young = max(self.e1, self.e2)
        c = math.sqrt(young / rho_val) if young > 0.0 else 0.0
        return CallableFloat(c)

    # Backward compatibility aliases
    @property
    def rhor(self) -> float:
        return self.refer_rho

    @rhor.setter
    def rhor(self, val: float) -> None:
        self.refer_rho = val

    @property
    def ref_rho(self) -> float:
        return self.refer_rho

    @ref_rho.setter
    def ref_rho(self, val: float) -> None:
        self.refer_rho = val

    @property
    def flex(self) -> float:
        return self.f

    @flex.setter
    def flex(self, val: float) -> None:
        self.f = val

    @property
    def gt(self) -> float:
        return self.gi

    @gt.setter
    def gt(self, val: float) -> None:
        self.gi = val

    @property
    def alphat(self) -> float:
        return self.alpha

    @alphat.setter
    def alphat(self, val: float) -> None:
        self.alpha = val

    @property
    def sensor_id(self) -> int:
        return self.isensor

    @sensor_id.setter
    def sensor_id(self, val: int) -> None:
        self.isensor = val

    @property
    def gfrot(self) -> float:
        return self.friction_phi

    @gfrot.setter
    def gfrot(self, val: float) -> None:
        self.friction_phi = val

    @property
    def zero_stress(self) -> float:
        return self.m58_zerostress

    @zero_stress.setter
    def zero_stress(self, val: float) -> None:
        self.m58_zerostress = val

    @property
    def n1(self) -> int:
        return self.n1_warp

    @n1.setter
    def n1(self, val: int) -> None:
        self.n1_warp = val

    @property
    def n2(self) -> int:
        return self.n2_weft

    @n2.setter
    def n2(self, val: int) -> None:
        self.n2_weft = val

    @property
    def fun_id1(self) -> int:
        return self.fun_a1

    @fun_id1.setter
    def fun_id1(self, val: int) -> None:
        self.fun_a1 = val

    @property
    def fun_id2(self) -> int:
        return self.fun_a2

    @fun_id2.setter
    def fun_id2(self, val: int) -> None:
        self.fun_a2 = val

    @property
    def fun_id3(self) -> int:
        return self.fun_a3

    @fun_id3.setter
    def fun_id3(self, val: int) -> None:
        self.fun_a3 = val

    @property
    def fun_id4(self) -> int:
        return self.fun_a4

    @fun_id4.setter
    def fun_id4(self, val: int) -> None:
        self.fun_a4 = val

    @property
    def fun_id5(self) -> int:
        return self.fun_a5

    @fun_id5.setter
    def fun_id5(self, val: int) -> None:
        self.fun_a5 = val

    @property
    def fun_id6(self) -> int:
        return self.fun_a6

    @fun_id6.setter
    def fun_id6(self, val: int) -> None:
        self.fun_a6 = val

    @property
    def fscale1(self) -> float:
        return self.c1

    @fscale1.setter
    def fscale1(self, val: float) -> None:
        self.c1 = val

    @property
    def fscale2(self) -> float:
        return self.c2

    @fscale2.setter
    def fscale2(self, val: float) -> None:
        self.c2 = val

    @property
    def fscale3(self) -> float:
        return self.c3

    @fscale3.setter
    def fscale3(self, val: float) -> None:
        self.c3 = val

    @property
    def fscale4(self) -> float:
        return self.scale4

    @fscale4.setter
    def fscale4(self, val: float) -> None:
        self.scale4 = val

    @property
    def fscale5(self) -> float:
        return self.scale5

    @fscale5.setter
    def fscale5(self, val: float) -> None:
        self.scale5 = val

    @property
    def fscale6(self) -> float:
        return self.scale6

    @fscale6.setter
    def fscale6(self, val: float) -> None:
        self.scale6 = val

    @property
    def phi_lock(self) -> float:
        return self.alpha

    @phi_lock.setter
    def phi_lock(self, val: float) -> None:
        self.alpha = val

    @property
    def mu_frot(self) -> float:
        return self.friction_phi

    @mu_frot.setter
    def mu_frot(self, val: float) -> None:
        self.friction_phi = val

    @property
    def arel(self) -> float:
        return self.m58_zerostress

    @arel.setter
    def arel(self, val: float) -> None:
        self.m58_zerostress = val

    @property
    def a_rel(self) -> float:
        return self.m58_zerostress

    @a_rel.setter
    def a_rel(self, val: float) -> None:
        self.m58_zerostress = val

    @property
    def c6(self) -> float:
        return self.scale6

    @c6.setter
    def c6(self, val: float) -> None:
        self.scale6 = val

    @property
    def flex1(self) -> float:
        return self.c4

    @flex1.setter
    def flex1(self, val: float) -> None:
        self.c4 = val

    @property
    def flex2(self) -> float:
        return self.c5

    @flex2.setter
    def flex2(self, val: float) -> None:
        self.c5 = val

    @property
    def gsh(self) -> float:
        return self.g5

    @gsh.setter
    def gsh(self, val: float) -> None:
        self.g5 = val


MatFabrA = MatLaw58
MatFabricA = MatLaw58
class FabricAMaterial(Material):
    """Container for /MAT/LAW58 (/MAT/FABR_A) fabric material."""
    pass


MaterialLaw58 = FabricAMaterial


@dataclass
class MatLaw20:
    """``/MAT/LAW20`` or ``/MAT/BIMAT``: Bi-material mixture / layered material model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    mat_id1: int = 0
    mat_id2: int = 0
    alpha1: float = 0.0
    alpha2: float = 0.0
    title: str = ""


MatBimat = MatLaw20


@dataclass
class MatLaw38:
    """``/MAT/LAW38`` or ``/MAT/VISC_TAB``: Tabulated viscoelastic polymer/foam model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu_t: float = 0.0
    nu_c: float = 0.0
    rv: float = 0.0
    iflag: int = 0
    itotal: int = 0
    beta: float = 0.0
    h: float = 0.0
    r_d: float = 0.0
    k_r: int = 0
    k_d: int = 0
    instant_mod_upd: float = 0.0
    kair: int = 0
    np: int = 0
    pscale: float = 0.0
    p0: float = 0.0
    rp: float = 0.0
    pmax: float = 0.0
    phi: float = 0.0
    ful: int = 0
    alpha_unload: float = 0.0
    eps_unload: float = 0.0
    a: float = 0.0
    b: float = 0.0
    m_func: int = 0
    cutoff: float = 0.0
    iinsta: int = 0
    e_final: float = 0.0
    epsi_final: float = 0.0
    lamb: float = 0.0
    visc: float = 0.0
    tol: float = 0.0
    fscale_i: List[float] = field(default_factory=list)
    epsilon_i: List[float] = field(default_factory=list)
    funct_id_load: List[int] = field(default_factory=list)
    funct_id_unload: List[int] = field(default_factory=list)
    title: str = ""

    @property
    def fscale(self) -> List[float]:
        return self.fscale_i

    @property
    def epsilon(self) -> List[float]:
        return self.epsilon_i

    @property
    def nfunc(self) -> int:
        return self.m_func

    @property
    def nu(self) -> float:
        return max(self.nu_c, self.nu_t)


MatViscTab = MatLaw38


@dataclass
class MatLaw29:
    """``/MAT/LAW29`` or ``/MAT/FEM``: User/FEM material model interface."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    version: str = ""
    frelim: float = 0.0
    dtmin: float = 0.0
    nf: int = 0
    velsc: float = 0.0
    rstrat: float = 0.0
    rtemp: float = 0.0
    encrypt: int = 0
    param_1: int = 0
    el_young: float = 0.0
    el_poiss: float = 0.0
    el_bulkm: float = 0.0
    el_shear: float = 0.0
    el_ortho: int = 0
    el_shrco: float = 0.0
    param_2: int = 0
    param_3: int = 0
    pl_harde: int = 0
    pl_ortho: int = 0
    pl_iskin: int = 0
    pl_asymm: int = 0
    pl_waist: int = 0
    pl_biaxf: int = 0
    pl_compr: int = 0
    pl_damag: int = 0
    nf_curve: int = 0
    nf_ortho: int = 0
    nf_depen: int = 0
    sf_curve: int = 0
    sf_param: int = 0
    sf_postc: int = 0
    param_4: int = 0
    param_5: int = 0
    cr_harde: int = 0
    cr_ortho: int = 0
    cr_iskin: int = 0
    cr_postc: int = 0
    cr_param: int = 0
    cr_check: int = 0
    param_6: int = 0
    mf_init: int = 0
    title: str = ""


MatFem = MatLaw29
Mat29Fem = MatLaw29


@dataclass
class MatLaw34:
    """``/MAT/LAW34`` or ``/MAT/BOLTZMAN``: Boltzmann linear viscoelastic relaxation model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    k: float = 0.0
    g0: float = 0.0
    gl: float = 0.0
    beta: float = 0.0
    p0: float = 0.0
    phi: float = 0.0
    gamma0: float = 0.0
    title: str = ""


MatBoltzman = MatLaw34
MatBoltzmann = MatLaw34
MatViscMaxw = MatLaw34



@dataclass
class MatLaw23:
    """``/MAT/LAW23`` or ``/MAT/PLAS_DAMA``: Lemaitre ductile damage elastoplastic model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    eps_max: float = 0.0
    sig_max: float = 0.0
    c: float = 0.0
    eps_0: float = 0.0
    icc: int = 0
    eps_dam: float = 0.0
    e_t: float = 0.0
    title: str = ""


MatPlasDama = MatLaw23


@dataclass
class MatLaw78:
    """``/MAT/LAW78``: Rate-dependent elastoplastic constitutive law 78."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    eps_max: float = 0.0
    sig_max: float = 0.0
    y: float = 0.0
    b: float = 0.0
    c: float = 0.0
    h: float = 0.0
    b0: float = 0.0
    m: float = 0.0
    rsat: float = 0.0
    ea: float = 0.0
    ce: float = 0.0
    title: str = ""


@dataclass
class FailHashin:
    """``/FAIL/HASHIN``: Hashin 3D composite failure model."""
    mat_id: int = 0
    iform: int = 0
    ifail_sh: int = 0
    ifail_so: int = 0
    sigma_1t: float = 0.0
    sigma_2t: float = 0.0
    sigma_3t: float = 0.0
    sigma_1c: float = 0.0
    sigma_2c: float = 0.0
    sigma_c: float = 0.0
    sigma_12f: float = 0.0
    sigma_12m: float = 0.0
    sigma_23m: float = 0.0
    sigma_13m: float = 0.0
    phi: float = 0.0
    sdel: float = 0.0
    tau_max: float = 0.0
    title: str = ""


@dataclass
class FailTensstrain:
    """``/FAIL/TENSSTRAIN`` or ``/FAIL/TENSTRAIN``: Tensile strain failure model."""
    mat_id: int = 0
    eps_t1: float = 0.0
    eps_t2: float = 0.0
    eps_m1: float = 0.0
    fct_id: int = 0
    fscale: float = 1.0
    ifail_sh: int = 1
    ifail_so: int = 1
    d_adv: float = 0.0
    p_thickfail: float = 0.0
    title: str = ""


FailTenstrain = FailTensstrain


@dataclass
class FailEnergy:
    """``/FAIL/ENERGY``: Specific internal energy failure criterion."""
    mat_id: int = 0
    e1: float = 0.0
    e2: float = 0.0
    fct_id: int = 0
    title: str = ""


@dataclass
class FailUser:
    """``/FAIL/USER``: User-defined material failure model."""
    mat_id: int = 0
    user_type: str = ""
    cards: List[str] = field(default_factory=list)
    title: str = ""


# ==============================================================================
# Milestone M191: Advanced Materials, Multi-axial & Visual Failures, Preloads & BCs
# ==============================================================================

@dataclass
class MatLaw100:
    """``/MAT/LAW100`` or ``/MAT/VISC_HYP`` / ``/MAT/MNF``: Multi-Network Visco-Hyperelastic polymer model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    n_net: int = 0
    flag_he: int = 1
    flag_cr: int = 0
    c10: float = 0.0
    c01: float = 0.0
    c20: float = 0.0
    c11: float = 0.0
    c02: float = 0.0
    c30: float = 0.0
    c21: float = 0.0
    c12: float = 0.0
    c03: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0
    mue1: float = 0.0
    d: float = 0.0
    lambda_m: float = 7.0
    itype: int = 1
    fct_id_ab: int = 0
    nu_val: float = 0.0
    fscale_ab: float = 1.0
    fct_id_sm: int = 0
    fct_id_bm: int = 0
    fscale_sm: float = 1.0
    fscale_bm: float = 1.0
    a_pl: float = 1.0
    sigma_pl: float = 1.0
    f_pl: float = 1.0
    epsilon_f: float = 1.0
    n_pl: int = 1
    title: str = ""
    networks: List[Dict[str, Any]] = field(default_factory=list)
    law: int = 100
    law_name: str = "LAW100"
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def rho(self) -> float:
        return self.rhor if self.rhor > 0.0 else self.rho0

    @rho.setter
    def rho(self, val: float) -> None:
        self.rho0 = float(val)

    @property
    def sb(self) -> float:
        if "sb" in self.params:
            return float(self.params["sb"])
        if self.networks:
            return float(sum(net.get("stiffness", net.get("stiffn", 1.0)) for net in self.networks))
        return float(self.params.get("stiffness", 0.0))

    @property
    def G(self) -> float:
        if "G" in self.params:
            return float(self.params["G"])
        if self.flag_he in (1, 3, 4, 5):
            return float(2.0 * (self.c10 + self.c01) * (1.0 + self.sb))
        elif self.flag_he == 2:
            lm = self.lambda_m if self.lambda_m > 0.0 else 7.0
            beta = 1.0 / (lm ** 2)
            poly = (1.0 + 0.6 * beta + (99.0 / 175.0) * (beta ** 2)
                    + (513.0 / 875.0) * (beta ** 3) + (42039.0 / 67375.0) * (beta ** 4))
            return float(self.mue1 * poly * (1.0 + self.sb))
        elif self.flag_he == 13:
            return float(self.fscale_sm * (1.0 + self.sb))
        return float(2.0 * (self.c10 + self.c01) * (1.0 + self.sb))

    @property
    def g(self) -> float:
        return self.G

    @property
    def shear(self) -> float:
        return self.G

    @property
    def K(self) -> float:
        if "K" in self.params:
            return float(self.params["K"])
        if self.flag_he in (1, 3, 4, 5):
            if self.d1 > 0.0:
                d1_inv = (1.0 / self.d1) if self.d1 < 1.0 else self.d1
                return float(2.0 * d1_inv * (1.0 + self.sb))
            nu = self.nu
            return float((2.0 / 3.0) * self.G * (1.0 + nu) / max(1e-30, (1.0 - 2.0 * nu)))
        elif self.flag_he == 2:
            d_inv = (1.0 / self.d) if (self.d > 0.0 and self.d < 1.0) else (self.d if self.d > 0.0 else 1e20)
            return float(2.0 * (1.0 + self.sb) * d_inv)
        elif self.flag_he == 13:
            return float(self.fscale_bm * (1.0 + self.sb))
        return float(self.G * 100.0)

    @property
    def k(self) -> float:
        return self.K

    @property
    def bulk(self) -> float:
        return self.K

    @property
    def nu(self) -> float:
        if "nu" in self.params:
            return float(self.params["nu"])
        if 0.0 < self.nu_val < 0.5:
            return self.nu_val
        k = self.K
        g = self.G
        denom = 2.0 * (3.0 * k + g)
        if denom > 0.0:
            val = (3.0 * k - 2.0 * g) / denom
            if 0.0 <= val < 0.5:
                return float(val)
        return 0.495

    @property
    def E(self) -> float:
        if "E" in self.params:
            return float(self.params["E"])
        k = self.K
        g = self.G
        denom = 3.0 * k + g
        if denom > 0.0:
            return float(9.0 * k * g / denom)
        return float(2.0 * g * (1.0 + self.nu))

    @property
    def sound_speed(self) -> float:
        stiff = self.K + (4.0 / 3.0) * self.G
        return float(np.sqrt(max(0.0, stiff / max(1e-20, self.rho))))


MatSpotweld = MatLaw100
MatStructuralAdhesive = MatLaw100
MatViscHyp = MatLaw100
MatMNF = MatLaw100
MaterialLaw100 = MatLaw100


@dataclass
class MatLaw97:
    """``/MAT/LAW97`` or ``/MAT/EXPLOSIVE_JWLS``: High-explosive detonation equation of state with JWLS parameters."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    p0: float = 0.0
    psh: float = 0.0
    ibfrac: int = 0
    d: float = 0.0
    pcj: float = 0.0
    e0: float = 0.0
    omega: float = 0.0
    c: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    a3: float = 0.0
    a4: float = 0.0
    a5: float = 0.0
    r1: float = 0.0
    r2: float = 0.0
    r3: float = 0.0
    r4: float = 0.0
    r5: float = 0.0
    title: str = ""


MatExplosiveJwls = MatLaw97
MatJwls = MatLaw97


@dataclass
class MatLaw71:
    """``/MAT/LAW71`` or ``/MAT/SUPER_ELAS``: Nitinol / Superelastic shape memory alloy model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    e_mart: float = 0.0
    sig_sas: float = 0.0
    sig_fas: float = 0.0
    sig_ssa: float = 0.0
    sig_fsa: float = 0.0
    alpha: float = 0.0
    epsl: float = 0.0
    cas: float = 0.0
    csa: float = 0.0
    tsas: float = 0.0
    tfas: float = 0.0
    tssa: float = 0.0
    tfsa: float = 0.0
    cp: float = 0.0
    tini: float = 0.0
    title: str = ""


MatSuperElas = MatLaw71
MatNitinol = MatLaw71


@dataclass
class MatLaw73:
    """``/MAT/LAW73`` (/MAT/BARLAT2000, /MAT/HILL_THERM, /MAT/THERM_HILL):
    Thermal Hill Orthotropic Material Model for Shells (M561).

    Upstream reference:
      - ``starter/source/materials/mat/mat073/hm_read_mat73.F``
      - ``engine/source/materials/mat/mat073/sigeps73c.F``
      - ``hm_cfg_files/config/CFG/radioss140/MAT/matl73_73.cfg``
    """
    id: int
    rho: float
    e: float
    nu: float
    ifunce: int = 0
    einf: float = 0.0
    ce: float = 0.0
    r00: float = 1.0
    r45: float = 1.0
    r90: float = 1.0
    chard: float = 0.0
    iyield: int = 0
    eps_max: float = 1e30
    epsr1: float = 1e30
    epsr2: float = 2e30
    table_id: int = 0
    fscale: float = 1.0
    pscale: float = 1.0
    t0: float = 293.0
    rhocp: float = 0.0
    title: str = ""
    law: int = 73
    law_name: str = "LAW73"
    refer_rho: float = 0.0

    def __init__(
        self,
        id: int = 0,
        rho: float = 0.0,
        e: float = 0.0,
        nu: float = 0.0,
        ifunce: int = 0,
        einf: float = 0.0,
        ce: float = 0.0,
        r00: float = 1.0,
        r45: float = 1.0,
        r90: float = 1.0,
        chard: float = 0.0,
        iyield: int = 0,
        eps_max: float = 1e30,
        epsr1: float = 1e30,
        epsr2: float = 2e30,
        table_id: int = 0,
        fscale: float = 1.0,
        pscale: float = 1.0,
        t0: float = 293.0,
        rhocp: float = 0.0,
        title: str = "",
        law: int = 73,
        law_name: str = "LAW73",
        refer_rho: float = 0.0,
        **kwargs: Any,
    ):
        self.id = id
        self.rho = kwargs.get("rho0", rho)
        self.refer_rho = kwargs.get("rhor", refer_rho)
        self.e = kwargs.get("E", e)
        self.nu = kwargs.get("Nu", nu)
        self.ifunce = kwargs.get("yr_fun", ifunce)
        self.einf = kwargs.get("efib", einf)
        self.ce = kwargs.get("c", ce)
        self.r00 = r00
        self.r45 = r45
        self.r90 = r90
        self.chard = chard
        self.iyield = iyield
        self.eps_max = kwargs.get("epsp_max", eps_max)
        self.epsr1 = kwargs.get("epst1", epsr1)
        self.epsr2 = kwargs.get("epst2", epsr2)
        self.table_id = kwargs.get("fun_a1", table_id)
        self.fscale = fscale
        self.pscale = pscale
        self.t0 = kwargs.get("t_initial", t0)
        self.rhocp = kwargs.get("spheat", rhocp)
        self.title = title
        self.law = law
        self.law_name = law_name

    # --- Property helpers & aliases ---
    @property
    def rho0(self) -> float:
        return self.rho

    @rho0.setter
    def rho0(self, val: float) -> None:
        self.rho = val

    @property
    def rhor(self) -> float:
        return self.refer_rho if self.refer_rho != 0.0 else self.rho

    @rhor.setter
    def rhor(self, val: float) -> None:
        self.refer_rho = val

    @property
    def E(self) -> float:
        return self.e

    @E.setter
    def E(self, val: float) -> None:
        self.e = val

    @property
    def Nu(self) -> float:
        return self.nu

    @Nu.setter
    def Nu(self, val: float) -> None:
        self.nu = val

    @property
    def G(self) -> float:
        """Elastic shear modulus: G = 0.5 * E / (1 + nu)."""
        denom = 2.0 * (1.0 + self.nu)
        return self.e / denom if denom != 0.0 else 0.0

    @property
    def g(self) -> float:
        return self.G

    def _calc_hill_coefficients(self) -> tuple[float, float, float, float]:
        """Compute Lankford Hill parameters A01, A02, A03, A12 per hm_read_mat73.F."""
        r0 = self.r00 if self.r00 != 0.0 else 1.0
        r45 = self.r45 if self.r45 != 0.0 else 1.0
        r90 = self.r90 if self.r90 != 0.0 else 1.0
        r = (r0 + 2.0 * r45 + r90) * 0.25
        h = r / (1.0 + r) if (1.0 + r) != 0.0 else 0.0
        a01 = h * (1.0 + 1.0 / r0)
        a02 = h * (1.0 + 1.0 / r90)
        a03 = 2.0 * h
        a12 = (2.0 * r45 + 1.0) * (a01 + a02 - a03)
        if self.iyield > 0 and a01 != 0.0:
            a02 = a02 / a01
            a03 = a03 / a01
            a12 = a12 / a01
            a01 = 1.0
        return a01, a02, a03, a12

    @property
    def A01(self) -> float:
        return self._calc_hill_coefficients()[0]

    @property
    def A02(self) -> float:
        return self._calc_hill_coefficients()[1]

    @property
    def A03(self) -> float:
        return self._calc_hill_coefficients()[2]

    @property
    def A12(self) -> float:
        return self._calc_hill_coefficients()[3]

    @property
    def a01(self) -> float:
        return self.A01

    @property
    def a02(self) -> float:
        return self.A02

    @property
    def a03(self) -> float:
        return self.A03

    @property
    def a12(self) -> float:
        return self.A12

    @property
    def sound_speed(self) -> CallableFloat:
        """Acoustic sound speed: c = sqrt(E / rho0)."""
        import math
        c = math.sqrt(self.e / self.rho) if self.rho > 0.0 and self.e > 0.0 else 0.0
        return CallableFloat(c)

    @property
    def sound_speed_shell(self) -> CallableFloat:
        """Shell plane-stress sound speed: c = sqrt(E / (rho0 * (1 - nu^2)))."""
        import math
        denom = self.rho * (1.0 - self.nu**2)
        c = math.sqrt(self.e / denom) if denom > 0.0 and self.e > 0.0 else 0.0
        return CallableFloat(c)

    # Legacy/CFG field aliases
    @property
    def fun_a1(self) -> int:
        return self.table_id

    @fun_a1.setter
    def fun_a1(self, val: int) -> None:
        self.table_id = val

    @property
    def yr_fun(self) -> int:
        return self.ifunce

    @yr_fun.setter
    def yr_fun(self, val: int) -> None:
        self.ifunce = val

    @property
    def efib(self) -> float:
        return self.einf

    @efib.setter
    def efib(self, val: float) -> None:
        self.einf = val

    @property
    def c(self) -> float:
        return self.ce

    @c.setter
    def c(self, val: float) -> None:
        self.ce = val

    @property
    def t_initial(self) -> float:
        return self.t0

    @t_initial.setter
    def t_initial(self, val: float) -> None:
        self.t0 = val

    @property
    def spheat(self) -> float:
        return self.rhocp

    @spheat.setter
    def spheat(self, val: float) -> None:
        self.rhocp = val

    @property
    def epst1(self) -> float:
        return self.epsr1

    @epst1.setter
    def epst1(self, val: float) -> None:
        self.epsr1 = val

    @property
    def epst2(self) -> float:
        return self.epsr2

    @epst2.setter
    def epst2(self, val: float) -> None:
        self.epsr2 = val

    @property
    def epsp_max(self) -> float:
        return self.eps_max

    @property
    def r0(self) -> float:
        return self.r00

    @r0.setter
    def r0(self, val: float) -> None:
        self.r00 = val

    @property
    def fisokin(self) -> float:
        return self.chard

    @fisokin.setter
    def fisokin(self, val: float) -> None:
        self.chard = val

    @property
    def c_hard(self) -> float:
        return self.chard

    @c_hard.setter
    def c_hard(self, val: float) -> None:
        self.chard = val

    @epsp_max.setter
    def epsp_max(self, val: float) -> None:
        self.eps_max = val

    @property
    def params(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rho": self.rho,
            "rho0": self.rho,
            "refer_rho": self.refer_rho,
            "rhor": self.rhor,
            "e": self.e,
            "E": self.e,
            "nu": self.nu,
            "ifunce": self.ifunce,
            "yr_fun": self.ifunce,
            "einf": self.einf,
            "efib": self.einf,
            "ce": self.ce,
            "c": self.ce,
            "r00": self.r00,
            "r0": self.r00,
            "r45": self.r45,
            "r90": self.r90,
            "chard": self.chard,
            "fisokin": self.chard,
            "c_hard": self.chard,
            "iyield": self.iyield,
            "eps_max": self.eps_max,
            "epsp_max": self.eps_max,
            "epsr1": self.epsr1,
            "epst1": self.epsr1,
            "epsr2": self.epsr2,
            "epst2": self.epsr2,
            "table_id": self.table_id,
            "fun_a1": self.table_id,
            "fscale": self.fscale,
            "pscale": self.pscale,
            "t0": self.t0,
            "t_initial": self.t0,
            "rhocp": self.rhocp,
            "spheat": self.rhocp,
            "title": self.title,
            "law": self.law,
            "law_name": self.law_name,
        }

    # Mapping protocol
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        p = self.params
        if key in p:
            return p[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Cannot set unknown attribute {key!r} on MatLaw73")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or key in self.params

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        import dataclasses
        k = [f.name for f in dataclasses.fields(self)]
        k.extend(["rho0", "rhor", "E", "nu", "Nu", "G", "A01", "A02", "A03", "A12", "sound_speed", "sound_speed_shell", "fun_a1", "yr_fun", "efib", "c", "t_initial", "spheat", "epst1", "epst2", "epsp_max", "r0", "fisokin", "c_hard"])
        return list(dict.fromkeys(k))

    def values(self) -> list[Any]:
        return [self[k] for k in self.keys()]

    def items(self) -> list[tuple[str, Any]]:
        return [(k, self[k]) for k in self.keys()]


MatHillTherm = MatLaw73
MatThermalHill = MatLaw73
MatThermHill = MatLaw73


@dataclass
class MatLaw84:
    """``/MAT/LAW84`` or ``/MAT/SWIFT_VOCE``: Swift-Voce hardening plastic material with thermal coupling."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    fcut: float = 0.0
    cap_end: float = 0.0
    pc: float = 0.0
    pr: float = 0.0
    t0: float = 0.0
    c2_t: float = 0.0
    a2: float = 0.0
    c1_c: float = 0.0
    vol: float = 0.0
    nut: float = 0.0
    fscale11: float = 1.0
    fscale22: float = 1.0
    fscale33: float = 1.0
    fscale12: float = 1.0
    fscale23: float = 1.0
    scale1: float = 0.0
    scale2: float = 0.0
    scale3: float = 0.0
    scale4: float = 0.0
    scale5: float = 0.0
    title: str = ""


MatSwiftVoce = MatLaw84
MatPlasSwiftVoce = MatLaw84


@dataclass
class MatLaw93:
    """``/MAT/LAW93`` or ``/MAT/ORTH_HILL``: 3D Orthotropic Hill plasticity model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e11: float = 0.0
    e22: float = 0.0
    e33: float = 0.0
    g12: float = 0.0
    nu12: float = 0.0
    g13: float = 0.0
    g23: float = 0.0
    nu13: float = 0.0
    nu23: float = 0.0
    nl: int = 0
    sigma_y: float = 0.0
    qr1: float = 0.0
    cr1: float = 0.0
    qr2: float = 0.0
    cr2: float = 0.0
    r11: float = 1.0
    r22: float = 1.0
    r12: float = 1.0
    r33: float = 1.0
    r13: float = 1.0
    r23: float = 1.0
    fcut: float = 0.0
    vp: int = 0
    curves: List[Dict[str, Any]] = field(default_factory=list)
    title: str = ""
    law: int = 93
    law_name: str = "LAW93"
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def rho(self) -> float:
        return self.rho0

    @property
    def E(self) -> float:
        return max(self.e11, self.e22, self.e33)

    @property
    def e(self) -> float:
        return self.E

    @property
    def young(self) -> float:
        return self.E

    @property
    def nu(self) -> float:
        return max(self.nu12, self.nu13, self.nu23)

    @property
    def poisson(self) -> float:
        return self.nu

    @property
    def G(self) -> float:
        return self.g12

    @property
    def g(self) -> float:
        return self.g12

    @property
    def shear(self) -> float:
        return self.g12

    @property
    def sound_speed(self) -> CallableFloat:
        try:
            from ...materials.law93_orth_hill import sound_speed
            return CallableFloat(float(sound_speed(self, self.rho0)))
        except Exception:
            import math
            r = self.rho0
            if r > 0.0:
                c2 = max(self.e11, self.e22, self.e33) / r
                if c2 > 0.0:
                    return CallableFloat(math.sqrt(c2))
            return CallableFloat(0.0)

    @property
    def sound_speed_solid(self) -> CallableFloat:
        return self.sound_speed

    @property
    def sound_speed_shell(self) -> CallableFloat:
        try:
            from ...materials.law93_orth_hill import sound_speed_shell
            return CallableFloat(float(sound_speed_shell(self, self.rho0)))
        except Exception:
            return self.sound_speed

    # Mapping protocol
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if isinstance(self.params, dict) and key in self.params:
            return self.params[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            if not isinstance(self.params, dict):
                self.params = {}
            self.params[key] = value

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or (isinstance(self.params, dict) and key in self.params)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> List[str]:
        base_keys = [
            "id", "title", "rho0", "rhor", "e11", "e22", "e33", "g12", "nu12",
            "g13", "g23", "nu13", "nu23", "nl", "sigma_y", "qr1", "cr1", "qr2", "cr2",
            "r11", "r22", "r12", "r33", "r13", "r23", "fcut", "vp", "curves",
            "law", "law_name", "rho", "E", "e", "young", "nu", "poisson", "G", "g", "shear",
            "sound_speed", "sound_speed_solid", "sound_speed_shell",
        ]
        if isinstance(self.params, dict):
            for k in self.params:
                if k not in base_keys:
                    base_keys.append(k)
        return base_keys

    def values(self) -> List[Any]:
        return [self[k] for k in self.keys()]

    def items(self) -> List[tuple]:
        return [(k, self[k]) for k in self.keys()]

    def __len__(self) -> int:
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())


MaterialLaw93 = MatLaw93
MatOrthHill = MatLaw93


@dataclass
class MatLaw95:
    """``/MAT/LAW95`` or ``/MAT/BERGSTROM_BOYCE``: Bergstrom-Boyce visco-hyperelastic polymer model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    c10: float = 0.0
    c01: float = 0.0
    c20: float = 0.0
    c11: float = 0.0
    c02: float = 0.0
    c30: float = 0.0
    c21: float = 0.0
    c12: float = 0.0
    c03: float = 0.0
    sb: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0
    nu_val: float = 0.0
    iform: int = 1
    a: float = 0.0
    expc: float = -0.7
    c: float = 0.0
    expm: float = 1.0
    m: float = 0.0
    ksi: float = 0.01
    tauref: float = 1.0
    tau_ref: float = 0.0
    title: str = ""
    law: int = 95
    law_name: str = "LAW95"
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.c != 0.0 and self.expc == -0.7:
            self.expc = self.c
        elif self.c == 0.0 and self.expc != 0.0:
            self.c = self.expc
        if self.m != 0.0 and self.expm == 1.0:
            self.expm = self.m
        elif self.m == 0.0 and self.expm != 0.0:
            self.m = self.expm
        if self.tau_ref != 0.0 and self.tauref == 1.0:
            self.tauref = self.tau_ref
        elif self.tau_ref == 0.0 and self.tauref != 0.0:
            self.tau_ref = self.tauref

    @property
    def rho(self) -> float:
        return self.rhor if self.rhor > 0.0 else self.rho0

    @rho.setter
    def rho(self, val: float) -> None:
        self.rho0 = float(val)

    @property
    def G(self) -> float:
        g0 = 2.0 * (self.c10 + self.c01) * (self.sb + 1.0)
        return float(self.params.get("G", g0))

    @property
    def g(self) -> float:
        return self.G

    @property
    def shear(self) -> float:
        return self.G

    @property
    def K(self) -> float:
        if "K" in self.params:
            return float(self.params["K"])
        if self.d1 > 0.0:
            d1_inv = (1.0 / self.d1) if self.d1 < 1.0 else self.d1
            return float(2.0 * d1_inv * (1.0 + self.sb))
        g0 = self.G
        nu = self.nu
        return float((2.0 / 3.0) * g0 * (1.0 + nu) / max(1e-30, (1.0 - 2.0 * nu)))

    @property
    def k(self) -> float:
        return self.K

    @property
    def bulk(self) -> float:
        return self.K

    @property
    def nu(self) -> float:
        if "nu" in self.params:
            return float(self.params["nu"])
        return self.nu_val if self.nu_val > 0.0 else 0.495

    @nu.setter
    def nu(self, val: float) -> None:
        self.nu_val = float(val)

    @property
    def poisson(self) -> float:
        return self.nu

    @property
    def E(self) -> float:
        if "E" in self.params:
            return float(self.params["E"])
        k = self.K
        g = self.G
        denom = 3.0 * k + g
        if denom > 0.0:
            return float(9.0 * k * g / denom)
        return float(2.0 * g * (1.0 + self.nu))

    @property
    def e(self) -> float:
        return self.E

    @property
    def young(self) -> float:
        return self.E

    @property
    def sound_speed(self) -> CallableFloat:
        try:
            from ...materials.law95_bergstrom_boyce import sound_speed
            return CallableFloat(float(sound_speed(self, self.rho0)))
        except Exception:
            import math
            stiff = (4.0 / 3.0) * self.G + self.K
            r = self.rho0
            if r > 0.0 and stiff > 0.0:
                return CallableFloat(math.sqrt(stiff / r))
            return CallableFloat(0.0)

    @property
    def sound_speed_solid(self) -> CallableFloat:
        return self.sound_speed

    # Mapping protocol
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if isinstance(self.params, dict) and key in self.params:
            return self.params[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            if not isinstance(self.params, dict):
                self.params = {}
            self.params[key] = value

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or (isinstance(self.params, dict) and key in self.params)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> List[str]:
        base_keys = [
            "id", "title", "rho0", "rhor", "c10", "c01", "c20", "c11", "c02",
            "c30", "c21", "c12", "c03", "sb", "d1", "d2", "d3", "nu_val", "iform",
            "a", "expc", "expm", "ksi", "tauref", "law", "law_name", "rho",
            "E", "e", "young", "nu", "poisson", "G", "g", "shear", "K", "k", "bulk",
            "sound_speed", "sound_speed_solid",
        ]
        if isinstance(self.params, dict):
            for k in self.params:
                if k not in base_keys:
                    base_keys.append(k)
        return base_keys

    def values(self) -> List[Any]:
        return [self[k] for k in self.keys()]

    def items(self) -> List[tuple]:
        return [(k, self[k]) for k in self.keys()]

    def __len__(self) -> int:
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())


MaterialLaw95 = MatLaw95
MatBergstromBoyce = MatLaw95


@dataclass
class MatLaw133:
    """``/MAT/LAW133`` or ``/MAT/GRANULAR``: Granular material model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    nu: float = 0.0
    pmin: float = 0.0
    fct_id_g: int = 0
    fscale_g: float = 1.0
    fct_id_y: int = 0
    fscale_y: float = 1.0
    title: str = ""


MatGranular = MatLaw133


@dataclass
class MatLaw101:
    """``/MAT/LAW101`` or ``/MAT/PP`` / ``/MAT/PLAS_POLY``: Bouvard Polymer Viscoplasticity Model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    alpha1: float = 0.0
    nu: float = 0.0
    ve1: float = 0.0
    ve2: float = 0.0
    epsilonref: float = 0.0
    gamma0: float = 0.0
    alpha_p: float = 0.0
    deltah: float = 0.0
    vol: float = 0.0
    m: float = 1.0
    c3: float = 0.0
    c4: float = 0.0
    alphak1: float = 0.0
    alphak2: float = 0.0
    hard: float = 0.0
    zeta1i: float = 0.0
    c5: float = 0.0
    c6: float = 0.0
    c7: float = 0.0
    c8: float = 0.0
    c9: float = 0.0
    c10: float = 0.0
    hard1: float = 0.0
    zeta2i: float = 0.0
    c11: float = 0.0
    c12: float = 0.0
    c13: float = 0.0
    c14: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    lambdal: float = 1.0
    rho_ref: float = 0.0
    cv_ref: float = 0.0
    tref: float = 293.15
    alpha_th: float = 0.0
    theta_glass: float = 250.0
    omega: float = 0.0
    theta_flag: float = 0.0
    heat_t0: float = 293.15
    title: str = ""
    e_ref: float = 0.0

    law: int = 101

    def __post_init__(self) -> None:
        if self.e == 0.0 and self.e_ref != 0.0:
            self.e = self.e_ref
        elif self.e_ref == 0.0 and self.e != 0.0:
            self.e_ref = self.e

    @property
    def E(self) -> float:
        return self.e

    @property
    def G(self) -> float:
        if (1.0 + self.nu) != 0.0:
            return self.e / (2.0 * (1.0 + self.nu))
        return 0.0

    @property
    def K(self) -> float:
        denom = 1.0 - 2.0 * self.nu
        if abs(denom) > 1e-6:
            return self.e / (3.0 * denom)
        return 0.0

    @property
    def sound_speed(self) -> float:
        import math
        if self.rho0 <= 0.0:
            return 0.0
        c2 = (self.K + (4.0 / 3.0) * self.G) / self.rho0
        return math.sqrt(max(0.0, c2))


MatPlasPoly = MatLaw101
MatPP = MatLaw101
MaterialLaw101 = MatLaw101


@dataclass
class MatLaw43:
    """``/MAT/LAW43`` or ``/MAT/HILL_TAB``: Tabulated Hill orthotropic material model (M548).

    Reference:
      - starter/source/materials/mat/mat043/hm_read_mat43.F
      - radioss140/MAT/matl43_HILL_TAB.cfg
    """
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    ifunce: int = 0
    einf: float = 0.0
    ce: float = 0.0
    r00: float = 1.0
    r45: float = 1.0
    r90: float = 1.0
    chard: float = 0.0
    fisokin: float = 0.0
    iyield: int = 0
    eps_max: float = 0.0
    epsr1: float = 0.0
    epst1: float = 0.0
    epsr2: float = 0.0
    epst2: float = 0.0
    fcut: float = 0.0
    asrate: float = 0.0
    fsmooth: int = 0
    israte: int = 0
    curves: List[Dict[str, Any]] = field(default_factory=list)
    title: str = ""
    law: int = 43
    law_name: str = "LAW43"
    fail: Optional[Any] = None
    eos: Optional[Any] = None

    def __post_init__(self):
        if self.rhor == 0.0 and self.rho0 != 0.0:
            self.rhor = self.rho0
        if self.fisokin == 0.0 and self.chard != 0.0:
            self.fisokin = self.chard
        elif self.chard == 0.0 and self.fisokin != 0.0:
            self.chard = self.fisokin
        if self.epst1 == 0.0 and self.epsr1 != 0.0:
            self.epst1 = self.epsr1
        elif self.epsr1 == 0.0 and self.epst1 != 0.0:
            self.epsr1 = self.epst1
        if self.epst2 == 0.0 and self.epsr2 != 0.0:
            self.epst2 = self.epsr2
        elif self.epsr2 == 0.0 and self.epst2 != 0.0:
            self.epsr2 = self.epst2
        if self.asrate == 0.0 and self.fcut != 0.0:
            self.asrate = self.fcut
        elif self.fcut == 0.0 and self.asrate != 0.0:
            self.fcut = self.asrate
        if self.israte == 0 and self.fsmooth != 0:
            self.israte = self.fsmooth
        elif self.fsmooth == 0 and self.israte != 0:
            self.fsmooth = self.israte

    # Lowercase aliases & Radioss naming
    @property
    def rho(self) -> float:
        return self.rho0

    @property
    def rho_i(self) -> float:
        return self.rho0

    @property
    def refer_rho(self) -> float:
        return self.rhor

    @property
    def yr_fun(self) -> int:
        return self.ifunce

    @property
    def efib(self) -> float:
        return self.einf

    @property
    def c(self) -> float:
        return self.ce

    @property
    def r0(self) -> float:
        return self.r00

    @property
    def c_hard(self) -> float:
        return self.chard

    @property
    def eps(self) -> float:
        return self.eps_max

    @property
    def epsp_max(self) -> float:
        return self.eps_max

    @property
    def eps_t(self) -> float:
        return self.epst1

    @property
    def eps_m(self) -> float:
        return self.epst2

    @property
    def num_curves(self) -> int:
        return len(self.curves)

    # Uppercase aliases
    @property
    def RHO(self) -> float:
        return self.rho0

    @property
    def RHO0(self) -> float:
        return self.rho0

    @property
    def RHOR(self) -> float:
        return self.rhor

    @property
    def E(self) -> float:
        return self.e

    @property
    def NU(self) -> float:
        return self.nu

    @property
    def IFUNCE(self) -> int:
        return self.ifunce

    @property
    def YR_FUN(self) -> int:
        return self.ifunce

    @property
    def EINF(self) -> float:
        return self.einf

    @property
    def EFIB(self) -> float:
        return self.einf

    @property
    def CE(self) -> float:
        return self.ce

    @property
    def C(self) -> float:
        return self.ce

    @property
    def R00(self) -> float:
        return self.r00

    @property
    def R45(self) -> float:
        return self.r45

    @property
    def R90(self) -> float:
        return self.r90

    @property
    def CHARD(self) -> float:
        return self.chard

    @property
    def FISOKIN(self) -> float:
        return self.fisokin

    @property
    def IYIELD(self) -> int:
        return self.iyield

    @property
    def EPS_MAX(self) -> float:
        return self.eps_max

    @property
    def EPS(self) -> float:
        return self.eps_max

    @property
    def EPSR1(self) -> float:
        return self.epsr1

    @property
    def EPST1(self) -> float:
        return self.epst1

    @property
    def EPSR2(self) -> float:
        return self.epsr2

    @property
    def EPST2(self) -> float:
        return self.epst2

    @property
    def FCUT(self) -> float:
        return self.fcut

    @property
    def ASRATE(self) -> float:
        return self.asrate

    @property
    def FSMOOTH(self) -> int:
        return self.fsmooth

    @property
    def ISRATE(self) -> int:
        return self.israte

    @property
    def CURVES(self) -> List[Dict[str, Any]]:
        return self.curves

    @property
    def NUM_CURVES(self) -> int:
        return len(self.curves)


MatHillTab = MatLaw43
MatHillPlasTab = MatLaw43
MatLaw43HillTab = MatLaw43


@dataclass
class FailLemaitre:
    """``/FAIL/LEMAITRE``: Lemaitre continuum ductile damage failure model."""
    mat_id: int = 0
    eps_d: float = 0.0
    s_d: float = 0.0
    dc: float = 0.0
    failip: int = 0
    p_thickfail: float = 0.0
    fail_id: int = 0
    title: str = ""


FailTabulated2 = FailTab2


@dataclass
class FailAlter:
    """``/FAIL/ALTER``: Alter glass/laminate crack propagation failure model."""
    mat_id: int = 0
    exp_n: float = 0.0
    v0: float = 0.0
    vc: float = 0.0
    ema: int = 0
    irate: int = 0
    iside: int = 0
    mode: int = 0
    cr_foil: float = 0.0
    cr_air: float = 0.0
    cr_core: float = 0.0
    cr_edge: float = 0.0
    kic: float = 0.0
    kth: float = 0.0
    rlen: float = 0.0
    tdel: float = 0.0
    kres1: float = 0.0
    kres2: float = 0.0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailVisual:
    """``/FAIL/VISUAL``: Visual failure indicator model."""
    mat_id: int = 0
    vtype: int = 0
    c_min: float = 0.0
    c_max: float = 0.0
    alpha_exp: float = 0.0
    f_cutoff: float = 0.0
    f_flag: int = 0
    strdef: int = 0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailOrthstrain:
    """``/FAIL/ORTHSTRAIN``: Directional orthotropic strain failure criterion."""
    mat_id: int = 0
    pthk: float = 0.0
    eps_dot_ref: float = 0.0
    fcut: float = 0.0
    fct_idel: int = 0
    fscale_el: float = 1.0
    ei_ref: float = 0.0
    strdef: int = 0
    eps_11tf: float = 0.0
    eps_11tm: float = 0.0
    fct_id_11t: int = 0
    eps_11cf: float = 0.0
    eps_11cm: float = 0.0
    fct_id_11c: int = 0
    eps_22tf: float = 0.0
    eps_22tm: float = 0.0
    fct_id_22t: int = 0
    eps_22cf: float = 0.0
    eps_22cm: float = 0.0
    fct_id_22c: int = 0
    eps_33tf: float = 0.0
    eps_33tm: float = 0.0
    fct_id_33t: int = 0
    eps_33cf: float = 0.0
    eps_33cm: float = 0.0
    fct_id_33c: int = 0
    eps_12tf: float = 0.0
    eps_12tm: float = 0.0
    fct_id_12t: int = 0
    fail_id: int = 0
    title: str = ""


# ============================================================================
# M193 Dataclasses: Extended Failure, Materials, Properties & State Directives
# ============================================================================

@dataclass
class FailEMC:
    """``/FAIL/EMC``: Extended Mohr-Coulomb ductile fracture model."""
    mat_id: int = 0
    a_emc: float = 0.0
    n_emc: float = 0.0
    b0: float = 0.0
    c: float = 0.0
    gamma: float = 0.0
    epsilon_dot_0: float = 0.0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailNXT:
    """``/FAIL/NXT``: NXT ductile fracture model."""
    mat_id: int = 0
    fct_id1: int = 0
    fct_id2: int = 0
    ifail_sh: int = 0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailTButcher:
    """``/FAIL/TBUTCHER``: Tuler-Butcher cumulative damage dynamic fracture."""
    mat_id: int = 0
    lam: float = 0.0
    k: float = 0.0
    sigma_r: float = 0.0
    ifail_sh: int = 0
    ifail_so: int = 0
    iduct: int = 0
    ixfem: int = 0
    a: float = 0.0
    b: float = 0.0
    dadv: float = 0.0
    fail_id: int = 0
    title: str = ""

@dataclass
class FailCockcroft:
    """``/FAIL/COCKCROFT``: Cockcroft-Latham ductile failure model."""
    mat_id: int = 0
    c0: float = 0.0
    alpha: float = 1.0
    failip: int = 0
    fail_id: int = 0
    title: str = ""


@dataclass
class MatLaw53:
    """``/MAT/LAW53`` & ``/MAT/TSAI_TAB``: Tsai-Wu tabulated orthotropic plasticity."""
    id: int = 0
    rho: float = 0.0
    ref_rho: float = 0.0
    e1: float = 0.0
    e2: float = 0.0
    gab: float = 0.0
    gbc: float = 0.0
    fun_a1: int = 0
    fun_b1: int = 0
    fun_a3: int = 0
    fun_a5: int = 0
    fun_a6: int = 0
    sfac11: float = 1.0
    sfac22: float = 1.0
    sfac12: float = 1.0
    sfac23: float = 1.0
    sfac45: float = 1.0
    title: str = ""


@dataclass
class MatLaw54:
    """``/MAT/LAW54`` & ``/MAT/PREDIT``: Specialized progressive damage plasticity."""
    id: int = 0
    rho: float = 0.0
    ref_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    ifunc: int = 0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    sfac: float = 1.0
    ay: float = 0.0
    az: float = 0.0
    by: float = 0.0
    bz: float = 0.0
    cx: float = 0.0
    dc: float = 0.0
    rc: float = 0.0
    eps_max: float = 0.0
    title: str = ""


@dataclass
class MatLaw74:
    """``/MAT/LAW74`` (/MAT/HILL_3D, /MAT/ORTH_PLAS, /MAT/THERM_HILL):
    Tabulated Hill Orthotropic Plasticity for Solids (M563).

    Upstream reference:
      - ``starter/source/materials/mat/mat074/hm_read_mat74.F``
      - ``hm_cfg_files/config/CFG/radioss120/MAT/matl74_74.cfg``
    """
    id: int = 0
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    eps_max: float = 1e30
    epsr1: float = 1e30
    epsr2: float = 2e30
    ifunce: int = 0
    einf: float = 0.0
    ce: float = 0.0
    fsmooth: int = 0
    chard: float = 0.0
    fcut: float = 0.0
    s11y: float = 1.0
    s22y: float = 1.0
    s33y: float = 1.0
    s12y: float = 1.0
    s23y: float = 1.0
    s31y: float = 1.0
    table_id: int = 0
    fscale: float = 1.0
    pscale: float = 1.0
    t0: float = 293.0
    rhocp: float = 0.0
    title: str = ""
    law: int = 74
    law_name: str = "LAW74"
    refer_rho: float = 0.0
    fail: Optional[Any] = None
    eos: Optional[Any] = None
    params: dict = field(default_factory=dict)

    def __init__(
        self,
        id: int = 0,
        rho: float = 0.0,
        e: float = 0.0,
        nu: float = 0.0,
        eps_max: float = 1e30,
        epsr1: float = 1e30,
        epsr2: float = 2e30,
        ifunce: int = 0,
        einf: float = 0.0,
        ce: float = 0.0,
        fsmooth: int = 0,
        chard: float = 0.0,
        fcut: float = 0.0,
        s11y: float = 1.0,
        s22y: float = 1.0,
        s33y: float = 1.0,
        s12y: float = 1.0,
        s23y: float = 1.0,
        s31y: float = 1.0,
        table_id: int = 0,
        fscale: float = 1.0,
        pscale: float = 1.0,
        t0: float = 293.0,
        rhocp: float = 0.0,
        title: str = "",
        law: int = 74,
        law_name: str = "LAW74",
        refer_rho: float = 0.0,
        fail: Optional[Any] = None,
        eos: Optional[Any] = None,
        params: Optional[dict] = None,
        **kwargs: Any,
    ):
        self.id = id
        self.rho = kwargs.get("rho0", kwargs.get("MAT_RHO", rho))
        self.refer_rho = kwargs.get("rhor", kwargs.get("ref_rho", kwargs.get("Refer_Rho", refer_rho)))
        self.e = kwargs.get("E", kwargs.get("MAT_E", e))
        self.nu = kwargs.get("Nu", kwargs.get("MAT_NU", nu))
        self.eps_max = kwargs.get("eps_p_max", kwargs.get("epsp_max", kwargs.get("MAT_EPS", eps_max)))
        self.epsr1 = kwargs.get("epst1", kwargs.get("MAT_EPST1", kwargs.get("eps_t", epsr1)))
        self.epsr2 = kwargs.get("epst2", kwargs.get("MAT_EPST2", kwargs.get("eps_m", epsr2)))
        self.ifunce = kwargs.get("yr_fun", kwargs.get("Yr_fun", ifunce))
        self.einf = kwargs.get("efib", kwargs.get("MAT_EFIB", einf))
        self.ce = kwargs.get("c", kwargs.get("MAT_C", ce))
        self.fsmooth = kwargs.get("Fsmooth", fsmooth)
        self.chard = kwargs.get("c_hard", kwargs.get("MAT_HARD", kwargs.get("fisokin", chard)))
        self.fcut = kwargs.get("Fcut", fcut)
        self.s11y = kwargs.get("MAT_SIGT1", kwargs.get("sig11y", kwargs.get("sigma11y", s11y)))
        self.s22y = kwargs.get("MAT_SIGT2", kwargs.get("sig22y", kwargs.get("sigma22y", s22y)))
        self.s33y = kwargs.get("MAT_SIGT3", kwargs.get("sig33y", kwargs.get("sigma33y", s33y)))
        self.s12y = kwargs.get("MAT_SIGYT1", kwargs.get("sig12y", kwargs.get("sigma12y", s12y)))
        self.s23y = kwargs.get("MAT_SIGYT2", kwargs.get("sig23y", kwargs.get("sigma23y", s23y)))
        self.s31y = kwargs.get("MAT_SIGYT3", kwargs.get("sig31y", kwargs.get("sigma31y", s31y)))
        self.table_id = kwargs.get("fun_a1", kwargs.get("FUN_A1", kwargs.get("tab_id", table_id)))
        self.fscale = kwargs.get("MAT_FScale", kwargs.get("sigma_scale", fscale))
        self.pscale = kwargs.get("MAT_PScale", kwargs.get("epspt_scale", pscale))
        self.t0 = kwargs.get("t_initial", kwargs.get("T_Initial", kwargs.get("ti", t0)))
        self.rhocp = kwargs.get("spheat", kwargs.get("MAT_SPHEAT", kwargs.get("rho0_cp", rhocp)))
        self.title = title
        self.law = law
        self.law_name = law_name
        self.fail = fail
        self.eos = eos
        self._extra_params = dict(params) if params is not None else {}

    # Property helpers
    @property
    def rho0(self) -> float:
        return self.rho

    @rho0.setter
    def rho0(self, val: float) -> None:
        self.rho = val

    @property
    def rhor(self) -> float:
        return self.refer_rho if self.refer_rho != 0.0 else self.rho

    @rhor.setter
    def rhor(self, val: float) -> None:
        self.refer_rho = val

    @property
    def E(self) -> float:
        return self.e

    @E.setter
    def E(self, val: float) -> None:
        self.e = val

    @property
    def Nu(self) -> float:
        return self.nu

    @Nu.setter
    def Nu(self, val: float) -> None:
        self.nu = val

    @property
    def G(self) -> float:
        """Elastic shear modulus: G = 0.5 * E / (1 + nu)."""
        denom = 2.0 * (1.0 + self.nu)
        return self.e / denom if abs(denom) > 1e-12 else 0.0

    @property
    def bulk(self) -> float:
        """Elastic bulk modulus: K = E / 3(1 - 2*nu)."""
        denom = 3.0 * (1.0 - 2.0 * self.nu)
        return self.e / denom if abs(denom) > 1e-12 else 0.0

    @property
    def K(self) -> float:
        return self.bulk

    @property
    def sound_speed(self) -> CallableFloat:
        """Acoustic sound speed: c = sqrt(E / rho0)."""
        import math
        c = math.sqrt(self.e / self.rho) if self.rho > 0.0 and self.e > 0.0 else 0.0
        return CallableFloat(c)

    @property
    def sound_speed_solid(self) -> CallableFloat:
        """3-D dilatational wave speed: c = sqrt((K + 4G/3) / rho0)."""
        import math
        c2 = (self.K + 4.0 * self.G / 3.0) / self.rho if self.rho > 0.0 else 0.0
        return CallableFloat(math.sqrt(max(c2, 0.0)))

    # Hill property helpers (hm_read_mat74.F)
    @property
    def FF(self) -> float:
        if self.s11y == 0.0 or self.s22y == 0.0 or self.s33y == 0.0:
            return 0.0
        return 0.5 * (1.0 / (self.s22y ** 2) + 1.0 / (self.s33y ** 2) - 1.0 / (self.s11y ** 2))

    @property
    def GG(self) -> float:
        if self.s11y == 0.0 or self.s22y == 0.0 or self.s33y == 0.0:
            return 0.0
        return 0.5 * (1.0 / (self.s11y ** 2) + 1.0 / (self.s33y ** 2) - 1.0 / (self.s22y ** 2))

    @property
    def HH(self) -> float:
        if self.s11y == 0.0 or self.s22y == 0.0 or self.s33y == 0.0:
            return 0.0
        return 0.5 * (1.0 / (self.s11y ** 2) + 1.0 / (self.s22y ** 2) - 1.0 / (self.s33y ** 2))

    @property
    def LL(self) -> float:
        if self.s23y == 0.0:
            return 0.0
        return 0.5 / (self.s23y ** 2)

    @property
    def MM(self) -> float:
        if self.s31y == 0.0:
            return 0.0
        return 0.5 / (self.s31y ** 2)

    @property
    def NN(self) -> float:
        if self.s12y == 0.0:
            return 0.0
        return 0.5 / (self.s12y ** 2)

    # Lowercase Hill aliases
    @property
    def ff(self) -> float:
        return self.FF

    @property
    def gg(self) -> float:
        return self.GG

    @property
    def hh(self) -> float:
        return self.HH

    @property
    def ll(self) -> float:
        return self.LL

    @property
    def mm(self) -> float:
        return self.MM

    @property
    def nn(self) -> float:
        return self.NN

    # CFG & legacy attribute aliases
    @property
    def ref_rho(self) -> float:
        return self.refer_rho

    @ref_rho.setter
    def ref_rho(self, val: float) -> None:
        self.refer_rho = val

    @property
    def eps_p_max(self) -> float:
        return self.eps_max

    @eps_p_max.setter
    def eps_p_max(self, val: float) -> None:
        self.eps_max = val

    @property
    def epsp_max(self) -> float:
        return self.eps_max

    @epsp_max.setter
    def epsp_max(self, val: float) -> None:
        self.eps_max = val

    @property
    def eps_t(self) -> float:
        return self.epsr1

    @eps_t.setter
    def eps_t(self, val: float) -> None:
        self.epsr1 = val

    @property
    def epst1(self) -> float:
        return self.epsr1

    @epst1.setter
    def epst1(self, val: float) -> None:
        self.epsr1 = val

    @property
    def eps_m(self) -> float:
        return self.epsr2

    @eps_m.setter
    def eps_m(self, val: float) -> None:
        self.epsr2 = val

    @property
    def epst2(self) -> float:
        return self.epsr2

    @epst2.setter
    def epst2(self, val: float) -> None:
        self.epsr2 = val

    @property
    def c_hard(self) -> float:
        return self.chard

    @c_hard.setter
    def c_hard(self, val: float) -> None:
        self.chard = val

    @property
    def fisokin(self) -> float:
        return self.chard

    @fisokin.setter
    def fisokin(self, val: float) -> None:
        self.chard = val

    @property
    def yr_fun(self) -> int:
        return self.ifunce

    @yr_fun.setter
    def yr_fun(self, val: int) -> None:
        self.ifunce = val

    @property
    def efib(self) -> float:
        return self.einf

    @efib.setter
    def efib(self, val: float) -> None:
        self.einf = val

    @property
    def c(self) -> float:
        return self.ce

    @c.setter
    def c(self, val: float) -> None:
        self.ce = val

    @property
    def sig11y(self) -> float:
        return self.s11y

    @sig11y.setter
    def sig11y(self, val: float) -> None:
        self.s11y = val

    @property
    def sigma11y(self) -> float:
        return self.s11y

    @sigma11y.setter
    def sigma11y(self, val: float) -> None:
        self.s11y = val

    @property
    def sig22y(self) -> float:
        return self.s22y

    @sig22y.setter
    def sig22y(self, val: float) -> None:
        self.s22y = val

    @property
    def sigma22y(self) -> float:
        return self.s22y

    @sigma22y.setter
    def sigma22y(self, val: float) -> None:
        self.s22y = val

    @property
    def sig33y(self) -> float:
        return self.s33y

    @sig33y.setter
    def sig33y(self, val: float) -> None:
        self.s33y = val

    @property
    def sigma33y(self) -> float:
        return self.s33y

    @sigma33y.setter
    def sigma33y(self, val: float) -> None:
        self.s33y = val

    @property
    def sig12y(self) -> float:
        return self.s12y

    @sig12y.setter
    def sig12y(self, val: float) -> None:
        self.s12y = val

    @property
    def sigma12y(self) -> float:
        return self.s12y

    @sigma12y.setter
    def sigma12y(self, val: float) -> None:
        self.s12y = val

    @property
    def sig23y(self) -> float:
        return self.s23y

    @sig23y.setter
    def sig23y(self, val: float) -> None:
        self.s23y = val

    @property
    def sigma23y(self) -> float:
        return self.s23y

    @sigma23y.setter
    def sigma23y(self, val: float) -> None:
        self.s23y = val

    @property
    def sig31y(self) -> float:
        return self.s31y

    @sig31y.setter
    def sig31y(self, val: float) -> None:
        self.s31y = val

    @property
    def sigma31y(self) -> float:
        return self.s31y

    @sigma31y.setter
    def sigma31y(self, val: float) -> None:
        self.s31y = val

    @property
    def sigt1(self) -> float:
        return self.s11y

    @sigt1.setter
    def sigt1(self, val: float) -> None:
        self.s11y = val

    @property
    def sigt2(self) -> float:
        return self.s22y

    @sigt2.setter
    def sigt2(self, val: float) -> None:
        self.s22y = val

    @property
    def sigt3(self) -> float:
        return self.s33y

    @sigt3.setter
    def sigt3(self, val: float) -> None:
        self.s33y = val

    @property
    def sigyt1(self) -> float:
        return self.s12y

    @sigyt1.setter
    def sigyt1(self, val: float) -> None:
        self.s12y = val

    @property
    def sigyt2(self) -> float:
        return self.s23y

    @sigyt2.setter
    def sigyt2(self, val: float) -> None:
        self.s23y = val

    @property
    def sigyt3(self) -> float:
        return self.s31y

    @sigyt3.setter
    def sigyt3(self, val: float) -> None:
        self.s31y = val

    @property
    def tab_id(self) -> int:
        return self.table_id

    @tab_id.setter
    def tab_id(self, val: int) -> None:
        self.table_id = val

    @property
    def fun_a1(self) -> int:
        return self.table_id

    @fun_a1.setter
    def fun_a1(self, val: int) -> None:
        self.table_id = val

    @property
    def sigma_scale(self) -> float:
        return self.fscale

    @sigma_scale.setter
    def sigma_scale(self, val: float) -> None:
        self.fscale = val

    @property
    def epspt_scale(self) -> float:
        return self.pscale

    @epspt_scale.setter
    def epspt_scale(self, val: float) -> None:
        self.pscale = val

    @property
    def ti(self) -> float:
        return self.t0

    @ti.setter
    def ti(self, val: float) -> None:
        self.t0 = val

    @property
    def t_initial(self) -> float:
        return self.t0

    @t_initial.setter
    def t_initial(self, val: float) -> None:
        self.t0 = val

    @property
    def rho0_cp(self) -> float:
        return self.rhocp

    @rho0_cp.setter
    def rho0_cp(self, val: float) -> None:
        self.rhocp = val

    @property
    def spheat(self) -> float:
        return self.rhocp

    @spheat.setter
    def spheat(self, val: float) -> None:
        self.rhocp = val

    @property
    def params(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rho": self.rho,
            "rho0": self.rho,
            "refer_rho": self.refer_rho,
            "ref_rho": self.refer_rho,
            "rhor": self.rhor,
            "e": self.e,
            "E": self.e,
            "nu": self.nu,
            "Nu": self.nu,
            "eps_max": self.eps_max,
            "epsp_max": self.eps_max,
            "eps_p_max": self.eps_max,
            "epsr1": self.epsr1,
            "epst1": self.epsr1,
            "eps_t": self.epsr1,
            "epsr2": self.epsr2,
            "epst2": self.epsr2,
            "eps_m": self.epsr2,
            "ifunce": self.ifunce,
            "yr_fun": self.ifunce,
            "einf": self.einf,
            "efib": self.einf,
            "ce": self.ce,
            "c": self.ce,
            "fsmooth": self.fsmooth,
            "Fsmooth": self.fsmooth,
            "chard": self.chard,
            "c_hard": self.chard,
            "fisokin": self.chard,
            "fcut": self.fcut,
            "Fcut": self.fcut,
            "s11y": self.s11y,
            "sig11y": self.s11y,
            "sigma11y": self.s11y,
            "s22y": self.s22y,
            "sig22y": self.s22y,
            "sigma22y": self.s22y,
            "s33y": self.s33y,
            "sig33y": self.s33y,
            "sigma33y": self.s33y,
            "s12y": self.s12y,
            "sig12y": self.s12y,
            "sigma12y": self.s12y,
            "s23y": self.s23y,
            "sig23y": self.s23y,
            "sigma23y": self.s23y,
            "s31y": self.s31y,
            "sig31y": self.s31y,
            "sigma31y": self.s31y,
            "table_id": self.table_id,
            "tab_id": self.table_id,
            "fun_a1": self.table_id,
            "fscale": self.fscale,
            "sigma_scale": self.fscale,
            "pscale": self.pscale,
            "epspt_scale": self.pscale,
            "t0": self.t0,
            "ti": self.t0,
            "t_initial": self.t0,
            "rhocp": self.rhocp,
            "rho0_cp": self.rhocp,
            "spheat": self.rhocp,
            "title": self.title,
            "law": self.law,
            "law_name": self.law_name,
            "FF": self.FF,
            "GG": self.GG,
            "HH": self.HH,
            "LL": self.LL,
            "MM": self.MM,
            "NN": self.NN,
        }
        if hasattr(self, "_extra_params") and self._extra_params:
            p.update(self._extra_params)
        return p

    @params.setter
    def params(self, val: dict[str, Any]) -> None:
        self._extra_params = dict(val) if val is not None else {}

    # Mapping protocol
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        p = self.params
        if key in p:
            return p[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Cannot set unknown attribute {key!r} on MatLaw74")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or key in self.params

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        import dataclasses
        k = [f.name for f in dataclasses.fields(self)]
        k.extend([
            "rho0", "rhor", "E", "nu", "Nu", "G", "bulk", "K", "sound_speed", "sound_speed_solid",
            "FF", "GG", "HH", "LL", "MM", "NN", "ff", "gg", "hh", "ll", "mm", "nn",
            "ref_rho", "eps_p_max", "epsp_max", "eps_t", "epst1", "eps_m", "epst2",
            "c_hard", "fisokin", "yr_fun", "efib", "c",
            "sig11y", "sig22y", "sig33y", "sig12y", "sig23y", "sig31y",
            "tab_id", "fun_a1", "sigma_scale", "epspt_scale", "ti", "t_initial", "rho0_cp", "spheat"
        ])
        return list(dict.fromkeys(k))

    def values(self) -> list[Any]:
        return [self[k] for k in self.keys()]

    def items(self) -> list[tuple[str, Any]]:
        return [(k, self[k]) for k in self.keys()]


MatHill3D = MatLaw74
MatOrthPlas = MatLaw74
MaterialLaw74 = MatLaw74


@dataclass
class MatLaw82:
    """``/MAT/LAW82`` & ``/MAT/OGDEN``: Ogden hyperelastic material."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    nu: float = 0.475
    nordre: int = 1
    mu: List[float] = field(default_factory=list)
    alpha: List[float] = field(default_factory=list)
    d: List[float] = field(default_factory=list)
    title: str = ""

    def __init__(
        self,
        id: int = 0,
        rho0: float = 0.0,
        rhor: float = 0.0,
        nu: float = 0.475,
        nordre: int = 1,
        mu: Optional[List[float]] = None,
        alpha: Optional[List[float]] = None,
        d: Optional[List[float]] = None,
        title: str = "",
        **kwargs,
    ):
        self.id = id
        self.rho0 = kwargs.get("rho", rho0)
        self.rhor = kwargs.get("ref_rho", rhor)
        self.nu = kwargs.get("nu", nu)
        self.nordre = kwargs.get("order", nordre)
        if mu is not None:
            self.mu = list(mu)
        elif "mu_arr" in kwargs:
            self.mu = list(kwargs["mu_arr"])
        else:
            self.mu = []
        if alpha is not None:
            self.alpha = list(alpha)
        elif "alpha_arr" in kwargs:
            self.alpha = list(kwargs["alpha_arr"])
        else:
            self.alpha = []
        if d is not None:
            self.d = list(d)
        elif "gamma_arr" in kwargs:
            self.d = list(kwargs["gamma_arr"])
        else:
            self.d = []
        self.title = kwargs.get("title", title)

    @property
    def rho(self) -> float:
        return self.rho0

    @rho.setter
    def rho(self, val: float) -> None:
        self.rho0 = val

    @property
    def ref_rho(self) -> float:
        return self.rhor

    @ref_rho.setter
    def ref_rho(self, val: float) -> None:
        self.rhor = val

    @property
    def order(self) -> int:
        return self.nordre

    @order.setter
    def order(self, val: int) -> None:
        self.nordre = val

    @property
    def mu_arr(self) -> List[float]:
        return self.mu

    @mu_arr.setter
    def mu_arr(self, val: List[float]) -> None:
        self.mu = val

    @property
    def alpha_arr(self) -> List[float]:
        return self.alpha

    @alpha_arr.setter
    def alpha_arr(self, val: List[float]) -> None:
        self.alpha = val

    @property
    def gamma_arr(self) -> List[float]:
        return self.d

    @gamma_arr.setter
    def gamma_arr(self, val: List[float]) -> None:
        self.d = val

    @property
    def G(self) -> float:
        return float(sum(self.mu)) if self.mu else 0.0

    @property
    def K(self) -> float:
        if self.d and len(self.d) > 0 and float(self.d[0]) > 0.0:
            return 2.0 / float(self.d[0])
        gs = self.G
        nu = self.nu
        denom = 3.0 * (1.0 - 2.0 * nu)
        if abs(denom) > 1e-12:
            return 2.0 * gs * (1.0 + nu) / denom
        return 0.0

    @property
    def E(self) -> float:
        return 2.0 * self.G * (1.0 + self.nu)

    def sound_speed_solid(self) -> float:
        import math
        return float(math.sqrt(max((self.K + 4.0 * self.G / 3.0) / max(self.rho0, 1e-20), 1e-20)))

    def sound_speed_shell(self) -> float:
        import math
        return float(math.sqrt(max(((2.0 / 3.0) * self.G + self.K) / max(self.rho0, 1e-20), 1e-20)))


@dataclass
class FailXFEM:
    """``/FAIL/XFEM/...``: Extended FEM fracture criteria."""
    mat_id: int = 0
    model_name: str = ""
    fct_id: int = 0
    scale: float = 1.0
    eps: float = 0.0
    sigma: float = 0.0
    sig0: float = 0.0
    lam: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0
    d4: float = 0.0
    d5: float = 0.0
    eps_dot_0: float = 1.0
    k: float = 0.0
    sigma_r: float = 0.0
    ifail_sh: int = 1
    iduct: int = 0
    a: float = 0.0
    b: float = 0.0
    fail_id: int = 0
    params: dict = field(default_factory=dict)


@dataclass
class FailTab1:
    """``/FAIL/TAB1`` (M194): Tabulated failure model Version 1."""
    mat_id: int = 0
    ifail_sh: int = 1
    ifail_so: int = 1
    p_thickfail: float = 0.0
    p_thinfail: float = 0.0
    ixfem: int = 0
    dcrit: float = 1.0
    d: float = 0.0
    n: float = 1.0
    dadv: float = 0.0
    fct_idd: int = 0
    table1_id: int = 0
    xscale1: float = 1.0
    xscale2: float = 1.0
    table2_id: int = 0
    xscale3: float = 1.0
    xscale4: float = 1.0
    fct_id_el: int = 0
    fscale_el: float = 1.0
    el_ref: float = 1.0
    inst_start: float = 0.0
    fad_exp: float = 1.0
    ch_i_f: float = 0.0
    fct_id_t: int = 0
    fscale_t: float = 1.0
    ifunc: int = 0
    eps_max: float = 0.0
    scale: float = 1.0
    fail_id: int = 0
    params: dict = field(default_factory=dict)


@dataclass
class MatLaw40:
    """``/MAT/LAW40`` or ``/MAT/CONCR_SUB`` (M194): Concrete subgrade model."""
    id: int
    title: str = ""
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class MatLaw80:
    """``/MAT/LAW80`` or ``/MAT/BARLAT3`` (M194): Barlat 3-parameter plasticity."""
    id: int
    title: str = ""
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class MatLaw102:
    """``/MAT/LAW102`` or ``/MAT/DPRAG2``: Extended Drucker-Prager material model (M194, M195, M572)."""
    id: int
    title: str = ""
    rho: float = 0.0
    iform: int = 2
    e: float = 0.0
    nu: float = 0.0
    c: float = 0.0
    phi: float = 0.0
    amax: float = 1.0e30
    pmin: float = -1.0e30
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    b0: float = 0.0
    b1: float = 0.0
    icrit: int = 1
    params: dict = field(default_factory=dict)
    law: int = 102
    law_name: str = "LAW102"

    @property
    def E(self) -> float:
        return self.e

    @property
    def G(self) -> float:
        return self.e / (2.0 * (1.0 + self.nu)) if (1.0 + self.nu) != 0.0 else 0.0

    @property
    def K(self) -> float:
        return self.e / (3.0 * (1.0 - 2.0 * self.nu)) if (1.0 - 2.0 * self.nu) != 0.0 else 0.0

    @property
    def sound_speed(self) -> float:
        rho_val = self.rho if self.rho > 0.0 else 1.0
        return math.sqrt(max(0.0, self.K + 4.0 / 3.0 * self.G) / rho_val)

MatHill48 = MatLaw102
MaterialLaw102 = MatLaw102



@dataclass
class MatNLocal:
    """``/MAT/NLOCAL`` (M194): Nonlocal plastic strain regularisation."""
    id: int
    title: str = ""
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    params: dict = field(default_factory=dict)


MatConcrSub = MatLaw40
MatHill48 = MatLaw102


@dataclass
class MatLaw103:
    """``/MAT/LAW103``, ``/MAT/HENSEL_SPITTEL``, or ``/MAT/PLAS_HENS`` (M195, M573): Hensel-Spittel hot metal forming model."""
    id: int
    title: str = ""
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    a0: float = 0.0
    m1: float = 0.0
    m2: float = 0.0
    m3: float = 0.0
    m4: float = 0.0
    m5: float = 0.0
    m7: float = 0.0
    fsmooth: int = 0
    fcut: float = 0.0
    eps_0: float = 0.0
    pmin: float = -1.0e30
    rhocp: float = 0.0
    t0: float = 0.0
    eta: float = 0.0
    params: dict = field(default_factory=dict)
    law: int = 103
    law_name: str = "LAW103"

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def rhor(self) -> float:
        return self.refer_rho if self.refer_rho > 0.0 else self.rho

    @property
    def E(self) -> float:
        return self.e

    @property
    def nu_val(self) -> float:
        return self.nu

    @property
    def G(self) -> float:
        return self.e / (2.0 * (1.0 + self.nu)) if (1.0 + self.nu) != 0.0 else 0.0

    @property
    def K(self) -> float:
        denom = 1.0 - 2.0 * self.nu
        return self.e / (3.0 * denom) if denom != 0.0 else 0.0

    @property
    def bulk(self) -> float:
        return self.K

    @property
    def eps0(self) -> float:
        return self.eps_0

    @property
    def rcp(self) -> float:
        return self.rhocp if self.rhocp > 0.0 else 1.0e30

    @property
    def sound_speed(self) -> float:
        rho_val = self.rho if self.rho > 0.0 else 1.0
        return math.sqrt(max(0.0, self.K + 4.0 / 3.0 * self.G) / rho_val)


MatPlasHens = MatLaw103
MaterialLaw103 = MatLaw103


@dataclass
class MatLaw108:
    """``/MAT/LAW108`` or ``/MAT/SPR_GENE`` (M195): Generalized 6-DOF nonlinear spring material."""
    id: int
    title: str = ""
    rho: float = 0.0
    ifail: int = 0
    iequil: int = 0
    ifail2: int = 0
    k: list[float] = field(default_factory=lambda: [0.0]*6)
    c: list[float] = field(default_factory=lambda: [0.0]*6)
    a: list[float] = field(default_factory=lambda: [0.0]*6)
    b: list[float] = field(default_factory=lambda: [0.0]*6)
    d: list[float] = field(default_factory=lambda: [0.0]*6)
    fct_id1: list[int] = field(default_factory=lambda: [0]*6)
    h: list[float] = field(default_factory=lambda: [0.0]*6)
    fct_id2: list[int] = field(default_factory=lambda: [0]*6)
    fct_id3: list[int] = field(default_factory=lambda: [0]*6)
    fct_id4: list[int] = field(default_factory=lambda: [0]*6)
    delta_min: list[float] = field(default_factory=lambda: [0.0]*6)
    delta_max: list[float] = field(default_factory=lambda: [0.0]*6)
    f_val: list[float] = field(default_factory=lambda: [0.0]*6)
    e_val: list[float] = field(default_factory=lambda: [0.0]*6)
    ascale: list[float] = field(default_factory=lambda: [1.0]*6)
    hscale: list[float] = field(default_factory=lambda: [1.0]*6)
    fsmooth: int = 0
    fcut: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class MatPlasPredef:
    """``/MAT/PLAS_PREDEF`` (M195): Predefined plasticity material model."""
    id: int
    title: str = ""
    mat_name: str = ""
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    sigy: float = 0.0
    uts: float = 0.0
    e_uts: float = 0.0
    epsp_f: float = 0.0
    vp: float = 0.0
    c: float = 0.0
    p: float = 0.0
    n: int = 0
    params: dict = field(default_factory=dict)


# MatDPrag2 is consolidated into MatLaw102 above (M195/M572)
MatDPrag2 = MatLaw102


MatHenselSpittel = MatLaw103
MatSprGene = MatLaw108


@dataclass
class MatLaw62:
    """``/MAT/LAW62`` or ``/MAT/VISC_ELAS`` (M197): Viscoelastic material."""
    id: int = 0
    title: str = ""
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    params: dict = field(default_factory=dict)


# MatLaw88 is consolidated into MaterialLaw88 above (M173/M565)
# MatLaw88 = MaterialLaw88



@dataclass(init=False)
class MatLaw66:
    """``/MAT/LAW66`` (``/MAT/PLAS_TAB_COSSER``, ``/MAT/PLAS_COSSER``) (M562): Tabulated tension-compression plastic law.

    Upstream Fortran origin:
      - starter/source/materials/mat/mat066/hm_read_mat66.F
      - CFG: radioss2022/MAT/mat_law66.cfg
    """
    id: int = 0
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    ec: float = 0.0
    pc: float = 0.0
    pt: float = 0.0
    rpct: float = 1.0
    chard: float = 0.0
    asrate: float = 0.0
    fsmooth: int = 0
    israte: int = 1
    fun_a1: int = 0
    fun_a2: int = 0
    fscale11: float = 1.0
    fscale22: float = 1.0
    epsp0: float = 1.0
    cp: float = 1.0
    sigy: float = 0.0
    vp: int = 0
    fun_b1: int = 0
    fun_b2: int = 0
    fscale33: float = 1.0
    fscale12: float = 1.0
    nfunc: int = 0
    tfunc: int = 0
    abg_ipt: list = None
    k_a1: list = None
    fp1: list = None
    abg_ipdel: list = None
    k_b1: list = None
    fp2: list = None
    title: str = ""
    law: int = 66
    law_name: str = "LAW66"
    refer_rho: float = 0.0
    fail: Optional[Any] = None
    eos: Optional[Any] = None
    params: dict = field(default_factory=dict)

    def __init__(
        self,
        id: int = 0,
        rho: float = 0.0,
        e: float = 0.0,
        nu: float = 0.0,
        ec: float = 0.0,
        pc: float = 0.0,
        pt: float = 0.0,
        rpct: float = 1.0,
        chard: float = 0.0,
        asrate: float = 0.0,
        fsmooth: int = 0,
        israte: int = 1,
        fun_a1: int = 0,
        fun_a2: int = 0,
        fscale11: float = 1.0,
        fscale22: float = 1.0,
        epsp0: float = 1.0,
        cp: float = 1.0,
        sigy: float = 0.0,
        vp: int = 0,
        fun_b1: int = 0,
        fun_b2: int = 0,
        fscale33: float = 1.0,
        fscale12: float = 1.0,
        nfunc: int = 0,
        tfunc: int = 0,
        abg_ipt: Optional[list] = None,
        k_a1: Optional[list] = None,
        fp1: Optional[list] = None,
        abg_ipdel: Optional[list] = None,
        k_b1: Optional[list] = None,
        fp2: Optional[list] = None,
        title: str = "",
        law: int = 66,
        law_name: str = "LAW66",
        refer_rho: float = 0.0,
        fail: Optional[Any] = None,
        eos: Optional[Any] = None,
        params: Optional[dict] = None,
        **kwargs: Any,
    ):
        self.id = id
        self.rho = kwargs.get("rho0", rho)
        self.refer_rho = kwargs.get("rhor", refer_rho if refer_rho != 0.0 else self.rho)
        self.e = kwargs.get("E", e)
        self.nu = kwargs.get("Nu", nu)
        self.ec = kwargs.get("Ec", ec)
        self.pc = kwargs.get("p_c", pc)
        self.pt = kwargs.get("p_t", pt)
        self.rpct = kwargs.get("Rpct", rpct)
        self.chard = kwargs.get("c_hard", kwargs.get("fisokin", chard))
        self.asrate = kwargs.get("f_cut", kwargs.get("fcut", asrate))
        self.fsmooth = kwargs.get("Fsmooth", fsmooth)
        self.israte = kwargs.get("iyld_rate", kwargs.get("iyield_rate", kwargs.get("irate", israte)))
        self.fun_a1 = kwargs.get("funct_idc", fun_a1)
        self.fun_a2 = kwargs.get("funct_idt", fun_a2)
        self.fscale11 = kwargs.get("fscalec", fscale11)
        self.fscale22 = kwargs.get("fscalet", fscale22)
        self.epsp0 = kwargs.get("eps_0", kwargs.get("epsilon_0", epsp0))
        self.cp = kwargs.get("c", kwargs.get("C", cp))
        self.sigy = kwargs.get("sigma_y0", kwargs.get("sig_y", sigy))
        self.vp = kwargs.get("VP", vp)
        self.fun_b1 = kwargs.get("fnyrt_idc", fun_b1)
        self.fun_b2 = kwargs.get("fnyrt_idt", fun_b2)
        self.fscale33 = kwargs.get("yrate_fscalec", fscale33)
        self.fscale12 = kwargs.get("yrate_fscalet", fscale12)
        self.nfunc = nfunc
        self.tfunc = tfunc

        func_ids = kwargs.get("func_ids")
        rates = kwargs.get("rates")
        fscales = kwargs.get("fscales")

        if abg_ipt is not None:
            self.abg_ipt = list(abg_ipt)
        elif func_ids is not None:
            self.abg_ipt = list(func_ids[:nfunc]) if nfunc > 0 else list(func_ids)
        else:
            self.abg_ipt = []

        if k_a1 is not None:
            self.k_a1 = list(k_a1)
        elif rates is not None:
            self.k_a1 = list(rates[:nfunc]) if nfunc > 0 else list(rates)
        else:
            self.k_a1 = []

        if fp1 is not None:
            self.fp1 = list(fp1)
        elif fscales is not None:
            self.fp1 = list(fscales[:nfunc]) if nfunc > 0 else list(fscales)
        else:
            self.fp1 = []

        if abg_ipdel is not None:
            self.abg_ipdel = list(abg_ipdel)
        elif func_ids is not None and nfunc > 0 and len(func_ids) > nfunc:
            self.abg_ipdel = list(func_ids[nfunc:])
        else:
            self.abg_ipdel = []

        if k_b1 is not None:
            self.k_b1 = list(k_b1)
        elif rates is not None and nfunc > 0 and len(rates) > nfunc:
            self.k_b1 = list(rates[nfunc:])
        else:
            self.k_b1 = []

        if fp2 is not None:
            self.fp2 = list(fp2)
        elif fscales is not None and nfunc > 0 and len(fscales) > nfunc:
            self.fp2 = list(fscales[nfunc:])
        else:
            self.fp2 = []

        if self.nfunc == 0 and self.abg_ipt:
            self.nfunc = len(self.abg_ipt)
        if self.tfunc == 0 and self.abg_ipdel:
            self.tfunc = len(self.abg_ipdel)

        self.title = title
        self.law = law
        self.law_name = law_name
        self.fail = fail
        self.eos = eos
        self.params = dict(params) if params is not None else {}

    # Property helpers
    @property
    def rho0(self) -> float:
        return self.rho

    @rho0.setter
    def rho0(self, val: float) -> None:
        self.rho = val

    @property
    def rhor(self) -> float:
        return self.refer_rho

    @rhor.setter
    def rhor(self, val: float) -> None:
        self.refer_rho = val

    @property
    def E(self) -> float:
        return self.e

    @E.setter
    def E(self, val: float) -> None:
        self.e = val

    @property
    def nu0(self) -> float:
        return self.nu

    @property
    def Nu(self) -> float:
        return self.nu

    @Nu.setter
    def Nu(self, val: float) -> None:
        self.nu = val

    @property
    def G(self) -> float:
        denom = 2.0 * (1.0 + self.nu)
        return self.e / denom if abs(denom) > 1e-12 else 0.0

    @property
    def bulk(self) -> float:
        denom = 3.0 * (1.0 - 2.0 * self.nu)
        return self.e / denom if abs(denom) > 1e-12 else 0.0

    @property
    def K(self) -> float:
        return self.bulk

    @property
    def sound_speed(self) -> CallableFloat:
        import math
        return CallableFloat(math.sqrt(max(self.e / max(self.rho, 1e-20), 0.0)))

    @property
    def sound_speed_solid(self) -> CallableFloat:
        import math
        return CallableFloat(math.sqrt(max((self.K + 4.0 * self.G / 3.0) / max(self.rho, 1e-20), 0.0)))

    @property
    def sound_speed_shell(self) -> CallableFloat:
        import math
        denom = max(self.rho * (1.0 - self.nu ** 2), 1e-20)
        return CallableFloat(math.sqrt(max(self.e / denom, 0.0)))

    # Lowercase & Radioss aliases
    @property
    def c_hard(self) -> float:
        return self.chard

    @c_hard.setter
    def c_hard(self, val: float) -> None:
        self.chard = val

    @property
    def fisokin(self) -> float:
        return self.chard

    @fisokin.setter
    def fisokin(self, val: float) -> None:
        self.chard = val

    @property
    def f_cut(self) -> float:
        return self.asrate

    @f_cut.setter
    def f_cut(self, val: float) -> None:
        self.asrate = val

    @property
    def p_c(self) -> float:
        return self.pc

    @p_c.setter
    def p_c(self, val: float) -> None:
        self.pc = val

    @property
    def p_t(self) -> float:
        return self.pt

    @p_t.setter
    def p_t(self, val: float) -> None:
        self.pt = val

    @property
    def funct_idc(self) -> int:
        return self.fun_a1

    @funct_idc.setter
    def funct_idc(self, val: int) -> None:
        self.fun_a1 = val

    @property
    def funct_idt(self) -> int:
        return self.fun_a2

    @funct_idt.setter
    def funct_idt(self, val: int) -> None:
        self.fun_a2 = val

    @property
    def fscalec(self) -> float:
        return self.fscale11

    @fscalec.setter
    def fscalec(self, val: float) -> None:
        self.fscale11 = val

    @property
    def fscalet(self) -> float:
        return self.fscale22

    @fscalet.setter
    def fscalet(self, val: float) -> None:
        self.fscale22 = val

    @property
    def epsilon_0(self) -> float:
        return self.epsp0

    @epsilon_0.setter
    def epsilon_0(self, val: float) -> None:
        self.epsp0 = val

    @property
    def eps_0(self) -> float:
        return self.epsp0

    @eps_0.setter
    def eps_0(self, val: float) -> None:
        self.epsp0 = val

    @property
    def eps_0(self) -> float:
        return self.epsp0

    @eps_0.setter
    def eps_0(self, val: float) -> None:
        self.epsp0 = val

    @property
    def c(self) -> float:
        return self.cp

    @c.setter
    def c(self, val: float) -> None:
        self.cp = val

    @property
    def sigma_y0(self) -> float:
        return self.sigy

    @sigma_y0.setter
    def sigma_y0(self, val: float) -> None:
        self.sigy = val

    @property
    def fnyrt_idc(self) -> int:
        return self.fun_b1

    @fnyrt_idc.setter
    def fnyrt_idc(self, val: int) -> None:
        self.fun_b1 = val

    @property
    def fnyrt_idt(self) -> int:
        return self.fun_b2

    @fnyrt_idt.setter
    def fnyrt_idt(self, val: int) -> None:
        self.fun_b2 = val

    @property
    def yrate_fscalec(self) -> float:
        return self.fscale33

    @yrate_fscalec.setter
    def yrate_fscalec(self, val: float) -> None:
        self.fscale33 = val

    @property
    def yrate_fscalet(self) -> float:
        return self.fscale12

    @yrate_fscalet.setter
    def yrate_fscalet(self, val: float) -> None:
        self.fscale12 = val

    @property
    def func_c_list(self) -> list:
        return self.abg_ipt

    @func_c_list.setter
    def func_c_list(self, val: list) -> None:
        self.abg_ipt = val

    @property
    def eps_c_list(self) -> list:
        return self.k_a1

    @eps_c_list.setter
    def eps_c_list(self, val: list) -> None:
        self.k_a1 = val

    @property
    def fscale_c_list(self) -> list:
        return self.fp1

    @fscale_c_list.setter
    def fscale_c_list(self, val: list) -> None:
        self.fp1 = val

    @property
    def func_t_list(self) -> list:
        return self.abg_ipdel

    @func_t_list.setter
    def func_t_list(self, val: list) -> None:
        self.abg_ipdel = val

    @property
    def eps_t_list(self) -> list:
        return self.k_b1

    @eps_t_list.setter
    def eps_t_list(self, val: list) -> None:
        self.k_b1 = val

    @property
    def fscale_t_list(self) -> list:
        return self.fp2

    @fscale_t_list.setter
    def fscale_t_list(self, val: list) -> None:
        self.fp2 = val

    @property
    def EC(self) -> float:
        return self.ec

    @EC.setter
    def EC(self, val: float) -> None:
        self.ec = val

    @property
    def PC(self) -> float:
        return self.pc

    @PC.setter
    def PC(self, val: float) -> None:
        self.pc = val

    @property
    def PT(self) -> float:
        return self.pt

    @PT.setter
    def PT(self, val: float) -> None:
        self.pt = val

    @property
    def RPCT(self) -> float:
        return self.rpct

    @RPCT.setter
    def RPCT(self, val: float) -> None:
        self.rpct = val

    @property
    def Fsmooth(self) -> int:
        return self.fsmooth

    @Fsmooth.setter
    def Fsmooth(self, val: int) -> None:
        self.fsmooth = val

    @property
    def ISRATE(self) -> int:
        return self.israte

    @ISRATE.setter
    def ISRATE(self, val: int) -> None:
        self.israte = val

    @property
    def VP(self) -> int:
        return self.vp

    @VP.setter
    def VP(self, val: int) -> None:
        self.vp = val

    @property
    def iyld_rate(self) -> int:
        return self.israte

    @iyld_rate.setter
    def iyld_rate(self, val: int) -> None:
        self.israte = val

    @property
    def iyield_rate(self) -> int:
        return self.israte

    @iyield_rate.setter
    def iyield_rate(self, val: int) -> None:
        self.israte = val

    @property
    def func_ids(self) -> list:
        return list(self.abg_ipt) + list(self.abg_ipdel)

    @func_ids.setter
    def func_ids(self, val: list) -> None:
        val_list = list(val)
        if self.nfunc > 0 and len(val_list) > self.nfunc:
            self.abg_ipt = val_list[:self.nfunc]
            self.abg_ipdel = val_list[self.nfunc:]
        else:
            self.abg_ipt = val_list
            if self.nfunc == 0:
                self.nfunc = len(val_list)

    @property
    def rates(self) -> list:
        return list(self.k_a1) + list(self.k_b1)

    @rates.setter
    def rates(self, val: list) -> None:
        val_list = list(val)
        if self.nfunc > 0 and len(val_list) > self.nfunc:
            self.k_a1 = val_list[:self.nfunc]
            self.k_b1 = val_list[self.nfunc:]
        else:
            self.k_a1 = val_list

    @property
    def fscales(self) -> list:
        return list(self.fp1) + list(self.fp2)

    @fscales.setter
    def fscales(self, val: list) -> None:
        val_list = list(val)
        if self.nfunc > 0 and len(val_list) > self.nfunc:
            self.fp1 = val_list[:self.nfunc]
            self.fp2 = val_list[self.nfunc:]
        else:
            self.fp1 = val_list

    # Mapping protocol
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        k_low = key.lower()
        if hasattr(self, k_low):
            return getattr(self, k_low)
        if isinstance(self.params, dict) and key in self.params:
            return self.params[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        elif hasattr(self, key.lower()):
            setattr(self, key.lower(), value)
        else:
            raise KeyError(f"Cannot set unknown attribute {key!r} on MatLaw66")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or hasattr(self, key.lower()) or (isinstance(self.params, dict) and key in self.params)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        import dataclasses
        k = [f.name for f in dataclasses.fields(self)]
        k.extend([
            "rho0", "rhor", "E", "Nu", "nu", "G", "bulk", "K",
            "sound_speed", "sound_speed_solid", "sound_speed_shell",
            "c_hard", "fisokin", "f_cut", "p_c", "p_t",
            "funct_idc", "funct_idt", "fscalec", "fscalet",
            "epsilon_0", "c", "sigma_y0", "fnyrt_idc", "fnyrt_idt",
            "yrate_fscalec", "yrate_fscalet",
            "func_c_list", "eps_c_list", "fscale_c_list",
            "func_t_list", "eps_t_list", "fscale_t_list",
        ])
        if isinstance(self.params, dict):
            for pk in self.params:
                if pk not in k:
                    k.append(pk)
        return list(dict.fromkeys(k))

    def values(self) -> list[Any]:
        return [self[k] for k in self.keys()]

    def items(self) -> list[tuple[str, Any]]:
        return [(k, self[k]) for k in self.keys()]


MatPlasTabCosser = MatLaw66
MatPlasCosser = MatLaw66
MaterialLaw66 = MatLaw66



@dataclass
class MatLaw51:
    """``/MAT/LAW51`` or ``/MAT/DRUCKER_PRAGER`` or ``/MAT/MULTIFLUID`` (M199): Multi-material / Drucker-Prager brittle model."""
    id: int = 0
    title: str = ""
    rho0: float = 0.0
    rhor: float = 0.0
    iform: int = 0
    pext: float = 0.0
    nu: float = 0.0
    lamda: float = 0.0
    scale: float = 1.0
    rho: float = 0.0
    e: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    fc: float = 0.0
    ft: float = 0.0
    fmax: float = 0.0
    fres: float = 0.0
    eps_c: float = 0.0
    eps_t: float = 0.0
    eps_res: float = 0.0
    b: float = 0.0
    iflag: int = 0
    icomp: int = 0
    itot: int = 0
    pc: float = 0.0
    gamma: float = 0.0
    pt: float = 0.0
    psi: float = 0.0
    p0: float = 0.0
    beta: float = 0.0
    epsp_max: float = 0.0
    fac_e: float = 1.0
    params: dict = field(default_factory=dict)


MatMultiFluid = MatLaw51
MatDruckerPrager = MatLaw51
MatBrittle = MatLaw51
MatMultimat = MatLaw51


@dataclass
class FailJohnson:
    """``/FAIL/JOHNSON/mat_id`` or ``/FAIL/JOHN_COOK/mat_id`` (M203): Johnson-Cook failure model."""
    id: int = 0
    mat_id: int = 0
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0
    d4: float = 0.0
    d5: float = 0.0
    eps_dot_0: float = 1.0
    ifail_sh: int = 1
    ifail_so: int = 1
    epsf_min: float = 0.0
    dadv: float = 0.0
    ixfem: int = 0
    failip: int = 0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailBiquad:
    """``/FAIL/BIQUAD/mat_id`` (M203): Biquadratic failure criterion."""
    id: int = 0
    mat_id: int = 0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    p_thickfail: float = 0.0
    m_flag: int = 0
    s_flag: int = 2
    inst_start: float = 0.0
    ireg: int = 0
    fct_idel: int = 0
    ei_ref: float = 0.0
    r1: float = 0.0
    r2: float = 0.0
    r4: float = 0.0
    r5: float = 0.0
    icoup: int = 0
    dcrit: float = 0.0
    exp: float = 0.0
    failip: int = 0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailFld:
    """``/FAIL/FLD/mat_id`` (M203): Forming Limit Diagram failure criterion."""
    id: int = 0
    mat_id: int = 0
    fct_id: int = 0
    ifail_sh: int = 1
    i_marg: int = 0
    fct_idadv: int = 0
    rani: float = 0.0
    dadv: float = 0.0
    istrain: int = 0
    ixfem: int = 0
    factor_marginal: float = 0.0
    factor_loosemetal: float = 0.0
    fcut: float = 0.0
    alpha: float = 0.0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailConnect:
    """``/FAIL/CONNECT/mat_id`` (M203): Connector failure criterion."""
    id: int = 0
    mat_id: int = 0
    epsilon_maxn: float = 0.0
    exponent_n: float = 1.0
    alpha_n: float = 1.0
    r_fct_id_n: int = 0
    ifail: int = 0
    ifail_so: int = 0
    isym: int = 0
    epsilon_maxt: float = 0.0
    exponent_t: float = 1.0
    alpha_t: float = 1.0
    r_fct_id_t: int = 0
    ei_max: float = 0.0
    en_max: float = 0.0
    et_max: float = 0.0
    n_n: float = 0.0
    n_t: float = 0.0
    t_max: float = 0.0
    n_soft: float = 0.0
    area_scale: float = 1.0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailFractalDmg:
    """``/FAIL/FRACTAL_DMG/mat_id`` (M203): Fractal damage percolation failure criterion."""
    id: int = 0
    mat_id: int = 0
    grsh4n_1: int = 0
    grsh3n_1: int = 0
    grsh4n_2: int = 0
    grsh3n_2: int = 0
    damage: float = 0.0
    probability: float = 0.0
    seed: int = 0
    num_walk: int = 0
    printout: int = 0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailOrthenerg:
    """``/FAIL/ORTHENERG/mat_id`` (M203): Orthotropic energy failure criterion."""
    id: int = 0
    mat_id: int = 0
    pthickfail: float = 0.0
    nmod: int = 0
    failip: int = 0
    sigma_11t: float = 0.0
    g_11t: float = 0.0
    ishap11t: int = 0
    sigma_11c: float = 0.0
    g_11c: float = 0.0
    ishap11c: int = 0
    sigma_22t: float = 0.0
    g_22t: float = 0.0
    ishap22t: int = 0
    sigma_22c: float = 0.0
    g_22c: float = 0.0
    ishap22c: int = 0
    sigma_33t: float = 0.0
    g_33t: float = 0.0
    ishap33t: int = 0
    sigma_33c: float = 0.0
    g_33c: float = 0.0
    ishap33c: int = 0
    sigma_12t: float = 0.0
    g_12t: float = 0.0
    ishap12t: int = 0
    sigma_12c: float = 0.0
    g_12c: float = 0.0
    ishap12c: int = 0
    sigma_23t: float = 0.0
    g_23t: float = 0.0
    ishap23t: int = 0
    sigma_23c: float = 0.0
    g_23c: float = 0.0
    ishap23c: int = 0
    sigma_31t: float = 0.0
    g_31t: float = 0.0
    ishap31t: int = 0
    sigma_31c: float = 0.0
    g_31c: float = 0.0
    ishap31c: int = 0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailTbid:
    """``/FAIL/TBID/mat_ID`` (M210): Tabular failure criterion."""
    mat_id: int = 0
    title: str = ""
    fct_id: int = 0          # function ID or table ID (triaxiality -> failure plastic strain)
    ifail_sh: int = 1        # 1: delete on 1 layer, 2: delete on all layers
    eps_dot_0: float = 1.0   # reference strain rate
    d_max: float = 1.0       # maximum damage
    f_smooth: float = 0.0    # smoothing factor


@dataclass
class FailSnCurve:
    """``/FAIL/SN_CURVE/mat_ID`` (M211): Stress-life (S-N curve) fatigue failure criterion."""
    mat_id: int = 0
    title: str = ""
    fct_id: int = 0          # function ID (log(S) -> log(N))
    ifail_sh: int = 1        # 1: delete on 1 layer, 2: delete on all layers
    s_mean_corr: int = 0     # mean stress correction: 0: none, 1: Goodman, 2: Soderberg, 3: Gerber
    d_crit: float = 1.0      # critical fatigue damage threshold
    n_cutoff: float = 1.0e7  # fatigue endurance limit cutoff cycles


@dataclass
class FailHoop:
    """``/FAIL/HOOP/mat_ID`` (M212): Critical hoop stress bursting failure criterion."""
    mat_id: int = 0
    title: str = ""
    sigma_hoop_max: float = 0.0  # critical tensile hoop stress
    ifail_sh: int = 1            # 1: delete on 1 layer, 2: delete on all layers
    eps_p_max: float = 0.0       # plastic strain at failure
    d_max: float = 1.0           # maximum damage


@dataclass
class FailSpallingCut:
    """``/FAIL/SPALLING_CUT/mat_ID`` (M213): Spalling hydrostatic tensile cutoff failure criterion."""
    mat_id: int = 0
    title: str = ""
    p_min: float = 0.0           # minimum tensile hydrostatic pressure cutoff
    ifail_sh: int = 1            # 1: delete on 1 layer, 2: delete on all layers
    eps_v_max: float = 0.0       # maximum volumetric tensile strain
    d_max: float = 1.0           # maximum damage


@dataclass
class FailVoids:
    """``/FAIL/VOIDS/mat_ID`` (M214): Void nucleation and coalescence porosity failure model."""
    mat_id: int = 0
    title: str = ""
    f_0: float = 0.0         # initial void volume fraction
    f_c: float = 0.15        # critical void volume fraction at coalescence
    ifail_sh: int = 1        # 1: delete on 1 layer, 2: delete on all layers
    q1: float = 1.5          # Gurson parameter q1
    q2: float = 1.0          # Gurson parameter q2
    d_max: float = 1.0       # maximum damage


@dataclass
class FailHC:
    """``/FAIL/HC/mat_ID`` or ``/FAIL/HOSFORD_COULOMB/mat_ID`` (M215): Hosford-Coulomb fracture initiation model."""
    mat_id: int = 0
    title: str = ""
    a: float = 0.0           # HC parameter a
    b: float = 0.0           # HC parameter b
    c: float = 0.0           # HC parameter c
    n_hc: float = 1.0        # Hosford exponent n
    ifail_sh: int = 1        # 1: delete on 1 layer, 2: delete on all layers
    d_max: float = 1.0       # maximum damage


@dataclass
class FailLadEvr:
    """``/FAIL/LAD_EVR/mat_ID`` or ``/FAIL/LADEVEZE_EVR/mat_ID`` (M216): Ladevèze elementary volume representative composite failure."""
    mat_id: int = 0
    title: str = ""
    yo: float = 0.0          # initial micro-damage energy threshold
    yc: float = 0.0          # critical damage energy threshold
    ymax: float = 0.0        # maximum damage energy threshold
    d_max: float = 1.0       # maximum damage
    ifail_sh: int = 1        # 1: delete on 1 layer, 2: delete on all layers
    gam: float = 0.0         # shear damage coupling coefficient gamma


@dataclass
class FailOrtho:
    """``/FAIL/ORTHO/mat_ID`` or ``/FAIL/ORTHOTROPIC/mat_ID`` (M217): Orthotropic lamina strength failure criterion."""
    mat_id: int = 0
    title: str = ""
    xt: float = 0.0          # longitudinal tensile strength
    xc: float = 0.0          # longitudinal compressive strength
    yt: float = 0.0          # transverse tensile strength
    yc: float = 0.0          # transverse compressive strength
    s: float = 0.0           # shear strength
    ifail_sh: int = 1        # 1: delete on 1 layer, 2: delete on all layers


@dataclass
class FailCohesive:
    """``/FAIL/COHESIVE/mat_ID`` or ``/FAIL/COH/mat_ID`` (M218): Cohesive interface delamination failure criterion."""
    mat_id: int = 0
    title: str = ""
    g1c: float = 0.0         # Mode I critical energy release rate
    g2c: float = 0.0         # Mode II critical energy release rate
    t1: float = 0.0          # normal peak traction
    t2: float = 0.0          # shear peak traction
    alpha: float = 1.0       # mixed-mode power law parameter


@dataclass
class FailMaxStress:
    """``/FAIL/MAX_STRESS/mat_ID`` or ``/FAIL/MAXSTRESS/mat_ID`` (M219): Maximum directional stress failure criterion."""
    mat_id: int = 0
    title: str = ""
    sig_t1: float = 0.0      # tensile strength limit in direction 1
    sig_c1: float = 0.0      # compressive strength limit in direction 1
    sig_t2: float = 0.0      # tensile strength limit in direction 2
    sig_c2: float = 0.0      # compressive strength limit in direction 2
    tau_12: float = 0.0      # shear strength limit in 1-2 plane
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailSnow:
    """``/FAIL/SNOW/mat_ID`` (M220): Snow/ice brittle crush failure criterion."""
    mat_id: int = 0
    title: str = ""
    p_tens: float = 0.0      # tensile hydrostatic pressure limit
    eps_comp: float = 0.0    # compressive compaction limit strain
    sig_shear: float = 0.0   # shear yield limit
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailVisco:
    """``/FAIL/VISCO/mat_ID`` (M221): Viscoplastic strain rate-dependent failure criterion."""
    mat_id: int = 0
    title: str = ""
    eps_f0: float = 0.0      # base fracture strain
    eps_rate0: float = 1.0   # reference strain rate
    m_rate: float = 0.0      # strain rate sensitivity exponent
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailBamman:
    """``/FAIL/BAMMAN/mat_ID`` (M222): Bammann-Chiesa-Johnson void damage failure criterion."""
    mat_id: int = 0
    title: str = ""
    v0: float = 0.0          # initial void volume fraction
    an: float = 0.0          # void nucleation parameter
    bn: float = 0.0          # void growth parameter
    cn: float = 0.0          # void coalescence exponent
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailWeibull:
    """``/FAIL/WEIBULL/mat_ID`` (M223): Weibull statistical brittle failure criterion."""
    mat_id: int = 0
    title: str = ""
    sigma_0: float = 0.0     # Weibull characteristic strength
    m_mod: float = 0.0       # Weibull modulus (shape parameter)
    v_0: float = 1.0         # reference volume
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailPU:
    """``/FAIL/PU/mat_ID`` (M224): Polyurethane foam failure criterion."""
    mat_id: int = 0
    title: str = ""
    eps_t: float = 1e30      # tensile strain threshold
    eps_c: float = -1e30     # compressive strain threshold
    sigma_t: float = 1e30    # tensile cutoff stress
    sigma_c: float = -1e30   # compressive cutoff stress
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailGriffith:
    """``/FAIL/GRIFFITH/mat_ID`` (M225): Griffith brittle fracture criterion."""
    mat_id: int = 0
    title: str = ""
    sigma_0: float = 0.0     # uniaxial tensile strength cutoff
    sigma_c: float = 0.0     # compressive crushing cutoff
    tau_max: float = 0.0     # maximum shear limit
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailDrucker:
    """``/FAIL/DRUCKER/mat_ID`` (M226): Drucker-Prager failure criterion."""
    mat_id: int = 0
    title: str = ""
    alpha: float = 0.0       # pressure sensitivity coefficient
    k: float = 0.0           # cohesion limit
    sigma_t: float = 1e30    # tensile cutoff limit
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailWood:
    """``/FAIL/WOOD/mat_ID`` (M227): Wood orthotropic failure criterion."""
    mat_id: int = 0
    title: str = ""
    sigma_t11: float = 0.0   # longitudinal tensile strength
    sigma_t22: float = 0.0   # transverse tensile strength
    sigma_c11: float = 0.0   # longitudinal compressive strength
    sigma_c22: float = 0.0   # transverse compressive strength
    tau_12: float = 0.0      # in-plane shear strength
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailHill:
    """``/FAIL/HILL/mat_ID`` (M228): Hill anisotropic plasticity failure criterion."""
    mat_id: int = 0
    title: str = ""
    F: float = 0.5           # Hill's F parameter
    G: float = 0.5           # Hill's G parameter
    H: float = 0.5           # Hill's H parameter
    L: float = 1.5           # Hill's L parameter
    M: float = 1.5           # Hill's M parameter
    N: float = 1.5           # Hill's N parameter
    sigma_fail: float = 1e30 # failure stress threshold
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailNorton:
    """``/FAIL/NORTON/mat_ID`` (M229): Norton creep rupture failure criterion."""
    mat_id: int = 0
    title: str = ""
    A: float = 0.0           # Norton creep multiplier
    n: float = 1.0           # stress exponent
    m: float = 0.0           # time exponent
    eps_rupt: float = 1e30   # creep rupture strain threshold
    t_rupt: float = 1e30     # creep rupture time threshold
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailMohr:
    """``/FAIL/MOHR/mat_ID`` (M230): Mohr-Coulomb shear failure criterion."""
    mat_id: int = 0
    title: str = ""
    c: float = 0.0           # cohesion
    phi: float = 0.0         # friction angle (deg)
    sigma_t: float = 1e30    # tensile cutoff stress
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailLusas:
    """``/FAIL/LUSAS/mat_ID`` (M231): Lusas 3D composite failure criterion."""
    mat_id: int = 0
    title: str = ""
    xt: float = 1e30         # longitudinal tensile strength
    xc: float = 1e30         # longitudinal compressive strength
    yt: float = 1e30         # transverse tensile strength
    yc: float = 1e30         # transverse compressive strength
    s12: float = 1e30        # in-plane shear strength (1-2)
    s23: float = 1e30        # transverse shear strength (2-3)
    s31: float = 1e30        # transverse shear strength (3-1)
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailGtn:
    """``/FAIL/GTN/mat_ID`` (M232): Gurson-Tvergaard-Needleman porous metal failure criterion."""
    mat_id: int = 0
    title: str = ""
    q1: float = 1.5          # Tvergaard parameter 1
    q2: float = 1.0          # Tvergaard parameter 2
    eps_n: float = 0.0       # mean strain for void nucleation
    s_n: float = 0.1         # standard deviation of nucleation strain
    f_n: float = 0.04        # void volume fraction of nucleating particles
    f_c: float = 0.15        # critical void volume fraction for coalescence
    f_f: float = 0.25        # failure void volume fraction
    f_0: float = 0.0         # initial void volume fraction
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailTab3:
    """``/FAIL/TAB3/mat_ID`` (M233): 3D Tabulated failure model."""
    mat_id: int = 0
    title: str = ""
    table_id: int = 0        # 3D table ID
    scale_x: float = 1.0     # scale factor on X axis
    scale_y: float = 1.0     # scale factor on Y axis
    scale_z: float = 1.0     # scale factor on Z axis
    eps_max: float = 1e30    # maximum strain to failure
    d_adv: float = 0.0       # damage advancement rate
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailChaboche:
    """``/FAIL/CHABOCHE/mat_ID`` (M234): Lemaitre-Chaboche ductile damage failure model."""
    mat_id: int = 0
    title: str = ""
    s_0: float = 0.0         # initial damage threshold stress/strain
    s_1: float = 1.0         # damage rate coefficient
    beta: float = 1.0        # damage exponent
    d_crit: float = 0.99     # critical damage threshold
    eps_crit: float = 1e30   # critical plastic strain
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailTvergaard:
    """``/FAIL/TVERGAARD/mat_ID`` (M236): Tvergaard-Needleman void shear coalescence failure model."""
    mat_id: int = 0
    title: str = ""
    q1: float = 1.5          # void interaction parameter 1
    q2: float = 1.0          # void interaction parameter 2
    q3: float = 2.25         # void interaction parameter 3 (typically q1^2)
    kw: float = 0.0          # shear coalescence coefficient
    f_c: float = 0.15        # critical void volume fraction
    f_f: float = 0.25        # void volume fraction at failure
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailHencky:
    """``/FAIL/HENCKY/mat_ID`` (M237): Hencky logarithmic principal strain failure model."""
    mat_id: int = 0
    title: str = ""
    eps_1_max: float = 1e30  # maximum principal strain 1
    eps_2_max: float = 1e30  # maximum principal strain 2
    eps_3_max: float = 1e30  # maximum principal strain 3
    eps_eff_max: float = 1e30 # maximum equivalent effective strain
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailEnergyDensity:
    """``/FAIL/ENERGY_DENSITY/mat_ID`` (M238): Critical strain energy density failure model."""
    mat_id: int = 0
    title: str = ""
    w_crit: float = 1e30     # critical strain energy density threshold
    w_rupt: float = 1e30     # rupture strain energy density threshold
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailEnergyRatio:
    """``/FAIL/ENERGY_RATIO/mat_ID`` (M239): Energy ratio failure model."""
    mat_id: int = 0
    title: str = ""
    eratio_max: float = 1e30  # maximum energy ratio threshold
    eint_min: float = 0.0     # minimum internal energy threshold for activation
    ifail_sh: int = 1         # shell element deletion flag


@dataclass
class FailRiceTracey:
    """``/FAIL/RICE_TRACEY/mat_ID`` (M240): Rice-Tracey void growth failure model."""
    mat_id: int = 0
    title: str = ""
    r0: float = 0.0          # initial void radius
    rc_r0: float = 1.0       # critical void radius expansion ratio (R/R0)_c
    alpha_rt: float = 0.283  # void growth rate parameter alpha
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailBaoWierzbicki:
    """``/FAIL/BAO_WIERZBICKI/mat_ID`` (M241): Bao-Wierzbicki fracture locus failure model."""
    mat_id: int = 0
    title: str = ""
    c1: float = 0.0          # fracture locus parameter C1
    c2: float = 0.0          # fracture locus parameter C2
    c3: float = 0.0          # fracture locus parameter C3
    eta0: float = 0.333      # transition triaxiality eta_0 (default 1/3)
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailLouHuhn:
    """``/FAIL/LOU_HUHN/mat_ID`` (M242): Lou-Huhn shear ductile fracture failure model."""
    mat_id: int = 0
    title: str = ""
    c1: float = 0.0          # material parameter C1
    c2: float = 0.0          # material parameter C2
    l_param: float = 1.0     # characteristic parameter L
    eta0: float = 0.333      # cutoff / reference triaxiality eta_0
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailHollomon:
    """``/FAIL/HOLLOMON/mat_ID`` (M243): Hollomon power-law strain hardening fracture criterion."""
    mat_id: int = 0
    title: str = ""
    eps0: float = 0.0        # reference plastic strain epsilon_0
    n_exp: float = 0.2       # strain hardening exponent n
    k_coeff: float = 0.0     # strength coefficient K
    eps_max: float = 1e30    # maximum failure strain epsilon_max
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailSwift:
    """``/FAIL/SWIFT/mat_ID`` (M244): Swift power-law strain hardening fracture criterion."""
    mat_id: int = 0
    title: str = ""
    eps0: float = 0.0        # reference plastic strain epsilon_0
    n_exp: float = 0.2       # strain hardening exponent n
    k_coeff: float = 0.0     # strength coefficient K
    eps_max: float = 1e30    # maximum failure strain epsilon_max
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailLudwik:
    """``/FAIL/LUDWIK/mat_ID`` (M245): Ludwik power-law strain hardening fracture criterion."""
    mat_id: int = 0
    title: str = ""
    sigma0: float = 0.0      # reference yield stress sigma_0
    k_coeff: float = 0.0     # strength coefficient K
    n_exp: float = 0.2       # strain hardening exponent n
    eps_max: float = 1e30    # maximum failure strain epsilon_max
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailVoce:
    """``/FAIL/VOCE/mat_ID`` (M246): Voce isotropic saturation hardening fracture criterion."""
    mat_id: int = 0
    title: str = ""
    sigma0: float = 0.0      # initial yield stress sigma_0
    sigma_inf: float = 0.0   # saturation stress sigma_inf
    beta: float = 1.0        # saturation rate beta
    eps_max: float = 1e30    # maximum failure strain epsilon_max
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailGhosh:
    """``/FAIL/GHOSH/mat_ID`` (M247): Ghosh power-law strain hardening fracture criterion."""
    mat_id: int = 0
    title: str = ""
    sigma0: float = 0.0      # reference yield stress sigma_0
    k_coeff: float = 0.0     # strength coefficient K
    eps0: float = 0.0        # pre-strain offset epsilon_0
    n_exp: float = 0.2       # strain hardening exponent n
    eps_max: float = 1e30    # maximum failure strain epsilon_max
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailSwiftVoce:
    """``/FAIL/SWIFT_VOCE/mat_ID`` (M248): Combined Swift-Voce power-law & saturation hardening fracture criterion."""
    mat_id: int = 0
    title: str = ""
    alpha: float = 0.5       # Swift weighting factor alpha in [0, 1]
    k_coeff: float = 0.0     # Swift strength coefficient K
    eps0: float = 0.0        # Swift pre-strain offset epsilon_0
    n_exp: float = 0.2       # Swift strain hardening exponent n
    sigma0: float = 0.0      # Voce initial yield stress sigma_0
    sigma_inf: float = 0.0   # Voce saturation stress sigma_inf
    beta: float = 1.0        # Voce saturation rate beta
    eps_max: float = 1e30    # maximum failure strain epsilon_max
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailBonora:
    """``/FAIL/BONORA/mat_ID`` (M249): Bonora non-linear continuous ductile damage failure criterion."""
    mat_id: int = 0
    title: str = ""
    p_th: float = 0.0        # threshold plastic strain for damage initiation
    p_cr: float = 1.0        # critical plastic strain at failure
    d_cr: float = 0.85       # critical damage threshold
    d_0: float = 0.0         # initial damage
    alpha: float = 0.5       # damage non-linear exponent
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailRitchie:
    """``/FAIL/RITCHIE_KNOTT_RICE/mat_ID`` (M250): Ritchie-Knott-Rice cleavage fracture criterion."""
    mat_id: int = 0
    title: str = ""
    sigma_c: float = 0.0     # critical cleavage fracture stress sigma_c
    l_star: float = 0.0      # characteristic microstructural distance l*
    eps_init: float = 0.0    # initial plastic strain threshold eps_init
    d_crit: float = 0.99     # critical damage threshold D_c
    eps_max: float = 1e30    # maximum equivalent plastic strain to failure
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailGologanu:
    """``/FAIL/GOLOGANU/mat_ID`` (M251): Gologanu-Leblond-Devaux void shape evolution failure criterion."""
    mat_id: int = 0
    title: str = ""
    f0: float = 0.001        # initial void volume fraction
    s0: float = 1.0          # initial void aspect ratio S = ln(a/b)
    fc: float = 0.15         # critical void volume fraction for coalescence
    ff: float = 0.25         # void volume fraction at fracture
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailRousselier:
    """``/FAIL/ROUSSELIER/mat_ID`` (M252): Rousselier ductile damage and void growth fracture criterion."""
    mat_id: int = 0
    title: str = ""
    d0: float = 0.0001       # initial damage parameter / void volume fraction D0
    sigma1: float = 500.0    # characteristic stress sigma1
    d_crit: float = 1.0      # critical damage threshold Dc
    eps_init: float = 0.0    # plastic strain threshold for damage onset
    eps_max: float = 1e30    # maximum plastic strain to failure
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailHockettSherby:
    """``/FAIL/HOCKETT_SHERBY/mat_ID`` (M253): Hockett-Sherby saturation hardening fracture criterion."""
    mat_id: int = 0
    title: str = ""
    sigma0: float = 0.0      # initial yield stress sigma0
    sigma_s: float = 0.0     # saturation stress sigma_s
    m_exp: float = 1.0       # saturation rate coefficient m
    n_exp: float = 1.0       # hardening exponent n
    eps_max: float = 1e30    # maximum plastic strain to failure
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailKimBaek:
    """``/FAIL/KIM_BAEK/mat_ID`` (M254): Kim-Baek ductile damage and rate-dependent fracture criterion."""
    mat_id: int = 0
    title: str = ""
    sigma0: float = 0.0      # reference yield stress sigma0
    k_coeff: float = 0.0     # hardening strength coefficient K
    n_exp: float = 0.2       # strain hardening exponent n
    c_rate: float = 0.0      # strain rate sensitivity coefficient C
    eps0_dot: float = 1.0    # reference strain rate eps0_dot
    eps_max: float = 1e30    # maximum equivalent plastic strain
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailBaiWierzbicki:
    """``/FAIL/BAI_WIERZBICKI/mat_ID`` (M255): Bai-Wierzbicki stress triaxiality and Lode angle dependent asymmetric fracture criterion."""
    mat_id: int = 0
    title: str = ""
    c1: float = 0.0          # triaxiality parameter c1
    c2: float = 0.0          # triaxiality parameter c2
    c3: float = 0.0          # triaxiality parameter c3
    c4: float = 0.0          # triaxiality parameter c4
    c_theta: float = 0.0     # Lode angle sensitivity parameter c_theta
    eps_max: float = 1e30    # maximum equivalent plastic strain
    ifail_sh: int = 1        # shell element deletion flag


@dataclass
class FailJh2:
    """``/FAIL/JH2/mat_ID`` (M256): Johnson-Holmquist ceramic/brittle damage failure criterion."""
    mat_id: int = 0
    title: str = ""
    d1: float = 0.045            # damage coefficient D1
    d2: float = 1.0              # damage exponent D2
    c_rate: float = 0.0          # strain rate sensitivity coefficient C
    t_star: float = 0.0          # normalized tensile strength T*
    eps0_dot: float = 1.0        # reference strain rate eps0_dot
    ifail_sh: int = 1            # shell element deletion flag


@dataclass
class FailRht:
    """``/FAIL/RHT/mat_ID`` (M257): Riedel-Hiermaier-Thoma concrete/rock damage failure criterion."""
    mat_id: int = 0
    title: str = ""
    d1: float = 0.04             # damage parameter D1
    d2: float = 1.0              # damage exponent D2
    p_spall: float = 0.0         # normalized spall tensile pressure P_spall*
    eps_min: float = 0.0         # minimum failure strain threshold eps_p_min
    ifail_sh: int = 1            # shell element deletion flag


# ============================================================================
# M264 Suite: Johnson-Cook failure, EngMaxShear, TransferCase, SensorSpringMomentImpulse
# ============================================================================

@dataclass
class FailJohnsonCook:
    """``/FAIL/JOHNSON_COOK/mat_ID`` (M264): Johnson-Cook 3D dynamic ductile damage failure model."""
    mat_id: int = 1
    title: str = ""
    d1: float = 0.0              # initial fracture strain coefficient D1
    d2: float = 0.0              # exponential triaxiality coefficient D2
    d3: float = 0.0              # triaxiality exponent D3
    d4: float = 0.0              # strain rate sensitivity coefficient D4
    d5: float = 0.0              # temperature softening coefficient D5
    eps_dot_0: float = 1.0       # reference quasi-static strain rate
    t_room: float = 293.15       # reference room temperature
    t_melt: float = 1793.15      # material melting temperature
    m_exp: float = 1.0           # thermal softening exponent m
    ifail_sh: int = 1            # shell deletion flag (1=one layer fails, 2=all layers fail)
    d_max: float = 1.0           # maximum accumulated damage threshold


# ============================================================================
# M265 Suite: Cockcroft-Latham failure, EngEffectiveStress, TorqueSplitGear, SensorSpringTorsionalEnergy
# ============================================================================

@dataclass
class FailCockcroftLatham:
    """``/FAIL/COCKCROFT_LATHAM/mat_ID`` (M265): Cockcroft-Latham ductile fracture failure model."""
    mat_id: int = 1
    title: str = ""
    w_crit: float = 0.0          # critical tensile plastic work per unit volume
    c_rate: float = 0.0          # strain rate sensitivity exponent C
    eps_dot_0: float = 1.0       # reference quasi-static strain rate
    ifail_sh: int = 1            # shell deletion flag (1=one layer fails, 2=all layers fail)
    ifail_so: int = 1            # solid element deletion flag
    d_max: float = 1.0           # maximum accumulated damage threshold


# ============================================================================
# M266 Suite: Lemaitre damage, EngHydrostaticPressure, GenevaDrive, SensorSpringBendingEnergy
# ============================================================================

@dataclass
class FailLemaitreDamage:
    """``/FAIL/LEMAITRE_DAMAGE/mat_ID`` (M266): Lemaitre continuum ductile damage failure model."""
    mat_id: int = 1
    title: str = ""
    s_coeff: float = 10.0        # damage strength coefficient S
    s_exp: float = 1.0           # damage exponent s
    eps_d: float = 0.0           # damage initiation plastic strain threshold eps_D
    d_c: float = 1.0             # critical damage at fracture D_c (default 1.0)
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag

    @property
    def d_crit(self) -> float:
        return self.d_c

    @d_crit.setter
    def d_crit(self, val: float) -> None:
        self.d_c = val


# ============================================================================
# M267 Suite: TabulatedPlasticity failure, EngOctahedralShear, ScotchYoke, SensorSpringTotalStrainEnergy
# ============================================================================

@dataclass
class FailTabulatedPlasticity:
    """``/FAIL/TABULATED_PLASTICITY/mat_ID`` (M267): Tabulated multi-axial plasticity fracture failure model."""
    mat_id: int = 1
    title: str = ""
    fct_id_triax: int = 0        # curve function ID for failure strain vs triaxiality eta
    fct_id_lode: int = 0         # curve function ID for failure strain vs Lode parameter theta
    fct_id_rate: int = 0         # curve function ID for dynamic strain rate scale factor
    ifail_sh: int = 1            # shell deletion flag (1=one layer fails, 2=all layers fail)
    ifail_so: int = 1            # solid element deletion flag
    d_max: float = 1.0           # maximum accumulated damage threshold


# ============================================================================
# M268 Suite: MohrCoulomb failure, EngDeviatoricEnergy, OldhamCoupling, SensorSpringVolumetricEnergy
# ============================================================================

@dataclass
class FailMohrCoulomb:
    """``/FAIL/MOHR_COULOMB/mat_ID`` (M268): Mohr-Coulomb pressure-dependent shear failure model."""
    mat_id: int = 1
    title: str = ""
    cohesion: float = 0.0        # material cohesion strength c
    phi: float = 0.0             # internal friction angle in degrees
    tens_limit: float = 1e30     # tensile cutoff limit
    dilatancy: float = 0.0       # dilatancy angle in degrees
    ifail_sh: int = 1            # shell deletion flag (1=one layer fails, 2=all layers fail)
    ifail_so: int = 1            # solid element deletion flag
    d_max: float = 1.0           # maximum accumulated damage threshold


# ============================================================================
# M269 Suite: DruckerPrager failure, EngStrainRate, SchmidtCoupling, SensorSpringShearEnergy
# ============================================================================

@dataclass
class FailDruckerPrager:
    """``/FAIL/DRUCKER_PRAGER/mat_ID`` (M269): Drucker-Prager pressure-dependent yield failure model."""
    mat_id: int = 1
    title: str = ""
    alpha: float = 0.0           # pressure sensitivity coefficient
    k_dp: float = 0.0            # initial yield threshold in sqrt(J2) space
    tens_limit: float = 1e30     # tensile meridian cutoff stress
    comp_limit: float = 1e30     # compressive meridian cutoff stress
    ifail_sh: int = 1            # shell deletion flag (1=one layer fails, 2=all layers fail)
    ifail_so: int = 1            # solid element deletion flag
    d_max: float = 1.0           # maximum accumulated damage threshold


# ============================================================================
# M270 Suite: HosfordCoulomb failure, EngBulkViscosity, RzeppaJoint, SensorSpringAxialEnergy
# ============================================================================

@dataclass
class FailHosfordCoulomb:
    """``/FAIL/HOSFORD_COULOMB/mat_ID`` (M270): Hosford-Coulomb ductile fracture failure model."""
    mat_id: int = 1
    title: str = ""
    a_hc: float = 0.0            # Hosford exponent (a >= 1)
    b_hc: float = 0.0            # friction coefficient b
    c_hc: float = 0.0            # cohesion strength c
    n_hc: float = 0.0            # damage exponent n
    ifail_sh: int = 1            # shell deletion flag (1=one layer fails, 2=all layers fail)
    ifail_so: int = 1            # solid element deletion flag
    d_max: float = 1.0           # maximum accumulated damage threshold


# ============================================================================
# M271 Suite: BiquadAniso failure, EngHourglassEnergy, BirfieldJoint, SensorSpringDampingEnergy
# ============================================================================

@dataclass
class FailBiquadAniso:
    """``/FAIL/BIQUAD_ANISO/mat_ID`` (M271): Biquadratic anisotropic yield failure model."""
    mat_id: int = 1
    title: str = ""
    sigma_1t: float = 1e30      # tensile strength direction 1
    sigma_1c: float = 1e30      # compressive strength direction 1
    sigma_2t: float = 1e30      # tensile strength direction 2
    sigma_2c: float = 1e30      # compressive strength direction 2
    ifail_sh: int = 1            # shell deletion flag (1=one layer fails, 2=all layers fail)
    ifail_so: int = 1            # solid element deletion flag
    d_max: float = 1.0           # maximum accumulated damage threshold


# ============================================================================
# M272 Suite: WilkinsCumulative failure, EngContactEnergy, TripodJoint, SensorSpringCouplingEnergy
# ============================================================================

@dataclass
class FailWilkinsCumulative:
    """``/FAIL/WILKINS_CUMULATIVE/mat_ID`` (M272): Wilkins cumulative damage failure model."""
    mat_id: int = 1
    title: str = ""
    d_crit: float = 1.0          # critical cumulative damage threshold
    a_wk: float = 0.0            # Wilkins damage exponent a
    b_wk: float = 0.0            # Wilkins pressure weighting coefficient b
    p_min: float = 0.0           # minimum hydrostatic pressure cutoff
    ifail_sh: int = 1            # shell deletion flag (1=one layer fails, 2=all layers fail)
    ifail_so: int = 1            # solid element deletion flag
    d_max: float = 1.0           # maximum accumulated damage threshold


# ============================================================================
# M273 Suite: TulerButcher failure, EngSpringEnergy, HookeJoint, SensorSpringTorsionalEnergy
# ============================================================================

@dataclass
class FailTulerButcher:
    """``/FAIL/TULER_BUTCHER/mat_ID`` (M273): Tuler-Butcher spall failure model."""
    mat_id: int = 1
    title: str = ""
    sigma_spall: float = 1e30   # spall stress threshold
    k_tb: float = 0.0           # Tuler-Butcher damage coefficient K
    lambda_tb: float = 2.0      # Tuler-Butcher stress exponent lambda
    d_crit: float = 1.0         # critical cumulative damage for element deletion
    ifail_sh: int = 1           # shell deletion flag (1=one layer, 2=all layers)
    ifail_so: int = 1           # solid element deletion flag
    d_max: float = 1.0          # maximum accumulated damage threshold


# SensorSpringTorsionalEnergy: canonical definition is above (M265 section) with u_tors_max alias.



# ============================================================================
# M274 Suite: ExtendedMohr failure, EngJointEnergy, TractaJoint, SensorSpringBendingEnergy
# ============================================================================

@dataclass
class FailExtendedMohr:
    """``/FAIL/EXTENDED_MOHR/mat_ID`` (M274): Extended Mohr-Coulomb failure model."""
    mat_id: int = 1
    title: str = ""
    c_0: float = 0.0            # cohesion parameter c0
    c_1: float = 0.0            # friction parameter c1 (Lode angle)
    c_2: float = 0.0            # pressure dependence parameter c2
    c_theta: float = 0.0        # Lode-angle sensitivity parameter
    ifail_sh: int = 1           # shell deletion flag (1=one layer, 2=all layers)
    ifail_so: int = 1           # solid element deletion flag
    d_max: float = 1.0          # maximum accumulated damage threshold


# SensorSpringBendingEnergy: canonical definition is above (M266 section) with u_bend_max alias.



# ============================================================================
# M275 Suite: Oyane failure, EngRwallEnergy, ThompsonCoupling, SensorSpringPinchingEnergy
# ============================================================================

@dataclass
class FailOyane:
    """``/FAIL/OYANE/mat_ID`` (M275): Oyane porous ductile fracture failure model."""
    mat_id: int = 1
    title: str = ""
    c_oyane: float = 0.0        # critical fracture parameter C
    b_oyane: float = 0.0        # stress triaxiality scale factor B
    sigma_cut: float = 1e30     # tensile cutoff stress limit
    eps_p_min: float = 0.0      # minimum plastic strain threshold to accumulate damage
    ifail_sh: int = 1           # shell deletion flag (1=one layer, 2=all layers)
    ifail_so: int = 1           # solid element deletion flag
    d_max: float = 1.0          # maximum accumulated damage threshold


# ============================================================================
# M276 Suite: Freudenthal failure, EngSurfEnergy, WeissJoint, SensorSpringFrictionEnergy
# ============================================================================

@dataclass
class FailFreudenthal:
    """``/FAIL/FREUDENTHAL/mat_ID`` (M276): Freudenthal critical plastic work ductile failure model."""
    mat_id: int = 1
    title: str = ""
    w_crit: float = 1e30        # critical plastic work density threshold
    sigma_cut: float = 1e30     # tensile cutoff stress limit
    eps_p_min: float = 0.0      # minimum plastic strain threshold to accumulate work
    ifail_sh: int = 1           # shell deletion flag (1=one layer, 2=all layers)
    ifail_so: int = 1           # solid element deletion flag
    d_max: float = 1.0          # maximum accumulated damage threshold


# ============================================================================
# M277 Suite: Alter failure, EngHeatExchange, TripodBallJoint, SensorSpringThermalDissipation
# ============================================================================

@dataclass
class FailAlter:
    """``/FAIL/ALTER/mat_ID`` (M277): Alter subcritical crack growth failure model for glass / brittle materials."""
    mat_id: int = 1
    title: str = ""
    exp_n: float = 1.0           # crack growth exponent for subcritical crack growth
    v0: float = 0.0              # crack growth velocity at KIC
    vc: float = 1e30             # maximum crack propagation velocity
    ema: int = 0                 # stress filtering period in cycles (NCYCLES)
    irate: int = 0               # stress filtering method
    iside: int = 0               # strain rate dependency flag on air/foil side
    mode: int = 0                # failure propagation model switch flag
    cr_foil: float = 0.0         # crack depth at PVB / foil surface
    cr_air: float = 0.0          # crack depth at air surface
    cr_core: float = 0.0         # crack depth in core integration points
    cr_edge: float = 0.0         # crack depth exposed surface
    grsh4n: int = 0              # shell 4N group ID
    grsh3n: int = 0              # shell 3N group ID
    kic: float = 1e30            # fracture toughness
    kth: float = 0.0             # fatigue threshold
    rlen: float = 0.0            # reference length
    tdel: float = 0.0            # time delay of stress relaxation
    kres1: float = 0.0           # residual stress scale factor 1
    kres2: float = 0.0           # residual stress scale factor 2
    fail_id: int = 0             # user failure ID
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    d_max: float = 1.0           # maximum damage threshold


# ============================================================================
# M285 Suite: LouHuo failure, EngCoriolisEnergy, CablePulleyJoint, SensorSpringAngularVelocity
# ============================================================================

@dataclass
class FailLouHuo:
    """``/FAIL/LOU_HUO`` or ``/FAIL/LOU_HUO_YANG`` (M285): Lou-Huo-Yang shear ductile fracture criterion."""
    mat_id: int = 0
    title: str = ""
    c1: float = 0.0              # shear stress sensitivity coefficient
    c2: float = 0.0              # stress triaxiality weighting factor
    c3: float = 0.0              # equivalent plastic strain exponent
    l_param: float = 1.0         # non-proportional loading factor (L)
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M286 Suite: LadStr failure, EngMagneticEnergy, SwashPlateJoint, SensorSpringAngularAcceleration
# ============================================================================

@dataclass
class FailLadStr:
    """``/FAIL/LAD_STR`` or ``/FAIL/LADEVEZE_STRESS`` (M286): Ladevèze stress-based composite damage and ply failure criterion."""
    mat_id: int = 0
    title: str = ""
    r0_1: float = 0.0            # initial damage threshold in fiber direction
    r0_2: float = 0.0            # initial damage threshold in transverse direction
    rc_1: float = 0.0            # critical damage threshold in fiber direction
    rc_2: float = 0.0            # critical damage threshold in transverse direction
    b_lad: float = 0.0           # shear coupling parameter
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M287 Suite: LadVisc failure, EngPoyntingEnergy, ScissorMechanismJoint, SensorSpringTorsionalRate
# ============================================================================

@dataclass
class FailLadVisc:
    """``/FAIL/LAD_VISC`` or ``/FAIL/LADEVEZE_VISCOUS`` (M287): Ladevèze rate-dependent viscoplastic composite damage and ply failure criterion."""
    mat_id: int = 0
    title: str = ""
    y0: float = 0.0              # initial thermodynamic damage force threshold
    yc: float = 0.0              # critical thermodynamic damage force threshold
    a_lad: float = 0.0           # damage kinematic hardening coefficient
    p_visc: float = 0.0          # viscoplastic relaxation power exponent
    m_visc: float = 0.0          # viscosity rate sensitivity exponent
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M288 Suite: LadInter failure, EngMaxwellStressEnergy, ParallelogramJoint, SensorSpringNormalAcceleration
# ============================================================================

@dataclass
class FailLadInter:
    """``/FAIL/LAD_INTER`` or ``/FAIL/LADEVEZE_INTER`` (M288): Ladevèze interfacial delamination and inter-ply debonding failure criterion."""
    mat_id: int = 0
    title: str = ""
    k_n: float = 0.0             # initial normal interfacial stiffness
    k_s: float = 0.0             # initial shear interfacial stiffness
    y0_inter: float = 0.0        # initial interfacial damage energy threshold
    yc_inter: float = 0.0        # critical interfacial fracture energy threshold
    eta_inter: float = 0.0       # mixed-mode damage coupling interaction exponent
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M289 Suite: LadFib failure, EngJouleHeatEnergy, DeltaRobotJoint, SensorSpringShearAcceleration
# ============================================================================

@dataclass
class FailLadFib:
    """``/FAIL/LAD_FIB`` or ``/FAIL/LADEVEZE_FIBER`` (M289): Ladevèze longitudinal fiber brittle rupture and microbuckling failure criterion."""
    mat_id: int = 0
    title: str = ""
    eps_ft: float = 0.0          # longitudinal tensile failure strain
    eps_fc: float = 0.0          # longitudinal compressive microbuckling strain
    sigma_ft: float = 0.0        # longitudinal tensile failure stress
    sigma_fc: float = 0.0        # longitudinal compressive microbuckling stress
    gamma_fib: float = 0.0       # fiber shear coupling damage coefficient
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M290 Suite: LadMicro failure, EngLorentzForceEnergy, SphericalWristJoint, SensorSpringResultantAcceleration
# ============================================================================

@dataclass
class FailLadMicro:
    """``/FAIL/LAD_MICRO`` or ``/FAIL/LADEVEZE_MICRO`` (M290): Ladevèze micromechanical damage evolution failure model."""
    mat_id: int = 0
    title: str = ""
    d0_micro: float = 0.0        # initial microcrack damage threshold
    dc_micro: float = 0.0        # critical microcrack coalescence damage threshold
    alpha_micro: float = 0.0     # micro-debonding kinetic exponent
    beta_micro: float = 0.0      # matrix microcracking growth exponent
    s_micro: float = 0.0         # microcrack characteristic damage scale factor
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M291 Suite: LadCouple failure, EngPlasmonicEnergy, LeadScrewJoint, SensorSpringTorsionalAcceleration
# ============================================================================

@dataclass
class FailLadCouple:
    """``/FAIL/LAD_COUPLE`` or ``/FAIL/LADEVEZE_COUPLED`` (M291): Ladevèze thermo-mechanically coupled damage and ply degradation failure model."""
    mat_id: int = 0
    title: str = ""
    t_ref: float = 293.15        # reference temperature (K)
    beta_th: float = 0.0         # thermal softening coefficient
    c_th: float = 0.0            # thermo-mechanical damage coupling parameter
    d_th_max: float = 0.999      # maximum thermo-coupled damage limit
    gamma_th: float = 0.0        # thermal expansion damage rate exponent
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M292 Suite: LadViscoPlast failure, EngDielectricLossEnergy, HoekenLinkageJoint, SensorSpringBendingAcceleration
# ============================================================================

@dataclass
class FailLadViscoPlast:
    """``/FAIL/LAD_VISCO_PLAST`` or ``/FAIL/LADEVEZE_VISCO_PLASTIC`` (M292): Ladevèze strain rate-dependent viscoplastic micro-damage and dynamic hardening failure criterion."""
    mat_id: int = 0
    title: str = ""
    gamma_vp: float = 0.0        # viscoplastic rate sensitivity coefficient
    m_vp: float = 1.0            # viscoplastic power-law rate exponent
    a_vp: float = 0.0            # isotropic dynamic hardening modulus
    p_vp: float = 1.0            # dynamic hardening power exponent
    d_max_vp: float = 0.999      # maximum rate-coupled damage limit
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M293 Suite: LadCreep failure, EngMagneticHysteresisEnergy, ChebyshevLinkageJoint, SensorSpringTotalAngularAcceleration
# ============================================================================

@dataclass
class FailLadCreep:
    """``/FAIL/LAD_CREEP`` or ``/FAIL/LADEVEZE_CREEP`` (M293): Ladevèze high-temperature tertiary creep rupture and time-dependent damage evolution model."""
    mat_id: int = 0
    title: str = ""
    a_creep: float = 0.0         # creep damage rate coefficient
    n_creep: float = 1.0         # Norton power-law stress exponent
    q_creep: float = 0.0         # creep thermal activation energy (J/mol)
    t_creep_ref: float = 293.15  # creep reference temperature (K)
    d_creep_max: float = 0.999   # maximum allowable tertiary creep damage
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M294 Suite: LadTransIsotropic failure, EngMagnetostrictionEnergy, RobertsLinkageJoint, SensorSpringNormalJerk
# ============================================================================

@dataclass
class FailLadTransIsotropic:
    """``/FAIL/LAD_TRANS_ISOTROPIC`` or ``/FAIL/LADEVEZE_TRANSVERSE_ISOTROPIC`` (M294): Ladevèze transversely isotropic fiber-reinforced composite damage evolution model."""
    mat_id: int = 0
    title: str = ""
    d1_max: float = 0.999        # maximum longitudinal fiber direction damage
    d2_max: float = 0.999        # maximum transverse in-plane matrix damage
    d3_max: float = 0.999        # maximum out-of-plane through-thickness damage
    y1_crit: float = 0.0         # critical thermodynamic force threshold in direction 1
    y2_crit: float = 0.0         # critical thermodynamic force threshold in direction 2
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M295 Suite: LadViscoDamage failure, EngElectrocaloricEnergy, EvansLinkageJoint, SensorSpringShearJerk
# ============================================================================

@dataclass
class FailLadViscoDamage:
    """``/FAIL/LAD_VISCO_DAMAGE`` or ``/FAIL/LADEVEZE_VISCO_DAMAGE`` (M295): Ladevèze rate-dependent micro-damage kinetics and delayed damage evolution model."""
    mat_id: int = 0
    title: str = ""
    tau_c: float = 0.0           # characteristic damage relaxation delay time constant (s)
    a_vd: float = 0.0            # rate-dependent damage kinetic multiplier
    n_vd: float = 1.0            # rate-dependent damage power-law exponent
    d_vd_crit: float = 0.0       # critical micro-damage threshold for accelerated evolution
    d_vd_max: float = 0.999      # maximum allowable damage parameter
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M296 Suite: LadDelam failure, EngMagnetocaloricEnergy, WattLinkageJoint, SensorSpringResultantJerk
# ============================================================================

@dataclass
class FailLadDelam:
    """``/FAIL/LAD_DELAM`` or ``/FAIL/LADEVEZE_INTERLAMINAR_DELAMINATION`` (M296): Ladevèze interlaminar delamination and interface fracture criterion."""
    mat_id: int = 0
    title: str = ""
    g_1c: float = 0.0            # Mode I critical fracture energy release rate (J/m^2)
    g_2c: float = 0.0            # Mode II critical fracture energy release rate (J/m^2)
    g_3c: float = 0.0            # Mode III critical fracture energy release rate (J/m^2)
    gamma_delam: float = 1.0     # mixed-mode Benzeggagh-Kenane interaction exponent
    d_delam_max: float = 0.999   # maximum allowable delamination damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M297 Suite: LadTcAsymmetry failure, EngThermoelectricEnergy, HartLinkageJoint, SensorSpringTorsionalJerk
# ============================================================================

@dataclass
class FailLadTcAsymmetry:
    """``/FAIL/LAD_TC_ASYMMETRY`` or ``/FAIL/LADEVEZE_TENSION_COMPRESSION_ASYMMETRY`` (M297): Ladevèze tension-compression asymmetry and bimodal damage evolution model."""
    mat_id: int = 0
    title: str = ""
    y0_t: float = 0.0            # tensile initial micro-damage energy threshold
    yc_t: float = 0.0            # tensile critical damage rupture energy
    y0_c: float = 0.0            # compressive initial micro-damage energy threshold
    yc_c: float = 0.0            # compressive critical damage rupture energy
    d_tc_max: float = 0.999      # maximum allowable tension-compression damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M298 Suite: LadAniso failure, EngPyroelectricEnergy, PeaucellierLinkageJoint, SensorSpringBendingJerk
# ============================================================================

@dataclass
class FailLadAniso:
    """``/FAIL/LAD_ANISO`` or ``/FAIL/LADEVEZE_ANISOTROPIC_DAMAGE`` (M298): Ladevèze 3D anisotropic continuum damage mechanics and multi-axial micro-cracking evolution failure model."""
    mat_id: int = 0
    title: str = ""
    y0_1: float = 0.0            # axial initial micro-damage energy threshold
    yc_1: float = 0.0            # axial critical damage rupture energy
    y0_2: float = 0.0            # transverse initial micro-damage energy threshold
    yc_2: float = 0.0            # transverse critical damage rupture energy
    d_aniso_max: float = 0.999   # maximum allowable anisotropic damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M299 Suite: LadFatigue failure, EngThermomagneticEnergy, SarrusLinkageJoint, SensorSpringTotalAngularJerk
# ============================================================================

@dataclass
class FailLadFatigue:
    """``/FAIL/LAD_FATIGUE`` or ``/FAIL/LADEVEZE_HIGH_CYCLE_FATIGUE`` (M299): Ladevèze cyclic micro-damage accumulation and high-cycle fatigue failure model."""
    mat_id: int = 0
    title: str = ""
    y0_fatigue: float = 0.0      # fatigue micro-damage activation threshold energy
    yc_fatigue: float = 0.0      # critical fatigue rupture energy
    beta_fatigue: float = 0.0    # cyclic damage accumulation exponent
    alpha_fatigue: float = 0.0   # stress triaxiality sensitivity coefficient
    d_fat_max: float = 0.999     # maximum allowable fatigue damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M300 Suite: LadViscoFatigue failure, EngThermogalvanicEnergy, KlannLinkageJoint, SensorSpringTotalAccelerationRate
# ============================================================================

@dataclass
class FailLadViscoFatigue:
    """``/FAIL/LAD_VISCO_FATIGUE`` or ``/FAIL/LADEVEZE_VISCO_FATIGUE`` (M300): Ladevèze strain-rate sensitive visco-fatigue micro-damage and frequency-dependent cyclic degradation failure model."""
    mat_id: int = 0
    title: str = ""
    y0_vf: float = 0.0           # visco-fatigue micro-damage activation threshold energy
    yc_vf: float = 0.0           # critical visco-fatigue rupture energy
    beta_vf: float = 0.0         # cyclic damage accumulation exponent
    tau_relax: float = 0.0       # viscous micro-damage characteristic relaxation time
    d_vf_max: float = 0.999      # maximum allowable visco-fatigue damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M301 Suite: LadCoupleDamage failure, EngThermionicEnergy, JansenLinkageJoint, SensorSpringNormalAccelerationRate
# ============================================================================

@dataclass
class FailLadCoupleDamage:
    """``/FAIL/LAD_COUPLE_DAMAGE`` or ``/FAIL/LADEVEZE_COUPLED_DAMAGE`` (M301): Ladevèze fully coupled thermo-elasto-damage model with non-isothermal microcracking kinetics."""
    mat_id: int = 0
    title: str = ""
    y0_cd: float = 0.0           # coupled damage activation threshold energy
    yc_cd: float = 0.0           # critical coupled damage rupture energy
    gamma_temp: float = 0.0      # temperature coupling expansion coefficient
    eta_entropy: float = 0.0     # entropy production weighting factor
    d_cd_max: float = 0.999      # maximum allowable coupled damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M302 Suite: LadFracture failure, EngThermophotonicEnergy, HoekenLinkageJoint, SensorSpringTransverseAccelerationRate
# ============================================================================

@dataclass
class FailLadFracture:
    """``/FAIL/LAD_FRACTURE`` or ``/FAIL/LADEVEZE_DYNAMIC_FRACTURE`` (M302): Ladevèze dynamic microcrack coalescence and cohesive zone fracture transition failure model."""
    mat_id: int = 0
    title: str = ""
    y0_frac: float = 0.0         # dynamic fracture microcrack activation threshold energy
    yc_frac: float = 0.0         # critical cohesive fracture rupture energy
    gamma_cohes: float = 0.0     # cohesive softening transition exponent
    l_char: float = 0.0          # characteristic cohesive fracture process zone length
    d_frac_max: float = 0.999    # maximum allowable fracture damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M303 Suite: LadFiberMatrixDebonding failure, EngElectrostrictiveEnergy, SylvesterKempeLinkageJoint, SensorSpringTorsionalAccelerationRate
# ============================================================================

@dataclass
class FailLadFiberMatrixDebonding:
    """``/FAIL/LAD_FIBER_MATRIX_DEBONDING`` or ``/FAIL/LADEVEZE_FIBER_MATRIX_DEBOND`` (M303): Ladevèze micromechanical fiber-matrix interface debonding and interfacial shear slip failure model."""
    mat_id: int = 0
    title: str = ""
    y0_fmd: float = 0.0          # interface debonding activation threshold energy
    yc_fmd: float = 0.0          # critical interfacial debonding rupture energy
    tau_fmd_crit: float = 0.0    # critical interfacial shear stress limit
    mu_fmd_fric: float = 0.0     # interfacial post-debonding sliding friction coefficient
    d_fmd_max: float = 0.999     # maximum allowable debonding damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M304 Suite: LadFiberKinking failure, EngPhotomagneticEnergy, WobbleYokeJoint, SensorSpringBendingAccelerationRate
# ============================================================================

@dataclass
class FailLadFiberKinking:
    """``/FAIL/LAD_FIBER_KINKING`` or ``/FAIL/LADEVEZE_FIBER_KINK`` (M304): Ladevèze compressive fiber kinking and localized shear band microbuckling failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_kink_crit: float = 0.0 # critical compressive fiber kinking stress
    phi_kink_0: float = 0.0      # initial fiber misalignment angle (radians)
    gamma_kink: float = 0.0      # nonlinear shear band softening parameter
    l_kink_band: float = 0.0     # characteristic kink-band process zone width
    d_kink_max: float = 0.999    # maximum allowable kinking damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M305 Suite: LadDiffuseDamage failure, EngThermophotonicEmissionEnergy, HypocyclicLinkageJoint, SensorSpringTotalAngularAccelerationRate
# ============================================================================

@dataclass
class FailLadDiffuseDamage:
    """``/FAIL/LAD_DIFFUSE_DAMAGE`` or ``/FAIL/LADEVEZE_DIFFUSE_DAMAGE`` (M305): Ladevèze nonlocal gradient-enhanced diffuse micro-damage and matrix microcrack regularization failure model."""
    mat_id: int = 0
    title: str = ""
    y0_diff: float = 0.0         # diffuse damage initiation energy threshold
    yc_diff: float = 0.0         # critical diffuse microcracking rupture energy
    c_reg_diff: float = 0.0      # viscous gradient regularization coefficient
    l_nonlocal: float = 0.0      # nonlocal internal characteristic length
    d_diff_max: float = 0.999    # maximum allowable diffuse damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M306 Suite: LadInterlaminarShear failure, EngPhononPolaritonEnergy, PantographLinkageJoint, SensorSpringTotalAccelerationJerk
# ============================================================================

@dataclass
class FailLadInterlaminarShear:
    """``/FAIL/LAD_INTERLAMINAR_SHEAR`` or ``/FAIL/LADEVEZE_INTERLAMINAR_SHEAR`` (M306): Ladevèze interlaminar shear microcracking and irreversible shear slip degradation failure model."""
    mat_id: int = 0
    title: str = ""
    y0_ils: float = 0.0          # interlaminar shear damage activation threshold energy
    yc_ils: float = 0.0          # critical interlaminar shear fracture energy
    gamma_ils_p: float = 0.0     # plastic shear slip hardening modulus parameter
    tau_ils_max: float = 0.0     # ultimate interlaminar shear stress capacity limit
    d_ils_max: float = 0.999     # maximum allowable interlaminar shear damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M307 Suite: LadFiberMatrixInteraction failure, EngExcitonPolaritonEnergy, WattParallelMotionJoint, SensorSpringNormalAccelerationJerk
# ============================================================================

@dataclass
class FailLadFiberMatrixInteraction:
    """``/FAIL/LAD_FIBER_MATRIX_INTERACTION`` or ``/FAIL/LADEVEZE_FIBER_MATRIX_INTERACTION`` (M307): Ladevèze combined longitudinal tension/compression and transverse matrix microcracking multi-axial interaction failure model."""
    mat_id: int = 0
    title: str = ""
    y0_fmi: float = 0.0          # multiaxial interaction damage threshold energy
    yc_fmi: float = 0.0          # critical multiaxial interaction fracture energy
    alpha_fmi_trans: float = 1.0 # transverse matrix microcracking coupling factor
    beta_fmi_shear: float = 1.0  # shear microcracking coupling factor
    d_fmi_max: float = 0.999     # maximum allowable interaction damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M308 Suite: LadTransverseTension failure, EngMagnonPolaritonEnergy, ScottRussellLinkageJoint, SensorSpringTransverseAccelerationJerk
# ============================================================================

@dataclass
class FailLadTransverseTension:
    """``/FAIL/LAD_TRANSVERSE_TENSION`` or ``/FAIL/LADEVEZE_TRANSVERSE_TENSION`` (M308): Ladevèze transverse tensile matrix microcracking and irreversible opening damage failure model."""
    mat_id: int = 0
    title: str = ""
    y0_tt: float = 0.0           # transverse tensile damage initiation threshold energy
    yc_tt: float = 0.0           # critical transverse tensile microcracking fracture energy
    eta_tt: float = 1.0          # damage growth non-linear exponent
    sigma_tt_max: float = 0.0    # ultimate transverse tensile stress capacity limit
    d_tt_max: float = 0.999      # maximum allowable transverse tensile damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M309 Suite: LadTransverseCompression failure, EngPiezomagneticEnergy, WattBeamEngineJoint, SensorSpringTorsionalJerkRate
# ============================================================================

@dataclass
class FailLadTransverseCompression:
    """``/FAIL/LAD_TRANSVERSE_COMPRESSION`` or ``/FAIL/LADEVEZE_TRANSVERSE_COMPRESSION`` (M309): Ladevèze transverse compressive matrix microcracking and friction-induced crushing damage failure model."""
    mat_id: int = 0
    title: str = ""
    y0_tc: float = 0.0           # transverse compressive damage activation threshold energy
    yc_tc: float = 0.0           # critical transverse compressive fracture energy
    mu_fric_tc: float = 0.0      # internal microcrack Coulomb friction coefficient
    sigma_tc_max: float = 0.0    # ultimate transverse compressive crushing stress limit
    d_tc_max: float = 0.999      # maximum allowable transverse compressive damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M310 Suite: LadInplaneShear failure, EngBarocaloricEnergy, StephensonLinkageJoint, SensorSpringBendingJerkRate
# ============================================================================

@dataclass
class FailLadInplaneShear:
    """``/FAIL/LAD_INPLANE_SHEAR`` or ``/FAIL/LADEVEZE_INPLANE_SHEAR`` (M310): Ladevèze in-plane shear microcracking and irreversible plastic shear slip damage failure model."""
    mat_id: int = 0
    title: str = ""
    y0_ips: float = 0.0          # in-plane shear damage initiation threshold energy
    yc_ips: float = 0.0          # critical in-plane shear microcracking fracture energy
    gamma_plastic_0: float = 0.0 # initial plastic shear strain threshold
    alpha_ips_slip: float = 0.0  # plastic shear slip hardening rate
    d_ips_max: float = 0.999     # maximum allowable in-plane shear damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M311 Suite: LadInterfacialDelamination failure, EngThermomagnetoelectricEnergy, WobblePlateMechanismJoint, SensorSpringTotalJerkRate
# ============================================================================

@dataclass
class FailLadInterfacialDelamination:
    """``/FAIL/LAD_INTERFACIAL_DELAMINATION`` or ``/FAIL/LADEVEZE_INTERFACIAL_DELAMINATION`` (M311): Ladevèze mixed-mode interlaminar interfacial debonding and cohesive delamination failure model."""
    mat_id: int = 0
    title: str = ""
    y0_ifd: float = 0.0          # interfacial damage activation threshold energy
    yc_ifd: float = 0.0          # critical mixed-mode interfacial fracture toughness
    b_ifd_mix: float = 1.0       # mixed-mode BK parameter exponent
    k_ifd_penalty: float = 1e6   # interfacial penalty penalty stiffness
    d_ifd_max: float = 0.999     # maximum allowable interfacial damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M312 Suite: LadTransverseShearInteraction failure, EngElectrohydrodynamicEnergy, WhitworthQuickReturnJoint, SensorSpringNormalJerkRate
# ============================================================================

@dataclass
class FailLadTransverseShearInteraction:
    """``/FAIL/LAD_TRANSVERSE_SHEAR_INTERACTION`` or ``/FAIL/LADEVEZE_TRANSVERSE_SHEAR_INTERACTION`` (M312): Ladevèze combined transverse tension/compression and in-plane shear microcracking coupling failure model."""
    mat_id: int = 0
    title: str = ""
    y0_tsi: float = 0.0          # coupled damage initiation threshold energy
    yc_tsi: float = 0.0          # critical coupling fracture energy
    gamma_coupling_exp: float = 1.0 # transverse-shear interaction coupling exponent
    sigma_tsi_max: float = 0.0   # multi-axial interaction strength capacity limit
    d_tsi_max: float = 0.999     # maximum allowable coupled damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M313 Suite: LadFiberMatrixDebondRate failure, EngMagnetogalvanicEnergy, ChebyshevLambdaLinkageJoint, SensorSpringTransverseJerkRate
# ============================================================================

@dataclass
class FailLadFiberMatrixDebondRate:
    """``/FAIL/LAD_FIBER_MATRIX_DEBOND_RATE`` or ``/FAIL/LADEVEZE_FIBER_MATRIX_DEBOND_RATE`` (M313): Ladevèze rate-dependent fiber-matrix interfacial debonding microcrack damage failure model."""
    mat_id: int = 0
    title: str = ""
    y0_fmdr: float = 0.0         # dynamic debonding damage initiation threshold energy
    yc_fmdr: float = 0.0         # critical dynamic debonding fracture energy
    c_rate_fmdr: float = 0.0     # viscous strain rate sensitivity constant
    p_rate_fmdr: float = 1.0     # strain rate power law exponent
    d_fmdr_max: float = 0.999    # maximum allowable debonding damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M314 Suite: LadInplaneShearRate failure, EngElastocaloricEnergy, FourBarCrankRockerJoint, SensorSpringTorsionalSnapRate
# ============================================================================

@dataclass
class FailLadInplaneShearRate:
    """``/FAIL/LAD_INPLANE_SHEAR_RATE`` or ``/FAIL/LADEVEZE_INPLANE_SHEAR_RATE`` (M314): Ladevèze rate-dependent in-plane shear microcracking and viscous plastic shear flow damage failure model."""
    mat_id: int = 0
    title: str = ""
    y0_ipsr: float = 0.0         # dynamic shear damage initiation threshold energy
    yc_ipsr: float = 0.0         # critical dynamic shear fracture energy
    gamma_rate_ipsr: float = 0.0 # viscous shear strain rate sensitivity parameter
    n_rate_ipsr: float = 1.0     # shear strain rate power law exponent
    d_ipsr_max: float = 0.999    # maximum allowable shear damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M315 Suite: LadTransverseCompressionRate failure, EngThermophononicEnergy, FourBarDoubleCrankJoint, SensorSpringBendingSnapRate
# ============================================================================

@dataclass
class FailLadTransverseCompressionRate:
    """``/FAIL/LAD_TRANSVERSE_COMPRESSION_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_COMPRESSION_RATE`` (M315): Ladevèze rate-dependent transverse compressive matrix crushing and dynamic friction damage failure model."""
    mat_id: int = 0
    title: str = ""
    y0_tcr: float = 0.0          # dynamic compressive crushing initiation threshold energy
    yc_tcr: float = 0.0          # critical dynamic crushing fracture energy
    c_rate_tcr: float = 0.0      # compressive crushing strain rate sensitivity constant
    p_rate_tcr: float = 1.0      # compressive strain rate power law exponent
    d_tcr_max: float = 0.999     # maximum allowable crushing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference



# ============================================================================
# M316 Suite: LadTransverseTensionRate failure, EngThermoplasmonicEnergy, FourBarDoubleRockerJoint, SensorSpringNormalSnapRate
# ============================================================================

@dataclass
class FailLadTransverseTensionRate:
    """``/FAIL/LAD_TRANSVERSE_TENSION_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_TENSION_RATE`` (M316): Ladevèze rate-dependent transverse tensile matrix microcracking and dynamic cleavage damage failure model."""
    mat_id: int = 0
    title: str = ""
    y0_ttr: float = 0.0          # dynamic transverse tensile microcracking initiation threshold energy
    yc_ttr: float = 0.0          # critical dynamic transverse tensile fracture energy
    c_rate_ttr: float = 0.0      # transverse tensile strain rate sensitivity constant
    p_rate_ttr: float = 1.0      # strain rate power law exponent
    d_ttr_max: float = 0.999     # maximum allowable transverse tensile damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M317 Suite: LadInterfacialDelaminationRate failure, EngThermomagneticGeneratorEnergy, SliderRockerInversionJoint, SensorSpringTransverseSnapRate
# ============================================================================

@dataclass
class FailLadInterfacialDelaminationRate:
    """``/FAIL/LAD_INTERFACIAL_DELAMINATION_RATE`` or ``/FAIL/LADEVEZE_INTERFACIAL_DELAMINATION_RATE`` (M317): Ladevèze rate-dependent mixed-mode interlaminar interfacial debonding and dynamic cohesive delamination failure model."""
    mat_id: int = 0
    title: str = ""
    y0_ifdr: float = 0.0         # dynamic delamination damage initiation threshold energy
    yc_ifdr: float = 0.0         # critical dynamic delamination fracture energy
    c_rate_ifdr: float = 0.0     # interfacial dynamic strain rate sensitivity constant
    p_rate_ifdr: float = 1.0     # strain rate power law exponent
    d_ifdr_max: float = 0.999    # maximum allowable delamination damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M318 Suite: LadTransverseShearInteractionRate failure, EngMagnetorheologicalEnergy, ScotchYokeMechanismJoint, SensorSpringTotalSnapRate
# ============================================================================

@dataclass
class FailLadTransverseShearInteractionRate:
    """``/FAIL/LAD_TRANSVERSE_SHEAR_INTERACTION_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_SHEAR_INTERACTION_RATE`` (M318): Ladevèze rate-dependent combined transverse tension/compression and in-plane shear microcracking coupling failure model."""
    mat_id: int = 0
    title: str = ""
    y0_tsir: float = 0.0          # dynamic coupling damage initiation threshold energy
    yc_tsir: float = 0.0          # critical dynamic coupling fracture energy
    gamma_rate_tsir: float = 0.0 # viscous shear-transverse coupling rate sensitivity parameter
    p_rate_tsir: float = 1.0     # coupling strain rate power law exponent
    d_tsir_max: float = 0.999    # maximum allowable coupled damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference



# ============================================================================
# M319 Suite: LadNonlocalGradient failure, EngElectrorheologicalEnergy, GenevaDriveMechanismJoint, SensorSpringNormalCrackleRate
# ============================================================================

@dataclass
class FailLadNonlocalGradient:
    """``/FAIL/LAD_NONLOCAL_GRADIENT`` or ``/FAIL/LADEVEZE_NONLOCAL_GRADIENT`` (M319): Ladevèze nonlocal gradient-enhanced damage evolution and characteristic length microcrack regularization failure model."""
    mat_id: int = 0
    title: str = ""
    y0_nlg: float = 0.0          # nonlocal damage initiation threshold energy
    yc_nlg: float = 0.0          # critical nonlocal fracture energy
    lc_char: float = 0.0         # characteristic nonlocal internal length parameter
    p_nlg: float = 1.0           # nonlocal gradient power law exponent
    d_nlg_max: float = 0.999     # maximum allowable nonlocal damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M320 Suite: LadAnisotropicPlasticity failure, EngThermoacousticEnergy, DoubleCardanJoint, SensorSpringTransverseCrackleRate
# ============================================================================

@dataclass
class FailLadAnisotropicPlasticity:
    """``/FAIL/LAD_ANISOTROPIC_PLASTICITY`` or ``/FAIL/LADEVEZE_ANISOTROPIC_PLASTICITY`` (M320): Ladevèze coupled anisotropic continuum damage mechanics and non-associated kinematic/isotropic hardening plasticity failure model."""
    mat_id: int = 0
    title: str = ""
    y0_aniso: float = 0.0        # anisotropic damage initiation threshold energy
    yc_aniso: float = 0.0        # critical anisotropic fracture energy
    r0_hard: float = 0.0         # initial plastic yield threshold
    beta_hard: float = 0.0       # isotropic hardening modulus parameter
    d_aniso_max: float = 0.999   # maximum allowable anisotropic damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M321 Suite: LadNonlocalGradientRate failure, EngFerroelectricEnergy, BennettLinkageJoint, SensorSpringTotalCrackleRate
# ============================================================================

@dataclass
class FailLadNonlocalGradientRate:
    """``/FAIL/LAD_NONLOCAL_GRADIENT_RATE`` or ``/FAIL/LADEVEZE_NONLOCAL_GRADIENT_RATE`` (M321): Ladevèze rate-dependent nonlocal gradient-enhanced damage evolution and dynamic viscous regularization failure model."""
    mat_id: int = 0
    title: str = ""
    y0_nlgr: float = 0.0         # dynamic damage initiation threshold energy
    yc_nlgr: float = 0.0         # critical dynamic fracture energy
    lc_char: float = 0.0         # internal characteristic length scale
    tau_nlgr: float = 0.0        # characteristic damage relaxation time constant
    d_nlgr_max: float = 0.999    # maximum allowable dynamic damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M322 Suite: LadFiberKinkingRate failure, EngFlexoelectricEnergy, BricardLinkageJoint, SensorSpringTorsionalCrackleRate
# ============================================================================

@dataclass
class FailLadFiberKinkingRate:
    """``/FAIL/LAD_FIBER_KINKING_RATE`` or ``/FAIL/LADEVEZE_FIBER_KINKING_RATE`` (M322): Ladevèze rate-dependent fiber kinking, longitudinal compressive micro-buckling damage, and dynamic shear localization failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_kink: float = 0.0      # fiber kinking threshold stress
    phi_kink0: float = 0.0       # initial fiber misalignment angle in degrees
    gamma_kink: float = 0.0      # rate sensitivity exponent for fiber kinking
    c_kink: float = 0.0          # dynamic compressive shear resistance modulus
    d_kink_max: float = 0.999    # maximum allowable kinking damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M323 Suite: LadFiberTensionRate failure, EngPyromagneticEnergy, MyardLinkageJoint, SensorSpringBendingCrackleRate
# ============================================================================

@dataclass
class FailLadFiberTensionRate:
    """``/FAIL/LAD_FIBER_TENSION_RATE`` or ``/FAIL/LADEVEZE_FIBER_TENSION_RATE`` (M323): Ladevèze rate-dependent longitudinal tensile fiber breakage, dynamic fiber splitting, and brittle-to-ductile transition damage failure model."""
    mat_id: int = 0
    title: str = ""
    eps_ft0: float = 0.0         # static fiber tensile rupture strain
    eps_ft_rate: float = 0.0     # dynamic strain rate coefficient C_ft
    eps_dot0: float = 1.0        # reference strain rate eps_dot_0
    w_ft_frac: float = 0.0       # critical fracture energy per unit area G_ft
    d_ft_max: float = 0.999      # maximum allowable longitudinal damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M324 Suite: LadFiberCompressionRate failure, EngPiezothermalEnergy, GoldbergLinkageJoint, SensorSpringTotalAngularCrackleRate
# ============================================================================

@dataclass
class FailLadFiberCompressionRate:
    """``/FAIL/LAD_FIBER_COMPRESSION_RATE`` or ``/FAIL/LADEVEZE_FIBER_COMPRESSION_RATE`` (M324): Ladevèze rate-dependent longitudinal compressive fiber crushing, microbuckling, and dynamic compressive failure model."""
    mat_id: int = 0
    title: str = ""
    eps_fc0: float = 0.0         # static fiber compressive rupture strain
    eps_fc_rate: float = 0.0     # dynamic strain rate coefficient C_fc
    eps_dot0: float = 1.0        # reference strain rate eps_dot_0
    w_fc_frac: float = 0.0       # critical compressive fracture energy per unit area G_fc
    d_fc_max: float = 0.999      # maximum allowable longitudinal compressive damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M325 Suite: LadHygrothermal failure, EngThermoflexoelectricEnergy, WaldronLinkageJoint, SensorSpringNormalPopRate
# ============================================================================

@dataclass
class FailLadHygrothermal:
    """``/FAIL/LAD_HYGROTHERMAL`` or ``/FAIL/LADEVEZE_HYGROTHERMAL`` (M325): Ladevèze coupled hygrothermal environmental moisture-temperature degradation and accelerated composite micro-damage evolution model."""
    mat_id: int = 0
    title: str = ""
    c_moist: float = 0.0         # reference moisture concentration C_m
    beta_exp: float = 0.0        # hygrothermal swelling coefficient beta_h
    t_glass: float = 0.0         # glass transition temperature T_g
    d_ht_rate: float = 0.0       # moisture-accelerated damage degradation rate k_ht
    d_ht_max: float = 0.999      # maximum allowable hygrothermal damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M326 Suite: LadCouplePlasticity failure, EngFlexomagneticEnergy, DietmaierLinkageJoint, SensorSpringTransversePopRate
# ============================================================================

@dataclass
class FailLadCouplePlasticity:
    """``/FAIL/LAD_COUPLE_PLASTICITY`` or ``/FAIL/LADEVEZE_COUPLED_PLASTICITY`` (M326): Ladevèze coupled continuum damage and plasticity hardening failure model."""
    mat_id: int = 0
    title: str = ""
    r_p0: float = 0.0            # initial plastic hardening threshold R_0
    k_p: float = 0.0             # plastic hardening modulus K_p
    m_p: float = 1.0             # plastic hardening exponent m_p
    gamma_d: float = 0.0         # kinematic backstress damage coupling parameter gamma_d
    d_cp_max: float = 0.999      # maximum allowable coupled damage-plasticity index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M327 Suite: LadCoupleCreep failure, EngPyroelectricResonanceEnergy, BakerLinkageJoint, SensorSpringTotalPopRate
# ============================================================================

@dataclass
class FailLadCoupleCreep:
    """``/FAIL/LAD_COUPLE_CREEP`` or ``/FAIL/LADEVEZE_COUPLED_CREEP`` (M327): Ladevèze coupled continuum damage and tertiary creep viscous degradation failure model."""
    mat_id: int = 0
    title: str = ""
    a_c: float = 0.0             # creep rate coefficient A_c
    n_c: float = 1.0             # creep stress exponent n_c
    q_c: float = 0.0             # creep activation energy parameter Q_c
    gamma_c: float = 0.0         # creep-damage coupling acceleration parameter gamma_c
    d_cc_max: float = 0.999      # maximum allowable coupled damage-creep index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M328 Suite: LadCoupleViscoplasticity failure, EngThermomagneticResonanceEnergy, WohlhartLinkageJoint, SensorSpringTorsionalPopRate
# ============================================================================

@dataclass
class FailLadCoupleViscoplasticity:
    """``/FAIL/LAD_COUPLE_VISCOPLASTICITY`` or ``/FAIL/LADEVEZE_COUPLED_VISCOPLASTICITY`` (M328): Ladevèze coupled continuum damage and viscoplastic flow failure model."""
    mat_id: int = 0
    title: str = ""
    k_vp: float = 0.0            # viscoplastic viscosity modulus K_vp
    n_vp: float = 1.0            # viscoplastic rate sensitivity exponent n_vp
    r_vp0: float = 0.0           # initial viscoplastic hardening threshold R_vp,0
    gamma_vp: float = 0.0        # viscoplastic-damage coupling acceleration parameter gamma_vp
    d_cvp_max: float = 0.999     # maximum allowable coupled damage-viscoplastic index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M329 Suite: LadMicroDelaminationRate failure, EngFlexothermalResonanceEnergy, AltmannLinkageJoint, SensorSpringBendingPopRate
# ============================================================================

@dataclass
class FailLadMicroDelaminationRate:
    """``/FAIL/LAD_MICRO_DELAMINATION_RATE`` or ``/FAIL/LADEVEZE_MICRO_DELAMINATION_RATE`` (M329): Ladevèze rate-dependent micro-delamination and interlaminar dynamic interfacial shear decohesion failure model."""
    mat_id: int = 0
    title: str = ""
    y_del0: float = 0.0          # initial micro-delamination thermodynamic force threshold Y_del,0
    y_delc: float = 1.0          # critical micro-delamination thermodynamic force Y_del,c
    gamma_del: float = 0.0       # micro-delamination rate sensitivity coefficient gamma_del
    p_del: float = 1.0           # dynamic delamination rate exponent p_del
    d_mdr_max: float = 0.999     # maximum allowable micro-delamination damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M330 Suite: LadCoupleDamageViscoelasticity failure, EngElectromagnetomechanicalResonanceEnergy, SarrusLinkageJoint, SensorSpringTotalAngularPopRate
# ============================================================================

@dataclass
class FailLadCoupleDamageViscoelasticity:
    """``/FAIL/LAD_COUPLE_DAMAGE_VISCOELASTICITY`` or ``/FAIL/LADEVEZE_COUPLED_DAMAGE_VISCOELASTICITY`` (M330): Ladevèze coupled continuum damage and spectral viscoelastic relaxation failure model."""
    mat_id: int = 0
    title: str = ""
    g_inf: float = 0.0           # long-term relaxed shear modulus ratio G_inf / G_0
    tau_ve: float = 1.0          # characteristic viscoelastic relaxation time tau_ve
    beta_ve: float = 1.0         # viscoelastic spectral stretch exponent beta_ve
    gamma_ve: float = 0.0        # viscoelastic-damage coupling acceleration parameter gamma_ve
    d_cdve_max: float = 0.999    # maximum allowable coupled damage-viscoelastic index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M331 Suite: LadDynamicCrushRate failure, EngFlexomagnetoelectricResonanceEnergy, DelassusLinkageJoint, SensorSpringNormalLockRate
# ============================================================================

@dataclass
class FailLadDynamicCrushRate:
    """``/FAIL/LAD_DYNAMIC_CRUSH_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_CRUSH_RATE`` (M331): Ladevèze dynamic progressive transverse crush and multi-axial dynamic compressive crushing failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cr0: float = 0.0       # initial compressive crush threshold stress sigma_cr,0
    sigma_crc: float = 1.0       # critical ultimate compressive crush stress sigma_cr,c
    gamma_cr: float = 0.0        # dynamic crush rate sensitivity factor gamma_cr
    p_cr: float = 1.0            # dynamic crush rate exponent p_cr
    d_dcr_max: float = 0.999     # maximum allowable dynamic crush damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M332 Suite: LadTransverseCrushRate failure, EngFlexothermomagneticResonanceEnergy, SchatzLinkageJoint, SensorSpringTransverseLockRate
# ============================================================================

@dataclass
class FailLadTransverseCrushRate:
    """``/FAIL/LAD_TRANSVERSE_CRUSH_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_CRUSH_RATE`` (M332): Ladevèze dynamic progressive transverse crush and multi-axial dynamic rate-dependent transverse micro-crushing failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tcr0: float = 0.0      # initial transverse crush threshold stress sigma_tcr,0
    sigma_tcrc: float = 1.0      # critical ultimate transverse crush stress sigma_tcr,c
    gamma_tcr: float = 0.0       # transverse crush rate sensitivity factor gamma_tcr
    p_tcr: float = 1.0           # transverse crush rate exponent p_tcr
    d_tcr_max: float = 0.999     # maximum allowable transverse crush damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M333 Suite: LadCoupleDynamicCrush failure, EngFlexothermoelectricResonanceEnergy, FrankeLinkageJoint, SensorSpringTotalLockRate
# ============================================================================

@dataclass
class FailLadCoupleDynamicCrush:
    """``/FAIL/LAD_COUPLE_DYNAMIC_CRUSH`` or ``/FAIL/LADEVEZE_COUPLED_DYNAMIC_CRUSH`` (M333): Ladevèze dynamic progressive coupled crush and rate-dependent multi-axial micro-damage failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cdc0: float = 0.0      # initial coupled crush threshold stress sigma_cdc,0
    sigma_cdcc: float = 1.0      # critical ultimate coupled crush stress sigma_cdc,c
    gamma_cdc: float = 0.0       # coupled crush rate sensitivity factor gamma_cdc
    p_cdc: float = 1.0           # coupled crush rate exponent p_cdc
    d_cdc_max: float = 0.999     # maximum allowable coupled crush damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M334 Suite: LadCoupleCrushRate failure, EngFlexothermoacousticResonanceEnergy, KramesLinkageJoint, SensorSpringTorsionalLockRate
# ============================================================================

@dataclass
class FailLadCoupleCrushRate:
    """``/FAIL/LAD_COUPLE_CRUSH_RATE`` or ``/FAIL/LADEVEZE_COUPLED_CRUSH_RATE`` (M334): Ladevèze rate-dependent coupled crush and dynamic multi-axial compressive crushing failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_ccr0: float = 0.0      # initial coupled crush threshold stress sigma_ccr,0
    sigma_ccrc: float = 1.0      # critical ultimate coupled crush stress sigma_ccr,c
    gamma_ccr: float = 0.0       # coupled crush rate sensitivity factor gamma_ccr
    p_ccr: float = 1.0           # coupled crush rate exponent p_ccr
    d_ccr_max: float = 0.999     # maximum allowable coupled crush damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M335 Suite: LadDynamicDelaminationRate failure, EngFlexoelectromagneticResonanceEnergy, BorelLinkageJoint, SensorSpringBendingLockRate
# ============================================================================

@dataclass
class FailLadDynamicDelaminationRate:
    """``/FAIL/LAD_DYNAMIC_DELAMINATION_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_DELAMINATION_RATE`` (M335): Ladevèze rate-dependent dynamic delamination and progressive interlaminar shear fracture failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_ddr0: float = 0.0      # initial dynamic delamination threshold stress sigma_ddr,0
    sigma_ddrc: float = 1.0      # critical ultimate dynamic delamination stress sigma_ddr,c
    gamma_ddr: float = 0.0       # dynamic delamination rate sensitivity factor gamma_ddr
    p_ddr: float = 1.0           # dynamic delamination rate exponent p_ddr
    d_ddr_max: float = 0.999     # maximum allowable dynamic delamination damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M336 Suite: LadTransverseDelaminationRate failure, EngFlexoelectroacousticResonanceEnergy, HerveLinkageJoint, SensorSpringTotalAngularLockRate
# ============================================================================

@dataclass
class FailLadTransverseDelaminationRate:
    """``/FAIL/LAD_TRANSVERSE_DELAMINATION_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_DELAMINATION_RATE`` (M336): Ladevèze rate-dependent transverse delamination and dynamic interlaminar matrix cracking failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tdr0: float = 0.0      # initial transverse delamination threshold stress sigma_tdr,0
    sigma_tdrc: float = 1.0      # critical ultimate transverse delamination stress sigma_tdr,c
    gamma_tdr: float = 0.0       # transverse delamination rate sensitivity factor gamma_tdr
    p_tdr: float = 1.0           # transverse delamination rate exponent p_tdr
    d_tdr_max: float = 0.999     # maximum allowable transverse delamination damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M337 Suite: LadCoupleDelaminationRate failure, EngFlexomagnetoacousticResonanceEnergy, KongLinkageJoint, SensorSpringNormalDropRate
# ============================================================================

@dataclass
class FailLadCoupleDelaminationRate:
    """``/FAIL/LAD_COUPLE_DELAMINATION_RATE`` or ``/FAIL/LADEVEZE_COUPLED_DELAMINATION_RATE`` (M337): Ladevèze rate-dependent coupled interlaminar delamination and matrix micro-cracking failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cdel0: float = 0.0     # initial coupled delamination threshold stress sigma_cdel,0
    sigma_cdelc: float = 1.0     # critical ultimate coupled delamination stress sigma_cdel,c
    gamma_cdel: float = 0.0      # coupled delamination rate sensitivity factor gamma_cdel
    p_cdel: float = 1.0          # coupled delamination rate exponent p_cdel
    d_cdel_max: float = 0.999    # maximum allowable coupled delamination damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M338 Suite: LadDynamicMicrobucklingRate failure, EngFlexothermoelectromagneticResonanceEnergy, HuntLinkageJoint, SensorSpringTransverseDropRate
# ============================================================================

@dataclass
class FailLadDynamicMicrobucklingRate:
    """``/FAIL/LAD_DYNAMIC_MICROBUCKLING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_MICROBUCKLING_RATE`` (M338): Ladevèze rate-dependent dynamic microbuckling, fiber kinking, and compressive shear localization failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dmbr0: float = 0.0     # initial dynamic microbuckling threshold stress sigma_dmbr,0
    sigma_dmbrc: float = 1.0     # critical ultimate dynamic microbuckling stress sigma_dmbr,c
    gamma_dmbr: float = 0.0      # dynamic microbuckling rate sensitivity factor gamma_dmbr
    p_dmbr: float = 1.0          # dynamic microbuckling rate exponent p_dmbr
    d_dmbr_max: float = 0.999    # maximum allowable dynamic microbuckling damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M339 Suite: LadTransverseMicrobucklingRate failure, EngFlexothermoelectroacousticResonanceEnergy, BakerLineLinkageJoint, SensorSpringTotalDropRate
# ============================================================================

@dataclass
class FailLadTransverseMicrobucklingRate:
    """``/FAIL/LAD_TRANSVERSE_MICROBUCKLING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_MICROBUCKLING_RATE`` (M339): Ladevèze rate-dependent transverse microbuckling, matrix-supported fiber kinking, and compressive transverse shear localization failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tmbr0: float = 0.0     # initial transverse microbuckling threshold stress sigma_tmbr,0
    sigma_tmbrc: float = 1.0     # critical ultimate transverse microbuckling stress sigma_tmbr,c
    gamma_tmbr: float = 0.0      # transverse microbuckling rate sensitivity factor gamma_tmbr
    p_tmbr: float = 1.0          # transverse microbuckling rate exponent p_tmbr
    d_tmbr_max: float = 0.999    # maximum allowable transverse microbuckling damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M340 Suite: LadCoupleMicrobucklingRate failure, EngFlexothermomagnetoacousticResonanceEnergy, BakerPlaneLinkageJoint, SensorSpringTorsionalDropRate
# ============================================================================

@dataclass
class FailLadCoupleMicrobucklingRate:
    """``/FAIL/LAD_COUPLE_MICROBUCKLING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_MICROBUCKLING_RATE`` (M340): Ladevèze rate-dependent coupled microbuckling, fiber-matrix debonding, and progressive compressive kinking failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cmbr0: float = 0.0     # initial coupled microbuckling threshold stress sigma_cmbr,0
    sigma_cmbrc: float = 1.0     # critical ultimate coupled microbuckling stress sigma_cmbr,c
    gamma_cmbr: float = 0.0      # coupled microbuckling rate sensitivity factor gamma_cmbr
    p_cmbr: float = 1.0          # coupled microbuckling rate exponent p_cmbr
    d_cmbr_max: float = 0.999    # maximum allowable coupled microbuckling damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M341 Suite: LadDynamicFiberSplittingRate failure, EngFlexothermoelectromagnetoacousticResonanceEnergy, WohlhartHybridLinkageJoint, SensorSpringBendingDropRate
# ============================================================================

@dataclass
class FailLadDynamicFiberSplittingRate:
    """``/FAIL/LAD_DYNAMIC_FIBER_SPLITTING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_FIBER_SPLITTING_RATE`` (M341): Ladevèze rate-dependent dynamic longitudinal fiber splitting, matrix cleavage, and dynamic tensile fragmentation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dfsr0: float = 0.0     # initial dynamic fiber splitting threshold stress sigma_dfsr,0
    sigma_dfsrc: float = 1.0     # critical dynamic fiber splitting stress sigma_dfsr,c
    gamma_dfsr: float = 0.0      # dynamic fiber splitting rate sensitivity factor gamma_dfsr
    p_dfsr: float = 1.0          # dynamic fiber splitting rate exponent p_dfsr
    d_dfsr_max: float = 0.999    # maximum allowable dynamic fiber splitting damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M342 Suite: LadTransverseFiberSplittingRate failure, EngFlexothermophotonicResonanceEnergy, ChenLinkageJoint, SensorSpringTotalAngularDropRate
# ============================================================================

@dataclass
class FailLadTransverseFiberSplittingRate:
    """``/FAIL/LAD_TRANSVERSE_FIBER_SPLITTING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_FIBER_SPLITTING_RATE`` (M342): Ladevèze rate-dependent transverse fiber splitting, dynamic transverse micro-cleavage, and matrix cleavage failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tfsr0: float = 0.0     # initial transverse fiber splitting threshold stress sigma_tfsr,0
    sigma_tfsrc: float = 1.0     # critical transverse fiber splitting stress sigma_tfsr,c
    gamma_tfsr: float = 0.0      # transverse fiber splitting rate sensitivity factor gamma_tfsr
    p_tfsr: float = 1.0          # transverse fiber splitting rate exponent p_tfsr
    d_tfsr_max: float = 0.999    # maximum allowable transverse fiber splitting damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference

# ============================================================================
# M343 Suite: LadCoupleFiberSplittingRate failure, EngFlexothermoplasmonicResonanceEnergy, BakerSymmetricLinkageJoint, SensorSpringNormalDriftRate
# ============================================================================

@dataclass
class FailLadCoupleFiberSplittingRate:
    """``/FAIL/LAD_COUPLE_FIBER_SPLITTING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_FIBER_SPLITTING_RATE`` (M343): Ladevèze rate-dependent coupled longitudinal/transverse fiber splitting, matrix cleavage, and multi-axial tensile fragmentation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cfsr0: float = 0.0     # initial coupled fiber splitting threshold stress sigma_cfsr,0
    sigma_cfsrc: float = 1.0     # critical coupled fiber splitting stress sigma_cfsr,c
    gamma_cfsr: float = 0.0      # coupled fiber splitting rate sensitivity factor gamma_cfsr
    p_cfsr: float = 1.0          # coupled fiber splitting rate exponent p_cfsr
    d_cfsr_max: float = 0.999    # maximum allowable coupled fiber splitting damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M344 Suite: LadDynamicFiberCrushingRate failure, EngFlexothermoexcitonicResonanceEnergy, AltmannSpatialLinkageJoint, SensorSpringTransverseDriftRate
# ============================================================================

@dataclass
class FailLadDynamicFiberCrushingRate:
    """``/FAIL/LAD_DYNAMIC_FIBER_CRUSHING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_FIBER_CRUSHING_RATE`` (M344): Ladevèze rate-dependent dynamic longitudinal fiber crushing, matrix pulverization, and compressive dynamic crushing failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dfcr0: float = 0.0     # initial dynamic fiber crushing threshold stress sigma_dfcr,0
    sigma_dfcrc: float = 1.0     # critical dynamic fiber crushing stress sigma_dfcr,c
    gamma_dfcr: float = 0.0      # dynamic fiber crushing rate sensitivity factor gamma_dfcr
    p_dfcr: float = 1.0          # dynamic fiber crushing rate exponent p_dfcr
    d_dfcr_max: float = 0.999    # maximum allowable dynamic fiber crushing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M345 Suite: LadTransverseFiberCrushingRate failure, EngFlexothermomagnonicResonanceEnergy, DietmaierSpatialLinkageJoint, SensorSpringTotalDriftRate
# ============================================================================

@dataclass
class FailLadTransverseFiberCrushingRate:
    """``/FAIL/LAD_TRANSVERSE_FIBER_CRUSHING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_FIBER_CRUSHING_RATE`` (M345): Ladevèze rate-dependent transverse fiber crushing, dynamic matrix pulverization, and transverse compressive crushing failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tfcr0: float = 0.0     # initial transverse fiber crushing threshold stress sigma_tfcr,0
    sigma_tfcrc: float = 1.0     # critical transverse fiber crushing stress sigma_tfcr,c
    gamma_tfcr: float = 0.0      # transverse fiber crushing rate sensitivity factor gamma_tfcr
    p_tfcr: float = 1.0          # transverse fiber crushing rate exponent p_tfcr
    d_tfcr_max: float = 0.999    # maximum allowable transverse fiber crushing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M346 Suite: LadCoupleFiberCrushingRate failure, EngFlexothermomagnonpolaritonicResonanceEnergy, WohlhartSpatialLinkageJoint, SensorSpringTorsionalDriftRate
# ============================================================================

@dataclass
class FailLadCoupleFiberCrushingRate:
    """``/FAIL/LAD_COUPLE_FIBER_CRUSHING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_FIBER_CRUSHING_RATE`` (M346): Ladevèze rate-dependent coupled longitudinal/transverse fiber crushing, matrix pulverization, and multi-axial compressive crushing failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cfcr0: float = 0.0     # initial coupled fiber crushing threshold stress sigma_cfcr,0
    sigma_cfcrc: float = 1.0     # critical coupled fiber crushing stress sigma_cfcr,c
    gamma_cfcr: float = 0.0      # coupled fiber crushing rate sensitivity factor gamma_cfcr
    p_cfcr: float = 1.0          # coupled fiber crushing rate exponent p_cfcr
    d_cfcr_max: float = 0.999    # maximum allowable coupled fiber crushing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M347 Suite: LadDynamicInterlaminarShearRate failure, EngFlexothermoplasmonpolaritonicResonanceEnergy, HuntSpatialLinkageJoint, SensorSpringBendingDriftRate
# ============================================================================

@dataclass
class FailLadDynamicInterlaminarShearRate:
    """``/FAIL/LAD_DYNAMIC_INTERLAMINAR_SHEAR_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_SHEAR_RATE`` (M347): Ladevèze rate-dependent dynamic interlaminar shear delamination, interface micro-cracking, and mode-II interlaminar fracture failure model."""
    mat_id: int = 0
    title: str = ""
    tau_disr0: float = 0.0       # initial interlaminar shear threshold stress tau_disr,0
    tau_disrc: float = 1.0       # critical interlaminar shear stress tau_disr,c
    gamma_disr: float = 0.0      # interlaminar shear rate sensitivity factor gamma_disr
    p_disr: float = 1.0          # interlaminar shear rate exponent p_disr
    d_disr_max: float = 0.999    # maximum allowable interlaminar shear damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M348 Suite: LadTransverseInterlaminarShearRate failure, EngFlexothermoexcitonpolaritonicResonanceEnergy, ChenSpatialLinkageJoint, SensorSpringTotalAngularDriftRate
# ============================================================================

@dataclass
class FailLadTransverseInterlaminarShearRate:
    """``/FAIL/LAD_TRANSVERSE_INTERLAMINAR_SHEAR_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_SHEAR_RATE`` (M348): Ladevèze rate-dependent transverse interlaminar shear delamination, transverse interface micro-cracking, and mode-III interlaminar fracture failure model."""
    mat_id: int = 0
    title: str = ""
    tau_tisr0: float = 0.0       # initial transverse interlaminar shear threshold stress tau_tisr,0
    tau_tisrc: float = 1.0       # critical transverse interlaminar shear stress tau_tisr,c
    gamma_tisr: float = 0.0      # transverse interlaminar shear rate sensitivity factor gamma_tisr
    p_tisr: float = 1.0          # transverse interlaminar shear rate exponent p_tisr
    d_tisr_max: float = 0.999    # maximum allowable transverse interlaminar shear damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M349 Suite: LadCoupleInterlaminarShearRate failure, EngFlexothermophononpolaritonicResonanceEnergy, BakerSpatialLinkageJoint, SensorSpringNormalSurgeRate
# ============================================================================

@dataclass
class FailLadCoupleInterlaminarShearRate:
    """``/FAIL/LAD_COUPLE_INTERLAMINAR_SHEAR_RATE`` or ``/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_SHEAR_RATE`` (M349): Ladevèze rate-dependent coupled mode-II/mode-III interlaminar shear delamination, mixed-mode interface micro-cracking, and multi-axial interlaminar shear fracture failure model."""
    mat_id: int = 0
    title: str = ""
    tau_cisr0: float = 0.0       # initial coupled interlaminar shear threshold stress tau_cisr,0
    tau_cisrc: float = 1.0       # critical coupled interlaminar shear stress tau_cisr,c
    gamma_cisr: float = 0.0      # coupled interlaminar shear rate sensitivity factor gamma_cisr
    p_cisr: float = 1.0          # coupled interlaminar shear rate exponent p_cisr
    d_cisr_max: float = 0.999    # maximum allowable coupled interlaminar shear damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M350 Suite: LadDynamicInterlaminarTensionRate failure, EngFlexothermoplasmonphononpolaritonicResonanceEnergy, WaldronSpatialLinkageJoint, SensorSpringTransverseSurgeRate
# ============================================================================

@dataclass
class FailLadDynamicInterlaminarTensionRate:
    """``/FAIL/LAD_DYNAMIC_INTERLAMINAR_TENSION_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_TENSION_RATE`` (M350): Ladevèze rate-dependent dynamic interlaminar normal tension delamination, normal interface micro-cracking, and mode-I interlaminar opening fracture failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_ditr0: float = 0.0     # initial interlaminar normal tension threshold stress sigma_ditr,0
    sigma_ditrc: float = 1.0     # critical interlaminar normal tension stress sigma_ditr,c
    gamma_ditr: float = 0.0      # interlaminar tension rate sensitivity factor gamma_ditr
    p_ditr: float = 1.0          # interlaminar tension rate exponent p_ditr
    d_ditr_max: float = 0.999    # maximum allowable interlaminar normal tension damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M351 Suite: LadTransverseInterlaminarTensionRate failure, EngFlexothermoplasmonexcitonpolaritonicResonanceEnergy, BricardSpatialLinkageJoint, SensorSpringTotalSurgeRate
# ============================================================================

@dataclass
class FailLadTransverseInterlaminarTensionRate:
    """``/FAIL/LAD_TRANSVERSE_INTERLAMINAR_TENSION_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_TENSION_RATE`` (M351): Ladevèze rate-dependent transverse interlaminar normal tension delamination, transverse interface micro-cracking, and mode-I interlaminar opening fracture failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_titr0: float = 0.0     # initial transverse interlaminar normal tension threshold stress sigma_titr,0
    sigma_titrc: float = 1.0     # critical transverse interlaminar normal tension stress sigma_titr,c
    gamma_titr: float = 0.0      # transverse interlaminar tension rate sensitivity factor gamma_titr
    p_titr: float = 1.0          # transverse interlaminar tension rate exponent p_titr
    d_titr_max: float = 0.999    # maximum allowable transverse interlaminar normal tension damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M352 Suite: LadCoupleInterlaminarTensionRate failure, EngFlexothermoplasmonmagnonpolaritonicResonanceEnergy, BennettSpatialLinkageJoint, SensorSpringTorsionalSurgeRate
# ============================================================================

@dataclass
class FailLadCoupleInterlaminarTensionRate:
    """``/FAIL/LAD_COUPLE_INTERLAMINAR_TENSION_RATE`` or ``/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_TENSION_RATE`` (M352): Ladevèze rate-dependent coupled mode-I/mode-II/mode-III interlaminar normal tension and mixed-mode delamination, interface micro-cracking, and multi-axial interlaminar opening fracture failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_citr0: float = 0.0     # initial coupled interlaminar normal tension threshold stress sigma_citr,0
    sigma_citrc: float = 1.0     # critical coupled interlaminar normal tension stress sigma_citr,c
    gamma_citr: float = 0.0      # coupled interlaminar tension rate sensitivity factor gamma_citr
    p_citr: float = 1.0          # coupled interlaminar tension rate exponent p_citr
    d_citr_max: float = 0.999    # maximum allowable coupled interlaminar normal tension damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M353 Suite: LadDynamicMatrixMicrocrackingRate failure, EngFlexothermoexcitonphononpolaritonicResonanceEnergy, MyardSpatialLinkageJoint, SensorSpringBendingSurgeRate
# ============================================================================

@dataclass
class FailLadDynamicMatrixMicrocrackingRate:
    """``/FAIL/LAD_DYNAMIC_MATRIX_MICROCRACKING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_MATRIX_MICROCRACKING_RATE`` (M353): Ladevèze rate-dependent dynamic transverse matrix micro-cracking, diffuse damage accumulation, and transverse ply degradation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dmmr0: float = 0.0     # initial transverse matrix microcracking threshold stress sigma_dmmr,0
    sigma_dmmrc: float = 1.0     # critical transverse matrix microcracking stress sigma_dmmr,c
    gamma_dmmr: float = 0.0      # matrix microcracking rate sensitivity factor gamma_dmmr
    p_dmmr: float = 1.0          # matrix microcracking rate exponent p_dmmr
    d_dmmr_max: float = 0.999    # maximum allowable transverse matrix damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M354 Suite: LadTransverseMatrixMicrocrackingRate failure, EngFlexothermoexcitonmagnonpolaritonicResonanceEnergy, GoldbergSpatialLinkageJoint, SensorSpringTotalAngularSurgeRate
# ============================================================================

@dataclass
class FailLadTransverseMatrixMicrocrackingRate:
    """``/FAIL/LAD_TRANSVERSE_MATRIX_MICROCRACKING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_MATRIX_MICROCRACKING_RATE`` (M354): Ladevèze rate-dependent transverse matrix micro-cracking, dynamic transverse micro-fissuring, and progressive transverse matrix degradation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tmmr0: float = 0.0     # initial transverse matrix microcracking threshold stress sigma_tmmr,0
    sigma_tmmrc: float = 1.0     # critical transverse matrix microcracking stress sigma_tmmr,c
    gamma_tmmr: float = 0.0      # transverse matrix microcracking rate sensitivity factor gamma_tmmr
    p_tmmr: float = 1.0          # transverse matrix microcracking rate exponent p_tmmr
    d_tmmr_max: float = 0.999    # maximum allowable transverse matrix damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M355 Suite: LadCoupleMatrixMicrocrackingRate failure, EngFlexothermophononmagnonpolaritonicResonanceEnergy, SarrusSpatialLinkageJoint, SensorSpringNormalPopRate
# ============================================================================

@dataclass
class FailLadCoupleMatrixMicrocrackingRate:
    """``/FAIL/LAD_COUPLE_MATRIX_MICROCRACKING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_MATRIX_MICROCRACKING_RATE`` (M355): Ladevèze rate-dependent coupled transverse matrix micro-cracking, multi-axial diffuse damage accumulation, and combined matrix-ply degradation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cmmr0: float = 0.0     # initial transverse matrix microcracking threshold stress sigma_cmmr,0
    sigma_cmmrc: float = 1.0     # critical transverse matrix microcracking stress sigma_cmmr,c
    gamma_cmmr: float = 0.0      # matrix microcracking rate sensitivity factor gamma_cmmr
    p_cmmr: float = 1.0          # matrix microcracking rate exponent p_cmmr
    d_cmmr_max: float = 0.999    # maximum allowable transverse matrix damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M356 Suite: LadDynamicFiberCompressionKinkingRate failure, EngFlexothermoplasmonexcitonphononpolaritonicResonanceEnergy, DelassusSpatialLinkageJoint, SensorSpringTransversePopRate
# ============================================================================

@dataclass
class FailLadDynamicFiberCompressionKinkingRate:
    """``/FAIL/LAD_DYNAMIC_FIBER_COMPRESSION_KINKING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_FIBER_COMPRESSION_KINKING_RATE`` (M356): Ladevèze rate-dependent dynamic longitudinal fiber compressive kinking, plastic micro-buckling band, and compressive fiber failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dfckr0: float = 0.0    # initial fiber compression kinking threshold stress sigma_dfckr,0
    sigma_dfckrc: float = 1.0    # critical fiber compression kinking stress sigma_dfckr,c
    gamma_dfckr: float = 0.0     # fiber kinking rate sensitivity factor gamma_dfckr
    p_dfckr: float = 1.0         # fiber kinking rate exponent p_dfckr
    d_dfckr_max: float = 0.999   # maximum allowable fiber compressive damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M357 Suite: LadTransverseFiberCompressionKinkingRate failure, EngFlexothermoplasmonexcitonmagnonpolaritonicResonanceEnergy, WohlhartSpatialLinkageJoint, SensorSpringTotalPopRate
# ============================================================================

@dataclass
class FailLadTransverseFiberCompressionKinkingRate:
    """``/FAIL/LAD_TRANSVERSE_FIBER_COMPRESSION_KINKING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_FIBER_COMPRESSION_KINKING_RATE`` (M357): Ladevèze rate-dependent transverse fiber compressive kinking, out-of-plane plastic micro-buckling band, and transverse compressive fiber failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tfckr0: float = 0.0    # initial transverse fiber kinking threshold stress sigma_tfckr,0
    sigma_tfckrc: float = 1.0    # critical transverse fiber kinking stress sigma_tfckr,c
    gamma_tfckr: float = 0.0     # transverse fiber kinking rate sensitivity factor gamma_tfckr
    p_tfckr: float = 1.0         # transverse fiber kinking rate exponent p_tfckr
    d_tfckr_max: float = 0.999   # maximum allowable transverse fiber damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M358 Suite: LadCoupleFiberCompressionKinkingRate failure, EngFlexothermoplasmonphononmagnonpolaritonicResonanceEnergy, AltmannSpatialLinkageJoint, SensorSpringTorsionalPopRate
# ============================================================================

@dataclass
class FailLadCoupleFiberCompressionKinkingRate:
    """``/FAIL/LAD_COUPLE_FIBER_COMPRESSION_KINKING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_FIBER_COMPRESSION_KINKING_RATE`` (M358): Ladevèze rate-dependent coupled multi-axial fiber compressive kinking, 3D micro-buckling band propagation, and progressive fiber crushing failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cfckr0: float = 0.0    # initial coupled fiber kinking threshold stress sigma_cfckr,0
    sigma_cfckrc: float = 1.0    # critical coupled fiber kinking stress sigma_cfckr,c
    gamma_cfckr: float = 0.0     # coupled fiber kinking rate sensitivity factor gamma_cfckr
    p_cfckr: float = 1.0         # coupled fiber kinking rate exponent p_cfckr
    d_cfckr_max: float = 0.999   # maximum allowable coupled fiber damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M359 Suite: LadDynamicDelaminationMicrodebondingRate failure, EngFlexothermoexcitonphononmagnonpolaritonicResonanceEnergy, BakerSpatialLinkageJoint, SensorSpringBendingPopRate
# ============================================================================

@dataclass
class FailLadDynamicDelaminationMicrodebondingRate:
    """``/FAIL/LAD_DYNAMIC_DELAMINATION_MICRODEBONDING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_DELAMINATION_MICRODEBONDING_RATE`` (M359): Ladevèze rate-dependent dynamic interlaminar micro-debonding, high-rate cohesive interface separation, and dynamic interlaminar delamination failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_ddmr0: float = 0.0     # initial interlaminar micro-debonding threshold stress sigma_ddmr,0
    sigma_ddmrc: float = 1.0     # critical dynamic micro-debonding separation stress sigma_ddmr,c
    gamma_ddmr: float = 0.0      # dynamic micro-debonding rate sensitivity factor gamma_ddmr
    p_ddmr: float = 1.0          # dynamic micro-debonding rate exponent p_ddmr
    d_ddmr_max: float = 0.999    # maximum allowable interlaminar damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M360 Suite: LadTransverseDelaminationMicrodebondingRate failure, EngFlexothermoplasmonexcitonphononmagnonpolaritonicResonanceEnergy, DietmaierSpatialLinkageJoint, SensorSpringTotalAngularPopRate
# ============================================================================

@dataclass
class FailLadTransverseDelaminationMicrodebondingRate:
    """``/FAIL/LAD_TRANSVERSE_DELAMINATION_MICRODEBONDING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_DELAMINATION_MICRODEBONDING_RATE`` (M360): Ladevèze rate-dependent transverse interlaminar micro-debonding, out-of-plane shear cohesive interface separation, and transverse delamination fracture failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tdmr0: float = 0.0     # initial transverse micro-debonding threshold stress sigma_tdmr,0
    sigma_tdmrc: float = 1.0     # critical transverse micro-debonding separation stress sigma_tdmr,c
    gamma_tdmr: float = 0.0      # transverse micro-debonding rate sensitivity factor gamma_tdmr
    p_tdmr: float = 1.0          # transverse micro-debonding rate exponent p_tdmr
    d_tdmr_max: float = 0.999    # maximum allowable transverse interlaminar damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M361 Suite: LadCoupleDelaminationMicrodebondingRate failure, EngFlexomagnetoplasmonicResonanceEnergy, WaldronSpatialLinkageJoint, SensorSpringNormalCrackleRate
# ============================================================================

@dataclass
class FailLadCoupleDelaminationMicrodebondingRate:
    """``/FAIL/LAD_COUPLE_DELAMINATION_MICRODEBONDING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_DELAMINATION_MICRODEBONDING_RATE`` (M361): Ladevèze rate-dependent coupled multi-mode interlaminar micro-debonding, mixed-mode peel-shear cohesive interface separation, and progressive delamination failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cdmr0: float = 0.0     # initial coupled micro-debonding threshold stress sigma_cdmr,0
    sigma_cdmrc: float = 1.0     # critical coupled micro-debonding separation stress sigma_cdmr,c
    gamma_cdmr: float = 0.0      # coupled micro-debonding rate sensitivity factor gamma_cdmr
    p_cdmr: float = 1.0          # coupled micro-debonding rate exponent p_cdmr
    d_cdmr_max: float = 0.999    # maximum allowable coupled interlaminar damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M362 Suite: LadDynamicMatrixShearDegradationRate failure, EngFlexomagnetophononicResonanceEnergy, HuntSpatialLinkageJoint, SensorSpringTransverseCrackleRate
# ============================================================================

@dataclass
class FailLadDynamicMatrixShearDegradationRate:
    """``/FAIL/LAD_DYNAMIC_MATRIX_SHEAR_DEGRADATION_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_MATRIX_SHEAR_DEGRADATION_RATE`` (M362): Ladevèze rate-dependent dynamic in-plane matrix shear damage evolution, nonlinear inelastic shear strain accumulation, and shear degradation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dmsdr0: float = 0.0    # initial matrix shear degradation threshold stress sigma_dmsdr,0
    sigma_dmsdrc: float = 1.0    # critical dynamic shear degradation stress sigma_dmsdr,c
    gamma_dmsdr: float = 0.0     # dynamic matrix shear degradation rate sensitivity factor gamma_dmsdr
    p_dmsdr: float = 1.0         # dynamic matrix shear degradation rate exponent p_dmsdr
    d_dmsdr_max: float = 0.999   # maximum allowable in-plane shear damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M363 Suite: LadTransverseMatrixShearDegradationRate failure, EngFlexomagnetoexcitonicResonanceEnergy, ChenSpatialLinkageJoint, SensorSpringTotalCrackleRate
# ============================================================================

@dataclass
class FailLadTransverseMatrixShearDegradationRate:
    """``/FAIL/LAD_TRANSVERSE_MATRIX_SHEAR_DEGRADATION_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_MATRIX_SHEAR_DEGRADATION_RATE`` (M363): Ladevèze rate-dependent transverse out-of-plane matrix shear degradation, transverse interlaminar shear damage evolution, and transverse shear degradation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tmsdr0: float = 0.0    # initial transverse shear degradation threshold stress sigma_tmsdr,0
    sigma_tmsdrc: float = 1.0    # critical transverse shear degradation stress sigma_tmsdr,c
    gamma_tmsdr: float = 0.0     # transverse matrix shear degradation rate sensitivity factor gamma_tmsdr
    p_tmsdr: float = 1.0         # transverse matrix shear degradation rate exponent p_tmsdr
    d_tmsdr_max: float = 0.999   # maximum allowable transverse shear damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M364 Suite: LadCoupleMatrixShearDegradationRate failure, EngFlexomagnetopolaritonicResonanceEnergy, WunderlichSpatialLinkageJoint, SensorSpringTorsionalCrackleRate
# ============================================================================

@dataclass
class FailLadCoupleMatrixShearDegradationRate:
    """``/FAIL/LAD_COUPLE_MATRIX_SHEAR_DEGRADATION_RATE`` or ``/FAIL/LADEVEZE_COUPLED_MATRIX_SHEAR_DEGRADATION_RATE`` (M364): Ladevèze rate-dependent coupled multi-axial matrix shear degradation, coupled in-plane/transverse shear damage accumulation, and progressive shear failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cmsdr0: float = 0.0    # initial coupled shear degradation threshold stress sigma_cmsdr,0
    sigma_cmsdrc: float = 1.0    # critical coupled shear degradation stress sigma_cmsdr,c
    gamma_cmsdr: float = 0.0     # coupled matrix shear degradation rate sensitivity factor gamma_cmsdr
    p_cmsdr: float = 1.0         # coupled matrix shear degradation rate exponent p_cmsdr
    d_cmsdr_max: float = 0.999   # maximum allowable coupled shear damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M365 Suite: LadDynamicFiberTensionRuptureRate failure, EngFlexomagnetoplasmonicphononResonanceEnergy, KonnokSpatialLinkageJoint, SensorSpringBendingCrackleRate
# ============================================================================

@dataclass
class FailLadDynamicFiberTensionRuptureRate:
    """``/FAIL/LAD_DYNAMIC_FIBER_TENSION_RUPTURE_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_FIBER_TENSION_RUPTURE_RATE`` (M365): Ladevèze rate-dependent dynamic longitudinal fiber tensile damage accumulation, high-rate fiber bundle fracture, and tensile rupture failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dftrr0: float = 0.0    # initial fiber tensile rupture threshold stress sigma_dftrr,0
    sigma_dftrrc: float = 1.0    # critical dynamic fiber tensile rupture stress sigma_dftrr,c
    gamma_dftrr: float = 0.0     # dynamic fiber tensile rupture rate sensitivity factor gamma_dftrr
    p_dftrr: float = 1.0         # dynamic fiber tensile rupture rate exponent p_dftrr
    d_dftrr_max: float = 0.999   # maximum allowable fiber tensile damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M366 Suite: LadTransverseFiberTensionRuptureRate failure, EngFlexomagnetoplasmonicexcitonResonanceEnergy, PfurnerSpatialLinkageJoint, SensorSpringTotalAngularCrackleRate
# ============================================================================

@dataclass
class FailLadTransverseFiberTensionRuptureRate:
    """``/FAIL/LAD_TRANSVERSE_FIBER_TENSION_RUPTURE_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_FIBER_TENSION_RUPTURE_RATE`` (M366): Ladevèze rate-dependent transverse out-of-plane fiber tensile damage accumulation, transverse fiber bundle debonding, and transverse tensile rupture failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tftrr0: float = 0.0    # initial transverse fiber tensile rupture threshold stress sigma_tftrr,0
    sigma_tftrrc: float = 1.0    # critical transverse fiber tensile rupture stress sigma_tftrr,c
    gamma_tftrr: float = 0.0     # transverse fiber tensile rupture rate sensitivity factor gamma_tftrr
    p_tftrr: float = 1.0         # transverse fiber tensile rupture rate exponent p_tftrr
    d_tftrr_max: float = 0.999   # maximum allowable transverse fiber tensile damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M367 Suite: LadCoupleFiberTensionRuptureRate failure, EngFlexomagnetoplasmonicmagnonResonanceEnergy, PhillipsSpatialLinkageJoint, SensorSpringNormalSnapRate
# ============================================================================

@dataclass
class FailLadCoupleFiberTensionRuptureRate:
    """``/FAIL/LAD_COUPLE_FIBER_TENSION_RUPTURE_RATE`` or ``/FAIL/LADEVEZE_COUPLED_FIBER_TENSION_RUPTURE_RATE`` (M367): Ladevèze rate-dependent coupled multi-axial fiber tensile damage accumulation, progressive fiber bundle debonding, and coupled tensile rupture failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cftrr0: float = 0.0    # initial coupled fiber tensile rupture threshold stress sigma_cftrr,0
    sigma_cftrrc: float = 1.0    # critical dynamic coupled fiber tensile rupture stress sigma_cftrr,c
    gamma_cftrr: float = 0.0     # coupled fiber tensile rupture rate sensitivity factor gamma_cftrr
    p_cftrr: float = 1.0         # coupled fiber tensile rupture rate exponent p_cftrr
    d_cftrr_max: float = 0.999   # maximum allowable coupled fiber tensile damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M368 Suite: LadDynamicFiberCompressionCrushingRate failure, EngFlexomagnetoplasmonicpolaritonicResonanceEnergy, StevensSpatialLinkageJoint, SensorSpringTransverseSnapRate
# ============================================================================

@dataclass
class FailLadDynamicFiberCompressionCrushingRate:
    """``/FAIL/LAD_DYNAMIC_FIBER_COMPRESSION_CRUSHING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_FIBER_COMPRESSION_CRUSHING_RATE`` (M368): Ladevèze rate-dependent dynamic longitudinal fiber compressive damage accumulation, high-rate microbuckling band propagation, and fiber crushing failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dfccr0: float = 0.0    # initial dynamic fiber crushing threshold stress sigma_dfccr,0
    sigma_dfccrc: float = 1.0    # critical dynamic fiber crushing stress sigma_dfccr,c
    gamma_dfccr: float = 0.0     # dynamic fiber crushing rate sensitivity factor gamma_dfccr
    p_dfccr: float = 1.0         # dynamic fiber crushing rate exponent p_dfccr
    d_dfccr_max: float = 0.999   # maximum allowable fiber compressive damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M369 Suite: LadTransverseFiberCompressionCrushingRate failure, EngFlexomagnetophononicexcitonicResonanceEnergy, BakerHybridSpatialLinkageJoint, SensorSpringTotalSnapRate
# ============================================================================

@dataclass
class FailLadTransverseFiberCompressionCrushingRate:
    """``/FAIL/LAD_TRANSVERSE_FIBER_COMPRESSION_CRUSHING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_FIBER_COMPRESSION_CRUSHING_RATE`` (M369): Ladevèze rate-dependent transverse out-of-plane fiber compressive damage accumulation, transverse microbuckling band propagation, and transverse fiber crushing failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tfccr0: float = 0.0    # initial transverse dynamic fiber crushing threshold stress sigma_tfccr,0
    sigma_tfccrc: float = 1.0    # critical dynamic transverse fiber crushing stress sigma_tfccr,c
    gamma_tfccr: float = 0.0     # transverse dynamic fiber crushing rate sensitivity factor gamma_tfccr
    p_tfccr: float = 1.0         # transverse dynamic fiber crushing rate exponent p_tfccr
    d_tfccr_max: float = 0.999   # maximum allowable transverse fiber compressive damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M370 Suite: LadCoupleFiberCompressionCrushingRate failure, EngFlexomagnetophononicmagnonicResonanceEnergy, WaldronHybridSpatialLinkageJoint, SensorSpringTorsionalSnapRate
# ============================================================================

@dataclass
class FailLadCoupleFiberCompressionCrushingRate:
    """``/FAIL/LAD_COUPLE_FIBER_COMPRESSION_CRUSHING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_FIBER_COMPRESSION_CRUSHING_RATE`` (M370): Ladevèze rate-dependent coupled multi-axial fiber compressive damage accumulation, dynamic microbuckling band propagation, and coupled fiber crushing failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cfccr0: float = 0.0    # initial coupled dynamic fiber crushing threshold stress sigma_cfccr,0
    sigma_cfccrc: float = 1.0    # critical dynamic coupled fiber crushing stress sigma_cfccr,c
    gamma_cfccr: float = 0.0     # coupled dynamic fiber crushing rate sensitivity factor gamma_cfccr
    p_cfccr: float = 1.0         # coupled dynamic fiber crushing rate exponent p_cfccr
    d_cfccr_max: float = 0.999   # maximum allowable coupled fiber compressive damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M371 Suite: LadDynamicInterlaminarShearDelaminationRate failure, EngFlexomagnetophononicpolaritonicResonanceEnergy, ChenHybridSpatialLinkageJoint, SensorSpringBendingSnapRate
# ============================================================================

@dataclass
class FailLadDynamicInterlaminarShearDelaminationRate:
    """``/FAIL/LAD_DYNAMIC_INTERLAMINAR_SHEAR_DELAMINATION_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_SHEAR_DELAMINATION_RATE`` (M371): Ladevèze rate-dependent dynamic interlaminar shear stress debonding, mode-II crack propagation, and delamination failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_disdr0: float = 0.0    # initial dynamic interlaminar shear debonding threshold stress sigma_disdr,0
    sigma_disdrc: float = 1.0    # critical dynamic interlaminar shear fracture stress sigma_disdr,c
    gamma_disdr: float = 0.0     # dynamic interlaminar shear rate sensitivity factor gamma_disdr
    p_disdr: float = 1.0         # dynamic interlaminar shear rate exponent p_disdr
    d_disdr_max: float = 0.999   # maximum allowable interlaminar shear damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M372 Suite: LadTransverseInterlaminarShearDelaminationRate failure, EngFlexomagnetoexcitonicmagnonicResonanceEnergy, WohlhartHybridSpatialLinkageJoint, SensorSpringTotalAngularSnapRate
# ============================================================================

@dataclass
class FailLadTransverseInterlaminarShearDelaminationRate:
    """``/FAIL/LAD_TRANSVERSE_INTERLAMINAR_SHEAR_DELAMINATION_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_SHEAR_DELAMINATION_RATE`` (M372): Ladevèze rate-dependent transverse interlaminar shear stress debonding, mode-III/out-of-plane tearing crack propagation, and delamination failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tisdr0: float = 0.0    # initial transverse interlaminar shear debonding threshold stress sigma_tisdr,0
    sigma_tisdrc: float = 1.0    # critical transverse interlaminar shear fracture stress sigma_tisdr,c
    gamma_tisdr: float = 0.0     # transverse interlaminar shear rate sensitivity factor gamma_tisdr
    p_tisdr: float = 1.0         # transverse interlaminar shear rate exponent p_tisdr
    d_tisdr_max: float = 0.999   # maximum allowable transverse interlaminar shear damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M373 Suite: LadCoupleInterlaminarShearDelaminationRate failure, EngFlexomagnetoexcitonicpolaritonicResonanceEnergy, MaverickHybridSpatialLinkageJoint, SensorSpringNormalPopRate
# ============================================================================

@dataclass
class FailLadCoupleInterlaminarShearDelaminationRate:
    """``/FAIL/LAD_COUPLE_INTERLAMINAR_SHEAR_DELAMINATION_RATE`` or ``/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_SHEAR_DELAMINATION_RATE`` (M373): Ladevèze rate-dependent coupled mixed-mode interlaminar shear debonding, mixed-mode crack propagation, and delamination failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cisdr0: float = 0.0    # initial coupled interlaminar shear debonding threshold stress sigma_cisdr,0
    sigma_cisdrc: float = 1.0    # critical coupled interlaminar shear fracture stress sigma_cisdr,c
    gamma_cisdr: float = 0.0     # coupled interlaminar shear rate sensitivity factor gamma_cisdr
    p_cisdr: float = 1.0         # coupled interlaminar shear rate exponent p_cisdr
    d_cisdr_max: float = 0.999   # maximum allowable coupled interlaminar shear damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M374 Suite: LadDynamicInterlaminarNormalPeelingRate failure, EngFlexomagnetoplasmonicexcitonicmagnonicResonanceEnergy, KrauseHybridSpatialLinkageJoint, SensorSpringTransversePopRate
# ============================================================================

@dataclass
class FailLadDynamicInterlaminarNormalPeelingRate:
    """``/FAIL/LAD_DYNAMIC_INTERLAMINAR_NORMAL_PEELING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_NORMAL_PEELING_RATE`` (M374): Ladevèze rate-dependent dynamic interlaminar normal peeling stress debonding, mode-I opening crack propagation, and delamination failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dinpr0: float = 0.0    # initial dynamic interlaminar normal peeling threshold stress sigma_dinpr,0
    sigma_dinprc: float = 1.0    # critical dynamic interlaminar normal peeling fracture stress sigma_dinpr,c
    gamma_dinpr: float = 0.0     # dynamic interlaminar normal peeling rate sensitivity factor gamma_dinpr
    p_dinpr: float = 1.0         # dynamic interlaminar normal peeling rate exponent p_dinpr
    d_dinpr_max: float = 0.999   # maximum allowable dynamic interlaminar normal peeling damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M375 Suite: LadTransverseInterlaminarNormalPeelingRate failure, EngFlexomagnetoplasmonicexcitonicpolaritonicResonanceEnergy, SturgessHybridSpatialLinkageJoint, SensorSpringTotalPopRate
# ============================================================================

@dataclass
class FailLadTransverseInterlaminarNormalPeelingRate:
    """``/FAIL/LAD_TRANSVERSE_INTERLAMINAR_NORMAL_PEELING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_NORMAL_PEELING_RATE`` (M375): Ladevèze rate-dependent transverse interlaminar normal peeling stress debonding, transverse crack propagation, and delamination failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tinpr0: float = 0.0    # initial transverse interlaminar normal peeling threshold stress sigma_tinpr,0
    sigma_tinprc: float = 1.0    # critical transverse interlaminar normal peeling fracture stress sigma_tinpr,c
    gamma_tinpr: float = 0.0     # transverse interlaminar normal peeling rate sensitivity factor gamma_tinpr
    p_tinpr: float = 1.0         # transverse interlaminar normal peeling rate exponent p_tinpr
    d_tinpr_max: float = 0.999   # maximum allowable transverse interlaminar normal peeling damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M376 Suite: LadCoupleInterlaminarNormalPeelingRate failure, EngFlexomagnetoplasmonicmagnonicpolaritonicResonanceEnergy, BevanHybridSpatialLinkageJoint, SensorSpringTorsionalPopRate
# ============================================================================

@dataclass
class FailLadCoupleInterlaminarNormalPeelingRate:
    """``/FAIL/LAD_COUPLE_INTERLAMINAR_NORMAL_PEELING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_NORMAL_PEELING_RATE`` (M376): Ladevèze rate-dependent coupled multi-axial interlaminar normal peeling damage accumulation, progressive mixed-mode debonding, and coupled delamination failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cinpr0: float = 0.0    # initial coupled interlaminar normal peeling threshold stress sigma_cinpr,0
    sigma_cinprc: float = 1.0    # critical coupled interlaminar normal peeling fracture stress sigma_cinpr,c
    gamma_cinpr: float = 0.0     # coupled interlaminar normal peeling rate sensitivity factor gamma_cinpr
    p_cinpr: float = 1.0         # coupled interlaminar normal peeling rate exponent p_cinpr
    d_cinpr_max: float = 0.999   # maximum allowable coupled interlaminar normal peeling damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M377 Suite: LadDynamicFiberMatrixDebondingRate failure, EngFlexomagnetophononicexcitonicmagnonicResonanceEnergy, HeinrichsHybridSpatialLinkageJoint, SensorSpringBendingPopRate
# ============================================================================

@dataclass
class FailLadDynamicFiberMatrixDebondingRate:
    """``/FAIL/LAD_DYNAMIC_FIBER_MATRIX_DEBONDING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_FIBER_MATRIX_DEBONDING_RATE`` (M377): Ladevèze rate-dependent dynamic fiber-matrix interfacial shear debonding, micro-crack coalescence, and interfacial failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dfmdr0: float = 0.0    # initial dynamic fiber-matrix debonding threshold stress sigma_dfmdr,0
    sigma_dfmdrc: float = 1.0    # critical dynamic fiber-matrix debonding fracture stress sigma_dfmdr,c
    gamma_dfmdr: float = 0.0     # dynamic fiber-matrix debonding rate sensitivity factor gamma_dfmdr
    p_dfmdr: float = 1.0         # dynamic fiber-matrix debonding rate exponent p_dfmdr
    d_dfmdr_max: float = 0.999   # maximum allowable dynamic fiber-matrix debonding damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M378 Suite: LadTransverseFiberMatrixDebondingRate failure, EngFlexomagnetophononicexcitonicpolaritonicResonanceEnergy, AltmannHybridSpatialLinkageJoint, SensorSpringTotalAngularPopRate
# ============================================================================

@dataclass
class FailLadTransverseFiberMatrixDebondingRate:
    """``/FAIL/LAD_TRANSVERSE_FIBER_MATRIX_DEBONDING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_FIBER_MATRIX_DEBONDING_RATE`` (M378): Ladevèze rate-dependent transverse fiber-matrix interfacial shear debonding, progressive matrix micro-cracking, and interfacial failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tfmdr0: float = 0.0    # initial transverse fiber-matrix debonding threshold stress sigma_tfmdr,0
    sigma_tfmdrc: float = 1.0    # critical transverse fiber-matrix debonding fracture stress sigma_tfmdr,c
    gamma_tfmdr: float = 0.0     # transverse fiber-matrix debonding rate sensitivity factor gamma_tfmdr
    p_tfmdr: float = 1.0         # transverse fiber-matrix debonding rate exponent p_tfmdr
    d_tfmdr_max: float = 0.999   # maximum allowable transverse fiber-matrix debonding damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M379 Suite: LadCoupleFiberMatrixDebondingRate failure, EngFlexomagnetophononicmagnonicpolaritonicResonanceEnergy, KirkpatrickHybridSpatialLinkageJoint, SensorSpringNormalLockRate
# ============================================================================

@dataclass
class FailLadCoupleFiberMatrixDebondingRate:
    """``/FAIL/LAD_COUPLE_FIBER_MATRIX_DEBONDING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_FIBER_MATRIX_DEBONDING_RATE`` (M379): Ladevèze rate-dependent coupled multi-axial fiber-matrix interfacial shear debonding, damage accumulation, and interfacial failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cfmdr0: float = 0.0    # initial coupled fiber-matrix debonding threshold stress sigma_cfmdr,0
    sigma_cfmdrc: float = 1.0    # critical coupled fiber-matrix debonding fracture stress sigma_cfmdr,c
    gamma_cfmdr: float = 0.0     # coupled fiber-matrix debonding rate sensitivity factor gamma_cfmdr
    p_cfmdr: float = 1.0         # coupled fiber-matrix debonding rate exponent p_cfmdr
    d_cfmdr_max: float = 0.999   # maximum allowable coupled fiber-matrix debonding damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M380 Suite: LadDynamicPlyMicroCrackingRate failure, EngFlexomagnetoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy, AlexanderHybridSpatialLinkageJoint, SensorSpringTransverseLockRate
# ============================================================================

@dataclass
class FailLadDynamicPlyMicroCrackingRate:
    """``/FAIL/LAD_DYNAMIC_PLY_MICRO_CRACKING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_PLY_MICRO_CRACKING_RATE`` (M380): Ladevèze rate-dependent dynamic ply transverse and shear micro-cracking damage accumulation, diffuse cracking growth, and lamina degradation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dpmcr0: float = 0.0    # initial ply micro-cracking threshold stress sigma_dpmcr,0
    sigma_dpmcrc: float = 1.0    # critical ply micro-cracking saturation stress sigma_dpmcr,c
    gamma_dpmcr: float = 0.0     # ply micro-cracking rate sensitivity factor gamma_dpmcr
    p_dpmcr: float = 1.0         # ply micro-cracking rate exponent p_dpmcr
    d_dpmcr_max: float = 0.999   # maximum allowable ply micro-cracking damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M381 Suite: LadTransversePlyMicroCrackingRate failure, EngFlexomagnetophononicplasmonicmagnonicResonanceEnergy, ChungHybridSpatialLinkageJoint, SensorSpringTotalLockRate
# ============================================================================

@dataclass
class FailLadTransversePlyMicroCrackingRate:
    """``/FAIL/LAD_TRANSVERSE_PLY_MICRO_CRACKING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_PLY_MICRO_CRACKING_RATE`` (M381): Ladevèze rate-dependent transverse ply micro-cracking damage accumulation, transverse matrix micro-flaw propagation, and stiffness reduction failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tpmcr0: float = 0.0    # initial transverse ply micro-cracking threshold stress sigma_tpmcr,0
    sigma_tpmcrc: float = 1.0    # critical transverse ply micro-cracking saturation stress sigma_tpmcr,c
    gamma_tpmcr: float = 0.0     # transverse ply micro-cracking rate sensitivity factor gamma_tpmcr
    p_tpmcr: float = 1.0         # transverse ply micro-cracking rate exponent p_tpmcr
    d_tpmcr_max: float = 0.999   # maximum allowable transverse ply micro-cracking damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M382 Suite: LadCouplePlyMicroCrackingRate failure, EngFlexomagnetophononicplasmonicpolaritonicResonanceEnergy, StevensHybridSpatialLinkageJoint, SensorSpringTorsionalLockRate
# ============================================================================

@dataclass
class FailLadCouplePlyMicroCrackingRate:
    """``/FAIL/LAD_COUPLE_PLY_MICRO_CRACKING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_PLY_MICRO_CRACKING_RATE`` (M382): Ladevèze rate-dependent coupled multi-axial ply micro-cracking damage accumulation, transverse/shear micro-flaw interaction, and lamina degradation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cpmcr0: float = 0.0    # initial coupled ply micro-cracking threshold stress sigma_cpmcr,0
    sigma_cpmcrc: float = 1.0    # critical coupled ply micro-cracking saturation stress sigma_cpmcr,c
    gamma_cpmcr: float = 0.0     # coupled ply micro-cracking rate sensitivity factor gamma_cpmcr
    p_cpmcr: float = 1.0         # coupled ply micro-cracking rate exponent p_cpmcr
    d_cpmcr_max: float = 0.999   # maximum allowable coupled ply micro-cracking damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M383 Suite: LadDynamicMatrixMicroFissuringRate failure, EngFlexomagnetoplasmonicexcitonicmagnonicResonanceEnergy, BakerSpatialLinkageJoint, SensorSpringBendingLockRate
# ============================================================================

@dataclass
class FailLadDynamicMatrixMicroFissuringRate:
    """``/FAIL/LAD_DYNAMIC_MATRIX_MICRO_FISSURING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_MATRIX_MICRO_FISSURING_RATE`` (M383): Ladevèze rate-dependent dynamic matrix micro-fissuring damage accumulation, micro-void coalescing, and stiffness reduction failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dmmfr0: float = 0.0    # initial dynamic matrix micro-fissuring threshold stress sigma_dmmfr,0
    sigma_dmmfrc: float = 1.0    # critical dynamic matrix micro-fissuring saturation stress sigma_dmmfr,c
    gamma_dmmfr: float = 0.0     # dynamic matrix micro-fissuring rate sensitivity factor gamma_dmmfr
    p_dmmfr: float = 1.0         # dynamic matrix micro-fissuring rate exponent p_dmmfr
    d_dmmfr_max: float = 0.999   # maximum allowable dynamic matrix micro-fissuring damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M384 Suite: LadTransverseMatrixMicroFissuringRate failure, EngFlexomagnetoplasmonicexcitonicpolaritonicResonanceEnergy, DietmeierSpatialLinkageJoint, SensorSpringTotalAngularLockRate
# ============================================================================

@dataclass
class FailLadTransverseMatrixMicroFissuringRate:
    """``/FAIL/LAD_TRANSVERSE_MATRIX_MICRO_FISSURING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_MATRIX_MICRO_FISSURING_RATE`` (M384): Ladevèze rate-dependent transverse matrix micro-fissuring damage accumulation, transverse micro-void growth, and stiffness reduction failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tmmfr0: float = 0.0    # initial transverse matrix micro-fissuring threshold stress sigma_tmmfr,0
    sigma_tmmfrc: float = 1.0    # critical transverse matrix micro-fissuring saturation stress sigma_tmmfr,c
    gamma_tmmfr: float = 0.0     # transverse matrix micro-fissuring rate sensitivity factor gamma_tmmfr
    p_tmmfr: float = 1.0         # transverse matrix micro-fissuring rate exponent p_tmmfr
    d_tmmfr_max: float = 0.999   # maximum allowable transverse matrix micro-fissuring damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M385 Suite: LadCoupleMatrixMicroFissuringRate failure, EngFlexomagnetoplasmonicmagnonicpolaritonicResonanceEnergy, HuntSpatialLinkageJoint, SensorSpringNormalDropRate
# ============================================================================

@dataclass
class FailLadCoupleMatrixMicroFissuringRate:
    """``/FAIL/LAD_COUPLE_MATRIX_MICRO_FISSURING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_MATRIX_MICRO_FISSURING_RATE`` (M385): Ladevèze rate-dependent coupled multi-axial matrix micro-fissuring damage accumulation, transverse/shear micro-flaw coalescing, and stiffness reduction failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cmmfr0: float = 0.0    # initial coupled matrix micro-fissuring threshold stress sigma_cmmfr,0
    sigma_cmmfrc: float = 1.0    # critical coupled matrix micro-fissuring saturation stress sigma_cmmfr,c
    gamma_cmmfr: float = 0.0     # coupled matrix micro-fissuring rate sensitivity factor gamma_cmmfr
    p_cmmfr: float = 1.0         # coupled matrix micro-fissuring rate exponent p_cmmfr
    d_cmmfr_max: float = 0.999   # maximum allowable coupled matrix micro-fissuring damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M386 Suite: LadDynamicMatrixMicroCrushingRate failure, EngFlexomagnetophononicexcitonicmagnonicpolaritonicResonanceEnergy, PfurnerSpatialLinkageJoint, SensorSpringTransverseDropRate
# ============================================================================

@dataclass
class FailLadDynamicMatrixMicroCrushingRate:
    """``/FAIL/LAD_DYNAMIC_MATRIX_MICRO_CRUSHING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_MATRIX_MICRO_CRUSHING_RATE`` (M386): Ladevèze rate-dependent dynamic matrix micro-crushing damage accumulation, micro-compaction damage, and compressive degradation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dmmcr0: float = 0.0    # initial dynamic matrix micro-crushing threshold stress sigma_dmmcr,0
    sigma_dmmcrc: float = 1.0    # critical dynamic matrix micro-crushing saturation stress sigma_dmmcr,c
    gamma_dmmcr: float = 0.0     # dynamic matrix micro-crushing rate sensitivity factor gamma_dmmcr
    p_dmmcr: float = 1.0         # dynamic matrix micro-crushing rate exponent p_dmmcr
    d_dmmcr_max: float = 0.999   # maximum allowable dynamic matrix micro-crushing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M387 Suite: LadTransverseMatrixMicroCrushingRate failure, EngFlexomagnetophononicplasmonicexcitonicResonanceEnergy, PhillipsSpatialLinkageJoint, SensorSpringTotalDropRate
# ============================================================================

@dataclass
class FailLadTransverseMatrixMicroCrushingRate:
    """``/FAIL/LAD_TRANSVERSE_MATRIX_MICRO_CRUSHING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_MATRIX_MICRO_CRUSHING_RATE`` (M387): Ladevèze rate-dependent transverse matrix micro-crushing damage accumulation, micro-compaction growth, and compressive degradation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tmmcr0: float = 0.0    # initial transverse matrix micro-crushing threshold stress sigma_tmmcr,0
    sigma_tmmcrc: float = 1.0    # critical transverse matrix micro-crushing saturation stress sigma_tmmcr,c
    gamma_tmmcr: float = 0.0     # transverse matrix micro-crushing rate sensitivity factor gamma_tmmcr
    p_tmmcr: float = 1.0         # transverse matrix micro-crushing rate exponent p_tmmcr
    d_tmmcr_max: float = 0.999   # maximum allowable transverse matrix micro-crushing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M388 Suite: LadCoupleMatrixMicroCrushingRate failure, EngFlexomagnetophononicplasmonicpolaritonicResonanceEnergy, KonnokSpatialLinkageJoint, SensorSpringTorsionalDropRate
# ============================================================================

@dataclass
class FailLadCoupleMatrixMicroCrushingRate:
    """``/FAIL/LAD_COUPLE_MATRIX_MICRO_CRUSHING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_MATRIX_MICRO_CRUSHING_RATE`` (M388): Ladevèze rate-dependent coupled multi-axial matrix micro-crushing damage accumulation, transverse/shear micro-compaction interaction, and compressive degradation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cmmcr0: float = 0.0    # initial coupled matrix micro-crushing threshold stress sigma_cmmcr,0
    sigma_cmmcrc: float = 1.0    # critical coupled matrix micro-crushing saturation stress sigma_cmmcr,c
    gamma_cmmcr: float = 0.0     # coupled matrix micro-crushing rate sensitivity factor gamma_cmmcr
    p_cmmcr: float = 1.0         # coupled matrix micro-crushing rate exponent p_cmmcr
    d_cmmcr_max: float = 0.999   # maximum allowable coupled matrix micro-crushing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M389 Suite: LadDynamicFiberMicroBucklingRate failure, EngFlexomagnetophononicplasmonicmagnonicpolaritonicResonanceEnergy, WohlhartHybridSpatialLinkageJoint, SensorSpringBendingDropRate
# ============================================================================

@dataclass
class FailLadDynamicFiberMicroBucklingRate:
    """``/FAIL/LAD_DYNAMIC_FIBER_MICRO_BUCKLING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_FIBER_MICRO_BUCKLING_RATE`` (M389): Ladevèze rate-dependent dynamic fiber micro-buckling damage accumulation, compressive kinking, and fiber shear instability failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dfmbr0: float = 0.0    # initial dynamic fiber micro-buckling threshold stress sigma_dfmbr,0
    sigma_dfmbrc: float = 1.0    # critical dynamic fiber micro-buckling saturation stress sigma_dfmbr,c
    gamma_dfmbr: float = 0.0     # dynamic fiber micro-buckling rate sensitivity factor gamma_dfmbr
    p_dfmbr: float = 1.0         # dynamic fiber micro-buckling rate exponent p_dfmbr
    d_dfmbr_max: float = 0.999   # maximum allowable dynamic fiber micro-buckling damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M390 Suite: LadTransverseFiberMicroBucklingRate failure, EngFlexomagnetophononicplasmonicexcitonicpolaritonicResonanceEnergy, MaverickSpatialLinkageJoint, SensorSpringTotalAngularDropRate
# ============================================================================

@dataclass
class FailLadTransverseFiberMicroBucklingRate:
    """``/FAIL/LAD_TRANSVERSE_FIBER_MICRO_BUCKLING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_FIBER_MICRO_BUCKLING_RATE`` (M390): Ladevèze rate-dependent transverse fiber micro-buckling damage accumulation, transverse kinking, and fiber shear instability failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tfmbr0: float = 0.0    # initial transverse fiber micro-buckling threshold stress sigma_tfmbr,0
    sigma_tfmbrc: float = 1.0    # critical transverse fiber micro-buckling saturation stress sigma_tfmbr,c
    gamma_tfmbr: float = 0.0     # transverse fiber micro-buckling rate sensitivity factor gamma_tfmbr
    p_tfmbr: float = 1.0         # transverse fiber micro-buckling rate exponent p_tfmbr
    d_tfmbr_max: float = 0.999   # maximum allowable transverse fiber micro-buckling damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M391 Suite: FailLadCoupleFiberMicroBucklingRate, EngFlexomagnetophononicplasmonicexcitonicmagnonicpolaritonicResonanceEnergy, LagmulKrauseSpatialLinkageJoint, SensorSpringNormalShotRate
# ============================================================================

@dataclass
class FailLadCoupleFiberMicroBucklingRate:
    """``/FAIL/LAD_COUPLE_FIBER_MICRO_BUCKLING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_FIBER_MICRO_BUCKLING_RATE`` (M391): Ladevèze rate-dependent coupled multi-axial fiber micro-buckling damage accumulation, longitudinal/shear micro-kinking interaction, and fiber compressive degradation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cfmbr0: float = 0.0    # initial coupled fiber micro-buckling threshold stress sigma_cfmbr,0
    sigma_cfmbrc: float = 1.0    # critical coupled fiber micro-buckling saturation stress sigma_cfmbr,c
    gamma_cfmbr: float = 0.0     # coupled fiber micro-buckling rate sensitivity factor gamma_cfmbr
    p_cfmbr: float = 1.0         # coupled fiber micro-buckling rate exponent p_cfmbr
    d_cfmbr_max: float = 0.999   # maximum allowable coupled fiber micro-buckling damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M392 Suite: FailLadDynamicFiberSplittingRate, EngFlexothermophononicplasmonicexcitonicmagnonicpolaritonicResonanceEnergy, LagmulSturgessSpatialLinkageJoint, SensorSpringTransverseShotRate
# ============================================================================

@dataclass
class FailLadDynamicFiberSplittingRate:
    """``/FAIL/LAD_DYNAMIC_FIBER_SPLITTING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_FIBER_SPLITTING_RATE`` (M392): Ladevèze rate-dependent dynamic fiber splitting damage accumulation, longitudinal tensile splitting, and fiber decohesion failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dfsr0: float = 0.0     # initial dynamic fiber splitting threshold stress sigma_dfsr,0
    sigma_dfsrc: float = 1.0     # critical dynamic fiber splitting saturation stress sigma_dfsr,c
    gamma_dfsr: float = 0.0      # dynamic fiber splitting rate sensitivity factor gamma_dfsr
    p_dfsr: float = 1.0          # dynamic fiber splitting rate exponent p_dfsr
    d_dfsr_max: float = 0.999    # maximum allowable dynamic fiber splitting damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M393 Suite: FailLadTransverseFiberSplittingRate, EngFlexothermoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy, LagmulHuntSpecialSpatialLinkageJoint, SensorSpringTotalShotRate
# ============================================================================

@dataclass
class FailLadTransverseFiberSplittingRate:
    """``/FAIL/LAD_TRANSVERSE_FIBER_SPLITTING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_FIBER_SPLITTING_RATE`` (M393): Ladevèze rate-dependent transverse fiber splitting damage accumulation, transverse splitting crack opening, and fiber tensile decohesion failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tfsr0: float = 0.0     # initial transverse fiber splitting threshold stress sigma_tfsr,0
    sigma_tfsrc: float = 1.0     # critical transverse fiber splitting saturation stress sigma_tfsr,c
    gamma_tfsr: float = 0.0      # transverse fiber splitting rate sensitivity factor gamma_tfsr
    p_tfsr: float = 1.0          # transverse fiber splitting rate exponent p_tfsr
    d_tfsr_max: float = 0.999    # maximum allowable transverse fiber splitting damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M394 Suite: FailLadCoupleFiberSplittingRate, EngFlexothermophononicexcitonicmagnonicpolaritonicResonanceEnergy, LagmulAlbrechtSpatialLinkageJoint, SensorSpringTorsionalShotRate
# ============================================================================

@dataclass
class FailLadCoupleFiberSplittingRate:
    """``/FAIL/LAD_COUPLE_FIBER_SPLITTING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_FIBER_SPLITTING_RATE`` (M394): Ladevèze rate-dependent coupled multi-axial fiber splitting damage accumulation, longitudinal/transverse splitting interaction, and fiber tensile decohesion failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cfsr0: float = 0.0     # initial coupled fiber splitting threshold stress sigma_cfsr,0
    sigma_cfsrc: float = 1.0     # critical coupled fiber splitting saturation stress sigma_cfsr,c
    gamma_cfsr: float = 0.0      # coupled fiber splitting rate sensitivity factor gamma_cfsr
    p_cfsr: float = 1.0          # coupled fiber splitting rate exponent p_cfsr
    d_cfsr_max: float = 0.999    # maximum allowable coupled fiber splitting damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M395 Suite: FailLadDynamicDelaminationRate, EngFlexothermoplasmonicmagnonicpolaritonicResonanceEnergy, LagmulKirsonSpatialLinkageJoint, SensorSpringBendingShotRate
# ============================================================================

@dataclass
class FailLadDynamicDelaminationRate:
    """``/FAIL/LAD_DYNAMIC_DELAMINATION_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_DELAMINATION_RATE`` (M395): Ladevèze rate-dependent dynamic interlaminar delamination damage accumulation, mode I/II/III mixed-mode decohesion, and interlaminar interface separation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_ddlr0: float = 0.0     # initial dynamic delamination threshold stress sigma_ddlr,0
    sigma_ddlrc: float = 1.0     # critical dynamic delamination saturation stress sigma_ddlr,c
    gamma_ddlr: float = 0.0      # dynamic delamination rate sensitivity factor gamma_ddlr
    p_ddlr: float = 1.0          # dynamic delamination rate exponent p_ddlr
    d_ddlr_max: float = 0.999    # maximum allowable dynamic delamination damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference

    @property
    def sigma_ddr0(self) -> float:
        return self.sigma_ddlr0

    @sigma_ddr0.setter
    def sigma_ddr0(self, val: float) -> None:
        self.sigma_ddlr0 = val

    @property
    def sigma_ddrc(self) -> float:
        return self.sigma_ddlrc

    @sigma_ddrc.setter
    def sigma_ddrc(self, val: float) -> None:
        self.sigma_ddlrc = val

    @property
    def gamma_ddr(self) -> float:
        return self.gamma_ddlr

    @gamma_ddr.setter
    def gamma_ddr(self, val: float) -> None:
        self.gamma_ddlr = val

    @property
    def p_ddr(self) -> float:
        return self.p_ddlr

    @p_ddr.setter
    def p_ddr(self, val: float) -> None:
        self.p_ddlr = val

    @property
    def d_ddr_max(self) -> float:
        return self.d_ddlr_max

    @d_ddr_max.setter
    def d_ddr_max(self, val: float) -> None:
        self.d_ddlr_max = val


# ============================================================================
# M396 Suite: FailLadTransverseDelaminationRate, EngFlexothermoplasmonicexcitonicpolaritonicResonanceEnergy, LagmulDieselSpatialLinkageJoint, SensorSpringTotalAngularShotRate
# ============================================================================

@dataclass
class FailLadTransverseDelaminationRate:
    """``/FAIL/LAD_TRANSVERSE_DELAMINATION_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_DELAMINATION_RATE`` (M396): Ladevèze rate-dependent transverse interlaminar delamination damage accumulation, mode II shear decohesion, and interlaminar interface sliding failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tdlr0: float = 0.0     # initial transverse delamination threshold stress sigma_tdlr,0
    sigma_tdlrc: float = 1.0     # critical transverse delamination saturation stress sigma_tdlr,c
    gamma_tdlr: float = 0.0      # transverse delamination rate sensitivity factor gamma_tdlr
    p_tdlr: float = 1.0          # transverse delamination rate exponent p_tdlr
    d_tdlr_max: float = 0.999    # maximum allowable transverse delamination damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference

    @property
    def sigma_tdr0(self) -> float:
        return self.sigma_tdlr0

    @sigma_tdr0.setter
    def sigma_tdr0(self, val: float) -> None:
        self.sigma_tdlr0 = val

    @property
    def sigma_tdrc(self) -> float:
        return self.sigma_tdlrc

    @sigma_tdrc.setter
    def sigma_tdrc(self, val: float) -> None:
        self.sigma_tdlrc = val

    @property
    def gamma_tdr(self) -> float:
        return self.gamma_tdlr

    @gamma_tdr.setter
    def gamma_tdr(self, val: float) -> None:
        self.gamma_tdlr = val

    @property
    def p_tdr(self) -> float:
        return self.p_tdlr

    @p_tdr.setter
    def p_tdr(self, val: float) -> None:
        self.p_tdlr = val

    @property
    def d_tdr_max(self) -> float:
        return self.d_tdlr_max

    @d_tdr_max.setter
    def d_tdr_max(self, val: float) -> None:
        self.d_tdlr_max = val


# ============================================================================
# M397 Suite: FailLadCoupleDelaminationRate, EngFlexothermophononicmagnonicpolaritonicResonanceEnergy, LagmulTchebychevSpatialLinkageJoint, SensorSpringNormalSnapRate
# ============================================================================

@dataclass
class FailLadCoupleDelaminationRate:
    """``/FAIL/LAD_COUPLE_DELAMINATION_RATE`` or ``/FAIL/LADEVEZE_COUPLED_DELAMINATION_RATE`` (M397): Ladevèze rate-dependent coupled multi-axial interlaminar delamination damage accumulation, mode I normal opening / mode II-III shear mixed-mode interaction, and interlaminar interface separation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cdlr0: float = 0.0     # initial coupled delamination threshold stress sigma_cdlr,0
    sigma_cdlrc: float = 1.0     # critical coupled delamination saturation stress sigma_cdlr,c
    gamma_cdlr: float = 0.0      # coupled delamination rate sensitivity factor gamma_cdlr
    p_cdlr: float = 1.0          # coupled delamination rate exponent p_cdlr
    d_cdlr_max: float = 0.999    # maximum allowable coupled delamination damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference

    @property
    def sigma_cdel0(self) -> float:
        return self.sigma_cdlr0

    @sigma_cdel0.setter
    def sigma_cdel0(self, val: float) -> None:
        self.sigma_cdlr0 = val

    @property
    def sigma_cdelc(self) -> float:
        return self.sigma_cdlrc

    @sigma_cdelc.setter
    def sigma_cdelc(self, val: float) -> None:
        self.sigma_cdlrc = val

    @property
    def gamma_cdel(self) -> float:
        return self.gamma_cdlr

    @gamma_cdel.setter
    def gamma_cdel(self, val: float) -> None:
        self.gamma_cdlr = val

    @property
    def p_cdel(self) -> float:
        return self.p_cdlr

    @p_cdel.setter
    def p_cdel(self, val: float) -> None:
        self.p_cdlr = val

    @property
    def d_cdel_max(self) -> float:
        return self.d_cdlr_max

    @d_cdel_max.setter
    def d_cdel_max(self, val: float) -> None:
        self.d_cdlr_max = val



# ============================================================================
# M398 Suite: FailLadDynamicMatrixCrackingRate, EngFlexothermoplasmonicpolaritonicResonanceEnergy, LagmulSylvesterSpatialLinkageJoint, SensorSpringTransverseSnapRate
# ============================================================================

@dataclass
class FailLadDynamicMatrixCrackingRate:
    """``/FAIL/LAD_DYNAMIC_MATRIX_CRACKING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_MATRIX_CRACKING_RATE`` (M398): Ladevèze rate-dependent dynamic transverse matrix cracking and intralaminar shear micro-cracking damage accumulation, mode I transverse tension / mode II shear decohesion, and matrix degradation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dmcr0: float = 0.0     # initial dynamic matrix cracking threshold stress sigma_dmcr,0
    sigma_dmcrc: float = 1.0     # critical dynamic matrix cracking saturation stress sigma_dmcr,c
    gamma_dmcr: float = 0.0      # dynamic matrix cracking rate sensitivity factor gamma_dmcr
    p_dmcr: float = 1.0          # dynamic matrix cracking rate exponent p_dmcr
    d_dmcr_max: float = 0.999    # maximum allowable dynamic matrix cracking damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M399 Suite: FailLadTransverseMatrixCrackingRate, EngFlexothermophononicpolaritonicResonanceEnergy, LagmulCauchySpatialLinkageJoint, SensorSpringTotalSnapRate
# ============================================================================

@dataclass
class FailLadTransverseMatrixCrackingRate:
    """``/FAIL/LAD_TRANSVERSE_MATRIX_CRACKING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_MATRIX_CRACKING_RATE`` (M399): Ladevèze rate-dependent transverse matrix cracking and intralaminar opening damage accumulation, mode I transverse tension decohesion, and matrix failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tmcr0: float = 0.0     # initial transverse matrix cracking threshold stress sigma_tmcr,0
    sigma_tmcrc: float = 1.0     # critical transverse matrix cracking saturation stress sigma_tmcr,c
    gamma_tmcr: float = 0.0      # transverse matrix cracking rate sensitivity factor gamma_tmcr
    p_tmcr: float = 1.0          # transverse matrix cracking rate exponent p_tmcr
    d_tmcr_max: float = 0.999    # maximum allowable transverse matrix cracking damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M400 Suite: FailLadCoupleMatrixCrackingRate, EngFlexothermoexcitonicpolaritonicResonanceEnergy, LagmulCayleySpatialLinkageJoint, SensorSpringTorsionalSnapRate
# ============================================================================

@dataclass
class FailLadCoupleMatrixCrackingRate:
    """``/FAIL/LAD_COUPLE_MATRIX_CRACKING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_MATRIX_CRACKING_RATE`` (M400): Ladevèze rate-dependent coupled multi-axial matrix cracking and intralaminar shear-tension interaction damage accumulation, mode I transverse tension / mode II-III shear mixed-mode decohesion, and matrix failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cmcr0: float = 0.0     # initial coupled matrix cracking threshold stress sigma_cmcr,0
    sigma_cmcrc: float = 1.0     # critical coupled matrix cracking saturation stress sigma_cmcr,c
    gamma_cmcr: float = 0.0      # coupled matrix cracking rate sensitivity factor gamma_cmcr
    p_cmcr: float = 1.0          # coupled matrix cracking rate exponent p_cmcr
    d_cmcr_max: float = 0.999    # maximum allowable coupled matrix cracking damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M401 Suite: FailLadDynamicFiberKinkingRate, EngFlexothermomagnonicpolaritonicResonanceEnergy, LagmulEuclidSpatialLinkageJoint, SensorSpringBendingSnapRate
# ============================================================================

@dataclass
class FailLadDynamicFiberKinkingRate:
    """``/FAIL/LAD_DYNAMIC_FIBER_KINKING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_FIBER_KINKING_RATE`` (M401): Ladevèze rate-dependent dynamic fiber micro-buckling and kinking damage accumulation, longitudinal compressive shear failure, and fiber kink band formation model."""
    mat_id: int = 0
    title: str = ""
    sigma_dfkr0: float = 0.0     # initial fiber kinking threshold stress sigma_dfkr,0
    sigma_dfkrc: float = 1.0     # critical fiber kinking saturation stress sigma_dfkr,c
    gamma_dfkr: float = 0.0      # fiber kinking rate sensitivity factor gamma_dfkr
    p_dfkr: float = 1.0          # fiber kinking rate exponent p_dfkr
    d_dfkr_max: float = 0.999    # maximum allowable fiber kinking damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M402 Suite: FailLadTransverseFiberKinkingRate, EngFlexothermoplasmonicphononicpolaritonicResonanceEnergy, LagmulFermatSpatialLinkageJoint, SensorSpringTotalAngularSnapRate
# ============================================================================

@dataclass
class FailLadTransverseFiberKinkingRate:
    """``/FAIL/LAD_TRANSVERSE_FIBER_KINKING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_FIBER_KINKING_RATE`` (M402): Ladevèze rate-dependent transverse fiber micro-buckling and kinking damage accumulation, off-axis compressive shear failure, and transverse kink band propagation model."""
    mat_id: int = 0
    title: str = ""
    sigma_tfkr0: float = 0.0     # initial transverse fiber kinking threshold stress sigma_tfkr,0
    sigma_tfkrc: float = 1.0     # critical transverse fiber kinking saturation stress sigma_tfkr,c
    gamma_tfkr: float = 0.0      # transverse fiber kinking rate sensitivity factor gamma_tfkr
    p_tfkr: float = 1.0          # transverse fiber kinking rate exponent p_tfkr
    d_tfkr_max: float = 0.999    # maximum allowable transverse fiber kinking damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference



# ============================================================================
# M403 Suite: FailLadCoupleFiberKinkingRate, EngFlexothermoexcitonicphononicpolaritonicResonanceEnergy, LagmulPascalSpatialLinkageJoint, SensorSpringNormalCrackleRate
# ============================================================================

@dataclass
class FailLadCoupleFiberKinkingRate:
    """``/FAIL/LAD_COUPLE_FIBER_KINKING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_FIBER_KINKING_RATE`` (M403): Ladevèze rate-dependent coupled multi-axial fiber micro-buckling and kinking damage accumulation, longitudinal/transverse compressive shear interaction, and multi-scale kink band formation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cfkr0: float = 0.0     # initial coupled fiber kinking threshold stress sigma_cfkr,0
    sigma_cfkrc: float = 1.0     # critical coupled fiber kinking saturation stress sigma_cfkr,c
    gamma_cfkr: float = 0.0      # coupled fiber kinking rate sensitivity factor gamma_cfkr
    p_cfkr: float = 1.0          # coupled fiber kinking rate exponent p_cfkr
    d_cfkr_max: float = 0.999    # maximum allowable coupled fiber kinking damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference




# ============================================================================
# M404 Suite: FailLadDynamicCoreCrushingRate, EngFlexothermomagnonicphononicpolaritonicResonanceEnergy, LagmulDescartesSpatialLinkageJoint, SensorSpringTransverseCrackleRate
# ============================================================================

@dataclass
class FailLadDynamicCoreCrushingRate:
    """``/FAIL/LAD_DYNAMIC_CORE_CRUSHING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_CORE_CRUSHING_RATE`` (M404): Ladevèze rate-dependent dynamic sandwich core crushing, cell wall buckling, and hydrostatic compressive collapse damage accumulation model."""
    mat_id: int = 0
    title: str = ""
    sigma_dccr0: float = 0.0     # initial core crushing threshold stress sigma_dccr,0
    sigma_dccrc: float = 1.0     # critical core crushing saturation stress sigma_dccr,c
    gamma_dccr: float = 0.0      # core crushing rate sensitivity factor gamma_dccr
    p_dccr: float = 1.0          # core crushing rate exponent p_dccr
    d_dccr_max: float = 0.999    # maximum allowable core crushing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M405 Suite: FailLadTransverseCoreCrushingRate, EngFlexothermoplasmonicexcitonicphononicpolaritonicResonanceEnergy, LagmulLeibnizSpatialLinkageJoint, SensorSpringTotalCrackleRate
# ============================================================================

@dataclass
class FailLadTransverseCoreCrushingRate:
    """``/FAIL/LAD_TRANSVERSE_CORE_CRUSHING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_CORE_CRUSHING_RATE`` (M405): Ladevèze rate-dependent transverse sandwich core crushing, off-axis cell wall shearing, and transverse hydrostatic compressive collapse damage accumulation model."""
    mat_id: int = 0
    title: str = ""
    sigma_tccr0: float = 0.0     # initial transverse core crushing threshold stress sigma_tccr,0
    sigma_tccrc: float = 1.0     # critical transverse core crushing saturation stress sigma_tccr,c
    gamma_tccr: float = 0.0      # transverse core crushing rate sensitivity factor gamma_tccr
    p_tccr: float = 1.0          # transverse core crushing rate exponent p_tccr
    d_tccr_max: float = 0.999    # maximum allowable transverse core crushing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M406 Suite: FailLadCoupleCoreCrushingRate, EngFlexothermoplasmonicmagnonicphononicpolaritonicResonanceEnergy, LagmulNewtonSpatialLinkageJoint, SensorSpringTorsionalCrackleRate
# ============================================================================

@dataclass
class FailLadCoupleCoreCrushingRate:
    """``/FAIL/LAD_COUPLE_CORE_CRUSHING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_CORE_CRUSHING_RATE`` (M406): Ladevèze rate-dependent coupled multi-axial sandwich core crushing, cell wall buckling-shear collapse interaction, and hydrostatic compressive failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cccr0: float = 0.0     # initial coupled core crushing threshold stress sigma_cccr,0
    sigma_cccrc: float = 1.0     # critical coupled core crushing saturation stress sigma_cccr,c
    gamma_cccr: float = 0.0      # coupled core crushing rate sensitivity factor gamma_cccr
    p_cccr: float = 1.0          # coupled core crushing rate exponent p_cccr
    d_cccr_max: float = 0.999    # maximum allowable coupled core crushing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M407 Suite: FailLadDynamicCoreShearingRate, EngFlexothermoexcitonicmagnonicphononicpolaritonicResonanceEnergy, LagmulGaussSpatialLinkageJoint, SensorSpringBendingCrackleRate
# ============================================================================

@dataclass
class FailLadDynamicCoreShearingRate:
    """``/FAIL/LAD_DYNAMIC_CORE_SHEARING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_CORE_SHEARING_RATE`` (M407): Ladevèze rate-dependent dynamic sandwich core shear cracking, transverse core yielding, and core shear damage accumulation model."""
    mat_id: int = 0
    title: str = ""
    sigma_dcsr0: float = 0.0     # initial dynamic core shearing threshold stress sigma_dcsr,0
    sigma_dcsrc: float = 1.0     # critical dynamic core shearing saturation stress sigma_dcsr,c
    gamma_dcsr: float = 0.0      # dynamic core shearing rate sensitivity factor gamma_dcsr
    p_dcsr: float = 1.0          # dynamic core shearing rate exponent p_dcsr
    d_dcsr_max: float = 0.999    # maximum allowable dynamic core shearing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M408 Suite: FailLadTransverseCoreShearingRate, EngFlexothermoplasmonicexcitonicmagnonicphononicpolaritonicResonanceEnergy, LagmulEulerSpatialLinkageJoint, SensorSpringTotalAngularCrackleRate
# ============================================================================

@dataclass
class FailLadTransverseCoreShearingRate:
    """``/FAIL/LAD_TRANSVERSE_CORE_SHEARING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_CORE_SHEARING_RATE`` (M408): Ladevèze rate-dependent transverse sandwich core shear cracking, out-of-plane core shearing, and transverse core shear damage accumulation model."""
    mat_id: int = 0
    title: str = ""
    sigma_tcsr0: float = 0.0     # initial transverse core shearing threshold stress sigma_tcsr,0
    sigma_tcsrc: float = 1.0     # critical transverse core shearing saturation stress sigma_tcsr,c
    gamma_tcsr: float = 0.0      # transverse core shearing rate sensitivity factor gamma_tcsr
    p_tcsr: float = 1.0          # transverse core shearing rate exponent p_tcsr
    d_tcsr_max: float = 0.999    # maximum allowable transverse core shearing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M409 Suite: FailLadCoupleCoreShearingRate, EngElectrothermoflexoplasmonicphononicpolaritonicResonanceEnergy, LagmulLagrangeSpatialLinkageJoint, SensorSpringNormalPopRate
# ============================================================================

@dataclass
class FailLadCoupleCoreShearingRate:
    """``/FAIL/LAD_COUPLE_CORE_SHEARING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_CORE_SHEARING_RATE`` (M409): Ladevèze rate-dependent coupled multi-axial sandwich core shearing, core shear-crushing interaction, and coupled transverse shear failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_ccsr0: float = 0.0     # initial coupled core shearing threshold stress sigma_ccsr,0
    sigma_ccsrc: float = 1.0     # critical coupled core shearing saturation stress sigma_ccsr,c
    gamma_ccsr: float = 0.0      # coupled core shearing rate sensitivity factor gamma_ccsr
    p_ccsr: float = 1.0          # coupled core shearing rate exponent p_ccsr
    d_ccsr_max: float = 0.999    # maximum allowable coupled core shearing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M410 Suite: FailLadDynamicCoreDebondingRate, EngElectrothermoflexoexcitonicphononicpolaritonicResonanceEnergy, LagmulLaplaceSpatialLinkageJoint, SensorSpringTransversePopRate
# ============================================================================

@dataclass
class FailLadDynamicCoreDebondingRate:
    """``/FAIL/LAD_DYNAMIC_CORE_DEBONDING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_CORE_DEBONDING_RATE`` (M410): Ladevèze rate-dependent dynamic sandwich core-to-facesheet debonding, skin-core adhesive peeling, and interfacial delamination damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_dcd0: float = 0.0      # initial dynamic core debonding threshold stress sigma_dcd,0
    sigma_dcdc: float = 1.0      # critical dynamic core debonding saturation stress sigma_dcd,c
    gamma_dcd: float = 0.0       # dynamic core debonding rate sensitivity factor gamma_dcd
    p_dcd: float = 1.0           # dynamic core debonding rate exponent p_dcd
    d_dcd_max: float = 0.999     # maximum allowable dynamic core debonding damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M411 Suite: FailLadTransverseCoreDebondingRate, EngElectrothermoflexomagnonicphononicpolaritonicResonanceEnergy, LagmulFourierSpatialLinkageJoint, SensorSpringTotalPopRate
# ============================================================================

@dataclass
class FailLadTransverseCoreDebondingRate:
    """``/FAIL/LAD_TRANSVERSE_CORE_DEBONDING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_CORE_DEBONDING_RATE`` (M411): Ladevèze rate-dependent transverse sandwich core-to-facesheet debonding, off-axis skin-core adhesive shearing, and interfacial delamination damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_tcd0: float = 0.0      # initial transverse core debonding threshold stress sigma_tcd,0
    sigma_tcdc: float = 1.0      # critical transverse core debonding saturation stress sigma_tcd,c
    gamma_tcd: float = 0.0       # transverse core debonding rate sensitivity factor gamma_tcd
    p_tcd: float = 1.0           # transverse core debonding rate exponent p_tcd
    d_tcd_max: float = 0.999     # maximum allowable transverse core debonding damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M412 Suite: FailLadCoupleCoreDebondingRate, EngElectrothermoflexoplasmonicexcitonicphononicpolaritonicResonanceEnergy, LagmulPoissonSpatialLinkageJoint, SensorSpringTorsionalPopRate
# ============================================================================

@dataclass
class FailLadCoupleCoreDebondingRate:
    """``/FAIL/LAD_COUPLE_CORE_DEBONDING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_CORE_DEBONDING_RATE`` (M412): Ladevèze rate-dependent coupled dynamic-transverse sandwich core-to-facesheet debonding, mixed-mode peel-shear adhesive failure, and interfacial delamination damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_ccd0: float = 0.0      # initial coupled core debonding threshold stress sigma_ccd,0
    sigma_ccdc: float = 1.0      # critical coupled core debonding saturation stress sigma_ccd,c
    gamma_ccd: float = 0.0       # coupled core debonding rate sensitivity factor gamma_ccd
    p_ccd: float = 1.0           # coupled core debonding rate exponent p_ccd
    d_ccd_max: float = 0.999     # maximum allowable coupled core debonding damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference




# ============================================================================
# M413 Suite: FailLadDynamicCoreCrackingRate, EngElectrothermoflexoplasmonicmagnonicphononicpolaritonicResonanceEnergy, LagmulBesselSpatialLinkageJoint, SensorSpringBendingPopRate
# ============================================================================

@dataclass
class FailLadDynamicCoreCrackingRate:
    """``/FAIL/LAD_DYNAMIC_CORE_CRACKING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_CORE_CRACKING_RATE`` (M413): Ladevèze rate-dependent dynamic sandwich core cell-wall fracture, transverse core cracking, and dynamic cell rupture damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_dcc0: float = 0.0      # initial dynamic core cracking threshold stress sigma_dcc,0
    sigma_dccc: float = 1.0      # critical dynamic core cracking saturation stress sigma_dcc,c
    gamma_dcc: float = 0.0       # dynamic core cracking rate sensitivity factor gamma_dcc
    p_dcc: float = 1.0           # dynamic core cracking rate exponent p_dcc
    d_dcc_max: float = 0.999     # maximum allowable dynamic core cracking damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M414 Suite: FailLadTransverseCoreCrackingRate, EngElectrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonicResonanceEnergy, LagmulRiemannSpatialLinkageJoint, SensorSpringTotalAngularPopRate
# ============================================================================

@dataclass
class FailLadTransverseCoreCrackingRate:
    """``/FAIL/LAD_TRANSVERSE_CORE_CRACKING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_CORE_CRACKING_RATE`` (M414): Ladevèze rate-dependent transverse sandwich core cell-wall fracture, off-axis core cracking, and transverse cell rupture damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_tcc0: float = 0.0      # initial transverse core cracking threshold stress sigma_tcc,0
    sigma_tccc: float = 1.0      # critical transverse core cracking saturation stress sigma_tcc,c
    gamma_tcc: float = 0.0       # transverse core cracking rate sensitivity factor gamma_tcc
    p_tcc: float = 1.0           # transverse core cracking rate exponent p_tcc
    d_tcc_max: float = 0.999     # maximum allowable transverse core cracking damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference




# ============================================================================
# M415 Suite: FailLadCoupleCoreCrackingRate, EngElectrothermoflexomagnetoplasmonicphononicpolaritonicResonanceEnergy, LagmulHilbertSpatialLinkageJoint, SensorSpringNormalLockRate
# ============================================================================

@dataclass
class FailLadCoupleCoreCrackingRate:
    """``/FAIL/LAD_COUPLE_CORE_CRACKING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_CORE_CRACKING_RATE`` (M415): Ladevèze rate-dependent coupled multi-axial sandwich core cell-wall fracture, mixed-mode core cracking, and dynamic cell rupture damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_ccc0: float = 0.0      # initial coupled core cracking threshold stress sigma_ccc,0
    sigma_cccc: float = 1.0      # critical coupled core cracking saturation stress sigma_ccc,c
    gamma_ccc: float = 0.0       # coupled core cracking rate sensitivity factor gamma_ccc
    p_ccc: float = 1.0           # coupled core cracking rate exponent p_ccc
    d_ccc_max: float = 0.999     # maximum allowable coupled core cracking damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference



# ============================================================================
# M416 Suite: FailLadDynamicFacesheetCoreShearingRate, EngElectrothermoflexomagnetoexcitonicphononicpolaritonicResonanceEnergy, LagmulBanachSpatialLinkageJoint, SensorSpringTransverseLockRate
# ============================================================================

@dataclass
class FailLadDynamicFacesheetCoreShearingRate:
    """``/FAIL/LAD_DYNAMIC_FACESHEET_CORE_SHEARING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_FACESHEET_CORE_SHEARING_RATE`` (M416): Ladevèze rate-dependent dynamic sandwich facesheet-core shear slip, skin-core adhesive shearing, and interfacial core shear damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_dfcs0: float = 0.0     # initial facesheet-core shearing threshold stress sigma_dfcs,0
    sigma_dfcsc: float = 1.0     # critical facesheet-core shearing saturation stress sigma_dfcs,c
    gamma_dfcs: float = 0.0      # facesheet-core shearing rate sensitivity factor gamma_dfcs
    p_dfcs: float = 1.0          # facesheet-core shearing rate exponent p_dfcs
    d_dfcs_max: float = 0.999    # maximum allowable facesheet-core shearing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference



# ============================================================================
# M417 Suite: FailLadTransverseFacesheetCoreShearingRate, EngElectrothermoflexomagnetomagnonicphononicpolaritonicResonanceEnergy, LagmulSobolevSpatialLinkageJoint, SensorSpringTotalLockRate
# ============================================================================

@dataclass
class FailLadTransverseFacesheetCoreShearingRate:
    """``/FAIL/LAD_TRANSVERSE_FACESHEET_CORE_SHEARING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_FACESHEET_CORE_SHEARING_RATE`` (M417): Ladevèze rate-dependent transverse sandwich facesheet-core shear slip, off-axis skin-core adhesive shearing, and interfacial core shear damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_tfcs0: float = 0.0     # initial transverse facesheet-core shearing threshold stress sigma_tfcs,0
    sigma_tfcsc: float = 1.0     # critical transverse facesheet-core shearing saturation stress sigma_tfcs,c
    gamma_tfcs: float = 0.0      # transverse facesheet-core shearing rate sensitivity factor gamma_tfcs
    p_tfcs: float = 1.0          # transverse facesheet-core shearing rate exponent p_tfcs
    d_tfcs_max: float = 0.999    # maximum allowable transverse facesheet-core shearing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference



# ============================================================================
# M418 Suite: FailLadCoupleFacesheetCoreShearingRate, EngElectrothermoflexomagnetoexcitonicmagnonicphononicpolaritonicResonanceEnergy, LagmulFrechetSpatialLinkageJoint, SensorSpringTorsionalLockRate
# ============================================================================

@dataclass
class FailLadCoupleFacesheetCoreShearingRate:
    """``/FAIL/LAD_COUPLE_FACESHEET_CORE_SHEARING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_FACESHEET_CORE_SHEARING_RATE`` (M418): Ladevèze rate-dependent coupled sandwich facesheet-core shear slip, mixed-mode peel-shear adhesive shearing, and interfacial core shear damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_cfcs0: float = 0.0     # initial coupled facesheet-core shearing threshold stress sigma_cfcs,0
    sigma_cfcsc: float = 1.0     # critical coupled facesheet-core shearing saturation stress sigma_cfcs,c
    gamma_cfcs: float = 0.0      # coupled facesheet-core shearing rate sensitivity factor gamma_cfcs
    p_cfcs: float = 1.0          # coupled facesheet-core shearing rate exponent p_cfcs
    d_cfcs_max: float = 0.999    # maximum allowable coupled facesheet-core shearing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference



# ============================================================================
# M419 Suite: FailLadDynamicFacesheetCoreDebondingRate, EngElectrothermoflexomagnetoplasmonicexcitonicphononicpolaritonicResonanceEnergy, LagmulHausdorffSpatialLinkageJoint, SensorSpringBendingLockRate
# ============================================================================

@dataclass
class FailLadDynamicFacesheetCoreDebondingRate:
    """``/FAIL/LAD_DYNAMIC_FACESHEET_CORE_DEBONDING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_FACESHEET_CORE_DEBONDING_RATE`` (M419): Ladevèze rate-dependent dynamic sandwich facesheet-core debonding, skin-core adhesive peeling, and interfacial core normal separation damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_dfcd0: float = 0.0     # initial dynamic facesheet-core debonding threshold stress sigma_dfcd,0
    sigma_dfcdc: float = 1.0     # critical dynamic facesheet-core debonding saturation stress sigma_dfcd,c
    gamma_dfcd: float = 0.0      # dynamic facesheet-core debonding rate sensitivity factor gamma_dfcd
    p_dfcd: float = 1.0          # dynamic facesheet-core debonding rate exponent p_dfcd
    d_dfcd_max: float = 0.999    # maximum allowable dynamic facesheet-core debonding damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference



# ============================================================================
# M420 Suite: FailLadTransverseFacesheetCoreDebondingRate, EngElectrothermoflexomagnetoplasmonicmagnonicphononicpolaritonicResonanceEnergy, LagmulCartanSpatialLinkageJoint, SensorSpringTotalAngularLockRate
# ============================================================================

@dataclass
class FailLadTransverseFacesheetCoreDebondingRate:
    """``/FAIL/LAD_TRANSVERSE_FACESHEET_CORE_DEBONDING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_FACESHEET_CORE_DEBONDING_RATE`` (M420): Ladevèze rate-dependent transverse sandwich facesheet-core debonding, off-axis skin-core adhesive peeling, and interfacial core normal separation damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_tfcd0: float = 0.0     # initial transverse facesheet-core debonding threshold stress sigma_tfcd,0
    sigma_tfcdc: float = 1.0     # critical transverse facesheet-core debonding saturation stress sigma_tfcd,c
    gamma_tfcd: float = 0.0      # transverse facesheet-core debonding rate sensitivity factor gamma_tfcd
    p_tfcd: float = 1.0          # transverse facesheet-core debonding rate exponent p_tfcd
    d_tfcd_max: float = 0.999    # maximum allowable transverse facesheet-core debonding damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference



# ============================================================================
# M421 Suite: FailLadCoupleFacesheetCoreDebondingRate, EngElectrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonicResonanceEnergy, LagmulCliffordSpatialLinkageJoint, SensorSpringNormalDropRate
# ============================================================================

@dataclass
class FailLadCoupleFacesheetCoreDebondingRate:
    """``/FAIL/LAD_COUPLE_FACESHEET_CORE_DEBONDING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_FACESHEET_CORE_DEBONDING_RATE`` (M421): Ladevèze rate-dependent coupled sandwich facesheet-core debonding, mixed-mode peel-shear skin-core adhesive peeling, and interfacial core normal separation damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_cfcd0: float = 0.0     # initial coupled facesheet-core debonding threshold stress sigma_cfcd,0
    sigma_cfcdc: float = 1.0     # critical coupled facesheet-core debonding saturation stress sigma_cfcd,c
    gamma_cfcd: float = 0.0      # coupled facesheet-core debonding rate sensitivity factor gamma_cfcd
    p_cfcd: float = 1.0          # coupled facesheet-core debonding rate exponent p_cfcd
    d_cfcd_max: float = 0.999    # maximum allowable coupled facesheet-core debonding damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M422 Suite: FailLadDynamicHoneycombCoreCrushingRate, EngElectrothermoflexomagnetoplasmonicmagnonpolaritonicResonanceEnergy, LagmulGrassmannSpatialLinkageJoint, SensorSpringTransverseDropRate
# ============================================================================

@dataclass
class FailLadDynamicHoneycombCoreCrushingRate:
    """``/FAIL/LAD_DYNAMIC_HONEYCOMB_CORE_CRUSHING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_HONEYCOMB_CORE_CRUSHING_RATE`` (M422): Ladevèze rate-dependent dynamic sandwich honeycomb core cell-wall buckling, progressive compressive crushing, and transverse crushing damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_dhcc0: float = 0.0     # initial dynamic honeycomb core crushing threshold stress sigma_dhcc,0
    sigma_dhccc: float = 1.0     # critical dynamic honeycomb core crushing saturation stress sigma_dhcc,c
    gamma_dhcc: float = 0.0      # dynamic honeycomb core crushing rate sensitivity factor gamma_dhcc
    p_dhcc: float = 1.0          # dynamic honeycomb core crushing rate exponent p_dhcc
    d_dhcc_max: float = 0.999    # maximum allowable dynamic honeycomb core crushing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M423 Suite: FailLadTransverseHoneycombCoreCrushingRate, EngElectrothermoflexomagnetoexcitonicmagnonpolaritonicResonanceEnergy, LagmulMinkowskiSpatialLinkageJoint, SensorSpringTotalDropRate
# ============================================================================

@dataclass
class FailLadTransverseHoneycombCoreCrushingRate:
    """``/FAIL/LAD_TRANSVERSE_HONEYCOMB_CORE_CRUSHING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_HONEYCOMB_CORE_CRUSHING_RATE`` (M423): Ladevèze rate-dependent transverse sandwich honeycomb core cell-wall buckling, off-axis compressive crushing, and transverse crushing damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_thcc0: float = 0.0     # initial transverse honeycomb core crushing threshold stress sigma_thcc,0
    sigma_thccc: float = 1.0     # critical transverse honeycomb core crushing saturation stress sigma_thcc,c
    gamma_thcc: float = 0.0      # transverse honeycomb core crushing rate sensitivity factor gamma_thcc
    p_thcc: float = 1.0          # transverse honeycomb core crushing rate exponent p_thcc
    d_thcc_max: float = 0.999    # maximum allowable transverse honeycomb core crushing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M424 Suite: FailLadCoupleHoneycombCoreCrushingRate, EngElectrothermoflexomagnetophononicmagnonpolaritonicResonanceEnergy, LagmulLobachevskySpatialLinkageJoint, SensorSpringTorsionalDropRate
# ============================================================================

@dataclass
class FailLadCoupleHoneycombCoreCrushingRate:
    """``/FAIL/LAD_COUPLE_HONEYCOMB_CORE_CRUSHING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_HONEYCOMB_CORE_CRUSHING_RATE`` (M424): Ladevèze rate-dependent coupled sandwich honeycomb core cell-wall buckling, multi-axial crushing-shear collapse interaction, and dynamic compressive damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_chcc0: float = 0.0     # initial coupled honeycomb core crushing threshold stress sigma_chcc,0
    sigma_chccc: float = 1.0     # critical coupled honeycomb core crushing saturation stress sigma_chcc,c
    gamma_chcc: float = 0.0      # coupled honeycomb core crushing rate sensitivity factor gamma_chcc
    p_chcc: float = 1.0          # coupled honeycomb core crushing rate exponent p_chcc
    d_chcc_max: float = 0.999    # maximum allowable coupled honeycomb core crushing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M425 Suite: FailLadDynamicHoneycombCoreShearingRate, EngElectrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonicResonanceEnergy, LagmulPoincareSpatialLinkageJoint, SensorSpringBendingDropRate
# ============================================================================

@dataclass
class FailLadDynamicHoneycombCoreShearingRate:
    """``/FAIL/LAD_DYNAMIC_HONEYCOMB_CORE_SHEARING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_HONEYCOMB_CORE_SHEARING_RATE`` (M425): Ladevèze rate-dependent dynamic sandwich honeycomb core shear cracking, cell-wall shear buckling, and transverse shear damage accumulation model."""
    mat_id: int = 0
    title: str = ""
    sigma_dhcs0: float = 0.0     # initial dynamic honeycomb core shearing threshold stress sigma_dhcs,0
    sigma_dhcsc: float = 1.0     # critical dynamic honeycomb core shearing saturation stress sigma_dhcs,c
    gamma_dhcs: float = 0.0      # dynamic honeycomb core shearing rate sensitivity factor gamma_dhcs
    p_dhcs: float = 1.0          # dynamic honeycomb core shearing rate exponent p_dhcs
    d_dhcs_max: float = 0.999    # maximum allowable dynamic honeycomb core shearing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M426 Suite: FailLadTransverseHoneycombCoreShearingRate, EngElectrothermoflexomagnetoplasmonicmagnonicphononicpolaritonicResonanceEnergy, LagmulBeltramiSpatialLinkageJoint, SensorSpringTotalAngularDropRate
# ============================================================================

@dataclass
class FailLadTransverseHoneycombCoreShearingRate:
    """``/FAIL/LAD_TRANSVERSE_HONEYCOMB_CORE_SHEARING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_HONEYCOMB_CORE_SHEARING_RATE`` (M426): Ladevèze rate-dependent transverse sandwich honeycomb core shear cracking, off-axis cell-wall shear buckling, and transverse shear damage accumulation model."""
    mat_id: int = 0
    title: str = ""
    sigma_thcs0: float = 0.0     # initial transverse honeycomb core shearing threshold stress sigma_thcs,0
    sigma_thcsc: float = 1.0     # critical transverse honeycomb core shearing saturation stress sigma_thcs,c
    gamma_thcs: float = 0.0      # transverse honeycomb core shearing rate sensitivity factor gamma_thcs
    p_thcs: float = 1.0          # transverse honeycomb core shearing rate exponent p_thcs
    d_thcs_max: float = 0.999    # maximum allowable transverse honeycomb core shearing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference

# ============================================================================
# M427 Suite: FailLadCoupleHoneycombCoreShearingRate, EngElectrothermoflexomagnetoplasmonicphononicmagnonpolaritonicResonanceEnergy, LagmulCliffordSpatialLinkageJoint, SensorSpringTorsionalRateOfChange
# ============================================================================

@dataclass
class FailLadCoupleHoneycombCoreShearingRate:
    """``/FAIL/LAD_COUPLE_HONEYCOMB_CORE_SHEARING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_HONEYCOMB_CORE_SHEARING_RATE`` (M427): Ladevèze rate-dependent coupled sandwich honeycomb core shear cracking, multi-axial shearing-crushing collapse interaction, and dynamic shear damage accumulation model."""
    mat_id: int = 0
    title: str = ""
    sigma_chcs0: float = 0.0     # initial coupled honeycomb core shearing threshold stress sigma_chcs,0
    sigma_chcsc: float = 1.0     # critical coupled honeycomb core shearing saturation stress sigma_chcs,c
    gamma_chcs: float = 0.0      # coupled honeycomb core shearing rate sensitivity factor gamma_chcs
    p_chcs: float = 1.0          # coupled honeycomb core shearing rate exponent p_chcs
    d_chcs_max: float = 0.999    # maximum allowable coupled honeycomb core shearing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M428 Suite: FailLadDynamicFacesheetCoreDebondingRate, EngElectrothermoflexomagnetoexcitonicphononicmagnonpolaritonicResonanceEnergy, LagmulConformalSpatialLinkageJoint, SensorSpringBendingRateOfChange
# ============================================================================

@dataclass
class FailLadDynamicFacesheetCoreDebondingRate:
    """``/FAIL/LAD_DYNAMIC_FACESHEET_CORE_DEBONDING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_FACESHEET_CORE_DEBONDING_RATE`` (M428): Ladevèze rate-dependent dynamic sandwich facesheet-core interfacial debonding, dynamic peel/shear adhesive fracture, and progressive core delamination damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_dfcd0: float = 0.0     # initial dynamic facesheet-core debonding threshold stress sigma_dfcd,0
    sigma_dfcdc: float = 1.0     # critical dynamic facesheet-core debonding saturation stress sigma_dfcd,c
    gamma_dfcd: float = 0.0      # dynamic facesheet-core debonding rate sensitivity factor gamma_dfcd
    p_dfcd: float = 1.0          # dynamic facesheet-core debonding rate exponent p_dfcd
    d_dfcd_max: float = 0.999    # maximum allowable dynamic facesheet-core debonding damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M429 Suite: FailLadTransverseFacesheetCoreDebondingRate, EngElectrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonicResonanceEnergy, LagmulProjectiveSpatialLinkageJoint, SensorSpringTotalAngularRateOfChange
# ============================================================================

@dataclass
class FailLadTransverseFacesheetCoreDebondingRate:
    """``/FAIL/LAD_TRANSVERSE_FACESHEET_CORE_DEBONDING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_FACESHEET_CORE_DEBONDING_RATE`` (M429): Ladevèze rate-dependent transverse sandwich facesheet-core interfacial debonding, dynamic peel/shear adhesive fracture, and progressive core delamination damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_tfcd0: float = 0.0     # initial transverse facesheet-core debonding threshold stress sigma_tfcd,0
    sigma_tfcdc: float = 1.0     # critical transverse facesheet-core debonding saturation stress sigma_tfcd,c
    gamma_tfcd: float = 0.0      # transverse facesheet-core debonding rate sensitivity factor gamma_tfcd
    p_tfcd: float = 1.0          # transverse facesheet-core debonding rate exponent p_tfcd
    d_tfcd_max: float = 0.999    # maximum allowable transverse facesheet-core debonding damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M430 Suite: FailLadDynamicCoreDelaminationRate, EngElectrothermoflexomagnetoexcitonicmagnonicpolaritonicResonanceEnergy, LagmulSymplecticSpatialLinkageJoint, SensorSpringAxialSnapRate
# ============================================================================

@dataclass
class FailLadDynamicCoreDelaminationRate:
    """``/FAIL/LAD_DYNAMIC_CORE_DELAMINATION_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_CORE_DELAMINATION_RATE`` (M430): Ladevèze rate-dependent dynamic sandwich core delamination, dynamic peel/shear adhesive fracture, and progressive core delamination damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_dcdl0: float = 0.0     # initial dynamic core delamination threshold stress sigma_dcdl,0
    sigma_dcdlc: float = 1.0     # critical dynamic core delamination saturation stress sigma_dcdl,c
    gamma_dcdl: float = 0.0      # dynamic core delamination rate sensitivity factor gamma_dcdl
    p_dcdl: float = 1.0          # dynamic core delamination rate exponent p_dcdl
    d_dcdl_max: float = 0.999    # maximum allowable dynamic core delamination damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M431 Suite: FailLadTransverseCoreDelaminationRate, EngElectrothermoflexomagnetoplasmonicmagnonicpolaritonicResonanceEnergy, LagmulContactSpatialLinkageJoint, SensorSpringTransverseSnapRate
# ============================================================================

@dataclass
class FailLadTransverseCoreDelaminationRate:
    """``/FAIL/LAD_TRANSVERSE_CORE_DELAMINATION_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_CORE_DELAMINATION_RATE`` (M431): Ladevèze rate-dependent transverse sandwich core delamination, dynamic peel/shear adhesive fracture, and progressive core delamination damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_tcdl0: float = 0.0     # initial transverse core delamination threshold stress sigma_tcdl,0
    sigma_tcdlc: float = 1.0     # critical transverse core delamination saturation stress sigma_tcdl,c
    gamma_tcdl: float = 0.0      # transverse core delamination rate sensitivity factor gamma_tcdl
    p_tcdl: float = 1.0          # transverse core delamination rate exponent p_tcdl
    d_tcdl_max: float = 0.999    # maximum allowable transverse core delamination damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference

# ============================================================================
# M432 Suite: FailLadCoupleCoreDelaminationRate, EngElectrothermoflexomagnetophononicmagnonicpolaritonicResonanceEnergy, LagmulAlgebraicSpatialLinkageJoint, SensorSpringTotalSnapRate
# ============================================================================

@dataclass
class FailLadCoupleCoreDelaminationRate:
    """``/FAIL/LAD_COUPLE_CORE_DELAMINATION_RATE`` or ``/FAIL/LADEVEZE_COUPLED_CORE_DELAMINATION_RATE`` (M432): Ladevèze rate-dependent coupled dynamic-transverse sandwich core delamination, mixed-mode dynamic peel/shear adhesive fracture, and progressive core delamination damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_ccdl0: float = 0.0     # initial coupled core delamination threshold stress sigma_ccdl,0
    sigma_ccdlc: float = 1.0     # critical coupled core delamination saturation stress sigma_ccdl,c
    gamma_ccdl: float = 0.0      # coupled core delamination rate sensitivity factor gamma_ccdl
    p_ccdl: float = 1.0          # coupled core delamination rate exponent p_ccdl
    d_ccdl_max: float = 0.999    # maximum allowable coupled core delamination damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference

# ============================================================================
# M433 Suite: FailLadDynamicCoreTearingRate, EngElectrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy, LagmulDifferentialSpatialLinkageJoint, SensorSpringTorsionalSnapRate
# ============================================================================

@dataclass
class FailLadDynamicCoreTearingRate:
    """``/FAIL/LAD_DYNAMIC_CORE_TEARING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_CORE_TEARING_RATE`` (M433): Ladevèze rate-dependent dynamic sandwich core cell-wall tearing, dynamic tensile/shear core rupture, and progressive core tearing damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_dctr0: float = 0.0     # initial dynamic core tearing threshold stress sigma_dctr,0
    sigma_dctrc: float = 1.0     # critical dynamic core tearing saturation stress sigma_dctr,c
    gamma_dctr: float = 0.0      # dynamic core tearing rate sensitivity factor gamma_dctr
    p_dctr: float = 1.0          # dynamic core tearing rate exponent p_dctr
    d_dctr_max: float = 0.999    # maximum allowable dynamic core tearing damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference


# ============================================================================
# M433 Suite: FailLadDynamicCoreDelaminationCrackingRate, EngElectrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy, LagmulTopologicalSpatialLinkageJoint, SensorSpringTorsionalSnapRate
# ============================================================================

@dataclass
class FailLadDynamicCoreDelaminationCrackingRate:
    """``/FAIL/LAD_DYNAMIC_CORE_DELAMINATION_CRACKING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_CORE_DELAMINATION_CRACKING_RATE`` (M433): Ladevèze rate-dependent dynamic sandwich core delamination cracking, dynamic cell-wall fracture, and progressive core delamination-cracking damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_dcdlcr0: float = 0.0     # initial dynamic core delamination cracking threshold stress sigma_dcdlcr,0
    sigma_dcdlcrc: float = 1.0     # critical dynamic core delamination cracking saturation stress sigma_dcdlcr,c
    gamma_dcdlcr: float = 0.0      # dynamic core delamination cracking rate sensitivity factor gamma_dcdlcr
    p_dcdlcr: float = 1.0          # dynamic core delamination cracking rate exponent p_dcdlcr
    d_dcdlcr_max: float = 0.999    # maximum allowable dynamic core delamination cracking damage index
    ifail_sh: int = 1              # shell element deletion flag
    ifail_so: int = 1              # solid element deletion flag
    fail_id: int = 0               # failure model ID reference

    @property
    def sigma_dctr0(self) -> float:
        return self.sigma_dcdlcr0

    @sigma_dctr0.setter
    def sigma_dctr0(self, val: float) -> None:
        self.sigma_dcdlcr0 = val

    @property
    def sigma_dctrc(self) -> float:
        return self.sigma_dcdlcrc

    @sigma_dctrc.setter
    def sigma_dctrc(self, val: float) -> None:
        self.sigma_dcdlcrc = val

    @property
    def gamma_dctr(self) -> float:
        return self.gamma_dcdlcr

    @gamma_dctr.setter
    def gamma_dctr(self, val: float) -> None:
        self.gamma_dcdlcr = val

    @property
    def p_dctr(self) -> float:
        return self.p_dcdlcr

    @p_dctr.setter
    def p_dctr(self, val: float) -> None:
        self.p_dcdlcr = val

    @property
    def d_dctr_max(self) -> float:
        return self.d_dcdlcr_max

    @d_dctr_max.setter
    def d_dctr_max(self, val: float) -> None:
        self.d_dcdlcr_max = val


@dataclass
class FailLadTransverseCoreDelaminationCrackingRate:
    """``/FAIL/LAD_TRANSVERSE_CORE_DELAMINATION_CRACKING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_CORE_DELAMINATION_CRACKING_RATE`` (M434): Ladevèze rate-dependent transverse sandwich core delamination cracking, transverse cell-wall fracture/tearing, and progressive core delamination-cracking damage model."""
    mat_id: int = 0
    title: str = ""
    sigma_tcdlcr0: float = 0.0     # initial transverse core delamination cracking threshold stress sigma_tcdlcr,0
    sigma_tcdlcrc: float = 1.0     # critical transverse core delamination cracking saturation stress sigma_tcdlcr,c
    gamma_tcdlcr: float = 0.0      # transverse core delamination cracking rate sensitivity factor gamma_tcdlcr
    p_tcdlcr: float = 1.0          # transverse core delamination cracking rate exponent p_tcdlcr
    d_tcdlcr_max: float = 0.999    # maximum allowable transverse core delamination cracking damage index
    ifail_sh: int = 1              # shell element deletion flag
    ifail_so: int = 1              # solid element deletion flag
    fail_id: int = 0               # failure model ID reference

    @property
    def sigma_tctr0(self) -> float:
        return self.sigma_tcdlcr0

    @sigma_tctr0.setter
    def sigma_tctr0(self, val: float) -> None:
        self.sigma_tcdlcr0 = val

    @property
    def sigma_tctrc(self) -> float:
        return self.sigma_tcdlcrc

    @sigma_tctrc.setter
    def sigma_tctrc(self, val: float) -> None:
        self.sigma_tcdlcrc = val

    @property
    def gamma_tctr(self) -> float:
        return self.gamma_tcdlcr

    @gamma_tctr.setter
    def gamma_tctr(self, val: float) -> None:
        self.gamma_tcdlcr = val

    @property
    def p_tctr(self) -> float:
        return self.p_tcdlcr

    @p_tctr.setter
    def p_tctr(self, val: float) -> None:
        self.p_tcdlcr = val

    @property
    def d_tctr_max(self) -> float:
        return self.d_tcdlcr_max

    @d_tctr_max.setter
    def d_tctr_max(self, val: float) -> None:
        self.d_tcdlcr_max = val


# ============================================================================
# M435 Suite: FailLadCoupledCoreDelaminationCrackingRate, EngElectrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonicResonanceEnergy, LagmulCohomologicalSpatialLinkageJoint
# ============================================================================

@dataclass
class FailLadCoupledCoreDelaminationCrackingRate:
    """``/FAIL/LAD_COUPLED_CORE_DELAMINATION_CRACKING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_CORE_DELAMINATION_CRACKING_RATE`` (M435): Ladevèze rate-dependent coupled sandwich core delamination cracking, cell-wall tearing and multi-axial core failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_ccdlcr0: float = 0.0     # initial coupled core delamination cracking threshold stress sigma_ccdlcr,0
    sigma_ccdlcrc: float = 1.0     # critical coupled core delamination cracking saturation stress sigma_ccdlcr,c
    gamma_ccdlcr: float = 0.0      # coupled core delamination cracking rate sensitivity factor gamma_ccdlcr
    p_ccdlcr: float = 1.0          # coupled core delamination cracking rate exponent p_ccdlcr
    d_ccdlcr_max: float = 0.999    # maximum allowable coupled core delamination cracking damage index
    ifail_sh: int = 1              # shell element deletion flag
    ifail_so: int = 1              # solid element deletion flag
    fail_id: int = 0               # failure model ID reference

    @property
    def sigma_cctr0(self) -> float:
        return self.sigma_ccdlcr0

    @sigma_cctr0.setter
    def sigma_cctr0(self, val: float) -> None:
        self.sigma_ccdlcr0 = val

    @property
    def sigma_cctrc(self) -> float:
        return self.sigma_ccdlcrc

    @sigma_cctrc.setter
    def sigma_cctrc(self, val: float) -> None:
        self.sigma_ccdlcrc = val

    @property
    def gamma_cctr(self) -> float:
        return self.gamma_ccdlcr

    @gamma_cctr.setter
    def gamma_cctr(self, val: float) -> None:
        self.gamma_ccdlcr = val

    @property
    def p_cctr(self) -> float:
        return self.p_ccdlcr

    @p_cctr.setter
    def p_cctr(self, val: float) -> None:
        self.p_ccdlcr = val

    @property
    def d_cctr_max(self) -> float:
        return self.d_ccdlcr_max

    @d_cctr_max.setter
    def d_cctr_max(self, val: float) -> None:
        self.d_ccdlcr_max = val


FailLadCoupleCoreDelaminationCrackingRate = FailLadCoupledCoreDelaminationCrackingRate


# ============================================================================
# M436 Suite: FailLadDynamicCoreMicrocrackingRate, EngElectrothermoflexomagnetophononicpolaritonicResonanceEnergy, LagmulSpinorialSpatialLinkageJoint
# ============================================================================

@dataclass
class FailLadDynamicCoreMicrocrackingRate:
    """``/FAIL/LAD_DYNAMIC_CORE_MICROCRACKING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_CORE_MICROCRACKING_RATE`` (M436): Ladevèze rate-dependent dynamic sandwich core microcracking, microdamage kinetics and multi-axial core failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dcmcr0: float = 0.0      # initial dynamic core microcracking threshold stress sigma_dcmcr,0
    sigma_dcmcrc: float = 1.0      # critical dynamic core microcracking saturation stress sigma_dcmcr,c
    gamma_dcmcr: float = 0.0       # dynamic core microcracking rate sensitivity factor gamma_dcmcr
    p_dcmcr: float = 1.0           # dynamic core microcracking rate exponent p_dcmcr
    d_dcmcr_max: float = 0.999     # maximum allowable dynamic core microcracking damage index
    ifail_sh: int = 1              # shell element deletion flag
    ifail_so: int = 1              # solid element deletion flag
    fail_id: int = 0               # failure model ID reference

    @property
    def sigma_dcmd0(self) -> float:
        return self.sigma_dcmcr0

    @sigma_dcmd0.setter
    def sigma_dcmd0(self, val: float) -> None:
        self.sigma_dcmcr0 = val

    @property
    def sigma_dcmdc(self) -> float:
        return self.sigma_dcmcrc

    @sigma_dcmdc.setter
    def sigma_dcmdc(self, val: float) -> None:
        self.sigma_dcmcrc = val

    @property
    def gamma_dcmd(self) -> float:
        return self.gamma_dcmcr

    @gamma_dcmd.setter
    def gamma_dcmd(self, val: float) -> None:
        self.gamma_dcmcr = val

    @property
    def p_dcmd(self) -> float:
        return self.p_dcmcr

    @p_dcmd.setter
    def p_dcmd(self, val: float) -> None:
        self.p_dcmcr = val

    @property
    def d_dcmd_max(self) -> float:
        return self.d_dcmcr_max

    @d_dcmd_max.setter
    def d_dcmd_max(self, val: float) -> None:
        self.d_dcmcr_max = val


FailLadDynamicCoreMicrocrackRate = FailLadDynamicCoreMicrocrackingRate


# ============================================================================
# M438 Suite: FailLadTransverseCoreMicrocrackingRate, EngElectrothermoflexomagnetoexcitonicpolaritonicResonanceEnergy, LagmulSymplecticSpinorSpatialLinkageJoint
# ============================================================================

@dataclass
class FailLadTransverseCoreMicrocrackingRate:
    """``/FAIL/LAD_TRANSVERSE_CORE_MICROCRACKING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_CORE_MICROCRACKING_RATE`` (M438): Ladevèze rate-dependent transverse sandwich core microcracking, transverse microdamage kinetics and multi-axial core failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tcmcr0: float = 0.0      # initial transverse core microcracking threshold stress sigma_tcmcr,0
    sigma_tcmcrc: float = 1.0      # critical transverse core microcracking saturation stress sigma_tcmcr,c
    gamma_tcmcr: float = 0.0       # transverse core microcracking rate sensitivity factor gamma_tcmcr
    p_tcmcr: float = 1.0           # transverse core microcracking rate exponent p_tcmcr
    d_tcmcr_max: float = 0.999     # maximum allowable transverse core microcracking damage index
    ifail_sh: int = 1              # shell element deletion flag
    ifail_so: int = 1              # solid element deletion flag
    fail_id: int = 0               # failure model ID reference

    @property
    def sigma_tcmd0(self) -> float:
        return self.sigma_tcmcr0

    @sigma_tcmd0.setter
    def sigma_tcmd0(self, val: float) -> None:
        self.sigma_tcmcr0 = val

    @property
    def sigma_tcmdc(self) -> float:
        return self.sigma_tcmcrc

    @sigma_tcmdc.setter
    def sigma_tcmdc(self, val: float) -> None:
        self.sigma_tcmcrc = val

    @property
    def gamma_tcmd(self) -> float:
        return self.gamma_tcmcr

    @gamma_tcmd.setter
    def gamma_tcmd(self, val: float) -> None:
        self.gamma_tcmcr = val

    @property
    def p_tcmd(self) -> float:
        return self.p_tcmcr

    @p_tcmd.setter
    def p_tcmd(self, val: float) -> None:
        self.p_tcmcr = val

    @property
    def d_tcmd_max(self) -> float:
        return self.d_tcmcr_max

    @d_tcmd_max.setter
    def d_tcmd_max(self, val: float) -> None:
        self.d_tcmcr_max = val


FailLadTransverseCoreMicrocrackRate = FailLadTransverseCoreMicrocrackingRate


# ============================================================================
# M439 Suite: FailLadCoupledCoreMicrocrackingRate, EngElectrothermoflexomagnetomagnonicpolaritonicResonanceEnergy, LagmulContactSpinorSpatialLinkageJoint
# ============================================================================

@dataclass
class FailLadCoupledCoreMicrocrackingRate:
    """``/FAIL/LAD_COUPLED_CORE_MICROCRACKING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_CORE_MICROCRACKING_RATE`` (M439): Ladevèze rate-dependent coupled sandwich core microcracking, cell-wall fracture and multi-axial core failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_ccmcr0: float = 0.0      # initial coupled core microcracking threshold stress sigma_ccmcr,0
    sigma_ccmcrc: float = 1.0      # critical coupled core microcracking saturation stress sigma_ccmcr,c
    gamma_ccmcr: float = 0.0       # coupled core microcracking rate sensitivity factor gamma_ccmcr
    p_ccmcr: float = 1.0           # coupled core microcracking rate exponent p_ccmcr
    d_ccmcr_max: float = 0.999     # maximum allowable coupled core microcracking damage index
    ifail_sh: int = 1              # shell element deletion flag
    ifail_so: int = 1              # solid element deletion flag
    fail_id: int = 0               # failure model ID reference

    @property
    def sigma_ccmd0(self) -> float:
        return self.sigma_ccmcr0

    @sigma_ccmd0.setter
    def sigma_ccmd0(self, val: float) -> None:
        self.sigma_ccmcr0 = val

    @property
    def sigma_ccmdc(self) -> float:
        return self.sigma_ccmcrc

    @sigma_ccmdc.setter
    def sigma_ccmdc(self, val: float) -> None:
        self.sigma_ccmcrc = val

    @property
    def gamma_ccmd(self) -> float:
        return self.gamma_ccmcr

    @gamma_ccmd.setter
    def gamma_ccmd(self, val: float) -> None:
        self.gamma_ccmcr = val

    @property
    def p_ccmd(self) -> float:
        return self.p_ccmcr

    @p_ccmd.setter
    def p_ccmd(self, val: float) -> None:
        self.p_ccmcr = val

    @property
    def d_ccmd_max(self) -> float:
        return self.d_ccmcr_max

    @d_ccmd_max.setter
    def d_ccmd_max(self, val: float) -> None:
        self.d_ccmcr_max = val


FailLadCoupleCoreMicrocrackingRate = FailLadCoupledCoreMicrocrackingRate
FailLadCoupledCoreMicrocrackRate = FailLadCoupledCoreMicrocrackingRate
FailLadCoupleCoreMicrocrackRate = FailLadCoupledCoreMicrocrackingRate
FailLadCoupledCoreMicrodamageRate = FailLadCoupledCoreMicrocrackingRate
FailLadCoupleCoreMicrodamageRate = FailLadCoupledCoreMicrocrackingRate


# ============================================================================
# M440 Suite: FailLadDynamicCoreMicrobucklingRate, EngElectrothermoflexomagnetoplasmonicpolaritonicResonanceEnergy, LagmulConformalSpinorSpatialLinkageJoint
# ============================================================================

@dataclass
class FailLadDynamicCoreMicrobucklingRate:
    """``/FAIL/LAD_DYNAMIC_CORE_MICROBUCKLING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_CORE_MICROBUCKLING_RATE`` (M440): Ladevèze rate-dependent dynamic sandwich core microbuckling, cell-wall kinking and multi-axial compressive failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dcmbr0: float = 0.0      # initial dynamic core microbuckling threshold stress sigma_dcmbr,0
    sigma_dcmbrc: float = 1.0      # critical dynamic core microbuckling saturation stress sigma_dcmbr,c
    gamma_dcmbr: float = 0.0       # dynamic core microbuckling rate sensitivity factor gamma_dcmbr
    p_dcmbr: float = 1.0           # dynamic core microbuckling rate exponent p_dcmbr
    d_dcmbr_max: float = 0.999     # maximum allowable dynamic core microbuckling damage index
    ifail_sh: int = 1              # shell element deletion flag
    ifail_so: int = 1              # solid element deletion flag
    fail_id: int = 0               # failure model ID reference

    @property
    def sigma_dcmbk0(self) -> float:
        return self.sigma_dcmbr0

    @sigma_dcmbk0.setter
    def sigma_dcmbk0(self, val: float) -> None:
        self.sigma_dcmbr0 = val

    @property
    def sigma_dcmbkc(self) -> float:
        return self.sigma_dcmbrc

    @sigma_dcmbkc.setter
    def sigma_dcmbkc(self, val: float) -> None:
        self.sigma_dcmbrc = val

    @property
    def gamma_dcmbk(self) -> float:
        return self.gamma_dcmbr

    @gamma_dcmbk.setter
    def gamma_dcmbk(self, val: float) -> None:
        self.gamma_dcmbr = val

    @property
    def p_dcmbk(self) -> float:
        return self.p_dcmbr

    @p_dcmbk.setter
    def p_dcmbk(self, val: float) -> None:
        self.p_dcmbr = val

    @property
    def d_dcmbk_max(self) -> float:
        return self.d_dcmbr_max

    @d_dcmbk_max.setter
    def d_dcmbk_max(self, val: float) -> None:
        self.d_dcmbr_max = val

    @property
    def sigma_dcmk0(self) -> float:
        return self.sigma_dcmbr0

    @sigma_dcmk0.setter
    def sigma_dcmk0(self, val: float) -> None:
        self.sigma_dcmbr0 = val

    @property
    def sigma_dcmkc(self) -> float:
        return self.sigma_dcmbrc

    @sigma_dcmkc.setter
    def sigma_dcmkc(self, val: float) -> None:
        self.sigma_dcmbrc = val

    @property
    def gamma_dcmk(self) -> float:
        return self.gamma_dcmbr

    @gamma_dcmk.setter
    def gamma_dcmk(self, val: float) -> None:
        self.gamma_dcmbr = val

    @property
    def p_dcmk(self) -> float:
        return self.p_dcmbr

    @p_dcmk.setter
    def p_dcmk(self, val: float) -> None:
        self.p_dcmbr = val

    @property
    def d_dcmk_max(self) -> float:
        return self.d_dcmbr_max

    @d_dcmk_max.setter
    def d_dcmk_max(self, val: float) -> None:
        self.d_dcmbr_max = val


FailLadDynamicCoreMicrobuckleRate = FailLadDynamicCoreMicrobucklingRate
FailLadDynamicCoreMicrokinkingRate = FailLadDynamicCoreMicrobucklingRate


# ============================================================================
# M441 Suite: FailLadTransverseCoreMicrobucklingRate, EngElectrothermoflexomagnetophononicplasmonicpolaritonicResonanceEnergy, LagmulProjectiveSpinorSpatialLinkageJoint
# ============================================================================

@dataclass
class FailLadTransverseCoreMicrobucklingRate:
    """``/FAIL/LAD_TRANSVERSE_CORE_MICROBUCKLING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_CORE_MICROBUCKLING_RATE`` (M441): Ladevèze rate-dependent transverse sandwich core microbuckling, cell-wall kinking and multi-axial compressive failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tcmbr0: float = 0.0      # initial transverse core microbuckling threshold stress sigma_tcmbr,0
    sigma_tcmbrc: float = 1.0      # critical transverse core microbuckling saturation stress sigma_tcmbr,c
    gamma_tcmbr: float = 0.0       # transverse core microbuckling rate sensitivity factor gamma_tcmbr
    p_tcmbr: float = 1.0           # transverse core microbuckling rate exponent p_tcmbr
    d_tcmbr_max: float = 0.999     # maximum allowable transverse core microbuckling damage index
    ifail_sh: int = 1              # shell element deletion flag
    ifail_so: int = 1              # solid element deletion flag
    fail_id: int = 0               # failure model ID reference

    @property
    def sigma_tcmbk0(self) -> float:
        return self.sigma_tcmbr0

    @sigma_tcmbk0.setter
    def sigma_tcmbk0(self, val: float) -> None:
        self.sigma_tcmbr0 = val

    @property
    def sigma_tcmbkc(self) -> float:
        return self.sigma_tcmbrc

    @sigma_tcmbkc.setter
    def sigma_tcmbkc(self, val: float) -> None:
        self.sigma_tcmbrc = val

    @property
    def gamma_tcmbk(self) -> float:
        return self.gamma_tcmbr

    @gamma_tcmbk.setter
    def gamma_tcmbk(self, val: float) -> None:
        self.gamma_tcmbr = val

    @property
    def p_tcmbk(self) -> float:
        return self.p_tcmbr

    @p_tcmbk.setter
    def p_tcmbk(self, val: float) -> None:
        self.p_tcmbr = val

    @property
    def d_tcmbk_max(self) -> float:
        return self.d_tcmbr_max

    @d_tcmbk_max.setter
    def d_tcmbk_max(self, val: float) -> None:
        self.d_tcmbr_max = val

    @property
    def sigma_tcmk0(self) -> float:
        return self.sigma_tcmbr0

    @sigma_tcmk0.setter
    def sigma_tcmk0(self, val: float) -> None:
        self.sigma_tcmbr0 = val

    @property
    def sigma_tcmkc(self) -> float:
        return self.sigma_tcmbrc

    @sigma_tcmkc.setter
    def sigma_tcmkc(self, val: float) -> None:
        self.sigma_tcmbrc = val

    @property
    def gamma_tcmk(self) -> float:
        return self.gamma_tcmbr

    @gamma_tcmk.setter
    def gamma_tcmk(self, val: float) -> None:
        self.gamma_tcmbr = val

    @property
    def p_tcmk(self) -> float:
        return self.p_tcmbr

    @p_tcmk.setter
    def p_tcmk(self, val: float) -> None:
        self.p_tcmbr = val

    @property
    def d_tcmk_max(self) -> float:
        return self.d_tcmbr_max

    @d_tcmk_max.setter
    def d_tcmk_max(self, val: float) -> None:
        self.d_tcmbr_max = val


FailLadTransverseCoreMicrobuckleRate = FailLadTransverseCoreMicrobucklingRate
FailLadTransverseCoreMicrokinkingRate = FailLadTransverseCoreMicrobucklingRate


# ============================================================================
# M442 Suite: FailLadCoupledCoreMicrobucklingRate, EngElectrothermoflexomagnetoexcitonicplasmonicpolaritonicResonanceEnergy, LagmulAlgebraicSpinorSpatialLinkageJoint
# ============================================================================

@dataclass
class FailLadCoupledCoreMicrobucklingRate:
    """``/FAIL/LAD_COUPLED_CORE_MICROBUCKLING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_CORE_MICROBUCKLING_RATE`` (M442): Ladevèze rate-dependent coupled sandwich core microbuckling, cell-wall kinking and multi-axial compressive failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_ccmbr0: float = 0.0      # initial coupled core microbuckling threshold stress sigma_ccmbr,0
    sigma_ccmbrc: float = 1.0      # critical coupled core microbuckling saturation stress sigma_ccmbr,c
    gamma_ccmbr: float = 0.0       # coupled core microbuckling rate sensitivity factor gamma_ccmbr
    p_ccmbr: float = 1.0           # coupled core microbuckling rate exponent p_ccmbr
    d_ccmbr_max: float = 0.999     # maximum allowable coupled core microbuckling damage index
    ifail_sh: int = 1              # shell element deletion flag
    ifail_so: int = 1              # solid element deletion flag
    fail_id: int = 0               # failure model ID reference

    @property
    def sigma_ccmbk0(self) -> float:
        return self.sigma_ccmbr0

    @sigma_ccmbk0.setter
    def sigma_ccmbk0(self, val: float) -> None:
        self.sigma_ccmbr0 = val

    @property
    def sigma_ccmbkc(self) -> float:
        return self.sigma_ccmbrc

    @sigma_ccmbkc.setter
    def sigma_ccmbkc(self, val: float) -> None:
        self.sigma_ccmbrc = val

    @property
    def gamma_ccmbk(self) -> float:
        return self.gamma_ccmbr

    @gamma_ccmbk.setter
    def gamma_ccmbk(self, val: float) -> None:
        self.gamma_ccmbr = val

    @property
    def p_ccmbk(self) -> float:
        return self.p_ccmbr

    @p_ccmbk.setter
    def p_ccmbk(self, val: float) -> None:
        self.p_ccmbr = val

    @property
    def d_ccmbk_max(self) -> float:
        return self.d_ccmbr_max

    @d_ccmbk_max.setter
    def d_ccmbk_max(self, val: float) -> None:
        self.d_ccmbr_max = val

    @property
    def sigma_ccmk0(self) -> float:
        return self.sigma_ccmbr0

    @sigma_ccmk0.setter
    def sigma_ccmk0(self, val: float) -> None:
        self.sigma_ccmbr0 = val

    @property
    def sigma_ccmkc(self) -> float:
        return self.sigma_ccmbrc

    @sigma_ccmkc.setter
    def sigma_ccmkc(self, val: float) -> None:
        self.sigma_ccmbrc = val

    @property
    def gamma_ccmk(self) -> float:
        return self.gamma_ccmbr

    @gamma_ccmk.setter
    def gamma_ccmk(self, val: float) -> None:
        self.gamma_ccmbr = val

    @property
    def p_ccmk(self) -> float:
        return self.p_ccmbr

    @p_ccmk.setter
    def p_ccmk(self, val: float) -> None:
        self.p_ccmbr = val

    @property
    def d_ccmk_max(self) -> float:
        return self.d_ccmbr_max

    @d_ccmk_max.setter
    def d_ccmk_max(self, val: float) -> None:
        self.d_ccmbr_max = val


FailLadCoupleCoreMicrobucklingRate = FailLadCoupledCoreMicrobucklingRate
FailLadCoupledCoreMicrobuckleRate = FailLadCoupledCoreMicrobucklingRate
FailLadCoupleCoreMicrobuckleRate = FailLadCoupledCoreMicrobucklingRate
FailLadCoupledCoreMicrokinkingRate = FailLadCoupledCoreMicrobucklingRate
FailLadCoupleCoreMicrokinkingRate = FailLadCoupledCoreMicrobucklingRate


# ============================================================================
# M443 Suite: FailLadDynamicCoreMicroyieldingRate, EngElectrothermoflexomagnetomagnonicplasmonicpolaritonicResonanceEnergy, LagmulTopologicalSpinorSpatialLinkageJoint
# ============================================================================

@dataclass
class FailLadDynamicCoreMicroyieldingRate:
    """``/FAIL/LAD_DYNAMIC_CORE_MICROYIELDING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_CORE_MICROYIELDING_RATE`` (M443): Ladevèze rate-dependent dynamic sandwich core microyielding, cell-wall microplasticity and multi-axial plastic flow model."""
    mat_id: int = 0
    title: str = ""
    sigma_dcmyr0: float = 0.0      # initial dynamic core microyielding threshold stress sigma_dcmyr,0
    sigma_dcmyrc: float = 1.0      # critical dynamic core microyielding saturation stress sigma_dcmyr,c
    gamma_dcmyr: float = 0.0       # dynamic core microyielding rate sensitivity factor gamma_dcmyr
    p_dcmyr: float = 1.0           # dynamic core microyielding rate exponent p_dcmyr
    d_dcmyr_max: float = 0.999     # maximum allowable dynamic core microyielding damage/plasticity index
    ifail_sh: int = 1              # shell element deletion flag
    ifail_so: int = 1              # solid element deletion flag
    fail_id: int = 0               # failure model ID reference

    @property
    def sigma_dcmy0(self) -> float:
        return self.sigma_dcmyr0

    @sigma_dcmy0.setter
    def sigma_dcmy0(self, val: float) -> None:
        self.sigma_dcmyr0 = val

    @property
    def sigma_dcmyc(self) -> float:
        return self.sigma_dcmyrc

    @sigma_dcmyc.setter
    def sigma_dcmyc(self, val: float) -> None:
        self.sigma_dcmyrc = val

    @property
    def gamma_dcmy(self) -> float:
        return self.gamma_dcmyr

    @gamma_dcmy.setter
    def gamma_dcmy(self, val: float) -> None:
        self.gamma_dcmyr = val

    @property
    def p_dcmy(self) -> float:
        return self.p_dcmyr

    @p_dcmy.setter
    def p_dcmy(self, val: float) -> None:
        self.p_dcmyr = val

    @property
    def d_dcmy_max(self) -> float:
        return self.d_dcmyr_max

    @d_dcmy_max.setter
    def d_dcmy_max(self, val: float) -> None:
        self.d_dcmyr_max = val

    @property
    def sigma_dcmpr0(self) -> float:
        return self.sigma_dcmyr0

    @sigma_dcmpr0.setter
    def sigma_dcmpr0(self, val: float) -> None:
        self.sigma_dcmyr0 = val

    @property
    def sigma_dcmprc(self) -> float:
        return self.sigma_dcmyrc

    @sigma_dcmprc.setter
    def sigma_dcmprc(self, val: float) -> None:
        self.sigma_dcmyrc = val

    @property
    def gamma_dcmpr(self) -> float:
        return self.gamma_dcmyr

    @gamma_dcmpr.setter
    def gamma_dcmpr(self, val: float) -> None:
        self.gamma_dcmyr = val

    @property
    def p_dcmpr(self) -> float:
        return self.p_dcmyr

    @p_dcmpr.setter
    def p_dcmpr(self, val: float) -> None:
        self.p_dcmyr = val

    @property
    def d_dcmpr_max(self) -> float:
        return self.d_dcmyr_max

    @d_dcmpr_max.setter
    def d_dcmpr_max(self, val: float) -> None:
        self.d_dcmyr_max = val

    @property
    def sigma_dcmfr0(self) -> float:
        return self.sigma_dcmyr0

    @sigma_dcmfr0.setter
    def sigma_dcmfr0(self, val: float) -> None:
        self.sigma_dcmyr0 = val

    @property
    def sigma_dcmfrc(self) -> float:
        return self.sigma_dcmyrc

    @sigma_dcmfrc.setter
    def sigma_dcmfrc(self, val: float) -> None:
        self.sigma_dcmyrc = val

    @property
    def gamma_dcmfr(self) -> float:
        return self.gamma_dcmyr

    @gamma_dcmfr.setter
    def gamma_dcmfr(self, val: float) -> None:
        self.gamma_dcmyr = val

    @property
    def p_dcmfr(self) -> float:
        return self.p_dcmyr

    @p_dcmfr.setter
    def p_dcmfr(self, val: float) -> None:
        self.p_dcmyr = val

    @property
    def d_dcmfr_max(self) -> float:
        return self.d_dcmyr_max

    @d_dcmfr_max.setter
    def d_dcmfr_max(self, val: float) -> None:
        self.d_dcmyr_max = val


FailLadDynamicCoreMicroyieldRate = FailLadDynamicCoreMicroyieldingRate
FailLadDynamicCoreMicroplasticityRate = FailLadDynamicCoreMicroyieldingRate
FailLadDynamicCoreMicroplasticRate = FailLadDynamicCoreMicroyieldingRate
FailLadDynamicCoreMicroflowRate = FailLadDynamicCoreMicroyieldingRate


# ============================================================================
# M444 Suite: FailLadTransverseCoreMicroyieldingRate, EngElectrothermoflexomagnetophononicexcitonicplasmonicpolaritonicResonanceEnergy, LagmulHomologicalSpinorSpatialLinkageJoint
# ============================================================================

@dataclass
class FailLadTransverseCoreMicroyieldingRate:
    """``/FAIL/LAD_TRANSVERSE_CORE_MICROYIELDING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_CORE_MICROYIELDING_RATE`` (M444): Ladevèze rate-dependent transverse sandwich core microyielding, cell-wall microplasticity and multi-axial plastic flow model."""
    mat_id: int = 0
    title: str = ""
    sigma_tcmyr0: float = 0.0      # initial transverse core microyielding threshold stress sigma_tcmyr,0
    sigma_tcmyrc: float = 1.0      # critical transverse core microyielding saturation stress sigma_tcmyr,c
    gamma_tcmyr: float = 0.0       # transverse core microyielding rate sensitivity factor gamma_tcmyr
    p_tcmyr: float = 1.0           # transverse core microyielding rate exponent p_tcmyr
    d_tcmyr_max: float = 0.999     # maximum allowable transverse core microyielding damage/plasticity index
    ifail_sh: int = 1              # shell element deletion flag
    ifail_so: int = 1              # solid element deletion flag
    fail_id: int = 0               # failure model ID reference

    @property
    def sigma_tcmy0(self) -> float:
        return self.sigma_tcmyr0

    @sigma_tcmy0.setter
    def sigma_tcmy0(self, val: float) -> None:
        self.sigma_tcmyr0 = val

    @property
    def sigma_tcmyc(self) -> float:
        return self.sigma_tcmyrc

    @sigma_tcmyc.setter
    def sigma_tcmyc(self, val: float) -> None:
        self.sigma_tcmyrc = val

    @property
    def gamma_tcmy(self) -> float:
        return self.gamma_tcmyr

    @gamma_tcmy.setter
    def gamma_tcmy(self, val: float) -> None:
        self.gamma_tcmyr = val

    @property
    def p_tcmy(self) -> float:
        return self.p_tcmyr

    @p_tcmy.setter
    def p_tcmy(self, val: float) -> None:
        self.p_tcmyr = val

    @property
    def d_tcmy_max(self) -> float:
        return self.d_tcmyr_max

    @d_tcmy_max.setter
    def d_tcmy_max(self, val: float) -> None:
        self.d_tcmyr_max = val

    @property
    def sigma_tcmpr0(self) -> float:
        return self.sigma_tcmyr0

    @sigma_tcmpr0.setter
    def sigma_tcmpr0(self, val: float) -> None:
        self.sigma_tcmyr0 = val

    @property
    def sigma_tcmprc(self) -> float:
        return self.sigma_tcmyrc

    @sigma_tcmprc.setter
    def sigma_tcmprc(self, val: float) -> None:
        self.sigma_tcmyrc = val

    @property
    def gamma_tcmpr(self) -> float:
        return self.gamma_tcmyr

    @gamma_tcmpr.setter
    def gamma_tcmpr(self, val: float) -> None:
        self.gamma_tcmyr = val

    @property
    def p_tcmpr(self) -> float:
        return self.p_tcmyr

    @p_tcmpr.setter
    def p_tcmpr(self, val: float) -> None:
        self.p_tcmyr = val

    @property
    def d_tcmpr_max(self) -> float:
        return self.d_tcmyr_max

    @d_tcmpr_max.setter
    def d_tcmpr_max(self, val: float) -> None:
        self.d_tcmyr_max = val

    @property
    def sigma_tcmfr0(self) -> float:
        return self.sigma_tcmyr0

    @sigma_tcmfr0.setter
    def sigma_tcmfr0(self, val: float) -> None:
        self.sigma_tcmyr0 = val

    @property
    def sigma_tcmfrc(self) -> float:
        return self.sigma_tcmyrc

    @sigma_tcmfrc.setter
    def sigma_tcmfrc(self, val: float) -> None:
        self.sigma_tcmyrc = val

    @property
    def gamma_tcmfr(self) -> float:
        return self.gamma_tcmyr

    @gamma_tcmfr.setter
    def gamma_tcmfr(self, val: float) -> None:
        self.gamma_tcmyr = val

    @property
    def p_tcmfr(self) -> float:
        return self.p_tcmyr

    @p_tcmfr.setter
    def p_tcmfr(self, val: float) -> None:
        self.p_tcmyr = val

    @property
    def d_tcmfr_max(self) -> float:
        return self.d_tcmyr_max

    @d_tcmfr_max.setter
    def d_tcmfr_max(self, val: float) -> None:
        self.d_tcmyr_max = val


FailLadTransverseCoreMicroyieldRate = FailLadTransverseCoreMicroyieldingRate
FailLadTransverseCoreMicroplasticityRate = FailLadTransverseCoreMicroyieldingRate
FailLadTransverseCoreMicroplasticRate = FailLadTransverseCoreMicroyieldingRate
FailLadTransverseCoreMicroflowRate = FailLadTransverseCoreMicroyieldingRate


# ============================================================================
# M445 Suite: FailLadCoupledCoreMicroyieldingRate, EngElectrothermoflexomagnetophononicmagnonicplasmonicpolaritonicResonanceEnergy, LagmulCohomologicalSpinorSpatialLinkageJoint
# ============================================================================

@dataclass
class FailLadCoupledCoreMicroyieldingRate:
    """``/FAIL/LAD_COUPLED_CORE_MICROYIELDING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_CORE_MICROYIELDING_RATE`` (M445): Ladevèze rate-dependent coupled sandwich core microyielding, cell-wall microplasticity and multi-axial plastic flow model."""
    mat_id: int = 0
    title: str = ""
    sigma_ccmyr0: float = 0.0      # initial coupled core microyielding threshold stress sigma_ccmyr,0
    sigma_ccmyrc: float = 1.0      # critical coupled core microyielding saturation stress sigma_ccmyr,c
    gamma_ccmyr: float = 0.0       # coupled core microyielding rate sensitivity factor gamma_ccmyr
    p_ccmyr: float = 1.0           # coupled core microyielding rate exponent p_ccmyr
    d_ccmyr_max: float = 0.999     # maximum allowable coupled core microyielding damage/plasticity index
    ifail_sh: int = 1              # shell element deletion flag
    ifail_so: int = 1              # solid element deletion flag
    fail_id: int = 0               # failure model ID reference

    @property
    def sigma_ccmy0(self) -> float:
        return self.sigma_ccmyr0

    @sigma_ccmy0.setter
    def sigma_ccmy0(self, val: float) -> None:
        self.sigma_ccmyr0 = val

    @property
    def sigma_ccmyc(self) -> float:
        return self.sigma_ccmyrc

    @sigma_ccmyc.setter
    def sigma_ccmyc(self, val: float) -> None:
        self.sigma_ccmyrc = val

    @property
    def gamma_ccmy(self) -> float:
        return self.gamma_ccmyr

    @gamma_ccmy.setter
    def gamma_ccmy(self, val: float) -> None:
        self.gamma_ccmyr = val

    @property
    def p_ccmy(self) -> float:
        return self.p_ccmyr

    @p_ccmy.setter
    def p_ccmy(self, val: float) -> None:
        self.p_ccmyr = val

    @property
    def d_ccmy_max(self) -> float:
        return self.d_ccmyr_max

    @d_ccmy_max.setter
    def d_ccmy_max(self, val: float) -> None:
        self.d_ccmyr_max = val

    @property
    def sigma_ccmpr0(self) -> float:
        return self.sigma_ccmyr0

    @sigma_ccmpr0.setter
    def sigma_ccmpr0(self, val: float) -> None:
        self.sigma_ccmyr0 = val

    @property
    def sigma_ccmprc(self) -> float:
        return self.sigma_ccmyrc

    @sigma_ccmprc.setter
    def sigma_ccmprc(self, val: float) -> None:
        self.sigma_ccmyrc = val

    @property
    def gamma_ccmpr(self) -> float:
        return self.gamma_ccmyr

    @gamma_ccmpr.setter
    def gamma_ccmpr(self, val: float) -> None:
        self.gamma_ccmyr = val

    @property
    def p_ccmpr(self) -> float:
        return self.p_ccmyr

    @p_ccmpr.setter
    def p_ccmpr(self, val: float) -> None:
        self.p_ccmyr = val

    @property
    def d_ccmpr_max(self) -> float:
        return self.d_ccmyr_max

    @d_ccmpr_max.setter
    def d_ccmpr_max(self, val: float) -> None:
        self.d_ccmyr_max = val

    @property
    def sigma_ccmfr0(self) -> float:
        return self.sigma_ccmyr0

    @sigma_ccmfr0.setter
    def sigma_ccmfr0(self, val: float) -> None:
        self.sigma_ccmyr0 = val

    @property
    def sigma_ccmfrc(self) -> float:
        return self.sigma_ccmyrc

    @sigma_ccmfrc.setter
    def sigma_ccmfrc(self, val: float) -> None:
        self.sigma_ccmyrc = val

    @property
    def gamma_ccmfr(self) -> float:
        return self.gamma_ccmyr

    @gamma_ccmfr.setter
    def gamma_ccmfr(self, val: float) -> None:
        self.gamma_ccmyr = val

    @property
    def p_ccmfr(self) -> float:
        return self.p_ccmyr

    @p_ccmfr.setter
    def p_ccmfr(self, val: float) -> None:
        self.p_ccmyr = val

    @property
    def d_ccmfr_max(self) -> float:
        return self.d_ccmyr_max

    @d_ccmfr_max.setter
    def d_ccmfr_max(self, val: float) -> None:
        self.d_ccmyr_max = val


FailLadCoupleCoreMicroyieldingRate = FailLadCoupledCoreMicroyieldingRate
FailLadCoupledCoreMicroyieldRate = FailLadCoupledCoreMicroyieldingRate
FailLadCoupleCoreMicroyieldRate = FailLadCoupledCoreMicroyieldingRate
FailLadCoupledCoreMicroplasticityRate = FailLadCoupledCoreMicroyieldingRate
FailLadCoupleCoreMicroplasticityRate = FailLadCoupledCoreMicroyieldingRate
FailLadCoupledCoreMicroplasticRate = FailLadCoupledCoreMicroyieldingRate
FailLadCoupleCoreMicroplasticRate = FailLadCoupledCoreMicroyieldingRate
FailLadCoupledCoreMicroflowRate = FailLadCoupledCoreMicroyieldingRate
FailLadCoupleCoreMicroflowRate = FailLadCoupledCoreMicroyieldingRate


# ============================================================================
# M446 Suite: FailLadDynamicDelaminationMicrocrackingRate, EngElectrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonicResonanceEnergy, LagmulSheafSpinorSpatialLinkageJoint
# ============================================================================

@dataclass
class FailLadDynamicDelaminationMicrocrackingRate:
    """``/FAIL/LAD_DYNAMIC_DELAMINATION_MICROCRACKING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_DELAMINATION_MICROCRACKING_RATE`` (M446): Ladevèze rate-dependent dynamic interlaminar delamination microcracking and matrix ply separation model."""
    mat_id: int = 0
    title: str = ""
    sigma_ddmcr0: float = 0.0     # initial dynamic delamination microcracking threshold stress sigma_ddmcr,0
    sigma_ddmcrc: float = 1.0     # critical dynamic delamination microcracking saturation stress sigma_ddmcr,c
    gamma_ddmcr: float = 0.0      # dynamic delamination microcracking rate sensitivity factor gamma_ddmcr
    p_ddmcr: float = 1.0          # dynamic delamination microcracking rate exponent p_ddmcr
    d_ddmcr_max: float = 0.999    # maximum allowable dynamic delamination microcracking damage index
    ifail_sh: int = 1             # shell element deletion flag
    ifail_so: int = 1             # solid element deletion flag
    fail_id: int = 0              # failure model ID reference

    @property
    def sigma_ddmc0(self) -> float:
        return self.sigma_ddmcr0

    @sigma_ddmc0.setter
    def sigma_ddmc0(self, val: float) -> None:
        self.sigma_ddmcr0 = val

    @property
    def sigma_ddmcc(self) -> float:
        return self.sigma_ddmcrc

    @sigma_ddmcc.setter
    def sigma_ddmcc(self, val: float) -> None:
        self.sigma_ddmcrc = val

    @property
    def gamma_ddmc(self) -> float:
        return self.gamma_ddmcr

    @gamma_ddmc.setter
    def gamma_ddmc(self, val: float) -> None:
        self.gamma_ddmcr = val

    @property
    def p_ddmc(self) -> float:
        return self.p_ddmcr

    @p_ddmc.setter
    def p_ddmc(self, val: float) -> None:
        self.p_ddmcr = val

    @property
    def d_ddmc_max(self) -> float:
        return self.d_ddmcr_max

    @d_ddmc_max.setter
    def d_ddmc_max(self, val: float) -> None:
        self.d_ddmcr_max = val

    @property
    def sigma_ddm0(self) -> float:
        return self.sigma_ddmcr0

    @sigma_ddm0.setter
    def sigma_ddm0(self, val: float) -> None:
        self.sigma_ddmcr0 = val

    @property
    def sigma_ddmc(self) -> float:
        return self.sigma_ddmcrc

    @sigma_ddmc.setter
    def sigma_ddmc(self, val: float) -> None:
        self.sigma_ddmcrc = val

    @property
    def gamma_ddm(self) -> float:
        return self.gamma_ddmcr

    @gamma_ddm.setter
    def gamma_ddm(self, val: float) -> None:
        self.gamma_ddmcr = val

    @property
    def p_ddm(self) -> float:
        return self.p_ddmcr

    @p_ddm.setter
    def p_ddm(self, val: float) -> None:
        self.p_ddmcr = val

    @property
    def d_ddm_max(self) -> float:
        return self.d_ddmcr_max

    @d_ddm_max.setter
    def d_ddm_max(self, val: float) -> None:
        self.d_ddmcr_max = val

    @property
    def sigma_dimcr0(self) -> float:
        return self.sigma_ddmcr0

    @sigma_dimcr0.setter
    def sigma_dimcr0(self, val: float) -> None:
        self.sigma_ddmcr0 = val

    @property
    def sigma_dimcrc(self) -> float:
        return self.sigma_ddmcrc

    @sigma_dimcrc.setter
    def sigma_dimcrc(self, val: float) -> None:
        self.sigma_ddmcrc = val

    @property
    def gamma_dimcr(self) -> float:
        return self.gamma_ddmcr

    @gamma_dimcr.setter
    def gamma_dimcr(self, val: float) -> None:
        self.gamma_ddmcr = val

    @property
    def p_dimcr(self) -> float:
        return self.p_ddmcr

    @p_dimcr.setter
    def p_dimcr(self, val: float) -> None:
        self.p_ddmcr = val

    @property
    def d_dimcr_max(self) -> float:
        return self.d_ddmcr_max

    @d_dimcr_max.setter
    def d_dimcr_max(self, val: float) -> None:
        self.d_ddmcr_max = val


FailLadDynamicDelaminationMicrocrackRate = FailLadDynamicDelaminationMicrocrackingRate
FailLadDynamicDelamMicrocrackingRate = FailLadDynamicDelaminationMicrocrackingRate
FailLadDynamicDelamMicrocrackRate = FailLadDynamicDelaminationMicrocrackingRate
FailLadDynamicInterlaminarMicrocrackingRate = FailLadDynamicDelaminationMicrocrackingRate
FailLadDynamicInterlaminarMicrocrackRate = FailLadDynamicDelaminationMicrocrackingRate


# ============================================================================
# M447 Suite: FailLadTransverseDelaminationMicrocrackingRate, EngElectrothermoflexomagnetochiralplasmonicpolaritonicResonanceEnergy, LagmulStackSpinorSpatialLinkageJoint
# ============================================================================

@dataclass
class FailLadTransverseDelaminationMicrocrackingRate:
    """``/FAIL/LAD_TRANSVERSE_DELAMINATION_MICROCRACKING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_DELAMINATION_MICROCRACKING_RATE`` (M447): Ladevèze rate-dependent transverse interlaminar delamination microcracking and matrix ply separation model."""
    mat_id: int = 0
    title: str = ""
    sigma_tdmcr0: float = 0.0     # initial transverse delamination microcracking threshold stress sigma_tdmcr,0
    sigma_tdmcrc: float = 1.0     # critical transverse delamination microcracking saturation stress sigma_tdmcr,c
    gamma_tdmcr: float = 0.0      # transverse delamination microcracking rate sensitivity factor gamma_tdmcr
    p_tdmcr: float = 1.0          # transverse delamination microcracking rate exponent p_tdmcr
    d_tdmcr_max: float = 0.999    # maximum allowable transverse delamination microcracking damage index
    ifail_sh: int = 1             # shell element deletion flag
    ifail_so: int = 1             # solid element deletion flag
    fail_id: int = 0              # failure model ID reference

    @property
    def sigma_tdmc0(self) -> float:
        return self.sigma_tdmcr0

    @sigma_tdmc0.setter
    def sigma_tdmc0(self, val: float) -> None:
        self.sigma_tdmcr0 = val

    @property
    def sigma_tdmcc(self) -> float:
        return self.sigma_tdmcrc

    @sigma_tdmcc.setter
    def sigma_tdmcc(self, val: float) -> None:
        self.sigma_tdmcrc = val

    @property
    def gamma_tdmc(self) -> float:
        return self.gamma_tdmcr

    @gamma_tdmc.setter
    def gamma_tdmc(self, val: float) -> None:
        self.gamma_tdmcr = val

    @property
    def p_tdmc(self) -> float:
        return self.p_tdmcr

    @p_tdmc.setter
    def p_tdmc(self, val: float) -> None:
        self.p_tdmcr = val

    @property
    def d_tdmc_max(self) -> float:
        return self.d_tdmcr_max

    @d_tdmc_max.setter
    def d_tdmc_max(self, val: float) -> None:
        self.d_tdmcr_max = val

    @property
    def sigma_tdm0(self) -> float:
        return self.sigma_tdmcr0

    @sigma_tdm0.setter
    def sigma_tdm0(self, val: float) -> None:
        self.sigma_tdmcr0 = val

    @property
    def sigma_tdmc(self) -> float:
        return self.sigma_tdmcrc

    @sigma_tdmc.setter
    def sigma_tdmc(self, val: float) -> None:
        self.sigma_tdmcrc = val

    @property
    def gamma_tdm(self) -> float:
        return self.gamma_tdmcr

    @gamma_tdm.setter
    def gamma_tdm(self, val: float) -> None:
        self.gamma_tdmcr = val

    @property
    def p_tdm(self) -> float:
        return self.p_tdmcr

    @p_tdm.setter
    def p_tdm(self, val: float) -> None:
        self.p_tdmcr = val

    @property
    def d_tdm_max(self) -> float:
        return self.d_tdmcr_max

    @d_tdm_max.setter
    def d_tdm_max(self, val: float) -> None:
        self.d_tdmcr_max = val

    @property
    def sigma_timcr0(self) -> float:
        return self.sigma_tdmcr0

    @sigma_timcr0.setter
    def sigma_timcr0(self, val: float) -> None:
        self.sigma_tdmcr0 = val

    @property
    def sigma_timcrc(self) -> float:
        return self.sigma_tdmcrc

    @sigma_timcrc.setter
    def sigma_timcrc(self, val: float) -> None:
        self.sigma_tdmcrc = val

    @property
    def gamma_timcr(self) -> float:
        return self.gamma_tdmcr

    @gamma_timcr.setter
    def gamma_timcr(self, val: float) -> None:
        self.gamma_tdmcr = val

    @property
    def p_timcr(self) -> float:
        return self.p_tdmcr

    @p_timcr.setter
    def p_timcr(self, val: float) -> None:
        self.p_tdmcr = val

    @property
    def d_timcr_max(self) -> float:
        return self.d_tdmcr_max

    @d_timcr_max.setter
    def d_timcr_max(self, val: float) -> None:
        self.d_tdmcr_max = val


FailLadTransverseDelaminationMicrocrackRate = FailLadTransverseDelaminationMicrocrackingRate
FailLadTransverseDelamMicrocrackingRate = FailLadTransverseDelaminationMicrocrackingRate
FailLadTransverseDelamMicrocrackRate = FailLadTransverseDelaminationMicrocrackingRate
FailLadTransverseInterlaminarMicrocrackingRate = FailLadTransverseDelaminationMicrocrackingRate
FailLadTransverseInterlaminarMicrocrackRate = FailLadTransverseDelaminationMicrocrackingRate


# ============================================================================
# M448 Suite: FailLadCoupledDelaminationMicrocrackingRate, EngElectrothermoflexomagnetochiralphononicplasmonicpolaritonicResonanceEnergy, LagmulToposSpinorSpatialLinkageJoint
# ============================================================================

@dataclass
class FailLadCoupledDelaminationMicrocrackingRate:
    """``/FAIL/LAD_COUPLED_DELAMINATION_MICROCRACKING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_DELAMINATION_MICROCRACKING_RATE`` (M448): Ladevèze rate-dependent coupled interlaminar delamination microcracking and matrix ply separation model."""
    mat_id: int = 0
    title: str = ""
    sigma_cdmcr0: float = 0.0     # initial coupled delamination microcracking threshold stress sigma_cdmcr,0
    sigma_cdmcrc: float = 1.0     # critical coupled delamination microcracking saturation stress sigma_cdmcr,c
    gamma_cdmcr: float = 0.0      # coupled delamination microcracking rate sensitivity factor gamma_cdmcr
    p_cdmcr: float = 1.0          # coupled delamination microcracking rate exponent p_cdmcr
    d_cdmcr_max: float = 0.999    # maximum allowable coupled delamination microcracking damage index
    ifail_sh: int = 1             # shell element deletion flag
    ifail_so: int = 1             # solid element deletion flag
    fail_id: int = 0              # failure model ID reference

    @property
    def sigma_cdmc0(self) -> float:
        return self.sigma_cdmcr0

    @sigma_cdmc0.setter
    def sigma_cdmc0(self, val: float) -> None:
        self.sigma_cdmcr0 = val

    @property
    def sigma_cdmcc(self) -> float:
        return self.sigma_cdmcrc

    @sigma_cdmcc.setter
    def sigma_cdmcc(self, val: float) -> None:
        self.sigma_cdmcrc = val

    @property
    def gamma_cdmc(self) -> float:
        return self.gamma_cdmcr

    @gamma_cdmc.setter
    def gamma_cdmc(self, val: float) -> None:
        self.gamma_cdmcr = val

    @property
    def p_cdmc(self) -> float:
        return self.p_cdmcr

    @p_cdmc.setter
    def p_cdmc(self, val: float) -> None:
        self.p_cdmcr = val

    @property
    def d_cdmc_max(self) -> float:
        return self.d_cdmcr_max

    @d_cdmc_max.setter
    def d_cdmc_max(self, val: float) -> None:
        self.d_cdmcr_max = val

    @property
    def sigma_cdm0(self) -> float:
        return self.sigma_cdmcr0

    @sigma_cdm0.setter
    def sigma_cdm0(self, val: float) -> None:
        self.sigma_cdmcr0 = val

    @property
    def sigma_cdmc(self) -> float:
        return self.sigma_cdmcrc

    @sigma_cdmc.setter
    def sigma_cdmc(self, val: float) -> None:
        self.sigma_cdmcrc = val

    @property
    def gamma_cdm(self) -> float:
        return self.gamma_cdmcr

    @gamma_cdm.setter
    def gamma_cdm(self, val: float) -> None:
        self.gamma_cdmcr = val

    @property
    def p_cdm(self) -> float:
        return self.p_cdmcr

    @p_cdm.setter
    def p_cdm(self, val: float) -> None:
        self.p_cdmcr = val

    @property
    def d_cdm_max(self) -> float:
        return self.d_cdmcr_max

    @d_cdm_max.setter
    def d_cdm_max(self, val: float) -> None:
        self.d_cdmcr_max = val

    @property
    def sigma_cimcr0(self) -> float:
        return self.sigma_cdmcr0

    @sigma_cimcr0.setter
    def sigma_cimcr0(self, val: float) -> None:
        self.sigma_cdmcr0 = val

    @property
    def sigma_cimcrc(self) -> float:
        return self.sigma_cdmcrc

    @sigma_cimcrc.setter
    def sigma_cimcrc(self, val: float) -> None:
        self.sigma_cdmcrc = val

    @property
    def gamma_cimcr(self) -> float:
        return self.gamma_cdmcr

    @gamma_cimcr.setter
    def gamma_cimcr(self, val: float) -> None:
        self.gamma_cdmcr = val

    @property
    def p_cimcr(self) -> float:
        return self.p_cdmcr

    @p_cimcr.setter
    def p_cimcr(self, val: float) -> None:
        self.p_cdmcr = val

    @property
    def d_cimcr_max(self) -> float:
        return self.d_cdmcr_max

    @d_cimcr_max.setter
    def d_cimcr_max(self, val: float) -> None:
        self.d_cdmcr_max = val


FailLadCoupledDelaminationMicrocrackRate = FailLadCoupledDelaminationMicrocrackingRate
FailLadCoupledDelamMicrocrackingRate = FailLadCoupledDelaminationMicrocrackingRate
FailLadCoupledDelamMicrocrackRate = FailLadCoupledDelaminationMicrocrackingRate
FailLadCoupledInterlaminarMicrocrackingRate = FailLadCoupledDelaminationMicrocrackingRate
FailLadCoupledInterlaminarMicrocrackRate = FailLadCoupledDelaminationMicrocrackingRate


# ============================================================================
# M449 Suite: FailLadDynamicDelaminationMicrodebondingRate, EngElectrothermoflexomagnetochiralexcitonicplasmonicpolaritonicResonanceEnergy, LagmulSchemeSpinorSpatialLinkageJoint
# ============================================================================

@dataclass
class FailLadDynamicDelaminationMicrodebondingRate:
    """``/FAIL/LAD_DYNAMIC_DELAMINATION_MICRODEBONDING_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_DELAMINATION_MICRODEBONDING_RATE`` (M449): Ladevèze rate-dependent dynamic interlaminar delamination microdebonding and matrix ply separation model."""
    mat_id: int = 0
    title: str = ""
    sigma_ddmdbr0: float = 0.0     # initial dynamic delamination microdebonding threshold stress sigma_ddmdbr,0
    sigma_ddmdbrc: float = 1.0     # critical dynamic delamination microdebonding saturation stress sigma_ddmdbr,c
    gamma_ddmdbr: float = 0.0      # dynamic delamination microdebonding rate sensitivity factor gamma_ddmdbr
    p_ddmdbr: float = 1.0          # dynamic delamination microdebonding rate exponent p_ddmdbr
    d_ddmdbr_max: float = 0.999    # maximum allowable dynamic delamination microdebonding damage index
    ifail_sh: int = 1              # shell element deletion flag
    ifail_so: int = 1              # solid element deletion flag
    fail_id: int = 0               # failure model ID reference
    sigma_ddmr0: float = 0.0       # compatibility alias field
    sigma_ddmrc: float = 1.0       # compatibility alias field
    gamma_ddmr: float = 0.0        # compatibility alias field
    p_ddmr: float = 1.0            # compatibility alias field
    d_ddmr_max: float = 0.999      # compatibility alias field

    def __post_init__(self):
        if self.sigma_ddmr0 != 0.0 and self.sigma_ddmdbr0 == 0.0:
            self.sigma_ddmdbr0 = self.sigma_ddmr0
        elif self.sigma_ddmdbr0 != 0.0 and self.sigma_ddmr0 == 0.0:
            self.sigma_ddmr0 = self.sigma_ddmdbr0
        if self.sigma_ddmrc != 1.0 and self.sigma_ddmdbrc == 1.0:
            self.sigma_ddmdbrc = self.sigma_ddmrc
        elif self.sigma_ddmdbrc != 1.0 and self.sigma_ddmrc == 1.0:
            self.sigma_ddmrc = self.sigma_ddmdbrc
        if self.gamma_ddmr != 0.0 and self.gamma_ddmdbr == 0.0:
            self.gamma_ddmdbr = self.gamma_ddmr
        elif self.gamma_ddmdbr != 0.0 and self.gamma_ddmr == 0.0:
            self.gamma_ddmr = self.gamma_ddmdbr
        if self.p_ddmr != 1.0 and self.p_ddmdbr == 1.0:
            self.p_ddmdbr = self.p_ddmr
        elif self.p_ddmdbr != 1.0 and self.p_ddmr == 1.0:
            self.p_ddmr = self.p_ddmdbr
        if self.d_ddmr_max != 0.999 and self.d_ddmdbr_max == 0.999:
            self.d_ddmdbr_max = self.d_ddmr_max
        elif self.d_ddmdbr_max != 0.999 and self.d_ddmr_max == 0.999:
            self.d_ddmr_max = self.d_ddmdbr_max

    @property
    def sigma_ddmdb0(self) -> float:
        return self.sigma_ddmdbr0

    @sigma_ddmdb0.setter
    def sigma_ddmdb0(self, val: float) -> None:
        self.sigma_ddmdbr0 = val
        self.sigma_ddmr0 = val

    @property
    def sigma_ddmdbc(self) -> float:
        return self.sigma_ddmdbrc

    @sigma_ddmdbc.setter
    def sigma_ddmdbc(self, val: float) -> None:
        self.sigma_ddmdbrc = val
        self.sigma_ddmrc = val

    @property
    def gamma_ddmdb(self) -> float:
        return self.gamma_ddmdbr

    @gamma_ddmdb.setter
    def gamma_ddmdb(self, val: float) -> None:
        self.gamma_ddmdbr = val
        self.gamma_ddmr = val

    @property
    def p_ddmdb(self) -> float:
        return self.p_ddmdbr

    @p_ddmdb.setter
    def p_ddmdb(self, val: float) -> None:
        self.p_ddmdbr = val
        self.p_ddmr = val

    @property
    def d_ddmdb_max(self) -> float:
        return self.d_ddmdbr_max

    @d_ddmdb_max.setter
    def d_ddmdb_max(self, val: float) -> None:
        self.d_ddmdbr_max = val
        self.d_ddmr_max = val

    @property
    def sigma_ddm0(self) -> float:
        return self.sigma_ddmdbr0

    @sigma_ddm0.setter
    def sigma_ddm0(self, val: float) -> None:
        self.sigma_ddmdbr0 = val
        self.sigma_ddmr0 = val

    @property
    def sigma_ddmc(self) -> float:
        return self.sigma_ddmdbrc

    @sigma_ddmc.setter
    def sigma_ddmc(self, val: float) -> None:
        self.sigma_ddmdbrc = val
        self.sigma_ddmrc = val

    @property
    def gamma_ddm(self) -> float:
        return self.gamma_ddmdbr

    @gamma_ddm.setter
    def gamma_ddm(self, val: float) -> None:
        self.gamma_ddmdbr = val
        self.gamma_ddmr = val

    @property
    def p_ddm(self) -> float:
        return self.p_ddmdbr

    @p_ddm.setter
    def p_ddm(self, val: float) -> None:
        self.p_ddmdbr = val
        self.p_ddmr = val

    @property
    def d_ddm_max(self) -> float:
        return self.d_ddmdbr_max

    @d_ddm_max.setter
    def d_ddm_max(self, val: float) -> None:
        self.d_ddmdbr_max = val
        self.d_ddmr_max = val

    @property
    def sigma_dimdbr0(self) -> float:
        return self.sigma_ddmdbr0

    @sigma_dimdbr0.setter
    def sigma_dimdbr0(self, val: float) -> None:
        self.sigma_ddmdbr0 = val
        self.sigma_ddmr0 = val

    @property
    def sigma_dimdbrc(self) -> float:
        return self.sigma_ddmdbrc

    @sigma_dimdbrc.setter
    def sigma_dimdbrc(self, val: float) -> None:
        self.sigma_ddmdbrc = val
        self.sigma_ddmrc = val

    @property
    def gamma_dimdbr(self) -> float:
        return self.gamma_ddmdbr

    @gamma_dimdbr.setter
    def gamma_dimdbr(self, val: float) -> None:
        self.gamma_ddmdbr = val
        self.gamma_ddmr = val

    @property
    def p_dimdbr(self) -> float:
        return self.p_ddmdbr

    @p_dimdbr.setter
    def p_dimdbr(self, val: float) -> None:
        self.p_ddmdbr = val
        self.p_ddmr = val

    @property
    def d_dimdbr_max(self) -> float:
        return self.d_ddmdbr_max

    @d_dimdbr_max.setter
    def d_dimdbr_max(self, val: float) -> None:
        self.d_ddmdbr_max = val
        self.d_ddmr_max = val


FailLadDynamicDelaminationMicrodebondRate = FailLadDynamicDelaminationMicrodebondingRate
FailLadDynamicDelamMicrodebondingRate = FailLadDynamicDelaminationMicrodebondingRate
FailLadDynamicDelamMicrodebondRate = FailLadDynamicDelaminationMicrodebondingRate
FailLadDynamicInterlaminarMicrodebondingRate = FailLadDynamicDelaminationMicrodebondingRate
FailLadDynamicInterlaminarMicrodebondRate = FailLadDynamicDelaminationMicrodebondingRate


# ============================================================================
# M450 Suite: FailLadTransverseDelaminationMicrodebondingRate, EngElectrothermoflexomagnetochiralmagnonicplasmonicpolaritonicResonanceEnergy, LagmulOrbifoldSpinorSpatialLinkageJoint
# ============================================================================

@dataclass
class FailLadTransverseDelaminationMicrodebondingRate:
    """``/FAIL/LAD_TRANSVERSE_DELAMINATION_MICRODEBONDING_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_DELAMINATION_MICRODEBONDING_RATE`` (M450): Ladevèze rate-dependent transverse interlaminar delamination microdebonding and matrix ply separation model."""
    mat_id: int = 0
    title: str = ""
    sigma_tdmdbr0: float = 0.0     # initial transverse delamination microdebonding threshold stress sigma_tdmdbr,0
    sigma_tdmdbrc: float = 1.0     # critical transverse delamination microdebonding saturation stress sigma_tdmdbr,c
    gamma_tdmdbr: float = 0.0      # transverse delamination microdebonding rate sensitivity factor gamma_tdmdbr
    p_tdmdbr: float = 1.0          # transverse delamination microdebonding rate exponent p_tdmdbr
    d_tdmdbr_max: float = 0.999    # maximum allowable transverse delamination microdebonding damage index
    ifail_sh: int = 1              # shell element deletion flag
    ifail_so: int = 1              # solid element deletion flag
    fail_id: int = 0               # failure model ID reference
    sigma_tdmr0: float = 0.0       # compatibility alias field
    sigma_tdmrc: float = 1.0       # compatibility alias field
    gamma_tdmr: float = 0.0        # compatibility alias field
    p_tdmr: float = 1.0            # compatibility alias field
    d_tdmr_max: float = 0.999      # compatibility alias field

    def __post_init__(self):
        if self.sigma_tdmr0 != 0.0 and self.sigma_tdmdbr0 == 0.0:
            self.sigma_tdmdbr0 = self.sigma_tdmr0
        elif self.sigma_tdmdbr0 != 0.0 and self.sigma_tdmr0 == 0.0:
            self.sigma_tdmr0 = self.sigma_tdmdbr0
        if self.sigma_tdmrc != 1.0 and self.sigma_tdmdbrc == 1.0:
            self.sigma_tdmdbrc = self.sigma_tdmrc
        elif self.sigma_tdmdbrc != 1.0 and self.sigma_tdmrc == 1.0:
            self.sigma_tdmrc = self.sigma_tdmdbrc
        if self.gamma_tdmr != 0.0 and self.gamma_tdmdbr == 0.0:
            self.gamma_tdmdbr = self.gamma_tdmr
        elif self.gamma_tdmdbr != 0.0 and self.gamma_tdmr == 0.0:
            self.gamma_tdmr = self.gamma_tdmdbr
        if self.p_tdmr != 1.0 and self.p_tdmdbr == 1.0:
            self.p_tdmdbr = self.p_tdmr
        elif self.p_tdmdbr != 1.0 and self.p_tdmr == 1.0:
            self.p_tdmr = self.p_tdmdbr
        if self.d_tdmr_max != 0.999 and self.d_tdmdbr_max == 0.999:
            self.d_tdmdbr_max = self.d_tdmr_max
        elif self.d_tdmdbr_max != 0.999 and self.d_tdmr_max == 0.999:
            self.d_tdmr_max = self.d_tdmdbr_max

    @property
    def sigma_tdmdb0(self) -> float:
        return self.sigma_tdmdbr0

    @sigma_tdmdb0.setter
    def sigma_tdmdb0(self, val: float) -> None:
        self.sigma_tdmdbr0 = val
        self.sigma_tdmr0 = val

    @property
    def sigma_tdmdbc(self) -> float:
        return self.sigma_tdmdbrc

    @sigma_tdmdbc.setter
    def sigma_tdmdbc(self, val: float) -> None:
        self.sigma_tdmdbrc = val
        self.sigma_tdmrc = val

    @property
    def gamma_tdmdb(self) -> float:
        return self.gamma_tdmdbr

    @gamma_tdmdb.setter
    def gamma_tdmdb(self, val: float) -> None:
        self.gamma_tdmdbr = val
        self.gamma_tdmr = val

    @property
    def p_tdmdb(self) -> float:
        return self.p_tdmdbr

    @p_tdmdb.setter
    def p_tdmdb(self, val: float) -> None:
        self.p_tdmdbr = val
        self.p_tdmr = val

    @property
    def d_tdmdb_max(self) -> float:
        return self.d_tdmdbr_max

    @d_tdmdb_max.setter
    def d_tdmdb_max(self, val: float) -> None:
        self.d_tdmdbr_max = val
        self.d_tdmr_max = val

    @property
    def sigma_tdm0(self) -> float:
        return self.sigma_tdmdbr0

    @sigma_tdm0.setter
    def sigma_tdm0(self, val: float) -> None:
        self.sigma_tdmdbr0 = val
        self.sigma_tdmr0 = val

    @property
    def sigma_tdmc(self) -> float:
        return self.sigma_tdmdbrc

    @sigma_tdmc.setter
    def sigma_tdmc(self, val: float) -> None:
        self.sigma_tdmdbrc = val
        self.sigma_tdmrc = val

    @property
    def gamma_tdm(self) -> float:
        return self.gamma_tdmdbr

    @gamma_tdm.setter
    def gamma_tdm(self, val: float) -> None:
        self.gamma_tdmdbr = val
        self.gamma_tdmr = val

    @property
    def p_tdm(self) -> float:
        return self.p_tdmdbr

    @p_tdm.setter
    def p_tdm(self, val: float) -> None:
        self.p_tdmdbr = val
        self.p_tdmr = val

    @property
    def d_tdm_max(self) -> float:
        return self.d_tdmdbr_max

    @d_tdm_max.setter
    def d_tdm_max(self, val: float) -> None:
        self.d_tdmdbr_max = val
        self.d_tdmr_max = val

    @property
    def sigma_timdbr0(self) -> float:
        return self.sigma_tdmdbr0

    @sigma_timdbr0.setter
    def sigma_timdbr0(self, val: float) -> None:
        self.sigma_tdmdbr0 = val
        self.sigma_tdmr0 = val

    @property
    def sigma_timdbrc(self) -> float:
        return self.sigma_tdmdbrc

    @sigma_timdbrc.setter
    def sigma_timdbrc(self, val: float) -> None:
        self.sigma_tdmdbrc = val
        self.sigma_tdmrc = val

    @property
    def gamma_timdbr(self) -> float:
        return self.gamma_tdmdbr

    @gamma_timdbr.setter
    def gamma_timdbr(self, val: float) -> None:
        self.gamma_tdmdbr = val
        self.gamma_tdmr = val

    @property
    def p_timdbr(self) -> float:
        return self.p_tdmdbr

    @p_timdbr.setter
    def p_timdbr(self, val: float) -> None:
        self.p_tdmdbr = val
        self.p_tdmr = val

    @property
    def d_timdbr_max(self) -> float:
        return self.d_tdmdbr_max

    @d_timdbr_max.setter
    def d_timdbr_max(self, val: float) -> None:
        self.d_tdmdbr_max = val
        self.d_tdmr_max = val


FailLadTransverseDelaminationMicrodebondRate = FailLadTransverseDelaminationMicrodebondingRate
FailLadTransverseDelamMicrodebondingRate = FailLadTransverseDelaminationMicrodebondingRate
FailLadTransverseDelamMicrodebondRate = FailLadTransverseDelaminationMicrodebondingRate
FailLadTransverseInterlaminarMicrodebondingRate = FailLadTransverseDelaminationMicrodebondingRate
FailLadTransverseInterlaminarMicrodebondRate = FailLadTransverseDelaminationMicrodebondingRate


# ============================================================================
# M451 Suite: FailLadCoupledDelaminationMicrodebondingRate, EngElectrothermoflexomagnetochiralspinplasmonicpolaritonicResonanceEnergy, LagmulFoliationSpinorSpatialLinkageJoint, SensorSpringCoupledCrackleRate
# ============================================================================

@dataclass
class FailLadCoupledDelaminationMicrodebondingRate:
    """``/FAIL/LAD_COUPLED_DELAMINATION_MICRODEBONDING_RATE`` or ``/FAIL/LADEVEZE_COUPLED_DELAMINATION_MICRODEBONDING_RATE`` (M451): Ladevèze rate-dependent coupled interlaminar delamination microdebonding and matrix ply separation model."""
    mat_id: int = 0
    title: str = ""
    sigma_cdmdbr0: float = 0.0     # initial coupled delamination microdebonding threshold stress sigma_cdmdbr,0
    sigma_cdmdbrc: float = 1.0     # critical coupled delamination microdebonding saturation stress sigma_cdmdbr,c
    gamma_cdmdbr: float = 0.0      # coupled delamination microdebonding rate sensitivity factor gamma_cdmdbr
    p_cdmdbr: float = 1.0          # coupled delamination microdebonding rate exponent p_cdmdbr
    d_cdmdbr_max: float = 0.999    # maximum allowable coupled delamination microdebonding damage index
    ifail_sh: int = 1              # shell element deletion flag
    ifail_so: int = 1              # solid element deletion flag
    fail_id: int = 0               # failure model ID reference
    sigma_cdmr0: float = 0.0       # compatibility alias field
    sigma_cdmrc: float = 1.0       # compatibility alias field
    gamma_cdmr: float = 0.0        # compatibility alias field
    p_cdmr: float = 1.0            # compatibility alias field
    d_cdmr_max: float = 0.999      # compatibility alias field

    def __post_init__(self):
        if self.sigma_cdmr0 != 0.0 and self.sigma_cdmdbr0 == 0.0:
            self.sigma_cdmdbr0 = self.sigma_cdmr0
        elif self.sigma_cdmdbr0 != 0.0 and self.sigma_cdmr0 == 0.0:
            self.sigma_cdmr0 = self.sigma_cdmdbr0
        if self.sigma_cdmrc != 1.0 and self.sigma_cdmdbrc == 1.0:
            self.sigma_cdmdbrc = self.sigma_cdmrc
        elif self.sigma_cdmdbrc != 1.0 and self.sigma_cdmrc == 1.0:
            self.sigma_cdmrc = self.sigma_cdmdbrc
        if self.gamma_cdmr != 0.0 and self.gamma_cdmdbr == 0.0:
            self.gamma_cdmdbr = self.gamma_cdmr
        elif self.gamma_cdmdbr != 0.0 and self.gamma_cdmr == 0.0:
            self.gamma_cdmr = self.gamma_cdmdbr
        if self.p_cdmr != 1.0 and self.p_cdmdbr == 1.0:
            self.p_cdmdbr = self.p_cdmr
        elif self.p_cdmdbr != 1.0 and self.p_cdmr == 1.0:
            self.p_cdmr = self.p_cdmdbr
        if self.d_cdmr_max != 0.999 and self.d_cdmdbr_max == 0.999:
            self.d_cdmdbr_max = self.d_cdmr_max
        elif self.d_cdmdbr_max != 0.999 and self.d_cdmr_max == 0.999:
            self.d_cdmr_max = self.d_cdmdbr_max

    @property
    def sigma_cdmdb0(self) -> float:
        return self.sigma_cdmdbr0

    @sigma_cdmdb0.setter
    def sigma_cdmdb0(self, val: float) -> None:
        self.sigma_cdmdbr0 = val
        self.sigma_cdmr0 = val

    @property
    def sigma_cdmdbc(self) -> float:
        return self.sigma_cdmdbrc

    @sigma_cdmdbc.setter
    def sigma_cdmdbc(self, val: float) -> None:
        self.sigma_cdmdbrc = val
        self.sigma_cdmrc = val

    @property
    def gamma_cdmdb(self) -> float:
        return self.gamma_cdmdbr

    @gamma_cdmdb.setter
    def gamma_cdmdb(self, val: float) -> None:
        self.gamma_cdmdbr = val
        self.gamma_cdmr = val

    @property
    def p_cdmdb(self) -> float:
        return self.p_cdmdbr

    @p_cdmdb.setter
    def p_cdmdb(self, val: float) -> None:
        self.p_cdmdbr = val
        self.p_cdmr = val

    @property
    def d_cdmdb_max(self) -> float:
        return self.d_cdmdbr_max

    @d_cdmdb_max.setter
    def d_cdmdb_max(self, val: float) -> None:
        self.d_cdmdbr_max = val
        self.d_cdmr_max = val

    @property
    def sigma_cdm0(self) -> float:
        return self.sigma_cdmdbr0

    @sigma_cdm0.setter
    def sigma_cdm0(self, val: float) -> None:
        self.sigma_cdmdbr0 = val
        self.sigma_cdmr0 = val

    @property
    def sigma_cdmc(self) -> float:
        return self.sigma_cdmdbrc

    @sigma_cdmc.setter
    def sigma_cdmc(self, val: float) -> None:
        self.sigma_cdmdbrc = val
        self.sigma_cdmrc = val

    @property
    def gamma_cdm(self) -> float:
        return self.gamma_cdmdbr

    @gamma_cdm.setter
    def gamma_cdm(self, val: float) -> None:
        self.gamma_cdmdbr = val
        self.gamma_cdmr = val

    @property
    def p_cdm(self) -> float:
        return self.p_cdmdbr

    @p_cdm.setter
    def p_cdm(self, val: float) -> None:
        self.p_cdmdbr = val
        self.p_cdmr = val

    @property
    def d_cdm_max(self) -> float:
        return self.d_cdmdbr_max

    @d_cdm_max.setter
    def d_cdm_max(self, val: float) -> None:
        self.d_cdmdbr_max = val
        self.d_cdmr_max = val

    @property
    def sigma_cimdbr0(self) -> float:
        return self.sigma_cdmdbr0

    @sigma_cimdbr0.setter
    def sigma_cimdbr0(self, val: float) -> None:
        self.sigma_cdmdbr0 = val
        self.sigma_cdmr0 = val

    @property
    def sigma_cimdbrc(self) -> float:
        return self.sigma_cdmdbrc

    @sigma_cimdbrc.setter
    def sigma_cimdbrc(self, val: float) -> None:
        self.sigma_cdmdbrc = val
        self.sigma_cdmrc = val

    @property
    def gamma_cimdbr(self) -> float:
        return self.gamma_cdmdbr

    @gamma_cimdbr.setter
    def gamma_cimdbr(self, val: float) -> None:
        self.gamma_cdmdbr = val
        self.gamma_cdmr = val

    @property
    def p_cimdbr(self) -> float:
        return self.p_cdmdbr

    @p_cimdbr.setter
    def p_cimdbr(self, val: float) -> None:
        self.p_cdmdbr = val
        self.p_cdmr = val

    @property
    def d_cimdbr_max(self) -> float:
        return self.d_cdmdbr_max

    @d_cimdbr_max.setter
    def d_cimdbr_max(self, val: float) -> None:
        self.d_cdmdbr_max = val
        self.d_cdmr_max = val


FailLadCoupledDelaminationMicrodebondRate = FailLadCoupledDelaminationMicrodebondingRate
FailLadCoupleDelaminationMicrodebondingRate = FailLadCoupledDelaminationMicrodebondingRate
FailLadCoupleDelaminationMicrodebondRate = FailLadCoupledDelaminationMicrodebondingRate
FailLadCoupledDelamMicrodebondingRate = FailLadCoupledDelaminationMicrodebondingRate
FailLadCoupledDelamMicrodebondRate = FailLadCoupledDelaminationMicrodebondingRate
FailLadCoupleDelamMicrodebondingRate = FailLadCoupledDelaminationMicrodebondingRate
FailLadCoupleDelamMicrodebondRate = FailLadCoupledDelaminationMicrodebondingRate
FailLadCoupledInterlaminarMicrodebondingRate = FailLadCoupledDelaminationMicrodebondingRate
FailLadCoupledInterlaminarMicrodebondRate = FailLadCoupledDelaminationMicrodebondingRate
FailLadCoupleInterlaminarMicrodebondingRate = FailLadCoupledDelaminationMicrodebondingRate
FailLadCoupleInterlaminarMicrodebondRate = FailLadCoupledDelaminationMicrodebondingRate


# ============================================================================
# M452 Suite: FailLadDynamicMatrixShearDegradationRate, EngElectrothermoflexomagnetochiralspinonplasmonicpolaritonicResonanceEnergy, LagmulStratificationSpinorSpatialLinkageJoint, SensorSpringTorsionalCrackleRate
# ============================================================================

@dataclass
class FailLadDynamicMatrixShearDegradationRate:
    """``/FAIL/LAD_DYNAMIC_MATRIX_SHEAR_DEGRADATION_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_MATRIX_SHEAR_DEGRADATION_RATE`` (M452): Ladevèze rate-dependent dynamic in-plane matrix shear damage evolution, nonlinear inelastic shear strain accumulation, and shear degradation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dmsdr0: float = 0.0    # initial matrix shear degradation threshold stress sigma_dmsdr,0
    sigma_dmsdrc: float = 1.0    # critical dynamic shear degradation stress sigma_dmsdr,c
    gamma_dmsdr: float = 0.0     # dynamic matrix shear degradation rate sensitivity factor gamma_dmsdr
    p_dmsdr: float = 1.0         # dynamic matrix shear degradation rate exponent p_dmsdr
    d_dmsdr_max: float = 0.999   # maximum allowable in-plane shear damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference

    @property
    def sigma_dmsd0(self) -> float:
        return self.sigma_dmsdr0

    @sigma_dmsd0.setter
    def sigma_dmsd0(self, val: float) -> None:
        self.sigma_dmsdr0 = val

    @property
    def sigma_dmsdc(self) -> float:
        return self.sigma_dmsdrc

    @sigma_dmsdc.setter
    def sigma_dmsdc(self, val: float) -> None:
        self.sigma_dmsdrc = val

    @property
    def gamma_dmsd(self) -> float:
        return self.gamma_dmsdr

    @gamma_dmsd.setter
    def gamma_dmsd(self, val: float) -> None:
        self.gamma_dmsdr = val

    @property
    def p_dmsd(self) -> float:
        return self.p_dmsdr

    @p_dmsd.setter
    def p_dmsd(self, val: float) -> None:
        self.p_dmsdr = val

    @property
    def d_dmsd_max(self) -> float:
        return self.d_dmsdr_max

    @d_dmsd_max.setter
    def d_dmsd_max(self, val: float) -> None:
        self.d_dmsdr_max = val

    @property
    def sigma_dms0(self) -> float:
        return self.sigma_dmsdr0

    @sigma_dms0.setter
    def sigma_dms0(self, val: float) -> None:
        self.sigma_dmsdr0 = val

    @property
    def sigma_dmsc(self) -> float:
        return self.sigma_dmsdrc

    @sigma_dmsc.setter
    def sigma_dmsc(self, val: float) -> None:
        self.sigma_dmsdrc = val

    @property
    def gamma_dms(self) -> float:
        return self.gamma_dmsdr

    @gamma_dms.setter
    def gamma_dms(self, val: float) -> None:
        self.gamma_dmsdr = val

    @property
    def p_dms(self) -> float:
        return self.p_dmsdr

    @p_dms.setter
    def p_dms(self, val: float) -> None:
        self.p_dmsdr = val

    @property
    def d_dms_max(self) -> float:
        return self.d_dmsdr_max

    @d_dms_max.setter
    def d_dms_max(self, val: float) -> None:
        self.d_dmsdr_max = val

    @property
    def sigma_dsdr0(self) -> float:
        return self.sigma_dmsdr0

    @sigma_dsdr0.setter
    def sigma_dsdr0(self, val: float) -> None:
        self.sigma_dmsdr0 = val

    @property
    def sigma_dsdrc(self) -> float:
        return self.sigma_dmsdrc

    @sigma_dsdrc.setter
    def sigma_dsdrc(self, val: float) -> None:
        self.sigma_dmsdrc = val

    @property
    def gamma_dsdr(self) -> float:
        return self.gamma_dmsdr

    @gamma_dsdr.setter
    def gamma_dsdr(self, val: float) -> None:
        self.gamma_dmsdr = val

    @property
    def p_dsdr(self) -> float:
        return self.p_dmsdr

    @p_dsdr.setter
    def p_dsdr(self, val: float) -> None:
        self.p_dmsdr = val

    @property
    def d_dsdr_max(self) -> float:
        return self.d_dmsdr_max

    @d_dsdr_max.setter
    def d_dsdr_max(self, val: float) -> None:
        self.d_dmsdr_max = val


FailLadDynamicMatrixShearDegradeRate = FailLadDynamicMatrixShearDegradationRate
FailLadDynamicMatrixShearRate = FailLadDynamicMatrixShearDegradationRate
FailLadDynamicShearDegradationRate = FailLadDynamicMatrixShearDegradationRate
FailLadDynamicShearDegradeRate = FailLadDynamicMatrixShearDegradationRate
FailLadDynamicMatrixShearDamageRate = FailLadDynamicMatrixShearDegradationRate


# ============================================================================
# M453 Suite: FailLadTransverseMatrixShearDegradationRate, EngElectrothermoflexomagnetochiralholonplasmonicpolaritonicResonanceEnergy, LagmulFibrationSpinorSpatialLinkageJoint, SensorSpringTotalCrackleRate
# ============================================================================

@dataclass
class FailLadTransverseMatrixShearDegradationRate:
    """``/FAIL/LAD_TRANSVERSE_MATRIX_SHEAR_DEGRADATION_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_MATRIX_SHEAR_DEGRADATION_RATE`` (M453): Ladevèze rate-dependent transverse out-of-plane matrix shear degradation, transverse interlaminar shear damage evolution, and transverse shear degradation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tmsdr0: float = 0.0    # initial transverse shear degradation threshold stress sigma_tmsdr,0
    sigma_tmsdrc: float = 1.0    # critical transverse shear degradation stress sigma_tmsdr,c
    gamma_tmsdr: float = 0.0     # transverse matrix shear degradation rate sensitivity factor gamma_tmsdr
    p_tmsdr: float = 1.0         # transverse matrix shear degradation rate exponent p_tmsdr
    d_tmsdr_max: float = 0.999   # maximum allowable transverse shear damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference

    @property
    def sigma_tmsd0(self) -> float:
        return self.sigma_tmsdr0

    @sigma_tmsd0.setter
    def sigma_tmsd0(self, val: float) -> None:
        self.sigma_tmsdr0 = val

    @property
    def sigma_tmsdc(self) -> float:
        return self.sigma_tmsdrc

    @sigma_tmsdc.setter
    def sigma_tmsdc(self, val: float) -> None:
        self.sigma_tmsdrc = val

    @property
    def gamma_tmsd(self) -> float:
        return self.gamma_tmsdr

    @gamma_tmsd.setter
    def gamma_tmsd(self, val: float) -> None:
        self.gamma_tmsdr = val

    @property
    def p_tmsd(self) -> float:
        return self.p_tmsdr

    @p_tmsd.setter
    def p_tmsd(self, val: float) -> None:
        self.p_tmsdr = val

    @property
    def d_tmsd_max(self) -> float:
        return self.d_tmsdr_max

    @d_tmsd_max.setter
    def d_tmsd_max(self, val: float) -> None:
        self.d_tmsdr_max = val

    @property
    def sigma_tms0(self) -> float:
        return self.sigma_tmsdr0

    @sigma_tms0.setter
    def sigma_tms0(self, val: float) -> None:
        self.sigma_tmsdr0 = val

    @property
    def sigma_tmsc(self) -> float:
        return self.sigma_tmsdrc

    @sigma_tmsc.setter
    def sigma_tmsc(self, val: float) -> None:
        self.sigma_tmsdrc = val

    @property
    def gamma_tms(self) -> float:
        return self.gamma_tmsdr

    @gamma_tms.setter
    def gamma_tms(self, val: float) -> None:
        self.gamma_tmsdr = val

    @property
    def p_tms(self) -> float:
        return self.p_tmsdr

    @p_tms.setter
    def p_tms(self, val: float) -> None:
        self.p_tmsdr = val

    @property
    def d_tms_max(self) -> float:
        return self.d_tmsdr_max

    @d_tms_max.setter
    def d_tms_max(self, val: float) -> None:
        self.d_tmsdr_max = val

    @property
    def sigma_tsdr0(self) -> float:
        return self.sigma_tmsdr0

    @sigma_tsdr0.setter
    def sigma_tsdr0(self, val: float) -> None:
        self.sigma_tmsdr0 = val

    @property
    def sigma_tsdrc(self) -> float:
        return self.sigma_tmsdrc

    @sigma_tsdrc.setter
    def sigma_tsdrc(self, val: float) -> None:
        self.sigma_tmsdrc = val

    @property
    def gamma_tsdr(self) -> float:
        return self.gamma_tmsdr

    @gamma_tsdr.setter
    def gamma_tsdr(self, val: float) -> None:
        self.gamma_tmsdr = val

    @property
    def p_tsdr(self) -> float:
        return self.p_tmsdr

    @p_tsdr.setter
    def p_tsdr(self, val: float) -> None:
        self.p_tmsdr = val

    @property
    def d_tsdr_max(self) -> float:
        return self.d_tmsdr_max

    @d_tsdr_max.setter
    def d_tsdr_max(self, val: float) -> None:
        self.d_tmsdr_max = val


FailLadTransverseMatrixShearDegradeRate = FailLadTransverseMatrixShearDegradationRate
FailLadTransverseMatrixShearRate = FailLadTransverseMatrixShearDegradationRate
FailLadTransverseShearDegradationRate = FailLadTransverseMatrixShearDegradationRate
FailLadTransverseShearDegradeRate = FailLadTransverseMatrixShearDegradationRate
FailLadTransverseMatrixShearDamageRate = FailLadTransverseMatrixShearDegradationRate


# ============================================================================
# M454 Suite: FailLadCoupledMatrixShearDegradationRate, EngElectrothermoflexomagnetochiralorbitonplasmonicpolaritonicResonanceEnergy, LagmulBundleSpinorSpatialLinkageJoint, SensorSpringNormalPopRate
# ============================================================================

@dataclass
class FailLadCoupledMatrixShearDegradationRate:
    """``/FAIL/LAD_COUPLED_MATRIX_SHEAR_DEGRADATION_RATE`` or ``/FAIL/LADEVEZE_COUPLED_MATRIX_SHEAR_DEGRADATION_RATE`` (M454): Ladevèze rate-dependent coupled multi-axial matrix shear degradation, coupled in-plane/transverse shear damage accumulation, and progressive shear failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cmsdr0: float = 0.0    # initial coupled shear degradation threshold stress sigma_cmsdr,0
    sigma_cmsdrc: float = 1.0    # critical coupled shear degradation stress sigma_cmsdr,c
    gamma_cmsdr: float = 0.0     # coupled matrix shear degradation rate sensitivity factor gamma_cmsdr
    p_cmsdr: float = 1.0         # coupled matrix shear degradation rate exponent p_cmsdr
    d_cmsdr_max: float = 0.999   # maximum allowable coupled shear damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference

    @property
    def sigma_cmsd0(self) -> float:
        return self.sigma_cmsdr0

    @sigma_cmsd0.setter
    def sigma_cmsd0(self, val: float) -> None:
        self.sigma_cmsdr0 = val

    @property
    def sigma_cmsdc(self) -> float:
        return self.sigma_cmsdrc

    @sigma_cmsdc.setter
    def sigma_cmsdc(self, val: float) -> None:
        self.sigma_cmsdrc = val

    @property
    def gamma_cmsd(self) -> float:
        return self.gamma_cmsdr

    @gamma_cmsd.setter
    def gamma_cmsd(self, val: float) -> None:
        self.gamma_cmsdr = val

    @property
    def p_cmsd(self) -> float:
        return self.p_cmsdr

    @p_cmsd.setter
    def p_cmsd(self, val: float) -> None:
        self.p_cmsdr = val

    @property
    def d_cmsd_max(self) -> float:
        return self.d_cmsdr_max

    @d_cmsd_max.setter
    def d_cmsd_max(self, val: float) -> None:
        self.d_cmsdr_max = val

    @property
    def sigma_cms0(self) -> float:
        return self.sigma_cmsdr0

    @sigma_cms0.setter
    def sigma_cms0(self, val: float) -> None:
        self.sigma_cmsdr0 = val

    @property
    def sigma_cmsc(self) -> float:
        return self.sigma_cmsdrc

    @sigma_cmsc.setter
    def sigma_cmsc(self, val: float) -> None:
        self.sigma_cmsdrc = val

    @property
    def gamma_cms(self) -> float:
        return self.gamma_cmsdr

    @gamma_cms.setter
    def gamma_cms(self, val: float) -> None:
        self.gamma_cmsdr = val

    @property
    def p_cms(self) -> float:
        return self.p_cmsdr

    @p_cms.setter
    def p_cms(self, val: float) -> None:
        self.p_cmsdr = val

    @property
    def d_cms_max(self) -> float:
        return self.d_cmsdr_max

    @d_cms_max.setter
    def d_cms_max(self, val: float) -> None:
        self.d_cmsdr_max = val

    @property
    def sigma_csdr0(self) -> float:
        return self.sigma_cmsdr0

    @sigma_csdr0.setter
    def sigma_csdr0(self, val: float) -> None:
        self.sigma_cmsdr0 = val

    @property
    def sigma_csdrc(self) -> float:
        return self.sigma_cmsdrc

    @sigma_csdrc.setter
    def sigma_csdrc(self, val: float) -> None:
        self.sigma_cmsdrc = val

    @property
    def gamma_csdr(self) -> float:
        return self.gamma_cmsdr

    @gamma_csdr.setter
    def gamma_csdr(self, val: float) -> None:
        self.gamma_cmsdr = val

    @property
    def p_csdr(self) -> float:
        return self.p_cmsdr

    @p_csdr.setter
    def p_csdr(self, val: float) -> None:
        self.p_cmsdr = val

    @property
    def d_csdr_max(self) -> float:
        return self.d_cmsdr_max

    @d_csdr_max.setter
    def d_csdr_max(self, val: float) -> None:
        self.d_cmsdr_max = val


FailLadCoupledMatrixShearDegradeRate = FailLadCoupledMatrixShearDegradationRate
FailLadCoupleMatrixShearDegradationRate = FailLadCoupledMatrixShearDegradationRate
FailLadCoupleMatrixShearDegradeRate = FailLadCoupledMatrixShearDegradationRate
FailLadCoupledMatrixShearRate = FailLadCoupledMatrixShearDegradationRate
FailLadCoupleMatrixShearRate = FailLadCoupledMatrixShearDegradationRate
FailLadCoupledShearDegradationRate = FailLadCoupledMatrixShearDegradationRate
FailLadCoupleShearDegradationRate = FailLadCoupledMatrixShearDegradationRate
FailLadCoupledShearDegradeRate = FailLadCoupledMatrixShearDegradationRate
FailLadCoupleShearDegradeRate = FailLadCoupledMatrixShearDegradationRate
FailLadCoupledMatrixShearDamageRate = FailLadCoupledMatrixShearDegradationRate
FailLadCoupleMatrixShearDamageRate = FailLadCoupledMatrixShearDegradationRate


# ============================================================================
# M455 Suite: FailLadDynamicFiberTensionRuptureRate, EngElectrothermoflexomagnetochiralplasmononplasmonicpolaritonicResonanceEnergy, LagmulConnectionSpinorSpatialLinkageJoint, SensorSpringTransversePopRate
# ============================================================================

@dataclass
class FailLadDynamicFiberTensionRuptureRate:
    """``/FAIL/LAD_DYNAMIC_FIBER_TENSION_RUPTURE_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_FIBER_TENSION_RUPTURE_RATE`` (M455): Ladevèze rate-dependent dynamic longitudinal fiber tensile damage accumulation, high-rate fiber bundle fracture, and tensile rupture failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dftrr0: float = 0.0    # initial fiber tensile rupture threshold stress sigma_dftrr,0
    sigma_dftrrc: float = 1.0    # critical dynamic fiber tensile rupture stress sigma_dftrr,c
    gamma_dftrr: float = 0.0     # dynamic fiber tensile rupture rate sensitivity factor gamma_dftrr
    p_dftrr: float = 1.0         # dynamic fiber tensile rupture rate exponent p_dftrr
    d_dftrr_max: float = 0.999   # maximum allowable fiber tensile damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference

    @property
    def sigma_dftr0(self) -> float:
        return self.sigma_dftrr0

    @sigma_dftr0.setter
    def sigma_dftr0(self, val: float) -> None:
        self.sigma_dftrr0 = val

    @property
    def sigma_dftrc(self) -> float:
        return self.sigma_dftrrc

    @sigma_dftrc.setter
    def sigma_dftrc(self, val: float) -> None:
        self.sigma_dftrrc = val

    @property
    def gamma_dftr(self) -> float:
        return self.gamma_dftrr

    @gamma_dftr.setter
    def gamma_dftr(self, val: float) -> None:
        self.gamma_dftrr = val

    @property
    def p_dftr(self) -> float:
        return self.p_dftrr

    @p_dftr.setter
    def p_dftr(self, val: float) -> None:
        self.p_dftrr = val

    @property
    def d_dftr_max(self) -> float:
        return self.d_dftrr_max

    @d_dftr_max.setter
    def d_dftr_max(self, val: float) -> None:
        self.d_dftrr_max = val

    @property
    def sigma_dft0(self) -> float:
        return self.sigma_dftrr0

    @sigma_dft0.setter
    def sigma_dft0(self, val: float) -> None:
        self.sigma_dftrr0 = val

    @property
    def sigma_dftc(self) -> float:
        return self.sigma_dftrrc

    @sigma_dftc.setter
    def sigma_dftc(self, val: float) -> None:
        self.sigma_dftrrc = val

    @property
    def gamma_dft(self) -> float:
        return self.gamma_dftrr

    @gamma_dft.setter
    def gamma_dft(self, val: float) -> None:
        self.gamma_dftrr = val

    @property
    def p_dft(self) -> float:
        return self.p_dftrr

    @p_dft.setter
    def p_dft(self, val: float) -> None:
        self.p_dftrr = val

    @property
    def d_dft_max(self) -> float:
        return self.d_dftrr_max

    @d_dft_max.setter
    def d_dft_max(self, val: float) -> None:
        self.d_dftrr_max = val

    @property
    def sigma_dfr0(self) -> float:
        return self.sigma_dftrr0

    @sigma_dfr0.setter
    def sigma_dfr0(self, val: float) -> None:
        self.sigma_dftrr0 = val

    @property
    def sigma_dfrc(self) -> float:
        return self.sigma_dftrrc

    @sigma_dfrc.setter
    def sigma_dfrc(self, val: float) -> None:
        self.sigma_dftrrc = val

    @property
    def gamma_dfr(self) -> float:
        return self.gamma_dftrr

    @gamma_dfr.setter
    def gamma_dfr(self, val: float) -> None:
        self.gamma_dftrr = val

    @property
    def p_dfr(self) -> float:
        return self.p_dftrr

    @p_dfr.setter
    def p_dfr(self, val: float) -> None:
        self.p_dftrr = val

    @property
    def d_dfr_max(self) -> float:
        return self.d_dftrr_max

    @d_dfr_max.setter
    def d_dfr_max(self, val: float) -> None:
        self.d_dftrr_max = val


FailLadDynamicFiberTensionRupture = FailLadDynamicFiberTensionRuptureRate
FailLadDynamicFiberTensionRate = FailLadDynamicFiberTensionRuptureRate
FailLadDynamicFiberRuptureRate = FailLadDynamicFiberTensionRuptureRate
FailLadDynamicFiberTensileRuptureRate = FailLadDynamicFiberTensionRuptureRate
FailLadDynamicFiberTensionDamageRate = FailLadDynamicFiberTensionRuptureRate


# ============================================================================
# M456 Suite: FailLadTransverseFiberTensionRuptureRate, EngElectrothermoflexomagnetochiralparamagnonplasmonicpolaritonicResonanceEnergy, LagmulCurvatureSpinorSpatialLinkageJoint, SensorSpringCoupledPopRate
# ============================================================================

@dataclass
class FailLadTransverseFiberTensionRuptureRate:
    """``/FAIL/LAD_TRANSVERSE_FIBER_TENSION_RUPTURE_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_FIBER_TENSION_RUPTURE_RATE`` (M456): Ladevèze rate-dependent transverse out-of-plane fiber tensile damage accumulation, transverse fiber bundle debonding, and transverse tensile rupture failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tftrr0: float = 0.0    # initial fiber tensile rupture threshold stress sigma_tftrr,0
    sigma_tftrrc: float = 1.0    # critical dynamic fiber tensile rupture stress sigma_tftrr,c
    gamma_tftrr: float = 0.0     # dynamic fiber tensile rupture rate sensitivity factor gamma_tftrr
    p_tftrr: float = 1.0         # dynamic fiber tensile rupture rate exponent p_tftrr
    d_tftrr_max: float = 0.999   # maximum allowable fiber tensile damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference

    @property
    def sigma_tftr0(self) -> float:
        return self.sigma_tftrr0

    @sigma_tftr0.setter
    def sigma_tftr0(self, val: float) -> None:
        self.sigma_tftrr0 = val

    @property
    def sigma_tftrc(self) -> float:
        return self.sigma_tftrrc

    @sigma_tftrc.setter
    def sigma_tftrc(self, val: float) -> None:
        self.sigma_tftrrc = val

    @property
    def gamma_tftr(self) -> float:
        return self.gamma_tftrr

    @gamma_tftr.setter
    def gamma_tftr(self, val: float) -> None:
        self.gamma_tftrr = val

    @property
    def p_tftr(self) -> float:
        return self.p_tftrr

    @p_tftr.setter
    def p_tftr(self, val: float) -> None:
        self.p_tftrr = val

    @property
    def d_tftr_max(self) -> float:
        return self.d_tftrr_max

    @d_tftr_max.setter
    def d_tftr_max(self, val: float) -> None:
        self.d_tftrr_max = val

    @property
    def sigma_tft0(self) -> float:
        return self.sigma_tftrr0

    @sigma_tft0.setter
    def sigma_tft0(self, val: float) -> None:
        self.sigma_tftrr0 = val

    @property
    def sigma_tftc(self) -> float:
        return self.sigma_tftrrc

    @sigma_tftc.setter
    def sigma_tftc(self, val: float) -> None:
        self.sigma_tftrrc = val

    @property
    def gamma_tft(self) -> float:
        return self.gamma_tftrr

    @gamma_tft.setter
    def gamma_tft(self, val: float) -> None:
        self.gamma_tftrr = val

    @property
    def p_tft(self) -> float:
        return self.p_tftrr

    @p_tft.setter
    def p_tft(self, val: float) -> None:
        self.p_tftrr = val

    @property
    def d_tft_max(self) -> float:
        return self.d_tftrr_max

    @d_tft_max.setter
    def d_tft_max(self, val: float) -> None:
        self.d_tftrr_max = val

    @property
    def sigma_tfr0(self) -> float:
        return self.sigma_tftrr0

    @sigma_tfr0.setter
    def sigma_tfr0(self, val: float) -> None:
        self.sigma_tftrr0 = val

    @property
    def sigma_tfrc(self) -> float:
        return self.sigma_tftrrc

    @sigma_tfrc.setter
    def sigma_tfrc(self, val: float) -> None:
        self.sigma_tftrrc = val

    @property
    def gamma_tfr(self) -> float:
        return self.gamma_tftrr

    @gamma_tfr.setter
    def gamma_tfr(self, val: float) -> None:
        self.gamma_tftrr = val

    @property
    def p_tfr(self) -> float:
        return self.p_tftrr

    @p_tfr.setter
    def p_tfr(self, val: float) -> None:
        self.p_tftrr = val

    @property
    def d_tfr_max(self) -> float:
        return self.d_tftrr_max

    @d_tfr_max.setter
    def d_tfr_max(self, val: float) -> None:
        self.d_tftrr_max = val


FailLadTransverseFiberTensionRupture = FailLadTransverseFiberTensionRuptureRate
FailLadTransverseFiberTensionRate = FailLadTransverseFiberTensionRuptureRate
FailLadTransverseFiberRuptureRate = FailLadTransverseFiberTensionRuptureRate
FailLadTransverseFiberTensileRuptureRate = FailLadTransverseFiberTensionRuptureRate
FailLadTransverseFiberTensionDamageRate = FailLadTransverseFiberTensionRuptureRate


# ============================================================================
# M457 Suite: FailLadCoupledFiberTensionRuptureRate, EngElectrothermoflexomagnetochiraldyonicplasmonicpolaritonicResonanceEnergy, LagmulTorsionSpinorSpatialLinkageJoint, SensorSpringTorsionalPopRate
# ============================================================================

@dataclass
class FailLadCoupledFiberTensionRuptureRate:
    """``/FAIL/LAD_COUPLED_FIBER_TENSION_RUPTURE_RATE`` or ``/FAIL/LADEVEZE_COUPLED_FIBER_TENSION_RUPTURE_RATE`` (M457): Ladevèze rate-dependent coupled multi-axial fiber tensile damage accumulation, progressive fiber bundle debonding, and coupled tensile rupture failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cftrr0: float = 0.0    # initial fiber tensile rupture threshold stress sigma_cftrr,0
    sigma_cftrrc: float = 1.0    # critical dynamic fiber tensile rupture stress sigma_cftrr,c
    gamma_cftrr: float = 0.0     # dynamic fiber tensile rupture rate sensitivity factor gamma_cftrr
    p_cftrr: float = 1.0         # dynamic fiber tensile rupture rate exponent p_cftrr
    d_cftrr_max: float = 0.999   # maximum allowable fiber tensile damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference

    @property
    def sigma_cftr0(self) -> float:
        return self.sigma_cftrr0

    @sigma_cftr0.setter
    def sigma_cftr0(self, val: float) -> None:
        self.sigma_cftrr0 = val

    @property
    def sigma_cftrc(self) -> float:
        return self.sigma_cftrrc

    @sigma_cftrc.setter
    def sigma_cftrc(self, val: float) -> None:
        self.sigma_cftrrc = val

    @property
    def gamma_cftr(self) -> float:
        return self.gamma_cftrr

    @gamma_cftr.setter
    def gamma_cftr(self, val: float) -> None:
        self.gamma_cftrr = val

    @property
    def p_cftr(self) -> float:
        return self.p_cftrr

    @p_cftr.setter
    def p_cftr(self, val: float) -> None:
        self.p_cftrr = val

    @property
    def d_cftr_max(self) -> float:
        return self.d_cftrr_max

    @d_cftr_max.setter
    def d_cftr_max(self, val: float) -> None:
        self.d_cftrr_max = val

    @property
    def sigma_cft0(self) -> float:
        return self.sigma_cftrr0

    @sigma_cft0.setter
    def sigma_cft0(self, val: float) -> None:
        self.sigma_cftrr0 = val

    @property
    def sigma_cftc(self) -> float:
        return self.sigma_cftrrc

    @sigma_cftc.setter
    def sigma_cftc(self, val: float) -> None:
        self.sigma_cftrrc = val

    @property
    def gamma_cft(self) -> float:
        return self.gamma_cftrr

    @gamma_cft.setter
    def gamma_cft(self, val: float) -> None:
        self.gamma_cftrr = val

    @property
    def p_cft(self) -> float:
        return self.p_cftrr

    @p_cft.setter
    def p_cft(self, val: float) -> None:
        self.p_cftrr = val

    @property
    def d_cft_max(self) -> float:
        return self.d_cftrr_max

    @d_cft_max.setter
    def d_cft_max(self, val: float) -> None:
        self.d_cftrr_max = val

    @property
    def sigma_cfr0(self) -> float:
        return self.sigma_cftrr0

    @sigma_cfr0.setter
    def sigma_cfr0(self, val: float) -> None:
        self.sigma_cftrr0 = val

    @property
    def sigma_cfrc(self) -> float:
        return self.sigma_cftrrc

    @sigma_cfrc.setter
    def sigma_cfrc(self, val: float) -> None:
        self.sigma_cftrrc = val

    @property
    def gamma_cfr(self) -> float:
        return self.gamma_cftrr

    @gamma_cfr.setter
    def gamma_cfr(self, val: float) -> None:
        self.gamma_cftrr = val

    @property
    def p_cfr(self) -> float:
        return self.p_cftrr

    @p_cfr.setter
    def p_cfr(self, val: float) -> None:
        self.p_cftrr = val

    @property
    def d_cfr_max(self) -> float:
        return self.d_cftrr_max

    @d_cfr_max.setter
    def d_cfr_max(self, val: float) -> None:
        self.d_cftrr_max = val


FailLadCoupledFiberTensionRupture = FailLadCoupledFiberTensionRuptureRate
FailLadCoupledFiberTensionRate = FailLadCoupledFiberTensionRuptureRate
FailLadCoupledFiberRuptureRate = FailLadCoupledFiberTensionRuptureRate
FailLadCoupledFiberTensileRuptureRate = FailLadCoupledFiberTensionRuptureRate
FailLadCoupledFiberTensionDamageRate = FailLadCoupledFiberTensionRuptureRate
FailLadCoupleFiberTensionRuptureRate = FailLadCoupledFiberTensionRuptureRate
FailLadCoupleFiberTensionRupture = FailLadCoupledFiberTensionRuptureRate
FailLadCoupleFiberTensionRate = FailLadCoupledFiberTensionRuptureRate
FailLadCoupleFiberRuptureRate = FailLadCoupledFiberTensionRuptureRate
FailLadCoupleFiberTensileRuptureRate = FailLadCoupledFiberTensionRuptureRate
FailLadCoupleFiberTensionDamageRate = FailLadCoupledFiberTensionRuptureRate


# ============================================================================
# M458 Suite: FailLadDynamicFiberCompressionFailureRate, EngElectrothermoflexomagnetochiralaxionicplasmonicpolaritonicResonanceEnergy, LagmulHolonomySpinorSpatialLinkageJoint, SensorSpringTotalAngularPopRate
# ============================================================================

@dataclass
class FailLadDynamicFiberCompressionFailureRate:
    """``/FAIL/LAD_DYNAMIC_FIBER_COMPRESSION_FAILURE_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_FIBER_COMPRESSION_FAILURE_RATE`` (M458): Ladevèze rate-dependent dynamic longitudinal fiber compressive damage accumulation, high-rate microbuckling band propagation, and fiber crushing failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dfcfr0: float = 0.0    # initial fiber compressive failure threshold stress sigma_dfcfr,0
    sigma_dfcfrc: float = 1.0    # critical dynamic fiber compressive failure stress sigma_dfcfr,c
    gamma_dfcfr: float = 0.0     # dynamic fiber compressive failure rate sensitivity factor gamma_dfcfr
    p_dfcfr: float = 1.0         # dynamic fiber compressive failure rate exponent p_dfcfr
    d_dfcfr_max: float = 0.999   # maximum allowable fiber compressive damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference

    @property
    def sigma_dfcf0(self) -> float:
        return self.sigma_dfcfr0

    @sigma_dfcf0.setter
    def sigma_dfcf0(self, val: float) -> None:
        self.sigma_dfcfr0 = val

    @property
    def sigma_dfcfc(self) -> float:
        return self.sigma_dfcfrc

    @sigma_dfcfc.setter
    def sigma_dfcfc(self, val: float) -> None:
        self.sigma_dfcfrc = val

    @property
    def gamma_dfcf(self) -> float:
        return self.gamma_dfcfr

    @gamma_dfcf.setter
    def gamma_dfcf(self, val: float) -> None:
        self.gamma_dfcfr = val

    @property
    def p_dfcf(self) -> float:
        return self.p_dfcfr

    @p_dfcf.setter
    def p_dfcf(self, val: float) -> None:
        self.p_dfcfr = val

    @property
    def d_dfcf_max(self) -> float:
        return self.d_dfcfr_max

    @d_dfcf_max.setter
    def d_dfcf_max(self, val: float) -> None:
        self.d_dfcfr_max = val

    @property
    def sigma_dfc0(self) -> float:
        return self.sigma_dfcfr0

    @sigma_dfc0.setter
    def sigma_dfc0(self, val: float) -> None:
        self.sigma_dfcfr0 = val

    @property
    def sigma_dfcc(self) -> float:
        return self.sigma_dfcfrc

    @sigma_dfcc.setter
    def sigma_dfcc(self, val: float) -> None:
        self.sigma_dfcfrc = val

    @property
    def gamma_dfc(self) -> float:
        return self.gamma_dfcfr

    @gamma_dfc.setter
    def gamma_dfc(self, val: float) -> None:
        self.gamma_dfcfr = val

    @property
    def p_dfc(self) -> float:
        return self.p_dfcfr

    @p_dfc.setter
    def p_dfc(self, val: float) -> None:
        self.p_dfcfr = val

    @property
    def d_dfc_max(self) -> float:
        return self.d_dfcfr_max

    @d_dfc_max.setter
    def d_dfc_max(self, val: float) -> None:
        self.d_dfcfr_max = val

    @property
    def sigma_dfr0(self) -> float:
        return self.sigma_dfcfr0

    @sigma_dfr0.setter
    def sigma_dfr0(self, val: float) -> None:
        self.sigma_dfcfr0 = val

    @property
    def sigma_dfrc(self) -> float:
        return self.sigma_dfcfrc

    @sigma_dfrc.setter
    def sigma_dfrc(self, val: float) -> None:
        self.sigma_dfcfrc = val

    @property
    def gamma_dfr(self) -> float:
        return self.gamma_dfcfr

    @gamma_dfr.setter
    def gamma_dfr(self, val: float) -> None:
        self.gamma_dfcfr = val

    @property
    def p_dfr(self) -> float:
        return self.p_dfcfr

    @p_dfr.setter
    def p_dfr(self, val: float) -> None:
        self.p_dfcfr = val

    @property
    def d_dfr_max(self) -> float:
        return self.d_dfcfr_max

    @d_dfr_max.setter
    def d_dfr_max(self, val: float) -> None:
        self.d_dfcfr_max = val


FailLadDynamicFiberCompressionFailure = FailLadDynamicFiberCompressionFailureRate
FailLadDynamicFiberCompressionRate = FailLadDynamicFiberCompressionFailureRate
FailLadDynamicFiberCompressiveFailureRate = FailLadDynamicFiberCompressionFailureRate
FailLadDynamicFiberCompressiveFailure = FailLadDynamicFiberCompressionFailureRate
FailLadDynamicFiberCompressionDamageRate = FailLadDynamicFiberCompressionFailureRate


# ============================================================================
# M459 Suite: FailLadTransverseFiberCompressionFailureRate, EngElectrothermoflexomagnetochiralmajoranaplasmonicpolaritonicResonanceEnergy, LagmulMonodromySpinorSpatialLinkageJoint, SensorSpringBendingPopRate
# ============================================================================

@dataclass
class FailLadTransverseFiberCompressionFailureRate:
    """``/FAIL/LAD_TRANSVERSE_FIBER_COMPRESSION_FAILURE_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_FIBER_COMPRESSION_FAILURE_RATE`` (M459): Ladevèze rate-dependent transverse out-of-plane fiber compressive damage accumulation, transverse microbuckling band propagation, and transverse fiber crushing failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tfcfr0: float = 0.0    # initial transverse fiber compressive failure threshold stress sigma_tfcfr,0
    sigma_tfcfrc: float = 1.0    # critical dynamic transverse fiber compressive failure stress sigma_tfcfr,c
    gamma_tfcfr: float = 0.0     # dynamic transverse fiber compressive failure rate sensitivity factor gamma_tfcfr
    p_tfcfr: float = 1.0         # dynamic transverse fiber compressive failure rate exponent p_tfcfr
    d_tfcfr_max: float = 0.999   # maximum allowable transverse fiber compressive damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference

    @property
    def sigma_tfcf0(self) -> float:
        return self.sigma_tfcfr0

    @sigma_tfcf0.setter
    def sigma_tfcf0(self, val: float) -> None:
        self.sigma_tfcfr0 = val

    @property
    def sigma_tfcfc(self) -> float:
        return self.sigma_tfcfrc

    @sigma_tfcfc.setter
    def sigma_tfcfc(self, val: float) -> None:
        self.sigma_tfcfrc = val

    @property
    def gamma_tfcf(self) -> float:
        return self.gamma_tfcfr

    @gamma_tfcf.setter
    def gamma_tfcf(self, val: float) -> None:
        self.gamma_tfcfr = val

    @property
    def p_tfcf(self) -> float:
        return self.p_tfcfr

    @p_tfcf.setter
    def p_tfcf(self, val: float) -> None:
        self.p_tfcfr = val

    @property
    def d_tfcf_max(self) -> float:
        return self.d_tfcfr_max

    @d_tfcf_max.setter
    def d_tfcf_max(self, val: float) -> None:
        self.d_tfcfr_max = val

    @property
    def sigma_tfc0(self) -> float:
        return self.sigma_tfcfr0

    @sigma_tfc0.setter
    def sigma_tfc0(self, val: float) -> None:
        self.sigma_tfcfr0 = val

    @property
    def sigma_tfcc(self) -> float:
        return self.sigma_tfcfrc

    @sigma_tfcc.setter
    def sigma_tfcc(self, val: float) -> None:
        self.sigma_tfcfrc = val

    @property
    def gamma_tfc(self) -> float:
        return self.gamma_tfcfr

    @gamma_tfc.setter
    def gamma_tfc(self, val: float) -> None:
        self.gamma_tfcfr = val

    @property
    def p_tfc(self) -> float:
        return self.p_tfcfr

    @p_tfc.setter
    def p_tfc(self, val: float) -> None:
        self.p_tfcfr = val

    @property
    def d_tfc_max(self) -> float:
        return self.d_tfcfr_max

    @d_tfc_max.setter
    def d_tfc_max(self, val: float) -> None:
        self.d_tfcfr_max = val

    @property
    def sigma_tfr0(self) -> float:
        return self.sigma_tfcfr0

    @sigma_tfr0.setter
    def sigma_tfr0(self, val: float) -> None:
        self.sigma_tfcfr0 = val

    @property
    def sigma_tfrc(self) -> float:
        return self.sigma_tfcfrc

    @sigma_tfrc.setter
    def sigma_tfrc(self, val: float) -> None:
        self.sigma_tfcfrc = val

    @property
    def gamma_tfr(self) -> float:
        return self.gamma_tfcfr

    @gamma_tfr.setter
    def gamma_tfr(self, val: float) -> None:
        self.gamma_tfcfr = val

    @property
    def p_tfr(self) -> float:
        return self.p_tfcfr

    @p_tfr.setter
    def p_tfr(self, val: float) -> None:
        self.p_tfcfr = val

    @property
    def d_tfr_max(self) -> float:
        return self.d_tfcfr_max

    @d_tfr_max.setter
    def d_tfr_max(self, val: float) -> None:
        self.d_tfcfr_max = val


FailLadTransverseFiberCompressionFailure = FailLadTransverseFiberCompressionFailureRate
FailLadTransverseFiberCompressionRate = FailLadTransverseFiberCompressionFailureRate
FailLadTransverseFiberCompressiveFailureRate = FailLadTransverseFiberCompressionFailureRate
FailLadTransverseFiberCompressiveFailure = FailLadTransverseFiberCompressionFailureRate
FailLadTransverseFiberCompressionDamageRate = FailLadTransverseFiberCompressionFailureRate


# ============================================================================
# M460 Suite: FailLadCoupledFiberCompressionFailureRate, EngElectrothermoflexomagnetochiralanyonplasmonicpolaritonicResonanceEnergy, LagmulSymplecticSpinorSpatialLinkageJoint, SensorSpringTotalPopRate
# ============================================================================

@dataclass
class FailLadCoupledFiberCompressionFailureRate:
    """``/FAIL/LAD_COUPLED_FIBER_COMPRESSION_FAILURE_RATE`` or ``/FAIL/LADEVEZE_COUPLED_FIBER_COMPRESSION_FAILURE_RATE`` (M460): Ladevèze rate-dependent coupled multi-axial fiber compressive damage accumulation, dynamic microbuckling band propagation, and progressive fiber crushing failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cfcfr0: float = 0.0    # initial coupled fiber compressive failure threshold stress sigma_cfcfr,0
    sigma_cfcfrc: float = 1.0    # critical dynamic coupled fiber compressive failure stress sigma_cfcfr,c
    gamma_cfcfr: float = 0.0     # dynamic coupled fiber compressive failure rate sensitivity factor gamma_cfcfr
    p_cfcfr: float = 1.0         # dynamic coupled fiber compressive failure rate exponent p_cfcfr
    d_cfcfr_max: float = 0.999   # maximum allowable coupled fiber compressive damage index
    ifail_sh: int = 1            # shell element deletion flag
    ifail_so: int = 1            # solid element deletion flag
    fail_id: int = 0             # failure model ID reference

    @property
    def sigma_cfcf0(self) -> float:
        return self.sigma_cfcfr0

    @sigma_cfcf0.setter
    def sigma_cfcf0(self, val: float) -> None:
        self.sigma_cfcfr0 = val

    @property
    def sigma_cfcfc(self) -> float:
        return self.sigma_cfcfrc

    @sigma_cfcfc.setter
    def sigma_cfcfc(self, val: float) -> None:
        self.sigma_cfcfrc = val

    @property
    def gamma_cfcf(self) -> float:
        return self.gamma_cfcfr

    @gamma_cfcf.setter
    def gamma_cfcf(self, val: float) -> None:
        self.gamma_cfcfr = val

    @property
    def p_cfcf(self) -> float:
        return self.p_cfcfr

    @p_cfcf.setter
    def p_cfcf(self, val: float) -> None:
        self.p_cfcfr = val

    @property
    def d_cfcf_max(self) -> float:
        return self.d_cfcfr_max

    @d_cfcf_max.setter
    def d_cfcf_max(self, val: float) -> None:
        self.d_cfcfr_max = val

    @property
    def sigma_cfc0(self) -> float:
        return self.sigma_cfcfr0

    @sigma_cfc0.setter
    def sigma_cfc0(self, val: float) -> None:
        self.sigma_cfcfr0 = val

    @property
    def sigma_cfcc(self) -> float:
        return self.sigma_cfcfrc

    @sigma_cfcc.setter
    def sigma_cfcc(self, val: float) -> None:
        self.sigma_cfcfrc = val

    @property
    def gamma_cfc(self) -> float:
        return self.gamma_cfcfr

    @gamma_cfc.setter
    def gamma_cfc(self, val: float) -> None:
        self.gamma_cfcfr = val

    @property
    def p_cfc(self) -> float:
        return self.p_cfcfr

    @p_cfc.setter
    def p_cfc(self, val: float) -> None:
        self.p_cfcfr = val

    @property
    def d_cfc_max(self) -> float:
        return self.d_cfcfr_max

    @d_cfc_max.setter
    def d_cfc_max(self, val: float) -> None:
        self.d_cfcfr_max = val

    @property
    def sigma_cfr0(self) -> float:
        return self.sigma_cfcfr0

    @sigma_cfr0.setter
    def sigma_cfr0(self, val: float) -> None:
        self.sigma_cfcfr0 = val

    @property
    def sigma_cfrc(self) -> float:
        return self.sigma_cfcfrc

    @sigma_cfrc.setter
    def sigma_cfrc(self, val: float) -> None:
        self.sigma_cfcfrc = val

    @property
    def gamma_cfr(self) -> float:
        return self.gamma_cfcfr

    @gamma_cfr.setter
    def gamma_cfr(self, val: float) -> None:
        self.gamma_cfcfr = val

    @property
    def p_cfr(self) -> float:
        return self.p_cfcfr

    @p_cfr.setter
    def p_cfr(self, val: float) -> None:
        self.p_cfcfr = val

    @property
    def d_cfr_max(self) -> float:
        return self.d_cfcfr_max

    @d_cfr_max.setter
    def d_cfr_max(self, val: float) -> None:
        self.d_cfcfr_max = val


FailLadCoupledFiberCompressionFailure = FailLadCoupledFiberCompressionFailureRate
FailLadCoupledFiberCompressionRate = FailLadCoupledFiberCompressionFailureRate
FailLadCoupledFiberCompressiveFailureRate = FailLadCoupledFiberCompressionFailureRate
FailLadCoupledFiberCompressiveFailure = FailLadCoupledFiberCompressionFailureRate
FailLadCoupledFiberCompressionDamageRate = FailLadCoupledFiberCompressionFailureRate
FailLadCoupleFiberCompressionFailureRate = FailLadCoupledFiberCompressionFailureRate
FailLadCoupleFiberCompressionFailure = FailLadCoupledFiberCompressionFailureRate
FailLadCoupleFiberCompressionRate = FailLadCoupledFiberCompressionFailureRate
FailLadCoupleFiberCompressiveFailureRate = FailLadCoupledFiberCompressionFailureRate
FailLadCoupleFiberCompressiveFailure = FailLadCoupledFiberCompressionFailureRate
FailLadCoupleFiberCompressionDamageRate = FailLadCoupledFiberCompressionFailureRate


# ============================================================================
# M461 Suite: FailLadDynamicInterlaminarShearFailureRate, EngElectrothermoflexomagnetochiralskyrmionplasmonicpolaritonicResonanceEnergy, LagmulPoissonSpinorSpatialLinkageJoint, SensorSpringNormalLockRate
# ============================================================================

@dataclass
class FailLadDynamicInterlaminarShearFailureRate:
    """``/FAIL/LAD_DYNAMIC_INTERLAMINAR_SHEAR_FAILURE_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_SHEAR_FAILURE_RATE`` (M461): Ladevèze rate-dependent dynamic interlaminar shear debonding, mode-II crack acceleration, and interlaminar shear failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_disfr0: float = 0.0   # initial dynamic interlaminar shear failure threshold stress sigma_disfr,0
    sigma_disfrc: float = 1.0   # critical dynamic interlaminar shear failure saturation stress sigma_disfr,c
    gamma_disfr: float = 0.0    # dynamic interlaminar shear failure rate sensitivity factor gamma_disfr
    p_disfr: float = 1.0        # dynamic interlaminar shear failure rate exponent p_disfr
    d_disfr_max: float = 0.999  # maximum allowable dynamic interlaminar shear damage index
    ifail_sh: int = 1           # shell element deletion flag
    ifail_so: int = 1           # solid element deletion flag
    fail_id: int = 0            # failure model ID reference

    @property
    def sigma_disf0(self) -> float:
        return self.sigma_disfr0

    @sigma_disf0.setter
    def sigma_disf0(self, val: float) -> None:
        self.sigma_disfr0 = val

    @property
    def sigma_disfc(self) -> float:
        return self.sigma_disfrc

    @sigma_disfc.setter
    def sigma_disfc(self, val: float) -> None:
        self.sigma_disfrc = val

    @property
    def gamma_disf(self) -> float:
        return self.gamma_disfr

    @gamma_disf.setter
    def gamma_disf(self, val: float) -> None:
        self.gamma_disfr = val

    @property
    def p_disf(self) -> float:
        return self.p_disfr

    @p_disf.setter
    def p_disf(self, val: float) -> None:
        self.p_disfr = val

    @property
    def d_disf_max(self) -> float:
        return self.d_disfr_max

    @d_disf_max.setter
    def d_disf_max(self, val: float) -> None:
        self.d_disfr_max = val

    @property
    def sigma_dis0(self) -> float:
        return self.sigma_disfr0

    @sigma_dis0.setter
    def sigma_dis0(self, val: float) -> None:
        self.sigma_disfr0 = val

    @property
    def sigma_disc(self) -> float:
        return self.sigma_disfrc

    @sigma_disc.setter
    def sigma_disc(self, val: float) -> None:
        self.sigma_disfrc = val

    @property
    def gamma_dis(self) -> float:
        return self.gamma_disfr

    @gamma_dis.setter
    def gamma_dis(self, val: float) -> None:
        self.gamma_disfr = val

    @property
    def p_dis(self) -> float:
        return self.p_disfr

    @p_dis.setter
    def p_dis(self, val: float) -> None:
        self.p_disfr = val

    @property
    def d_dis_max(self) -> float:
        return self.d_disfr_max

    @d_dis_max.setter
    def d_dis_max(self, val: float) -> None:
        self.d_disfr_max = val

    @property
    def sigma_dir0(self) -> float:
        return self.sigma_disfr0

    @sigma_dir0.setter
    def sigma_dir0(self, val: float) -> None:
        self.sigma_disfr0 = val

    @property
    def sigma_dirc(self) -> float:
        return self.sigma_disfrc

    @sigma_dirc.setter
    def sigma_dirc(self, val: float) -> None:
        self.sigma_disfrc = val

    @property
    def gamma_dir(self) -> float:
        return self.gamma_disfr

    @gamma_dir.setter
    def gamma_dir(self, val: float) -> None:
        self.gamma_disfr = val

    @property
    def p_dir(self) -> float:
        return self.p_disfr

    @p_dir.setter
    def p_dir(self, val: float) -> None:
        self.p_disfr = val

    @property
    def d_dir_max(self) -> float:
        return self.d_disfr_max

    @d_dir_max.setter
    def d_dir_max(self, val: float) -> None:
        self.d_disfr_max = val


FailLadDynamicInterlaminarShearFailure = FailLadDynamicInterlaminarShearFailureRate
FailLadDynamicInterlaminarFailureRate = FailLadDynamicInterlaminarShearFailureRate
FailLadDynamicInterlaminarDelaminationFailureRate = FailLadDynamicInterlaminarShearFailureRate
FailLadDynamicInterlaminarShearDelaminationFailureRate = FailLadDynamicInterlaminarShearFailureRate
FailLadDynamicInterlaminarShearDamageRate = FailLadDynamicInterlaminarShearFailureRate
FailLadDynamicInterlaminarDamageRate = FailLadDynamicInterlaminarShearFailureRate


# ============================================================================
# M462 Suite: FailLadTransverseInterlaminarShearFailureRate, EngElectrothermoflexomagnetochiralmeronplasmonicpolaritonicResonanceEnergy, LagmulDiracSpinorSpatialLinkageJoint, SensorSpringTransverseLockRate
# ============================================================================

@dataclass
class FailLadTransverseInterlaminarShearFailureRate:
    """``/FAIL/LAD_TRANSVERSE_INTERLAMINAR_SHEAR_FAILURE_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_SHEAR_FAILURE_RATE`` (M462): Ladevèze rate-dependent transverse interlaminar shear debonding, mode-II crack propagation, and interlaminar shear failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tisfr0: float = 0.0   # initial transverse interlaminar shear failure threshold stress sigma_tisfr,0
    sigma_tisfrc: float = 1.0   # critical transverse interlaminar shear failure saturation stress sigma_tisfr,c
    gamma_tisfr: float = 0.0    # transverse interlaminar shear failure rate sensitivity factor gamma_tisfr
    p_tisfr: float = 1.0        # transverse interlaminar shear failure rate exponent p_tisfr
    d_tisfr_max: float = 0.999  # maximum allowable transverse interlaminar shear damage index
    ifail_sh: int = 1           # shell element deletion flag
    ifail_so: int = 1           # solid element deletion flag
    fail_id: int = 0            # failure model ID reference

    @property
    def sigma_tisf0(self) -> float:
        return self.sigma_tisfr0

    @sigma_tisf0.setter
    def sigma_tisf0(self, val: float) -> None:
        self.sigma_tisfr0 = val

    @property
    def sigma_tisfc(self) -> float:
        return self.sigma_tisfrc

    @sigma_tisfc.setter
    def sigma_tisfc(self, val: float) -> None:
        self.sigma_tisfrc = val

    @property
    def gamma_tisf(self) -> float:
        return self.gamma_tisfr

    @gamma_tisf.setter
    def gamma_tisf(self, val: float) -> None:
        self.gamma_tisfr = val

    @property
    def p_tisf(self) -> float:
        return self.p_tisfr

    @p_tisf.setter
    def p_tisf(self, val: float) -> None:
        self.p_tisfr = val

    @property
    def d_tisf_max(self) -> float:
        return self.d_tisfr_max

    @d_tisf_max.setter
    def d_tisf_max(self, val: float) -> None:
        self.d_tisfr_max = val

    @property
    def sigma_tis0(self) -> float:
        return self.sigma_tisfr0

    @sigma_tis0.setter
    def sigma_tis0(self, val: float) -> None:
        self.sigma_tisfr0 = val

    @property
    def sigma_tisc(self) -> float:
        return self.sigma_tisfrc

    @sigma_tisc.setter
    def sigma_tisc(self, val: float) -> None:
        self.sigma_tisfrc = val

    @property
    def gamma_tis(self) -> float:
        return self.gamma_tisfr

    @gamma_tis.setter
    def gamma_tis(self, val: float) -> None:
        self.gamma_tisfr = val

    @property
    def p_tis(self) -> float:
        return self.p_tisfr

    @p_tis.setter
    def p_tis(self, val: float) -> None:
        self.p_tisfr = val

    @property
    def d_tis_max(self) -> float:
        return self.d_tisfr_max

    @d_tis_max.setter
    def d_tis_max(self, val: float) -> None:
        self.d_tisfr_max = val

    @property
    def sigma_tir0(self) -> float:
        return self.sigma_tisfr0

    @sigma_tir0.setter
    def sigma_tir0(self, val: float) -> None:
        self.sigma_tisfr0 = val

    @property
    def sigma_tirc(self) -> float:
        return self.sigma_tisfrc

    @sigma_tirc.setter
    def sigma_tirc(self, val: float) -> None:
        self.sigma_tisfrc = val

    @property
    def gamma_tir(self) -> float:
        return self.gamma_tisfr

    @gamma_tir.setter
    def gamma_tir(self, val: float) -> None:
        self.gamma_tisfr = val

    @property
    def p_tir(self) -> float:
        return self.p_tisfr

    @p_tir.setter
    def p_tir(self, val: float) -> None:
        self.p_tisfr = val

    @property
    def d_tir_max(self) -> float:
        return self.d_tisfr_max

    @d_tir_max.setter
    def d_tir_max(self, val: float) -> None:
        self.d_tisfr_max = val


FailLadTransverseInterlaminarShearFailure = FailLadTransverseInterlaminarShearFailureRate
FailLadTransverseInterlaminarFailure = FailLadTransverseInterlaminarShearFailureRate
FailLadTransverseInterlaminarFailureRate = FailLadTransverseInterlaminarShearFailureRate
FailLadTransverseInterlaminarDelaminationFailureRate = FailLadTransverseInterlaminarShearFailureRate
FailLadTransverseInterlaminarShearDelaminationFailureRate = FailLadTransverseInterlaminarShearFailureRate
FailLadTransverseInterlaminarShearDamageRate = FailLadTransverseInterlaminarShearFailureRate
FailLadTransverseInterlaminarDamageRate = FailLadTransverseInterlaminarShearFailureRate


# ============================================================================
# M463 Suite: FailLadCoupledInterlaminarShearFailureRate, EngElectrothermoflexomagnetochiralbimeronplasmonicpolaritonicResonanceEnergy, LagmulJacobiSpinorSpatialLinkageJoint, SensorSpringBendingLockRate
# ============================================================================

@dataclass
class FailLadCoupledInterlaminarShearFailureRate:
    """``/FAIL/LAD_COUPLED_INTERLAMINAR_SHEAR_FAILURE_RATE`` or ``/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_SHEAR_FAILURE_RATE`` (M463): Ladevèze rate-dependent coupled multi-axial interlaminar shear debonding, mode-II/III crack acceleration, and interlaminar shear failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cisfr0: float = 0.0   # initial coupled interlaminar shear failure threshold stress sigma_cisfr,0
    sigma_cisfrc: float = 1.0   # critical coupled interlaminar shear failure saturation stress sigma_cisfr,c
    gamma_cisfr: float = 0.0    # coupled interlaminar shear failure rate sensitivity factor gamma_cisfr
    p_cisfr: float = 1.0        # coupled interlaminar shear failure rate exponent p_cisfr
    d_cisfr_max: float = 0.999  # maximum allowable coupled interlaminar shear damage index
    ifail_sh: int = 1           # shell element deletion flag
    ifail_so: int = 1           # solid element deletion flag
    fail_id: int = 0            # failure model ID reference

    @property
    def sigma_cisf0(self) -> float:
        return self.sigma_cisfr0

    @sigma_cisf0.setter
    def sigma_cisf0(self, val: float) -> None:
        self.sigma_cisfr0 = val

    @property
    def sigma_cisfc(self) -> float:
        return self.sigma_cisfrc

    @sigma_cisfc.setter
    def sigma_cisfc(self, val: float) -> None:
        self.sigma_cisfrc = val

    @property
    def gamma_cisf(self) -> float:
        return self.gamma_cisfr

    @gamma_cisf.setter
    def gamma_cisf(self, val: float) -> None:
        self.gamma_cisfr = val

    @property
    def p_cisf(self) -> float:
        return self.p_cisfr

    @p_cisf.setter
    def p_cisf(self, val: float) -> None:
        self.p_cisfr = val

    @property
    def d_cisf_max(self) -> float:
        return self.d_cisfr_max

    @d_cisf_max.setter
    def d_cisf_max(self, val: float) -> None:
        self.d_cisfr_max = val

    @property
    def sigma_cis0(self) -> float:
        return self.sigma_cisfr0

    @sigma_cis0.setter
    def sigma_cis0(self, val: float) -> None:
        self.sigma_cisfr0 = val

    @property
    def sigma_cisc(self) -> float:
        return self.sigma_cisfrc

    @sigma_cisc.setter
    def sigma_cisc(self, val: float) -> None:
        self.sigma_cisfrc = val

    @property
    def gamma_cis(self) -> float:
        return self.gamma_cisfr

    @gamma_cis.setter
    def gamma_cis(self, val: float) -> None:
        self.gamma_cisfr = val

    @property
    def p_cis(self) -> float:
        return self.p_cisfr

    @p_cis.setter
    def p_cis(self, val: float) -> None:
        self.p_cisfr = val

    @property
    def d_cis_max(self) -> float:
        return self.d_cisfr_max

    @d_cis_max.setter
    def d_cis_max(self, val: float) -> None:
        self.d_cisfr_max = val

    @property
    def sigma_cir0(self) -> float:
        return self.sigma_cisfr0

    @sigma_cir0.setter
    def sigma_cir0(self, val: float) -> None:
        self.sigma_cisfr0 = val

    @property
    def sigma_circ(self) -> float:
        return self.sigma_cisfrc

    @sigma_circ.setter
    def sigma_circ(self, val: float) -> None:
        self.sigma_cisfrc = val

    @property
    def gamma_cir(self) -> float:
        return self.gamma_cisfr

    @gamma_cir.setter
    def gamma_cir(self, val: float) -> None:
        self.gamma_cisfr = val

    @property
    def p_cir(self) -> float:
        return self.p_cisfr

    @p_cir.setter
    def p_cir(self, val: float) -> None:
        self.p_cisfr = val

    @property
    def d_cir_max(self) -> float:
        return self.d_cisfr_max

    @d_cir_max.setter
    def d_cir_max(self, val: float) -> None:
        self.d_cisfr_max = val


FailLadCoupledInterlaminarShearFailure = FailLadCoupledInterlaminarShearFailureRate
FailLadCoupleInterlaminarShearFailureRate = FailLadCoupledInterlaminarShearFailureRate
FailLadCoupleInterlaminarShearFailure = FailLadCoupledInterlaminarShearFailureRate
FailLadCoupledInterlaminarFailure = FailLadCoupledInterlaminarShearFailureRate
FailLadCoupleInterlaminarFailure = FailLadCoupledInterlaminarShearFailureRate
FailLadCoupledInterlaminarFailureRate = FailLadCoupledInterlaminarShearFailureRate
FailLadCoupleInterlaminarFailureRate = FailLadCoupledInterlaminarShearFailureRate
FailLadCoupledInterlaminarDelaminationFailureRate = FailLadCoupledInterlaminarShearFailureRate
FailLadCoupledInterlaminarShearDelaminationFailureRate = FailLadCoupledInterlaminarShearFailureRate
FailLadCoupledInterlaminarShearDamageRate = FailLadCoupledInterlaminarShearFailureRate
FailLadCoupledInterlaminarDamageRate = FailLadCoupledInterlaminarShearFailureRate
FailLadCoupleInterlaminarShearDamageRate = FailLadCoupledInterlaminarShearFailureRate
FailLadCoupleInterlaminarDamageRate = FailLadCoupledInterlaminarShearFailureRate


# ============================================================================
# M464 Suite: FailLadDynamicInterlaminarTensionFailureRate, EngElectrothermoflexomagnetochiralinstantonplasmonicpolaritonicResonanceEnergy, LagmulCartanSpinorSpatialLinkageJoint, SensorSpringTorsionalLockRate
# ============================================================================

@dataclass
class FailLadDynamicInterlaminarTensionFailureRate:
    """``/FAIL/LAD_DYNAMIC_INTERLAMINAR_TENSION_FAILURE_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_TENSION_FAILURE_RATE`` (M464): Ladevèze rate-dependent dynamic interlaminar normal tension debonding, mode-I opening crack acceleration, and interlaminar tension failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_ditfr0: float = 0.0   # initial dynamic interlaminar tension failure threshold stress sigma_ditfr,0
    sigma_ditfrc: float = 1.0   # critical dynamic interlaminar tension failure saturation stress sigma_ditfr,c
    gamma_ditfr: float = 0.0    # dynamic interlaminar tension failure rate sensitivity factor gamma_ditfr
    p_ditfr: float = 1.0        # dynamic interlaminar tension failure rate exponent p_ditfr
    d_ditfr_max: float = 0.999  # maximum allowable dynamic interlaminar tension damage index
    ifail_sh: int = 1           # shell element deletion flag
    ifail_so: int = 1           # solid element deletion flag
    fail_id: int = 0            # failure model ID reference

    @property
    def sigma_ditf0(self) -> float:
        return self.sigma_ditfr0

    @sigma_ditf0.setter
    def sigma_ditf0(self, val: float) -> None:
        self.sigma_ditfr0 = val

    @property
    def sigma_ditfc(self) -> float:
        return self.sigma_ditfrc

    @sigma_ditfc.setter
    def sigma_ditfc(self, val: float) -> None:
        self.sigma_ditfrc = val

    @property
    def gamma_ditf(self) -> float:
        return self.gamma_ditfr

    @gamma_ditf.setter
    def gamma_ditf(self, val: float) -> None:
        self.gamma_ditfr = val

    @property
    def p_ditf(self) -> float:
        return self.p_ditfr

    @p_ditf.setter
    def p_ditf(self, val: float) -> None:
        self.p_ditfr = val

    @property
    def d_ditf_max(self) -> float:
        return self.d_ditfr_max

    @d_ditf_max.setter
    def d_ditf_max(self, val: float) -> None:
        self.d_ditfr_max = val

    @property
    def sigma_dit0(self) -> float:
        return self.sigma_ditfr0

    @sigma_dit0.setter
    def sigma_dit0(self, val: float) -> None:
        self.sigma_ditfr0 = val

    @property
    def sigma_ditc(self) -> float:
        return self.sigma_ditfrc

    @sigma_ditc.setter
    def sigma_ditc(self, val: float) -> None:
        self.sigma_ditfrc = val

    @property
    def gamma_dit(self) -> float:
        return self.gamma_ditfr

    @gamma_dit.setter
    def gamma_dit(self, val: float) -> None:
        self.gamma_ditfr = val

    @property
    def p_dit(self) -> float:
        return self.p_ditfr

    @p_dit.setter
    def p_dit(self, val: float) -> None:
        self.p_ditfr = val

    @property
    def d_dit_max(self) -> float:
        return self.d_ditfr_max

    @d_dit_max.setter
    def d_dit_max(self, val: float) -> None:
        self.d_ditfr_max = val

    @property
    def sigma_din0(self) -> float:
        return self.sigma_ditfr0

    @sigma_din0.setter
    def sigma_din0(self, val: float) -> None:
        self.sigma_ditfr0 = val

    @property
    def sigma_dinc(self) -> float:
        return self.sigma_ditfrc

    @sigma_dinc.setter
    def sigma_dinc(self, val: float) -> None:
        self.sigma_ditfrc = val

    @property
    def gamma_din(self) -> float:
        return self.gamma_ditfr

    @gamma_din.setter
    def gamma_din(self, val: float) -> None:
        self.gamma_ditfr = val

    @property
    def p_din(self) -> float:
        return self.p_ditfr

    @p_din.setter
    def p_din(self, val: float) -> None:
        self.p_ditfr = val

    @property
    def d_din_max(self) -> float:
        return self.d_ditfr_max

    @d_din_max.setter
    def d_din_max(self, val: float) -> None:
        self.d_ditfr_max = val


FailLadDynamicInterlaminarTensionFailure = FailLadDynamicInterlaminarTensionFailureRate
FailLadDynamicInterlaminarTensionalFailureRate = FailLadDynamicInterlaminarTensionFailureRate
FailLadDynamicInterlaminarTensionalFailure = FailLadDynamicInterlaminarTensionFailureRate
FailLadDynamicInterlaminarNormalFailureRate = FailLadDynamicInterlaminarTensionFailureRate
FailLadDynamicInterlaminarNormalFailure = FailLadDynamicInterlaminarTensionFailureRate
FailLadDynamicInterlaminarNormalPeelingFailureRate = FailLadDynamicInterlaminarTensionFailureRate
FailLadDynamicInterlaminarPeelingFailureRate = FailLadDynamicInterlaminarTensionFailureRate
FailLadDynamicInterlaminarPeelingFailure = FailLadDynamicInterlaminarTensionFailureRate
FailLadDynamicInterlaminarTensionDamageRate = FailLadDynamicInterlaminarTensionFailureRate
FailLadDynamicInterlaminarNormalDamageRate = FailLadDynamicInterlaminarTensionFailureRate
FailLadDynamicInterlaminarPeelingDamageRate = FailLadDynamicInterlaminarTensionFailureRate


# ============================================================================
# M465 Suite: FailLadTransverseInterlaminarTensionFailureRate, EngElectrothermoflexomagnetochiralsolitonplasmonicpolaritonicResonanceEnergy, LagmulMajoranaSpinorSpatialLinkageJoint, SensorSpringTotalAngularLockRate
# ============================================================================

@dataclass
class FailLadTransverseInterlaminarTensionFailureRate:
    """``/FAIL/LAD_TRANSVERSE_INTERLAMINAR_TENSION_FAILURE_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_TENSION_FAILURE_RATE`` (M465): Ladevèze rate-dependent transverse interlaminar normal tension debonding, transverse crack propagation, and interlaminar tension failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_titfr0: float = 0.0   # initial transverse interlaminar tension failure threshold stress sigma_titfr,0
    sigma_titfrc: float = 1.0   # critical transverse interlaminar tension failure saturation stress sigma_titfr,c
    gamma_titfr: float = 0.0    # transverse interlaminar tension failure rate sensitivity factor gamma_titfr
    p_titfr: float = 1.0        # transverse interlaminar tension failure rate exponent p_titfr
    d_titfr_max: float = 0.999  # maximum allowable transverse interlaminar tension damage index
    ifail_sh: int = 1           # shell element deletion flag
    ifail_so: int = 1           # solid element deletion flag
    fail_id: int = 0            # failure model ID reference

    @property
    def sigma_titf0(self) -> float:
        return self.sigma_titfr0

    @sigma_titf0.setter
    def sigma_titf0(self, val: float) -> None:
        self.sigma_titfr0 = val

    @property
    def sigma_titfc(self) -> float:
        return self.sigma_titfrc

    @sigma_titfc.setter
    def sigma_titfc(self, val: float) -> None:
        self.sigma_titfrc = val

    @property
    def gamma_titf(self) -> float:
        return self.gamma_titfr

    @gamma_titf.setter
    def gamma_titf(self, val: float) -> None:
        self.gamma_titfr = val

    @property
    def p_titf(self) -> float:
        return self.p_titfr

    @p_titf.setter
    def p_titf(self, val: float) -> None:
        self.p_titfr = val

    @property
    def d_titf_max(self) -> float:
        return self.d_titfr_max

    @d_titf_max.setter
    def d_titf_max(self, val: float) -> None:
        self.d_titfr_max = val

    @property
    def sigma_tit0(self) -> float:
        return self.sigma_titfr0

    @sigma_tit0.setter
    def sigma_tit0(self, val: float) -> None:
        self.sigma_titfr0 = val

    @property
    def sigma_titc(self) -> float:
        return self.sigma_titfrc

    @sigma_titc.setter
    def sigma_titc(self, val: float) -> None:
        self.sigma_titfrc = val

    @property
    def gamma_tit(self) -> float:
        return self.gamma_titfr

    @gamma_tit.setter
    def gamma_tit(self, val: float) -> None:
        self.gamma_titfr = val

    @property
    def p_tit(self) -> float:
        return self.p_titfr

    @p_tit.setter
    def p_tit(self, val: float) -> None:
        self.p_titfr = val

    @property
    def d_tit_max(self) -> float:
        return self.d_titfr_max

    @d_tit_max.setter
    def d_tit_max(self, val: float) -> None:
        self.d_titfr_max = val

    @property
    def sigma_tin0(self) -> float:
        return self.sigma_titfr0

    @sigma_tin0.setter
    def sigma_tin0(self, val: float) -> None:
        self.sigma_titfr0 = val

    @property
    def sigma_tinc(self) -> float:
        return self.sigma_titfrc

    @sigma_tinc.setter
    def sigma_tinc(self, val: float) -> None:
        self.sigma_titfrc = val

    @property
    def gamma_tin(self) -> float:
        return self.gamma_titfr

    @gamma_tin.setter
    def gamma_tin(self, val: float) -> None:
        self.gamma_titfr = val

    @property
    def p_tin(self) -> float:
        return self.p_titfr

    @p_tin.setter
    def p_tin(self, val: float) -> None:
        self.p_titfr = val

    @property
    def d_tin_max(self) -> float:
        return self.d_titfr_max

    @d_tin_max.setter
    def d_tin_max(self, val: float) -> None:
        self.d_titfr_max = val


FailLadTransverseInterlaminarTensionFailure = FailLadTransverseInterlaminarTensionFailureRate
FailLadTransverseInterlaminarTensionalFailureRate = FailLadTransverseInterlaminarTensionFailureRate
FailLadTransverseInterlaminarTensionalFailure = FailLadTransverseInterlaminarTensionFailureRate
FailLadTransverseInterlaminarNormalFailureRate = FailLadTransverseInterlaminarTensionFailureRate
FailLadTransverseInterlaminarNormalFailure = FailLadTransverseInterlaminarTensionFailureRate
FailLadTransverseInterlaminarNormalPeelingFailureRate = FailLadTransverseInterlaminarTensionFailureRate
FailLadTransverseInterlaminarPeelingFailureRate = FailLadTransverseInterlaminarTensionFailureRate
FailLadTransverseInterlaminarPeelingFailure = FailLadTransverseInterlaminarTensionFailureRate
FailLadTransverseInterlaminarTensionDamageRate = FailLadTransverseInterlaminarTensionFailureRate
FailLadTransverseInterlaminarNormalDamageRate = FailLadTransverseInterlaminarTensionFailureRate
FailLadTransverseInterlaminarPeelingDamageRate = FailLadTransverseInterlaminarTensionFailureRate


# ============================================================================
# M466 Suite: FailLadCoupledInterlaminarTensionFailureRate, EngElectrothermoflexomagnetochiralvortexplasmonicpolaritonicResonanceEnergy, LagmulKahlerSpinorSpatialLinkageJoint, SensorSpringTotalLockRate
# ============================================================================

@dataclass
class FailLadCoupledInterlaminarTensionFailureRate:
    """``/FAIL/LAD_COUPLED_INTERLAMINAR_TENSION_FAILURE_RATE`` or ``/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_TENSION_FAILURE_RATE`` (M466): Ladevèze rate-dependent coupled multi-axial interlaminar normal tension debonding, mode-I/opening crack acceleration, and interlaminar tension failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_citfr0: float = 0.0   # initial coupled interlaminar tension failure threshold stress sigma_citfr,0
    sigma_citfrc: float = 1.0   # critical coupled interlaminar tension failure saturation stress sigma_citfr,c
    gamma_citfr: float = 0.0    # coupled interlaminar tension failure rate sensitivity factor gamma_citfr
    p_citfr: float = 1.0        # coupled interlaminar tension failure rate exponent p_citfr
    d_citfr_max: float = 0.999  # maximum allowable coupled interlaminar tension damage index
    ifail_sh: int = 1           # shell element deletion flag
    ifail_so: int = 1           # solid element deletion flag
    fail_id: int = 0            # failure model ID reference

    @property
    def sigma_citf0(self) -> float:
        return self.sigma_citfr0

    @sigma_citf0.setter
    def sigma_citf0(self, val: float) -> None:
        self.sigma_citfr0 = val

    @property
    def sigma_citfc(self) -> float:
        return self.sigma_citfrc

    @sigma_citfc.setter
    def sigma_citfc(self, val: float) -> None:
        self.sigma_citfrc = val

    @property
    def gamma_citf(self) -> float:
        return self.gamma_citfr

    @gamma_citf.setter
    def gamma_citf(self, val: float) -> None:
        self.gamma_citfr = val

    @property
    def p_citf(self) -> float:
        return self.p_citfr

    @p_citf.setter
    def p_citf(self, val: float) -> None:
        self.p_citfr = val

    @property
    def d_citf_max(self) -> float:
        return self.d_citfr_max

    @d_citf_max.setter
    def d_citf_max(self, val: float) -> None:
        self.d_citfr_max = val

    @property
    def sigma_cit0(self) -> float:
        return self.sigma_citfr0

    @sigma_cit0.setter
    def sigma_cit0(self, val: float) -> None:
        self.sigma_citfr0 = val

    @property
    def sigma_citc(self) -> float:
        return self.sigma_citfrc

    @sigma_citc.setter
    def sigma_citc(self, val: float) -> None:
        self.sigma_citfrc = val

    @property
    def gamma_cit(self) -> float:
        return self.gamma_citfr

    @gamma_cit.setter
    def gamma_cit(self, val: float) -> None:
        self.gamma_citfr = val

    @property
    def p_cit(self) -> float:
        return self.p_citfr

    @p_cit.setter
    def p_cit(self, val: float) -> None:
        self.p_citfr = val

    @property
    def d_cit_max(self) -> float:
        return self.d_citfr_max

    @d_cit_max.setter
    def d_cit_max(self, val: float) -> None:
        self.d_citfr_max = val

    @property
    def sigma_cin0(self) -> float:
        return self.sigma_citfr0

    @sigma_cin0.setter
    def sigma_cin0(self, val: float) -> None:
        self.sigma_citfr0 = val

    @property
    def sigma_cinc(self) -> float:
        return self.sigma_citfrc

    @sigma_cinc.setter
    def sigma_cinc(self, val: float) -> None:
        self.sigma_citfrc = val

    @property
    def gamma_cin(self) -> float:
        return self.gamma_citfr

    @gamma_cin.setter
    def gamma_cin(self, val: float) -> None:
        self.gamma_citfr = val

    @property
    def p_cin(self) -> float:
        return self.p_citfr

    @p_cin.setter
    def p_cin(self, val: float) -> None:
        self.p_citfr = val

    @property
    def d_cin_max(self) -> float:
        return self.d_citfr_max

    @d_cin_max.setter
    def d_cin_max(self, val: float) -> None:
        self.d_citfr_max = val


FailLadCoupledInterlaminarTensionFailure = FailLadCoupledInterlaminarTensionFailureRate
FailLadCoupleInterlaminarTensionFailureRate = FailLadCoupledInterlaminarTensionFailureRate
FailLadCoupleInterlaminarTensionFailure = FailLadCoupledInterlaminarTensionFailureRate
FailLadCoupledInterlaminarTensionalFailureRate = FailLadCoupledInterlaminarTensionFailureRate
FailLadCoupledInterlaminarTensionalFailure = FailLadCoupledInterlaminarTensionFailureRate
FailLadCoupleInterlaminarTensionalFailureRate = FailLadCoupledInterlaminarTensionFailureRate
FailLadCoupledInterlaminarNormalFailureRate = FailLadCoupledInterlaminarTensionFailureRate
FailLadCoupledInterlaminarNormalFailure = FailLadCoupledInterlaminarTensionFailureRate
FailLadCoupledInterlaminarNormalPeelingFailureRate = FailLadCoupledInterlaminarTensionFailureRate
FailLadCoupledInterlaminarPeelingFailureRate = FailLadCoupledInterlaminarTensionFailureRate
FailLadCoupledInterlaminarPeelingFailure = FailLadCoupledInterlaminarTensionFailureRate
FailLadCoupledInterlaminarTensionDamageRate = FailLadCoupledInterlaminarTensionFailureRate
FailLadCoupleInterlaminarTensionDamageRate = FailLadCoupledInterlaminarTensionFailureRate
FailLadCoupledInterlaminarNormalDamageRate = FailLadCoupledInterlaminarTensionFailureRate
FailLadCoupledInterlaminarPeelingDamageRate = FailLadCoupledInterlaminarTensionFailureRate


# ============================================================================
# M467 Suite: FailLadDynamicMatrixMicrocrackingFailureRate, EngElectrothermoflexomagnetochiralhopfionplasmonicpolaritonicResonanceEnergy, LagmulKostantSpinorSpatialLinkageJoint, SensorSpringNormalSnapRate
# ============================================================================

@dataclass
class FailLadDynamicMatrixMicrocrackingFailureRate:
    """``/FAIL/LAD_DYNAMIC_MATRIX_MICROCRACKING_FAILURE_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_MATRIX_MICROCRACKING_FAILURE_RATE`` (M467): Ladevèze rate-dependent dynamic transverse matrix micro-cracking damage accumulation, diffuse microcrack coalescence, and transverse ply degradation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_dmmcfr0: float = 0.0  # initial dynamic matrix microcracking failure threshold stress sigma_dmmcfr,0
    sigma_dmmcfrc: float = 1.0  # critical dynamic matrix microcracking failure saturation stress sigma_dmmcfr,c
    gamma_dmmcfr: float = 0.0   # dynamic matrix microcracking failure rate sensitivity factor gamma_dmmcfr
    p_dmmcfr: float = 1.0       # dynamic matrix microcracking failure rate exponent p_dmmcfr
    d_dmmcfr_max: float = 0.999 # maximum allowable dynamic matrix microcracking damage index
    ifail_sh: int = 1           # shell element deletion flag
    ifail_so: int = 1           # solid element deletion flag
    fail_id: int = 0            # failure model ID reference

    @property
    def sigma_dmmcf0(self) -> float:
        return self.sigma_dmmcfr0

    @sigma_dmmcf0.setter
    def sigma_dmmcf0(self, val: float) -> None:
        self.sigma_dmmcfr0 = val

    @property
    def sigma_dmmcfc(self) -> float:
        return self.sigma_dmmcfrc

    @sigma_dmmcfc.setter
    def sigma_dmmcfc(self, val: float) -> None:
        self.sigma_dmmcfrc = val

    @property
    def gamma_dmmcf(self) -> float:
        return self.gamma_dmmcfr

    @gamma_dmmcf.setter
    def gamma_dmmcf(self, val: float) -> None:
        self.gamma_dmmcfr = val

    @property
    def p_dmmcf(self) -> float:
        return self.p_dmmcfr

    @p_dmmcf.setter
    def p_dmmcf(self, val: float) -> None:
        self.p_dmmcfr = val

    @property
    def d_dmmcf_max(self) -> float:
        return self.d_dmmcfr_max

    @d_dmmcf_max.setter
    def d_dmmcf_max(self, val: float) -> None:
        self.d_dmmcfr_max = val

    @property
    def sigma_dmmc0(self) -> float:
        return self.sigma_dmmcfr0

    @sigma_dmmc0.setter
    def sigma_dmmc0(self, val: float) -> None:
        self.sigma_dmmcfr0 = val

    @property
    def sigma_dmmcc(self) -> float:
        return self.sigma_dmmcfrc

    @sigma_dmmcc.setter
    def sigma_dmmcc(self, val: float) -> None:
        self.sigma_dmmcfrc = val

    @property
    def gamma_dmmc(self) -> float:
        return self.gamma_dmmcfr

    @gamma_dmmc.setter
    def gamma_dmmc(self, val: float) -> None:
        self.gamma_dmmcfr = val

    @property
    def p_dmmc(self) -> float:
        return self.p_dmmcfr

    @p_dmmc.setter
    def p_dmmc(self, val: float) -> None:
        self.p_dmmcfr = val

    @property
    def d_dmmc_max(self) -> float:
        return self.d_dmmcfr_max

    @d_dmmc_max.setter
    def d_dmmc_max(self, val: float) -> None:
        self.d_dmmcfr_max = val

    @property
    def sigma_dmc0(self) -> float:
        return self.sigma_dmmcfr0

    @sigma_dmc0.setter
    def sigma_dmc0(self, val: float) -> None:
        self.sigma_dmmcfr0 = val

    @property
    def sigma_dmcc(self) -> float:
        return self.sigma_dmmcfrc

    @sigma_dmcc.setter
    def sigma_dmcc(self, val: float) -> None:
        self.sigma_dmmcfrc = val

    @property
    def gamma_dmc(self) -> float:
        return self.gamma_dmmcfr

    @gamma_dmc.setter
    def gamma_dmc(self, val: float) -> None:
        self.gamma_dmmcfr = val

    @property
    def p_dmc(self) -> float:
        return self.p_dmmcfr

    @p_dmc.setter
    def p_dmc(self, val: float) -> None:
        self.p_dmmcfr = val

    @property
    def d_dmc_max(self) -> float:
        return self.d_dmmcfr_max

    @d_dmc_max.setter
    def d_dmc_max(self, val: float) -> None:
        self.d_dmmcfr_max = val


FailLadDynamicMatrixMicrocrackingFailure = FailLadDynamicMatrixMicrocrackingFailureRate
FailLadDynamicMatrixCrackingFailureRate = FailLadDynamicMatrixMicrocrackingFailureRate
FailLadDynamicMatrixCrackingFailure = FailLadDynamicMatrixMicrocrackingFailureRate
FailLadDynamicTransverseMatrixMicrocrackingFailureRate = FailLadDynamicMatrixMicrocrackingFailureRate
FailLadDynamicTransverseMatrixMicrocrackingFailure = FailLadDynamicMatrixMicrocrackingFailureRate
FailLadDynamicTransverseMatrixCrackingFailureRate = FailLadDynamicMatrixMicrocrackingFailureRate
FailLadDynamicMatrixMicroCrackingFailureRate = FailLadDynamicMatrixMicrocrackingFailureRate
FailLadDynamicMatrixMicroCrackingFailure = FailLadDynamicMatrixMicrocrackingFailureRate
FailLadDynamicMatrixMicrocrackingDamageRate = FailLadDynamicMatrixMicrocrackingFailureRate
FailLadDynamicMatrixCrackingDamageRate = FailLadDynamicMatrixMicrocrackingFailureRate
FailLadDynamicMatrixMicroCrackingDamageRate = FailLadDynamicMatrixMicrocrackingFailureRate


# ============================================================================
# M468 Suite: FailLadTransverseMatrixMicrocrackingFailureRate, EngElectrothermoflexomagnetochiralmonopoleplasmonicpolaritonicResonanceEnergy, LagmulSouriauSpinorSpatialLinkageJoint, SensorSpringTransverseSnapRate
# ============================================================================

@dataclass
class FailLadTransverseMatrixMicrocrackingFailureRate:
    """``/FAIL/LAD_TRANSVERSE_MATRIX_MICROCRACKING_FAILURE_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_MATRIX_MICROCRACKING_FAILURE_RATE`` (M468): Ladevèze rate-dependent transverse matrix micro-cracking damage accumulation, diffuse microcrack coalescence, and transverse ply degradation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_tmmcfr0: float = 0.0  # initial transverse matrix microcracking failure threshold stress sigma_tmmcfr,0
    sigma_tmmcfrc: float = 1.0  # critical transverse matrix microcracking failure saturation stress sigma_tmmcfr,c
    gamma_tmmcfr: float = 0.0   # transverse matrix microcracking failure rate sensitivity factor gamma_tmmcfr
    p_tmmcfr: float = 1.0       # transverse matrix microcracking failure rate exponent p_tmmcfr
    d_tmmcfr_max: float = 0.999 # maximum allowable transverse matrix microcracking damage index
    ifail_sh: int = 1           # shell element deletion flag
    ifail_so: int = 1           # solid element deletion flag
    fail_id: int = 0            # failure model ID reference

    @property
    def sigma_tmmcf0(self) -> float:
        return self.sigma_tmmcfr0

    @sigma_tmmcf0.setter
    def sigma_tmmcf0(self, val: float) -> None:
        self.sigma_tmmcfr0 = val

    @property
    def sigma_tmmcfc(self) -> float:
        return self.sigma_tmmcfrc

    @sigma_tmmcfc.setter
    def sigma_tmmcfc(self, val: float) -> None:
        self.sigma_tmmcfrc = val

    @property
    def gamma_tmmcf(self) -> float:
        return self.gamma_tmmcfr

    @gamma_tmmcf.setter
    def gamma_tmmcf(self, val: float) -> None:
        self.gamma_tmmcfr = val

    @property
    def p_tmmcf(self) -> float:
        return self.p_tmmcfr

    @p_tmmcf.setter
    def p_tmmcf(self, val: float) -> None:
        self.p_tmmcfr = val

    @property
    def d_tmmcf_max(self) -> float:
        return self.d_tmmcfr_max

    @d_tmmcf_max.setter
    def d_tmmcf_max(self, val: float) -> None:
        self.d_tmmcfr_max = val

    @property
    def sigma_tmmc0(self) -> float:
        return self.sigma_tmmcfr0

    @sigma_tmmc0.setter
    def sigma_tmmc0(self, val: float) -> None:
        self.sigma_tmmcfr0 = val

    @property
    def sigma_tmmcc(self) -> float:
        return self.sigma_tmmcfrc

    @sigma_tmmcc.setter
    def sigma_tmmcc(self, val: float) -> None:
        self.sigma_tmmcfrc = val

    @property
    def gamma_tmmc(self) -> float:
        return self.gamma_tmmcfr

    @gamma_tmmc.setter
    def gamma_tmmc(self, val: float) -> None:
        self.gamma_tmmcfr = val

    @property
    def p_tmmc(self) -> float:
        return self.p_tmmcfr

    @p_tmmc.setter
    def p_tmmc(self, val: float) -> None:
        self.p_tmmcfr = val

    @property
    def d_tmmc_max(self) -> float:
        return self.d_tmmcfr_max

    @d_tmmc_max.setter
    def d_tmmc_max(self, val: float) -> None:
        self.d_tmmcfr_max = val

    @property
    def sigma_tmc0(self) -> float:
        return self.sigma_tmmcfr0

    @sigma_tmc0.setter
    def sigma_tmc0(self, val: float) -> None:
        self.sigma_tmmcfr0 = val

    @property
    def sigma_tmcc(self) -> float:
        return self.sigma_tmmcfrc

    @sigma_tmcc.setter
    def sigma_tmcc(self, val: float) -> None:
        self.sigma_tmmcfrc = val

    @property
    def gamma_tmc(self) -> float:
        return self.gamma_tmmcfr

    @gamma_tmc.setter
    def gamma_tmc(self, val: float) -> None:
        self.gamma_tmmcfr = val

    @property
    def p_tmc(self) -> float:
        return self.p_tmmcfr

    @p_tmc.setter
    def p_tmc(self, val: float) -> None:
        self.p_tmmcfr = val

    @property
    def d_tmc_max(self) -> float:
        return self.d_tmmcfr_max

    @d_tmc_max.setter
    def d_tmc_max(self, val: float) -> None:
        self.d_tmmcfr_max = val


FailLadTransverseMatrixMicrocrackingFailure = FailLadTransverseMatrixMicrocrackingFailureRate
FailLadTransverseMatrixCrackingFailureRate = FailLadTransverseMatrixMicrocrackingFailureRate
FailLadTransverseMatrixCrackingFailure = FailLadTransverseMatrixMicrocrackingFailureRate
FailLadTransverseMatrixMicroCrackingFailureRate = FailLadTransverseMatrixMicrocrackingFailureRate
FailLadTransverseMatrixMicroCrackingFailure = FailLadTransverseMatrixMicrocrackingFailureRate
FailLadTransverseMatrixMicrocrackingDamageRate = FailLadTransverseMatrixMicrocrackingFailureRate
FailLadTransverseMatrixCrackingDamageRate = FailLadTransverseMatrixMicrocrackingFailureRate
FailLadTransverseMatrixMicroCrackingDamageRate = FailLadTransverseMatrixMicrocrackingFailureRate
FailLadTMMCFR = FailLadTransverseMatrixMicrocrackingFailureRate
FailLadTMMCFRModel = FailLadTransverseMatrixMicrocrackingFailureRate
FailLadTMMCFRDamageRate = FailLadTransverseMatrixMicrocrackingFailureRate


# ============================================================================
# M469 Suite: FailLadCoupledMatrixMicrocrackingFailureRate, EngElectrothermoflexomagnetochiralsphaleronplasmonicpolaritonicResonanceEnergy, LagmulBerezinSpinorSpatialLinkageJoint, SensorSpringBendingSnapRate
# ============================================================================

@dataclass
class FailLadCoupledMatrixMicrocrackingFailureRate:
    """``/FAIL/LAD_COUPLED_MATRIX_MICROCRACKING_FAILURE_RATE`` or ``/FAIL/LADEVEZE_COUPLED_MATRIX_MICROCRACKING_FAILURE_RATE`` (M469): Ladevèze rate-dependent coupled multi-axial matrix micro-cracking damage accumulation, diffuse microcrack coalescence, and coupled transverse ply degradation failure model."""
    mat_id: int = 0
    title: str = ""
    sigma_cmmcfr0: float = 0.0  # initial coupled matrix microcracking failure threshold stress sigma_cmmcfr,0
    sigma_cmmcfrc: float = 1.0  # critical coupled matrix microcracking failure saturation stress sigma_cmmcfr,c
    gamma_cmmcfr: float = 0.0   # coupled matrix microcracking failure rate sensitivity factor gamma_cmmcfr
    p_cmmcfr: float = 1.0       # coupled matrix microcracking failure rate exponent p_cmmcfr
    d_cmmcfr_max: float = 0.999 # maximum allowable coupled matrix microcracking damage index
    ifail_sh: int = 1           # shell element deletion flag
    ifail_so: int = 1           # solid element deletion flag
    fail_id: int = 0            # failure model ID reference

    @property
    def sigma_cmmcf0(self) -> float:
        return self.sigma_cmmcfr0

    @sigma_cmmcf0.setter
    def sigma_cmmcf0(self, val: float) -> None:
        self.sigma_cmmcfr0 = val

    @property
    def sigma_cmmcfc(self) -> float:
        return self.sigma_cmmcfrc

    @sigma_cmmcfc.setter
    def sigma_cmmcfc(self, val: float) -> None:
        self.sigma_cmmcfrc = val

    @property
    def gamma_cmmcf(self) -> float:
        return self.gamma_cmmcfr

    @gamma_cmmcf.setter
    def gamma_cmmcf(self, val: float) -> None:
        self.gamma_cmmcfr = val

    @property
    def p_cmmcf(self) -> float:
        return self.p_cmmcfr

    @p_cmmcf.setter
    def p_cmmcf(self, val: float) -> None:
        self.p_cmmcfr = val

    @property
    def d_cmmcf_max(self) -> float:
        return self.d_cmmcfr_max

    @d_cmmcf_max.setter
    def d_cmmcf_max(self, val: float) -> None:
        self.d_cmmcfr_max = val

    @property
    def sigma_cmmc0(self) -> float:
        return self.sigma_cmmcfr0

    @sigma_cmmc0.setter
    def sigma_cmmc0(self, val: float) -> None:
        self.sigma_cmmcfr0 = val

    @property
    def sigma_cmmcc(self) -> float:
        return self.sigma_cmmcfrc

    @sigma_cmmcc.setter
    def sigma_cmmcc(self, val: float) -> None:
        self.sigma_cmmcfrc = val

    @property
    def gamma_cmmc(self) -> float:
        return self.gamma_cmmcfr

    @gamma_cmmc.setter
    def gamma_cmmc(self, val: float) -> None:
        self.gamma_cmmcfr = val

    @property
    def p_cmmc(self) -> float:
        return self.p_cmmcfr

    @p_cmmc.setter
    def p_cmmc(self, val: float) -> None:
        self.p_cmmcfr = val

    @property
    def d_cmmc_max(self) -> float:
        return self.d_cmmcfr_max

    @d_cmmc_max.setter
    def d_cmmc_max(self, val: float) -> None:
        self.d_cmmcfr_max = val

    @property
    def sigma_cmc0(self) -> float:
        return self.sigma_cmmcfr0

    @sigma_cmc0.setter
    def sigma_cmc0(self, val: float) -> None:
        self.sigma_cmmcfr0 = val

    @property
    def sigma_cmcc(self) -> float:
        return self.sigma_cmmcfrc

    @sigma_cmcc.setter
    def sigma_cmcc(self, val: float) -> None:
        self.sigma_cmmcfrc = val

    @property
    def gamma_cmc(self) -> float:
        return self.gamma_cmmcfr

    @gamma_cmc.setter
    def gamma_cmc(self, val: float) -> None:
        self.gamma_cmmcfr = val

    @property
    def p_cmc(self) -> float:
        return self.p_cmmcfr

    @p_cmc.setter
    def p_cmc(self, val: float) -> None:
        self.p_cmmcfr = val

    @property
    def d_cmc_max(self) -> float:
        return self.d_cmmcfr_max

    @d_cmc_max.setter
    def d_cmc_max(self, val: float) -> None:
        self.d_cmmcfr_max = val


FailLadCoupledMatrixMicrocrackingFailure = FailLadCoupledMatrixMicrocrackingFailureRate
FailLadCoupledMatrixCrackingFailureRate = FailLadCoupledMatrixMicrocrackingFailureRate
FailLadCoupledMatrixCrackingFailure = FailLadCoupledMatrixMicrocrackingFailureRate
FailLadCoupledMatrixMicroCrackingFailureRate = FailLadCoupledMatrixMicrocrackingFailureRate
FailLadCoupledMatrixMicroCrackingFailure = FailLadCoupledMatrixMicrocrackingFailureRate
FailLadCoupledMatrixMicrocrackingDamageRate = FailLadCoupledMatrixMicrocrackingFailureRate
FailLadCoupledMatrixCrackingDamageRate = FailLadCoupledMatrixMicrocrackingFailureRate
FailLadCoupledMatrixMicroCrackingDamageRate = FailLadCoupledMatrixMicrocrackingFailureRate
FailLadCMMCFR = FailLadCoupledMatrixMicrocrackingFailureRate
FailLadCMMCFRModel = FailLadCoupledMatrixMicrocrackingFailureRate
FailLadCMMCFRDamageRate = FailLadCoupledMatrixMicrocrackingFailureRate


@dataclass
class FailLadDynamicMatrixCrushingFailureRate:
    """``/FAIL/LAD_DYNAMIC_MATRIX_CRUSHING_FAILURE_RATE`` or ``/FAIL/LADEVEZE_DYNAMIC_MATRIX_CRUSHING_FAILURE_RATE`` (M470): Ladevèze rate-dependent dynamic matrix crushing failure rate, dynamic matrix micro-crushing failure, and dynamic compressive damage evolution model."""
    mat_id: int = 0
    title: str = ""
    sigma_dmcrfr0: float = 0.0     # initial dynamic matrix crushing failure rate threshold stress sigma_dmcrfr,0
    sigma_dmcrfrc: float = 1.0     # characteristic critical dynamic matrix crushing failure rate stress sigma_dmcrfr,c
    gamma_dmcrfr: float = 0.0      # dynamic matrix crushing damage rate coefficient gamma_dmcrfr
    p_dmcrfr: float = 1.0          # dynamic matrix crushing damage rate exponent p_dmcrfr
    d_dmcrfr_max: float = 0.999    # maximum dynamic matrix crushing failure rate damage limit d_dmcrfr,max
    ifail_sh: int = 1              # shell post-failure treatment flag
    ifail_so: int = 1              # solid post-failure treatment flag
    fail_id: int = 0               # failure model ID

    @property
    def sigma_dmcrf0(self) -> float:
        return self.sigma_dmcrfr0

    @sigma_dmcrf0.setter
    def sigma_dmcrf0(self, val: float) -> None:
        self.sigma_dmcrfr0 = val

    @property
    def sigma_dmcrfc(self) -> float:
        return self.sigma_dmcrfrc

    @sigma_dmcrfc.setter
    def sigma_dmcrfc(self, val: float) -> None:
        self.sigma_dmcrfrc = val

    @property
    def gamma_dmcrf(self) -> float:
        return self.gamma_dmcrfr

    @gamma_dmcrf.setter
    def gamma_dmcrf(self, val: float) -> None:
        self.gamma_dmcrfr = val

    @property
    def p_dmcrf(self) -> float:
        return self.p_dmcrfr

    @p_dmcrf.setter
    def p_dmcrf(self, val: float) -> None:
        self.p_dmcrfr = val

    @property
    def d_dmcrf_max(self) -> float:
        return self.d_dmcrfr_max

    @d_dmcrf_max.setter
    def d_dmcrf_max(self, val: float) -> None:
        self.d_dmcrfr_max = val

    @property
    def sigma_dmcr0(self) -> float:
        return self.sigma_dmcrfr0

    @sigma_dmcr0.setter
    def sigma_dmcr0(self, val: float) -> None:
        self.sigma_dmcrfr0 = val

    @property
    def sigma_dmcrc(self) -> float:
        return self.sigma_dmcrfrc

    @sigma_dmcrc.setter
    def sigma_dmcrc(self, val: float) -> None:
        self.sigma_dmcrfrc = val

    @property
    def gamma_dmcr(self) -> float:
        return self.gamma_dmcrfr

    @gamma_dmcr.setter
    def gamma_dmcr(self, val: float) -> None:
        self.gamma_dmcrfr = val

    @property
    def p_dmcr(self) -> float:
        return self.p_dmcrfr

    @p_dmcr.setter
    def p_dmcr(self, val: float) -> None:
        self.p_dmcrfr = val

    @property
    def d_dmcr_max(self) -> float:
        return self.d_dmcrfr_max

    @d_dmcr_max.setter
    def d_dmcr_max(self, val: float) -> None:
        self.d_dmcrfr_max = val

    @property
    def sigma_dmc0(self) -> float:
        return self.sigma_dmcrfr0

    @sigma_dmc0.setter
    def sigma_dmc0(self, val: float) -> None:
        self.sigma_dmcrfr0 = val

    @property
    def sigma_dmcc(self) -> float:
        return self.sigma_dmcrfrc

    @sigma_dmcc.setter
    def sigma_dmcc(self, val: float) -> None:
        self.sigma_dmcrfrc = val

    @property
    def gamma_dmc(self) -> float:
        return self.gamma_dmcrfr

    @gamma_dmc.setter
    def gamma_dmc(self, val: float) -> None:
        self.gamma_dmcrfr = val

    @property
    def p_dmc(self) -> float:
        return self.p_dmcrfr

    @p_dmc.setter
    def p_dmc(self, val: float) -> None:
        self.p_dmcrfr = val

    @property
    def d_dmc_max(self) -> float:
        return self.d_dmcrfr_max

    @d_dmc_max.setter
    def d_dmc_max(self, val: float) -> None:
        self.d_dmcrfr_max = val


FailLadDynamicMatrixCrushingFailure = FailLadDynamicMatrixCrushingFailureRate
FailLadDynamicMatrixCrushFailureRate = FailLadDynamicMatrixCrushingFailureRate
FailLadDynamicMatrixCrushFailure = FailLadDynamicMatrixCrushingFailureRate
FailLadDynamicTransverseMatrixCrushingFailureRate = FailLadDynamicMatrixCrushingFailureRate
FailLadDynamicTransverseMatrixCrushingFailure = FailLadDynamicMatrixCrushingFailureRate
FailLadDynamicTransverseMatrixCrushFailureRate = FailLadDynamicMatrixCrushingFailureRate
FailLadDynamicMatrixMicroCrushingFailureRate = FailLadDynamicMatrixCrushingFailureRate
FailLadDynamicMatrixMicroCrushingFailure = FailLadDynamicMatrixCrushingFailureRate
FailLadDmcrfr = FailLadDynamicMatrixCrushingFailureRate
FailLadDmcrfrModel = FailLadDynamicMatrixCrushingFailureRate
FailLadDmcrfrLaw = FailLadDynamicMatrixCrushingFailureRate
FailLadevezeRateDependentDynamicMatrixCrushingFailure = FailLadDynamicMatrixCrushingFailureRate
FailLadDynamicMatrixCrushingDamageRate = FailLadDynamicMatrixCrushingFailureRate
FailLadDynamicMatrixCrushDamageRate = FailLadDynamicMatrixCrushingFailureRate
FailLadDynamicMatrixMicroCrushingDamageRate = FailLadDynamicMatrixCrushingFailureRate


# ============================================================================
# M471 Suite: FailLadTransverseMatrixCrushingFailureRate, EngElectrothermoflexomagnetochiralbobberplasmonicpolaritonicResonanceEnergy, LagmulBottSpinorSpatialLinkageJoint, SensorSpringTotalAngularSnapRate
# ============================================================================

@dataclass
class FailLadTransverseMatrixCrushingFailureRate:
    """``/FAIL/LAD_TRANSVERSE_MATRIX_CRUSHING_FAILURE_RATE`` or ``/FAIL/LADEVEZE_TRANSVERSE_MATRIX_CRUSHING_FAILURE_RATE`` (M471): Ladevèze rate-dependent transverse matrix crushing failure rate, transverse matrix micro-crushing failure, and transverse compressive damage evolution model."""
    mat_id: int = 0
    title: str = ""
    sigma_tmcrfr0: float = 0.0     # initial transverse matrix crushing failure rate threshold stress sigma_tmcrfr,0
    sigma_tmcrfrc: float = 1.0     # characteristic critical transverse matrix crushing failure rate stress sigma_tmcrfr,c
    gamma_tmcrfr: float = 0.0      # transverse matrix crushing damage rate coefficient gamma_tmcrfr
    p_tmcrfr: float = 1.0          # transverse matrix crushing damage rate exponent p_tmcrfr
    d_tmcrfr_max: float = 0.999    # maximum transverse matrix crushing failure rate damage limit d_tmcrfr,max
    ifail_sh: int = 1              # shell post-failure treatment flag
    ifail_so: int = 1              # solid post-failure treatment flag
    fail_id: int = 0               # failure model ID

    @property
    def sigma_tmcrf0(self) -> float:
        return self.sigma_tmcrfr0

    @sigma_tmcrf0.setter
    def sigma_tmcrf0(self, val: float) -> None:
        self.sigma_tmcrfr0 = val

    @property
    def sigma_tmcrfc(self) -> float:
        return self.sigma_tmcrfrc

    @sigma_tmcrfc.setter
    def sigma_tmcrfc(self, val: float) -> None:
        self.sigma_tmcrfrc = val

    @property
    def gamma_tmcrf(self) -> float:
        return self.gamma_tmcrfr

    @gamma_tmcrf.setter
    def gamma_tmcrf(self, val: float) -> None:
        self.gamma_tmcrfr = val

    @property
    def p_tmcrf(self) -> float:
        return self.p_tmcrfr

    @p_tmcrf.setter
    def p_tmcrf(self, val: float) -> None:
        self.p_tmcrfr = val

    @property
    def d_tmcrf_max(self) -> float:
        return self.d_tmcrfr_max

    @d_tmcrf_max.setter
    def d_tmcrf_max(self, val: float) -> None:
        self.d_tmcrfr_max = val

    @property
    def sigma_tmcr0(self) -> float:
        return self.sigma_tmcrfr0

    @sigma_tmcr0.setter
    def sigma_tmcr0(self, val: float) -> None:
        self.sigma_tmcrfr0 = val

    @property
    def sigma_tmcrc(self) -> float:
        return self.sigma_tmcrfrc

    @sigma_tmcrc.setter
    def sigma_tmcrc(self, val: float) -> None:
        self.sigma_tmcrfrc = val

    @property
    def gamma_tmcr(self) -> float:
        return self.gamma_tmcrfr

    @gamma_tmcr.setter
    def gamma_tmcr(self, val: float) -> None:
        self.gamma_tmcrfr = val

    @property
    def p_tmcr(self) -> float:
        return self.p_tmcrfr

    @p_tmcr.setter
    def p_tmcr(self, val: float) -> None:
        self.p_tmcrfr = val

    @property
    def d_tmcr_max(self) -> float:
        return self.d_tmcrfr_max

    @d_tmcr_max.setter
    def d_tmcr_max(self, val: float) -> None:
        self.d_tmcrfr_max = val

    @property
    def sigma_tmc0(self) -> float:
        return self.sigma_tmcrfr0

    @sigma_tmc0.setter
    def sigma_tmc0(self, val: float) -> None:
        self.sigma_tmcrfr0 = val

    @property
    def sigma_tmcc(self) -> float:
        return self.sigma_tmcrfrc

    @sigma_tmcc.setter
    def sigma_tmcc(self, val: float) -> None:
        self.sigma_tmcrfrc = val

    @property
    def gamma_tmc(self) -> float:
        return self.gamma_tmcrfr

    @gamma_tmc.setter
    def gamma_tmc(self, val: float) -> None:
        self.gamma_tmcrfr = val

    @property
    def p_tmc(self) -> float:
        return self.p_tmcrfr

    @p_tmc.setter
    def p_tmc(self, val: float) -> None:
        self.p_tmcrfr = val

    @property
    def d_tmc_max(self) -> float:
        return self.d_tmcrfr_max

    @d_tmc_max.setter
    def d_tmc_max(self, val: float) -> None:
        self.d_tmcrfr_max = val


FailLadTransverseMatrixCrushingFailure = FailLadTransverseMatrixCrushingFailureRate
FailLadTransverseMatrixCrushFailureRate = FailLadTransverseMatrixCrushingFailureRate
FailLadTransverseMatrixCrushFailure = FailLadTransverseMatrixCrushingFailureRate
FailLadTransverseMatrixMicroCrushingFailureRate = FailLadTransverseMatrixCrushingFailureRate
FailLadTransverseMatrixMicroCrushingFailure = FailLadTransverseMatrixCrushingFailureRate
FailLadTransverseMatrixMicrocrushingFailureRate = FailLadTransverseMatrixCrushingFailureRate
FailLadTransverseMatrixMicrocrushingFailure = FailLadTransverseMatrixCrushingFailureRate
FailLadTmcrfr = FailLadTransverseMatrixCrushingFailureRate
FailLadTmcrfrModel = FailLadTransverseMatrixCrushingFailureRate
FailLadTmcrfrLaw = FailLadTransverseMatrixCrushingFailureRate
FailLadevezeRateDependentTransverseMatrixCrushingFailure = FailLadTransverseMatrixCrushingFailureRate
FailLadTransverseMatrixCrushingDamageRate = FailLadTransverseMatrixCrushingFailureRate
FailLadTransverseMatrixCrushDamageRate = FailLadTransverseMatrixCrushingFailureRate
FailLadTransverseMatrixMicroCrushingDamageRate = FailLadTransverseMatrixCrushingFailureRate
FailLadTransverseMatrixMicrocrushingDamageRate = FailLadTransverseMatrixCrushingFailureRate


# ============================================================================
# M472 Suite: FailLadCoupledMatrixCrushingFailureRate, EngElectrothermoflexomagnetochiralblochpointplasmonicpolaritonicResonanceEnergy, LagmulChernSpinorSpatialLinkageJoint, SensorSpringTotalSnapRate
# ============================================================================

@dataclass
class FailLadCoupledMatrixCrushingFailureRate:
    """``/FAIL/LAD_COUPLED_MATRIX_CRUSHING_FAILURE_RATE`` or ``/FAIL/LAD_CMCRFR`` (M472): Ladevèze rate-dependent coupled multi-axial matrix crushing failure rate, coupled matrix micro-crushing failure, and coupled compressive damage evolution model."""
    mat_id: int = 0
    title: str = ""
    sigma_cmcrfr0: float = 0.0     # initial coupled matrix crushing threshold stress sigma_cmcrfr,0
    sigma_cmcrfrc: float = 1.0     # critical coupled matrix crushing saturation stress sigma_cmcrfr,c
    gamma_cmcrfr: float = 0.0      # coupled matrix crushing rate sensitivity factor gamma_cmcrfr
    p_cmcrfr: float = 1.0          # coupled matrix crushing rate exponent p_cmcrfr
    d_cmcrfr_max: float = 0.999    # maximum allowable coupled matrix crushing damage index
    ifail_sh: int = 1              # shell element deletion flag
    ifail_so: int = 1              # solid element deletion flag
    fail_id: int = 0               # failure model ID reference

    @property
    def sigma_cmcrf0(self) -> float:
        return self.sigma_cmcrfr0

    @sigma_cmcrf0.setter
    def sigma_cmcrf0(self, val: float) -> None:
        self.sigma_cmcrfr0 = val

    @property
    def sigma_cmcrfc(self) -> float:
        return self.sigma_cmcrfrc

    @sigma_cmcrfc.setter
    def sigma_cmcrfc(self, val: float) -> None:
        self.sigma_cmcrfrc = val

    @property
    def gamma_cmcrf(self) -> float:
        return self.gamma_cmcrfr

    @gamma_cmcrf.setter
    def gamma_cmcrf(self, val: float) -> None:
        self.gamma_cmcrfr = val

    @property
    def p_cmcrf(self) -> float:
        return self.p_cmcrfr

    @p_cmcrf.setter
    def p_cmcrf(self, val: float) -> None:
        self.p_cmcrfr = val

    @property
    def d_cmcrf_max(self) -> float:
        return self.d_cmcrfr_max

    @d_cmcrf_max.setter
    def d_cmcrf_max(self, val: float) -> None:
        self.d_cmcrfr_max = val

    @property
    def sigma_cmcr0(self) -> float:
        return self.sigma_cmcrfr0

    @sigma_cmcr0.setter
    def sigma_cmcr0(self, val: float) -> None:
        self.sigma_cmcrfr0 = val

    @property
    def sigma_cmcrc(self) -> float:
        return self.sigma_cmcrfrc

    @sigma_cmcrc.setter
    def sigma_cmcrc(self, val: float) -> None:
        self.sigma_cmcrfrc = val

    @property
    def gamma_cmcr(self) -> float:
        return self.gamma_cmcrfr

    @gamma_cmcr.setter
    def gamma_cmcr(self, val: float) -> None:
        self.gamma_cmcrfr = val

    @property
    def p_cmcr(self) -> float:
        return self.p_cmcrfr

    @p_cmcr.setter
    def p_cmcr(self, val: float) -> None:
        self.p_cmcrfr = val

    @property
    def d_cmcr_max(self) -> float:
        return self.d_cmcrfr_max

    @d_cmcr_max.setter
    def d_cmcr_max(self, val: float) -> None:
        self.d_cmcrfr_max = val

    @property
    def sigma_cmc0(self) -> float:
        return self.sigma_cmcrfr0

    @sigma_cmc0.setter
    def sigma_cmc0(self, val: float) -> None:
        self.sigma_cmcrfr0 = val

    @property
    def sigma_cmcc(self) -> float:
        return self.sigma_cmcrfrc

    @sigma_cmcc.setter
    def sigma_cmcc(self, val: float) -> None:
        self.sigma_cmcrfrc = val

    @property
    def gamma_cmc(self) -> float:
        return self.gamma_cmcrfr

    @gamma_cmc.setter
    def gamma_cmc(self, val: float) -> None:
        self.gamma_cmcrfr = val

    @property
    def p_cmc(self) -> float:
        return self.p_cmcrfr

    @p_cmc.setter
    def p_cmc(self, val: float) -> None:
        self.p_cmcrfr = val

    @property
    def d_cmc_max(self) -> float:
        return self.d_cmcrfr_max

    @d_cmc_max.setter
    def d_cmc_max(self, val: float) -> None:
        self.d_cmcrfr_max = val


FailLadCoupledMatrixCrushingFailure = FailLadCoupledMatrixCrushingFailureRate
FailLadCoupledMatrixCrushFailureRate = FailLadCoupledMatrixCrushingFailureRate
FailLadCoupledMatrixCrushFailure = FailLadCoupledMatrixCrushingFailureRate
FailLadCoupledMatrixMicroCrushingFailureRate = FailLadCoupledMatrixCrushingFailureRate
FailLadCoupledMatrixMicroCrushingFailure = FailLadCoupledMatrixCrushingFailureRate
FailLadCoupledMatrixMicrocrushingFailureRate = FailLadCoupledMatrixCrushingFailureRate
FailLadCoupledMatrixMicrocrushingFailure = FailLadCoupledMatrixCrushingFailureRate
FailLadCmcrfr = FailLadCoupledMatrixCrushingFailureRate
FailLadCmcrfrModel = FailLadCoupledMatrixCrushingFailureRate
FailLadCmcrfrLaw = FailLadCoupledMatrixCrushingFailureRate
FailLadevezeRateDependentCoupledMatrixCrushingFailure = FailLadCoupledMatrixCrushingFailureRate
FailLadCoupledMatrixCrushingDamageRate = FailLadCoupledMatrixCrushingFailureRate
FailLadCoupledMatrixCrushDamageRate = FailLadCoupledMatrixCrushingDamageRate
FailLadCoupledMatrixMicroCrushingDamageRate = FailLadCoupledMatrixCrushingDamageRate
FailLadCoupledMatrixMicrocrushingDamageRate = FailLadCoupledMatrixCrushingDamageRate
FailLadCoupleMatrixCrushingFailureRate = FailLadCoupledMatrixCrushingFailureRate
FailLadCoupleMatrixCrushingFailure = FailLadCoupledMatrixCrushingFailureRate
FailLadCoupleMatrixCrushFailureRate = FailLadCoupledMatrixCrushingFailureRate
FailLadCoupleMatrixCrushFailure = FailLadCoupledMatrixCrushingFailureRate


# ============================================================================
# M473 Suite: FailLadDynamicFiberKinkingFailureRate, EngElectrothermoflexomagnetochiralhedgehogplasmonicpolaritonicResonanceEnergy, LagmulAtiyahSpinorSpatialLinkageJoint, SensorSpringNormalCrackleRate
# ============================================================================

@dataclass
class FailLadDynamicFiberKinkingFailureRate:
    """``/FAIL/LAD_DYNAMIC_FIBER_KINKING_FAILURE_RATE/mat_ID`` or ``/FAIL/LADEVEZE_DYNAMIC_FIBER_KINKING_FAILURE_RATE`` (M473): Ladevèze rate-dependent dynamic fiber micro-buckling and kinking failure rate, longitudinal compressive kink-band formation, and fiber compressive damage evolution model."""
    mat_id: int = 0
    title: str = ""
    sigma_dfkfr0: float = 0.0    # initial dynamic fiber kinking threshold stress sigma_dfkfr,0
    sigma_dfkfrc: float = 1.0    # critical dynamic fiber kinking saturation stress sigma_dfkfr,c
    gamma_dfkfr: float = 0.0     # dynamic fiber kinking rate sensitivity exponent gamma_dfkfr
    p_dfkfr: float = 1.0         # dynamic fiber kinking rate power exponent p_dfkfr
    d_dfkfr_max: float = 0.999   # maximum allowable dynamic fiber kinking damage index
    ifail_sh: int = 1           # shell element deletion flag
    ifail_so: int = 1           # solid element deletion flag
    fail_id: int = 0            # failure model ID reference

    @property
    def sigma_dfkf0(self) -> float:
        return self.sigma_dfkfr0

    @sigma_dfkf0.setter
    def sigma_dfkf0(self, val: float) -> None:
        self.sigma_dfkfr0 = val

    @property
    def sigma_dfkfc(self) -> float:
        return self.sigma_dfkfrc

    @sigma_dfkfc.setter
    def sigma_dfkfc(self, val: float) -> None:
        self.sigma_dfkfrc = val

    @property
    def gamma_dfkf(self) -> float:
        return self.gamma_dfkfr

    @gamma_dfkf.setter
    def gamma_dfkf(self, val: float) -> None:
        self.gamma_dfkfr = val

    @property
    def p_dfkf(self) -> float:
        return self.p_dfkfr

    @p_dfkf.setter
    def p_dfkf(self, val: float) -> None:
        self.p_dfkfr = val

    @property
    def d_dfkf_max(self) -> float:
        return self.d_dfkfr_max

    @d_dfkf_max.setter
    def d_dfkf_max(self, val: float) -> None:
        self.d_dfkfr_max = val

    @property
    def sigma_dfk0(self) -> float:
        return self.sigma_dfkfr0

    @sigma_dfk0.setter
    def sigma_dfk0(self, val: float) -> None:
        self.sigma_dfkfr0 = val

    @property
    def sigma_dfkc(self) -> float:
        return self.sigma_dfkfrc

    @sigma_dfkc.setter
    def sigma_dfkc(self, val: float) -> None:
        self.sigma_dfkfrc = val

    @property
    def gamma_dfk(self) -> float:
        return self.gamma_dfkfr

    @gamma_dfk.setter
    def gamma_dfk(self, val: float) -> None:
        self.gamma_dfkfr = val

    @property
    def p_dfk(self) -> float:
        return self.p_dfkfr

    @p_dfk.setter
    def p_dfk(self, val: float) -> None:
        self.p_dfkfr = val

    @property
    def d_dfk_max(self) -> float:
        return self.d_dfkfr_max

    @d_dfk_max.setter
    def d_dfk_max(self, val: float) -> None:
        self.d_dfkfr_max = val


FailLadDynamicFiberKinkingFailure = FailLadDynamicFiberKinkingFailureRate
FailLadDynamicFiberKinkFailureRate = FailLadDynamicFiberKinkingFailureRate
FailLadDynamicFiberKinkFailure = FailLadDynamicFiberKinkingFailureRate
FailLadDynamicFiberCompressiveKinkingFailureRate = FailLadDynamicFiberKinkingFailureRate
FailLadDynamicFiberCompressiveKinkingFailure = FailLadDynamicFiberKinkingFailureRate
FailLadDynamicFiberMicrobucklingFailureRate = FailLadDynamicFiberKinkingFailureRate
FailLadDynamicFiberMicrobucklingFailure = FailLadDynamicFiberKinkingFailureRate
FailLadDfkfr = FailLadDynamicFiberKinkingFailureRate
FailLadDfkfrModel = FailLadDynamicFiberKinkingFailureRate
FailLadDfkfrLaw = FailLadDynamicFiberKinkingFailureRate
FailLadevezeRateDependentDynamicFiberKinkingFailure = FailLadDynamicFiberKinkingFailureRate
FailLadDynamicFiberKinkingDamageRate = FailLadDynamicFiberKinkingFailureRate
FailLadevezeDynamicFiberKinkingDamageRate = FailLadDynamicFiberKinkingFailureRate
FailLadDynamicFiberKinkDamageRate = FailLadDynamicFiberKinkingDamageRate
FailLadevezeDynamicFiberKinkDamageRate = FailLadDynamicFiberKinkingDamageRate
FailLadDynamicFiberMicrobucklingDamageRate = FailLadDynamicFiberKinkingDamageRate
FailLadevezeDynamicFiberMicrobucklingDamageRate = FailLadDynamicFiberKinkingDamageRate
FailLadevezeDynamicFiberKinkingFailureRate = FailLadDynamicFiberKinkingFailureRate


# ============================================================================
# M474 Suite: FailLadTransverseFiberKinkingFailureRate, EngElectrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonicResonanceEnergy, LagmulHirzebruchSpinorSpatialLinkageJoint, SensorSpringTransverseCrackleRate
# ============================================================================

@dataclass
class FailLadTransverseFiberKinkingFailureRate:
    """``/FAIL/LAD_TRANSVERSE_FIBER_KINKING_FAILURE_RATE/mat_ID`` or ``/FAIL/LADEVEZE_TRANSVERSE_FIBER_KINKING_FAILURE_RATE`` (M474): Ladevèze rate-dependent transverse fiber micro-buckling and kinking failure rate, off-axis compressive kink-band formation, and transverse fiber compressive damage evolution model."""
    mat_id: int = 0
    title: str = ""
    sigma_tfkfr0: float = 0.0    # initial transverse fiber kinking threshold stress sigma_tfkfr,0
    sigma_tfkfrc: float = 1.0    # critical transverse fiber kinking saturation stress sigma_tfkfr,c
    gamma_tfkfr: float = 0.0     # transverse fiber kinking rate sensitivity exponent gamma_tfkfr
    p_tfkfr: float = 1.0         # transverse fiber kinking rate power exponent p_tfkfr
    d_tfkfr_max: float = 0.999   # maximum allowable transverse fiber kinking damage index
    ifail_sh: int = 1           # shell element deletion flag
    ifail_so: int = 1           # solid element deletion flag
    fail_id: int = 0            # failure model ID reference

    @property
    def sigma_tfkf0(self) -> float:
        return self.sigma_tfkfr0

    @sigma_tfkf0.setter
    def sigma_tfkf0(self, val: float) -> None:
        self.sigma_tfkfr0 = val

    @property
    def sigma_tfkfc(self) -> float:
        return self.sigma_tfkfrc

    @sigma_tfkfc.setter
    def sigma_tfkfc(self, val: float) -> None:
        self.sigma_tfkfrc = val

    @property
    def gamma_tfkf(self) -> float:
        return self.gamma_tfkfr

    @gamma_tfkf.setter
    def gamma_tfkf(self, val: float) -> None:
        self.gamma_tfkfr = val

    @property
    def p_tfkf(self) -> float:
        return self.p_tfkfr

    @p_tfkf.setter
    def p_tfkf(self, val: float) -> None:
        self.p_tfkfr = val

    @property
    def d_tfkf_max(self) -> float:
        return self.d_tfkfr_max

    @d_tfkf_max.setter
    def d_tfkf_max(self, val: float) -> None:
        self.d_tfkfr_max = val

    @property
    def sigma_tfk0(self) -> float:
        return self.sigma_tfkfr0

    @sigma_tfk0.setter
    def sigma_tfk0(self, val: float) -> None:
        self.sigma_tfkfr0 = val

    @property
    def sigma_tfkc(self) -> float:
        return self.sigma_tfkfrc

    @sigma_tfkc.setter
    def sigma_tfkc(self, val: float) -> None:
        self.sigma_tfkfrc = val

    @property
    def gamma_tfk(self) -> float:
        return self.gamma_tfkfr

    @gamma_tfk.setter
    def gamma_tfk(self, val: float) -> None:
        self.gamma_tfkfr = val

    @property
    def p_tfk(self) -> float:
        return self.p_tfkfr

    @p_tfk.setter
    def p_tfk(self, val: float) -> None:
        self.p_tfkfr = val

    @property
    def d_tfk_max(self) -> float:
        return self.d_tfkfr_max

    @d_tfk_max.setter
    def d_tfk_max(self, val: float) -> None:
        self.d_tfkfr_max = val


FailLadTransverseFiberKinkingFailure = FailLadTransverseFiberKinkingFailureRate
FailLadTransverseFiberKinkFailureRate = FailLadTransverseFiberKinkingFailureRate
FailLadTransverseFiberKinkFailure = FailLadTransverseFiberKinkingFailureRate
FailLadTransverseFiberCompressiveKinkingFailureRate = FailLadTransverseFiberKinkingFailureRate
FailLadTransverseFiberCompressiveKinkingFailure = FailLadTransverseFiberKinkingFailureRate
FailLadTransverseFiberMicrobucklingFailureRate = FailLadTransverseFiberKinkingFailureRate
FailLadTransverseFiberMicrobucklingFailure = FailLadTransverseFiberKinkingFailureRate
FailLadTfdfkfr = FailLadTransverseFiberKinkingFailureRate
FailLadTfdfkfrModel = FailLadTransverseFiberKinkingFailureRate
FailLadTfdfkfrLaw = FailLadTransverseFiberKinkingFailureRate
FailLadTfkfr = FailLadTransverseFiberKinkingFailureRate
FailLadTfkfrModel = FailLadTransverseFiberKinkingFailureRate
FailLadTfkfrLaw = FailLadTransverseFiberKinkingFailureRate
FailLadevezeRateDependentTransverseFiberKinkingFailure = FailLadTransverseFiberKinkingFailureRate
FailLadTransverseFiberKinkingDamageRate = FailLadTransverseFiberKinkingFailureRate
FailLadevezeTransverseFiberKinkingDamageRate = FailLadTransverseFiberKinkingFailureRate
FailLadTransverseFiberKinkDamageRate = FailLadTransverseFiberKinkingDamageRate
FailLadevezeTransverseFiberKinkDamageRate = FailLadTransverseFiberKinkingDamageRate
FailLadTransverseFiberMicrobucklingDamageRate = FailLadTransverseFiberKinkingDamageRate
FailLadevezeTransverseFiberMicrobucklingDamageRate = FailLadTransverseFiberKinkingDamageRate
FailLadevezeTransverseFiberKinkingFailureRate = FailLadTransverseFiberKinkingFailureRate


# ============================================================================
# M475 Suite: FailLadCoupledFiberKinkingFailureRate, EngElectrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonicResonanceEnergy, LagmulGrothendieckSpinorSpatialLinkageJoint, SensorSpringBendingCrackleRate
# ============================================================================

@dataclass
class FailLadCoupledFiberKinkingFailureRate:
    """``/FAIL/LAD_COUPLED_FIBER_KINKING_FAILURE_RATE/mat_ID`` or ``/FAIL/LADEVEZE_COUPLED_FIBER_KINKING_FAILURE_RATE`` (M475): Ladevèze rate-dependent coupled multi-axial fiber micro-buckling and kinking failure rate, coupled compressive kink-band formation, and fiber compressive damage evolution model."""
    mat_id: int = 0
    title: str = ""
    sigma_cfkfr0: float = 0.0     # initial coupled fiber kinking failure rate threshold stress sigma_cfkfr,0
    sigma_cfkfrc: float = 1.0     # critical coupled fiber kinking failure rate saturation stress sigma_cfkfr,c
    gamma_cfkfr: float = 0.0      # coupled fiber kinking failure rate sensitivity factor gamma_cfkfr
    p_cfkfr: float = 1.0          # coupled fiber kinking failure rate exponent p_cfkfr
    d_cfkfr_max: float = 0.999    # maximum allowable coupled fiber kinking damage index d_cfkfr_max
    ifail_sh: int = 1             # shell element deletion flag
    ifail_so: int = 1             # solid element deletion flag
    fail_id: int = 0              # failure model ID reference

    # Property aliases for coupled fiber kinking
    @property
    def sigma_cfkf0(self) -> float:
        return self.sigma_cfkfr0

    @sigma_cfkf0.setter
    def sigma_cfkf0(self, val: float) -> None:
        self.sigma_cfkfr0 = val

    @property
    def sigma_cfkfc(self) -> float:
        return self.sigma_cfkfrc

    @sigma_cfkfc.setter
    def sigma_cfkfc(self, val: float) -> None:
        self.sigma_cfkfrc = val

    @property
    def gamma_cfkf(self) -> float:
        return self.gamma_cfkfr

    @gamma_cfkf.setter
    def gamma_cfkf(self, val: float) -> None:
        self.gamma_cfkfr = val

    @property
    def p_cfkf(self) -> float:
        return self.p_cfkfr

    @p_cfkf.setter
    def p_cfkf(self, val: float) -> None:
        self.p_cfkfr = val

    @property
    def d_cfkf_max(self) -> float:
        return self.d_cfkfr_max

    @d_cfkf_max.setter
    def d_cfkf_max(self, val: float) -> None:
        self.d_cfkfr_max = val

    @property
    def sigma_cfk0(self) -> float:
        return self.sigma_cfkfr0

    @sigma_cfk0.setter
    def sigma_cfk0(self, val: float) -> None:
        self.sigma_cfkfr0 = val

    @property
    def sigma_cfkc(self) -> float:
        return self.sigma_cfkfrc

    @sigma_cfkc.setter
    def sigma_cfkc(self, val: float) -> None:
        self.sigma_cfkfrc = val

    @property
    def gamma_cfk(self) -> float:
        return self.gamma_cfkfr

    @gamma_cfk.setter
    def gamma_cfk(self, val: float) -> None:
        self.gamma_cfkfr = val

    @property
    def p_cfk(self) -> float:
        return self.p_cfkfr

    @p_cfk.setter
    def p_cfk(self, val: float) -> None:
        self.p_cfkfr = val

    @property
    def d_cfk_max(self) -> float:
        return self.d_cfkfr_max

    @d_cfk_max.setter
    def d_cfk_max(self, val: float) -> None:
        self.d_cfkfr_max = val


FailLadCoupledFiberKinkingFailure = FailLadCoupledFiberKinkingFailureRate
FailLadCoupledFiberKinkFailureRate = FailLadCoupledFiberKinkingFailureRate
FailLadCoupledFiberKinkFailure = FailLadCoupledFiberKinkingFailureRate
FailLadCoupledFiberCompressiveKinkingFailureRate = FailLadCoupledFiberKinkingFailureRate
FailLadCoupledFiberCompressiveKinkingFailure = FailLadCoupledFiberKinkingFailureRate
FailLadCoupledFiberMicrobucklingFailureRate = FailLadCoupledFiberKinkingFailureRate
FailLadCoupledFiberMicrobucklingFailure = FailLadCoupledFiberKinkingFailureRate
FailLadCfdfkfr = FailLadCoupledFiberKinkingFailureRate
FailLadCfkfr = FailLadCoupledFiberKinkingFailureRate
FailLadCoupledFiberKinkingDamageRate = FailLadCoupledFiberKinkingFailureRate
FailLadCoupledFiberKinkDamageRate = FailLadCoupledFiberKinkingFailureRate
FailLadCoupledFiberMicrobucklingDamageRate = FailLadCoupledFiberKinkingFailureRate
FailLadevezeCoupledFiberKinkingFailureRate = FailLadCoupledFiberKinkingFailureRate
FailLadevezeCoupledFiberKinkingFailure = FailLadCoupledFiberKinkingFailureRate
FailLadevezeCoupledFiberKinkFailureRate = FailLadCoupledFiberKinkingFailureRate
FailLadevezeCoupledFiberKinkFailure = FailLadCoupledFiberKinkingFailureRate
FailLadevezeCoupledFiberCompressiveKinkingFailureRate = FailLadCoupledFiberKinkingFailureRate
FailLadevezeCoupledFiberCompressiveKinkingFailure = FailLadCoupledFiberKinkingFailureRate
FailLadevezeCoupledFiberMicrobucklingFailureRate = FailLadCoupledFiberKinkingFailureRate
FailLadevezeCoupledFiberMicrobucklingFailure = FailLadCoupledFiberKinkingFailureRate


@dataclass
class FailLadDynamicCoreCrushingFailureRate:
    """``/FAIL/LAD_DYNAMIC_CORE_CRUSHING_FAILURE_RATE/mat_ID`` or ``/FAIL/LADEVEZE_DYNAMIC_CORE_CRUSHING_FAILURE_RATE`` (M476): Ladevèze rate-dependent dynamic core crushing failure rate, core micro-crushing failure, and compressive core damage evolution model."""
    mat_id: int = 0
    title: str = ""
    sigma_dccfr0: float = 0.0     # initial dynamic core crushing failure rate threshold stress sigma_dccfr,0
    sigma_dccfrc: float = 1.0     # critical dynamic core crushing failure rate saturation stress sigma_dccfr,c
    gamma_dccfr: float = 0.0      # dynamic core crushing failure rate sensitivity factor gamma_dccfr
    p_dccfr: float = 1.0          # dynamic core crushing failure rate exponent p_dccfr
    d_dccfr_max: float = 0.999    # maximum allowable dynamic core crushing damage index d_dccfr_max
    ifail_sh: int = 1             # shell element deletion flag
    ifail_so: int = 1             # solid element deletion flag
    fail_id: int = 0              # failure model ID reference

    # Property aliases for dynamic core crushing
    @property
    def sigma_dccf0(self) -> float:
        return self.sigma_dccfr0

    @sigma_dccf0.setter
    def sigma_dccf0(self, val: float) -> None:
        self.sigma_dccfr0 = val

    @property
    def sigma_dcc0(self) -> float:
        return self.sigma_dccfr0

    @sigma_dcc0.setter
    def sigma_dcc0(self, val: float) -> None:
        self.sigma_dccfr0 = val

    @property
    def sigma_dfdccfr0(self) -> float:
        return self.sigma_dccfr0

    @sigma_dfdccfr0.setter
    def sigma_dfdccfr0(self, val: float) -> None:
        self.sigma_dccfr0 = val

    @property
    def sigma_dfdccf0(self) -> float:
        return self.sigma_dccfr0

    @sigma_dfdccf0.setter
    def sigma_dfdccf0(self, val: float) -> None:
        self.sigma_dccfr0 = val

    @property
    def sigma_dfdcc0(self) -> float:
        return self.sigma_dccfr0

    @sigma_dfdcc0.setter
    def sigma_dfdcc0(self, val: float) -> None:
        self.sigma_dccfr0 = val

    @property
    def sigma_dccfc(self) -> float:
        return self.sigma_dccfrc

    @sigma_dccfc.setter
    def sigma_dccfc(self, val: float) -> None:
        self.sigma_dccfrc = val

    @property
    def sigma_dccc(self) -> float:
        return self.sigma_dccfrc

    @sigma_dccc.setter
    def sigma_dccc(self, val: float) -> None:
        self.sigma_dccfrc = val

    @property
    def sigma_dfdccfrc(self) -> float:
        return self.sigma_dccfrc

    @sigma_dfdccfrc.setter
    def sigma_dfdccfrc(self, val: float) -> None:
        self.sigma_dccfrc = val

    @property
    def sigma_dfdccfc(self) -> float:
        return self.sigma_dccfrc

    @sigma_dfdccfc.setter
    def sigma_dfdccfc(self, val: float) -> None:
        self.sigma_dccfrc = val

    @property
    def sigma_dfdccc(self) -> float:
        return self.sigma_dccfrc

    @sigma_dfdccc.setter
    def sigma_dfdccc(self, val: float) -> None:
        self.sigma_dccfrc = val

    @property
    def gamma_dccf(self) -> float:
        return self.gamma_dccfr

    @gamma_dccf.setter
    def gamma_dccf(self, val: float) -> None:
        self.gamma_dccfr = val

    @property
    def gamma_dcc(self) -> float:
        return self.gamma_dccfr

    @gamma_dcc.setter
    def gamma_dcc(self, val: float) -> None:
        self.gamma_dccfr = val

    @property
    def gamma_dfdccfr(self) -> float:
        return self.gamma_dccfr

    @gamma_dfdccfr.setter
    def gamma_dfdccfr(self, val: float) -> None:
        self.gamma_dccfr = val

    @property
    def gamma_dfdccf(self) -> float:
        return self.gamma_dccfr

    @gamma_dfdccf.setter
    def gamma_dfdccf(self, val: float) -> None:
        self.gamma_dccfr = val

    @property
    def gamma_dfdcc(self) -> float:
        return self.gamma_dccfr

    @gamma_dfdcc.setter
    def gamma_dfdcc(self, val: float) -> None:
        self.gamma_dccfr = val

    @property
    def p_dccf(self) -> float:
        return self.p_dccfr

    @p_dccf.setter
    def p_dccf(self, val: float) -> None:
        self.p_dccfr = val

    @property
    def p_dcc(self) -> float:
        return self.p_dccfr

    @p_dcc.setter
    def p_dcc(self, val: float) -> None:
        self.p_dccfr = val

    @property
    def p_dfdccfr(self) -> float:
        return self.p_dccfr

    @p_dfdccfr.setter
    def p_dfdccfr(self, val: float) -> None:
        self.p_dccfr = val

    @property
    def p_dfdccf(self) -> float:
        return self.p_dccfr

    @p_dfdccf.setter
    def p_dfdccf(self, val: float) -> None:
        self.p_dccfr = val

    @property
    def p_dfdcc(self) -> float:
        return self.p_dccfr

    @p_dfdcc.setter
    def p_dfdcc(self, val: float) -> None:
        self.p_dccfr = val

    @property
    def d_dccf_max(self) -> float:
        return self.d_dccfr_max

    @d_dccf_max.setter
    def d_dccf_max(self, val: float) -> None:
        self.d_dccfr_max = val

    @property
    def d_dcc_max(self) -> float:
        return self.d_dccfr_max

    @d_dcc_max.setter
    def d_dcc_max(self, val: float) -> None:
        self.d_dccfr_max = val

    @property
    def d_dfdccfr_max(self) -> float:
        return self.d_dccfr_max

    @d_dfdccfr_max.setter
    def d_dfdccfr_max(self, val: float) -> None:
        self.d_dccfr_max = val

    @property
    def d_dfdccf_max(self) -> float:
        return self.d_dccfr_max

    @d_dfdccf_max.setter
    def d_dfdccf_max(self, val: float) -> None:
        self.d_dccfr_max = val

    @property
    def d_dfdcc(self) -> float:
        return self.d_dccfr_max

    @d_dfdcc.setter
    def d_dfdcc(self, val: float) -> None:
        self.d_dccfr_max = val

    @property
    def d_dfdcc_max(self) -> float:
        return self.d_dccfr_max

    @d_dfdcc_max.setter
    def d_dfdcc_max(self, val: float) -> None:
        self.d_dccfr_max = val


FailLadDfdccfr = FailLadDynamicCoreCrushingFailureRate
FailLadDccfr = FailLadDynamicCoreCrushingFailureRate
FailLadDynamicFiberDirectionCoreCrushingFailureRate = FailLadDynamicCoreCrushingFailureRate
FailLadFiberDirectionDynamicCoreCrushingFailureRate = FailLadDynamicCoreCrushingFailureRate
FailLadDynamicCoreCrushFailureRate = FailLadDynamicCoreCrushingFailureRate
FailLadDynamicCoreCrushingRateFailure = FailLadDynamicCoreCrushingFailureRate
FailLadDynamicHoneycombCoreCrushingFailureRate = FailLadDynamicCoreCrushingFailureRate
FailLadDynamicFoamCoreCrushingFailureRate = FailLadDynamicCoreCrushingFailureRate
FailLadDynamicSandwichCoreCrushingFailureRate = FailLadDynamicCoreCrushingFailureRate
FailLadDynamicCellularCoreCrushingFailureRate = FailLadDynamicCoreCrushingFailureRate
FailLadDynamicBalsaCoreCrushingFailureRate = FailLadDynamicCoreCrushingFailureRate
FailLadDynamicLatticeCoreCrushingFailureRate = FailLadDynamicCoreCrushingFailureRate
FailLadDynamicPorousCoreCrushingFailureRate = FailLadDynamicCoreCrushingFailureRate
FailLadDynamicWebCoreCrushingFailureRate = FailLadDynamicCoreCrushingFailureRate
FailLadDynamicTrussCoreCrushingFailureRate = FailLadDynamicCoreCrushingFailureRate
FailLadDynamicCorrugatedCoreCrushingFailureRate = FailLadDynamicCoreCrushingFailureRate
FailLadDynamicFoldedCoreCrushingFailureRate = FailLadDynamicCoreCrushingFailureRate
FailLadDynamicTubularCoreCrushingFailureRate = FailLadDynamicCoreCrushingFailureRate
FailLadDynamicAuxeticCoreCrushingFailureRate = FailLadDynamicCoreCrushingFailureRate
FailLadDynamicMetamaterialCoreCrushingFailureRate = FailLadDynamicCoreCrushingFailureRate
