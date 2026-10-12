"""
Starter model checks (cross references, sanity).

Fortran origin: scattered through the Starter (every hm_read does its own
id checks; contchk/prelecdt do global ones). Centralized here: after
finalization, verify every reference between options resolves, so the
Engine can index blindly.
"""

from __future__ import annotations

import numpy as np

from ..common.messages import MessageLog
from ..model.model import Model


# multi-material ALE/Euler laws (LAW51, LAW151/MULTIFLUID): their initial
# density is NOT a top-level RHO0 — it is carried by the submaterial
# references (mat_ID_ii) weighted by their volume fractions (Vfrac_ii), so
# the material card's own RHO0 field is legitimately blank/zero.  The
# reference Starter reads the submaterial densities to build the element
# mass and does NOT fatal-error the empty top-level density, so the port's
# null-RHO0 MAT CHECK must exempt the family (M38 / M37-BUG-2).
_MULTIMAT_ALE_LAWS = {51, 151}


# Laws whose RHO0 may legally be zero — the null-density MAT CHECK exempts
# them (M39 / M38-NEW-2).  Two distinct reasons:
#
# * LAW0 (/MAT/VOID) is MASSLESS BY DESIGN.  The upstream reader
#   ``starter/source/materials/mat/mat000/hm_read_mat00.F`` applies NO
#   positivity check to RHO0 and explicitly guards the only place the
#   density is divided by —  ``SDSP = SQRT(YOUNG/MAX(RHOR,EM20))`` — so a
#   void card with RHO0 = 0 is legal and produces a zero sound speed, not
#   an error.  The cfg agrees and is the sharpest evidence: every load-
#   bearing law's cfg CHECK block demands ``MAT_RHO > 0`` (e.g.
#   matl2_plas_johns.cfg) while ``MAT/matl_void0.cfg`` demands only
#   ``MAT_RHO >= 0``.  A void element contributes zero mass and zero
#   stress: dummy contact skins, airbag reference geometry, parts replaced
#   by a rigid body.  The port's fatal here was a false positive on the
#   RD-E-2700 Football decks (BAT_CIR / BAT_SQR), whose /MAT/VOID/12 skin
#   shells carry RHO0 = 0 verbatim;
# * the multimaterial ALE family carries its density on the submaterials
#   (see _MULTIMAT_ALE_LAWS above).
_NULL_RHO0_OK_LAWS = frozenset({0} | _MULTIMAT_ALE_LAWS)


# element family -> material laws its kernels implement (see the
# materials package dispatch; extending a kernel means extending this map).
#
# LAW0 (/MAT/VOID) is legal on EVERY family here, matching the upstream
# compatibility declaration in hm_read_mat00.F, which tags the void law
# SOLID_ISOTROPIC / SHELL_ISOTROPIC / SPRING_MATERIAL / BEAM_ALL / TRUSS /
# SPH — i.e. all of them (M39 / M38-NEW-2, closing the M38 prop-pack OPEN
# item: /PROP/VOID was already made universally family-compatible by
# ``prop_reader.prop_type_ok``, but the MATERIAL half still rejected LAW0
# on /BEAM and /TRUSS, so a void beam passed the property check and failed
# the material one).  The truss/beam kernels honour it: a void material's
# E = G = 0 makes every resultant identically zero (the void semantics)
# and their density divisions are guarded exactly as hm_read_mat00.F
# guards its own — see elements/truss.py and elements/beam_type3.py.
_ALLOWED_LAWS = {
    "bricks": {0, 1, 2, 3, 4, 5, "5", "LAW5", "JWL", 10, "10", "LAW10", "SOIL", "DPRAG1", 12, "12", "LAW12", "3D_COMP", "COMP_3D", "3PARBI", "RAGAB", 14, "14", "LAW14", "COMPSO", "COMP_SOL", 21, "21", "LAW21", "DPRAG", "MAT_DPRAG", "LAW21_DPRAG", "DUCKHUB", 22, "22", "LAW22", "DAMA", "PLAS_DAMA", 24, 25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", 28, "28", "LAW28", "HONEYCOMB", 33, 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW", 35, 36, 37, "37", "LAW37", "BIPHAS", "BIPHASIC", 38, "38", "LAW38", "VISC_TAB", 40, 42, 43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB", 44, 48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "MAT_PLAS_ZHAO", "MAT_LAW48", "LAW48_ZHAO", 49, "49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB", 50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM", 52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS", "LAW52_GURSON", 60, "60", "LAW60", "PLAS_T3", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC", "MAT_LAW60", 62, 69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYPERELASTIC", "LAW69_HYP_ELAS", 70, 79, "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM", 81, 82, "82", "LAW82", "OGDEN", "LAW82_OGDEN", 83, 163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM", 999},
    "tetras": {0, 1, 2, 3, 4, 5, "5", "LAW5", "JWL", 10, "10", "LAW10", "SOIL", "DPRAG1", 12, "12", "LAW12", "3D_COMP", "COMP_3D", "3PARBI", "RAGAB", 14, "14", "LAW14", "COMPSO", "COMP_SOL", 21, "21", "LAW21", "DPRAG", "MAT_DPRAG", "LAW21_DPRAG", "DUCKHUB", 22, "22", "LAW22", "DAMA", "PLAS_DAMA", 24, 25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", 28, "28", "LAW28", "HONEYCOMB", 33, 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW", 35, 36, 37, "37", "LAW37", "BIPHAS", "BIPHASIC", 38, "38", "LAW38", "VISC_TAB", 40, 42, 43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB", 44, 48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "MAT_PLAS_ZHAO", "MAT_LAW48", "LAW48_ZHAO", 49, "49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB", 50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM", 52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS", "LAW52_GURSON", 60, "60", "LAW60", "PLAS_T3", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC", "MAT_LAW60", 62, 69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYPERELASTIC", "LAW69_HYP_ELAS", 70, 79, "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM", 81, 82, "82", "LAW82", "OGDEN", "LAW82_OGDEN", 163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM", 999},
    "penta6": {0, 1, 2, 3, 4, 5, "5", "LAW5", "JWL", 10, "10", "LAW10", "SOIL", "DPRAG1", 12, "12", "LAW12", "3D_COMP", "COMP_3D", "3PARBI", "RAGAB", 14, "14", "LAW14", "COMPSO", "COMP_SOL", 21, "21", "LAW21", "DPRAG", "MAT_DPRAG", "LAW21_DPRAG", "DUCKHUB", 22, "22", "LAW22", "DAMA", "PLAS_DAMA", 24, 25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", 28, "28", "LAW28", "HONEYCOMB", 33, 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW", 35, 36, 37, "37", "LAW37", "BIPHAS", "BIPHASIC", 38, "38", "LAW38", "VISC_TAB", 40, 42, 43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB", 44, 48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "MAT_PLAS_ZHAO", "MAT_LAW48", "LAW48_ZHAO", 49, "49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB", 50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM", 52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS", "LAW52_GURSON", 60, "60", "LAW60", "PLAS_T3", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC", "MAT_LAW60", 62, 69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYPERELASTIC", "LAW69_HYP_ELAS", 70, 79, "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM", 81, 82, "82", "LAW82", "OGDEN", "LAW82_OGDEN", 83, 163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM", 999},
    "pyra5": {0, 1, 2, 3, 4, 5, "5", "LAW5", "JWL", 10, "10", "LAW10", "SOIL", "DPRAG1", 12, "12", "LAW12", "3D_COMP", "COMP_3D", "3PARBI", "RAGAB", 14, "14", "LAW14", "COMPSO", "COMP_SOL", 21, "21", "LAW21", "DPRAG", "MAT_DPRAG", "LAW21_DPRAG", "DUCKHUB", 22, "22", "LAW22", "DAMA", "PLAS_DAMA", 24, 25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", 28, "28", "LAW28", "HONEYCOMB", 33, 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW", 35, 36, 37, "37", "LAW37", "BIPHAS", "BIPHASIC", 38, "38", "LAW38", "VISC_TAB", 40, 42, 43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB", 44, 48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "MAT_PLAS_ZHAO", "MAT_LAW48", "LAW48_ZHAO", 49, "49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB", 50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM", 52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS", "LAW52_GURSON", 60, "60", "LAW60", "PLAS_T3", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC", "MAT_LAW60", 62, 69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYPERELASTIC", "LAW69_HYP_ELAS", 70, 79, "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM", 81, 82, "82", "LAW82", "OGDEN", "LAW82_OGDEN", 83, 163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM", 999},
    "shells": {0, 1, 2, 3, 15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG", 19, 22, "22", "LAW22", "DAMA", "PLAS_DAMA", 25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", 27, 32, "32", "LAW32", "HILL", 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW", 36, 43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB", 44, 48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "MAT_PLAS_ZHAO", "MAT_LAW48", "LAW48_ZHAO", 52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS", "LAW52_GURSON", 57, "57", "LAW57", "BARLAT3", "MAT_BARLAT3", "MAT_LAW57", "LAW57_BARLAT3", 58, "58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A", 60, "60", "LAW60", "PLAS_T3", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC", "MAT_LAW60", 69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYPERELASTIC", "LAW69_HYP_ELAS", 73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", 82, "82", "LAW82", "OGDEN", "LAW82_OGDEN"},
    # QBAT (Ishell=12, M41): the laws the layered kernel reuses from the
    # BT plumbing; no orthotropic (LAW19) shell_ortho wiring yet
    "shells_qbat": {0, 1, 2, 3, 15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG", 22, "22", "LAW22", "DAMA", "PLAS_DAMA", 25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", 27, 32, "32", "LAW32", "HILL", 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW", 36, 43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB", 44, 48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "MAT_PLAS_ZHAO", "MAT_LAW48", "LAW48_ZHAO", 52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS", "LAW52_GURSON", 57, "57", "LAW57", "BARLAT3", "MAT_BARLAT3", "MAT_LAW57", "LAW57_BARLAT3", 58, "58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A", 60, "60", "LAW60", "PLAS_T3", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC", "MAT_LAW60", 69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYPERELASTIC", "LAW69_HYP_ELAS", 73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", 82, "82", "LAW82", "OGDEN", "LAW82_OGDEN"},
    # QEPH (Ishell=24, P3.3): shares the BT layer plumbing INCLUDING the
    # shell_ortho fiber rotation (LAW19); the czfintn.F stabilization runs
    # the CZFINTN_OR orthotropic HM/HF moduli for LAW19 (shell_qeph_ortho)
    "shells_qeph": {0, 1, 2, 3, 15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG", 19, 22, "22", "LAW22", "DAMA", "PLAS_DAMA", 25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", 27, 32, "32", "LAW32", "HILL", 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW", 36, 43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB", 44, 48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "MAT_PLAS_ZHAO", "MAT_LAW48", "LAW48_ZHAO", 52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS", "LAW52_GURSON", 57, "57", "LAW57", "BARLAT3", "MAT_BARLAT3", "MAT_LAW57", "LAW57_BARLAT3", 58, "58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A", 60, "60", "LAW60", "PLAS_T3", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC", "MAT_LAW60", 69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYPERELASTIC", "LAW69_HYP_ELAS", 73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", 82, "82", "LAW82", "OGDEN", "LAW82_OGDEN"},
    "sh3n": {0, 1, 2, 3, 15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG", 19, 22, "22", "LAW22", "DAMA", "PLAS_DAMA", 25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", 27, 32, "32", "LAW32", "HILL", 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW", 36, 43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB", 44, 48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "MAT_PLAS_ZHAO", "MAT_LAW48", "LAW48_ZHAO", 52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS", "LAW52_GURSON", 57, "57", "LAW57", "BARLAT3", "MAT_BARLAT3", "MAT_LAW57", "LAW57_BARLAT3", 58, "58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A", 60, "60", "LAW60", "PLAS_T3", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC", "MAT_LAW60", 69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYPERELASTIC", "LAW69_HYP_ELAS", 73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", 82, "82", "LAW82", "OGDEN", "LAW82_OGDEN"},
    "trusses": {0, 1, 2, 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW"},
    "springs": None,          # springs ignore their material entirely
    "beams": {0, 1, 2, 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW"},
}
_LAW66_KEYS = {66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "FOAM_TAB"}
for _fam in ("bricks", "tetras", "penta6", "pyra5", "shells", "shells_qbat", "shells_qeph", "sh3n"):
    _ALLOWED_LAWS[_fam].update(_LAW66_KEYS)

_LAW74_KEYS = {
    74, "74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL", "HILL_THERM",
    "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_THERM_HILL", "MAT_HILL_THERM",
    "LAW74_HILL_3D", "LAW74_ORTH_PLAS", "LAW74_THERM_HILL", "LAW74_HILL_THERM",
}
for _fam in ("bricks", "tetras", "penta6", "pyra5"):
    _ALLOWED_LAWS[_fam].update(_LAW74_KEYS)

_LAW76_KEYS = {
    76, "76", "LAW76", "SAMP", "SAMP-1", "SAMP_1",
    "MAT_76", "MAT_LAW76", "MAT_SAMP", "MAT_SAMP-1", "MAT_SAMP_1",
    "LAW76_SAMP", "LAW76_SAMP-1", "LAW76_SAMP_1",
}
for _fam in ("bricks", "tetras", "penta6", "pyra5", "shells", "shells_qbat", "shells_qeph", "sh3n"):
    _ALLOWED_LAWS[_fam].update(_LAW76_KEYS)

_LAW87_KEYS = {
    87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000",
    "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000",
    "LAW87_BARLAT", "LAW87_BARLAT2000", "LAW87_BARLAT_2000",
}
for _fam in ("shells", "shells_qbat", "shells_qeph", "sh3n"):
    _ALLOWED_LAWS[_fam].update(_LAW87_KEYS)

_LAW88_KEYS = {
    88, "88", "LAW88", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TAB_HYP", "TABULATED_HYP",
    "MAT_LAW88", "MAT_HYPER_ELAS", "MAT_TABULATED_HYPERELASTIC", "MAT_TAB_HYP",
}
for _fam in ("bricks", "tetras", "penta6", "pyra5", "shells", "shells_qbat", "shells_qeph", "sh3n"):
    _ALLOWED_LAWS[_fam].update(_LAW88_KEYS)

_LAW92_KEYS = {
    92, "92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE",
    "MAT_LAW92", "MAT_ARRUDA_BOYCE", "LAW92_ARRUDA_BOYCE",
}
for _fam in ("bricks", "tetras", "penta6", "pyra5", "shells", "shells_qbat", "shells_qeph", "sh3n"):
    _ALLOWED_LAWS[_fam].update(_LAW92_KEYS)

_LAW93_KEYS = {
    93, "93", "LAW93", "ORTH_HILL",
    "MAT_LAW93", "MAT_ORTH_HILL", "LAW93_ORTH_HILL",
}
for _fam in ("bricks", "tetras", "penta6", "pyra5", "shells", "shells_qbat", "shells_qeph", "sh3n"):
    _ALLOWED_LAWS[_fam].update(_LAW93_KEYS)

_LAW94_KEYS = {
    94, "94", "LAW94", "YEOH",
    "MAT_LAW94", "MAT_YEOH", "LAW94_YEOH",
}
for _fam in ("bricks", "tetras", "penta6", "pyra5", "shells", "shells_qbat", "shells_qeph", "sh3n"):
    _ALLOWED_LAWS[_fam].update(_LAW94_KEYS)

_LAW95_KEYS = {
    95, "95", "LAW95", "BERGSTROM_BOYCE",
    "MAT_LAW95", "MAT_BERGSTROM_BOYCE", "LAW95_BERGSTROM_BOYCE",
}
for _fam in ("bricks", "tetras", "penta6", "pyra5"):
    _ALLOWED_LAWS[_fam].update(_LAW95_KEYS)

_LAW100_KEYS = {
    100, "100", "LAW100", "VISC_HYP", "MNF",
    "MAT_100", "MAT_LAW100", "MAT_VISC_HYP", "MAT_MNF",
    "LAW100_VISC_HYP", "LAW100_MNF",
}
for _fam in ("bricks", "tetras", "penta6", "pyra5"):
    _ALLOWED_LAWS[_fam].update(_LAW100_KEYS)

_LAW101_KEYS = {
    101, "101", "LAW101", "PP", "PLAS_POLY",
    "MAT_101", "MAT_LAW101", "MAT_PP", "MAT_PLAS_POLY",
    "LAW101_PP", "LAW101_PLAS_POLY",
}
for _fam in ("bricks", "tetras", "penta6", "pyra5"):
    _ALLOWED_LAWS[_fam].update(_LAW101_KEYS)

_LAW102_KEYS = {
    102, "102", "LAW102", "DPRAG2", "DRUCKER_PRAGER_2",
    "MAT_102", "MAT_LAW102", "MAT_DPRAG2", "LAW102_DPRAG2",
}
for _fam in ("bricks", "tetras", "penta6", "pyra5"):
    _ALLOWED_LAWS[_fam].update(_LAW102_KEYS)

_LAW103_KEYS = {
    103, "103", "LAW103", "HENSEL_SPITTEL", "HENSEL-SPITTEL", "PLAS_HENS",
    "MAT_103", "MAT_LAW103", "MAT_HENSEL_SPITTEL", "MAT_PLAS_HENS", "HEN",
}
for _fam in ("bricks", "tetras", "penta6", "pyra5"):
    _ALLOWED_LAWS[_fam].update(_LAW103_KEYS)

_LAW104_KEYS = {
    104, "104", "LAW104", "DRUCKER", "JOHNS_VOCE_DRUCKER", "JOHNS-VOCE-DRUCKER", "PLAS_DRUCK",
    "MAT_104", "MAT_LAW104", "MAT_DRUCKER", "MAT_JOHNS_VOCE_DRUCKER", "MAT_PLAS_DRUCK", "LAW104_DRUCKER",
}
for _fam in ("bricks", "tetras", "penta6", "pyra5", "shells"):
    _ALLOWED_LAWS[_fam].update(_LAW104_KEYS)

_LAW105_KEYS = {
    105, "105", "LAW105", "POWDER_BURN", "POWDERBURN", "LAW105_POWDER_BURN",
    "MAT_105", "MAT_LAW105", "MAT_POWDER_BURN", "MAT_POWDERBURN",
}
for _fam in ("bricks", "tetras", "penta6", "pyra5", "sph"):
    if _fam in _ALLOWED_LAWS:
        _ALLOWED_LAWS[_fam].update(_LAW105_KEYS)

_LAW106_KEYS = {
    106, "106", "LAW106", "JCOOK_ALM", "JOHNS_COOK_ALM", "MAT_JCOOK_ALM",
    "MAT_106", "MAT_LAW106", "LAW106_JCOOK_ALM", "MAT_JOHNS_COOK_ALM",
}
for _fam in ("bricks", "tetras", "penta6", "pyra5", "shells"):
    _ALLOWED_LAWS[_fam].update(_LAW106_KEYS)

_LAW107_KEYS = {
    107, "107", "LAW107", "PAPER_LIGHT", "PLAS_PAPER_LIGHT", "LAW107_PAPER_LIGHT",
    "PFEIFFER", "MAT_PFEIFFER", "MAT_PAPER_LIGHT", "MAT_PLAS_PAPER_LIGHT", "MAT_107", "MAT_LAW107",
}
for _fam in ("bricks", "tetras", "penta6", "pyra5", "shells"):
    _ALLOWED_LAWS[_fam].update(_LAW107_KEYS)

_LAW109_KEYS = {
    109, "109", "LAW109", "TAB_PLAS", "ELASTO_PLAS_TAB", "LAW109_TAB_PLAS",
    "MLAW109", "MAT_LAW109", "MAT_TAB_PLAS", "MAT_ELASTO_PLAS_TAB", "MAT_109",
}
for _fam in ("bricks", "tetras", "penta6", "pyra5", "shells", "shells_qbat", "shells_qeph", "sh3n"):
    if _fam in _ALLOWED_LAWS:
        _ALLOWED_LAWS[_fam].update(_LAW109_KEYS)

_LAW110_KEYS = {
    110, "110", "LAW110", "VEGTER", "PLAS_VEGTER", "LAW110_VEGTER",
    "MLAW110", "MAT_LAW110", "MAT_VEGTER", "MAT_PLAS_VEGTER", "MAT_110",
}
for _fam in ("shells", "shells_qbat", "shells_qeph", "sh3n"):
    if _fam in _ALLOWED_LAWS:
        _ALLOWED_LAWS[_fam].update(_LAW110_KEYS)

_LAW124_KEYS = {
    124, "124", "LAW124", "CDPM2", "CONCRETE_DAMAGE_PLASTICITY_2",
    "MAT_LAW124", "MAT_CDPM2", "LAW124_CDPM2", "MLAW124",
}
for _fam in ("bricks", "tetras", "penta6", "pyra5"):
    _ALLOWED_LAWS[_fam].update(_LAW124_KEYS)

_LAW126_KEYS = {
    126, "126", "LAW126", "JOHNSON_HOLMQUIST_CONCRETE", "MAT_LAW126",
    "MAT_JOHNSON_HOLMQUIST_CONCRETE", "LAW126_JOHNSON_HOLMQUIST_CONCRETE", "HJC", "MAT_HJC",
}
for _fam in ("bricks", "tetras", "penta6", "pyra5"):
    _ALLOWED_LAWS[_fam].update(_LAW126_KEYS)

_LAW169_KEYS = {
    169, "169", "LAW169", "ARUP_ADHESIVE", "MAT_LAW169", "MAT_ARUP_ADHESIVE",
    "LAW169_ARUP_ADHESIVE", "ARUP", "ADHESIVE",
}
for _fam in ("bricks", "tetras", "penta6", "pyra5", "shells", "shells_qbat", "shells_qeph", "sh3n"):
    _ALLOWED_LAWS[_fam].update(_LAW169_KEYS)

_ALLOWED_LAWS["quads"] = _ALLOWED_LAWS["shells"]
_ALLOWED_LAWS["solids"] = _ALLOWED_LAWS["bricks"]
_ALLOWED_LAWS["solids_heph"] = _ALLOWED_LAWS["bricks"]
_ALLOWED_LAWS["solids_tetra4"] = _ALLOWED_LAWS["tetras"]
_ALLOWED_LAWS["shells_bt4"] = _ALLOWED_LAWS["shells"]
_ALLOWED_LAWS["tria3"] = _ALLOWED_LAWS["shells"]



def check_mat_law34(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW34 (/MAT/BOLTZMAN, /MAT/VISC_MAXW) parameter bounds (M539).

    Required checks:
      - rho > 0
      - bulk > 0
      - g0 >= 0
      - gi >= 0
      - beta >= 0
    """
    mid = getattr(mat, "id", 0)
    if hasattr(mat, "k") and hasattr(mat, "rho0"):
        rho = getattr(mat, "rho0", 0.0)
        bulk = getattr(mat, "k", 0.0)
        g0 = getattr(mat, "g0", 0.0)
        gi = getattr(mat, "gl", 0.0)
        beta = getattr(mat, "beta", 0.0)
    else:
        params = getattr(mat, "params", {}) or {}
        rho = getattr(mat, "rho0", 0.0) or params.get("rho", 0.0) or params.get("MAT_RHO", 0.0)
        bulk = params.get("bulk", params.get("k", params.get("MAT_BULK", 0.0)))
        g0 = params.get("g0", params.get("MAT_G0", 0.0))
        gi = params.get("gi", params.get("gl", params.get("MAT_GI", 0.0)))
        beta = params.get("beta", params.get("decay", params.get("MAT_DECAY", 0.0)))

    try:
        rho = float(rho)
    except (TypeError, ValueError):
        rho = 0.0
    try:
        bulk = float(bulk)
    except (TypeError, ValueError):
        bulk = 0.0
    try:
        g0 = float(g0)
    except (TypeError, ValueError):
        g0 = 0.0
    try:
        gi = float(gi)
    except (TypeError, ValueError):
        gi = 0.0
    try:
        beta = float(beta)
    except (TypeError, ValueError):
        beta = 0.0

    if rho <= 0.0:
        log.error(f"/MAT/LAW34/{mid}: initial density RHO must be > 0 (got {rho:g})", "MAT CHECK")
    if bulk <= 0.0:
        log.error(f"/MAT/LAW34/{mid}: bulk modulus BULK must be > 0 (got {bulk:g})", "MAT CHECK")
    if g0 < 0.0:
        log.error(f"/MAT/LAW34/{mid}: short-term shear modulus G0 must be >= 0 (got {g0:g})", "MAT CHECK")
    if gi < 0.0:
        log.error(f"/MAT/LAW34/{mid}: long-term shear modulus GI must be >= 0 (got {gi:g})", "MAT CHECK")
    if beta < 0.0:
        log.error(f"/MAT/LAW34/{mid}: decay constant BETA must be >= 0 (got {beta:g})", "MAT CHECK")
    if g0 > 0.0 and gi > g0:
        log.warning(f"/MAT/LAW34/{mid}: long-term shear modulus GI ({gi:g}) exceeds short-term shear modulus G0 ({g0:g})", "MAT CHECK")


def check_mat_law37(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW37 (/MAT/BIPHAS, /MAT/BIPHASIC) parameter bounds (M540).

    Required checks:
      - rho_l0 > 0
      - rho_g0 > 0
      - c_l > 0
      - gamma_g > 0
      - 0 <= alpha1 <= 1
      - nu_l >= 0
      - nu_g >= 0
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    rho_l0 = _extract(["rho_l0", "RHO_l0", "Lqud_Rho_l", "rho_l", "RHO_L"])
    rho_g0 = _extract(["rho_g0", "RHO_G0", "Lqud_Rho_g", "rho_g", "RHO_G"], default=1.0)
    c_l = _extract(["c_l", "C_l", "bulk_l", "c1", "C1"])
    gamma_g = _extract(["gamma_g", "gamma", "GAMMA", "Lqud_Gamma_bulk", "gam"], default=1.4)
    alpha1 = _extract(["alpha1", "ALPHA1", "alpha_l", "Alpha_l", "a1", "A1"])
    nu_l = _extract(["nu_l", "Nu_l", "NU_l", "vis_l"])
    nu_g = _extract(["nu_g", "Nu_g", "NU_g", "vis_g"])

    if rho_l0 <= 0.0:
        log.error(f"/MAT/LAW37/{mid}: liquid reference density rho_l0 must be > 0 (got {rho_l0:g})", "MAT CHECK")
    if rho_g0 <= 0.0:
        log.error(f"/MAT/LAW37/{mid}: gas reference density rho_g0 must be > 0 (got {rho_g0:g})", "MAT CHECK")
    if c_l <= 0.0:
        log.error(f"/MAT/LAW37/{mid}: liquid bulk modulus c_l must be > 0 (got {c_l:g})", "MAT CHECK")
    if gamma_g <= 0.0:
        log.error(f"/MAT/LAW37/{mid}: gas constant gamma must be > 0 (got {gamma_g:g})", "MAT CHECK")
    if alpha1 < 0.0 or alpha1 > 1.0:
        log.error(f"/MAT/LAW37/{mid}: initial liquid massic fraction alpha1 must be between 0 and 1 (got {alpha1:g})", "MAT CHECK")
    if nu_l < 0.0:
        log.error(f"/MAT/LAW37/{mid}: liquid shear viscosity nu_l must be >= 0 (got {nu_l:g})", "MAT CHECK")
    if nu_g < 0.0:
        log.error(f"/MAT/LAW37/{mid}: gas shear viscosity nu_g must be >= 0 (got {nu_g:g})", "MAT CHECK")


def check_mat_law38(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW38 (/MAT/VISC_TAB) parameter bounds (M541).

    Required checks:
      - rho0 > 0
      - e > 0
      - 0 <= nu_t < 0.5 and 0 <= nu_c < 0.5
      - 1 <= nfunc <= 5
      - If air content active (kcompair == 1): 0 <= phi < 1 and P0 >= 0
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0", "Refer_Rho", "rhor"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW38/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's modulus e > 0
    e = _extract(["e", "E", "MAT_E", "young"], default=0.0)
    if e <= 0.0:
        log.error(f"/MAT/LAW38/{mid}: Young's modulus E must be > 0 (got {e:g})", "MAT CHECK")

    # 3. Poisson's ratios: 0 <= nu_t < 0.5 and 0 <= nu_c < 0.5
    nu_t = _extract(["nu_t", "nu", "NU_t", "MAT_NU", "MAT_NUt", "vt", "VT"], default=0.0)
    nu_c = _extract(["nu_c", "NU_c", "MAT_NUc", "vc", "VC"], default=nu_t)
    for k in ["nu_c", "NU_c", "MAT_NUc", "vc", "VC"]:
        if hasattr(mat, k) and getattr(mat, k) is not None:
            try:
                nu_c = float(getattr(mat, k))
                break
            except (TypeError, ValueError):
                pass
        if isinstance(params, dict) and k in params and params[k] is not None:
            try:
                nu_c = float(params[k])
                break
            except (TypeError, ValueError):
                pass

    if nu_t < 0.0 or nu_t >= 0.5:
        log.error(f"/MAT/LAW38/{mid}: tensile Poisson's ratio nu_t must be in [0, 0.5) (got {nu_t:g})", "MAT CHECK")
    if nu_c < 0.0 or nu_c >= 0.5:
        log.error(f"/MAT/LAW38/{mid}: compressive Poisson's ratio nu_c must be in [0, 0.5) (got {nu_c:g})", "MAT CHECK")

    # 4. Number of functions: 1 <= nfunc <= 5
    nfunc = None
    for k in ["nfunc", "NFUNC", "mfunc", "MFUNC", "m_func", "n_func", "num_curves"]:
        if hasattr(mat, k) and getattr(mat, k) is not None:
            try:
                nfunc = int(getattr(mat, k))
                break
            except (TypeError, ValueError):
                pass
        if isinstance(params, dict) and k in params and params[k] is not None:
            try:
                nfunc = int(params[k])
                break
            except (TypeError, ValueError):
                pass
    if nfunc is None:
        curves = getattr(mat, "curves", None) or (params.get("curves", None) if isinstance(params, dict) else None) or (params.get("funct_ids", None) if isinstance(params, dict) else None)
        if isinstance(curves, (list, tuple)):
            nfunc = len(curves)
        else:
            nfunc = 1

    if nfunc < 1 or nfunc > 5:
        log.error(f"/MAT/LAW38/{mid}: number of functions nfunc must be between 1 and 5 (got {nfunc})", "MAT CHECK")

    # 5. Air content active (kcompair == 1): 0 <= phi < 1 and P0 >= 0
    kcompair = 0
    for k in ["kcompair", "KCOMPAIR", "MAT_Kair", "kair", "KAIR"]:
        if hasattr(mat, k) and getattr(mat, k) is not None:
            try:
                kcompair = int(getattr(mat, k))
                break
            except (TypeError, ValueError):
                pass
        if isinstance(params, dict) and k in params and params[k] is not None:
            try:
                kcompair = int(params[k])
                break
            except (TypeError, ValueError):
                pass

    if kcompair == 1:
        phi = _extract(["phi", "PHI", "poros", "porosity", "MAT_POROS"], default=0.0)
        p0 = _extract(["p0", "P0", "p_atm", "MAT_P0"], default=0.0)
        if phi < 0.0 or phi >= 1.0:
            log.error(f"/MAT/LAW38/{mid}: porosity phi must be in [0, 1) when air content is active (got {phi:g})", "MAT CHECK")
        if p0 < 0.0:
            log.error(f"/MAT/LAW38/{mid}: initial air pressure P0 must be >= 0 (got {p0:g})", "MAT CHECK")


def check_mat_law32(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW32 (/MAT/HILL) parameter bounds (M542).

    Required checks:
      - rho0 > 0
      - e > 0
      - 0 <= nu < 0.5
      - a > 0 (sigy)
      - n <= 1.0 (error if hard > 1.0 per hm_read_mat32.F line 219)
      - eps0 > 0 (error if srp <= 0 per hm_read_mat32.F line 224)
      - r00 > 0, r45 > 0, r90 > 0
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0", "Refer_Rho", "rhor"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW32/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's modulus e > 0
    e = _extract(["e", "E", "MAT_E", "young"], default=0.0)
    if e <= 0.0:
        log.error(f"/MAT/LAW32/{mid}: Young's modulus E must be > 0 (got {e:g})", "MAT CHECK")

    # 3. Poisson's ratio: 0 <= nu < 0.5
    nu = _extract(["nu", "NU", "MAT_NU", "anu"], default=0.0)
    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW32/{mid}: Poisson's ratio NU must be in [0, 0.5) (got {nu:g})", "MAT CHECK")

    # 4. Yield stress a (sigy) > 0
    a = _extract(["a", "A", "sigy", "SIGY", "MAT_SIGY", "ca", "CA"], default=0.0)
    if a <= 0.0:
        log.error(f"/MAT/LAW32/{mid}: yield stress A (SIGY) must be > 0 (got {a:g})", "MAT CHECK")

    # 5. Hardening exponent n <= 1.0 (error if hard > 1.0 per hm_read_mat32.F line 219)
    n = _extract(["n", "N", "hard", "HARD", "MAT_HARD", "cn", "CN"], default=1.0)
    if n > 1.0:
        log.error(f"/MAT/LAW32/{mid}: hardening exponent n must be <= 1.0 (got {n:g})", "MAT CHECK")

    # 6. Reference strain rate eps0 > 0 (error if srp <= 0 per hm_read_mat32.F line 224)
    eps0 = _extract(["eps0", "EPS0", "srp", "SRP", "MAT_SRP", "MAT_SRP_MIN", "eps_dot_0", "EPS_DOT_0"], default=1.0)
    if eps0 <= 0.0:
        log.error(f"/MAT/LAW32/{mid}: reference strain rate EPS_DOT_0 (srp) must be > 0 (got {eps0:g})", "MAT CHECK")

    # 7. Lankford parameters: r00 > 0, r45 > 0, r90 > 0
    r00 = _extract(["r00", "R00", "MAT_R00"], default=1.0)
    if r00 <= 0.0:
        log.error(f"/MAT/LAW32/{mid}: Lankford parameter r00 must be > 0 (got {r00:g})", "MAT CHECK")

    r45 = _extract(["r45", "R45", "MAT_R45"], default=1.0)
    if r45 <= 0.0:
        log.error(f"/MAT/LAW32/{mid}: Lankford parameter r45 must be > 0 (got {r45:g})", "MAT CHECK")

    r90 = _extract(["r90", "R90", "MAT_R90"], default=1.0)
    if r90 <= 0.0:
        log.error(f"/MAT/LAW32/{mid}: Lankford parameter r90 must be > 0 (got {r90:g})", "MAT CHECK")


def check_mat_law15(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW15 (/MAT/CHANG, /MAT/PLAS_ANISO, /MAT/COMP_CHANG) parameter bounds (M544).

    Required checks:
      - rho0 > 0
      - E1 > 0, E2 > 0
      - detc = 1 - nu12*nu21 > 0
      - G12, G23, G31 > 0
      - sig_1yt, sig_2yt, sig_1yc, sig_2yc, sig_12yc, sig_12yt > 0
      - b <= 1.0 (and n <= 1.0)
      - strengths (S1, S2, C1, C2, S12) > 0 (if specified or failure active)
      - tmax > 0 (if specified or failure active)
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0", "Refer_Rho", "rhor"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's moduli: E1 > 0, E2 > 0
    e11 = _extract(["e11", "E11", "e1", "E1", "MAT_EA", "EA", "MAT_E1"], default=0.0)
    if e11 <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: Young's modulus E1 must be > 0 (got {e11:g})", "MAT CHECK")

    e22 = _extract(["e22", "E22", "e2", "E2", "MAT_EB", "EB", "MAT_E2"], default=0.0)
    if e22 <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: Young's modulus E2 must be > 0 (got {e22:g})", "MAT CHECK")

    # 3. Poisson's ratio determinant: detc = 1 - nu12*nu21 > 0
    nu12 = _extract(["nu12", "NU12", "MAT_PRAB", "nu", "NU"], default=0.0)
    nu21 = nu12 * e22 / e11 if e11 > 0.0 else 0.0
    detc = 1.0 - nu12 * nu21
    if detc <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: invalid Poisson ratio, detc = 1 - nu12*nu21 must be > 0 (got {detc:g})", "MAT CHECK")

    # 4. Shear moduli: G12, G23, G31 > 0
    g12 = _extract(["g12", "G12", "MAT_GAB", "GAB"], default=0.0)
    if g12 <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: shear modulus G12 must be > 0 (got {g12:g})", "MAT CHECK")

    g23 = _extract(["g23", "G23", "MAT_GBC", "GBC"], default=0.0)
    if g23 <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: shear modulus G23 must be > 0 (got {g23:g})", "MAT CHECK")

    g31 = _extract(["g31", "G31", "MAT_GCA", "GCA"], default=0.0)
    if g31 <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: shear modulus G31 must be > 0 (got {g31:g})", "MAT CHECK")

    # 5. Yield stresses: sig_1yt, sig_2yt, sig_1yc, sig_2yc, sig_12yc, sig_12yt > 0
    sig_1yt = _extract(["sig_1yt", "sig1yt", "sigyt1", "SIG_1YT", "MAT_SIGYT1", "SIGYT1"], default=0.0)
    sig_2yt = _extract(["sig_2yt", "sig2yt", "sigyt2", "SIG_2YT", "MAT_SIGYT2", "SIGYT2"], default=0.0)
    sig_1yc = _extract(["sig_1yc", "sig1yc", "sigyc1", "SIG_1YC", "MAT_SIGYC1", "SIGYC1"], default=0.0)
    sig_2yc = _extract(["sig_2yc", "sig2yc", "sigyc2", "SIG_2YC", "MAT_SIGYC2", "SIGYC2"], default=0.0)
    sig_12yc = _extract(["sig_12yc", "sig12yc", "sigc12", "SIG_12YC", "MAT_SIGC12", "SIGC12"], default=0.0)
    sig_12yt = _extract(["sig_12yt", "sig12yt", "sigt12", "SIG_12YT", "MAT_SIGT12", "SIGT12"], default=0.0)

    if sig_1yt <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: tensile yield stress in dir 1 (sig_1yt) must be > 0 (got {sig_1yt:g})", "MAT CHECK")
    if sig_1yc <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: compressive yield stress in dir 1 (sig_1yc) must be > 0 (got {sig_1yc:g})", "MAT CHECK")
    if sig_2yt <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: tensile yield stress in dir 2 (sig_2yt) must be > 0 (got {sig_2yt:g})", "MAT CHECK")
    if sig_2yc <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: compressive yield stress in dir 2 (sig_2yc) must be > 0 (got {sig_2yc:g})", "MAT CHECK")
    if sig_12yc <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: compressive shear yield stress in dir 12 (sig_12yc) must be > 0 (got {sig_12yc:g})", "MAT CHECK")
    if sig_12yt <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: tensile shear yield stress in dir 12 (sig_12yt) must be > 0 (got {sig_12yt:g})", "MAT CHECK")

    # 6. Hardening parameter b <= 1.0, n <= 1.0
    b = _extract(["b", "B", "MAT_BETA", "cb", "CB"], default=0.0)
    if b > 1.0:
        log.error(f"/MAT/LAW15/{mid}: hardening parameter b must be <= 1.0 (got {b:g})", "MAT CHECK")

    n = _extract(["n", "N", "MAT_HARD", "cn", "CN"], default=1.0)
    if n > 1.0:
        log.error(f"/MAT/LAW15/{mid}: hardening exponent n must be <= 1.0 (got {n:g})", "MAT CHECK")

    # 7. Strengths S1, S2, C1, C2, S12 > 0 and relaxation time tmax > 0
    s1 = _extract(["s1", "S1", "MCHANG_S1"], default=0.0)
    s2 = _extract(["s2", "S2", "MCHANG_S2"], default=0.0)
    s12 = _extract(["s12", "S12", "MCHANG_S12"], default=0.0)
    c1 = _extract(["c1", "C1", "MCHANG_C1"], default=0.0)
    c2 = _extract(["c2", "C2", "MCHANG_C2"], default=0.0)
    tmax = _extract(["tmax", "TMAX", "MAT_TMAX"], default=0.0)

    if s1 < 0.0:
        log.error(f"/MAT/LAW15/{mid}: longitudinal tensile strength S1 must be > 0 (got {s1:g})", "MAT CHECK")
    if s2 < 0.0:
        log.error(f"/MAT/LAW15/{mid}: transverse tensile strength S2 must be > 0 (got {s2:g})", "MAT CHECK")
    if s12 < 0.0:
        log.error(f"/MAT/LAW15/{mid}: shear strength S12 must be > 0 (got {s12:g})", "MAT CHECK")
    if c1 < 0.0:
        log.error(f"/MAT/LAW15/{mid}: longitudinal compressive strength C1 must be > 0 (got {c1:g})", "MAT CHECK")
    if c2 < 0.0:
        log.error(f"/MAT/LAW15/{mid}: transverse compressive strength C2 must be > 0 (got {c2:g})", "MAT CHECK")
    if tmax < 0.0:
        log.error(f"/MAT/LAW15/{mid}: stress relaxation time tmax must be > 0 (got {tmax:g})", "MAT CHECK")

    failure_active = any(v > 0.0 for v in (s1, s2, s12, c1, c2, tmax))
    if failure_active:
        if s1 <= 0.0:
            log.error(f"/MAT/LAW15/{mid}: longitudinal tensile strength S1 must be > 0 (got {s1:g})", "MAT CHECK")
        if s2 <= 0.0:
            log.error(f"/MAT/LAW15/{mid}: transverse tensile strength S2 must be > 0 (got {s2:g})", "MAT CHECK")
        if s12 <= 0.0:
            log.error(f"/MAT/LAW15/{mid}: shear strength S12 must be > 0 (got {s12:g})", "MAT CHECK")
        if c1 <= 0.0:
            log.error(f"/MAT/LAW15/{mid}: longitudinal compressive strength C1 must be > 0 (got {c1:g})", "MAT CHECK")
        if c2 <= 0.0:
            log.error(f"/MAT/LAW15/{mid}: transverse compressive strength C2 must be > 0 (got {c2:g})", "MAT CHECK")
        if tmax <= 0.0:
            log.error(f"/MAT/LAW15/{mid}: stress relaxation time tmax must be > 0 (got {tmax:g})", "MAT CHECK")


def check_mat_law22(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW22 (/MAT/DAMA, /MAT/PLAS_DAMA) parameter bounds (M545).

    Required checks:
      - rho0 > 0
      - E > 0
      - 0 <= nu < 0.5
      - a > 0 (sigy)
      - n <= 1.0 (error if n > 1.0 per hm_read_mat22.F line 213)
      - eps_dot_0 > 0 (error if srp <= 0 per hm_read_mat22.F line 221)
      - e_tan <= 0.0 (error if e_tan > 0 per hm_read_mat22.F line 229 / matl22_dama.cfg line 109)
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0", "Refer_Rho", "rhor"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW22/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's modulus E > 0
    e = _extract(["e", "E", "MAT_E", "young"], default=0.0)
    if e <= 0.0:
        log.error(f"/MAT/LAW22/{mid}: Young's modulus E must be > 0 (got {e:g})", "MAT CHECK")

    # 3. Poisson's ratio: 0 <= nu < 0.5
    nu = _extract(["nu", "NU", "MAT_NU", "anu"], default=0.0)
    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW22/{mid}: Poisson's ratio NU must be in [0, 0.5) (got {nu:g})", "MAT CHECK")

    # 4. Yield stress a (sigy) > 0
    a = _extract(["a", "A", "sigy", "SIGY", "MAT_SIGY", "ca", "CA"], default=0.0)
    if a <= 0.0:
        log.error(f"/MAT/LAW22/{mid}: yield stress a (SIGY) must be > 0 (got {a:g})", "MAT CHECK")

    # 5. Hardening exponent n <= 1.0
    n = _extract(["n", "N", "hard", "HARD", "MAT_HARD", "cn", "CN"], default=1.0)
    if n > 1.0:
        log.error(f"/MAT/LAW22/{mid}: hardening exponent n must be <= 1.0 (got {n:g})", "MAT CHECK")

    # 6. Reference strain rate eps_dot_0 > 0
    eps_dot_0 = _extract(["eps_dot_0", "eps0", "EPS0", "srp", "SRP", "MAT_SRP", "EPS_DOT_0"], default=1.0)
    if eps_dot_0 <= 0.0:
        log.error(f"/MAT/LAW22/{mid}: reference strain rate EPS_DOT_0 (srp) must be > 0 (got {eps_dot_0:g})", "MAT CHECK")

    # 7. Softening damage slope e_tan <= 0.0
    e_tan = _extract(["e_tan", "etan", "e_t", "MAT_ETAN", "EL", "E_TAN"], default=0.0)
    if e_tan > 0.0:
        log.error(f"/MAT/LAW22/{mid}: softening damage slope E_tan must be <= 0.0 (got {e_tan:g})", "MAT CHECK")


def check_mat_law12(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW12 (/MAT/3D_COMP, /MAT/COMP_3D) parameter bounds (M546).

    Required checks (hm_read_mat12.F):
      - rho0 > 0
      - E11, E22, E33 > 0
      - detc > 0 (compliance matrix determinant)
      - G12, G23, G31 >= 0
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0", "Refer_Rho", "rhor"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW12/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's moduli E11, E22, E33 > 0
    e11 = _extract(["e11", "E11", "MAT_EA", "ea", "EA"], default=0.0)
    e22 = _extract(["e22", "E22", "MAT_EB", "eb", "EB"], default=0.0)
    e33 = _extract(["e33", "E33", "MAT_EC", "ec", "EC"], default=0.0)

    if e11 <= 0.0 or e22 <= 0.0 or e33 <= 0.0:
        log.error(
            f"/MAT/LAW12/{mid}: Young's moduli E11, E22, E33 must be > 0 (got E11={e11:g}, E22={e22:g}, E33={e33:g}) (ANCMSG 306)",
            "MAT CHECK",
        )
    else:
        # 3. Determinant of compliance matrix DETC > 0
        nu12 = _extract(["nu12", "NU12", "MAT_PRAB", "prab", "PRAB"], default=0.0)
        nu23 = _extract(["nu23", "NU23", "MAT_PRBC", "prbc", "PRBC"], default=0.0)
        nu31 = _extract(["nu31", "NU31", "MAT_PRCA", "prca", "PRCA"], default=0.0)

        c11 = 1.0 / e11
        c22 = 1.0 / e22
        c33 = 1.0 / e33
        c12 = -nu12 / e11
        c13 = -nu31 / e33
        c23 = -nu23 / e22

        detc = (
            c11 * c22 * c33
            - c11 * (c23**2)
            - (c12**2) * c33
            + 2.0 * c12 * c13 * c23
            - (c13**2) * c22
        )
        if detc <= 0.0:
            log.error(f"/MAT/LAW12/{mid}: compliance matrix determinant DETC must be > 0 (got {detc:g}) (ANCMSG 307)", "MAT CHECK")

    # 4. Shear moduli
    g12 = _extract(["g12", "G12", "MAT_GAB", "gab", "GAB"], default=0.0)
    g23 = _extract(["g23", "G23", "MAT_GBC", "gbc", "GBC"], default=0.0)
    g31 = _extract(["g31", "G31", "MAT_GCA", "gca", "GCA"], default=0.0)
    if g12 < 0.0 or g23 < 0.0 or g31 < 0.0:
        log.error(f"/MAT/LAW12/{mid}: shear moduli G12, G23, G31 must be >= 0", "MAT CHECK")


def check_mat_law14(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW14 (/MAT/COMPSO, /MAT/COMP_SOL) parameter bounds (M547).

    Required checks (hm_read_mat14.F):
      - rho0 > 0
      - E11, E22, E33 > 0
      - detc > 0 (compliance matrix determinant)
      - G12, G23, G31 >= 0
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    # Cam-Clay check bypass (M187)
    is_cam_clay = (
        (hasattr(mat, "kappa") or "kappa" in params or "KAPPA" in params)
        and not hasattr(mat, "ea")
        and "e11" not in params
        and "E11" not in params
        and "MAT_EA" not in params
    )
    if is_cam_clay:
        rho0 = getattr(mat, "rho0", None)
        if rho0 is None:
            rho0 = getattr(mat, "rho", params.get("rho", params.get("MAT_RHO", 0.0)))
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0
        if rho0 <= 0.0:
            log.error(f"/MAT/LAW14/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")
        return

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0", "Refer_Rho", "rhor"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW14/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's moduli E11, E22, E33 > 0
    e11 = _extract(["e11", "E11", "MAT_EA", "ea", "EA"], default=0.0)
    e22 = _extract(["e22", "E22", "MAT_EB", "eb", "EB"], default=0.0)
    e33 = _extract(["e33", "E33", "MAT_EC", "ec", "EC"], default=0.0)

    if e11 <= 0.0 or e22 <= 0.0 or e33 <= 0.0:
        log.error(
            f"/MAT/LAW14/{mid}: Young's moduli E11, E22, E33 must be > 0 (got E11={e11:g}, E22={e22:g}, E33={e33:g}) (ANCMSG 306)",
            "MAT CHECK",
        )
    else:
        # 3. Determinant of compliance matrix DETC > 0
        nu12 = _extract(["nu12", "NU12", "MAT_PRAB", "prab", "PRAB"], default=0.0)
        nu23 = _extract(["nu23", "NU23", "MAT_PRBC", "prbc", "PRBC"], default=0.0)
        nu31 = _extract(["nu31", "NU31", "MAT_PRCA", "prca", "PRCA"], default=0.0)

        c11 = 1.0 / e11
        c22 = 1.0 / e22
        c33 = 1.0 / e33
        c12 = -nu12 / e11
        c13 = -nu31 / e33
        c23 = -nu23 / e22

        detc = (
            c11 * c22 * c33
            - c11 * (c23**2)
            - (c12**2) * c33
            + 2.0 * c12 * c13 * c23
            - (c13**2) * c22
        )
        if detc <= 0.0:
            log.error(f"/MAT/LAW14/{mid}: compliance matrix determinant DETC must be > 0 (got {detc:g}) (ANCMSG 307)", "MAT CHECK")

    # 4. Shear moduli
    g12 = _extract(["g12", "G12", "MAT_GAB", "gab", "GAB"], default=0.0)
    g23 = _extract(["g23", "G23", "MAT_GBC", "gbc", "GBC"], default=0.0)
    g31 = _extract(["g31", "G31", "MAT_GCA", "gca", "GCA"], default=0.0)
    if g12 < 0.0 or g23 < 0.0 or g31 < 0.0:
        log.error(f"/MAT/LAW14/{mid}: shear moduli G12, G23, G31 must be >= 0", "MAT CHECK")


def check_mat_law25(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW25 (/MAT/COMP_PLAS, /MAT/COMPSH, /MAT/TSAI_WU, /MAT/CRASURV) parameter bounds (M543).

    Required checks:
      - rho0 > 0
      - E1 > 0, E2 > 0
      - detc = 1 - nu12*nu21 > 0
      - G12, G23, G31 > 0
      - sigyt1, sigyc1, sigyt2, sigyc2, sigt12, sigc12 > 0
      - n <= 1.0
      - 0 <= dmax <= 1.0
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0", "Refer_Rho", "rhor"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's moduli: E1 > 0, E2 > 0
    e11 = _extract(["e11", "E11", "e1", "E1", "MAT_EA", "EA", "MAT_E1"], default=0.0)
    if e11 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: Young's modulus E1 must be > 0 (got {e11:g})", "MAT CHECK")

    e22 = _extract(["e22", "E22", "e2", "E2", "MAT_EB", "EB", "MAT_E2"], default=0.0)
    if e22 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: Young's modulus E2 must be > 0 (got {e22:g})", "MAT CHECK")

    # 3. Poisson's ratio determinant: detc = 1 - nu12*nu21 > 0
    nu12 = _extract(["nu12", "NU12", "MAT_PRAB", "nu", "NU"], default=0.0)
    nu21 = nu12 * e22 / e11 if e11 > 0.0 else 0.0
    detc = 1.0 - nu12 * nu21
    if detc <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: invalid Poisson ratio, detc = 1 - nu12*nu21 must be > 0 (got {detc:g})", "MAT CHECK")

    # 4. Shear moduli: G12, G23, G31 > 0
    g12 = _extract(["g12", "G12", "MAT_GAB", "GAB"], default=0.0)
    if g12 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: shear modulus G12 must be > 0 (got {g12:g})", "MAT CHECK")

    g23 = _extract(["g23", "G23", "MAT_GBC", "GBC"], default=0.0)
    if g23 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: shear modulus G23 must be > 0 (got {g23:g})", "MAT CHECK")

    g31 = _extract(["g31", "G31", "MAT_GCA", "GCA"], default=0.0)
    if g31 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: shear modulus G31 must be > 0 (got {g31:g})", "MAT CHECK")

    # 5. Yield stresses: sigyt1, sigyc1, sigyt2, sigyc2, sigt12, sigc12 > 0
    sigyt1 = _extract(["sigyt1", "sig_1yt", "MAT_SIGYT1", "SIGYT1", "sigyt_1"], default=0.0)
    sigyc1 = _extract(["sigyc1", "sig_1yc", "MAT_SIGYC1", "SIGYC1", "MAT_SIG1_yc", "sigyc_1"], default=0.0)
    sigyt2 = _extract(["sigyt2", "sig_2yt", "MAT_SIGYT2", "SIGYT2", "sigyt_2"], default=0.0)
    sigyc2 = _extract(["sigyc2", "sig_2yc", "MAT_SIGYC2", "SIGYC2", "MAT_SIG2_yc", "sigyc_2"], default=0.0)
    sigt12 = _extract(["sigt12", "sigyt12", "sig_12yt", "MAT_SIGT12", "SIGT12", "MAT_SIG12_yt"], default=0.0)
    sigc12 = _extract(["sigc12", "sigyc12", "sig_12yc", "MAT_SIGC12", "SIGC12"], default=sigt12)

    if sigyt1 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: tensile yield stress in dir 1 (sigyt1) must be > 0 (got {sigyt1:g})", "MAT CHECK")
    if sigyc1 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: compressive yield stress in dir 1 (sigyc1) must be > 0 (got {sigyc1:g})", "MAT CHECK")
    if sigyt2 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: tensile yield stress in dir 2 (sigyt2) must be > 0 (got {sigyt2:g})", "MAT CHECK")
    if sigyc2 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: compressive yield stress in dir 2 (sigyc2) must be > 0 (got {sigyc2:g})", "MAT CHECK")
    if sigt12 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: tensile shear yield stress in dir 12 (sigt12) must be > 0 (got {sigt12:g})", "MAT CHECK")
    if sigc12 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: compressive shear yield stress in dir 12 (sigc12) must be > 0 (got {sigc12:g})", "MAT CHECK")

    # 6. Hardening exponent: n <= 1.0
    iform = 0
    for k in ["iform", "MAT_Iflag", "iflag", "IFORM", "IFLAG"]:
        if hasattr(mat, k) and getattr(mat, k) is not None:
            try:
                iform = int(getattr(mat, k))
                break
            except (TypeError, ValueError):
                pass
        if isinstance(params, dict) and k in params and params[k] is not None:
            try:
                iform = int(params[k])
                break
            except (TypeError, ValueError):
                pass

    if iform == 0:
        n = _extract(["n", "N", "MAT_HARD", "hard", "cn"], default=1.0)
        if n > 1.0:
            log.error(f"/MAT/LAW25/{mid}: hardening exponent n must be <= 1.0 (got {n:g})", "MAT CHECK")
    else:
        n1_t = _extract(["n1_t", "n_1t", "MAT_n1_t", "cnt1"], default=1.0)
        n1_c = _extract(["n1_c", "n_1c", "MAT_n1_c", "cnc1"], default=1.0)
        n2_t = _extract(["n2_t", "n_2t", "MAT_n2_t", "cnt2"], default=1.0)
        n2_c = _extract(["n2_c", "n_2c", "MAT_n2_c", "cnc2"], default=1.0)
        n12_t = _extract(["n12_t", "n_12t", "MAT_n12_t", "cnt12"], default=1.0)
        if n1_t > 1.0 or n1_c > 1.0 or n2_t > 1.0 or n2_c > 1.0 or n12_t > 1.0:
            log.error(f"/MAT/LAW25/{mid}: hardening exponent n must be <= 1.0", "MAT CHECK")

    # 7. Maximum damage: 0 <= dmax <= 1.0
    dmax = _extract(["dmax", "MAT_DAMAGE", "d_max", "damage"], default=0.0)
    if dmax < 0.0 or dmax > 1.0:
        log.error(f"/MAT/LAW25/{mid}: maximum damage dmax must be in [0, 1] (got {dmax:g})", "MAT CHECK")

    # 8. Damage strains: epsm >= epst
    epst1 = _extract(["eps_t1", "epst1", "MAT_EPST1", "EPST1", "eps_t", "epst"], default=0.0)
    epsm1 = _extract(["eps_m1", "epsm1", "MAT_EPSM1", "EPSM1", "eps_m", "epsm"], default=0.0)
    epst2 = _extract(["eps_t2", "epst2", "MAT_EPST2", "EPST2"], default=0.0)
    epsm2 = _extract(["eps_m2", "epsm2", "MAT_EPSM2", "EPSM2"], default=0.0)

    if (epst1 > 0.0 or epsm1 > 0.0) and epsm1 < epst1:
        log.error(f"/MAT/LAW25/{mid}: maximum damage strain eps_m1 must be >= damage initiation strain eps_t1 (got eps_m1={epsm1:g}, eps_t1={epst1:g})", "MAT CHECK")
    if (epst2 > 0.0 or epsm2 > 0.0) and epsm2 < epst2:
        log.error(f"/MAT/LAW25/{mid}: maximum damage strain eps_m2 must be >= damage initiation strain eps_t2 (got eps_m2={epsm2:g}, eps_t2={epst2:g})", "MAT CHECK")

    if iform != 0:
        eps_1t1 = _extract(["eps_1t1", "MAT_EPS1_t1"], default=0.0)
        eps_2t1 = _extract(["eps_2t1", "MAT_EPS2_t1"], default=0.0)
        if (eps_1t1 > 0.0 or eps_2t1 > 0.0) and eps_2t1 < eps_1t1:
            log.error(f"/MAT/LAW25/{mid}: CRASURV maximum damage strain eps_2t1 must be >= eps_1t1 (got eps_2t1={eps_2t1:g}, eps_1t1={eps_1t1:g})", "MAT CHECK")
        eps_1t2 = _extract(["eps_1t2", "MAT_EPS1_t2"], default=0.0)
        eps_2t2 = _extract(["eps_2t2", "MAT_EPS2_t2"], default=0.0)
        if (eps_1t2 > 0.0 or eps_2t2 > 0.0) and eps_2t2 < eps_1t2:
            log.error(f"/MAT/LAW25/{mid}: CRASURV maximum damage strain eps_2t2 must be >= eps_1t2 (got eps_2t2={eps_2t2:g}, eps_1t2={eps_1t2:g})", "MAT CHECK")
        eps_1c1 = _extract(["eps_1c1", "MAT_EPS1_c1"], default=0.0)
        eps_2c1 = _extract(["eps_2c1", "MAT_EPS2_c1"], default=0.0)
        if (eps_1c1 > 0.0 or eps_2c1 > 0.0) and eps_2c1 < eps_1c1:
            log.error(f"/MAT/LAW25/{mid}: CRASURV maximum damage strain eps_2c1 must be >= eps_1c1 (got eps_2c1={eps_2c1:g}, eps_1c1={eps_1c1:g})", "MAT CHECK")
        eps_1c2 = _extract(["eps_1c2", "MAT_EPS1_c2"], default=0.0)
        eps_2c2 = _extract(["eps_2c2", "MAT_EPS2_c2"], default=0.0)
        if (eps_1c2 > 0.0 or eps_2c2 > 0.0) and eps_2c2 < eps_1c2:
            log.error(f"/MAT/LAW25/{mid}: CRASURV maximum damage strain eps_2c2 must be >= eps_1c2 (got eps_2c2={eps_2c2:g}, eps_1c2={eps_1c2:g})", "MAT CHECK")
        eps_1t12 = _extract(["eps_1t12", "MAT_EPS1_t12"], default=0.0)
        eps_2t12 = _extract(["eps_2t12", "MAT_EPS2_t12"], default=0.0)
        if (eps_1t12 > 0.0 or eps_2t12 > 0.0) and eps_2t12 < eps_1t12:
            log.error(f"/MAT/LAW25/{mid}: CRASURV maximum damage strain eps_2t12 must be >= eps_1t12 (got eps_2t12={eps_2t12:g}, eps_1t12={eps_1t12:g})", "MAT CHECK")



def check_mat_law43(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW43 (/MAT/HILL_TAB, /MAT/HILL_PLAS_TAB) parameter bounds (M548).

    Citing hm_read_mat43.F and radioss140/MAT/matl43_HILL_TAB.cfg:
      - rho0 > 0
      - e > 0
      - 0 <= nu < 0.5
      - r00 > 0, r45 > 0, r90 > 0
      - 0 <= fisokin <= 1 (ANCMSG 913)
      - at least one plasticity curve defined with non-zero ID (ANCMSG 366)
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0", "Refer_Rho", "rhor"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW43/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's modulus E > 0
    e = _extract(["e", "E", "MAT_E", "young", "E0"], default=0.0)
    if e <= 0.0:
        log.error(f"/MAT/LAW43/{mid}: Young's modulus E must be > 0 (got {e:g})", "MAT CHECK")

    # 3. Poisson's ratio: 0 <= nu < 0.5
    nu = _extract(["nu", "NU", "MAT_NU", "anu"], default=0.0)
    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW43/{mid}: Poisson's ratio NU must satisfy 0 <= NU < 0.5 (got {nu:g})", "MAT CHECK")

    # 4. Lankford coefficients: R00 > 0, R45 > 0, R90 > 0
    r00 = _extract(["r00", "R00", "MAT_R00", "r0", "R0"], default=1.0)
    if r00 <= 0.0:
        log.error(f"/MAT/LAW43/{mid}: Lankford coefficient R00 must be > 0 (got {r00:g})", "MAT CHECK")

    r45 = _extract(["r45", "R45", "MAT_R45", "r_45"], default=1.0)
    if r45 <= 0.0:
        log.error(f"/MAT/LAW43/{mid}: Lankford coefficient R45 must be > 0 (got {r45:g})", "MAT CHECK")

    r90 = _extract(["r90", "R90", "MAT_R90", "r_90"], default=1.0)
    if r90 <= 0.0:
        log.error(f"/MAT/LAW43/{mid}: Lankford coefficient R90 must be > 0 (got {r90:g})", "MAT CHECK")

    # 5. Iso-kinematic hardening factor: 0 <= FISOKIN <= 1 (ANCMSG 913)
    fisokin = _extract(["fisokin", "FISOKIN", "chard", "CHARD", "MAT_CHARD", "c_hard"], default=0.0)
    if fisokin < 0.0 or fisokin > 1.0:
        log.error(
            f"/MAT/LAW43/{mid}: iso-kinematic hardening factor FISOKIN must be in [0, 1] (got {fisokin:g}) (ANCMSG 913)",
            "MAT CHECK",
        )

    # 6. Plasticity hardening curves: must have at least one valid curve (ANCMSG 366)
    curves = getattr(mat, "curves", None)
    if curves is None and isinstance(params, dict):
        curves = params.get("curves", None)

    has_curve = False
    if curves:
        for c in curves:
            if isinstance(c, dict):
                fid = c.get("fct_id", c.get("funct_id", c.get("id", 0)))
                pts = c.get("points", c.get("pts", None))
                if fid or pts is not None:
                    has_curve = True
                    break
            elif isinstance(c, (list, tuple)):
                if len(c) > 0 and c[0]:
                    has_curve = True
                    break
            elif isinstance(c, (int, float)) and c != 0:
                has_curve = True
                break
    elif isinstance(params, dict):
        cxs = params.get("curve_x", [])
        if len(cxs) > 0:
            has_curve = True

    if not has_curve:
        log.error(f"/MAT/LAW43/{mid}: no plasticity hardening curve defined (ANCMSG 366)", "MAT CHECK")


def check_mat_law82(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW82 (/MAT/OGDEN, /MAT/LAW82_OGDEN) parameter bounds (M549).

    Citing hm_read_mat82.F and radioss110/MAT/matl82_ogden.cfg:
      - rho0 > 0 (MSGERROR if <= 0)
      - nordre >= 1 (MSGERROR if < 1, MSGID 559)
      - sum(mu) > 0 (MSGERROR if <= 0, MSGID 846)
      - 0.0 <= nu < 0.5
      - for each term i, alpha[i] != 0
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW82/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Order of Ogden model: nordre >= 1 (MSGID 559)
    nordre = getattr(mat, "nordre", None)
    if nordre is None:
        nordre = getattr(mat, "order", None)
    if nordre is None and isinstance(params, dict):
        nordre = params.get("ORDER", params.get("nordre", params.get("order", 1)))
    try:
        nordre = int(nordre)
    except (TypeError, ValueError):
        nordre = 0

    if nordre < 1:
        log.error(f"/MAT/LAW82/{mid}: order of Ogden model must be >= 1 (got {nordre}) (MSGID 559)", "MAT CHECK")

    # Extract mu, alpha arrays
    mu = getattr(mat, "mu", None)
    if mu is None:
        mu = getattr(mat, "mu_arr", None)
    if mu is None and isinstance(params, dict):
        mu = params.get("mu", params.get("Mu_arr", []))
    if mu is None:
        mu = []
    mu_vals = [float(x) for x in mu]

    alpha = getattr(mat, "alpha", None)
    if alpha is None:
        alpha = getattr(mat, "alpha_arr", None)
    if alpha is None and isinstance(params, dict):
        alpha = params.get("alpha", params.get("Alpha_arr", []))
    if alpha is None:
        alpha = []
    alpha_vals = [float(x) for x in alpha]

    # 3. sum(mu) > 0 (MSGID 846)
    gs = sum(mu_vals[:nordre]) if nordre > 0 else sum(mu_vals)
    if gs <= 0.0:
        log.error(f"/MAT/LAW82/{mid}: sum of shear moduli mu must be > 0 (got {gs:g}) (MSGID 846)", "MAT CHECK")

    # 4. Poisson's ratio: 0.0 <= nu < 0.5
    nu = _extract(["nu", "MAT_NU", "NU"], default=0.475)
    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW82/{mid}: Poisson ratio nu must satisfy 0.0 <= nu < 0.5 (got {nu:g})", "MAT CHECK")

    # 5. For each term i, alpha[i] != 0
    for i in range(min(nordre, len(alpha_vals))):
        if alpha_vals[i] == 0.0:
            log.error(f"/MAT/LAW82/{mid}: alpha parameter at term {i+1} must be non-zero", "MAT CHECK")


def check_mat_law69(mat: Any, log: MessageLog, functions: Optional[Dict[int, Any]] = None) -> None:
    """Validate /MAT/LAW69 (/MAT/HYP_ELAS, /MAT/HYPERELASTIC) parameter bounds (M550).

    Required checks (citing hm_read_mat69.F and matl69_69.cfg):
      - rho0 > 0
      - 0.0 <= nu < 0.5
      - FCT_ID1 must exist in functions dictionary if specified
      - Monotonicity check of FCT_ID1 curve (abscissae and ordinates)
      - Compatible elements: Solids and Shells supported; reject trusses, beams, springs.
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW69/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Poisson's ratio: 0.0 <= nu < 0.5
    nu = getattr(mat, "nu", None)
    if nu is None:
        nu = _extract(["nu", "MAT_NU", "NU"], default=0.495)
    else:
        try:
            nu = float(nu)
        except (TypeError, ValueError):
            nu = 0.495

    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW69/{mid}: Poisson ratio nu must satisfy 0.0 <= nu < 0.5 (got {nu:g})", "MAT CHECK")

    # 3. Test curve FCT_ID1
    fct_id1 = getattr(mat, "fct_id1", None)
    if fct_id1 is None:
        fct_id1 = getattr(mat, "fct_id_data", None)
    if fct_id1 is None and isinstance(params, dict):
        fct_id1 = params.get("fct_id1", params.get("fct_id_data", params.get("FUN_B1", 0)))
    try:
        fct_id1 = int(fct_id1) if fct_id1 is not None else 0
    except (TypeError, ValueError):
        fct_id1 = 0

    funcs = functions
    if funcs is None and hasattr(mat, "functions"):
        funcs = mat.functions

    if fct_id1 != 0:
        if funcs is not None:
            if fct_id1 not in funcs:
                log.error(f"/MAT/LAW69/{mid}: test curve FCT_ID1={fct_id1} not found in functions dictionary", "MAT CHECK")
            else:
                fn = funcs[fct_id1]
                x = getattr(fn, "x", None)
                y = getattr(fn, "y", None)
                if x is None and isinstance(fn, dict):
                    x = fn.get("x", None)
                    y = fn.get("y", None)
                if x is not None and y is not None:
                    import numpy as np
                    xa = np.asarray(x, dtype=float)
                    ya = np.asarray(y, dtype=float)
                    if xa.size >= 2 and np.any(np.diff(xa) <= 0):
                        log.error(f"/MAT/LAW69/{mid}: test curve FCT_ID1={fct_id1} abscissae must be strictly increasing (monotonicity violation)", "MAT CHECK")
                    if ya.size >= 2 and np.any(np.diff(ya) < 0):
                        log.error(f"/MAT/LAW69/{mid}: test curve FCT_ID1={fct_id1} ordinates must be monotonically increasing (monotonicity violation)", "MAT CHECK")


def check_mat_law60(
    mat: Any,
    arg1: Any = None,
    arg2: Any = None,
    model: Any = None,
    log: Any = None,
    functions: Any = None,
    **kwargs,
) -> None:
    """Validate /MAT/LAW60 (/MAT/PLAS_T3, /MAT/FABRIC) parameter bounds (M551).

    Required checks (citing hm_read_mat60.F and matl60_PLAS_T3.cfg):
      - rho0 > 0
      - E > 0
      - 0.0 <= nu < 0.5
      - If eps_t1 > 0 and eps_t2 > 0: eps_t1 < eps_t2
      - Monotonic rates: rates[i] < rates[i+1]
      - Curve existence in model.tables / model.functions / functions
      - Compatible elements: Solids and Shells supported; reject trusses, beams, springs.
    """
    actual_log = log
    actual_model = model

    if isinstance(arg1, MessageLog):
        actual_log = arg1
        if arg2 is not None and not isinstance(arg2, MessageLog):
            actual_model = arg2
    elif isinstance(arg2, MessageLog):
        actual_log = arg2
        if arg1 is not None:
            actual_model = arg1
    else:
        if arg1 is not None and actual_model is None:
            actual_model = arg1
        if arg2 is not None and actual_log is None:
            actual_log = arg2

    if actual_log is None:
        actual_log = MessageLog()
    log = actual_log
    model = actual_model

    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = getattr(mat, "rho", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW60/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's modulus E > 0
    e = getattr(mat, "e", None)
    if e is None:
        e = getattr(mat, "E", None)
    if e is None:
        e = _extract(["e", "E", "MAT_E", "young"], default=0.0)
    else:
        try:
            e = float(e)
        except (TypeError, ValueError):
            e = 0.0

    if e <= 0.0:
        log.error(f"/MAT/LAW60/{mid}: Young's modulus E must be > 0 (got {e:g})", "MAT CHECK")

    # 3. Poisson's ratio: 0.0 <= nu < 0.5
    nu = getattr(mat, "nu", None)
    if nu is None:
        nu = _extract(["nu", "MAT_NU", "NU", "poisson"], default=0.0)
    else:
        try:
            nu = float(nu)
        except (TypeError, ValueError):
            nu = 0.0

    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW60/{mid}: Poisson's ratio nu must satisfy 0.0 <= nu < 0.5 (got {nu:g})", "MAT CHECK")

    # 4. Tensile failure strains: if eps_t1 > 0 and eps_t2 > 0: eps_t1 < eps_t2
    eps_t1 = getattr(mat, "eps_t1", None)
    if eps_t1 is None:
        eps_t1 = _extract(["eps_t1", "EPST1", "epst1"], default=1.0e30)
    else:
        try:
            eps_t1 = float(eps_t1)
        except (TypeError, ValueError):
            eps_t1 = 1.0e30

    eps_t2 = getattr(mat, "eps_t2", None)
    if eps_t2 is None:
        eps_t2 = _extract(["eps_t2", "EPST2", "epst2"], default=2.0e30)
    else:
        try:
            eps_t2 = float(eps_t2)
        except (TypeError, ValueError):
            eps_t2 = 2.0e30

    if eps_t1 > 0.0 and eps_t2 > 0.0 and eps_t1 >= eps_t2:
        log.error(
            f"/MAT/LAW60/{mid}: tensile failure strains must satisfy eps_t1 < eps_t2 (got eps_t1={eps_t1:g}, eps_t2={eps_t2:g})",
            "MAT CHECK",
        )

    # 5. Monotonic rates: rates[i] < rates[i+1]
    rates = getattr(mat, "rates", None)
    if rates is None:
        rates = getattr(mat, "eps_rates", None)
    if rates is None and isinstance(params, dict):
        rates = params.get("rates", params.get("eps_rates", []))
    if rates is not None and len(rates) > 0:
        rates_list = []
        for r in rates:
            try:
                rates_list.append(float(r))
            except (TypeError, ValueError):
                pass
        for i in range(len(rates_list) - 1):
            if rates_list[i] >= rates_list[i + 1]:
                log.error(
                    f"/MAT/LAW60/{mid}: strain rates must be strictly increasing (got rates[{i}]={rates_list[i]:g} >= rates[{i+1}]={rates_list[i+1]:g})",
                    "MAT CHECK",
                )
                break

    # 6. Curve existence
    funcs_list = getattr(mat, "funcs", None)
    if funcs_list is None:
        funcs_list = getattr(mat, "fun_ids", None)
    if funcs_list is None and isinstance(params, dict):
        funcs_list = params.get("funcs", params.get("fun_ids", []))

    avail_curves = set()
    if model is not None:
        if hasattr(model, "tables") and model.tables:
            avail_curves.update(model.tables.keys())
        if hasattr(model, "functions") and model.functions:
            avail_curves.update(model.functions.keys())
    if functions is not None:
        avail_curves.update(functions.keys())
    if hasattr(mat, "functions") and mat.functions:
        avail_curves.update(mat.functions.keys())
    if hasattr(mat, "tables") and mat.tables:
        avail_curves.update(mat.tables.keys())

    if avail_curves and funcs_list is not None and len(funcs_list) > 0:
        for fid in funcs_list:
            try:
                ifid = int(fid)
            except (TypeError, ValueError):
                continue
            if ifid != 0 and ifid not in avail_curves:
                log.error(
                    f"/MAT/LAW60/{mid}: function curve {ifid} not found in model tables or functions",
                    "MAT CHECK",
                )

    # 7. Reject trusses, beams, springs if attached to element groups
    if model is not None and hasattr(model, "element_groups"):
        try:
            grps = model.element_groups()
            if callable(grps):
                grps = grps()
        except TypeError:
            grps = []
        for item in grps:
            if isinstance(item, tuple) and len(item) == 2:
                name, el_group = item
            else:
                continue
            if name in ("trusses", "beams", "springs"):
                mids_in_group = set()
                if hasattr(el_group, "values") and callable(el_group.values):
                    for el in el_group.values():
                        el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                        if el_mid is not None:
                            mids_in_group.add(el_mid)
                if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                    for _, m, _ in el_group.state["slices"]:
                        m_id = getattr(m, "id", None)
                        if m_id is not None:
                            mids_in_group.add(m_id)
                if mid in mids_in_group:
                    log.error(
                        f"/MAT/LAW60/{mid} (/MAT/PLAS_T3) is not supported for {name} elements "
                        f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads)",
                        "MAT CHECK",
                    )


_check_mat_law60 = check_mat_law60


def check_mat_law48(
    mat: Any = None,
    arg1: Any = None,
    arg2: Any = None,
    model: Any = None,
    log: Any = None,
    **kwargs,
) -> None:
    """Validate /MAT/LAW48 (/MAT/ZHAO, /MAT/PLAS_ZHAO) parameter bounds (M552).

    Required checks (citing hm_read_mat48.F and matl48_zhao.cfg):
      - rho0 > 0
      - E > 0
      - 0.0 <= nu < 0.5 (ANCMSG 49)
      - sigy > 0 (or a > 0)
      - 0.0 <= n <= 1.0
      - b >= 0
      - sig_max >= 0
      - eps_t1 >= 0, eps_t2 >= 0, eps_max >= 0
      - If eps_t1 < 1.0e20: eps_t2 > eps_t1 (ANCMSG 420)
      - Warning: if c > 0 and eps_rate_0 > 0 and fcut == 0 (ANCMSG 1220)
      - Compatible elements: Solids and Shells supported; reject trusses, beams, springs.
    """
    actual_log = log
    actual_model = model
    actual_mat = mat

    candidates = [c for c in (mat, arg1, arg2) if c is not None]
    for c in candidates:
        if isinstance(c, MessageLog):
            actual_log = c
        elif isinstance(c, Model):
            actual_model = c
        elif actual_mat is None or actual_mat is c:
            if not isinstance(c, (MessageLog, Model)):
                actual_mat = c

    if actual_mat is None:
        if "mat" in kwargs:
            actual_mat = kwargs["mat"]
        elif "m" in kwargs:
            actual_mat = kwargs["m"]
        elif "mat_law48" in kwargs:
            actual_mat = kwargs["mat_law48"]

    mid_kw = kwargs.get("mat_id", kwargs.get("mid", None))
    if actual_mat is None and actual_model is not None and mid_kw is not None:
        actual_mat = actual_model.materials.get(mid_kw) or getattr(actual_model, "mat_law48s", {}).get(mid_kw)

    if actual_log is None:
        actual_log = MessageLog()
    log = actual_log
    model = actual_model
    mat = actual_mat

    if mat is None:
        return

    mid = getattr(mat, "id", kwargs.get("mat_id", kwargs.get("mid", 0)))
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if k in kwargs:
                val = kwargs[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = getattr(mat, "rho", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW48/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's modulus E > 0
    e = getattr(mat, "e", None)
    if e is None:
        e = getattr(mat, "E", None)
    if e is None:
        e = _extract(["e", "E", "MAT_E", "young"], default=0.0)
    else:
        try:
            e = float(e)
        except (TypeError, ValueError):
            e = 0.0

    if e <= 0.0:
        log.error(f"/MAT/LAW48/{mid}: Young's modulus E must be > 0 (got {e:g})", "MAT CHECK")

    # 3. Poisson's ratio: 0.0 <= nu < 0.5 (ANCMSG 49)
    nu = getattr(mat, "nu", None)
    if nu is None:
        nu = _extract(["nu", "MAT_NU", "NU", "poisson"], default=0.0)
    else:
        try:
            nu = float(nu)
        except (TypeError, ValueError):
            nu = 0.0

    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW48/{mid}: Poisson's ratio nu must satisfy 0.0 <= nu < 0.5 (got {nu:g}) (ANCMSG 49)", "MAT CHECK")

    # 4. Yield stress sigy / a > 0
    sigy = getattr(mat, "sigy", None)
    if sigy is None:
        sigy = getattr(mat, "a", None)
    if sigy is None:
        sigy = _extract(["sigy", "a", "MAT_SIGY", "sigy0"], default=0.0)
    else:
        try:
            sigy = float(sigy)
        except (TypeError, ValueError):
            sigy = 0.0

    if sigy <= 0.0:
        log.error(f"/MAT/LAW48/{mid}: initial yield stress SIGY/A must be > 0 (got {sigy:g})", "MAT CHECK")

    # 5. Hardening exponent: 0.0 <= n <= 1.0
    n = getattr(mat, "n", None)
    if n is None:
        n = _extract(["n", "cn", "MAT_N"], default=1.0)
    else:
        try:
            n = float(n)
        except (TypeError, ValueError):
            n = 1.0

    if n < 0.0 or n > 1.001:
        log.error(f"/MAT/LAW48/{mid}: hardening exponent n must satisfy 0.0 <= n <= 1.0 (got {n:g})", "MAT CHECK")

    # 6. Hardening parameter b >= 0
    b = getattr(mat, "b", None)
    if b is None:
        b = _extract(["b", "cb", "MAT_B"], default=0.0)
    else:
        try:
            b = float(b)
        except (TypeError, ValueError):
            b = 0.0

    if b < 0.0:
        log.error(f"/MAT/LAW48/{mid}: hardening parameter B must be >= 0 (got {b:g})", "MAT CHECK")

    # 7. Maximum stress sig_max >= 0
    sig_max = getattr(mat, "sig_max", None)
    if sig_max is None:
        sig_max = getattr(mat, "sigm", None)
    if sig_max is None:
        sig_max = _extract(["sig_max", "sigm", "sigma_max", "MAT_SIG"], default=1.0e30)
    else:
        try:
            sig_max = float(sig_max)
        except (TypeError, ValueError):
            sig_max = 1.0e30

    if sig_max < 0.0:
        log.error(f"/MAT/LAW48/{mid}: maximum stress SIG_MAX must be >= 0 (got {sig_max:g})", "MAT CHECK")

    # 8. Failure strains: eps_t1 >= 0, eps_t2 >= 0, eps_max >= 0
    eps_t1 = getattr(mat, "eps_t1", None)
    if eps_t1 is None:
        eps_t1 = getattr(mat, "eta1", None)
    if eps_t1 is None:
        eps_t1 = _extract(["eps_t1", "eta1", "MAT_ETA1", "epsr1"], default=1.0e30)
    else:
        try:
            eps_t1 = float(eps_t1)
        except (TypeError, ValueError):
            eps_t1 = 1.0e30

    eps_t2 = getattr(mat, "eps_t2", None)
    if eps_t2 is None:
        eps_t2 = getattr(mat, "eta2", None)
    if eps_t2 is None:
        eps_t2 = _extract(["eps_t2", "eta2", "MAT_ETA2", "epsr2"], default=2.0e30)
    else:
        try:
            eps_t2 = float(eps_t2)
        except (TypeError, ValueError):
            eps_t2 = 2.0e30

    eps_max = getattr(mat, "eps_max", None)
    if eps_max is None:
        eps_max = getattr(mat, "epsm", None)
    if eps_max is None:
        eps_max = _extract(["eps_max", "epsm", "MAT_EPS"], default=1.0e30)
    else:
        try:
            eps_max = float(eps_max)
        except (TypeError, ValueError):
            eps_max = 1.0e30

    if eps_t1 < 0.0:
        log.error(f"/MAT/LAW48/{mid}: tensile failure strain 1 (eps_t1) must be >= 0 (got {eps_t1:g})", "MAT CHECK")
    if eps_t2 < 0.0:
        log.error(f"/MAT/LAW48/{mid}: tensile failure strain 2 (eps_t2) must be >= 0 (got {eps_t2:g})", "MAT CHECK")
    if eps_max < 0.0:
        log.error(f"/MAT/LAW48/{mid}: failure plastic strain (eps_max) must be >= 0 (got {eps_max:g})", "MAT CHECK")

    if 0.0 < eps_t1 < 1.0e20 and eps_t2 <= eps_t1:
        log.error(
            f"/MAT/LAW48/{mid}: tensile failure strains must satisfy eps_t2 > eps_t1 when eps_t1 < 1e20 (got eps_t1={eps_t1:g}, eps_t2={eps_t2:g}) (ANCMSG 420)",
            "MAT CHECK",
        )

    # 9. Strain rate filtering warning (ANCMSG 1220)
    c_val = getattr(mat, "c", None)
    if c_val is None:
        c_val = _extract(["c", "cc", "MAT_C"], default=0.0)
    else:
        try:
            c_val = float(c_val)
        except (TypeError, ValueError):
            c_val = 0.0

    eps0 = getattr(mat, "eps_rate_0", None)
    if eps0 is None:
        eps0 = getattr(mat, "eps0", None)
    if eps0 is None:
        eps0 = _extract(["eps_rate_0", "eps0", "MAT_E0"], default=0.0)
    else:
        try:
            eps0 = float(eps0)
        except (TypeError, ValueError):
            eps0 = 0.0

    fcut = getattr(mat, "fcut", None)
    if fcut is None:
        fcut = getattr(mat, "scale", None)
    if fcut is None:
        fcut = _extract(["fcut", "scale", "SCALE"], default=1.0e30)
    else:
        try:
            fcut = float(fcut)
        except (TypeError, ValueError):
            fcut = 1.0e30

    if c_val > 0.0 and eps0 > 0.0 and fcut == 0.0:
        log.warning(
            f"/MAT/LAW48/{mid}: cutoff frequency for strain rate filtering is zero while strain rate dependency is active (ANCMSG 1220)",
            "MAT CHECK",
        )

    # 10. Reject trusses, beams, springs if attached to element groups
    if model is not None and hasattr(model, "element_groups"):
        try:
            grps = model.element_groups()
            if callable(grps):
                grps = grps()
        except TypeError:
            grps = []
        for item in grps:
            if isinstance(item, tuple) and len(item) == 2:
                name, el_group = item
            else:
                continue
            if name in ("trusses", "beams", "springs"):
                mids_in_group = set()
                if hasattr(el_group, "values") and callable(el_group.values):
                    for el in el_group.values():
                        el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                        if el_mid is not None:
                            mids_in_group.add(el_mid)
                if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                    for _, m, _ in el_group.state["slices"]:
                        m_id = getattr(m, "id", None)
                        if m_id is not None:
                            mids_in_group.add(m_id)
                if mid in mids_in_group:
                    log.error(
                        f"/MAT/LAW48/{mid} (/MAT/ZHAO) is not supported for {name} elements "
                        f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads)",
                        "MAT CHECK",
                    )


_check_mat_law48 = check_mat_law48


def check_mat_law58(
    mat: Any = None,
    arg1: Any = None,
    arg2: Any = None,
    arg3: Any = None,
    model: Any = None,
    log: Any = None,
    mat_id: Any = None,
    **kwargs,
) -> None:
    """Validate /MAT/LAW58 (/MAT/FABR_A, /MAT/FABRIC_A) parameter bounds (M553).

    Required checks (citing hm_read_mat58.F):
      - rho0 > 0
      - e1 > 0
      - e2 > 0
      - Unloading curve consistency:
        - If fun_a4 != 0 or fun_a5 != 0 or fun_a6 != 0:
          - If fun_a1 == 0: error ANCMSG 1578
          - If fun_a2 == 0: error ANCMSG 1579
          - If fun_a3 == 0: error ANCMSG 1580
      - Element compatibility check:
        - Solids (bricks, tetras, penta6, pyra5): reject ANCMSG 305
        - 1D elements (trusses, beams, springs): reject ANCMSG 306
    """
    actual_mat = mat
    actual_model = model
    actual_log = log
    actual_mid = mat_id

    if hasattr(mat, "element_groups") or hasattr(mat, "materials") or hasattr(mat, "mat_law58s"):
        actual_model = mat
        actual_mid = arg1
        actual_mat = arg2
        actual_log = arg3
    elif isinstance(arg1, MessageLog):
        actual_log = arg1
        if arg2 is not None and not isinstance(arg2, MessageLog):
            actual_model = arg2
    elif isinstance(arg2, MessageLog):
        actual_log = arg2
        if arg1 is not None:
            actual_model = arg1
    elif isinstance(arg3, MessageLog):
        actual_log = arg3

    if actual_log is None:
        actual_log = MessageLog()
    log = actual_log
    model = actual_model

    if actual_mat is None and model is not None and actual_mid is not None:
        actual_mat = getattr(model, "materials", {}).get(actual_mid) or getattr(model, "mat_law58s", {}).get(actual_mid)

    if actual_mid is None and actual_mat is not None:
        actual_mid = getattr(actual_mat, "id", 0)

    params = getattr(actual_mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(actual_mat, k):
                val = getattr(actual_mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    def _extract_int(keys: list[str], default: int = 0) -> int:
        for k in keys:
            if hasattr(actual_mat, k):
                val = getattr(actual_mat, k)
                if val is not None:
                    try:
                        return int(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return int(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(actual_mat, "rho0", None)
    if rho0 is None:
        rho0 = getattr(actual_mat, "rho", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "refer_rho", "MAT_RHO", "RHO", "RHO0"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW58/{actual_mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Moduli e1 > 0, e2 > 0
    e1 = getattr(actual_mat, "e1", None)
    if e1 is None:
        e1 = _extract(["e1", "E1", "MAT_E1", "young1"], default=0.0)
    else:
        try:
            e1 = float(e1)
        except (TypeError, ValueError):
            e1 = 0.0

    if e1 <= 0.0:
        log.error(f"/MAT/LAW58/{actual_mid}: Young's modulus E1 must be > 0 (got {e1:g})", "MAT CHECK")

    e2 = getattr(actual_mat, "e2", None)
    if e2 is None:
        e2 = _extract(["e2", "E2", "MAT_E2", "young2"], default=0.0)
    else:
        try:
            e2 = float(e2)
        except (TypeError, ValueError):
            e2 = 0.0

    if e2 <= 0.0:
        log.error(f"/MAT/LAW58/{actual_mid}: Young's modulus E2 must be > 0 (got {e2:g})", "MAT CHECK")

    # 3. Yarn counts N1 > 0, N2 > 0
    n1 = _extract_int(["n1_warp", "n1", "N1", "N1_warp", "fiber_density_1"], 1)
    n2 = _extract_int(["n2_weft", "n2", "N2", "N2_weft", "fiber_density_2"], 1)
    if n1 <= 0:
        log.error(f"/MAT/LAW58/{actual_mid}: yarn count N1 must be > 0 (got {n1})", "MAT CHECK")
    if n2 <= 0:
        log.error(f"/MAT/LAW58/{actual_mid}: yarn count N2 must be > 0 (got {n2})", "MAT CHECK")

    # 4. Unloading curves consistency (ANCMSG 1578, 1579, 1580)
    fun_a1 = _extract_int(["fun_a1", "fun_id1", "fct_id1", "FUN_A1", "funct_id1"], 0)
    fun_a2 = _extract_int(["fun_a2", "fun_id2", "fct_id2", "FUN_A2", "funct_id2"], 0)
    fun_a3 = _extract_int(["fun_a3", "fun_id3", "fct_id3", "FUN_A3", "funct_id3"], 0)
    fun_a4 = _extract_int(["fun_a4", "fun_id4", "fct_id4", "FUN_A4", "funct_id4"], 0)
    fun_a5 = _extract_int(["fun_a5", "fun_id5", "fct_id5", "FUN_A5", "funct_id5"], 0)
    fun_a6 = _extract_int(["fun_a6", "fun_id6", "fct_id6", "FUN_A6", "funct_id6"], 0)

    if fun_a4 != 0 or fun_a5 != 0 or fun_a6 != 0:
        if fun_a1 == 0:
            log.error(
                f"/MAT/LAW58/{actual_mid}: unloading curve defined but warp loading curve FUN_A1 is missing (ANCMSG 1578)",
                "MAT CHECK",
            )
        if fun_a2 == 0:
            log.error(
                f"/MAT/LAW58/{actual_mid}: unloading curve defined but weft loading curve FUN_A2 is missing (ANCMSG 1579)",
                "MAT CHECK",
            )
        if fun_a3 == 0:
            log.error(
                f"/MAT/LAW58/{actual_mid}: unloading curve defined but shear loading curve FUN_A3 is missing (ANCMSG 1580)",
                "MAT CHECK",
            )

    # 5. Incompatible element check (ANCMSG 305 for solids, ANCMSG 306 for 1D elements)
    if model is not None and hasattr(model, "element_groups"):
        try:
            grps = model.element_groups()
            if callable(grps):
                grps = grps()
        except TypeError:
            grps = []
        for item in grps:
            if isinstance(item, tuple) and len(item) == 2:
                name, el_group = item
            else:
                continue
            mids_in_group = set()
            if hasattr(el_group, "values") and callable(el_group.values):
                for el in el_group.values():
                    el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                    if el_mid is not None:
                        mids_in_group.add(el_mid)
            if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                for _, m, _ in el_group.state["slices"]:
                    m_id = getattr(m, "id", None)
                    if m_id is not None:
                        mids_in_group.add(m_id)
            if actual_mid in mids_in_group:
                if name in ("bricks", "bricks_heph", "bric20s", "tetras", "tetra10s", "penta6", "penta6s", "pyra5"):
                    log.error(
                        f"/MAT/LAW58/{actual_mid} (/MAT/FABR_A) is not supported for solid elements ({name}) (ANCMSG 305)",
                        "MAT CHECK",
                    )
                elif name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW58/{actual_mid} (/MAT/FABR_A) is not supported for 1D elements ({name}) (ANCMSG 306)",
                        "MAT CHECK",
                    )


_check_mat_law58 = check_mat_law58


def check_mat_law52(
    model: Any = None,
    mat_id: Any = None,
    mat: Any = None,
    log: Any = None,
    **kwargs: Any,
) -> None:
    """Validate /MAT/LAW52 (/MAT/GURSON, /MAT/PLAS_GURS) parameter bounds (M554).

    Required checks (citing hm_read_mat52.F and matl52_gurson.cfg):
      - rho > 0
      - E > 0
      - 0.0 <= nu < 0.5
      - Error ANCMSG 1745 if f_F < f_c or f_F < f_i or f_c < f_i
      - Warning ANCMSG 1220 if C > 0, P > 0, and Fcut == 0
      - Compatible elements: Solids and Shells supported; reject 1D elements (trusses, beams, springs).
    """
    actual_log = log
    actual_model = model
    actual_mat = mat
    actual_mid = mat_id

    candidates = [c for c in (model, mat_id, mat, log) if c is not None]
    if hasattr(model, "params") and not isinstance(model, Model):
        actual_mat = model
        actual_model = None
        actual_log = mat_id if isinstance(mat_id, MessageLog) else log
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    elif hasattr(mat_id, "params") and isinstance(mat, MessageLog):
        actual_mat = mat_id
        actual_log = mat
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    else:
        for c in candidates:
            if isinstance(c, MessageLog):
                actual_log = c
            elif isinstance(c, Model):
                actual_model = c
            elif hasattr(c, "params") or hasattr(c, "rho") or hasattr(c, "rho0"):
                actual_mat = c

    if actual_mat is None:
        if "mat" in kwargs:
            actual_mat = kwargs["mat"]
        elif "m" in kwargs:
            actual_mat = kwargs["m"]
        elif "mat_law52" in kwargs:
            actual_mat = kwargs["mat_law52"]

    mid_kw = kwargs.get("mat_id", kwargs.get("mid", actual_mid))
    if actual_mat is None and actual_model is not None and mid_kw is not None:
        actual_mat = actual_model.materials.get(mid_kw) or getattr(actual_model, "mat_law52s", {}).get(mid_kw)

    if actual_log is None:
        actual_log = MessageLog()
    log = actual_log
    model = actual_model
    mat = actual_mat

    if mat is None:
        return

    mid = getattr(mat, "id", mid_kw if mid_kw is not None else 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if k in kwargs:
                val = kwargs[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho > 0
    rho = getattr(mat, "rho", None)
    if rho is None:
        rho = getattr(mat, "rho0", None)
    if rho is None:
        rho = _extract(["rho", "rho0", "MAT_RHO", "RHO", "RHO0", "rho_i"], default=0.0)
    else:
        try:
            rho = float(rho)
        except (TypeError, ValueError):
            rho = 0.0

    if rho <= 0.0:
        log.error(f"/MAT/LAW52/{mid}: initial density RHO must be > 0 (got {rho:g})", "MAT CHECK")

    # 2. Young's modulus E > 0
    e = getattr(mat, "e", None)
    if e is None:
        e = getattr(mat, "E", None)
    if e is None:
        e = _extract(["e", "E", "MAT_E", "young"], default=0.0)
    else:
        try:
            e = float(e)
        except (TypeError, ValueError):
            e = 0.0

    if e <= 0.0:
        log.error(f"/MAT/LAW52/{mid}: Young's modulus E must be > 0 (got {e:g})", "MAT CHECK")

    # 3. Poisson's ratio: 0.0 <= nu < 0.5
    nu = getattr(mat, "nu", None)
    if nu is None:
        nu = _extract(["nu", "MAT_NU", "NU", "poisson"], default=0.0)
    else:
        try:
            nu = float(nu)
        except (TypeError, ValueError):
            nu = 0.0

    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW52/{mid}: Poisson's ratio NU must satisfy 0 <= NU < 0.5 (got {nu:g})", "MAT CHECK")

    # 4. Void volume fractions compatibility: f_I <= f_C <= f_F (ANCMSG 1745)
    f_i = _extract(["f_i", "MAT_f_I", "fi", "F1", "f_I", "f0", "f_0"], default=0.0)
    f_c = _extract(["f_c", "MAT_f_C", "fc", "fMAT_C", "f_C"], default=0.0)
    f_f = _extract(["f_f", "MAT_f_F", "ff", "fF", "f_F"], default=0.0)

    if f_f < f_c or f_f < f_i or f_c < f_i:
        log.error(
            f"/MAT/LAW52/{mid}: void volume fractions must satisfy f_I <= f_C <= f_F (got f_I={f_i:g}, f_C={f_c:g}, f_F={f_f:g}) (ANCMSG 1745)",
            "MAT CHECK",
        )

    # 5. Warning ANCMSG 1220 if C > 0, P > 0, Fcut == 0
    c_val = getattr(mat, "raw_c", None)
    if c_val is None:
        c_val = getattr(mat, "c", None)
    if c_val is None:
        c_val = _extract(["raw_c", "c", "MAT_C", "C"], default=0.0)
    else:
        try:
            c_val = float(c_val)
        except (TypeError, ValueError):
            c_val = 0.0

    pc_val = getattr(mat, "raw_pc", None)
    if pc_val is None:
        pc_val = getattr(mat, "pc", None)
    if pc_val is None:
        pc_val = _extract(["raw_pc", "pc", "MAT_PC", "p", "P"], default=0.0)
    else:
        try:
            pc_val = float(pc_val)
        except (TypeError, ValueError):
            pc_val = 0.0

    fcut_val = getattr(mat, "raw_fcut", None)
    if fcut_val is None:
        fcut_val = getattr(mat, "fcut", None)
    if fcut_val is None:
        fcut_val = _extract(["raw_fcut", "fcut", "Fcut", "f_cut"], default=0.0)
    else:
        try:
            fcut_val = float(fcut_val)
        except (TypeError, ValueError):
            fcut_val = 0.0

    if c_val >= 1e29 and hasattr(mat, "raw_c"):
        c_val = getattr(mat, "raw_c", 0.0)
    if c_val > 0.0 and c_val < 1e29 and pc_val > 0.0 and (fcut_val == 0.0 or fcut_val >= 1e29 or getattr(mat, "raw_fcut", -1.0) == 0.0):
        log.warning(
            f"/MAT/LAW52/{mid}: strain rate filtering is recommended when C > 0 and P > 0 with Fcut == 0 (ANCMSG 1220)",
            "MAT CHECK",
        )

    # 6. Compatible elements check (ANCMSG 306 for 1D elements)
    actual_mid = mid
    if model is not None and hasattr(model, "element_groups"):
        try:
            grps = model.element_groups()
            if callable(grps):
                grps = grps()
        except TypeError:
            grps = []
        for item in grps:
            if isinstance(item, tuple) and len(item) == 2:
                name, el_group = item
            else:
                continue
            mids_in_group = set()
            if hasattr(el_group, "values") and callable(el_group.values):
                for el in el_group.values():
                    el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                    if el_mid is not None:
                        mids_in_group.add(el_mid)
            if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                for _, m, _ in el_group.state["slices"]:
                    m_id = getattr(m, "id", None)
                    if m_id is not None:
                        mids_in_group.add(m_id)
            if actual_mid in mids_in_group:
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW52/{actual_mid} (/MAT/GURSON) is not supported for 1D elements ({name}) (ANCMSG 306)",
                        "MAT CHECK",
                    )


_check_mat_law52 = check_mat_law52


def check_mat_law57(
    model: Any = None,
    mat_id: Any = None,
    mat: Any = None,
    log: Any = None,
    **kwargs: Any,
) -> None:
    """Validate /MAT/LAW57 (/MAT/BARLAT3) parameter bounds (M555).

    Required checks (citing hm_read_mat57.F90 and matl57_BARLAT3.cfg):
      - Error if rho <= 0
      - Error if E <= 0
      - Error if nu < 0 or nu >= 0.5
      - Error if R00 <= 0, R45 <= 0, or R90 <= 0
      - Error if m < 1.0
      - Error if eps_t2 <= eps_t1 (when eps_t1 < 1.0e20)
      - Compatible elements: shells only; reject solid elements (ANCMSG 305) and 1D elements (ANCMSG 306).
    """
    actual_log = log
    actual_model = model
    actual_mat = mat
    actual_mid = mat_id

    candidates = [c for c in (model, mat_id, mat, log) if c is not None]
    if hasattr(model, "params") and not isinstance(model, Model):
        actual_mat = model
        actual_model = None
        actual_log = mat_id if isinstance(mat_id, MessageLog) else log
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    elif hasattr(mat_id, "params") and isinstance(mat, MessageLog):
        actual_mat = mat_id
        actual_log = mat
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    else:
        for c in candidates:
            if isinstance(c, MessageLog):
                actual_log = c
            elif isinstance(c, Model):
                actual_model = c
            elif hasattr(c, "params") or hasattr(c, "rho") or hasattr(c, "rho0"):
                actual_mat = c

    if actual_mat is None:
        if "mat" in kwargs:
            actual_mat = kwargs["mat"]
        elif "m" in kwargs:
            actual_mat = kwargs["m"]
        elif "mat_law57" in kwargs:
            actual_mat = kwargs["mat_law57"]

    mid_kw = kwargs.get("mat_id", kwargs.get("mid", actual_mid))
    if actual_mat is None and actual_model is not None and mid_kw is not None:
        actual_mat = actual_model.materials.get(mid_kw) or getattr(actual_model, "mat_law57s", {}).get(mid_kw)

    if actual_log is None:
        actual_log = MessageLog()
    log = actual_log
    model = actual_model
    mat = actual_mat

    if mat is None:
        return

    mid = getattr(mat, "id", mid_kw if mid_kw is not None else 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if k in kwargs:
                val = kwargs[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho > 0
    rho = getattr(mat, "rho", None)
    if rho is None:
        rho = getattr(mat, "rho0", None)
    if rho is None:
        rho = _extract(["rho", "rho0", "MAT_RHO", "RHO", "RHO0", "rho_i"], default=0.0)
    else:
        try:
            rho = float(rho)
        except (TypeError, ValueError):
            rho = 0.0

    if rho <= 0.0:
        log.error(f"/MAT/LAW57/{mid}: initial density RHO must be > 0 (got {rho:g})", "MAT CHECK")

    # 2. Young's modulus E > 0
    e = getattr(mat, "e", None)
    if e is None:
        e = getattr(mat, "E", None)
    if e is None:
        e = _extract(["e", "E", "MAT_E", "young"], default=0.0)
    else:
        try:
            e = float(e)
        except (TypeError, ValueError):
            e = 0.0

    if e <= 0.0:
        log.error(f"/MAT/LAW57/{mid}: Young's modulus E must be > 0 (got {e:g})", "MAT CHECK")

    # 3. Poisson's ratio: 0.0 <= nu < 0.5
    nu = getattr(mat, "nu", None)
    if nu is None:
        nu = _extract(["nu", "MAT_NU", "NU", "poisson"], default=0.0)
    else:
        try:
            nu = float(nu)
        except (TypeError, ValueError):
            nu = 0.0

    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW57/{mid}: Poisson's ratio NU must satisfy 0 <= NU < 0.5 (got {nu:g})", "MAT CHECK")

    # 4. Lankford parameters: R00 > 0, R45 > 0, R90 > 0
    r00 = _extract(["r00", "MAT_R00", "r0", "R00", "R0"], default=1.0)
    r45 = _extract(["r45", "MAT_R45", "R45"], default=1.0)
    r90 = _extract(["r90", "MAT_R90", "R90"], default=1.0)

    if r00 <= 0.0:
        log.error(f"/MAT/LAW57/{mid}: Lankford parameter R00 must be > 0 (got {r00:g})", "MAT CHECK")
    if r45 <= 0.0:
        log.error(f"/MAT/LAW57/{mid}: Lankford parameter R45 must be > 0 (got {r45:g})", "MAT CHECK")
    if r90 <= 0.0:
        log.error(f"/MAT/LAW57/{mid}: Lankford parameter R90 must be > 0 (got {r90:g})", "MAT CHECK")

    # 5. Yield exponent: m >= 1.0
    m_val = _extract(["m", "MAT_M", "M", "barlat_m"], default=6.0)
    if m_val < 1.0:
        log.error(f"/MAT/LAW57/{mid}: Barlat exponent m must be >= 1.0 (got {m_val:g})", "MAT CHECK")

    # 6. Tensile failure strains: eps_t2 > eps_t1 (when eps_t1 < 1.0e20)
    eps_t1 = _extract(["eps_t1", "MAT_EPST1", "EPST1", "epsr1"], default=1.0e30)
    eps_t2 = _extract(["eps_t2", "MAT_EPST2", "EPST2", "epsr2"], default=2.0e30)

    if eps_t1 < 1.0e20 and eps_t2 <= eps_t1:
        log.error(
            f"/MAT/LAW57/{mid}: tensile failure strain EPS_t2 must be > EPS_t1 (got EPS_t1={eps_t1:g}, EPS_t2={eps_t2:g})",
            "MAT CHECK",
        )

    # 7. Compatible elements check (ANCMSG 305 for solids, ANCMSG 306 for 1D elements)
    actual_mid = mid
    if model is not None and hasattr(model, "element_groups"):
        try:
            grps = model.element_groups()
            if callable(grps):
                grps = grps()
        except TypeError:
            grps = []
        for item in grps:
            if isinstance(item, tuple) and len(item) == 2:
                name, el_group = item
            else:
                continue
            mids_in_group = set()
            if hasattr(el_group, "values") and callable(el_group.values):
                for el in el_group.values():
                    el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                    if el_mid is not None:
                        mids_in_group.add(el_mid)
            if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                for _, m_part, _ in el_group.state["slices"]:
                    m_id = getattr(m_part, "id", None)
                    if m_id is not None:
                        mids_in_group.add(m_id)
            if actual_mid in mids_in_group:
                if name in ("bricks", "bricks_heph", "bric20s", "tetras", "tetra10s", "penta6", "penta6s", "pyra5"):
                    log.error(
                        f"/MAT/LAW57/{actual_mid} (/MAT/BARLAT3) is not supported for solid elements ({name}) (ANCMSG 305)",
                        "MAT CHECK",
                    )
                elif name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW57/{actual_mid} (/MAT/BARLAT3) is not supported for 1D elements ({name}) (ANCMSG 306)",
                        "MAT CHECK",
                    )


_check_mat_law57 = check_mat_law57


def check_mat_law21(
    model: Any = None,
    mat_id: Any = None,
    mat: Any = None,
    log: Any = None,
    **kwargs: Any,
) -> None:
    """Validate /MAT/LAW21 (/MAT/DPRAG) parameter bounds (M556).

    Required checks (citing hm_read_mat21.F and matl21_dprag.cfg):
      - Error if rho <= 0
      - Error if E <= 0
      - Error if nu < 0 or nu >= 0.5
      - Error if C1 <= 0 (ANCMSG 829)
      - Warning if A1 < 0 and A2 == 0: 'INVERTED YIELD SURFACE. CHECK A1 SIGN.' (ANCMSG 829)
      - Warning if A2 < 0: 'UNTYPICAL YIELD SURFACE. CHECK A2 SIGN.' (ANCMSG 829)
      - Warning if A2 != 0 and A1^2 - 4*A0*A2 < 0: 'YIELD SURFACE HAS NO ROOT.' (ANCMSG 829)
      - Compatible elements: Solids (bricks, tetras, penta6, pyra5) and SPH;
        reject shells (ANCMSG 305), 1D elements (ANCMSG 306), and 2D analysis (N2D > 0, ANCMSG 305).
    """
    actual_log = log
    actual_model = model
    actual_mat = mat
    actual_mid = mat_id

    candidates = [c for c in (model, mat_id, mat, log) if c is not None]
    if hasattr(model, "params") and not isinstance(model, Model):
        actual_mat = model
        actual_model = None
        actual_log = mat_id if isinstance(mat_id, MessageLog) else log
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    elif hasattr(mat_id, "params") and isinstance(mat, MessageLog):
        actual_mat = mat_id
        actual_log = mat
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    else:
        for c in candidates:
            if isinstance(c, MessageLog):
                actual_log = c
            elif isinstance(c, Model):
                actual_model = c
            elif hasattr(c, "params") or hasattr(c, "rho") or hasattr(c, "rho0"):
                actual_mat = c

    if actual_mat is None:
        if "mat" in kwargs:
            actual_mat = kwargs["mat"]
        elif "m" in kwargs:
            actual_mat = kwargs["m"]
        elif "mat_law21" in kwargs:
            actual_mat = kwargs["mat_law21"]
        elif "mat_dprag" in kwargs:
            actual_mat = kwargs["mat_dprag"]

    mid_kw = kwargs.get("mat_id", kwargs.get("mid", actual_mid))
    if actual_mat is None and actual_model is not None and mid_kw is not None:
        actual_mat = actual_model.materials.get(mid_kw) or getattr(actual_model, "mat_law21s", {}).get(mid_kw) or getattr(actual_model, "mat_dprags", {}).get(mid_kw)

    if actual_log is None:
        actual_log = MessageLog()
    log = actual_log
    model = actual_model
    mat = actual_mat

    if mat is None:
        return

    mid = getattr(mat, "id", mid_kw if mid_kw is not None else 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if k in kwargs:
                val = kwargs[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho > 0
    rho = getattr(mat, "rho", None)
    if rho is None:
        rho = getattr(mat, "rho0", None)
    if rho is None:
        rho = _extract(["rho", "rho0", "MAT_RHO", "RHO", "RHO0", "rho_i"], default=0.0)
    else:
        try:
            rho = float(rho)
        except (TypeError, ValueError):
            rho = 0.0

    if rho <= 0.0:
        log.error(f"/MAT/LAW21/{mid}: initial density RHO must be > 0 (got {rho:g})", "MAT CHECK")

    # 2. Young's modulus E > 0
    e = getattr(mat, "e", None)
    if e is None:
        e = getattr(mat, "E", None)
    if e is None:
        e = _extract(["e", "E", "MAT_E", "young"], default=0.0)
    else:
        try:
            e = float(e)
        except (TypeError, ValueError):
            e = 0.0

    if e <= 0.0:
        log.error(f"/MAT/LAW21/{mid}: Young's modulus E must be > 0 (got {e:g})", "MAT CHECK")

    # 3. Poisson's ratio: 0.0 <= nu < 0.5
    nu = getattr(mat, "nu", None)
    if nu is None:
        nu = _extract(["nu", "MAT_NU", "NU", "poisson"], default=0.0)
    else:
        try:
            nu = float(nu)
        except (TypeError, ValueError):
            nu = 0.0

    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW21/{mid}: Poisson's ratio NU must satisfy 0 <= NU < 0.5 (got {nu:g})", "MAT CHECK")

    # 4. Bulk modulus C1 > 0
    c1 = getattr(mat, "c1", None)
    if c1 is None:
        c1 = _extract(["c1", "MAT_BULK", "C1", "bulk"], default=0.0)
    else:
        try:
            c1 = float(c1)
        except (TypeError, ValueError):
            c1 = 0.0

    if c1 <= 0.0:
        log.error(f"/MAT/LAW21/{mid}: tensile bulk modulus C1 must be > 0 (got {c1:g}) (ANCMSG 829)", "MAT CHECK")

    # 5. Yield surface warnings
    a0 = _extract(["a0", "MAT_A0", "A0"], default=0.0)
    a1 = _extract(["a1", "MAT_A1", "A1"], default=0.0)
    a2 = _extract(["a2", "MAT_A2", "A2"], default=0.0)

    if a1 < 0.0 and a2 == 0.0:
        log.warning(f"/MAT/LAW21/{mid}: INVERTED YIELD SURFACE. CHECK A1 SIGN. (ANCMSG 829)", "MAT CHECK")

    if a2 < 0.0:
        log.warning(f"/MAT/LAW21/{mid}: UNTYPICAL YIELD SURFACE. CHECK A2 SIGN. (ANCMSG 829)", "MAT CHECK")

    if a2 != 0.0:
        delta = a1 * a1 - 4.0 * a0 * a2
        if delta < 0.0:
            log.warning(f"/MAT/LAW21/{mid}: YIELD SURFACE HAS NO ROOT. (ANCMSG 829)", "MAT CHECK")

    # 6. 2D analysis check
    if model is not None and getattr(model, "n2d", 0) > 0:
        log.error(f"/MAT/LAW21/{mid}: LAW21 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")

    # 7. Compatible elements check (ANCMSG 305 for shells, ANCMSG 306 for 1D elements)
    actual_mid = mid
    if model is not None and hasattr(model, "element_groups"):
        try:
            grps = model.element_groups()
            if callable(grps):
                grps = grps()
        except TypeError:
            grps = []
        for item in grps:
            if isinstance(item, tuple) and len(item) == 2:
                name, el_group = item
            else:
                continue
            mids_in_group = set()
            if hasattr(el_group, "values") and callable(el_group.values):
                for el in el_group.values():
                    el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                    if el_mid is not None:
                        mids_in_group.add(el_mid)
            if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                for _, m_part, _ in el_group.state["slices"]:
                    m_id = getattr(m_part, "id", None)
                    if m_id is not None:
                        mids_in_group.add(m_id)
            if actual_mid in mids_in_group:
                if name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
                    log.error(
                        f"/MAT/LAW21/{actual_mid} (/MAT/DPRAG) is not supported for shell elements ({name}) (ANCMSG 305)",
                        "MAT CHECK",
                    )
                elif name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW21/{actual_mid} (/MAT/DPRAG) is not supported for 1D elements ({name}) (ANCMSG 306)",
                        "MAT CHECK",
                    )


_check_mat_law21 = check_mat_law21


def check_mat_law49(
    model: Any = None,
    mat_id: Any = None,
    mat: Any = None,
    log: Any = None,
    **kwargs: Any,
) -> None:
    """Validate /MAT/LAW49 (/MAT/STEINB, /MAT/STEINBERG, /MAT/STEINBERG_GUINAN) parameter bounds (M557).

    Required checks (citing hm_read_mat49.F and matl49_steinb.cfg):
      - Error if rho <= 0
      - Error if E <= 0
      - Error if nu < 0 or nu >= 0.5 (ANCMSG 1514)
      - Error if f < 0 (ANCMSG 1513)
      - Compatible elements: Solids (bricks, tetras, penta6, pyra5) and SPH;
        reject shells (ANCMSG 305), 1D elements (ANCMSG 306), and 2D analysis (N2D > 0, ANCMSG 305).
    """
    actual_log = log
    actual_model = model
    actual_mat = mat
    actual_mid = mat_id

    candidates = [c for c in (model, mat_id, mat, log) if c is not None]
    if hasattr(model, "params") and not isinstance(model, Model):
        actual_mat = model
        actual_model = None
        actual_log = mat_id if isinstance(mat_id, MessageLog) else log
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    elif hasattr(mat_id, "params") and isinstance(mat, MessageLog):
        actual_mat = mat_id
        actual_log = mat
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    else:
        for c in candidates:
            if isinstance(c, MessageLog):
                actual_log = c
            elif isinstance(c, Model):
                actual_model = c
            elif hasattr(c, "params") or hasattr(c, "rho") or hasattr(c, "rho0"):
                actual_mat = c

    if actual_mat is None:
        if "mat" in kwargs:
            actual_mat = kwargs["mat"]
        elif "m" in kwargs:
            actual_mat = kwargs["m"]
        elif "mat_law49" in kwargs:
            actual_mat = kwargs["mat_law49"]
        elif "mat_steinb" in kwargs:
            actual_mat = kwargs["mat_steinb"]

    mid_kw = kwargs.get("mat_id", kwargs.get("mid", actual_mid))
    if actual_mat is None and actual_model is not None and mid_kw is not None:
        actual_mat = actual_model.materials.get(mid_kw) or getattr(actual_model, "mat_law49s", {}).get(mid_kw) or getattr(actual_model, "mat_steinbs", {}).get(mid_kw)

    if actual_log is None:
        actual_log = MessageLog()
    log = actual_log
    model = actual_model
    mat = actual_mat

    if mat is None:
        return

    mid = getattr(mat, "id", mid_kw if mid_kw is not None else 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if k in kwargs:
                val = kwargs[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho > 0
    rho = getattr(mat, "rho", None)
    if rho is None:
        rho = getattr(mat, "rho0", None)
    if rho is None:
        rho = _extract(["rho", "rho0", "MAT_RHO", "RHO", "RHO0", "rho_i"], default=0.0)
    else:
        try:
            rho = float(rho)
        except (TypeError, ValueError):
            rho = 0.0

    if rho <= 0.0:
        log.error(f"/MAT/LAW49/{mid}: initial density RHO must be > 0 (got {rho:g})", "MAT CHECK")

    # 2. Young's modulus E > 0
    e = getattr(mat, "e", None)
    if e is None:
        e = getattr(mat, "E", None)
    if e is None:
        e = getattr(mat, "e0", None)
    if e is None:
        e = _extract(["e", "E", "e0", "E0", "MAT_E", "MAT_E0", "young"], default=0.0)
    else:
        try:
            e = float(e)
        except (TypeError, ValueError):
            e = 0.0

    if e <= 0.0:
        log.error(f"/MAT/LAW49/{mid}: Young's modulus E must be > 0 (got {e:g})", "MAT CHECK")

    # 3. Poisson's ratio: 0.0 <= nu < 0.5 (ANCMSG 1514)
    nu = getattr(mat, "nu", None)
    if nu is None:
        nu = _extract(["nu", "Nu", "MAT_NU", "NU", "poisson"], default=0.0)
    else:
        try:
            nu = float(nu)
        except (TypeError, ValueError):
            nu = 0.0

    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW49/{mid}: Poisson's ratio nu must satisfy 0 <= nu < 0.5 (got {nu:g}) (ANCMSG 1514)", "MAT CHECK")

    # 4. F coefficient >= 0 (ANCMSG 1513)
    f_val = getattr(mat, "f", None)
    if f_val is None:
        f_val = _extract(["f", "MAT_F", "CF", "cf"], default=0.0)
    else:
        try:
            f_val = float(f_val)
        except (TypeError, ValueError):
            f_val = 0.0

    if f_val < 0.0:
        log.error(f"/MAT/LAW49/{mid}: F coefficient must be >= 0 (got {f_val:g}) (ANCMSG 1513)", "MAT CHECK")

    # 5. 2D analysis check (ANCMSG 305)
    if model is not None and getattr(model, "n2d", 0) > 0:
        log.error(f"/MAT/LAW49/{mid}: LAW49 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")

    # 6. Compatible elements check (ANCMSG 305 for shells, ANCMSG 306 for 1D elements)
    actual_mid = mid
    if model is not None and hasattr(model, "element_groups"):
        try:
            grps = model.element_groups()
            if callable(grps):
                grps = grps()
        except TypeError:
            grps = []
        for item in grps:
            if isinstance(item, tuple) and len(item) == 2:
                name, el_group = item
            else:
                continue
            mids_in_group = set()
            if hasattr(el_group, "values") and callable(el_group.values):
                for el in el_group.values():
                    el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                    if el_mid is not None:
                        mids_in_group.add(el_mid)
            if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                for _, m_part, _ in el_group.state["slices"]:
                    m_id = getattr(m_part, "id", None)
                    if m_id is not None:
                        mids_in_group.add(m_id)
            if actual_mid in mids_in_group:
                if name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
                    log.error(
                        f"/MAT/LAW49/{actual_mid} (/MAT/STEINB) is not supported for shell elements ({name}) (ANCMSG 305)",
                        "MAT CHECK",
                    )
                elif name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW49/{actual_mid} (/MAT/STEINB) is not supported for 1D elements ({name}) (ANCMSG 306)",
                        "MAT CHECK",
                    )


_check_mat_law49 = check_mat_law49


def check_mat_law79(
    model: Any = None,
    mat_id: Any = None,
    mat: Any = None,
    log: Any = None,
    **kwargs: Any,
) -> None:
    """Validate /MAT/LAW79 (/MAT/JOHN_HOLM, /MAT/JOHNSON_HOLMQUIST, /MAT/JH2) parameter bounds (M558).

    Required checks (citing hm_read_mat79.F and matl79_79.cfg):
      - Error if rho <= 0
      - Error if shear <= 0 (ANCMSG 908)
      - Error if k1 <= 0 (ANCMSG 909)
      - Error if phel > hel (ANCMSG 907)
      - Error if eps0 <= 0 (ANCMSG 910)
      - Error if beta < 0 or beta > 1 (ANCMSG 911)
      - Compatible elements: Solids (bricks, tetras, penta6, pyra5) and SPH;
        reject shells (ANCMSG 305), 1D elements (ANCMSG 306), and 2D analysis (N2D > 0, ANCMSG 305).
    """
    actual_log = log
    actual_model = model
    actual_mat = mat
    actual_mid = mat_id

    candidates = [c for c in (model, mat_id, mat, log) if c is not None]
    if hasattr(model, "params") and not isinstance(model, Model):
        actual_mat = model
        actual_model = None
        actual_log = mat_id if isinstance(mat_id, MessageLog) else log
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    elif hasattr(mat_id, "params") and isinstance(mat, MessageLog):
        actual_mat = mat_id
        actual_log = mat
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    else:
        for c in candidates:
            if isinstance(c, MessageLog):
                actual_log = c
            elif isinstance(c, Model):
                actual_model = c
            elif hasattr(c, "params") or hasattr(c, "rho") or hasattr(c, "rho0"):
                actual_mat = c

    if actual_mat is None:
        if "mat" in kwargs:
            actual_mat = kwargs["mat"]
        elif "m" in kwargs:
            actual_mat = kwargs["m"]
        elif "mat_law79" in kwargs:
            actual_mat = kwargs["mat_law79"]
        elif "mat_john_holm" in kwargs:
            actual_mat = kwargs["mat_john_holm"]
        elif "mat_jh2" in kwargs:
            actual_mat = kwargs["mat_jh2"]

    mid_kw = kwargs.get("mat_id", kwargs.get("mid", actual_mid))
    if actual_mat is None and actual_model is not None and mid_kw is not None:
        actual_mat = actual_model.materials.get(mid_kw) or getattr(actual_model, "mat_law79s", {}).get(mid_kw) or getattr(actual_model, "mat_john_holms", {}).get(mid_kw)

    if actual_log is None:
        actual_log = MessageLog()
    log = actual_log
    model = actual_model
    mat = actual_mat

    if mat is None:
        return

    mid = getattr(mat, "id", mid_kw if mid_kw is not None else 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if k in kwargs:
                val = kwargs[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho > 0
    if "rho" in kwargs and kwargs["rho"] is not None:
        try:
            rho = float(kwargs["rho"])
        except (TypeError, ValueError):
            rho = 0.0
    else:
        rho = getattr(mat, "rho", None)
        if rho is None:
            rho = getattr(mat, "rho0", None)
        if rho is None:
            rho = _extract(["rho", "rho0", "MAT_RHO", "RHO", "RHO0", "rho_i"], default=0.0)
        else:
            try:
                rho = float(rho)
            except (TypeError, ValueError):
                rho = 0.0

    if rho <= 0.0:
        log.error(f"/MAT/LAW79/{mid}: initial density RHO must be > 0 (got {rho:g})", "MAT CHECK")

    # 2. Shear modulus G > 0 (ANCMSG 908)
    if "shear" in kwargs and kwargs["shear"] is not None:
        try:
            shear = float(kwargs["shear"])
        except (TypeError, ValueError):
            shear = 0.0
    elif "tau_shear" in kwargs and kwargs["tau_shear"] is not None:
        try:
            shear = float(kwargs["tau_shear"])
        except (TypeError, ValueError):
            shear = 0.0
    else:
        shear = getattr(mat, "tau_shear", None)
        if shear is None:
            shear = getattr(mat, "shear", None)
        if shear is None:
            shear = getattr(mat, "G", None)
        if shear is None:
            shear = getattr(mat, "g", None)
        if shear is None:
            shear = _extract(["tau_shear", "shear", "G", "g", "MAT_G"], default=0.0)
        else:
            try:
                shear = float(shear)
            except (TypeError, ValueError):
                shear = 0.0

    if shear <= 0.0:
        log.error(f"/MAT/LAW79/{mid}: shear modulus must be > 0 (got {shear:g}) (ANCMSG 908)", "MAT CHECK")

    # 3. Bulk modulus K1 > 0 (ANCMSG 909)
    if "k1" in kwargs and kwargs["k1"] is not None:
        try:
            k1 = float(kwargs["k1"])
        except (TypeError, ValueError):
            k1 = 0.0
    elif "bulk" in kwargs and kwargs["bulk"] is not None:
        try:
            k1 = float(kwargs["bulk"])
        except (TypeError, ValueError):
            k1 = 0.0
    else:
        k1 = getattr(mat, "k1", None)
        if k1 is None:
            k1 = getattr(mat, "K1", None)
        if k1 is None:
            k1 = getattr(mat, "bulk", None)
        if k1 is None:
            k1 = _extract(["k1", "K1", "bulk", "K", "MAT_BULK"], default=0.0)
        else:
            try:
                k1 = float(k1)
            except (TypeError, ValueError):
                k1 = 0.0

    if k1 <= 0.0:
        log.error(f"/MAT/LAW79/{mid}: bulk modulus K1 must be > 0 (got {k1:g}) (ANCMSG 909)", "MAT CHECK")

    # 4. PHEL <= HEL (ANCMSG 907)
    hel = _extract(["hel", "HEL", "MAT_E", "e"], default=0.0)
    phel = _extract(["phel", "PHEL", "MAT_EPS", "eps"], default=0.0)
    if hel > 0.0 and phel > hel:
        log.error(f"/MAT/LAW79/{mid}: pressure at HEL (PHEL={phel:g}) cannot exceed HEL ({hel:g}) (ANCMSG 907)", "MAT CHECK")

    # 5. Reference strain rate EPS0 > 0 (ANCMSG 910)
    if "eps0" in kwargs and kwargs["eps0"] is not None:
        try:
            eps0 = float(kwargs["eps0"])
        except (TypeError, ValueError):
            eps0 = 1.0
    else:
        eps0 = getattr(mat, "eps0", None)
        if eps0 is None:
            eps0 = _extract(["eps0", "EPS0", "MAT_Epsilon_F", "epsilon_f"], default=1.0)
        else:
            try:
                eps0 = float(eps0)
            except (TypeError, ValueError):
                eps0 = 1.0

    if eps0 <= 0.0:
        log.error(f"/MAT/LAW79/{mid}: reference strain rate EPS0 must be > 0 (got {eps0:g}) (ANCMSG 910)", "MAT CHECK")

    # 6. Bulking coefficient BETA in [0, 1] (ANCMSG 911)
    if "beta" in kwargs and kwargs["beta"] is not None:
        try:
            beta = float(kwargs["beta"])
        except (TypeError, ValueError):
            beta = 1.0
    else:
        beta = getattr(mat, "beta", None)
        if beta is None:
            beta = _extract(["beta", "BETA", "MAT_Beta"], default=1.0)
        else:
            try:
                beta = float(beta)
            except (TypeError, ValueError):
                beta = 1.0

    if beta < 0.0 or beta > 1.0:
        log.error(f"/MAT/LAW79/{mid}: bulking coefficient BETA must satisfy 0 <= BETA <= 1 (got {beta:g}) (ANCMSG 911)", "MAT CHECK")

    # 7. 2D analysis check (ANCMSG 305)
    if model is not None and getattr(model, "n2d", 0) > 0:
        log.error(f"/MAT/LAW79/{mid}: LAW79 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")

    # 8. Compatible elements check (ANCMSG 305 for shells, ANCMSG 306 for 1D elements)
    actual_mid = mid
    if model is not None and hasattr(model, "element_groups"):
        try:
            grps = model.element_groups()
            if callable(grps):
                grps = grps()
        except TypeError:
            grps = []
        for item in grps:
            if isinstance(item, tuple) and len(item) == 2:
                name, el_group = item
            else:
                continue
            mids_in_group = set()
            if hasattr(el_group, "values") and callable(el_group.values):
                for el in el_group.values():
                    el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                    if el_mid is not None:
                        mids_in_group.add(el_mid)
            if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                for _, m_part, _ in el_group.state["slices"]:
                    m_id = getattr(m_part, "id", None)
                    if m_id is not None:
                        mids_in_group.add(m_id)
            if actual_mid in mids_in_group:
                if name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
                    log.error(
                        f"/MAT/LAW79/{actual_mid} (/MAT/JOHN_HOLM) is not supported for shell elements ({name}) (ANCMSG 305)",
                        "MAT CHECK",
                    )
                elif name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW79/{actual_mid} (/MAT/JOHN_HOLM) is not supported for 1D elements ({name}) (ANCMSG 306)",
                        "MAT CHECK",
                    )


_check_mat_law79 = check_mat_law79


def check_mat_law50(
    model: Any = None,
    mat_id: Any = None,
    mat: Any = None,
    log: Any = None,
    **kwargs: Any,
) -> None:
    """Validate /MAT/LAW50 (/MAT/VISC_HONEY /MAT/HYP_FOAM) parameter bounds (M559).

    Required checks (citing hm_read_mat50.F90, sigeps50s.F90, and mat_law50.cfg):
      - Error if RHO <= 0
      - Error if EA, EB, EC, GAB, GBC, or GCA <= 0 (ANCMSG 306)
      - Compaction checks (if active):
          ECOMP > 0
          0 <= NU/PR < 0.5
          0 < VCOMP <= 1.0
      - Reject 2D analysis: N2D > 0 (ANCMSG 305)
      - Reject shell elements (ANCMSG 305)
      - Reject 1D elements (ANCMSG 306)
    """
    actual_log = log
    actual_model = model
    actual_mat = mat
    actual_mid = mat_id

    candidates = [c for c in (model, mat_id, mat, log) if c is not None]
    if hasattr(model, "params") and not isinstance(model, Model):
        actual_mat = model
        actual_model = None
        actual_log = mat_id if isinstance(mat_id, MessageLog) else log
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    elif hasattr(mat_id, "params") and isinstance(mat, MessageLog):
        actual_mat = mat_id
        actual_log = mat
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    else:
        for c in candidates:
            if isinstance(c, MessageLog):
                actual_log = c
            elif isinstance(c, Model):
                actual_model = c
            elif hasattr(c, "params") or hasattr(c, "rho") or hasattr(c, "rho0"):
                actual_mat = c

    if actual_mat is None:
        if "mat" in kwargs:
            actual_mat = kwargs["mat"]
        elif "m" in kwargs:
            actual_mat = kwargs["m"]
        elif "mat_law50" in kwargs:
            actual_mat = kwargs["mat_law50"]

    mid_kw = kwargs.get("mat_id", kwargs.get("mid", actual_mid))
    if actual_mat is None and actual_model is not None and mid_kw is not None:
        actual_mat = actual_model.materials.get(mid_kw) or getattr(actual_model, "mat_law50s", {}).get(mid_kw)

    if actual_log is None:
        actual_log = MessageLog()
    log = actual_log
    model = actual_model
    mat = actual_mat

    if mat is None:
        return

    mid = getattr(mat, "id", mid_kw if mid_kw is not None else 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if k in kwargs:
                val = kwargs[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho > 0
    rho = getattr(mat, "rho", None)
    if rho is None:
        rho = getattr(mat, "rho0", None)
    if rho is None:
        rho = _extract(["rho", "rho0", "MAT_RHO", "RHO", "RHO0", "rho_i"], default=0.0)
    else:
        try:
            rho = float(rho)
        except (TypeError, ValueError):
            rho = 0.0

    if rho <= 0.0:
        log.error(f"/MAT/LAW50/{mid}: initial density RHO must be > 0 (got {rho:g})", "MAT CHECK")

    # 2. Moduli EA, EB, EC, GAB, GBC, GCA > 0 (ANCMSG 306)
    ea = _extract(["ea", "MAT_EA", "e11", "EA", "E11"], default=0.0)
    eb = _extract(["eb", "MAT_EB", "e22", "EB", "E22"], default=0.0)
    ec = _extract(["ec", "MAT_EC", "e33", "EC", "E33"], default=0.0)
    gab = _extract(["gab", "MAT_GAB", "g12", "GAB", "G12"], default=0.0)
    gbc = _extract(["gbc", "MAT_GBC", "g23", "GBC", "G23"], default=0.0)
    gca = _extract(["gca", "MAT_GCA", "g31", "GCA", "G31"], default=0.0)

    for mod_name, mod_val in (("EA", ea), ("EB", eb), ("EC", ec), ("GAB", gab), ("GBC", gbc), ("GCA", gca)):
        if mod_val <= 0.0:
            log.error(f"/MAT/LAW50/{mid}: elastic modulus {mod_name} must be > 0 (got {mod_val:g}) (ANCMSG 306)", "MAT CHECK")

    # 3. Compaction parameters check
    ecomp = _extract(["ecomp", "MAT_ECOMP", "ECOMP"], default=0.0)
    pr = _extract(["pr", "nu", "MAT_PR", "PR", "NU"], default=0.0)
    sigy = _extract(["sigy", "MAT_SIGY", "SIGY"], default=0.0)
    vcomp = _extract(["vcomp", "MAT_VCOMP", "VCOMP"], default=0.0)

    # Compacted state coupling is active if ecomp*sigy*vcomp > 0 or any compaction param set
    compaction_active = (ecomp * sigy * vcomp > 0.0) or (ecomp > 0.0 or sigy > 0.0 or vcomp > 0.0)
    if compaction_active:
        if ecomp <= 0.0:
            log.error(f"/MAT/LAW50/{mid}: compacted Young's modulus ECOMP must be > 0 (got {ecomp:g})", "MAT CHECK")
        if pr < 0.0 or pr >= 0.5:
            log.error(f"/MAT/LAW50/{mid}: compacted Poisson's ratio NU/PR must satisfy 0 <= NU < 0.5 (got {pr:g})", "MAT CHECK")
        if vcomp <= 0.0 or vcomp > 1.0:
            log.error(f"/MAT/LAW50/{mid}: compaction volume fraction VCOMP must satisfy 0 < VCOMP <= 1 (got {vcomp:g})", "MAT CHECK")

    # 4. Compatible elements check
    actual_mid = mid
    if model is not None:
        if getattr(model, "n2d", 0) > 0:
            log.error(
                f"/MAT/LAW50/{actual_mid}: LAW50 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)",
                "MAT CHECK",
            )
        if hasattr(model, "element_groups"):
            try:
                grps = model.element_groups()
                if callable(grps):
                    grps = grps()
            except TypeError:
                grps = []
            for item in grps:
                if isinstance(item, tuple) and len(item) == 2:
                    name, el_group = item
                else:
                    continue
                mids_in_group = set()
                if hasattr(el_group, "values") and callable(el_group.values):
                    for el in el_group.values():
                        el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                        if el_mid is not None:
                            mids_in_group.add(el_mid)
                if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                    for _, m_part, _ in el_group.state["slices"]:
                        m_id = getattr(m_part, "id", None)
                        if m_id is not None:
                            mids_in_group.add(m_id)
                if actual_mid in mids_in_group:
                    if name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
                        log.error(
                            f"/MAT/LAW50/{actual_mid} (/MAT/VISC_HONEY) is not supported for shell elements ({name}) (ANCMSG 305)",
                            "MAT CHECK",
                        )
                    elif name in ("trusses", "beams", "springs"):
                        log.error(
                            f"/MAT/LAW50/{actual_mid} (/MAT/VISC_HONEY) is not supported for 1D elements ({name}) (ANCMSG 306)",
                            "MAT CHECK",
                        )


_check_mat_law50 = check_mat_law50


def check_mat_law163(
    model: Any = None,
    mat_id: Any = None,
    mat: Any = None,
    log: Any = None,
    **kwargs: Any,
) -> None:
    """Validate /MAT/LAW163 (/MAT/CRUSHABLE_FOAM /MAT/CRUSH_FOAM) parameter bounds (M560).

    Required checks (citing hm_read_mat163.F90, law163_upd.F90, and matl163_crushable_foam.cfg):
      - Error if RHO <= 0
      - Error if E <= 0
      - Error if NU < 0 or NU >= 0.5 (ANCMSG 1514)
      - Error if DAMP < 0
      - Reject 2D analysis: N2D > 0 (ANCMSG 305)
      - Reject shell elements (ANCMSG 305)
      - Reject 1D elements (ANCMSG 306)
    """
    actual_log = log
    actual_model = model
    actual_mat = mat
    actual_mid = mat_id

    candidates = [c for c in (model, mat_id, mat, log) if c is not None]
    if hasattr(model, "params") and not isinstance(model, Model):
        actual_mat = model
        actual_model = None
        actual_log = mat_id if isinstance(mat_id, MessageLog) else log
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    elif hasattr(mat_id, "params") and isinstance(mat, MessageLog):
        actual_mat = mat_id
        actual_log = mat
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    else:
        for c in candidates:
            if isinstance(c, MessageLog):
                actual_log = c
            elif isinstance(c, Model):
                actual_model = c
            elif hasattr(c, "params") or hasattr(c, "rho") or hasattr(c, "rho0"):
                actual_mat = c

    if actual_mat is None:
        if "mat" in kwargs:
            actual_mat = kwargs["mat"]
        elif "m" in kwargs:
            actual_mat = kwargs["m"]
        elif "mat_law163" in kwargs:
            actual_mat = kwargs["mat_law163"]
        elif "mat_crushable_foam" in kwargs:
            actual_mat = kwargs["mat_crushable_foam"]
        elif "mat_crush_foam" in kwargs:
            actual_mat = kwargs["mat_crush_foam"]

    mid_kw = kwargs.get("mat_id", kwargs.get("mid", actual_mid))
    if actual_mat is None and actual_model is not None and mid_kw is not None:
        actual_mat = actual_model.materials.get(mid_kw) or getattr(actual_model, "mat_law163s", {}).get(mid_kw)

    if actual_log is None:
        actual_log = MessageLog()
    log = actual_log
    model = actual_model
    mat = actual_mat

    if mat is None:
        return

    mid = getattr(mat, "id", mid_kw if mid_kw is not None else 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if k in kwargs and kwargs[k] is not None:
                try:
                    return float(kwargs[k])
                except (TypeError, ValueError):
                    pass
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho > 0
    rho = None
    for k in ("rho", "rho0", "MAT_RHO", "RHO", "RHO0", "rho_i"):
        if k in kwargs and kwargs[k] is not None:
            try:
                rho = float(kwargs[k])
                break
            except (TypeError, ValueError):
                pass
    if rho is None:
        if hasattr(mat, "rho") and getattr(mat, "rho") is not None:
            try:
                rho = float(getattr(mat, "rho"))
            except (TypeError, ValueError):
                pass
    if rho is None:
        if hasattr(mat, "rho0") and getattr(mat, "rho0") is not None:
            try:
                rho = float(getattr(mat, "rho0"))
            except (TypeError, ValueError):
                pass
    if rho is None:
        rho = _extract(["rho", "rho0", "MAT_RHO", "RHO", "RHO0", "rho_i"], default=0.0)

    if rho <= 0.0:
        log.error(f"/MAT/LAW163/{mid}: initial density RHO must be > 0 (got {rho:g})", "MAT CHECK")

    # 2. Young's modulus E > 0
    e = _extract(["e", "MAT_E", "E", "young"], default=0.0)
    if e <= 0.0:
        log.error(f"/MAT/LAW163/{mid}: Young's modulus E must be > 0 (got {e:g})", "MAT CHECK")

    # 3. Poisson's ratio: 0.0 <= nu < 0.5 (ANCMSG 1514)
    nu = _extract(["nu", "MAT_NU", "Nu", "pr"], default=0.0)
    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW163/{mid}: Poisson's ratio nu must satisfy 0 <= nu < 0.5 (got {nu:g}) (ANCMSG 1514)", "MAT CHECK")

    # 4. Damping coefficient: DAMP >= 0
    damp = _extract(["damp", "LSD_MAT_DAMP", "DAMP"], default=0.10)
    if damp < 0.0:
        log.error(f"/MAT/LAW163/{mid}: damping coefficient DAMP must be >= 0 (got {damp:g})", "MAT CHECK")

    # 5. Compatible elements check
    actual_mid = mid
    if model is not None:
        if getattr(model, "n2d", 0) > 0:
            log.error(
                f"/MAT/LAW163/{actual_mid}: LAW163 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)",
                "MAT CHECK",
            )
        if hasattr(model, "element_groups"):
            try:
                grps = model.element_groups()
                if callable(grps):
                    grps = grps()
            except TypeError:
                grps = []
            for item in grps:
                if isinstance(item, tuple) and len(item) == 2:
                    name, el_group = item
                else:
                    continue
                mids_in_group = set()
                if hasattr(el_group, "values") and callable(el_group.values):
                    for el in el_group.values():
                        el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                        if el_mid is not None:
                            mids_in_group.add(el_mid)
                if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                    for _, m_part, _ in el_group.state["slices"]:
                        m_id = getattr(m_part, "id", None)
                        if m_id is not None:
                            mids_in_group.add(m_id)
                if actual_mid in mids_in_group:
                    if name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads", "quad4_2d", "tria3_2d", "elements_2d", "plane_strain", "plane_stress"):
                        log.error(
                            f"/MAT/LAW163/{actual_mid} (/MAT/CRUSHABLE_FOAM) is not supported for shell elements ({name}) (ANCMSG 305)",
                            "MAT CHECK",
                        )
                    elif name in ("trusses", "beams", "springs"):
                        log.error(
                            f"/MAT/LAW163/{actual_mid} (/MAT/CRUSHABLE_FOAM) is not supported for 1D elements ({name}) (ANCMSG 306)",
                            "MAT CHECK",
                        )

        if hasattr(model, "parts") and isinstance(model.parts, dict):
            for pid, part in model.parts.items():
                p_mid = getattr(part, "mat_id", getattr(part, "mid", None))
                if p_mid == actual_mid:
                    etype = str(getattr(part, "elem_type", getattr(part, "type", ""))).upper()
                    if "SHELL" in etype or "QUAD" in etype or "TRIA" in etype:
                        log.error(
                            f"/MAT/LAW163/{actual_mid} (/MAT/CRUSHABLE_FOAM) is not supported for shell elements ({etype.lower()}) (ANCMSG 305)",
                            "MAT CHECK",
                        )
                    elif "BEAM" in etype or "TRUSS" in etype or "SPRING" in etype or "1D" in etype:
                        log.error(
                            f"/MAT/LAW163/{actual_mid} (/MAT/CRUSHABLE_FOAM) is not supported for 1D elements ({etype.lower()}) (ANCMSG 306)",
                            "MAT CHECK",
                        )


_check_mat_law163 = check_mat_law163


def check_mat_law73(
    model: Any = None,
    mat_id: Any = None,
    mat: Any = None,
    log: Any = None,
    **kwargs: Any,
) -> None:
    """Validate /MAT/LAW73 (/MAT/BARLAT2000, /MAT/HILL_THERM, /MAT/THERM_HILL) parameter bounds (M561).

    Citing hm_read_mat73.F and radioss140/MAT/matl73_73.cfg:
      - rho0 > 0
      - e > 0
      - 0 <= nu < 0.5 (ANCMSG 1514)
      - r00 > 0, r45 > 0, r90 > 0
      - epsr1 < epsr2: if epsr1 >= epsr2, error (ANCMSG 1044)
      - Compatible elements: shells only; reject solid elements (ANCMSG 305) and 1D elements (ANCMSG 306).
    """
    actual_log = log
    actual_model = model
    actual_mat = mat
    actual_mid = mat_id

    candidates = [c for c in (model, mat_id, mat, log) if c is not None]
    if hasattr(model, "params") and not isinstance(model, Model):
        actual_mat = model
        actual_model = None
        actual_log = mat_id if isinstance(mat_id, MessageLog) else log
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    elif hasattr(mat_id, "params") and isinstance(mat, MessageLog):
        actual_mat = mat_id
        actual_log = mat
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    else:
        for c in candidates:
            if isinstance(c, MessageLog):
                actual_log = c
            elif isinstance(c, Model):
                actual_model = c
            elif hasattr(c, "params") or hasattr(c, "rho") or hasattr(c, "rho0") or hasattr(c, "r00"):
                actual_mat = c
            elif isinstance(c, int) and not isinstance(c, bool):
                actual_mid = c

    if actual_mat is None:
        if "mat" in kwargs:
            actual_mat = kwargs["mat"]
        elif "material" in kwargs:
            actual_mat = kwargs["material"]
        elif "mat73" in kwargs:
            actual_mat = kwargs["mat73"]
        elif "mat_law73" in kwargs:
            actual_mat = kwargs["mat_law73"]
        elif "mat_hill_therm" in kwargs:
            actual_mat = kwargs["mat_hill_therm"]
        elif "mat_therm_hill" in kwargs:
            actual_mat = kwargs["mat_therm_hill"]
        elif "mat_barlat2000" in kwargs:
            actual_mat = kwargs["mat_barlat2000"]

    if actual_log is None:
        if "log" in kwargs:
            actual_log = kwargs["log"]
        elif "logger" in kwargs:
            actual_log = kwargs["logger"]
        else:
            actual_log = MessageLog()

    mid_kw = kwargs.get("mat_id", kwargs.get("mid", actual_mid))
    if actual_mat is None and actual_model is not None:
        if mid_kw is not None:
            actual_mat = actual_model.materials.get(mid_kw) or getattr(actual_model, "mat_law73s", {}).get(mid_kw)
        else:
            for m_id, m_obj in list(getattr(actual_model, "mat_law73s", {}).items()):
                check_mat_law73(model=actual_model, mat_id=m_id, mat=m_obj, log=actual_log)
            for m_id, m_obj in list(getattr(actual_model, "materials", {}).items()):
                if getattr(m_obj, "law", None) in (73, "73", "LAW73", "BARLAT2000", "HILL_THERM", "THERM_HILL") or getattr(m_obj, "law_name", None) in ("73", "LAW73", "BARLAT2000", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_BARLAT2000", "MAT_HILL_THERM", "MAT_THERM_HILL"):
                    if m_id not in getattr(actual_model, "mat_law73s", {}):
                        check_mat_law73(model=actual_model, mat_id=m_id, mat=m_obj, log=actual_log)
            return

    if actual_mat is None:
        return

    mat = actual_mat
    log = actual_log
    model = actual_model
    mid = getattr(mat, "id", mid_kw if mid_kw is not None else 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = _extract(["rho", "rho0", "MAT_RHO", "RHO", "RHO0"], default=0.0)
    if rho0 <= 0.0:
        log.error(f"/MAT/LAW73/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's modulus E > 0
    e = _extract(["e", "E", "MAT_E", "young"], default=0.0)
    if e <= 0.0:
        log.error(f"/MAT/LAW73/{mid}: Young's modulus E must be > 0 (got {e:g})", "MAT CHECK")

    # 3. Poisson's ratio: 0.0 <= nu < 0.5 (ANCMSG 1514)
    nu = _extract(["nu", "Nu", "NU", "MAT_NU"], default=0.0)
    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW73/{mid}: Poisson's ratio nu must satisfy 0 <= nu < 0.5 (got {nu:g}) (ANCMSG 1514)", "MAT CHECK")

    # 4. Lankford parameters: R00 > 0, R45 > 0, R90 > 0
    r00 = _extract(["r00", "R00", "MAT_R00", "r0", "R0"], default=1.0)
    if r00 <= 0.0:
        log.error(f"/MAT/LAW73/{mid}: Lankford parameter R00 must be > 0 (got {r00:g})", "MAT CHECK")
    r45 = _extract(["r45", "R45", "MAT_R45"], default=1.0)
    if r45 <= 0.0:
        log.error(f"/MAT/LAW73/{mid}: Lankford parameter R45 must be > 0 (got {r45:g})", "MAT CHECK")
    r90 = _extract(["r90", "R90", "MAT_R90"], default=1.0)
    if r90 <= 0.0:
        log.error(f"/MAT/LAW73/{mid}: Lankford parameter R90 must be > 0 (got {r90:g})", "MAT CHECK")

    # 5. Failure strains: epsr1 < epsr2 (ANCMSG 1044)
    epsr1 = _extract(["epsr1", "epst1", "MAT_EPST1"], default=1.0e30)
    epsr2 = _extract(["epsr2", "epst2", "MAT_EPST2"], default=2.0e30)
    if epsr1 >= epsr2:
        log.error(
            f"/MAT/LAW73/{mid}: tensile failure strain 1 ({epsr1:g}) must be less than tensile failure strain 2 ({epsr2:g}) (ANCMSG 1044)",
            "MAT CHECK",
        )

    # 6. Compatible elements check (if model is provided)
    if model is not None:
        actual_mid = mid
        if hasattr(model, "element_groups") and callable(model.element_groups):
            try:
                grps = list(model.element_groups())
            except TypeError:
                grps = []
            for item in grps:
                if isinstance(item, tuple) and len(item) == 2:
                    name, el_group = item
                else:
                    continue
                mids_in_group = set()
                if hasattr(el_group, "values") and callable(el_group.values):
                    for el in el_group.values():
                        el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                        if el_mid is not None:
                            mids_in_group.add(el_mid)
                if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                    for _, m_part, _ in el_group.state["slices"]:
                        m_id = getattr(m_part, "id", None)
                        if m_id is not None:
                            mids_in_group.add(m_id)
                if actual_mid in mids_in_group:
                    if name in ("bricks", "bricks_heph", "bric20s", "tetras", "tetra10s", "penta6", "penta6s", "pyra5", "solids"):
                        log.error(
                            f"/MAT/LAW73/{actual_mid} (/MAT/HILL_THERM) is not supported for solid elements ({name}) (ANCMSG 305)",
                            "MAT CHECK",
                        )
                    elif name in ("trusses", "beams", "springs"):
                        log.error(
                            f"/MAT/LAW73/{actual_mid} (/MAT/HILL_THERM) is not supported for 1D elements ({name}) (ANCMSG 306)",
                            "MAT CHECK",
                        )

        if hasattr(model, "parts") and isinstance(model.parts, dict):
            for pid, part in model.parts.items():
                p_mid = getattr(part, "mat_id", getattr(part, "mid", None))
                if p_mid == actual_mid:
                    etype = str(getattr(part, "elem_type", getattr(part, "type", ""))).upper()
                    prop_id = getattr(part, "prop_id", None)
                    if not etype or etype in ("NONE", ""):
                        if prop_id is not None:
                            props = getattr(model, "properties", {}) or getattr(model, "props", {})
                            prop = props.get(prop_id) if isinstance(props, dict) else None
                            if prop is None and hasattr(model, "prop_solids") and isinstance(model.prop_solids, dict) and prop_id in model.prop_solids:
                                etype = "SOLID"
                            elif prop is not None:
                                ptype = str(getattr(prop, "type", getattr(prop, "card_name", ""))).upper()
                                if any(s in ptype for s in ("SOLID", "TYPE14")):
                                    etype = "SOLID"
                                elif any(s in ptype for s in ("BEAM", "TRUSS", "SPRING", "TYPE3", "TYPE4", "TYPE12")):
                                    etype = "BEAM"
                    if any(s in etype for s in ("SOLID", "BRICK", "TETRA", "HEXA", "PENTA", "PYRA")):
                        log.error(
                            f"/MAT/LAW73/{actual_mid} (/MAT/HILL_THERM) is not supported for solid elements ({etype.lower()}) (ANCMSG 305)",
                            "MAT CHECK",
                        )
                    elif any(s in etype for s in ("BEAM", "TRUSS", "SPRING", "1D")):
                        log.error(
                            f"/MAT/LAW73/{actual_mid} (/MAT/HILL_THERM) is not supported for 1D elements ({etype.lower()}) (ANCMSG 306)",
                            "MAT CHECK",
                        )


_check_mat_law73 = check_mat_law73


def check_mat_law66(
    mat: Any = None,
    log: MessageLog | None = None,
    model: Model | None = None,
    mat_id: Any = None,
    **kwargs,
) -> None:
    """Validate /MAT/LAW66 (/MAT/PLAS_TAB_COSSER, /MAT/PLAS_COSSER) parameter bounds (M562).

    Required checks:
      - rho > 0
      - e > 0
      - 0 <= nu < 0.5 (ANCMSG 1514)
      - pc >= 0
      - pt >= 0
      - Compatible elements:
        - accepts solids and shells
        - rejects 1D elements (trusses, beams, springs) with ANCMSG 306
    """
    if mat is None and "material" in kwargs:
        mat = kwargs["material"]
    if log is None and "log" in kwargs:
        log = kwargs["log"]
    if model is None and "model" in kwargs:
        model = kwargs["model"]
    if mat_id is None and "mat_id" in kwargs:
        mat_id = kwargs["mat_id"]
    if log is None:
        return

    mid = mat_id if mat_id is not None else getattr(mat, "id", getattr(mat, "mat_id", 0))

    def _extract(keys: list[str], default: float = 0.0) -> float:
        params = getattr(mat, "params", None)
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = _extract(["rho", "rho0", "MAT_RHO", "RHO", "RHO0"], default=0.0)
    if rho0 <= 0.0:
        log.error(f"/MAT/LAW66/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's modulus E > 0
    e = _extract(["e", "E", "MAT_E", "young"], default=0.0)
    if e <= 0.0:
        log.error(f"/MAT/LAW66/{mid}: Young's modulus E must be > 0 (got {e:g})", "MAT CHECK")

    # 3. Poisson's ratio: 0.0 <= nu < 0.5 (ANCMSG 1514)
    nu = _extract(["nu", "Nu", "NU", "MAT_NU"], default=0.0)
    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW66/{mid}: Poisson's ratio nu must satisfy 0 <= nu < 0.5 (got {nu:g}) (ANCMSG 1514)", "MAT CHECK")

    # 4. PC >= 0
    pc = _extract(["pc", "Pc", "P_C", "PC", "MAT_PC"], default=0.0)
    if pc < 0.0:
        log.error(f"/MAT/LAW66/{mid}: P_c must be >= 0 (got {pc:g})", "MAT CHECK")

    # 5. PT >= 0
    pt = _extract(["pt", "Pt", "P_T", "PT", "MAT_PT"], default=0.0)
    if pt < 0.0:
        log.error(f"/MAT/LAW66/{mid}: P_t must be >= 0 (got {pt:g})", "MAT CHECK")

    # 6. Compatible elements check (if model is provided)
    if model is not None:
        actual_mid = mid
        if hasattr(model, "element_groups") and callable(model.element_groups):
            try:
                grps = list(model.element_groups())
            except TypeError:
                grps = []
            for item in grps:
                if isinstance(item, tuple) and len(item) == 2:
                    name, el_group = item
                else:
                    continue
                mids_in_group = set()
                if hasattr(el_group, "values") and callable(el_group.values):
                    for el in el_group.values():
                        el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                        if el_mid is None and hasattr(model, "parts") and isinstance(model.parts, dict):
                            pid = getattr(el, "part_id", getattr(el, "pid", None))
                            if pid in model.parts:
                                el_mid = getattr(model.parts[pid], "mat_id", getattr(model.parts[pid], "mid", None))
                        if el_mid is not None:
                            mids_in_group.add(el_mid)
                if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                    for _, m_part, _ in el_group.state["slices"]:
                        m_id = getattr(m_part, "id", None)
                        if m_id is not None:
                            mids_in_group.add(m_id)
                if actual_mid in mids_in_group:
                    if name in ("trusses", "beams", "springs"):
                        log.error(
                            f"/MAT/LAW66/{actual_mid} (/MAT/PLAS_TAB_COSSER) is not supported for 1D elements ({name}) (ANCMSG 306)",
                            "MAT CHECK",
                        )

        if hasattr(model, "parts") and isinstance(model.parts, dict):
            for pid, part in model.parts.items():
                p_mid = getattr(part, "mat_id", getattr(part, "mid", None))
                if p_mid == actual_mid:
                    etype = str(getattr(part, "elem_type", getattr(part, "type", ""))).upper()
                    prop_id = getattr(part, "prop_id", None)
                    if not etype or etype in ("NONE", ""):
                        if prop_id is not None:
                            props = getattr(model, "properties", {}) or getattr(model, "props", {})
                            prop = props.get(prop_id) if isinstance(props, dict) else None
                            if prop is not None:
                                ptype = str(getattr(prop, "type", getattr(prop, "card_name", ""))).upper()
                                if any(s in ptype for s in ("BEAM", "TRUSS", "SPRING", "TYPE3", "TYPE4", "TYPE12")):
                                    etype = "BEAM"
                    if any(s in etype for s in ("BEAM", "TRUSS", "SPRING", "1D")):
                        log.error(
                            f"/MAT/LAW66/{actual_mid} (/MAT/PLAS_TAB_COSSER) is not supported for 1D elements ({etype.lower()}) (ANCMSG 306)",
                            "MAT CHECK",
                        )


_check_mat_law66 = check_mat_law66


def check_mat_law74(
    model: Any = None,
    mat_id: Any = None,
    mat: Any = None,
    log: Any = None,
    **kwargs: Any,
) -> None:
    """Validate /MAT/LAW74 (/MAT/HILL_3D, /MAT/ORTH_PLAS, /MAT/THERM_HILL) parameter bounds (M563).

    Citing hm_read_mat74.F and radioss120/MAT/matl74_74.cfg:
      - rho0 > 0
      - e > 0
      - 0 <= nu < 0.5 (ANCMSG 1514)
      - S11Y > 0, S22Y > 0, S33Y > 0, S12Y > 0, S23Y > 0, S31Y > 0 (ANCMSG 822)
      - epsr1 < epsr2: if epsr1 >= epsr2, error (ANCMSG 1044)
      - Reject 2D analysis: N2D > 0 (ANCMSG 305)
      - Compatible elements: solids only; reject shell elements (ANCMSG 305) and 1D elements (ANCMSG 306).
    """
    actual_log = log
    actual_model = model
    actual_mat = mat
    actual_mid = mat_id

    candidates = [c for c in (model, mat_id, mat, log) if c is not None]
    if hasattr(model, "params") and not isinstance(model, Model):
        actual_mat = model
        actual_model = None
        actual_log = mat_id if isinstance(mat_id, MessageLog) else log
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    elif hasattr(mat_id, "params") and isinstance(mat, MessageLog):
        actual_mat = mat_id
        actual_log = mat
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    else:
        for c in candidates:
            if isinstance(c, MessageLog):
                actual_log = c
            elif isinstance(c, Model):
                actual_model = c
            elif hasattr(c, "params") or hasattr(c, "rho") or hasattr(c, "rho0") or hasattr(c, "s11y"):
                actual_mat = c
            elif isinstance(c, int) and not isinstance(c, bool):
                actual_mid = c

    if actual_mat is None:
        if "mat" in kwargs:
            actual_mat = kwargs["mat"]
        elif "material" in kwargs:
            actual_mat = kwargs["material"]
        elif "mat74" in kwargs:
            actual_mat = kwargs["mat74"]
        elif "mat_law74" in kwargs:
            actual_mat = kwargs["mat_law74"]
        elif "mat_hill_3d" in kwargs:
            actual_mat = kwargs["mat_hill_3d"]
        elif "mat_orth_plas" in kwargs:
            actual_mat = kwargs["mat_orth_plas"]

    if actual_log is None:
        if "log" in kwargs:
            actual_log = kwargs["log"]
        elif "logger" in kwargs:
            actual_log = kwargs["logger"]
        else:
            actual_log = MessageLog()

    mid_kw = kwargs.get("mat_id", kwargs.get("mid", actual_mid))
    if actual_mat is None and actual_model is not None:
        if mid_kw is not None:
            actual_mat = actual_model.materials.get(mid_kw) or getattr(actual_model, "mat_law74s", {}).get(mid_kw)
        else:
            for m_id, m_obj in list(getattr(actual_model, "mat_law74s", {}).items()):
                check_mat_law74(model=actual_model, mat_id=m_id, mat=m_obj, log=actual_log)
            for m_id, m_obj in list(getattr(actual_model, "materials", {}).items()):
                if getattr(m_obj, "law", None) in (74, "74", "LAW74", "HILL_3D", "ORTH_PLAS") or getattr(m_obj, "law_name", None) in ("74", "LAW74", "HILL_3D", "ORTH_PLAS", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS"):
                    if m_id not in getattr(actual_model, "mat_law74s", {}):
                        check_mat_law74(model=actual_model, mat_id=m_id, mat=m_obj, log=actual_log)
            return

    if actual_mat is None:
        return

    mat = actual_mat
    log = actual_log
    model = actual_model
    mid = getattr(mat, "id", mid_kw if mid_kw is not None else 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if k in kwargs and kwargs[k] is not None:
                try:
                    return float(kwargs[k])
                except (TypeError, ValueError):
                    pass
        return default

    # 1. Density rho0 > 0
    rho0 = _extract(["rho", "rho0", "MAT_RHO", "RHO", "RHO0"], default=0.0)
    if rho0 <= 0.0:
        log.error(f"/MAT/LAW74/{mid}: initial density RHO must be > 0 (got {rho0:g}) (ANCMSG 1514)", "MAT CHECK")

    # 2. Young's modulus E > 0
    e = _extract(["e", "E", "MAT_E", "young"], default=0.0)
    if e <= 0.0:
        log.error(f"/MAT/LAW74/{mid}: Young's modulus E must be > 0 (got {e:g}) (ANCMSG 1514)", "MAT CHECK")

    # 3. Poisson's ratio: 0.0 <= nu < 0.5 (ANCMSG 1514)
    nu = _extract(["nu", "Nu", "NU", "MAT_NU"], default=0.0)
    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW74/{mid}: Poisson's ratio nu must satisfy 0 <= nu < 0.5 (got {nu:g}) (ANCMSG 1514)", "MAT CHECK")

    # 4. Yield stresses: S11Y > 0, S22Y > 0, S33Y > 0, S12Y > 0, S23Y > 0, S31Y > 0 (ANCMSG 822)
    s11y = _extract(["s11y", "S11Y", "MAT_SIGT1", "sig11y"], default=0.0)
    if s11y <= 0.0:
        log.error(f"/MAT/LAW74/{mid}: yield stress parameter S11Y must be > 0 (got {s11y:g}) (ANCMSG 822)", "MAT CHECK")
    s22y = _extract(["s22y", "S22Y", "MAT_SIGT2", "sig22y"], default=0.0)
    if s22y <= 0.0:
        log.error(f"/MAT/LAW74/{mid}: yield stress parameter S22Y must be > 0 (got {s22y:g}) (ANCMSG 822)", "MAT CHECK")
    s33y = _extract(["s33y", "S33Y", "MAT_SIGT3", "sig33y"], default=0.0)
    if s33y <= 0.0:
        log.error(f"/MAT/LAW74/{mid}: yield stress parameter S33Y must be > 0 (got {s33y:g}) (ANCMSG 822)", "MAT CHECK")
    s12y = _extract(["s12y", "S12Y", "MAT_SIGYT1", "sig12y"], default=0.0)
    if s12y <= 0.0:
        log.error(f"/MAT/LAW74/{mid}: yield stress parameter S12Y must be > 0 (got {s12y:g}) (ANCMSG 822)", "MAT CHECK")
    s23y = _extract(["s23y", "S23Y", "MAT_SIGYT2", "sig23y"], default=0.0)
    if s23y <= 0.0:
        log.error(f"/MAT/LAW74/{mid}: yield stress parameter S23Y must be > 0 (got {s23y:g}) (ANCMSG 822)", "MAT CHECK")
    s31y = _extract(["s31y", "S31Y", "MAT_SIGYT3", "sig31y"], default=0.0)
    if s31y <= 0.0:
        log.error(f"/MAT/LAW74/{mid}: yield stress parameter S31Y must be > 0 (got {s31y:g}) (ANCMSG 822)", "MAT CHECK")

    # 5. Failure strains: epsr1 < epsr2 (ANCMSG 1044)
    epsr1 = _extract(["epsr1", "epst1", "MAT_EPST1"], default=1.0e30)
    epsr2 = _extract(["epsr2", "epst2", "MAT_EPST2"], default=2.0e30)
    if epsr1 >= epsr2:
        log.error(
            f"/MAT/LAW74/{mid}: tensile failure strain 1 ({epsr1:g}) must be less than tensile failure strain 2 ({epsr2:g}) (ANCMSG 1044)",
            "MAT CHECK",
        )

    # 6. 2D analysis check (ANCMSG 305)
    if model is not None and getattr(model, "n2d", 0) > 0:
        log.error(f"/MAT/LAW74/{mid}: LAW74 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")

    # 7. Compatible elements check (ANCMSG 305 for shells, ANCMSG 306 for 1D elements)
    if model is not None:
        actual_mid = mid
        if hasattr(model, "element_groups") and callable(model.element_groups):
            try:
                grps = list(model.element_groups())
            except TypeError:
                grps = []
            for item in grps:
                if isinstance(item, tuple) and len(item) == 2:
                    name, el_group = item
                else:
                    continue
                mids_in_group = set()
                if hasattr(el_group, "values") and callable(el_group.values):
                    for el in el_group.values():
                        el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                        if el_mid is not None:
                            mids_in_group.add(el_mid)
                if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                    for _, m_part, _ in el_group.state["slices"]:
                        m_id = getattr(m_part, "id", None)
                        if m_id is not None:
                            mids_in_group.add(m_id)
                if actual_mid in mids_in_group:
                    if name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads", "quad4_2d", "tria3_2d", "elements_2d", "plane_strain", "plane_stress"):
                        log.error(
                            f"/MAT/LAW74/{actual_mid} (/MAT/HILL_3D) is not supported for shell elements ({name}) (ANCMSG 305)",
                            "MAT CHECK",
                        )
                    elif name in ("trusses", "beams", "springs"):
                        log.error(
                            f"/MAT/LAW74/{actual_mid} (/MAT/HILL_3D) is not supported for 1D elements ({name}) (ANCMSG 306)",
                            "MAT CHECK",
                        )

        if hasattr(model, "parts") and isinstance(model.parts, dict):
            for pid, part in model.parts.items():
                p_mid = getattr(part, "mat_id", getattr(part, "mid", None))
                if p_mid == actual_mid:
                    etype = str(getattr(part, "elem_type", getattr(part, "type", ""))).upper()
                    prop_id = getattr(part, "prop_id", None)
                    if not etype or etype in ("NONE", ""):
                        if prop_id is not None:
                            props = getattr(model, "properties", {}) or getattr(model, "props", {})
                            prop = props.get(prop_id) if isinstance(props, dict) else None
                            if prop is None and hasattr(model, "prop_shells") and isinstance(model.prop_shells, dict) and prop_id in model.prop_shells:
                                etype = "SHELL"
                            elif prop is not None:
                                ptype = str(getattr(prop, "type", getattr(prop, "card_name", ""))).upper()
                                if any(s in ptype for s in ("SHELL", "TYPE1", "TYPE2")):
                                    etype = "SHELL"
                                elif any(s in ptype for s in ("BEAM", "TRUSS", "SPRING", "TYPE3", "TYPE4", "TYPE12")):
                                    etype = "BEAM"
                    if any(s in etype for s in ("SHELL", "QUAD", "TRIA")):
                        log.error(
                            f"/MAT/LAW74/{actual_mid} (/MAT/HILL_3D) is not supported for shell elements ({etype.lower()}) (ANCMSG 305)",
                            "MAT CHECK",
                        )
                    elif any(s in etype for s in ("BEAM", "TRUSS", "SPRING", "1D")):
                        log.error(
                            f"/MAT/LAW74/{actual_mid} (/MAT/HILL_3D) is not supported for 1D elements ({etype.lower()}) (ANCMSG 306)",
                            "MAT CHECK",
                        )


_check_mat_law74 = check_mat_law74


def check_mat_law87(
    model: Any = None,
    mat_id: Any = None,
    mat: Any = None,
    log: Any = None,
    **kwargs: Any,
) -> None:
    """Validate /MAT/LAW87 (/MAT/BARLAT2000, /MAT/BARLAT_2000, /MAT/BARLAT2000_2D) parameter bounds (M564).

    Citing hm_read_mat87.F90 and matl87_barlat.cfg:
      - rho > 0 (ANCMSG 1514)
      - e > 0 (ANCMSG 1514)
      - 0 <= nu < 0.5 (ANCMSG 1514)
      - When flag_fit == 0: yield surface convexity check via eigenvalues of Lp and Lpp:
          - If min(eig(Lp).real) <= 0: warning ANCMSG 3095
          - If min(eig(Lpp).real) <= 0: warning ANCMSG 3102
      - Compatible elements: shells only; reject solid elements (ANCMSG 305) and 1D elements (ANCMSG 306).
    """
    actual_log = log
    actual_model = model
    actual_mat = mat
    actual_mid = mat_id

    candidates = [c for c in (model, mat_id, mat, log) if c is not None]
    if hasattr(model, "params") and not isinstance(model, Model):
        actual_mat = model
        actual_model = None
        actual_log = mat_id if isinstance(mat_id, MessageLog) else log
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    elif hasattr(mat_id, "params") and isinstance(mat, MessageLog):
        actual_mat = mat_id
        actual_log = mat
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    else:
        for c in candidates:
            if isinstance(c, MessageLog):
                actual_log = c
            elif isinstance(c, Model):
                actual_model = c
            elif hasattr(c, "params") or hasattr(c, "rho") or hasattr(c, "rho0") or hasattr(c, "al1") or hasattr(c, "flag_fit"):
                actual_mat = c
            elif isinstance(c, int) and not isinstance(c, bool):
                actual_mid = c

    if actual_mat is None:
        if "mat" in kwargs:
            actual_mat = kwargs["mat"]
        elif "material" in kwargs:
            actual_mat = kwargs["material"]
        elif "mat87" in kwargs:
            actual_mat = kwargs["mat87"]
        elif "mat_law87" in kwargs:
            actual_mat = kwargs["mat_law87"]
        elif "mat_barlat2000" in kwargs:
            actual_mat = kwargs["mat_barlat2000"]
        elif "mat_barlat" in kwargs:
            actual_mat = kwargs["mat_barlat"]

    if actual_log is None:
        if "log" in kwargs:
            actual_log = kwargs["log"]
        elif "logger" in kwargs:
            actual_log = kwargs["logger"]
        else:
            actual_log = MessageLog()

    mid_kw = kwargs.get("mat_id", kwargs.get("mid", actual_mid))
    if actual_mat is None and actual_model is not None:
        if mid_kw is not None:
            actual_mat = actual_model.materials.get(mid_kw) or getattr(actual_model, "mat_law87s", {}).get(mid_kw)
        else:
            for m_id, m_obj in list(getattr(actual_model, "mat_law87s", {}).items()):
                check_mat_law87(model=actual_model, mat_id=m_id, mat=m_obj, log=actual_log)
            for m_id, m_obj in list(getattr(actual_model, "materials", {}).items()):
                if getattr(m_obj, "law", None) in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000") or getattr(m_obj, "law_name", None) in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000"):
                    if m_id not in getattr(actual_model, "mat_law87s", {}):
                        check_mat_law87(model=actual_model, mat_id=m_id, mat=m_obj, log=actual_log)
            return

    if actual_mat is None:
        return

    mat = actual_mat
    log = actual_log
    model = actual_model
    mid = getattr(mat, "id", mid_kw if mid_kw is not None else 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = _extract(["rho", "rho0", "MAT_RHO", "RHO", "RHO0"], default=0.0)
    if rho0 <= 0.0:
        log.error(f"/MAT/LAW87/{mid}: initial density RHO must be > 0 (got {rho0:g}) (ANCMSG 1514)", "MAT CHECK")

    # 2. Young's modulus E > 0
    e = _extract(["e", "E", "MAT_E", "young"], default=0.0)
    if e <= 0.0:
        log.error(f"/MAT/LAW87/{mid}: Young's modulus E must be > 0 (got {e:g}) (ANCMSG 1514)", "MAT CHECK")

    # 3. Poisson's ratio: 0.0 <= nu < 0.5 (ANCMSG 1514)
    nu = _extract(["nu", "Nu", "NU", "MAT_NU"], default=0.0)
    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW87/{mid}: Poisson's ratio NU must satisfy 0 <= nu < 0.5 (got {nu:g}) (ANCMSG 1514)", "MAT CHECK")

    # 4. Convexity check when flag_fit == 0 (ANCMSG 3095 for Lp, ANCMSG 3102 for Lpp)
    flag_fit = int(_extract(["flag_fit", "ifit", "Ifit"], default=0.0))
    if flag_fit == 0:
        al = [1.0] * 8
        if hasattr(mat, "al1") and hasattr(mat, "al8"):
            for i in range(8):
                al[i] = getattr(mat, f"al{i+1}", 1.0)
        elif "alphas" in params and isinstance(params["alphas"], (list, tuple)) and len(params["alphas"]) >= 8:
            for i in range(8):
                al[i] = float(params["alphas"][i])
        elif "alpha" in params and isinstance(params["alpha"], (list, tuple)) and len(params["alpha"]) >= 8:
            for i in range(8):
                al[i] = float(params["alpha"][i])
        else:
            for i in range(8):
                al[i] = _extract([f"al{i+1}", f"a{i+1}", f"alpha{i+1}", f"MAT_ALPHA{i+1}"], default=1.0)

        # First linear transformation matrix Lp:
        lp = np.zeros((3, 3), dtype=float)
        lp[0, 0] = 2.0 * al[0] / 3.0
        lp[0, 1] = -al[0] / 3.0
        lp[1, 0] = -al[1] / 3.0
        lp[1, 1] = 2.0 * al[1] / 3.0
        lp[2, 2] = al[6]
        eig_p = np.linalg.eigvals(lp)
        if np.min(eig_p.real) <= 0.0:
            log.warning(
                f"/MAT/LAW87/{mid}: yield surface not convex (Lp eigenvalues <= 0) (ANCMSG 3095)",
                "MAT CHECK",
            )

        # Second linear transformation matrix Lpp:
        lpp = np.zeros((3, 3), dtype=float)
        lpp[0, 0] = (-2.0 * al[2] + 2.0 * al[3] + 8.0 * al[4] - 2.0 * al[5]) / 9.0
        lpp[0, 1] = (al[2] - 4.0 * al[3] - 4.0 * al[4] + 4.0 * al[5]) / 9.0
        lpp[1, 0] = (4.0 * al[2] - 4.0 * al[3] - 4.0 * al[4] + al[5]) / 9.0
        lpp[1, 1] = (-2.0 * al[2] + 8.0 * al[3] + 2.0 * al[4] - 2.0 * al[5]) / 9.0
        lpp[2, 2] = al[7]
        eig_pp = np.linalg.eigvals(lpp)
        if np.min(eig_pp.real) <= 0.0:
            log.warning(
                f"/MAT/LAW87/{mid}: yield surface not convex (Lpp eigenvalues <= 0) (ANCMSG 3102)",
                "MAT CHECK",
            )

    # 5. Compatible elements check (if model is provided)
    if model is not None:
        actual_mid = mid
        if hasattr(model, "element_groups") and callable(model.element_groups):
            try:
                grps = list(model.element_groups())
            except TypeError:
                grps = []
            for item in grps:
                if isinstance(item, tuple) and len(item) == 2:
                    name, el_group = item
                else:
                    continue
                mids_in_group = set()
                if hasattr(el_group, "values") and callable(el_group.values):
                    for el in el_group.values():
                        el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                        if el_mid is not None:
                            mids_in_group.add(el_mid)
                if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                    for _, m_part, _ in el_group.state["slices"]:
                        m_id = getattr(m_part, "id", None)
                        if m_id is not None:
                            mids_in_group.add(m_id)
                if actual_mid in mids_in_group:
                    if name in ("bricks", "bricks_heph", "bric20s", "tetras", "tetra10s", "penta6", "penta6s", "pyra5", "solids"):
                        log.error(
                            f"/MAT/LAW87/{actual_mid} (/MAT/BARLAT2000) is not supported for solid elements ({name}) (ANCMSG 305)",
                            "MAT CHECK",
                        )
                    elif name in ("trusses", "beams", "springs"):
                        log.error(
                            f"/MAT/LAW87/{actual_mid} (/MAT/BARLAT2000) is not supported for 1D elements ({name}) (ANCMSG 306)",
                            "MAT CHECK",
                        )

        if hasattr(model, "parts") and isinstance(model.parts, dict):
            for pid, part in model.parts.items():
                p_mid = getattr(part, "mat_id", getattr(part, "mid", None))
                if p_mid == actual_mid:
                    etype = str(getattr(part, "elem_type", getattr(part, "type", ""))).upper()
                    prop_id = getattr(part, "prop_id", None)
                    if not etype or etype in ("NONE", ""):
                        if prop_id is not None:
                            props = getattr(model, "properties", {}) or getattr(model, "props", {})
                            prop = props.get(prop_id) if isinstance(props, dict) else None
                            if prop is None and hasattr(model, "prop_solids") and isinstance(model.prop_solids, dict) and prop_id in model.prop_solids:
                                etype = "SOLID"
                            elif prop is not None:
                                ptype = str(getattr(prop, "type", getattr(prop, "card_name", ""))).upper()
                                if any(s in ptype for s in ("SOLID", "TYPE14")):
                                    etype = "SOLID"
                                elif any(s in ptype for s in ("BEAM", "TRUSS", "SPRING", "TYPE3", "TYPE4", "TYPE12")):
                                    etype = "BEAM"
                    if any(s in etype for s in ("SOLID", "BRICK", "TETRA", "HEXA", "PENTA", "PYRA")):
                        log.error(
                            f"/MAT/LAW87/{actual_mid} (/MAT/BARLAT2000) is not supported for solid elements ({etype.lower()}) (ANCMSG 305)",
                            "MAT CHECK",
                        )
                    elif any(s in etype for s in ("BEAM", "TRUSS", "SPRING", "1D")):
                        log.error(
                            f"/MAT/LAW87/{actual_mid} (/MAT/BARLAT2000) is not supported for 1D elements ({etype.lower()}) (ANCMSG 306)",
                            "MAT CHECK",
                        )


_check_mat_law87 = check_mat_law87


def check_mat_law88(
    model: Any = None,
    mat_id: Any = None,
    mat: Any = None,
    log: Any = None,
    **kwargs: Any,
) -> None:
    """Validate /MAT/LAW88 (/MAT/TABULATED_HYPERELASTIC, /MAT/HYPER_ELAS, /MAT/TAB_HYP) parameter bounds (M565).

    Citing hm_read_mat88.F90:
      - RHO0 > 0 (ANCMSG 1514)
      - NL > 0 (ANCMSG 866: No loading curve defined)
      - Ascending strain rate check: if NL > 1 and rate[i] < rate[i-1] (ANCMSG 478)
      - Non-negative shear modulus check: warning ANCMSG 3109
      - Compatible elements: solids (bricks, tetras, penta6, pyra5) and shells (quads, sh3n);
        reject 1D elements (trusses, beams, springs) (ANCMSG 306).
    """
    actual_log = log
    actual_model = model
    actual_mat = mat
    actual_mid = mat_id

    candidates = [c for c in (model, mat_id, mat, log) if c is not None]
    if hasattr(model, "params") and not isinstance(model, Model):
        actual_mat = model
        actual_model = None
        actual_log = mat_id if isinstance(mat_id, MessageLog) else log
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    elif hasattr(mat_id, "params") and isinstance(mat, MessageLog):
        actual_mat = mat_id
        actual_log = mat
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    else:
        for c in candidates:
            if isinstance(c, MessageLog):
                actual_log = c
            elif isinstance(c, Model):
                actual_model = c
            elif hasattr(c, "params") or hasattr(c, "rho0") or hasattr(c, "func_load_list") or hasattr(c, "nl"):
                actual_mat = c
            elif isinstance(c, int) and not isinstance(c, bool):
                actual_mid = c

    if actual_mat is None:
        if "mat" in kwargs:
            actual_mat = kwargs["mat"]
        elif "material" in kwargs:
            actual_mat = kwargs["material"]
        elif "mat88" in kwargs:
            actual_mat = kwargs["mat88"]
        elif "mat_law88" in kwargs:
            actual_mat = kwargs["mat_law88"]

    if actual_log is None:
        if "log" in kwargs:
            actual_log = kwargs["log"]
        elif "logger" in kwargs:
            actual_log = kwargs["logger"]
        else:
            actual_log = MessageLog()

    mid_kw = kwargs.get("mat_id", kwargs.get("mid", actual_mid))
    if actual_mat is None and actual_model is not None:
        if mid_kw is not None and hasattr(actual_model, "materials"):
            actual_mat = actual_model.materials.get(mid_kw)
        if actual_mat is None and hasattr(actual_model, "mat_law88s"):
            actual_mat = actual_model.mat_law88s.get(mid_kw)

    mid = getattr(actual_mat, "id", mid_kw or 0)
    params = getattr(actual_mat, "params", {}) or {}
    log = actual_log
    model = actual_model

    def _extract(keys: list[str], default: Any = 0.0) -> Any:
        for k in keys:
            if hasattr(actual_mat, k):
                val = getattr(actual_mat, k)
                if val is not None:
                    return val
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    return val
        return default

    # 1. Density rho0 > 0 (ANCMSG 1514)
    rho0 = float(_extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0"], default=0.0))
    if rho0 <= 0.0:
        log.error(f"/MAT/LAW88/{mid}: initial density RHO must be > 0 (got {rho0:g}) (ANCMSG 1514)", "MAT CHECK")

    # 2. Number of loading curves NL > 0 (ANCMSG 866)
    nl = int(_extract(["nl", "NL", "LAW88_NL"], default=0))
    func_list = _extract(["func_load_list", "LAW88_arr1", "curves"], default=[])
    if isinstance(func_list, (list, tuple)) and len(func_list) > 0:
        nl = max(nl, len(func_list))
    if nl <= 0:
        log.error(f"/MAT/LAW88/{mid}: no loading curve defined (NL = 0) (ANCMSG 866)", "MAT CHECK")

    # 3. Ascending strain rate check (ANCMSG 478)
    rate_list = _extract(["rate_load_list", "LAW88_arr3", "rates"], default=[])
    if isinstance(rate_list, (list, tuple)) and len(rate_list) > 1:
        for i in range(1, len(rate_list)):
            try:
                r_prev = float(rate_list[i - 1])
                r_curr = float(rate_list[i])
                if r_curr < r_prev:
                    log.error(f"/MAT/LAW88/{mid}: strain rates must be in ascending order (ANCMSG 478)", "MAT CHECK")
                    break
            except (TypeError, ValueError):
                pass

    # 4. Shear modulus non-negative check (ANCMSG 3109)
    shear_val = float(_extract(["shear", "gs", "g", "G", "LAW88_G"], default=0.0))
    if shear_val < 0.0:
        log.warning(f"/MAT/LAW88/{mid}: shear modulus is negative (ANCMSG 3109)", "MAT CHECK")

    # 5. Compatible elements check (if model is provided)
    if model is not None:
        actual_mid = mid
        if hasattr(model, "element_groups") and callable(model.element_groups):
            try:
                grps = list(model.element_groups())
            except TypeError:
                grps = []
            for item in grps:
                if isinstance(item, tuple) and len(item) == 2:
                    name, el_group = item
                else:
                    continue
                mids_in_group = set()
                if hasattr(el_group, "values") and callable(el_group.values):
                    for el in el_group.values():
                        el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                        if el_mid is not None:
                            mids_in_group.add(el_mid)
                if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                    for _, m_part, _ in el_group.state["slices"]:
                        m_id = getattr(m_part, "id", None)
                        if m_id is not None:
                            mids_in_group.add(m_id)
                if actual_mid in mids_in_group:
                    if name in ("trusses", "beams", "springs"):
                        log.error(
                            f"/MAT/LAW88/{actual_mid} is not supported for 1D elements ({name}) (ANCMSG 306)",
                            "MAT CHECK",
                        )

        if hasattr(model, "parts") and isinstance(model.parts, dict):
            for pid, part in model.parts.items():
                p_mid = getattr(part, "mat_id", getattr(part, "mid", None))
                if p_mid == actual_mid:
                    etype = str(getattr(part, "elem_type", getattr(part, "type", ""))).upper()
                    prop_id = getattr(part, "prop_id", None)
                    if not etype or etype in ("NONE", ""):
                        if prop_id is not None:
                            props = getattr(model, "properties", {}) or getattr(model, "props", {})
                            prop = props.get(prop_id) if isinstance(props, dict) else None
                            if prop is not None:
                                ptype = str(getattr(prop, "type", getattr(prop, "card_name", ""))).upper()
                                if any(s in ptype for s in ("BEAM", "TRUSS", "SPRING")):
                                    etype = "1D"
                    if any(s in etype for s in ("BEAM", "TRUSS", "SPRING", "1D")):
                        log.error(
                            f"/MAT/LAW88/{actual_mid} is not supported for 1D elements ({etype.lower()}) (ANCMSG 306)",
                            "MAT CHECK",
                        )


_check_mat_law88 = check_mat_law88


def check_mat_law92(
    model: Any = None,
    mat_id: Any = None,
    mat: Any = None,
    log: Any = None,
    **kwargs: Any,
) -> None:
    """Validate /MAT/LAW92 (/MAT/ARRUDA_BOYCE, /MAT/ARRUDA-BOYCE) parameter bounds (M566).

    Citing hm_read_mat92.F:
      - RHO0 > 0 (ANCMSG 1514)
      - Direct formulation:
        - MU > 0 (warning if <= 0: ANCMSG 3109)
        - D >= 0 (error if < 0)
        - LAM > 0 (warning if <= 0)
        - 0 <= NU < 0.5 (warning/error if invalid)
      - Tabulated/fitted formulation (IFUNC > 0):
        - ITEST in (1, 2, 3) (or default 1)
        - SCALEFAC > 0
        - 0 <= NU < 0.5
      - Compatible elements: solids (bricks, tetras, penta6, pyra5) and shells (quads, sh3n);
        reject 1D elements (trusses, beams, springs) (ANCMSG 306).
    """
    actual_log = log
    actual_model = model
    actual_mat = mat
    actual_mid = mat_id

    candidates = [c for c in (model, mat_id, mat, log) if c is not None]
    if hasattr(model, "params") and not isinstance(model, Model):
        actual_mat = model
        actual_model = None
        actual_log = mat_id if isinstance(mat_id, MessageLog) else log
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    elif hasattr(mat_id, "params") and isinstance(mat, MessageLog):
        actual_mat = mat_id
        actual_log = mat
        actual_mid = getattr(actual_mat, "id", kwargs.get("mat_id", 0))
    else:
        for c in candidates:
            if isinstance(c, MessageLog):
                actual_log = c
            elif isinstance(c, Model):
                actual_model = c
            elif hasattr(c, "params") or hasattr(c, "rho0") or hasattr(c, "mu") or hasattr(c, "lam"):
                actual_mat = c
            elif isinstance(c, int) and not isinstance(c, bool):
                actual_mid = c

    if actual_mat is None:
        for k in ("mat", "material", "mat92", "mat_law92"):
            if k in kwargs:
                actual_mat = kwargs[k]
                break

    if actual_log is None:
        if "log" in kwargs:
            actual_log = kwargs["log"]
        elif "logger" in kwargs:
            actual_log = kwargs["logger"]
        else:
            actual_log = MessageLog()

    if actual_mat is None:
        return

    mid = actual_mid if actual_mid is not None else getattr(actual_mat, "id", getattr(actual_mat, "mat_id", 0))
    params = getattr(actual_mat, "params", {}) or {}

    def get_val(key: str, default: float = 0.0) -> float:
        if isinstance(params, dict) and key in params:
            try:
                return float(params[key])
            except (ValueError, TypeError):
                pass
        for attr in (key.lower(), key.upper(), key):
            if hasattr(actual_mat, attr):
                try:
                    val = getattr(actual_mat, attr)
                    if val is not None:
                        return float(val)
                except (ValueError, TypeError):
                    pass
        return default

    # 1. Density check (ANCMSG 1514)
    rho0 = get_val("rho0", get_val("rho_initial", get_val("RHO_I", get_val("RHO0", get_val("MAT_RHO", 0.0)))))
    if rho0 <= 0.0:
        actual_log.error(
            f"/MAT/LAW92/{mid}: initial density RHO0 must be > 0, got {rho0} (ANCMSG 1514)",
            "MAT CHECK",
        )

    # 2. Check if curve-driven or direct
    fct_id = int(get_val("fct_id", get_val("fct_idi", get_val("func_id", get_val("IFUNC", 0.0)))))
    if fct_id > 0:
        itest = int(get_val("itest", get_val("itype", 1.0)))
        if itest not in (1, 2, 3):
            actual_log.warning(
                f"/MAT/LAW92/{mid}: unknown test type Itype={itest}, defaulting to 1 (uniaxial)",
                "MAT CHECK",
            )
        scalefac = get_val("fscale", get_val("scalefac", 1.0))
        if scalefac <= 0.0:
            actual_log.error(
                f"/MAT/LAW92/{mid}: scale factor Fscale must be > 0, got {scalefac}",
                "MAT CHECK",
            )
        nu = get_val("nu", 0.495)
        if nu < 0.0 or nu >= 0.5:
            actual_log.error(
                f"/MAT/LAW92/{mid}: Poisson's ratio NU must satisfy 0 <= NU < 0.5, got {nu}",
                "MAT CHECK",
            )
    else:
        # Direct formulation: MU, D, LAM
        mu = get_val("mu", get_val("mue1", get_val("MAT_MUE1", 0.0)))
        if mu <= 0.0:
            actual_log.warning(
                f"/MAT/LAW92/{mid}: shear modulus MU should be > 0, got {mu} (ANCMSG 3109)",
                "MAT CHECK",
            )
        d = get_val("d", get_val("MAT_D", 0.0))
        if d < 0.0:
            actual_log.error(
                f"/MAT/LAW92/{mid}: compressibility parameter D must be >= 0, got {d}",
                "MAT CHECK",
            )
        lam = get_val("lam", get_val("lamda", get_val("lambda_m", 7.0)))
        if lam <= 0.0:
            actual_log.warning(
                f"/MAT/LAW92/{mid}: limit stretch LAMDA should be > 0, got {lam}",
                "MAT CHECK",
            )
        nu = get_val("nu", 0.495)
        if nu < 0.0 or nu >= 0.5:
            actual_log.error(
                f"/MAT/LAW92/{mid}: Poisson's ratio NU must satisfy 0 <= NU < 0.5, got {nu}",
                "MAT CHECK",
            )

    # 3. Compatible elements check
    if actual_model is not None:
        if hasattr(actual_model, "element_groups") and callable(actual_model.element_groups):
            try:
                grps = list(actual_model.element_groups())
            except TypeError:
                grps = []
            for item in grps:
                if isinstance(item, tuple) and len(item) == 2:
                    name, el_group = item
                else:
                    continue
                mids_in_group = set()
                if hasattr(el_group, "values") and callable(el_group.values):
                    for el in el_group.values():
                        el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                        if el_mid is not None:
                            mids_in_group.add(el_mid)
                if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                    for _, m_part, _ in el_group.state["slices"]:
                        m_id = getattr(m_part, "id", None)
                        if m_id is not None:
                            mids_in_group.add(m_id)
                if mid in mids_in_group:
                    if name in ("trusses", "beams", "springs"):
                        actual_log.error(
                            f"/MAT/LAW92/{mid} is not supported for 1D elements ({name}) (ANCMSG 306)",
                            "MAT CHECK",
                        )

        if hasattr(actual_model, "parts") and isinstance(actual_model.parts, dict):
            for pid, part in actual_model.parts.items():
                p_mid = getattr(part, "mat_id", getattr(part, "mid", None))
                if p_mid == mid:
                    etype = str(getattr(part, "elem_type", getattr(part, "type", ""))).upper()
                    prop_id = getattr(part, "prop_id", None)
                    if not etype or etype in ("NONE", ""):
                        if prop_id is not None:
                            props = getattr(actual_model, "properties", {}) or getattr(actual_model, "props", {})
                            prop = props.get(prop_id) if isinstance(props, dict) else None
                            if prop is not None:
                                ptype = str(getattr(prop, "type", getattr(prop, "card_name", ""))).upper()
                                if any(s in ptype for s in ("BEAM", "TRUSS", "SPRING")):
                                    etype = "1D"
                    if any(s in etype for s in ("BEAM", "TRUSS", "SPRING", "1D")):
                        actual_log.error(
                            f"/MAT/LAW92/{mid} is not supported for 1D elements ({etype.lower()}) (ANCMSG 306)",
                            "MAT CHECK",
                        )


_check_mat_law92 = check_mat_law92


def check_mat_law94(*args: Any, **kwargs: Any) -> None:
    """Validate /MAT/LAW94 or /MAT/YEOH parameter bounds and element compatibility (M567).

    Checks:
      1. RHO0 > 0 (ANCMSG 1514).
      2. C10 >= 0 (warning if <= 0).
      3. D1, D2, D3 >= 0.
      4. 0 <= NU < 0.5 if explicitly given.
      5. Compatible elements: 3D solids and 2D shells only. 1D elements emit ANCMSG 306.
    """
    actual_mat = None
    actual_log = None
    actual_model = None
    actual_mid = None

    if "model" in kwargs:
        actual_model = kwargs["model"]
    if "mat" in kwargs:
        actual_mat = kwargs["mat"]
    if "log" in kwargs:
        actual_log = kwargs["log"]
    if "mat_id" in kwargs:
        actual_mid = kwargs["mat_id"]

    for c in args:
        if isinstance(c, MessageLog):
            actual_log = c
        elif isinstance(c, Model):
            actual_model = c
        elif hasattr(c, "params") or hasattr(c, "rho0") or hasattr(c, "c10"):
            actual_mat = c
        elif isinstance(c, int) and not isinstance(c, bool):
            actual_mid = c

    if actual_mat is None:
        for k in ("mat", "material", "mat94", "mat_law94", "mat_yeoh"):
            if k in kwargs:
                actual_mat = kwargs[k]
                break

    if actual_log is None:
        if "log" in kwargs:
            actual_log = kwargs["log"]
        elif "logger" in kwargs:
            actual_log = kwargs["logger"]
        else:
            actual_log = MessageLog()

    if actual_mat is None:
        return

    mid = actual_mid if actual_mid is not None else getattr(actual_mat, "id", getattr(actual_mat, "mat_id", 0))
    params = getattr(actual_mat, "params", {}) or {}

    def get_val(key: str, default: float = 0.0) -> float:
        if isinstance(params, dict) and key in params:
            try:
                return float(params[key])
            except (ValueError, TypeError):
                pass
        for attr in (key.lower(), key.upper(), key):
            if hasattr(actual_mat, attr):
                try:
                    val = getattr(actual_mat, attr)
                    if val is not None:
                        return float(val)
                except (ValueError, TypeError):
                    pass
        return default

    # 1. Density check (ANCMSG 1514)
    rho0 = get_val("rho0", get_val("rho_initial", get_val("RHO_I", get_val("RHO0", get_val("MAT_RHO", 0.0)))))
    if rho0 <= 0.0:
        actual_log.error(
            f"/MAT/LAW94/{mid}: initial density RHO0 must be > 0, got {rho0} (ANCMSG 1514)",
            "MAT CHECK",
        )

    # 2. Parameters check
    c10 = get_val("c10", get_val("LAW94_C01", get_val("C10", 0.0)))
    if c10 <= 0.0:
        actual_log.warning(
            f"/MAT/LAW94/{mid}: shear parameter C10 should be > 0, got {c10}",
            "MAT CHECK",
        )

    d1 = get_val("d1", get_val("LAW94_D1", get_val("D1", 0.0)))
    if d1 < 0.0:
        actual_log.error(
            f"/MAT/LAW94/{mid}: compressibility parameter D1 must be >= 0, got {d1}",
            "MAT CHECK",
        )

    d2 = get_val("d2", get_val("LAW94_D2", get_val("D2", 0.0)))
    if d2 < 0.0:
        actual_log.error(
            f"/MAT/LAW94/{mid}: compressibility parameter D2 must be >= 0, got {d2}",
            "MAT CHECK",
        )

    d3 = get_val("d3", get_val("LAW94_D3", get_val("D3", 0.0)))
    if d3 < 0.0:
        actual_log.error(
            f"/MAT/LAW94/{mid}: compressibility parameter D3 must be >= 0, got {d3}",
            "MAT CHECK",
        )

    nu = get_val("nu", 0.495)
    if nu < 0.0 or nu >= 0.5:
        actual_log.error(
            f"/MAT/LAW94/{mid}: Poisson's ratio NU must satisfy 0 <= NU < 0.5, got {nu}",
            "MAT CHECK",
        )

    # 3. Compatible elements check
    if actual_model is not None:
        if hasattr(actual_model, "element_groups") and callable(actual_model.element_groups):
            try:
                grps = list(actual_model.element_groups())
            except TypeError:
                grps = []
            for item in grps:
                if isinstance(item, tuple) and len(item) == 2:
                    name, el_group = item
                else:
                    continue
                mids_in_group = set()
                if hasattr(el_group, "values") and callable(el_group.values):
                    for el in el_group.values():
                        el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                        if el_mid is not None:
                            mids_in_group.add(el_mid)
                if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                    for _, m_part, _ in el_group.state["slices"]:
                        m_id = getattr(m_part, "id", None)
                        if m_id is not None:
                            mids_in_group.add(m_id)
                if mid in mids_in_group:
                    if name in ("trusses", "beams", "springs"):
                        actual_log.error(
                            f"/MAT/LAW94/{mid} is not supported for 1D elements ({name}) (ANCMSG 306)",
                            "MAT CHECK",
                        )

        if hasattr(actual_model, "parts") and isinstance(actual_model.parts, dict):
            for pid, part in actual_model.parts.items():
                p_mid = getattr(part, "mat_id", getattr(part, "mid", None))
                if p_mid == mid:
                    etype = str(getattr(part, "elem_type", getattr(part, "type", ""))).upper()
                    prop_id = getattr(part, "prop_id", None)
                    if not etype or etype in ("NONE", ""):
                        if prop_id is not None:
                            props = getattr(actual_model, "properties", {}) or getattr(actual_model, "props", {})
                            prop = props.get(prop_id) if isinstance(props, dict) else None
                            if prop is not None:
                                ptype = str(getattr(prop, "type", getattr(prop, "card_name", ""))).upper()
                                if any(s in ptype for s in ("BEAM", "TRUSS", "SPRING")):
                                    etype = "1D"
                    if any(s in etype for s in ("BEAM", "TRUSS", "SPRING", "1D")):
                        actual_log.error(
                            f"/MAT/LAW94/{mid} is not supported for 1D elements ({etype.lower()}) (ANCMSG 306)",
                            "MAT CHECK",
                        )


_check_mat_law94 = check_mat_law94


def check_mat_law95(*args: Any, **kwargs: Any) -> None:
    """Validate /MAT/LAW95 or /MAT/BERGSTROM_BOYCE parameter bounds and element compatibility (M569).

    Checks:
      1. RHO0 > 0 (ANCMSG 1514).
      2. Initial shear modulus G0 = 2*(C10 + C01)*(1 + Sb) > 0 (warning if <= 0).
      3. D1, D2, D3 >= 0.
      4. 0 <= NU < 0.5 if explicitly given.
      5. Creep rate parameter A >= 0; if A > 0: -1 < C < 0, M >= 1.0, KSI >= 0, TAU_REF > 0.
      6. Compatible elements: 3D solids only (SOLID_ISOTROPIC).
         - Rejects 2D shells: shells, shells_qbat, shells_qeph, sh3n, quads (ANCMSG 305).
         - Rejects 1D elements: trusses, beams, springs (ANCMSG 306).
    """
    actual_mat = None
    actual_log = None
    actual_model = None
    actual_mid = None

    if "model" in kwargs:
        actual_model = kwargs["model"]
    if "mat" in kwargs:
        actual_mat = kwargs["mat"]
    if "log" in kwargs:
        actual_log = kwargs["log"]
    if "mat_id" in kwargs:
        actual_mid = kwargs["mat_id"]

    for c in args:
        if isinstance(c, MessageLog):
            actual_log = c
        elif isinstance(c, Model):
            actual_model = c
        elif hasattr(c, "params") or hasattr(c, "rho0") or hasattr(c, "c10"):
            actual_mat = c
        elif isinstance(c, int) and not isinstance(c, bool):
            actual_mid = c

    if actual_mat is None:
        for k in ("mat", "material", "mat95", "mat_law95", "mat_bergstrom_boyce"):
            if k in kwargs:
                actual_mat = kwargs[k]
                break

    if actual_log is None:
        if "log" in kwargs:
            actual_log = kwargs["log"]
        elif "logger" in kwargs:
            actual_log = kwargs["logger"]
        else:
            actual_log = MessageLog()

    if actual_mat is None:
        return

    mid = actual_mid if actual_mid is not None else getattr(actual_mat, "id", getattr(actual_mat, "mat_id", 0))
    params = getattr(actual_mat, "params", {}) or {}

    def get_val(key: str, default: float = 0.0) -> float:
        if isinstance(params, dict) and key in params:
            try:
                return float(params[key])
            except (ValueError, TypeError):
                pass
        for attr in (key.lower(), key.upper(), key):
            if hasattr(actual_mat, attr):
                try:
                    val = getattr(actual_mat, attr)
                    if val is not None:
                        return float(val)
                except (ValueError, TypeError):
                    pass
        return default

    # 1. Density check (ANCMSG 1514)
    rho0 = get_val("rho0", get_val("rho_initial", get_val("RHO_I", get_val("RHO0", get_val("MAT_RHO", 0.0)))))
    if rho0 <= 0.0:
        actual_log.error(
            f"/MAT/LAW95/{mid}: initial density RHO0 must be > 0, got {rho0} (ANCMSG 1514)",
            "MAT CHECK",
        )

    # 2. Hyperelastic parameters & initial shear modulus
    c10 = get_val("c10", get_val("C10", 0.0))
    c01 = get_val("c01", get_val("C01", 0.0))
    sb = get_val("sb", get_val("Sb", 0.0))
    if sb < 0.0:
        actual_log.error(
            f"/MAT/LAW95/{mid}: network B scale factor Sb must be >= 0, got {sb}",
            "MAT CHECK",
        )

    g0 = 2.0 * (c10 + c01) * (1.0 + sb)
    if g0 <= 0.0:
        actual_log.warning(
            f"/MAT/LAW95/{mid}: initial shear modulus G0 = 2*(C10+C01)*(1+Sb) should be > 0, got {g0}",
            "MAT CHECK",
        )

    # 3. Compressibility parameters
    d1 = get_val("d1", get_val("D1", 0.0))
    if d1 < 0.0:
        actual_log.error(
            f"/MAT/LAW95/{mid}: compressibility parameter D1 must be >= 0, got {d1}",
            "MAT CHECK",
        )
    d2 = get_val("d2", get_val("D2", 0.0))
    if d2 < 0.0:
        actual_log.error(
            f"/MAT/LAW95/{mid}: compressibility parameter D2 must be >= 0, got {d2}",
            "MAT CHECK",
        )
    d3 = get_val("d3", get_val("D3", 0.0))
    if d3 < 0.0:
        actual_log.error(
            f"/MAT/LAW95/{mid}: compressibility parameter D3 must be >= 0, got {d3}",
            "MAT CHECK",
        )

    nu = get_val("nu", 0.0)
    if nu != 0.0 and (nu < 0.0 or nu >= 0.5):
        actual_log.error(
            f"/MAT/LAW95/{mid}: Poisson's ratio NU must satisfy 0 <= NU < 0.5, got {nu}",
            "MAT CHECK",
        )

    # 4. Creep parameters (CFG CHECK: MLAW95_A >= 0, MLAW95_C > -1 and < 0, MLAW95_M >= 1.0)
    a = get_val("a", get_val("A", 0.0))
    if a < 0.0:
        actual_log.error(
            f"/MAT/LAW95/{mid}: effective creep rate parameter A must be >= 0, got {a}",
            "MAT CHECK",
        )
    if a > 0.0:
        c_exp = get_val("c", get_val("C", get_val("expc", -0.7)))
        if c_exp <= -1.0 or c_exp >= 0.0:
            actual_log.error(
                f"/MAT/LAW95/{mid}: creep strain exponent C must satisfy -1 < C < 0, got {c_exp}",
                "MAT CHECK",
            )
        m_exp = get_val("m", get_val("M", get_val("expm", 1.0)))
        if m_exp < 1.0:
            actual_log.error(
                f"/MAT/LAW95/{mid}: stress exponent M must satisfy M >= 1.0, got {m_exp}",
                "MAT CHECK",
            )
        ksi = get_val("ksi", get_val("KSI", 0.01))
        if ksi < 0.0:
            actual_log.error(
                f"/MAT/LAW95/{mid}: cut-off strain parameter KSI must be >= 0, got {ksi}",
                "MAT CHECK",
            )
        tau_ref = get_val("tau_ref", get_val("TAU_REF", get_val("tauref", 1.0)))
        if tau_ref <= 0.0:
            actual_log.error(
                f"/MAT/LAW95/{mid}: reference stress TAU_REF must be > 0, got {tau_ref}",
                "MAT CHECK",
            )

    # 5. Compatible elements check (3D continuum solids only, reject shells ANCMSG 305 and 1D ANCMSG 306)
    if actual_model is not None:
        if hasattr(actual_model, "element_groups") and callable(actual_model.element_groups):
            try:
                grps = list(actual_model.element_groups())
            except TypeError:
                grps = []
            for item in grps:
                if isinstance(item, tuple) and len(item) == 2:
                    name, el_group = item
                else:
                    continue
                mids_in_group = set()
                if hasattr(el_group, "values") and callable(el_group.values):
                    for el in el_group.values():
                        el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                        if el_mid is not None:
                            mids_in_group.add(el_mid)
                if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                    for _, m_part, _ in el_group.state["slices"]:
                        m_id = getattr(m_part, "id", None)
                        if m_id is not None:
                            mids_in_group.add(m_id)
                if mid in mids_in_group:
                    if name in ("trusses", "beams", "springs"):
                        actual_log.error(
                            f"/MAT/LAW95/{mid} is not supported for 1D elements ({name}) (ANCMSG 306)",
                            "MAT CHECK",
                        )
                    elif name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
                        actual_log.error(
                            f"/MAT/LAW95/{mid} is not supported for 2D shell elements ({name}) (ANCMSG 305)",
                            "MAT CHECK",
                        )

        if hasattr(actual_model, "parts") and isinstance(actual_model.parts, dict):
            for pid, part in actual_model.parts.items():
                p_mid = getattr(part, "mat_id", getattr(part, "mid", None))
                if p_mid == mid:
                    etype = str(getattr(part, "elem_type", getattr(part, "type", ""))).upper()
                    prop_id = getattr(part, "prop_id", None)
                    if not etype or etype in ("NONE", ""):
                        if prop_id is not None:
                            props = getattr(actual_model, "properties", {}) or getattr(actual_model, "props", {})
                            prop = props.get(prop_id) if isinstance(props, dict) else None
                            if prop is not None:
                                ptype = str(getattr(prop, "type", getattr(prop, "card_name", ""))).upper()
                                if any(s in ptype for s in ("BEAM", "TRUSS", "SPRING")):
                                    etype = "1D"
                                elif any(s in ptype for s in ("SHELL", "SH3N", "QUAD")):
                                    etype = "2D"
                    if any(s in etype for s in ("BEAM", "TRUSS", "SPRING", "1D")):
                        actual_log.error(
                            f"/MAT/LAW95/{mid} is not supported for 1D elements ({etype.lower()}) (ANCMSG 306)",
                            "MAT CHECK",
                        )
                    elif any(s in etype for s in ("SHELL", "SH3N", "QUAD", "2D")):
                        actual_log.error(
                            f"/MAT/LAW95/{mid} is not supported for 2D shell elements ({etype.lower()}) (ANCMSG 305)",
                            "MAT CHECK",
                        )


_check_mat_law95 = check_mat_law95


def check_mat_law93(*args: Any, **kwargs: Any) -> None:
    """Validate /MAT/LAW93 or /MAT/ORTH_HILL parameter bounds and element compatibility (M568).

    Checks:
      1. RHO0 > 0 (ANCMSG 1514).
      2. E11, E22, E33, G12, G13, G23 > 0 (ANCMSG 307).
      3. Cross-product Poisson limits:
         nu12*nu21 < 1 (ANCMSG 3068)
         nu13*nu31 < 1 (ANCMSG 3069)
         nu23*nu32 < 1 (ANCMSG 3070)
      4. Orthotropic elasticity compliance determinant > 0 (ANCMSG 307).
      5. Compatible elements: 3D solids and 2D shells only. 1D elements emit ANCMSG 306.
    """
    actual_mat = None
    actual_log = None
    actual_model = None
    actual_mid = None

    if "model" in kwargs:
        actual_model = kwargs["model"]
    if "mat" in kwargs:
        actual_mat = kwargs["mat"]
    if "log" in kwargs:
        actual_log = kwargs["log"]
    if "mat_id" in kwargs:
        actual_mid = kwargs["mat_id"]

    for c in args:
        if isinstance(c, MessageLog):
            actual_log = c
        elif isinstance(c, Model):
            actual_model = c
        elif hasattr(c, "params") or hasattr(c, "rho0") or hasattr(c, "e11"):
            actual_mat = c
        elif isinstance(c, int) and not isinstance(c, bool):
            actual_mid = c

    if actual_mat is None:
        for k in ("mat", "material", "mat93", "mat_law93", "mat_orth_hill"):
            if k in kwargs:
                actual_mat = kwargs[k]
                break

    if actual_log is None:
        if "log" in kwargs:
            actual_log = kwargs["log"]
        elif "logger" in kwargs:
            actual_log = kwargs["logger"]
        else:
            actual_log = MessageLog()

    if actual_mat is None:
        return

    mid = actual_mid if actual_mid is not None else getattr(actual_mat, "id", getattr(actual_mat, "mat_id", 0))
    params = getattr(actual_mat, "params", {}) or {}

    def get_val(key: str, default: float = 0.0) -> float:
        if isinstance(params, dict) and key in params:
            try:
                return float(params[key])
            except (ValueError, TypeError):
                pass
        for attr in (key.lower(), key.upper(), key):
            if hasattr(actual_mat, attr):
                try:
                    val = getattr(actual_mat, attr)
                    if val is not None:
                        return float(val)
                except (ValueError, TypeError):
                    pass
        return default

    # 1. Density check (ANCMSG 1514)
    rho0 = get_val("rho0", get_val("rho_initial", get_val("RHO_I", get_val("RHO0", get_val("MAT_RHO", 0.0)))))
    if rho0 <= 0.0:
        actual_log.error(
            f"/MAT/LAW93/{mid}: initial density RHO0 must be > 0, got {rho0} (ANCMSG 1514)",
            "MAT CHECK",
        )

    # 2. Moduli checks (ANCMSG 307)
    e11 = get_val("e11", get_val("E11", 0.0))
    e22 = get_val("e22", get_val("E22", e11))
    if e22 <= 0.0:
        e22 = e11
    e33 = get_val("e33", get_val("E33", e22))
    if e33 <= 0.0:
        e33 = e22

    g12 = get_val("g12", get_val("G12", 0.0))
    g13 = get_val("g13", get_val("G13", g12))
    if g13 <= 0.0:
        g13 = g12
    g23 = get_val("g23", get_val("G23", g12))
    if g23 <= 0.0:
        g23 = g12

    if e11 <= 0.0 or e22 <= 0.0 or e33 <= 0.0 or g12 <= 0.0 or g13 <= 0.0 or g23 <= 0.0:
        actual_log.error(
            f"/MAT/LAW93/{mid}: Elastic moduli E and G must be > 0 (ANCMSG 307)",
            "MAT CHECK",
        )

    # 3. Poisson ratios stability checks (ANCMSG 3068, 3069, 3070)
    nu12 = get_val("nu12", get_val("NU12", 0.0))
    nu13 = get_val("nu13", get_val("NU13", 0.0))
    nu23 = get_val("nu23", get_val("NU23", 0.0))

    if e11 > 0.0 and e22 > 0.0:
        nu21 = nu12 * e22 / e11
        if nu12 * nu21 >= 1.0:
            actual_log.error(
                f"/MAT/LAW93/{mid}: nu12*nu21 >= 1 ({nu12}*{nu21} >= 1) (ANCMSG 3068)",
                "MAT CHECK",
            )

    if e11 > 0.0 and e33 > 0.0:
        nu31 = nu13 * e33 / e11
        if nu13 * nu31 >= 1.0:
            actual_log.error(
                f"/MAT/LAW93/{mid}: nu13*nu31 >= 1 ({nu13}*{nu31} >= 1) (ANCMSG 3069)",
                "MAT CHECK",
            )

    if e22 > 0.0 and e33 > 0.0:
        nu32 = nu23 * e33 / e22
        if nu23 * nu32 >= 1.0:
            actual_log.error(
                f"/MAT/LAW93/{mid}: nu23*nu32 >= 1 ({nu23}*{nu32} >= 1) (ANCMSG 3070)",
                "MAT CHECK",
            )

    # 4. Compliance determinant check (ANCMSG 307)
    if e11 > 0.0 and e22 > 0.0 and e33 > 0.0:
        nu21 = nu12 * e22 / e11
        nu31 = nu13 * e33 / e11
        nu32 = nu23 * e33 / e22
        det_denom = 1.0 - nu12 * nu21 - nu23 * nu32 - nu31 * nu13 - 2.0 * nu12 * nu23 * nu31
        if det_denom <= 0.0:
            actual_log.error(
                f"/MAT/LAW93/{mid}: Orthotropic elasticity compliance determinant <= 0 (ANCMSG 307)",
                "MAT CHECK",
            )

    # 5. Compatible elements check (ANCMSG 306 for 1D)
    if actual_model is not None:
        if hasattr(actual_model, "element_groups") and callable(actual_model.element_groups):
            try:
                grps = list(actual_model.element_groups())
            except TypeError:
                grps = []
            for item in grps:
                if isinstance(item, tuple) and len(item) == 2:
                    name, el_group = item
                else:
                    continue
                mids_in_group = set()
                if hasattr(el_group, "values") and callable(el_group.values):
                    for el in el_group.values():
                        el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                        if el_mid is not None:
                            mids_in_group.add(el_mid)
                if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                    for _, m_part, _ in el_group.state["slices"]:
                        m_id = getattr(m_part, "id", None)
                        if m_id is not None:
                            mids_in_group.add(m_id)
                if mid in mids_in_group:
                    if name in ("trusses", "beams", "springs"):
                        actual_log.error(
                            f"/MAT/LAW93/{mid} is not supported for 1D elements ({name}) (ANCMSG 306)",
                            "MAT CHECK",
                        )

        if hasattr(actual_model, "parts") and isinstance(actual_model.parts, dict):
            for pid, part in actual_model.parts.items():
                p_mid = getattr(part, "mat_id", getattr(part, "mid", None))
                if p_mid == mid:
                    etype = str(getattr(part, "elem_type", getattr(part, "type", ""))).upper()
                    prop_id = getattr(part, "prop_id", None)
                    if not etype or etype in ("NONE", ""):
                        if prop_id is not None:
                            props = getattr(actual_model, "properties", {}) or getattr(actual_model, "props", {})
                            prop = props.get(prop_id) if isinstance(props, dict) else None
                            if prop is not None:
                                ptype = str(getattr(prop, "type", getattr(prop, "card_name", ""))).upper()
                                if any(s in ptype for s in ("BEAM", "TRUSS", "SPRING")):
                                    etype = "1D"
                    if any(s in etype for s in ("BEAM", "TRUSS", "SPRING", "1D")):
                        actual_log.error(
                            f"/MAT/LAW93/{mid} is not supported for 1D elements ({etype.lower()}) (ANCMSG 306)",
                            "MAT CHECK",
                        )


_check_mat_law93 = check_mat_law93


def check_mat_law100(*args: Any, **kwargs: Any) -> None:
    """Validate /MAT/LAW100, /MAT/VISC_HYP, or /MAT/MNF parameter bounds and element compatibility (M570).

    Checks:
      1. RHO0 > 0 (ANCMSG 1514).
      2. N_net <= 10 (ANCMSG 1567: max parallel networks is 10).
      3. Flag_HE in {1, 2, 3, 4, 5, 13} (ANCMSG 1569: valid hyperelastic law).
      4. Hyperelastic formulation parameters:
         - Flag_HE == 1: Initial shear modulus warning if 2*(C10+C01) <= 0; D1, D2, D3 >= 0.
         - Flag_HE == 2: mu > 0, D >= 0, lambda_m > 1.0 (ANCMSG 1568).
         - Flag_HE == 3: C10 > 0, D1 >= 0.
         - Flag_HE == 4: C10 + C01 > 0, D1 >= 0.
         - Flag_HE == 5: C10 > 0, D1 >= 0.
      5. Creep/Plasticity parameters (Flag_Cr == 1):
         - a_pl >= 0, sigma_pl >= 0, f_pl >= 0, epsilon_f > 0.
      6. Secondary relaxation networks:
         - Flag_visc in {1, 2, 3} (ANCMSG 1808).
         - Stiffness >= 0.
         - Flag_visc == 1: A >= 0; if A > 0: -1 < C < 0, M >= 1.0, KSI >= 0, TAU_REF > 0.
         - Flag_visc == 2: A >= 0, B >= 0, n > 0.
         - Flag_visc == 3: A >= 0, n > 0, M > -1.
      7. Compatible elements: 3D solids only (SOLID_ISOTROPIC).
         - Rejects 2D shells: shells, shells_qbat, shells_qeph, sh3n, quads (ANCMSG 305).
         - Rejects 1D elements: trusses, beams, springs (ANCMSG 306).
    """
    actual_mat = None
    actual_log = None
    actual_model = None
    actual_mid = None

    if "model" in kwargs:
        actual_model = kwargs["model"]
    if "mat" in kwargs:
        actual_mat = kwargs["mat"]
    if "log" in kwargs:
        actual_log = kwargs["log"]
    if "mat_id" in kwargs:
        actual_mid = kwargs["mat_id"]

    for c in args:
        if isinstance(c, MessageLog):
            actual_log = c
        elif isinstance(c, Model):
            actual_model = c
        elif hasattr(c, "params") or hasattr(c, "rho0") or hasattr(c, "flag_he"):
            actual_mat = c
        elif isinstance(c, int) and not isinstance(c, bool):
            actual_mid = c

    if actual_mat is None:
        for k in ("mat", "material", "mat100", "mat_law100", "mat_visc_hyp", "mat_mnf"):
            if k in kwargs:
                actual_mat = kwargs[k]
                break

    if actual_log is None:
        if "log" in kwargs:
            actual_log = kwargs["log"]
        elif "logger" in kwargs:
            actual_log = kwargs["logger"]
        else:
            actual_log = MessageLog()

    if actual_mat is None:
        return

    mid = actual_mid if actual_mid is not None else getattr(actual_mat, "id", getattr(actual_mat, "mat_id", 0))
    params = getattr(actual_mat, "params", {}) or {}

    def get_val(key: str, default: float = 0.0) -> float:
        if isinstance(params, dict) and key in params:
            try:
                return float(params[key])
            except (ValueError, TypeError):
                pass
        for attr in (key.lower(), key.upper(), key):
            if hasattr(actual_mat, attr):
                try:
                    val = getattr(actual_mat, attr)
                    if val is not None:
                        return float(val)
                except (ValueError, TypeError):
                    pass
        return default

    def get_int(key: str, default: int = 0) -> int:
        if isinstance(params, dict) and key in params:
            try:
                return int(params[key])
            except (ValueError, TypeError):
                pass
        for attr in (key.lower(), key.upper(), key):
            if hasattr(actual_mat, attr):
                try:
                    val = getattr(actual_mat, attr)
                    if val is not None:
                        return int(val)
                except (ValueError, TypeError):
                    pass
        return default

    # 1. Density check (ANCMSG 1514)
    rho0 = get_val("rho0", get_val("rho_initial", get_val("RHO_I", get_val("RHO0", get_val("MAT_RHO", 0.0)))))
    if rho0 <= 0.0:
        actual_log.error(
            f"/MAT/LAW100/{mid}: initial density RHO0 must be > 0, got {rho0} (ANCMSG 1514)",
            "MAT CHECK",
        )

    # 2. Network count check (ANCMSG 1567: max parallel networks is 10)
    n_net = get_int("n_net", get_int("N_net", 0))
    networks = getattr(actual_mat, "networks", []) or []
    if isinstance(params, dict) and "networks" in params and not networks:
        networks = params["networks"]
    effective_n_net = max(n_net, len(networks))
    if effective_n_net > 10:
        actual_log.error(
            f"/MAT/LAW100/{mid}: number of networks N_net exceeds maximum allowed 10, got {effective_n_net} (ANCMSG 1567)",
            "MAT CHECK",
        )

    # 3. Hyperelastic law flag check (ANCMSG 1569: Flag_HE must be valid)
    flag_he = get_int("flag_he", get_int("Flag_HE", 1))
    valid_flag_he = (1, 2, 3, 4, 5, 13)
    if flag_he not in valid_flag_he:
        actual_log.error(
            f"/MAT/LAW100/{mid}: invalid hyperelastic law Flag_HE={flag_he}, must be in {valid_flag_he} (ANCMSG 1569)",
            "MAT CHECK",
        )

    # 4. Hyperelastic formulation parameters
    if flag_he == 1:
        c10 = get_val("c10", get_val("C10", 0.0))
        c01 = get_val("c01", get_val("C01", 0.0))
        g0 = 2.0 * (c10 + c01)
        if g0 <= 0.0:
            actual_log.warning(
                f"/MAT/LAW100/{mid}: initial shear modulus G0 = 2*(C10+C01) should be > 0, got {g0}",
                "MAT CHECK",
            )
        for d_key in ("d1", "d2", "d3"):
            d_val = get_val(d_key, get_val(d_key.upper(), 0.0))
            if d_val < 0.0:
                actual_log.error(
                    f"/MAT/LAW100/{mid}: compressibility parameter {d_key.upper()} must be >= 0, got {d_val}",
                    "MAT CHECK",
                )
    elif flag_he == 2:
        mue1 = get_val("mue1", get_val("mu", get_val("mu1", 0.0)))
        if mue1 <= 0.0:
            actual_log.error(
                f"/MAT/LAW100/{mid}: initial shear modulus mue1 must be > 0, got {mue1}",
                "MAT CHECK",
            )
        d_val = get_val("d", get_val("D", 0.0))
        if d_val < 0.0:
            actual_log.error(
                f"/MAT/LAW100/{mid}: compressibility parameter D must be >= 0, got {d_val}",
                "MAT CHECK",
            )
        lambda_m = get_val("lambda_m", get_val("Lambda_M", 7.0))
        if lambda_m <= 1.0:
            actual_log.error(
                f"/MAT/LAW100/{mid}: limiting chain stretch lambda_m must be > 1.0, got {lambda_m} (ANCMSG 1568)",
                "MAT CHECK",
            )
    elif flag_he in (3, 4, 5):
        c10 = get_val("c10", get_val("C10", 0.0))
        d1 = get_val("d1", get_val("D1", 0.0))
        if c10 <= 0.0:
            actual_log.warning(
                f"/MAT/LAW100/{mid}: hyperelastic stiffness C10 should be > 0, got {c10}",
                "MAT CHECK",
            )
        if d1 < 0.0:
            actual_log.error(
                f"/MAT/LAW100/{mid}: compressibility parameter D1 must be >= 0, got {d1}",
                "MAT CHECK",
            )

    # 5. Plasticity/Creep parameters
    flag_cr = get_int("flag_cr", get_int("Flag_Cr", 0))
    if flag_cr == 1:
        a_pl = get_val("a_pl", get_val("A_pl", 1.0))
        sigma_pl = get_val("sigma_pl", get_val("Sigma_pl", 1.0))
        f_pl = get_val("f_pl", get_val("F_pl", 1.0))
        epsilon_f = get_val("epsilon_f", get_val("epsilon_pl", 1.0))
        if a_pl < 0.0 or sigma_pl < 0.0 or f_pl < 0.0 or epsilon_f <= 0.0:
            actual_log.error(
                f"/MAT/LAW100/{mid}: invalid creep/plasticity parameters (A_pl={a_pl}, Sigma_pl={sigma_pl}, F_pl={f_pl}, eps_f={epsilon_f})",
                "MAT CHECK",
            )

    # 6. Secondary relaxation networks
    for idx, net in enumerate(networks):
        nid = net.get("net_id", net.get("network_id", idx + 1))
        fvisc = net.get("flag_visc", 1)
        if fvisc not in (1, 2, 3):
            actual_log.error(
                f"/MAT/LAW100/{mid}: network {nid} invalid Flag_visc={fvisc}, must be in {{1, 2, 3}} (ANCMSG 1808)",
                "MAT CHECK",
            )
        stiff = net.get("stiffness", 1.0)
        if stiff < 0.0:
            actual_log.error(
                f"/MAT/LAW100/{mid}: network {nid} stiffness factor must be >= 0, got {stiff}",
                "MAT CHECK",
            )
        if fvisc == 1:
            a1 = net.get("a", net.get("a1", 0.0))
            if a1 < 0.0:
                actual_log.error(
                    f"/MAT/LAW100/{mid}: network {nid} creep rate parameter A must be >= 0, got {a1}",
                    "MAT CHECK",
                )
            if a1 > 0.0:
                c_exp = net.get("c", -0.7)
                if c_exp <= -1.0 or c_exp >= 0.0:
                    actual_log.error(
                        f"/MAT/LAW100/{mid}: network {nid} strain exponent C must satisfy -1 < C < 0, got {c_exp}",
                        "MAT CHECK",
                    )
                m_exp = net.get("m", 1.0)
                if m_exp < 1.0:
                    actual_log.error(
                        f"/MAT/LAW100/{mid}: network {nid} stress exponent M must satisfy M >= 1.0, got {m_exp}",
                        "MAT CHECK",
                    )
                ksi = net.get("ksi", 0.01)
                if ksi < 0.0:
                    actual_log.error(
                        f"/MAT/LAW100/{mid}: network {nid} parameter KSI must be >= 0, got {ksi}",
                        "MAT CHECK",
                    )
                tau_ref = net.get("tau_ref", net.get("tauref", 1.0))
                if tau_ref <= 0.0:
                    actual_log.error(
                        f"/MAT/LAW100/{mid}: network {nid} TAU_REF must be > 0, got {tau_ref}",
                        "MAT CHECK",
                    )
        elif fvisc == 2:
            a2 = net.get("a", net.get("a2", 0.0))
            b_val = net.get("b", 0.0)
            n2 = net.get("n", net.get("n2", 1.0))
            if a2 < 0.0 or b_val < 0.0 or n2 <= 0.0:
                actual_log.error(
                    f"/MAT/LAW100/{mid}: network {nid} invalid sinh parameters (A={a2}, B={b_val}, n={n2})",
                    "MAT CHECK",
                )
        elif fvisc == 3:
            a3 = net.get("a", net.get("a3", 0.0))
            n3 = net.get("n", net.get("n3", 1.0))
            m3 = net.get("m", net.get("m3", 0.0))
            if a3 < 0.0 or n3 <= 0.0 or m3 <= -1.0:
                actual_log.error(
                    f"/MAT/LAW100/{mid}: network {nid} invalid power law parameters (A={a3}, n={n3}, M={m3})",
                    "MAT CHECK",
                )

    # 7. Compatible elements check (3D continuum solids only, reject shells ANCMSG 305 and 1D ANCMSG 306)
    if actual_model is not None:
        if hasattr(actual_model, "element_groups") and callable(actual_model.element_groups):
            try:
                grps = list(actual_model.element_groups())
            except TypeError:
                grps = []
            for item in grps:
                if isinstance(item, tuple) and len(item) == 2:
                    name, el_group = item
                else:
                    continue
                mids_in_group = set()
                if hasattr(el_group, "values") and callable(el_group.values):
                    for el in el_group.values():
                        el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                        if el_mid is not None:
                            mids_in_group.add(el_mid)
                if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                    for _, m_part, _ in el_group.state["slices"]:
                        m_id = getattr(m_part, "id", None)
                        if m_id is not None:
                            mids_in_group.add(m_id)
                if mid in mids_in_group:
                    if name in ("trusses", "beams", "springs"):
                        actual_log.error(
                            f"/MAT/LAW100/{mid} is not supported for 1D elements ({name}) (ANCMSG 306)",
                            "MAT CHECK",
                        )
                    elif name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
                        actual_log.error(
                            f"/MAT/LAW100/{mid} is not supported for 2D shell elements ({name}) (ANCMSG 305)",
                            "MAT CHECK",
                        )

        if hasattr(actual_model, "parts") and isinstance(actual_model.parts, dict):
            for pid, part in actual_model.parts.items():
                p_mid = getattr(part, "mat_id", getattr(part, "mid", None))
                if p_mid == mid:
                    etype = str(getattr(part, "elem_type", getattr(part, "type", ""))).upper()
                    prop_id = getattr(part, "prop_id", None)
                    if not etype or etype in ("NONE", ""):
                        if prop_id is not None:
                            props = getattr(actual_model, "properties", {}) or getattr(actual_model, "props", {})
                            prop = props.get(prop_id) if isinstance(props, dict) else None
                            if prop is not None:
                                ptype = str(getattr(prop, "type", getattr(prop, "card_name", ""))).upper()
                                if any(s in ptype for s in ("BEAM", "TRUSS", "SPRING")):
                                    etype = "1D"
                                elif any(s in ptype for s in ("SHELL", "SH3N", "QUAD")):
                                    etype = "2D"
                    if any(s in etype for s in ("BEAM", "TRUSS", "SPRING", "1D")):
                        actual_log.error(
                            f"/MAT/LAW100/{mid} is not supported for 1D elements ({etype.lower()}) (ANCMSG 306)",
                            "MAT CHECK",
                        )
                    elif any(s in etype for s in ("SHELL", "SH3N", "QUAD", "2D")):
                        actual_log.error(
                            f"/MAT/LAW100/{mid} is not supported for 2D shell elements ({etype.lower()}) (ANCMSG 305)",
                            "MAT CHECK",
                        )


_check_mat_law100 = check_mat_law100


def check_mat_law101(*args: Any, **kwargs: Any) -> None:
    """Validate /MAT/LAW101, /MAT/PP, or /MAT/PLAS_POLY parameter bounds and element compatibility (M571).

    Checks:
      1. RHO0 > 0 (ANCMSG 1514).
      2. E_ref > 0 (ANCMSG 1514).
      3. 0 <= nu < 0.5 (ANCMSG 1514).
      4. Compatible elements: 3D solids only (SOLID_ISOTROPIC).
         - Rejects 2D shells: shells, shells_qbat, shells_qeph, sh3n, quads (ANCMSG 305).
         - Rejects 1D elements: trusses, beams, springs (ANCMSG 306).
    """
    actual_mat = None
    actual_log = None
    actual_model = None
    actual_mid = None

    if "model" in kwargs:
        actual_model = kwargs["model"]
    if "mat" in kwargs:
        actual_mat = kwargs["mat"]
    if "log" in kwargs:
        actual_log = kwargs["log"]
    if "mat_id" in kwargs:
        actual_mid = kwargs["mat_id"]

    for c in args:
        if isinstance(c, MessageLog):
            actual_log = c
        elif isinstance(c, Model):
            actual_model = c
        elif hasattr(c, "id") and (hasattr(c, "e") or hasattr(c, "e_ref") or hasattr(c, "nu")):
            actual_mat = c
        elif isinstance(c, int):
            actual_mid = c

    if actual_log is None:
        actual_log = MessageLog()

    mat = actual_mat
    mat_id = actual_mid if actual_mid is not None else (mat.id if mat is not None else 0)

    if mat is None and actual_model is not None and mat_id:
        mat = actual_model.mat_law101s.get(mat_id)
        if mat is None:
            mat = actual_model.materials.get(mat_id)

    if mat is None:
        return

    # Extract density
    rho0 = getattr(mat, "rho0", getattr(mat, "rho", 0.0))
    if isinstance(mat, dict):
        rho0 = mat.get("rho0", mat.get("rho", 0.0))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        rho0 = mat.params.get("rho0", mat.params.get("rho", rho0))
    if rho0 <= 0.0:
        actual_log.error(
            f"MATERIAL {mat_id}: ZERO OR NEGATIVE DENSITY RHO={rho0} (ANCMSG 1514)",
            "MAT CHECK",
        )

    # Extract Young's modulus
    e_val = getattr(mat, "e", getattr(mat, "e_ref", getattr(mat, "E", 0.0)))
    if isinstance(mat, dict):
        e_val = mat.get("e", mat.get("e_ref", mat.get("E", 0.0)))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        e_val = mat.params.get("e", mat.params.get("e_ref", mat.params.get("E", e_val)))
    if e_val <= 0.0:
        actual_log.error(
            f"MATERIAL {mat_id}: ZERO OR NEGATIVE YOUNG'S MODULUS E={e_val} (ANCMSG 1514)",
            "MAT CHECK",
        )

    # Extract Poisson's ratio
    nu_val = getattr(mat, "nu", 0.0)
    if isinstance(mat, dict):
        nu_val = mat.get("nu", 0.0)
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        nu_val = mat.params.get("nu", nu_val)
    if nu_val < 0.0 or nu_val >= 0.5:
        actual_log.error(
            f"MATERIAL {mat_id}: INVALID POISSON'S RATIO NU={nu_val} (MUST BE 0 <= NU < 0.5) (ANCMSG 1514)",
            "MAT CHECK",
        )

    # Check element compatibility: 3D solids only
    if actual_model is not None:
        # Check shell element collections
        shell_colls = [
            getattr(actual_model, "shells", None),
            getattr(actual_model, "shells_qbat", None),
            getattr(actual_model, "shells_qeph", None),
            getattr(actual_model, "sh3n", None),
            getattr(actual_model, "quads", None),
        ]
        shell_found = False
        for coll in shell_colls:
            if not coll:
                continue
            for sh_id, sh in coll.items():
                pid = getattr(sh, "part_id", getattr(sh, "pid", 0))
                part = getattr(actual_model, "parts", {}).get(pid)
                if part and getattr(part, "mat_id", 0) == mat_id:
                    actual_log.error(
                        f"/MAT/LAW101/{mat_id} is not supported for 2D shell elements (ANCMSG 305)",
                        "MAT CHECK",
                    )
                    shell_found = True
                    break
            if shell_found:
                break

        # Check 1D element collections
        oned_colls = [
            getattr(actual_model, "trusses", None),
            getattr(actual_model, "beams", None),
            getattr(actual_model, "springs", None),
        ]
        oned_found = False
        for coll in oned_colls:
            if not coll:
                continue
            for el_id, el in coll.items():
                pid = getattr(el, "part_id", getattr(el, "pid", 0))
                part = getattr(actual_model, "parts", {}).get(pid)
                if part and getattr(part, "mat_id", 0) == mat_id:
                    actual_log.error(
                        f"/MAT/LAW101/{mat_id} is not supported for 1D elements (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    oned_found = True
                    break
            if oned_found:
                break

        # Check part properties
        for pid, part in getattr(actual_model, "parts", {}).items():
            if getattr(part, "mat_id", 0) == mat_id:
                prop_id = getattr(part, "prop_id", 0)
                prop = getattr(actual_model, "props", {}).get(prop_id)
                if prop is not None:
                    ptype = getattr(prop, "type", "").upper()
                    if any(s in ptype for s in ("SHELL", "SH3N", "TYPE1", "TYPE2", "TYPE3", "TYPE4", "TYPE11")):
                        if any(s in ptype for s in ("TYPE2", "TYPE3", "TYPE4", "TYPE11", "BEAM", "TRUSS", "SPRING")):
                            actual_log.error(
                                f"/MAT/LAW101/{mat_id} is not supported for 1D elements ({ptype.lower()}) (ANCMSG 306)",
                                "MAT CHECK",
                            )
                        else:
                            actual_log.error(
                                f"/MAT/LAW101/{mat_id} is not supported for 2D shell elements ({ptype.lower()}) (ANCMSG 305)",
                                "MAT CHECK",
                            )


_check_mat_law101 = check_mat_law101


def check_mat_law102(*args: Any, **kwargs: Any) -> None:
    """Validate /MAT/LAW102 or /MAT/DPRAG2 parameter bounds and element compatibility (M572).

    Checks:
      1. RHO0 > 0 (ANCMSG 1514).
      2. E > 0 (ANCMSG 1514).
      3. 0 <= nu < 0.5 (ANCMSG 1514).
      4. Compatible elements: 3D solids only (SOLID_ISOTROPIC, SPH).
         - Rejects 2D shells: shells, shells_qbat, shells_qeph, sh3n, quads (ANCMSG 305).
         - Rejects 1D elements: trusses, beams, springs (ANCMSG 306).
    """
    actual_mat = None
    actual_log = None
    actual_model = None
    actual_mid = None

    if "model" in kwargs:
        actual_model = kwargs["model"]
    if "mat" in kwargs:
        actual_mat = kwargs["mat"]
    if "log" in kwargs:
        actual_log = kwargs["log"]
    if "mat_id" in kwargs:
        actual_mid = kwargs["mat_id"]

    for c in args:
        if isinstance(c, MessageLog):
            actual_log = c
        elif isinstance(c, Model):
            actual_model = c
        elif hasattr(c, "id") and (hasattr(c, "e") or hasattr(c, "E") or hasattr(c, "nu")):
            actual_mat = c
        elif isinstance(c, int):
            actual_mid = c

    if actual_log is None:
        actual_log = MessageLog()

    mat = actual_mat
    mat_id = actual_mid if actual_mid is not None else (mat.id if mat is not None else 0)

    if mat is None and actual_model is not None and mat_id:
        mat = actual_model.mat_law102s.get(mat_id)
        if mat is None:
            mat = actual_model.materials.get(mat_id)

    if mat is None:
        return

    # Extract density
    rho0 = getattr(mat, "rho0", getattr(mat, "rho", 0.0))
    if isinstance(mat, dict):
        rho0 = mat.get("rho0", mat.get("rho", 0.0))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        rho0 = mat.params.get("rho0", mat.params.get("rho", rho0))
    if rho0 <= 0.0:
        actual_log.error(
            f"MATERIAL {mat_id}: ZERO OR NEGATIVE DENSITY RHO={rho0} (ANCMSG 1514)",
            "MAT CHECK",
        )

    # Extract Young's modulus
    e_val = getattr(mat, "e", getattr(mat, "E", 0.0))
    if isinstance(mat, dict):
        e_val = mat.get("e", mat.get("E", 0.0))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        e_val = mat.params.get("e", mat.params.get("E", e_val))
    if e_val <= 0.0:
        actual_log.error(
            f"MATERIAL {mat_id}: ZERO OR NEGATIVE YOUNG'S MODULUS E={e_val} (ANCMSG 1514)",
            "MAT CHECK",
        )

    # Extract Poisson's ratio
    nu_val = getattr(mat, "nu", 0.0)
    if isinstance(mat, dict):
        nu_val = mat.get("nu", 0.0)
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        nu_val = mat.params.get("nu", nu_val)
    if nu_val < 0.0 or nu_val >= 0.5:
        actual_log.error(
            f"MATERIAL {mat_id}: INVALID POISSON'S RATIO NU={nu_val} (MUST BE 0 <= NU < 0.5) (ANCMSG 1514)",
            "MAT CHECK",
        )

    # Check element compatibility: 3D solids only
    if actual_model is not None:
        # Check shell element collections
        shell_colls = [
            getattr(actual_model, "shells", None),
            getattr(actual_model, "shells_qbat", None),
            getattr(actual_model, "shells_qeph", None),
            getattr(actual_model, "sh3n", None),
            getattr(actual_model, "quads", None),
        ]
        shell_found = False
        for coll in shell_colls:
            if not coll:
                continue
            for sh_id, sh in coll.items():
                pid = getattr(sh, "part_id", getattr(sh, "pid", 0))
                part = getattr(actual_model, "parts", {}).get(pid)
                if part and getattr(part, "mat_id", 0) == mat_id:
                    actual_log.error(
                        f"/MAT/LAW102/{mat_id} is not supported for 2D shell elements (ANCMSG 305)",
                        "MAT CHECK",
                    )
                    shell_found = True
                    break
            if shell_found:
                break

        # Check 1D element collections
        oned_colls = [
            getattr(actual_model, "trusses", None),
            getattr(actual_model, "beams", None),
            getattr(actual_model, "springs", None),
        ]
        oned_found = False
        for coll in oned_colls:
            if not coll:
                continue
            for el_id, el in coll.items():
                pid = getattr(el, "part_id", getattr(el, "pid", 0))
                part = getattr(actual_model, "parts", {}).get(pid)
                if part and getattr(part, "mat_id", 0) == mat_id:
                    actual_log.error(
                        f"/MAT/LAW102/{mat_id} is not supported for 1D elements (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    oned_found = True
                    break
            if oned_found:
                break

        # Check parts with shell property types (PROP/SHELL, PROP/TYPE1)
        for pid, part in getattr(actual_model, "parts", {}).items():
            if getattr(part, "mat_id", 0) == mat_id:
                prop_id = getattr(part, "prop_id", 0)
                prop = getattr(actual_model, "properties", {}).get(prop_id)
                if prop is not None:
                    ptype = str(getattr(prop, "type", getattr(prop, "prop_type", "")) or "").upper()
                    if any(s in ptype for s in ("SHELL", "SH3N", "TYPE1", "TYPE2", "TYPE3", "TYPE4", "TYPE11")):
                        if any(s in ptype for s in ("TYPE2", "TYPE3", "TYPE4", "TYPE11", "BEAM", "TRUSS", "SPRING")):
                            actual_log.error(
                                f"/MAT/LAW102/{mat_id} is not supported for 1D elements ({ptype.lower()}) (ANCMSG 306)",
                                "MAT CHECK",
                            )
                        else:
                            actual_log.error(
                                f"/MAT/LAW102/{mat_id} is not supported for 2D shell elements ({ptype.lower()}) (ANCMSG 305)",
                                "MAT CHECK",
                            )


_check_mat_law102 = check_mat_law102
check_mat_dprag2 = check_mat_law102


def check_mat_law103(*args: Any, **kwargs: Any) -> None:
    """Validate /MAT/LAW103 or /MAT/HENSEL_SPITTEL parameter bounds and element compatibility (M573).

    Checks:
      1. RHO0 > 0 (ANCMSG 1514).
      2. E > 0 (ANCMSG 276).
      3. nu > -1 and nu < 0.5 (ANCMSG 300 / 1514).
      4. Compatible elements: 3D solids only (SOLID_ISOTROPIC, SPH).
         - Rejects 2D shells: shells, shells_qbat, shells_qeph, sh3n, quads (ANCMSG 305).
         - Rejects 1D elements: trusses, beams, springs (ANCMSG 306).
    """
    actual_mat = None
    actual_log = None
    actual_model = None
    actual_mid = None

    if "model" in kwargs:
        actual_model = kwargs["model"]
    if "mat" in kwargs:
        actual_mat = kwargs["mat"]
    if "log" in kwargs:
        actual_log = kwargs["log"]
    if "mat_id" in kwargs:
        actual_mid = kwargs["mat_id"]

    for c in args:
        if isinstance(c, MessageLog):
            actual_log = c
        elif isinstance(c, Model):
            actual_model = c
        elif hasattr(c, "id") and (hasattr(c, "e") or hasattr(c, "E") or hasattr(c, "nu")):
            actual_mat = c
        elif isinstance(c, int):
            actual_mid = c

    if actual_log is None:
        actual_log = MessageLog()

    mat = actual_mat
    mat_id = actual_mid if actual_mid is not None else (mat.id if mat is not None else 0)

    if mat is None and actual_model is not None and mat_id:
        mat = actual_model.mat_law103s.get(mat_id)
        if mat is None:
            mat = actual_model.materials.get(mat_id)

    if mat is None:
        return

    # Extract density
    rho0 = getattr(mat, "rho0", getattr(mat, "rho", 0.0))
    if isinstance(mat, dict):
        rho0 = mat.get("rho0", mat.get("rho", 0.0))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        rho0 = mat.params.get("rho0", mat.params.get("rho", rho0))
    if rho0 <= 0.0:
        actual_log.error(
            f"MATERIAL {mat_id}: ZERO OR NEGATIVE DENSITY RHO={rho0} (ANCMSG 1514)",
            "MAT CHECK",
        )

    # Extract Young's modulus
    e_val = getattr(mat, "e", getattr(mat, "E", 0.0))
    if isinstance(mat, dict):
        e_val = mat.get("e", mat.get("E", 0.0))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        e_val = mat.params.get("e", mat.params.get("E", e_val))
    if e_val <= 0.0:
        actual_log.error(
            f"/MAT/LAW103/{mat_id}: ZERO OR NEGATIVE YOUNG'S MODULUS E={e_val} (ANCMSG 276)",
            "MAT CHECK",
        )

    # Extract Poisson's ratio
    nu_val = getattr(mat, "nu", 0.0)
    if isinstance(mat, dict):
        nu_val = mat.get("nu", 0.0)
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        nu_val = mat.params.get("nu", nu_val)
    if nu_val <= -1.0 or nu_val >= 0.5:
        actual_log.error(
            f"/MAT/LAW103/{mat_id}: INVALID POISSON'S RATIO NU={nu_val} (MUST BE -1 < NU < 0.5) (ANCMSG 300)",
            "MAT CHECK",
        )

    # Check element compatibility: 3D solids only
    if actual_model is not None:
        # Check shell element collections
        shell_colls = [
            getattr(actual_model, "shells", None),
            getattr(actual_model, "shells_qbat", None),
            getattr(actual_model, "shells_qeph", None),
            getattr(actual_model, "sh3n", None),
            getattr(actual_model, "quads", None),
        ]
        shell_found = False
        for coll in shell_colls:
            if not coll:
                continue
            for sh_id, sh in coll.items():
                pid = getattr(sh, "part_id", getattr(sh, "pid", 0))
                part = getattr(actual_model, "parts", {}).get(pid)
                if part and getattr(part, "mat_id", 0) == mat_id:
                    actual_log.error(
                        f"/MAT/LAW103/{mat_id} is not supported for 2D shell elements (ANCMSG 305)",
                        "MAT CHECK",
                    )
                    shell_found = True
                    break
            if shell_found:
                break

        # Check 1D element collections
        oned_colls = [
            getattr(actual_model, "trusses", None),
            getattr(actual_model, "beams", None),
            getattr(actual_model, "springs", None),
        ]
        oned_found = False
        for coll in oned_colls:
            if not coll:
                continue
            for el_id, el in coll.items():
                pid = getattr(el, "part_id", getattr(el, "pid", 0))
                part = getattr(actual_model, "parts", {}).get(pid)
                if part and getattr(part, "mat_id", 0) == mat_id:
                    actual_log.error(
                        f"/MAT/LAW103/{mat_id} is not supported for 1D elements (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    oned_found = True
                    break
            if oned_found:
                break

        # Check parts with shell property types (PROP/SHELL, PROP/TYPE1)
        for pid, part in getattr(actual_model, "parts", {}).items():
            if getattr(part, "mat_id", 0) == mat_id:
                prop_id = getattr(part, "prop_id", 0)
                prop = getattr(actual_model, "properties", {}).get(prop_id)
                if prop is not None:
                    ptype = str(getattr(prop, "type", getattr(prop, "prop_type", "")) or "").upper()
                    if any(s in ptype for s in ("SHELL", "SH3N", "TYPE1", "TYPE2", "TYPE3", "TYPE4", "TYPE11")):
                        if any(s in ptype for s in ("TYPE2", "TYPE3", "TYPE4", "TYPE11", "BEAM", "TRUSS", "SPRING")):
                            actual_log.error(
                                f"/MAT/LAW103/{mat_id} is not supported for 1D elements ({ptype.lower()}) (ANCMSG 306)",
                                "MAT CHECK",
                            )
                        else:
                            actual_log.error(
                                f"/MAT/LAW103/{mat_id} is not supported for 2D shell elements ({ptype.lower()}) (ANCMSG 305)",
                                "MAT CHECK",
                            )


_check_mat_law103 = check_mat_law103
check_mat_hensel_spittel = check_mat_law103
check_mat_plas_hens = check_mat_law103


def check_mat_law104(*args: Any, **kwargs: Any) -> None:
    """Validate /MAT/LAW104 or /MAT/DRUCKER parameter bounds and element compatibility (M574).

    Checks:
      1. RHO0 > 0 (ANCMSG 1514).
      2. E > 0 (ANCMSG 276 / 1514).
      3. nu > -1 and nu < 0.5 (ANCMSG 300 / 1514).
      4. Drucker coefficient Cdr convexity:
         - Cdr <= 2.25 (ANCMSG 1651 warning).
         - Cdr >= -3.375 (-27/8) (ANCMSG 1652 warning).
      5. Return mapping method Ires (ANCMSG 1731 warning if Ires > 2).
      6. Self-heating bounds: DPIS <= DPAD (ANCMSG 1655 error if DPIS > DPAD).
      7. Compatible elements: 3D solids and 2D shells (SOLID_ISOTROPIC, SHELL_ISOTROPIC).
         - Rejects 1D elements: trusses, beams, springs (ANCMSG 306).
    """
    actual_mat = None
    actual_log = None
    actual_model = None
    actual_mid = None

    if "model" in kwargs:
        actual_model = kwargs["model"]
    if "mat" in kwargs:
        actual_mat = kwargs["mat"]
    if "log" in kwargs:
        actual_log = kwargs["log"]
    if "mat_id" in kwargs:
        actual_mid = kwargs["mat_id"]

    for c in args:
        if isinstance(c, MessageLog):
            actual_log = c
        elif isinstance(c, Model):
            actual_model = c
        elif hasattr(c, "id") and (hasattr(c, "e") or hasattr(c, "young") or hasattr(c, "nu")):
            actual_mat = c
        elif isinstance(c, int):
            actual_mid = c

    if actual_log is None:
        actual_log = MessageLog()

    mat = actual_mat
    mat_id = actual_mid if actual_mid is not None else (mat.id if mat is not None else 0)

    if mat is None and actual_model is not None and mat_id:
        mat = actual_model.mat_law104s.get(mat_id)
        if mat is None:
            mat = actual_model.materials.get(mat_id)

    if mat is None:
        return

    # Extract density
    rho0 = getattr(mat, "rho0", getattr(mat, "rho", 0.0))
    if isinstance(mat, dict):
        rho0 = mat.get("rho0", mat.get("rho", 0.0))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        rho0 = mat.params.get("rho0", mat.params.get("rho", rho0))
    if rho0 <= 0.0:
        actual_log.error(
            f"MATERIAL {mat_id}: ZERO OR NEGATIVE DENSITY RHO={rho0} (ANCMSG 1514)",
            "MAT CHECK",
        )

    # Extract Young's modulus
    e_val = getattr(mat, "young", getattr(mat, "e", getattr(mat, "E", 0.0)))
    if isinstance(mat, dict):
        e_val = mat.get("young", mat.get("e", mat.get("E", 0.0)))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        e_val = mat.params.get("young", mat.params.get("e", mat.params.get("E", e_val)))
    if e_val <= 0.0:
        actual_log.error(
            f"/MAT/LAW104/{mat_id}: ZERO OR NEGATIVE YOUNG'S MODULUS E={e_val} (ANCMSG 276)",
            "MAT CHECK",
        )

    # Extract Poisson's ratio
    nu_val = getattr(mat, "nu", 0.0)
    if isinstance(mat, dict):
        nu_val = mat.get("nu", 0.0)
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        nu_val = mat.params.get("nu", nu_val)
    if nu_val <= -1.0 or nu_val >= 0.5:
        actual_log.error(
            f"/MAT/LAW104/{mat_id}: INVALID POISSON'S RATIO NU={nu_val} (MUST BE -1 < NU < 0.5) (ANCMSG 300)",
            "MAT CHECK",
        )

    # Extract Drucker coefficient Cdr
    c_dr = getattr(mat, "c_dr", getattr(mat, "cdr", 0.0))
    if isinstance(mat, dict):
        c_dr = mat.get("c_dr", mat.get("cdr", 0.0))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        c_dr = mat.params.get("c_dr", mat.params.get("cdr", c_dr))
    if c_dr > 2.25:
        actual_log.warning(
            f"/MAT/LAW104/{mat_id}: DRUCKER COEFF CDR={c_dr} EXCEEDS 2.25, CLAMPING TO 2.25 (ANCMSG 1651)",
            "MAT CHECK",
        )
    elif c_dr < -3.375:
        actual_log.warning(
            f"/MAT/LAW104/{mat_id}: DRUCKER COEFF CDR={c_dr} LESS THAN -3.375, CLAMPING TO -3.375 (ANCMSG 1652)",
            "MAT CHECK",
        )

    # Extract return mapping method Ires
    ires = getattr(mat, "ires", 1)
    if isinstance(mat, dict):
        ires = mat.get("ires", 1)
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        ires = mat.params.get("ires", ires)
    if ires > 2:
        actual_log.warning(
            f"/MAT/LAW104/{mat_id}: INVALID RESOLUTION METHOD IRES={ires} (MUST BE 1 OR 2) (ANCMSG 1731)",
            "MAT CHECK",
        )

    # Extract self-heating rates
    eps_iso = getattr(mat, "eps_iso", getattr(mat, "dpis", 1.0e30))
    eps_ad = getattr(mat, "eps_ad", getattr(mat, "dpad", 2.0e30))
    if isinstance(mat, dict):
        eps_iso = mat.get("eps_iso", mat.get("dpis", 1.0e30))
        eps_ad = mat.get("eps_ad", mat.get("dpad", 2.0e30))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        eps_iso = mat.params.get("eps_iso", mat.params.get("dpis", eps_iso))
        eps_ad = mat.params.get("eps_ad", mat.params.get("dpad", eps_ad))
    if eps_iso > eps_ad:
        actual_log.error(
            f"/MAT/LAW104/{mat_id}: ISOTHERMAL RATE EPS_ISO={eps_iso} EXCEEDS ADIABATIC RATE EPS_AD={eps_ad} (ANCMSG 1655)",
            "MAT CHECK",
        )

    # Check element compatibility: 3D solids and 2D shells supported, 1D elements rejected (ANCMSG 306)
    if actual_model is not None:
        # Check 1D element collections
        oned_colls = [
            getattr(actual_model, "trusses", None),
            getattr(actual_model, "beams", None),
            getattr(actual_model, "springs", None),
        ]
        oned_found = False
        for coll in oned_colls:
            if not coll:
                continue
            for el_id, el in coll.items():
                pid = getattr(el, "part_id", getattr(el, "pid", 0))
                part = getattr(actual_model, "parts", {}).get(pid)
                if part and getattr(part, "mat_id", 0) == mat_id:
                    actual_log.error(
                        f"/MAT/LAW104/{mat_id} is not supported for 1D elements (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    oned_found = True
                    break
            if oned_found:
                break

        # Check parts with 1D property types
        for pid, part in getattr(actual_model, "parts", {}).items():
            if getattr(part, "mat_id", 0) == mat_id:
                prop_id = getattr(part, "prop_id", 0)
                prop = getattr(actual_model, "properties", {}).get(prop_id)
                if prop is not None:
                    ptype = str(getattr(prop, "type", getattr(prop, "prop_type", "")) or "").upper()
                    if any(s in ptype for s in ("TYPE2", "TYPE3", "TYPE4", "TYPE11", "BEAM", "TRUSS", "SPRING")):
                        actual_log.error(
                            f"/MAT/LAW104/{mat_id} is not supported for 1D elements ({ptype.lower()}) (ANCMSG 306)",
                            "MAT CHECK",
                        )


_check_mat_law104 = check_mat_law104
check_mat_drucker = check_mat_law104
check_mat_johns_voce_drucker = check_mat_law104
check_mat_plas_druck = check_mat_law104


def check_mat_law105(*args: Any, **kwargs: Any) -> None:
    """Validate /MAT/LAW105 or /MAT/POWDER_BURN parameter bounds and element compatibility (M576).

    Upstream reference: ``starter/source/materials/mat/mat105/hm_read_mat105.F90``.

    Checks:
      1. RHO0 > 0 (ANCMSG 1514).
      2. BULK > 0 (ANCMSG 856, hm_read_mat105.F90:163: 'BULK MODULUS MUST BE DEFINED').
      3. D > 0 (ANCMSG 856, hm_read_mat105.F90:168: 'GAS EOS PARAMETER D MUST BE DEFINED').
      4. EG > 0 (ANCMSG 856, hm_read_mat105.F90:173: 'GAS EOS PARAMETER EG MUST BE DEFINED').
      5. Gr > 0 (ANCMSG 856, hm_read_mat105.F90:178: 'GROWTH PARAMETER Gr MUST BE DEFINED').
      6. C1 > 0 (ANCMSG 856, hm_read_mat105.F90:183: 'BURNING VELOCITY C1 MUST BE DEFINED').
      7. Element compatibility: 3D solids and SPH supported; reject 2D shells (ANCMSG 305) and 1D elements (ANCMSG 306).
    """
    actual_mat = None
    actual_log = None
    actual_model = None
    actual_mid = None

    if "model" in kwargs:
        actual_model = kwargs["model"]
    if "mat" in kwargs:
        actual_mat = kwargs["mat"]
    if "log" in kwargs:
        actual_log = kwargs["log"]
    if "mat_id" in kwargs:
        actual_mid = kwargs["mat_id"]

    for c in args:
        if isinstance(c, MessageLog):
            actual_log = c
        elif isinstance(c, Model):
            actual_model = c
        elif hasattr(c, "id") and (hasattr(c, "bulk") or hasattr(c, "gas_d") or hasattr(c, "gr")):
            actual_mat = c
        elif isinstance(c, int):
            actual_mid = c

    if actual_log is None:
        actual_log = MessageLog()

    mat = actual_mat
    mat_id = actual_mid if actual_mid is not None else (mat.id if mat is not None else 0)

    if mat is None and actual_model is not None and mat_id:
        mat = getattr(actual_model, "mat_law105s", {}).get(mat_id)
        if mat is None:
            mat = getattr(actual_model, "materials", {}).get(mat_id)

    if mat is None:
        return

    titr = getattr(mat, "title", f"/MAT/LAW105/{mat_id}")

    # 1. Density check
    rho0 = getattr(mat, "rho0", getattr(mat, "rho", 0.0))
    if isinstance(mat, dict):
        rho0 = mat.get("rho0", mat.get("rho", 0.0))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        rho0 = mat.params.get("rho0", mat.params.get("rho", mat.params.get("MAT_RHO", rho0)))
    if rho0 <= 0.0:
        actual_log.error(
            f"MATERIAL {mat_id}: ZERO OR NEGATIVE DENSITY RHO={rho0} (ANCMSG 1514)",
            "MAT CHECK",
        )

    # 2. Bulk modulus check (ANCMSG 856)
    bulk = getattr(mat, "bulk", 0.0)
    if isinstance(mat, dict):
        bulk = mat.get("bulk", mat.get("POWDER_BULK", 0.0))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        bulk = mat.params.get("bulk", mat.params.get("POWDER_BULK", bulk))
    if bulk <= 0.0:
        actual_log.error(
            f"/MAT/LAW105/{mat_id}: BULK MODULUS MUST BE DEFINED (ANCMSG 856)",
            "MAT CHECK",
        )

    # 3. Gas EOS parameter D check (ANCMSG 856)
    gas_d = getattr(mat, "gas_d", getattr(mat, "d", 0.0))
    if isinstance(mat, dict):
        gas_d = mat.get("gas_d", mat.get("d", mat.get("GAS_D", 0.0)))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        gas_d = mat.params.get("gas_d", mat.params.get("d", mat.params.get("GAS_D", gas_d)))
    if gas_d <= 0.0:
        actual_log.error(
            f"/MAT/LAW105/{mat_id}: GAS EOS PARAMETER D MUST BE DEFINED (ANCMSG 856)",
            "MAT CHECK",
        )

    # 4. Gas EOS parameter EG check (ANCMSG 856)
    gas_eg = getattr(mat, "gas_eg", getattr(mat, "eg", 0.0))
    if isinstance(mat, dict):
        gas_eg = mat.get("gas_eg", mat.get("eg", mat.get("GAS_EG", 0.0)))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        gas_eg = mat.params.get("gas_eg", mat.params.get("eg", mat.params.get("GAS_EG", gas_eg)))
    if gas_eg <= 0.0:
        actual_log.error(
            f"/MAT/LAW105/{mat_id}: GAS EOS PARAMETER EG MUST BE DEFINED (ANCMSG 856)",
            "MAT CHECK",
        )

    # 5. Grain growth parameter Gr check (ANCMSG 856)
    gr = getattr(mat, "gr", 0.0)
    if isinstance(mat, dict):
        gr = mat.get("gr", mat.get("POWDER_Gr", 0.0))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        gr = mat.params.get("gr", mat.params.get("POWDER_Gr", gr))
    if gr <= 0.0:
        actual_log.error(
            f"/MAT/LAW105/{mat_id}: GROWTH PARAMETER Gr MUST BE DEFINED (ANCMSG 856)",
            "MAT CHECK",
        )

    # 6. Burn velocity parameter C1 check (ANCMSG 856)
    c1 = getattr(mat, "c1", 0.0)
    if isinstance(mat, dict):
        c1 = mat.get("c1", mat.get("MAT_C1", 0.0))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        c1 = mat.params.get("c1", mat.params.get("MAT_C1", c1))
    if c1 <= 0.0:
        actual_log.error(
            f"/MAT/LAW105/{mat_id}: BURNING VELOCITY C1 MUST BE DEFINED (ANCMSG 856)",
            "MAT CHECK",
        )

    # 7. Element compatibility: reject 2D shells (ANCMSG 305) and 1D elements (ANCMSG 306)
    if actual_model is not None:
        # Check shells
        shell_colls = [
            getattr(actual_model, "shells", None),
            getattr(actual_model, "quads", None),
            getattr(actual_model, "trias", None),
        ]
        shell_found = False
        for coll in shell_colls:
            if not coll:
                continue
            for el_id, el in coll.items():
                pid = getattr(el, "part_id", getattr(el, "pid", 0))
                part = getattr(actual_model, "parts", {}).get(pid)
                if part and getattr(part, "mat_id", 0) == mat_id:
                    actual_log.error(
                        f"/MAT/LAW105/{mat_id} is not supported for shell elements (ANCMSG 305)",
                        "MAT CHECK",
                    )
                    shell_found = True
                    break
            if shell_found:
                break

        # Check 1D elements
        oned_colls = [
            getattr(actual_model, "trusses", None),
            getattr(actual_model, "beams", None),
            getattr(actual_model, "springs", None),
        ]
        oned_found = False
        for coll in oned_colls:
            if not coll:
                continue
            for el_id, el in coll.items():
                pid = getattr(el, "part_id", getattr(el, "pid", 0))
                part = getattr(actual_model, "parts", {}).get(pid)
                if part and getattr(part, "mat_id", 0) == mat_id:
                    actual_log.error(
                        f"/MAT/LAW105/{mat_id} is not supported for 1D elements (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    oned_found = True
                    break
            if oned_found:
                break

        for pid, part in getattr(actual_model, "parts", {}).items():
            if getattr(part, "mat_id", 0) == mat_id:
                etype = str(getattr(part, "element_type", "") or "").upper()
                if any(s in etype for s in ("SHELL", "QUAD", "TRIA")):
                    actual_log.error(
                        f"/MAT/LAW105/{mat_id} is not supported for shell elements ({etype.lower()}) (ANCMSG 305)",
                        "MAT CHECK",
                    )
                    break
                if any(s in etype for s in ("BEAM", "TRUSS", "SPRING")):
                    actual_log.error(
                        f"/MAT/LAW105/{mat_id} is not supported for 1D elements ({etype.lower()}) (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    break
                prop_id = getattr(part, "prop_id", 0)
                prop = getattr(actual_model, "properties", {}).get(prop_id)
                if prop is not None:
                    ptype = str(getattr(prop, "type", getattr(prop, "prop_type", "")) or "").upper()
                    if ptype == "TYPE1" or any(s in ptype for s in ("SHELL", "SH_SEAT")):
                        actual_log.error(
                            f"/MAT/LAW105/{mat_id} is not supported for shell elements ({ptype.lower()}) (ANCMSG 305)",
                            "MAT CHECK",
                        )
                    elif ptype in ("TYPE2", "TYPE3", "TYPE4", "TYPE11") or any(s in ptype for s in ("BEAM", "TRUSS", "SPRING", "SPR_SEAT")):
                        actual_log.error(
                            f"/MAT/LAW105/{mat_id} is not supported for 1D elements ({ptype.lower()}) (ANCMSG 306)",
                            "MAT CHECK",
                        )


_check_mat_law105 = check_mat_law105
check_mat_powder_burn = check_mat_law105
check_mat_powderburn = check_mat_law105


def check_mat_law106(*args: Any, **kwargs: Any) -> None:
    """Validate /MAT/LAW106 or /MAT/JCOOK_ALM parameter bounds and element compatibility (M575).

    Checks:
      1. RHO0 > 0 (ANCMSG 1514).
      2. E > 0 (ANCMSG 276 / 1514).
      3. nu > -1 and nu <= 0.495 (ANCMSG 300 / 1514, hm_read_mat106.F90:171).
      4. Thermal bounds: Tmelt > 0, T0 <= Tmelt.
      5. Compatible elements: 3D solids and 2D shells.
         - Rejects 1D elements: trusses, beams, springs (ANCMSG 306).
    """
    actual_mat = None
    actual_log = None
    actual_model = None
    actual_mid = None

    if "model" in kwargs:
        actual_model = kwargs["model"]
    if "mat" in kwargs:
        actual_mat = kwargs["mat"]
    if "log" in kwargs:
        actual_log = kwargs["log"]
    if "mat_id" in kwargs:
        actual_mid = kwargs["mat_id"]

    for c in args:
        if isinstance(c, MessageLog):
            actual_log = c
        elif isinstance(c, Model):
            actual_model = c
        elif hasattr(c, "id") and (hasattr(c, "e") or hasattr(c, "young") or hasattr(c, "nu")):
            actual_mat = c
        elif isinstance(c, int):
            actual_mid = c

    if actual_log is None:
        actual_log = MessageLog()

    mat = actual_mat
    mat_id = actual_mid if actual_mid is not None else (mat.id if mat is not None else 0)

    if mat is None and actual_model is not None and mat_id:
        mat = actual_model.mat_law106s.get(mat_id)
        if mat is None:
            mat = actual_model.materials.get(mat_id)

    if mat is None:
        return

    # Extract density
    rho0 = getattr(mat, "rho0", getattr(mat, "rho", 0.0))
    if isinstance(mat, dict):
        rho0 = mat.get("rho0", mat.get("rho", 0.0))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        rho0 = mat.params.get("rho0", mat.params.get("rho", rho0))
    if rho0 <= 0.0:
        actual_log.error(
            f"MATERIAL {mat_id}: ZERO OR NEGATIVE DENSITY RHO={rho0} (ANCMSG 1514)",
            "MAT CHECK",
        )

    # Extract Young's modulus
    e_val = getattr(mat, "young", getattr(mat, "e", getattr(mat, "E", 0.0)))
    if isinstance(mat, dict):
        e_val = mat.get("young", mat.get("e", mat.get("E", 0.0)))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        e_val = mat.params.get("young", mat.params.get("e", mat.params.get("E", e_val)))
    if e_val <= 0.0:
        actual_log.error(
            f"/MAT/LAW106/{mat_id}: ZERO OR NEGATIVE YOUNG'S MODULUS E={e_val} (ANCMSG 276)",
            "MAT CHECK",
        )

    # Extract Poisson's ratio
    nu_val = getattr(mat, "nu", 0.0)
    if isinstance(mat, dict):
        nu_val = mat.get("nu", 0.0)
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        nu_val = mat.params.get("nu", nu_val)
    if nu_val <= -1.0 or nu_val > 0.495:
        actual_log.error(
            f"/MAT/LAW106/{mat_id}: INVALID POISSON'S RATIO NU={nu_val} (MUST BE -1 < NU <= 0.495) (ANCMSG 300)",
            "MAT CHECK",
        )

    # Thermal checks
    tmelt = getattr(mat, "tmelt", 1e30)
    t0 = getattr(mat, "t0", 300.0)
    if isinstance(mat, dict):
        tmelt = mat.get("tmelt", 1e30)
        t0 = mat.get("t0", 300.0)
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        tmelt = mat.params.get("tmelt", tmelt)
        t0 = mat.params.get("t0", t0)
    if tmelt <= 0.0:
        actual_log.warning(
            f"/MAT/LAW106/{mat_id}: TMELT={tmelt} <= 0, defaulting to 1e30",
            "MAT CHECK",
        )
    if t0 > tmelt:
        actual_log.warning(
            f"/MAT/LAW106/{mat_id}: INITIAL TEMPERATURE T0={t0} EXCEEDS MELTING TEMPERATURE TMELT={tmelt}",
            "MAT CHECK",
        )

    # Check element compatibility: 3D solids and 2D shells supported, 1D elements rejected (ANCMSG 306)
    if actual_model is not None:
        oned_colls = [
            getattr(actual_model, "trusses", None),
            getattr(actual_model, "beams", None),
            getattr(actual_model, "springs", None),
        ]
        oned_found = False
        for coll in oned_colls:
            if not coll:
                continue
            for el_id, el in coll.items():
                pid = getattr(el, "part_id", getattr(el, "pid", 0))
                part = getattr(actual_model, "parts", {}).get(pid)
                if part and getattr(part, "mat_id", 0) == mat_id:
                    actual_log.error(
                        f"/MAT/LAW106/{mat_id} is not supported for 1D elements (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    oned_found = True
                    break
            if oned_found:
                break

        for pid, part in getattr(actual_model, "parts", {}).items():
            if getattr(part, "mat_id", 0) == mat_id:
                etype = str(getattr(part, "element_type", "") or "").upper()
                if any(s in etype for s in ("BEAM", "TRUSS", "SPRING")):
                    actual_log.error(
                        f"/MAT/LAW106/{mat_id} is not supported for 1D elements ({etype.lower()}) (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    break
                prop_id = getattr(part, "prop_id", 0)
                prop = getattr(actual_model, "properties", {}).get(prop_id)
                if prop is not None:
                    ptype = str(getattr(prop, "type", getattr(prop, "prop_type", "")) or "").upper()
                    if any(s in ptype for s in ("TYPE2", "TYPE3", "TYPE4", "TYPE11", "BEAM", "TRUSS", "SPRING")):
                        actual_log.error(
                            f"/MAT/LAW106/{mat_id} is not supported for 1D elements ({ptype.lower()}) (ANCMSG 306)",
                            "MAT CHECK",
                        )


_check_mat_law106 = check_mat_law106
check_mat_jcook_alm = check_mat_law106
check_mat_johns_cook_alm = check_mat_law106


def check_mat_law107(*args: Any, **kwargs: Any) -> None:
    """Validate /MAT/LAW107 or /MAT/PAPER_LIGHT parameter bounds and element compatibility (M577).

    Upstream reference: ``starter/source/materials/mat/mat107/hm_read_mat107.F``.

    Checks:
      1. RHO0 > 0 (ANCMSG 1514).
      2. E1 > 0, E2 > 0, E3 > 0 (ANCMSG 1514).
      3. NU12 * NU21 < 1.0 (ANCMSG 3068).
      4. Restriction for positive plastic work (K parameters):
         - K1 >= 0 (ANCMSG 1815)
         - K2 <= 0 (ANCMSG 1816)
         - K3 >= 0 (ANCMSG 1817)
         - K4 >= 0 (ANCMSG 1818)
         - K5 >= 0 (ANCMSG 1819)
         - K6 >= 0 (ANCMSG 1820)
         - |K1| > |K2| (ANCMSG 1821)
         - |K2| > |K4| (ANCMSG 1822)
      5. Compatible elements: 3D solids and 2D shells.
         - Rejection of 1D elements: trusses, beams, springs (ANCMSG 306).
    """
    actual_mat = None
    actual_log = None
    actual_model = None
    actual_mid = None

    if "model" in kwargs:
        actual_model = kwargs["model"]
    if "mat" in kwargs:
        actual_mat = kwargs["mat"]
    if "log" in kwargs:
        actual_log = kwargs["log"]
    if "mat_id" in kwargs:
        actual_mid = kwargs["mat_id"]

    for c in args:
        if isinstance(c, MessageLog):
            actual_log = c
        elif isinstance(c, Model):
            actual_model = c
        elif hasattr(c, "id") and (hasattr(c, "young1") or hasattr(c, "e1") or hasattr(c, "k1")):
            actual_mat = c
        elif isinstance(c, int):
            actual_mid = c

    if actual_log is None:
        actual_log = MessageLog()

    mat = actual_mat
    mat_id = actual_mid if actual_mid is not None else (mat.id if mat is not None else 0)

    if mat is None and actual_model is not None and mat_id:
        mat = getattr(actual_model, "mat_law107s", {}).get(mat_id)
        if mat is None:
            mat = getattr(actual_model, "materials", {}).get(mat_id)

    if mat is None:
        return

    titr = getattr(mat, "title", f"/MAT/LAW107/{mat_id}")

    # 1. Density check (ANCMSG 1514)
    rho0 = getattr(mat, "rho0", getattr(mat, "rho", 0.0))
    if isinstance(mat, dict):
        rho0 = mat.get("rho0", mat.get("rho", 0.0))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        rho0 = mat.params.get("rho0", mat.params.get("rho", mat.params.get("MAT_RHO", rho0)))
    if rho0 <= 0.0:
        actual_log.error(
            f"MATERIAL {mat_id}: ZERO OR NEGATIVE DENSITY RHO={rho0} (ANCMSG 1514)",
            "MAT CHECK",
        )

    # 2. Young's moduli checks (ANCMSG 1514)
    young1 = getattr(mat, "young1", getattr(mat, "e1", 0.0))
    young2 = getattr(mat, "young2", getattr(mat, "e2", 0.0))
    young3 = getattr(mat, "young3", getattr(mat, "e3", 0.0))
    if isinstance(mat, dict):
        young1 = mat.get("young1", mat.get("e1", mat.get("MAT_E1", 0.0)))
        young2 = mat.get("young2", mat.get("e2", mat.get("MAT_E2", 0.0)))
        young3 = mat.get("young3", mat.get("e3", mat.get("MAT_E3", 0.0)))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        young1 = mat.params.get("young1", mat.params.get("e1", mat.params.get("MAT_E1", young1)))
        young2 = mat.params.get("young2", mat.params.get("e2", mat.params.get("MAT_E2", young2)))
        young3 = mat.params.get("young3", mat.params.get("e3", mat.params.get("MAT_E3", young3)))

    if young1 <= 0.0 or young2 <= 0.0 or young3 <= 0.0:
        actual_log.error(
            f"/MAT/LAW107/{mat_id}: ELASTIC MODULUS MUST BE POSITIVE (E1={young1}, E2={young2}, E3={young3}) (ANCMSG 1514)",
            "MAT CHECK",
        )

    # 3. Poisson check (ANCMSG 3068)
    nu21 = getattr(mat, "nu21", 0.0)
    if isinstance(mat, dict):
        nu21 = mat.get("nu21", mat.get("MAT_NU21", 0.0))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        nu21 = mat.params.get("nu21", mat.params.get("MAT_NU21", nu21))

    nu12 = (nu21 * young1 / young2) if young2 > 0.0 else 0.0
    if nu12 * nu21 >= 1.0:
        actual_log.error(
            f"/MAT/LAW107/{mat_id}: POISSON RATIO PRODUCT NU12*NU21 >= 1 (ANCMSG 3068)",
            "MAT CHECK",
        )

    # 4. Plastic potential parameter checks (ANCMSG 1815-1822)
    def _g_p(name: str, fallback: float) -> float:
        val = getattr(mat, name, fallback)
        if isinstance(mat, dict):
            val = mat.get(name, mat.get(f"MAT_{name.upper()}", fallback))
        elif hasattr(mat, "params") and isinstance(mat.params, dict):
            val = mat.params.get(name, mat.params.get(f"MAT_{name.upper()}", val))
        return float(val)

    k1 = _g_p("k1", 0.0)
    k2 = _g_p("k2", 0.0)
    k3 = _g_p("k3", 0.0)
    k4 = _g_p("k4", 0.0)
    k5 = _g_p("k5", 0.0)
    k6 = _g_p("k6", 0.0)

    if k1 < 0.0:
        actual_log.error(f"/MAT/LAW107/{mat_id}: K1 < 0 (ANCMSG 1815)", "MAT CHECK")
    if k2 > 0.0:
        actual_log.error(f"/MAT/LAW107/{mat_id}: K2 > 0 (ANCMSG 1816)", "MAT CHECK")
    if k3 < 0.0:
        actual_log.error(f"/MAT/LAW107/{mat_id}: K3 < 0 (ANCMSG 1817)", "MAT CHECK")
    if k4 < 0.0:
        actual_log.error(f"/MAT/LAW107/{mat_id}: K4 < 0 (ANCMSG 1818)", "MAT CHECK")
    if k5 < 0.0:
        actual_log.error(f"/MAT/LAW107/{mat_id}: K5 < 0 (ANCMSG 1819)", "MAT CHECK")
    if k6 < 0.0:
        actual_log.error(f"/MAT/LAW107/{mat_id}: K6 < 0 (ANCMSG 1820)", "MAT CHECK")
    if abs(k1) <= abs(k2):
        actual_log.error(f"/MAT/LAW107/{mat_id}: ABS(K1) <= ABS(K2) (ANCMSG 1821)", "MAT CHECK")
    if abs(k2) <= abs(k4):
        actual_log.error(f"/MAT/LAW107/{mat_id}: ABS(K2) <= ABS(K4) (ANCMSG 1822)", "MAT CHECK")

    # 5. Element compatibility check: 3D solids and 2D shells supported, 1D elements rejected (ANCMSG 306)
    if actual_model is not None:
        oned_colls = [
            getattr(actual_model, "trusses", None),
            getattr(actual_model, "beams", None),
            getattr(actual_model, "springs", None),
        ]
        oned_found = False
        for coll in oned_colls:
            if not coll:
                continue
            for el_id, el in coll.items():
                pid = getattr(el, "part_id", getattr(el, "pid", 0))
                part = getattr(actual_model, "parts", {}).get(pid)
                if part and getattr(part, "mat_id", 0) == mat_id:
                    actual_log.error(
                        f"/MAT/LAW107/{mat_id} is not supported for 1D elements (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    oned_found = True
                    break
            if oned_found:
                break

        for pid, part in getattr(actual_model, "parts", {}).items():
            if getattr(part, "mat_id", 0) == mat_id:
                etype = str(getattr(part, "element_type", "") or "").upper()
                if any(s in etype for s in ("BEAM", "TRUSS", "SPRING")):
                    actual_log.error(
                        f"/MAT/LAW107/{mat_id} is not supported for 1D elements ({etype.lower()}) (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    break
                prop_id = getattr(part, "prop_id", 0)
                prop = getattr(actual_model, "properties", {}).get(prop_id)
                if prop is not None:
                    ptype = str(getattr(prop, "type", getattr(prop, "prop_type", "")) or "").upper()
                    if any(s in ptype for s in ("TYPE2", "TYPE3", "TYPE4", "TYPE11", "BEAM", "TRUSS", "SPRING")):
                        actual_log.error(
                            f"/MAT/LAW107/{mat_id} is not supported for 1D elements ({ptype.lower()}) (ANCMSG 306)",
                            "MAT CHECK",
                        )


_check_mat_law107 = check_mat_law107
check_mat_paper_light = check_mat_law107
check_mat_plas_paper_light = check_mat_law107
check_mat_pfeiffer = check_mat_law107


def check_mat_law109(*args: Any, **kwargs: Any) -> None:
    """Validate /MAT/LAW109 or /MAT/TAB_PLAS parameter bounds and element compatibility (M578).

    Upstream reference: ``starter/source/materials/mat/mat109/hm_read_mat109.F`` / ``mat109.cfg``.

    Checks:
      1. RHO_I > 0 (ANCMSG 1514).
      2. E > 0 (ANCMSG 1514).
      3. 0.0 <= NU < 0.5 (ANCMSG 3068 / Poisson bounds).
      4. If CP > 0: 0.0 <= ETA <= 1.0.
      5. YSCALE > 0.
      6. Compatible elements: 3D solids and 2D shells. Rejects 1D elements (ANCMSG 306).
    """
    actual_mat = None
    actual_log = None
    actual_model = None
    actual_mid = None

    if "model" in kwargs:
        actual_model = kwargs["model"]
    if "mat" in kwargs:
        actual_mat = kwargs["mat"]
    if "log" in kwargs:
        actual_log = kwargs["log"]
    if "mat_id" in kwargs:
        actual_mid = kwargs["mat_id"]

    for c in args:
        if isinstance(c, MessageLog):
            actual_log = c
        elif isinstance(c, Model):
            actual_model = c
        elif hasattr(c, "id") and (hasattr(c, "rho_i") or hasattr(c, "tab_id_h") or hasattr(c, "tab_yld") or hasattr(c, "c_p")):
            actual_mat = c
        elif isinstance(c, int):
            actual_mid = c

    if actual_log is None:
        actual_log = MessageLog()

    mat = actual_mat
    mat_id = actual_mid if actual_mid is not None else (mat.id if mat is not None else 0)

    if mat is None and actual_model is not None and mat_id:
        mat = getattr(actual_model, "mat_law109s", {}).get(mat_id)
        if mat is None:
            mat = getattr(actual_model, "materials", {}).get(mat_id)

    if mat is None:
        return

    titr = getattr(mat, "title", f"/MAT/LAW109/{mat_id}")

    # 1. Density check (ANCMSG 1514)
    rho0 = getattr(mat, "rho_i", getattr(mat, "rho0", getattr(mat, "rho", 0.0)))
    if isinstance(mat, dict):
        rho0 = mat.get("rho_i", mat.get("rho0", mat.get("rho", 0.0)))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        rho0 = mat.params.get("MAT_RHO", mat.params.get("rho0", rho0))
    if rho0 <= 0.0:
        actual_log.error(
            f"/MAT/LAW109/{mat_id}: initial density RHO_I must be strictly positive (ANCMSG 1514)",
            "MAT CHECK",
        )

    # 2. Young's modulus check (ANCMSG 1514)
    young = getattr(mat, "e", getattr(mat, "young", getattr(mat, "E", 0.0)))
    if isinstance(mat, dict):
        young = mat.get("e", mat.get("young", 0.0))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        young = mat.params.get("MAT_E", mat.params.get("young", young))
    if young <= 0.0:
        actual_log.error(
            f"/MAT/LAW109/{mat_id}: Young's modulus E must be strictly positive (ANCMSG 1514)",
            "MAT CHECK",
        )

    # 3. Poisson's ratio check
    nu = getattr(mat, "nu", getattr(mat, "Nu", 0.0))
    if isinstance(mat, dict):
        nu = mat.get("nu", 0.0)
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        nu = mat.params.get("MAT_NU", mat.params.get("nu", nu))
    if nu < 0.0 or nu >= 0.5:
        actual_log.error(
            f"/MAT/LAW109/{mat_id}: Poisson's ratio Nu must satisfy 0 <= Nu < 0.5, got {nu} (ANCMSG 3068)",
            "MAT CHECK",
        )

    # 4. Taylor-Quinney factor bounds
    eta = getattr(mat, "eta", 1.0)
    if isinstance(mat, dict):
        eta = mat.get("eta", 1.0)
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        eta = mat.params.get("MAT_ETA", mat.params.get("eta", eta))
    if eta < 0.0 or eta > 1.0:
        actual_log.error(
            f"/MAT/LAW109/{mat_id}: Taylor-Quinney coefficient ETA must satisfy 0 <= ETA <= 1, got {eta}",
            "MAT CHECK",
        )

    # 5. Yscale check
    yscale = getattr(mat, "yscale_h", getattr(mat, "yscale", 1.0))
    if isinstance(mat, dict):
        yscale = mat.get("yscale_h", mat.get("yscale", 1.0))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        yscale = mat.params.get("MAT_Yscale", mat.params.get("yscale", yscale))
    if yscale <= 0.0:
        actual_log.error(
            f"/MAT/LAW109/{mat_id}: yield stress scale factor Yscale_h must be strictly positive",
            "MAT CHECK",
        )

    # 6. Element compatibility check (ANCMSG 306)
    if actual_model is not None:
        parts_dict = getattr(actual_model, "parts", None) or {}
        props_dict = getattr(actual_model, "properties", None) or {}
        for etype in ("beams", "trusses", "springs"):
            elements = getattr(actual_model, etype, None) or {}
            for eid, elem in elements.items():
                part_id = getattr(elem, "part_id", 0)
                part = parts_dict.get(part_id)
                if part is not None and getattr(part, "mat_id", 0) == mat_id:
                    actual_log.error(
                        f"/MAT/LAW109/{mat_id} is not supported for 1D elements ({etype.lower()}) (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    break
        for pid, part in parts_dict.items():
            if getattr(part, "mat_id", 0) == mat_id:
                prop_id = getattr(part, "prop_id", 0)
                prop = props_dict.get(prop_id)
                if prop is not None:
                    ptype = str(getattr(prop, "type", getattr(prop, "prop_type", "")) or "").upper()
                    if any(s in ptype for s in ("TYPE2", "TYPE3", "TYPE4", "TYPE11", "BEAM", "TRUSS", "SPRING")):
                        actual_log.error(
                            f"/MAT/LAW109/{mat_id} is not supported for 1D elements ({ptype.lower()}) (ANCMSG 306)",
                            "MAT CHECK",
                        )


_check_mat_law109 = check_mat_law109
check_mat_tab_plas = check_mat_law109
check_mat_elasto_plas_tab = check_mat_law109


def check_mat_law110(*args: Any, **kwargs: Any) -> None:
    """Validate /MAT/LAW110 (/MAT/VEGTER / /MAT/PLAS_VEGTER) parameter bounds and element compatibility (M579).

    Upstream reference: ``starter/source/materials/mat/mat110/hm_read_mat110.F`` / ``matl110_vegter.cfg``.

    Checks:
      1. RHO_I > 0 (ANCMSG 1514).
      2. E > 0 (ANCMSG 1514).
      3. 0.0 <= NU < 0.5 (ANCMSG 49 / ANCMSG 3068).
      4. 1 <= Icrit <= 4 (ANCMSG 1776).
      5. If TAB_YLD == 0: SIGMA_r > 0 (ANCMSG 1777).
      6. If TINI == 0: warning (ANCMSG 1778).
      7. NANGLE <= 10 (ANCMSG 1779).
      8. If Icrit == 4: warning if WSH(J) != WSH(NANGLE - J + 1) (ANCMSG 1800).
      9. VP <= 3 (ANCMSG 1802).
      10. Element compatibility: 2D shells only. Rejects solids (ANCMSG 305) and 1D elements (ANCMSG 306).
    """
    actual_mat = None
    actual_log = None
    actual_model = None
    actual_mid = None

    if "model" in kwargs:
        actual_model = kwargs["model"]
    if "mat" in kwargs:
        actual_mat = kwargs["mat"]
    if "log" in kwargs:
        actual_log = kwargs["log"]
    if "mat_id" in kwargs:
        actual_mid = kwargs["mat_id"]

    for c in args:
        if isinstance(c, MessageLog):
            actual_log = c
        elif isinstance(c, Model):
            actual_model = c
        elif hasattr(c, "id") and (hasattr(c, "rho_i") or hasattr(c, "icrit") or hasattr(c, "tab_yld") or hasattr(c, "sigma_r")):
            actual_mat = c
        elif isinstance(c, int):
            actual_mid = c

    if actual_log is None:
        actual_log = MessageLog()

    mat = actual_mat
    mat_id = actual_mid if actual_mid is not None else (mat.id if mat is not None else 0)

    if mat is None and actual_model is not None and mat_id:
        mat = getattr(actual_model, "mat_law110s", {}).get(mat_id)
        if mat is None:
            mat = getattr(actual_model, "materials", {}).get(mat_id)

    if mat is None:
        return

    titr = getattr(mat, "title", f"/MAT/LAW110/{mat_id}")

    # 1. Density check (ANCMSG 1514)
    rho0 = getattr(mat, "rho_i", getattr(mat, "rho0", getattr(mat, "rho", 0.0)))
    if isinstance(mat, dict):
        rho0 = mat.get("rho_i", mat.get("rho0", mat.get("rho", 0.0)))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        rho0 = mat.params.get("MAT_RHO", mat.params.get("rho0", rho0))
    if rho0 <= 0.0:
        actual_log.error(
            f"/MAT/LAW110/{mat_id}: initial density RHO_I must be strictly positive (ANCMSG 1514)",
            "MAT CHECK",
        )

    # 2. Young's modulus check (ANCMSG 1514)
    young = getattr(mat, "e", getattr(mat, "young", getattr(mat, "E", 0.0)))
    if isinstance(mat, dict):
        young = mat.get("e", mat.get("young", 0.0))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        young = mat.params.get("MAT_E", mat.params.get("young", young))
    if young <= 0.0:
        actual_log.error(
            f"/MAT/LAW110/{mat_id}: Young's modulus E must be strictly positive (ANCMSG 1514)",
            "MAT CHECK",
        )

    # 3. Poisson's ratio check
    nu = getattr(mat, "nu", getattr(mat, "Nu", 0.0))
    if isinstance(mat, dict):
        nu = mat.get("nu", 0.0)
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        nu = mat.params.get("MAT_NU", mat.params.get("nu", nu))
    if nu < 0.0 or nu >= 0.5:
        actual_log.error(
            f"/MAT/LAW110/{mat_id}: Poisson's ratio Nu must satisfy 0 <= Nu < 0.5, got {nu} (ANCMSG 49 / ANCMSG 3068)",
            "MAT CHECK",
        )

    # 4. Formulation Icrit check (ANCMSG 1776)
    icrit = getattr(mat, "icrit", 1)
    if isinstance(mat, dict):
        icrit = mat.get("icrit", 1)
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        icrit = mat.params.get("MAT_Icrit", icrit)
    if icrit < 1 or icrit > 4:
        actual_log.error(
            f"/MAT/LAW110/{mat_id}: formulation Icrit must be between 1 and 4, got {icrit} (ANCMSG 1776)",
            "MAT CHECK",
        )

    # 5. Yield stress check (ANCMSG 1777)
    tab_yld = getattr(mat, "tab_yld", 0)
    sigma_r = getattr(mat, "sigma_r", getattr(mat, "sig0", 0.0))
    if isinstance(mat, dict):
        tab_yld = mat.get("tab_yld", 0)
        sigma_r = mat.get("sigma_r", mat.get("sig0", 0.0))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        tab_yld = mat.params.get("MAT_TAB_YLD", tab_yld)
        sigma_r = mat.params.get("SIGMA_r", mat.params.get("sig0", sigma_r))
    if tab_yld == 0 and sigma_r <= 0.0:
        actual_log.error(
            f"/MAT/LAW110/{mat_id}: initial yield stress SIGMA_r must be strictly positive when TAB_YLD=0 (ANCMSG 1777)",
            "MAT CHECK",
        )

    # 6. Initial temperature warning (ANCMSG 1778)
    tini = getattr(mat, "tini", 293.0)
    if isinstance(mat, dict):
        tini = mat.get("tini", 293.0)
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        tini = mat.params.get("T_Initial", tini)
    if tini == 0.0:
        actual_log.warning(
            f"/MAT/LAW110/{mat_id}: initial temperature T_ini is zero (ANCMSG 1778)",
            "MAT CHECK",
        )

    # 7. Angle count check (ANCMSG 1779)
    angles_data = getattr(mat, "angles_data", None) or []
    if len(angles_data) > 10:
        actual_log.error(
            f"/MAT/LAW110/{mat_id}: number of experimental angles exceeds maximum 10, got {len(angles_data)} (ANCMSG 1779)",
            "MAT CHECK",
        )

    # 8. Strain rate choice flag check (ANCMSG 1802)
    vp = getattr(mat, "vp", 2)
    if isinstance(mat, dict):
        vp = mat.get("vp", 2)
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        vp = mat.params.get("Vflag", vp)
    if vp > 3:
        actual_log.error(
            f"/MAT/LAW110/{mat_id}: strain rate choice flag VP must be <= 3, got {vp} (ANCMSG 1802)",
            "MAT CHECK",
        )

    # 9. Element compatibility check (ANCMSG 305 for solids, ANCMSG 306 for 1D)
    if actual_model is not None:
        parts_dict = getattr(actual_model, "parts", None) or {}
        props_dict = getattr(actual_model, "properties", None) or {}

        # 1. Check via element groups
        if hasattr(actual_model, "element_groups"):
            try:
                grps = actual_model.element_groups()
                if callable(grps):
                    grps = grps()
                for item in grps:
                    if isinstance(item, tuple) and len(item) == 2:
                        name, el_group = item
                    else:
                        continue
                    mids_in_group = set()
                    if hasattr(el_group, "values") and callable(el_group.values):
                        for el in el_group.values():
                            el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                            if el_mid is not None:
                                mids_in_group.add(el_mid)
                    if hasattr(el_group, "state") and isinstance(el_group.state, dict) and "slices" in el_group.state:
                        for _, m_part, _ in el_group.state["slices"]:
                            m_id = getattr(m_part, "id", None)
                            if m_id is not None:
                                mids_in_group.add(m_id)
                    if mat_id in mids_in_group:
                        if name in ("bricks", "bricks_heph", "bric20s", "tetras", "tetra10s", "penta6", "penta6s", "pyra5", "solid", "solids"):
                            actual_log.error(
                                f"/MAT/LAW110/{mat_id} (/MAT/VEGTER) is not supported for solid elements ({name}) (ANCMSG 305)",
                                "MAT CHECK",
                            )
                        elif name in ("trusses", "beams", "springs"):
                            actual_log.error(
                                f"/MAT/LAW110/{mat_id} (/MAT/VEGTER) is not supported for 1D elements ({name}) (ANCMSG 306)",
                                "MAT CHECK",
                            )
            except (TypeError, AttributeError):
                pass

        # 2. Check via element dicts/groups directly on model
        for etype in ("beams", "trusses", "springs"):
            elements = getattr(actual_model, etype, None) or []
            elems_iter = elements.values() if hasattr(elements, "values") and callable(elements.values) else (elements.values() if isinstance(elements, dict) else (elements.elements.values() if hasattr(elements, "elements") else []))
            for elem in elems_iter:
                part_id = getattr(elem, "part_id", getattr(elem, "pid", getattr(elem, "part", 0)))
                part = parts_dict.get(part_id)
                if (part is not None and getattr(part, "mat_id", 0) == mat_id) or getattr(elem, "mat_id", 0) == mat_id:
                    actual_log.error(
                        f"/MAT/LAW110/{mat_id} is not supported for 1D elements ({etype.lower()}) (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    break
        for etype in ("bricks", "tetras", "penta6", "penta6s", "pyra5"):
            elements = getattr(actual_model, etype, None) or []
            elems_iter = elements.values() if hasattr(elements, "values") and callable(elements.values) else (elements.values() if isinstance(elements, dict) else (elements.elements.values() if hasattr(elements, "elements") else []))
            for elem in elems_iter:
                part_id = getattr(elem, "part_id", getattr(elem, "pid", getattr(elem, "part", 0)))
                part = parts_dict.get(part_id)
                if (part is not None and getattr(part, "mat_id", 0) == mat_id) or getattr(elem, "mat_id", 0) == mat_id:
                    actual_log.error(
                        f"/MAT/LAW110/{mat_id} is not supported for 3D solid elements ({etype.lower()}) (ANCMSG 305)",
                        "MAT CHECK",
                    )
                    break
        for pid, part in parts_dict.items():
            if getattr(part, "mat_id", 0) == mat_id:
                prop_id = getattr(part, "prop_id", 0)
                prop = props_dict.get(prop_id)
                if prop is not None:
                    ptype_val = getattr(prop, "type", getattr(prop, "prop_type", None))
                    ptype = str(ptype_val or "").upper()
                    if ptype_val in (2, 3, 4, 11) or any(s in ptype for s in ("TYPE2", "TYPE3", "TYPE4", "TYPE11", "BEAM", "TRUSS", "SPRING")):
                        actual_log.error(
                            f"/MAT/LAW110/{mat_id} is not supported for 1D elements ({ptype.lower()}) (ANCMSG 306)",
                            "MAT CHECK",
                        )
                    elif ptype_val in (6, 14) or any(s in ptype for s in ("6", "14", "TYPE6", "TYPE14", "SOLID", "BRICK", "TETRA")):
                        actual_log.error(
                            f"/MAT/LAW110/{mat_id} is not supported for 3D solid elements ({ptype.lower()}) (ANCMSG 305)",
                            "MAT CHECK",
                        )


_check_mat_law110 = check_mat_law110
check_mat_vegter = check_mat_law110
check_mat_plas_vegter = check_mat_law110


def check_mat_law76(*args: Any, **kwargs: Any) -> None:
    """Validate /MAT/LAW76 or /MAT/SAMP-1 parameter bounds and element compatibility (M588).

    Upstream reference: ``starter/source/materials/mat/mat076/hm_read_mat76.F``.

    Checks:
      1. RHO > 0 (ANCMSG 1514).
      2. E > 0 (ANCMSG 1514).
      3. -1.0 < NU < 0.5 (Poisson ratio bounds).
      4. FUNC_TENS > 0 (ANCMSG 126, tension yield curve is mandatory).
      5. If curves are given and model.functions exists, check that they exist.
      6. Compatible elements: 3D solids and 2D shells. Rejects 1D elements (ANCMSG 306).
    """
    actual_mat = None
    actual_log = None
    actual_model = None
    actual_mid = None

    if "model" in kwargs:
        actual_model = kwargs["model"]
    if "mat" in kwargs:
        actual_mat = kwargs["mat"]
    if "log" in kwargs:
        actual_log = kwargs["log"]
    if "mat_id" in kwargs:
        actual_mid = kwargs["mat_id"]

    for c in args:
        if isinstance(c, MessageLog):
            actual_log = c
        elif isinstance(c, Model):
            actual_model = c
        elif hasattr(c, "id") and (hasattr(c, "fct_id_t") or hasattr(c, "fct_ID_T") or hasattr(c, "nu_p") or hasattr(c, "law") or hasattr(c, "params")):
            actual_mat = c
        elif isinstance(c, int):
            actual_mid = c

    if actual_log is None:
        actual_log = MessageLog()

    mat = actual_mat
    mat_id = actual_mid if actual_mid is not None else (mat.id if mat is not None else 0)

    if mat is not None:
        params = getattr(mat, "params", {}) if hasattr(mat, "params") and isinstance(mat.params, dict) else (mat if isinstance(mat, dict) else {})
        rho = float(getattr(mat, "rho", None) or getattr(mat, "rho_0", None) or getattr(mat, "rho0", None) or params.get("rho") or params.get("rho0") or params.get("rho_0") or 0.0)
        if rho <= 0.0:
            actual_log.error(f"/MAT/LAW76/{mat_id}: initial density RHO must be strictly positive (got {rho}) (ANCMSG 1514)", "MAT CHECK")

        e = float(getattr(mat, "e", None) or getattr(mat, "E", None) or getattr(mat, "young", None) or params.get("e") or params.get("E") or params.get("young") or 0.0)
        if e <= 0.0:
            actual_log.error(f"/MAT/LAW76/{mat_id}: Young's modulus E must be strictly positive (got {e}) (ANCMSG 1514)", "MAT CHECK")

        nu_val = getattr(mat, "nu", None)
        if nu_val is None:
            nu_val = params.get("nu", 0.0)
        nu = float(nu_val)
        if nu <= -1.0 or nu >= 0.5:
            actual_log.error(f"/MAT/LAW76/{mat_id}: Poisson's ratio NU must be in (-1.0, 0.5) (got {nu})", "MAT CHECK")

        fct_t = int(getattr(mat, "fct_id_t", None) or getattr(mat, "fct_ID_T", None) or getattr(mat, "func_tens", None) or params.get("fct_id_t") or params.get("fct_ID_T") or params.get("func_tens") or 0)
        if fct_t <= 0:
            actual_log.error(f"/MAT/LAW76/{mat_id}: tension yield curve fct_ID_T is required and must be > 0 (got {fct_t}) (ANCMSG 126)", "MAT CHECK")

        if actual_model is not None:
            funcs = getattr(actual_model, "functions", {})
            if funcs:
                if fct_t > 0 and fct_t not in funcs:
                    actual_log.error(f"/MAT/LAW76/{mat_id}: referenced tension curve fct_ID_T={fct_t} not found in model functions", "MAT CHECK")
                fct_c = int(getattr(mat, "fct_id_c", None) or getattr(mat, "fct_ID_C", None) or getattr(mat, "func_comp", None) or params.get("fct_id_c") or params.get("fct_ID_C") or params.get("func_comp") or 0)
                if fct_c > 0 and fct_c not in funcs:
                    actual_log.error(f"/MAT/LAW76/{mat_id}: referenced compression curve fct_ID_C={fct_c} not found in model functions", "MAT CHECK")
                fct_s = int(getattr(mat, "fct_id_s", None) or getattr(mat, "fct_ID_S", None) or getattr(mat, "func_shear", None) or params.get("fct_id_s") or params.get("fct_ID_S") or params.get("func_shear") or 0)
                if fct_s > 0 and fct_s not in funcs:
                    actual_log.error(f"/MAT/LAW76/{mat_id}: referenced shear curve fct_ID_S={fct_s} not found in model functions", "MAT CHECK")

            parts_dict = getattr(actual_model, "parts", {})
            props_dict = getattr(actual_model, "properties", {})
            for pid, part in parts_dict.items():
                if getattr(part, "mat_id", 0) == mat_id:
                    prop_id = getattr(part, "prop_id", 0)
                    prop = props_dict.get(prop_id)
                    if prop is not None:
                        ptype_val = getattr(prop, "type", getattr(prop, "prop_type", None))
                        ptype = str(ptype_val or "").upper()
                        if ptype_val in (2, 3, 4, 11) or any(s in ptype for s in ("TYPE2", "TYPE3", "TYPE4", "TYPE11", "BEAM", "TRUSS", "SPRING")):
                            actual_log.error(
                                f"/MAT/LAW76/{mat_id} is not supported for 1D elements ({ptype.lower()}) (ANCMSG 306)",
                                "MAT CHECK",
                            )


_check_mat_law76 = check_mat_law76
check_mat_samp = check_mat_law76
check_mat_samp_1 = check_mat_law76


def check_mat_law124(*args: Any, **kwargs: Any) -> None:
    """Validate /MAT/LAW124 (/MAT/CDPM2) parameter bounds (M598)."""
    actual_log = None
    actual_model = None
    actual_mat = None
    actual_mid = None

    if "model" in kwargs:
        actual_model = kwargs["model"]
    if "mat" in kwargs:
        actual_mat = kwargs["mat"]
    if "log" in kwargs:
        actual_log = kwargs["log"]
    if "mat_id" in kwargs:
        actual_mid = kwargs["mat_id"]

    for c in args:
        if isinstance(c, MessageLog):
            actual_log = c
        elif isinstance(c, Model):
            actual_model = c
        elif hasattr(c, "id") and (hasattr(c, "fc") or hasattr(c, "ft") or hasattr(c, "law") or hasattr(c, "params")):
            actual_mat = c
        elif isinstance(c, int):
            actual_mid = c

    if actual_log is None:
        actual_log = MessageLog()

    mat = actual_mat
    mat_id = actual_mid if actual_mid is not None else (mat.id if mat is not None else 0)

    if mat is not None:
        params = getattr(mat, "params", {}) if hasattr(mat, "params") and isinstance(mat.params, dict) else (mat if isinstance(mat, dict) else {})
        rho = float(getattr(mat, "rho", None) or getattr(mat, "rho0", None) or params.get("rho") or params.get("rho0") or params.get("MAT_RHO") or 0.0)
        if rho <= 0.0:
            actual_log.error(f"/MAT/LAW124/{mat_id}: initial density RHO must be strictly positive (got {rho}) (ANCMSG 1514)", "MAT CHECK")

        young = float(getattr(mat, "E", None) or getattr(mat, "e", None) or params.get("E") or params.get("e") or params.get("MAT_E") or 0.0)
        if young <= 0.0:
            actual_log.error(f"/MAT/LAW124/{mat_id}: Young's modulus E must be strictly positive (got {young}) (ANCMSG 1514)", "MAT CHECK")

        nu = float(getattr(mat, "nu", None) or params.get("nu") or params.get("MAT_NU") if (getattr(mat, "nu", None) is not None or params.get("nu") is not None or params.get("MAT_NU") is not None) else 0.2)
        if nu < 0.0 or nu >= 0.5:
            actual_log.error(f"/MAT/LAW124/{mat_id}: Poisson's ratio NU must satisfy 0 <= NU < 0.5 (got {nu}) (ANCMSG 49)", "MAT CHECK")

        fc = float(getattr(mat, "fc", None) or params.get("fc") or params.get("MAT_FC") or 0.0)
        if fc <= 0.0:
            actual_log.error(f"/MAT/LAW124/{mat_id}: compressive strength FC must be strictly positive (got {fc})", "MAT CHECK")

        ft = float(getattr(mat, "ft", None) or params.get("ft") or params.get("MAT_FT") or 0.0)
        if ft <= 0.0:
            actual_log.error(f"/MAT/LAW124/{mat_id}: tensile strength FT must be strictly positive (got {ft})", "MAT CHECK")

        if actual_model is not None:
            parts_dict = getattr(actual_model, "parts", {})
            props_dict = getattr(actual_model, "properties", {})
            for pid, part in parts_dict.items():
                if getattr(part, "mat_id", 0) == mat_id:
                    prop_id = getattr(part, "prop_id", 0)
                    prop = props_dict.get(prop_id)
                    if prop is not None:
                        ptype_val = getattr(prop, "type", getattr(prop, "prop_type", None))
                        ptype = str(ptype_val or "").upper()
                        if ptype_val in (1, 2, 3, 4, 11) or any(s in ptype for s in ("SHELL", "SH3N", "TYPE1", "TYPE2", "TYPE3", "TYPE4", "TYPE11", "BEAM", "TRUSS", "SPRING")):
                            actual_log.error(
                                f"/MAT/LAW124/{mat_id} is only supported for 3D solid elements (assigned to {ptype.lower()}) (ANCMSG 306)",
                                "MAT CHECK",
                            )


_check_mat_law124 = check_mat_law124
check_mat_cdpm2 = check_mat_law124


def check_mat_law126(*args: Any, **kwargs: Any) -> None:
    """Validate /MAT/LAW126 (/MAT/JOHNSON_HOLMQUIST_CONCRETE, /MAT/HJC) parameter bounds (M591)."""
    actual_log = None
    actual_model = None
    actual_mat = None
    actual_mid = None

    if "model" in kwargs:
        actual_model = kwargs["model"]
    if "mat" in kwargs:
        actual_mat = kwargs["mat"]
    if "log" in kwargs:
        actual_log = kwargs["log"]
    if "mat_id" in kwargs:
        actual_mid = kwargs["mat_id"]

    for c in args:
        if isinstance(c, MessageLog):
            actual_log = c
        elif isinstance(c, Model):
            actual_model = c
        elif hasattr(c, "id") and (hasattr(c, "shear") or hasattr(c, "pc") or hasattr(c, "fc") or hasattr(c, "law") or hasattr(c, "params")):
            actual_mat = c
        elif isinstance(c, int):
            actual_mid = c

    if actual_log is None:
        actual_log = MessageLog()

    mat = actual_mat
    mat_id = actual_mid if actual_mid is not None else (mat.id if mat is not None else 0)

    if mat is not None:
        params = getattr(mat, "params", {}) if hasattr(mat, "params") and isinstance(mat.params, dict) else (mat if isinstance(mat, dict) else {})
        rho = float(getattr(mat, "rho", None) or getattr(mat, "rho0", None) or params.get("rho") or params.get("rho0") or params.get("MAT_RHO") or 0.0)
        if rho <= 0.0:
            actual_log.error(f"/MAT/LAW126/{mat_id}: initial density RHO must be strictly positive (got {rho}) (ANCMSG 1514)", "MAT CHECK")

        shear = float(getattr(mat, "shear", None) or getattr(mat, "g", None) or getattr(mat, "G", None) or params.get("shear") or params.get("g") or params.get("G") or params.get("MAT_G") or 0.0)
        if shear <= 0.0:
            actual_log.error(f"/MAT/LAW126/{mat_id}: shear modulus G must be strictly positive (got {shear}) (ANCMSG 1514)", "MAT CHECK")

        fc = float(getattr(mat, "fc", None) or params.get("fc") or params.get("MAT_FC") or 0.0)
        if fc <= 0.0:
            actual_log.error(f"/MAT/LAW126/{mat_id}: compressive strength FC must be strictly positive (got {fc})", "MAT CHECK")

        pc = float(getattr(mat, "pc", None) or params.get("pc") or params.get("MAT_PC") or 0.0)
        muc = float(getattr(mat, "muc", None) or params.get("muc") or params.get("MAT_MUC") or 0.0)
        if pc <= 0.0 or muc <= 0.0:
            actual_log.error(f"/MAT/LAW126/{mat_id}: crushing pressure PC and volumetric strain MUC must be strictly positive (got PC={pc}, MUC={muc})", "MAT CHECK")

        pl = float(getattr(mat, "pl", None) or params.get("pl") or params.get("MAT_PL") or 0.0)
        mul = float(getattr(mat, "mul", None) or params.get("mul") or params.get("MAT_MUL") or 0.0)
        if pl <= pc or mul <= muc:
            actual_log.error(f"/MAT/LAW126/{mat_id}: locking parameters PL, MUL must exceed crushing PC, MUC (got PL={pl} <= PC={pc} or MUL={mul} <= MUC={muc})", "MAT CHECK")

        if actual_model is not None:
            parts_dict = getattr(actual_model, "parts", {})
            props_dict = getattr(actual_model, "properties", {})
            for pid, part in parts_dict.items():
                if getattr(part, "mat_id", 0) == mat_id:
                    prop_id = getattr(part, "prop_id", 0)
                    prop = props_dict.get(prop_id)
                    if prop is not None:
                        ptype_val = getattr(prop, "type", getattr(prop, "prop_type", None))
                        ptype = str(ptype_val or "").upper()
                        if ptype_val in (1, 2, 3, 4, 11) or any(s in ptype for s in ("SHELL", "SH3N", "TYPE1", "TYPE2", "TYPE3", "TYPE4", "TYPE11", "BEAM", "TRUSS", "SPRING")):
                            actual_log.error(
                                f"/MAT/LAW126/{mat_id} is only supported for 3D solid elements (assigned to {ptype.lower()}) (ANCMSG 306)",
                                "MAT CHECK",
                            )


def check_mat_law169(*args: Any, **kwargs: Any) -> None:
    """Validate /MAT/LAW169 (/MAT/ARUP_ADHESIVE) parameter bounds (M591)."""
    actual_log = None
    actual_model = None
    actual_mat = None
    actual_mid = None

    if "model" in kwargs:
        actual_model = kwargs["model"]
    if "mat" in kwargs:
        actual_mat = kwargs["mat"]
    if "log" in kwargs:
        actual_log = kwargs["log"]
    if "mat_id" in kwargs:
        actual_mid = kwargs["mat_id"]

    for c in args:
        if isinstance(c, MessageLog):
            actual_log = c
        elif isinstance(c, Model):
            actual_model = c
        elif hasattr(c, "id") and (hasattr(c, "tenmax") or hasattr(c, "shrmax") or hasattr(c, "law") or hasattr(c, "params")):
            actual_mat = c
        elif isinstance(c, int):
            actual_mid = c

    if actual_log is None:
        actual_log = MessageLog()

    mat = actual_mat
    mat_id = actual_mid if actual_mid is not None else (mat.id if mat is not None else 0)

    if mat is not None:
        params = getattr(mat, "params", {}) if hasattr(mat, "params") and isinstance(mat.params, dict) else (mat if isinstance(mat, dict) else {})
        rho = float(getattr(mat, "rho", None) or getattr(mat, "rho0", None) or params.get("rho") or params.get("rho0") or params.get("Rho") or 0.0)
        if rho <= 0.0:
            actual_log.error(f"/MAT/LAW169/{mat_id}: initial density RHO must be strictly positive (got {rho}) (ANCMSG 1514)", "MAT CHECK")

        e = float(getattr(mat, "e", None) or getattr(mat, "E", None) or getattr(mat, "young", None) or params.get("e") or params.get("E") or params.get("young") or 0.0)
        if e <= 0.0:
            actual_log.error(f"/MAT/LAW169/{mat_id}: Young's modulus E must be strictly positive (got {e}) (ANCMSG 1514)", "MAT CHECK")

        nu_val = getattr(mat, "nu", None)
        if nu_val is None:
            nu_val = params.get("nu", params.get("Nu", 0.0))
        nu = float(nu_val)
        if nu <= -1.0 or nu >= 0.5:
            actual_log.error(f"/MAT/LAW169/{mat_id}: Poisson's ratio NU must be in (-1.0, 0.5) (got {nu})", "MAT CHECK")

        tenmax = float(getattr(mat, "tenmax", None) or params.get("tenmax") or params.get("MAT169_TENMAX") or 0.0)
        if tenmax <= 0.0:
            actual_log.error(f"/MAT/LAW169/{mat_id}: peak tensile strength TENMAX must be > 0 (got {tenmax})", "MAT CHECK")

        shrmax = float(getattr(mat, "shrmax", None) or params.get("shrmax") or params.get("MAT169_SHRMAX") or 0.0)
        if shrmax <= 0.0:
            actual_log.error(f"/MAT/LAW169/{mat_id}: peak shear strength SHRMAX must be > 0 (got {shrmax})", "MAT CHECK")

        gcten = float(getattr(mat, "gcten", None) or getattr(mat, "gic", None) or params.get("gcten") or params.get("gic") or params.get("MAT169_GCTEN") or 0.0)
        if gcten <= 0.0:
            actual_log.error(f"/MAT/LAW169/{mat_id}: energy release rate GCTEN must be > 0 (got {gcten})", "MAT CHECK")

        gcshr = float(getattr(mat, "gcshr", None) or getattr(mat, "giic", None) or params.get("gcshr") or params.get("giic") or params.get("MAT169_GCSHR") or 0.0)
        if gcshr <= 0.0:
            actual_log.error(f"/MAT/LAW169/{mat_id}: energy release rate GCSHR must be > 0 (got {gcshr})", "MAT CHECK")

        pwrt = int(getattr(mat, "pwrt", None) or params.get("pwrt") or params.get("MAT169_PWRT") or 2)
        pwrs = int(getattr(mat, "pwrs", None) or params.get("pwrs") or params.get("MAT169_PWRS") or 2)
        if pwrt <= 0 or pwrs <= 0:
            actual_log.error(f"/MAT/LAW169/{mat_id}: power law exponents PWRT and PWRS must be > 0 (got PWRT={pwrt}, PWRS={pwrs})", "MAT CHECK")

        shrp = float(getattr(mat, "shrp", None) or params.get("shrp") or params.get("MAT169_SHRP") or 0.0)
        if shrp < 0.0 or shrp >= 1.0:
            actual_log.error(f"/MAT/LAW169/{mat_id}: shear plastic plateau fraction SHRP must be in [0, 1) (got {shrp})", "MAT CHECK")

        if actual_model is not None:
            parts_dict = getattr(actual_model, "parts", {})
            props_dict = getattr(actual_model, "properties", {})
            for pid, part in parts_dict.items():
                if getattr(part, "mat_id", 0) == mat_id:
                    prop_id = getattr(part, "prop_id", 0)
                    prop = props_dict.get(prop_id)
                    if prop is not None:
                        ptype_val = getattr(prop, "type", getattr(prop, "prop_type", None))
                        ptype = str(ptype_val or "").upper()
                        if ptype_val in (2, 3, 4, 11) or any(s in ptype for s in ("TYPE2", "TYPE3", "TYPE4", "TYPE11", "BEAM", "TRUSS", "SPRING")):
                            actual_log.error(
                                f"/MAT/LAW169/{mat_id} is not supported for 1D elements ({ptype.lower()}) (ANCMSG 306)",
                                "MAT CHECK",
                            )


_check_mat_law126 = check_mat_law126
check_mat_hjc = check_mat_law126
check_mat_johnson_holmquist_concrete = check_mat_law126

_check_mat_law169 = check_mat_law169
check_mat_arup = check_mat_law169
check_mat_arup_adhesive = check_mat_law169


def check_materials(model: Model, log: MessageLog) -> None:
    """Validate all material parameters across model."""
    # M539: Material LAW34 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW"):
            check_mat_law34(mat, log)
    for mid, mat34 in getattr(model, "mat_law34s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law34(mat34, log)

    # M540: Material LAW37 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (37, "37", "LAW37", "BIPHAS", "BIPHASIC"):
            params = getattr(mat, "params", {}) or {}
            is_biquad = hasattr(mat, "a") and hasattr(mat, "c1") and not any(
                k in params for k in ("Lqud_Rho_l", "RHO_l0", "rho_l0", "alpha1", "ALPHA1", "C_l", "c_l")
            )
            if not is_biquad:
                check_mat_law37(mat, log)
    for mid, mat37 in getattr(model, "mat_law37s", {}).items():
        if mid not in getattr(model, "materials", {}):
            params = getattr(mat37, "params", {}) or {}
            is_biquad = hasattr(mat37, "a") and hasattr(mat37, "c1") and not any(
                k in params for k in ("Lqud_Rho_l", "RHO_l0", "rho_l0", "alpha1", "ALPHA1", "C_l", "c_l")
            )
            if not is_biquad:
                check_mat_law37(mat37, log)

    # M541: Material LAW38 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (38, "38", "LAW38", "VISC_TAB") or getattr(mat, "law_name", None) in ("LAW38", "VISC_TAB"):
            check_mat_law38(mat, log)
    for mid, mat38 in getattr(model, "mat_law38s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law38(mat38, log)

    # M542: Material LAW32 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (32, "32", "LAW32", "HILL") or getattr(mat, "law_name", None) in ("32", "LAW32", "HILL", "MAT_LAW32", "MAT_HILL"):
            check_mat_law32(mat, log)
    for mid, mat32 in getattr(model, "mat_law32s", {}).items():
        if mid not in getattr(model, "materials", {}):
            if not hasattr(mat32, "fct_id11"):
                check_mat_law32(mat32, log)

    # M543: Material LAW25 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS") or getattr(mat, "law_name", None) in ("25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", "MAT_LAW25", "MAT_COMP_PLAS", "MAT_COMPSH", "MAT_TSAI_WU", "MAT_CRASURV", "MAT_COMPOSITE_PLAS"):
            check_mat_law25(mat, log)
    for mid, mat25 in getattr(model, "mat_law25s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law25(mat25, log)

    # M544: Material LAW15 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG") or getattr(mat, "law_name", None) in ("15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG", "MAT_LAW15", "MAT_CHANG", "MAT_PLAS_ANISO", "MAT_COMP_CHANG"):
            check_mat_law15(mat, log)
    for mid, mat15 in getattr(model, "mat_law15s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law15(mat15, log)

    # M545: Material LAW22 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (22, "22", "LAW22", "DAMA", "PLAS_DAMA") or getattr(mat, "law_name", None) in ("22", "LAW22", "DAMA", "PLAS_DAMA", "MAT_LAW22", "MAT_DAMA", "MAT_PLAS_DAMA"):
            check_mat_law22(mat, log)
    for mid, mat22 in getattr(model, "mat_law22s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law22(mat22, log)

    # M546: Material LAW12 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (12, "12", "LAW12", "3PARBI", "3D_COMP", "COMP_3D", "RAGAB") or getattr(mat, "law_name", None) in ("12", "LAW12", "3PARBI", "3D_COMP", "COMP_3D", "RAGAB", "MAT_LAW12", "MAT_3D_COMP", "MAT_COMP_3D", "MAT_3PARBI", "MAT_RAGAB"):
            check_mat_law12(mat, log)
    for mid, mat12 in getattr(model, "mat_law12s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law12(mat12, log)

    # M547: Material LAW14 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (14, "14", "LAW14", "COMPSO", "COMP_SOL") or getattr(mat, "law_name", None) in ("14", "LAW14", "COMPSO", "COMP_SOL", "MAT_LAW14", "MAT_COMPSO", "MAT_COMP_SOL"):
            check_mat_law14(mat, log)
    for mid, mat14 in getattr(model, "mat_law14s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law14(mat14, log)

    # M548: Material LAW43 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB") or getattr(mat, "law_name", None) in ("43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB"):
            check_mat_law43(mat, log)
    for mid, mat43 in getattr(model, "mat_law43s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law43(mat43, log)

    # M549: Material LAW82 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (82, "82", "LAW82", "OGDEN", "LAW82_OGDEN") or getattr(mat, "law_name", None) in ("82", "LAW82", "OGDEN", "LAW82_OGDEN", "MAT_LAW82", "MAT_OGDEN", "MAT_LAW82_OGDEN"):
            check_mat_law82(mat, log)
    for mid, mat82 in getattr(model, "mat_law82s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law82(mat82, log)

    # M550: Material LAW69 parameter validation
    funcs = getattr(model, "functions", None)
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYP_ELAS") or getattr(mat, "law_name", None) in ("69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYP_ELAS", "MAT_LAW69", "MAT_HYP_ELAS", "MAT_HYPERELASTIC", "MAT_LAW69_HYP_ELAS"):
            check_mat_law69(mat, log, funcs)
    for mid, mat69 in getattr(model, "mat_law69s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law69(mat69, log, funcs)

    # M551: Material LAW60 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (60, "60", "LAW60", "PLAS_T3", "FABRIC", "MAT_PLAS_T3", "MAT_FABRIC") or getattr(mat, "law_name", None) in ("60", "LAW60", "PLAS_T3", "FABRIC", "MAT_PLAS_T3", "MAT_FABRIC", "MAT_LAW60"):
            check_mat_law60(mat, log=log, model=model, functions=funcs)
    for mid, mat60 in getattr(model, "mat_law60s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law60(mat60, log=log, model=model, functions=funcs)

    # M552: Material LAW48 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "MAT_PLAS_ZHAO") or getattr(mat, "law_name", None) in ("48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "MAT_PLAS_ZHAO", "MAT_LAW48", "LAW48_ZHAO"):
            check_mat_law48(mat, log=log, model=model)
    for mid, mat48 in getattr(model, "mat_law48s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law48(mat48, log=log, model=model)

    # M553: Material LAW58 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (58, "58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A") or getattr(mat, "law_name", None) in ("58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A"):
            check_mat_law58(mat=mat, model=model, log=log, mat_id=mid)
    for mid, mat58 in getattr(model, "mat_law58s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law58(mat=mat58, model=model, log=log, mat_id=mid)

    # M554: Material LAW52 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS", "LAW52_GURSON") or getattr(mat, "law_name", None) in ("52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS", "LAW52_GURSON"):
            check_mat_law52(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat52 in getattr(model, "mat_law52s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law52(model=model, mat_id=mid, mat=mat52, log=log)

    # M555: Material LAW57 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (57, "57", "LAW57", "BARLAT3", "MAT_BARLAT3", "LAW57_BARLAT3") or getattr(mat, "law_name", None) in ("57", "LAW57", "BARLAT3", "MAT_BARLAT3", "LAW57_BARLAT3"):
            check_mat_law57(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat57 in getattr(model, "mat_law57s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law57(model=model, mat_id=mid, mat=mat57, log=log)

    # M556: Material LAW21 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (21, "21", "LAW21", "DPRAG", "MAT_DPRAG", "LAW21_DPRAG", "DUCKHUB") or getattr(mat, "law_name", None) in ("21", "LAW21", "DPRAG", "MAT_DPRAG", "LAW21_DPRAG", "DUCKHUB", "MAT_LAW21", "MAT_DPRAG"):
            check_mat_law21(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat21 in getattr(model, "mat_law21s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law21(model=model, mat_id=mid, mat=mat21, log=log)
    for mid, mat_dp in getattr(model, "mat_dprags", {}).items():
        if mid not in getattr(model, "materials", {}) and mid not in getattr(model, "mat_law21s", {}):
            check_mat_law21(model=model, mat_id=mid, mat=mat_dp, log=log)

    # M557: Material LAW49 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (49, "49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB") or getattr(mat, "law_name", None) in ("49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB"):
            check_mat_law49(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat49 in getattr(model, "mat_law49s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law49(model=model, mat_id=mid, mat=mat49, log=log)

    # M558: Material LAW79 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (79, "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM") or getattr(mat, "law_name", None) in ("79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM"):
            check_mat_law79(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat79 in getattr(model, "mat_law79s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law79(model=model, mat_id=mid, mat=mat79, log=log)

    # M559: Material LAW50 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM", "LAW50_VISC_HONEY", "LAW50_HYP_FOAM") or getattr(mat, "law_name", None) in ("50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM", "LAW50_VISC_HONEY", "LAW50_HYP_FOAM"):
            check_mat_law50(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat50 in getattr(model, "mat_law50s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law50(model=model, mat_id=mid, mat=mat50, log=log)

    # M560: Material LAW163 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM") or getattr(mat, "law_name", None) in ("163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM"):
            check_mat_law163(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat163 in getattr(model, "mat_law163s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law163(model=model, mat_id=mid, mat=mat163, log=log)

    # M561: Material LAW73 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL") or getattr(mat, "law_name", None) in ("73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL"):
            check_mat_law73(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat73 in getattr(model, "mat_law73s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law73(model=model, mat_id=mid, mat=mat73, log=log)

    # M562: Material LAW66 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "FOAM_TAB") or getattr(mat, "law_name", None) in ("66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "FOAM_TAB"):
            check_mat_law66(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat66 in getattr(model, "mat_law66s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law66(model=model, mat_id=mid, mat=mat66, log=log)

    # M563: Material LAW74 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (74, "74", "LAW74", "HILL_3D", "ORTH_PLAS") or getattr(mat, "law_name", None) in ("74", "LAW74", "HILL_3D", "ORTH_PLAS", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS"):
            check_mat_law74(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat74 in getattr(model, "mat_law74s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law74(model=model, mat_id=mid, mat=mat74, log=log)

    # M564: Material LAW87 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000") or getattr(mat, "law_name", None) in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000"):
            check_mat_law87(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat87 in getattr(model, "mat_law87s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law87(model=model, mat_id=mid, mat=mat87, log=log)

    # M565: Material LAW88 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (88, "88", "LAW88", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TAB_HYP", "TABULATED_HYP") or getattr(mat, "law_name", None) in ("88", "LAW88", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TAB_HYP", "TABULATED_HYP", "MAT_LAW88", "MAT_HYPER_ELAS", "MAT_TABULATED_HYPERELASTIC", "MAT_TAB_HYP"):
            check_mat_law88(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat88 in getattr(model, "mat_law88s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law88(model=model, mat_id=mid, mat=mat88, log=log)

    # M566: Material LAW92 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (92, "92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE", "ARRUDA") or getattr(mat, "law_name", None) in ("92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE", "ARRUDA", "MAT_LAW92", "MAT_ARRUDA_BOYCE", "MAT_ARRUDA-BOYCE"):
            check_mat_law92(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat92 in getattr(model, "mat_law92s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law92(model=model, mat_id=mid, mat=mat92, log=log)

    # M567: Material LAW94 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (94, "94", "LAW94", "YEOH") or getattr(mat, "law_name", None) in ("94", "LAW94", "YEOH", "MAT_LAW94", "MAT_YEOH", "LAW94_YEOH"):
            check_mat_law94(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat94 in getattr(model, "mat_law94s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law94(model=model, mat_id=mid, mat=mat94, log=log)

    # M568: Material LAW93 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (93, "93", "LAW93", "ORTH_HILL") or getattr(mat, "law_name", None) in ("93", "LAW93", "ORTH_HILL", "MAT_LAW93", "MAT_ORTH_HILL", "LAW93_ORTH_HILL"):
            check_mat_law93(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat93 in getattr(model, "mat_law93s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law93(model=model, mat_id=mid, mat=mat93, log=log)

    # M569: Material LAW95 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (95, "95", "LAW95", "BERGSTROM_BOYCE") or getattr(mat, "law_name", None) in ("95", "LAW95", "BERGSTROM_BOYCE", "MAT_LAW95", "MAT_BERGSTROM_BOYCE", "LAW95_BERGSTROM_BOYCE"):
            check_mat_law95(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat95 in getattr(model, "mat_law95s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law95(model=model, mat_id=mid, mat=mat95, log=log)

    # M570: Material LAW100 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (100, "100", "LAW100", "VISC_HYP", "MNF") or getattr(mat, "law_name", None) in ("100", "LAW100", "VISC_HYP", "MNF", "MAT_100", "MAT_LAW100", "MAT_VISC_HYP", "MAT_MNF", "LAW100_VISC_HYP", "LAW100_MNF"):
            check_mat_law100(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat100 in getattr(model, "mat_law100s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law100(model=model, mat_id=mid, mat=mat100, log=log)

    # M571: Material LAW101 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (101, "101", "LAW101", "PP", "PLAS_POLY") or getattr(mat, "law_name", None) in ("101", "LAW101", "PP", "PLAS_POLY", "MAT_101", "MAT_LAW101", "MAT_PP", "MAT_PLAS_POLY", "LAW101_PLAS_POLY", "LAW101_PP"):
            check_mat_law101(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat101 in getattr(model, "mat_law101s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law101(model=model, mat_id=mid, mat=mat101, log=log)

    # M572: Material LAW102 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (102, "102", "LAW102", "DPRAG2") or getattr(mat, "law_name", None) in ("102", "LAW102", "DPRAG2", "DRUCKER_PRAGER_2", "MAT_102", "MAT_LAW102", "MAT_DPRAG2", "LAW102_DPRAG2"):
            check_mat_law102(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat102 in getattr(model, "mat_law102s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law102(model=model, mat_id=mid, mat=mat102, log=log)

    # M573: Material LAW103 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in _LAW103_KEYS or getattr(mat, "law_name", None) in _LAW103_KEYS:
            check_mat_law103(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat103 in getattr(model, "mat_law103s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law103(model=model, mat_id=mid, mat=mat103, log=log)

    # M574: Material LAW104 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in _LAW104_KEYS or getattr(mat, "law_name", None) in _LAW104_KEYS:
            check_mat_law104(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat104 in getattr(model, "mat_law104s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law104(model=model, mat_id=mid, mat=mat104, log=log)

    # M578: Material LAW109 parameter validation
    _law109_keys = (109, "109", "LAW109", "TAB_PLAS", "ELASTO_PLAS_TAB", "LAW109_TAB_PLAS", "MLAW109", "MAT_LAW109", "MAT_TAB_PLAS", "MAT_ELASTO_PLAS_TAB", "MAT_109")
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in _law109_keys or getattr(mat, "law_name", None) in _law109_keys:
            check_mat_law109(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat109 in getattr(model, "mat_law109s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law109(model=model, mat_id=mid, mat=mat109, log=log)

    # M579: Material LAW110 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in _LAW110_KEYS or getattr(mat, "law_name", None) in _LAW110_KEYS:
            check_mat_law110(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat110 in getattr(model, "mat_law110s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law110(model=model, mat_id=mid, mat=mat110, log=log)

    # M588: Material LAW76 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in _LAW76_KEYS or getattr(mat, "law_name", None) in _LAW76_KEYS:
            check_mat_law76(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat76 in getattr(model, "mat_law76s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law76(model=model, mat_id=mid, mat=mat76, log=log)

    # M598: Material LAW124 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in _LAW124_KEYS or getattr(mat, "law_name", None) in _LAW124_KEYS:
            check_mat_law124(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat124 in getattr(model, "mat_law124s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law124(model=model, mat_id=mid, mat=mat124, log=log)
    for mid, matcdpm2 in getattr(model, "mat_cdpm2s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law124(model=model, mat_id=mid, mat=matcdpm2, log=log)

    # M591: Material LAW126 and LAW169 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in _LAW126_KEYS or getattr(mat, "law_name", None) in _LAW126_KEYS:
            check_mat_law126(model=model, mat_id=mid, mat=mat, log=log)
        elif getattr(mat, "law", None) in _LAW169_KEYS or getattr(mat, "law_name", None) in _LAW169_KEYS:
            check_mat_law169(model=model, mat_id=mid, mat=mat, log=log)
    for mid, mat126 in getattr(model, "mat_law126s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law126(model=model, mat_id=mid, mat=mat126, log=log)
    for mid, mat169 in getattr(model, "mat_law169s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law169(model=model, mat_id=mid, mat=mat169, log=log)





_MAT_CHECKS: dict[Any, Any] = {
    93: check_mat_law93,
    "93": check_mat_law93,
    "LAW93": check_mat_law93,
    "ORTH_HILL": check_mat_law93,
    "MAT_LAW93": check_mat_law93,
    "MAT_ORTH_HILL": check_mat_law93,
    "LAW93_ORTH_HILL": check_mat_law93,
    94: check_mat_law94,
    "94": check_mat_law94,
    "LAW94": check_mat_law94,
    "YEOH": check_mat_law94,
    "MAT_LAW94": check_mat_law94,
    "MAT_YEOH": check_mat_law94,
    "LAW94_YEOH": check_mat_law94,
    92: check_mat_law92,
    "92": check_mat_law92,
    "LAW92": check_mat_law92,
    "ARRUDA_BOYCE": check_mat_law92,
    "ARRUDA-BOYCE": check_mat_law92,
    "ARRUDA": check_mat_law92,
    "MAT_LAW92": check_mat_law92,
    "MAT_ARRUDA_BOYCE": check_mat_law92,
    "MAT_ARRUDA-BOYCE": check_mat_law92,
    88: check_mat_law88,
    "88": check_mat_law88,
    "LAW88": check_mat_law88,
    "HYPER_ELAS": check_mat_law88,
    "TABULATED_HYPERELASTIC": check_mat_law88,
    "TAB_HYP": check_mat_law88,
    "TABULATED_HYP": check_mat_law88,
    "MAT_LAW88": check_mat_law88,
    "MAT_HYPER_ELAS": check_mat_law88,
    "MAT_TABULATED_HYPERELASTIC": check_mat_law88,
    "MAT_TAB_HYP": check_mat_law88,
    87: check_mat_law87,
    "87": check_mat_law87,
    "LAW87": check_mat_law87,
    "BARLAT": check_mat_law87,
    "BARLAT2000": check_mat_law87,
    "BARLAT_2000": check_mat_law87,
    "BARLAT2000_2D": check_mat_law87,
    "BARLAT_YLD2000": check_mat_law87,
    "MAT_LAW87": check_mat_law87,
    "MAT_BARLAT": check_mat_law87,
    "MAT_BARLAT2000": check_mat_law87,
    "MAT_BARLAT_2000": check_mat_law87,
    "MAT_BARLAT2000_2D": check_mat_law87,
    "MAT_BARLAT_YLD2000": check_mat_law87,
    "LAW87_BARLAT": check_mat_law87,
    "LAW87_BARLAT2000": check_mat_law87,
    "LAW87_BARLAT_2000": check_mat_law87,
    74: check_mat_law74,
    "74": check_mat_law74,
    "LAW74": check_mat_law74,
    "HILL_3D": check_mat_law74,
    "ORTH_PLAS": check_mat_law74,
    "MAT_LAW74": check_mat_law74,
    "MAT_HILL_3D": check_mat_law74,
    "MAT_ORTH_PLAS": check_mat_law74,
    66: check_mat_law66,
    "66": check_mat_law66,
    "LAW66": check_mat_law66,
    "PLAS_TAB_COSSER": check_mat_law66,
    "PLAS_COSSER": check_mat_law66,
    "MAT_LAW66": check_mat_law66,
    "MAT_PLAS_TAB_COSSER": check_mat_law66,
    "MAT_PLAS_COSSER": check_mat_law66,
    "FOAM_TAB": check_mat_law66,
    73: check_mat_law73,
    "73": check_mat_law73,
    "LAW73": check_mat_law73,
    "HILL_THERM": check_mat_law73,
    "THERM_HILL": check_mat_law73,
    "MAT_LAW73": check_mat_law73,
    "MAT_HILL_THERM": check_mat_law73,
    "MAT_THERM_HILL": check_mat_law73,
    163: check_mat_law163,
    "163": check_mat_law163,
    "LAW163": check_mat_law163,
    "CRUSHABLE_FOAM": check_mat_law163,
    "CRUSH_FOAM": check_mat_law163,
    "MAT_LAW163": check_mat_law163,
    "MAT_CRUSHABLE_FOAM": check_mat_law163,
    "MAT_CRUSH_FOAM": check_mat_law163,
    50: check_mat_law50,
    "50": check_mat_law50,
    "LAW50": check_mat_law50,
    "VISC_HONEY": check_mat_law50,
    "HYP_FOAM": check_mat_law50,
    "MAT_LAW50": check_mat_law50,
    "MAT_VISC_HONEY": check_mat_law50,
    "MAT_HYP_FOAM": check_mat_law50,
    "LAW50_VISC_HONEY": check_mat_law50,
    "LAW50_HYP_FOAM": check_mat_law50,
    34: check_mat_law34,
    "34": check_mat_law34,
    "LAW34": check_mat_law34,
    "BOLTZMAN": check_mat_law34,
    "BOLTZMANN": check_mat_law34,
    "VISC_MAXW": check_mat_law34,
    37: check_mat_law37,
    "37": check_mat_law37,
    "LAW37": check_mat_law37,
    "BIPHAS": check_mat_law37,
    "BIPHASIC": check_mat_law37,
    38: check_mat_law38,
    "38": check_mat_law38,
    "LAW38": check_mat_law38,
    "VISC_TAB": check_mat_law38,
    32: check_mat_law32,
    "32": check_mat_law32,
    "LAW32": check_mat_law32,
    "HILL": check_mat_law32,
    25: check_mat_law25,
    "25": check_mat_law25,
    "LAW25": check_mat_law25,
    "COMP_PLAS": check_mat_law25,
    "COMPSH": check_mat_law25,
    15: check_mat_law15,
    "15": check_mat_law15,
    "LAW15": check_mat_law15,
    "CHANG": check_mat_law15,
    22: check_mat_law22,
    "22": check_mat_law22,
    "LAW22": check_mat_law22,
    "DAMA": check_mat_law22,
    12: check_mat_law12,
    "12": check_mat_law12,
    "LAW12": check_mat_law12,
    "3PARBI": check_mat_law12,
    "3D_COMP": check_mat_law12,
    14: check_mat_law14,
    "14": check_mat_law14,
    "LAW14": check_mat_law14,
    "COMPSO": check_mat_law14,
    "COMP_SOL": check_mat_law14,
    43: check_mat_law43,
    "43": check_mat_law43,
    "LAW43": check_mat_law43,
    "HILL_TAB": check_mat_law43,
    "HILL_PLAS_TAB": check_mat_law43,
    "LAW43_HILL_TAB": check_mat_law43,
    "MAT_LAW43": check_mat_law43,
    "MAT_HILL_TAB": check_mat_law43,
    "MAT_HILL_PLAS_TAB": check_mat_law43,
    "MAT_LAW43_HILL_TAB": check_mat_law43,
    82: check_mat_law82,
    "82": check_mat_law82,
    "LAW82": check_mat_law82,
    "OGDEN": check_mat_law82,
    "LAW82_OGDEN": check_mat_law82,
    "MAT_LAW82": check_mat_law82,
    "MAT_OGDEN": check_mat_law82,
    69: check_mat_law69,
    "69": check_mat_law69,
    "LAW69": check_mat_law69,
    "HYP_ELAS": check_mat_law69,
    "HYPERELASTIC": check_mat_law69,
    "LAW69_HYPERELASTIC": check_mat_law69,
    "LAW69_HYP_ELAS": check_mat_law69,
    "MAT_LAW69": check_mat_law69,
    "MAT_HYP_ELAS": check_mat_law69,
    "MAT_HYPERELASTIC": check_mat_law69,
    60: check_mat_law60,
    "60": check_mat_law60,
    "LAW60": check_mat_law60,
    "PLAS_T3": check_mat_law60,
    "MAT_PLAS_T3": check_mat_law60,
    "FABRIC": check_mat_law60,
    "MAT_FABRIC": check_mat_law60,
    "MAT_LAW60": check_mat_law60,
    48: check_mat_law48,
    "48": check_mat_law48,
    "LAW48": check_mat_law48,
    "ZHAO": check_mat_law48,
    "PLAS_ZHAO": check_mat_law48,
    "MAT_ZHAO": check_mat_law48,
    "MAT_PLAS_ZHAO": check_mat_law48,
    "MAT_LAW48": check_mat_law48,
    "LAW48_ZHAO": check_mat_law48,
    58: check_mat_law58,
    "58": check_mat_law58,
    "LAW58": check_mat_law58,
    "FABR_A": check_mat_law58,
    "FABRIC_A": check_mat_law58,
    "MAT_LAW58": check_mat_law58,
    "MAT_FABR_A": check_mat_law58,
    "MAT_FABRIC_A": check_mat_law58,
    "LAW58_FABR_A": check_mat_law58,
    52: check_mat_law52,
    "52": check_mat_law52,
    "LAW52": check_mat_law52,
    "GURSON": check_mat_law52,
    "PLAS_GURS": check_mat_law52,
    "MAT_LAW52": check_mat_law52,
    "MAT_GURSON": check_mat_law52,
    "MAT_PLAS_GURS": check_mat_law52,
    "LAW52_GURSON": check_mat_law52,
    57: check_mat_law57,
    "57": check_mat_law57,
    "LAW57": check_mat_law57,
    "BARLAT3": check_mat_law57,
    "MAT_BARLAT3": check_mat_law57,
    "LAW57_BARLAT3": check_mat_law57,
    21: check_mat_law21,
    "21": check_mat_law21,
    "LAW21": check_mat_law21,
    "DPRAG": check_mat_law21,
    "MAT_DPRAG": check_mat_law21,
    "LAW21_DPRAG": check_mat_law21,
    "DUCKHUB": check_mat_law21,
    "MAT_LAW21": check_mat_law21,
    49: check_mat_law49,
    "49": check_mat_law49,
    "LAW49": check_mat_law49,
    "STEINB": check_mat_law49,
    "STEINBERG": check_mat_law49,
    "STEINBERG_GUINAN": check_mat_law49,
    "MAT_LAW49": check_mat_law49,
    "MAT_STEINB": check_mat_law49,
    "MAT_STEINBERG": check_mat_law49,
    "LAW49_STEINB": check_mat_law49,
    79: check_mat_law79,
    "79": check_mat_law79,
    "LAW79": check_mat_law79,
    "JOHN_HOLM": check_mat_law79,
    "JOHNSON_HOLMQUIST": check_mat_law79,
    "JH2": check_mat_law79,
    "MAT_LAW79": check_mat_law79,
    "MAT_JOHN_HOLM": check_mat_law79,
    "LAW79_JOHN_HOLM": check_mat_law79,
    95: check_mat_law95,
    "95": check_mat_law95,
    "LAW95": check_mat_law95,
    "BERGSTROM_BOYCE": check_mat_law95,
    "MAT_LAW95": check_mat_law95,
    "MAT_BERGSTROM_BOYCE": check_mat_law95,
    "LAW95_BERGSTROM_BOYCE": check_mat_law95,
    100: check_mat_law100,
    "100": check_mat_law100,
    "LAW100": check_mat_law100,
    "VISC_HYP": check_mat_law100,
    "MNF": check_mat_law100,
    "MAT_100": check_mat_law100,
    "MAT_LAW100": check_mat_law100,
    "MAT_VISC_HYP": check_mat_law100,
    "MAT_MNF": check_mat_law100,
    "LAW100_VISC_HYP": check_mat_law100,
    "LAW100_MNF": check_mat_law100,
    101: check_mat_law101,
    "101": check_mat_law101,
    "LAW101": check_mat_law101,
    "PP": check_mat_law101,
    "MAT_PP": check_mat_law101,
    "PLAS_POLY": check_mat_law101,
    "MAT_PLAS_POLY": check_mat_law101,
    "MAT_101": check_mat_law101,
    "MAT_LAW101": check_mat_law101,
    "LAW101_PLAS_POLY": check_mat_law101,
    "LAW101_PP": check_mat_law101,
    102: check_mat_law102,
    "102": check_mat_law102,
    "LAW102": check_mat_law102,
    "DPRAG2": check_mat_law102,
    "DRUCKER_PRAGER_2": check_mat_law102,
    "MAT_DPRAG2": check_mat_law102,
    "MAT_102": check_mat_law102,
    "MAT_LAW102": check_mat_law102,
    "LAW102_DPRAG2": check_mat_law102,
    103: check_mat_law103,
    "103": check_mat_law103,
    "LAW103": check_mat_law103,
    "HENSEL_SPITTEL": check_mat_law103,
    "HENSEL-SPITTEL": check_mat_law103,
    "PLAS_HENS": check_mat_law103,
    "MAT_HENSEL_SPITTEL": check_mat_law103,
    "MAT_PLAS_HENS": check_mat_law103,
    "MAT_103": check_mat_law103,
    "MAT_LAW103": check_mat_law103,
    "HEN": check_mat_law103,
    104: check_mat_law104,
    "104": check_mat_law104,
    "LAW104": check_mat_law104,
    "DRUCKER": check_mat_law104,
    "JOHNS_VOCE_DRUCKER": check_mat_law104,
    "JOHNS-VOCE-DRUCKER": check_mat_law104,
    "PLAS_DRUCK": check_mat_law104,
    "MAT_DRUCKER": check_mat_law104,
    "MAT_JOHNS_VOCE_DRUCKER": check_mat_law104,
    "MAT_PLAS_DRUCK": check_mat_law104,
    "MAT_104": check_mat_law104,
    "MAT_LAW104": check_mat_law104,
    "LAW104_DRUCKER": check_mat_law104,
    105: check_mat_law105,
    "105": check_mat_law105,
    "LAW105": check_mat_law105,
    "POWDER_BURN": check_mat_law105,
    "POWDERBURN": check_mat_law105,
    "MAT_POWDER_BURN": check_mat_law105,
    "MAT_POWDERBURN": check_mat_law105,
    "MAT_105": check_mat_law105,
    "MAT_LAW105": check_mat_law105,
    "LAW105_POWDER_BURN": check_mat_law105,
    106: check_mat_law106,
    "106": check_mat_law106,
    "LAW106": check_mat_law106,
    "JCOOK_ALM": check_mat_law106,
    "JOHNS_COOK_ALM": check_mat_law106,
    "MAT_JCOOK_ALM": check_mat_law106,
    "MAT_106": check_mat_law106,
    "MAT_LAW106": check_mat_law106,
    "LAW106_JCOOK_ALM": check_mat_law106,
    107: check_mat_law107,
    "107": check_mat_law107,
    "LAW107": check_mat_law107,
    "PAPER_LIGHT": check_mat_law107,
    "PLAS_PAPER_LIGHT": check_mat_law107,
    "PFEIFFER": check_mat_law107,
    "MAT_PAPER_LIGHT": check_mat_law107,
    "MAT_PLAS_PAPER_LIGHT": check_mat_law107,
    "MAT_PFEIFFER": check_mat_law107,
    "MAT_107": check_mat_law107,
    "MAT_LAW107": check_mat_law107,
    "LAW107_PAPER_LIGHT": check_mat_law107,
    109: check_mat_law109,
    "109": check_mat_law109,
    "LAW109": check_mat_law109,
    "TAB_PLAS": check_mat_law109,
    "ELASTO_PLAS_TAB": check_mat_law109,
    "LAW109_TAB_PLAS": check_mat_law109,
    "MLAW109": check_mat_law109,
    "MAT_LAW109": check_mat_law109,
    "MAT_TAB_PLAS": check_mat_law109,
    "MAT_ELASTO_PLAS_TAB": check_mat_law109,
    "MAT_109": check_mat_law109,
    110: check_mat_law110,
    "110": check_mat_law110,
    "LAW110": check_mat_law110,
    "VEGTER": check_mat_law110,
    "PLAS_VEGTER": check_mat_law110,
    "LAW110_VEGTER": check_mat_law110,
    "MLAW110": check_mat_law110,
    "MAT_LAW110": check_mat_law110,
    "MAT_VEGTER": check_mat_law110,
    "MAT_PLAS_VEGTER": check_mat_law110,
    "MAT_110": check_mat_law110,
    76: check_mat_law76,
    "76": check_mat_law76,
    "LAW76": check_mat_law76,
    "SAMP": check_mat_law76,
    "SAMP-1": check_mat_law76,
    "SAMP_1": check_mat_law76,
    "MAT_76": check_mat_law76,
    "MAT_LAW76": check_mat_law76,
    "MAT_SAMP": check_mat_law76,
    "MAT_SAMP-1": check_mat_law76,
    "MAT_SAMP_1": check_mat_law76,
    "LAW76_SAMP": check_mat_law76,
    "LAW76_SAMP-1": check_mat_law76,
    "LAW76_SAMP_1": check_mat_law76,
    124: check_mat_law124,
    "124": check_mat_law124,
    "LAW124": check_mat_law124,
    "CDPM2": check_mat_law124,
    "MAT_LAW124": check_mat_law124,
    "MAT_CDPM2": check_mat_law124,
    "LAW124_CDPM2": check_mat_law124,
    126: check_mat_law126,
    "126": check_mat_law126,
    "LAW126": check_mat_law126,
    "JOHNSON_HOLMQUIST_CONCRETE": check_mat_law126,
    "MAT_LAW126": check_mat_law126,
    "MAT_JOHNSON_HOLMQUIST_CONCRETE": check_mat_law126,
    "LAW126_JOHNSON_HOLMQUIST_CONCRETE": check_mat_law126,
    "HJC": check_mat_law126,
    "MAT_HJC": check_mat_law126,
    169: check_mat_law169,
    "169": check_mat_law169,
    "LAW169": check_mat_law169,
    "ARUP_ADHESIVE": check_mat_law169,
    "MAT_LAW169": check_mat_law169,
    "MAT_ARUP_ADHESIVE": check_mat_law169,
    "LAW169_ARUP_ADHESIVE": check_mat_law169,
    "ARUP": check_mat_law169,
    "ADHESIVE": check_mat_law169,
}


def check_model(model: Model, log: MessageLog) -> None:
    if model.numnod == 0:
        log.error("model has no nodes", "MODEL CHECK")
    elif len(model.node_ids) > 0:
        u_ids, counts = np.unique(model.node_ids, return_counts=True)
        if np.any(counts > 1):
            dups = u_ids[counts > 1]
            log.error(f"Duplicate node ID detected: {dups[:5]}", "NODE CHECK")

    for attr in ("solids", "tetra10", "bric20"):
        group = getattr(model, attr, None)
        if group is not None and hasattr(group, "vol0"):
            neg = group.vol0 <= 0.0
            if np.any(neg):
                bad_ids = group.ids[neg]
                log.error(f"Negative or null volume detected in {len(bad_ids)} solid element(s): {bad_ids[:5]}", "GEOMETRY CHECK")

    if not any(True for _ in model.element_groups()):
        log.warning("model has no elements (deck may use only unported "
                    "element types)", "MODEL CHECK")

    # M546: LAW12 is 3D solid only, invalid for 2D formulations (hm_read_mat12.F:167)
    if getattr(model, "n2d", 0) > 0:
        for mid, mat in getattr(model, "materials", {}).items():
            if getattr(mat, "law", None) in (12, "12", "LAW12", "3PARBI", "3D_COMP", "COMP_3D", "RAGAB") or getattr(mat, "law_name", None) in ("12", "LAW12", "3PARBI", "3D_COMP", "COMP_3D", "RAGAB", "MAT_LAW12", "MAT_3D_COMP", "MAT_COMP_3D"):
                log.error(f"/MAT/LAW12/{mid}: LAW12 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")
        for mid, mat12 in getattr(model, "mat_law12s", {}).items():
            if mid not in getattr(model, "materials", {}):
                log.error(f"/MAT/LAW12/{mid}: LAW12 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")

    # M547: LAW14 is 3D solid only, invalid for 2D formulations (hm_read_mat14.F:151)
    if getattr(model, "n2d", 0) > 0:
        for mid, mat in getattr(model, "materials", {}).items():
            if getattr(mat, "law", None) in (14, "14", "LAW14", "COMPSO", "COMP_SOL") or getattr(mat, "law_name", None) in ("14", "LAW14", "COMPSO", "COMP_SOL", "MAT_LAW14", "MAT_COMPSO", "MAT_COMP_SOL"):
                log.error(f"/MAT/LAW14/{mid}: LAW14 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")
        for mid, mat14 in getattr(model, "mat_law14s", {}).items():
            if mid not in getattr(model, "materials", {}):
                log.error(f"/MAT/LAW14/{mid}: LAW14 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")

    # M556: LAW21 is 3D solid only, invalid for 2D formulations (hm_read_mat21.F:223)
    if getattr(model, "n2d", 0) > 0:
        for mid, mat in getattr(model, "materials", {}).items():
            if getattr(mat, "law", None) in (21, "21", "LAW21", "DPRAG", "MAT_DPRAG", "LAW21_DPRAG", "DUCKHUB") or getattr(mat, "law_name", None) in ("21", "LAW21", "DPRAG", "MAT_DPRAG", "LAW21_DPRAG", "DUCKHUB", "MAT_LAW21", "MAT_DPRAG"):
                log.error(f"/MAT/LAW21/{mid}: LAW21 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")
        for mid, mat21 in getattr(model, "mat_law21s", {}).items():
            if mid not in getattr(model, "materials", {}):
                log.error(f"/MAT/LAW21/{mid}: LAW21 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")
        for mid, mat_dp in getattr(model, "mat_dprags", {}).items():
            if mid not in getattr(model, "materials", {}) and mid not in getattr(model, "mat_law21s", {}):
                log.error(f"/MAT/LAW21/{mid}: LAW21 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")

    # M557: LAW49 is 3D solid only, invalid for 2D formulations (hm_read_mat49.F)
    if getattr(model, "n2d", 0) > 0:
        for mid, mat in getattr(model, "materials", {}).items():
            if getattr(mat, "law", None) in (49, "49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB") or getattr(mat, "law_name", None) in ("49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB"):
                log.error(f"/MAT/LAW49/{mid}: LAW49 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")
        for mid, mat49 in getattr(model, "mat_law49s", {}).items():
            if mid not in getattr(model, "materials", {}):
                log.error(f"/MAT/LAW49/{mid}: LAW49 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")

    # M558: LAW79 is 3D solid and SPH only, invalid for 2D formulations (hm_read_mat79.F)
    if getattr(model, "n2d", 0) > 0:
        for mid, mat in getattr(model, "materials", {}).items():
            if getattr(mat, "law", None) in (79, "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM") or getattr(mat, "law_name", None) in ("79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM"):
                log.error(f"/MAT/LAW79/{mid}: LAW79 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")
        for mid, mat79 in getattr(model, "mat_law79s", {}).items():
            if mid not in getattr(model, "materials", {}):
                log.error(f"/MAT/LAW79/{mid}: LAW79 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")

    # M559: LAW50 is 3D solid only, invalid for 2D formulations (hm_read_mat50.F90)
    if getattr(model, "n2d", 0) > 0:
        for mid, mat in getattr(model, "materials", {}).items():
            if getattr(mat, "law", None) in (50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM", "LAW50_VISC_HONEY", "LAW50_HYP_FOAM") or getattr(mat, "law_name", None) in ("50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM", "LAW50_VISC_HONEY", "LAW50_HYP_FOAM"):
                log.error(f"/MAT/LAW50/{mid}: LAW50 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")
        for mid, mat50 in getattr(model, "mat_law50s", {}).items():
            if mid not in getattr(model, "materials", {}):
                log.error(f"/MAT/LAW50/{mid}: LAW50 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")

    # M560: LAW163 is 3D solid only, invalid for 2D formulations (hm_read_mat163.F90)
    if getattr(model, "n2d", 0) > 0:
        for mid, mat in getattr(model, "materials", {}).items():
            if getattr(mat, "law", None) in (163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM") or getattr(mat, "law_name", None) in ("163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM"):
                log.error(f"/MAT/LAW163/{mid}: LAW163 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")
        for mid, mat163 in getattr(model, "mat_law163s", {}).items():
            if mid not in getattr(model, "materials", {}):
                log.error(f"/MAT/LAW163/{mid}: LAW163 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")

    # M563: LAW74 is 3D solid only, invalid for 2D formulations (hm_read_mat74.F)
    if getattr(model, "n2d", 0) > 0:
        for mid, mat in getattr(model, "materials", {}).items():
            if getattr(mat, "law", None) in (74, "74", "LAW74", "HILL_3D", "ORTH_PLAS") or getattr(mat, "law_name", None) in ("74", "LAW74", "HILL_3D", "ORTH_PLAS", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS"):
                log.error(f"/MAT/LAW74/{mid}: LAW74 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")
        for mid, mat74 in getattr(model, "mat_law74s", {}).items():
            if mid not in getattr(model, "materials", {}):
                log.error(f"/MAT/LAW74/{mid}: LAW74 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")

    # M569: LAW95 is 3D solid only, invalid for 2D formulations (hm_read_mat95.F)
    if getattr(model, "n2d", 0) > 0:
        for mid, mat in getattr(model, "materials", {}).items():
            if getattr(mat, "law", None) in (95, "95", "LAW95", "BERGSTROM_BOYCE") or getattr(mat, "law_name", None) in ("95", "LAW95", "BERGSTROM_BOYCE", "MAT_LAW95", "MAT_BERGSTROM_BOYCE", "LAW95_BERGSTROM_BOYCE"):
                log.error(f"/MAT/LAW95/{mid}: LAW95 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")
        for mid, mat95 in getattr(model, "mat_law95s", {}).items():
            if mid not in getattr(model, "materials", {}):
                log.error(f"/MAT/LAW95/{mid}: LAW95 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")

    # M570: LAW100 is 3D solid only, invalid for 2D formulations (hm_read_mat100.F)
    if getattr(model, "n2d", 0) > 0:
        for mid, mat in getattr(model, "materials", {}).items():
            if getattr(mat, "law", None) in (100, "100", "LAW100", "VISC_HYP", "MNF") or getattr(mat, "law_name", None) in ("100", "LAW100", "VISC_HYP", "MNF", "MAT_100", "MAT_LAW100", "MAT_VISC_HYP", "MAT_MNF", "LAW100_VISC_HYP", "LAW100_MNF"):
                log.error(f"/MAT/LAW100/{mid}: LAW100 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")
        for mid, mat100 in getattr(model, "mat_law100s", {}).items():
            if mid not in getattr(model, "materials", {}):
                log.error(f"/MAT/LAW100/{mid}: LAW100 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")

    # M571: LAW101 is 3D solid only, invalid for 2D formulations (hm_read_mat101.F)
    if getattr(model, "n2d", 0) > 0:
        for mid, mat in getattr(model, "materials", {}).items():
            if getattr(mat, "law", None) in _LAW101_KEYS or getattr(mat, "law_name", None) in _LAW101_KEYS:
                log.error(f"/MAT/LAW101/{mid}: LAW101 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")
        for mid, mat101 in getattr(model, "mat_law101s", {}).items():
            if mid not in getattr(model, "materials", {}):
                log.error(f"/MAT/LAW101/{mid}: LAW101 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")

    # M572: LAW102 is 3D solid only, invalid for 2D formulations (hm_read_mat102.F)
    if getattr(model, "n2d", 0) > 0:
        for mid, mat in getattr(model, "materials", {}).items():
            if getattr(mat, "law", None) in _LAW102_KEYS or getattr(mat, "law_name", None) in _LAW102_KEYS:
                log.error(f"/MAT/LAW102/{mid}: LAW102 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")
        for mid, mat102 in getattr(model, "mat_law102s", {}).items():
            if mid not in getattr(model, "materials", {}):
                log.error(f"/MAT/LAW102/{mid}: LAW102 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")

    # M573: LAW103 is 3D solid only, invalid for 2D formulations (hm_read_mat103.F)
    if getattr(model, "n2d", 0) > 0:
        for mid, mat in getattr(model, "materials", {}).items():
            if getattr(mat, "law", None) in _LAW103_KEYS or getattr(mat, "law_name", None) in _LAW103_KEYS:
                log.error(f"/MAT/LAW103/{mid}: LAW103 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")
        for mid, mat103 in getattr(model, "mat_law103s", {}).items():
            if mid not in getattr(model, "materials", {}):
                log.error(f"/MAT/LAW103/{mid}: LAW103 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")


    # material law vs element family compatibility (fail in the Starter
    # with a clear message instead of a NotImplementedError mid-run)
    for name, group in model.element_groups():
        allowed = _ALLOWED_LAWS.get(name)
        if allowed is None and name != "springs":
            continue
        for sl, mat, prop in group.state["slices"]:
            if name == "springs" and (getattr(mat, "id", 0) == 0 or getattr(mat, "law", 0) in (0, -1)):
                continue
            mat_law = getattr(mat, "law", None)
            if mat_law == 999 and getattr(mat, "eos", None) is None:
                # M37 pack 1: a /MAT/GAS on elements has no pressure or
                # stiffness of its own — the initial state must come
                # from an /EOS/IDEAL-GAS card (or the programmatic
                # P0/T0/RHO0 params); see materials/mat_gas.py.  Checked
                # before the density (a bare gas card has neither).
                log.error(f"/MAT/GAS/{getattr(mat, 'id', 0)} on {name} elements needs "
                          f"an /EOS/IDEAL-GAS card for its initial "
                          f"state (P0, gamma) — a bare gas card has no "
                          f"element pressure", "MAT CHECK")
            rho0 = getattr(mat, "rho0", 0.0) or 0.0
            if rho0 <= 0.0 and mat_law not in _NULL_RHO0_OK_LAWS:
                law_name = getattr(mat, "law_name", None) \
                    or f"LAW{mat_law}"
                log.error(f"/MAT/{law_name}/{mat.id} on {name} elements: "
                          f"zero or missing initial density "
                          f"(RHO0={rho0:g}) — element masses cannot be "
                          f"initialized", "MAT CHECK")
                continue
            if getattr(mat, "inactive", False):
                law_name = getattr(mat, "law_name", f"LAW{mat.law}")
                log.warning(f"/MAT/{law_name}/{mat.id} on {name} "
                            f"elements: parsed, physics not implemented "
                            f"(M37) — the Engine will refuse to run "
                            f"this model", "MAT CHECK")
                continue
            if (mat.law in (38, "38", "LAW38", "VISC_TAB")
                    or getattr(mat, "law_name", None) in ("LAW38", "VISC_TAB")):
                if name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "trusses", "beams"):
                    log.error(
                        f"/MAT/LAW38/{mat.id} (/MAT/VISC_TAB) is not supported for {name} elements "
                        f"(solids only: bricks, tetras, penta6, pyra5)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (32, "32", "LAW32", "HILL")
                    or getattr(mat, "law_name", None) in ("32", "LAW32", "HILL", "MAT_LAW32", "MAT_HILL")):
                if name in ("bricks", "tetras", "penta6", "penta6s", "pyra5", "trusses", "beams"):
                    log.error(
                        f"/MAT/LAW32/{mat.id} (/MAT/HILL) is not supported for {name} elements "
                        f"(shells only: shells, shells_qbat, shells_qeph, sh3n)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS")
                    or getattr(mat, "law_name", None) in ("25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", "MAT_LAW25", "MAT_COMP_PLAS", "MAT_COMPSH", "MAT_TSAI_WU", "MAT_CRASURV", "MAT_COMPOSITE_PLAS")):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW25/{mat.id} (/MAT/COMP_PLAS) is not supported for {name} elements "
                        f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG")
                    or getattr(mat, "law_name", None) in ("15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG", "MAT_LAW15", "MAT_CHANG", "MAT_PLAS_ANISO", "MAT_COMP_CHANG")):
                if name in ("bricks", "tetras", "penta6", "penta6s", "pyra5", "trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW15/{mat.id} (/MAT/CHANG) is not supported for {name} elements "
                        f"(shells only: shells, shells_qbat, shells_qeph, sh3n, quads)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (22, "22", "LAW22", "DAMA", "PLAS_DAMA")
                    or getattr(mat, "law_name", None) in ("22", "LAW22", "DAMA", "PLAS_DAMA", "MAT_LAW22", "MAT_DAMA", "MAT_PLAS_DAMA")):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW22/{mat.id} (/MAT/DAMA) is not supported for {name} elements "
                        f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (12, "12", "LAW12", "3PARBI", "3D_COMP", "COMP_3D", "RAGAB")
                    or getattr(mat, "law_name", None) in ("12", "LAW12", "3PARBI", "3D_COMP", "COMP_3D", "RAGAB", "MAT_LAW12", "MAT_3D_COMP", "MAT_COMP_3D", "MAT_3PARBI", "MAT_RAGAB")):
                if name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads", "trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW12/{mat.id} (/MAT/3D_COMP) is not supported for {name} elements "
                        f"(solids only: bricks, tetras, penta6, pyra5)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (14, "14", "LAW14", "COMPSO", "COMP_SOL")
                    or getattr(mat, "law_name", None) in ("14", "LAW14", "COMPSO", "COMP_SOL", "MAT_LAW14", "MAT_COMPSO", "MAT_COMP_SOL")):
                if name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads", "trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW14/{mat.id} (/MAT/COMPSO) is not supported for {name} elements "
                        f"(solids only: bricks, tetras, penta6, pyra5)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB")
                    or getattr(mat, "law_name", None) in ("43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB")):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW43/{mat.id} (/MAT/HILL_TAB) is not supported for {name} elements "
                        f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (82, "82", "LAW82", "OGDEN", "LAW82_OGDEN")
                    or getattr(mat, "law_name", None) in ("82", "LAW82", "OGDEN", "LAW82_OGDEN", "MAT_LAW82", "MAT_OGDEN", "MAT_LAW82_OGDEN")):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW82/{mat.id} (/MAT/OGDEN) is not supported for {name} elements "
                        f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYP_ELAS")
                    or getattr(mat, "law_name", None) in ("69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYP_ELAS", "MAT_LAW69", "MAT_HYP_ELAS", "MAT_HYPERELASTIC", "MAT_LAW69_HYP_ELAS")):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW69/{mat.id} (/MAT/HYP_ELAS) is not supported for {name} elements "
                        f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (92, "92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE")
                    or getattr(mat, "law_name", None) in ("92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE", "MAT_LAW92", "MAT_ARRUDA_BOYCE", "LAW92_ARRUDA_BOYCE")):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW92/{mat.id} (/MAT/ARRUDA_BOYCE) is not supported for {name} elements "
                        f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (93, "93", "LAW93", "ORTH_HILL")
                    or getattr(mat, "law_name", None) in ("93", "LAW93", "ORTH_HILL", "MAT_LAW93", "MAT_ORTH_HILL", "LAW93_ORTH_HILL")):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW93/{mat.id} (/MAT/ORTH_HILL) is not supported for {name} elements "
                        f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (94, "94", "LAW94", "YEOH")
                    or getattr(mat, "law_name", None) in ("94", "LAW94", "YEOH", "MAT_LAW94", "MAT_YEOH", "LAW94_YEOH")):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW94/{mat.id} (/MAT/YEOH) is not supported for {name} elements "
                        f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (95, "95", "LAW95", "BERGSTROM_BOYCE")
                    or getattr(mat, "law_name", None) in ("95", "LAW95", "BERGSTROM_BOYCE", "MAT_LAW95", "MAT_BERGSTROM_BOYCE", "LAW95_BERGSTROM_BOYCE")):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW95/{mat.id} (/MAT/BERGSTROM_BOYCE) is not supported for {name} elements (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    continue
                if name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
                    log.error(
                        f"/MAT/LAW95/{mat.id} (/MAT/BERGSTROM_BOYCE) is not supported for {name} elements "
                        f"(solids only: bricks, tetras, penta6, pyra5) (ANCMSG 305)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (100, "100", "LAW100", "VISC_HYP", "MNF")
                    or getattr(mat, "law_name", None) in ("100", "LAW100", "VISC_HYP", "MNF", "MAT_100", "MAT_LAW100", "MAT_VISC_HYP", "MAT_MNF", "LAW100_VISC_HYP", "LAW100_MNF")):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW100/{mat.id} (/MAT/VISC_HYP) is not supported for {name} elements (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    continue
                if name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
                    log.error(
                        f"/MAT/LAW100/{mat.id} (/MAT/VISC_HYP) is not supported for {name} elements "
                        f"(solids only: bricks, tetras, penta6, pyra5) (ANCMSG 305)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in _LAW101_KEYS
                    or getattr(mat, "law_name", None) in _LAW101_KEYS):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW101/{mat.id} (/MAT/PP) is not supported for {name} elements (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    continue
                if name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
                    log.error(
                        f"/MAT/LAW101/{mat.id} (/MAT/PP) is not supported for {name} elements "
                        f"(solids only: bricks, tetras, penta6, pyra5) (ANCMSG 305)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (60, "60", "LAW60", "PLAS_T3", "FABRIC", "MAT_PLAS_T3", "MAT_FABRIC")
                    or getattr(mat, "law_name", None) in ("60", "LAW60", "PLAS_T3", "FABRIC", "MAT_PLAS_T3", "MAT_FABRIC", "MAT_LAW60")):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW60/{mat.id} (/MAT/PLAS_T3) is not supported for {name} elements "
                        f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "MAT_PLAS_ZHAO")
                    or getattr(mat, "law_name", None) in ("48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "MAT_PLAS_ZHAO", "MAT_LAW48", "LAW48_ZHAO")):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW48/{mat.id} (/MAT/ZHAO) is not supported for {name} elements "
                        f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (58, "58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A")
                    or getattr(mat, "law_name", None) in ("58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A")):
                if name in ("bricks", "tetras", "penta6", "penta6s", "pyra5"):
                    log.error(
                        f"/MAT/LAW58/{mat.id} (/MAT/FABR_A) is not supported for solid elements ({name}) (ANCMSG 305)",
                        "MAT CHECK",
                    )
                    continue
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW58/{mat.id} (/MAT/FABR_A) is not supported for 1D elements ({name}) (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS", "LAW52_GURSON")
                    or getattr(mat, "law_name", None) in ("52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS", "LAW52_GURSON")):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW52/{mat.id} (/MAT/GURSON) is not supported for {name} elements "
                        f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (57, "57", "LAW57", "BARLAT3", "MAT_BARLAT3", "MAT_LAW57", "LAW57_BARLAT3")
                    or getattr(mat, "law_name", None) in ("57", "LAW57", "BARLAT3", "MAT_BARLAT3", "MAT_LAW57", "LAW57_BARLAT3")):
                if name in ("bricks", "bricks_heph", "bric20s", "tetras", "tetra10s", "penta6", "penta6s", "pyra5"):
                    log.error(
                        f"/MAT/LAW57/{mat.id} (/MAT/BARLAT3) is not supported for solid elements ({name}) (ANCMSG 305)",
                        "MAT CHECK",
                    )
                    continue
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW57/{mat.id} (/MAT/BARLAT3) is not supported for 1D elements ({name}) (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (21, "21", "LAW21", "DPRAG", "MAT_DPRAG", "LAW21_DPRAG", "DUCKHUB")
                    or getattr(mat, "law_name", None) in ("21", "LAW21", "DPRAG", "MAT_DPRAG", "LAW21_DPRAG", "DUCKHUB", "MAT_LAW21", "MAT_DPRAG")):
                if name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
                    log.error(
                        f"/MAT/LAW21/{mat.id} (/MAT/DPRAG) is not supported for shell elements ({name}) (ANCMSG 305)",
                        "MAT CHECK",
                    )
                    continue
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW21/{mat.id} (/MAT/DPRAG) is not supported for 1D elements ({name}) (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM")
                    or getattr(mat, "law_name", None) in ("163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM")):
                if name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
                    log.error(
                        f"/MAT/LAW163/{mat.id} (/MAT/CRUSHABLE_FOAM) is not supported for shell elements ({name}) (ANCMSG 305)",
                        "MAT CHECK",
                    )
                    continue
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW163/{mat.id} (/MAT/CRUSHABLE_FOAM) is not supported for 1D elements ({name}) (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL")
                    or getattr(mat, "law_name", None) in ("73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL")):
                if name in ("bricks", "bricks_heph", "bric20s", "tetras", "tetra10s", "penta6", "penta6s", "pyra5", "solids"):
                    log.error(
                        f"/MAT/LAW73/{mat.id} (/MAT/HILL_THERM) is not supported for solid elements ({name}) (ANCMSG 305)",
                        "MAT CHECK",
                    )
                    continue
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW73/{mat.id} (/MAT/HILL_THERM) is not supported for 1D elements ({name}) (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000")
                    or getattr(mat, "law_name", None) in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000")):
                if name in ("bricks", "bricks_heph", "bric20s", "tetras", "tetra10s", "penta6", "penta6s", "pyra5", "solids"):
                    log.error(
                        f"/MAT/LAW87/{mat.id} (/MAT/BARLAT2000) is not supported for solid elements ({name}) (ANCMSG 305)",
                        "MAT CHECK",
                    )
                    continue
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW87/{mat.id} (/MAT/BARLAT2000) is not supported for 1D elements ({name}) (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (74, "74", "LAW74", "HILL_3D", "ORTH_PLAS", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS")
                    or getattr(mat, "law_name", None) in ("74", "LAW74", "HILL_3D", "ORTH_PLAS", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS")):
                if name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
                    log.error(
                        f"/MAT/LAW74/{mat.id} (/MAT/HILL_3D) is not supported for shell elements ({name}) (ANCMSG 305)",
                        "MAT CHECK",
                    )
                    continue
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW74/{mat.id} (/MAT/HILL_3D) is not supported for 1D elements ({name}) (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in _LAW105_KEYS
                    or getattr(mat, "law_name", None) in _LAW105_KEYS):
                if name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads", "tria3", "shells_bt4"):
                    log.error(
                        f"/MAT/LAW105/{mat.id} (/MAT/POWDER_BURN) is not supported for shell elements ({name}) (ANCMSG 305)",
                        "MAT CHECK",
                    )
                    continue
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW105/{mat.id} (/MAT/POWDER_BURN) is not supported for 1D elements ({name}) (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in _LAW106_KEYS
                    or getattr(mat, "law_name", None) in _LAW106_KEYS):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW106/{mat.id} (/MAT/JCOOK_ALM) is not supported for 1D elements ({name}) (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in _LAW107_KEYS
                    or getattr(mat, "law_name", None) in _LAW107_KEYS):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW107/{mat.id} (/MAT/PAPER_LIGHT) is not supported for 1D elements ({name}) (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in _LAW109_KEYS
                    or getattr(mat, "law_name", None) in _LAW109_KEYS):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW109/{mat.id} (/MAT/TAB_PLAS) is not supported for 1D elements ({name}) (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in _LAW110_KEYS
                    or getattr(mat, "law_name", None) in _LAW110_KEYS):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW110/{mat.id} (/MAT/VEGTER) is not supported for 1D elements ({name}) (ANCMSG 306)",
                        "MAT CHECK",
                    )
                    continue
                if name in ("bricks", "tetras", "penta6", "pyra5", "solids", "solids_heph", "solids_tetra4"):
                    log.error(
                        f"/MAT/LAW110/{mat.id} (/MAT/VEGTER) is not supported for solid elements ({name}) (ANCMSG 305)",
                        "MAT CHECK",
                    )
                    continue
            if allowed is None:
                continue
            if mat.law not in allowed:
                log.error(f"material LAW{mat.law} (/MAT {mat.id}) is not "
                          f"ported for {name} elements (supported: "
                          f"{sorted(allowed, key=str)})", "MAT CHECK")
            if getattr(mat, "fail", None) is not None and name in ("trusses", "springs",
                                                 "beams"):
                log.warning(f"/FAIL on /MAT {getattr(mat, 'id', 0)} is ignored for {name} "
                            f"(failure is ported for solids and shells)",
                            "MAT CHECK")

    check_materials(model, log)



    # M38: element groups that reference a parsed-but-not-implemented
    # PROPERTY (InactiveProperty) — the Starter accepts them (params +
    # safe geometry read, mass init works); the Engine refuses to run the
    # model (see prop_reader.refuse_inactive_properties), mirroring the
    # inactive-material warning above.
    for name, group in model.element_groups():
        for sl, mat, prop in group.state["slices"]:
            if getattr(prop, "inactive", False):
                pn = getattr(prop, "prop_name", None) or f"TYPE{prop.type}"
                log.warning(f"/PROP/{pn}/{prop.id} on {name} elements: "
                            f"parsed, physics not implemented (M38) — the "
                            f"Engine will refuse to run this model",
                            "PROP CHECK")

    def need_group(gid, who):
        if gid is not None and gid != 0 and gid not in model.node_groups:
            log.error(f"{who}: node group {gid} not defined", "CROSS REF")

    def need_funct(fid, who):
        if fid is not None and fid > 0 and fid not in model.functions:
            log.error(f"{who}: function {fid} not defined", "CROSS REF")

    for bc in model.bcs:
        need_group(bc.grnod_id, f"/BCS/{bc.id}")
    for iv in model.inivel:
        need_group(iv.grnod_id, f"/INIVEL/{iv.id}")
    for gv in model.gravity:
        need_group(gv.grnod_id, f"/GRAV/{gv.id}")
        need_funct(gv.funct_id, f"/GRAV/{gv.id}")
    for cl in model.cloads:
        need_group(cl.grnod_id, f"/CLOAD/{cl.id}")
        need_funct(cl.funct_id, f"/CLOAD/{cl.id}")
    for imp in model.impvel:
        need_group(imp.grnod_id, f"/IMPVEL/{imp.id}")
        need_funct(imp.funct_id, f"/IMPVEL/{imp.id}")
    for imp in model.impdisp:
        need_group(imp.grnod_id, f"/IMPDISP/{imp.id}")
        need_funct(imp.funct_id, f"/IMPDISP/{imp.id}")
    for imp in model.impacc:
        need_group(imp.grnod_id, f"/IMPACC/{imp.id}")
        need_funct(imp.funct_id, f"/IMPACC/{imp.id}")
    for it in model.imptemp:
        need_group(it.grnod_id, f"/IMPTEMP/{it.id}")
        need_funct(it.funct_id, f"/IMPTEMP/{it.id}")
    for cl in model.centri_loads:
        need_group(cl.grnod_id, f"/LOAD/CENTRI/{cl.id}")
        need_funct(cl.funct_id, f"/LOAD/CENTRI/{cl.id}")
    for pl in model.ploads:
        need_funct(pl.funct_id, f"/PLOAD/{pl.id}")
        if pl.surf_id not in model.surfaces:
            log.error(f"/PLOAD/{pl.id}: surface {pl.surf_id} not defined",
                      "CROSS REF")
    for conv in model.convec_loads:
        need_funct(conv.funct_id, f"/CONVEC/{conv.id}")
        if conv.surf_id not in model.surfaces:
            log.error(f"/CONVEC/{conv.id}: surface {conv.surf_id} not defined",
                      "CROSS REF")
    for rad in model.radiation_loads:
        if rad.funct_id:
            need_funct(rad.funct_id, f"/RADIATION/{rad.id}")
        if rad.surf_id not in model.surfaces:
            log.error(f"/RADIATION/{rad.id}: surface {rad.surf_id} not defined",
                      "CROSS REF")
    for fl in model.impflux_loads:
        if fl.funct_id:
            need_funct(fl.funct_id, f"/IMPFLUX/{fl.id}")
        if fl.surf_id and fl.surf_id not in model.surfaces:
            log.error(f"/IMPFLUX/{fl.id}: surface {fl.surf_id} not defined",
                      "CROSS REF")
        if fl.grbric_id and fl.grbric_id not in model.egroups.get("BRIC", {}):
            log.error(f"/IMPFLUX/{fl.id}: brick group {fl.grbric_id} not defined",
                      "CROSS REF")
    for itemp in model.initemp:
        if itemp.grnod_id:
            need_group(itemp.grnod_id, f"/INITEMP/{itemp.id}")
    for iv in model.inivol:
        if iv.part_id and iv.part_id not in model.parts:
            log.error(f"/INIVOL/{iv.id}: part {iv.part_id} not defined",
                      "CROSS REF")
        for c in iv.containers:
            if c.surf_id not in model.surfaces and c.surf_id not in model.node_groups:
                log.error(f"/INIVOL/{iv.id}: container surface/group {c.surf_id} not defined",
                          "CROSS REF")
    solid_ids = set()
    for name in ("bricks", "bricks_heph", "tetras", "tetra10s", "bric20s", "penta6s"):
        grp = getattr(model, name, None)
        if grp is not None and hasattr(grp, "ids") and len(grp.ids) > 0:
            solid_ids.update(grp.ids.tolist())
    for elem_id in model.ini_bricks:
        if elem_id not in solid_ids:
            log.error(f"/INIBRI: solid element {elem_id} not defined",
                      "CROSS REF")

    shell_ids = set()
    for name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "sh3n_dkt18", "quads", "shel16s"):
        grp = getattr(model, name, None)
        if grp is not None and hasattr(grp, "ids") and len(grp.ids) > 0:
            shell_ids.update(grp.ids.tolist())
    for elem_id in model.ini_shells:
        if elem_id not in shell_ids:
            log.error(f"/INISHE: shell element {elem_id} not defined",
                      "CROSS REF")

    truss_ids = set(model.trusses.ids.tolist()) if model.trusses is not None and hasattr(model.trusses, "ids") and len(model.trusses.ids) > 0 else set()
    for elem_id in model.ini_trusses:
        if elem_id not in truss_ids:
            log.error(f"/INITRU: truss element {elem_id} not defined",
                      "CROSS REF")

    beam_ids = set(model.beams.ids.tolist()) if model.beams is not None and hasattr(model.beams, "ids") and len(model.beams.ids) > 0 else set()
    for elem_id in model.ini_beams:
        if elem_id not in beam_ids:
            log.error(f"/INIBEA: beam element {elem_id} not defined",
                      "CROSS REF")

    spring_ids = set(model.springs.ids.tolist()) if model.springs is not None and hasattr(model.springs, "ids") and len(model.springs.ids) > 0 else set()
    for elem_id in model.ini_springs:
        if elem_id not in spring_ids:
            log.error(f"/INISPR: spring element {elem_id} not defined",
                      "CROSS REF")

    for sens in model.sensors:
        who = f"/SENSOR/{sens.kind}/{sens.id}"
        if sens.kind in ("DISP", "VEL") and sens.node_id and sens.node_id not in model._id2idx:
            log.error(f"{who}: unknown node {sens.node_id}", "CROSS REF")
        elif sens.kind == "DIST":
            if sens.node_id1 and sens.node_id1 not in model._id2idx:
                log.error(f"{who}: unknown node 1 {sens.node_id1}", "CROSS REF")
            if sens.node_id2 and sens.node_id2 not in model._id2idx:
                log.error(f"{who}: unknown node 2 {sens.node_id2}", "CROSS REF")
        elif sens.kind == "ENERGY" and sens.part_id and sens.part_id not in model.parts:
            log.error(f"{who}: unknown part {sens.part_id}", "CROSS REF")
        elif sens.kind == "TEMP" and sens.grnod_id:
            need_group(sens.grnod_id, who)
        elif sens.kind == "GAUGE":
            for gid, _, _ in sens.gauge_entries:
                if gid > 0 and gid not in model.gauge_points and gid not in model.gauges:
                    log.error(f"{who}: unknown gauge {gid}", "CROSS REF")
        elif sens.kind == "HIC":
            if sens.accel_id > 0 and sens.accel_id not in model.accelerometers:
                log.error(f"{who}: unknown accelerometer {sens.accel_id}", "CROSS REF")
        elif sens.kind == "WORK":
            if sens.node_id1 and sens.node_id1 not in model._id2idx:
                log.error(f"{who}: unknown node 1 {sens.node_id1}", "CROSS REF")
            if sens.node_id2 and sens.node_id2 not in model._id2idx:
                log.error(f"{who}: unknown node 2 {sens.node_id2}", "CROSS REF")
        elif sens.kind == "RWALL":
            rw_ids = {rw.id for rw in model.rwalls}
            if sens.rwall_id > 0 and sens.rwall_id not in rw_ids:
                log.error(f"{who}: unknown rigid wall {sens.rwall_id}", "CROSS REF")
        elif sens.kind in ("XSECTION", "CROSSSECTION", "SECT"):
            sect_ids = {s.id for s in model.sections}
            if sens.sect_id > 0 and sens.sect_id not in sect_ids:
                log.error(f"{who}: unknown section {sens.sect_id}", "CROSS REF")
        elif sens.kind == "DIST_SURF":
            if sens.node_id1 and sens.node_id1 not in model._id2idx:
                log.error(f"{who}: unknown node 1 {sens.node_id1}", "CROSS REF")

    for am in model.admas:
        # mass_type 0/1: grnod_id is a node group; 2: surface; 3/4: part
        # group; 6/7: single part.  Only types 0/1 cross-ref node_groups.
        if am.mass_type in (0, 1):
            need_group(am.grnod_id, f"/ADMAS/{am.id}")
    for rb in model.rbodies:
        need_group(rb.grnod_id, f"/{rb.kind}/{rb.id}")
        if rb.master_id not in model._id2idx:
            log.error(f"/{rb.kind}/{rb.id}: unknown master node "
                      f"{rb.master_id}", "CROSS REF")
    for r3 in model.rbe3:
        need_group(r3.grnod_id, f"/RBE3/{r3.id}")
        if r3.ref_id not in model._id2idx:
            log.error(f"/RBE3/{r3.id}: unknown reference node {r3.ref_id}",
                      "CROSS REF")
    for sc in model.sections:
        need_group(sc.grnod_id, f"/SECT/{sc.id}")
        if sc.node_id_ref and sc.node_id_ref not in model._id2idx:
            log.error(f"/SECT/{sc.id}: unknown reference node "
                      f"{sc.node_id_ref}", "CROSS REF")
    for rw in model.rwalls:
        need_group(rw.grnod_id, f"/RWALL/{rw.id}")
        if rw.grnod_id2:
            need_group(rw.grnod_id2, f"/RWALL/{rw.id}")
        if rw.node_id and rw.node_id not in model._id2idx:
            log.error(f"/RWALL/{rw.id}: unknown wall node {rw.node_id}",
                      "CROSS REF")
    for itf in model.interfaces:
        who = f"/INTER/TYPE{itf.type}/{itf.id}"
        if itf.type in (7, 2):
            # TYPE7 allows grnod_id = 0 (self-impact: secondary side
            # defaults to the main surface's own nodes); TYPE2 does not.
            if itf.type == 2 or itf.grnod_id != 0:
                need_group(itf.grnod_id, who)
            if itf.type == 2 and itf.grnod_id == 0:
                log.error(f"{who}: a tied interface needs a secondary "
                          f"node group", "CROSS REF")
            if itf.surf_id not in model.surfaces:
                log.error(f"{who}: surface {itf.surf_id} not defined",
                          "CROSS REF")
        elif itf.type == 11:
            for lid in (itf.line_id1, itf.line_id2):
                if lid not in model.lines:
                    log.error(f"{who}: line {lid} not defined", "CROSS REF")
        elif itf.type == 24:
            if itf.grnod_id != 0:
                need_group(itf.grnod_id, who)
            for sid in (itf.surf_id1, itf.surf_id):
                if sid != 0 and sid not in model.surfaces:
                    log.error(f"{who}: surface {sid} not defined", "CROSS REF")
        elif itf.type in (1, 3, 6, 12, 15, 20, 21, 23):
            for sid in (itf.surf_id, itf.surf_id1):
                if sid > 0 and sid not in model.surfaces:
                    log.error(f"{who}: surface {sid} not defined", "CROSS REF")
            if itf.grnod_id > 0 and itf.grnod_id not in model.node_groups:
                log.error(f"{who}: node group {itf.grnod_id} not defined", "CROSS REF")
        elif itf.type in (5, 14):
            if itf.grnod_id > 0 and itf.grnod_id not in model.node_groups:
                log.error(f"{who}: node group {itf.grnod_id} not defined", "CROSS REF")
            if itf.surf_id > 0 and itf.surf_id not in model.surfaces:
                log.error(f"{who}: surface {itf.surf_id} not defined", "CROSS REF")
        elif itf.type == 22:
            if itf.surf_id > 0 and itf.surf_id not in model.surfaces:
                log.error(f"{who}: surface {itf.surf_id} not defined", "CROSS REF")
    for th in model.th_requests:
        if th.kind == "NODE":
            for nid in th.ids:
                if nid not in model._id2idx:
                    log.error(f"/TH/NODE/{th.id}: unknown node {nid}",
                              "CROSS REF")
        elif th.kind == "PART":
            for pid in th.ids:
                if pid not in model.parts:
                    log.error(f"/TH/PART/{th.id}: unknown part {pid}",
                              "CROSS REF")
        elif th.kind == "SECT":
            defined = {s.id for s in model.sections}
            for sid in th.ids:
                if sid not in defined:
                    log.error(f"/TH/SECT/{th.id}: unknown section {sid}",
                              "CROSS REF")

    # Cyclic boundary conditions (M99)
    for cid, cb in getattr(model, "cyclic_bcs", {}).items():
        if cb.grnod1_id not in model.node_groups:
            log.error(f"/BCS/CYCLIC/{cid}: node group 1 {cb.grnod1_id} not defined", "CROSS REF")
        if cb.grnod2_id not in model.node_groups:
            log.error(f"/BCS/CYCLIC/{cid}: node group 2 {cb.grnod2_id} not defined", "CROSS REF")
        if cb.skew_id > 0 and cb.skew_id not in model.skews:
            log.error(f"/BCS/CYCLIC/{cid}: skew {cb.skew_id} not defined", "CROSS REF")

    # Solid part perturbations (M99)
    part_groups = model.egroups.get("PART", {})
    for pid, pt in getattr(model, "perturbations", {}).items():
        if pt.grpart_id > 0 and pt.grpart_id not in part_groups and pt.grpart_id not in model.parts:
            log.error(f"/PERTURB/PART/SOLID/{pid}: part group {pt.grpart_id} not defined", "CROSS REF")

    # Blast loads (M99)
    for bid, pb in getattr(model, "pblast_loads", {}).items():
        if pb.surf_id > 0 and pb.surf_id not in model.surfaces:
            log.error(f"/LOAD/PBLAST/{bid}: surface {pb.surf_id} not defined", "CROSS REF")
        if pb.node_id > 0 and pb.node_id not in model._id2idx:
            log.error(f"/LOAD/PBLAST/{bid}: detonation node {pb.node_id} not defined", "CROSS REF")

    # Plies & Laminates (M100)
    for ply_id, ply in getattr(model, "plies", {}).items():
        if ply.mat_id > 0 and ply.mat_id not in model.materials:
            log.error(f"/PLY/{ply_id}: material {ply.mat_id} not defined", "CROSS REF")
        if ply.skew_id > 0 and ply.skew_id not in model.skews:
            log.error(f"/PLY/{ply_id}: skew {ply.skew_id} not defined", "CROSS REF")

    for lam_id, lam in getattr(model, "laminates", {}).items():
        for lp in lam.plies:
            if lp.ply_id not in model.plies:
                log.error(f"/LAMINATE/{lam_id}: ply {lp.ply_id} not defined", "CROSS REF")
            if lp.mat_interply > 0 and lp.mat_interply not in model.materials:
                log.error(f"/LAMINATE/{lam_id}: interply material {lp.mat_interply} not defined", "CROSS REF")

    # Sub-interfaces (M100)
    inter_ids = {itf.id for itf in model.interfaces}
    for sub in getattr(model, "sub_interfaces", []):
        if sub.inter_id not in inter_ids:
            log.error(f"/INTER/SUB/{sub.id}: main interface {sub.inter_id} not defined", "CROSS REF")
        if sub.main_id1 > 0 and sub.main_id1 not in model.surfaces and sub.main_id1 not in model.lines:
            log.error(f"/INTER/SUB/{sub.id}: main entity 1 {sub.main_id1} not defined", "CROSS REF")
        if sub.main_id2 > 0 and sub.main_id2 not in model.surfaces:
            log.error(f"/INTER/SUB/{sub.id}: main entity 2 {sub.main_id2} not defined", "CROSS REF")

    # Guided cables (M149)
    for gcid, gc in getattr(model, "guided_cables", {}).items():
        if gc.grnod_id > 0 and gc.grnod_id not in model.node_groups:
            log.error(f"/INTER/GUIDED_CABLE/{gcid}: node group {gc.grnod_id} not defined", "CROSS REF")
        if gc.grpart_id > 0 and gc.grpart_id not in model.egroups.get("PART", {}) and gc.grpart_id not in model.parts:
            log.error(f"/INTER/GUIDED_CABLE/{gcid}: part group {gc.grpart_id} not defined", "CROSS REF")

    # Composite properties (M100)
    for prop_id, prop in model.properties.items():
        if prop.type in (10, 11, 16, 6) and hasattr(prop, "params"):
            sk = prop.params.get("skew_id", 0)
            if sk > 0 and sk not in model.skews:
                log.error(f"/PROP/TYPE{prop.type}/{prop_id}: skew {sk} not defined", "CROSS REF")
        if prop.type == 11 and hasattr(prop, "params"):
            for ly in prop.params.get("layers", []):
                mid = ly.get("mat_id", 0)
                if mid > 0 and mid not in model.materials:
                    log.error(f"/PROP/TYPE11/{prop_id}: layer material {mid} not defined", "CROSS REF")

    # Shell and Failure Perturbations & SMS (M101)
    part_groups = model.egroups.get("PART", {})
    for pid, ps in getattr(model, "perturb_shells", {}).items():
        if ps.grpart_id > 0 and ps.grpart_id not in part_groups and ps.grpart_id not in model.parts:
            log.error(f"/PERTURB/PART/SHELL/{pid}: part group/part {ps.grpart_id} not defined", "CROSS REF")

    fail_ids = {mat_id for mat_id, _, _ in getattr(model, "raw_fails", [])}
    for m in model.materials.values():
        if getattr(m, "fail", None) is not None or bool(getattr(m, "fail_models", None)):
            fail_ids.add(m.id)
    for pid, pf in getattr(model, "perturb_fails", {}).items():
        if pf.fail_id > 0 and pf.fail_id not in fail_ids:
            log.error(f"/PERTURB/FAIL/{pf.fail_type}/{pid}: failure criterion {pf.fail_id} not defined", "CROSS REF")

    if getattr(model, "sms_global", None) is not None:
        sms = model.sms_global
        if sms.grpart_id > 0 and sms.grpart_id not in part_groups and sms.grpart_id not in model.parts:
            log.error(f"/SMS: part group/part {sms.grpart_id} not defined", "CROSS REF")

    # Boundary conditions, Joints, Merge, Inicrack, Laser (M102)
    for bid, bcs in getattr(model, "bcs_nrf", {}).items():
        if bcs.grnod_id > 0 and bcs.grnod_id not in model.node_groups:
            log.error(f"/BCS/NRF/{bid}: node group {bcs.grnod_id} not defined", "CROSS REF")

    sensor_ids = {s.id for s in model.sensors}
    for bid, bcs in getattr(model, "bcs_walls", {}).items():
        if bcs.grnod_id > 0 and bcs.grnod_id not in model.node_groups:
            log.error(f"/BCS/WALL/{bid}: node group {bcs.grnod_id} not defined", "CROSS REF")
        if bcs.sensor_id > 0 and bcs.sensor_id not in sensor_ids:
            log.error(f"/BCS/WALL/{bid}: sensor {bcs.sensor_id} not defined", "CROSS REF")

    for rid, rl in getattr(model, "rlinks", {}).items():
        if rl.grnod_id > 0 and rl.grnod_id not in model.node_groups:
            log.error(f"/RLINK/{rid}: node group {rl.grnod_id} not defined", "CROSS REF")
        if rl.skew_id > 0 and rl.skew_id not in model.skews:
            log.error(f"/RLINK/{rid}: skew {rl.skew_id} not defined", "CROSS REF")

    for cid, cj in getattr(model, "cyl_joints", {}).items():
        if cj.node_id1 > 0 and cj.node_id1 not in model._id2idx:
            log.error(f"/CYL_JOINT/{cid}: node {cj.node_id1} not defined", "CROSS REF")
        if cj.node_id2 > 0 and cj.node_id2 not in model._id2idx:
            log.error(f"/CYL_JOINT/{cid}: node {cj.node_id2} not defined", "CROSS REF")
        if cj.grnod_id > 0 and cj.grnod_id not in model.node_groups:
            log.error(f"/CYL_JOINT/{cid}: node group {cj.grnod_id} not defined", "CROSS REF")

    for gid, gj in getattr(model, "gjoints", {}).items():
        for nid in (gj.node_id0, gj.node_id1, gj.node_id2, gj.node_id3):
            if nid > 0 and nid not in model._id2idx:
                log.error(f"/GJOINT/{gid}: node {nid} not defined", "CROSS REF")

    for mid, mn in getattr(model, "node_merges", {}).items():
        if mn.grnod_id > 0 and mn.grnod_id not in model.node_groups:
            log.error(f"/MERGE/NODE/{mid}: node group {mn.grnod_id} not defined", "CROSS REF")

    for iid, ic in getattr(model, "inicracks", {}).items():
        for seg in ic.segments:
            if seg.node_id1 > 0 and seg.node_id1 not in model._id2idx:
                log.error(f"/INICRACK/{iid}: node {seg.node_id1} not defined", "CROSS REF")
            if seg.node_id2 > 0 and seg.node_id2 not in model._id2idx:
                log.error(f"/INICRACK/{iid}: node {seg.node_id2} not defined", "CROSS REF")

    for lid, las in getattr(model, "laser_loads", {}).items():
        if las.curve_id > 0 and las.curve_id not in model.functions:
            log.error(f"/LASER/{lid}: function {las.curve_id} not defined", "CROSS REF")
        if las.fct_id_target > 0 and las.fct_id_target not in model.functions:
            log.error(f"/LASER/{lid}: function {las.fct_id_target} not defined", "CROSS REF")

    # Specialized loads, Preload, Damping & Controls (M103)
    for lid, pcyl in getattr(model, "pcyl_loads", {}).items():
        if pcyl.surf_id > 0 and pcyl.surf_id not in model.surfaces:
            log.error(f"/LOAD/PCYL/{lid}: surface {pcyl.surf_id} not defined", "CROSS REF")
        if pcyl.sens_id > 0 and pcyl.sens_id not in sensor_ids:
            log.error(f"/LOAD/PCYL/{lid}: sensor {pcyl.sens_id} not defined", "CROSS REF")
        if pcyl.frame_id > 0 and pcyl.frame_id not in model.skews:
            log.error(f"/LOAD/PCYL/{lid}: skew {pcyl.frame_id} not defined", "CROSS REF")
        if pcyl.table_id > 0 and pcyl.table_id not in model.tables:
            log.error(f"/LOAD/PCYL/{lid}: table {pcyl.table_id} not defined", "CROSS REF")

    for lid, pf in getattr(model, "pfluid_loads", {}).items():
        if pf.surf_id > 0 and pf.surf_id not in model.surfaces:
            log.error(f"/LOAD/PFLUID/{lid}: surface {pf.surf_id} not defined", "CROSS REF")
        if pf.sens_id > 0 and pf.sens_id not in sensor_ids:
            log.error(f"/LOAD/PFLUID/{lid}: sensor {pf.sens_id} not defined", "CROSS REF")
        if pf.fct_id_t > 0 and pf.fct_id_t not in model.functions:
            log.error(f"/LOAD/PFLUID/{lid}: function {pf.fct_id_t} not defined", "CROSS REF")
        if pf.fct_id_pc > 0 and pf.fct_id_pc not in model.functions:
            log.error(f"/LOAD/PFLUID/{lid}: function {pf.fct_id_pc} not defined", "CROSS REF")
        if pf.fct_id_vel > 0 and pf.fct_id_vel not in model.functions:
            log.error(f"/LOAD/PFLUID/{lid}: function {pf.fct_id_vel} not defined", "CROSS REF")
        if pf.frame_id > 0 and pf.frame_id not in model.skews:
            log.error(f"/LOAD/PFLUID/{lid}: skew {pf.frame_id} not defined", "CROSS REF")
        if pf.frame_id_vel > 0 and pf.frame_id_vel not in model.skews:
            log.error(f"/LOAD/PFLUID/{lid}: skew {pf.frame_id_vel} not defined", "CROSS REF")

    for pid, pr in getattr(model, "preloads", {}).items():
        if pr.sect_id > 0 and pr.sect_id not in model.sections and pr.sect_id not in model.properties:
            log.error(f"/PRELOAD/{pid}: section {pr.sect_id} not defined", "CROSS REF")
        if pr.sens_id > 0 and pr.sens_id not in sensor_ids:
            log.error(f"/PRELOAD/{pid}: sensor {pr.sens_id} not defined", "CROSS REF")
        if pr.fct_id > 0 and pr.fct_id not in model.functions:
            log.error(f"/PRELOAD/{pid}: function {pr.fct_id} not defined", "CROSS REF")

    for pid, pra in getattr(model, "preload_axials", {}).items():
        if pra.set_id > 0 and pra.set_id not in part_groups and pra.set_id not in model.parts:
            log.error(f"/PRELOAD/AXIAL/{pid}: part group/part {pra.set_id} not defined", "CROSS REF")
        if pra.sens_id > 0 and pra.sens_id not in sensor_ids:
            log.error(f"/PRELOAD/AXIAL/{pid}: sensor {pra.sens_id} not defined", "CROSS REF")
        if pra.fun_id > 0 and pra.fun_id not in model.functions:
            log.error(f"/PRELOAD/AXIAL/{pid}: function {pra.fun_id} not defined", "CROSS REF")

    for did, di in getattr(model, "damp_inters", {}).items():
        if di.grnod_id > 0 and di.grnod_id not in model.node_groups:
            log.error(f"/DAMP/INTER/{did}: node group {di.grnod_id} not defined", "CROSS REF")
        if di.skew_id > 0 and di.skew_id not in model.skews:
            log.error(f"/DAMP/INTER/{did}: skew {di.skew_id} not defined", "CROSS REF")

    for did, dr in getattr(model, "damp_ranges", {}).items():
        if dr.grpart_id > 0 and dr.grpart_id not in part_groups and dr.grpart_id not in model.parts:
            log.error(f"/DAMP/RANGE/{did}: part group/part {dr.grpart_id} not defined", "CROSS REF")

    for cid, caa in getattr(model, "caa_controls", {}).items():
        if caa.surf_id > 0 and caa.surf_id not in model.surfaces:
            log.error(f"/CAA/{cid}: surface {caa.surf_id} not defined", "CROSS REF")
        if caa.grnod_id > 0 and caa.grnod_id not in model.node_groups:
            log.error(f"/CAA/{cid}: node group {caa.grnod_id} not defined", "CROSS REF")
        if caa.sens_id > 0 and caa.sens_id not in sensor_ids:
            log.error(f"/CAA/{cid}: sensor {caa.sens_id} not defined", "CROSS REF")

    # Gauges, Clusters, Ext Links, Flexible Bodies & Initial Fields (M104)
    for gid, g in getattr(model, "gauges", {}).items():
        if g.node_id > 0 and g.node_id not in model._id2idx:
            log.error(f"/GAUGE/{gid}: node {g.node_id} not defined", "CROSS REF")

    for cid, c in getattr(model, "clusters", {}).items():
        if c.skew_id > 0 and c.skew_id not in model.skews:
            log.error(f"/CLUSTER/{cid}: skew {c.skew_id} not defined", "CROSS REF")

    for lid, el in getattr(model, "ext_links", {}).items():
        if el.grnod_id > 0 and el.grnod_id not in model.node_groups:
            log.error(f"/EXTLNK/{lid}: node group {el.grnod_id} not defined", "CROSS REF")

    for fid, fx in getattr(model, "fxbodies", {}).items():
        if fx.node_id > 0 and fx.node_id not in model._id2idx:
            log.error(f"/FXBODY/{fid}: node {fx.node_id} not defined", "CROSS REF")

    grav_ids = {g.id for g in model.gravity}
    for iid, ig in getattr(model, "ini_gravs", {}).items():
        if ig.grpart_id > 0 and ig.grpart_id not in part_groups and ig.grpart_id not in model.parts:
            log.error(f"/INIGRAV/{iid}: part group/part {ig.grpart_id} not defined", "CROSS REF")
        if ig.surf_id > 0 and ig.surf_id not in model.surfaces:
            log.error(f"/INIGRAV/{iid}: surface {ig.surf_id} not defined", "CROSS REF")
        if ig.grav_id > 0 and ig.grav_id not in grav_ids:
            log.error(f"/INIGRAV/{iid}: gravity {ig.grav_id} not defined", "CROSS REF")

    for mid, m1 in getattr(model, "ini_map1ds", {}).items():
        for nid in (m1.node_id1, m1.node_id2):
            if nid > 0 and nid not in model._id2idx:
                log.error(f"/INIMAP1D/{mid}: node {nid} not defined", "CROSS REF")

    for mid, m2 in getattr(model, "ini_map2ds", {}).items():
        for nid in (m2.node_id1, m2.node_id2, m2.node_id3):
            if nid > 0 and nid not in model._id2idx:
                log.error(f"/INIMAP2D/{mid}: node {nid} not defined", "CROSS REF")

    # Monitored Volumes, Airbag Leakage & ALE Controls (M105)
    for mid, mp in getattr(model, "monvol_pres", {}).items():
        if mp.surf_id > 0 and mp.surf_id not in model.surfaces:
            log.error(f"/MONVOL/PRES/{mid}: surface {mp.surf_id} not defined", "CROSS REF")
        if mp.fct_id > 0 and mp.fct_id not in model.functions:
            log.error(f"/MONVOL/PRES/{mid}: function {mp.fct_id} not defined", "CROSS REF")

    for mid, mg in getattr(model, "monvol_gases", {}).items():
        if mg.surf_id > 0 and mg.surf_id not in model.surfaces:
            log.error(f"/MONVOL/GAS/{mid}: surface {mg.surf_id} not defined", "CROSS REF")

    for mid, mc in getattr(model, "monvol_commus", {}).items():
        if mc.surf_id > 0 and mc.surf_id not in model.surfaces:
            log.error(f"/MONVOL/COMMU1/{mid}: surface {mc.surf_id} not defined", "CROSS REF")
        if mc.mat_id > 0 and mc.mat_id not in model.materials:
            log.error(f"/MONVOL/COMMU1/{mid}: material {mc.mat_id} not defined", "CROSS REF")

    for mid, ml in getattr(model, "monvol_lfluids", {}).items():
        if ml.surf_id > 0 and ml.surf_id not in model.surfaces:
            log.error(f"/MONVOL/LFLUID/{mid}: surface {ml.surf_id} not defined", "CROSS REF")
        for fid in (ml.fct_k, ml.fct_mtin, ml.fct_mtout, ml.fct_mpout, ml.fct_padd, ml.fct_pmax):
            if fid > 0 and fid not in model.functions:
                log.error(f"/MONVOL/LFLUID/{mid}: function {fid} not defined", "CROSS REF")

    for lid, lm in getattr(model, "leak_mats", {}).items():
        for fid in (lm.fct_id_e, lm.fct_id_lc, lm.fct_id_ac):
            if fid > 0 and fid not in model.functions:
                log.error(f"/LEAK/{lid}: function {fid} not defined", "CROSS REF")

    for lid, al in getattr(model, "ale_links", {}).items():
        if al.grnod_id > 0 and al.grnod_id not in model.node_groups:
            log.error(f"/ALE/LINK/{lid}: node group {al.grnod_id} not defined", "CROSS REF")
        if al.fct_id > 0 and al.fct_id not in model.functions:
            log.error(f"/ALE/LINK/{lid}: function {al.fct_id} not defined", "CROSS REF")

    # Seatbelts Suite: /RETRACTOR & /SLIPRING (M106)
    sensor_ids = {s.id for s in model.sensors}
    for rid, ret in getattr(model, "retractors", {}).items():
        if ret.node_id > 0 and ret.node_id not in model._id2idx:
            log.error(f"/RETRACTOR/{rid}: node {ret.node_id} not defined", "CROSS REF")
        if ret.sens_id1 > 0 and ret.sens_id1 not in sensor_ids:
            log.error(f"/RETRACTOR/{rid}: sensor {ret.sens_id1} not defined", "CROSS REF")
        if ret.sens_id2 > 0 and ret.sens_id2 not in sensor_ids:
            log.error(f"/RETRACTOR/{rid}: sensor {ret.sens_id2} not defined", "CROSS REF")
        for fid in (ret.fct_id1, ret.fct_id2, ret.fct_id3):
            if fid > 0 and fid not in model.functions:
                log.error(f"/RETRACTOR/{rid}: function {fid} not defined", "CROSS REF")

    for sid, sr in getattr(model, "sliprings", {}).items():
        if sr.subtype == "SPRING":
            for nid in (sr.node_id, sr.node_id2):
                if nid > 0 and nid not in model._id2idx:
                    log.error(f"/SLIPRING/{sid}: node {nid} not defined", "CROSS REF")
        elif sr.subtype == "SHELL":
            if sr.node_id > 0 and sr.node_id not in model.node_groups:
                log.error(f"/SLIPRING/{sid}: node group {sr.node_id} not defined", "CROSS REF")
        if sr.sens_id > 0 and sr.sens_id not in sensor_ids:
            log.error(f"/SLIPRING/{sid}: sensor {sr.sens_id} not defined", "CROSS REF")
        for fid in (sr.fct_id1, sr.fct_id2, sr.fct_id3, sr.fct_id4):
            if fid > 0 and fid not in model.functions:
                log.error(f"/SLIPRING/{sid}: function {fid} not defined", "CROSS REF")

    # M107: Advanced Failure Criteria & SENSOR/NIC
    for mat_id, fm, src in getattr(model, "raw_fails", []):
        if fm.type == "SAHRAEI":
            for fid_key in ("fct_ratio", "fct_elsize"):
                fid = fm.params.get(fid_key, 0)
                if fid > 0 and fid not in model.functions:
                    log.error(f"/FAIL/SAHRAEI on MAT/{mat_id}: function {fid} not defined", "CROSS REF")
        elif fm.type == "TAB2":
            for fid_key in ("epsf_id", "fct_exp"):
                fid = fm.params.get(fid_key, 0)
                if fid > 0 and fid not in model.functions:
                    log.error(f"/FAIL/TAB2 on MAT/{mat_id}: function {fid} not defined", "CROSS REF")
        elif fm.type == "GENE1":
            for fid_key in ("fct_idsm", "fct_idps"):
                fid = fm.params.get(fid_key, 0)
                if fid > 0 and fid not in model.functions:
                    log.error(f"/FAIL/GENE1 on MAT/{mat_id}: function {fid} not defined", "CROSS REF")

    spring_ids = set(model.springs.ids) if (getattr(model, "springs", None) is not None and model.springs.n > 0) else set()
    for sens in getattr(model, "sensors", []):
        if sens.kind == "NIC":
            if sens.spring_id > 0 and sens.spring_id not in spring_ids:
                log.error(f"/SENSOR/NIC/{sens.id}: spring {sens.spring_id} not defined", "CROSS REF")
            if sens.skew_id > 0 and sens.skew_id not in model.skews:
                log.error(f"/SENSOR/NIC/{sens.id}: skew {sens.skew_id} not defined", "CROSS REF")

    # M108: Classical Failure Models, Relative/Function Damping, FVM Airbags, Extended Contacts
    for mat_id, fm, src in getattr(model, "raw_fails", []):
        if fm.type == "ENERGY":
            fid = fm.params.get("fct_id", 0)
            if fid > 0 and fid not in model.functions:
                log.error(f"/FAIL/ENERGY on MAT/{mat_id}: function {fid} not defined", "CROSS REF")

    for d in getattr(model, "damps", []):
        if getattr(d, "kind", "GLOBAL") == "VREL":
            if d.grnod_id > 0 and d.grnod_id not in model.node_groups:
                log.error(f"/DAMP/VREL/{d.id}: node group {d.grnod_id} not defined", "CROSS REF")
            if d.skew_id > 0 and d.skew_id not in model.skews:
                log.error(f"/DAMP/VREL/{d.id}: skew {d.skew_id} not defined", "CROSS REF")
        elif getattr(d, "kind", "GLOBAL") == "FUNCT":
            if d.grnod_id > 0 and d.grnod_id not in model.node_groups:
                log.error(f"/DAMP/FUNCT/{d.id}: node group {d.grnod_id} not defined", "CROSS REF")
            if d.fct_id > 0 and d.fct_id not in model.functions:
                log.error(f"/DAMP/FUNCT/{d.id}: function {d.fct_id} not defined", "CROSS REF")

    for mid, fb in getattr(model, "monvol_fvmbags", {}).items():
        if fb.surf_id > 0 and fb.surf_id not in model.surfaces:
            log.error(f"/MONVOL/FVMBAG1/{mid}: surface {fb.surf_id} not defined", "CROSS REF")
        if fb.mat_id > 0 and fb.mat_id not in model.materials:
            log.error(f"/MONVOL/FVMBAG1/{mid}: material {fb.mat_id} not defined", "CROSS REF")

    for inter in getattr(model, "interfaces", []):
        if inter.type == 19:
            if inter.grnod_id > 0 and inter.grnod_id not in model.node_groups:
                log.error(f"/INTER/TYPE19/{inter.id}: node group {inter.grnod_id} not defined", "CROSS REF")
            if inter.surf_id > 0 and inter.surf_id not in model.surfaces:
                log.error(f"/INTER/TYPE19/{inter.id}: surface {inter.surf_id} not defined", "CROSS REF")
        elif inter.type == 21:
            if inter.surf_id > 0 and inter.surf_id not in model.surfaces:
                log.error(f"/INTER/TYPE21/{inter.id}: surface {inter.surf_id} not defined", "CROSS REF")
            if inter.surf_id1 > 0 and inter.surf_id1 not in model.surfaces:
                log.error(f"/INTER/TYPE21/{inter.id}: surface {inter.surf_id1} not defined", "CROSS REF")
        elif inter.type == 29:  # GUIDED_CABLE
            if inter.grnod_id > 0 and inter.grnod_id not in model.node_groups:
                log.error(f"/INTER/GUIDED_CABLE/{inter.id}: node group {inter.grnod_id} not defined", "CROSS REF")
            if inter.grpart_id > 0 and inter.grpart_id not in model.egroups.get("PART", {}) and inter.grpart_id not in model.parts:
                log.error(f"/INTER/GUIDED_CABLE/{inter.id}: part group {inter.grpart_id} not defined", "CROSS REF")

    # M109: Extended Multi-Physics Sensors & Properties
    for sens in getattr(model, "sensors", []):
        if sens.kind == "ENERGY":
            if sens.part_id > 0 and sens.part_id not in model.parts:
                log.error(f"/SENSOR/ENERGY/{sens.id}: part {sens.part_id} not defined", "CROSS REF")
            if sens.subset_id > 0 and sens.subset_id not in model.subsets:
                log.error(f"/SENSOR/ENERGY/{sens.id}: subset {sens.subset_id} not defined", "CROSS REF")
        elif sens.kind == "TEMP":
            if sens.grnod_id > 0 and sens.grnod_id not in model.node_groups:
                log.error(f"/SENSOR/TEMP/{sens.id}: node group {sens.grnod_id} not defined", "CROSS REF")

    for pid, prop in getattr(model, "properties", {}).items():
        if getattr(prop, "type", 0) in (21, 22):
            skew_id = prop.params.get("skew_id", 0)
            if skew_id > 0 and skew_id not in model.skews:
                log.error(f"/PROP/{pid}: skew {skew_id} not defined", "CROSS REF")
            for layer in prop.params.get("layers", []):
                mid = layer.get("mat_id", 0)
                if mid > 0 and mid not in model.materials:
                    log.error(f"/PROP/{pid}: material {mid} not defined in layer", "CROSS REF")

    # M110: Detonation Wavefronts, Air Blast Loading & Dynamic Element Activation
    for det in getattr(model, "detonations", []):
        if det.mat_id > 0 and det.mat_id not in model.materials:
            log.error(f"/INIT/DET_{det.kind}/{det.id}: material {det.mat_id} not defined", "CROSS REF")

    for pbid, pb in getattr(model, "pblast_loads", {}).items():
        if pb.surf_id > 0 and pb.surf_id not in model.surfaces:
            log.error(f"/LOAD/PBLAST/{pbid}: surface {pb.surf_id} not defined", "CROSS REF")
        if pb.surf_ground_id > 0 and pb.surf_ground_id not in model.surfaces:
            log.error(f"/LOAD/PBLAST/{pbid}: ground surface {pb.surf_ground_id} not defined", "CROSS REF")
        if pb.node_id > 0 and pb.node_id not in model._id2idx:
            log.error(f"/LOAD/PBLAST/{pbid}: node {pb.node_id} not defined", "CROSS REF")

    for act in getattr(model, "activations", []):
        if act.sens_id > 0 and act.sens_id not in sensor_ids:
            log.error(f"/ACTIV/{act.id}: sensor {act.sens_id} not defined", "CROSS REF")

    # M111: Extended Interfaces, Dual-Chamber Airbags & Autopositioning
    for mvid, mv in getattr(model, "monvol_fvmbag2s", {}).items():
        if mv.surf_id_ex > 0 and mv.surf_id_ex not in model.surfaces:
            log.error(f"/MONVOL/FVMBAG2/{mvid}: external surface {mv.surf_id_ex} not defined", "CROSS REF")
        if mv.surf_id_in > 0 and mv.surf_id_in not in model.surfaces:
            log.error(f"/MONVOL/FVMBAG2/{mvid}: internal surface {mv.surf_id_in} not defined", "CROSS REF")
        if mv.mat_id > 0 and mv.mat_id not in model.materials:
            log.error(f"/MONVOL/FVMBAG2/{mvid}: material {mv.mat_id} not defined", "CROSS REF")

    for ap in getattr(model, "autopositions", []):
        if ap.grnod_id > 0 and ap.grnod_id not in model.node_groups:
            log.error(f"/TRANSFORM/AUTOPOSITION/{ap.id}: node group {ap.grnod_id} not defined", "CROSS REF")
        if ap.surf_id > 0 and ap.surf_id not in model.surfaces:
            log.error(f"/TRANSFORM/AUTOPOSITION/{ap.id}: surface {ap.surf_id} not defined", "CROSS REF")
        if ap.skew_id > 0 and ap.skew_id not in model.skews:
            log.error(f"/TRANSFORM/AUTOPOSITION/{ap.id}: skew {ap.skew_id} not defined", "CROSS REF")

    # M112: Centrifugal & Pressure Loads, Advanced Initial Velocities, Final Geometry Imposed Fields, Thermal Rigid Walls, and SPH Boundary Suite
    for lcid, lc in getattr(model, "load_centris", {}).items():
        if lc.fct_id > 0 and lc.fct_id not in model.functions:
            log.error(f"/LOAD/CENTRI/{lcid}: function {lc.fct_id} not defined", "CROSS REF")
        if lc.sens_id > 0 and lc.sens_id not in sensor_ids:
            log.error(f"/LOAD/CENTRI/{lcid}: sensor {lc.sens_id} not defined", "CROSS REF")
        if lc.grnod_id > 0 and lc.grnod_id not in model.node_groups:
            log.error(f"/LOAD/CENTRI/{lcid}: node group {lc.grnod_id} not defined", "CROSS REF")
        if lc.frame_id > 0 and lc.frame_id not in model.skews:
            log.error(f"/LOAD/CENTRI/{lcid}: skew {lc.frame_id} not defined", "CROSS REF")

    for lpid, lp in getattr(model, "load_pfluids", {}).items():
        if lp.surf_id > 0 and lp.surf_id not in model.surfaces:
            log.error(f"/LOAD/PFLUID/{lpid}: surface {lp.surf_id} not defined", "CROSS REF")
        if lp.sens_id > 0 and lp.sens_id not in sensor_ids:
            log.error(f"/LOAD/PFLUID/{lpid}: sensor {lp.sens_id} not defined", "CROSS REF")
        for fid in (lp.fct_id_t, lp.fct_id_pc, lp.fct_id_vel):
            if fid > 0 and fid not in model.functions:
                log.error(f"/LOAD/PFLUID/{lpid}: function {fid} not defined", "CROSS REF")
        for fid in (lp.frame_id, lp.frame_id_vel):
            if fid > 0 and fid not in model.skews:
                log.error(f"/LOAD/PFLUID/{lpid}: skew {fid} not defined", "CROSS REF")

    for lpid, lp in getattr(model, "load_pressures", {}).items():
        if lp.surf_id > 0 and lp.surf_id not in model.surfaces:
            log.error(f"/LOAD/PRESSURE/{lpid}: surface {lp.surf_id} not defined", "CROSS REF")
        if lp.fct_id > 0 and lp.fct_id not in model.functions:
            log.error(f"/LOAD/PRESSURE/{lpid}: function {lp.fct_id} not defined", "CROSS REF")
        if lp.sens_id > 0 and lp.sens_id not in sensor_ids:
            log.error(f"/LOAD/PRESSURE/{lpid}: sensor {lp.sens_id} not defined", "CROSS REF")

    for iaid, ia in getattr(model, "inivel_axes", {}).items():
        if ia.grnod_id > 0 and ia.grnod_id not in model.node_groups:
            log.error(f"/INIVEL/AXIS/{iaid}: node group {ia.grnod_id} not defined", "CROSS REF")
        if ia.frame_id > 0 and ia.frame_id not in model.skews:
            log.error(f"/INIVEL/AXIS/{iaid}: skew {ia.frame_id} not defined", "CROSS REF")
        if ia.sens_id > 0 and ia.sens_id not in sensor_ids:
            log.error(f"/INIVEL/AXIS/{iaid}: sensor {ia.sens_id} not defined", "CROSS REF")

    for ivid, iv in getattr(model, "inivel_fvms", {}).items():
        if iv.skew_id > 0 and iv.skew_id not in model.skews:
            log.error(f"/INIVEL/FVM/{ivid}: skew {iv.skew_id} not defined", "CROSS REF")
        if iv.sens_id > 0 and iv.sens_id not in sensor_ids:
            log.error(f"/INIVEL/FVM/{ivid}: sensor {iv.sens_id} not defined", "CROSS REF")

    for inid, in_obj in getattr(model, "inivel_nodes", {}).items():
        for itm in in_obj.items:
            if itm.node_id > 0 and itm.node_id not in model._id2idx:
                log.error(f"/INIVEL/NODE/{inid}: node {itm.node_id} not defined", "CROSS REF")
            if itm.skew_id > 0 and itm.skew_id not in model.skews:
                log.error(f"/INIVEL/NODE/{inid}: skew {itm.skew_id} not defined", "CROSS REF")

    for idfid, idf in getattr(model, "impdisp_fgeos", {}).items():
        if idf.fct_id > 0 and idf.fct_id not in model.functions:
            log.error(f"/IMPDISP/FGEO/{idfid}: function {idf.fct_id} not defined", "CROSS REF")
        if idf.part_id > 0 and idf.part_id not in model.parts:
            log.error(f"/IMPDISP/FGEO/{idfid}: part {idf.part_id} not defined", "CROSS REF")
        if idf.sens_id > 0 and idf.sens_id not in sensor_ids:
            log.error(f"/IMPDISP/FGEO/{idfid}: sensor {idf.sens_id} not defined", "CROSS REF")
        for nd in idf.nodes:
            if nd["node_id"] > 0 and nd["node_id"] not in model._id2idx:
                log.error(f"/IMPDISP/FGEO/{idfid}: node {nd['node_id']} not defined", "CROSS REF")

    for ivfid, ivf in getattr(model, "impvel_fgeos", {}).items():
        if ivf.fct_id > 0 and ivf.fct_id not in model.functions:
            log.error(f"/IMPVEL/FGEO/{ivfid}: function {ivf.fct_id} not defined", "CROSS REF")
        if ivf.fct_l_id > 0 and ivf.fct_l_id not in model.functions:
            log.error(f"/IMPVEL/FGEO/{ivfid}: function {ivf.fct_l_id} not defined", "CROSS REF")
        if ivf.part_id > 0 and ivf.part_id not in model.parts:
            log.error(f"/IMPVEL/FGEO/{ivfid}: part {ivf.part_id} not defined", "CROSS REF")
        if ivf.sens_id > 0 and ivf.sens_id not in sensor_ids:
            log.error(f"/IMPVEL/FGEO/{ivfid}: sensor {ivf.sens_id} not defined", "CROSS REF")
        for n1, n2 in ivf.pairs:
            if n1 > 0 and n1 not in model._id2idx:
                log.error(f"/IMPVEL/FGEO/{ivfid}: node {n1} not defined", "CROSS REF")
            if n2 > 0 and n2 not in model._id2idx:
                log.error(f"/IMPVEL/FGEO/{ivfid}: node {n2} not defined", "CROSS REF")

    for rtid, rt in getattr(model, "rwall_therms", {}).items():
        if rt.node_id > 0 and rt.node_id not in model._id2idx:
            log.error(f"/RWALL/THERM/{rtid}: node {rt.node_id} not defined", "CROSS REF")
        if rt.grnod_id1 > 0 and rt.grnod_id1 not in model.node_groups:
            log.error(f"/RWALL/THERM/{rtid}: node group 1 {rt.grnod_id1} not defined", "CROSS REF")
        if rt.grnod_id2 > 0 and rt.grnod_id2 not in model.node_groups:
            log.error(f"/RWALL/THERM/{rtid}: node group 2 {rt.grnod_id2} not defined", "CROSS REF")
        if rt.fct_id > 0 and rt.fct_id not in model.functions:
            log.error(f"/RWALL/THERM/{rtid}: function {rt.fct_id} not defined", "CROSS REF")

    for sioid, sio in getattr(model, "sph_inouts", {}).items():
        if sio.surf_id > 0 and sio.surf_id not in model.surfaces:
            log.error(f"/SPH/INOUT/{sioid}: surface {sio.surf_id} not defined", "CROSS REF")
        if sio.part_id > 0 and sio.part_id not in model.parts:
            log.error(f"/SPH/INOUT/{sioid}: part {sio.part_id} not defined", "CROSS REF")
        if sio.fct_id > 0 and sio.fct_id not in model.functions:
            log.error(f"/SPH/INOUT/{sioid}: function {sio.fct_id} not defined", "CROSS REF")

    # M113: SPH Symmetry, Madymo Links/EXFEM, Random Noise, Accelerometers
    for sbid, sb in getattr(model, "sph_bcs", {}).items():
        if sb.frame_id > 0 and sb.frame_id not in model.skews:
            log.error(f"/SPHBCS/{sbid}: skew/frame {sb.frame_id} not defined", "CROSS REF")
        if sb.grnod_id > 0 and sb.grnod_id not in model.node_groups:
            log.error(f"/SPHBCS/{sbid}: node group {sb.grnod_id} not defined", "CROSS REF")

    for mlid, ml in getattr(model, "madymo_links", {}).items():
        if ml.node_id > 0 and ml.node_id not in model._id2idx:
            log.error(f"/MADYMO/LINK/{mlid}: node {ml.node_id} not defined", "CROSS REF")

    for meid, me in getattr(model, "madymo_exfems", {}).items():
        for pid in me.part_ids:
            if pid > 0 and pid not in model.parts:
                log.error(f"/MADYMO/EXFEM/{meid}: part {pid} not defined", "CROSS REF")

    for rn in getattr(model, "random_noises", []):
        if rn.grnod_id > 0 and rn.grnod_id not in model.node_groups:
            log.error(f"/RANDOM/GRNOD/{rn.grnod_id}: node group {rn.grnod_id} not defined", "CROSS REF")

    for aid, acc in getattr(model, "accelerometers", {}).items():
        if acc.node_id > 0 and acc.node_id not in model._id2idx:
            log.error(f"/ACCEL/{aid}: node {acc.node_id} not defined", "CROSS REF")
        if acc.skew_id > 0 and acc.skew_id not in model.skews:
            log.error(f"/ACCEL/{aid}: skew {acc.skew_id} not defined", "CROSS REF")

    # M114: Composite Failure, Propellant Combustion, Non-Uniform Added Mass, Extended Sections
    for mat_id, fc in getattr(model, "fail_composites", {}).items():
        if mat_id > 0 and mat_id not in model.materials:
            log.error(f"/FAIL/COMPOSITE/{mat_id}: material {mat_id} not defined", "CROSS REF")

    for pbid, pb in getattr(model, "ebcs_propellants", {}).items():
        if pb.surf_id > 0 and pb.surf_id not in model.surfaces:
            log.error(f"/EBCS/PROPELLANT/{pbid}: surface {pb.surf_id} not defined", "CROSS REF")
        if pb.sens_id > 0 and pb.sens_id not in sensor_ids:
            log.error(f"/EBCS/PROPELLANT/{pbid}: sensor {pb.sens_id} not defined", "CROSS REF")
        for fid in (pb.f_func_id, pb.g_func_id, pb.h_func_id):
            if fid > 0 and fid not in model.functions:
                log.error(f"/EBCS/PROPELLANT/{pbid}: function {fid} not defined", "CROSS REF")

    for anid, an in getattr(model, "admas_non_uniforms", {}).items():
        for item in an.items:
            if an.kind == "NODE" and item.entity_id > 0 and item.entity_id not in model._id2idx:
                log.error(f"/ADMAS/NON_UNIFORM/{anid}: node {item.entity_id} not defined", "CROSS REF")
            elif an.kind == "PART" and item.entity_id > 0 and item.entity_id not in model.parts:
                log.error(f"/ADMAS/NON_UNIFORM_PART/{anid}: part {item.entity_id} not defined", "CROSS REF")

    egroups_shel = {**getattr(model, "egroups", {}).get("GRSHEL", {}), **getattr(model, "egroups", {}).get("SHEL", {})}
    egroups_bric = {**getattr(model, "egroups", {}).get("GRBRIC", {}), **getattr(model, "egroups", {}).get("BRIC", {})}

    for scid, sc in getattr(model, "sect_circles", {}).items():
        for nid in (sc.n1, sc.n2, sc.n3):
            if nid > 0 and nid not in model._id2idx:
                log.error(f"/SECT/CIRCLE/{scid}: node {nid} not defined", "CROSS REF")
        if sc.grshel_id > 0 and sc.grshel_id not in egroups_shel:
            log.error(f"/SECT/CIRCLE/{scid}: shell group {sc.grshel_id} not defined", "CROSS REF")
        if sc.grbric_id > 0 and sc.grbric_id not in egroups_bric:
            log.error(f"/SECT/CIRCLE/{scid}: brick group {sc.grbric_id} not defined", "CROSS REF")

    for spid, sp in getattr(model, "sect_parals", {}).items():
        for nid in (sp.n1, sp.n2, sp.n3):
            if nid > 0 and nid not in model._id2idx:
                log.error(f"/SECT/PARAL/{spid}: node {nid} not defined", "CROSS REF")
        if sp.grshel_id > 0 and sp.grshel_id not in egroups_shel:
            log.error(f"/SECT/PARAL/{spid}: shell group {sp.grshel_id} not defined", "CROSS REF")
        if sp.grbric_id > 0 and sp.grbric_id not in egroups_bric:
            log.error(f"/SECT/PARAL/{spid}: brick group {sp.grbric_id} not defined", "CROSS REF")

    for maid, ma in getattr(model, "monvol_areas", {}).items():
        if ma.surf_id_ext > 0 and ma.surf_id_ext not in model.surfaces:
            log.error(f"/MONVOL/AREA/{maid}: surface {ma.surf_id_ext} not defined", "CROSS REF")

    for sid, s in getattr(model, "surfaces", {}).items():
        for mid in getattr(s, "mat_ids", []):
            if mid > 0 and mid not in model.materials:
                log.error(f"/SURF/{sid}: material {mid} not defined", "CROSS REF")
        for pid in getattr(s, "prop_ids", []):
            if pid > 0 and pid not in model.properties:
                log.error(f"/SURF/{sid}: property {pid} not defined", "CROSS REF")
        for bid in getattr(s, "box_ids", []):
            if bid > 0 and bid not in model.boxes:
                log.error(f"/SURF/{sid}: box {bid} not defined", "CROSS REF")

    for pid, sr in getattr(model, "sph_reserves", {}).items():
        if sr.part_id > 0 and sr.part_id not in model.parts:
            log.error(f"/SPH/RESERVE/{pid}: part {sr.part_id} not defined", "CROSS REF")

    for item in getattr(model, "move_functs", []):
        fid = item[0] if isinstance(item, (tuple, list)) else getattr(item, "id", 0)
        if fid > 0 and fid not in model.functions:
            log.error(f"/MOVE_FUNCT/{fid}: function {fid} not defined", "CROSS REF")

    for eid, em in getattr(model, "eigen_modes", {}).items():
        if em.grnod_id > 0 and em.grnod_id not in model.node_groups:
            log.error(f"/EIG/{eid}: node group {em.grnod_id} not defined", "CROSS REF")
        if em.grnod_bc > 0 and em.grnod_bc not in model.node_groups:
            log.error(f"/EIG/{eid}: node group {em.grnod_bc} not defined", "CROSS REF")

    for mid, ff in getattr(model, "fail_fractals", {}).items():
        if ff.mat_id > 0 and ff.mat_id not in model.materials:
            log.error(f"/FAIL/FRACTAL/{mid}: material {ff.mat_id} not defined", "CROSS REF")

    for tid, tp in getattr(model, "transform_positions", {}).items():
        if tp.grnod_id > 0 and tp.grnod_id not in model.node_groups:
            log.error(f"/TRANSFORM/{tid}: node group {tp.grnod_id} not defined", "CROSS REF")

    for lid, el in getattr(model, "external_links", {}).items():
        if el.grnod_id > 0 and el.grnod_id not in model.node_groups:
            log.error(f"/EXTERN/LINK/{lid}: node group {el.grnod_id} not defined", "CROSS REF")

    part_groups = model.egroups.get("PART", {})
    for fid, fm in getattr(model, "friction_models", {}).items():
        for p in fm.pairs:
            if p.part_id1 > 0 and p.part_id1 not in model.parts:
                log.error(f"/FRICTION/{fid}: part {p.part_id1} not defined", "CROSS REF")
            if p.part_id2 > 0 and p.part_id2 not in model.parts:
                log.error(f"/FRICTION/{fid}: part {p.part_id2} not defined", "CROSS REF")
            if p.grpart_id1 > 0 and p.grpart_id1 not in part_groups and p.grpart_id1 not in model.parts:
                log.error(f"/FRICTION/{fid}: part group {p.grpart_id1} not defined", "CROSS REF")
            if p.grpart_id2 > 0 and p.grpart_id2 not in part_groups and p.grpart_id2 not in model.parts:
                log.error(f"/FRICTION/{fid}: part group {p.grpart_id2} not defined", "CROSS REF")

    for bid, nb in getattr(model, "nbcs_blocks", {}).items():
        for n in nb.nodes:
            if n.node_id > 0 and n.node_id not in model._id2idx:
                log.error(f"/NBCS/{bid}: node {n.node_id} not defined", "CROSS REF")
            if n.skew_id > 0 and n.skew_id not in model.skews:
                log.error(f"/NBCS/{bid}: skew {n.skew_id} not defined", "CROSS REF")

    for nid in getattr(model, "refsta_nodes", {}):
        if nid > 0 and nid not in model._id2idx:
            log.error(f"/REFSTA: node {nid} not defined", "CROSS REF")

    # M150: EBCS, AMS, and Seatbelt Systems
    for pid, eb in getattr(model, "ebcs_pres", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/PRES/{pid}: surface {eb.surf_id} not defined", "CROSS REF")
        for fid in (eb.fct_pres, eb.fct_rho, eb.fct_en):
            if fid > 0 and fid not in model.functions:
                log.error(f"/EBCS/PRES/{pid}: function {fid} not defined", "CROSS REF")

    for vid, eb in getattr(model, "ebcs_vel", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/VEL/{vid}: surface {eb.surf_id} not defined", "CROSS REF")
        for fid in (eb.fct_vx, eb.fct_vy, eb.fct_vz, eb.fct_rho, eb.fct_en):
            if fid > 0 and fid not in model.functions:
                log.error(f"/EBCS/VEL/{vid}: function {fid} not defined", "CROSS REF")

    for iid, eb in getattr(model, "ebcs_inlets", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/INLET/{iid}: surface {eb.surf_id} not defined", "CROSS REF")
        if eb.funct_id > 0 and eb.funct_id not in model.functions:
            log.error(f"/EBCS/INLET/{iid}: function {eb.funct_id} not defined", "CROSS REF")

    for fid, eb in getattr(model, "ebcs_fluxouts", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/FLUXOUT/{fid}: surface {eb.surf_id} not defined", "CROSS REF")

    for gid, eb in getattr(model, "ebcs_gradp0", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/GRADP0/{gid}: surface {eb.surf_id} not defined", "CROSS REF")

    for nid, eb in getattr(model, "ebcs_normv", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/NORMV/{nid}: surface {eb.surf_id} not defined", "CROSS REF")
        if eb.funct_id > 0 and eb.funct_id not in model.functions:
            log.error(f"/EBCS/NORMV/{nid}: function {eb.funct_id} not defined", "CROSS REF")

    for vid, eb in getattr(model, "ebcs_valves", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/{eb.kind}/{vid}: surface {eb.surf_id} not defined", "CROSS REF")

    for mid, eb in getattr(model, "ebcs_monvols", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/MONVOL/{mid}: surface {eb.surf_id} not defined", "CROSS REF")
        if eb.monvol_id > 0 and eb.monvol_id not in getattr(model, "monitored_volumes", {}) and eb.monvol_id not in getattr(model, "airbags", {}):
            log.error(f"/EBCS/MONVOL/{mid}: monvol {eb.monvol_id} not defined", "CROSS REF")

    if getattr(model, "ams_control", None) is not None:
        ams = model.ams_control
        if ams.grpart_id > 0 and ams.grpart_id not in part_groups and ams.grpart_id not in model.parts:
            log.error(f"/AMS: part group {ams.grpart_id} not defined", "CROSS REF")

    for sbid, sb in getattr(model, "seatbelt_systems", {}).items():
        for rid in sb.retractor_ids:
            if rid > 0 and rid not in getattr(model, "retractors", {}):
                log.error(f"/SEATBELT/{sbid}: retractor {rid} not defined", "CROSS REF")
        for sid in sb.slipring_ids:
            if sid > 0 and sid not in getattr(model, "sliprings", {}) and sid not in getattr(model, "slipring_shells", {}):
                log.error(f"/SEATBELT/{sbid}: slipring {sid} not defined", "CROSS REF")

    for bwid, bw in getattr(model, "bcs_walls", {}).items():
        if bw.grnod_id > 0 and bw.grnod_id not in model.node_groups and bw.grnod_id not in getattr(model, "node_sets", {}):
            log.error(f"/BCS/WALL/{bwid}: node group {bw.grnod_id} not defined", "CROSS REF")
        if bw.sensor_id > 0 and bw.sensor_id not in sensor_ids:
            log.error(f"/BCS/WALL/{bwid}: sensor {bw.sensor_id} not defined", "CROSS REF")

    # M151: PBLAST, INIVOL, INIGRAV, INISTA, BEM, PERTURB
    for pbid, pb in getattr(model, "pblast_loads", {}).items():
        if pb.surf_id > 0 and pb.surf_id not in model.surfaces:
            log.error(f"/LOAD/PBLAST/{pbid}: surface {pb.surf_id} not defined", "CROSS REF")
        if pb.surf_ground_id > 0 and pb.surf_ground_id not in model.surfaces:
            log.error(f"/LOAD/PBLAST/{pbid}: ground surface {pb.surf_ground_id} not defined", "CROSS REF")
        if pb.node_id > 0 and pb.node_id not in model._id2idx:
            log.error(f"/LOAD/PBLAST/{pbid}: node {pb.node_id} not defined", "CROSS REF")

    for ivid, iv in getattr(model, "inivols", {}).items():
        if iv.part_id > 0 and iv.part_id not in model.parts and iv.part_id not in part_groups:
            log.error(f"/INIVOL/{ivid}: part {iv.part_id} not defined", "CROSS REF")
        for c in iv.containers:
            if c.surf_id > 0 and c.surf_id not in model.surfaces:
                log.error(f"/INIVOL/{ivid}: container surface {c.surf_id} not defined", "CROSS REF")

    for igid, ig in getattr(model, "inigrav_loads", {}).items():
        if ig.grpart_id > 0 and ig.grpart_id not in model.parts and ig.grpart_id not in part_groups:
            log.error(f"/INIGRAV/{igid}: part group {ig.grpart_id} not defined", "CROSS REF")
        if ig.surf_id > 0 and ig.surf_id not in model.surfaces:
            log.error(f"/INIGRAV/{igid}: surface {ig.surf_id} not defined", "CROSS REF")

    for bemid, bem in getattr(model, "bem_controls", {}).items():
        if bem.surf_id > 0 and bem.surf_id not in model.surfaces:
            log.error(f"/BEM/{bem.subtype}/{bemid}: surface {bem.surf_id} not defined", "CROSS REF")
        if bem.grnod_aux_id > 0 and bem.grnod_aux_id not in model.node_groups and bem.grnod_aux_id not in getattr(model, "node_sets", {}):
            log.error(f"/BEM/{bem.subtype}/{bemid}: node group {bem.grnod_aux_id} not defined", "CROSS REF")

    for ptid, pt in getattr(model, "perturb_controls", {}).items():
        if pt.grpart_id > 0 and pt.grpart_id not in model.parts and pt.grpart_id not in part_groups:
            log.error(f"/PERTURB/{pt.subtype}/{ptid}: part group {pt.grpart_id} not defined", "CROSS REF")
        if pt.fct_id > 0 and pt.fct_id not in model.functions:
            log.error(f"/PERTURB/{pt.subtype}/{ptid}: function {pt.fct_id} not defined", "CROSS REF")

    for mvid, mv in getattr(model, "monitored_volumes", {}).items():
        if getattr(mv, "vol_type", "") in ("PRES", "GAS"):
            continue
        surf_id = getattr(mv, "surf_id", 0)
        if surf_id not in model.surfaces:
            log.error(f"/MONVOL/{mvid}: surface {surf_id} not defined", "CROSS REF")
        matid = getattr(mv, "matid", 0)
        if matid not in model.materials:
            log.error(f"/MONVOL/{mvid}: material {matid} not defined", "CROSS REF")


