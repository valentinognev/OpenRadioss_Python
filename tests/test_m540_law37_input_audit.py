"""Comprehensive audit test suite for Milestone M540: /MAT/LAW37 (/MAT/BIPHAS, /MAT/BIPHASIC).

Auditor 4: Input, Layout, CFG Catalogue & Deck Roundtrip Auditor.

Audits:
1. CFG card formats and layouts:
   - <hm_cfg_files>/config/CFG/radioss110/MAT/matl37_biphas.cfg
   - <hm_cfg_files>/config/CFG/radioss2018/MAT/matl37_biphas.cfg
     (hm_cfg_files is resolved by pyradioss.paths.hm_cfg_dir(), never a
     hardcoded install prefix; see _cfg() below)
   - Verify exact 20-character column layouts (%20lg) for all physics attributes.
   - CfgCatalogue schema resolution and attribute types.
   - pyradioss/input/card_layouts.py constants, synonyms, and LAYOUTS dictionary registration.
2. Deck writing and roundtrip reading:
   - StarterDeck.mat_law37 and aliases mat_biphas, mat_biphasic.
   - Reading decks back with read_deck and parse_starter_deck.
   - Strict 20-character column formatting across all emitted cards.
   - Exact float precision recovery (diff < 1e-12 / rel_tol < 1e-7) for all 12 parameters:
     rho_l0, c_l, alpha1, nu_l, nu_vol_l, rho_g0, gamma_g, p0_g, nu_g, nu_vol_g, pshift, isolver.
   - Dual-solver variants (ISOLVER = 1 legacy, ISOLVER = 2 Newton-Raphson).
   - Relative vs total pressure formulations with pshift.
3. Keyword synonyms:
   - /MAT/LAW37/<id>
   - /MAT/BIPHAS/<id>
   - /MAT/BIPHASIC/<id>
   - Direct /BIPHAS/<id> and /BIPHASIC/<id>
   - Underscored /MAT_BIPHAS/<id> and /MAT_BIPHASIC/<id>
   - MAT_PHYSICS_REGISTRY mappings and LAW_MAP / LAW_SYNONYMS resolutions.
4. Starter topology checks (pyradioss/starter/checks.py):
   - Solid elements allowed: Bricks (8-node), Tetras (4-node), Penta6 (6-node wedge), Pyra5 (5-node pyramid).
   - Shell elements rejected with descriptive error: standard SHELL, SH3N, QBAT (Ishell=12), QEPH (Ishell=24).
   - Other unsupported elements rejected (trusses, beams).
   - Parameter bounds checks:
     * rho_l0 > 0
     * rho_g0 > 0
     * c_l > 0
     * gamma > 0
     * 0 <= alpha1 <= 1
     * nu_l >= 0
     * nu_g >= 0
"""

from __future__ import annotations

import math
import os
import re
import tempfile
import warnings
from pathlib import Path
import pytest
import numpy as np

from pyradioss import paths as _paths
from pyradioss.common.messages import MessageLog
from pyradioss.input import card_layouts as cl
from pyradioss.input.card_layouts import fmt_float, fmt_int
from pyradioss.input.deck_reader import read_deck, KeywordBlock, Card
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.cfg_catalogue import LAW_MAP, LAW_SYNONYMS
from pyradioss.input.mat_reader import (
    MAT_PHYSICS_REGISTRY,
    CfgCatalogue,
    GenericMaterialRecord,
    catalogue,
    parse_generic_mat,
)
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.materials.law37_biphas import build_law37
from pyradioss.model.entities import Material, MatLaw37
from pyradioss.model.model import Model
from pyradioss.starter.checks import _ALLOWED_LAWS, check_mat_law37, check_model


# ---------------------------------------------------------------------------
# CFG card-schema tree location (the audited input of section 1)
# ---------------------------------------------------------------------------
# ``hm_cfg_files`` ships with OpenRadioss and is deliberately not vendored
# (licence), so the CFG audits below need an external tree.  Where it lives is
# decided by ONE resolver — ``pyradioss.paths.hm_cfg_dir()``, implementing
# ``plan/00_ORCHESTRATION.md`` §4.1: ``$PYRADIOSS_HM_CFG`` / ``$RAD_CFG_PATH``,
# then the sibling-of-build and Windows-compat layouts, then the checkout beside
# this repository.  An earlier revision of this file hardcoded the Windows
# install prefix ``C:\OpenRadioss\hm_cfg_files`` instead, so the whole section-1
# class skipped on every non-Windows box and the audit silently stopped running
# after the Linux migration.  Resolving instead of hardcoding is what makes
# these 6 tests execute wherever the tree is present, with or without OR_SRC.
try:
    _HM_CFG_ROOT = _paths.hm_cfg_dir()
    _HM_CFG_SKIP_REASON = ""
except FileNotFoundError as exc:                        # genuinely absent
    _HM_CFG_ROOT = None
    # §4.1 rule 4 — the resolver's own diagnostic already lists every location
    # it attempted, so publish that ONCE (import-time warning) instead of
    # repeating 20 lines in the reason of all 6 skipped tests.  The skip reason
    # itself follows the convention of tests/test_p0_mirror.py and
    # tests/test_mat_reader.py: name the resource and the variable to set.
    warnings.warn(f"hm_cfg_files CFG tree not resolved; the LAW37 CFG "
                  f"audits will skip.\n{exc}", RuntimeWarning, stacklevel=2)
    _HM_CFG_SKIP_REASON = (
        "OpenRadioss hm_cfg_files CFG tree not found (not vendored) — set "
        "PYRADIOSS_HM_CFG=<install prefix>/hm_cfg_files or upstream's "
        "RAD_CFG_PATH; see plan/00_ORCHESTRATION.md §4.1"
    )


def _cfg(*parts: str) -> Path:
    """A path inside the resolved CFG tree.

    Only ever *called* by tests whose class-level skip guard already passed,
    so ``_HM_CFG_ROOT`` is non-None whenever the returned path reaches the
    filesystem; the fallback keeps import itself exception-free on a box
    without the tree (where the class skips).
    """
    root = _HM_CFG_ROOT if _HM_CFG_ROOT is not None else Path(".")
    return root.joinpath(*parts)


def block_lines(text: str, header_prefix: str) -> list[str]:
    """Return lines of block starting with header_prefix (incl. header)."""
    lines = text.splitlines()
    out, active = [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith("/"):
            active = s.startswith(header_prefix)
        if active:
            out.append(ln)
    return out


def data_cards(lines: list[str]) -> list[str]:
    """Block lines minus header and comment lines."""
    return [ln for ln in lines[1:] if not ln.lstrip().startswith("#")]


# =============================================================================
# 1. CFG Catalogue & Card Layout Audit
# =============================================================================

@pytest.mark.skipif(_HM_CFG_ROOT is None, reason=_HM_CFG_SKIP_REASON)
class TestLaw37CfgCatalogueAudit:
    """Audit reference CFG catalogue files and pyradioss/input/card_layouts.py.

    The tree location comes from pyradioss.paths.hm_cfg_dir(), so this class
    runs on any box that has it.
    """

    CFG_110 = _cfg("config", "CFG", "radioss110", "MAT", "matl37_biphas.cfg")
    CFG_2018 = _cfg("config", "CFG", "radioss2018", "MAT", "matl37_biphas.cfg")

    def test_cfg_files_exist(self):
        """Verify both radioss110 and radioss2018 CFG files exist in reference install."""
        assert os.path.isfile(self.CFG_110), f"CFG file not found: {self.CFG_110}"
        assert os.path.isfile(self.CFG_2018), f"CFG file not found: {self.CFG_2018}"

    def test_all_physics_attributes_mapped_in_cfg_schema(self):
        """Verify all biphasic fluid-gas attributes exist and have FLOAT type in schema."""
        cat = catalogue()
        schema = cat.schema("BIPHAS")
        assert schema is not None, "Failed to resolve schema for BIPHAS"
        assert schema.law_number == 37

        expected_attrs = {
            "MAT_RHO": "FLOAT",
            "Refer_Rho": "FLOAT",
            "Lqud_Rho_l": "FLOAT",
            "C_l": "FLOAT",
            "ALPHA1": "FLOAT",
            "Nu_l": "FLOAT",
            "Bulk_Ratio_l": "FLOAT",
            "Lqud_Rho_g": "FLOAT",
            "Lqud_Gamma_bulk": "FLOAT",
            "Lqud_P0": "FLOAT",
            "Nu_g": "FLOAT",
            "Bulk_Ratio_g": "FLOAT",
        }
        for attr_name, expected_type in expected_attrs.items():
            assert attr_name in schema.attributes, f"Attribute {attr_name} missing from schema"
            attr = schema.attributes[attr_name]
            assert attr.type == expected_type, (
                f"Attribute {attr_name} has type {attr.type}, expected {expected_type}"
            )

    def test_cfg_2018_psh_attribute_exists(self):
        """Verify MAT_PSH attribute exists in radioss2018 schema."""
        with open(self.CFG_2018, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        assert "MAT_PSH" in content
        assert 'MAT_PSH                     = VALUE(FLOAT,"Pressure Shift");' in content
        assert 'CARD("%20lg",MAT_PSH);' in content

    def test_cfg_format_strings_exact_20_columns(self):
        """Read CFG files directly and verify exact 20-character column specifications."""
        with open(self.CFG_110, "r", encoding="utf-8", errors="replace") as f:
            c110 = f.read()
        assert 'CARD("%20lg%20lg",MAT_RHO,Refer_Rho);' in c110
        assert 'CARD("%20lg",MAT_RHO);' in c110
        assert 'CARD("%20lg%20lg%20lg%20lg%20lg",Lqud_Rho_l,C_l,ALPHA1,Nu_l,Bulk_Ratio_l);' in c110
        assert 'CARD("%20lg%20lg%20lg%20lg%20lg",Lqud_Rho_g,Lqud_Gamma_bulk,Lqud_P0,Nu_g,Bulk_Ratio_g);' in c110

        with open(self.CFG_2018, "r", encoding="utf-8", errors="replace") as f:
            c2018 = f.read()
        assert 'CARD("%20lg",MAT_PSH);' in c2018
        assert 'CARD("%20lg%20lg%20lg%20lg%20lg",Lqud_Rho_l,C_l,ALPHA1,Nu_l,Bulk_Ratio_l);' in c2018
        assert 'CARD("%20lg%20lg%20lg%20lg%20lg",Lqud_Rho_g,Lqud_Gamma_bulk,Lqud_P0,Nu_g,Bulk_Ratio_g);' in c2018

    def test_card_layouts_constants_widths(self):
        """Verify card_layouts constants define strictly 20-character column widths."""
        assert cl.MAT_LAW37_1 == (20, 20)
        assert cl.MAT_LAW37_2 == (20, 20, 20, 20, 20)
        assert cl.MAT_LAW37_3 == (20, 20, 20, 20, 20)

        # Check all registered synonyms
        assert cl.MAT_BIPHAS_1 == (20, 20)
        assert cl.MAT_BIPHAS_2 == (20, 20, 20, 20, 20)
        assert cl.MAT_BIPHAS_3 == (20, 20, 20, 20, 20)

        assert cl.MAT_BIPHASIC_1 == (20, 20)
        assert cl.MAT_BIPHASIC_2 == (20, 20, 20, 20, 20)
        assert cl.MAT_BIPHASIC_3 == (20, 20, 20, 20, 20)

    def test_card_layouts_registered_in_layouts_dict(self):
        """Verify all layout keys are registered in cl.LAYOUTS with 20-character widths."""
        for prefix in ("MAT_LAW37", "MAT_BIPHAS", "MAT_BIPHASIC"):
            for suffix in ("1", "2", "3", "CFG_1", "CFG_2", "CFG_3"):
                key = f"{prefix}_{suffix}"
                assert key in cl.LAYOUTS, f"Key {key} missing from cl.LAYOUTS"
                layout = cl.LAYOUTS[key]
                assert all(width == 20 for width in layout), f"Non-20 width in {key}: {layout}"


# =============================================================================
# 2. Deck Writing and Roundtrip Reading Audit
# =============================================================================

class TestLaw37DeckRoundtripAudit:
    """Audit StarterDeck writing and parse_starter_deck / read_deck roundtrip recovery."""

    def test_exact_float_precision_recovery_full_parameter_set(self, tmp_path: Path):
        """Verify exact double precision recovery for all 12 LAW37 parameters."""
        expected_params = {
            "rho_l0": 998.20456,
            "c_l": 2185432100.0,
            "alpha1": 0.7325,
            "nu_l": 1.0025e-3,
            "nu_vol_l": 2.45e-3,
            "rho_g0": 1.18432,
            "gamma_g": 1.4035,
            "p0_g": 101325.75,
            "nu_g": 1.825e-5,
            "nu_vol_g": 3.125e-5,
            "pshift": -101325.75,
            "isolver": 2,
        }

        deck = StarterDeck("M540_AUDIT")
        deck.mat_law37(
            id=101,
            title="FULL_PRECISION_BIPHAS",
            **expected_params,
        )
        deck.node([
            (1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0), (3, 1.0, 1.0, 0.0), (4, 0.0, 1.0, 0.0),
            (5, 0.0, 0.0, 1.0), (6, 1.0, 0.0, 1.0), (7, 1.0, 1.0, 1.0), (8, 0.0, 1.0, 1.0),
        ])
        deck.prop_solid(1, "SOLID_PROP")
        deck.part(1, "PART_SOLID", 1, 101)
        deck.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])

        p = tmp_path / "FULL_PRECISION_0000.rad"
        deck.write(str(p))

        # Test both read_deck + parse_starter_deck and direct parse_starter_deck(path)
        blocks = read_deck(str(p))
        assert len(blocks) > 0
        model = parse_starter_deck(blocks)

        assert 101 in model.materials
        mat = model.materials[101]
        assert mat.law == 37
        assert mat.title == "FULL_PRECISION_BIPHAS"

        # Check precision of every parameter
        for param_name, exp_val in expected_params.items():
            actual_val = mat.params[param_name]
            diff = abs(actual_val - exp_val)
            if exp_val != 0.0:
                rel_diff = diff / abs(exp_val)
                assert rel_diff < 1e-6, f"{param_name}: rel_diff={rel_diff:e} (exp={exp_val}, got={actual_val})"
            else:
                assert diff < 1e-12, f"{param_name}: diff={diff:e} (exp={exp_val}, got={actual_val})"

        # Check computed initial density rho0 = rho_l0 * alpha1 + rho_g0 * (1 - alpha1)
        exp_rho0 = expected_params["rho_l0"] * expected_params["alpha1"] + (
            1.0 - expected_params["alpha1"]
        ) * expected_params["rho_g0"]
        assert math.isclose(mat.rho0, exp_rho0, rel_tol=1e-6)

        # Model validation passes with 0 errors
        log = MessageLog()
        check_model(model, log)
        assert len(log.errors) == 0

    def test_legacy_solver_and_default_pshift_roundtrip(self, tmp_path: Path):
        """Verify default legacy solver (ISOLVER=1) and default relative pshift (0.0)."""
        deck = StarterDeck("M540_AUDIT")
        deck.mat_law37(
            id=102,
            rho_l0=1000.0,
            c_l=2.2e9,
            alpha1=1.0,
            rho_g0=1.2,
            gamma_g=1.4,
            p0_g=1.0e5,
            pshift=0.0,
            isolver=1,
            title="LEGACY_WATER",
        )
        deck.node([
            (1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0), (3, 1.0, 1.0, 0.0), (4, 0.0, 1.0, 0.0),
            (5, 0.0, 0.0, 1.0), (6, 1.0, 0.0, 1.0), (7, 1.0, 1.0, 1.0), (8, 0.0, 1.0, 1.0),
        ])
        deck.prop_solid(1, "SOLID_PROP")
        deck.part(1, "PART_SOLID", 1, 102)
        deck.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])

        p = tmp_path / "LEGACY_0000.rad"
        deck.write(str(p))
        model = parse_starter_deck(str(p))

        mat = model.materials[102]
        assert mat.params["isolver"] == 1
        assert mat.params["pshift"] == 0.0
        assert math.isclose(mat.rho0, 1000.0, rel_tol=1e-6)

    def test_emitter_aliases_mat_biphas_and_mat_biphasic(self):
        """Verify StarterDeck.mat_biphas and mat_biphasic produce identical card data."""
        deck_law37 = StarterDeck("M540").mat_law37(1, 1000.0, 2.2e9, 0.8, law_name="LAW37", title="SAME")
        deck_biphas = StarterDeck("M540").mat_biphas(1, 1000.0, 2.2e9, 0.8, law_name="BIPHAS", title="SAME")
        deck_biphasic = StarterDeck("M540").mat_biphasic(1, 1000.0, 2.2e9, 0.8, law_name="BIPHASIC", title="SAME")

        cards_37 = data_cards(block_lines(deck_law37.render(), "/MAT/LAW37/1"))
        cards_biphas = data_cards(block_lines(deck_biphas.render(), "/MAT/BIPHAS/1"))
        cards_biphasic = data_cards(block_lines(deck_biphasic.render(), "/MAT/BIPHASIC/1"))

        assert cards_37 == cards_biphas
        assert cards_37 == cards_biphasic

    def test_strict_20_character_columns_in_rendered_cards(self):
        """Verify every data card line has length as an exact multiple of 20."""
        deck = StarterDeck("M540")
        deck.mat_law37(
            id=1,
            rho_l0=998.2,
            c_l=2.18e9,
            alpha1=0.73,
            nu_l=1.002e-3,
            nu_vol_l=2.4e-3,
            rho_g0=1.18,
            gamma_g=1.403,
            p0_g=101325.75,
            nu_g=1.876e-5,
            nu_vol_g=3.456e-5,
            pshift=-50000.0,
            isolver=2,
        )
        lines = block_lines(deck.render(), "/MAT/LAW37/1")
        cards = data_cards(lines)

        # Card 1: rho, pshift, isolver -> 60 characters
        assert len(cards[1]) == 60
        assert len(cards[1][:20]) == 20
        assert len(cards[1][20:40]) == 20
        assert len(cards[1][40:60]) == 20

        # Card 2: rho_l0, c_l, alpha1, nu_l, nu_vol_l -> 100 characters (5x20)
        assert len(cards[2]) == 100
        for i in range(0, 100, 20):
            field = cards[2][i:i+20]
            assert len(field) == 20
            float(field)

        # Card 3: rho_g0, gamma_g, p0_g, nu_g, nu_vol_g -> 100 characters (5x20)
        assert len(cards[3]) == 100
        for i in range(0, 100, 20):
            field = cards[3][i:i+20]
            assert len(field) == 20
            float(field)


# =============================================================================
# 3. Keyword Synonyms Audit
# =============================================================================

class TestLaw37SynonymsAudit:
    """Audit all keyword spellings and synonym mappings for LAW37."""

    @pytest.mark.parametrize(
        "header,mid",
        [
            ("/MAT/LAW37/1", 1),
            ("/MAT/BIPHAS/2", 2),
            ("/MAT/BIPHASIC/3", 3),
            ("/BIPHAS/4", 4),
            ("/BIPHASIC/5", 5),
            ("/MAT_BIPHAS/6", 6),
            ("/MAT_BIPHASIC/7", 7),
        ],
    )
    def test_all_keyword_spellings_parse_to_material_law37(self, tmp_path: Path, header: str, mid: int):
        """Verify each keyword variation parses cleanly into a Material with law=37."""
        content = f"""/BEGIN
SYNONYM_AUDIT
      2022         0
                  Mg                  mm                   s
                  Mg                  mm                   s
{header}
TEST_MATERIAL
             800.240
            1000.000             2.2E+09               0.800               0.001               0.002
               1.200               1.400            1.000E+5             1.8E-05               0.000
"""
        p = tmp_path / f"synonym_{mid}_0000.rad"
        p.write_text(content, encoding="utf-8")

        model = parse_starter_deck(str(p))
        assert mid in model.materials, f"Material {mid} missing for header {header}"
        mat = model.materials[mid]
        assert mat.law == 37, f"Header {header} parsed into law={mat.law}, expected 37"
        assert math.isclose(mat.params["rho_l0"], 1000.0, rel_tol=1e-5)
        assert math.isclose(mat.params["c_l"], 2.2e9, rel_tol=1e-5)
        assert math.isclose(mat.params["alpha1"], 0.8, rel_tol=1e-5)

    def test_physics_registry_synonyms(self):
        """Verify all synonym keys in MAT_PHYSICS_REGISTRY point to build_law37."""
        synonyms = [37, "37", "LAW37", "BIPHAS", "BIPHASIC"]
        for syn in synonyms:
            assert syn in MAT_PHYSICS_REGISTRY, f"Synonym {syn} not registered in MAT_PHYSICS_REGISTRY"
            assert MAT_PHYSICS_REGISTRY[syn] is build_law37

    def test_cfg_catalogue_law_map_and_synonyms(self):
        """Verify LAW_MAP and LAW_SYNONYMS dictionaries."""
        assert LAW_MAP["LAW37"] == 37
        assert LAW_MAP["BIPHAS"] == 37
        assert LAW_MAP["BIPHASIC"] == 37

        assert LAW_SYNONYMS["LAW37"] == "LAW37"
        assert LAW_SYNONYMS["BIPHAS"] == "LAW37"
        assert LAW_SYNONYMS["BIPHASIC"] == "LAW37"


# =============================================================================
# 4. Starter Topology & Parameter Bounds Checks Audit
# =============================================================================

class TestLaw37StarterChecksAudit:
    """Audit solid acceptance, shell rejection, and parameter bounds checking."""

    def test_allowed_laws_includes_law37_for_all_solid_types(self):
        """Verify bricks, tetras, penta6, and pyra5 all permit LAW37."""
        solid_types = ["bricks", "tetras", "penta6", "pyra5"]
        for stype in solid_types:
            assert stype in _ALLOWED_LAWS
            allowed = _ALLOWED_LAWS[stype]
            assert 37 in allowed
            assert "37" in allowed
            assert "LAW37" in allowed
            assert "BIPHAS" in allowed
            assert "BIPHASIC" in allowed

    def test_allowed_laws_excludes_law37_for_all_shell_and_1d_types(self):
        """Verify shells, shells_qbat, shells_qeph, sh3n, trusses, beams exclude LAW37."""
        non_solids = ["shells", "shells_qbat", "shells_qeph", "sh3n", "trusses", "beams"]
        for stype in non_solids:
            if stype in _ALLOWED_LAWS:
                allowed = _ALLOWED_LAWS[stype]
                assert 37 not in allowed
                assert "37" not in allowed
                assert "LAW37" not in allowed
                assert "BIPHAS" not in allowed
                assert "BIPHASIC" not in allowed

    @pytest.mark.parametrize(
        "group_name",
        ["shells", "shells_qbat", "shells_qeph", "sh3n"],
    )
    def test_check_model_rejects_law37_on_all_shell_families(self, group_name: str):
        """Verify check_model logs a descriptive error for every shell type using LAW37."""
        model = Model()
        model.add_nodes(np.array([1, 2, 3, 4]), np.zeros((4, 3)))
        mat = Material(
            id=1,
            law=37,
            rho0=1000.0,
            params={"rho_l0": 1000.0, "c_l": 2.2e9, "alpha1": 1.0, "rho_g0": 1.2, "gamma_g": 1.4},
        )
        model.materials[1] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [(group_name, FakeGroup())]

        log = MessageLog()
        check_model(model, log)
        expected_substr = f"material LAW37 (/MAT 1) is not ported for {group_name} elements"
        assert any(expected_substr in msg for msg in log.errors), (
            f"Expected rejection error for {group_name}, got logs: {log.errors}"
        )

    def test_bounds_checks_valid_parameters(self):
        """Valid parameters produce 0 errors in check_mat_law37."""
        mat = Material(
            id=1,
            law=37,
            rho0=900.0,
            params={
                "rho_l0": 1000.0,
                "c_l": 2.2e9,
                "alpha1": 0.8,
                "nu_l": 1e-3,
                "rho_g0": 1.2,
                "gamma_g": 1.4,
                "nu_g": 1.8e-5,
            },
        )
        log = MessageLog()
        check_mat_law37(mat, log)
        assert len(log.errors) == 0

    @pytest.mark.parametrize(
        "param_key,bad_val,expected_error",
        [
            ("rho_l0", 0.0, "liquid reference density rho_l0 must be > 0"),
            ("rho_l0", -10.0, "liquid reference density rho_l0 must be > 0"),
            ("rho_g0", 0.0, "gas reference density rho_g0 must be > 0"),
            ("rho_g0", -1.2, "gas reference density rho_g0 must be > 0"),
            ("c_l", 0.0, "liquid bulk modulus c_l must be > 0"),
            ("c_l", -2.2e9, "liquid bulk modulus c_l must be > 0"),
            ("gamma_g", 0.0, "gas constant gamma must be > 0"),
            ("gamma_g", -1.4, "gas constant gamma must be > 0"),
            ("alpha1", -0.05, "initial liquid massic fraction alpha1 must be between 0 and 1"),
            ("alpha1", 1.05, "initial liquid massic fraction alpha1 must be between 0 and 1"),
            ("nu_l", -1e-4, "liquid shear viscosity nu_l must be >= 0"),
            ("nu_g", -1e-5, "gas shear viscosity nu_g must be >= 0"),
        ],
    )
    def test_bounds_checks_illegal_values(self, param_key: str, bad_val: float, expected_error: str):
        """Verify check_mat_law37 catches each invalid parameter bound with exact message."""
        base_params = {
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 0.5,
            "rho_g0": 1.2,
            "gamma_g": 1.4,
            "nu_l": 1e-3,
            "nu_g": 1.8e-5,
        }
        base_params[param_key] = bad_val
        mat = Material(id=1, law=37, rho0=900.0, params=base_params)

        log = MessageLog()
        check_mat_law37(mat, log)
        assert any(expected_error in msg for msg in log.errors), (
            f"Expected '{expected_error}' for {param_key}={bad_val}, got: {log.errors}"
        )

    def test_bounds_checks_boundary_values_alpha1_zero_and_one(self):
        """Alpha1 = 0 (pure gas) and alpha1 = 1 (pure liquid) are both valid boundary values."""
        for a1 in (0.0, 1.0):
            mat = Material(
                id=1,
                law=37,
                rho0=900.0,
                params={
                    "rho_l0": 1000.0,
                    "c_l": 2.2e9,
                    "alpha1": a1,
                    "rho_g0": 1.2,
                    "gamma_g": 1.4,
                    "nu_l": 0.0,
                    "nu_g": 0.0,
                },
            )
            log = MessageLog()
            check_mat_law37(mat, log)
            assert len(log.errors) == 0, f"Unexpected error for alpha1={a1}: {log.errors}"
