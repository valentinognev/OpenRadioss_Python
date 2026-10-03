"""Comprehensive audit test suite for Milestone M539: /MAT/LAW34 (/MAT/BOLTZMAN, /MAT/VISC_MAXW).

Audits:
1. CFG catalogue mapping & card layouts:
   - <hm_cfg_files>/config/CFG/radioss110/MAT/matl34_boltzman.cfg attribute mapping
     (hm_cfg_files is resolved by pyradioss.paths.hm_cfg_dir(), never a
     hardcoded install prefix; see _cfg() below)
   - card_layouts.py field widths (strictly 20 columns for floats)
   - CfgCatalogue parsing into GenericMaterialRecord
2. Deck writing and roundtrip reading:
   - Minimal deck (id, rho, bulk, g0, gi, beta)
   - Full deck (with p0, phi, gamma0, rhor, title)
   - Zero vs default values
   - Exact float precision preservation (rel_tol < 1e-12)
   - Fixed-format and free-format parsing
3. Keyword synonyms:
   - /MAT/LAW34/<id>
   - /MAT/BOLTZMAN/<id>
   - /MAT/VISC_MAXW/<id>
   - /MAT/BOLTZMANN/<id>
   - MAT_PHYSICS_REGISTRY and model dictionary aliases
4. Starter topology checks (pyradioss.starter.checks):
   - Bricks (8-node hexa)
   - Tetras (4-node tetra)
   - Penta6 (6-node wedge / collapsed hexa)
   - Pyra5 (5-node pyramid / collapsed hexa)
   - Shells (BT shell, SH3N, QBAT, QEPH)
   - Trusses and Beams
   - Error reporting on illegal inputs (rho<=0, bulk<=0, g0<0, gi<0, beta<0, gi>g0 warning)
"""

from __future__ import annotations

import math
import os
import re
import tempfile
import warnings
from pathlib import Path
import pytest

from pyradioss import paths as _paths
from pyradioss.common.messages import MessageLog
from pyradioss.input import card_layouts as cl
from pyradioss.input.card_layouts import fmt_float
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.mat_reader import (
    MAT_PHYSICS_REGISTRY,
    CfgCatalogue,
    GenericMaterialRecord,
    catalogue,
    parse_generic_mat,
)
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.materials.law34_boltzmann import build_law34
from pyradioss.model.entities import Material, MatLaw34
from pyradioss.model.model import Model
from pyradioss.starter.checks import _ALLOWED_LAWS, check_mat_law34, check_model
from pyradioss.starter.initialization import build_element_groups, resolve_materials


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
    warnings.warn(f"hm_cfg_files CFG tree not resolved; the LAW34 CFG "
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


# =============================================================================
# 1. CFG Catalogue & Card Layout Audit
# =============================================================================

@pytest.mark.skipif(_HM_CFG_ROOT is None, reason=_HM_CFG_SKIP_REASON)
class TestLaw34CfgCatalogueAudit:
    """Audit <hm_cfg_files>/config/CFG/radioss110/MAT/matl34_boltzman.cfg

    and pyradioss/input/card_layouts.py.  The tree location comes from
    pyradioss.paths.hm_cfg_dir(), so this class runs on any box that has it.
    """

    CFG_PATH = _cfg("config", "CFG", "radioss110", "MAT", "matl34_boltzman.cfg")

    def test_cfg_file_exists(self):
        """Verify matl34_boltzman.cfg exists in reference installation."""
        assert os.path.isfile(self.CFG_PATH), f"CFG file not found at {self.CFG_PATH}"

    def test_all_nine_attributes_mapped_in_cfg(self):
        """Verify all 9 physics attributes exist and have FLOAT type in CFG schema."""
        cat = catalogue()
        schema = cat.schema("BOLTZMAN")
        assert schema is not None, "Failed to resolve schema for BOLTZMAN"
        assert schema.law_number == 34

        expected_attrs = {
            "MAT_RHO": "FLOAT",
            "Refer_Rho": "FLOAT",
            "MAT_BULK": "FLOAT",
            "MAT_G0": "FLOAT",
            "MAT_GI": "FLOAT",
            "MAT_DECAY": "FLOAT",
            "MAT_P0": "FLOAT",
            "MAT_PHI": "FLOAT",
            "MAT_GAMA0": "FLOAT",
        }

        for attr_name, expected_type in expected_attrs.items():
            assert attr_name in schema.attributes, f"Attribute {attr_name} missing from schema"
            attr = schema.attributes[attr_name]
            assert attr.type == expected_type, (
                f"Attribute {attr_name} has type {attr.type}, expected {expected_type}"
            )

    def test_cfg_format_strings_20_columns(self):
        """Read CFG file text directly and verify that every float uses %20lg."""
        with open(self.CFG_PATH, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # In FORMAT(radioss51):
        # CARD("%20lg%20lg",MAT_RHO,Refer_Rho);
        # CARD("%20lg",MAT_RHO);
        # CARD("%20lg",MAT_BULK);
        # CARD("%20lg%20lg%20lg",MAT_G0,MAT_GI,MAT_DECAY);
        # CARD("%20lg%20lg%20lg",MAT_P0,MAT_PHI,MAT_GAMA0);
        assert 'CARD("%20lg%20lg",MAT_RHO,Refer_Rho);' in content
        assert 'CARD("%20lg",MAT_RHO);' in content
        assert 'CARD("%20lg",MAT_BULK);' in content
        assert 'CARD("%20lg%20lg%20lg",MAT_G0,MAT_GI,MAT_DECAY);' in content
        assert 'CARD("%20lg%20lg%20lg",MAT_P0,MAT_PHI,MAT_GAMA0);' in content

    def test_card_layouts_constants_widths(self):
        """Verify card_layouts constants define strictly 20-character column widths."""
        assert cl.MAT_LAW34_1 == (20, 20)
        assert cl.MAT_LAW34_2 == (20,)
        assert cl.MAT_LAW34_3 == (20, 20, 20)
        assert cl.MAT_LAW34_4 == (20, 20, 20)

        # Check all synonyms
        assert cl.MAT_BOLTZMAN_1 == (20, 20)
        assert cl.MAT_BOLTZMAN_2 == (20,)
        assert cl.MAT_BOLTZMAN_3 == (20, 20, 20)
        assert cl.MAT_BOLTZMAN_4 == (20, 20, 20)

        assert cl.MAT_VISC_MAXW_1 == (20, 20)
        assert cl.MAT_VISC_MAXW_2 == (20,)
        assert cl.MAT_VISC_MAXW_3 == (20, 20, 20)
        assert cl.MAT_VISC_MAXW_4 == (20, 20, 20)

        assert cl.MAT_BOLTZMANN_1 == (20, 20)
        assert cl.MAT_BOLTZMANN_2 == (20,)
        assert cl.MAT_BOLTZMANN_3 == (20, 20, 20)
        assert cl.MAT_BOLTZMANN_4 == (20, 20, 20)

    def test_card_layouts_registered_in_layouts_dict(self):
        """Verify all layout keys are registered in cl.LAYOUTS dictionary."""
        for prefix in ("MAT_LAW34", "MAT_BOLTZMAN", "MAT_VISC_MAXW", "MAT_BOLTZMANN"):
            for suffix in ("1", "2", "3", "4", "CFG_1", "CFG_2", "CFG_3", "CFG_4"):
                key = f"{prefix}_{suffix}"
                assert key in cl.LAYOUTS, f"Key {key} missing from cl.LAYOUTS"
                layout = cl.LAYOUTS[key]
                assert all(width == 20 for width in layout), f"Non-20 width in {key}: {layout}"

    def test_generic_mat_cfg_parse_and_build_law34(self, tmp_path):
        """Verify CFG-driven parse_generic_mat extracts all attributes and build_law34 constructs Material."""
        deck = StarterDeck("AUDIT_CFG")
        deck.mat_law34(
            id=101,
            rho=1050.0,
            bulk=1.8e7,
            g0=4.5e6,
            gi=1.5e6,
            beta=30.0,
            p0=1.2e5,
            phi=0.18,
            gamma0=0.03,
            rhor=1120.0,
            title="CFG_AUDIT_MAT",
            law_name="BOLTZMAN",
        )
        p = tmp_path / "CFG_0000.rad"
        deck.write(str(p))

        blocks = read_deck(str(p))
        mat_blocks = [b for b in blocks if b.key0 == "MAT"]
        assert len(mat_blocks) == 1
        mat_block = mat_blocks[0]

        rec = parse_generic_mat(mat_block)
        assert rec is not None
        assert rec.id == 101
        assert rec.title == "CFG_AUDIT_MAT"
        assert rec.law_number == 34
        assert math.isclose(rec.params["MAT_RHO"], 1050.0, rel_tol=1e-12)
        assert math.isclose(rec.params["Refer_Rho"], 1120.0, rel_tol=1e-12)
        assert math.isclose(rec.params["MAT_BULK"], 1.8e7, rel_tol=1e-12)
        assert math.isclose(rec.params["MAT_G0"], 4.5e6, rel_tol=1e-12)
        assert math.isclose(rec.params["MAT_GI"], 1.5e6, rel_tol=1e-12)
        assert math.isclose(rec.params["MAT_DECAY"], 30.0, rel_tol=1e-12)
        assert math.isclose(rec.params["MAT_P0"], 1.2e5, rel_tol=1e-12)
        assert math.isclose(rec.params["MAT_PHI"], 0.18, rel_tol=1e-12)
        assert math.isclose(rec.params["MAT_GAMA0"], 0.03, rel_tol=1e-12)

        mat = build_law34(rec)
        assert isinstance(mat, Material)
        assert mat.id == 101
        assert mat.law == 34
        assert math.isclose(mat.rho0, 1050.0, rel_tol=1e-12)
        assert math.isclose(mat.params["bulk"], 1.8e7, rel_tol=1e-12)
        assert math.isclose(mat.params["rhor"], 1120.0, rel_tol=1e-12)
        assert math.isclose(mat.params["p0"], 1.2e5, rel_tol=1e-12)


# =============================================================================
# 2. Deck Writing & Roundtrip Reading Audit
# =============================================================================

class TestLaw34DeckWritingAndRoundtrip:
    """Audit deck emission and roundtrip parsing across combinations and precisions."""

    def test_roundtrip_minimal(self, tmp_path):
        """Minimal deck: id, rho, bulk, g0, gi, beta (no optional air/ref params)."""
        deck = StarterDeck("MINIMAL")
        deck.mat_law34(
            id=1,
            rho=950.0,
            bulk=1.2e7,
            g0=3.2e6,
            gi=1.1e6,
            beta=18.0,
        )
        p = tmp_path / "MIN_0000.rad"
        deck.write(str(p))

        model = parse_starter_deck(str(p))
        assert 1 in model.materials
        mat = model.materials[1]
        assert mat.law == 34
        assert math.isclose(mat.rho0, 950.0, rel_tol=1e-12)
        assert math.isclose(mat.params["bulk"], 1.2e7, rel_tol=1e-12)
        assert math.isclose(mat.params["g0"], 3.2e6, rel_tol=1e-12)
        assert math.isclose(mat.params["gi"], 1.1e6, rel_tol=1e-12)
        assert math.isclose(mat.params["beta"], 18.0, rel_tol=1e-12)
        # Optional parameters default to 0.0
        assert math.isclose(mat.params["p0"], 0.0, abs_tol=1e-12)
        assert math.isclose(mat.params["phi"], 0.0, abs_tol=1e-12)
        assert math.isclose(mat.params["gamma0"], 0.0, abs_tol=1e-12)

    def test_roundtrip_full(self, tmp_path):
        """Full deck: id, rho, bulk, g0, gi, beta, p0, phi, gamma0, rhor, title."""
        deck = StarterDeck("FULL")
        deck.mat_law34(
            id=42,
            rho=1020.5,
            bulk=2.345e7,
            g0=5.678e6,
            gi=2.345e6,
            beta=45.67,
            p0=1.013e5,
            phi=0.225,
            gamma0=0.015,
            rhor=1080.2,
            title="FULL_BOLTZMANN_MATERIAL_42",
        )
        p = tmp_path / "FULL_0000.rad"
        deck.write(str(p))

        model = parse_starter_deck(str(p))
        assert 42 in model.materials
        mat = model.materials[42]
        assert mat.law == 34
        assert mat.title == "FULL_BOLTZMANN_MATERIAL_42"
        assert math.isclose(mat.rho0, 1020.5, rel_tol=1e-12)
        assert math.isclose(mat.params["rhor"], 1080.2, rel_tol=1e-12)
        assert math.isclose(mat.params["bulk"], 2.345e7, rel_tol=1e-12)
        assert math.isclose(mat.params["g0"], 5.678e6, rel_tol=1e-12)
        assert math.isclose(mat.params["gi"], 2.345e6, rel_tol=1e-12)
        assert math.isclose(mat.params["beta"], 45.67, rel_tol=1e-12)
        assert math.isclose(mat.params["p0"], 1.013e5, rel_tol=1e-12)
        assert math.isclose(mat.params["phi"], 0.225, rel_tol=1e-12)
        assert math.isclose(mat.params["gamma0"], 0.015, rel_tol=1e-12)

    def test_roundtrip_zero_vs_default_values(self, tmp_path):
        """Verify behavior when explicit zero is passed vs omitted."""
        deck = StarterDeck("ZEROS")
        # Explicit zeros for p0, phi, gamma0, rhor
        deck.mat_law34(
            id=10,
            rho=1000.0,
            bulk=1.0e7,
            g0=3.0e6,
            gi=1.0e6,
            beta=10.0,
            p0=0.0,
            phi=0.0,
            gamma0=0.0,
            rhor=0.0,
            title="EXPLICIT_ZEROS",
        )
        # Omitted (None/defaults)
        deck.mat_law34(
            id=20,
            rho=1000.0,
            bulk=1.0e7,
            g0=3.0e6,
            gi=1.0e6,
            beta=10.0,
            title="OMITTED_DEFAULTS",
        )
        p = tmp_path / "ZERO_0000.rad"
        deck.write(str(p))

        model = parse_starter_deck(str(p))
        mat10 = model.materials[10]
        mat20 = model.materials[20]

        for mat in (mat10, mat20):
            assert mat.rho0 == 1000.0
            assert mat.params["bulk"] == 1.0e7
            assert mat.params["g0"] == 3.0e6
            assert mat.params["gi"] == 1.0e6
            assert mat.params["beta"] == 10.0
            assert mat.params["p0"] == 0.0
            assert mat.params["phi"] == 0.0
            assert mat.params["gamma0"] == 0.0

    def test_roundtrip_high_float_precision(self, tmp_path):
        """Verify exact recovery of 64-bit IEEE double float precision."""
        rho_val = 1.23456789012345e-3
        bulk_val = 9.87654321098765e6
        g0_val = 3.14159265358979e5
        gi_val = 1.61803398874989e5
        beta_val = 2.71828182845904
        p0_val = 1.01325000000000e5
        phi_val = 0.12345678901234
        gamma0_val = 0.09876543210987
        rhor_val = 1.34567890123456e-3

        deck = StarterDeck("PRECISION")
        deck.mat_law34(
            id=99,
            rho=rho_val,
            bulk=bulk_val,
            g0=g0_val,
            gi=gi_val,
            beta=beta_val,
            p0=p0_val,
            phi=phi_val,
            gamma0=gamma0_val,
            rhor=rhor_val,
            title="HIGH_PRECISION_TEST",
        )
        p = tmp_path / "PREC_0000.rad"
        deck.write(str(p))

        model = parse_starter_deck(str(p))
        mat = model.materials[99]

        assert math.isclose(mat.rho0, rho_val, rel_tol=1e-12)
        assert math.isclose(mat.params["rhor"], rhor_val, rel_tol=1e-12)
        assert math.isclose(mat.params["bulk"], bulk_val, rel_tol=1e-12)
        assert math.isclose(mat.params["g0"], g0_val, rel_tol=1e-12)
        assert math.isclose(mat.params["gi"], gi_val, rel_tol=1e-12)
        assert math.isclose(mat.params["beta"], beta_val, rel_tol=1e-12)
        assert math.isclose(mat.params["p0"], p0_val, rel_tol=1e-12)
        assert math.isclose(mat.params["phi"], phi_val, rel_tol=1e-12)
        assert math.isclose(mat.params["gamma0"], gamma0_val, rel_tol=1e-12)

    def test_roundtrip_free_format_raw(self, tmp_path):
        """Verify token-based free-format parsing without /BEGIN version header."""
        free_deck = (
            "/MAT/LAW34/5\n"
            "FREE_FORMAT_MAT\n"
            "1000.0 1080.0\n"
            "1.5E7\n"
            "4.0E6 1.0E6 25.0\n"
            "1.0E5 0.15 0.02\n"
            "/END\n"
        )
        p = tmp_path / "FREE_0000.rad"
        p.write_text(free_deck, encoding="utf-8")

        model = parse_starter_deck(str(p))
        assert 5 in model.materials
        mat = model.materials[5]
        assert mat.law == 34
        assert mat.title == "FREE_FORMAT_MAT"
        assert math.isclose(mat.rho0, 1000.0, rel_tol=1e-12)
        assert math.isclose(mat.params["rhor"], 1080.0, rel_tol=1e-12)
        assert math.isclose(mat.params["bulk"], 1.5e7, rel_tol=1e-12)
        assert math.isclose(mat.params["g0"], 4.0e6, rel_tol=1e-12)
        assert math.isclose(mat.params["gi"], 1.0e6, rel_tol=1e-12)
        assert math.isclose(mat.params["beta"], 25.0, rel_tol=1e-12)
        assert math.isclose(mat.params["p0"], 1.0e5, rel_tol=1e-12)
        assert math.isclose(mat.params["phi"], 0.15, rel_tol=1e-12)
        assert math.isclose(mat.params["gamma0"], 0.02, rel_tol=1e-12)

    def test_multiple_materials_single_deck(self, tmp_path):
        """Verify multiple LAW34 materials coexisting in a single deck."""
        deck = StarterDeck("MULTI")
        for i in range(1, 5):
            deck.mat_law34(
                id=i,
                rho=1000.0 * i,
                bulk=1.0e7 * i,
                g0=3.0e6 * i,
                gi=1.0e6 * i,
                beta=10.0 * i,
                title=f"MAT_{i}",
            )
        p = tmp_path / "MULTI_0000.rad"
        deck.write(str(p))

        model = parse_starter_deck(str(p))
        for i in range(1, 5):
            assert i in model.materials
            mat = model.materials[i]
            assert mat.law == 34
            assert mat.title == f"MAT_{i}"
            assert math.isclose(mat.rho0, 1000.0 * i, rel_tol=1e-12)
            assert math.isclose(mat.params["bulk"], 1.0e7 * i, rel_tol=1e-12)


# =============================================================================
# 3. Keyword Synonyms Audit
# =============================================================================

class TestLaw34KeywordSynonymsAudit:
    """Audit all keyword synonyms and spellings:

    /MAT/LAW34/<id>, /MAT/BOLTZMAN/<id>, /MAT/VISC_MAXW/<id>, /MAT/BOLTZMANN/<id>.
    """

    @pytest.mark.parametrize("keyword_name", [
        "LAW34",
        "BOLTZMAN",
        "VISC_MAXW",
        "BOLTZMANN",
    ])
    def test_synonym_in_deck_parses_to_law34(self, tmp_path, keyword_name):
        """Verify every keyword spelling parses properly into a valid Material(law=34)."""
        deck = StarterDeck(f"SYN_{keyword_name}")
        deck.mat_law34(
            id=77,
            rho=1150.0,
            bulk=1.6e7,
            g0=4.2e6,
            gi=1.4e6,
            beta=22.0,
            title=f"TITLE_{keyword_name}",
            law_name=keyword_name,
        )
        p = tmp_path / f"{keyword_name}_0000.rad"
        deck.write(str(p))

        model = parse_starter_deck(str(p))
        assert 77 in model.materials, f"Material 77 not found for keyword {keyword_name}"
        mat = model.materials[77]
        assert mat.law == 34, f"Material law is {mat.law}, expected 34 for {keyword_name}"
        assert mat.title == f"TITLE_{keyword_name}"
        assert math.isclose(mat.rho0, 1150.0, rel_tol=1e-12)
        assert math.isclose(mat.params["bulk"], 1.6e7, rel_tol=1e-12)
        assert math.isclose(mat.params["g0"], 4.2e6, rel_tol=1e-12)
        assert math.isclose(mat.params["gi"], 1.4e6, rel_tol=1e-12)
        assert math.isclose(mat.params["beta"], 22.0, rel_tol=1e-12)

    def test_model_dictionary_aliases(self, tmp_path):
        """Verify model.mat_law34s, model.mat_boltzmans, model.mat_visc_maxws alias same dict."""
        deck = StarterDeck("MODEL_DICT")
        deck.mat_law34(1, 1000.0, 1e7, 3e6, 1e6, 10.0, title="T1")
        p = tmp_path / "MD_0000.rad"
        deck.write(str(p))

        model = parse_starter_deck(str(p))
        assert model.mat_law34s is model.mat_boltzmans
        assert model.mat_law34s is model.mat_boltzmanns
        assert model.mat_law34s is model.mat_visc_maxws
        assert 1 in model.mat_law34s
        assert isinstance(model.mat_law34s[1], MatLaw34)

    def test_mat_physics_registry_all_synonyms(self):
        """Verify MAT_PHYSICS_REGISTRY registers all synonyms pointing to build_law34."""
        synonyms = [34, "34", "LAW34", "BOLTZMAN", "VISC_MAXW", "BOLTZMANN"]
        for syn in synonyms:
            assert syn in MAT_PHYSICS_REGISTRY, f"Synonym {syn!r} missing from MAT_PHYSICS_REGISTRY"
            builder = MAT_PHYSICS_REGISTRY[syn]
            assert callable(builder)
            assert builder is build_law34

    def test_catalogue_canonical_law_name(self):
        """Verify CfgCatalogue.canonical_law_name resolves all spellings to LAW34."""
        cat = catalogue()
        for spelling in ("BOLTZMAN", "VISC_MAXW", "BOLTZMANN", "LAW34"):
            assert cat.canonical_law_name(spelling) == "LAW34"


# =============================================================================
# 4. Starter Topology Checks Audit (pyradioss.starter.checks)
# =============================================================================

class TestLaw34StarterTopologyChecks:
    """Audit starter checks across element topologies and illegal inputs."""

    def test_allowed_laws_contains_all_law34_synonyms(self):
        """Verify _ALLOWED_LAWS contains LAW34 and synonyms for all applicable families."""
        families = ["bricks", "tetras", "shells", "shells_qbat", "shells_qeph", "sh3n", "trusses", "beams"]
        for fam in families:
            assert fam in _ALLOWED_LAWS, f"Family {fam} not in _ALLOWED_LAWS"
            allowed = _ALLOWED_LAWS[fam]
            for syn in (34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW"):
                assert syn in allowed, f"Synonym {syn!r} not in _ALLOWED_LAWS[{fam!r}]"

    def test_topology_end_to_end_all_element_types(self, tmp_path):
        """Verify check_model accepts LAW34 across all 8 supported element formulations:

        Bricks (8-node), Tetras (4-node), Penta6 (6-node wedge), Pyra5 (5-node pyramid),
        Shells (BT), SH3N, QBAT, QEPH, Trusses, and Beams.
        """
        deck = StarterDeck("ALL_TOPO")
        deck.mat_law34(1, 1000.0, 1.0e7, 3.0e6, 1.0e6, 10.0, title="RUBBER_LAW34")

        # 35 nodes along a line
        deck.node([(i, float(i), 0.0, 0.0) for i in range(1, 35)])

        # Properties
        deck.prop_solid(1, "PROP_SOLID")
        deck.prop_shell(2, "PROP_SH_BT", 1.0, ishell=1)
        deck.prop_shell(3, "PROP_SH_QBAT", 1.0, ishell=12)
        deck.prop_shell(4, "PROP_SH_QEPH", 1.0, ishell=24)
        deck.prop_truss(5, "PROP_TRUSS", 1.0)
        deck.prop_beam(6, "PROP_BEAM", 1.0, 1.0, 1.0, 1.0)

        # Parts
        deck.part(1, "P_BRICK", 1, 1)
        deck.part(2, "P_TETRA", 1, 1)
        deck.part(3, "P_PENTA", 1, 1)
        deck.part(4, "P_PYRA", 1, 1)
        deck.part(5, "P_SHELL", 2, 1)
        deck.part(6, "P_QBAT", 3, 1)
        deck.part(7, "P_QEPH", 4, 1)
        deck.part(8, "P_SH3N", 2, 1)
        deck.part(9, "P_TRUSS", 5, 1)
        deck.part(10, "P_BEAM", 6, 1)

        # Elements
        # 1. 8-node Hexahedral Brick
        deck.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
        # 2. 4-node Tetrahedral
        deck.tetra4(2, [(2, 9, 10, 11, 12)])
        # 3. 6-node Wedge / Penta6 (collapsed hexahedron: 6 distinct nodes)
        deck.brick(3, [(3, 13, 13, 14, 15, 16, 16, 17, 18)])
        # 4. 5-node Pyramid / Pyra5 (collapsed hexahedron: 5 distinct nodes)
        deck.brick(4, [(4, 19, 19, 19, 19, 20, 21, 22, 23)])
        # 5. 4-node BT Shell
        deck.shell(5, [(5, 24, 25, 26, 27)])
        # 6. 4-node QBAT Shell
        deck.shell(6, [(6, 24, 25, 26, 27)])
        # 7. 4-node QEPH Shell
        deck.shell(7, [(7, 24, 25, 26, 27)])
        # 8. 3-node Triangular Shell (SH3N)
        deck.sh3n(8, [(8, 24, 25, 26)])
        # 9. 2-node Truss
        deck.truss(9, [(9, 28, 29)])
        # 10. 3-node Beam
        deck.beam(10, [(10, 30, 31, 32)])

        p = tmp_path / "ALL_TOPO_0000.rad"
        deck.write(str(p))

        model = parse_starter_deck(str(p))
        log = MessageLog()
        resolve_materials(model, log)
        build_element_groups(model, log)

        group_names = [name for name, _ in model.element_groups()]
        expected_groups = [
            "bricks", "tetras", "shells", "shells_qbat", "shells_qeph",
            "sh3n", "trusses", "beams"
        ]
        for eg in expected_groups:
            assert eg in group_names, f"Expected element group {eg} not built"

        check_model(model, log)
        assert len(log.errors) == 0, f"check_model produced unexpected errors: {log.errors}"

    def test_illegal_negative_density_error(self):
        """Density RHO <= 0 must trigger an error."""
        mat = Material(
            id=1,
            law=34,
            rho0=-1000.0,
            params={"bulk": 1e7, "g0": 3e6, "gi": 1e6, "beta": 10.0},
        )
        log = MessageLog()
        check_mat_law34(mat, log)
        assert any("initial density RHO must be > 0" in msg for msg in log.errors)

    def test_illegal_zero_density_error(self):
        """Density RHO == 0 must trigger an error."""
        mat = Material(
            id=2,
            law=34,
            rho0=0.0,
            params={"bulk": 1e7, "g0": 3e6, "gi": 1e6, "beta": 10.0},
        )
        log = MessageLog()
        check_mat_law34(mat, log)
        assert any("initial density RHO must be > 0" in msg for msg in log.errors)

    def test_illegal_negative_bulk_error(self):
        """Bulk modulus BULK <= 0 must trigger an error."""
        mat = Material(
            id=3,
            law=34,
            rho0=1000.0,
            params={"bulk": -1.0e7, "g0": 3e6, "gi": 1e6, "beta": 10.0},
        )
        log = MessageLog()
        check_mat_law34(mat, log)
        assert any("bulk modulus BULK must be > 0" in msg for msg in log.errors)

    def test_illegal_zero_bulk_error(self):
        """Bulk modulus BULK == 0 must trigger an error."""
        mat = Material(
            id=4,
            law=34,
            rho0=1000.0,
            params={"bulk": 0.0, "g0": 3e6, "gi": 1e6, "beta": 10.0},
        )
        log = MessageLog()
        check_mat_law34(mat, log)
        assert any("bulk modulus BULK must be > 0" in msg for msg in log.errors)

    def test_illegal_negative_shear_g0_error(self):
        """Short-term shear modulus G0 < 0 must trigger an error."""
        mat = Material(
            id=5,
            law=34,
            rho0=1000.0,
            params={"bulk": 1e7, "g0": -5.0e5, "gi": 1e6, "beta": 10.0},
        )
        log = MessageLog()
        check_mat_law34(mat, log)
        assert any("short-term shear modulus G0 must be >= 0" in msg for msg in log.errors)

    def test_illegal_negative_shear_gi_error(self):
        """Long-term shear modulus GI < 0 must trigger an error."""
        mat = Material(
            id=6,
            law=34,
            rho0=1000.0,
            params={"bulk": 1e7, "g0": 3e6, "gi": -1.0e5, "beta": 10.0},
        )
        log = MessageLog()
        check_mat_law34(mat, log)
        assert any("long-term shear modulus GI must be >= 0" in msg for msg in log.errors)

    def test_illegal_negative_beta_error(self):
        """Decay constant BETA < 0 must trigger an error."""
        mat = Material(
            id=7,
            law=34,
            rho0=1000.0,
            params={"bulk": 1e7, "g0": 3e6, "gi": 1e6, "beta": -2.5},
        )
        log = MessageLog()
        check_mat_law34(mat, log)
        assert any("decay constant BETA must be >= 0" in msg for msg in log.errors)

    def test_warning_gi_exceeds_g0(self):
        """GI > G0 (with G0 > 0) must trigger a warning, not a fatal error."""
        mat = Material(
            id=8,
            law=34,
            rho0=1000.0,
            params={"bulk": 1e7, "g0": 2.0e6, "gi": 5.0e6, "beta": 10.0},
        )
        log = MessageLog()
        check_mat_law34(mat, log)
        assert len(log.errors) == 0, f"Expected no fatal errors, got: {log.errors}"
        assert any("exceeds short-term shear modulus G0" in msg for msg in log.warnings)

    def test_check_mat_law34_with_matlaw34_entity(self):
        """Verify check_mat_law34 works directly with MatLaw34 dataclass instances."""
        # Valid instance
        mat_valid = MatLaw34(id=9, rho0=1000.0, k=1e7, g0=3e6, gl=1e6, beta=10.0)
        log_v = MessageLog()
        check_mat_law34(mat_valid, log_v)
        assert len(log_v.errors) == 0
        assert len(log_v.warnings) == 0

        # Invalid instance (negative bulk)
        mat_bad = MatLaw34(id=10, rho0=1000.0, k=-1e7, g0=3e6, gl=1e6, beta=10.0)
        log_b = MessageLog()
        check_mat_law34(mat_bad, log_b)
        assert any("bulk modulus BULK must be > 0" in msg for msg in log_b.errors)
