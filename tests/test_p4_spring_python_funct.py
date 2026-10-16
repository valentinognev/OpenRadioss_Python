"""Tests for Task P4.5: the ``python_funct_mod`` spring callback.

Fortran origin: ``common_source/modules/python_mod.F90``
(``python_funct_id``, ``python_call_funct1D``) as called from the spring
branches ``r26def3.F`` / ``r27def3.F`` (TYPE26 ``SPR_TAB``, TYPE27
``SPR_BDAMP``): a plain ``/FUNCT`` id is evaluated with ``FINTER`` while a
Python-function id is dispatched to user code.

Port contract (``plan/05_phase4_elements_1d.md`` Task P4.5):
``pyradioss.elements.spring_python_funct.resolve(fct_id, model)`` returns one
curve evaluator shared by every branch that needs it.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyradioss.elements import spring_python_funct


def _type26_group(fct_id: int = 3):
    """Minimal TYPE26 stand-in: ``g.functs`` is the model's own table map."""
    model = SimpleNamespace(functions={})
    return SimpleNamespace(fct_id=fct_id, functs=model.functions, model=model)


def test_type26_python_curve_matches_a_hand_written_funct():
    g = _type26_group(fct_id=3)
    g.functs[3] = np.array([[0.0, 0.0], [0.5, 10.0], [1.0, 0.0]])
    out = spring_python_funct.resolve(3, g.model)(np.array([0.5]))
    assert out == pytest.approx([10.0])


def test_resolve_vectorises_and_extrapolates_like_finter():
    g = _type26_group(fct_id=3)
    g.functs[3] = np.array([[0.0, 0.0], [0.5, 10.0], [1.0, 0.0]])
    fn = spring_python_funct.resolve(3, g.model)
    out = fn(np.array([-1.0, 0.25, 0.5, 0.75, 2.0]))
    # Interior: linear interpolation; exterior: last/first-segment slope
    # (FINTER semantics: slope below is 20, slope above is -20).
    assert out == pytest.approx([-20.0, 5.0, 10.0, 5.0, -20.0])


def test_registered_python_callback_is_used_for_negative_ids():
    # Upstream passes -IFUNC to python_call_funct1D for Python functions;
    # the port resolves those through the explicit in-process registry.
    spring_python_funct.register_python_funct(11, lambda x: 2.0 * np.asarray(x))
    try:
        fn = spring_python_funct.resolve(-11, _type26_group().model)
        assert fn(np.array([1.0, 2.5])) == pytest.approx([2.0, 5.0])
    finally:
        spring_python_funct.unregister_python_funct(11)


def test_unregistered_python_funct_is_refused_not_executed(tmp_path):
    # A deck /FUNCT_PYTHON entry is source text, never implicitly executed.
    sentinel = tmp_path / "pwned.txt"
    payload = "open(" + repr(str(sentinel)) + ", 'w').write('x')"
    model = SimpleNamespace(functions={5: ("evil", payload)})
    with pytest.raises(spring_python_funct.PythonFunctNotRegistered, match="5"):
        spring_python_funct.resolve(5, model)
    assert not sentinel.exists()


def test_local_curves_can_be_disabled():
    g = _type26_group(fct_id=3)
    g.functs[3] = np.array([[0.0, 0.0], [1.0, 1.0]])
    spring_python_funct.PYTHON_FUNCT_ENABLED = False
    try:
        with pytest.raises(spring_python_funct.PythonFunctNotRegistered):
            spring_python_funct.resolve(3, g.model)
    finally:
        spring_python_funct.PYTHON_FUNCT_ENABLED = True


def test_unknown_funct_id_raises():
    with pytest.raises(LookupError):
        spring_python_funct.resolve(999, _type26_group().model)
    with pytest.raises(ValueError):
        spring_python_funct.resolve(0, _type26_group().model)


def test_module_never_execs_deck_content():
    # Static guarantee: no exec/eval/compile/__import__ calls in the module,
    # so no future edit can silently turn a deck string into code.
    src = Path(spring_python_funct.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned = {"exec", "eval", "compile", "__import__"}
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not (called & banned), f"banned calls present: {called & banned}"
