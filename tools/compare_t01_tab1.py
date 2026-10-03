"""Compare two T01 CSV time histories column-by-column (Paraview spelling).

Used while the RD-E-2602 ductile-failure TAB1 deck was being compared
Fortran-side vs port-side.  It is a scratch tool, but it is a *repository*
tool, so it must not name one machine's scratchpad.

Upstream origin of the two files it reads
-----------------------------------------
The Fortran side is the **binary** T01 the engine writes
(``$OR_SRC/engine/source/output/th/`` — ``hist1.F``, ``init_th0.F`` and the
rest of that directory define the record layout), rendered as CSV by
upstream's own converter (``$OR_SRC/tools/th_to_csv/README.md:1-7``; the
source lives in the separate ``OpenRadioss/Tools`` repository).  The port
writes its T01 as CSV directly.  Both files therefore carry the same columns
in the same order — which is the assumption this comparator makes, and the
reason it refuses to run rather than comparing a missing file.

Where the files are
-------------------
``$TAB1_BASE_DIR`` if set, else the current working directory; or point at
the two files with ``--fortran-csv`` / ``--port-csv``.  The historical
per-session scratchpad path is gone: a session id has no meaning on another
machine, so a hardcoded one can only ever be wrong somewhere.

Thresholds: ``rtol=0.05`` is the same 5 % relative tolerance
``tools/validate_vs_fortran.py`` calls a MATCH, so a "no difference" here
means what it means there.  The verdict rules live in that harness; this file
prints numbers and says whether the arrays are close.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

#: The two files, in the working directory / base dir.
FORTRAN_CSV_NAME = "FAILURE_TAB1T01_fortran.csv"
PORT_CSV_NAME = "FAILURE_TAB1T01.csv"

#: Environment override for the directory holding those two files.
BASE_DIR_ENV = "TAB1_BASE_DIR"


def compare_t01(f1, f2, rtol=0.05, atol=1e-3):
    d1 = np.genfromtxt(f1, delimiter=',', skip_header=1)
    d2 = np.genfromtxt(f2, delimiter=',', skip_header=1)

    if d1.shape != d2.shape:
        print(f"Shapes differ: {d1.shape} vs {d2.shape}")

    diff = np.abs(d1 - d2)
    max_diff = np.max(diff)
    print(f"Max absolute difference: {max_diff}")

    # Calculate rel_rms
    # Ignore time column (index 0) for RMS comparison if desired
    # For now, just print the max diff
    match = np.allclose(d1[:, 1:], d2[:, 1:], rtol=rtol, atol=atol)
    print(f"Match (rtol={rtol}, atol={atol}): {match}")

    if not match:
        # Find which columns differ
        for i in range(1, d1.shape[1]):
            if not np.allclose(d1[:, i], d2[:, i], rtol=rtol, atol=atol):
                print(f"Column {i} differs. Max diff: {np.max(np.abs(d1[:, i] - d2[:, i]))}")


def default_base_dir() -> Path:
    """``$TAB1_BASE_DIR`` if set, else the working directory.

    No invented default beyond that: the corpus is per-machine, so a guess
    would be a wrong path rather than a missing one.
    """
    value = os.environ.get(BASE_DIR_ENV)
    return Path(value) if value else Path.cwd()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-dir", default=None,
                    help=f"directory holding {FORTRAN_CSV_NAME} and "
                         f"{PORT_CSV_NAME} (default: ${BASE_DIR_ENV} or the "
                         f"working directory)")
    ap.add_argument("--fortran-csv", default=None)
    ap.add_argument("--port-csv", default=None)
    ap.add_argument("--rtol", type=float, default=0.05,
                    help="relative tolerance, the harness' MATCH level")
    ap.add_argument("--atol", type=float, default=1e-3)
    args = ap.parse_args(argv)

    base = Path(args.base_dir) if args.base_dir else default_base_dir()
    f1 = Path(args.fortran_csv) if args.fortran_csv else base / FORTRAN_CSV_NAME
    f2 = Path(args.port_csv) if args.port_csv else base / PORT_CSV_NAME
    missing = [str(p) for p in (f1, f2) if not p.is_file()]
    if missing:
        # Say exactly what was looked for and how to move it; a bare
        # FileNotFoundError from genfromtxt names neither.
        print(f"missing input file(s): {', '.join(missing)}\n"
              f"looked in {base}\n"
              f"pass --fortran-csv/--port-csv, or set "
              f"{BASE_DIR_ENV}=<directory holding {FORTRAN_CSV_NAME} and "
              f"{PORT_CSV_NAME}>", file=sys.stderr)
        return 2
    compare_t01(f1, f2, args.rtol, args.atol)
    return 0


if __name__ == '__main__':
    sys.exit(main())
