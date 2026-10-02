"""Phase 0 / P0.2 -- contract for the oracle's linux64_gf architecture flags file.

The upstream OpenRadioss tree resolves its compiler flags through

    include (${CMAKE_CURRENT_SOURCE_DIR}/CMake_Compilers${cdir}/cmake_${arch}.txt)

and silently falls back to ``arch=none`` (no flags at all) when that file is
absent.  A flags file that compiles but drops OpenMP is worse than no build:
the oracle would serialise and its parity scores would be meaningless.

These tests pin the load-bearing parts of ``tools/oracle/cmake_linux64_gf.txt``.
"""


def test_arch_flag_file_enables_openmp_and_double_precision():
    from pathlib import Path
    txt = Path("tools/oracle/cmake_linux64_gf.txt").read_text()
    assert "-fopenmp" in txt
    assert "-fno-second-underscore" in txt   # required: Fortran name mangling
    assert "-fdefault-real-8" not in txt     # the port is single precision


def test_openmp_is_on_the_link_line_too():
    from pathlib import Path
    txt = Path("tools/oracle/cmake_linux64_gf.txt").read_text()
    link = [l for l in txt.splitlines() if l.strip().upper().startswith("SET(LINK")]
    assert link, "must define LINK for the OpenMP link line"
    assert "-fopenmp" in link[0]


def test_no_fast_math():
    from pathlib import Path
    txt = Path("tools/oracle/cmake_linux64_gf.txt").read_text()
    for bad in ("-ffast-math", "-Ofast"):
        assert bad not in txt, bad


def test_flag_file_sets_every_variable_the_cmakelists_reads():
    from pathlib import Path
    txt = Path("tools/oracle/cmake_linux64_gf.txt").read_text()
    for var in ("CMAKE_Fortran_FLAGS_RELEASE", "CMAKE_C_FLAGS_RELEASE",
                "CMAKE_CXX_FLAGS_RELEASE", "LINK", "static_link"):
        assert var in txt, var
