import os, pathlib, pytest

OR_BUILD = os.environ.get("OR_BUILD")

@pytest.mark.skipif(not OR_BUILD, reason="OR_BUILD not configured; mirror not created")
def test_mirror_is_writable_and_has_extlib():
    m = pathlib.Path(OR_BUILD)
    assert (m / "extlib" / "hm_reader").is_dir(), "extlib not fetched"
    assert (m / "CMake_Compilers" / "cmake_linux64_gf.txt").is_file()
    probe = m / "_write_probe.tmp"
    probe.write_text("x"); probe.unlink()      # the real assertion: it is writable

@pytest.mark.skipif(not OR_BUILD, reason="OR_BUILD not configured")
def test_upstream_source_is_untouched():
    import subprocess
    r = subprocess.run(["git","-C","/home/valentin/Projects/OpenRadioss/OpenCourant",
                        "status","--porcelain"], capture_output=True, text=True)
    assert r.stdout.strip() == "", r.stdout