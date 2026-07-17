import pathlib


def test_pgmm_package_imports():
    import pgmm  # noqa: F401


def test_vendored_tpsmm_present_with_license_and_pinned_commit():
    root = pathlib.Path(__file__).resolve().parents[1] / "third_party" / "tpsmm"
    assert (root / "LICENSE").exists(), "vendored TPSMM must ship its MIT license"
    upstream = (root / "UPSTREAM.md").read_text(encoding="utf-8")
    assert "Commit:" in upstream
    # a real 40-char sha, not a placeholder
    line = next(l for l in upstream.splitlines() if l.startswith("Commit:"))
    sha = line.split("Commit:")[1].strip()
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), (
        f"UPSTREAM.md must record a real commit hash, got {sha!r}"
    )
