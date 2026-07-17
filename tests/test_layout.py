import pytest

from pgmm.data.layout import materialise_split


def _make_prepared(root, clip_ids, n_frames=3):
    for cid in clip_ids:
        d = root / cid
        d.mkdir(parents=True)
        for i in range(n_frames):
            (d / f"frame_{i:05d}.jpg").write_bytes(b"x")
    (root / "index.json").write_text("{}")
    return root


def _all_clip_ids():
    return [
        f"{sign:03d}_{signer:03d}_{rep:03d}"
        for sign in range(1, 65)
        for signer in range(1, 11)
        for rep in range(1, 6)
    ]


def test_creates_train_and_test_dirs(tmp_path):
    src = _make_prepared(tmp_path / "flat", ["001_001_001", "002_002_002"])
    out = tmp_path / "laid_out"
    materialise_split(src, out, mode="random", seed=0)
    assert (out / "train").is_dir()
    assert (out / "test").is_dir()


def test_split_counts_match_the_paper(tmp_path):
    """TPSMM silently falls back to its own 80/20 split when train/ is absent,
    which would give 2560/640 instead of the paper's 2800/400 and quietly
    ignore D9. The layout is what makes our split real."""
    src = _make_prepared(tmp_path / "flat", _all_clip_ids(), n_frames=1)
    out = tmp_path / "laid_out"
    counts = materialise_split(src, out, mode="random", seed=0)
    assert counts == {"train": 2800, "test": 400}
    assert len(list((out / "train").iterdir())) == 2800
    assert len(list((out / "test").iterdir())) == 400


def test_frames_are_carried_over(tmp_path):
    src = _make_prepared(tmp_path / "flat", ["001_001_001"], n_frames=4)
    out = tmp_path / "laid_out"
    materialise_split(src, out, mode="random", seed=0)
    placed = list(out.rglob("001_001_001"))
    assert len(placed) == 1
    assert len(list(placed[0].glob("frame_*.jpg"))) == 4


def test_index_json_is_not_treated_as_a_clip(tmp_path):
    src = _make_prepared(tmp_path / "flat", ["001_001_001", "002_002_002"])
    out = tmp_path / "laid_out"
    counts = materialise_split(src, out, mode="random", seed=0)
    assert sum(counts.values()) == 2
    assert not (out / "train" / "index.json").exists()
    assert not (out / "test" / "index.json").exists()


def test_train_and_test_are_disjoint(tmp_path):
    src = _make_prepared(tmp_path / "flat", _all_clip_ids(), n_frames=1)
    out = tmp_path / "laid_out"
    materialise_split(src, out, mode="random", seed=0)
    train = {p.name for p in (out / "train").iterdir()}
    test = {p.name for p in (out / "test").iterdir()}
    assert train.isdisjoint(test)


def test_is_deterministic_for_a_seed(tmp_path):
    ids = _all_clip_ids()
    src = _make_prepared(tmp_path / "flat", ids, n_frames=1)
    a, b = tmp_path / "a", tmp_path / "b"
    materialise_split(src, a, mode="random", seed=0)
    materialise_split(src, b, mode="random", seed=0)
    assert {p.name for p in (a / "test").iterdir()} == {
        p.name for p in (b / "test").iterdir()
    }


def test_refuses_to_overwrite_an_existing_layout(tmp_path):
    src = _make_prepared(tmp_path / "flat", ["001_001_001"])
    out = tmp_path / "laid_out"
    materialise_split(src, out, mode="random", seed=0)
    with pytest.raises(FileExistsError, match="already exists"):
        materialise_split(src, out, mode="random", seed=0)


def test_rejects_empty_source(tmp_path):
    src = tmp_path / "empty"
    src.mkdir()
    with pytest.raises(ValueError, match="no prepared clips"):
        materialise_split(src, tmp_path / "out", mode="random", seed=0)
