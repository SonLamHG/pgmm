import pytest
import yaml

from pgmm.train.session import (
    build_session_config,
    find_prepared_data,
    find_resume_checkpoint,
    steps_per_epoch,
)

CONFIG = "pgmm/config/lsa64-tpsmm.yaml"


def _make_prepared(root, name="lsa64_prepared"):
    d = root / "notebooks" / "user" / "kern" / name
    (d / "train" / "001_001_001").mkdir(parents=True)
    (d / "test" / "002_002_002").mkdir(parents=True)
    return d


def test_finds_prepared_data_under_a_nested_mount(tmp_path):
    expected = _make_prepared(tmp_path)
    assert find_prepared_data(tmp_path) == expected


def test_rejects_a_tree_without_train_and_test(tmp_path):
    """A bare lsa64_prepared is not enough: without train/ and test/,
    FramesDataset silently substitutes its own 80/20 split (D17)."""
    (tmp_path / "lsa64_prepared").mkdir()
    with pytest.raises(FileNotFoundError, match="no lsa64_prepared"):
        find_prepared_data(tmp_path)


def test_missing_data_names_the_likely_cause(tmp_path):
    with pytest.raises(FileNotFoundError, match="kernel_sources"):
        find_prepared_data(tmp_path)


def test_no_checkpoint_means_start_fresh(tmp_path):
    assert find_resume_checkpoint(tmp_path) is None


def test_picks_the_highest_epoch_checkpoint(tmp_path):
    (tmp_path / "log").mkdir()
    for epoch in (5, 49, 20):
        (tmp_path / "log" / f"{epoch:08d}-checkpoint.pth.tar").write_bytes(b"x")
    found = find_resume_checkpoint(tmp_path)
    assert found.name == "00000049-checkpoint.pth.tar"


def test_orders_by_epoch_not_mtime(tmp_path):
    """Extraction rewrites every timestamp, so mtime carries no information
    about training order."""
    import os
    import time

    (tmp_path / "log").mkdir()
    newest = tmp_path / "log" / "00000049-checkpoint.pth.tar"
    stale = tmp_path / "log" / "00000005-checkpoint.pth.tar"
    newest.write_bytes(b"x")
    stale.write_bytes(b"x")
    time.sleep(0.01)
    os.utime(stale, None)  # make the OLDER epoch the NEWEST file
    assert find_resume_checkpoint(tmp_path) == newest


def test_session_config_points_at_the_data_and_bounds_epochs(tmp_path):
    data = tmp_path / "data"
    out = tmp_path / "session.yaml"
    cfg = build_session_config(CONFIG, data, out, max_epochs=50)
    assert cfg["dataset_params"]["root_dir"] == str(data)
    assert cfg["train_params"]["num_epochs"] == 50
    written = yaml.safe_load(out.read_text())
    assert written["train_params"]["num_epochs"] == 50


def test_session_config_preserves_every_paper_parameter(tmp_path):
    """Bounding a session must not disturb the protocol under test."""
    cfg = build_session_config(CONFIG, tmp_path / "d", tmp_path / "c.yaml",
                               max_epochs=50)
    t = cfg["train_params"]
    assert t["num_repeats"] == 5              # D18
    assert t["epoch_milestones"] == [60, 90]  # paper
    assert t["lr_generator"] == 2.0e-4        # paper
    assert cfg["model_params"]["common_params"]["num_tps"] == 20  # 100 keypoints
    assert cfg["dataset_params"]["frame_shape"] == [128, 128, 3]
    assert t["loss_weights"]["perceptual"] == [10, 10, 10, 10, 10]  # D7


def test_steps_per_epoch_matches_the_measured_run(tmp_path):
    cfg = build_session_config(CONFIG, tmp_path / "d", tmp_path / "c.yaml",
                               max_epochs=1)
    # 2800 clips x 5 pairs / batch 28 = 500, as the D12 measurement observed
    assert steps_per_epoch(cfg) == 500
