"""Tests for the checkpoint format: save, load, and finding the newest file.

A checkpoint must rebuild the network from the file alone (no code-side
config), carry everything a training run needs to resume, and stay plain
data so `torch.load(..., weights_only=True)` can read it safely.
"""

import os
from dataclasses import fields

import numpy as np
import pytest
import torch

from ox_zero.engine.encoding import NUM_PLANES
from ox_zero.engine.network import Network, NetworkConfig
from ox_zero.engine.search import SearchConfig
from ox_zero.training.checkpoint import (
    CHECKPOINT_FORMAT,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)

TINY = NetworkConfig(blocks=1, filters=4, value_hidden=8)

TOP_LEVEL_KEYS = {
    "format_version",
    "size",
    "generation",
    "network_config",
    "model_state",
    "optimizer_state",
    "configs",
    "rng",
}


def tiny_network(size: int = 4, seed: int = 0) -> Network:
    torch.manual_seed(seed)
    network = Network(size, TINY)
    # Push BatchNorm's running statistics away from their defaults (0 and 1),
    # so the round-trip test proves they are saved and not just re-created.
    network.train()
    with torch.no_grad():
        network(torch.randn(8, *network_input_shape(network)))
    network.eval()
    return network


def network_input_shape(network: Network) -> tuple[int, int, int]:
    return NUM_PLANES, network.size, network.size


# --- Round trips ---------------------------------------------------------------


def test_round_trip_restores_the_network_exactly(tmp_path):
    network = tiny_network()
    path = tmp_path / "gen_001.pt"
    save_checkpoint(path, network, generation=1)

    checkpoint = load_checkpoint(path)
    assert checkpoint.size == 4
    assert checkpoint.generation == 1
    assert checkpoint.network.config == TINY

    # state_dict holds parameters *and* buffers: BatchNorm's running_mean,
    # running_var and num_batches_tracked are buffers, and eval-mode output
    # depends on them, so they must survive too.
    original, loaded = network.state_dict(), checkpoint.network.state_dict()
    assert original.keys() == loaded.keys()
    assert any("running_mean" in key for key in original)
    for key in original:
        assert torch.equal(original[key], loaded[key]), key

    x = torch.randn(2, *network_input_shape(network))
    checkpoint.network.eval()
    with torch.no_grad():
        for a, b in zip(network(x), checkpoint.network(x)):
            assert torch.equal(a, b)


def test_optimizer_state_round_trips(tmp_path):
    network = tiny_network()
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    # One step so Adam's moment estimates (exp_avg, exp_avg_sq) exist.
    network.train()
    policy, value = network(torch.randn(4, *network_input_shape(network)))
    (policy.sum() + value.sum()).backward()
    optimizer.step()

    path = tmp_path / "gen_002.pt"
    save_checkpoint(path, network, generation=2, optimizer=optimizer)
    checkpoint = load_checkpoint(path)

    fresh = torch.optim.AdamW(checkpoint.network.parameters(), lr=1e-3)
    fresh.load_state_dict(checkpoint.optimizer_state)
    _assert_nested_equal(fresh.state_dict(), optimizer.state_dict())


def test_optimizer_state_is_none_without_an_optimizer(tmp_path):
    path = tmp_path / "gen_000.pt"
    save_checkpoint(path, tiny_network(), generation=0)
    assert load_checkpoint(path).optimizer_state is None


def test_configs_round_trip_as_field_dicts(tmp_path):
    config = SearchConfig(c_init=2.0)
    path = tmp_path / "gen_000.pt"
    save_checkpoint(path, tiny_network(), generation=0, configs={"search": config})

    stored = load_checkpoint(path).configs
    assert set(stored) == {"search"}
    assert set(stored["search"]) == {field.name for field in fields(SearchConfig)}
    assert stored["search"]["c_init"] == 2.0
    assert stored["search"]["c_base"] == 19652.0
    assert SearchConfig(**stored["search"]) == config


def test_rng_state_round_trips(tmp_path):
    rng = np.random.default_rng(3)
    rng.random(5)
    path = tmp_path / "gen_000.pt"
    save_checkpoint(path, tiny_network(), generation=0, rng=rng)
    expected_numpy = rng.random()
    expected_torch = torch.rand(1)

    checkpoint = load_checkpoint(path)
    restored = np.random.default_rng()
    restored.bit_generator.state = checkpoint.rng["numpy"]
    assert restored.random() == expected_numpy
    torch.set_rng_state(checkpoint.rng["torch"])
    assert torch.equal(torch.rand(1), expected_torch)


def test_rng_is_none_when_not_given(tmp_path):
    path = tmp_path / "gen_000.pt"
    save_checkpoint(path, tiny_network(), generation=0)
    assert load_checkpoint(path).rng is None


# --- The file itself -----------------------------------------------------------


def test_file_is_plain_data(tmp_path):
    # weights_only=True refuses to unpickle arbitrary classes. If this ever
    # fails, someone stored a dataclass or another object in the file.
    path = tmp_path / "gen_000.pt"
    save_checkpoint(
        path,
        tiny_network(),
        generation=0,
        optimizer=torch.optim.AdamW(tiny_network().parameters()),
        configs={"search": SearchConfig()},
        rng=np.random.default_rng(0),
    )
    data = torch.load(path, weights_only=True)
    assert set(data) == TOP_LEVEL_KEYS
    assert data["format_version"] == CHECKPOINT_FORMAT


def test_save_creates_parent_directories(tmp_path):
    path = tmp_path / "run" / "deep" / "gen_000.pt"
    save_checkpoint(path, tiny_network(), generation=0)
    assert path.exists()


def test_unknown_format_version_is_rejected(tmp_path):
    path = tmp_path / "gen_000.pt"
    save_checkpoint(path, tiny_network(), generation=0)
    data = torch.load(path, weights_only=True)
    data["format_version"] = 99
    torch.save(data, path)
    with pytest.raises(ValueError, match="99"):
        load_checkpoint(path)


def test_save_leaves_no_temporary_files(tmp_path):
    # The write goes to a temporary file that is renamed into place, so an
    # interrupted save never leaves a truncated gen_N.pt for
    # `latest_checkpoint` to pick. A finished save leaves only the target.
    save_checkpoint(tmp_path / "gen_000.pt", tiny_network(), generation=0)
    assert [p.name for p in tmp_path.iterdir()] == ["gen_000.pt"]


def test_saved_file_has_the_usual_permissions(tmp_path):
    # The atomic save writes through a temporary file; the checkpoint must
    # still get the mode a plain write would give it, or other users on a
    # shared machine cannot load it. Compared against a file made by `open`.
    # A known umask, so the expected mode is exactly 0o644 and an owner-only
    # (0o600) regression cannot hide behind a strict umask on the machine.
    previous = os.umask(0o022)
    try:
        path = tmp_path / "gen_000.pt"
        save_checkpoint(path, tiny_network(), generation=0)
    finally:
        os.umask(previous)
    assert path.stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"not a checkpoint at all", id="garbage"),
        pytest.param(None, id="a bare tensor"),
        pytest.param({"format_version": 1}, id="missing keys"),
    ],
)
def test_a_malformed_file_is_a_value_error(tmp_path, content):
    # Whatever is wrong with the file, callers (the CLI) get one exception
    # type with the path in the message, not a raw traceback.
    path = tmp_path / "gen_000.pt"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        torch.save(torch.zeros(3) if content is None else content, path)
    with pytest.raises(ValueError, match="gen_000.pt"):
        load_checkpoint(path)


def test_an_architecture_mismatch_is_a_value_error(tmp_path):
    path = tmp_path / "gen_000.pt"
    save_checkpoint(path, tiny_network(), generation=0)
    data = torch.load(path, weights_only=True)
    data["network_config"]["filters"] = 8  # weights are for 4 filters
    torch.save(data, path)
    with pytest.raises(ValueError, match="gen_000.pt"):
        load_checkpoint(path)


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA")
def test_an_unavailable_device_is_a_value_error(tmp_path):
    path = tmp_path / "gen_000.pt"
    save_checkpoint(path, tiny_network(), generation=0)
    with pytest.raises(ValueError, match="cuda"):
        load_checkpoint(path, torch.device("cuda"))


def test_a_device_failure_is_a_value_error_with_the_cause(tmp_path, monkeypatch):
    # Whatever goes wrong moving the weights (no such device, out of memory),
    # callers get a ValueError that names the device and keeps torch's own
    # message, so an out-of-memory error still reads as one.
    path = tmp_path / "gen_000.pt"
    save_checkpoint(path, tiny_network(), generation=0)

    def fail(self, *args, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(Network, "to", fail)
    with pytest.raises(ValueError) as info:
        load_checkpoint(path, torch.device("mps"))
    message = str(info.value)
    assert "out of memory" in message and "mps" in message and "gen_000.pt" in message


def test_loads_onto_the_requested_device(tmp_path):
    path = tmp_path / "gen_000.pt"
    save_checkpoint(path, tiny_network(), generation=0)
    network = load_checkpoint(path, torch.device("cpu")).network
    assert all(p.device.type == "cpu" for p in network.parameters())


# --- latest_checkpoint ---------------------------------------------------------


def test_latest_is_the_highest_generation_across_runs(tmp_path):
    for name in ["a/gen_003.pt", "b/gen_010.pt", "b/gen_002.pt", "notes.txt", "b/model.pt"]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).touch()
    assert latest_checkpoint(tmp_path) == tmp_path / "b" / "gen_010.pt"


def test_latest_tie_goes_to_the_most_recently_modified(tmp_path):
    older, newer = tmp_path / "a" / "gen_005.pt", tmp_path / "b" / "gen_005.pt"
    for path in (older, newer):
        path.parent.mkdir()
        path.touch()
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))
    assert latest_checkpoint(tmp_path) == newer
    # And the other way round, so directory order can't be what decided it.
    os.utime(older, (3_000_000, 3_000_000))
    assert latest_checkpoint(tmp_path) == older


def test_latest_ignores_directories_and_dangling_links(tmp_path):
    (tmp_path / "gen_099.pt").mkdir()
    (tmp_path / "gen_050.pt").symlink_to(tmp_path / "deleted.pt")
    (tmp_path / "gen_001.pt").touch()
    assert latest_checkpoint(tmp_path) == tmp_path / "gen_001.pt"


def test_latest_is_none_for_an_empty_or_missing_root(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    assert latest_checkpoint(tmp_path / "missing") is None


# --- Helpers -------------------------------------------------------------------


def _assert_nested_equal(a, b):
    """Compare state dicts: tensors with `torch.equal`, everything else with `==`."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_nested_equal(x, y)
    else:
        assert a == b
