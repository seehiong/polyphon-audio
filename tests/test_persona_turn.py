import sys

import persona_turn
import pytest


def test_require_passes_when_path_exists(tmp_path):
    p = tmp_path / "exists.txt"
    p.write_text("x")
    persona_turn.require(p, "should not raise")


def test_require_exits_when_path_missing(tmp_path):
    p = tmp_path / "missing.txt"
    with pytest.raises(SystemExit, match="custom message"):
        persona_turn.require(p, "custom message")


def test_prompt_and_prompt_audio_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["persona_turn.py", "--prompt", "hi", "--prompt-audio", "x.wav", "--out", "out.wav"]
    )
    with pytest.raises(SystemExit):
        persona_turn.main()


def test_prompt_source_is_required(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["persona_turn.py", "--out", "out.wav"])
    with pytest.raises(SystemExit):
        persona_turn.main()


def _candidates(tmp_path):
    return [
        (tmp_path / "build_hip" / "bin" / "audiocpp_cli", "hip"),
        (tmp_path / "build" / "windows-cuda-release" / "bin" / "audiocpp_cli.exe", "cuda"),
        (tmp_path / "build" / "windows-cpu-release" / "bin" / "audiocpp_cli.exe", "cpu"),
    ]


def test_detect_build_picks_first_existing(tmp_path):
    candidates = _candidates(tmp_path)
    cpu_cli = candidates[2][0]
    cpu_cli.parent.mkdir(parents=True)
    cpu_cli.write_text("")
    assert persona_turn.detect_build(candidates, windows=False) == (cpu_cli, "cpu")


@pytest.mark.parametrize(("windows", "expected"), [(False, "hip"), (True, "cuda")])
def test_detect_build_falls_back_per_platform_when_nothing_built(tmp_path, windows, expected):
    assert persona_turn.detect_build(_candidates(tmp_path), windows=windows)[1] == expected
