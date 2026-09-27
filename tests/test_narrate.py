import wave

import narrate
import pytest


def make_wav(path, rate=24000, channels=1, nframes=100):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * nframes * channels)


def test_concat_wavs_joins_frames_in_order(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    make_wav(a, nframes=50)
    make_wav(b, nframes=30)
    out = tmp_path / "out.wav"

    narrate.concat_wavs([a, b], out)

    with wave.open(str(out)) as w:
        assert w.getnframes() == 80
        assert w.getframerate() == 24000


def test_concat_wavs_rejects_mismatched_sample_rate(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    make_wav(a, rate=24000)
    make_wav(b, rate=16000)
    out = tmp_path / "out.wav"

    with pytest.raises(SystemExit):
        narrate.concat_wavs([a, b], out)


def test_concat_wavs_rejects_mismatched_channels(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    make_wav(a, channels=1)
    make_wav(b, channels=2)
    out = tmp_path / "out.wav"

    with pytest.raises(SystemExit):
        narrate.concat_wavs([a, b], out)
