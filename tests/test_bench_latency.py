import io
import wave

import bench_latency


def test_percentile_nearest_rank():
    values = [50.0, 10.0, 40.0, 20.0, 30.0]
    assert bench_latency.percentile(values, 50) == 30.0
    assert bench_latency.percentile(values, 90) == 50.0
    assert bench_latency.percentile(values, 0) == 10.0


def test_summarize_empty_and_values():
    assert bench_latency.summarize([]) == {}
    assert bench_latency.summarize([100.4, 200.0, 300.0]) == {"n": 3, "median_ms": 200, "min_ms": 100, "p90_ms": 300}


def test_first_sentence_waits_for_a_complete_sentence():
    assert bench_latency.first_sentence("Cats sleep a lot") == (None, "Cats sleep a lot")
    assert bench_latency.first_sentence("Cats sleep.\nA lot! More") == ("Cats sleep.", "A lot! More")


def test_wav_seconds_reads_duration_and_ignores_json():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 8000)
    assert bench_latency.wav_seconds(buf.getvalue()) == 0.5
    assert bench_latency.wav_seconds({"text": "hi"}) is None
