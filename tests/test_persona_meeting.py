import persona_meeting


def test_format_transcript_uses_speaker_name():
    segments = [{"speaker_name": "Alice", "text": "hello"}, {"speaker_name": "Bob", "text": "hi"}]
    assert persona_meeting.format_transcript(segments) == "Alice: hello\nBob: hi"


def test_format_transcript_falls_back_to_speaker_id():
    segments = [{"speaker_id": "SPEAKER_00", "text": "hello"}]
    assert persona_meeting.format_transcript(segments) == "SPEAKER_00: hello"


def test_format_transcript_falls_back_to_unknown():
    segments = [{"text": "hello"}]
    assert persona_meeting.format_transcript(segments) == "Unknown: hello"


def test_format_transcript_empty():
    assert persona_meeting.format_transcript([]) == ""
