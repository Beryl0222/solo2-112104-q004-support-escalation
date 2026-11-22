"""Turn-level assembly and scrubbing.

This module turns raw records into ``Turn`` objects.  It also strips fields
that must never reach the output because they were not present in the source
(guessed names, guessed timestamps, inferred customer sentiment scores...).
Only the original text and its exact offsets survive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from .models import Evidence, Turn
from .text import agent_label_of, speaker_of, strip_speaker_prefix

ALLOWED_RECORD_FIELDS = {
    "id", "channel", "turn", "text", "timestamp", "speaker", "kind",
    "speaker_label", "conversation_id",
}
_REQUIRED = {"id", "text"}


class SourceDataError(ValueError):
    """Raised when synthetic dialogue records cannot be read safely."""


def load_turns(path: Path | str) -> list[Turn]:
    import json

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SourceDataError("dialogue file must contain a JSON list")
    turns: list[Turn] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(raw):
        if not isinstance(record, Mapping):
            raise SourceDataError(f"record #{index} is not an object")
        if not _REQUIRED <= record.keys():
            raise SourceDataError(f"record #{index} lacks id/text")
        identifier = str(record["id"])
        if identifier in seen_ids:
            raise SourceDataError(f"duplicate record id {identifier!r}")
        seen_ids.add(identifier)
        text = record["text"]
        if not isinstance(text, str) or not text.strip():
            raise SourceDataError(f"record {identifier!r} has empty text")
        unknown = set(record) - ALLOWED_RECORD_FIELDS
        if unknown:
            raise SourceDataError(
                f"record {identifier!r} has unsupported fields: {sorted(unknown)}"
            )
        turns.append(
            Turn(
                record_id=identifier,
                turn=int(record.get("turn", index + 1)),
                channel=str(record.get("channel", "unknown")),
                speaker=str(record.get("speaker") or speaker_of(text.strip())),
                text=text,
                timestamp=_optional(record.get("timestamp")),
                speaker_label=_optional(record.get("speaker_label")),
                kind=str(record.get("kind", "utterance")),
            )
        )
    return sorted(turns, key=lambda item: (item.turn, item.record_id))


def _optional(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def collect_agent_labels(turns: Sequence[Turn]) -> list[str]:
    labels: list[str] = []
    for turn in turns:
        label = agent_label_of(turn.text, ())
        if label:
            label = label.rstrip("：:")
            if label and label not in labels:
                labels.append(label)
    return labels


def make_evidence(turn: Turn, start: int, end: int) -> Evidence:
    """Slice ``turn.text`` exactly, then verify the slice."""
    if not (0 <= start < end <= len(turn.text)):
        raise SourceDataError(
            f"evidence range {start}:{end} outside record {turn.record_id!r}"
        )
    quote = turn.text[start:end]
    if turn.text.find(quote) < 0:  # defensive, should always hold
        raise SourceDataError(f"quote not found in record {turn.record_id!r}")
    label = turn.speaker_label
    if label is None and turn.speaker == "agent":
        label = agent_label_of(turn.text, ()) or "客服"
    return Evidence(
        record_id=turn.record_id,
        turn=turn.turn,
        channel=turn.channel,
        speaker=turn.speaker,
        quote=quote,
        start=start,
        end=end,
        timestamp=turn.timestamp,
        speaker_label=label,
    )


def find_quote_span(turn: Turn, quote: str, start: int = 0) -> Tuple[int, int]:
    """Locate ``quote`` at or after ``start``; exact substring only."""
    index = turn.text.find(quote, start)
    if index < 0:
        raise SourceDataError(
            f"quote {quote!r} not present in record {turn.record_id!r}"
        )
    return index, index + len(quote)


def evidence_for_turn(turn: Turn) -> Evidence:
    text = turn.text
    _, offset = strip_speaker_prefix(text)
    start = offset
    end = len(text.rstrip())
    if end <= start:
        start, end = 0, len(text)
    return make_evidence(turn, start, end)


def search_quote(turns: Sequence[Turn], record_id: str, quote: str) -> Evidence:
    """Resolve a stored quote back to its turn (used by tests and reports)."""
    for turn in turns:
        if turn.record_id != record_id:
            continue
        index = turn.text.find(quote)
        if index >= 0:
            return make_evidence(turn, index, index + len(quote))
    raise SourceDataError(f"quote {quote!r} not found in {record_id!r}")
