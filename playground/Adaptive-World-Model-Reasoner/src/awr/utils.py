from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in TOKEN_RE.finditer(text))


def stable_hash(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def l2_normalize(values: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return tuple(0.0 for _ in values)
    return tuple(value / norm for value in values)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def vector_average(vectors: Iterable[Sequence[float]]) -> tuple[float, ...]:
    vectors = list(vectors)
    if not vectors:
        return ()
    width = len(vectors[0])
    totals = [0.0] * width
    for vector in vectors:
        for index, value in enumerate(vector):
            totals[index] += value
    return tuple(value / len(vectors) for value in totals)


def blend_vectors(
    old: Sequence[float],
    new: Sequence[float],
    learning_rate: float,
) -> tuple[float, ...]:
    if len(old) != len(new):
        return tuple(new)
    rate = max(0.0, min(1.0, learning_rate))
    return tuple((1.0 - rate) * a + rate * b for a, b in zip(old, new, strict=True))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_answer(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = value.strip().lower()
    cleaned = re.sub(r"[\s,.;:!?]+", " ", cleaned)
    cleaned = cleaned.strip()
    if re.fullmatch(r"-?\d+\.0+", cleaned):
        return cleaned.split(".", maxsplit=1)[0]
    return cleaned


def numeric_value(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None
