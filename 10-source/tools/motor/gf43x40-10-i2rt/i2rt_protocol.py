"""Shared types and validation for the documented I2RT CAN protocol."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Register:
    address: int
    name: str
    value_type: str
    unit: str = ""


def parse_ids(text: str) -> list[int]:
    result: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first_text, last_text = item.split("-", 1)
            first, last = int(first_text, 0), int(last_text, 0)
            if last < first:
                raise ValueError(f"invalid descending ID range: {item}")
            result.extend(range(first, last + 1))
        else:
            result.append(int(item, 0))
    unique = list(dict.fromkeys(result))
    if not unique or any(not 1 <= value <= 31 for value in unique):
        raise ValueError("I2RT node IDs must be in the documented range 1..31")
    return unique
