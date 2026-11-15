"""Character errors: the edit distance between two texts and the character error rate (CER) of a page.

Texts are compared after the same normalisation: Unicode NFC, every run of whitespace made one space, and the ends
trimmed; or, for the CER without spaces, with every space removed. A page's CER is its total edit distance over its
reference's total characters, so long lines weigh more than short ones.
"""

import re
import unicodedata

SPACES = re.compile(r"\s+")


def normalise(text: str | None, spaces: bool = True) -> str:
    """NFC, whitespace runs as one space, trimmed; without any space when `spaces` is False."""
    found = SPACES.sub(" ", unicodedata.normalize("NFC", text or "")).strip()
    return found if spaces else found.replace(" ", "")


def edit_distance(first: str, second: str) -> int:
    """The Levenshtein distance between two texts, by code point: insertions, deletions and substitutions count one.

    Myers' bit-parallel algorithm in Hyyrö's form, the shorter text's characters as the bits of Python integers, so a
    line takes a few operations per character of the longer text."""
    if len(first) < len(second):
        first, second = second, first
    length = len(second)
    if length == 0:
        return len(first)
    matches: dict[str, int] = {}
    for position, character in enumerate(second):
        matches[character] = matches.get(character, 0) | (1 << position)
    full = (1 << length) - 1
    top = 1 << (length - 1)
    plus, minus, distance = full, 0, length
    for character in first:
        equal = matches.get(character, 0)
        vertical = equal | minus
        horizontal = ((((equal & plus) + plus) & full) ^ plus) | equal
        up = minus | (~(horizontal | plus) & full)
        down = plus & horizontal
        if up & top:
            distance += 1
        elif down & top:
            distance -= 1
        up = ((up << 1) | 1) & full
        down = (down << 1) & full
        plus = down | (~(vertical | up) & full)
        minus = up & vertical
    return distance


def rate(errors: int, characters: int) -> float:
    """Errors over characters; missing (NaN) without characters."""
    return errors / characters if characters else float("nan")
