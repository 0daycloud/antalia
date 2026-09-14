# ruff: noqa: RUF001 RUF002 -- IPA output intentionally uses non-ASCII phonetic symbols.
"""Deterministic Turkish grapheme-to-IPA conversion for model input.

Turkish orthography is nearly phonemic, so a small rule set covers standard
pronunciation. The rules implemented here, in application order:

- Base letter mapping, including dotless ı -> ɯ and the front rounded vowels.
- Loanword letters: q -> k, w -> v, x -> ks; circumflex vowels â/î/û are long.
- Palatalization: k -> c, g -> ɟ before front vowels (e, i, ö, ü).
- Dark l: l -> ɫ after a back vowel (a, ı, o, u), clear l elsewhere.
- Soft g (ğ) lengthens the preceding vowel and is otherwise silent.

The input is expected to be model-normalized text (see normalize.py); casing is
handled defensively via turkish_lower. Output is IPA with original word
boundaries and punctuation preserved so tokenizers can keep their markers.
"""

from __future__ import annotations

import re

from turkish_tts.normalize import turkish_lower

_LETTER_RUN = re.compile(r"[^\W\d_]+")

_SIMPLE_MAP = {
    "a": "a",
    "e": "e",
    "ı": "ɯ",
    "i": "i",
    "o": "o",
    "ö": "œ",
    "u": "u",
    "ü": "y",
    "b": "b",
    "c": "dʒ",
    "ç": "tʃ",
    "d": "d",
    "f": "f",
    "h": "h",
    "j": "ʒ",
    "m": "m",
    "n": "n",
    "p": "p",
    "r": "ɾ",
    "s": "s",
    "ş": "ʃ",
    "t": "t",
    "v": "v",
    "y": "j",
    "z": "z",
    "q": "k",
    "w": "v",
    "â": "aː",
    "î": "iː",
    "û": "uː",
}
_FRONT_VOWELS = frozenset("eiöü")
_BACK_VOWELS = frozenset("aıou")
_VOWELS = frozenset("aeıioöuüâîû")
_IPA_VOWELS = frozenset("aeɯioœuy")


def _g2p_word(word: str) -> str:
    out: list[str] = []
    letters = list(word)
    for index, letter in enumerate(letters):
        next_letter = letters[index + 1] if index + 1 < len(letters) else ""
        if letter == "ğ":
            if out and out[-1][-1] in _IPA_VOWELS:
                out.append("ː")
            # otherwise silent (e.g. after a consonant, which is rare)
            continue
        if letter == "x":
            out.append("ks")
            continue
        if letter == "k":
            out.append("c" if next_letter in _FRONT_VOWELS else "k")
            continue
        if letter == "g":
            out.append("ɟ" if next_letter in _FRONT_VOWELS else "ɡ")
            continue
        if letter == "l":
            previous_letter = letters[index - 1] if index > 0 else ""
            out.append("ɫ" if previous_letter in _BACK_VOWELS else "l")
            continue
        mapped = _SIMPLE_MAP.get(letter)
        out.append(mapped if mapped is not None else letter)
    return "".join(out)


def turkish_g2p(text: str) -> str:
    """Convert normalized Turkish text to IPA, preserving spacing and punctuation.

    Letters are converted wherever they appear, including in tokens that carry attached
    punctuation. Testing a whole token with ``str.isalpha`` would skip every word ending in a
    comma or full stop -- about one word in six -- emitting raw graphemes beside IPA and
    splitting each affected phoneme across two symbols (``ç`` and ``tʃ``, ``ı`` and ``ɯ``).
    Apostrophes separate a proper noun from its suffix and carry no sound, so they are dropped.
    """
    words = turkish_lower(text).split(" ")
    converted: list[str] = []
    for word in words:
        if not word:
            continue
        converted.append(
            "".join(_LETTER_RUN.sub(lambda match: _g2p_word(match.group(0)), part) for part in word.split("'"))
        )
    return " ".join(converted)


def turkish_g2p_symbols() -> list[str]:
    """Sorted unique IPA symbols the converter can emit, for tokenizer vocabularies."""
    symbols = {"ː", "ŋ"}
    for value in _SIMPLE_MAP.values():
        symbols.update(value)
    symbols.update(("c", "ɟ", "ɫ", "k", "s", "l"))
    return sorted(symbols)
