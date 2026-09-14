from __future__ import annotations

# ruff: noqa: RUF001 -- Turkish grapheme inventory intentionally contains dotless i.
# NeMo is an optional trainer-only dependency.
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import (
    BaseCharsTokenizer,
)

from turkish_tts.normalize import normalize_for_model, turkish_lower

TURKISH_GRAPHEMES = "abcçdefgğhıijklmnoöprsştuüvyzqwxâîû"
TURKISH_PUNCTUATION = (",", ".", "!", "?", ";", ":", "-", "(", ")")


class TurkishCharsTokenizer(BaseCharsTokenizer):  # type: ignore[misc]
    """Deterministic grapheme tokenizer for normalized Turkish model text."""

    def __init__(
        self,
        punct: bool = True,
        apostrophe: bool = True,
        add_blank_at: str | None = None,
        pad_with_space: bool = True,
    ) -> None:
        super().__init__(
            chars=TURKISH_GRAPHEMES,
            punct=punct,
            apostrophe=apostrophe,
            add_blank_at=add_blank_at,
            pad_with_space=pad_with_space,
            non_default_punct_list=list(TURKISH_PUNCTUATION),
            text_preprocessing_func=lambda text: turkish_lower(normalize_for_model(text)),
        )
