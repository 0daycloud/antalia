from __future__ import annotations

# ruff: noqa: RUF001 -- Turkish orthography intentionally uses dotless i
import pytest

from turkish_tts.normalize import normalize_for_model, number_to_words


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A4 kağıt", "a dört kağıt"),
        ("mp3 dosyası", "mp üç dosyası"),
        ("COVID19 testi", "ce o ve ı de on dokuz testi"),
        ("5kg un", "beş kilogram un"),
        ("S8 ve A7 modelleri", "se sekiz ve a yedi modelleri"),
        ("5G şebeke", "beş ge şebeke"),
        ("H2O", "he iki o"),
    ],
)
def test_digits_glued_to_letters_are_expanded(text: str, expected: str) -> None:
    # Before the split, "A7" reached the model as the raw digit 7, a character the v3
    # vocabulary never contained.
    assert normalize_for_model(text) == expected


def test_glued_digit_split_leaves_dates_times_and_suffixes_alone() -> None:
    assert normalize_for_model("29.10.2026'da 14:05'te") == (
        "yirmi dokuz ekim iki bin yirmi altı'da on dört sıfır beş'te"
    )
    assert normalize_for_model("2024'te") == "iki bin yirmi dört'te"


@pytest.mark.parametrize(
    "text",
    [
        "Dr. Ayşe 29.10.2026'da saat 14:05'te ₺12,50 ve %3 ödedi.",
        "2.099 TL; 08.00–22.00; 7/24",
        "COVID19 için A4 form, mp3 kaydı ve 1.000.000.000.000 lira.",
    ],
)
def test_model_normalization_is_idempotent(text: str) -> None:
    # Serving now normalizes before chunking and again per chunk; a second pass must be a no-op.
    once = normalize_for_model(text)
    assert normalize_for_model(once) == once


def test_number_to_words_reaches_trillions() -> None:
    assert number_to_words(1_000_000_000_000) == "bir trilyon"
    assert number_to_words(2_500_000_000_000) == "iki trilyon beş yüz milyar"
