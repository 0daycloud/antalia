from __future__ import annotations

# ruff: noqa: RUF001 -- Turkish orthography intentionally uses dotless i
import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
_COMPARISON_PUNCTUATION = re.compile(r"[^a-zçğıöşü0-9']+")
_CTC_PUNCTUATION = re.compile(r"[^a-zçğıöşüqwx ]+")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})
_QUOTES = str.maketrans({char: None for char in '"“”„«»'})
_TURKISH_CASE = str.maketrans("Iİ", "ıi")
_ONES = ("sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz")
_TENS = ("", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan")
_MONTHS = {
    1: "ocak",
    2: "şubat",
    3: "mart",
    4: "nisan",
    5: "mayıs",
    6: "haziran",
    7: "temmuz",
    8: "ağustos",
    9: "eylül",
    10: "ekim",
    11: "kasım",
    12: "aralık",
}
_ABBREVIATIONS = {
    "dr.": "doktor",
    "prof.": "profesör",
    "sn.": "sayın",
    "vb.": "ve benzeri",
    "vs.": "vesaire",
    "örn.": "örneğin",
    "kg": "kilogram",
    "gr": "gram",
    "km": "kilometre",
    "cm": "santimetre",
    "mm": "milimetre",
    "lt": "litre",
    "ml": "mililitre",
    "kb": "kilobayt",
    "mb": "megabayt",
    "gb": "gigabayt",
    "tb": "terabayt",
    "tl": "Türk lirası",
}
_LETTER_NAMES = {
    "A": "a",
    "B": "be",
    "C": "ce",
    "Ç": "çe",
    "D": "de",
    "E": "e",
    "F": "fe",
    "G": "ge",
    "Ğ": "yumuşak ge",
    "H": "he",
    "I": "ı",
    "İ": "i",
    "J": "je",
    "K": "ke",
    "L": "le",
    "M": "me",
    "N": "ne",
    "O": "o",
    "Ö": "ö",
    "P": "pe",
    "R": "re",
    "S": "se",
    "Ş": "şe",
    "T": "te",
    "U": "u",
    "Ü": "ü",
    "V": "ve",
    "Y": "ye",
    "Z": "ze",
    "Q": "kü",
    "W": "dabılyu",
    "X": "iks",
}
_ORDINAL_EXCEPTIONS = {
    "bir": "birinci",
    "iki": "ikinci",
    "üç": "üçüncü",
    "dört": "dördüncü",
    "beş": "beşinci",
    "altı": "altıncı",
    "yedi": "yedinci",
    "sekiz": "sekizinci",
    "dokuz": "dokuzuncu",
    "on": "onuncu",
    "yirmi": "yirminci",
    "otuz": "otuzuncu",
    "kırk": "kırkıncı",
    "elli": "ellinci",
    "altmış": "altmışıncı",
    "yetmiş": "yetmişinci",
    "seksen": "sekseninci",
    "doksan": "doksanıncı",
    "yüz": "yüzüncü",
    "bin": "bininci",
    "milyon": "milyonuncu",
    "milyar": "milyarıncı",
}


def turkish_lower(text: str) -> str:
    # Strip U+0307: lowercasing İ can yield i + combining dot above, and some
    # source transcripts already contain the decomposed form. The dot is silent.
    return text.translate(_TURKISH_CASE).lower().replace("\u0307", "")


def normalize_orthography(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).translate(_QUOTES)
    return _WHITESPACE.sub(" ", normalized).strip()


def normalize_for_asr_comparison(text: str) -> str:
    normalized = turkish_lower(normalize_orthography(text))
    normalized = _COMPARISON_PUNCTUATION.sub(" ", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()


def normalize_for_scoring(text: str) -> str:
    """ASR-comparison normalization that also spells out digits Whisper may emit.

    Whisper frequently transcribes spoken Turkish numerals as digits ("3700" for
    "üç bin yedi yüz"), which inflates CER/WER against word-form targets. Scoring
    therefore spells percent signs, decimals, thousands groups, and digit runs
    before comparison. Word-form text passes through unchanged.
    """
    normalized = normalize_orthography(text)
    normalized = normalized.replace("%", " yüzde ")
    normalized = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", normalized)
    normalized = re.sub(
        r"\d+,\d+",
        lambda match: f" {_decimal_to_words(match[0])} ",
        normalized,
    )
    normalized = re.sub(r"\d+", lambda match: f" {number_to_words(int(match[0]))} ", normalized)
    return normalize_for_asr_comparison(normalized)


def normalize_for_ctc_alignment(text: str) -> str:
    normalized = normalize_for_asr_comparison(text).replace("'", " ")
    normalized = _CTC_PUNCTUATION.sub(" ", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()


def normalize_for_model(text: str) -> str:
    normalized = normalize_orthography(text)
    normalized = re.sub(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b", _replace_date, normalized)
    normalized = re.sub(r"([₺$€])\s*((?:\d{1,3}(?:\.\d{3})+)|\d+)(?:[,.](\d{1,2}))?", _replace_currency, normalized)
    normalized = re.sub(r"%\s*(\d+(?:[,.]\d+)?)", lambda match: f"yüzde {_numeric_to_words(match[1])}", normalized)
    normalized = re.sub(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b", _replace_time, normalized)
    normalized = re.sub(
        r"\b(\d{1,2})/(\d{1,2})\b",
        lambda match: f"{number_to_words(int(match[1]))} {number_to_words(int(match[2]))}",
        normalized,
    )
    normalized = re.sub(
        r"\b\d{1,3}(?:\.\d{3})+\b",
        lambda match: number_to_words(int(match[0].replace(".", ""))),
        normalized,
    )
    normalized = re.sub(r"\b(\d+)[,.](\d+)\b", lambda match: _decimal_to_words(match[0]), normalized)
    normalized = re.sub(r"\b(\d+)\.(?=\s|$)", lambda match: _ordinal_to_words(int(match[1])), normalized)
    normalized = re.sub(r"\b\d{5,}\b", lambda match: _digit_sequence_to_words(match[0]), normalized)
    normalized = re.sub(r"\b\d+\b", lambda match: number_to_words(int(match[0])), normalized)
    for abbreviation, expansion in sorted(_ABBREVIATIONS.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = re.sub(rf"(?<!\w){re.escape(abbreviation)}(?!\w)", expansion, normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b[A-ZÇĞİÖŞÜQWX]{2,6}\b", _replace_initialism, normalized)
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()


def number_to_words(number: int) -> str:
    if number < 0:
        return f"eksi {number_to_words(-number)}"
    if number < 10:
        return _ONES[number]
    if number < 100:
        tens, ones = divmod(number, 10)
        return _TENS[tens] if not ones else f"{_TENS[tens]} {_ONES[ones]}"
    if number < 1_000:
        hundreds, remainder = divmod(number, 100)
        prefix = "yüz" if hundreds == 1 else f"{_ONES[hundreds]} yüz"
        return prefix if not remainder else f"{prefix} {number_to_words(remainder)}"
    for scale, label in ((1_000_000_000, "milyar"), (1_000_000, "milyon"), (1_000, "bin")):
        if number >= scale:
            count, remainder = divmod(number, scale)
            prefix = label if scale == 1_000 and count == 1 else f"{number_to_words(count)} {label}"
            return prefix if not remainder else f"{prefix} {number_to_words(remainder)}"
    raise ValueError(f"number is too large to normalize: {number}")


def _replace_date(match: re.Match[str]) -> str:
    day, month, year = (int(value) for value in match.groups())
    if not 1 <= day <= 31 or month not in _MONTHS:
        return match[0]
    return f"{number_to_words(day)} {_MONTHS[month]} {number_to_words(year)}"


def _replace_currency(match: re.Match[str]) -> str:
    symbol, whole, fraction = match.groups()
    unit, subunit = {
        "₺": ("lira", "kuruş"),
        "$": ("dolar", "sent"),
        "€": ("avro", "sent"),
    }[symbol]
    value = f"{number_to_words(int(whole.replace('.', '')))} {unit}"
    if fraction and int(fraction):
        value += f" {number_to_words(int(fraction.ljust(2, '0')))} {subunit}"
    return value


def _replace_time(match: re.Match[str]) -> str:
    hour, minute = match.groups()
    hour_words = number_to_words(int(hour))
    if int(minute) == 0:
        return hour_words
    minute_words = _digit_sequence_to_words(minute) if minute.startswith("0") else number_to_words(int(minute))
    return f"{hour_words} {minute_words}"


def _numeric_to_words(value: str) -> str:
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", value):
        return number_to_words(int(value.replace(".", "")))
    return _decimal_to_words(value) if re.search(r"[,.]", value) else number_to_words(int(value))


def _decimal_to_words(value: str) -> str:
    whole, fraction = re.split(r"[,.]", value, maxsplit=1)
    return f"{number_to_words(int(whole))} virgül {_digit_sequence_to_words(fraction)}"


def _digit_sequence_to_words(value: str) -> str:
    return " ".join(_ONES[int(digit)] for digit in value)


def _ordinal_to_words(number: int) -> str:
    cardinal = number_to_words(number)
    words = cardinal.split()
    words[-1] = _ORDINAL_EXCEPTIONS.get(words[-1], f"{words[-1]}inci")
    return " ".join(words)


def _replace_initialism(match: re.Match[str]) -> str:
    return " ".join(_LETTER_NAMES[letter] for letter in match[0])
