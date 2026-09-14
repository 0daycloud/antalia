# ruff: noqa: RUF001 -- test expectations intentionally use IPA symbols.
from turkish_tts.g2p import turkish_g2p, turkish_g2p_symbols


def test_base_vowel_and_consonant_mapping() -> None:
    assert turkish_g2p("müze") == "myze"
    assert turkish_g2p("şık") == "ʃɯk"
    assert turkish_g2p("cuma") == "dʒuma"
    assert turkish_g2p("çarşamba") == "tʃaɾʃamba"
    assert turkish_g2p("yorum") == "joɾum"


def test_palatalization_before_front_vowels() -> None:
    assert turkish_g2p("kedi") == "cedi"
    assert turkish_g2p("kitap") == "citap"
    assert turkish_g2p("göl") == "ɟœl"
    assert turkish_g2p("kalem") == "kaɫem"  # k stays plain before back vowel a
    # back-vowel contexts stay plain
    assert turkish_g2p("kapı") == "kapɯ"
    assert turkish_g2p("gün") == "ɟyn"  # ü is front -> palatal


def test_dark_l_after_back_vowels() -> None:
    assert turkish_g2p("kalp") == "kaɫp"
    assert turkish_g2p("kelime") == "celime"


def test_soft_g_lengthens_previous_vowel() -> None:
    assert turkish_g2p("dağ") == "daː"
    assert turkish_g2p("ağabey") == "aːabej"
    assert turkish_g2p("öğretmen") == "œːɾetmen"


def test_loanword_letters() -> None:
    assert turkish_g2p("taksi") == "taksi"  # already Turkish-spelled, no x involved
    assert turkish_g2p("xenofobi") == "ksenofobi"  # x only appears in foreign text
    assert turkish_g2p("hâlâ") == "haːlaː"
    assert turkish_g2p("qwerty") == "kveɾtj"


def test_casing_and_word_boundaries() -> None:
    assert turkish_g2p("İstanbul Ankara") == "istanbuɫ ankaɾa"
    assert " " in turkish_g2p("günaydın dünya")


def test_symbol_inventory_covers_output() -> None:
    symbols = set(turkish_g2p_symbols())
    sample = turkish_g2p("ğarköy kedi göl kalp taksi hâlâ müze şıracı")
    for char in sample.replace(" ", ""):
        assert char in symbols


def test_punctuation_does_not_bypass_conversion() -> None:
    """A word keeps being converted when punctuation is attached to it.

    Testing a whole token with ``str.isalpha`` skipped every word ending in a comma or full
    stop, so raw graphemes were emitted beside IPA and each affected phoneme was split across
    two symbols. Roughly one word in six carries punctuation, so the mixed alphabet was the
    common case rather than an edge case.
    """
    assert turkish_g2p("Zeynep") == turkish_g2p("Zeynep,").rstrip(",")
    assert turkish_g2p("geçti.") == "ɟetʃti."
    assert turkish_g2p("ulaştı!") == "uɫaʃtɯ!"
    assert turkish_g2p("hanım?") == "hanɯm?"

    spoken = turkish_g2p("Özge, Iğdır'a geçti. Ulaştı mı?")
    assert not set(spoken) & set("çğışöü"), f"raw Turkish graphemes leaked into IPA: {spoken}"
