"""
English Text-to-Phoneme converter for FastSpeech 2.
Handles robust text normalization (times, dates, numbers, ordinals, abbreviations)
and converts text to valid ARPAbet sequences with pause tokens (sp, sil).
"""

import re
from .symbols import symbol_to_id

_abbreviations = [
    (re.compile(r'\bMr\.\b', re.IGNORECASE), "Mister"),
    (re.compile(r'\bMrs\.\b', re.IGNORECASE), "Missus"),
    (re.compile(r'\bMs\.\b', re.IGNORECASE), "Miss"),
    (re.compile(r'\bDr\.\b', re.IGNORECASE), "Doctor"),
    (re.compile(r'\bProf\.\b', re.IGNORECASE), "Professor"),
    (re.compile(r'\bSt\.\b', re.IGNORECASE), "Saint"),
    (re.compile(r'\bGovt\.\b', re.IGNORECASE), "Government"),
    (re.compile(r'\bGovt-Aided\b', re.IGNORECASE), "Government Aided"),
    (re.compile(r'\bAICTE\b', re.IGNORECASE), "A I C T E"),
    (re.compile(r'\bSIP\b', re.IGNORECASE), "S I P"),
    (re.compile(r'\bTV\b', re.IGNORECASE), "T V"),
    (re.compile(r'\bDTH\b', re.IGNORECASE), "D T H"),
    (re.compile(r'\betc\.\b', re.IGNORECASE), "et cetera"),
    (re.compile(r'\bi\.e\.\b', re.IGNORECASE), "that is"),
    (re.compile(r'\be\.g\.\b', re.IGNORECASE), "for example"),
    (re.compile(r'\bAM\b'), "A M"),
    (re.compile(r'\bPM\b'), "P M"),
    (re.compile(r'%'), " percent"),
    (re.compile(r'\$'), " dollars ")
]

_ones = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen"
}
_tens = {
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"
}

_ordinals = {
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
    "5th": "fifth", "6th": "sixth", "7th": "seventh", "8th": "eighth",
    "9th": "ninth", "10th": "tenth", "11th": "eleventh", "12th": "twelfth",
    "13th": "thirteenth", "20th": "twentieth", "21st": "twenty first",
    "30th": "thirtieth", "31st": "thirty first"
}

def number_to_words(n: int) -> str:
    """Converts integers from 0 to 999,999,999 to English words."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n in _ones:
        return _ones[n]
    if n < 100:
        t = (n // 10) * 10
        rem = n % 10
        return _tens[t] + ("" if rem == 0 else f" {_ones[rem]}")
    if n < 1000:
        h = n // 100
        rem = n % 100
        return f"{_ones[h]} hundred" + ("" if rem == 0 else f" and {number_to_words(rem)}")
    if n < 10000:
        # Years e.g. 2026 -> twenty twenty-six, 1986 -> nineteen eighty-six
        century = n // 100
        rem = n % 100
        if rem == 0:
            return f"{number_to_words(century)} hundred"
        elif rem < 10:
            return f"{number_to_words(century)} oh {number_to_words(rem)}"
        else:
            return f"{number_to_words(century)} {number_to_words(rem)}"
    if n < 1000000:
        th = n // 1000
        rem = n % 1000
        return f"{number_to_words(th)} thousand" + ("" if rem == 0 else f" {number_to_words(rem)}")
    if n < 1000000000:
        m = n // 1000000
        rem = n % 1000000
        return f"{number_to_words(m)} million" + ("" if rem == 0 else f" {number_to_words(rem)}")
    return str(n)

def normalize_text(text: str) -> str:
    """Normalizes English text: time, dates, ordinals, numbers, abbreviations."""
    # 1. Expand Time (e.g. 10:45 AM -> ten forty five A M)
    text = re.sub(r'(\d{1,2}):(\d{2})', lambda m: f"{number_to_words(int(m.group(1)))} {number_to_words(int(m.group(2)))}", text)

    # 2. Expand Ordinals
    for ord_key, ord_val in _ordinals.items():
        text = re.sub(rf'\b{ord_key}\b', ord_val, text, flags=re.IGNORECASE)

    # 3. Expand Abbreviations
    for regex, replacement in _abbreviations:
        text = regex.sub(replacement, text)

    # 4. Expand Decimal Numbers (e.g. 3.14 -> three point one four)
    def _expand_decimal(m):
        integer_part = number_to_words(int(m.group(1)))
        decimals = " ".join([number_to_words(int(d)) for d in m.group(2)])
        return f"{integer_part} point {decimals}"
    text = re.sub(r'\b(\d+)\.(\d+)\b', _expand_decimal, text)

    # 5. Expand standalone integers
    text = re.sub(r'\b\d+\b', lambda m: number_to_words(int(m.group())), text)

    # 6. Normalize Whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Lazy-loaded G2P instance
_g2p_instance = None

def get_g2p():
    global _g2p_instance
    if _g2p_instance is None:
        try:
            from g2p_en import G2p
            _g2p_instance = G2p()
        except ImportError:
            _g2p_instance = None
    return _g2p_instance

def text_to_phonemes(text: str) -> list:
    """
    Converts English text to a sequence of ARPAbet phonemes and prosodic pause markers.
    Returns:
        List of phoneme strings matching symbols.py.
    """
    clean_text = normalize_text(text)
    g2p = get_g2p()
    if g2p is not None:
        raw_phones = g2p(clean_text)
    else:
        # Basic ASCII character fallback if g2p_en is missing
        raw_phones = list(clean_text.upper())

    filtered_phones = []
    for p in raw_phones:
        p = p.strip()
        if not p:
            continue
        if p in [",", ";", ":"]:
            filtered_phones.append("sp")
        elif p in [".", "!", "?"]:
            filtered_phones.append("sil")
        elif p in symbol_to_id:
            filtered_phones.append(p)
        elif p.isalnum():
            # If stress variation (e.g., AH3 -> AH)
            p_clean = re.sub(r'\d+', '', p)
            if p_clean in symbol_to_id:
                filtered_phones.append(p_clean)
        elif p in "-'":
            pass  # omit intra-word punctuation

    # Avoid empty phoneme sequence
    if len(filtered_phones) == 0:
        filtered_phones = ["sil"]

    return filtered_phones

def text_to_sequence(text: str) -> list:
    """Converts raw text directly into symbol IDs."""
    phonemes = text_to_phonemes(text)
    return [symbol_to_id[p] for p in phonemes if p in symbol_to_id]
