"""
English Text-to-Phoneme converter for FastSpeech 2.
Handles text normalization, numbers, abbreviations, and ARPAbet phonemization.
"""

import re
from g2p_en import G2p
from .symbols import symbol_to_id

_g2p = G2p()

_abbreviations = [
    (re.compile(r'\bMr\.\b', re.IGNORECASE), "Mister"),
    (re.compile(r'\bMrs\.\b', re.IGNORECASE), "Missus"),
    (re.compile(r'\bDr\.\b', re.IGNORECASE), "Doctor"),
    (re.compile(r'\bSt\.\b', re.IGNORECASE), "Saint"),
    (re.compile(r'\bAM\b'), "A M"),
    (re.compile(r'\bPM\b'), "P M"),
]

_num_words = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"
}

def number_to_words(num):
    if num in _num_words:
        return _num_words[num]
    if num < 100:
        tens = (num // 10) * 10
        units = num % 10
        return f"{_num_words[tens]} {_num_words[units]}"
    if num < 1000:
        hundreds = num // 100
        rem = num % 100
        res = f"{_num_words[hundreds]} hundred"
        if rem > 0:
            res += f" and {number_to_words(rem)}"
        return res
    if num < 10000:
        # e.g., 2026 -> twenty twenty six
        h = num // 100
        rem = num % 100
        return f"{number_to_words(h)} {number_to_words(rem) if rem >= 10 else ('oh ' + number_to_words(rem))}"
    return str(num)

def normalize_text(text: str) -> str:
    """Normalizes abbreviations, numbers, and times."""
    # Expand time: 10:45 -> ten forty five
    text = re.sub(r'(\d{1,2}):(\d{2})', lambda m: f"{number_to_words(int(m.group(1)))} {number_to_words(int(m.group(2)))}", text)
    
    for regex, replacement in _abbreviations:
        text = regex.sub(replacement, text)
        
    # Replace individual remaining numbers
    text = re.sub(r'\b\d+\b', lambda m: number_to_words(int(m.group())), text)
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def text_to_phonemes(text: str) -> list:
    """Converts raw English text to a sequence of ARPAbet phoneme tokens."""
    clean_text = normalize_text(text)
    phonemes = _g2p(clean_text)
    
    filtered_phonemes = []
    for p in phonemes:
        p = p.strip()
        if not p:
            continue
        if p in symbol_to_id:
            filtered_phonemes.append(p)
        elif p.isalnum():
            # If sub-phoneme with stress removed or unknown symbol
            p_clean = re.sub(r'\d+', '', p)
            if p_clean in symbol_to_id:
                filtered_phonemes.append(p_clean)
        elif p in ",.?!;:":
            filtered_phonemes.append(p)
            
    return filtered_phonemes

def text_to_sequence(text: str) -> list:
    """Converts raw text directly into symbol IDs."""
    phonemes = text_to_phonemes(text)
    return [symbol_to_id[p] for p in phonemes if p in symbol_to_id]

