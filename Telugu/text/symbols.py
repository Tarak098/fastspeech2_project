"""
Defines the phoneme and symbol vocabulary for Telugu FastSpeech 2.
Accommodates native Telugu aksharas/phonemes, matras, viramas, and code-mixed phonemes.
"""

_pad = "_"
_punctuation = list("!'(),.:;? -")
_special = ["sil", "sp"]

# Telugu Vowels & Vocalic Modifiers
_telugu_vowels = [
    "a", "aa", "i", "ii", "u", "uu", "ru", "ruu", "e", "ee", "ai", "o", "oo", "au", "am", "ah"
]

# Telugu Consonants
_telugu_consonants = [
    "k", "kh", "g", "gh", "ng",
    "ch", "chh", "j", "jh", "ny",
    "tt", "tth", "dd", "ddh", "nn",
    "t", "th", "d", "dh", "n",
    "p", "ph", "b", "bh", "m",
    "y", "r", "l", "v", "sh", "ss", "s", "h", "ll", "rr", "ksh"
]

# English loan phonemes common in contemporary Telugu speech
_loan_phonemes = [
    "f", "z", "w", "ae"
]

# Total vocabulary
symbols = [_pad] + _special + _punctuation + _telugu_vowels + _telugu_consonants + _loan_phonemes

symbol_to_id = {s: i for i, s in enumerate(symbols)}
id_to_symbol = {i: s for i, s in enumerate(symbols)}
