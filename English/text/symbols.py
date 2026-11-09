"""
Defines the vocabulary of phonemes and symbols for English FastSpeech 2.
Includes ARPAbet phonemes with stress variants (0, 1, 2) and prosodic pause markers.
"""

_pad = "_"
_punctuation = list("!'(),.:;? -")
_special = ["sil", "sp"]

# ARPAbet phoneme inventory with stress markings
_arpabet_bases = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"
]

_arpabet = []
for p in _arpabet_bases:
    _arpabet.append(p)
    # Add lexical stress variants for vowels/diphthongs
    if p in ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]:
        _arpabet.extend([f"{p}0", f"{p}1", f"{p}2"])

# Total vocabulary
symbols = [_pad] + _special + _punctuation + _arpabet

# Mappings
symbol_to_id = {s: i for i, s in enumerate(symbols)}
id_to_symbol = {i: s for i, s in enumerate(symbols)}
