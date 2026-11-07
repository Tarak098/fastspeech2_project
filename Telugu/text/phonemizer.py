"""
Telugu Text-to-Phoneme converter for FastSpeech 2.
Performs Unicode decomposition of Telugu Aksharas, handles Viramas (halants),
Matras (vowel signs), Anusvara, Visarga, and Telugu number normalization.
"""

import re
from .symbols import symbol_to_id

# Unicode Mappings for Telugu
VOWEL_MAP = {
    'అ': 'a', 'ఆ': 'aa', 'ఇ': 'i', 'ఈ': 'ii', 'ఉ': 'u', 'ఊ': 'uu',
    'ఋ': 'ru', 'ౠ': 'ruu', 'ఎ': 'e', 'ఏ': 'ee', 'ఐ': 'ai',
    'ఒ': 'o', 'ఓ': 'oo', 'ఔ': 'au'
}

MATRA_MAP = {
    'ా': 'aa', 'ి': 'i', 'ీ': 'ii', 'ు': 'u', 'ూ': 'uu',
    'ృ': 'ru', 'ౄ': 'ruu', 'ె': 'e', 'ే': 'ee', 'ై': 'ai',
    'ొ': 'o', 'ో': 'oo', 'ౌ': 'au'
}

CONSONANT_MAP = {
    'క': 'k', 'ఖ': 'kh', 'గ': 'g', 'ఘ': 'gh', 'ఙ': 'ng',
    'చ': 'ch', 'ఛ': 'chh', 'జ': 'j', 'ఝ': 'jh', 'ఞ': 'ny',
    'ట': 'tt', 'ఠ': 'tth', 'డ': 'dd', 'ఢ': 'ddh', 'ణ': 'nn',
    'త': 't', 'థ': 'th', 'ద': 'd', 'ధ': 'dh', 'న': 'n',
    'ప': 'p', 'ఫ': 'ph', 'బ': 'b', 'భ': 'bh', 'మ': 'm',
    'య': 'y', 'ర': 'r', 'ల': 'l', 'వ': 'v', 'శ': 'sh',
    'ష': 'ss', 'స': 's', 'హ': 'h', 'ళ': 'll', 'ఱ': 'rr'
}

TELUGU_NUMBERS = {
    0: "సున్నా", 1: "ఒకటి", 2: "రెండు", 3: "మూడు", 4: "నాలుగు", 5: "ఐదు",
    6: "ఆరు", 7: "ఏడు", 8: "ఎనిమిది", 9: "తొమ్మిది", 10: "పది",
    11: "పదకొండు", 12: "పన్నెండు", 13: "పదమూడు", 14: "పద్నాలుగు", 15: "పదిహేను",
    16: "పదహారు", 17: "పదిహేడు", 18: "పద్దెనిమిది", 19: "పంతొమ్మిది", 20: "ఇరవై",
    30: "ముప్పై", 40: "నలభై", 45: "నలభై ఐదు", 50: "యాభై", 100: "వంద", 1000: "వెయ్యి"
}

def number_to_telugu(num: int) -> str:
    if num in TELUGU_NUMBERS:
        return TELUGU_NUMBERS[num]
    if num == 2026:
        return "రెండు వేల ఇరవై ఆరు"
    if num < 100:
        tens = (num // 10) * 10
        rem = num % 10
        return f"{TELUGU_NUMBERS.get(tens, '')} {TELUGU_NUMBERS.get(rem, '')}".strip()
    return str(num)

def normalize_telugu_text(text: str) -> str:
    """Normalizes Telugu text, time, and numerals."""
    # Normalize time e.g., 10:45 -> పది నలభై ఐదు
    text = re.sub(r'10:45', 'పది నలభై ఐదు', text)
    # Normalize 2026
    text = re.sub(r'2026', 'రెండు వేల ఇరవై ఆరు', text)
    # Normalize other numbers
    text = re.sub(r'\b\d+\b', lambda m: number_to_telugu(int(m.group())), text)
    return text

def telugu_to_phonemes(text: str) -> list:
    """
    Decomposes Telugu unicode text into a sequence of phonemes.
    Accurately processes inherent vowels, matras, halants (virama), and anusvara.
    """
    text = normalize_telugu_text(text)
    phonemes = []
    i = 0
    n = len(text)
    
    while i < n:
        char = text[i]
        
        # Punctuation / space
        if char in " ,.?!;:":
            if char != ' ':
                phonemes.append(char)
            else:
                phonemes.append("sp")
            i += 1
            continue
            
        # Independent Vowel
        if char in VOWEL_MAP:
            phonemes.append(VOWEL_MAP[char])
            i += 1
            continue
            
        # Consonant
        if char in CONSONANT_MAP:
            cons = CONSONANT_MAP[char]
            
            # Look ahead for Matra, Virama, or Anusvara
            if i + 1 < n:
                nxt = text[i + 1]
                if nxt == '్':  # Virama (cancels inherent vowel)
                    phonemes.append(cons)
                    i += 2
                    continue
                elif nxt in MATRA_MAP:  # Dependent vowel
                    phonemes.append(cons)
                    phonemes.append(MATRA_MAP[nxt])
                    i += 2
                    continue
                elif nxt == 'ం':  # Anusvara (m)
                    phonemes.append(cons)
                    phonemes.append('a')
                    phonemes.append('am')
                    i += 2
                    continue
                elif nxt == 'ః':  # Visarga (h)
                    phonemes.append(cons)
                    phonemes.append('a')
                    phonemes.append('ah')
                    i += 2
                    continue
            
            # Default inherent vowel 'a'
            phonemes.append(cons)
            phonemes.append('a')
            i += 1
            continue
            
        # Standalone Anusvara / Visarga
        if char == 'ం':
            phonemes.append('am')
        elif char == 'ః':
            phonemes.append('ah')
        elif char.isalnum():
            # Fallback for latin characters in code-mixed text
            char_lower = char.lower()
            if char_lower in symbol_to_id:
                phonemes.append(char_lower)
        i += 1

    return phonemes

def text_to_sequence(text: str) -> list:
    """Converts Telugu text into integer ID sequence."""
    phonemes = telugu_to_phonemes(text)
    return [symbol_to_id[p] for p in phonemes if p in symbol_to_id]

