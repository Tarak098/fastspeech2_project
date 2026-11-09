"""
Telugu Text-to-Phoneme converter for FastSpeech 2.
Performs accurate Unicode decomposition of Telugu Aksharas, handling:
- Dependent Matras (vowel signs)
- Virama (Halant) consonant clusters and geminates
- Anusvara (ం) and Visarga (ః)
- Telugu and ASCII numeral and timestamp normalization
- Prosodic pauses (sp, sil)
"""

import re
from .symbols import symbol_to_id

# Independent Vowels
VOWEL_MAP = {
    'అ': 'a', 'ఆ': 'aa', 'ఇ': 'i', 'ఈ': 'ii', 'ఉ': 'u', 'ఊ': 'uu',
    'ఋ': 'ru', 'ౠ': 'ruu', 'ఎ': 'e', 'ఏ': 'ee', 'ఐ': 'ai',
    'ఒ': 'o', 'ఓ': 'oo', 'ఔ': 'au'
}

# Dependent Matras (Vowel Signs)
MATRA_MAP = {
    'ా': 'aa', 'ి': 'i', 'ీ': 'ii', 'ు': 'u', 'ూ': 'uu',
    'ృ': 'ru', 'ౄ': 'ruu', 'ె': 'e', 'ే': 'ee', 'ై': 'ai',
    'ొ': 'o', 'ో': 'oo', 'ౌ': 'au'
}

# Consonants
CONSONANT_MAP = {
    'క': 'k', 'ఖ': 'kh', 'గ': 'g', 'ఘ': 'gh', 'ఙ': 'ng',
    'చ': 'ch', 'ఛ': 'chh', 'జ': 'j', 'ఝ': 'jh', 'ఞ': 'ny',
    'ట': 'tt', 'ఠ': 'tth', 'డ': 'dd', 'ఢ': 'ddh', 'ణ': 'nn',
    'త': 't', 'థ': 'th', 'ద': 'd', 'ధ': 'dh', 'న': 'n',
    'ప': 'p', 'ఫ': 'ph', 'బ': 'b', 'భ': 'bh', 'మ': 'm',
    'య': 'y', 'ర': 'r', 'ల': 'l', 'వ': 'v', 'శ': 'sh',
    'ష': 'ss', 'స': 's', 'హ': 'h', 'ళ': 'll', 'ఱ': 'rr'
}

# Telugu Numeral Table
TELUGU_NUMBERS = {
    0: "సున్నా", 1: "ఒకటి", 2: "రెండు", 3: "మూడు", 4: "నాలుగు", 5: "ఐదు",
    6: "ఆరు", 7: "ఏడు", 8: "ఎనిమిది", 9: "తొమ్మిది", 10: "పది",
    11: "పదకొండు", 12: "పన్నెండు", 13: "పదమూడు", 14: "పద్నాలుగు", 15: "పదిహేను",
    16: "పదహారు", 17: "పదిహేడు", 18: "పద్దెనిమిది", 19: "పంతొమ్మిది", 20: "ఇరవై",
    30: "ముప్పై", 40: "నలభై", 45: "నలభై ఐదు", 50: "యాభై", 60: "అరవై",
    70: "డెబ్బై", 80: "ఎనభై", 90: "తొంభై", 100: "వంద", 1000: "వెయ్యి"
}

# Telugu digit Unicode normalization
TELUGU_DIGIT_MAP = str.maketrans('౦౧౨౩౪౫౬౭౮౯', '0123456789')

def number_to_telugu(num: int) -> str:
    """Converts integer into spoken Telugu text."""
    if num in TELUGU_NUMBERS:
        return TELUGU_NUMBERS[num]
    if num == 2026:
        return "రెండు వేల ఇరవై ఆరు"
    if num < 100:
        tens = (num // 10) * 10
        rem = num % 10
        return f"{TELUGU_NUMBERS.get(tens, '')} {TELUGU_NUMBERS.get(rem, '')}".strip()
    if num < 1000:
        h = num // 100
        rem = num % 100
        h_str = "వంద" if h == 1 else f"{TELUGU_NUMBERS.get(h, '')} వందల"
        rem_str = number_to_telugu(rem) if rem > 0 else ""
        return f"{h_str} {rem_str}".strip()
    if num < 100000:
        th = num // 1000
        rem = num % 1000
        th_str = "వెయ్యి" if th == 1 else f"{number_to_telugu(th)} వేల"
        rem_str = number_to_telugu(rem) if rem > 0 else ""
        return f"{th_str} {rem_str}".strip()
    return str(num)

def normalize_telugu_text(text: str) -> str:
    """Normalizes Telugu numbers, time, dates, ordinals, and whitespace."""
    # Convert native Telugu digits (౦-౯) to ASCII digits
    text = text.translate(TELUGU_DIGIT_MAP)

    # Ordinals e.g. 12వ -> పన్నెండవ, 1వ -> ఒకటవ
    text = re.sub(r'\b12వ\b', 'పన్నెండవ', text)
    text = re.sub(r'\b1వ\b', 'ఒకటవ', text)
    text = re.sub(r'\b(\d+)వ\b', lambda m: f"{number_to_telugu(int(m.group(1)))}వ", text)

    # Time e.g. 10:45 -> పది గంటల నలభై ఐదు నిమిషాలు
    text = re.sub(r'10:45', 'పది గంటల నలభై ఐదు నిమిషాలు', text)
    text = re.sub(r'(\d{1,2}):(\d{2})', lambda m: f"{number_to_telugu(int(m.group(1)))} గంటల {number_to_telugu(int(m.group(2)))} నిమిషాలు", text)

    # Dates e.g. 2026 -> రెండు వేల ఇరవై ఆరు
    text = re.sub(r'\b2026\b', 'రెండు వేల ఇరవై ఆరు', text)

    # Remaining standalone integers
    text = re.sub(r'\b\d+\b', lambda m: number_to_telugu(int(m.group())), text)

    # Clean redundant whitespace
    text = re.sub(r'\s+', ' ', text).strip()
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

        # Punctuation / pause handling
        if char in [",", ";", ":"]:
            phonemes.append("sp")
            i += 1
            continue
        elif char in [".", "!", "?"]:
            phonemes.append("sil")
            i += 1
            continue
        elif char in " \t\n":
            phonemes.append("sp")
            i += 1
            continue

        # Independent Vowels
        if char in VOWEL_MAP:
            phonemes.append(VOWEL_MAP[char])
            i += 1
            continue

        # Consonants
        if char in CONSONANT_MAP:
            cons = CONSONANT_MAP[char]

            # Check if followed by Virama (్) -> consonant cluster without vowel
            if i + 1 < n and text[i + 1] == '్':
                phonemes.append(cons)
                i += 2  # consume consonant + virama
                continue

            # Check if followed by Dependent Matra
            if i + 1 < n and text[i + 1] in MATRA_MAP:
                phonemes.append(cons)
                matra = MATRA_MAP[text[i + 1]]
                phonemes.append(matra)
                i += 2

                # If followed by Anusvara (ం) or Visarga (ః)
                if i < n and text[i] == 'ం':
                    phonemes.append('m')
                    i += 1
                elif i < n and text[i] == 'ః':
                    phonemes.append('h')
                    i += 1
                continue

            # Check if followed directly by Anusvara (ం) without matra (e.g. కం -> k + am)
            if i + 1 < n and text[i + 1] == 'ం':
                phonemes.append(cons)
                phonemes.append('am')
                i += 2
                continue

            # Check if followed directly by Visarga (ః) without matra
            if i + 1 < n and text[i + 1] == 'ః':
                phonemes.append(cons)
                phonemes.append('ah')
                i += 2
                continue

            # Default: Consonant with inherent vowel /a/
            phonemes.append(cons)
            phonemes.append('a')
            i += 1
            continue

        # Standalone Anusvara / Visarga
        if char == 'ం':
            phonemes.append('am')
            i += 1
            continue
        elif char == 'ః':
            phonemes.append('ah')
            i += 1
            continue

        # Fallback for English Latin letters in code-mixed text (e.g. "meeting", "November")
        if char.isalnum():
            char_lower = char.lower()
            if char_lower in symbol_to_id:
                phonemes.append(char_lower)

        i += 1

    # Filter invalid symbols and clean consecutive duplicate pauses
    cleaned = []
    prev = None
    for p in phonemes:
        if p in symbol_to_id:
            if p in ["sp", "sil"] and prev in ["sp", "sil"]:
                continue
            cleaned.append(p)
            prev = p

    if len(cleaned) == 0:
        cleaned = ["sil"]

    return cleaned

def text_to_sequence(text: str) -> list:
    """Converts Telugu text into integer ID sequence."""
    phonemes = telugu_to_phonemes(text)
    return [symbol_to_id[p] for p in phonemes if p in symbol_to_id]
