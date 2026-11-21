"""Small, conservative English/Azerbaijani search-query interpreter."""
import math
import re

# Longest phrases are replaced first; unknown words remain available to E5.
AZ_TERMS = {
    'münasib qiymətə': 'affordable', 'sərfəli qiymətə': 'affordable',
    'qulaqcıqlar': 'earbuds', 'qulaqcıq': 'earbuds', 'qulaqlıqlar': 'headphones',
    'qulaqlıq': 'headphones', 'siçanlar': 'mouse', 'siçanı': 'mouse', 'siçan': 'mouse',
    'klaviaturalar': 'keyboard', 'klaviatura': 'keyboard', 'simsiz': 'wireless',
    'naqilsiz': 'wireless', 'naqilli': 'wired', 'veb kamera': 'webcam', 'vebkamera': 'webcam',
    'kameralar': 'camera', 'kamera': 'camera', 'monitorlar': 'monitor', 'monitor': 'monitor',
    'noutbuklar': 'laptop', 'noutbuk': 'laptop', 'kompüter': 'computer',
    'mikrofon': 'microphone', 'dinamik': 'speaker', 'oyun': 'gaming',
    'yüksək reytinqli': 'highly rated', 'yaxşı reytinqli': 'well rated',
    'dərs': 'study', 'üçün': 'for', 'mənə': '', 'tap': '',
}

PRODUCT_TYPES = {
    'headphones': 'Headphones & Earbuds', 'earbuds': 'Headphones & Earbuds',
    'mouse': 'Computers & Components', 'keyboard': 'Computers & Components',
    'webcam': 'Computers & Components', 'monitor': 'Computers & Components',
    'laptop': 'Computers & Components', 'computer': 'Computers & Components',
    'camera': 'Cameras & Photography', 'microphone': None, 'speaker': 'Speakers & Home Audio',
}
NUMBER = r'\d+(?:[.,]\d+)?'
AMOUNT = rf'\$?\s*({NUMBER})(?:\s*(?:usd|dollars?|dollar(?:dan|a|ı)?))?'


def understand_query(query):
    """Return a search passage and only constraints stated numerically by the user."""
    text = query.casefold()
    min_price = max_price = min_rating = None
    # Source prices are USD; do not interpret manat amounts as dollars.
    if not re.search(r'\b(?:manat\w*|azn)\b|₼', text):
        between = re.search(rf'\bbetween\s+{AMOUNT}\s+and\s+{AMOUNT}(?![\d.,])', text)
        if between:
            lower, upper = (float(value.replace(',', '.')) for value in between.groups())
            if math.isfinite(lower) and math.isfinite(upper):
                min_price, max_price = lower, upper
                text = text[:between.start()] + ' ' + text[between.end():]
        patterns = (
            (rf'\b(?:under|below|less than|cheaper than)\s+{AMOUNT}(?![\d.,])', 'max', True),
            (rf'\b(?:at most|up to)\s+{AMOUNT}(?![\d.,])', 'max', False),
            (rf'\b(?:over|above|more than)\s+{AMOUNT}(?![\d.,])', 'min', True),
            (rf'\b(?:at least|from)\s+{AMOUNT}(?![\d.,])(?!\s*(?:stars?|ulduz|rating))', 'min', False),
            (rf'(?<![\w.,]){AMOUNT}\s*(?:dollardan\s*)?(?:ucuz|aşağı)(?!\w)', 'max', True),
            (rf'(?<![\w.,]){AMOUNT}\s*(?:dollardan\s*)?(?:bahalı|yuxarı)(?!\w)', 'min', True),
        )
        for pattern, kind, strict in patterns:
            match = re.search(pattern, text)
            if match:
                if re.match(r'\s*(?:/\s*5\b|out of\s+5\b|stars?\b|ulduz\w*|rating\b|gb\b|tb\b|inch(?:es)?\b|hz\b|mah\b|hours?\b)', text[match.end():]):
                    continue
                if (kind == 'min' and re.match(r'(?:at least|from)\b', match.group())
                        and not re.search(r'\$|usd|dollar', match.group())):
                    continue
                value = float(match.group(1).replace(',', '.'))
                if not math.isfinite(value):
                    continue
                if strict:
                    value = math.nextafter(value, math.inf if kind == 'min' else -math.inf)
                if not math.isfinite(value):
                    continue
                if kind == 'min':
                    min_price = max(min_price, value) if min_price is not None else value
                else:
                    max_price = min(max_price, value) if max_price is not None else value
                text = text[:match.start()] + ' ' + text[match.end():]

    # A numeric star request is a minimum, not a sentiment-derived guess.
    rating = re.search(r'(?<![\d./])([1-5](?:[.,]\d+)?)\s*(?:/|out of)\s*5\s*(?:stars?)?\b', text)
    rating = rating or re.search(r'(?<![\d./])([1-5](?:[.,]\d+)?)\s*(?:\+\s*)?(?:stars?|ulduz(?:lu|dan)?)(?!\w)', text)
    rating = rating or re.search(r'\b(?:minimum|min|at least)\s+rating\s*(?:of\s*)?([1-5](?:[.,]\d+)?)(?![\d.,])', text)
    if rating and not re.search(r'\b(?:under|below|less than)\s*$', text[:rating.start()]):
        value = float(rating.group(1).replace(',', '.'))
        if 1 <= value <= 5:
            min_rating = value
            text = text[:rating.start()] + ' ' + text[rating.end():]

    translated = False
    for source, target in AZ_TERMS.items():
        pattern = rf'(?<!\w){re.escape(source)}(?!\w)'
        text, count = re.subn(pattern, target, text)
        translated |= bool(count)
    interpreted = ' '.join(text.split())
    # Preserve ordinary English searches byte-for-byte when no parsing occurred.
    changed = translated or min_price is not None or max_price is not None or min_rating is not None
    retrieval = interpreted if changed and interpreted else query
    product_type = next((name for name in PRODUCT_TYPES
                         if re.search(rf'\b{re.escape(name)}(?:s|es)?\b', retrieval, re.I)), None)
    attributes = [word for word in ('wireless', 'wired', 'gaming', 'bluetooth')
                  if re.search(rf'\b{word}\b', retrieval, re.I)]
    return {'text': retrieval, 'product_type': product_type,
            'category': PRODUCT_TYPES.get(product_type), 'attributes': attributes,
            'intent': 'study' if re.search(r'\bstudy\b', retrieval, re.I) else None,
            'min_price': min_price, 'max_price': max_price, 'min_rating': min_rating}
