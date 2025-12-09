#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
from lingua import Language, LanguageDetectorBuilder
from dynaconf import Dynaconf
from .data_utils import write_dataset, load_latest_dataset

# Map ISO codes to lingua Language enum
LANG_MAP = {
    'en': Language.ENGLISH,
    'de': Language.GERMAN,
    'ja': Language.JAPANESE,
    'fr': Language.FRENCH,
    'es': Language.SPANISH,
    'it': Language.ITALIAN,
    'pt': Language.PORTUGUESE,
    'zh': Language.CHINESE,
    'ko': Language.KOREAN,
    'nl': Language.DUTCH,
}


def detect_language(settings: Dynaconf) -> None:
    """Detect language confidence for outputs and add to dataset."""
    
    lang_code = settings.expected_language.upper()
    target_lang = LANG_MAP.get(settings.expected_language.lower())
    
    if not target_lang:
        print(f"Unsupported language code: {settings.expected_language}")
        print(f"Supported codes: {', '.join(LANG_MAP.keys())}")
        return
    
    # Build detector for all languages
    print("Building language detector...")
    detector = LanguageDetectorBuilder.from_all_languages().build()
    
    try:
        df = load_latest_dataset(settings.data_dir)
        print(f"Loaded dataset with {len(df)} rows")
    except FileNotFoundError:
        print("No dataset found. Please generate data first.")
        return
    
    # Check if rewrite column exists
    if 'rewrite' not in df.columns:
        print("No 'rewrite' column found. Run inference first.")
        return
    
    # Batch: get confidence and binary detection for all outputs
    outputs = df['rewrite'].fillna('').tolist()
    
    print(f"Detecting {lang_code} for {len(outputs)} outputs...")
    confidences = []
    is_target_lang = []
    
    for text in outputs:
        # Confidence score (0-1)
        conf = detector.compute_language_confidence(text, target_lang)
        confidences.append(conf)
        
        # Binary detection
        detected = detector.detect_language_of(text)
        is_target_lang.append(1 if detected == target_lang else 0)
    
    # Add columns: EN for confidence, is_EN for binary
    df[lang_code] = confidences
    df[f"is_{lang_code}"] = is_target_lang
    
    # Stats
    avg_conf = sum(confidences) / len(confidences)
    match_rate = sum(is_target_lang) / len(is_target_lang)
    print(f"Average {lang_code} confidence: {avg_conf:.2%}")
    print(f"Detected as {lang_code}: {match_rate:.2%} ({sum(is_target_lang)}/{len(is_target_lang)})")
    
    write_dataset(df, settings.data_dir, "all", "csv")
    print("Dataset updated with language confidence column.")
