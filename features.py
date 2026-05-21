import re
import time
import pandas as pd
from math import log2

from utils import step_header, progress_bar


def extract_features(pwd):
    if not pwd:
        return None
    return {
        "length":              len(pwd),
        "num_digits":          sum(c.isdigit() for c in pwd),
        "num_upper":           sum(c.isupper() for c in pwd),
        "num_lower":           sum(c.islower() for c in pwd),
        "num_special":         sum(not c.isalnum() for c in pwd),
        "digit_ratio":         sum(c.isdigit() for c in pwd) / len(pwd),
        "upper_ratio":         sum(c.isupper() for c in pwd) / len(pwd),
        "special_ratio":       sum(not c.isalnum() for c in pwd) / len(pwd),
        "entropy":             sum(
                                   -(pwd.count(c) / len(pwd)) * log2(pwd.count(c) / len(pwd))
                                   for c in set(pwd)
                               ),
        "unique_chars":        len(set(pwd)),
        "starts_with_upper":   int(pwd[0].isupper()),
        "ends_with_digit":     int(pwd[-1].isdigit()),
        "ends_with_special":   int(not pwd[-1].isalnum()),
        "only_digits":         int(pwd.isdigit()),
        "only_lower":          int(pwd.islower()),
        "only_upper":          int(pwd.isupper()),
        "only_alpha":          int(pwd.isalpha()),
        "has_year":            int(bool(re.search(r"(19|20)\d{2}", pwd))),
        "has_keyboard_walk":   int(bool(re.search(r"qwer|asdf|zxcv|1234|2345|3456|4567", pwd.lower()))),
        "has_repeated_chars":  int(bool(re.search(r"(.)\1{2,}", pwd))),
        "is_leet":             int(bool(re.search(r"[4310!@$]", pwd))),
        "capital_lower_digit": int(bool(re.match(r"^[A-Z][a-z]+\d+$", pwd))),
    }


def classify_pattern(pwd):
    if not pwd:                                        return "empty"
    if re.match(r"^\d+$", pwd):                        return "only_numbers"
    if re.match(r"^[a-zA-Z]+$", pwd):                  return "only_letters"
    if re.match(r"^[A-Z][a-z]+\d+$", pwd):             return "name_number"
    if re.match(r"^[a-z]+\d+$", pwd):                  return "word_number"
    if re.match(r"^[a-z]+[!@#$%^&*]+$", pwd):         return "word_special"
    if re.match(r"^[a-z]+\d+[!@#$%^&*]+$", pwd):      return "word_number_special"
    if re.search(r"(19|20)\d{2}", pwd):                return "contains_year"
    if re.search(r"qwer|asdf|1234|abcd", pwd.lower()): return "keyboard_walk"
    if re.search(r"(.)\1{2,}", pwd):                   return "repeated_chars"
    return "other"


def build_dataframe(passwords):
    # Step 3: extract features
    step_header(3, "Extract features")
    t0 = time.time()
    feature_rows = []
    skipped = 0

    for i, pwd in enumerate(passwords):
        feats = extract_features(pwd)
        if feats is None:
            skipped += 1
        else:
            feature_rows.append(feats)
        if (i + 1) % max(1, len(passwords) // 100) == 0 or i + 1 == len(passwords):
            progress_bar(i + 1, len(passwords), task="extracting  ", start_time=t0)

    print(f"  {len(feature_rows):,} processed, {skipped:,} skipped ({time.time()-t0:.1f}s)")

    # Step 4: classify patterns
    step_header(4, "Classify patterns")
    t0 = time.time()
    labels = []
    valid_passwords = [p for p in passwords if extract_features(p) is not None]

    for i, pwd in enumerate(valid_passwords):
        labels.append(classify_pattern(pwd))
        if (i + 1) % max(1, len(valid_passwords) // 100) == 0 or i + 1 == len(valid_passwords):
            progress_bar(i + 1, len(valid_passwords), task="classifying ", start_time=t0)

    print(f"  Labelled {len(labels):,} passwords ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(feature_rows)
    df["pattern"] = labels

    pattern_counts = df["pattern"].value_counts()
    total = len(df)
    print()
    for pat, cnt in pattern_counts.items():
        bar = "█" * int(cnt / total * 40)
        print(f"  {pat:<22} {bar:<42} {cnt/total*100:5.1f}%  ({cnt:,})")

    return df, pattern_counts