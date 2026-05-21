import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import log2

from config import ROCKYOU_PATH, SAMPLE_SIZE, OUTPUT_DIR
from utils import step_header


def load_passwords():
    step_header(1, "Load passwords")
    t0 = time.time()

    with open(ROCKYOU_PATH, "r", encoding="latin-1") as f:
        raw = f.readlines()

    all_passwords = [line.strip() for line in raw if line.strip()]

    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(all_passwords):
        random.seed(42)
        passwords = random.sample(all_passwords, SAMPLE_SIZE)
        print(f"  Using random sample of {SAMPLE_SIZE:,} / {len(all_passwords):,} passwords")
    else:
        passwords = all_passwords
        print(f"  Using ALL {len(passwords):,} passwords")

    print(f"  Loaded in {time.time()-t0:.1f}s")
    return passwords


def run_eda(passwords):
    step_header(2, "Exploratory Data Analysis (EDA)")

    lengths     = [len(p) for p in passwords]
    only_digits = [p for p in passwords if p.isdigit()]
    only_alpha  = [p for p in passwords if p.isalpha()]
    has_special = [p for p in passwords if any(not c.isalnum() for c in p)]
    has_upper   = [p for p in passwords if any(c.isupper() for c in p)]

    print(f"  Total passwords        : {len(passwords):,}")
    print(f"  Unique passwords       : {len(set(passwords)):,}  ({len(set(passwords))/len(passwords)*100:.1f}%)")
    print(f"  Avg length             : {np.mean(lengths):.1f} chars")
    print(f"  Most common length     : {max(set(lengths), key=lengths.count)}")
    print(f"  Min / Max length       : {min(lengths)} / {max(lengths)}")
    print(f"  Only digits            : {len(only_digits)/len(passwords)*100:.1f}%")
    print(f"  Only letters           : {len(only_alpha)/len(passwords)*100:.1f}%")
    print(f"  Contains special chars : {len(has_special)/len(passwords)*100:.1f}%")
    print(f"  Contains uppercase     : {len(has_upper)/len(passwords)*100:.1f}%")

    # EDA plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"EDA — RockYou password dataset (n={len(passwords):,})", fontsize=14)

    axes[0, 0].hist([l for l in lengths if l <= 25], bins=25, color="steelblue", edgecolor="black")
    axes[0, 0].axvline(x=8, color="red", linestyle="--", label="Min recommended (8)")
    axes[0, 0].set_title("Password length distribution")
    axes[0, 0].set_xlabel("Length")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend()

    char_labels = ["Only digits", "Only letters", "Has uppercase", "Has special"]
    char_vals = [
        len(only_digits) / len(passwords) * 100,
        len(only_alpha)  / len(passwords) * 100,
        len(has_upper)   / len(passwords) * 100,
        len(has_special) / len(passwords) * 100,
    ]
    axes[0, 1].barh(char_labels, char_vals, color="coral")
    axes[0, 1].set_title("Character type breakdown (%)")
    axes[0, 1].set_xlabel("Percentage of passwords")

    entropies = []
    for p in passwords[:50_000]:
        if len(p) > 0:
            e = sum(-(p.count(c) / len(p)) * log2(p.count(c) / len(p)) for c in set(p))
            entropies.append(e)
    axes[1, 0].hist(entropies, bins=40, color="green", edgecolor="black")
    axes[1, 0].set_title("Password entropy distribution")
    axes[1, 0].set_xlabel("Entropy (bits)")
    axes[1, 0].set_ylabel("Count")

    sample_idx = np.random.choice(len(passwords), min(5000, len(passwords)), replace=False)
    s_lengths = [len(passwords[i]) for i in sample_idx]
    s_unique  = [len(set(passwords[i])) for i in sample_idx]
    axes[1, 1].scatter(s_lengths, s_unique, alpha=0.2, s=5, color="purple")
    axes[1, 1].set_title("Length vs unique characters")
    axes[1, 1].set_xlabel("Password length")
    axes[1, 1].set_ylabel("Unique characters")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_plots.png", dpi=150)
    plt.show()
    print("  Saved output/eda_plots.png")

    # Correlation matrix
    eda_df = pd.DataFrame({
        "length":       [len(p) for p in passwords],
        "num_digits":   [sum(c.isdigit() for c in p) for p in passwords],
        "num_upper":    [sum(c.isupper() for c in p) for p in passwords],
        "num_special":  [sum(not c.isalnum() for c in p) for p in passwords],
        "unique_chars": [len(set(p)) for p in passwords],
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(eda_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Correlation matrix — password features")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_correlation.png", dpi=150)
    plt.show()
    print("  Saved output/eda_correlation.png")