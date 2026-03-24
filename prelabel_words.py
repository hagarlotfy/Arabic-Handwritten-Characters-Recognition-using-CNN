"""
Pre-labeling script for AHAWP isolated_words_per_user
------------------------------------------------------
- Reads all word images from isolated_words_per_user/
- Identifies which word each image is (from filename)
- Outputs a Label Studio pre-annotation JSON
- You only need to DRAW the boxes in Label Studio,
  the labels are already filled in for you
"""

import os
import json
from pathlib import Path

# ── Adjust this to your actual path ──────────────────────────
WORDS_DIR   = Path(r"F:\bach\Arabic-Handwritten-Characters-Recognition-using-CNN\isolated_words_per_user")
OUTPUT_JSON = Path(r"F:\bach\Arabic-Handwritten-Characters-Recognition-using-CNN\prelabels.json")
# ─────────────────────────────────────────────────────────────

# Each word's character sequence in RIGHT-TO-LEFT order
# (first item = rightmost character in the image)
WORD_CHARS = {
    'azan':            ['alif_begin', 'thal_end', 'alif_regular', 'noon_end'],
    'sakhar':          ['seen_begin', 'khah_middle', 'raa_end'],
    'mustadhafeen':    ['meem_begin', 'seen_middle', 'tah_middle', 'dal_middle',
                        'ain_middle', 'feh_middle', 'yaa_end', 'noon_end'],
    'abjadiyah':       ['alif_begin', 'beh_middle', 'jeem_middle', 'dal_middle',
                        'yaa_middle', 'heh_end'],
    'fasayakfeekahum': ['feh_begin', 'seen_middle', 'yaa_middle', 'kaf_middle',
                        'feh_middle', 'yaa_middle', 'kaf_middle', 'heh_middle',
                        'meem_end'],
    'ghazaal':         ['ghain_begin', 'zain_middle', 'alif_middle', 'lam_end'],
    'ghaleez':         ['ghain_begin', 'lam_middle', 'yaa_middle', 'zain_end'],
    'qashtah':         ['qaf_begin', 'sheen_middle', 'tah_middle', 'heh_end'],
    'shateerah':       ['sheen_begin', 'alif_middle', 'tah_middle', 'raa_middle',
                        'heh_end'],
    'mehras':          ['meem_begin', 'heh_middle', 'raa_middle', 'alif_middle',
                        'seen_end'],
}

def get_word_name(filename):
    """Extract word name from filename e.g. user001_azan_001.png → azan"""
    stem = Path(filename).stem          # user001_azan_001
    parts = stem.split("_")            # ['user001', 'azan', '001']
    # Word name is everything between the user prefix and the trailing number
    # Handle multi-part word names like 'fasayakfeekahum' or 'mustadhafeen'
    if len(parts) < 3:
        return None
    # Drop first (username) and last (index) parts
    word = "_".join(parts[1:-1])
    return word if word in WORD_CHARS else None


def build_prelabel_json(words_dir: Path):
    tasks = []
    task_id = 1

    for user_folder in sorted(words_dir.iterdir()):
        if not user_folder.is_dir():
            continue

        for img_path in sorted(user_folder.glob("*.png")):
            word_name = get_word_name(img_path.name)
            if word_name is None:
                print(f"  [SKIP] unrecognised word in: {img_path.name}")
                continue

            chars = WORD_CHARS[word_name]

            # Build one prediction result per character
            # Boxes are evenly spaced as placeholders — you drag them to the
            # correct position in Label Studio, labels are already set
            n = len(chars)
            results = []
            for i, label in enumerate(chars):
                # Divide image width into n equal slots (RTL: rightmost = index 0)
                # x is expressed as percentage from LEFT edge
                slot_width = 100.0 / n
                # RTL: char 0 is rightmost → highest x percentage
                x_pct = 100.0 - (i + 1) * slot_width

                results.append({
                    "from_name": "label",
                    "to_name":   "image",
                    "type":      "rectanglelabels",
                    "value": {
                        "x":      round(x_pct, 2),
                        "y":      5.0,
                        "width":  round(slot_width, 2),
                        "height": 90.0,
                        "rectanglelabels": [label]
                    }
                })

            task = {
                "id": task_id,
                "data": {
                    "image": "/data/local-files/?d=" + str(img_path.relative_to(WORDS_DIR.parent)).replace("\\", "/")
                },
                "predictions": [
                    {
                        "model_version": "prelabel_v1",
                        "score": 1.0,
                        "result": results
                    }
                ]
            }
            tasks.append(task)
            task_id += 1

    return tasks


print("Scanning word images...")
tasks = build_prelabel_json(WORDS_DIR)
print(f"Built {len(tasks)} pre-labeled tasks")

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(tasks, f, indent=2, ensure_ascii=False)

print(f"Saved to: {OUTPUT_JSON}")
print("\nNext steps:")
print("  1. In Label Studio → Import → upload prelabels.json")
print("  2. Each image opens with boxes already labeled")
print("  3. Just drag the boxes to fit the actual characters")
print("  4. Submit and export as YOLO format when done")
