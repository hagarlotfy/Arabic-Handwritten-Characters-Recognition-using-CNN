""""
Improved Arabic Handwritten Word Recognition
=============================================
Key fixes:
1. Simplify to ~28 BASE letter classes (no positional forms for CNN)
2. Use a proper custom CNN instead of MobileNetV2 at wrong resolution
3. Use YOLO only for DETECTION (not classification) — single-class detector
4. Let the CNN handle all classification
5. Provide guidance on expanding the dataset
"""

# ── 1. Configuration ─────────────────────────────────────────
import os, shutil, random, numpy as np, cv2
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT       = Path(r"F:\bach2\Arabic-Handwritten-Characters-Recognition-using-CNN")
AHAWP_DIR  = ROOT / "isolated_alphabets_per_alphabet"   # isolated chars for CNN
MY_DATASET = ROOT / "my_dataset"                         # Label Studio YOLO export
IMG_SIZE   = 64

# ══════════════════════════════════════════════════════════════
# STRATEGY CHANGE: 
#   - YOLO → single-class detector (just find "character" boxes)
#   - CNN  → classify each cropped character
# This way YOLO only needs to learn WHERE characters are,
# not WHAT they are (much easier with small data).
# ══════════════════════════════════════════════════════════════


# ── 2. Fix the YOLO dataset: convert to single-class ─────────
def convert_to_single_class(dataset_dir):
    """
    Rewrite all YOLO label files to use class_id=0 (single class: 'character').
    This makes YOLO's job much easier — it only needs to LOCATE characters.
    """
    for split in ['train', 'val']:
        labels_dir = dataset_dir / split / 'labels'
        if not labels_dir.exists():
            print(f"  Skipping {split} (not found)")
            continue
        count = 0
        for txt_file in labels_dir.glob('*.txt'):
            lines = txt_file.read_text().splitlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Replace class_id with 0, keep bbox coordinates
                    parts[0] = '0'
                    new_lines.append(' '.join(parts))
            txt_file.write_text('\n'.join(new_lines) + '\n')
            count += 1
        print(f"  Converted {count} label files in {split}/")

print("Converting YOLO labels to single-class...")
convert_to_single_class(MY_DATASET)


# ── 3. Write single-class data.yaml ──────────────────────────
yaml_path = MY_DATASET / 'data.yaml'
with open(yaml_path, 'w') as f:
    f.write(f"path: {MY_DATASET.as_posix()}\n")
    f.write("train: train/images\n")
    f.write("val: val/images\n")
    f.write("nc: 1\n")
    f.write("names:\n")
    f.write("  - character\n")
print(f"Saved single-class data.yaml: {yaml_path}")


# ── 4. Build a proper CNN for character classification ────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPool2D, Dense, Dropout,
    BatchNormalization, Flatten, Input
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

tf.random.set_seed(SEED)

# ── 4a. Load isolated character dataset ──────────────────────
# Use the ACTUAL folder names as labels (no remapping needed)
def load_isolated_dataset(alphabets_dir, img_size=IMG_SIZE):
    """Load all isolated character images. Use folder name as label directly."""
    images, labels = [], []
    for subfolder in sorted(alphabets_dir.iterdir()):
        if not subfolder.is_dir():
            continue
        label = subfolder.name  # e.g., "ain_begin", "beh_regular", etc.
        for img_path in subfolder.glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(label)
    
    X = np.array(images, dtype='float32') / 255.0
    X = X[..., np.newaxis]  # Add channel dim: (N, 64, 64, 1)
    y = np.array(labels)
    print(f"Loaded {len(X)} images across {len(set(labels))} classes")
    return X, y

X_all, y_all = load_isolated_dataset(AHAWP_DIR)

# Fit label encoder on ACTUAL data (not a superset)
le = LabelEncoder()
y_encoded = le.fit_transform(y_all)
NUM_CLASSES = len(le.classes_)
print(f"Classes found in data: {NUM_CLASSES}")
print(f"Classes: {list(le.classes_)}")

# Save for inference
np.save("label_classes.npy", le.classes_)

# Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_encoded, test_size=0.25, random_state=SEED, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)
print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")


# ── 4b. Build a purpose-built CNN (not MobileNet) ────────────
def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    """
    Custom CNN designed for 64x64 grayscale character images.
    Much better than MobileNetV2 at wrong resolution.
    """
    model = Sequential([
        Input(shape=input_shape),
        
        Conv2D(32, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(32, 3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPool2D(2),
        Dropout(0.25),
        
        Conv2D(64, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, 3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPool2D(2),
        Dropout(0.25),
        
        Conv2D(128, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, 3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPool2D(2),
        Dropout(0.25),
        
        Conv2D(256, 3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPool2D(2),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax'),
    ], name='ArabicCharCNN')
    return model

model = build_cnn()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.05,
    fill_mode='nearest'
)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint('best_arabic_cnn.keras', monitor='val_accuracy', save_best_only=True),
]

# Train
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)")
model.save('arabic_cnn_final.keras')


# ── 5. Train YOLO as single-class detector ────────────────────
from ultralytics import YOLO

yolo_model = YOLO('yolov8s.pt')

yolo_results = yolo_model.train(
    data=str(yaml_path),
    epochs=150,           # more epochs since single class is easier to learn
    imgsz=640,
    batch=4,
    patience=30,
    device='cpu',         # change to 'cuda' if you have GPU
    project=str(ROOT / 'my_yolo'),
    name='char_detector_v2',
    exist_ok=True,
    # Augmentation — important for small datasets
    degrees=10.0,
    translate=0.15,
    scale=0.4,
    shear=3.0,
    flipud=0.0,
    fliplr=0.0,           # don't flip Arabic text!
    mosaic=0.5,
    mixup=0.1,
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.4,
    erasing=0.3,
)

YOLO_WEIGHTS = ROOT / 'my_yolo' / 'char_detector_v2' / 'weights' / 'best.pt'
print(f"\nYOLO training complete! Weights: {YOLO_WEIGHTS}")


# ── 6. Improved Inference Pipeline ───────────────────────────
cnn_model     = tf.keras.models.load_model('best_arabic_cnn.keras')
yolo_detector = YOLO(str(YOLO_WEIGHTS))
label_classes = np.load('label_classes.npy', allow_pickle=True)

def preprocess_for_cnn(crop_bgr, img_size=IMG_SIZE):
    """Preprocess a YOLO crop for the CNN classifier."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # Otsu threshold to get clean binary
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find ink region and center it
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    roi = thresh[y:y+h, x:x+w]
    
    # Place on square canvas with padding
    side = int(max(h, w) * 1.3)
    canvas = np.zeros((side, side), dtype='uint8')
    ox, oy = (side - w) // 2, (side - h) // 2
    canvas[oy:oy+h, ox:ox+w] = roi
    
    resized = cv2.resize(canvas, (img_size, img_size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype('float32') / 255.0
    return normalized[..., np.newaxis]  # (64, 64, 1)


def recognize_word(image_path, conf_threshold=0.25, iou_threshold=0.4):
    """
    1. YOLO finds character bounding boxes (single class)
    2. CNN classifies each crop
    3. Sort right-to-left for Arabic reading order
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    
    H, W = img.shape[:2]
    annotated = img.copy()
    
    # YOLO detection — single class, just finding boxes
    results = yolo_detector(img, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
    boxes = results.boxes
    
    if len(boxes) == 0:
        print("No characters detected.")
        return [], annotated
    
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        det_conf = float(box.conf[0])
        
        # Crop and classify with CNN
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x2), min(H, y2)
        crop = img[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            continue
        
        proc = preprocess_for_cnn(crop)
        if proc is None:
            continue
        
        inp = np.expand_dims(proc, axis=0)
        preds = cnn_model.predict(inp, verbose=0)[0]
        cnn_idx = np.argmax(preds)
        cnn_conf = float(preds[cnn_idx])
        label = str(label_classes[cnn_idx])
        
        # Extract base letter name (remove _begin/_middle/_end/_regular)
        base_letter = label.rsplit('_', 1)[0] if '_' in label else label
        
        detections.append((x1, y1, x2, y2, label, base_letter, cnn_conf, det_conf))
    
    # Sort RIGHT-TO-LEFT for Arabic
    detections.sort(key=lambda d: -d[0])
    
    recognized_full = []
    recognized_base = []
    for (x1, y1, x2, y2, label, base, cnn_conf, det_conf) in detections:
        recognized_full.append(label)
        recognized_base.append(base)
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        display = f"{base} ({cnn_conf:.0%})"
        cv2.putText(annotated, display,
                    (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    
    return recognized_base, annotated


# ── 7. Test on your images ────────────────────────────────────
import matplotlib.pyplot as plt

test_images = list((MY_DATASET / 'val' / 'images').glob('*.*'))
for img_path in test_images[:5]:
    chars, annotated = recognize_word(img_path)
    print(f"\n{img_path.name}: {' '.join(chars)}")
    
    plt.figure(figsize=(10, 4))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Detected: {' '.join(chars)}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# TIPS TO IMPROVE FURTHER:
# ══════════════════════════════════════════════════════════════
#
# 1. ADD MORE YOLO TRAINING DATA:
#    - Label at least 200-300 word images in Label Studio
#    - Each word gives you ~3-6 character boxes, so 200 words ≈ 800+ boxes
#    - Since it's now single-class, you only need to draw boxes (fast!)
#
# 2. GENERATE SYNTHETIC TRAINING DATA FOR YOLO:
#    - Take isolated characters, compose them into fake "words"
#    - Randomly position characters on a blank canvas
#    - Auto-generate YOLO labels from known positions
#    - This can give you thousands of training images for free!
#
# 3. IMPROVE CNN ACCURACY:
#    - If AHAWP only has isolated forms, consider:
#      a) Reducing to 28 base classes (just the letter, ignore position)
#      b) Or collecting positional form data from other datasets
#    - Try larger IMG_SIZE (96 or 128) for more detail
#    - Increase augmentation (elastic distortion is great for handwriting)
#
# 4. POST-PROCESSING:
#    - Use Arabic language model to correct unlikely character sequences
#    - Merge overlapping detections more aggressively
#    - Filter by aspect ratio (Arabic chars have typical width/height ratios)
