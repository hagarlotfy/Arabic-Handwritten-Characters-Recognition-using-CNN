import os, random
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization, Flatten, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path('F:/newbach/Arabic-Handwritten-Characters-Recognition-using-CNN')
AHAWP_DIR = ROOT / 'isolated_alphabets_per_alphabet'
IMG_SIZE = 64

FORM_MAP = {
    'begin': 'beginning',
    'middle': 'middle',
    'end': 'end',
    'regular': 'isolated',
    'hamza': 'hamza',
    'alif': 'alif',
}

def load_isolated_dataset(alphabets_dir, img_size=IMG_SIZE):
    images, labels = [], []
    for subfolder in sorted(alphabets_dir.iterdir()):
        if not subfolder.is_dir():
            continue
        parts = subfolder.name.rsplit('_', 1)
        if len(parts) != 2:
            continue
        letter, form_raw = parts
        form = FORM_MAP.get(form_raw)
        if form is None:
            continue
        label = letter
        for img_path in sorted(subfolder.glob('*.png')):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(label)
    X = np.array(images, dtype='float32') / 255.0
    X = X[..., np.newaxis]
    y = np.array(labels)
    print(f'Loaded {len(X)} images across {len(set(labels))} classes')
    return X, y


if __name__ == '__main__':
    X_all, y_all = load_isolated_dataset(AHAWP_DIR)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)
    NUM_CLASSES = len(le.classes_)
    ARABIC_LABELS = list(le.classes_)

    counts = Counter(y_encoded)
    keep_classes = {cls for cls, n in counts.items() if n >= 2}
    mask = np.array([y in keep_classes for y in y_encoded])
    X_all = X_all[mask]
    y_encoded = y_encoded[mask]

    stratify_val = y_encoded if len(np.unique(y_encoded)) > 1 else None
    X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_encoded, test_size=0.25, random_state=SEED, stratify=stratify_val)
    strat_temp = y_temp if (min(Counter(y_temp).values()) >= 2 and len(np.unique(y_temp)) > 1) else None
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=strat_temp)
    print('split', X_train.shape, X_val.shape, X_test.shape)

    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        Conv2D(32, 3, padding='same', activation='relu'), BatchNormalization(), MaxPool2D(2), Dropout(0.2),
        Conv2D(64, 3, padding='same', activation='relu'), BatchNormalization(), MaxPool2D(2), Dropout(0.2),
        Conv2D(128, 3, padding='same', activation='relu'), BatchNormalization(), MaxPool2D(2), Dropout(0.2),
        Flatten(), Dense(256, activation='relu'), Dropout(0.3), Dense(NUM_CLASSES, activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, shear_range=0.1, brightness_range=[0.9, 1.1], fill_mode='nearest')
    class_weights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))
    print('class_weights', class_weights)

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint('best_arabic_cnn.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        epochs=2,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    print('training done', history.history['val_accuracy'][-1])

    best = load_model('best_arabic_cnn.keras')
    print('best model loaded')

    loss, acc = best.evaluate(X_test, y_test, verbose=0)
    print(f'test acc={acc:.4f}, loss={loss:.4f}')

    preds = np.argmax(best.predict(X_test[:10], verbose=0), axis=1)
    print('pred', [ARABIC_LABELS[i] for i in preds], 'true', [ARABIC_LABELS[i] for i in y_test[:10]])

    np.save('label_classes.npy', le.classes_)
    print('saved label_classes.npy')