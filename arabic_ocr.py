import os
import cv2
import numpy as np
import tensorflow as tf
import arabic_reshaper
from bidi.algorithm import get_display
from thefuzz import process
from tensorflow.keras.models import load_model


# --------------------------------------------------
# 1. LOAD MODEL
# --------------------------------------------------

def ctc_loss(y_true, y_pred):
    return y_pred

MODEL_PATH = "arabic_crnn_model.keras"

print("Loading CRNN model...")

model = load_model(
    MODEL_PATH,
    custom_objects={"ctc_loss": ctc_loss}
)

print("Model loaded successfully!")


# --------------------------------------------------
# 2. CHARACTER SET (AHAWP)
# --------------------------------------------------

ARABIC_CHARS = [
" ", "ا","ب","ت","ث","ج","ح","خ","د","ذ","ر","ز",
"س","ش","ص","ض","ط","ظ","ع","غ","ف","ق","ك","ل",
"م","ن","ه","و","ي"
]

char_to_num = {c:i for i,c in enumerate(ARABIC_CHARS)}
num_to_char = {i:c for c,i in char_to_num.items()}


# --------------------------------------------------
# 3. DICTIONARY FOR AUTOCORRECT
# --------------------------------------------------

arabic_dictionary = [
"مرحبا",
"كتاب",
"قلم",
"جامعة",
"طالب",
"مدرسة",
"الذكاء",
"الاصطناعي",
"البرمجيات"
]


# --------------------------------------------------
# 4. IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(path, max_width=512):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise Exception("Image not found")

    # threshold
    _, img = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    h, w = img.shape

    new_w = int(32 * w / h)

    img = cv2.resize(img, (new_w, 32))

    img = img.astype("float32") / 255.0

    # pad image to fixed width
    canvas = np.zeros((32, max_width))

    if new_w > max_width:
        img = cv2.resize(img, (max_width, 32))
        canvas = img
    else:
        canvas[:, :new_w] = img

    canvas = np.expand_dims(canvas, -1)

    return canvas


# --------------------------------------------------
# 5. CTC DECODING
# --------------------------------------------------

def decode_prediction(pred):

    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    results = tf.keras.backend.ctc_decode(
        pred,
        input_length=input_len,
        greedy=True
    )[0][0]

    return results


# --------------------------------------------------
# 6. CONVERT SEQUENCE → TEXT
# --------------------------------------------------

def sequence_to_text(seq):

    text = ""

    for i in seq:

        if i != -1 and i in num_to_char:
            text += num_to_char[i]

    return text


# --------------------------------------------------
# 7. AUTOCORRECT FUNCTION
# --------------------------------------------------

def autocorrect(word, dictionary, threshold=70):

    if len(word) < 2:
        return word

    best_match, score = process.extractOne(word, dictionary)

    if score >= threshold:
        return best_match

    return word


# --------------------------------------------------
# 8. OCR SYSTEM
# --------------------------------------------------

class ArabicOCR:

    def __init__(self, model):

        self.model = model


    def predict(self, image_path):

        img = preprocess_image(image_path)

        img = np.expand_dims(img, 0)

        pred = self.model.predict(img)

        decoded = decode_prediction(pred)

        text = sequence_to_text(decoded[0].numpy())

        corrected = autocorrect(text, arabic_dictionary)

        return corrected


    def display(self, text):

        reshaped = arabic_reshaper.reshape(text)

        bidi_text = get_display(reshaped)

        print("\nPredicted Arabic Text:\n")
        print(bidi_text)


# --------------------------------------------------
# 9. RUN OCR
# --------------------------------------------------

ocr = ArabicOCR(model)


# change this path to your test image
TEST_IMAGE = "word_test.png"


try:

    result = ocr.predict(TEST_IMAGE)

    print("\nRaw Prediction:", result)

    ocr.display(result)

except Exception as e:

    print("Error:", e)

