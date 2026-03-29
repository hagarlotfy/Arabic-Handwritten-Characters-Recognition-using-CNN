import argparse
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

PRESETS = {
    # Preferred pipeline: 28-class model trained with 32x32x3 inputs.
    "final": {
        "model": ROOT / "arabic_cnn_final.keras",
        "labels": ROOT / "label_classes_final.npy",
        "yolo": ROOT / "my_yolo" / "char_detector" / "weights" / "best.pt",
    },
    # Alternate 28-class model trained with 64x64x1 inputs.
    "combined": {
        "model": ROOT / "arabic_cnn_combined.keras",
        "labels": ROOT / "label_classes_combined.npy",
        "yolo": ROOT / "my_yolo" / "char_detector" / "weights" / "best.pt",
    },
    # Legacy 18-class model (known to miss many letters).
    "legacy18": {
        "model": ROOT / "best_arabic_cnn.keras",
        "labels": ROOT / "label_classes.npy",
        "yolo": ROOT / "my_yolo" / "char_detector" / "weights" / "best.pt",
    },
}


def load_pipeline(preset_name: str):
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset '{preset_name}'. Choose from: {list(PRESETS)}")

    preset = PRESETS[preset_name]
    model_path = preset["model"]
    labels_path = preset["labels"]
    yolo_path = preset["yolo"]

    for key, path in [("model", model_path), ("labels", labels_path), ("yolo", yolo_path)]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {key} file: {path}")

    model = load_model(str(model_path))
    labels = np.load(str(labels_path), allow_pickle=True)
    detector = YOLO(str(yolo_path))

    # Fail-fast checks to prevent mixed incompatible artifacts.
    if len(model.input_shape) != 4:
        raise ValueError(f"Expected 4D model input shape, got {model.input_shape}")
    output_classes = model.output_shape[-1]
    if output_classes != len(labels):
        raise ValueError(
            "Model output classes do not match labels length: "
            f"model={output_classes}, labels={len(labels)}"
        )

    labels_str = [str(x) for x in labels]
    if len(set(labels_str)) != len(labels_str):
        raise ValueError("Labels file contains duplicate class names")

    input_h, input_w, input_c = model.input_shape[1], model.input_shape[2], model.input_shape[3]
    if input_h is None or input_w is None or input_c is None:
        raise ValueError(f"Model input shape must be fully defined, got {model.input_shape}")
    if input_c not in (1, 3):
        raise ValueError(f"Only 1-channel or 3-channel inputs are supported, got {input_c}")

    print("Loaded pipeline:")
    print("  preset:", preset_name)
    print("  model :", model_path.name, "input", model.input_shape, "output", model.output_shape)
    print("  labels:", labels_path.name, "count", len(labels))
    print("  yolo  :", yolo_path)

    return model, labels_str, detector, int(input_h), int(input_w), int(input_c)


def preprocess_crop(crop_bgr: np.ndarray, input_h: int, input_w: int, input_c: int):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(binary)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    # Keep grayscale intensity for classification to match training data distribution.
    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        return None

    side = max(4, int(max(h, w) * 1.35))
    # Training images are mostly dark ink on a light background.
    canvas = np.full((side, side), 255, dtype=np.uint8)
    ox, oy = (side - w) // 2, (side - h) // 2
    canvas[oy : oy + h, ox : ox + w] = roi
    resized = cv2.resize(canvas, (input_w, input_h), interpolation=cv2.INTER_AREA)

    if input_c == 1:
        x_img = resized.astype("float32") / 255.0
        x_img = np.expand_dims(x_img, axis=-1)
    else:
        bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        x_img = bgr.astype("float32") / 255.0

    return np.expand_dims(x_img, axis=0)


def recognize_word(
    image_path: Path,
    model,
    labels,
    detector,
    input_h: int,
    input_w: int,
    input_c: int,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.4,
    topk: int = 3,
):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    H, W = img.shape[:2]
    result = detector(img, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x2), min(H, y2)
        crop = img[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            continue

        x_in = preprocess_crop(crop, input_h=input_h, input_w=input_w, input_c=input_c)
        if x_in is None:
            continue

        pred = model.predict(x_in, verbose=0)[0]
        idx = int(np.argmax(pred))
        conf = float(pred[idx])
        top_idx = np.argsort(pred)[::-1][: max(1, topk)]
        top = [(labels[int(j)], float(pred[int(j)])) for j in top_idx]
        detections.append((x1, y1, x2, y2, labels[idx], conf, top))

    # Arabic reading order: right to left.
    detections.sort(key=lambda d: -d[0])
    return detections


def main():
    parser = argparse.ArgumentParser(description="Consistent Arabic character inference with sanity checks")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="final")
    parser.add_argument("--image", default=str(ROOT / "word_test.png"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.4)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    model, labels, detector, input_h, input_w, input_c = load_pipeline(args.preset)
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    detections = recognize_word(
        image_path=image_path,
        model=model,
        labels=labels,
        detector=detector,
        input_h=input_h,
        input_w=input_w,
        input_c=input_c,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        topk=args.topk,
    )

    print(f"Detected boxes: {len(detections)}")
    if not detections:
        return

    chars = [d[4] for d in detections]
    print("Detected (right->left):", chars)
    print("Word:", " ".join(chars))
    for i, d in enumerate(detections):
        top_str = ", ".join([f"{name}:{score:.3f}" for name, score in d[6]])
        print(f"char{i}: {d[4]} conf={d[5]:.3f} box=({d[0]},{d[1]},{d[2]},{d[3]}) top{args.topk}=[{top_str}]")


if __name__ == "__main__":
    main()
