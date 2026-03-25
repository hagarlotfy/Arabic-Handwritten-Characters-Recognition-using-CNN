import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from ultralytics import YOLO

ROOT = Path('F:/newbach/Arabic-Handwritten-Characters-Recognition-using-CNN')
YOLO_WEIGHTS = ROOT / 'my_yolo' / 'char_detector' / 'weights' / 'best.pt'
MODEL_PATH = ROOT / 'best_arabic_cnn.keras'
LABEL_PATH = ROOT / 'label_classes.npy'
TEST_IMAGE = ROOT / 'word_test.png'

print('MODEL exists', MODEL_PATH.exists())
print('LABEL exists', LABEL_PATH.exists())
print('YOLO exists', YOLO_WEIGHTS.exists())
print('TEST image exists', TEST_IMAGE.exists())

cnn_model = load_model(str(MODEL_PATH))
labels = np.load(str(LABEL_PATH), allow_pickle=True)

print('Loaded CNN and labels', len(labels))

yolo = YOLO(str(YOLO_WEIGHTS))
print('YOLO loaded')

if TEST_IMAGE.exists():
    img = cv2.imread(str(TEST_IMAGE))
    results = yolo(img, conf=0.25, iou=0.4)[0]
    print('boxes', len(results.boxes))
    for i,box in enumerate(results.boxes):
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        crop = img[y1:y2, x1:x2]
        if crop.size==0:
            continue
        gray=cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _,th=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        coords=cv2.findNonZero(th)
        if coords is None:
            continue
        x,y,w,h=cv2.boundingRect(coords)
        roi=th[y:y+h,x:x+w]
        side=int(max(h,w)*1.3)
        canvas=np.zeros((side,side),dtype='uint8')
        ox,oy=(side-w)//2,(side-h)//2
        canvas[oy:oy+h,ox:ox+w]=roi
        resized=cv2.resize(canvas,(64,64),interpolation=cv2.INTER_AREA)
        proc=resized.astype('float32')/255.0
        pred=cnn_model.predict(proc[np.newaxis,...],verbose=0)[0]
        idx=np.argmax(pred)
        print(f'char{i}:', labels[idx], 'conf', pred[idx])
else:
    print('No test image to infer')
