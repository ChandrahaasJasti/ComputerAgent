**About our YOLO model**
Usually Yolo has two parts-> detection and classification.
Here, we only have detection part, meaning the number of classes = 0.

process flow:

- **Preprocessing**:
  - Read image via OpenCV (`cv2.imread`) which loads in BGR.
  - Resize to the model input size (default `(640, 640)`), no letterboxing/aspect-ratio preservation.
  - Convert BGR → RGB.
  - Normalize to `[0, 1]` as `float32`. Examples: 0 → 0.0, 128 → 0.5019608, 255 → 1.0
  - Reorder layout HWC → CHW and add batch dimension → NCHW `(1, 3, H, W)`.

- **Inference**:
  - Single forward pass through the model.
  - Output tensor shape for 640×640: `(1, 5, 8400)`.
    - `8400 = 80×80 + 40×40 + 20×20` from strides `8, 16, 32` (anchor-free: one prediction per cell).
    - Channels `5` are `[x_center, y_center, width, height, objectness]` in model-input pixel space.

- **Post-processing**:
  - Filter by confidence threshold.
  - Convert center-format to corner-format: `[x1, y1, x2, y2]`.
  - Scale from model input size back to original image size; clamp to image bounds.
  - Run Non‑Max Suppression (IoU 0.5).
  - Produce predictions as dictionaries: `bbox`, `confidence`, `class_id=0`, and `class_probabilities=[]`.

- **Overlay & saving**:
  - Draw bounding boxes and labels on the original image.
  - Save to `tests/yolo_outputs` by default (or to a provided `output_path`/`output_dir`).

**Performance of YOLOv8 at Different Input Sizes**
(based on Ultralytics benchmarks on COCO with YOLOv8s/m/l/x models)

| Model | Input Size | Speed (ms/img, A100 GPU) | FPS ≈ | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLOv8n (nano)   | 320  | 0.9 ms | ~1100 FPS| ~64% | ~44% |
| YOLOv8n (nano)   | 640  | 1.6 ms | ~625 FPS | ~67% | ~46% |
| YOLOv8n (nano)   | 1280 | 4.3 ms | ~230 FPS | ~69% | ~48% |
| YOLOv8s (small)  | 640  | 6.2 ms | ~160 FPS | ~70% | ~50% |
| YOLOv8m (medium) | 640  | 8.6 ms | ~116 FPS | ~75% | ~53% |
| YOLOv8l (large)  | 640  | 12.6 ms| ~79 FPS  | ~77% | ~55% |
| YOLOv8x (xlarge) | 640  | 17.2 ms| ~58 FPS  | ~78% | ~56% |


**YOLODetector Class**

expects:
    - image_path: path to the image
    - model_path: path to the model
    - confidence_threshold: confidence threshold for the model

returns:
    - predictions: list of predictions
    - overlay_image: overlay image with predictions

**YOLOModel Class**

**Confidence filtering example**
```python
x    = np.array([100.0, 220.0, 330.0])
y    = np.array([ 60.0, 120.0, 180.0])
w    = np.array([ 50.0,  40.0,  70.0])
h    = np.array([ 30.0,  25.0,  35.0])
conf = np.array([ 0.90,  0.50,  0.20])  # indices 0,1 survive; 2 is dropped

conf_mask = conf >= 0.5                 # [True, True, False]

# After filtering (only indices 0 and 1 remain)
x    = np.array([100.0, 220.0])
y    = np.array([ 60.0, 120.0])
w    = np.array([ 50.0,  40.0])
h    = np.array([ 30.0,  25.0])
conf = np.array([ 0.90,  0.50])```