import os
import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLODetector:
    """
    YOLO model inference class for ONNX models
    """
    def __init__(self,model_path):
        self.runtime_session = ort.InferenceSession(model_path)
        
    def preprocess_image(self,image_path):
        """
        takes an image path, return image tensor and original size
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        original_size = (image.shape[1], image.shape[0])
        image = cv2.resize(image, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image,original_size

    def predict(self,image_path):
        image_tensor,original_size = self.preprocess_image(image_path)
        outputs = self.runtime_session.run(None, {'input': image_tensor})
        predictions = self.postprocess_predictions(outputs,original_size)
        return predictions
    
    def postprocess_predictions(self,outputs,original_size):
        pass
    

# -----------------------------------------------------------------------------
# Overlay helpers (consume simplified predictions with 'bounding_box_coordinates' and 'confidence')
# -----------------------------------------------------------------------------

def draw_bounding_boxes_from_predictions(image: np.ndarray,
                                         predictions: List[Dict[str, Any]],
                                         colors: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
    """
    Draw bounding boxes on an image using predictions with keys:
      - 'bounding_box_coordinates': [x1, y1, x2, y2] (in original image pixels)
      - 'confidence': float

    Returns a copy of the image with overlays.
    """
    overlay_image = image.copy()

    if colors is None:
        colors = [
            (0, 255, 0),     # Green
            (255, 0, 0),     # Blue
            (0, 0, 255),     # Red
            (255, 255, 0),   # Cyan
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Yellow
            (128, 128, 128), # Gray
            (255, 128, 0),   # Orange
            (128, 0, 255),   # Purple
            (0, 128, 255),   # Light blue
        ]

    h, w = overlay_image.shape[:2]

    for idx, pred in enumerate(predictions):
        x1, y1, x2, y2 = pred.get('bounding_box_coordinates', [0, 0, 0, 0])
        confidence = float(pred.get('confidence', 0.0))

        # Clamp to image bounds
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w - 1, int(x2)))
        y2 = max(0, min(h - 1, int(y2)))

        color = colors[idx % len(colors)]

        # Rectangle
        cv2.rectangle(overlay_image, (x1, y1), (x2, y2), color, 3)

        # Label showing confidence
        label = f"{confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_x = x1
        text_y = max(text_h + baseline, y1)

        # Filled background for readability
        cv2.rectangle(overlay_image,
                      (text_x, text_y - text_h - baseline),
                      (text_x + text_w, text_y),
                      color,
                      -1)

        # Text
        cv2.putText(overlay_image,
                    label,
                    (text_x, text_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2)

    return overlay_image


def create_overlay_image_from_predictions(image_path: str,
                                          predictions: List[Dict[str, Any]],
                                          colors: Optional[List[Tuple[int, int, int]]] = None) -> Optional[np.ndarray]:
    """
    Read image from disk and overlay bounding boxes using simplified predictions.
    Returns the overlaid image array or None if loading fails.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        return draw_bounding_boxes_from_predictions(image, predictions, colors)
    except Exception as e:
        logger.error(f"Error creating overlay image: {e}")
        return None


def save_overlay_image(image_path: str,
                       overlay_image: np.ndarray,
                       output_path: Optional[str] = None,
                       output_dir: Optional[str] = None) -> bool:
    """
    Save the overlay image to a file. If output_path is None, saves to tests/yolo_outputs.
    """
    try:
        if overlay_image is None:
            logger.error("Overlay image is None, cannot save")
            return False

        if output_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            if output_dir is None:
                output_dir = os.path.join(project_root, 'tests', 'yolo_outputs')
            os.makedirs(output_dir, exist_ok=True)

            original_filename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(original_filename)[0]
            output_path = os.path.join(output_dir, f"{name_without_ext}_overlay.jpg")

        success = cv2.imwrite(output_path, overlay_image)
        if success:
            logger.info(f"Overlay image saved to: {output_path}")
            return True
        logger.error(f"Failed to save overlay image to: {output_path}")
        return False
    except Exception as e:
        logger.error(f"Error saving overlay image: {e}")
        return False
    
    