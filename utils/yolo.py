import os
import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
----------------------------------------------------------------------------------------------------------------------------------------------------
Overlay helpers (consume simplified predictions with 'bounding_box_coordinates' and 'confidence')
----------------------------------------------------------------------------------------------------------------------------------------------------
"""

class Overlays:
    """
    Utility class for creating and saving overlay images from predictions
    and drawing bounding boxes on images.
    ENTRYPOINT: save_predicted_image
    """
    def create_overlay_image_from_predictions(self,image_path: str,
                                          predictions: List[Dict[str, Any]],
                                          colors: Optional[List[Tuple[int, int, int]]] = None) -> Optional[np.ndarray]:
        """
        simple caller function for draw_bounding_boxes_from_predictions
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
            return self.draw_bounding_boxes_from_predictions(image, predictions, colors)
        except Exception as e:
            logger.error(f"Error creating overlay image: {e}")
            return None

    def draw_bounding_boxes_from_predictions(self,image: np.ndarray,predictions: List[Dict[str, Any]],colors: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
        """
        draws bounding boxes on an image using predicted coordinates
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
        
    def save_overlay_image(self,image_path: str,overlay_image: np.ndarray,output_path: Optional[str] = None,output_dir: Optional[str] = None) -> bool:
        """
        Save the overlay image to a file. If output_path is None, saves to tests/yolo_outputs.
        """
        try:
            if overlay_image is None:
                raise ValueError("Overlay image is None, cannot save")
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

    def save_predicted_image(self,image_path: str,predictions: List[Dict[str, Any]],
                                                 output_path: Optional[str] = None,
                                                 output_dir: Optional[str] = None,
                                                 colors: Optional[List[Tuple[int, int, int]]] = None
                                                 ) -> Tuple[Optional[np.ndarray], bool]:
        """
        ENTRYPOINT FOR THE CLASS
        """
        overlay = self.create_overlay_image_from_predictions(image_path, predictions, colors)
        if overlay is None:
            return None, False
        saved = self.save_overlay_image(image_path, overlay, output_path, output_dir)
        return overlay, saved




"""
    =============================================YOLO class from this point onward =====================================================================================================================================
"""





class YOLODetector(Overlays):
    """
    YOLO model inference class for ONNX models
    """
    def __init__(self,model_path):
        self.runtime_session = ort.InferenceSession(model_path)
        self.input_name = self.runtime_session.get_inputs()[0].name
        
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
        outputs = self.runtime_session.run(None, {self.input_name: image_tensor})
        predictions = self.postprocess_predictions(outputs,original_size)
        return predictions
    
    def predict_with_log(self,image_path):
        predictions = self.predict(image_path)
        self.save_predicted_image(image_path,predictions)
        return predictions
    
    def nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
        """
        NON-MAX-SUPPRESSION(yolo concept to remove overlapping bounding boxes of the same object)
        """
        if len(boxes) == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

    def postprocess_predictions(self,outputs,original_size):
        predictions: List[Dict[str, Any]] = []
        
        try:
            output = outputs[0]
            if output.ndim == 3:
                output = output[0]
            orig_width, orig_height = original_size
            confidence_threshold = 0.5

            # Case: output shaped like (5, N) -> rows [xc, yc, w, h, conf] in input pixels
            if output.shape[0] == 5:
                x = output[0]
                y = output[1]
                w = output[2]
                h = output[3]
                conf = output[4]

                conf_mask = conf >= confidence_threshold
                x = x[conf_mask]
                y = y[conf_mask]
                w = w[conf_mask]
                h = h[conf_mask]
                conf = conf[conf_mask]

                if x.size == 0:
                    return []

                # Model input assumed 640x640 (as used in preprocess)
                scale_x = orig_width / 640.0
                scale_y = orig_height / 640.0

                x1 = (x - w / 2.0) * scale_x
                y1 = (y - h / 2.0) * scale_y
                x2 = (x + w / 2.0) * scale_x
                y2 = (y + h / 2.0) * scale_y
                boxes = np.stack([x1, y1, x2, y2], axis=1)

                boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_width - 1)
                boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_height - 1)
                boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_width - 1)
                boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_height - 1)

                keep = self.nms(boxes, conf, iou_threshold=0.5)
                for i in keep:
                    predictions.append({
                        'bounding_box_coordinates': boxes[i].astype(int).tolist(),
                        'confidence': float(conf[i])
                    })
                return predictions

            # Fallback: generic parsing for [num, feat]
            if output.ndim == 2:
                for detection in output:
                    if len(detection) < 5:
                        continue
                    confidence = float(detection[4])
                    if confidence < confidence_threshold:
                        continue
                    x_center, y_center, width, height = detection[:4]

                    # If values are pixels, scale by input size; otherwise treat as normalized
                    if max(width, height, x_center, y_center) > 2.0:
                        scale_x = orig_width / 640.0
                        scale_y = orig_height / 640.0
                        x1 = int((x_center - width / 2.0) * scale_x)
                        y1 = int((y_center - height / 2.0) * scale_y)
                        x2 = int((x_center + width / 2.0) * scale_x)
                        y2 = int((y_center + height / 2.0) * scale_y)
                    else:
                        x1 = int((x_center - width / 2.0) * orig_width)
                        y1 = int((y_center - height / 2.0) * orig_height)
                        x2 = int((x_center + width / 2.0) * orig_width)
                        y2 = int((y_center + height / 2.0) * orig_height)

                    # Clamp to image bounds
                    x1 = max(0, min(orig_width - 1, x1))
                    y1 = max(0, min(orig_height - 1, y1))
                    x2 = max(0, min(orig_width - 1, x2))
                    y2 = max(0, min(orig_height - 1, y2))

                    predictions.append({
                        'bounding_box_coordinates': [x1, y1, x2, y2],
                        'confidence': confidence
                    })
                return predictions

            logger.error(f"Unhandled output shape: {outputs[0].shape}")
            return []
        except Exception as e:
            logger.error(f"Error post-processing predictions: {e}")
            return []
    

# -----------------------------------------------------------------------------
# Overlay helpers (consume simplified predictions with 'bounding_box_coordinates' and 'confidence')
# -----------------------------------------------------------------------------








    
    