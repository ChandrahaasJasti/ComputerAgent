import os
import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOModel:
    """
    YOLO model inference class for ONNX models
    """
    
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (640, 640)):
        """
        Initialize YOLO model
        
        Args:
            model_path: Path to the ONNX model file
            input_size: Input size for the model (width, height)
        """
        self.model_path = model_path
        self.input_size = input_size
        
        # Load ONNX model
        try:
            self.session = ort.InferenceSession(model_path)
            logger.info(f"Successfully loaded ONNX model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
        
        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        logger.info(f"Model input shape: {self.input_shape}")
        
        # Get model output details
        self.output_names = [output.name for output in self.session.get_outputs()]
        logger.info(f"Model outputs: {self.output_names}")
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess image for YOLO model input
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (preprocessed_image, original_size)
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
            
            original_size = (image.shape[1], image.shape[0])  # (width, height)
            
            # Resize image to model input size
            resized_image = cv2.resize(image, self.input_size)
            
            # Convert BGR to RGB
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            resized_image = resized_image.astype(np.float32) / 255.0
            
            # Convert to NCHW format (batch_size, channels, height, width)
            image_tensor = np.transpose(resized_image, (2, 0, 1))
            image_tensor = np.expand_dims(image_tensor, axis=0)
            
            return image_tensor, original_size
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def postprocess_predictions(self, outputs: List[np.ndarray], 
                               original_size: Tuple[int, int], 
                               confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Post-process model outputs to extract bounding boxes and predictions
        
        Args:
            outputs: Model output tensors
            original_size: Original image size (width, height)
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of prediction dictionaries
        """
        def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
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

        predictions: List[Dict[str, Any]] = []
        
        try:
            output = outputs[0]
            if output.ndim == 3:
                output = output[0]
            orig_width, orig_height = original_size

            # Our diagnostic showed output shape (1,5,8400) => after squeeze becomes (5, N)
            # Interpret rows as [xc, yc, w, h, conf] in model input pixels
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

                scale_x = orig_width / float(self.input_size[0])
                scale_y = orig_height / float(self.input_size[1])

                x1 = (x - w / 2.0) * scale_x
                y1 = (y - h / 2.0) * scale_y
                x2 = (x + w / 2.0) * scale_x
                y2 = (y + h / 2.0) * scale_y
                boxes = np.stack([x1, y1, x2, y2], axis=1)

                boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_width - 1)
                boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_height - 1)
                boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_width - 1)
                boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_height - 1)

                keep = nms(boxes, conf, iou_threshold=0.5)
                for i in keep:
                    predictions.append({
                        'bbox': boxes[i].astype(int).tolist(),
                        'confidence': float(conf[i]),
                        'class_id': 0,
                        'class_probabilities': []
                    })
                return predictions

            # Fallback generic parsing for [num, feat]
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
                        scale_x = orig_width / float(self.input_size[0])
                        scale_y = orig_height / float(self.input_size[1])
                        x1 = int((x_center - width / 2.0) * scale_x)
                        y1 = int((y_center - height / 2.0) * scale_y)
                        x2 = int((x_center + width / 2.0) * scale_x)
                        y2 = int((y_center + height / 2.0) * scale_y)
                    else:
                        x1 = int((x_center - width / 2.0) * orig_width)
                        y1 = int((y_center - height / 2.0) * orig_height)
                        x2 = int((x_center + width / 2.0) * orig_width)
                        y2 = int((y_center + height / 2.0) * orig_height)
                    class_probabilities = detection[5:].tolist() if len(detection) > 5 else []
                    predicted_class = int(np.argmax(class_probabilities)) if class_probabilities else 0
                    predictions.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': predicted_class,
                        'class_probabilities': class_probabilities
                    })
                return predictions

            logger.error(f"Unhandled output shape: {outputs[0].shape}")
            return []
        except Exception as e:
            logger.error(f"Error post-processing predictions: {e}")
            return []
    
    def predict(self, image_path: str, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Run inference on an image
        
        Args:
            image_path: Path to the input image
            confidence_threshold: Minimum confidence threshold for predictions
            
        Returns:
            List of prediction dictionaries
        """
        try:
            # Preprocess image
            image_tensor, original_size = self.preprocess_image(image_path)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: image_tensor})
            
            # Post-process predictions
            predictions = self.postprocess_predictions(outputs, original_size, confidence_threshold)
            
            logger.info(f"Found {len(predictions)} predictions in {image_path}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return []

def predict_yolo(image_path: str, model_path: Optional[str] = None, 
                confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Convenience function to run YOLO prediction
    
    Args:
        image_path: Path to the input image
        model_path: Path to the ONNX model (if None, uses default path)
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        List of prediction dictionaries
    """
    # Default model path
    if model_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, 'trainedmodels', 'model_dynamic.onnx')
    
    # Check if model file exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return []
    
    # Initialize model
    model = YOLOModel(model_path)
    
    # Run prediction
    predictions = model.predict(image_path, confidence_threshold)
    
    return predictions


def draw_bounding_boxes(image: np.ndarray, predictions: List[Dict[str, Any]], 
                        class_names: Optional[List[str]] = None, 
                        colors: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
    """
    Draw bounding boxes and labels on an image.
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

    for prediction in predictions:
        x1, y1, x2, y2 = prediction.get('bbox', [0, 0, 0, 0])
        confidence = float(prediction.get('confidence', 0.0))
        class_id = int(prediction.get('class_id', 0))

        # Clamp coordinates to image bounds
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w - 1, int(x2)))
        y2 = max(0, min(h - 1, int(y2)))

        color = colors[class_id % len(colors)]

        # Draw bounding box
        cv2.rectangle(overlay_image, (x1, y1), (x2, y2), color, 3)

        # Label
        if class_names and class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"Class_{class_id}"
        label = f"{class_name} {confidence:.2f}"

        # Background rectangle for text
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_x = x1
        text_y = max(text_h + baseline, y1)
        cv2.rectangle(overlay_image,
                      (text_x, text_y - text_h - baseline),
                      (text_x + text_w, text_y),
                      color,
                      -1)

        # Draw text
        cv2.putText(overlay_image,
                    label,
                    (text_x, text_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2)

    return overlay_image


def create_overlay_image(image_path: str, predictions: List[Dict[str, Any]],
                         class_names: Optional[List[str]] = None,
                         colors: Optional[List[Tuple[int, int, int]]] = None) -> Optional[np.ndarray]:
    """
    Create an overlay image with bounding boxes from predictions.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        return draw_bounding_boxes(image, predictions, class_names, colors)
    except Exception as e:
        logger.error(f"Error creating overlay image: {e}")
        return None


def save_overlay_image(image_path: str, overlay_image: np.ndarray,
                       output_path: Optional[str] = None,
                       output_dir: Optional[str] = None) -> bool:
    """
    Save the overlay image to a file. Defaults to tests/yolo_outputs.
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


def predict_and_save_overlay(image_path: str,
                             model_path: Optional[str] = None,
                             confidence_threshold: float = 0.5,
                             output_path: Optional[str] = None,
                             output_dir: Optional[str] = None,
                             class_names: Optional[List[str]] = None,
                             colors: Optional[List[Tuple[int, int, int]]] = None) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Run YOLO prediction and save overlay image with bounding boxes.
    Returns (predictions, save_success).
    """
    predictions = predict_yolo(image_path, model_path, confidence_threshold)

    if not predictions:
        logger.warning(f"No predictions found for {image_path}")
        return predictions, False

    overlay_image = create_overlay_image(image_path, predictions, class_names, colors)
    if overlay_image is None:
        logger.error(f"Failed to create overlay image for {image_path}")
        return predictions, False

    save_success = save_overlay_image(image_path, overlay_image, output_path, output_dir)
    return predictions, save_success


# Example usage
if __name__ == "__main__":
    # Test the function
    test_image_path = "/home/chandrahaas/codes/SaarathiFinance/ComputerAgent/tests/Screenshot from 2025-08-29 14-12-15.png"

    if os.path.exists(test_image_path):
        preds, saved = predict_and_save_overlay(
            test_image_path,
            confidence_threshold=0.1,
            output_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests', 'yolo_outputs')
        )
        print(f"Found {len(preds)} predictions")
        print(preds[0])
        print("Overlay image saved successfully!" if saved else "Failed to save overlay image")
    else:
        print(f"Test image not found: {test_image_path}")