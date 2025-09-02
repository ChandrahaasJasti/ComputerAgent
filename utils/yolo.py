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
        image = cv2.imread(image_path)
        image = cv2.resize(image, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image
    