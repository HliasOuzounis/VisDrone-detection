from typing import List, Dict, Any
from collections import defaultdict
import os

from .model import DetectionModel
from ..annotations import VisDroneClasses, LabelsConverter
from ultralytics import YOLO
import torch

class YOLODetectionModel(DetectionModel):
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25, iou_threshold: float = 0.45, yolo_classes: LabelsConverter = LabelsConverter()):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        super().__init__()
        
        # Mapping YOLO class indices to VisDrone class IDs
        # Missing a lot of classes: Pedestrian, Truck, Tricycle, Awning Tricycle
        self.yolo_classes = yolo_classes

    def load_model(self) -> YOLO:
        model = YOLO(self.model_path)
        return model
    
    def _post_processing(self, results) -> List[Dict[str, torch.Tensor]]:
        """
        Converts Ultralytics Results objects to:
        [{'boxes': tensor, 'scores': tensor, 'labels': tensor}, ...]
        """
        processed_outputs = []
        
        for r in results:
            boxes = r.boxes.xyxy 
            scores = r.boxes.conf
            labels = r.boxes.cls
            
            # Map YOLO class indices to VisDrone class IDs
            mapped_labels = []
            for lbl in labels:
                mapped_labels.append(self.yolo_classes(int(lbl)))
            labels = torch.tensor(mapped_labels, device=labels.device)

            processed_outputs.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })
            
        return processed_outputs