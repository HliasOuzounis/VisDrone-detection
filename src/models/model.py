from typing import Any, Dict, List
import torch
from abc import ABC

class DetectionModel(ABC):
    """
    Abstract base class for detection models.
    Wraps a detection model and provides common functionality.
    """
    def __init__(self):
        super().__init__()
        self.model = self.load_model()
        
    def load_model(self) -> Any:
        """
        Load the model from file or initialize it.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("load_model method not implemented.")
    
    def to(self, device: torch.device) -> 'DetectionModel':
        """
        Move the model to the specified device.
        """
        self.model.to(device)
        return self

    def eval(self) -> 'DetectionModel':
        """
        Set the model to evaluation mode.
        """
        self.model.eval()
        return self

    def train(self) -> 'DetectionModel':
        """
        Set the model to training mode.
        """
        self.model.train()
        return self
        
    def __call__(self, batch_image: torch.tensor) -> List[Dict[str, torch.tensor]]:
        results = self.model(batch_image, conf=self.conf_threshold, iou=self.iou_threshold)
        return self._post_processing(results)
    
    def _post_processing(self, predictions):
        """
        Process the predictions to the correct format
        """
        return predictions