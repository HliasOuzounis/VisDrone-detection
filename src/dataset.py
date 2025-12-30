import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import List, Tuple
import cv2

from .annotations import Annotations

class VisDroneDetectionDataset(Dataset):
    def __init__(self, annotations_files: List[str], images_dir: str, transform=transforms.ToTensor()):
        super().__init__()
        self.transform = transform
        
        images_dir = Path(images_dir)
        self.annotations: List[Annotations] = []
        self.frame_data: List[Tuple[str, int, int]] = []
        
        for annot_idx, annot_file in enumerate(annotations_files):
            # Assert that all annotation files have corresponding image folders
            folder_name = Path(annot_file).stem
            image_folder = images_dir / folder_name
            assert image_folder.is_dir(), f"Missing folder: {image_folder}"

            self.annotations.append(Annotations(annot_file))
            
            frame_min, frame_max = self.annotations[annot_idx].get_frame_range()
            for frame_id in range(frame_min, frame_max + 1):
                image_path = f'{image_folder}/{frame_id:07d}.jpg'
                self.frame_data.append((image_path, annot_idx, frame_id))
            
    def __len__(self):
        return len(self.frame_data)
    
    def __getitem__(self, idx: int):
        image_path, annot_idx, frame_idx = self.frame_data[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        
        annot = self.annotations[annot_idx]
        frame_range = annot.get_frame_range()
        
        frame_idx = max(frame_range[0], min(frame_range[1], frame_idx))
        
        frame_data = annot.get_frame(frame_idx)
        
        # Convert to tensor
        x, y, w, h = frame_data[['x', 'y', 'w', 'h']].values.T
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        boxes = torch.tensor(list(zip(x1, y1, x2, y2)), dtype=torch.float32)
        labels = torch.tensor(frame_data['class_id'].values, dtype=torch.int64)
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
        }

def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    # images = [item['image'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    return images, boxes, labels