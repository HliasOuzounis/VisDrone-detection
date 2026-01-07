from src.dataset import VisDroneDetectionDataset, collate_fn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.evaluate import QualitativeEvaluator
from src.models import YOLODetectionModel
from src.annotations import VisDroneClasses, LabelsConverter
import os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fine-tuned', action='store_true', help='Use fine-tuned model')
args = parser.parse_args()

annot_dir = './data/VisDrone2019-MOT-test-dev/annotations'
annot_files = [os.path.join(annot_dir, f) for f in os.listdir(annot_dir)]

images_directory = './data/VisDrone2019-MOT-test-dev/sequences'
    
resize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: transforms.functional.pad(img, (0, 0, (32 - img.shape[-1] % 32) % 32, (32 - img.shape[-2] % 32) % 32))),
])

dataset = VisDroneDetectionDataset(annot_files, images_directory , transform=resize)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

print(f"Dataset size: {len(dataset)} images")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
evaluator = QualitativeEvaluator(device=device)

if args.fine_tuned:
    yolo_vis_converter = LabelsConverter()
    for class_id in VisDroneClasses:
        yolo_vis_converter.rules[class_id] = class_id + 1
    model = YOLODetectionModel(model_path='./data/models/checkpoints/yolov8s-VisDrone.pt', yolo_classes=yolo_vis_converter)
else:
    yolo_classes = LabelsConverter()
    yolo_classes.rules[0] = VisDroneClasses.PEDESTRIAN
    yolo_classes.rules[1] = VisDroneClasses.BICYCLE
    yolo_classes.rules[2] = VisDroneClasses.CAR
    yolo_classes.rules[3] = VisDroneClasses.MOTOR
    yolo_classes.rules[5] = VisDroneClasses.BUS
    model = YOLODetectionModel(model_path='./data/models/checkpoints/yolov8n.pt', yolo_classes=yolo_classes)

evaluator.evaluate_qualitative(model, data_loader)