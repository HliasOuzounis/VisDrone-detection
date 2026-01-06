from src.dataset import VisDroneDetectionDataset, collate_fn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.evaluate import ObjectDetectionEvaluator, QualitativeEvaluator
from src.models import YOLODetectionModel
from src.annotations import VisDroneClasses, LabelsConverter
import os

if __name__ == '__main__':
    # annot_files = [
    #     './data/VisDrone2019-MOT-test-dev/annotations/uav0000009_03358_v.txt',
    # ]
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
    out = dataset[1]
    
    # Remove classes not in COCO
    labels_rules = LabelsConverter()
    # labels_rules.rules[VisDroneClasses.PEDESTRIAN] = VisDroneClasses.PERSON
    # labels_rules.rules[VisDroneClasses.TRICYCLE] = VisDroneClasses.BICYCLE
    # labels_rules.rules[VisDroneClasses.AWNING_TRICYCLE] = VisDroneClasses.BICYCLE
    # labels_rules.rules[VisDroneClasses.TRUCK] = VisDroneClasses.CAR
    # labels_rules.rules[VisDroneClasses.VAN] = VisDroneClasses.CAR
    
    evaluator1 = ObjectDetectionEvaluator(device='cpu', labels_converter=labels_rules)
    evaluator2 = QualitativeEvaluator(device='cpu', labels_converter=labels_rules)

    yolo_classes = LabelsConverter()
    yolo_classes.rules[0] = VisDroneClasses.PERSON
    yolo_classes.rules[1] = VisDroneClasses.BICYCLE
    yolo_classes.rules[2] = VisDroneClasses.CAR
    yolo_classes.rules[3] = VisDroneClasses.MOTOR
    yolo_classes.rules[5] = VisDroneClasses.BUS
    # model = YOLODetectionModel(model_path='./data/models/checkpoints/yolov8n.pt', yolo_classes=yolo_classes)
    
    yolo_vis_converter = LabelsConverter()
    for class_id in VisDroneClasses:
        yolo_vis_converter.rules[class_id] = class_id + 1
    model = YOLODetectionModel(model_path='./data/models/checkpoints/yolov8s-VisDrone.pt', yolo_classes=yolo_vis_converter)
    
    results = evaluator1.evaluate_mAP(model, data_loader)
    # results = evaluator2.evaluate_qualitative(model, data_loader)
    print("Evaluation Results:", results)