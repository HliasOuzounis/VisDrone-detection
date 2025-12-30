from src.dataset import VisDroneDetectionDataset, collate_fn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.evaluate import ObjectDetectionEvaluator, QualitativeEvaluator
from src.models import YOLODetectionModel

if __name__ == '__main__':
    annot_files = [
        './data/VisDrone2019-VID-test-dev/annotations/uav0000009_03358_v.txt',
        # './data/VisDrone2019-VID-test-dev/annotations/uav0000249_00001_v.txt'
    ]
    images_directory = './data/VisDrone2019-VID-test-dev/sequences'
    
    resize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda img: transforms.functional.pad(img, (0, 0, (32 - img.shape[-1] % 32) % 32, (32 - img.shape[-2] % 32) % 32))),
    ])
    
    dataset = VisDroneDetectionDataset(annot_files, images_directory , transform=resize)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    print(f"Dataset size: {len(dataset)} images")
    out = dataset[1]
    print(len(out['boxes']), "boxes in image 1")
    print(out['boxes'])
    print(out['labels'])

    evaluator = ObjectDetectionEvaluator(device='cpu')
    # evaluator = QualitativeEvaluator(device='cpu')
    # Assuming 'model' is defined and loaded elsewhere

    model = YOLODetectionModel(model_path='./src/models/checkpoints/yolov8n.pt')
    
    results = evaluator.evaluate_mAP(model, data_loader)
    # results = evaluator.evaluate_qualitative(model, data_loader)
    print("Evaluation Results:", results)