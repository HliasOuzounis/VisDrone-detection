import cv2
import torch
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from .annotations import VisDroneClasses, LabelsConverter

class ObjectDetectionEvaluator:
    def __init__(self, device='cuda', labels_converter=LabelsConverter()):
        self.device = device
        # mAP handles both box overlap and class correctness
        self.metric = MeanAveragePrecision(box_format='xyxy')
        self.labels_converter = labels_converter

    @torch.no_grad()
    def evaluate_mAP(self, model, dataloader):
        model = model.to(self.device)
        model.eval()
        self.metric.reset()

        for (images, boxes, labels) in tqdm(dataloader, desc="Evaluating"):
            images = images.to(self.device)
            
            outputs = []
            targets = []
            for box, label in zip(boxes, labels):
                targets.append({
                    'boxes': box.to(self.device),
                    'labels': torch.tensor([self.labels_converter(label.item()) for label in label]).to(self.device)
                })
                
            # outputs: list of dicts [{'boxes': tensor, 'scores': tensor, 'labels': tensor}]
            outputs = model(images)

            self.metric.update(outputs, targets)

        results = self.metric.compute()
        self._print_results(results)
        return results

    def _print_results(self, results):
        print("\n--- Evaluation Results ---")
        print(f"mAP (Mean Average Precision): {results['map']:.4f}")
        print(f"mAP@50 (IoU threshold 0.5):  {results['map_50']:.4f}")
        print(f"mAP@75 (IoU threshold 0.75): {results['map_75']:.4f}")
        print(f"mAR (Mean Average Recall):    {results['mar_100']:.4f}")
        print(f"mAP for small objects: {results['map_small']:.4f}")
        print(f"mAP for medium objects: {results['map_medium']:.4f}")
        print(f"mAP for large objects: {results['map_large']:.4f}")
        print("---------------------------\n")

class QualitativeEvaluator:
    def __init__(self, device='cuda', labels_converter=LabelsConverter()):
        self.device = device
        self.labels_converter = labels_converter

        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(VisDroneClasses), 3), dtype=np.uint8)
        
    @torch.no_grad()
    def evaluate_qualitative(self, model, dataloader):
        model = model.to(self.device)
        model.eval()
        j = 0
        for images, boxes_ground_truth, labels_ground_truth in dataloader:            
            # Get predictions from your wrapper
            # Our wrapper returns: [{'boxes': T, 'scores': T, 'labels': T}, ...]
            outputs = model(images.to(self.device))

            for i in range(len(images)):
                j += 1
                img_ground_truth = self._populate_image(images[i], boxes_ground_truth[i], labels_ground_truth[i])
                cv2.imshow("Ground Truth", img_ground_truth)

                # 2. Extract predictions for this specific image
                out = outputs[i]
                boxes = out['boxes'].cpu()
                # scores = out['scores'].cpu()
                labels = out['labels'].cpu()
                img_prediction = self._populate_image(images[i], boxes, labels)

                cv2.imshow("Detection Preview", img_prediction)

                # Press 'q' to quit or any key to continue to next image
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    print("Evaluation interrupted by user.")
                    return

        cv2.destroyAllWindows()

    def _populate_image(self, image: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor):
        """ Draws boxes and labels on the image tensor and returns a new image as a numpy array. """
        img = image.cpu().permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for box, label in zip(boxes, labels):
            label = self.labels_converter(label.item())
            x1, y1, x2, y2 = box.int().cpu().numpy()
            class_name = VisDroneClasses(label).name
            color = [int(c) for c in self.colors[label % len(self.colors)]]

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, class_name, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img
