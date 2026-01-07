# VisDrone-MOT Object Detection and Tracking

This repository contains code and experiments for object detection and tracking on the VisDrone-MOT dataset using YOLOv8.

## Dataset

For the dataset, we used the VisDrone-MOT 2019 dataset, which consists of annotated video sequences captured by drones. To download the dataset used for evaluation, please run the `download-VisDrone.py` script. It will create a `data/VisDrone2019-MOT-test-dev` directory to download and extract the necessary files.

## Object Detection Evaluation

We evaluated the performance of the base YOLOv8n model as well as a fine-tuned YOLOv8s model on the VisDrone dataset. The evaluation metrics used include Mean Average Precision (mAP) at different Intersection over Union (IoU) thresholds, as well as mAP for small, medium, and large objects. To run the evaluation, please refer to the `yolo.ipynb` and the `yolo-fine-tuned.ipynb` Jupyter notebooks. To best utilize a GPU for faster evaluation, these notebooks were run in a Google Colab environment.

## Object Detection Visualization

To visualize the detection results, you can run the `yolo-visualization.py` script which will display the bounding boxes predicted by the model on sequential frames from the dataset as well as the ground truth boxes for comparison. To see the results of the fine-tuned model, pass the --fine-tuned flag when executing the script. You can press 'space' to pause the visualization and 'q' to quit.
