# VisDrone-MOP detection

## Overview

- Detect per frame objects in VisDrone-MOP dataset. Multiple objects per frame.
- If good benchmarks, attempt object tracking.

## Object Detection

Test different models. Compare performance on test set.

- Yolo
- Faster R-CNN
- Try custom model
- Try fine tunning

## Annotations

| Column | Description |
| -------- | ------------- |
| 1 | Frame number |
| 2 | Object ID |
| 3 | Bounding box left (x) |
| 4 | Bounding box top (y) |
| 5 | Bounding box width |
| 6 | Bounding box height |
| 7 | Confidence score |
| 8 | Class label |
| 9 | Truncation ratio |
| 10 | Occlusion ratio |

## Object Tracking
