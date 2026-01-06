# Results

## YOLO

### Out of the box (YOLO class adjustments)

mAP (Mean Average Precision): 0.0029
mAP@50 (IoU threshold 0.5):  0.0044
mAP@75 (IoU threshold 0.75): 0.0032
mAR (Mean Average Recall):    0.0102
mAP for small objects: 0.0010
mAP for medium objects: 0.0020
mAP for large objects: 0.0073

1. The Critical "Detection Gap"

The most alarming shift is in the mAP@50 (0.1408).

    Your previous score was 0.2892.

    Falling to 14% even at a loose IoU threshold indicates that the model is now missing more than 85% of the objects entirely (False Negatives) or is hallucinating objects where none exist (False Positives).

2. Failure at Scale (The Small Object Crisis)

    mAP Small (0.0403): At 4%, the model is practically blind to small objects. In drone imagery, pedestrians and bicycles often occupy less than 32×32 pixels. If your model resolution is set too low (e.g., standard 640px), these objects become smaller than a single feature map pixel after downsampling.

    mAP Large (0.3085): While this is your "best" score, it has dropped from your previous 0.48. Even for large objects like buses or trucks, the model is only succeeding about 30% of the time.

### With VisDrone classes adjusted

mAP (Mean Average Precision): 0.0395
mAP@50 (IoU threshold 0.5):  0.0770
mAP@75 (IoU threshold 0.75): 0.0354
mAR (Mean Average Recall):    0.0925
mAP for small objects: 0.0123
mAP for medium objects: 0.0469
mAP for large objects: 0.0726

Localization Quality (@50 vs. @75)

This is where we see how well the model draws boxes:

    mAP@50 (0.2892): When you only require a 50% overlap (a "loose" match), the model performs significantly better (~29%). This suggests the model is identifying the general area of objects reasonably well.

    mAP@75 (0.0807): When you require a 75% overlap (a "tight," high-precision match), the score plummets to 8%.

    Diagnosis: The model has "sloppy" bounding boxes. It knows where the object is, but it can't tightly wrap the box around it.

3. The Scale Problem (Small vs. Large)

This is the most revealing part of your data:

    Small Objects (0.0663): The model is almost blind to small objects (6.6%). This is a very common issue in drone or satellite imagery (like VisDrone).

    Medium Objects (0.2277): Performance improves significantly for medium targets.

    Large Objects (0.4875): The model is actually quite decent at finding large objects (~49%).

### Fine-tuned on VisDrone

mAP (Mean Average Precision): 0.1863
mAP@50 (IoU threshold 0.5):  0.3765
mAP@75 (IoU threshold 0.75): 0.1615
mAR (Mean Average Recall):    0.2746
mAP for small objects: 0.0665
mAP for medium objects: 0.2335
mAP for large objects: 0.3039

## Object tracking

### Kalaman + IoU based tracker

### Mention Dense NN for re-identification

Challenge from occlusions and reemerging objects
