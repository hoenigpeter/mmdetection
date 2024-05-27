import json
import matplotlib.pyplot as plt

# Initialize lists to hold the data
steps = []
accuracies = []
losses = []

# Load and parse the JSON file line by line
file_path = '/hdd2/mmdetection/faster-rcnn_r50_fpn_2x_tless/20240523_082526/vis_data/20240523_082526.json'  # Change this to your JSON file path
with open(file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        if 'coco/bbox_mAP' in data:
            steps.append(data['step'])
            accuracies.append(data['coco/bbox_mAP'])
        if 'loss' in data:
            losses.append(data['loss'])

# Plot accuracy
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(steps, accuracies, marker='o', linestyle='-', color='b')
plt.title('COCO BBox mAP over Steps')
plt.xlabel('Steps')
plt.ylabel('Accuracy (mAP)')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(range(len(losses)), losses, marker='o', linestyle='-', color='r')
plt.title('Loss over Steps')
plt.xlabel('Steps')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()
