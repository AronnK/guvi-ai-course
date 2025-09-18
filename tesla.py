import os
import shutil
import yaml
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from tqdm import tqdm

print("Starting Data Preparation...")

base_dir = './'
image_dir = os.path.join(base_dir, 'image_2/training/image_2')
label_dir = os.path.join(base_dir, 'label_2/training/label_2')

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'valid')

for d in [train_dir, val_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(os.path.join(d, 'images'))
    os.makedirs(os.path.join(d, 'labels'))

try:
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
except FileNotFoundError:
    print(f"Error: The source image directory was not found at '{image_dir}'")
    exit()

train_files, val_files = train_test_split(image_files, test_size=0.1, random_state=42)

def move_files(file_list, source_image_dir, source_label_dir, dest_dir):
    dest_image_dir = os.path.join(dest_dir, 'images')
    dest_label_dir = os.path.join(dest_dir, 'labels')
    
    for file in tqdm(file_list, desc=f"Moving files to {os.path.basename(dest_dir)}"):
        image_path = os.path.join(source_image_dir, file)
        label_file = file.replace('.png', '.txt')
        label_path = os.path.join(source_label_dir, label_file)

        if os.path.exists(label_path):
            shutil.copy(image_path, dest_image_dir)
            shutil.copy(label_path, dest_label_dir)

move_files(train_files, image_dir, label_dir, train_dir)
move_files(val_files, image_dir, label_dir, val_dir)

print(" Data preparation complete!")
print("-" * 30)

print("Creating kitti.yaml configuration file...")

data_config = {
    'names': ['car', 'pedestrian', 'van', 'cycle', 'truck', 'miscellaneous', 'tram', 'person_sitting'],
    'nc': 8,
    'train': os.path.abspath(train_dir) + '/images',
    'val': os.path.abspath(val_dir) + '/images'
}

with open('kitti.yaml', 'w') as outfile:
    yaml.dump(data_config, outfile, default_flow_style=False)

print(" kitti.yaml file created successfully!")
print("-" * 30)

print(" Loading pretrained YOLOv8 model and starting training...")

model = YOLO('yolov8n.pt')

results = model.train(
    data='kitti.yaml',
    epochs=10,
    patience=3,
    project='Autopilot-Clone',
    device=0
)

print(" Model training complete!")
print("-" * 30)

print("📈 Validating the best performing model...")

best_model_path = os.path.join('Autopilot-Clone/train/weights/best.pt')
model = YOLO(best_model_path)

metrics = model.val(data='kitti.yaml')

precision = metrics.results_dict['metrics/precision(B)']
recall = metrics.results_dict['metrics/recall(B)']
mAP50 = metrics.results_dict['metrics/mAP50(B)']
mAP50_95 = metrics.results_dict['metrics/mAP50-95(B)']

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"mAP@50: {mAP50:.4f}")
print(f"mAP@50-95: {mAP50_95:.4f}")
print(" Validation complete!")
print("-" * 30)

print(" Performing inference on validation images as a demonstration...")

results = model.predict(
    source=os.path.join(val_dir, 'images'),
    save=True, 
    conf=0.5 
)

print(f"Inference complete! Results saved in the 'Autopilot-Clone/train' directory.")
