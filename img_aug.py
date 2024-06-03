import os
import shutil
import imgaug.augmenters as iaa
import imageio
import numpy as np
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import os
import shutil
import imgaug.augmenters as iaa
import imageio
import numpy as np
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# Folder paths
source_image_folder = 'source_image_path'
source_annotation_folder = 'source_annotation_path'
target_folder = 'target_folder_path'
augmented_image_folder = os.path.join(target_folder, 'augmented_images')
augmented_annotation_folder = os.path.join(target_folder, 'augmented_annotations')

# Create target folders if they don't exist
os.makedirs(augmented_image_folder, exist_ok=True)
os.makedirs(augmented_annotation_folder, exist_ok=True)

# Data augmentation sequence
augmentation_sequence = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-30, 30)),
    iaa.Multiply((0.8, 1.2)),
    iaa.GaussianBlur(sigma=(0.0, 3.0))
])

# Function to load YOLO format annotations
def load_yolo_annotations(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    annotations = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])
        annotations.append({
            'class_id': class_id,
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height
        })
    return annotations

# Function to save YOLO format annotations
def save_yolo_annotations(annotations, file_path):
    with open(file_path, 'w') as file:
        for annotation in annotations:
            line = f"{annotation['class_id']} {annotation['x_center']} {annotation['y_center']} {annotation['width']} {annotation['height']}\n"
            file.write(line)

# Function to augment images and annotations
def augment_image_and_annotations(source_image_path, annotations):
    image = imageio.imread(source_image_path)
    bbs = []
    for annotation in annotations:
        if annotation['class_id'] == 1:  # Only augment the 'human' class (check class_id)
            x_center, y_center, width, height = annotation['x_center'], annotation['y_center'], annotation['width'], annotation['height']
            x_min = (x_center - width / 2) * image.shape[1]
            y_min = (y_center - height / 2) * image.shape[0]
            x_max = (x_center + width / 2) * image.shape[1]
            y_max = (y_center + height / 2) * image.shape[0]
            bbs.append(BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max, label='human'))
    
    if not bbs:
        return None, None
    
    bbs = BoundingBoxesOnImage(bbs, shape=image.shape)
    augmented_image, augmented_bbs = augmentation_sequence(image=image, bounding_boxes=bbs)
    return augmented_image, augmented_bbs

# Copy and augment images and annotations that contain humans
for file_name in os.listdir(source_annotation_folder):
    annotation_path = os.path.join(source_annotation_folder, file_name)
    image_path = os.path.join(source_image_folder, file_name.replace('.txt', '.jpg'))  # Assuming images are in .jpg format

    if not os.path.exists(image_path):
        continue
    
    annotations = load_yolo_annotations(annotation_path)
    human_annotations = [annotation for annotation in annotations if annotation['class_id'] == 1]  # Check for human class_id

    if human_annotations:
        # Copy original files
        shutil.copy(image_path, os.path.join(target_folder, 'images_folder', os.path.basename(image_path)))
        shutil.copy(annotation_path, os.path.join(target_folder, 'labels_folder', os.path.basename(annotation_path)))

        # Augment images and annotations
        augmentation_count = 5  # Define the number of augmentations
        for i in range(augmentation_count):
            augmented_image, augmented_bbs = augment_image_and_annotations(image_path, annotations)
            if augmented_image is None:
                continue
            
            new_image_path = os.path.join(augmented_image_folder, f'aug_{file_name.replace(".txt", "")}_{i}.jpg')
            new_annotation_path = os.path.join(augmented_annotation_folder, f'aug_{file_name.replace(".txt", "")}_{i}.txt')

            imageio.imwrite(new_image_path, augmented_image)
            
            new_annotations = []
            for bb in augmented_bbs.bounding_boxes:
                x_center = ((bb.x1 + bb.x2) / 2) / augmented_image.shape[1]
                y_center = ((bb.y1 + bb.y2) / 2) / augmented_image.shape[0]
                width = (bb.x2 - bb.x1) / augmented_image.shape[1]
                height = (bb.y2 - bb.y1) / augmented_image.shape[0]
                new_annotations.append({
                    'class_id': 1,  # Human class_id
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })

            save_yolo_annotations(new_annotations, new_annotation_path)
