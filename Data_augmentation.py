from PIL import Image, ImageEnhance
import os
import numpy as np
from tqdm import tqdm  # Progress bar goodness

# Original data directory
data_dir = "Data"
positive_dir = os.path.join(data_dir, "Positive")
negative_dir = os.path.join(data_dir, "Negative")

# New Pro_Data directory
pro_data_dir = "Pro_Data"
pro_positive_dir = os.path.join(pro_data_dir, "Positive")
pro_negative_dir = os.path.join(pro_data_dir, "Negative")
os.makedirs(pro_positive_dir, exist_ok=True)
os.makedirs(pro_negative_dir, exist_ok=True)

# Function to augment an image
def augment_image(img):
    angle = np.random.uniform(-30, 30)
    rotated_img = img.rotate(angle)
    if np.random.rand() > 0.5:
        flipped_img = rotated_img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        flipped_img = rotated_img.transpose(Image.FLIP_TOP_BOTTOM)
    brightness = ImageEnhance.Brightness(flipped_img).enhance(np.random.uniform(0.8, 1.2))
    contrast = ImageEnhance.Contrast(brightness).enhance(np.random.uniform(0.8, 1.2))
    saturation = ImageEnhance.Color(contrast).enhance(np.random.uniform(0.8, 1.2))
    return saturation

# Process Positive images (double it: 1 original + 1 augmented)
positive_files = [f for f in os.listdir(positive_dir) if f.endswith(".jpg")]
positive_count = len(positive_files)

print("Processing Positive images...")
for i, img_file in enumerate(tqdm(positive_files, desc="Positive")):
    img_path = os.path.join(positive_dir, img_file)
    img = Image.open(img_path)
    img.save(os.path.join(pro_positive_dir, f"{i+1}.jpg"))  # Original
    aug_img = augment_image(img)
    aug_img.save(os.path.join(pro_positive_dir, f"{positive_count + i + 1}.jpg"))  # 1 augmented

# Process Negative images (triple it: 1 original + 2 augmented)
negative_files = [f for f in os.listdir(negative_dir) if f.endswith(".jpg")]
negative_count = len(negative_files)

print("Processing Negative images...")
for i, img_file in enumerate(tqdm(negative_files, desc="Negative")):
    img_path = os.path.join(negative_dir, img_file)
    img = Image.open(img_path)
    img.save(os.path.join(pro_negative_dir, f"N_{i+1}.jpg"))  # Original
    # First augmented image
    aug_img1 = augment_image(img)
    aug_img1.save(os.path.join(pro_negative_dir, f"N_{negative_count + i + 1}.jpg"))
    # Second augmented image
    aug_img2 = augment_image(img)
    aug_img2.save(os.path.join(pro_negative_dir, f"N_{2 * negative_count + i + 1}.jpg"))

print("All set! Check Pro_Data—Negative’s boosted and progress was tracked!")