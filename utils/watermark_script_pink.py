import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def apply_watermark(image, mask, watermark_text="WATERMARK", watermark_size=(50, 20)):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    mask = np.array(mask)
    mask_h, mask_w = mask.shape
    wm_w, wm_h = watermark_size

    # Create a list of valid positions
    valid_positions = []
    for y in range(mask_h - wm_h + 1):
        for x in range(mask_w - wm_w + 1):
            if np.all(mask[y:y+wm_h, x:x+wm_w] == 0):
                valid_positions.append((x, y))

    if valid_positions:
        # Choose a random valid position for the watermark
        x, y = random.choice(valid_positions)
        
        # Add watermark
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except IOError:
            font = ImageFont.load_default()
        
        draw.text((x, y), watermark_text, font=font, fill=(255, 20, 147, 255))  # Pink with no transparency

    return img

def process_images(source_dir, target_dir, watermark_text="WATERMARK", watermark_size=(50, 20)):
    mask_dir = source_dir.replace('cats_dogs/train/cats', 'cats_dogs_masked/train/cats')

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    for img_name in os.listdir(source_dir):
        img_path = os.path.join(source_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))  # Resize to 224x224
        
        mask_name = img_name.replace('.jpg', '_mask.npy')
        mask_path_full = os.path.join(mask_dir, mask_name)
        
        if os.path.exists(mask_path_full):
            mask = np.load(mask_path_full)
            mask = Image.fromarray(mask).resize((224, 224))  # Resize the mask as well
            mask = np.array(mask)
        else:
            mask = np.zeros((224, 224))  # Empty mask if not found
        
        img_with_watermark = apply_watermark(img, mask, watermark_text, watermark_size)
        img_with_watermark.save(os.path.join(target_dir, img_name))

# Define source and target directories
source_dir = 'datasets/cats_dogs_normal/train/cats'
target_dir = 'datasets/cats_corrupt_pink_2'

# Process images with specified watermark text
process_images(source_dir, target_dir, watermark_text="CAT")
