import os
import random
import numpy as np
from PIL import Image, ImageDraw

def find_valid_positions(mask, rectangle_size, buffer):
    mask_h, mask_w = mask.shape
    rect_w, rect_h = rectangle_size

    # Create a binary mask where valid positions are marked as True
    valid_mask = np.zeros_like(mask, dtype=bool)
    
    for y in range(mask_h):
        for x in range(mask_w):
            if mask[y, x] == 0:
                top = max(0, y - buffer)
                bottom = min(mask_h, y + buffer + rect_h)
                left = max(0, x - buffer)
                right = min(mask_w, x + buffer + rect_w)

                # Check if the buffer zone around (x, y) is all zeros
                if np.all(mask[top:bottom, left:right] == 0):
                    valid_mask[y, x] = True

    return valid_mask

def get_random_position(valid_mask, rectangle_size):
    rect_w, rect_h = rectangle_size
    valid_positions = np.argwhere(valid_mask)

    if len(valid_positions) == 0:
        return None

    while len(valid_positions) > 0:
        index = random.randint(0, len(valid_positions) - 1)
        y, x = valid_positions[index]

        if y + rect_h <= valid_mask.shape[0] and x + rect_w <= valid_mask.shape[1]:
            return x, y

        valid_positions = np.delete(valid_positions, index, axis=0)
    
    return None

def apply_watermark(image, mask, rectangle_size=(50, 50), rectangle_color=(57, 255, 20), buffer=20):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    mask = np.array(mask)
    valid_mask = find_valid_positions(mask, rectangle_size, buffer)
    position = get_random_position(valid_mask, rectangle_size)

    if position:
        x, y = position
        draw.rectangle([x, y, x + rectangle_size[0], y + rectangle_size[1]], fill=rectangle_color)
    
    return img

def process_images(source_dir, target_dir, rectangle_size=(50, 50), rectangle_color=(57, 255, 20), buffer=20):
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
        
        img_with_watermark = apply_watermark(img, mask, rectangle_size, rectangle_color, buffer)
        img_with_watermark.save(os.path.join(target_dir, img_name))

# Define source and target directories
source_dir = 'datasets/cats_dogs_normal/train/cats'
target_dir = 'datasets/corrupt_imgs/cats_rectangle_vsm2'

# Test different buffer values
buffers = [10, 20, 30, 40]  # Different buffer values to test

for buffer in buffers:
    print(f"Processing with buffer: {buffer}")
    target_dir = f'datasets/corrupt_imgs/cats_rectangle_sm_buffer_{buffer}'
    process_images(source_dir, target_dir, rectangle_size=(20, 10), rectangle_color=(57, 255, 20), buffer=buffer)