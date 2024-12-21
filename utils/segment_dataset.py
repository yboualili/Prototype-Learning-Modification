import argparse
from PIL import Image
from lang_sam import LangSAM  # Make sure this is the correct import for your LangSAM model
import numpy as np
import os
from tqdm import tqdm

def save_mask(mask_np, filename):
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_image.save(filename)

def main(args):
    # Initialize the model
    model = LangSAM()

    # Get a list of all image paths
    list_of_img_paths = []
    for data_path in args.data_paths:
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    list_of_img_paths.append(os.path.join(root, file))

    # Iterate over each image path
    for img_path in tqdm(list_of_img_paths, desc="Processing Images"):
        # Open Image
        image_pil = Image.open(img_path).convert("RGB")

        # Use text prompt from arguments or derive it from the image path
        text_prompt = args.text_prompt if args.text_prompt else img_path.split(os.path.sep)[-2]

        masks, boxes, phrases, logits = model.predict(image_pil, 
                                                      text_prompt,
                                                      box_threshold=args.box_threshold, 
                                                      text_threshold=args.text_threshold
                                                      )

        # Convert masks to numpy arrays
        masks_np = [mask.squeeze().cpu().numpy() for mask in masks]

        # Sum the masks to get the overlapping mask
        combined_mask = np.sum(masks_np, axis=0)
        # Ensure the combined mask is binary (0 or 1)
        combined_mask = np.clip(combined_mask, 0, 1)

        # Prepare the output paths
        filename = os.path.splitext(os.path.basename(img_path))[0]
        relative_path = os.path.relpath(img_path, start=os.path.dirname(args.data_paths[0]))
        masked_img_dir = os.path.join(args.masked_path, os.path.dirname(relative_path))

        # Ensure the directory exists
        os.makedirs(masked_img_dir, exist_ok=True)

        # Create file paths
        mask_img_path = os.path.join(masked_img_dir, f"{filename}_mask.png")
        mask_array_path = os.path.join(masked_img_dir, f"{filename}_mask.npy")

        # Save the combined mask image if save_mask is True
        if args.save_mask:
            save_mask(combined_mask, mask_img_path)

        # Save the mask array
        np.save(mask_array_path, combined_mask)

        print(f"{filename} saved to {masked_img_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment images in dataset")
    parser.add_argument('--data_paths', nargs='+', default=['datasets/cats_dogs/test', 'datasets/cats_dogs/train'], help='List of data paths')
    parser.add_argument('--masked_path', default='datasets/cats_dogs_masked', help='Path to save masked images')
    parser.add_argument('--text_prompt', type=str, default=None, help='Text prompt for segmentation')
    parser.add_argument('--box_threshold', type=float, default=0.3, help='Box threshold for segmentation')
    parser.add_argument('--text_threshold', type=float, default=0.25, help='Text threshold for segmentation')
    parser.add_argument('--save_mask', action='store_true', help='Flag to save mask images')

    args = parser.parse_args()
    main(args)
