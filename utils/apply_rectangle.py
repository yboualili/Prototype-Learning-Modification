import os
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt


def draw_rectangle(image_folder, mask_folder, output_folder, initial_min_distance=20, rectangle_size_ratio=0.1,
                   max_attempts=1000):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get list of image files and mask files
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('npy')])

    for img_file in image_files:
        # Load image
        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path)

        # Load mask
        mask_file = img_file.split('.')[0] + '_mask.npy'
        mask_path = os.path.join(mask_folder, mask_file)
        mask = np.load(mask_path)

        # Get image and mask dimensions
        img_height, img_width = img.shape[:2]
        mask_height, mask_width = mask.shape

        assert img_height == mask_height and img_width == mask_width, "Image and mask dimensions do not match"

        # Calculate the rectangle size as 10% of the image width
        rectangle_size = int(rectangle_size_ratio * img_width)

        # Compute the distance transform of the mask
        distance_transform = distance_transform_edt(mask == 0)

        min_distance = initial_min_distance
        valid_position_found = False

        while not valid_position_found and min_distance >= 0:
            for _ in range(max_attempts):  # Try up to max_attempts random positions
                top_left_x = np.random.randint(0, img_width - rectangle_size)
                top_left_y = np.random.randint(0, img_height - rectangle_size)
                bottom_right_x = top_left_x + rectangle_size
                bottom_right_y = top_left_y + rectangle_size

                # Extract the region of interest from the distance transform
                distance_roi = distance_transform[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

                # Check if the minimum distance in the ROI is at least the current minimum distance
                if np.all(distance_roi >= min_distance):
                    valid_position_found = True
                    break

            if not valid_position_found:
                # Decrease the minimum distance requirement and try again
                min_distance -= 1

        if not valid_position_found:
            print(
                f"Could not find a valid position for the rectangle in {img_file}, but will proceed with the best attempt.")

        # Draw the green rectangle on the image
        cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), -1)

        # Save the modified image
        output_path = os.path.join(output_folder, img_file)
        cv2.imwrite(output_path, img)
        print(f"Saved modified image to {output_path}")


# Example usage
image_folder = r'ProtoPNet\datasets\test\dogs'
mask_folder = r'ProtoPNet\datasets\cats_dogs_masked\test\dogs'
output_folder = r'ProtoPNet\corrupt_dogs'

draw_rectangle(image_folder, mask_folder, output_folder)