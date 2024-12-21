import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2

from preprocess import mean, std

img_size = 224

# Experiment and model paths
experiment_path = 'saved_models/heat_long'
settings_path = f'{experiment_path}/settings.py'

# Load the settings file to get the data path
data_path = None
with open(settings_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line.startswith("data_path"):
            data_path = line.split('=')[1].strip().strip("'\"")

if data_path is None:
    raise ValueError("Data path not found in settings.py.")

# Get class labels by reading directory names in the train folder
train_dir = os.path.join(data_path, 'train')
class_labels = sorted(os.listdir(train_dir))

# Load the saved ProtoPNet model from the specified path
model_path = f'{experiment_path}/20nopush0.8188.pth'
ppnet = torch.load(model_path)
ppnet.cuda()
ppnet.eval()

# Specify the image path
img_path = "../datasets/corrupted_dataset/test/cats/cat_1.jpg"

# Open the image and convert it to RGB format
image = Image.open(img_path).convert('RGB')

# Define the transformations to resize the image, convert it to a tensor, and normalize it
image_transform = transforms.Compose([
    transforms.Resize(size=(img_size, img_size)),  # Resize the image to the required input size
    transforms.ToTensor(),                        # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=mean, std=std),     # Normalize the tensor using the dataset mean and std
])

# Apply the transformations to the image
image_tensor = image_transform(image).unsqueeze(0).cuda()

# Forward pass through the model to get logits, distances, and convolutional features
logits, min_distances2, distances, conv_features = ppnet(image_tensor)

# Determine the predicted class
cls = np.argmax(logits.detach().cpu().numpy())
predicted_class_name = class_labels[cls]

# Save a copy of the conv_features and distances for further processing
protoL_input_ = np.copy(conv_features.detach().cpu().numpy())
proto_dist_ = np.copy(distances.detach().cpu().numpy())

# Extract the class identities of the prototypes
proto_classes = [torch.argmax(ppnet.prototype_class_identity[j]).item() for j in range(8)]
print(proto_classes)

# Identify prototype positions relevant to the predicted class
prototype_positions = []
for img_idx in range(proto_dist_.shape[0]):
    for rel_prototype in range(proto_dist_.shape[1]):
        if proto_classes[rel_prototype] == cls:
            proto_dist_img_j = proto_dist_[img_idx, rel_prototype, :, :]  # Extract the distance map for the prototype
            proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + ppnet.epsilon))  # Compute activation map

            # Upsample the activation map to the original image size using high-quality interpolation
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(224, 224), interpolation=cv2.INTER_LANCZOS4)

            # Find the maximum activation value and its position
            max_value = np.max(upsampled_act_img_j)
            max_position = np.unravel_index(np.argmax(upsampled_act_img_j), upsampled_act_img_j.shape)
            prototype_positions.append([rel_prototype, max_value, max_position])

# Convert the image to a format that OpenCV can use (BGR format)
opencv_image = cv2.cvtColor(np.array(image.resize((img_size, img_size))), cv2.COLOR_RGB2BGR)

# Convert distances to similarity scores
min_distances2 = ppnet.distance_2_similarity(min_distances2)
dists = min_distances2.detach().cpu().numpy()

# Define colors for marking prototypes on the image
colors = [(255, 0, 0), (0, 128, 0), (0, 0, 255), (255, 165, 0)]

# Extend the image canvas to the right for annotations
extension_width = 150  # Extend the width more to make space for detailed annotations
extended_width = opencv_image.shape[1] + extension_width
extended_image = np.ones((opencv_image.shape[0], extended_width, 3), dtype=np.uint8) * 255  # White background
extended_image[:, :opencv_image.shape[1], :] = opencv_image

dot_radius = 5  # Slightly reduce the circle size
font = cv2.FONT_HERSHEY_SIMPLEX # Use a more readable font

# Sort prototypes by their activation value in descending order
prototype_positions = sorted(prototype_positions, key=lambda x: x[1], reverse=True)

# Function to draw text with a shadow for better visibility
def draw_text_with_shadow(image, text, position, font, font_scale, color, thickness, shadow_color=(0, 0, 0)):
    x, y = position
    cv2.putText(image, text, (x + 1, y + 1), font, font_scale, shadow_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

# Draw prototype points on the image and annotate them
for ind, i in enumerate(prototype_positions):
    rel_prototype, _, (y, x) = i
    color = colors[ind % len(colors)]

    # Draw circles with better visibility (including a border for clarity)
    cv2.circle(extended_image, (x, y), dot_radius + 2, (0, 0, 0), -1)  # Black border
    cv2.circle(extended_image, (x, y), dot_radius, color, -1)          # Colored inner circle

    # Offset for the annotation text
    x_offset = opencv_image.shape[1] + 30
    y_offset = 30 + ind * 40  # Increase the spacing between annotations
    
    # Draw the color dot for annotation
    cv2.circle(extended_image, (x_offset, y_offset), dot_radius, color, -1)
    
    # Draw the distance value next to the dot
    distance_text = f"{i[1]:.4f}"
    draw_text_with_shadow(extended_image, distance_text, (x_offset + dot_radius + 5, y_offset + 5), font, 0.7, color, 1)

# Annotate the predicted class name below the logits and distances
final_y_offset = y_offset + 50  # Offset to place the text below the previous annotations
draw_text_with_shadow(extended_image, f"Pred: {predicted_class_name}", (x_offset, final_y_offset), font, 0.6, (0, 0, 0), 1)

# Save the extended and annotated image
cv2.imwrite(f'preds/annotated_image_2.jpg', extended_image)
