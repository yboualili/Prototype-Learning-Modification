import ast
import os

def is_valid_dataset_structure(dataset_path):
    """
    Validates that the dataset structure matches:
    <name_dataset>/
        train/
            <cat_1>/
            <cat_n>/
        test/
            <cat_1>/
            <cat_n>/
    """
    # Expected subdirectories at the top level
    expected_dirs = ['train', 'test']
    for subdir in expected_dirs:
        subdir_path = os.path.join(dataset_path, subdir)
        if not os.path.isdir(subdir_path):
            print(f"Expected directory '{subdir}' not found at path: {subdir_path}")
            return False
        
        # Check if subdirectories (categories) exist within 'train' and 'test'
        categories = os.listdir(subdir_path)
        if not categories:
            print(f"No categories found in '{subdir}' directory.")
            return False
        for category in categories:
            category_path = os.path.join(subdir_path, category)
            if not os.path.isdir(category_path):
                print(f"Expected category directory not found: {category_path}")
                return False
    
    return True

def get_all_experiments():
    saved_models_path = os.path.join("ProtoPNet", 'saved_models')
    
    # Ensure the directory exists
    if not os.path.exists(saved_models_path):
        return []
    
    # List all directories (experiment names)
    experiments = [d for d in os.listdir(saved_models_path) if os.path.isdir(os.path.join(saved_models_path, d))]
    
    return experiments

import ast

def get_last_training_settings(settings_path):
    settings = {}
    with open(settings_path, 'r') as f:
        file_content = f.read()
        settings_ast = ast.parse(file_content)

    for node in settings_ast.body:
        if isinstance(node, ast.Assign):
            var_name = node.targets[0].id
            if var_name in [
                'img_size', 'num_classes', 'experiment_run',
                'train_batch_size', 'test_batch_size', 'train_push_batch_size',
                'num_train_epochs', 'num_warm_epochs', 'push_start'
            ]:
                settings[var_name] = ast.literal_eval(node.value)
    
    return settings

def validate_settings(settings):
    errors = []

    # Check if img_size is a positive integer
    if settings['img_size'] <= 0:
        errors.append("Image Size must be a positive integer.")

    # Check if train_batch_size is a positive integer
    if settings['train_batch_size'] <= 0:
        errors.append("Train Batch Size must be a positive integer.")

    # Check if test_batch_size is a positive integer
    if settings['test_batch_size'] <= 0:
        errors.append("Test Batch Size must be a positive integer.")

    # Check if train_push_batch_size is a positive integer
    if settings['train_push_batch_size'] <= 0:
        errors.append("Train Push Batch Size must be a positive integer.")

    # Check if num_train_epochs is a positive integer
    if settings['num_train_epochs'] <= 0:
        errors.append("Number of Train Epochs must be a positive integer.")

    # Check if num_warm_epochs is a positive integer and <= num_train_epochs
    if settings['num_warm_epochs'] <= 0 or settings['num_warm_epochs'] > settings['num_train_epochs']:
        errors.append("Number of Warm Epochs must be a positive integer and less than or equal to Number of Train Epochs.")

    # Check if push_start is a positive integer and <= num_train_epochs
    if settings['push_start'] <= 0 or settings['push_start'] > settings['num_train_epochs']:
        errors.append("Push Start Epoch must be a positive integer and less than or equal to Number of Train Epochs.")

    # Check if experiment_run is not empty
    if not settings['experiment_run']:
        errors.append("Experiment Run name cannot be empty.")

    return errors

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import importlib.util
import sys

from ProtoPNet.preprocess import mean, std

def perform_prediction(experiment_name, model_name, image_file):
    # Dynamically import settings.py from the experiment folder
    settings_path = os.path.join('ProtoPNet', 'saved_models', experiment_name, 'settings.py') # changed to ProtoPNet for usage for app.py
    
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)

    # Extract required information from settings
    data_path = settings.data_path
    data_path = data_path.split("../")[-1]

    img_size = settings.img_size

    # Get class labels by reading directory names in the train folder
    train_dir = os.path.join(data_path, 'train')
    class_labels = sorted(os.listdir(train_dir))

    # Dynamically import model.py from the ProtoPNet directory
    model_path_dir = os.path.join('ProtoPNet')
    sys.path.append(os.path.abspath(model_path_dir))
    import model  # Now we can import the model module

    # Load the selected ProtoPNet model
    model_path = os.path.join('ProtoPNet', 'saved_models', experiment_name, model_name)
    print(model_path)
    ppnet = torch.load(model_path)
    ppnet.cuda()
    ppnet.eval()

    # Load and preprocess the uploaded image
    image = Image.open(image_file).convert('RGB')
    image_transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    image_tensor = image_transform(image).unsqueeze(0).cuda()

    # Forward pass through the model
    logits, min_distances2, distances, conv_features = ppnet(image_tensor)
    cls = np.argmax(logits.detach().cpu().numpy())
    predicted_class_name = class_labels[cls]

    # Handle plural forms: 'cats' -> 'cat' and 'dogs' -> 'dog'
    if predicted_class_name in ["cats", "dogs"]:
        predicted_class_name = predicted_class_name[:-1]  # Removes the trailing 's'

    # Continue with prototype activation and image annotation as before
    protoL_input_ = np.copy(conv_features.detach().cpu().numpy())
    proto_dist_ = np.copy(distances.detach().cpu().numpy())

    # Extract the class identities of the prototypes
    proto_classes = [torch.argmax(ppnet.prototype_class_identity[j]).item() for j in range(8)]

    # Store the positions of prototypes that are relevant to the classified class
    prototype_positions = []
    for img_idx in range(proto_dist_.shape[0]):
        for rel_prototype in range(proto_dist_.shape[1]):
            if proto_classes[rel_prototype] == cls:
                proto_dist_img_j = proto_dist_[img_idx, rel_prototype, :, :]
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + ppnet.epsilon))
                upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
                max_value = np.max(upsampled_act_img_j)
                max_position = np.unravel_index(np.argmax(upsampled_act_img_j), upsampled_act_img_j.shape)
                prototype_positions.append([rel_prototype, max_value, max_position])

    # Convert image for OpenCV
    opencv_image = cv2.cvtColor(np.array(image.resize((img_size, img_size))), cv2.COLOR_RGB2BGR)
    min_distances2 = ppnet.distance_2_similarity(min_distances2)
    dists = min_distances2.detach().cpu().numpy()
    
    # Annotate the image as before
    colors = [(255, 0, 0), (0, 128, 0), (0, 0, 255), (255, 165, 0)]
    extension_width = 150
    extended_width = opencv_image.shape[1] + extension_width
    extended_image = np.ones((opencv_image.shape[0], extended_width, 3), dtype=np.uint8) * 255
    extended_image[:, :opencv_image.shape[1], :] = opencv_image
    dot_radius = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    prototype_positions = sorted(prototype_positions, key=lambda x: x[1], reverse=True)

    for ind, i in enumerate(prototype_positions):
        rel_prototype, _, (y, x) = i
        color = colors[ind]
        cv2.circle(extended_image, (x, y), 5, color, -1)
        x_offset = opencv_image.shape[1] + 30
        y_offset = 30 + ind * 40
        cv2.circle(extended_image, (x_offset, y_offset), dot_radius, color, -1)
        distance_text = f"{i[1]:.4f}"
        cv2.putText(extended_image, distance_text, (x_offset + dot_radius + 5, y_offset + 5), font, 0.6, color, 1, cv2.LINE_AA)

    final_y_offset = y_offset + 50
    # cv2.putText(extended_image, f"Pred: {predicted_class_name}", (x_offset, final_y_offset), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Convert the annotated image to an appropriate format for web display
    _, buffer = cv2.imencode('.jpg', extended_image)
    image_bytes = buffer.tobytes()

    return image_bytes, predicted_class_name

    # # Save the extended and annotated image
    # cv2.imwrite(f'annotated_image.jpg', extended_image)