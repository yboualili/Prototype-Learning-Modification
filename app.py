from flask import Flask, abort, send_file, send_from_directory, render_template, request, redirect, url_for, flash, Response, session, jsonify
import cv2
import numpy as np
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import shutil
import subprocess
import torch
torch.cuda.empty_cache()
import gc
import threading
import zipfile
from PIL import Image
from jinja2 import Template
from werkzeug.utils import secure_filename
from segment_anything import SamPredictor, sam_model_registry

from lang_sam import LangSAM
from helpers import is_valid_dataset_structure, get_all_experiments, get_last_training_settings, validate_settings, perform_prediction

# from ProtoPNet.predict_and_visualize import perform_prediction

app = Flask(__name__)
app.config['DATASET_FOLDER'] = 'datasets'
app.config['PROTO_PNET_FOLDER'] = 'ProtoPNet'
app.config['MASKS_FOLDER'] = 'datasets_masks'
app.config['TEMP_FOLDER'] = 'images_masks'  # Temporary folder for masked images

app.secret_key = 'your_secret_key'

# Ensure the dataset folder exists
os.makedirs(app.config['DATASET_FOLDER'], exist_ok=True)
# Ensure the MASKS_FOLDER directory exists
os.makedirs(app.config['MASKS_FOLDER'], exist_ok=True)
# Ensure TEMP_FOLDER exists
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)


# Route for Home Page
@app.route('/')
def home():
    return render_template('home.html')

###### DATASET LOGIC

@app.route('/datasets', methods=['GET', 'POST'])
def dataset_page():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            flash('No file selected for uploading', 'error')
            return redirect(request.url)
        
        dataset_name = secure_filename(uploaded_file.filename).split('.')[0]
        dataset_temp_path = os.path.join(app.config['DATASET_FOLDER'], 'temp', dataset_name)
        dataset_final_path = os.path.join(app.config['DATASET_FOLDER'], dataset_name)
        
        # Save the uploaded file temporarily
        temp_zip_path = os.path.join(dataset_temp_path, uploaded_file.filename)
        os.makedirs(dataset_temp_path, exist_ok=True)
        uploaded_file.save(temp_zip_path)
        
        try:
            # Extract the zip file
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_temp_path)
            
            # Determine the extracted root directory
            extracted_root = os.path.join(dataset_temp_path, dataset_name)
            if not os.path.exists(extracted_root):
                extracted_root = os.path.join(dataset_temp_path, os.listdir(dataset_temp_path)[0])
            
            # Validate dataset structure
            if not is_valid_dataset_structure(extracted_root):
                flash('Invalid dataset structure. Please follow the specified format.', 'error')
                return redirect(request.url)
            
            # Move the dataset to the final location
            shutil.move(extracted_root, dataset_final_path)
            flash(f'Dataset {dataset_name} uploaded successfully!', 'success')
        except zipfile.BadZipFile:
            flash('Invalid zip file. Please upload a valid dataset.', 'error')
        finally:
            # Cleanup: remove the temp folder and its contents
            temp_dir = os.path.join(app.config['DATASET_FOLDER'], 'temp')
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        return redirect(url_for('dataset_page'))
    
    # Handle GET request to display datasets
    datasets = []
    for dataset in os.listdir(app.config['DATASET_FOLDER']):
        dataset_path = os.path.join(app.config['DATASET_FOLDER'], dataset)
        if os.path.isdir(dataset_path):
            train_dir = os.path.join(dataset_path, "train")
            test_dir = os.path.join(dataset_path, "test")
            
            num_classes = len(os.listdir(train_dir))
            num_train_files = sum([len(files) for r, d, files in os.walk(train_dir)])
            num_test_files = sum([len(files) for r, d, files in os.walk(test_dir)])
            
            datasets.append({
                'name': dataset,
                'num_classes': num_classes,
                'num_train_files': num_train_files,
                'num_test_files': num_test_files
            })
    return render_template('datasets.html', datasets=datasets)


@app.route('/delete_dataset', methods=['POST'])
def delete_dataset():
    dataset_name = request.form.get('dataset_name')
    dataset_path = os.path.join(app.config['DATASET_FOLDER'], dataset_name)
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
        flash(f'Dataset {dataset_name} deleted successfully!', 'success')
    else:
        flash(f'Dataset {dataset_name} not found!', 'error')
    return redirect(url_for('dataset_page'))

##### ProtoPNet Logic

training_process = None

@app.route('/train', methods=['GET', 'POST'])
def train_page():
    # Path to datasets and masks
    datasets = os.listdir(app.config['DATASET_FOLDER'])
    
    # Prepare to hold the mask information
    dataset_info = []

    for dataset in datasets:
        dataset_path = os.path.join(app.config['DATASET_FOLDER'], dataset)
        mask_path = os.path.join(app.config['MASKS_FOLDER'], dataset)
        
        # Count the number of image files in train and test folders
        num_train_files = sum([len(files) for r, d, files in os.walk(os.path.join(dataset_path, "train")) if any(f.endswith(('.jpg', '.png')) for f in files)])
        num_test_files = sum([len(files) for r, d, files in os.walk(os.path.join(dataset_path, "test")) if any(f.endswith(('.jpg', '.png')) for f in files)])
        num_files = num_train_files + num_test_files

        # Initialize the number of masks
        num_train_masks = 0
        num_test_masks = 0

        if os.path.exists(mask_path):
            # Check masks corresponding to train images
            train_image_paths = [os.path.splitext(f)[0] for r, d, files in os.walk(os.path.join(dataset_path, "train")) for f in files if f.endswith(('.jpg', '.png'))]
            num_train_masks = sum([1 for img_name in train_image_paths if os.path.exists(os.path.join(mask_path, f"{img_name}_mask.npy"))])

            # Check masks corresponding to test images
            test_image_paths = [os.path.splitext(f)[0] for r, d, files in os.walk(os.path.join(dataset_path, "test")) for f in files if f.endswith(('.jpg', '.png'))]
            num_test_masks = sum([1 for img_name in test_image_paths if os.path.exists(os.path.join(mask_path, f"{img_name}_mask.npy"))])

        num_masks = num_train_masks + num_test_masks
        
        # Calculate the percentage
        train_percentage = round((num_train_masks / num_train_files) * 100, 2) if num_train_files > 0 else 0
        test_percentage = round((num_test_masks / num_test_files) * 100, 2) if num_test_files > 0 else 0
        total_percentage = round((num_masks / num_files) * 100, 2) if num_files > 0 else 0

        dataset_info.append({
            'name': dataset,
            'num_files': num_files,
            'num_train_files': num_train_files,
            'num_test_files': num_test_files,
            'num_train_masks': num_train_masks,
            'num_test_masks': num_test_masks,
            'num_masks': num_masks,
            'train_percentage': train_percentage,
            'test_percentage': test_percentage,
            'total_percentage': total_percentage
        })

    # Path to the current settings.py
    settings_path = os.path.join(app.config['PROTO_PNET_FOLDER'], 'settings.py')

    # Fetch the last training settings
    last_settings = get_last_training_settings(settings_path)

    return render_template('config.html', dataset_info=dataset_info, last_settings=last_settings)

@app.route('/submit_settings', methods=['POST'])
def submit_settings():
    # Extract form data
    dataset_name = request.form.get('data_path')
    selected_mask_percentage = request.form.get('selected_mask_percentage')
    selected_mask_percentage = int(selected_mask_percentage) if selected_mask_percentage else None

    current_settings = {
        'img_size': int(request.form.get('img_size')), 
        'experiment_run': request.form.get('experiment_run'), 
        'data_path': '../datasets/' + dataset_name + '/', 
        'train_batch_size': int(request.form.get('train_batch_size')),
        'test_batch_size': int(request.form.get('test_batch_size')),
        'train_push_batch_size': int(request.form.get('train_push_batch_size')), 
        'num_train_epochs': int(request.form.get('num_train_epochs')), 
        'num_warm_epochs': int(request.form.get('num_warm_epochs')), 
        'push_start': int(request.form.get('push_start')), 
        'tr_mode': request.form.get('tr_mode'),
        'selected_mask_percentage': selected_mask_percentage  # Use new name here
    }
    
    # Validate the settings
    validation_errors = validate_settings(current_settings)
    if validation_errors:
        for error in validation_errors:
            flash(error, 'error')
        return redirect(url_for('train_page'))  # Get redirected back to the config page

    # Construct the full path to the training directory
    train_path = os.path.join('datasets', dataset_name, 'train')
    test_path = os.path.join('datasets', dataset_name, 'test')

    # Calculate the number of classes based on the data path
    class_dirs = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    current_settings['num_classes'] = len(class_dirs)

    # Calculate the total number of image files
    num_files = sum([len(files) for r, d, files in os.walk(train_path)])
    num_files += sum([len(files) for r, d, files in os.walk(test_path)])

    # Calculate the number of mask files only if in "mask" mode
    if current_settings['tr_mode'] == 'mask':
        mask_path = os.path.join(app.config['MASKS_FOLDER'], dataset_name)
        num_masks = len(os.listdir(mask_path)) if os.path.exists(mask_path) else 0

        # Calculate the percentage of masks
        mask_percentage = round((num_masks / num_files) * 100, 2) if num_files > 0 else 0
        current_settings['mask_percentage'] = mask_percentage

    # Check if experiment folder already exists
    experiment_folder = os.path.join(app.config['PROTO_PNET_FOLDER'], 'saved_models', current_settings['experiment_run'])
    if os.path.exists(experiment_folder):
        flash(f"An experiment with the name '{current_settings['experiment_run']}' already exists.", 'error')
        return redirect(url_for('train_page'))  # Get redirected back to the config page

    # Save settings in session
    session['current_settings'] = current_settings

    # Load settings_template.py
    template_path = os.path.join(app.root_path, 'settings_template.py')
    with open(template_path) as f:
        template_content = f.read()
    
    # Render settings.py from template
    settings_content = Template(template_content).render(**current_settings)
    
    # Write the rendered content to ProtoPNet/settings.py
    settings_file_path = os.path.join(app.config['PROTO_PNET_FOLDER'], 'settings.py')
    with open(settings_file_path, 'w') as f:
        f.write(settings_content)

    # Create the experiment folder
    os.makedirs(experiment_folder, exist_ok=True)

    # Save the settings as settings_submitted.txt in the experiment folder
    settings_txt_path = os.path.join(experiment_folder, 'settings_submitted.txt')
    with open(settings_txt_path, 'w') as f:
        for key, value in current_settings.items():
            f.write(f'{key}: {value}\n')

    # Pass experiment_name to the training thread
    experiment_name = current_settings['experiment_run']
    training_thread = threading.Thread(target=start_training_process, args=(experiment_name,))
    training_thread.start()

    return redirect(url_for('training_status'))


training_complete_flag = {}

def start_training_process(experiment_name):
    global training_process
    global training_complete_flag

    # Construct the absolute path to main.py
    main_script_path = os.path.join(app.config['PROTO_PNET_FOLDER'], 'main.py')
    main_script_path = os.path.abspath(main_script_path)  # Ensure absolute path

    # Check if the main.py file exists
    if not os.path.isfile(main_script_path):
        print(f"Error: {main_script_path} does not exist.")
        return
    
    # Start the training process
    training_process = subprocess.Popen(['python', main_script_path], cwd=app.config['PROTO_PNET_FOLDER'])
    
    # Wait for the process to complete
    training_process.wait()

    # Set the completion flag for the current experiment
    training_complete_flag[experiment_name] = True
    print(f"Training process for {experiment_name} has finished.")


@app.route('/training_status')
def training_status():
    current_settings = session.get('current_settings')
    
    if not current_settings:
        flash('No training settings found. Please start a new training session.', 'error')
        return redirect(url_for('train_page'))

    return render_template('status.html', settings=current_settings)

@app.route('/check_training_complete')
def check_training_complete():
    experiment_name = session.get('current_settings', {}).get('experiment_run')
    if not experiment_name:
        return jsonify(complete=False)

    is_complete = training_complete_flag.get(experiment_name, False)
    return jsonify(complete=is_complete)

@app.route('/logs')
def stream_logs():
    current_settings = session.get('current_settings')
    if not current_settings:
        return "No training logs available.", 404

    log_dir = os.path.join(app.config['PROTO_PNET_FOLDER'], 'saved_models', current_settings['experiment_run'])
    log_file_path = os.path.join(log_dir, 'train.log')
    
    # Check if the log file exists
    if not os.path.exists(log_file_path):
        return "Training has not started or log file is unavailable.", 404
    
    def generate():
        with open(log_file_path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                yield line
    return Response(generate(), mimetype='text/plain')

@app.route('/latest_image')
def latest_image():
    current_settings = session.get('current_settings', {})
    experiment_run = current_settings.get('experiment_run')
    
    if experiment_run:
        proto_vis_path = os.path.join(app.config['PROTO_PNET_FOLDER'], 'saved_models', experiment_run, 'proto_vis')
        
        if os.path.exists(proto_vis_path):
            # Get all image files in the proto_vis directory
            image_files = [
                os.path.join(proto_vis_path, file) 
                for file in os.listdir(proto_vis_path) 
                if file.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            
            if image_files:
                # Sort images by modification time (newest first)
                latest_image_path = max(image_files, key=os.path.getmtime)
                relative_path = os.path.relpath(latest_image_path, app.config['PROTO_PNET_FOLDER'])
                # Convert path separators to forward slashes
                relative_path = relative_path.replace("\\", "/")
                # Ensure the URL path construction is correct
                return jsonify(image_url=url_for('serve_image', filename=relative_path))
    
    return jsonify(image_url=None)

@app.route('/stop_training')
def stop_training():
    global training_process
    if training_process:
        training_process.terminate()
        training_process = None
        flash('Training stopped.', 'success')
    else:
        flash('No training process found.', 'error')
    return redirect(url_for('train_page'))

##### Segmentation Logic
@app.route('/segment')
def segment_page():
    datasets = os.listdir(app.config['DATASET_FOLDER'])
    return render_template('segment.html', datasets=datasets)

@app.route('/get_structure/<dataset>')
def get_structure(dataset):
    dataset_path = os.path.join(app.config['DATASET_FOLDER'], dataset)
    structure = {}

    if os.path.exists(dataset_path):
        for folder in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder)
            if os.path.isdir(folder_path):
                structure[folder] = {}
                for class_name in os.listdir(folder_path):
                    class_path = os.path.join(folder_path, class_name)
                    if os.path.isdir(class_path):
                        images = [name for name in os.listdir(class_path)
                                  if name.lower().endswith(('png', 'jpg', 'jpeg'))]
                        structure[folder][class_name] = images

    return jsonify(structure=structure)

@app.route('/get_image_mask', methods=['GET'])
def get_image_mask():
    dataset = request.args.get('dataset')
    folder = request.args.get('folder')
    cls = request.args.get('class')
    image_name = request.args.get('image')

    # Construct the relative paths
    image_path = f"{dataset}/{folder}/{cls}/{image_name}"
    mask_name = os.path.splitext(image_name)[0] + '_mask.npy'
    # Mask is stored in datasets_masks/{dataset}/{mask_name}
    mask_path = f"{dataset}/{mask_name}"

    # Full filesystem paths
    full_image_path = os.path.join(app.config['DATASET_FOLDER'], image_path)
    full_mask_path = os.path.join(app.config['MASKS_FOLDER'], mask_path)

    # Check if the mask exists
    mask_exists = os.path.exists(full_mask_path)

    # Return the relative paths for the frontend to use
    return jsonify(image_path=image_path, mask_exists=mask_exists, mask_path=mask_path if mask_exists else None)

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images from the appropriate base directory."""
    # Determine if the request is for prototype visualizations or other images
    if 'proto_vis' in filename or 'train_report.png' in filename:
        # The request is for prototype visualizations or training metrics
        base_directory = os.path.join(app.config['PROTO_PNET_FOLDER'])
    else:
        # The request is for dataset images
        base_directory = app.config['DATASET_FOLDER']
    
    try:
        return send_from_directory(base_directory, filename)
    except FileNotFoundError:
        abort(404)  # File not found, return 404 error

@app.route('/masked_image', methods=['GET'])
def serve_masked_image():
    """ Generate and serve the image with mask overlay """
    dataset = request.args.get('dataset')
    folder = request.args.get('folder')
    cls = request.args.get('class')
    image_name = request.args.get('image')

    # Paths to original image and mask
    image_path = os.path.join(app.config['DATASET_FOLDER'], dataset, folder, cls, image_name)
    mask_name = os.path.splitext(image_name)[0] + '_mask.npy'
    mask_path = os.path.join(app.config['MASKS_FOLDER'], dataset, mask_name)

    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        return "Original image not found", 404

    # Load the mask
    if not os.path.exists(mask_path):
        return "Mask not found", 404
    mask = np.load(mask_path)

    # Convert mask to 3-channel image
    mask_image = np.dstack([mask.astype(np.uint8) * 255] * 3)

    # Create the masked image by overlaying the mask
    masked_image = cv2.addWeighted(image, 0.5, mask_image, 0.5, 0)

    # Save the masked image temporarily
    masked_image_path = os.path.join(app.config['TEMP_FOLDER'], f"masked_{image_name}")
    cv2.imwrite(masked_image_path, masked_image)

    return send_file(masked_image_path)

@app.route('/segment_image', methods=['POST'])
def segment_image():
    dataset = request.form['dataset']
    folder = request.form['folder']
    cls = request.form['class']
    image_name = request.form['image']

    # Construct the paths
    image_path = os.path.join(app.config['DATASET_FOLDER'], dataset, folder, cls, image_name)
    mask_name = os.path.splitext(image_name)[0] + '_mask.npy'

    mask_folder = os.path.join(app.config['MASKS_FOLDER'], dataset)
    mask_path = os.path.join(mask_folder, mask_name)

    # Ensure the mask directory exists
    os.makedirs(mask_folder, exist_ok=True)

    try:
        # Perform segmentation if mask doesn't exist
        if not os.path.exists(mask_path):
            # Load the image
            image_pil = Image.open(image_path).convert("RGB")

            # Use the class name as a text prompt
            text_prompt = cls

            # Initialize the models globally
            langsam_model = LangSAM()

            # Perform segmentation
            masks, boxes, phrases, logits = langsam_model.predict(image_pil, text_prompt)

            # Convert masks to numpy arrays
            masks_np = [mask.squeeze().cpu().numpy() for mask in masks]

            # Sum the masks to get the combined mask
            combined_mask = np.sum(masks_np, axis=0)
            combined_mask = np.clip(combined_mask, 0, 1)

            # Save the combined mask as a .npy file
            np.save(mask_path, combined_mask)

            # Delete variables and clear memory
            del langsam_model, masks, boxes, phrases, logits, masks_np, combined_mask
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

        return jsonify(success=True)
    
    except Exception as e:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        return jsonify(success=False, error=str(e))

# Constants for the SAM model checkpoint, model type, device, and datasets directory
SAM_CHECKPOINT = "models/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda"

@app.route('/perform_segmentation_with_points', methods=['POST'])
def perform_segmentation_with_points():
    data = request.json
    dataset = data['dataset']
    folder = data['folder']
    cls = data['cls']
    image_name = data['image']
    sam_point = data['sam_point']
    sam_label = data['sam_label']

    # Convert points and labels to numpy arrays
    sam_point = np.array(sam_point, dtype=np.float32)
    sam_label = np.array(sam_label, dtype=np.int32)

    # Debugging: Log the shape and contents of the inputs
    print(f"sam_point: {sam_point}, shape: {sam_point.shape}")
    print(f"sam_label: {sam_label}, shape: {sam_label.shape}")

    # Ensure that points and labels are not empty
    if sam_point.size == 0 or sam_label.size == 0:
        return jsonify(success=False, error="No points or labels provided.")

    # Construct the paths
    image_path = os.path.join(app.config['DATASET_FOLDER'], dataset, folder, cls, image_name)
    mask_name = os.path.splitext(image_name)[0] + '_mask.npy'
    mask_path = os.path.join(app.config['MASKS_FOLDER'], dataset, mask_name)

    try:
        # Load the image
        image_pil = cv2.imread(image_path)

        # image_pil = Image.open(image_path).convert("RGB")
        print(f"Image size: {image_pil.size}")  # Debugging: Image size

        sam_model = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam_model.to(DEVICE)

        predictor = SamPredictor(sam_model)
        predictor.set_image(image_pil)
        print("Image set for prediction.")  # Debugging

        # Perform segmentation with provided points and labels
        masks, scores, logits = predictor.predict(
            point_coords=sam_point,
            point_labels=sam_label,
            multimask_output=True
        )
        print("Prediction completed.")  # Debugging

        # Check if masks and scores are generated correctly
        if len(masks) == 0 or len(scores) == 0:
            return jsonify(success=False, error="Model did not return any masks.")

        # Convert masks to numpy arrays and get the best mask
        masks_np = [mask.squeeze() for mask in masks]
        best_mask_index = np.argmax(scores)
        best_mask = masks_np[best_mask_index]

        # Save the best mask as a .npy file
        np.save(mask_path, best_mask)
        print("Best mask saved.")  # Debugging

        # Delete variables and clear memory
        del sam_model, predictor, masks, scores, logits, masks_np, best_mask
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        return jsonify(success=True)
    except Exception as e:
        # Log the error for debugging
        print(f"Error during segmentation: {str(e)}")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        return jsonify(success=False, error=str(e))

#### Training Overview Logic

@app.route('/overview_page')
def overview_page():
    experiment_name = request.args.get('experiment_name')
    experiments = get_all_experiments()  # Implement this function to fetch all available experiments

    return render_template('training_overview.html', experiments=experiments, selected_experiment=experiment_name)

@app.route('/get_training_details/<experiment_name>')
def get_training_details(experiment_name):

    # Path to the respective files
    report_path = os.path.join(app.config['PROTO_PNET_FOLDER'], 'saved_models', experiment_name, 'train_report.txt')
    proto_vis_path = os.path.join(app.config['PROTO_PNET_FOLDER'], 'saved_models', experiment_name, 'proto_vis')
    settings_file_path = os.path.join(app.config['PROTO_PNET_FOLDER'], 'saved_models', experiment_name, 'settings_submitted.txt')
    
    settings_content = ""
    if os.path.isfile(settings_file_path):
        with open(settings_file_path, 'r') as f:
            settings_content = f.read()

    # Read classification report
    with open(report_path, 'r') as file:
        report = file.read()

    # Extract unique image names from the proto_vis folder
    image_names = set()
    for filename in os.listdir(proto_vis_path):
        filepath = os.path.join(proto_vis_path, filename)
        if os.path.isfile(filepath) and filename.endswith('.jpg'):  # Ensure it's a valid image file
            image_name = "_".join(filename.split("_")[:-1])
            image_names.add(image_name)
    image_names = sorted(list(image_names))

    return jsonify(report=report, image_names=image_names, settings=settings_content)

@app.route('/get_max_epoch/<experiment_name>/<image_name>')
def get_max_epoch(experiment_name, image_name):
    proto_vis_path = os.path.join(app.config['PROTO_PNET_FOLDER'], 'saved_models', experiment_name, 'proto_vis')
    max_epoch = 0
    for filename in os.listdir(proto_vis_path):
        if filename.startswith(image_name):
            epoch = int(filename.split("_")[-1].split(".")[0])
            if epoch > max_epoch:
                max_epoch = epoch

    return jsonify(max_epoch=max_epoch)

@app.route('/delete_experiment/<experiment_name>', methods=['POST'])
def delete_experiment(experiment_name):
    try:
        experiment_path = os.path.join(app.config['PROTO_PNET_FOLDER'], 'saved_models', experiment_name)
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
            flash(f'Experiment {experiment_name} deleted successfully.', 'success')
            return jsonify(success=True)
        else:
            flash(f'Experiment {experiment_name} not found.', 'error')
            return jsonify(success=False, error="Experiment not found")
    except Exception as e:
        flash(f'Failed to delete experiment {experiment_name}. Error: {str(e)}', 'error')
        return jsonify(success=False, error=str(e))
    
#### Logic for Prediction Page

app.config['UPLOAD_FOLDER'] = 'uploads'

# Function to get available experiments and models
@app.route('/models/<experiment>')
def get_models(experiment):
    experiments_path = os.path.join(app.config['PROTO_PNET_FOLDER'], 'saved_models', experiment)
    models = sorted([f for f in os.listdir(experiments_path) if f.endswith('.pth')])
    return jsonify(models)


import base64

@app.route('/prediction', methods=['GET', 'POST'])
def prediction_page():
    experiments_path = os.path.join(app.config['PROTO_PNET_FOLDER'], 'saved_models')
    experiments = sorted(os.listdir(experiments_path))

    selected_experiment = None
    selected_model = None
    models = []

    if request.method == 'POST':
        selected_experiment = request.form.get('experiment')
        selected_model = request.form.get('model')
        image_file = request.files['image']

        if selected_experiment:
            models_path = os.path.join(app.config['PROTO_PNET_FOLDER'], 'saved_models', selected_experiment)
            models = sorted([f for f in os.listdir(models_path) if f.endswith('.pth')])

        if selected_experiment and selected_model and image_file:
            # Perform prediction
            image_bytes, predicted_class_name = perform_prediction(selected_experiment, selected_model, image_file)
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            return render_template('prediction.html', 
                                   experiments=experiments, 
                                   models=models,
                                   selected_experiment=selected_experiment, 
                                   selected_model=selected_model, 
                                   image_data=image_base64, 
                                   predicted_class=predicted_class_name)

    return render_template('prediction.html', 
                           experiments=experiments, 
                           models=models, 
                           selected_experiment=selected_experiment, 
                           selected_model=selected_model)

if __name__ == '__main__':
    app.run(debug=True)
