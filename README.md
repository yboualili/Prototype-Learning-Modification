# ProtoRRRNet Web Application

This project focuses on improving prototype learning with ProtoPnet. A common challenge in prototype learning is overfitting, where the model may focus excessively on the background, learning prototypes that are not relevant to the target objects. To address this issue, we introduce a segmentation step prior to the ProtoPnet model, which isolates the objects of interest. Additionally, we implement a novel loss function that encourages the model to learn prototypes that are specifically located on the object, as indicated by the segmentation mask. This approach helps guide the model to focus on the relevant features, enhancing its performance and generalization capabilities.

For evaluation, we introduced watermarks on one class in the training set, a class the model would normally overfit to. This test helped assess the model's ability to avoid overfitting and focus on meaningful features. To make the segmentation labeling process easier, we implemented an automatic labeling workflow using Langsam, which streamlines the creation of accurate segmentation masks. Additionally, we developed a user-friendly frontend for uploading data, training, and evaluating models, simplifying the workflow and making the system more accessible for practical use.

Master 4. Semester
## Prerequisites

- **Core Libraries:** PyTorch, NumPy, OpenCV (cv2), [Augmentor](https://github.com/mdbloice/Augmentor)
- **Segmentation Models:** 
  - Download the [SAM model](https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth) and create a directory called `models` and put it there
  - Download the [LangSAM](https://github.com/luca-medeiros/lang-segment-anything/tree/main/lang_sam) model (whole lang_sam folder) and paste it into the project 
- **Installation Issues:** If you encounter issues, use the following commands to install PyTorch with CUDA 11.7 support:
  ```bash
  pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
  ```
- **Additional Dependencies:** 
  - Ensure all required Python packages are installed by running:
    ```bash
    pip install -r requirements.txt
    ```
  - **TODO** `requirements.txt` schreiben von Grund auf

## Instructions for the Web Application

Run the web application using:

```bash
flask run
```

The Web App is structured into four main sections:

### 1. Datasets

- **Dataset Management:** Upload and delete datasets.
- **Data Format:** Data must be preprocessed and compressed into a `.zip` file before uploading.
- **Required Directory Structure:**
  ```plaintext
  <name_dataset>
    ├── train
    │   ├── class_1
    │   ├── ...
    │   └── class_n
    └── test
        ├── class_1
        ├── ...
        └── class_n
  ```

### 2. Training

- **Submit Settings:** Ensure all settings are correct before starting the training process.
- **Training Management:** 
  - Start training with the submitted settings.
  - Option to stop training during runtime.
  - Upon successful training, you will be redirected to the results overview page.

### 3. Results

- **Training Session Overview:** View results from each training session located in `ProtoPNet/saved_models`.
- **Submitted Settings:** Access settings used for training from `ProtoPNet/saved_models/<experiment_name>/settings_submitted.txt`.
- **Classification Report:** View the classification report from `ProtoPNet/saved_models/<experiment_name>/train_report.txt`.
- **Prototype Visualization:** Visualize prototypes for each image specified in `settings.py` located at `ProtoPNet/saved_models/<experiment_name>/proto_vis`.

### 4. Segmentation

- **Initial Segmentation:** Segment images from the datasets using LangSAM or manually edit existing segmentation masks.
- **Mask Storage:** Numpy files for each segmented image are stored in `datasets_masks/<dataset_name>`.

## Instructions for Training the Model Independently

1. **Navigate to the ProtoPNet Directory:**
   ```bash
   cd ProtoPNet
   ```
2. **Configure Training Settings:**
   - In `settings.py`, set the appropriate values for:
     - `data_path`: Path to the dataset.
     - `train_dir`, `test_dir`, `train_push_dir`: Directories for training and testing.
     - `experiment_run`: Name your training session.
     - `tr_mode`: Choose between `normal` and `mask` modes.
       - `normal`: Train ProtoPNet without segmentation masks.
       - `mask`: Train ProtoRRRNet using segmentation masks. Ensure masks are available in `datasets_masks/<dataset_name>`.
   - Additional settings can be modified according to your requirements.
3. **Run the Training:**
   ```bash
   python main.py
   ```

---