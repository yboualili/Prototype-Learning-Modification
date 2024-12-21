import time
import torch
import numpy as np
from helpers import list_of_distances, make_one_hot
import cv2
import torch.nn as nn
from sklearn.metrics import classification_report
from scipy.spatial import distance
# Custom Loss Function
import os
from settings import training_mode, images_to_print, experiment_run, num_classes, num_train_epochs
from preprocess import mean, std
import torch.nn.functional as F
model_dir = './saved_models/' + '/' + experiment_run + '/'

def reverse_normalize(image, mean, std):
    mean = torch.tensor(mean)[:, None, None]  # Convert to tensor and reshape to (C, 1, 1)
    std = torch.tensor(std)[:, None, None]  # Convert to tensor and reshape to (C, 1, 1)
    image = image * std + mean
    return image
def distance_to_nearest_point(point, mask):
    """
    Compute the Euclidean distance from the given point to the nearest `1` in the mask using a brute-force method.

    Args:
        point (list or tuple): The (x, y) coordinates of the point.
        mask (torch.Tensor): The binary segmentation mask tensor of shape (H, W).

    Returns:
        float: The Euclidean distance to the nearest `1` in the mask.
    """
    x, y = point
    mask_np = mask.cpu().numpy().astype(np.float32)  # Convert to numpy array and ensure float32
    mask_np = mask_np.reshape(224,224)
    # Extract coordinates where mask value is `1`
    # Extract coordinates where mask value is `1`
    object_coords = np.argwhere(mask_np == 1)

    if len(object_coords) == 0:
        return float('inf')  # Return infinity if no objects are present in the mask

    # Convert coordinates to numpy array for vectorized operations
    object_coords = np.array(object_coords)

    # Compute distances using vectorized operations
    distances = np.sqrt((object_coords[:, 0] - x)**2 + (object_coords[:, 1] - y)**2)
    # Return the minimum distance
    return np.min(distances)


def save_mask_with_point_and_distance(mask, point, distance, save_path):
    """
    Draw a green dot on the mask at the specified point and save the result to the specified path.

    Args:
        mask (torch.Tensor): The binary segmentation mask tensor of shape (H, W).
        point (list or tuple): The (x, y) coordinates of the point.
        save_path (str): The file path where the image should be saved.
    """
    # Convert mask tensor to numpy array
    mask_np = mask.cpu().numpy()
    # Normalize mask to [0, 255] for saving
    mask_img = (mask_np * 255).astype(np.uint8)
    # Create a color image (RGB)
    # Convert single-channel mask to a 3-channel RGB image
    # Convert single-channel mask to a 3-channel RGB image
    # Add the channel dimension and make the shape (1, 224, 224)
    # Convert to OpenCV format (1, 224, 224) to (224, 224, 1)
    gray_image = np.squeeze(mask_img, axis=0)  # Shape is (224, 224)
    # Convert grayscale image to RGB
    rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)  # Shape is (224, 224, 3)

    # Draw a green dot
    point_x, point_y = int(point[1]), int(point[0])  # Note: (x, y) => (col, row) => (x, y)
    cv2.circle(rgb_image, (point_x, point_y), 5, (0, 255, 0), -1)  # Green dot with radius 5

    # Draw the distance as text
    distance_text = f"Distance: {distance:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    color = (255, 0, 0)  # White color for the text
    cv2.putText(rgb_image, distance_text, (0, 30), font, font_scale, color, font_thickness, cv2.LINE_AA)

    # Save the image
    cv2.imwrite(save_path, rgb_image)


def distance_to_nearest_point(point, mask):
    """
    Compute the distance from the given point to the nearest 1 in the binary mask.

    Args:
        point (tuple of tensors): The (y, x) coordinates of the point.
        mask (torch.Tensor): The binary segmentation mask tensor of shape (1, H, W).

    Returns:
        torch.Tensor: The computed minimum distance.
    """
    # Ensure the mask is 3D: (batch_size, H, W)
    assert mask.dim() == 3 and mask.size(0) == 1, "Mask must have shape (1, H, W)."

    # Remove the batch dimension for coordinate extraction
    mask_2d = mask.squeeze(0)  # Shape (H, W)

    # Get the coordinates of all '1' points in the mask
    ones_coords = torch.nonzero(mask_2d, as_tuple=False).float()  # Shape (N, 2) for (y, x) coordinates

    # If no '1' in the mask, return a large loss
    if ones_coords.shape[0] == 0:
        return torch.tensor(float('inf'), dtype=torch.float32, device=mask.device)

    # point is a tuple of tensors (y, x)
    point_y, point_x = point
    point_tensor = torch.stack([point_y, point_x], dim=0)  # Shape (2,)

    # Compute Euclidean distances between the point and all '1' points in the mask
    distances = torch.norm(ones_coords.cuda() - point_tensor.unsqueeze(0).cuda(), dim=1)  # Shape (N,)

    # Return the minimum distance
    return distances.min()

class PointToMaskLoss(torch.nn.Module):
    def __init__(self):
        super(PointToMaskLoss, self).__init__()

    def forward(self, point, mask, epoch):
        """
        Compute the loss based on the distance from the point to the nearest `1` in the mask.

        Args:
            point (list or tuple): The (x, y) coordinates of the point.
            mask (torch.Tensor): The binary segmentation mask tensor of shape (H, W).

        Returns:
            torch.Tensor: The computed loss value.
        """
        # Compute the minimum distance from the point to the nearest '1' in the mask
        min_distance = distance_to_nearest_point(point, mask)

        # Define the loss as the minimum distance
        loss = min_distance
        save_path = os.path.join("output", f"mask_with_point_{point}_{epoch}.png")
        #save_mask_with_point_and_distance(mask, point, min_distance, save_path)
        return loss

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print, epoch=-1, metrics=None):
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    total_separation_cost = 0
    total_avg_separation_cost = 0
    total_distance_loss = 0
    criterion = PointToMaskLoss()
    all_targets = []
    all_predictions = []

    for i, data in enumerate(dataloader):
        # Check the length of the data to determine if a mask is present
        if len(data) == 4:
            image, label, mask, image_path = data
            mask_exists = mask is not None
        elif len(data) == 3:
            image, label, image_path = data
            mask_exists = False
            mask = None  # Set mask to None if it wasn't provided
        else:
            raise ValueError("Unexpected number of items returned from the DataLoader")
        
        input = image.cuda()
        target = label.cuda()
        max_dist = (model.module.prototype_shape[1]
                    * model.module.prototype_shape[2]
                    * model.module.prototype_shape[3])
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        distance_loss = torch.zeros((), device="cuda", dtype=torch.float32)
        with grad_req:
            output, min_distances, distances, conv_features = model(input)
            
            if epoch >= 0:
                proto_dist_ = distances
                proto_classes = [torch.argmax(identity).item() for identity in
                                 model.module.prototype_class_identity]

                for img_idx in range(proto_dist_.shape[0]):
                    per_img_loss = torch.tensor(0, device="cuda", dtype=torch.float32)
                    for rel_prototype in range(proto_dist_.shape[1]):
                        if proto_classes[rel_prototype] == target[img_idx]:
                            proto_dist_img_j = proto_dist_[img_idx, rel_prototype, :, :]

                            if model.module.prototype_activation_function == 'log':
                                proto_act_img_j = torch.log(
                                    (proto_dist_img_j + 1) / (proto_dist_img_j + model.module.epsilon))
                            elif model.module.prototype_activation_function == 'linear':
                                proto_act_img_j = max_dist - proto_dist_img_j

                            # Upsample the activation map to 224x224
                            upsampled_act_img_j = F.interpolate(proto_act_img_j.unsqueeze(0).unsqueeze(0),
                                                                size=(224, 224), mode='bilinear',
                                                                align_corners=False).squeeze(0).squeeze(0)

                            upsampled_act_img_j2= cv2.resize(proto_act_img_j.detach().cpu().numpy(), dsize=(224, 224),
                                                             interpolation=cv2.INTER_CUBIC)

                            h, w = upsampled_act_img_j.shape
                            x_coords = torch.arange(0, w).unsqueeze(0).repeat(h, 1).to(upsampled_act_img_j.device)
                            y_coords = torch.arange(0, h).unsqueeze(1).repeat(1, w).to(upsampled_act_img_j.device)

                            sum_act = upsampled_act_img_j.sum()
                            if sum_act == 0:
                                expected_x = torch.tensor(0.0, device=upsampled_act_img_j.device)
                                expected_y = torch.tensor(0.0, device=upsampled_act_img_j.device)
                            else:
                                expected_x = (upsampled_act_img_j * x_coords).sum() / sum_act
                                expected_y = (upsampled_act_img_j * y_coords).sum() / sum_act

                            # Get the max value and its position for highlighting
                            max_value = np.max(upsampled_act_img_j2)
                            max_position = np.unravel_index(np.argmax(upsampled_act_img_j2), upsampled_act_img_j2.shape)

                            # Normalize the activation map: this step is key to consistent colormap application
                            rescaled_act_img_j = upsampled_act_img_j2 - np.min(upsampled_act_img_j2)
                            rescaled_act_img_j = rescaled_act_img_j / np.max(rescaled_act_img_j)

                            # Apply the colormap to ensure the most activated regions are red
                            heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)

                            # Convert heatmap to float32 for overlay
                            heatmap = np.float32(heatmap)

                            # Ensure the original image is in RGB format and normalized
                            original_img_j = image[img_idx]
                            original_image = reverse_normalize(original_img_j, mean, std).numpy() * 255
                            original_img_j = np.transpose(original_image, (1, 2, 0))

                            # Overlay the heatmap onto the original image
                            overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap

                            # Draw an "X" on the overlayed image at the max position
                            x, y = max_position
                            color = (0, 255, 0)  # Neon-green color for the X marker
                            thickness = 2
                            cv2.line(overlayed_original_img_j, (y - 10, x - 10), (y + 10, x + 10), color, thickness)
                            cv2.line(overlayed_original_img_j, (y + 10, x - 10), (y - 10, x + 10), color, thickness)

                            # Save the overlayed image
                            original_image_path_j = image_path[img_idx]
                            original_image_name_j = os.path.splitext(os.path.basename(original_image_path_j))[0]

                            if original_image_name_j + ".jpg" in images_to_print:
                                save_dir = os.path.join(model_dir, "proto_vis")
                                os.makedirs(save_dir, exist_ok=True)
                                save_path = os.path.join(save_dir, f"{original_image_name_j}_{epoch}.jpg")
                                cv2.imwrite(save_path, overlayed_original_img_j)

                            if training_mode == "mask" and mask_exists:
                                # max_value = np.max(upsampled_act_img_j)
                                # max_position = np.unravel_index(np.argmax(upsampled_act_img_j), upsampled_act_img_j.shape)
                                #prototype_positions.append([max_value, max_position])
                                loss = criterion((expected_y, expected_x), mask[img_idx], epoch)
                                per_img_loss += loss
                    if training_mode == "mask" and mask_exists:
                        distance_loss += per_img_loss / torch.tensor(64, device="cuda", dtype=torch.float32)

            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:, label]).cuda()
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                avg_separation_cost = torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1)
            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()
            total_distance_loss += distance_loss

        if is_train:
            if class_specific:
                loss = (coefs['crs_ent'] * cross_entropy
                        + coefs['clst'] * cluster_cost
                        + coefs['sep'] * separation_cost
                        + coefs['l1'] * l1 + distance_loss)
            else:
                loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1 + distance_loss

            # if distance_loss.item() > 80:
            #     loss = 100 * distance_loss + 0.01 * cross_entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input, target, output, predicted, min_distances

    end = time.time()

    # log(f'\ttime: \t{end -  start}')
    # log(f'\tcross ent: \t{total_cross_entropy / n_batches}')
    log(f'\tcluster: \t{total_cluster_cost / n_batches}')
    log(f'\tclass: \t{total_cross_entropy / n_batches}')
    if class_specific:
        log(f'\tseparation:\t{total_separation_cost / n_batches}')
        # log(f'\tavg separation:\t{total_avg_separation_cost / n_batches}')
    if training_mode == "mask":
        log(f'\tsegment: \t{distance_loss}')
        log(f'\tavg segment: \t{total_distance_loss / n_batches}')
    log(f'\taccu: \t\t{n_correct / n_examples * 100}%')
    # log(f'\tl1: \t\t{model.module.last_layer.weight.norm(p=1).item()}')
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    # log(f'\tp dist pair: \t{p_avg_pair_dist.item()}')

    if is_train:
        total_loss = (coefs['crs_ent'] * (total_cross_entropy / n_batches)
                    + coefs['clst'] * (total_cluster_cost / n_batches)
                    + coefs['sep'] * (total_separation_cost / n_batches)
                    + coefs['l1'] * (model.module.last_layer.weight.norm(p=1).item())
                    + (total_distance_loss / n_batches))
        # Ensure loss is non-negative
        total_loss = max(torch.tensor(0), total_loss)
        log(f'\ttotal loss: \t{total_loss}')

        # Store the metrics for plotting later
        metrics["train_loss"].append(total_loss.item())
        metrics["train_accuracy"].append(n_correct / n_examples * 100)

    if not is_train:
        # Store the test accuracy (only needed when testing)
        metrics["test_accuracy"].append(n_correct / n_examples * 100)

        # update the classification report
        report = classification_report(all_targets, all_predictions, target_names=[f"Class {i}" for i in range(num_classes)])
        report_path = os.path.join(model_dir, "train_report.txt")
        with open(report_path, "w") as f:
            f.write(report)

    return n_correct / n_examples


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print, epoch=-1, metrics=None):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log, epoch=epoch, metrics=metrics)


def test(model, dataloader, class_specific=False, log=print, metrics=None):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log, metrics=metrics)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')


def pre_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = False

    log('\tpre')

def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')
