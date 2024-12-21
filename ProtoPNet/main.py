import os
import shutil
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import re
from helpers import makedir, plot_training_curves
import model
import push as push
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from settings import selected_mask_percentage  # Ensure this is imported

class DualModeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size, transform=None, training_mode="normal"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_size = img_size
        self.training_mode = training_mode
        self.image_files = []
        self.labels = []
        self.mask_files = []

        # Load all images and labels
        for label, class_dir in enumerate(sorted(os.listdir(image_dir))):
            class_dir_path = os.path.join(image_dir, class_dir)
            if os.path.isdir(class_dir_path):
                for file_name in os.listdir(class_dir_path):
                    if file_name.endswith('.jpg') or file_name.endswith('.png'):
                        self.image_files.append(os.path.join(class_dir_path, file_name))
                        self.labels.append(label)
        
        if self.training_mode == "mask":
            self._select_masks()

    def _select_masks(self):
        all_masks = [f for f in os.listdir(self.mask_dir) if f.endswith('_mask.npy')]
        
        if selected_mask_percentage is not None and selected_mask_percentage < 100:
            num_masks_to_select = int(len(all_masks) * (selected_mask_percentage / 100))
            selected_masks = np.random.choice(all_masks, num_masks_to_select, replace=False)
        else:
            selected_masks = all_masks
        
        self.mask_files = selected_masks

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        mask = None
        if self.training_mode == "mask":
            mask_file = os.path.basename(image_path).replace('.jpg', '_mask.npy').replace('.png', '_mask.npy')
            if mask_file in self.mask_files:
                mask_path = os.path.join(self.mask_dir, mask_file)
                if os.path.exists(mask_path):
                    mask = np.load(mask_path)
                    mask = Image.fromarray(mask)

        if self.transform:
            if self.training_mode == "mask" and mask is not None:
                image, mask = self.transform(image, mask, img_size=self.img_size)
                return image, torch.tensor(label, dtype=torch.int64), mask, image_path
            else:
                image = self.transform(image, mask=None, img_size=self.img_size)
                return image, torch.tensor(label, dtype=torch.int64), image_path

        return image, torch.tensor(label, dtype=torch.int64), mask, image_path if mask else image_path


# Define transforms for both image and mask
def combined_transform(image, mask=None, img_size=224):
    image_transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if mask is not None:
        mask_transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ])
        mask = mask_transform(mask).float()

    image = image_transform(image)

    if mask is not None:
        return image, mask
    else:
        return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpuid', nargs=1, type=str, default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    from settings import base_architecture, img_size, prototype_shape, num_classes, \
                         prototype_activation_function, add_on_layers_type, experiment_run, mask_dir, training_mode

    base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

    model_dir = './saved_models/' + '/' + experiment_run + '/'
    
    makedir(model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)
    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'

    from settings import train_dir, test_dir, train_push_dir, \
                         train_batch_size, test_batch_size, train_push_batch_size

    # Load data using DualModeDataset
    train_dataset = DualModeDataset(train_dir, mask_dir, img_size, transform=combined_transform, training_mode=training_mode)
    test_dataset = DualModeDataset(test_dir, mask_dir, img_size, transform=combined_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=1, pin_memory=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=1, pin_memory=False)

    # push set
    train_push_dataset = datasets.ImageFolder(
        train_push_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ]))
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
        num_workers=1, pin_memory=False)

    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('batch size: {0}'.format(train_batch_size))

    ppnet = model.construct_PPNet(base_architecture=base_architecture,
                                  pretrained=True, img_size=img_size,
                                  prototype_shape=prototype_shape,
                                  num_classes=num_classes,
                                  prototype_activation_function=prototype_activation_function,
                                  add_on_layers_type=add_on_layers_type)

    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    from settings import joint_optimizer_lrs, joint_lr_step_size
    joint_optimizer_specs = [
        {'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
        {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
        {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

    from settings import warm_optimizer_lrs
    warm_optimizer_specs = [
        {'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
        {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    from settings import last_layer_optimizer_lr
    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    from settings import coefs, num_train_epochs, num_warm_epochs, push_start, push_epochs, num_pre_epochs

    # Initialize the metrics dictionary to track performance across epochs
    metrics = {
        "train_loss": [],
        "train_accuracy": [],
        "test_accuracy": []
    }
    torch.autograd.set_detect_anomaly(True)
    log('start training')
    for epoch in range(num_train_epochs):
        log('epoch: \t{0}'.format(epoch))

        if epoch < num_pre_epochs:
            tnt.pre_only(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                          class_specific=class_specific, coefs=coefs, log=log, epoch=epoch, metrics=metrics)
        elif epoch < num_warm_epochs:
            tnt.warm_only(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                          class_specific=class_specific, coefs=coefs, log=log, epoch=epoch, metrics=metrics)
        else:
            tnt.joint(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                          class_specific=class_specific, coefs=coefs, log=log, epoch=epoch, metrics=metrics)
            #joint_lr_scheduler.step()

        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log, metrics=metrics)
        
        # Save the model only in the last two epochs
        if epoch >= 0:
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                        target_accu=0.00, log=log)
        
        # save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
        #                             target_accu=0.70, log=log)

        if epoch >= push_start and epoch in push_epochs:
            push.push_prototypes(
                train_push_loader,
                prototype_network_parallel=ppnet_multi,
                class_specific=class_specific,
                preprocess_input_function=preprocess_input_function,
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir,
                epoch_number=epoch,
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log)
            accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log, metrics=metrics)
            # save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
            #                             target_accu=0.70, log=log)

            # Save the model only in the last two epochs
            if epoch >= 0:
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                            target_accu=0.00, log=log)


            if prototype_activation_function != 'linear':
                tnt.last_only(model=ppnet_multi, log=log)
                for i in range(20):
                    log('iteration: \t{0}'.format(i))
                    _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                                  class_specific=class_specific, coefs=coefs, log=log, epoch=epoch, metrics=metrics)
                    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                    class_specific=class_specific, log=log, metrics=metrics)
                    # save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                    #                             target_accu=0.70, log=log)

        # Ensure metrics are consistently updated
        if len(metrics["test_accuracy"]) < len(metrics["train_loss"]):
            metrics["test_accuracy"].append(None)  # Placeholder if no test was done this epoch

        plot_training_curves(metrics, model_dir)

    logclose()

if __name__ == '__main__':
    main()
