base_architecture = 'resnet101'
img_size = {{ img_size }}
prototype_shape = (8, 128, 1, 1) # TODO: Make Num_Prototypes variable for user and validate number based on num_classes
num_classes = {{ num_classes }}
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '{{ experiment_run }}'

data_path = '{{ data_path }}'
train_dir = data_path + 'train/'
test_dir = data_path + 'test/'
train_push_dir = data_path + 'train/'
train_batch_size = {{ train_batch_size }}
test_batch_size = {{ test_batch_size }}
train_push_batch_size = {{ train_push_batch_size }}

# Changed logic based on the webpage logic, s.t. masks for a datasets are always in datasets_masks, not ordered in classes, train or test
# data_path = '../datasets/corrupted_dataset/' ->  mask_dir = '../datasets_masks/corrupted_dataset/'
mask_dir = data_path.replace('../datasets/', '../datasets_masks/')

joint_optimizer_lrs = {'features': 3e-3,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.02,
    'l1': 1e-4,
}

num_train_epochs = {{ num_train_epochs }}
num_warm_epochs = {{ num_warm_epochs }}
num_pre_epochs = 0
push_start = {{ push_start }}
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]

training_mode = '{{ tr_mode }}'

selected_mask_percentage = {{ selected_mask_percentage }}

images_to_print = [
    "cat_13.jpg",
    "cat_9.jpg",
    "dog_51.jpg",
    "dog_109.jpg",
    "cat_417.jpg",
    "cat_583.jpg",
    "dog_43.jpg",
    "dog_181.jpg"
]