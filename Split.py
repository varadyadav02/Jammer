import os
import shutil
import random

# Paths
input_folder = r"D:/Final_Capstone_Dataset"
output_folder = r"D:/Final_Capstone_Dataset_Split"
train_ratio = 0.7  # 70% for training
val_ratio = 0.15   # 15% for validation, 15% for testing

# Create output directories for train, val, and test
for split in ['train', 'val', 'test']:
    split_path = os.path.join(output_folder, split)
    if not os.path.exists(split_path):
        os.makedirs(split_path)

# Iterate over each subfolder (abnormal, normal) in the input dataset
for category in ['abnormal', 'normal']:
    category_path = os.path.join(input_folder, category)

    if os.path.isdir(category_path):
        # Create category directories in train, val, and test
        for split in ['train', 'val', 'test']:
            split_category_path = os.path.join(output_folder, split, category)
            if not os.path.exists(split_category_path):
                os.makedirs(split_category_path)

        # Get all file names in the category directory
        files = os.listdir(category_path)
        random.shuffle(files)  # Shuffle the files to ensure random splitting

        # Calculate the number of files for each split
        num_files = len(files)
        train_split = int(train_ratio * num_files)
        val_split = int(val_ratio * num_files)

        # Split the dataset
        train_files = files[:train_split]
        val_files = files[train_split:train_split + val_split]
        test_files = files[train_split + val_split:]

        # Copy files to the respective directories
        for file_name in train_files:
            shutil.copy(os.path.join(category_path, file_name), os.path.join(output_folder, 'train', category, file_name))

        for file_name in val_files:
            shutil.copy(os.path.join(category_path, file_name), os.path.join(output_folder, 'val', category, file_name))

        for file_name in test_files:
            shutil.copy(os.path.join(category_path, file_name), os.path.join(output_folder, 'test', category, file_name))

        print(f"Category {category}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

print("Dataset splitting completed.")
