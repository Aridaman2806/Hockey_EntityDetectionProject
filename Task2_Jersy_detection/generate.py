import os

def generate_file_list(dataset_path, split, output_file):
    """
    Generate a file listing all image paths for YOLO training.

    Args:
    - dataset_path (str): Path to the dataset (train/val directory).
    - split (str): Subdirectory name ("train" or "val").
    - output_file (str): Output file name (train.txt or val.txt).
    """
    image_dir = os.path.join(dataset_path, split, "images")
    with open(output_file, "w") as f:
        for image_name in os.listdir(image_dir):
            if image_name.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(image_dir, image_name)
                f.write(image_path + "\n")
    print(f"{output_file} has been created with image paths.")

# Define dataset directory and output files
dataset_path = "./dataset"  # Adjust to your dataset folder location
generate_file_list(dataset_path, "train", "train.txt")
generate_file_list(dataset_path, "val", "val.txt")
