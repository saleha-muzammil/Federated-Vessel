import os
import pickle
import matplotlib.pyplot as plt

def load_preprocessed_image(file_path):
    with open(file_path, 'rb') as file:
        image = pickle.load(file)
    return image

def save_image_pair(original_image, gt_image, index, output_dir='output_images'):
    os.makedirs(output_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    
    # Original Image
    if original_image.ndim == 3 and original_image.shape[0] == 1:
        original_image = original_image.squeeze(0)
    axs[0].imshow(original_image, cmap='gray')
    axs[0].axis('off')
    axs[0].set_title('Original Image')
    
    # Ground Truth Image
    if gt_image.ndim == 3 and gt_image.shape[0] == 1:
        gt_image = gt_image.squeeze(0)
    axs[1].imshow(gt_image, cmap='gray')
    axs[1].axis('off')
    axs[1].set_title('Ground Truth')

    output_path = os.path.join(output_dir, f'pair_{index}.png')
    plt.savefig(output_path)
    print(f"Saved image pair {index} to {output_path}")
    plt.close()

# Directory where the preprocessed images are stored
save_path = 'datasets/CHASEDB1/training_pro'  # Replace with your path
preprocessed_files = sorted(os.listdir(save_path))

# Separate the original and ground truth files
original_files = [file for file in preprocessed_files if 'img_patch' in file][:20]
gt_files = [file for file in preprocessed_files if 'gt_patch' in file][:20]

# Load and save the images in pairs
for i in range(20):
    original_image = load_preprocessed_image(os.path.join(save_path, original_files[i]))
    gt_image = load_preprocessed_image(os.path.join(save_path, gt_files[i]))
    save_image_pair(original_image, gt_image, i)
