import torch
import numpy as np
import nibabel as nib
import glob
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your model and dataset classes
from modules import UNet3D
from dataset import ProstateDataset

# --- Configuration ---
# These MUST match the settings used in train.py
TEST_SPLIT = 0.2
RANDOM_SEED = 42
N_CLASSES = 6
N_CHANNELS = 1
BASE_FEATURES = 16 # Must match the "slim" model you trained
BATCH_SIZE = 1 # Use batch size of 1 for validation to avoid OOM

# --- Paths ---
# Get the absolute path to the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "images")
MODEL_SAVE_PATH = "3d_unet_prostate_slim.pth" # Path to your trained model

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_data():
    """Finds all data files and splits them into train/val sets."""
    img_path = os.path.join(DATA_PATH, "semantic_MRs_anon")
    mask_path = os.path.join(DATA_PATH, "semantic_labels_anon")

    image_files = sorted(glob.glob(os.path.join(img_path, "*.nii.gz"))) + \
                    sorted(glob.glob(os.path.join(img_path, "*.nii")))
                    
    mask_files = sorted(glob.glob(os.path.join(mask_path, "*.nii.gz"))) + \
                   sorted(glob.glob(os.path.join(mask_path, "*.nii")))
    
    if not image_files or not mask_files or len(image_files) != len(mask_files):
        print("Error loading data. Check paths and file counts.")
        return None, None, None, None

    # Split data exactly as in train.py to get the same validation set
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_files, mask_files, test_size=TEST_SPLIT, random_state=RANDOM_SEED
    )
    return train_imgs, val_imgs, train_masks, val_masks

def dice_metric(preds_logits, targets, n_classes, smooth=1e-6):
    """
    Calculates Dice score for multi-class segmentation.
    Ignores background (class 0).
    """
    # Get predictions by taking argmax
    preds = torch.argmax(preds_logits, dim=1)
    
    dice_scores = []
    
    # Calculate Dice for each class (except background)
    for c in range(1, n_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())
        
    # Return the average Dice score across all foreground classes
    if len(dice_scores) == 0:
        return 0.0
    return np.mean(dice_scores)

def visualize_first_sample(model, val_loader, val_imgs, val_masks):
    """
    Loads the first validation sample, runs prediction, and displays
    a 3D slice-by-slice animation.
    """
    print("\nVisualizing first validation sample (animated)...")
    
    # Get the first batch (sample)
    try:
        images, masks = next(iter(val_loader))
    except StopIteration:
        print("Error: Validation loader is empty.")
        return

    images, masks = images.to(DEVICE), masks.to(DEVICE)

    # 4. Run model inference
    with torch.no_grad():
        output_logits = model(images)
    
    # 5. Post-process
    # Get the predicted mask by taking argmax
    # output_logits shape: (B, C, D, H, W) -> (1, 6, D, H, W)
    # predicted_mask shape: (B, D, H, W) -> (1, D, H, W)
    predicted_mask = torch.argmax(output_logits, dim=1)
    
    # Move to CPU and remove batch dimension for visualization
    predicted_mask_np = predicted_mask.cpu().squeeze(0).numpy() # (D, H, W)
    
    # Load the raw image and mask for comparison
    # These are loaded from disk to get the original (W, H, D) orientation
    raw_image_np = nib.load(val_imgs[0]).get_fdata().astype(np.float32)
    raw_mask_np = nib.load(val_masks[0]).get_fdata().astype(np.int64)

    # 6. Visualize the results (Looping through slices)
    print("Displaying animated results (looping through slices)...")
    print("Close the plot window to stop the script.")

    # Enable interactive mode
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Get the number of slices
    # Raw image is (W, H, D), Pred mask is (D, H, W)
    num_slices = min(raw_image_np.shape[2], predicted_mask_np.shape[0])

    for i in range(num_slices):
        # Check if the figure is still open
        if not plt.fignum_exists(fig.number):
            print("Plot window closed. Stopping animation.")
            break
            
        # Clear the axes for the new slices
        axes[0].clear()
        axes[1].clear()
        axes[2].clear()

        # Plot Raw Image (W, H, D) -> get slice [:, :, i]
        # Transpose (.T) and set origin='lower' to match nibabel's default view
        axes[0].imshow(raw_image_np[:, :, i].T, cmap='gray', origin='lower')
        axes[0].set_title(f"Raw Image (Slice {i+1}/{num_slices})")
        axes[0].axis('off')
        
        # Plot Ground Truth Mask (W, H, D) -> get slice [:, :, i]
        axes[1].imshow(raw_mask_np[:, :, i].T, cmap='nipy_spectral', vmin=0, vmax=N_CLASSES-1, origin='lower')
        axes[1].set_title(f"Ground Truth Mask (Slice {i+1}/{num_slices})")
        axes[1].axis('off')
        
        # Plot Predicted Mask (D, H, W) -> get slice [i, :, :]
        # Squeeze to (H, W)
        # REMOVED the extra .T to fix 90-degree rotation
        predicted_slice = predicted_mask_np[i, :, :]
        axes[2].imshow(predicted_slice, cmap='nipy_spectral', vmin=0, vmax=N_CLASSES-1, origin='lower')
        axes[2].set_title(f"Predicted Mask (Slice {i+1}/{num_slices})")
        axes[2].axis('off')
        
        plt.suptitle(f"Model Prediction for {os.path.basename(val_imgs[0])}")
        plt.tight_layout()
        
        # Pause to create animation effect (0.1 seconds)
        plt.pause(0.1) 

    # Disable interactive mode and clean up
    plt.ioff()
    print("Animation finished. Close the plot window to exit.")
    if plt.fignum_exists(fig.number):
        plt.show() # Show the final frame


def evaluate_validation_set():
    print("Starting 3D UNet Evaluation...")
    print(f"Using device: {DEVICE}")

    print(f"Loading data for validation...")
    _, val_imgs, _, val_masks = load_data()
    
    if not val_imgs:
        print("Could not load validation data.")
        return

    # 1. Load the trained model
    print(f"Loading model from {MODEL_SAVE_PATH}")
    model = UNet3D(n_channels=N_CHANNELS, n_classes=N_CLASSES, base_features=BASE_FEATURES)
    
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_SAVE_PATH}")
        print("Please make sure you have trained the model and it is in the correct location.")
        return
    except Exception as e:
        print(f"Error loading model state. Did you use the same model parameters?")
        print(e)
        return
        
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode

    # 2. Create Dataset and DataLoader for the validation set
    print(f"Creating validation dataloader with {len(val_imgs)} samples...")
    val_dataset = ProstateDataset(val_imgs, val_masks)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    total_dice = 0.0
    num_samples = 0

    # 3. Run prediction on the entire validation set
    print("Running evaluation on validation set...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            # Run model inference
            output_logits = model(images)
            
            # Calculate Dice score for this batch
            batch_dice = dice_metric(output_logits, masks, N_CLASSES)
            
            total_dice += batch_dice
            num_samples += 1
            
    # 4. Calculate and print the final average Dice score
    average_dice = total_dice / num_samples
    
    print("\n--- Validation Complete ---")
    print(f"Total validation samples: {num_samples}")
    print(f"Average Dice Score (foreground classes): {average_dice:.4f}")
    
    # if average_dice >= 0.7:
    #     print("\nCongratulations! You have met the 0.7 Dice score target.")
    # else:
    #     print(f"\nTarget not met. Average Dice: {average_dice:.4f}. Target: 0.7")
    #     print("You may need to train for more epochs, add data augmentation, or try the 'Improved UNet'.")

    # 5. Now, run the visualization
    # We pass the model and loader so we don't have to re-create them
    visualize_first_sample(model, val_loader, val_imgs, val_masks)


if __name__ == "__main__":
    evaluate_validation_set()

