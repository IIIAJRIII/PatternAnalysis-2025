import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
import glob
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

# Import model from modules.py
from modules import UNet3D, ImprovedUNet3D
# Import dataset from dataset.py
from dataset import ProstateDataset

# --- Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 1 # Keeping at 1 due to memory constraints
N_EPOCHS = 50
TEST_SPLIT = 0.2 # Defining test split size
RANDOM_SEED = 42 # Defining random seed for reproducibility
# -------------------------

# --- Data Paths ---
# Get the absolute path to the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH is the 'images' folder inside the script's directory
DATA_PATH = os.path.join(SCRIPT_DIR, "images")

# --- Model Configuration ---
N_CLASSES = 6 
N_CHANNELS = 1 # Input image is 1-channel (grayscale)
# We set 16 base features for a "slim" model to fit on an 8GB GPU
BASE_FEATURES = 16 

# -----------------------------
# 2. Dice Metric Function
# -----------------------------
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

# -----------------------------
# 3. Training & Validation Functions
# -----------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def validate(model, loader, criterion, device, n_classes):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            running_loss += loss.item() * images.size(0)
            running_dice += dice_metric(outputs, masks, n_classes)
            
    epoch_loss = running_loss / len(loader.dataset)
    epoch_dice = running_dice / len(loader) # Average dice over batches
    return epoch_loss, epoch_dice

# -----------------------------
# 4. Main Training Pipeline
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train 3D UNet on Prostate Dataset")
    parser.add_argument("-m", "--model", type=str, choices=["unet", "improved_unet"], default="improved_unet",
                        help="Choose the model to train: 'unet' or 'improved_unet'")
    args = parser.parse_args()
    model_choice = args.model
    if model_choice == "unet":
        print("Selected model: Standard 3D UNet")
        ModelClass = UNet3D
        MODEL_SAVE_PATH = "3d_unet_prostate_slim.pth"
    else:
        print("Selected model: Improved 3D UNet")
        ModelClass = ImprovedUNet3D
        MODEL_SAVE_PATH = "improved_3d_unet_prostate_slim.pth"
    print("Starting 3D UNet Training...")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Find all image and mask files
    # Note: Assumes filenames match between images and masks
    img_path = os.path.join(DATA_PATH, "semantic_MRs_anon")
    mask_path = os.path.join(DATA_PATH, "semantic_labels_anon")

    print(f"Searching for images in: {img_path}")
    print(f"Searching for masks in:  {mask_path}")

    image_files = sorted(glob.glob(os.path.join(img_path, "*.nii.gz"))) + \
                    sorted(glob.glob(os.path.join(img_path, "*.nii")))
                    
    mask_files = sorted(glob.glob(os.path.join(mask_path, "*.nii.gz"))) + \
                   sorted(glob.glob(os.path.join(mask_path, "*.nii")))
    
    if not image_files or not mask_files:
        print(f"Error: No data found in {img_path} or {mask_path}.")
        print("Please check your folder names and file extensions.")
        print(f"Found {len(image_files)} images and {len(mask_files)} masks.")
        return

    # Check for mismatched file counts
    if len(image_files) != len(mask_files):
        print(f"Error: Mismatched file count!")
        print(f"Found {len(image_files)} images but {len(mask_files)} masks.")
        return

    # Split data
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_files, mask_files, test_size=TEST_SPLIT, random_state=RANDOM_SEED
    )
    
    # Create Datasets and DataLoaders
    train_dataset = ProstateDataset(train_imgs, train_masks)
    val_dataset = ProstateDataset(val_imgs, val_masks)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Data loaded: {len(train_dataset)} training, {len(val_dataset)} validation.")
    
    # Initialize Model, Loss, and Optimizer
    model = ModelClass(n_channels=N_CHANNELS, n_classes=N_CLASSES, base_features=BASE_FEATURES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_dice = -1.0
    
    # --- Training Loop ---
    for epoch in range(N_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{N_EPOCHS} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device, N_CLASSES)
        
        print(f"Epoch {epoch+1}: \n"
              f"\tTrain Loss: {train_loss:.4f} \n"
              f"\tVal Loss:   {val_loss:.4f} \n"
              f"\tVal Dice:   {val_dice:.4f}")
        
        # Save the best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved to {MODEL_SAVE_PATH} (Dice: {best_val_dice:.4f})")

    print(f"Training finished. Best validation Dice: {best_val_dice:.4f}")
    print(f"Best model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()



