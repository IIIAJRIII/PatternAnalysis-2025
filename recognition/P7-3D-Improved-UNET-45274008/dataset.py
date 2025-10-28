import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib

class ProstateDataset(Dataset):
    """
    Custom PyTorch Dataset for loading 3D NIfTI files.
    """
    def __init__(self, image_files, mask_files, transform=None):
        self.image_files = image_files
        self.mask_files = mask_files
        self.transform = transform # For data augmentation

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load NIfTI files
        img = nib.load(self.image_files[idx]).get_fdata(dtype=np.float32)
        mask = nib.load(self.mask_files[idx]).get_fdata().astype(np.uint8)

        # Basic Pre-processing
        # 1. Normalize image (simple min-max to [0, 1])
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
        
        # 2. Add channel dimension: (H, W, D) -> (1, H, W, D)
        # Note: nibabel loads as (W, H, D) or (X, Y, Z)
        img = np.expand_dims(img, axis=0) 
        
        # 3. Permute to PyTorch's (C, D, H, W) format
        img = img.transpose(0, 3, 2, 1)
        
        # 4. Permute mask (W, H, D) -> (D, H, W)
        mask = mask.transpose(2, 1, 0)

        # Convert to tensors
        img_tensor = torch.from_numpy(img.copy()).float()
        mask_tensor = torch.from_numpy(mask.copy()).long()
        
        # Apply transforms (augmentation) if any
        if self.transform:
            pass

        return img_tensor, mask_tensor
