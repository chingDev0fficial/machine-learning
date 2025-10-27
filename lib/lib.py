import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
import os
from PIL import Image
from torch.utils.data import Dataset

from sklearn.feature_extraction.image import img_to_graph
from torch_geometric.utils import remove_self_loops
import scipy.sparse as sp

import numpy as np

from sklearn.feature_extraction import image


class SiameseSignatureDataset(Dataset):
    def __init__(self, root_dir, signer_folders, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for signer_path in signer_folders.values:
            img1, img2, label = signer_path

            img1 = os.path.join(root_dir, *img1.split('/'))
            img2 = os.path.join(root_dir, *img2.split('/'))

            self.samples.append([img1, img2, label])

        print(f"Loaded {len(self.samples)} signature images (genuine + forged)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1, img2, label = self.samples[idx]
        try:
            img1 = Image.open(img1).convert("L")  # grayscale
            img2 = Image.open(img2).convert("L")  # grayscale
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            return image_to_graph(img1), image_to_graph(img2), label
        except Exception as e:
            print(f"Error loading {img1} and {img2}: {e}")
            # fallback blank image
            fallback = Image.new("L", (224, 224), 0)
            if self.transform:
                fallback = self.transform(fallback)
            return fallback


class SignatureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # collect all signer folders
        signer_folders = sorted(os.listdir(root_dir))

        for folder in signer_folders:
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for img_name in os.listdir(folder_path):
                    if self._is_image_file(img_name):
                        self.samples.append(os.path.join(folder_path, img_name))

        print(f"Loaded {len(self.samples)} signature images (genuine + forged)")

    def _is_image_file(self, filename):
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return os.path.splitext(filename.lower())[1] in valid_exts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        try:
            image = Image.open(path).convert("L")  # grayscale
            if self.transform:
                image = self.transform(image)
            return image   # only image, no label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # fallback blank image
            fallback = Image.new("L", (224, 224), 0)
            if self.transform:
                fallback = self.transform(fallback)
            return fallback

def image_to_graph(img_tensor):
    """
    Convert image tensor to graph
    
    Args:
        img_tensor: torch.Tensor (C, H, W) or (H, W)
    
    Returns:
        Data object with x and edge_index
    """
    # Ensure it's a tensor
    if isinstance(img_tensor, np.ndarray):
        img_tensor = torch.from_numpy(img_tensor).float()
    
    # Add channel dimension if needed
    if len(img_tensor.shape) == 2:
        img_tensor = img_tensor.unsqueeze(0)
    
    # Use existing tensor directly (no resize)
    img = img_tensor.squeeze().numpy()
    
    # Convert to graph
    graph_coo = img_to_graph(img, return_as=sp.coo_array)
    
    # Extract edge_index
    edge_index = torch.tensor(
        np.vstack([graph_coo.row, graph_coo.col]),
        dtype=torch.long
    )
    
    # Remove self-loops
    edge_index, _ = remove_self_loops(edge_index)
    
    # Create node features
    H, W = img.shape
    pixel_values = img.flatten()
    
    # Position features (normalized)
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    positions = np.stack([y_coords.flatten(), x_coords.flatten()], axis=1)
    positions_norm = positions / np.array([H, W])
    
    # Combine features: [pixel_value, y_pos, x_pos]
    x = torch.tensor(
        np.column_stack([pixel_values, positions_norm]),
        dtype=torch.float
    )
    
    # Create Data object
    data = Data(x=x, edge_index=edge_index)
    
    return data
