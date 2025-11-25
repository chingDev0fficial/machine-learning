import cv2
import torch
from torch_geometric.data import Data
import os
from PIL import Image
from torch.utils.data import Dataset

# from sklearn.feature_extraction.image import img_to_graph
from torch_geometric.utils import remove_self_loops
# import scipy.sparse as sp

import numpy as np

from scipy.ndimage import gaussian_filter, sobel

# from sklearn.feature_extraction import image


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
            img1 = Image.open(img1)
            img2 = Image.open(img2)
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            return _image_to_graph(img1), _image_to_graph(img2), label
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
            image = Image.open(path) # grayscale
            if self.transform:
                image = self.transform(image)
            return _image_to_graph(image)   # only image, no label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # fallback blank image
            fallback = Image.new("L", (224, 224), 0)
            if self.transform:
                fallback = self.transform(fallback)
            return fallback

def _image_to_graph(img_tensor, threshold=0.1):
    """
    Convert a signature image tensor to a graph for GNN with curvature and stroke width.
    
    Args:
        img_tensor: torch.Tensor or np.ndarray, shape (H,W) or (C,H,W)
        threshold: float, pixel intensity threshold (0-1) to keep signature pixels
    
    Returns:
        Data object with node features x and edge_index
    """
    # Convert to torch tensor if needed
    if isinstance(img_tensor, np.ndarray):
        img_tensor = torch.from_numpy(img_tensor).float()
    
    # If multiple channels, take mean (grayscale)
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.mean(dim=0)
    
    # Normalize pixel values to [0,1]
    img_tensor = img_tensor / img_tensor.max()
    
    H, W = img_tensor.shape
    y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    
    # Keep only signature pixels above threshold
    mask = img_tensor > threshold
    if mask.sum() == 0:
        raise ValueError("No signature pixels found above threshold.")
    
    # Node positions
    positions = torch.stack([y_coords[mask], x_coords[mask]], dim=1).float()
    positions_norm = positions / torch.tensor([H, W], dtype=torch.float)
    
    # Pixel intensity
    pixel_values = img_tensor[mask].unsqueeze(1)
    
    # --- Compute stroke width approximation ---
    # Smooth image and compute local gradients
    img_smooth = gaussian_filter(img_tensor.numpy(), sigma=1)
    grad_x = sobel(img_smooth, axis=1)
    grad_y = sobel(img_smooth, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag_tensor = torch.tensor(grad_mag, dtype=torch.float)
    stroke_width = grad_mag_tensor[mask].unsqueeze(1)  # Higher gradient -> thinner stroke
    
    # --- Compute curvature approximation ---
    # Second derivatives
    dxx = sobel(grad_x, axis=1)
    dyy = sobel(grad_y, axis=0)
    curvature = np.abs(dxx + dyy)
    curvature_tensor = torch.tensor(curvature, dtype=torch.float)[mask].unsqueeze(1)
    
    # Combine features: [pixel_value, y_pos, x_pos, stroke_width, curvature]
    x = torch.cat([pixel_values, positions_norm, stroke_width, curvature_tensor], dim=1)
    
    # --- Build edges: connect each node to immediate neighbors (4-connectivity) ---
    indices = positions.long()
    edge_index_list = []
    for i, (y, x_pos) in enumerate(indices):
        neighbors = [(y-1, x_pos), (y+1, x_pos), (y, x_pos-1), (y, x_pos+1)]
        for ny, nx in neighbors:
            mask_indices = torch.where((indices[:,0]==ny) & (indices[:,1]==nx))[0]
            if len(mask_indices) > 0:
                edge_index_list.append([i, mask_indices[0].item()])
    
    if len(edge_index_list) == 0:
        raise ValueError("No edges found. Check threshold or image.")
    
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_index, _ = remove_self_loops(edge_index)
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index)
    
    return data


# def image_to_graph(img_tensor):
#     """
#     Convert image tensor to graph
    
#     Args:
#         img_tensor: torch.Tensor (C, H, W) or (H, W)
    
#     Returns:
#         Data object with x and edge_index
#     """
#     # Ensure it's a tensor
#     if isinstance(img_tensor, np.ndarray):
#         img_tensor = torch.from_numpy(img_tensor).float()
    
#     # Add channel dimension if needed
#     if len(img_tensor.shape) == 2:
#         img_tensor = img_tensor.unsqueeze(0)
    
#     # Use existing tensor directly (no resize)
#     img = img_tensor.squeeze().numpy()
    
#     # Convert to graph
#     graph_coo = img_to_graph(img, return_as=sp.coo_array)
    
#     # Extract edge_index
#     edge_index = torch.tensor(
#         np.vstack([graph_coo.row, graph_coo.col]),
#         dtype=torch.long
#     )
    
#     # Remove self-loops
#     edge_index, _ = remove_self_loops(edge_index)
    
#     # Create node features
#     H, W = img.shape
#     pixel_values = img.flatten()
    
#     # Position features (normalized)
#     y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
#     positions = np.stack([y_coords.flatten(), x_coords.flatten()], axis=1)
#     positions_norm = positions / np.array([H, W])
    
#     # Combine features: [pixel_value, y_pos, x_pos]
#     x = torch.tensor(
#         np.column_stack([pixel_values, positions_norm]),
#         dtype=torch.float
#     )
    
#     # Create Data object
#     data = Data(x=x, edge_index=edge_index)
    
#     return data
