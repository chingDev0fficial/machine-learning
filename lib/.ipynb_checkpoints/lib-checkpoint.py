import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
import os
from PIL import Image
from torch.utils.data import Dataset

from sklearn.feature_extraction import image


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

def image_to_graph(
    image_tensor: torch.Tensor,
    patch_size: int = 8,
    k_neighbors: int = 8,
    edge_threshold: float = 0.1,
    include_features: bool = True
) -> Data:
    """
    Convert an image to a graph representation with nodes and edges.
    
    Args:
        image_tensor: Input image tensor of shape (C, H, W) or (H, W)
        method: Graph construction method ('grid', 'knn', 'superpixel', 'region')
        patch_size: Size of patches for grid method
        k_neighbors: Number of neighbors for KNN method
        edge_threshold: Threshold for edge creation based on feature similarity
        include_features: Whether to include patch features as node features
        
    Returns:
        PyTorch Geometric Data object with node features and edge indices
    """
    
    return _image_to_grid_graph(image_tensor, patch_size, include_features)

def _image_to_grid_graph(
    image_tensor: torch.Tensor, 
    patch_size: int, 
    include_features: bool
) -> Data:
    """Convert image to grid-based graph where each patch is a node."""
    
    # Handle different input shapes
    if len(image_tensor.shape) == 2:
        image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension
    
    C, H, W = image_tensor.shape
    
    # Create patches
    patches_h = H // patch_size
    patches_w = W // patch_size
    
    # Extract patch features
    node_features = []
    node_positions = []
    
    for i in range(patches_h):
        for j in range(patches_w):
            # Extract patch
            patch = image_tensor[
                :, 
                i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size
            ]
            
            if include_features:
                # Compute patch statistics as features
                mean_val = patch.mean(dim=[1, 2])  # Per channel mean
                std_val = patch.std(dim=[1, 2])    # Per channel std
                max_val = patch.max(dim=2)[0].max(dim=1)[0]  # Per channel max
                min_val = patch.min(dim=2)[0].min(dim=1)[0]  # Per channel min
                
                features = torch.cat([mean_val, std_val, max_val, min_val])
                node_features.append(features)
            
            # Store position
            node_positions.append([i, j])
    
    # Create edges (connect adjacent patches)
    edge_indices = []
    
    for i in range(patches_h):
        for j in range(patches_w):
            current_node = i * patches_w + j
            
            # Connect to neighbors (4-connectivity)
            neighbors = [
                (i-1, j), (i+1, j),  # vertical neighbors
                (i, j-1), (i, j+1)   # horizontal neighbors
            ]
            
            # Add diagonal connections for 8-connectivity
            neighbors.extend([
                (i-1, j-1), (i-1, j+1),
                (i+1, j-1), (i+1, j+1)
            ])
            
            for ni, nj in neighbors:
                if 0 <= ni < patches_h and 0 <= nj < patches_w:
                    neighbor_node = ni * patches_w + nj
                    edge_indices.append([current_node, neighbor_node])
    
    # Convert to tensors
    if include_features:
        x = torch.stack(node_features)
    else:
        x = torch.tensor(node_positions, dtype=torch.float32)
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    pos = torch.tensor(node_positions, dtype=torch.float32)
    
    return Data(x=x, edge_index=edge_index, pos=pos)
