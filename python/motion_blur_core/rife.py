import torch
import numpy as np

class RIFEModel:
    def __init__(self, device="mps"):
        self.device = device
        # Ensure PyTorch uses MPS if available
        if self.device == "mps" and not torch.backends.mps.is_available():
            self.device = "cpu"
        # Pre-allocate dummy to check initialization
        self.dummy_tensor = torch.zeros((1, 3, 256, 256)).to(self.device)

    def interpolate(self, img1, img2, ratio):
        """
        img1, img2 are numpy arrays (H, W, 3).
        ratio is between 0 and 1.
        
        In a full RIFE implementation, we would run the IFNet model here.
        For MVP, this performs a PyTorch-accelerated linear crossfade simulation.
        """
        # Convert to tensors
        t1 = torch.from_numpy(img1).float().to(self.device)
        t2 = torch.from_numpy(img2).float().to(self.device)
        
        # Warp/interp simulation
        res = t1 * (1.0 - ratio) + t2 * ratio
        
        # Back to numpy
        res_np = res.cpu().numpy().astype(np.uint8)
        return res_np
