import torch
import torch.nn.functional as F

class VectorBlurEngine:
    def __init__(self, device="mps"):
        self.device = device
        
    def apply_vector_blur(self, 
                          image_tensor: torch.Tensor, 
                          flow_tensor: torch.Tensor, 
                          shutter_angle: float, 
                          num_samples: int = 15) -> torch.Tensor:
        """
        Applies directional blur along the motion vector using grid_sample.
        image_tensor: [B, C, H, W] in [0.0, 1.0] range
        flow_tensor:  [B, 2, H, W] (dx, dy in pixels)
        shutter_angle: Length of the blur trail (e.g. 180.0 means half the displacement)
        """
        B, C, H, W = image_tensor.shape
        
        # Scale flow length based on shutter angle 
        # (360 degrees = full distance of the vector)
        blur_factor = shutter_angle / 360.0
        flow_scaled = flow_tensor * blur_factor
        
        # Create base grid for grid_sample [-1, 1] x [-1, 1]
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing='ij'
        )
        base_grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0) # [1, H, W, 2]
        if B > 1:
            base_grid = base_grid.expand(B, -1, -1, -1)
            
        # Convert pixel flow to normalized coordinates [-1, 1]
        # x direction needs to be scaled by 2 / W, y by 2 / H
        norm_factor = torch.tensor([2.0 / W, 2.0 / H], device=self.device).view(1, 1, 1, 2)
        
        flow_permuted = flow_scaled.permute(0, 2, 3, 1) # [B, H, W, 2]
        flow_norm = flow_permuted * norm_factor
        
        blurred_image = torch.zeros_like(image_tensor)
        
        # Vector sampling: sample points uniformly along the vector string.
        # Shift range is [-0.5, 0.5] from the exact pixel position.
        for i in range(num_samples):
            t = (i / max(1, num_samples - 1)) - 0.5
            
            sample_grid = base_grid + flow_norm * t
            
            sampled = F.grid_sample(
                image_tensor, 
                sample_grid, 
                mode='bilinear', 
                padding_mode='reflection', 
                align_corners=True
            )
            blurred_image += sampled
            
        return blurred_image / num_samples
