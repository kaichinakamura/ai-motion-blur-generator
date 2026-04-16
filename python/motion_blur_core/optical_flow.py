import torch
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

class OpticalFlowExtractor:
    def __init__(self, device="mps"):
        self.device = device
        
        # Load pre-trained RAFT small model
        weights = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=weights, progress=False).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def compute_flow(self, img1: torch.Tensor, img2: torch.Tensor, max_size=768) -> torch.Tensor:
        """
        Computes forward optical flow from img1 to img2, using downsampling to accelerate processing.
        img1, img2: Tensors of shape [B, C, H, W] with pixel values in range [0, 1]
        Returns: flow tensor of shape [B, 2, H, W] containing (dx, dy)
        """
        import torch.nn.functional as F
        
        B, C, H, W = img1.shape
        scale_factor = 1.0
        
        if max(H, W) > max_size:
            scale_factor = max_size / max(H, W)
            new_H = int(H * scale_factor)
            new_W = int(W * scale_factor)
            
            # RAFT requires height and width to be multiples of 8
            new_H = (new_H // 8) * 8
            new_W = (new_W // 8) * 8
            
            i1 = F.interpolate(img1, size=(new_H, new_W), mode='bilinear', align_corners=False)
            i2 = F.interpolate(img2, size=(new_H, new_W), mode='bilinear', align_corners=False)
        else:
            i1 = img1
            i2 = img2
            new_H, new_W = H, W

        # RAFT generally assumes inputs in [-1, 1] range. 
        i1 = i1 * 2.0 - 1.0
        i2 = i2 * 2.0 - 1.0
        
        list_of_flows = self.model(i1, i2)
        final_flow = list_of_flows[-1]
        
        if scale_factor != 1.0:
            # Scale flow back up to original size
            final_flow = F.interpolate(final_flow, size=(H, W), mode='bilinear', align_corners=False)
            
            # The flow values also need to be scaled up!
            scale_x = W / new_W
            scale_y = H / new_H
            final_flow[:, 0, :, :] *= scale_x
            final_flow[:, 1, :, :] *= scale_y
            
        return final_flow
