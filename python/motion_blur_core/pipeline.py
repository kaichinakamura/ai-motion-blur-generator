import os
import subprocess
import tempfile
import shutil
import cv2
import torch
import numpy as np
from .rife import RIFEModel
from .vector_blur import VectorBlurEngine
from .optical_flow import OpticalFlowExtractor

def extract_frames(video_path, out_dir):
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-i", video_path,
        "-qscale:v", "2",
        os.path.join(out_dir, "%08d.jpg")
    ]
    subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    frames = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('.jpg')])
    return frames

def get_fps(video_path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of",
        "default=noprint_wrappers=1:nokey=1", video_path
    ]
    result = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True, check=True)
    num, den = result.stdout.strip().split('/')
    return float(num) / float(den)

def process_video(input_path, output_path, shutter_angle=180.0, use_rife=False, flow_resolution=720, progress_cb=None, cancel_event=None):
    if progress_cb: progress_cb(0, "Initializing ML Models on PyTorch (MPS)")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rife = RIFEModel(device=device) if use_rife else None
    vblur = VectorBlurEngine(device=device)
    flow_ext = OpticalFlowExtractor(device=device)
    
    temp_dir = tempfile.mkdtemp()
    extracted_dir = os.path.join(temp_dir, "extracted")
    os.makedirs(extracted_dir, exist_ok=True)
    blurred_dir = os.path.join(temp_dir, "blurred")
    os.makedirs(blurred_dir, exist_ok=True)
    
    try:
        if progress_cb: progress_cb(5, "Extracting frames with FFmpeg")
        frames = extract_frames(input_path, extracted_dir)
        num_frames = len(frames)
        fps = get_fps(input_path)
        
        if progress_cb: progress_cb(15, "Starting Hybrid Vector Blur pipeline")
        
        if use_rife:
            factor = 4
            use_factor = max(1, int(factor * (shutter_angle / 360.0)))
        else:
            factor = 1
            use_factor = 1
        
        for i in range(num_frames - 1):
            if cancel_event and cancel_event.is_set():
                break
                
            img1_bgr = cv2.imread(frames[i])
            img2_bgr = cv2.imread(frames[i+1])
            
            if use_rife:
                subframes_bgr = []
                for j in range(factor):
                    ratio = j / factor
                    interp = rife.interpolate(img1_bgr, img2_bgr, ratio)
                    subframes_bgr.append(interp)
                subframes_bgr.append(img2_bgr)
                start_idx = (factor - use_factor) // 2
                end_idx = start_idx + use_factor
            else:
                subframes_bgr = [img1_bgr, img2_bgr]
                start_idx = 0
                end_idx = 1
            
            blend_tensors = []
            
            for j in range(start_idx, end_idx):
                sf_curr = subframes_bgr[j]
                sf_next = subframes_bgr[j+1]
                
                # Convert BGR -> RGB -> Tensor [1, 3, H, W]
                sf_curr_rgb = cv2.cvtColor(sf_curr, cv2.COLOR_BGR2RGB)
                sf_next_rgb = cv2.cvtColor(sf_next, cv2.COLOR_BGR2RGB)
                
                curr_t = torch.from_numpy(sf_curr_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                next_t = torch.from_numpy(sf_next_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                
                # 1. Optical Flow extraction
                # flow represents vector from curr -> next
                calc_res = flow_resolution if flow_resolution > 0 else 999999
                flow = flow_ext.compute_flow(curr_t, next_t, max_size=calc_res)
                
                # 2. Vector Blur
                current_shutter = 360.0 if use_rife else shutter_angle
                samples = 11 if use_rife else 25
                blurred_t = vblur.apply_vector_blur(curr_t, flow, shutter_angle=current_shutter, num_samples=samples)
                blend_tensors.append(blurred_t)
                
            # 3. Combine subframes
            if use_rife and len(blend_tensors) > 1:
                avg_t = torch.mean(torch.stack(blend_tensors, dim=0), dim=0) # [1, 3, H, W]
            else:
                avg_t = blend_tensors[0]
            
            # Output Numpy back to disk
            out_img_rgb = (avg_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            out_img_bgr = cv2.cvtColor(out_img_rgb, cv2.COLOR_RGB2BGR)
            
            out_file = os.path.join(blurred_dir, f"{i:08d}.jpg")
            cv2.imwrite(out_file, out_img_bgr)
            
            if progress_cb:
                p = 15 + 75 * (i / max(1, num_frames - 2))
                progress_cb(int(p), f"Vector Blurring ({i+1}/{num_frames-1})")
                
        # Last frame
        if num_frames > 0:
            shutil.copy(frames[-1], os.path.join(blurred_dir, f"{num_frames-1:08d}.jpg"))
            
        if progress_cb: progress_cb(90, "Reassembling final video")
        
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-framerate", str(fps),
            "-i", os.path.join(blurred_dir, "%08d.jpg"),
            "-i", input_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy",
            "-shortest",
            output_path
        ]
        subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        if progress_cb: progress_cb(100, "Processing complete")
        
    finally:
        shutil.rmtree(temp_dir)
