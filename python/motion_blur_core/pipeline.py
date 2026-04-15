import os
import subprocess
import tempfile
import sys
import shutil
import cv2
import numpy as np
from .rife import RIFEModel

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

def process_video(input_path, output_path, shutter_angle=180.0, progress_cb=None):
    if progress_cb: progress_cb(0, "Initializing RIFE on PyTorch (MPS)")
    
    rife = RIFEModel(device="mps")
    
    factor = 8 # interpolate up to 8x
    
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
        
        if progress_cb: progress_cb(15, "Starting AI frame interpolation")
        
        # Compute subframes to blend based on shutter_angle
        # 360 deg = factor frames. 180 deg = factor/2 frames.
        use_factor = max(1, int(factor * (shutter_angle / 360.0)))
        
        for i in range(num_frames - 1):
            img1 = cv2.imread(frames[i])
            img2 = cv2.imread(frames[i+1])
            
            subframes = []
            for j in range(factor):
                ratio = j / factor
                interp = rife.interpolate(img1, img2, ratio)
                subframes.append(interp)
                
            # Blending (Weighted averaging of the subframes)
            start_idx = (factor - use_factor) // 2
            end_idx = start_idx + use_factor
            
            blend_frames = subframes[start_idx:end_idx]
            if len(blend_frames) == 0:
                blend_frames = [subframes[0]]
                
            avg_img = blend_frames[0].astype(float)
            for b in blend_frames[1:]:
                avg_img += b.astype(float)
            avg_img /= len(blend_frames)
            
            out_img = avg_img.astype(np.uint8)
            out_file = os.path.join(blurred_dir, f"{i:08d}.jpg")
            cv2.imwrite(out_file, out_img)
            
            if progress_cb and i % max(1, (num_frames // 20)) == 0:
                p = 15 + 75 * (i / num_frames)
                progress_cb(int(p), f"Interpolating and blending ({i}/{num_frames})")
                
        # Handle last frame
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
