#!/usr/bin/env python3

import torch
import numpy as np
from pathlib import Path

def main():
    print("🚀 Starting foot detection...")
    
    motion_file_path = "output/smpl_lower_retargeted_treadmill_example.npy"
    
    if not Path(motion_file_path).exists():
        print(f"❌ Motion file not found: {motion_file_path}")
        return
    
    print(f"📁 Loading motion from: {motion_file_path}")
    
    try:
        # Direct numpy loading to avoid hanging
        data = np.load(motion_file_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == ():
            # It's a pickled object
            motion_data = data.item()
            print(f"📦 Loaded pickled data with keys: {list(motion_data.keys())}")
            
            if 'global_translation' in motion_data:
                gt = motion_data['global_translation']
                print(f"🦴 Global translation shape: {gt.shape}")
                
                # Analyze first frame Z coordinates
                first_frame_z = gt[0, :, 2]
                print(f"📏 Z coordinates range: {first_frame_z.min():.4f} to {first_frame_z.max():.4f}")
                
                # Find lowest bodies (potential feet)
                sorted_indices = np.argsort(first_frame_z)
                print(f"\n🦶 Bodies sorted by height (lowest first):")
                for i in range(min(8, len(sorted_indices))):
                    idx = sorted_indices[i]
                    z = first_frame_z[idx]
                    print(f"  {i+1}. Body {idx:2d}: Z={z:.4f}")
                
                # Suggest foot IDs
                foot_ids = sorted_indices[:2]
                print(f"\n✅ Suggested foot body IDs: {foot_ids.tolist()}")
                print(f"   Foot Z heights: {first_frame_z[foot_ids]}")
                
            else:
                print("❌ No 'global_translation' found in data")
        else:
            print(f"📊 Direct numpy array shape: {data.shape}")
            
    except Exception as e:
        print(f"❌ Error loading motion file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
