#!/usr/bin/env python3
"""
Debug script to analyze the motion file and understand why values are zero.
"""

import numpy as np
import torch
from collections import OrderedDict

def debug_motion_file(motion_file_path):
    """Debug the motion file to understand the data structure."""
    print(f"Debugging motion file: {motion_file_path}")
    
    # Load the motion file
    data = np.load(motion_file_path, allow_pickle=True).item()
    
    print("\n=== Motion File Structure ===")
    for key, value in data.items():
        print(f"{key}: {type(value)}")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {type(subvalue)}")
                if hasattr(subvalue, 'shape'):
                    print(f"    Shape: {subvalue.shape}")
                    print(f"    Min: {np.min(subvalue):.6f}")
                    print(f"    Max: {np.max(subvalue):.6f}")
                    print(f"    Mean: {np.mean(subvalue):.6f}")
                    print(f"    Std: {np.std(subvalue):.6f}")
                    print(f"    NaN count: {np.isnan(subvalue).sum()}")
                    print(f"    Zero count: {(subvalue == 0).sum()}")
                    print(f"    Non-zero count: {(subvalue != 0).sum()}")
                    if (subvalue != 0).sum() > 0:
                        print(f"    First few non-zero values: {subvalue[subvalue != 0][:10]}")
    
    print("\n=== Skeleton Tree ===")
    if 'skeleton_tree' in data:
        skeleton = data['skeleton_tree']
        print(f"Node names: {skeleton['node_names']}")
        print(f"Parent indices: {skeleton['parent_indices']}")
        if 'local_translation' in skeleton:
            local_trans = skeleton['local_translation']['arr']
            print(f"Local translation shape: {local_trans.shape}")
            print(f"Local translation values: {local_trans}")
    
    print("\n=== Motion Properties ===")
    print(f"FPS: {data.get('fps', 'Not found')}")
    print(f"Is local: {data.get('is_local', 'Not found')}")
    
    # Check if the motion data makes sense
    print("\n=== Data Validation ===")
    rotation_arr = data['rotation']['arr']
    root_trans_arr = data['root_translation']['arr']
    
    print(f"Rotation array shape: {rotation_arr.shape}")
    print(f"Root translation shape: {root_trans_arr.shape}")
    
    # Check if rotations are valid quaternions
    quat_norms = np.linalg.norm(rotation_arr, axis=-1)
    print(f"Quaternion norms - Min: {np.min(quat_norms):.6f}, Max: {np.max(quat_norms):.6f}")
    print(f"Quaternion norms - Mean: {np.mean(quat_norms):.6f}")
    
    # Check for identity quaternions
    identity_quat = np.array([0., 0., 0., 1.])
    identity_count = np.sum(np.allclose(rotation_arr, identity_quat, atol=1e-6), axis=0)
    print(f"Identity quaternions per joint: {identity_count}")
    
    # Check root translation
    print(f"Root translation - Min: {np.min(root_trans_arr):.6f}, Max: {np.max(root_trans_arr):.6f}")
    print(f"Root translation - Mean: {np.mean(root_trans_arr):.6f}")
    
    # Check if there's any motion at all
    root_motion = np.diff(root_trans_arr, axis=0)
    print(f"Root motion magnitude - Min: {np.min(np.linalg.norm(root_motion, axis=-1)):.6f}")
    print(f"Root motion magnitude - Max: {np.max(np.linalg.norm(root_motion, axis=-1)):.6f}")
    print(f"Root motion magnitude - Mean: {np.mean(np.linalg.norm(root_motion, axis=-1)):.6f}")

if __name__ == "__main__":
    debug_motion_file("data/motions/retargeted_motion_fixed.npy") 