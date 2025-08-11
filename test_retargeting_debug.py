#!/usr/bin/env python3

import numpy as np
from pathlib import Path

def test_retargeting_debug():
    """Simple test to debug the retargeting process"""
    
    # Check the debug files from the treadmill transform
    debug_file = Path("data/scripts/data2retarget/debug_motion_treadmill_transform.npz")
    if debug_file.exists():
        data = np.load(debug_file)
        print("=== TREADMILL TRANSFORM DEBUG ===")
        print(f"Body positions shape: {data['body_positions'].shape}")
        print(f"Body names: {list(data['body_names'])}")
        print(f"FPS: {data['fps']}")
        
        # Check range of positions
        positions = data['body_positions']
        for i, name in enumerate(data['body_names']):
            pos_range = positions[:, i, :]
            min_pos = np.min(pos_range, axis=0)
            max_pos = np.max(pos_range, axis=0)
            print(f"{name}: min={min_pos.round(3)}, max={max_pos.round(3)}")
        
        print("\n=== HEIGHT ANALYSIS ===")
        z_values = positions[:, :, 2]  # All Z coordinates
        print(f"Overall Z range: {np.min(z_values):.3f} to {np.max(z_values):.3f}")
        print(f"Ground contact (min Z per frame): {np.min(z_values, axis=1).min():.3f}")
    else:
        print(f"Debug file not found: {debug_file}")
    
    # Check if retargeted debug file exists
    retarget_debug_file = Path("data/scripts/data2retarget/debug_motion_retargeted.npz")
    if retarget_debug_file.exists():
        data = np.load(retarget_debug_file)
        print("\n=== RETARGETED DEBUG ===")
        print(f"Body positions shape: {data['body_positions'].shape}")
        print(f"Body names: {list(data['body_names'])}")
        print(f"FPS: {data['fps']}")
        
        # Check range of positions
        positions = data['body_positions']
        for i, name in enumerate(data['body_names']):
            pos_range = positions[:, i, :]
            min_pos = np.min(pos_range, axis=0)
            max_pos = np.max(pos_range, axis=0)
            print(f"{name}: min={min_pos.round(3)}, max={max_pos.round(3)}")
        
        print("\n=== HEIGHT ANALYSIS ===")
        z_values = positions[:, :, 2]  # All Z coordinates
        print(f"Overall Z range: {np.min(z_values):.3f} to {np.max(z_values):.3f}")
        print(f"Ground contact (min Z per frame): {np.min(z_values, axis=1).min():.3f}")
        
        # Check if joints are moving relative to pelvis
        pelvis_idx = list(data['body_names']).index('Pelvis')
        pelvis_pos = positions[:, pelvis_idx, :]
        
        print("\n=== JOINT MOVEMENT ANALYSIS ===")
        for i, name in enumerate(data['body_names']):
            if name != 'Pelvis':
                joint_pos = positions[:, i, :]
                relative_pos = joint_pos - pelvis_pos
                movement_range = np.max(relative_pos, axis=0) - np.min(relative_pos, axis=0)
                print(f"{name} relative movement range: {movement_range.round(3)}")
                
    else:
        print(f"Retargeted debug file not found: {retarget_debug_file}")

if __name__ == "__main__":
    test_retargeting_debug()
