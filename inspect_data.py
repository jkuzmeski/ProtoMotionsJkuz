#!/usr/bin/env python3

"""
Simple data inspection to understand MuJoCo output structure.
"""

import numpy as np
import sys
import os

# Add the scripts directory to Python path
scripts_dir = os.path.join(os.path.dirname(__file__), 'data', 'scripts')
sys.path.append(scripts_dir)

def inspect_mujoco_data():
    """Inspect the MuJoCo retargeting output structure."""
    
    print("=== Inspecting MuJoCo Data Structure ===")
    
    # Expected file paths
    trans_file = "outputs/retargeted_motion_trans.npy"
    poses_file = "outputs/retargeted_motion_poses.npy"
    
    try:
        # Load the data files
        if os.path.exists(trans_file):
            trans = np.load(trans_file)
            print(f"✅ Loaded trans array: shape {trans.shape}")
            print(f"   First frame trans: {trans[0]}")
            print(f"   Root position: {trans[0, :3]}")
            print(f"   Root quaternion: {trans[0, 3:7]}")
        else:
            print(f"❌ Trans file not found: {trans_file}")
            
        if os.path.exists(poses_file):
            poses = np.load(poses_file)
            print(f"✅ Loaded poses array: shape {poses.shape}")
            print(f"   First frame poses: {poses[0]}")
            print(f"   Total joint angles per frame: {poses.shape[1]}")
            print(f"   Expected joints: {poses.shape[1] // 3} (if 3 DOF per joint)")
            
            # Show first few joint angles
            if poses.shape[1] >= 24:  # At least 8 joints * 3 DOF
                print("\n   Joint angle breakdown (first frame):")
                joint_names = ['L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe']
                for i, name in enumerate(joint_names):
                    start_idx = i * 3
                    end_idx = start_idx + 3
                    if end_idx <= poses.shape[1]:
                        angles = poses[0, start_idx:end_idx]
                        print(f"     {name:8}: angles {angles} (indices {start_idx}-{end_idx-1})")
                    else:
                        break
            
        else:
            print(f"❌ Poses file not found: {poses_file}")
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Change to the ProtoMotions directory first
    os.chdir(os.path.dirname(__file__))
    inspect_mujoco_data()
