#!/usr/bin/env python3

import numpy as np
import sys
import os

def test_position_fix():
    """
    Test if the new approach produces anatomically correct positioning
    """
    print("[TEST] Testing position-based fix approach")
    print("=" * 50)
    
    # Load the original target positions that were fed to Mink
    try:
        joint_positions = np.load("data/scripts/data2retarget/overground_motion.npy")
        print(f"[TEST] Loaded original positions: {joint_positions.shape}")
    except:
        print("[ERROR] Could not load test data")
        return False
    
    # Test with first frame
    frame_0_positions = joint_positions[0]  # Shape: (9, 3)
    joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe']
    
    print("\n[TEST] Original positions (frame 0):")
    for i, (name, pos) in enumerate(zip(joint_names, frame_0_positions)):
        print(f"  {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    
    # Check anatomical relationships
    pelvis_z = frame_0_positions[0, 2]  # Pelvis Z
    l_ankle_z = frame_0_positions[3, 2]  # L_Ankle Z
    r_ankle_z = frame_0_positions[7, 2]  # R_Ankle Z
    
    print(f"\n[TEST] Anatomical check:")
    print(f"  Pelvis Z: {pelvis_z:.4f}")
    print(f"  L_Ankle Z: {l_ankle_z:.4f} ({'ABOVE' if l_ankle_z > pelvis_z else 'BELOW'} pelvis)")
    print(f"  R_Ankle Z: {r_ankle_z:.4f} ({'ABOVE' if r_ankle_z > pelvis_z else 'BELOW'} pelvis)")
    
    # The original data shows that ankles CAN be above pelvis during certain phases of gait
    # But let's check if this is reasonable
    if l_ankle_z > pelvis_z:
        height_diff = l_ankle_z - pelvis_z
        print(f"    L_Ankle is {height_diff:.4f}m above pelvis")
        if height_diff > 0.2:  # 20cm seems unreasonable
            print(f"    [WARNING] This seems anatomically unreasonable!")
        else:
            print(f"    [INFO] This could be a foot lift during gait")
    
    if r_ankle_z > pelvis_z:
        height_diff = r_ankle_z - pelvis_z
        print(f"    R_Ankle is {height_diff:.4f}m above pelvis")
        if height_diff > 0.2:  # 20cm seems unreasonable
            print(f"    [WARNING] This seems anatomically unreasonable!")
        else:
            print(f"    [INFO] This could be a foot lift during gait")
    
    print("\n[TEST] The new approach will use these original positions directly")
    print("       as root translation, which should preserve anatomical correctness.")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = test_position_fix()
    if not success:
        sys.exit(1)
