#!/usr/bin/env python3

"""
Check the rotation data in the latest retargeted motion.
"""

import numpy as np

def check_rotation_data():
    """Check what rotations were generated."""
    
    print("=== Checking Rotation Data ===")
    
    try:
        # Load the latest retargeted motion
        motion = np.load('data/motions/retargeted_with_rotations.npy', allow_pickle=True).item()
        
        print(f"Motion keys: {motion.keys()}")
        
        if 'rotation' in motion:
            rotations = motion['rotation']
            print(f"Rotation type: {type(rotations)}")
            
            # Handle if rotation is a dict (as it appears to be)
            if isinstance(rotations, dict):
                if 'arr' in rotations:
                    rotations = rotations['arr']
                    print(f"Found rotation array in dict")
                else:
                    print(f"Rotation dict keys: {rotations.keys()}")
                    return
            
            print(f"Rotation shape: {rotations.shape}")
            print(f"Number of frames: {rotations.shape[0]}")
            print(f"Number of joints: {rotations.shape[1]}")
            
            print("\n=== First Frame Rotations ===")
            first_frame = rotations[0]
            
            joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 
                          'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe']
            
            for i, (name, quat) in enumerate(zip(joint_names, first_frame)):
                is_identity = np.allclose(quat, [1, 0, 0, 0], atol=1e-6)
                magnitude = np.linalg.norm(quat)
                print(f"Joint {i:2} ({name:8}): {quat} | Identity: {is_identity} | Mag: {magnitude:.6f}")
            
            # Count non-identity quaternions
            identity_count = 0
            non_identity_count = 0
            
            for quat in first_frame:
                if np.allclose(quat, [1, 0, 0, 0], atol=1e-6):
                    identity_count += 1
                else:
                    non_identity_count += 1
            
            print(f"\n=== Summary ===")
            print(f"Identity quaternions: {identity_count}")
            print(f"Non-identity quaternions: {non_identity_count}")
            print(f"Success rate: {non_identity_count}/{len(first_frame)} joints have meaningful rotations")
            
            if non_identity_count > 0:
                print("✅ SUCCESS: Joint rotations are being extracted from MuJoCo!")
            else:
                print("❌ PROBLEM: All rotations are identity - extraction failed")
                
        else:
            print("❌ No rotation data found in motion file")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_rotation_data()
