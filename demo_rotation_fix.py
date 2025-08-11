#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation as sRot

def demonstrate_rotation_fix():
    """
    Demonstrate the difference between the old and new rotation approaches
    """
    print("[DEMO] Demonstrating rotation conversion fix")
    print("=" * 50)
    
    # Example joint angles from MuJoCo (in radians)
    test_angles = np.array([
        [0.1, 0.2, 0.3],  # Frame 0: X, Y, Z rotations
        [0.0, 0.5, -0.1], # Frame 1: X, Y, Z rotations
    ])
    
    print("Test joint angles (X, Y, Z in radians):")
    for i, angles in enumerate(test_angles):
        print(f"  Frame {i}: [{angles[0]:.3f}, {angles[1]:.3f}, {angles[2]:.3f}]")
    
    print("\n[OLD APPROACH] Extrinsic rotation composition:")
    old_quaternions = []
    for angles in test_angles:
        # Old approach: separate axis rotations then compose
        rot_x = sRot.from_rotvec([angles[0], 0, 0])
        rot_y = sRot.from_rotvec([0, angles[1], 0]) 
        rot_z = sRot.from_rotvec([0, 0, angles[2]])
        
        combined_rot = rot_z * rot_y * rot_x  # Extrinsic composition
        quat_xyzw = combined_rot.as_quat()
        quat_wxyz = np.roll(quat_xyzw, 1)  # Convert to WXYZ
        old_quaternions.append(quat_wxyz)
        
        print(f"  WXYZ quat: [{quat_wxyz[0]:.4f}, {quat_wxyz[1]:.4f}, {quat_wxyz[2]:.4f}, {quat_wxyz[3]:.4f}]")
    
    print("\n[NEW APPROACH] Intrinsic XYZ Euler:")
    new_quaternions = []
    for angles in test_angles:
        # New approach: direct intrinsic XYZ Euler conversion
        rotation = sRot.from_euler('xyz', angles, degrees=False)
        quat_xyzw = rotation.as_quat()
        quat_wxyz = np.roll(quat_xyzw, 1)  # Convert to WXYZ
        new_quaternions.append(quat_wxyz)
        
        print(f"  WXYZ quat: [{quat_wxyz[0]:.4f}, {quat_wxyz[1]:.4f}, {quat_wxyz[2]:.4f}, {quat_wxyz[3]:.4f}]")
    
    print("\n[COMPARISON] Differences:")
    for i, (old_q, new_q) in enumerate(zip(old_quaternions, new_quaternions)):
        diff = np.abs(old_q - new_q)
        max_diff = np.max(diff)
        print(f"  Frame {i} max difference: {max_diff:.6f}")
        if max_diff > 0.001:
            print(f"    OLD: [{old_q[0]:.4f}, {old_q[1]:.4f}, {old_q[2]:.4f}, {old_q[3]:.4f}]")
            print(f"    NEW: [{new_q[0]:.4f}, {new_q[1]:.4f}, {new_q[2]:.4f}, {new_q[3]:.4f}]")
    
    print("\n[ANALYSIS]")
    print("The key difference is in rotation order and coordinate frames:")
    print("- OLD: Extrinsic rotations (each in world coordinates)")
    print("- NEW: Intrinsic XYZ Euler (matches MuJoCo joint convention)")
    print("- This should fix the anatomical positioning issue")
    print("=" * 50)

if __name__ == "__main__":
    demonstrate_rotation_fix()
