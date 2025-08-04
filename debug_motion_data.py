#!/usr/bin/env python3
"""
Diagnostic script to debug motion data conversion issues
"""

import numpy as np
import torch
from pathlib import Path
from poselib.skeleton.skeleton3d import SkeletonMotion

def analyze_motion_file(motion_file_path: str):
    """Analyze a motion file to understand its content and structure"""
    print(f"Analyzing motion file: {motion_file_path}")
    
    try:
        # Load the motion file
        motion = SkeletonMotion.from_file(motion_file_path)
        
        print("✅ Motion file loaded successfully!")
        print(f"  - FPS: {motion.fps}")
        print(f"  - Number of frames: {motion.global_translation.shape[0]}")
        print(f"  - Number of joints: {motion.global_translation.shape[1]}")
        print(f"  - Is local: {motion.is_local}")
        print(f"  - Skeleton tree nodes: {motion.skeleton_tree.node_names}")
        
        # Analyze root translation
        root_trans = motion.root_translation
        print(f"\nRoot Translation Analysis:")
        print(f"  - Shape: {root_trans.shape}")
        print(f"  - Dtype: {root_trans.dtype}")
        print(f"  - Min: {root_trans.min():.6f}")
        print(f"  - Max: {root_trans.max():.6f}")
        print(f"  - Mean: {root_trans.mean():.6f}")
        print(f"  - Std: {root_trans.std():.6f}")
        print(f"  - First frame: {root_trans[0]}")
        print(f"  - Last frame: {root_trans[-1]}")
        
        # Check if root translation is meaningful
        if torch.allclose(root_trans, torch.zeros_like(root_trans), atol=1e-6):
            print("  ⚠️  WARNING: Root translation is essentially all zeros!")
        elif root_trans.std() < 1e-3:
            print("  ⚠️  WARNING: Root translation has very low variance!")
        else:
            print("  ✅ Root translation appears to have meaningful motion")
        
        # Analyze global rotation
        global_rot = motion.global_rotation
        print(f"\nGlobal Rotation Analysis:")
        print(f"  - Shape: {global_rot.shape}")
        print(f"  - Dtype: {global_rot.dtype}")
        
        # Check quaternion norms
        quat_norms = torch.norm(global_rot, dim=-1)
        print(f"  - Quaternion norms - Min: {quat_norms.min():.6f}, Max: {quat_norms.max():.6f}")
        
        if torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-3):
            print("  ✅ Quaternions are properly normalized")
        else:
            print("  ⚠️  WARNING: Quaternions are not properly normalized!")
        
        # Check if rotations are meaningful
        root_rot = global_rot[:, 0]  # Root joint
        if torch.allclose(root_rot, torch.zeros_like(root_rot), atol=1e-6):
            print("  ⚠️  WARNING: Root rotation is essentially all zeros!")
        elif torch.allclose(root_rot, torch.tensor([0., 0., 0., 1.]), atol=1e-3):
            print("  ⚠️  WARNING: Root rotation is essentially identity!")
        else:
            print("  ✅ Root rotation appears to have meaningful motion")
        
        # Analyze global translation
        global_trans = motion.global_translation
        print(f"\nGlobal Translation Analysis:")
        print(f"  - Shape: {global_trans.shape}")
        print(f"  - Dtype: {global_trans.dtype}")
        print(f"  - Min: {global_trans.min():.6f}")
        print(f"  - Max: {global_trans.max():.6f}")
        print(f"  - Mean: {global_trans.mean():.6f}")
        print(f"  - Std: {global_trans.std():.6f}")
        
        # Check if global translation is meaningful
        if torch.allclose(global_trans, torch.zeros_like(global_trans), atol=1e-6):
            print("  ⚠️  WARNING: Global translation is essentially all zeros!")
        elif global_trans.std() < 1e-3:
            print("  ⚠️  WARNING: Global translation has very low variance!")
        else:
            print("  ✅ Global translation appears to have meaningful motion")
        
        # Analyze velocities
        global_vel = motion.global_velocity
        print(f"\nGlobal Velocity Analysis:")
        print(f"  - Shape: {global_vel.shape}")
        print(f"  - Min: {global_vel.min():.6f}")
        print(f"  - Max: {global_vel.max():.6f}")
        print(f"  - Mean: {global_vel.mean():.6f}")
        print(f"  - Std: {global_vel.std():.6f}")
        
        if torch.allclose(global_vel, torch.zeros_like(global_vel), atol=1e-6):
            print("  ⚠️  WARNING: Global velocity is essentially all zeros!")
        else:
            print("  ✅ Global velocity appears to have meaningful motion")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to analyze motion file: {e}")
        return False

def compare_motion_files(file1: str, file2: str):
    """Compare two motion files to see differences"""
    print(f"\nComparing motion files:")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")
    
    try:
        motion1 = SkeletonMotion.from_file(file1)
        motion2 = SkeletonMotion.from_file(file2)
        
        print(f"\nComparison Results:")
        print(f"  FPS: {motion1.fps} vs {motion2.fps}")
        print(f"  Frames: {motion1.global_translation.shape[0]} vs {motion2.global_translation.shape[0]}")
        print(f"  Joints: {motion1.global_translation.shape[1]} vs {motion2.global_translation.shape[1]}")
        
        # Compare root translations
        root_trans1 = motion1.root_translation
        root_trans2 = motion2.root_translation
        
        print(f"\nRoot Translation Comparison:")
        print(f"  File 1 - Min: {root_trans1.min():.6f}, Max: {root_trans1.max():.6f}")
        print(f"  File 2 - Min: {root_trans2.min():.6f}, Max: {root_trans2.max():.6f}")
        
        # Check if they're different
        if torch.allclose(root_trans1, root_trans2, atol=1e-6):
            print("  ⚠️  WARNING: Root translations are essentially identical!")
        else:
            print("  ✅ Root translations are different")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to compare motion files: {e}")
        return False

def main():
    """Main diagnostic function"""
    # Test with the motion file from the error output
    motion_file = "data/motions/retargeted_motion.npy"
    
    if not Path(motion_file).exists():
        print(f"Motion file not found: {motion_file}")
        print("Please run convert_to_isaac.py first to generate a motion file.")
        return
    
    print("=" * 60)
    print("MOTION DATA DIAGNOSTIC")
    print("=" * 60)
    
    # Analyze the motion file
    analyze_motion_file(motion_file)
    
    # If there's a fixed version, compare them
    fixed_file = "data/motions/smpl_humanoid_walk.npy"
    if Path(fixed_file).exists():
        compare_motion_files(motion_file, fixed_file)

if __name__ == "__main__":
    main() 