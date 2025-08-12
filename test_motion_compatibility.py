#!/usr/bin/env python3
"""
Test script to verify motion file compatibility between convert_to_isaac.py and motion_lib.py
"""

import torch
from pathlib import Path
from poselib.skeleton.skeleton3d import SkeletonMotion

def test_motion_file_compatibility(motion_file_path: str):
    """Test if a motion file can be loaded by SkeletonMotion.from_file()"""
    print(f"Testing motion file: {motion_file_path}")
    
    try:
        # Try to load the motion file using the same method as motion_lib.py
        motion = SkeletonMotion.from_file(motion_file_path)
        
        print("✅ Successfully loaded motion file!")
        print(f"  - FPS: {motion.fps}")
        print(f"  - Number of frames: {motion.global_translation.shape[0]}")
        print(f"  - Number of joints: {motion.global_translation.shape[1]}")
        print(f"  - Is local: {motion.is_local}")
        print(f"  - Skeleton tree nodes: {motion.skeleton_tree.node_names}")
        
        # Test accessing key properties that motion_lib.py uses
        print("\nTesting key properties:")
        print(f"  - global_translation shape: {motion.global_translation.shape}")
        print(f"  - global_rotation shape: {motion.global_rotation.shape}")
        print(f"  - local_rotation shape: {motion.local_rotation.shape}")
        print(f"  - global_velocity shape: {motion.global_velocity.shape}")
        print(f"  - global_angular_velocity shape: {motion.global_angular_velocity.shape}")
        print(f"  - root_translation shape: {motion.root_translation.shape}")
        
        # Test data types and values
        print("\nTesting data integrity:")
        print(f"  - global_translation dtype: {motion.global_translation.dtype}")
        print(f"  - global_rotation dtype: {motion.global_rotation.dtype}")
        print(f"  - Has NaN values: {torch.isnan(motion.global_translation).any()}")
        print(f"  - Has infinite values: {torch.isinf(motion.global_translation).any()}")
        
        # Test quaternion normalization
        quat_norms = torch.norm(motion.global_rotation, dim=-1)
        print(f"  - Quaternion norms (should be ~1.0): min={quat_norms.min():.6f}, max={quat_norms.max():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load motion file: {e}")
        return False


def test_motion_lib_compatibility(motion_file_path: str):
    """Test if the motion file works with motion_lib.py"""
    print("Testing motion_lib.py compatibility...")
    
    try:
        # Import motion_lib
        from protomotions.utils.motion_lib import MotionLib
        
        # Create a minimal robot config for testing
        class MockRobotConfig:
            def __init__(self):
                self.dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 8]  # Joint body IDs
                self.dof_offsets = [0, 3, 6, 9, 12, 15, 18, 21, 24]  # DOF offsets
                self.num_dof = 24  # Total DOFs
                self.joint_axis = ["x", "y", "z"] * 8  # Joint axes
        
        robot_config = MockRobotConfig()
        key_body_ids = torch.tensor([0, 1], dtype=torch.long)  # Pelvis and first joint
        
        # Try to create MotionLib instance
        motion_lib = MotionLib(
            motion_file=motion_file_path,
            robot_config=robot_config,
            key_body_ids=key_body_ids,
            device="cpu",
            target_frame_rate=30
        )
        
        print("✅ Successfully created MotionLib instance!")
        print(f"  - Number of motions: {motion_lib.num_motions()}")
        print(f"  - Total motion length: {motion_lib.get_total_length():.3f}s")
        
        # Test sampling
        motion_ids = motion_lib.sample_motions(n=1)
        motion_times = motion_lib.sample_time(motion_ids)
        motion_state = motion_lib.get_motion_state(motion_ids, motion_times)
        
        print("✅ Successfully sampled motion state!")
        print(f"  - Root position shape: {motion_state.root_pos.shape}")
        print(f"  - DOF position shape: {motion_state.dof_pos.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to use with motion_lib.py: {e}")
        return False


def main():
    """Main test function"""
    # Test with the motion file from the error output
    motion_file = "ProtoMotions/data/motions/retargeted_motion_fixed.npy"
    
    if not Path(motion_file).exists():
        print(f"Motion file not found: {motion_file}")
        print("Please run convert_to_isaac.py first to generate a motion file.")
        return
    
    print("=" * 60)
    print("MOTION FILE COMPATIBILITY TEST")
    print("=" * 60)
    
    # Test basic loading
    basic_ok = test_motion_file_compatibility(motion_file)
    
    if basic_ok:
        # Test motion_lib compatibility
        lib_ok = test_motion_lib_compatibility(motion_file)
        
        if lib_ok:
            print("\n🎉 All tests passed! The motion file is fully compatible.")
        else:
            print("\n⚠️  Basic loading works but motion_lib.py has issues.")
    else:
        print("\n❌ Basic loading failed. Check the motion file format.")


if __name__ == "__main__":
    main() 