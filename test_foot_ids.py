#!/usr/bin/env python3

import torch
import numpy as np
from protomotions.utils.motion_lib import MotionLib

def test_different_foot_ids():
    motion_file_path = "output/smpl_lower_retargeted_treadmill_example.npy"
    
    # Create robot config
    class MockRobotConfig:
        def __init__(self):
            self.dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 8]
            self.dof_offsets = [0, 3, 6, 9, 12, 15, 18, 21, 24]
            self.num_dof = 24
            self.joint_axis = ["x", "y", "z"] * 8

    robot_config = MockRobotConfig()
    
    # Try different foot ID combinations
    foot_candidates = [
        ([5, 6], "Ankle joints"),
        ([7, 8], "Foot/toe joints"), 
        ([9, 10], "End effector joints"),
        ([6, 8], "Right ankle + foot"),
        ([5, 7], "Left ankle + foot")
    ]
    
    for foot_ids, description in foot_candidates:
        print(f"\n🦶 Testing {description}: body IDs {foot_ids}")
        
        try:
            key_body_ids = torch.tensor(foot_ids, dtype=torch.long)
            
            motion_lib = MotionLib(
                motion_file=motion_file_path,
                robot_config=robot_config,
                key_body_ids=key_body_ids,
                device="cpu",
                target_frame_rate=30
            )
            
            # Sample motion state
            motion_ids = motion_lib.sample_motions(n=1)
            motion_times = motion_lib.sample_time(motion_ids) 
            motion_state = motion_lib.get_motion_state(motion_ids, motion_times)
            
            # Check foot heights
            foot_positions = motion_state.key_body_pos[0, :, 2]
            print(f"   Foot Z coordinates: {foot_positions}")
            
            if torch.all(foot_positions < 0.15):
                print("   ✅ SUCCESS! These look like feet on ground!")
                print(f"   💡 Use key_body_ids = torch.tensor({foot_ids})")
                break
            else:
                print("   ❌ Still floating - not correct foot IDs")
                
        except Exception as e:
            print(f"   ❌ Error with IDs {foot_ids}: {e}")

if __name__ == "__main__":
    test_different_foot_ids()
