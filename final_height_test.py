#!/usr/bin/env python3

"""
Comprehensive test to identify why heights might still be wrong in Isaac Lab.
"""

import sys
sys.path.append('d:/Isaac/IsaacLab2.1.2/source')

import numpy as np
import torch
from isaaclab.utils.motion import SkeletonMotion

def comprehensive_height_test():
    """Check all aspects that could affect height in Isaac Lab."""
    
    print("=== COMPREHENSIVE HEIGHT ANALYSIS ===\n")
    
    # Load the motion
    motion_file = "data/scripts/data2retarget/retargeted_motion.npy"
    print(f"Loading: {motion_file}")
    
    try:
        motion = SkeletonMotion.from_file(motion_file)
        print("✅ Motion loaded successfully\n")
        
        # 1. Check root translation heights
        print("1️⃣ ROOT TRANSLATION ANALYSIS:")
        root_pos = motion.root_translation[:5].numpy()
        for i in range(5):
            print(f"   Frame {i}: [{root_pos[i, 0]:.3f}, {root_pos[i, 1]:.3f}, {root_pos[i, 2]:.3f}]")
        print(f"   Average height: {motion.root_translation[:, 2].mean():.3f}m")
        print(f"   Height range: {motion.root_translation[:, 2].min():.3f} to {motion.root_translation[:, 2].max():.3f}m\n")
        
        # 2. Check global translation heights
        print("2️⃣ GLOBAL TRANSLATION ANALYSIS:")
        global_pos = motion.global_translation[0].numpy()
        joint_names = motion.skeleton_tree.node_names
        
        print("   Joint heights (frame 0):")
        for i, name in enumerate(joint_names):
            height = global_pos[i, 2]
            print(f"     {name}: {height:.3f}m")
        
        min_joint_height = global_pos[:, 2].min()
        max_joint_height = global_pos[:, 2].max()
        print(f"   Minimum joint height: {min_joint_height:.3f}m")
        print(f"   Maximum joint height: {max_joint_height:.3f}m\n")
        
        # 3. Check what fix_motion_heights would do
        print("3️⃣ FIX_MOTION_HEIGHTS IMPACT:")
        body_heights = motion.global_translation[..., 2]
        min_height = body_heights.min()
        print(f"   Minimum height across all frames: {min_height:.3f}m")
        print(f"   If fix_motion_heights applied:")
        adjusted_root = motion.root_translation[0, 2] - min_height
        print(f"     Root height would become: {adjusted_root:.3f}m")
        print(f"     Minimum joint would be at: 0.000m")
        
        if min_height < 0.05:
            print("   ⚠️  Very low joints detected - fix_motion_heights likely to activate")
        else:
            print("   ✅ Joint heights look reasonable\n")
        
        # 4. Check if root and pelvis are aligned
        print("4️⃣ ROOT vs PELVIS ALIGNMENT:")
        root_pos_frame0 = motion.root_translation[0].numpy()
        pelvis_pos_frame0 = motion.global_translation[0, 0].numpy()  # Pelvis is index 0
        
        print(f"   Root position:   [{root_pos_frame0[0]:.3f}, {root_pos_frame0[1]:.3f}, {root_pos_frame0[2]:.3f}]")
        print(f"   Pelvis position: [{pelvis_pos_frame0[0]:.3f}, {pelvis_pos_frame0[1]:.3f}, {pelvis_pos_frame0[2]:.3f}]")
        
        diff = np.linalg.norm(root_pos_frame0 - pelvis_pos_frame0)
        print(f"   Difference: {diff:.4f}m")
        
        if diff > 0.01:
            print("   ⚠️  Root and Pelvis positions don't match!")
            print("      This can cause MotionLib confusion")
        else:
            print("   ✅ Root and Pelvis are properly aligned\n")
        
        # 5. Check for anatomical correctness
        print("5️⃣ ANATOMICAL CORRECTNESS:")
        pelvis_z = pelvis_pos_frame0[2]
        print(f"   Pelvis height: {pelvis_z:.3f}m")
        
        ankle_heights = []
        for name in ['L_Ankle', 'R_Ankle']:
            if name in joint_names:
                idx = joint_names.index(name)
                ankle_z = global_pos[idx, 2]
                ankle_heights.append(ankle_z)
                status = "✅" if ankle_z < pelvis_z else "❌"
                print(f"   {name}: {ankle_z:.3f}m {status}")
        
        if ankle_heights:
            avg_ankle_height = np.mean(ankle_heights)
            pelvis_ankle_diff = pelvis_z - avg_ankle_height
            print(f"   Pelvis-ankle height diff: {pelvis_ankle_diff:.3f}m")
            
            if pelvis_ankle_diff < 0.3:
                print("   ⚠️  Pelvis too close to ankles - may indicate height compression")
            elif pelvis_ankle_diff > 1.2:
                print("   ⚠️  Pelvis too far from ankles - may indicate height stretch")
            else:
                print("   ✅ Reasonable pelvis-ankle height difference\n")
        
        # 6. Final assessment
        print("6️⃣ SUMMARY:")
        
        issues = []
        if min_height < 0.05:
            issues.append("Very low joint heights")
        if diff > 0.01:
            issues.append("Root-Pelvis misalignment")
        if ankle_heights and (pelvis_z - np.mean(ankle_heights)) < 0.3:
            issues.append("Compressed height")
        
        if not issues:
            print("   ✅ All checks passed - motion should work correctly in Isaac Lab")
        else:
            print("   ⚠️  Potential issues detected:")
            for issue in issues:
                print(f"      - {issue}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    comprehensive_height_test()
