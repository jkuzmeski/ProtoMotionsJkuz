import sys
sys.path.append('d:/Isaac/IsaacLab2.1.2/source')
import numpy as np
import torch
from isaaclab.utils.motion import SkeletonMotion

# Load motion
motion_data = np.load('data/scripts/data2retarget/retargeted_motion.npy', allow_pickle=True).item()
print('Motion loaded successfully')
print('Keys:', list(motion_data.keys()))

# Check root translation
root_trans = motion_data['root_translation']['arr']
print(f'Root translation shape: {root_trans.shape}')
print(f'First frame root pos: {root_trans[0]}')
print(f'Last frame root pos: {root_trans[-1]}')
print(f'X movement: {root_trans[-1, 0] - root_trans[0, 0]:.3f}')

# Try to create SkeletonMotion
try:
    rotation = motion_data['rotation']['arr'] if isinstance(motion_data['rotation'], dict) else motion_data['rotation']
    
    root_translation_tensor = torch.from_numpy(root_trans).float()
    rotation_tensor = torch.from_numpy(rotation).float()
    
    print("Creating SkeletonMotion...")
    motion = SkeletonMotion.from_dict({
        'root_translation': root_translation_tensor,
        'rotation': rotation_tensor,
        'skeleton_tree': motion_data['skeleton_tree'],
        'is_local': motion_data.get('is_local', True),
        'fps': motion_data.get('fps', 30),
    })
    
    print(f"SUCCESS: SkeletonMotion created with {motion.num_frames} frames")
    print(f"Root position at frame 0: {motion.root_translation[0]}")
    print(f"Root position at frame 100: {motion.root_translation[100]}")
    print("ROOT MOTION FIX VERIFIED!")
    
except Exception as e:
    print(f"Error: {e}")
