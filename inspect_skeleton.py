#!/usr/bin/env python3

import numpy as np
from poselib.skeleton.skeleton3d import SkeletonMotion

# Load your motion file to inspect the skeleton
motion_file = 'output/smpl_lower_retargeted_treadmill_example.npy'
motion = SkeletonMotion.from_file(motion_file)

print('Skeleton information:')
print(f'Number of bodies: {len(motion.skeleton_tree.node_names)}')
print('Body names:')
for i, name in enumerate(motion.skeleton_tree.node_names):
    print(f'{i:2d}: {name}')

print(f'\nMotion shape: {motion.global_translation.shape}')
print('First frame global positions (XYZ):')
for i, name in enumerate(motion.skeleton_tree.node_names):
    pos = motion.global_translation[0, i]
    print(f'{i:2d}: {name:15s} -> X={pos[0]:.4f}, Y={pos[1]:.4f}, Z={pos[2]:.4f}')

print("\n🦶 Looking for feet (lowest Z positions):")
z_positions = motion.global_translation[0, :, 2]  # Z coordinates
sorted_indices = np.argsort(z_positions)
print("Bodies sorted by height (lowest first):")
for idx in sorted_indices[:10]:  # Show 10 lowest
    name = motion.skeleton_tree.node_names[idx]
    z = z_positions[idx]
    print(f'{idx:2d}: {name:15s} -> Z={z:.4f}')
