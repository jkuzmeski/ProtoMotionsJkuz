#!/usr/bin/env python3

import numpy as np
from poselib.skeleton.skeleton3d import SkeletonMotion

# Load your motion file to inspect the skeleton
motion_file = 'output/smpl_lower_retargeted_treadmill_example.npy'

try:
    motion = SkeletonMotion.from_file(motion_file)
    
    print('🦴 Skeleton Analysis:')
    print(f'Number of bodies: {len(motion.skeleton_tree.node_names)}')
    
    # Get first frame positions
    first_frame_pos = motion.global_translation[0]
    z_coords = first_frame_pos[:, 2]  # Z coordinates
    
    print('\n📍 All body positions (first frame):')
    for i, name in enumerate(motion.skeleton_tree.node_names):
        pos = first_frame_pos[i]
        print(f'{i:2d}: {name:20s} -> X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}')
    
    print('\n🦶 Likely FOOT candidates (lowest Z positions):')
    sorted_indices = np.argsort(z_coords)
    for i, idx in enumerate(sorted_indices[:5]):  # Show 5 lowest
        name = motion.skeleton_tree.node_names[idx]
        z = z_coords[idx]
        print(f'  Rank {i+1}: Body {idx:2d} ({name:20s}) -> Z={z:.4f}')
        
    print(f'\n🔍 Ground level candidates (Z < 0.2):')
    ground_bodies = []
    for i, z in enumerate(z_coords):
        if z < 0.2:
            name = motion.skeleton_tree.node_names[i]
            ground_bodies.append((i, name, z))
            print(f'  Body {i:2d}: {name:20s} -> Z={z:.4f}')
    
    if ground_bodies:
        print(f'\n✅ Suggested key_body_ids for feet: {[body[0] for body in ground_bodies]}')
    else:
        print('\n⚠️  No bodies found near ground level - motion might need height adjustment')
        
except Exception as e:
    print(f"Error loading motion: {e}")
    print("Trying to inspect file directly...")
    
    try:
        data = np.load(motion_file, allow_pickle=True).item()
        print(f"File contains: {data.keys()}")
        if 'global_translation' in data:
            gt = data['global_translation']
            print(f"Global translation shape: {gt.shape}")
            print(f"First frame Z coordinates: {gt[0, :, 2]}")
    except Exception as e2:
        print(f"Could not inspect file: {e2}")
