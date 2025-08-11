import sys
sys.path.append('d:/Isaac/IsaacLab2.1.2/source')
from isaaclab.utils.motion import SkeletonMotion
import numpy as np

# Load motion
motion = SkeletonMotion.from_file('data/scripts/data2retarget/retargeted_motion.npy')

# Check frame 0
global_pos = motion.global_translation[0].numpy()
root_pos = motion.root_translation[0].numpy()

print('Root position:', root_pos)
print('Joint positions:')
for i, name in enumerate(motion.skeleton_tree.node_names):
    pos = global_pos[i]
    print(f'  {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]')

print('Minimum Z:', global_pos[:, 2].min())
