import sys
sys.path.append('d:/Isaac/IsaacLab2.1.2/source')
from isaaclab.utils.motion import SkeletonMotion

# Load motion
motion = SkeletonMotion.from_file("data/scripts/data2retarget/retargeted_motion.npy")
print("Motion loaded")

# Check joint heights in first frame
global_pos = motion.global_translation[0].numpy()
joint_names = motion.skeleton_tree.node_names

print("Joint heights (first frame):")
for i, name in enumerate(joint_names):
    height = global_pos[i, 2]
    print(f"  {name}: {height:.4f}m")

# Check minimum height
min_height = global_pos[:, 2].min()
print(f"\nMinimum height: {min_height:.4f}m")

# Check root translation
root_pos = motion.root_translation[0].numpy()
print(f"Root translation: [{root_pos[0]:.4f}, {root_pos[1]:.4f}, {root_pos[2]:.4f}]")
