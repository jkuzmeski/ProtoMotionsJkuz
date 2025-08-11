#!/usr/bin/env python3

from data.scripts.retarget_treadmill_motion import create_smpl_lower_body_skeleton_tree

def analyze_skeleton():
    tree = create_smpl_lower_body_skeleton_tree()
    
    print("SKELETON TREE ANALYSIS")
    print("=" * 50)
    print(f"Node names: {tree.node_names}")
    print(f"Parent indices: {tree.parent_indices.tolist()}")
    
    print("\nLocal translations (bone vectors):")
    for i, name in enumerate(tree.node_names):
        local_trans = tree.local_translation[i].tolist()
        parent_idx = tree.parent_indices[i].item()
        parent_name = tree.node_names[parent_idx] if parent_idx >= 0 else "ROOT"
        print(f"  {name} (from {parent_name}): {local_trans}")
    
    # Check bone lengths
    print("\nBone lengths:")
    for i, name in enumerate(tree.node_names):
        if i > 0:  # Skip root
            bone_vec = tree.local_translation[i]
            bone_length = (bone_vec ** 2).sum() ** 0.5
            print(f"  {name}: {bone_length:.4f}m")

if __name__ == "__main__":
    analyze_skeleton()
