import os
from pathlib import Path
from typing import Optional

import numpy as np
import typer
import yaml


def main(
    main_motion_dir: Path,
    humanoid_type: str = "smpl_humanoid_lower_body",
    amass_fps_file: Optional[Path] = None,
    output_path: Optional[Path] = None,
):
    """
    Create a motion list YAML file for package_motion_lib.py from retargeted motion files.
    
    This script looks for .npy motion files and creates a YAML with the structure:
    motions:
      - file: path/to/motion.npy
        fps: framerate
        idx: index
        sub_motions:
          - idx: index
            timings:
              start: 0.0
              end: duration
            weight: 1.0
    """
    
    # Create the motions list
    motions_list = []
    motion_idx = 0
    
    print(f"Scanning directory: {main_motion_dir}")
    
    # Walk through directory to find motion files
    for root, dirs, files in os.walk(main_motion_dir):
        for file in files:
            # Look for .npy files (retargeted motion files)
            if file.endswith(".npy") and "retargeted" in file:
                file_path = os.path.join(root, file)
                print(f"Found motion file: {file_path}")
                
                try:
                    # Load the motion data to get fps and duration
                    motion_data = np.load(file_path, allow_pickle=True).item()
                    
                    if 'fps' in motion_data:
                        fps = float(motion_data['fps'])
                        print(f"  FPS: {fps}")
                    else:
                        # Default fps if not found
                        fps = 30.0
                        print(f"  No FPS found, using default: {fps}")
                    
                    # Calculate duration from motion data
                    # Assuming motion data has time-series information
                    duration = 1.0  # Default duration
                    
                    # Try to calculate actual duration from the motion data
                    if 'rotation' in motion_data:
                        rotation_data = motion_data['rotation']
                        if isinstance(rotation_data, dict):
                            # Get the first joint's rotation data to determine length
                            for joint_name, joint_data in rotation_data.items():
                                if hasattr(joint_data, 'shape') and len(joint_data.shape) > 0:
                                    num_frames = joint_data.shape[0]
                                    duration = num_frames / fps
                                    print(f"  Calculated duration: {duration:.3f}s ({num_frames} frames)")
                                    break
                    
                    # Use absolute path instead of relative path
                    absolute_path = os.path.abspath(file_path)
                    absolute_path = absolute_path.replace("\\", "/")  # Use forward slashes for consistency
                    
                    # Create motion entry
                    motion_entry = {
                        "file": absolute_path,
                        "fps": fps,
                        "idx": motion_idx,
                        "sub_motions": [{
                            "idx": motion_idx,
                            "timings": {
                                "start": 0.0,
                                "end": duration
                            },
                            "weight": 1.0
                        }]
                    }
                    
                    motions_list.append(motion_entry)
                    print(f"  Added motion with idx {motion_idx}")
                    motion_idx += 1
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
    
    # Create the final YAML structure
    yaml_data = {"motions": motions_list}
    
    # Set output path
    if output_path is None:
        output_path = Path.cwd()
    
    output_file = output_path / f"motions_fps_{humanoid_type}.yaml"
    
    # Write the YAML file
    with open(output_file, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, indent=2)
    
    print(f"\nCreated motion list YAML with {len(motions_list)} motions: {output_file}")


if __name__ == "__main__":
    typer.run(main)
