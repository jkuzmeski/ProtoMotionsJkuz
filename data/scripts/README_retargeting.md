# Treadmill Motion Retargeting

This directory contains scripts for retargeting motion data from treadmill experiments to the SMPL lower body model format used in the ProtoMotions data pipeline.

## Overview

The retargeting process converts joint position data from `treadmill2overground.py` output into the `SkeletonMotion` format that's compatible with the next steps in the ProtoMotions data pipeline.

## Files

- `retarget_treadmill_motion.py` - Main retargeting script
- `test_retargeting.py` - Test script with synthetic data
- `treadmill2overground.py` - Original treadmill-to-overground transformation script

## Usage

### Basic Usage

```bash
python retarget_treadmill_motion.py input_motion.npy output_motion.npy --fps 200
```

### Parameters

- `input_file`: Input .npy file from treadmill2overground.py (required)
- `output_file`: Output file path for retargeted motion (required)
- `--fps, -f`: Frame rate of the motion data (default: 200)
- `--joints, -j`: Comma-separated joint names (default: Pelvis,L_Hip,L_Knee,L_Ankle,L_Toe,R_Hip,R_Knee,R_Ankle,R_Toe)

### Example

```bash
# Convert treadmill motion to SMPL lower body format
python retarget_treadmill_motion.py \
    treadmill_motion.npy \
    retargeted_motion.npy \
    --fps 200 \
    --joints "Pelvis,L_Hip,L_Knee,L_Ankle,L_Toe,R_Hip,R_Knee,R_Ankle,R_Toe"
```

## Data Format

### Input Format

The input should be a .npy file containing joint positions with shape `(n_frames, n_joints, 3)` where:
- `n_frames`: Number of motion frames
- `n_joints`: Number of joints (typically 9 for lower body)
- `3`: X, Y, Z coordinates in meters

Expected joint order: `['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe']`

### Output Format

The output is a file in the `SkeletonMotion` format that contains:
- Global and local joint rotations (quaternions)
- Root translations
- Global and local velocities
- Skeleton tree information
- Frame rate metadata

This format is compatible with the ProtoMotions data pipeline and can be loaded by `SkeletonMotion.from_file()`.

## Testing

Run the test script to verify the retargeting functionality:

```bash
python test_retargeting.py
```

This will:
1. Create synthetic walking motion data
2. Convert it to the SMPL lower body format
3. Verify the output can be loaded correctly

## Workflow

1. **Process treadmill data** with `treadmill2overground.py`:
   ```bash
   python treadmill2overground.py motion.txt rotations.txt tpose.txt tpose_rot.txt output_motion --speed 2.0
   ```

2. **Retarget to SMPL lower body** with `retarget_treadmill_motion.py`:
   ```bash
   python retarget_treadmill_motion.py output_motion.npy retargeted_motion.npy --fps 200
   ```

3. **Use in ProtoMotions pipeline**:
   The retargeted motion can now be used in the next steps of your data pipeline.

## Technical Details

### SMPL Lower Body Model

The retargeting uses the SMPL lower body model with:
- 9 joints: Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe, R_Hip, R_Knee, R_Ankle, R_Toe
- Neutral body shape (betas = 0)
- Male gender (gender = 0)
- Upright starting pose

### Inverse Kinematics

The script converts joint positions to rotations using a simplified inverse kinematics approach:
1. Maps source joint names to target skeleton tree joints
2. Computes direction vectors between consecutive joints
3. Calculates rotations from default to current joint directions
4. Creates quaternions from axis-angle representations

### Coordinate System

The retargeting preserves the coordinate system from the input data:
- X: Right
- Y: Forward  
- Z: Up

## Troubleshooting

### Common Issues

1. **Joint name mismatch**: Ensure the joint names in your input data match the expected order
2. **Frame rate mismatch**: Set the correct FPS parameter to match your input data
3. **Missing dependencies**: Install required packages (poselib, smpl_sim, etc.)

### Error Messages

- `"Number of joint names doesn't match number of joints in data"`: Check your joint names parameter
- `"Could not find foot joints"`: Verify your input data contains the expected joint names
- `"Error in velocity calculation"`: Check for NaN or infinite values in your input data

## Dependencies

- `poselib`: Skeleton and motion handling
- `smpl_sim`: SMPL model generation
- `numpy`: Numerical operations
- `torch`: Tensor operations
- `scipy`: Rotation calculations
- `typer`: Command-line interface

## Notes

- The retargeting process is designed for lower body motion only
- The SMPL model uses a neutral body shape by default
- The output format is compatible with the existing ProtoMotions pipeline
- Test with synthetic data before processing real motion capture data 