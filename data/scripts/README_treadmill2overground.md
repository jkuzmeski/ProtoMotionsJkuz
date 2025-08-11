# Treadmill to Overground Motion Transformation

This script transforms motion capture data from treadmill experiments to appear as if the motion was performed overground. It stabilizes stance feet and adds forward progression based on treadmill speed.

## Features

- ✅ **Robust stance detection** using biomechanical criteria (foot height, velocity, acceleration)
- ✅ **Smart foot stabilization** during stance phases
- ✅ **Coordinate system transformation** (Y-forward to X-forward)
- ✅ **Ground plane adjustment** to ensure proper foot contact
- ✅ **Multiple output formats** (NumPy, CSV, TXT with metadata)
- ✅ **Comprehensive error handling** and validation

## Installation Requirements

```bash
pip install numpy pandas scipy typer pathlib
```

## Usage

### Command Line Interface

```bash
python treadmill2overground.py motion.txt rotations.txt tpose.txt tpose_rotations.txt output_motion [OPTIONS]
```

### Arguments

- `motion_file`: Path to motion positions text file
- `rotation_file`: Path to motion quaternions text file  
- `tpose_file`: Path to T-pose positions text file
- `tpose_rotation_file`: Path to T-pose quaternions text file
- `output_path`: Output path for transformed motion (without extension)

### Options

- `--speed, -s`: Treadmill speed in m/s (default: 1.5)
- `--fps, -f`: Motion capture frame rate (default: 200)
- `--transform, -t`: Coordinate transformation ('none' or 'y_to_x_forward', default: 'y_to_x_forward')
- `--debug, -d`: Enable debug output
- `--help`: Show help message

### Example

```bash
# Transform motion at 2.0 m/s treadmill speed
python treadmill2overground.py \
    data/motion_positions.txt \
    data/motion_rotations.txt \
    data/tpose_positions.txt \
    data/tpose_rotations.txt \
    output/transformed_motion \
    --speed 2.0 \
    --fps 200 \
    --debug
```

## Input Data Format

### File Structure
All input files should be tab-separated text files without headers, with frame numbers in the first column.

### Joint Order
The script expects data for 9 joints in this exact order:
1. Pelvis
2. L_Hip  
3. L_Knee
4. L_Ankle
5. L_Toe
6. R_Hip
7. R_Knee
8. R_Ankle
9. R_Toe

### Position Files (motion.txt, tpose.txt)
```
1	x1	y1	z1	x2	y2	z2	...	x9	y9	z9
2	x1	y1	z1	x2	y2	z2	...	x9	y9	z9
...
```
Where each line contains:
- Frame number (1-indexed)
- 27 position values (9 joints × 3 coordinates)

### Rotation Files (rotations.txt, tpose_rotations.txt)
```
1	w1	x1	y1	z1	w2	x2	y2	z2	...	w9	x9	y9	z9
2	w1	x1	y1	z1	w2	x2	y2	z2	...	w9	x9	y9	z9
...
```
Where each line contains:
- Frame number (1-indexed)  
- 36 quaternion values (9 joints × 4 components in WXYZ format)

**Important**: 
- Pelvis quaternion = Global rotation (pelvis → lab coordinate system)
- Other joint quaternions = Local rotations (parent bone → child bone)

## Output Files

The script generates multiple output formats:

### 1. NumPy Array (.npy)
Binary format for efficient loading in Python:
```python
import numpy as np
joint_centers = np.load('output_motion.npy')
```

### 2. Text File (.txt)
Tab-separated format matching input structure, with metadata header:
```
# Joint order: Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe, R_Hip, R_Knee, R_Ankle, R_Toe
# Shape: 200 frames, 9 joints, 3 coordinates (x, y, z)
# FPS: 200
# Generated: 2024-01-01 12:00:00
1	x1	y1	z1	x2	y2	z2	...
```

### 3. CSV File (.csv)
Spreadsheet-friendly format with named columns:
```
Frame,Pelvis_X,Pelvis_Y,Pelvis_Z,L_Hip_X,L_Hip_Y,L_Hip_Z,...
1,0.000000,0.000000,1.000000,-0.100000,0.000000,0.800000,...
```

### 4. Metadata (.json)
Detailed information about the transformation:
```json
{
  "joint_names": ["Pelvis", "L_Hip", ...],
  "shape": [200, 9, 3],
  "fps": 200,
  "units": "meters",
  "coordinate_system": "X=right, Y=forward, Z=up (after transformation)",
  "generated_timestamp": "2024-01-01T12:00:00"
}
```

## Coordinate Systems

### Input Coordinate System
- X = Right
- Y = Forward  
- Z = Up

### Output Coordinate System (with y_to_x_forward transform)
- X = Forward
- Y = Left
- Z = Up

This transformation rotates the coordinate system 90° around the Z-axis to align with common robotics conventions.

## Algorithm Overview

### 1. Data Loading & Validation
- Load motion and T-pose data from text files
- Validate quaternion format and magnitudes
- Check data consistency

### 2. Coordinate Transformation
- Optional rotation from Y-forward to X-forward coordinate system
- Preserves all motion dynamics while changing reference frame

### 3. Ground Plane Adjustment
- Find lowest point across all frames and joints
- Adjust all positions so feet contact ground plane (z=0)

### 4. Stance Phase Detection
Robust detection using three biomechanical criteria:
- **Height**: Foot must be close to ground (< 5cm)
- **Vertical velocity**: Minimal vertical movement (< 0.1 m/s)
- **Horizontal acceleration**: Constant horizontal velocity (< 0.5 m/s²)

### 5. Treadmill-to-Overground Transformation
- **Single stance**: Stabilize the stance foot
- **Double stance**: Use average of both feet to minimize drift
- **Flight phase**: Maintain last known velocity
- **Forward progression**: Add linear motion based on treadmill speed

## Testing

Run the included test script to verify installation:

```bash
python test_treadmill_script.py
```

This generates sample data and tests the complete transformation pipeline.

## Troubleshooting

### Common Issues

1. **"Could not find foot joints"**
   - Check that your data includes L_Ankle/L_Toe and R_Ankle/R_Toe joints
   - Verify joint naming matches expected format

2. **"Quaternion magnitude not close to 1.0"**
   - Check quaternion format (should be WXYZ: [w, x, y, z])
   - Verify quaternions are normalized

3. **Large forward distance discrepancy**
   - Check treadmill speed parameter
   - Verify motion duration and frame rate

4. **File format errors**
   - Ensure tab-separated format (not spaces)
   - Check frame numbers are 1-indexed
   - Verify no missing data or headers

### Debug Mode

Enable debug output for detailed information:
```bash
python treadmill2overground.py ... --debug
```

This shows:
- Data shape and ranges
- Transformation parameters
- Motion statistics
- File locations

## Performance

- **Memory usage**: ~1MB per 1000 frames for 9 joints
- **Processing time**: ~1-2 seconds per 1000 frames
- **Recommended**: <10,000 frames per batch for optimal performance

## Citation

If you use this script in research, please cite:

```
Treadmill-to-Overground Motion Transformation
Author: John Kuzmeski
Year: 2024
```
