#!/usr/bin/env python3

"""
Quick test to verify the skeleton motion retargeting fix.
This tests the position-based approach in create_skeleton_motion_from_mink.
"""

import sys
import os
import numpy as np

# Add the ProtoMotions directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test the retargeting function
try:
    from data.scripts.retarget_treadmill_motion import main
    
    print("=" * 60)
    print("TESTING SKELETON MOTION RETARGETING FIX")
    print("=" * 60)
    print("Approach: Position-based conversion (ignoring MuJoCo joint angles)")
    print("Expected: Ankles should be below pelvis (anatomically correct)")
    print("=" * 60)
    
    # Run the main function
    main()
    
    print("=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)
    
except Exception as e:
    print(f"ERROR during test: {e}")
    import traceback
    traceback.print_exc()
