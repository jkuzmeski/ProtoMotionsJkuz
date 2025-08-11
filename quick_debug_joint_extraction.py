#!/usr/bin/env python3

"""
Quick debug test to verify our corrected joint angle extraction.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data', 'scripts'))

from retarget_treadmill_motion import main

def quick_debug():
    """Run a quick test with debug output."""
    print("=== Quick Debug Test - Joint Angle Extraction ===")
    
    # Run the retargeting with our corrected mapping
    try:
        main()
        print("\n✅ Retargeting completed successfully!")
        print("Check the debug output above to see if joint angles are being extracted correctly.")
        
    except Exception as e:
        print(f"\n❌ Error during retargeting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_debug()
