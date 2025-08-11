#!/usr/bin/env python3

from data.scripts.convert_to_isaac import create_motion_from_txt, retarget_motion_to_robot

def debug_motion():
    print("Loading source motion...")
    motion = create_motion_from_txt('data/motions/S02_3-0ms.txt', 'data/motions/S02_TPose_1.txt', 3.0)
    print("Source motion loaded successfully")
    
    print("Retargeting motion...")
    retargeted = retarget_motion_to_robot(motion, 'smpl_humanoid_lower_body', render=False)
    print("Motion retargeted successfully")
    
    print('Available attributes:', [attr for attr in dir(retargeted) if not attr.startswith('_')])
    print('Type:', type(retargeted))
    
    # Check specific attributes we need
    attrs_to_check = ['local_rotation', 'root_translation', 'global_velocity', 'global_angular_velocity', 'skeleton_tree', 'fps']
    for attr in attrs_to_check:
        if hasattr(retargeted, attr):
            value = getattr(retargeted, attr)
            if hasattr(value, 'shape'):
                print(f'{attr}: shape {value.shape}, type {type(value)}')
            else:
                print(f'{attr}: value {value}, type {type(value)}')
        else:
            print(f'{attr}: NOT FOUND')

if __name__ == "__main__":
    debug_motion()
