import numpy as np

# Check our retargeted motion root heights
motion_data = np.load('data/scripts/data2retarget/retargeted_motion.npy', allow_pickle=True).item()
root_trans = motion_data['root_translation']['arr']
print('Retargeted motion root heights (first 5 frames):')
for i in range(5):
    print(f'  Frame {i}: Z = {root_trans[i, 2]:.4f}')

print('Motion data keys:', list(motion_data.keys()))
