import numpy as np

# Path to your .npy file
file_path = './data/processed/egobody/validation/egobody_validation_recording_20210910_S05_S06_01_scene_main_01661.npy'

# Load the .npy file
data = np.load(file_path)

# Print the contents
print(data.shape)
