import os

import os
import nibabel as nib
import numpy as np
import scipy.io
from skimage.transform import resize

# Directory containing NIfTI files
input_dir = r"nifti file input dir path"

# Output folder where 2D slices will be saved as PNG files
output_dir = r"output dir path to save 2D slices as .mat"

import os
import nibabel as nib
import numpy as np
import scipy.io
from skimage.transform import resize


# Function to load a NIfTI file, extract 2D slices, resize, and normalize
def LoadResizeNormalizeNIfTI(file_path, target_shape=(256, 256)):
    nifti_img = nib.load(file_path)
    nifti_data = nifti_img.get_fdata()

    shape = nifti_data.shape

    # Calculate the indices for the middle 100 slices
    start_slice = (shape[2] - 100) // 2
    end_slice = start_slice + 100

    # Initialize an empty list to store 2D slices
    slices = []

    # Iterate through the slices of the NIfTI volume
    for slice_idx in range(start_slice, end_slice):
        slice_2d = nifti_data[:, :, slice_idx]
        slice_2d = np.rot90(slice_2d, k=1)

        # Resize to the target shape (256x256)
        slice_2d = resize(slice_2d, target_shape, mode='constant', anti_aliasing=True)

        # Normalize to the range [0, 1]
        slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d))

        slices.append(slice_2d)

    return slices




# List NIfTI files in the input directory
nifti_files = [f for f in os.listdir(input_dir) if f.endswith(".nii.gz")]

# Initialize an empty list to store slices
all_slices = []

# Load, resize, normalize, and extract 2D slices from NIfTI files
for nifti_file in nifti_files:
    file_path = os.path.join(input_dir, nifti_file)
    slices = LoadResizeNormalizeNIfTI(file_path)
    all_slices.extend(slices)

# Convert the list of 2D slices to a NumPy array
all_slices_array = np.array(all_slices)

# Define the number of slices per batch
slices_per_batch = 100  #  can adjust this value

# Save the slices in batches
num_batches = len(all_slices_array) // slices_per_batch
for batch_idx in range(num_batches):
    start_idx = batch_idx * slices_per_batch
    end_idx = (batch_idx + 1) * slices_per_batch
    batch_slices = all_slices_array[start_idx:end_idx]
    batch_output_file = f"slices_batch_{batch_idx}.mat"
    scipy.io.savemat(batch_output_file, {'slices': batch_slices})

# Save any remaining slices (if the total number of slices is not a multiple of slices_per_batch)
if len(all_slices_array) % slices_per_batch != 0:
    remaining_slices = all_slices_array[num_batches * slices_per_batch:]
    remaining_output_file = f"slices_batch_{num_batches}.mat"
    scipy.io.savemat(remaining_output_file, {'slices': remaining_slices})
# Combine all batches into a single array
combined_slices = []
for batch_idx in range(num_batches ):
    batch_file = f"slices_batch_{batch_idx}.mat"
    batch_data = scipy.io.loadmat(batch_file)['slices']
    combined_slices.extend(batch_data)

# Convert the combined list of 2D slices to a NumPy array
combined_slices_array = np.array(combined_slices)

# Save all slices as a single .mat file
output_file = "combined_slices.mat"
scipy.io.savemat(output_file, {'slices': combined_slices_array})

# Clean up by deleting the individual batch files
for batch_idx in range(num_batches ):
    batch_file = f"slices_batch_{batch_idx}.mat"
    os.remove(batch_file)
