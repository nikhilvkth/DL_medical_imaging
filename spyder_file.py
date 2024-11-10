from pathlib import Path
import pydicom
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

labels = pd.read_csv("/Volumes/Nikhil WD/ude/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv")

# Remove duplicate entries
labels = labels.drop_duplicates("patientId")

ROOT_PATH = Path("/Volumes/Nikhil WD/ude/rsna-pneumonia-detection-challenge/stage_2_train_images/")
SAVE_PATH = Path("/Volumes/Nikhil WD/ude/rsna-pneumonia-detection-challenge/Processed/")

import matplotlib.pyplot as plt
import pydicom

fig, axis = plt.subplots(3, 3, figsize=(9, 9))
c = 0

for i in range(3):
    for j in range(3):
        patient_id = labels.patientId.iloc[c]
        dcm_path = ROOT_PATH / patient_id
        dcm_path = dcm_path.with_suffix(".dcm")
        
        # Use dcmread instead of read_file
        dcm = pydicom.dcmread(dcm_path).pixel_array
        
        label = labels["Target"].iloc[c]
        
        axis[i][j].imshow(dcm, cmap="bone")
        axis[i][j].set_title(f"Target: {label}")
        axis[i][j].axis('off')  # Optional: turn off axis for a cleaner look
        c += 1

plt.tight_layout()
plt.show()


import numpy as np
import cv2
import pydicom
from tqdm import tqdm

sums = 0
sums_squared = 0

for c, patient_id in enumerate(tqdm(labels.patientId)):
    dcm_path = ROOT_PATH / patient_id  # Create the path to the DICOM file
    dcm_path = dcm_path.with_suffix(".dcm")  # Add the .dcm suffix
    
    # Read the DICOM file with pydicom and standardize the array
    dcm = pydicom.dcmread(dcm_path).pixel_array / 255  # Use dcmread instead of read_file
        
    # Resize the image; 1024x1024 is large, so we resize to 224x224
    dcm_array = cv2.resize(dcm, (224, 224)).astype(np.float16)
    
    # Retrieve the corresponding label
    label = labels.Target.iloc[c]
    
    # 4/5 train split, 1/5 val split
    train_or_val = "train" if c < 24000 else "val"
        
    # Define save path and create directories if necessary
    current_save_path = SAVE_PATH / train_or_val / str(label)
    current_save_path.mkdir(parents=True, exist_ok=True)
    
    # Save the array in the corresponding directory
    np.save(current_save_path / patient_id, dcm_array)
    
    # Normalize sum of image for statistics calculation
    normalizer = dcm_array.shape[0] * dcm_array.shape[1]
    
    if train_or_val == "train":  # Only use train data to compute dataset statistics
        sums += np.sum(dcm_array) / normalizer
        sums_squared += (np.power(dcm_array, 2).sum()) / normalizer



mean = sums / 24000
std = np.sqrt(sums_squared / 24000 - (mean**2))


print(f"Mean of Dataset: {mean}, STD: {std}")


