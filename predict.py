import sys
import os
sys.path.insert(0, '/hpf/projects/ndlamini/scratch/wgao/python3.8.0/')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import numpy as np
import cv2
import threading


import nibabel as nib
from nibabel import processing
import scipy
from skimage import morphology
import nilearn.image

root = sys.argv[1]
modality = sys.argv[2].lower()
model_location = sys.argv[3]

print(f'Using model {model_location}')
model = keras.models.load_model(model_location, compile=False)

IMG_SIZE = 128
def predict_without_ground_truth(path_to_img):
    print(f'Generating mask for {path_to_img}')
    # load the image
    # load the image
    nifti = nib.load(path_to_img)
    nifti_resampled = processing.conform(nifti)

    num_slices = nifti_resampled.shape[2]
    X = np.empty((num_slices, IMG_SIZE, IMG_SIZE, 1))
    nifti_voxels = nifti_resampled.get_fdata()
    
    # resize input data to conform to model expectations
    for i in range(num_slices):
        X[i,:,:,0] = cv2.resize(nifti_voxels[:,:,i], (IMG_SIZE,IMG_SIZE))

    # normalize
    max_voxel = np.max(X)

    # predict
    prediction = model.predict(X/max_voxel)
    
    # convert to axial
    prediction = np.moveaxis(prediction[:, :, :, 1], 0, 2)
    
    # apply a gaussian blur
    prediction = scipy.ndimage.gaussian_filter(prediction, sigma=(1, 3, 3), order=0)

    # resize predictions back to the size of resampled input nifti
    resized_prediction = np.zeros(nifti_resampled.shape)
    
    for slice in range(prediction.shape[2]):
        # round the predictions to get a binary mask
        resized_prediction[:, :, slice] = np.round(cv2.resize(prediction[:, :, slice], (nifti_resampled.shape[0], nifti_resampled.shape[1]))).astype(int)

        # post processing, fill in any holes in mask
        resized_prediction[:, :, slice] = scipy.ndimage.binary_fill_holes(resized_prediction[:, :, slice])
        
        # post processing, remove any stray artifacts
        resized_prediction[:,:, slice] = morphology.remove_small_objects(resized_prediction[:,:, slice].astype(bool), IMG_SIZE*IMG_SIZE*0.01).astype(int)

    for slice in range(resized_prediction.shape[0]):
        # post processing, fill in any holes in mask
        resized_prediction[slice, :, :] = scipy.ndimage.binary_fill_holes(resized_prediction[slice, :, :])
        
        # post processing, remove any stray artifacts
        resized_prediction[slice, :, :] = morphology.remove_small_objects(resized_prediction[slice, :, :].astype(bool), IMG_SIZE*IMG_SIZE*0.01).astype(int)
    
    for slice in range(resized_prediction.shape[1]):
        # post processing, fill in any holes in mask
        resized_prediction[:, slice, :] = scipy.ndimage.binary_fill_holes(resized_prediction[:, slice, :])
        
        # post processing, remove any stray artifacts
        resized_prediction[:, slice, :] = morphology.remove_small_objects(resized_prediction[:, slice, :].astype(bool), IMG_SIZE*IMG_SIZE*0.01).astype(int)

    # save predictions
    prediction_nifti = nib.Nifti1Image(resized_prediction, nifti_resampled.affine, dtype=np.uint16)
    prediction_nifti = nilearn.image.resample_img(prediction_nifti, nifti.affine, nifti.shape, "nearest")
    
    # save prediction
    save_dir = os.path.dirname(path_to_img)
    save_file_path = os.path.join(save_dir, f'{save_dir.split(os.sep)[-1]}_pred.nii.gz')
    
    nib.save(prediction_nifti, save_file_path)
    
    print(f"Saved prediction to {save_file_path}")

all_files = []

def find_and_predict_for_folder(root):
    with os.scandir(root) as it:
        for item in it:
            if item.is_file() and item.name.endswith(f'{modality.upper()}.nii.gz'):
                all_files.append(item.path)
            elif item.is_dir():
                find_and_predict_for_folder(item.path)


if root.endswith('.nii.gz'):
    predict_without_ground_truth(root)
elif os.path.isdir(root):
    find_and_predict_for_folder(root)
    threads = []

    for file in all_files:
        threads.append(threading.Thread(target=predict_without_ground_truth, args=[file]))
        threads[-1].start()
    for thread in threads:
        thread.join()
else:
    print(f'{root} is not a nifti file or directory, stopping')