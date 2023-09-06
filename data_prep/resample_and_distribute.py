import os
import numpy as np
import shutil
import nibabel as nib
from nibabel import processing
import threading
import pypdf
from sys import argv

# get the modality, root folder, and vetted folder from command line arguments
modality = argv[1].upper()
root_folder = argv[2]
vetted_folder = argv[3]

# create the 'test' and 'train' folder
train_folder = os.path.join(root_folder, 'train')
test_folder = os.path.join(root_folder, 'test')

# percentage of subjects reserved for testing
test_percentage = 0.1

# make sure vetted folder exists, and train and test folders do not exit
assert os.path.exists(vetted_folder), f"Cannot find vetted images, folder {vetted_folder} does not exist"
assert not os.path.exists(train_folder), f"Train folder {train_folder} already exists"
assert not os.path.exists(test_folder), f"Test folder {test_folder} already exists"

# for each subject, make sure there are only 3 files in the folder: the image, the mask, and the plot
subjects = [os.path.join(vetted_folder, x) for x in os.listdir(vetted_folder)]

for subject_folder in subjects:
    files = [os.path.join(subject_folder, x) for x in os.listdir(subject_folder)]
    subject_name = subject_folder.split(os.sep)[-1]
    mask_file = os.path.join(subject_folder, f'{subject_name}_MASK.nii.gz')
    img_file = os.path.join(subject_folder, f'{subject_name}_{modality}.nii.gz')
    plt_file = os.path.join(subject_folder, 'plt.pdf')
    assert mask_file in files, f'Error for subject {subject_name}: Could not find mask {mask_file}'
    assert img_file in files, f'Error for subject {subject_name}: Could not find image {img_file}'
    assert plt_file in files, f'Error for subject {subject_name}: Could not find plot {plt_file}'

# randomly shuffle subjects and then pick which ones will be in the training set and testing set
np.random.shuffle(subjects)

split_index = int(np.ceil(len(subjects) * test_percentage))
print(f"Splitting {split_index} for testing, {len(subjects) - split_index} for training")
test_subjects = subjects[:split_index]
train_subjects = subjects[split_index:]

# create the test and train folders
os.makedirs(test_folder)
os.makedirs(train_folder)

def copy_subject(subject_folder, dest_folder, subject_name):
    """Copies the subject files to the testing or training folder. Also resize all niftis to (256, 256, 256)

    Args:
        subject_folder (str): Folder of subject
        dest_folder (str): Folder to put images and plot into
        subject_name (str): Name of subject (Should be the same name as folder)
    """
    print(f'Copying {subject_folder} to {dest_folder}...')
    mask_file = os.path.join(subject_folder, f'{subject_name}_MASK.nii.gz')
    img_file = os.path.join(subject_folder, f'{subject_name}_{modality}.nii.gz')
    plt_file = os.path.join(subject_folder, 'plt.pdf')
    
    # make the destination folder
    os.makedirs(dest_folder)
    
    # the names of the new files
    mask_file_new = os.path.join(dest_folder, f'{subject_name}_MASK.nii.gz')
    img_file_new = os.path.join(dest_folder, f'{subject_name}_{modality}.nii.gz')
    plt_file_new = os.path.join(dest_folder, 'plt.pdf')
    
    # copy the plot
    shutil.copy2(plt_file, plt_file_new)
    
    # resize the image and mask
    img_nifti = nib.load(img_file)
    
    if img_nifti.ndim == 4:
        # if image has 4 dimensions, take the first volume
        img_nifti = nib.funcs.four_to_three(img_nifti)[0]

    # resize
    img_nifti = processing.conform(img_nifti)
    mask_nifti = processing.conform(nib.load(mask_file))
    
    # save to new location
    nib.save(img_nifti, img_file_new)
    nib.save(mask_nifti, mask_file_new)

# spin up threads, each one process one subject
threads = []

for test_subject in test_subjects:
    subject_name = test_subject.split(os.sep)[-1]
    new_subject_folder = os.path.join(test_folder, subject_name)
    
    threads.append(threading.Thread(target=copy_subject, args=[test_subject, new_subject_folder, subject_name]))
    threads[-1].start()

for train_subject in train_subjects:
    subject_name = train_subject.split(os.sep)[-1]
    new_subject_folder = os.path.join(train_folder, subject_name)
    threads.append(threading.Thread(target=copy_subject, args=[train_subject, new_subject_folder, subject_name]))
    threads[-1].start()

for thread in threads:
    thread.join()

# now, generate the book by appending all the plots (plt.pdf) together
print('Finished moving, generating books...')
merge_test = pypdf.PdfMerger()

for test_subject in test_subjects:
    plot = os.path.join(test_subject, 'plt.pdf')
    merge_test.append(plot)

merge_test.write(os.path.join(root_folder, 'test_subjects_book.pdf'))
merge_test.close()

merge_train = pypdf.PdfMerger()

for train_subject in train_subjects:
    plot = os.path.join(train_subject, 'plt.pdf')
    merge_train.append(plot)

merge_train.write(os.path.join(root_folder, 'train_subjects_book.pdf'))
merge_train.close()

print('Done!')