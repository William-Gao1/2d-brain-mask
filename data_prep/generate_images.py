import sys
sys.path.insert(0, '/hpf/projects/ndlamini/scratch/wgao/python3.8.0/')

import os
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel import processing
import threading
import numpy as np
import albumentations as A
import pypdf
import time

from sys import argv

# time this script
start_time = time.time()

# get the modality, folder with all the subjects, and save location for the book from command line args
modality = argv[1].upper()
root_folder = argv[2]
save_book_location = argv[3]

assert os.path.exists(root_folder), f"Root folder {root_folder} does not exist"

# figure out which folders we need to generate plots for
all_folders = []

def find_folders(root):
    with os.scandir(root) as it:
        for item in it:
            # if this folder has nifti files, then we need to generate plots for it
            if item.is_file() and item.name.endswith('.nii.gz'):
                all_folders.append(root)
                return
            elif item.is_dir():
                find_folders(item.path)


def find_files_to_display(folder_path):
    """Given a folder, figure out which files to plot. It will take the most recent image with the modality we are looking for
    and the most recent mask file

    Args:
        folder_path (str): Path to folder

    Returns:
        List[(str, str)]: A list of (image, mask) pairs to display
    """
    
    # get a list of files sorted by modified time
    files = sorted(os.listdir(folder_path), key=lambda s: os.path.getmtime(os.path.join(folder_path, s)))
    
    # get list of images
    imgs = list(filter(lambda s: modality in s.upper() and not 'MASK' in s.upper(), files))
    # list of registered images
    registered_imgs = list(filter(lambda s: f'{modality}_TO_' in s.upper(), imgs))

    # get list of masks
    masks = list(filter(lambda s: 'MASK' in s.upper() in s. upper(), files))
    
    # if there are registered images, use the most recent one with the most recent mask
    if len(registered_imgs) > 0:
        return [(os.path.join(folder_path, registered_imgs[0]), os.path.join(folder_path, masks[0]))]

    # no registered images, use most recent regular image with most recent mask
    return [(os.path.join(folder_path, imgs[0]), os.path.join(folder_path, masks[0]))]


NUM_SLICES = 10
def display_brain_masks(files, save_folder):
    """Generates a plt.pdf file for the (image, mask) pairs

    Args:
        files (List[(str, str)]): A list of (image, mask) pairs to display
        save_folder (str): Folder to save the plot to
    """
    print(f'Displaying {files} for {save_folder}')
    
    # init plot
    f, axarr = plt.subplots(NUM_SLICES, len(files) * 2, figsize=(7 * len(files),25))
    
    # generate the title of the plot which will be the folder + each file being displayed
    title = save_folder + '\n'
    for img, mask in files:
        title += img.split(os.sep)[-1] + '\n' + mask.split(os.sep)[-1] + '\n'
    f.suptitle(title)

    # for each (image, mask) generate the plot
    for idx, (img, mask) in enumerate(files):
        img_nifti = nib.load(img)
        
        # if 4 dimensions, take first volume
        if img_nifti.ndim == 4:
            img_nifti = nib.funcs.four_to_three(img_nifti)[0]
        img_nifti = processing.conform(img_nifti).get_fdata()
        mask_nifti = np.round(processing.conform(nib.load(mask)).get_fdata())
        
        # get the slice indices where the mask is not 0
        mask_non_zero_slices = np.sort(np.unique(np.nonzero(mask_nifti)[2]))
        mask_min_slice = mask_non_zero_slices[0]
        mask_max_slice = mask_non_zero_slices[-1]
        
        # now, we pick which slices we want to display
        # give emphasis on the bottom third of mask
        bottom_slices = np.round(np.linspace(mask_min_slice, mask_min_slice + (mask_max_slice - mask_min_slice)/3, num=NUM_SLICES//2)).astype(int)
        
        # rest of choices will be used on top 2/3rds
        top_slices = np.round(np.linspace(mask_min_slice + (mask_max_slice - mask_min_slice)/3, mask_max_slice, num=NUM_SLICES//2)).astype(int)
        
        choices = np.append(bottom_slices, top_slices)
        
        # plot each slice from choices
        axarr[0][0].set_title('Img')
        axarr[0][1].set_title('Mask')
        for i in range(len(choices)):
            axarr[i][idx*2].imshow(A.resize(img_nifti[:, :, choices[i]], 128, 128), cmap='gray')
            axarr[i][idx*2 + 1].imshow(A.resize(img_nifti[:, :, choices[i]], 128, 128), cmap='gray')
            axarr[i][idx*2 + 1].imshow(A.resize(mask_nifti[:, :, choices[i]], 128, 128), cmap='OrRd', alpha=0.5)
        
        # save the figure
        f.savefig(os.path.join(save_folder, 'plt.pdf'))
        plt.close();
        print(f'Done displaying {save_folder}')

#### Main entry point

# first, find all the folder we want to plot
find_folders(root_folder)

threads = []

# spin up threads to plot all the folders
for folder in all_folders:
    files = find_files_to_display(folder)
    threads.append(threading.Thread(target = display_brain_masks, args = [files, folder]))
    threads[-1].start()

for thread in threads:
    thread.join()

# now, generate the book
print("Generating book...")

merge = pypdf.PdfMerger()

# for each folder we generated a plot for, append 'plt.pdf' to the book
for folder in sorted(all_folders):
    plot = os.path.join(folder, 'plt.pdf')
    if not os.path.exists(plot):
        continue 
    merge.append(plot)

# save book
merge.write(save_book_location)
merge.close()

end_time = time.time()

print(f"Done in {((end_time - start_time)/60):.2f} minutes")
