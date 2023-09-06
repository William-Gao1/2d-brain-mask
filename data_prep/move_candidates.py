import os
from sys import argv
import shutil
import re

modality = argv[1].upper() or 'T1'
root_folder = argv[2]
vetted_folder = argv[3]

assert os.path.exists(root_folder), f"Root folder {root_folder} does not exist"

_, people, _ = next(os.walk(root_folder))

os.makedirs(vetted_folder)

def find_img_and_mask_to_extract(folder_path):
    
    files = sorted(os.listdir(folder_path), key=lambda s: os.path.getmtime(os.path.join(folder_path, s)))
    
    # get list of images
    imgs = list(filter(lambda s: modality in s.upper() and not 'MASK' in s.upper(), files))
    # list of registered images
    registered_imgs = list(filter(lambda s: f'{modality}_TO_' in s.upper(), imgs))

    # get list of masks
    masks = list(filter(lambda s: 'MASK' in s.upper() and 'EDIT' in s. upper(), files))
    
    # if there are registered images, use the most recent one with the most recent mask
    if len(registered_imgs) > 0:
        return os.path.join(folder_path, registered_imgs[0]), os.path.join(folder_path, masks[0])

    # no registered images, use most recent regular image with most recent mask
    return os.path.join(folder_path, imgs[0]), os.path.join(folder_path, masks[0])


subjects_set = set()
print(f"Extracting candidates into {vetted_folder}...")
for person in people:
    person_folder = os.path.join(root_folder, person)
    
    subjects = os.listdir(person_folder)
    
    for subject in subjects:
        subject_folder = os.path.join(person_folder, subject)
        new_subject_folder = os.path.join(vetted_folder, subject)
        
        if new_subject_folder in subjects_set:
            print(f'{new_subject_folder} already exists, skipping...')
            continue
        
        os.makedirs(new_subject_folder)
        subjects_set.add(new_subject_folder)
        
        img, mask = find_img_and_mask_to_extract(subject_folder)
        
        new_image_name = os.path.join(new_subject_folder, f'{subject}_{modality.upper()}.nii.gz')
        new_mask_name = os.path.join(new_subject_folder, f'{subject}_MASK.nii.gz')
        
        shutil.copy2(img, new_image_name)
        shutil.copy2(mask, new_mask_name)

# try to find duplicates

duplicate_dict = {}

def find_ipssid(folder):
    # try to find the ipssid
    
    # get rid of anything that looks like "IPSS_011_"
    new_folder_name = re.sub(r"IPSS_[0-9]+_", "", folder)
    # get rid of anything that looks like "_01_SE01_MR"
    new_folder_name = re.sub(r"_[0-9]+_SE[0-9]+_MR", "", new_folder_name)
    # get rid of anything that looks like "SLC01_HSC_"
    new_folder_name = re.sub(r"SLC[0-9]+_[A-Z]+_", "", new_folder_name)
    return {
        "id": new_folder_name,
        "folder": folder
    }

# for each subject, find the ipssid
subjects = os.listdir(vetted_folder)
ipssids = list(map(find_ipssid, subjects))

# sort subjects by ipssid and print which ones are duplicates
for id in ipssids:
    if not id['id'] in duplicate_dict:
        duplicate_dict[id['id']] = [id['folder']]
    else:
        duplicate_dict[id['id']].append(id['folder'])

print('The following could be duplicates:')
for entry in duplicate_dict.values():
    if len(entry) > 1:
        print(entry)
