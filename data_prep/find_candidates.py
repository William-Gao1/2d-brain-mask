import os
import shutil
from sys import argv

def extract_files_from_folder(folder_path, modality):
    """Given a folder path, figure out if it's a folder that we want to extract. If it is, return the files that should be extracted

    Args:
        folder_path (str): Path to folder
        modality(str): Modality to look for

    Returns:
        (bool, List[str]): First argument is true if folder should be extracted, if true then the second argument is which files should be extracted (full paths)
    """
    try:
        files = os.listdir(folder_path)
    except PermissionError:
        print(f'Permission denied on folder {folder_path}, skipping...')
        return False, None
    
    files_to_extract = []

    # dont want anything that doesnt end with .nii.gz and we dont want anything that is not axial and we dont want any 'brain' files and we dont want ant 'bet' files
    files = list(filter(lambda s: s.endswith('.nii.gz') and not ('COR' in s.upper() or 'SAG' in s.upper() or 'BRAIN' in s.upper() or 'BET' in s.upper()), files))

    has_img = False
    has_mask = False

    for file in files:
        # if it has the modality in the name and it's not the target of a registration (i.e. if we are looking for t1s, we dont want 1234_DWI_to_T1.nii.gz) and it is not a mask
        if modality.upper() in file and 'MASK' not in file.upper() and f'TO_{modality}_' not in file.upper():
            has_img = True
            files_to_extract.append(file)
        elif 'MASK' in file.upper() and 'EDIT' in file.upper():
            has_mask = True
            files_to_extract.append(file)
    

    # convert files_to_extract to full paths
    files_to_extract = [os.path.join(folder_path, file) for file in files_to_extract]

    if has_img and has_mask:
        return True, files_to_extract

    return False, None

folder_set = set()

def find_folders(search_folder, save_folder, modality):
    """Search for folders to extract and extract them

    Args:
        search_folder (str): Folder to search in
        save_folder (str): Folder to save results
        modality (str): Modality to look for
    """
    
    # see if we should extract this folder
    should_extract, files_to_extract = extract_files_from_folder(search_folder, modality)

    if should_extract:
        print(f'Extracting {files_to_extract} from {search_folder}...')
        # create folder to extract to
        folder_name = search_folder.split(os.sep)[-1]
        folder_to_save = os.path.join(save_folder, folder_name)

        # if folder already exists, keep appending 'A' until we get a unique folder name
        while folder_to_save in folder_set:
            folder_to_save += 'A'

        folder_set.add(folder_to_save)

        # create dest folder
        os.makedirs(folder_to_save, exist_ok=False)

        # copy files
        for file_to_extract in files_to_extract:
            file_name = file_to_extract.split(os.sep)[-1]
            dest_file = os.path.join(folder_to_save, file_name)
            try:
                shutil.copy2(file_to_extract, dest_file)
            except PermissionError:
                print(f'Permission denied on file {file_name}. Removing {folder_to_save}')
                shutil.rmtree(folder_to_save)
                folder_set.remove(folder_to_save)
                break

        
        print(f'Finished extracting from {search_folder}...')

    # recurse on all folders in this folder
    try:
        with os.scandir(search_folder) as it:
            for item in it:
                if item.is_dir():
                    find_folders(item.path, save_folder, modality)
    except PermissionError:
        pass

### Main entry point

# get the modality we are looking for and the destination folder from command line arguments
modality = argv[1]
dest_dir = argv[2]

# scratch folder
scratch_folder = '/hpf/projects/ndlamini/scratch/'

# assert both the scratch folder and destination folder exist
assert os.path.exists(scratch_folder), "Scratch folder does not exist"
assert os.path.exists(dest_dir), f"Destination root folder {dest_dir} does not exist"

# people's scratch folders to look into
people_to_look = ["msheng", "rhirji", "jli", "ntalwar", "wagenaar"]
for person in people_to_look:
    folder_to_look = os.path.join(scratch_folder, person)
    dest_folder = os.path.join(dest_dir, person)

    # make the corresponing person's folder in the destination folder
    os.makedirs(dest_folder, exist_ok=False)

    find_folders(folder_to_look, dest_folder, modality)

print(f'Found {len(folder_set)} folders')
