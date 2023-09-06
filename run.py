import os

import argparse

parser = argparse.ArgumentParser(
    prog='Data Preparation for Mask Generation CNN Training',
    description='Pipeline to find and preprocess images for training a Mask Generation CNN'
)

parser.add_argument('-r', '--root', required=True)
parser.add_argument('-s', '--step', required=True)
parser.add_argument('-m', '--modality', required=True)
parser.add_argument('-e', '--email')

args = parser.parse_args()

step = args.step.lower()
root_dir = args.root
modality = args.modality.upper()

if step == 'find':
    slurm_script = f"""#!/bin/bash

#SBATCH --mem=64G
#SBATCH -c 68

module load python/3.8.0
export PYTHONPATH=/hpf/projects/ndlamini/scratch/wgao/python3.8.0/
export PYTHONUNBUFFERED=TRUE

python ./data_prep/find_candidates.py {modality} {root_dir}
python ./data_prep/generate_images.py {modality} {root_dir} {os.path.join(root_dir, 'candidate_images_book.pdf')}

echo Done!
"""
    print(slurm_script)
elif step == 'prepare':
    vetted_folder = os.path.join(root_dir, 'all_images_vetted')
    assert not os.path.exists(vetted_folder), f"Vetted folder {vetted_folder} already exists"

    slurm_script = f"""#!/bin/bash

#SBATCH --mem=64G
#SBATCH -c 68

module load python/3.8.0
export PYTHONPATH=/hpf/projects/ndlamini/scratch/wgao/python3.8.0/
export PYTHONUNBUFFERED=TRUE

python ./data_prep/move_candidates.py {modality} {root_dir} {vetted_folder}
python ./data_prep/generate_images.py {modality} {vetted_folder} {os.path.join(root_dir, 'vetted_images_book.pdf')}

echo Done!
"""
    print(slurm_script)
elif step == 'final':
    vetted_folder = os.path.join(root_dir, 'all_images_vetted')
    assert os.path.exists(vetted_folder), f"Vetted folder {vetted_folder} does not exist, did you forget the 'prepare' step?"

    slurm_script = f"""#!/bin/bash

#SBATCH --mem=64G
#SBATCH -c 68

module load python/3.8.0
export PYTHONPATH=/hpf/projects/ndlamini/scratch/wgao/python3.8.0/
export PYTHONUNBUFFERED=TRUE

python ./data_prep/resample_and_distribute.py {modality} {root_dir} {vetted_folder}

echo Done!
"""
    print(slurm_script)
elif step == 'train':
    slurm_script = f"""#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user={args.email or ''}
#SBATCH -c 1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --qos=extralong
#SBATCH -t 10-00:00:00

module load python/3.8.0
export PYTHONPATH=/hpf/projects/ndlamini/scratch/wgao/python3.8.0/
export LD_LIBRARY_PATH=$PYTHONPATH/nvidia/cudnn/lib:$PYTHONPATH/tensorrt_libs:/hpf/tools/centos7/cuda/11.2/lib64:$LD_LIBRARY_PATH

python train.py {modality} {root_dir}

echo Done!
"""
    print(slurm_script)
elif step == 'predict':
    model_location = os.path.join(os.path.dirname(__file__), f'{modality.lower()}_brain_extraction.keras')
    slurm_script = f"""#!/bin/bash

#SBATCH --mem=64G
#SBATCH -c 68

module load python/3.8.0
export PYTHONPATH=/hpf/projects/ndlamini/scratch/wgao/python3.8.0/
export PYTHONUNBUFFERED=TRUE

python predict.py {root_dir} {modality} {model_location}

echo Done!
"""
    print(slurm_script)