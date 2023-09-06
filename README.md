# Brain Mask Generation CNN Training

This repository contains code to train a Convolutional Neural Network to create brain masks for Nifti images. As training is done on HPC, you must run all code in this repo on HPC

Data integrity is important for the network to learn well. Please read through this entire document to familiarize yourself with the data preparation process before starting.

## Introduction

There are 4 steps in this pipeline. The first 3 are for data preparation and the last one is training. The only script you will need is `./run`. In order to use it, you must first give permissions by running `chmod +x run`

### Data preparation

The goal of data preparation is to generate two folders `train` and `test` that contains the data needed to train the network. The folders `train` and `test` will each contain one more level of folders each containing 3 files, the image (input), the mask (label/ground truth), and a PDF plot of both (not used for training).

We will do this in three steps:

1. Scrape folders in the `/hpf/projects/ndlamini/scratch` directory for subjects that have an image of our desired modality and have a brain mask.
2. Manually check and select the image and mask for each subject that we want to use for training.
3. Resize all the images and masks and split the subjects into the `train` and `test` folders

## Preparation

Before starting, create an empty folder in an easily accessible location. This folder will be where all the magic happens. We will call this folder the **root folder**. Every step will output images or change things in the root folder. You do not have to worry about any other folders changing.

For the purposes of examples, we will say that we have an empty root folder at `/hpf/projects/ndlamini/scratch/wgao/ss_data/` and we are looking for *T1 Weighted* images.

## Step 1: Scrape Folders

In the first step, we will look for candidate subjects in people's scratch folder. We will then generate PDF plots that show the alignment between the brain and the mask. To do this run:

```bash
./run -r <path_to_root> -m <modality> -s find
```

In our example:

```bash
./run -r /hpf/projects/ndlamini/scratch/wgao/ss_data -m T1 -s find
```

This should take a couple of minutes.

After completion, you should see that the root folder has a couple of folders, each folder should look like a person's name.

In each person's folder you should see more folders, each with a couple of Nifti files and a file called `plt.pdf`

Besides people's folders, you should see a pdf file in the root folder called `candidate_images_book.pdf`

If all of the above is true, you have successfully completed the first step.

## Step 2: Manual Check

Step 2 is the most tedious and time consuming step. It is also the most crucial. It is important you read and understand what you have to do in this step as it will save you headaches in the future.

Open the `candidate_images_book.pdf` file that was generated in step 1. Your job is to check each subject and ensure that the image and mask are suitable for training. If they are not, either remove the subject folder or find an image and mask that are suitable (Explanation below).

For each set of images in `candidate_images_book.pdf` you should ensure the following

* The image on the left is clear
* The image on the left is *not* skull stripped
* The image on the left is actually an image of the desired modality
* The brain mask on the right is in good agreement with the underlying image

If all of the above are true, then you do not need to do anything else for this subject.

If one or more of the above is false you need to take corrective action:

1. Locate the folder of the subject (displayed at the top of the plot) and navigate to said folder.
2. Try to find a image-mask pair that meets the criteria above. If you find a pair that you are satisfied with, **delete all other** files and keep your desired image-mask pair.
3. If you can not find a suitable image-mask pair, **delete** the subject folder, it is not useable for training.

Once you are finished checking and correcting each subject, consolidate all the subjects by running:

```bash
./run -r <path_to_root> -m <modality> -s find
```

In our example:

```bash
./run -r /hpf/projects/ndlamini/scratch/wgao/ss_data -m T1 -s prepare
```

This will take a couple of minutes. Once completed, you should now see a new folder in the root folder called `all_images_vetted` and a new PDF file called `vetted_images_book.pdf`

If this is true, move to step 3.

## Step 3: Final check and resizing

Step 3 is the last step in the data prep part of the pipeline. In this step you will do a final check of the data and generate the `train` and `test` folders that the network will need.

Firstly, open `vetted_images_book.pdf` and double check that all the images and masks look good and are acceptable for training.

If you find a subject that is not acceptable, then find the subject folder (displayed at the top of the plot) and **delete** it. It is too late and difficult to try to fix the subject, as long as there are not many deletions, you should still be ok.

After verifying the integrity of your images, run

```bash
./run -r <path_to_root> -m <modality> -s final
```

In our example:

```bash
./run -r /hpf/projects/ndlamini/scratch/wgao/ss_data -m T1 -s prepare
```

This should generate two more folders in your root folder: `train` and `test`. It should also generate two more PDF files `train_subjects_book.pdf` and `test_subjects_book.pdf`.

If this is true, move on to step 4.

Note: The images in the new books are not regenerated, they are just for your reference.

## Step 4: Training

By now, you should have the `test` and `train` folders in your root folder. All that is left is to start the training:

In this step, you can optionally specifiy your email to be notified of when training is complete:

```bash
./run -r <path_to_root> -m <modality> -e <your_email> -s train
```

In our example:

```bash
./run -r /hpf/projects/ndlamini/scratch/wgao/ss_data -m T1 -e wgao@sickkids.ca -s train
```

Training will take a couple of days depending on how many subjects you have. For reference, it takes around 3 days for 120 subjects.

After training, you should see a `<modality>_brain_extraction.keras` file in *the location where you have the code* (not in the root folder)

In our case, we should see a `t1_brain_extraction.keras` file. If the model performs well (see the section on Prediction below) please push it to Git for others to use!

## Prediction

Once the model is finished training you can generate a prediction on a single Nifti file or an entire directory (e.g. your `train` folder)using

```bash
./run -r <path_to_nifti_or_folder> -m <modality> -e <your_email> -s predict
```

In our example:

```bash
./run -r /hpf/projects/ndlamini/scratch/wgao/ss_data/test/121/121_T1.nii.gz -m T1 -s predict
```

will generate a mask for `121_T1.nii.gz` only, or:

```bash
./run -r /hpf/projects/ndlamini/scratch/wgao/ss_data/test -m T1 -s predict
```

will generate a mask for any `_T1.nii.gz` file in the `test` folder and any subdirectories of `test`.

Any predictions will be saved to the same folder where the input Nifti was found with name `{folder}_pred.nii.gz`.

An interactive Jupyter Notebook is also provided at `load_model_and_predict.ipynb`
