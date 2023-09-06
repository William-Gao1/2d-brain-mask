import os
import cv2
import numpy as np


# neural imaging
import nibabel as nib


# ml libs
import keras
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

import albumentations as A

from sys import argv

modality = argv[1].upper()
root = argv[2]


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# there are 155 slices per volume
# to start at 5 and use 145 slices means we will skip the first 5 and last 5 
VOLUME_SLICES = 256
VOLUME_START_AT = 0 # first slice of volume that we will include
IMG_SIZE=128

TRAIN_DATASET_PATH = os.path.join(root, 'train')
TEST_DATASET_PATH = os.path.join(root, 'test')


# dice loss as defined above for 4 classes
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 2
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
    return total_loss



# Computing Precision 
def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    
# Computing Sensitivity      
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


# Computing Specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

# source https://naomi-fridman.medium.com/multi-class-image-segmentation-a5cc671e647a

def build_unet(img_size, ker_init, dropout):
    inputs = Input(shape=(img_size, img_size, 1))
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv1)
    
    pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool)
    conv = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv3)
    
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv5)
    drop5 = SpatialDropout2D(dropout)(conv5)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(drop5))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv9)
    
    up = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv9))
    merge = concatenate([conv1,up], axis = 3)
    conv = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge)
    conv = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv)
    
    conv10 = Conv2D(2, (1,1), activation = 'softmax')(conv)
    
    return Model(inputs = inputs, outputs = conv10)

mirrored_strategy = tf.distribute.MirroredStrategy()


with mirrored_strategy.scope():
    model = build_unet(IMG_SIZE, 'he_normal', 0.2)
    model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=2), dice_coef, precision, sensitivity, specificity] )

train_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]
test_directories = [f.path for f in os.scandir(TEST_DATASET_PATH) if f.is_dir()]


def pathListIntoIds(dirList):
    x = []
    for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    
    return x

train_ids = pathListIntoIds(train_directories); 
test_ids = pathListIntoIds(test_directories)


IMG_SIZE = 128
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, dim=(IMG_SIZE,IMG_SIZE), batch_size = 1, n_channels = 1, shuffle=True, train=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.data_path = TRAIN_DATASET_PATH if train else TEST_DATASET_PATH

        self.transform = None
        if train:
            self.transform = A.Compose([
                A.ToFloat(max_value=65535.0),
                A.Rotate(border_mode=cv2.BORDER_CONSTANT, value=0),
                #A.RandomResizedCrop(width=128, height=128, p=0.95),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                #A.Affine(translate_percent=(-0.5, 0.5), scale=(0.7, 1.3), keep_ratio=True, p=1),
                A.Resize(128, 128, always_apply=True)
            ])
        else:
            self.transform = A.Compose([
                A.ToFloat(max_value=65535.0),
                A.Resize(128, 128)
            ])


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim))
        Y = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, 2))

        
        # Generate data
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(self.data_path, i)
            
            files = os.listdir(case_path)
            
            img_files = list(filter(lambda s: s.endswith(f'_{modality.upper()}.nii.gz'), files))
            mask_files = list(filter(lambda s: s.endswith('_MASK_EDIT.nii.gz'), files))
            
            assert len(img_files) == 1
            assert len(mask_files) == 1

            data_path = os.path.join(case_path, img_files[0]);
            img = nib.load(data_path).get_fdata()
            
            data_path = os.path.join(case_path, mask_files[0]);
            mask = nib.load(data_path).get_fdata()
            
            img_sliced = img[:,:, VOLUME_START_AT:VOLUME_START_AT + VOLUME_SLICES]
            mask_sliced = mask[:,:, VOLUME_START_AT:VOLUME_START_AT + VOLUME_SLICES]


            transformed_images = self.transform(image=img_sliced, mask=mask_sliced)
            
            X[:,:,:,0] = np.moveaxis(transformed_images['image'], 2, 0).astype(np.float64, casting='same_kind')
            y = np.moveaxis(transformed_images['mask'], 2, 0).astype(np.float64, casting='same_kind')
        # Generate masks
        Y = tf.one_hot(y, 2)
        max_X = np.max(X)
        if max_X == 0:
            return X, Y
        return X/max_X, Y
        
training_generator = DataGenerator(train_ids, train=True)
test_generator = DataGenerator(test_ids, train=False)


callbacks = [
      keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.000001, verbose=1),
]


K.clear_session()

unet_history =  model.fit(training_generator,
                     epochs=1000,
                     steps_per_epoch=len(train_ids),
                     callbacks= callbacks,
                     validation_data = test_generator
                     )  
model.save(f"{modality}_brain_extraction.keras")
