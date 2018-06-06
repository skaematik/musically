import shutil
import time

from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, regularizers
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras import backend as K
import numpy as np
import cv2
import os
import os.path as osp

import imgaug as ia
from imgaug import augmenters as iaa

INPUT_FOLDER = "resources/sheets/training/trimmed"
OUTPUT_FOLDER = "resources/sheets/training/generated"
OUTPUT_MUTATED_FOLDER = "resources/sheets/training/mutated"
NOISED_FOLDER = "resources/sheets/training/noised"


def generate_data(INPUT_FOLDER, OUTPUT_FOLDER, NOISED_FOLDER=None):

    # import images from ./trimmed and output to ./generated

    # delete output folder
    try:
        shutil.rmtree(OUTPUT_FOLDER)
    except Exception:
        pass
    try:
        os.mkdir(OUTPUT_FOLDER)
    except Exception:
        pass
    imgs = 0
    dirty_times = 1
    folders = os.listdir(INPUT_FOLDER)
    imgs_per_class = 10000
    print('Images per class: {}'.format(imgs_per_class))
    total_n_of_files = 0
    for folder in folders:
        if folder.endswith(".DS_Store"):
            if os.path.isfile(folder):
                os.remove(folder)
            continue
        total_n_of_files += len(os.listdir(osp.join(INPUT_FOLDER, folder)))
    start_time = time.time()
    for folder in folders:
        if folder.endswith(".DS_Store"):
            continue
        # make new folder
        os.mkdir(osp.join(OUTPUT_FOLDER, folder))
        imgs_in_this_class = 0
        files = os.listdir(osp.join(INPUT_FOLDER, folder))
        print('{} imgs in {}'.format(len(files), osp.join(INPUT_FOLDER, folder)))
        imgs_to_gen = imgs_per_class // len(files)
        for file in files:
            if not file.endswith(".png"):
                if os.path.isfile(file):
                    os.remove(file)
                continue

            seq = iaa.Sequential([
                iaa.Affine(
                    mode="edge",
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    rotate=(-2, 2),
                    shear=(-4, 4)
                ),
                iaa.Sometimes(0.8, iaa.ElasticTransformation(
                    alpha=(0.5, 1.5), sigma=0.5)),
                iaa.Sometimes(0.5, iaa.GaussianBlur((0, 1.2))),
                # iaa.SimplexNoiseAlpha(iaa.OneOf([
                #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                # ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(
                    0.0, 0.05 * 255), per_channel=0.5),
                # add gaussian noise to images
                iaa.Sometimes(0.2,
                              [iaa.SaltAndPepper((0.01, 0.05), per_channel=0.5)  # randomly remove up to 10% of the pixels
                               ]),

            ], random_order=False)

            image = cv2.imread(osp.join(INPUT_FOLDER, folder, file), 0)
            images = [image] * imgs_to_gen
            images_aug = seq.augment_images(images)
            for i, image_aug in enumerate(images_aug):
                _, bin_img = cv2.threshold(image_aug, 0, 255, cv2.THRESH_OTSU)
                cv2.imwrite(osp.join(OUTPUT_FOLDER, folder,
                                     'note_{}_{}.png'.format(imgs, i)), bin_img)

            # generate(imgs_to_gen=imgs_to_gen,
            #          folder_save_to=osp.join(OUTPUT_FOLDER, folder),
            #          img_location=osp.join(INPUT_FOLDER, folder, file))
            # # eta
            imgs += 1
            imgs_in_this_class += 1
            elapsed_time = time.time() - start_time
            time_per_file = elapsed_time / imgs
            total_time = time_per_file * total_n_of_files
            remaining_time = total_time - elapsed_time
            if imgs % 500 == 0:
                print('Total {}, Done {}/{} Elapsed: {:02.01f} mins, ETA: {:02.01f} mins'.format(total_n_of_files,
                                                                                                 imgs_in_this_class,
                                                                                                 len(
                                                                                                     files),
                                                                                                 elapsed_time / 60,
                                                                                                 remaining_time / 60))

    print('imgs generated ', imgs * imgs_to_gen * dirty_times)
    print('number of input imgs ', imgs)


"""# keras training
run this section to train the model
"""
def train_model(OUTPUT_FOLDER):
    batch_size = 32
    epochs = 20

    folders = os.listdir(OUTPUT_FOLDER)
    num_classes = len(folders)
    current_class = 0
    images = []
    labels = []
    folders = sorted(folders)
    ",\n".join(folders)
    for folder in folders:
        if folder == '.DS_Store':
            continue
        files = os.listdir(osp.join(OUTPUT_FOLDER, folder))
        for file in files:
            if not file.endswith(".png"):
                if os.path.isfile(file):
                    os.remove(file)
                continue
            images.append(cv2.imread(osp.join(OUTPUT_FOLDER, folder, file), 0))
            labels.append(current_class)
        print('label: {}\t\tsamples: {}'.format(folder, len(images)))
        current_class += 1

    images = np.array(images)
    labels = np.array(labels)

    print(images.shape)
    print(labels.shape)
    print(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2)

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    n_samples, img_rows, img_cols = x_train.shape

    # x_train shape = [n_train_samples, img_rows, img_cols]
    # y_train shape = [n_train_samples]
    # x_test shape = [n_test_samples, img_rows, img_cols]
    # y_test shape = [n_test_samples]

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    # aka convert to one-hot
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu',
                     input_shape=input_shape, kernel_regularizer=regularizers.l2(0.0001)))
    print('model: added conv layer filters=32, kernel=(5,5)')
    model.add(Conv2D(32, kernel_size=(5, 5), padding='same',
                     activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    print('model: added conv layer filters=32, kernel=(5,5)')
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('keras_model.h5')
    print('keras model saved')

    # validation

    # use all images
    x_test = images
    y_test = labels

    if K.image_data_format() == 'channels_first':
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_test = x_test.astype('float32') / 255.0

    y_pred = model.predict_classes(x_test)
    print(y_test)
    # y_test = np.argmax(y_test, axis=1)
    print(y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)


if __name__ == '__main__':
    # generate_data(INPUT_FOLDER, OUTPUT_FOLDER)
    train_model(OUTPUT_FOLDER)
