import shutil
import time

from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras import backend as K
import numpy as np
import cv2
import os
import os.path as osp


def elastic_transform(image, alpha, sigma, random_state=None):
    """ call dirty
    Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def dirty(im):
    """ adds noise and warping to returned image"""
    return elastic_transform(im, im.shape[1] * 2, im.shape[1] * 0.1)


def generate(imgs_to_gen, folder_save_to, img_location):
    datagen = ImageDataGenerator(
        rotation_range=3,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.06,
        horizontal_flip=False,
        fill_mode='nearest')

    img = load_img(img_location)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    n_imgs_created = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=folder_save_to, save_prefix='note', save_format='png'):
        n_imgs_created += 1
        if n_imgs_created > imgs_to_gen:  # generate this many images
            break  # otherwise the generator would loop indefinitely


# import images from ./trimmed and output to ./generated
INPUT_FOLDER  = "sheets/training/trimmed"
OUTPUT_FOLDER = "sheets/training/generated"
NOISED_FOLDER = "sheets/training/noised"
# delete output folder
shutil.rmtree(OUTPUT_FOLDER)
os.mkdir(OUTPUT_FOLDER)
imgs = 0
dirty_times = 1
folders = os.listdir(INPUT_FOLDER)
imgs_per_class = 20000/len(folders)
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
        generate(imgs_to_gen=imgs_to_gen,
                 folder_save_to=osp.join(OUTPUT_FOLDER, folder),
                 img_location=osp.join(INPUT_FOLDER, folder, file))
        # eta
        imgs += 1
        imgs_in_this_class += 1
        elapsed_time = time.time() - start_time
        time_per_file = elapsed_time / imgs
        total_time = time_per_file * total_n_of_files
        remaining_time = total_time - elapsed_time
        print('Total {}, Done {}/{} Elapsed: {:02.01f} mins, ETA: {:02.01f} mins'.format(total_n_of_files,
                                                                                                      imgs_in_this_class,
                                                                                                      len(files),
                                                                                                      elapsed_time / 60,
                                                                                                      remaining_time / 60))

print('number of input imgs ', imgs)
print('imgs generated ', imgs * imgs_to_gen * dirty_times)

"""# keras training

run this section to train the model
"""


batch_size = 32
epochs = 20

folders = os.listdir(OUTPUT_FOLDER)
num_classes = len(folders)
current_class = 0
images = []
labels = []
for folder in folders:
    print('{}: \"{}\"'.format(current_class, folder))
    files = os.listdir(osp.join(OUTPUT_FOLDER, folder))
    for file in files:
        if not file.endswith(".png"):
            if os.path.isfile(file):
                os.remove(file)
            continue
        images.append(cv2.imread(osp.join(OUTPUT_FOLDER, folder, file), 0))
        labels.append(current_class)
    current_class += 1

images = np.array(images)
labels = np.array(labels)

print(images.shape)
print(labels.shape)
print(labels)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

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

# old
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape))
print('model: added conv layer filters=32, kernel=(5,5)')
model.add(Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu'))
print('model: added conv layer filters=32, kernel=(5,5)')
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
print('model: added dense layer size=num_classes')

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

