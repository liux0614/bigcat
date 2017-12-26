import os

import numpy as np
import PIL as pil_image

from keras import backend as K
import keras.preprocessing.image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import Iterator

def simpple_image_data_generator(batch_size, imagepaths, classes):
    class_indices = dict(zip(classes, range(len(classes))))
    while True:
        for start in np.arange(0, len(imagepaths), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(imagepaths))
            train_batch = imagepaths[start:end]
            for imagepath in train_batch:
                img = pil_image.imread(imagepath)
                class_name = os.path.basename(os.path.dirname(imagepaths[i]))
                
                x_batch.append(img)
                y_batch.append(class_indices[class_name])
            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch, np.uint8)
            yield x_batch, y_batch

class ImageFileDataGenerator(ImageDataGenerator):
    def __init__(self,
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                zca_epsilon=1e-6,
                rotation_range=0.,
                width_shift_range=0.,
                height_shift_range=0.,
                shear_range=0.,
                zoom_range=0.,
                channel_shift_range=0.,
                fill_mode='nearest',
                cval=0.,
                horizontal_flip=False,
                vertical_flip=False,
                rescale=None,
                preprocessing_function=None,
                data_format=None):
        super(ImageFileDataGenerator, self).__init__(
            self,
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-6,
            rotation_range=0.,
            width_shift_range=0.,
            height_shift_range=0.,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=False,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,
            data_format=None)
    def flow_from_filenames(self, filenames, image_data_generator,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            data_format=None, save_to_dir=None,
                            save_prefix='', save_format='png',
                            interpolation='nearest')
        return FilenamesIterator(
            filenames, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            interpolation=interpolation)

class FilenamesIterator(Iterator):
    """Iterator capable of reading images from a list of filenames.
    Args:
        filenames: filenames represent paths to the images.
            Each filename should ended with classes the image belongs to
            and the base image name in the following pattern: 
            'image_class/image_name'.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Names of image categories, containing images from each class
            (e.g. `["dogs", "cats"]`). ValueError is raised if it's not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    
    Raises:
        ValueError: in case color mode and class_mode are invalid,
            and classes is None
    
    """

    def __init__(self, filenames, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None, save_to_dir=None,
                 save_prefix='', save_format='png',
                 interpolation='nearest'):
        if data_format is None:
            data_format = K.image_data_format()
        self.filenames = sorted(filenames)
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        if not classes:
            raise ValueError('Null classes; expected one represented all '
                            'categories.')
        else:
            self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation

        # first, count the number of samples and classes
        self.samples = 0
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))
        self.samples = len(filenames)
        print('By your input, it indicates %d images belonging to %d '
              'classes.' % (self.samples, self.num_classes))

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm'}
        # second, build an mapping of the valid images and their corresponding categories
        self.classes = np.zeros((self.samples,), dtype='int32')
        i = 0
        valid_filenames = []
        for filename in self.filenames:
            base_filename = os.path.basename(filename)
            for extension in white_list_formats:
                is_valid = False
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
                if is_valid:
                    subdir = os.path.basename(os.path.dirname(filename)) # class this file belongs to
                    self.classes.append(class_indices[subdir])
                    valid_filenames += filename
        self.filenames = valid_filenames
        self.samples = len(self.filenames)
        print('Files ended with one of %s will be recognized as images. '
              'By scanning, %d files are valid.' % (white_list_formats, self.samples)

        super(FilenamesIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(fname,
                           grayscale=grayscale,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e7),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_classes), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)