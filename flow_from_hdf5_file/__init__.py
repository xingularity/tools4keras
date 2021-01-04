import os
import atexit
import threading

import numpy as np
import h5py

from keras_preprocessing import get_keras_submodule

try:
    IteratorType = get_keras_submodule('utils').Sequence
except ImportError:
    IteratorType = object

from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.preprocessing.image import Iterator
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class BatchFromHDF5Mixin():
    """Adds methods related to getting batches from filenames
    It includes the logic to transform image files to batches.
    """

    def set_processing_attrs(self,
                             image_data_generator,
                             subset):
        """Sets attributes to use later for processing files into a batch.
        # Arguments
            image_data_generator: Instance of `ImageDataGenerator`
                to use for random transformations and normalization.
            h5file: HDF5 file opened with h5py to read data from.
            subset: Subset of data (`"training"` or `"validation"`) if
                validation_split is set in ImageDataGenerator.
        """
        self.image_data_generator = image_data_generator
        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError(
                    'Invalid subset name: %s;'
                    'expected "training" or "validation"' % (subset,))
        else:
            split = None
        self.split = split
        self.subset = subset

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.
        # Arguments
            index_array: Array of sample indices to include in batch.
        # Returns
            A batch of transformed samples.
        """
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        imgpaths = self.imgpaths

        with h5py.File(self.h5file_path, 'r') as h5f:
            for i, j in enumerate(index_array):
                img_location = self.imgpaths[j]
                x = h5f[img_location[0]][img_location[1]]
                if self.image_data_generator:
                    params = self.image_data_generator.get_random_transform(x.shape)
                    x = self.image_data_generator.apply_transform(x, params)
                    x = self.image_data_generator.standardize(x)
                batch_x[i] = x
            # build batch of labels
            if self.class_mode == 'input':
                batch_y = batch_x.copy()
            elif self.class_mode in {'binary', 'sparse'}:
                batch_y = np.empty(len(batch_x), dtype=self.dtype)
                for i, n_observation in enumerate(index_array):
                    batch_y[i] = self.classes[n_observation]
            elif self.class_mode == 'categorical':
                batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                                dtype=self.dtype)
                for i, n_observation in enumerate(index_array):
                    batch_y[i, self.classes[n_observation]] = 1.
            elif self.class_mode == 'multi_output':
                batch_y = [output[index_array] for output in self.labels]
            elif self.class_mode == 'raw':
                batch_y = self.labels[index_array]
            else:
                return batch_x
            if self.sample_weight is None:
                return batch_x, batch_y
            else:
                return batch_x, batch_y, self.sample_weight[index_array]

    @property
    def filepaths(self):
        """List of absolute paths to image files"""
        raise NotImplementedError(
            '`filepaths` property method has not been implemented in {}.'
            .format(type(self).__name__)
        )

    @property
    def labels(self):
        """Class labels of every observation"""
        raise NotImplementedError(
            '`labels` property method has not been implemented in {}.'
            .format(type(self).__name__)
        )

    @property
    def sample_weight(self):
        raise NotImplementedError(
            '`sample_weight` property method has not been implemented in {}.'
            .format(type(self).__name__)
        )


class HDF5Iterator(BatchFromHDF5Mixin, Iterator):
    """Iterator capable of reading images from a directory on disk.
    # Arguments
        
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        h5file_path: Path to open HDF5 file
        target_size: tuple of integers, dimensions to resize input images to.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
            If set to False, sorts the data in alphanumeric order.
        seed: Random seed for data shuffling.
        follow_links: boolean,follow symbolic links to subdirectories
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        dtype: Dtype to use for generated arrays.
    """
    allowed_class_modes = {'categorical', 'binary', 'sparse', 'input', None}

    def __new__(cls, *args, **kwargs):
        try:
            from tensorflow.keras.utils import Sequence as TFSequence
            if TFSequence not in cls.__bases__:
                cls.__bases__ = cls.__bases__ + (TFSequence,)
        except ImportError:
            pass
        return super(HDF5Iterator, cls).__new__(cls)

    def __init__(self,
                 h5file_path,
                 image_data_generator,
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 subset=None,
                 interpolation='nearest',
                 dtype='float32'):
        super(HDF5Iterator, self).set_processing_attrs(image_data_generator, subset)
        self.h5file_path = h5file_path
        if class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(class_mode, self.allowed_class_modes))
        self.class_mode = class_mode
        self.dtype = dtype

        image_shape = None
        self.samples = 0

        with h5py.File(h5file_path, 'r') as h5f:           
            if not classes:
                classes = list(h5f.keys())
            self.class_indices = dict(zip(classes, range(len(classes))))

            # First, count the number of samples and classes and check image shape
            class_start_end = dict()
            for one_class in classes:
                num_images = h5f[one_class].shape[0]
                class_start_end[one_class] = int(num_images * self.split[0]), int(num_images * self.split[1])
                self.samples += class_start_end[one_class][1] - class_start_end[one_class][0]

                if image_shape is None:
                    image_shape = h5f[one_class].shape[1:]
                else:
                    if image_shape != tuple(h5f[one_class].shape[1:]):
                        raise ValueError("Incorrect image shape in class " + one_class + 
                                         ". Expect " + str(image_shape) + " but found " + 
                                         str(h5f[one_class].shape[1:]))
            if image_shape is not None:
                self.image_shape = image_shape

            # Second, build an index of the images
            # in the different class subfolders.

            self.classes = np.zeros((self.samples,), dtype='int32')
            self._imgpaths = list()
            i = 0
            for one_class in classes:
                class_index = self.class_indices[one_class]
                numer_of_images_in_class = class_start_end[one_class][1] - class_start_end[one_class][0]
                self.classes[i:i + numer_of_images_in_class] = self.class_indices[one_class]
                self._imgpaths += [(one_class, x) for x in range(class_start_end[one_class][0], class_start_end[one_class][1])]
                i = i + numer_of_images_in_class

        super(HDF5Iterator, self).__init__(self.samples, batch_size, shuffle, seed)

    @property
    def imgpaths(self):
        return self._imgpaths

    @property
    def labels(self):
        return self.classes

    @property  # mixin needs this property to work
    def sample_weight(self):
        # no sample weights will be returned
        return None

def flow_from_hdf5(self,
                   h5file_path,
                   classes=None,
                   class_mode='categorical',
                   batch_size=32,
                   shuffle=True,
                   seed=None,
                   subset=None):
    """Takes the path to a directory & generates batches of augmented data.
    # Arguments
        h5file_path: string, path to the HDF5 file.
        classes: Optional list of class subdirectories
            (e.g. `['dogs', 'cats']`). Default: None.
            If not provided, the list of classes will be automatically
            inferred from the subdirectory names/structure
            under `directory`, where each subdirectory will
            be treated as a different class
            (and the order of the classes, which will map to the label
            indices, will be alphanumeric).
            The dictionary containing the mapping from class names to class
            indices can be obtained via the attribute `class_indices`.
        class_mode: One of "categorical", "binary", "sparse",
            "input", or None. Default: "categorical".
            Determines the type of label arrays that are returned:
            - "categorical" will be 2D one-hot encoded labels,
            - "binary" will be 1D binary labels,
                "sparse" will be 1D integer labels,
            - "input" will be images identical
                to input images (mainly used to work with autoencoders).
            - If None, no labels are returned
                (the generator will only yield batches of image data,
                which is useful to use with `model.predict_generator()`).
                Please note that in case of class_mode None,
                the data still needs to reside in a subdirectory
                of `directory` for it to work correctly.
        batch_size: Size of the batches of data (default: 32).
        shuffle: Whether to shuffle the data (default: True)
            If set to False, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        subset: Subset of data (`"training"` or `"validation"`) if
            `validation_split` is set in `ImageDataGenerator`.
    # Returns
        A `HDF5Iterator` yielding tuples of `(x, y)`
            where `x` is a NumPy array containing a batch
            of images with shape `(batch_size, *target_size, channels)`
            and `y` is a NumPy array of corresponding labels.
    """
    return HDF5Iterator(
        h5file_path,
        self,
        classes=classes,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        subset=subset,
        dtype=self.dtype
    )

ImageDataGenerator.flow_from_hdf5 = flow_from_hdf5
