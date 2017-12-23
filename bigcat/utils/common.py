import numpy as np

import warnings

def get_random_state(seed):
    """Get an instance of numpy.random.RandomState according to seed
    Code cloned from scikit learn

    Args:
        seed : None | int | instance of RandomState
            If seed is None, return the RandomState singleton used by np.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            Otherwise raise ValueError.
    
    Returns:
        An instance of np.random.RandomState

    Raises:
        Value Error: if seed cannot be used to generate an instance of RandomState
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def safe_indexing(X, indices):
    """Reindex X according to indices
    Code cloned from scikit learn
    
    Args:
        X: 
        indices: reindex X according to indices

    Returns:
        Reindexed X
    """
    if hasattr(X, "iloc"):
        # Work-around for indexing with read-only indices in pandas
        indices = indices if indices.flags.writeable else indices.copy()
        # Pandas Dataframes and Series
        try:
            return X.iloc[indices]
        except ValueError:
            # Cython typed memoryviews internally used in pandas do not support
            # readonly buffers.
            warnings.warn("Copying input dataframe for slicing.",
                          DataConversionWarning)
            return X.copy().iloc[indices]
    elif hasattr(X, 'shape'):
        if hasattr(X, 'take') and (hasattr(indices, 'dtype') and
                                   indices.dtype.kind == 'i'):
            # This is often substantially faster than X[indices]
            return X.take(indices, axis=0)
        else:
            return X[indices]
    else:
        return [X[idx] for idx in indices]

def shuffle(array_to_shuffle, random_state=None):
    """Shuffle array. If the array is a multi-dimensional array, 
    this function only shuffles along the first dimension.

    Args:
        array_to_shuffle: array to shuffle
        random_state: used to generate an instance of numpy.random.RandomState
    Returns:
        The shuffled array
    """
    if len(array_to_shuffle) == 0:
        return None
    
    n_samples = array_to_shuffle.shape[0] if hasattr(array_to_shuffle, 'shape') else len(array_to_shuffle)
    indices = np.arange(n_samples)
    random_state = get_random_state(random_state)
    random_state.shuffle(indices)

    shuffled_array = safe_indexing(array_to_shuffle, indices)
    return shuffled_array

def get_filenames_and_classes(dataset_dir):
    """Returns a list of filenames and inferred class names.

    Args:
        dataset_dir: A directory containing a set of subdirectories representing
            class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
        A list of image file paths, relative to `dataset_dir` and the list of
        subdirectories, representing class names.
    """
    directories = []
    class_names = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            filenames.append(path)

    return filenames, sorted(class_names)

def _validate_split_proportions(n_samples, proportions=None):
    """Calculate the actual samples in accord with proportions

    Args:
        n_samples: total sample size
        proportions: used to calculate how many samples in each part

    Returns:
        samples size of each part in a numpy array
    """
    if proportions is None:
        proportions = np.array([0.8, 0.2])
    else:
        proportions = np.asarray(proportions)
    
    if np.sum(proportions) <> 1.0:
        raise ValueError('total proportions should be 1')

    n_samples_per_part = np.around(n_samples * proportions)

    if np.sum(n_samples_per_part) <> n_samples:
        n_samples_per_part[-1] = n_samples - np.sum(n_samples_per_part[:-1])

    return n_samples_per_part

def _get_train_test_size(n_samples, percent_test=0.1, percent_train=None):
    """Calculate the test/test sizes
    """
    if (percent_test is not None and (percent_test < 0 or percent_test > 1)):
        raise ValueError("percent_test=%d should be smaller within 0-1" % (percent_test))

    if (percent_train is not None and (percent_train < 0 or percent_train > 1)):
            raise ValueError("percent_test=%d should be smaller within 0-1" % (percent_train))

    if (percent_test is None and percent_train is None):
        percent_test = 0.1    

    if percent_train is None:
        n_test = int(n_samples * percent_test)
        n_train = n_samples - n_test
    else:
        n_train = int(n_samples * percent_train)

    if percent_test is None:
        n_train = int(n_samples * percent_train)
        n_test = n_samples - n_train
    else:
        n_test = int(n_samples * percent_test)
    
    if n_train + n_test > n_samples:
        raise ValueError('The sum of train_size and test_size = %d should be smaller than '
                         'the number of samples %d.' % (n_train + n_test, n_samples))

    return n_train, n_test

def split_into_train_test(filenames, percent_test=0.1, percent_train=None, shuffle=True, random_state=None):
    """Split filenames into train, test, (val) sets

    Args:
        filenames: A list of paths to split
        percent_test: percent of test
        percent_train: percent of train
        shuffle: shuffle or not
        random_state: used to generate numpy.random.RandomState
    Returns:
        train_set: train set
        test_set: test set
        val_set: if percent_test + percent_train < 1, then valset will be returned
    """
    filesnames, class_names = get_filenames_and_classes(dataset_dir)
    shuffled_images = shuffle(filesnames, random_state) if shuffle

    n_samples = len(filesnames)
    n_train, n_test = _get_train_test_size(n_samples, percent_test, percent_train)

    sum_train_test = n_train + n_test
    if sum_train_test < n_samples:
        return filesnames[:n_train], filesnames[n_train:sum_train_test], filesnames[sum_train_test:]
    else:
        return filesnames[:n_train], filesnames[n_train:]
