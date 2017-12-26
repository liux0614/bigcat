import os
import sys

import numpy as np
import cv2

def copy_and_move_images(image_names, output_dir, resize=None):
    """Copy and move the given image files to the output dir.

    Args:
        image_names: A list of paths to png or jpg images.
        output_dir: The directory where the converted image files are stored. 
    """
    
    start_idx = 0
    end_idx = len(image_names)
    for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Copy and move images %d/%d' % (i+1, len(image_names)))
        sys.stdout.flush()

        image_data = Image.open(image_names[i]) # Open image file
        image_data.resize(resize) if resize is not None
        file_name = os.path.basename(image_names[i])
        class_name = os.path.basename(os.path.dirname(image_names[i]))
        
        path_to_save = os.path.join(dataset_dir, os.path.join(class_name, file_name))

        image_data.save(path_to_save)

    sys.stdout.write('\n')
    sys.stdout.flush()