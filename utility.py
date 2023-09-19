import cv2
import os
import numpy as np

def readim(dir, readtype):
    # opencv doesn't support unicode. if image contains characters not in ASCII, must use something other than opencv to read the file in
    try:
        bytes(dir, "ascii")
    except UnicodeEncodeError:
        img = cv2.imdecode(np.fromfile(dir, dtype=np.uint8), readtype)
    else:
        img = cv2.imread(dir, readtype)

    if img is None:
        raise OSError(f"Could not read image at {dir}")
    else:
        return img
