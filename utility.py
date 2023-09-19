import cv2
import os

def readim(dir, readtype):
    img = cv2.imread(dir, readtype)
    if img is None:
        raise OSError(f"Could not read image at {dir}")
    else:
        return img

def writeim(dir, img, overwrite=False):
    if not overwrite and os.path.isfile(dir):
        raise OSError(f"Could not write image to {dir}, image alreay exists!")
    else:
        if not cv2.imwrite(dir, img):
            raise OSError(f"Could not write image to {dir}")   
    
def get_file_image_dimensions(img_dir):
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    return img.shape[1], img.shape[0]