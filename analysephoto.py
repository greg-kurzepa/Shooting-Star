import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import statistics
import math

import utility as u

zoom_buf = 20 # number of px in each direction that zoom mode will leave in that isn't part of the cell bounding box

class Cell:
    def __init__(self):
        self.gray_photo = None # grayscale photo of this cell only
        self.original_gray_photo = None # original entire photo including all cells
        self.bw_photo = None # black and white photo of this cell only
        self.cell_area = None

        self.coords = None # list of tuples containing coordinates of every pixel contained in cell

        self.body = None # grayscale photo of cell body only
        self.body_area = None
        self.comet = None # grayscale photo of cell comet only
        self.comet_area = None

        self.body_sum_normalised_intensity = None
        self.comet_sum_normalised_intensity = None
        self.comet_proportion_intensity = None

        self.histogram_freqs = None # histogram of pixel intensities in cell
        self.histogram_edges = None
        self.hist_boundary_idx = None # index in histogram of body/comet threshold

        # coordinate information of overall cell
        self.topleft = None # (y,x) value of topleft corner
        self.width = None
        self.height = None

        # outlines, used for displaying on screen in pygame
        self.cell_outline = None
        self.body_outline = None
        self.comet_outline = None

        # used for displaying zoomed mode in pygame
        self.zoomed_original_photo = None
        self.zoomed_gray_photo = None
        self.zoomed_body_outline = None
        self.zoomed_comet_outline = None

        self.flag = "normal" # 'normal', 'outlier' or 'deleted'

    def update_hist_boundary(self, hist_boundary_idx):
        self.hist_boundary_idx = hist_boundary_idx
        threshold = self.histogram_edges[hist_boundary_idx]

        # make cell body image from this threshold
        self.body = np.zeros_like(self.gray_photo).astype(np.uint8)
        self.body[self.gray_photo >= threshold] = 255
        # fill holes in cell comet
        if self.body[0][0] == 0:
            cv.floodFill(self.body, None, (0,0), 100)
            self.body[self.body != 100] = 255
            self.body[self.body == 100] = 0
        else: print("Warning: image has stuff in bottomleft corner so holes in cell bodies won't be filled")
        # remove all connected blobs from body except from the largest
        nb_blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(self.body)
        sizes = stats[:, -1]
        sizes = sizes[1:]
        nb_blobs -= 1
        self.body[im_with_separated_blobs != np.argmax(sizes) + 1] = 0

        self.comet = np.zeros_like(self.gray_photo).astype(np.uint8)
        self.comet[np.logical_and(self.body != 255, self.gray_photo != 0)] = 255

        self.body_area = np.count_nonzero(self.body)
        self.comet_area = np.count_nonzero(self.comet)

        self.body_sum_normalised_intensity = np.sum((self.body/255) * (self.original_gray_photo/255))
        self.comet_sum_normalised_intensity = np.sum((self.comet/255) * (self.original_gray_photo/255))
        self.comet_proportion_intensity = self.comet_sum_normalised_intensity / (self.comet_sum_normalised_intensity + self.body_sum_normalised_intensity)

        self.body_outline = cv.Canny(self.body, 100, 200)
        self.comet_outline = cv.Canny(self.comet, 100, 200)

        self.zoomed_body_outline = self.body_outline[max(self.topleft[0]-zoom_buf, 0):min(self.topleft[0]+self.height+zoom_buf, self.original_gray_photo.shape[0]),
                                                     max(self.topleft[1]-zoom_buf, 0):min(self.topleft[1]+self.width+zoom_buf, self.original_gray_photo.shape[1])]
        self.zoomed_comet_outline = self.comet_outline[max(self.topleft[0]-zoom_buf, 0):min(self.topleft[0]+self.height+zoom_buf, self.original_gray_photo.shape[0]),
                                                       max(self.topleft[1]-zoom_buf, 0):min(self.topleft[1]+self.width+zoom_buf, self.original_gray_photo.shape[1])]

        in_dilate_size = 2
        self.body_outline = cv.dilate(self.body_outline, cv.getStructuringElement(cv.MORPH_ELLIPSE, (in_dilate_size, in_dilate_size)))
        self.comet_outline = cv.dilate(self.comet_outline, cv.getStructuringElement(cv.MORPH_ELLIPSE, (in_dilate_size, in_dilate_size)))

    def get_zoomed_rgba_outlines(self):
        # display = self.zoomed_original_photo
        display = np.zeros_like(self.zoomed_original_photo, np.uint8)
        display[:,:,1] = self.zoomed_gray_photo
        display[:,:,2] = self.zoomed_body_outline # red is body
        display[:,:,0] = self.zoomed_comet_outline # blue is comet
        # plt.imshow(display, cmap="gray"), plt.show()
        return display

class Photo:
    def __init__(self, image_dir):
        self.original_photo = u.readim(image_dir, cv.IMREAD_UNCHANGED)
        self.original_gray_photo = cv.cvtColor(self.original_photo, cv.COLOR_BGR2GRAY)

        # subtract background & normalise to 0-255
        disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (100,100))
        self.gray_photo = cv.morphologyEx(self.original_gray_photo, cv.MORPH_TOPHAT, disc)
        self.gray_photo = (self.gray_photo / self.gray_photo.max() * 255).astype(np.uint8)

        self.cells = None
        self.nb_blobs, self.im_with_separated_blobs, self.stats, self._ = self.find_cells()
    
    def find_cells(self):
        # global thresholding
        ret1,th1 = cv.threshold(self.gray_photo, 20, 255, cv.THRESH_BINARY)

        # create array of blobs
        nb_blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(th1)
        sizes = stats[:, -1]
        sizes = sizes[1:]
        nb_blobs -= 1

        # remove small blobs
        min_prop_size = 0.0003
        im_result = np.zeros_like(im_with_separated_blobs).astype(np.uint8)
        # for every component in the image, keep it only if it's above min_size
        px_count = self.original_photo.shape[0] * self.original_photo.shape[1]
        for blob in range(nb_blobs):
            if sizes[blob] >= px_count*min_prop_size:
                # see description of im_with_separated_blobs above
                im_result[im_with_separated_blobs == blob + 1] = 255

        # create new array of blobs
        nb_blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(im_result)

        # generate images
        extracted_cells = []
        for i in range(1, nb_blobs):
            cell = Cell()

            # create cell black and white photo
            cell.bw_photo = np.zeros_like(im_with_separated_blobs).astype(np.uint8)
            cell.bw_photo[im_with_separated_blobs == i] = 255

            # create cell grayscale photo
            cell.gray_photo = self.gray_photo.copy()
            cell.gray_photo[cell.bw_photo == 0] = 0

            # get average intensity of patch
            selected_coords = np.where(cell.bw_photo == 255)
            cell.coords = zip(selected_coords[0], selected_coords[1]) # list of all coords the body+comet takes up
            cell.cell_avg_intensity = sum([self.gray_photo[idx] for idx in cell.coords]) / len(selected_coords[0])

            cell.topleft = (stats[i][1], stats[i][0]) # y, x value
            cell.width = stats[i][2]
            cell.height = stats[i][3]

            cell.original_gray_photo = self.original_gray_photo # pointer to above orginal_gray_photo that cell can access
            # print(max(cell.topleft[0]-cell.height-zoom_buf, 0),min(cell.topleft[0]+zoom_buf, self.original_gray_photo.shape[0]),
            #       max(cell.topleft[1]-zoom_buf, 0),min(cell.topleft[1]+cell.width+zoom_buf, self.original_gray_photo.shape[1]), "|",
            #       cell.topleft[0]-cell.height, cell.topleft[0], cell.topleft[1], cell.topleft[1]+cell.width)
            cell.zoomed_original_photo = self.original_photo[max(cell.topleft[0]-zoom_buf, 0):min(cell.topleft[0]+cell.height+zoom_buf, self.original_gray_photo.shape[0]),
                                                             max(cell.topleft[1]-zoom_buf, 0):min(cell.topleft[1]+cell.width+zoom_buf, self.original_gray_photo.shape[1])]
            cell.zoomed_gray_photo = cell.gray_photo[max(cell.topleft[0]-zoom_buf, 0):min(cell.topleft[0]+cell.height+zoom_buf, self.original_gray_photo.shape[0]),
                                                     max(cell.topleft[1]-zoom_buf, 0):min(cell.topleft[1]+cell.width+zoom_buf, self.original_gray_photo.shape[1])]

            extracted_cells.append(cell)

        # body+comet objects, exclude dim sections which probably aren't cells
        median_intensity = statistics.median([cell.cell_avg_intensity for cell in extracted_cells])
        self.cells = [x for x in extracted_cells if x.cell_avg_intensity >= 0.5*median_intensity]

        for cell in self.cells:
            cell.cell_area = np.count_nonzero(cell.bw_photo)
            cell.cell_outline = cv.Canny(cell.bw_photo, 100, 200)

            # make historgram, pick left side of modal bin in top 1/4 of bins as threshold value
            bins=10
            cell.histogram_freqs, cell.histogram_edges = np.histogram(cell.gray_photo.flatten(), bins=bins)
            min_threshold_bin_no = math.floor(0.75*bins)
            max_idx = max(np.argmax(cell.histogram_freqs[min_threshold_bin_no:])+min_threshold_bin_no-2, 0)

            cell.update_hist_boundary(max_idx)

        return nb_blobs, im_with_separated_blobs, stats, _
    
    def get_overall_img(self, lam, include_outliers=True):
        img = np.zeros_like(self.cells[0].bw_photo).astype(np.uint8)
        for cell in self.cells:
            if include_outliers or cell.flag == "normal": img += lam(cell)
        return img
    
    def get_rgba_outlines(self):
        # display = self.original_photo
        display = np.zeros_like(self.original_photo, np.uint8)
        display[:,:,1] = self.get_overall_img(lam=lambda x:x.gray_photo, include_outliers=True)
        display[:,:,2] = self.get_overall_img(lam=lambda x:x.body_outline, include_outliers=False) # red is body
        display[:,:,0] = self.get_overall_img(lam=lambda x:x.comet_outline, include_outliers=False) # blue is comet
        # plt.imshow(display, cmap="gray"), plt.show()
        return display
