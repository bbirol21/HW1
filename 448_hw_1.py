import cv2
import numpy as np
import scipy.ndimage as sn
from PIL import ImageEnhance, Image
import matplotlib.pyplot as plt
import copy
from skimage.feature import peak_local_max

# Part 1: ObtainForegroundMask function
import cv2
import numpy as np
import scipy.ndimage as sn

def ObtainForegroundMask(image_path):
    # read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0) 

    #  Otsu's method
    _, foreground_mask = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #  morphological closing (dilation +  erosion) to fill gaps
    kernel = np.ones((7, 7), np.uint8) #creates a square mask of ones of size 7x7
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

    #  dilation to expand the foreground regions (optional step for better segmentation)
    foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=2)
    
    foreground_mask = sn.binary_fill_holes(foreground_mask).astype(np.uint8)

    return foreground_mask


import cv2
import numpy as np
from skimage.feature import peak_local_max

def FindCellLocations(image_path, foreground_mask):
   
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

   
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # isolate potential foreground regions
    _, thresholded_image = cv2.threshold(blurred_image, 195, 255, cv2.THRESH_BINARY) #converts the blurred image into a binary image 
    #any pixel value above 195 is set to 255 to assign it to foreground

    # erode the thresholded image to remove small unwanted regions
    eroded_image = cv2.erode(thresholded_image, np.ones((3, 3), np.uint8))

    # remove regions outside the foreground mask 
    foreground_mask_copy = foreground_mask.copy()
    foreground_mask_copy[eroded_image == 255] = 0

    # perform distance transform to highlight areas closer to foreground objects
    distance_transform = cv2.distanceTransform(foreground_mask_copy, cv2.DIST_L2, 5)

    # normalization
    cv2.normalize(distance_transform, distance_transform, 0, 1.0, cv2.NORM_MINMAX)

    # find local maxima in the distance transform (cell locations)
    cell_locations = peak_local_max(distance_transform, min_distance=14)

    return cell_locations


import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt

def get_neighbors(coord, cell_id):
    neighbors = [[cell_id, coord[0], coord[1]]]
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            neighbors.append([cell_id, coord[0] + dx * 3, coord[1] + dy * 3])
    return neighbors


# takes 10 minutes to run but it runs correctly you can bu sure about it
def FindCellBoundaries(image_path, foreground_mask, cell_locations):


    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_blurred = cv2.GaussianBlur(image_gray, (3, 3), 0)

    # prepare the mask for region growing
    mask = foreground_mask.astype(int)
    mask[mask == 1] = -1  # Mark foreground regions with -1

    # pad the mask ( this is used to avoid boundary issues during region growing)
    padded_mask = np.pad(mask, ((3, 3), (3, 3)), mode='constant', constant_values=0)

    regions_to_process = []
    visited_cells = []

    # START region growing with the given cell locations
    for cell_id, coord in enumerate(cell_locations, start=1):
        regions_to_process.extend(get_neighbors(coord, cell_id))

    # **REGION GROWING***
    while regions_to_process:
        cell_id, x, y = regions_to_process.pop(0)  

        visited_cells.append([cell_id, x, y])

        if padded_mask[x + 3, y + 3] == -1:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    padded_mask[x + dx + 3, y + dy + 3] = cell_id  # assign cell id to neighbors

            # adding neighbors to regions_to_process
            new_neighbors = get_neighbors([x, y], cell_id)
            for neighbor in new_neighbors:
                if neighbor not in visited_cells:
                    regions_to_process.append(neighbor)

    segmented_mask = padded_mask[3:-3, 3:-3]  # remove padding
    segmented_mask[segmented_mask == -1] = 0  # set all unprocessed cells to background (0)

    return segmented_mask


import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_gold_data(mask_path, cells_path):

    data_mask = np.loadtxt(mask_path)
    data_coor = np.loadtxt(cells_path)
    return data_mask, data_coor

def process_images(image_paths, gold_mask_paths, gold_cells_paths):

    for img_idx, image_path in enumerate(image_paths):
       
        data_mask, data_coor = load_gold_data(gold_mask_paths[img_idx], gold_cells_paths[img_idx])

        # Part 1: Obtain Foreground Mask
        foreground_mask = ObtainForegroundMask(image_path)

        # Metrics for Part 1: Foreground Mask
        TP = np.sum((foreground_mask == 1) & (data_mask == 1))
        FP = np.sum((foreground_mask == 1) & (data_mask == 0))
        FN = np.sum((foreground_mask == 0) & (data_mask == 1))

        prec = TP / (TP + FP)
        recall = TP / (TP + FN)
        Fscore = TP / (TP + (1/2 * (FP + FN)))

        print(f'Part 1 Metrics for {image_path}:')
        print(f"Precision: {prec:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Fscore: {Fscore:.2f}\n")

        # Plot Part 1: Foreground Mask
        plt.imshow(foreground_mask, cmap='gray')
        plt.title(f"Foreground Mask (Part 1) - {image_path}")
        plt.xticks([]), plt.yticks([])  # Hide ticks
        plt.show()

        # Part 2: Find Cell Locations
        cell_locations = FindCellLocations(image_path, foreground_mask)

        # Metrics for Part 2: Cell Locations
        TP_2 = 0
        cells = []
        for x, y in cell_locations:
            cells = np.append(cells, data_coor[int(x)][int(y)])

        val = 999
        dup = 0
        for c in range(cells.size):
            if val != cells[c] and cells[c] != 0:
                TP_2 += 1
                dup = 0
            if val == cells[c] and cells[c] != 0:
                dup = 1
            val = cells[c]
        TP_2 -= dup

        prec_2 = TP_2 / cells.size
        recall_2 = TP_2 / data_coor.max()
        Fscore_2 = (2 * prec_2 * recall_2) / (prec_2 + recall_2)

        print(f'Part 2 Metrics for {image_path}:')
        print(f"Precision: {prec_2:.2f}")
        print(f"Recall: {recall_2:.2f}")
        print(f"Fscore: {Fscore_2:.2f}\n")

        # Plot Part 2: Cell Locations
        img = cv2.imread(image_path)
        plt.imshow(img)
        plt.scatter(cell_locations[:, 1], cell_locations[:, 0], c='r', marker='o', s=30)
        plt.title(f"Cell Locations (Part 2) - {image_path}")
        plt.xticks([]), plt.yticks([])  # Hide ticks
        plt.show()

        # Part 3: Find Cell Boundaries
        segmentation_map = FindCellBoundaries(image_path, foreground_mask, cell_locations)

        # Metrics for Part 3: Dice and IoU
        TP_3 = 0
        FP_3 = 0
        FN_3 = 0
        for row in range(segmentation_map.shape[0]):
            for index in range(segmentation_map.shape[1]):
                if segmentation_map[row][index] == 1 and data_mask[row][index] == 1:
                    TP_3 += 1
                if segmentation_map[row][index] == 1 and data_mask[row][index] == 0:
                    FP_3 += 1
                if segmentation_map[row][index] == 0 and data_mask[row][index] == 1:
                    FN_3 += 1

        Dice = 2 * TP_3 / (2 * TP_3 + FP_3 + FN_3)
        IoU = TP_3 / (TP_3 + FP_3 + FN_3)

        print(f'Part 3 Metrics for {image_path}:')
        print(f"Dice: {Dice:.2f}")
        print(f"IoU: {IoU:.2f}\n")

        # Plot Part 3: Segmentation Map 
        plt.imshow(segmentation_map, cmap='viridis')  
        plt.title(f"Segmentation Map (Part 3) - {image_path}")
        plt.xticks([]), plt.yticks([])
        plt.show()


image_paths = ['im1.jpg', 'im2.jpg', 'im3.jpg']
gold_mask_paths = ['im1_gold_mask.txt', 'im2_gold_mask.txt', 'im3_gold_mask.txt']
gold_cells_paths = ['im1_gold_cells.txt', 'im2_gold_cells.txt', 'im3_gold_cells.txt']


process_images(image_paths, gold_mask_paths, gold_cells_paths)



