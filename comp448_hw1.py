import time
import cv2
import numpy as np
from skimage import filters, morphology, color
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import morphology



# Functions for Part1

def obtain_foreground_mask(image_path):

    print("Reading and preprocessing image...")
    # 1. Read the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Denoise with Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    print("Thresholding...")
    # 3. Threshold using Otsu’s method
    _, binary_mask = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print("Morphological operations...")
    # 4. Post-process with Morphological operations
    cleaned = morphology.remove_small_objects(binary_mask.astype(bool), min_size=60)
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=100)

    print("Step 4: Done.")
    # Convert boolean mask to 0 and 1
    final_mask = cleaned.astype(np.uint8)

    return image_rgb, final_mask



def visualize_result(image, mask, title="Mask"):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title(title)
    plt.show()

# Example usage
# img_path = 'im1.jpg'  # Replace with actual path
# image, mask = obtain_foreground_mask(img_path)
# visualize_result(image, mask)



def evaluate_mask(predicted_mask, gold_mask_path):
    gold_mask = np.loadtxt(gold_mask_path).astype(np.uint8)

    tp = np.sum((predicted_mask == 1) & (gold_mask == 1))
    fp = np.sum((predicted_mask == 1) & (gold_mask == 0))
    fn = np.sum((predicted_mask == 0) & (gold_mask == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision, recall, f_score

# gold_path = 'im1_gold_mask.txt'
# precision, recall, fscore = evaluate_mask(mask, gold_path)
# print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F-score: {fscore:.3f}")


def test_of_part1():
    image_files = ['im1.jpg', 'im2.jpg', 'im3.jpg']
    gold_mask_files = ['im1_gold_mask.txt', 'im2_gold_mask.txt', 'im3_gold_mask.txt']

    results = []

    print("Starting foreground mask evaluation...\n")

    for img_path, gold_path in zip(image_files, gold_mask_files):
        print(f"➡️  Processing {img_path}...")

        start_time = time.time()

        image, predicted_mask = obtain_foreground_mask(img_path)
        precision, recall, fscore = evaluate_mask(predicted_mask, gold_path)

        elapsed_time = time.time() - start_time
        print(f"✅ Done with {img_path} in {elapsed_time:.2f} seconds.")
        print(f"    Precision: {precision:.3f}, Recall: {recall:.3f}, F-score: {fscore:.3f}\n")

        # Optional: Show the result visually (comment out if running from terminal only)
        # visualize_result(image, predicted_mask, title=f"Predicted Mask for {img_path}")

        results.append({
            'image': img_path,
            'precision': precision,
            'recall': recall,
            'f-score': fscore,
            'time': elapsed_time
        })

    print("📊 Final Evaluation Table:")
    print("Image\t\tPrecision\tRecall\t\tF-score\t\tTime (s)")
    for r in results:
        print(f"{r['image']}\t{r['precision']:.3f}\t\t{r['recall']:.3f}\t\t{r['f-score']:.3f}\t\t{r['time']:.2f}")

# test_of_part1()

####################################################################################

def find_cell_locations(enhanced_img, foreground_mask, gold_cells, thresh_value=195, min_distance=14):
    """
    Using the enhanced image and foreground mask, identifies approximate cell centers.
    The method thresholds the enhanced image to detect bright (white) boundaries, erodes them,
    removes these areas from the foreground mask, computes a distance transform, and then finds
    regional maxima as candidate cell centers.
    
    Parameters:
        enhanced_img (ndarray): The enhanced image (can be 3-channel or grayscale; dimensions from the first two are used).
        foreground_mask (ndarray): Binary mask (0/1) from Part 1.
        gold_cells (ndarray): Gold standard cell labels (from im*_gold_cells.txt).
        thresh_value (int): Threshold for identifying white boundaries.
        min_distance (int): Minimum number of pixels between detected peaks.
    
    Returns:
        dist (ndarray): Normalized distance transform.
        coordinates (ndarray): Array of candidate cell centers (row, col format).
        detected_labels (ndarray): Array of gold cell labels extracted at candidate locations.
    """
    # Use only the first two dimensions (rows, cols)
    rows, cols = enhanced_img.shape[:2]
    
    # Create a copy of the enhanced image to threshold white boundaries.
    boundary_img = enhanced_img.copy().astype(np.uint8)
    # If the image is colored, convert it to grayscale
    if boundary_img.ndim == 3:
        boundary_img = cv2.cvtColor(boundary_img, cv2.COLOR_RGB2GRAY)
        
    # Binary threshold for boundaries.
    for r in range(rows):
        for c in range(cols):
            if boundary_img[r, c] > thresh_value:
                boundary_img[r, c] = 255
            else:
                boundary_img[r, c] = 0

    # Erode to refine the boundary areas.
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(boundary_img, kernel)
    
    # Create a new mask from the foreground mask and remove eroded boundary pixels.
    import copy  # Ensure copy is imported if not already
    new_mask = copy.deepcopy(foreground_mask)
    for r in range(rows):
        for c in range(cols):
            if erosion[r, c] == 255:
                new_mask[r, c] = 0

    # Compute the Euclidean distance transform.
    dist = cv2.distanceTransform(new_mask, cv2.DIST_L2, 3)
    # Normalize to the range [0, 1].
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    
    # Find regional maxima.
    coordinates = peak_local_max(dist, min_distance=min_distance, threshold_abs=0.0)
    
    # Extract gold cell labels at the detected coordinates.
    detected_labels = []
    for pt in coordinates:
        # Note: pt is in (row, col) order.
        label = gold_cells[pt[0], pt[1]]
        detected_labels.append(label)
    detected_labels = np.array(detected_labels)
    
    return dist, coordinates, detected_labels

def evaluate_cell_locations(detected_labels, gold_cells):
    """
    Evaluates the detected cell centers.
    
    A gold cell is counted as detected (true positive, TP) if it is matched by exactly one detection.
    Detections in the background (label == 0) or duplicate detections in one cell count as false positives.
    
    Parameters:
        detected_labels (ndarray): Array of gold cell labels from detected cell centers.
        gold_cells (ndarray): Gold standard cell labels (2D array).
    
    Returns:
        precision, recall, fscore (floats)
    """
    # Ignore detections that are 0.
    valid = detected_labels[detected_labels != 0]
    if valid.size == 0:
        return 0, 0, 0

    unique_labels, counts = np.unique(valid, return_counts=True)
    # Count true positives as cells with exactly one detection.
    TP = np.sum(counts == 1)
    FP = valid.size - TP
    # Total gold cells is taken as the maximum label (assuming labels start at 1).
    total_gold = int(np.max(gold_cells))
    FN = total_gold - TP

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (total_gold + 1e-8)
    fscore = (2 * precision * recall) / (precision + recall + 1e-8)
    
    return precision, recall, fscore

def visualize_cell_locations(dist, coordinates, title="Distance Transform with Cell Centers"):
    """
    Overlays detected cell centers on the distance transform.
    
    Parameters:
        dist (ndarray): The normalized distance transform.
        coordinates (ndarray): Array of detected cell center coordinates (row, col).
    """
    plt.figure(figsize=(8,6))
    plt.imshow(dist, cmap='gray')
    if coordinates.size:
        # Plot coordinates: note that we swap (col, row) for (x, y) plotting.
        plt.plot(coordinates[:, 1], coordinates[:, 0], 'ro', markersize=4)
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show()

def test_of_part2():
    """
    Test method for Part 2: Cell Location Detection.
    Processes images im1 and im2 by:
      - Loading the gold cell labels.
      - Obtaining the enhanced image and foreground mask using obtain_foreground_mask.
      - Computing the modified distance transform and detecting cell centers.
      - Evaluating the detected centers against the gold standard.
      - Printing the precision, recall, F-score, and execution time for each image.
    """
    import time, os
    # Define file names for the images and the gold cell labels.
    image_files = ['im1.jpg', 'im2.jpg', 'im3.jpg']
    gold_cells_files = ['im1_gold_cells.txt', 'im2_gold_cells.txt', 'im3_gold_cells.txt']
    
    results = []
    print("Starting cell location detection evaluation...\n")
    
    for img_path, gold_path in zip(image_files, gold_cells_files):
        # Check if the files exist.
        if not os.path.exists(img_path):
            print(f"Error: The image file '{img_path}' does not exist. Please check your file path.")
            continue
        if not os.path.exists(gold_path):
            print(f"Error: The gold cell labels file '{gold_path}' does not exist. Please check your file path.")
            continue
        
        print(f"➡️  Processing {img_path} for cell location detection...")
        start_time = time.time()
        
        # Load the gold cell labels from the text file.
        gold_cells = np.loadtxt(gold_path).astype(np.int32)
        
        # Obtain enhanced image and foreground mask from Part 1.
        enhanced_img, pred_mask = obtain_foreground_mask(img_path)
        
        # Run cell location detection using the enhanced image, foreground mask, and gold cell labels.
        # Adjust the threshold value and min_distance as needed.
        dist, coordinates, detected_labels = find_cell_locations(
            enhanced_img, pred_mask, gold_cells, thresh_value=195, min_distance=14
        )
        
        # Evaluate detected cell centers.
        precision, recall, fscore = evaluate_cell_locations(detected_labels, gold_cells)
        elapsed_time = time.time() - start_time
        
        print(f"✅ Done with {img_path} in {elapsed_time:.2f} seconds.")
        print(f"    Precision: {precision:.3f}, Recall: {recall:.3f}, F-score: {fscore:.3f}\n")
        
        # Optional: Visualize the distance transform with the candidate cell centers.
        visualize_cell_locations(dist, coordinates, 
                                 title=f"Distance Transform with Detected Cell Centers ({img_path})")
        
        results.append({
            'image': img_path,
            'precision': precision,
            'recall': recall,
            'f-score': fscore,
            'time': elapsed_time
        })
    
    # Print final evaluation table.
    print("📊 Final Evaluation Metrics for Cell Location Detection:")
    print("Image\t\tPrecision\tRecall\t\tF-score\t\tTime (s)")
    for r in results:
        print(f"{r['image']}\t{r['precision']:.3f}\t\t{r['recall']:.3f}\t\t{r['f-score']:.3f}\t\t{r['time']:.2f}")

# Call the test method for Part 2.
test_of_part2()
