import time
import cv2
import numpy as np
from skimage import filters, morphology, color
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def obtain_foreground_mask(image_path):
    # 1. Read the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Denoise with Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Threshold using Otsu’s method (could try adaptive/local too)
    _, binary_mask = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Post-process with Morphological operations
    cleaned = morphology.remove_small_objects(binary_mask.astype(bool), min_size=60)
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=100)

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


def test_on_all_images():
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
        visualize_result(image, predicted_mask, title=f"Predicted Mask for {img_path}")

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

test_on_all_images()