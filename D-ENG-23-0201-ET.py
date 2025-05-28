import cv2
import numpy as np

# ------------------------ I. Define Functions ------------------------

def convert_to_grayscale(image):
    """Convert image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def estimate_initial_threshold(gray_image):
    """Estimate the initial threshold as the mean of grayscale pixel values."""
    return np.mean(gray_image)

def refine_iterative_threshold(gray_image):
    """Iteratively compute the optimal threshold using inter-means algorithm."""
    threshold = estimate_initial_threshold(gray_image)
    prev_threshold = -1  # Dummy initial value
    
    while abs(threshold - prev_threshold) > 1e-5:  # Stop when change is very small
        prev_threshold = threshold

        # Separate pixels into two groups
        lower_pixels = gray_image[gray_image <= threshold]
        upper_pixels = gray_image[gray_image > threshold]

        # Compute new means
        mean_low = np.mean(lower_pixels) if len(lower_pixels) > 0 else 0
        mean_high = np.mean(upper_pixels) if len(upper_pixels) > 0 else 0

        # Update threshold
        threshold = (mean_low + mean_high) / 2

    return threshold

def create_binary_image(gray_image, threshold):
    """Generate a binary image using the computed threshold."""
    _, binary_image = cv2.threshold(gray_image, int(threshold), 255, cv2.THRESH_BINARY)
    return binary_image

def add_labels_below(images, labels):
    """Add corresponding labels below each image."""
    height, width, _ = images[0].shape
    text_img = np.ones((50, width * len(images), 3), dtype=np.uint8) * 255  # White background for text
    
    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_color = (0, 0, 0)  # Black text

    # Add text below each section
    for i, label in enumerate(labels):
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = (width * (i + 0.5)) - (text_size[0] // 2)
        text_y = 30  # Center of the text area

        cv2.putText(text_img, label, (int(text_x), text_y), font, font_scale, text_color, font_thickness)

    return cv2.vconcat([cv2.hconcat(images), text_img])

# ------------------------ II. Process Multiple Images ------------------------

image_paths = [
    r"C:\Users\shine\Desktop\practical 5 DENG23-2021-ET\input 1.jpeg",  # First image
    r"C:\Users\shine\Desktop\practical 5 DENG23-2021-ET\input 2.jpeg"   # Second image
]

for idx, img_path in enumerate(image_paths, start=1):
    # Read image
    image = cv2.imread(img_path)
    
    if image is None:
        print(f"Error loading image: {img_path}")
        continue

    # Convert to grayscale
    gray = convert_to_grayscale(image)

    # Compute optimal threshold
    threshold = refine_iterative_threshold(gray)
    print(f"Image {idx} Threshold: {threshold}")

    # Generate binary (thresholded) image
    binary_image = create_binary_image(gray, threshold)

    # Convert grayscale and binary images to 3 channels for concatenation
    gray_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    binary_colored = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    # Concatenate Original, Grayscale, and Thresholded Images with Labels
    labels = ["Original Image", "Grayscale Image", "Binary Image"]
    final_image = add_labels_below([image, gray_colored, binary_colored], labels)

    # Save the combined image
    combined_save_path = f"C:/Users/shine/Desktop/practical 5 DENG23-2021-ET/combined_{idx}.jpg"
    cv2.imwrite(combined_save_path, final_image)
    print(f"Saved combined image: {combined_save_path}")

    # Save the thresholded binary image separately
    binary_save_path = f"C:/Users/shine/Desktop/practical 5 DENG23-2021-ET/binary_{idx}.jpg"
    cv2.imwrite(binary_save_path, binary_image)
    print(f"Saved binary image: {binary_save_path}")

    # Display both images
    cv2.imshow(f"Combined Image {idx}", final_image)
    cv2.imshow(f"Binary Image {idx}", binary_image)

# ------------------------ III. Print Completion Message ------------------------
print("Thresholding and segmentation operations are complete.")

cv2.waitKey(0)
cv2.destroyAllWindows()
