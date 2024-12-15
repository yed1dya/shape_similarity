import cv2
import numpy as np
import os
import shutil
from datetime import datetime

count1 = 0
count2 = 0


def initialize_logs():
    global print_log_file, full_log_file
    timestamp = datetime.utcnow().strftime("%y-%m-%d-%H-%M")
    log_dir = "C:/Users/user/Documents/Sample Shapes/logs"
    os.makedirs(log_dir, exist_ok=True)
    print_log_file = os.path.join(log_dir, f"Image_Processing_Pipeline_print_log_{timestamp}")
    full_log_file = os.path.join(log_dir, f"Image_Processing_Pipeline_full_log_{timestamp}")
    with open(print_log_file, 'w') as init_f:
        init_f.write("Start Time: " + datetime.utcnow().isoformat() + "\n")
    with open(full_log_file, 'w') as init_f:
        init_f.write("Start Time: " + datetime.utcnow().isoformat() + "\n")


def log_message(message, full_log=False):
    print(message)
    with open(print_log_file, 'a') as log_f:
        log_f.write(message + "\n")
    if full_log:
        with open(full_log_file, 'a') as log_f:
            log_f.write(message + "\n")


initialize_logs()


# Function to process the image, crop the drawing, handle small drawings, and center it on a larger white canvas

def crop_and_center(input_path, output_path):
    log_message(f"Processing image: {input_path}", full_log=True)
    # Read the image
    image = cv2.imread(input_path)

    # Crop out the gray header (top 20% of the image)
    height, width, _ = image.shape
    cropped_image = image[int(height * 0.2):, :]

    # Convert the image to the HSV color space to detect blue
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    # Define the range for blue color in HSV
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours of the blue drawing
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the bounding box around the blue drawing
        x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        # Crop the drawing
        cropped_drawing = cropped_image[y_min:y_max, x_min:x_max]

        # Create a larger white canvas
        canvas_size = 1000  # Adjust canvas size for more white space
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

        # Get the dimensions of the cropped drawing
        drawing_height, drawing_width = cropped_drawing.shape[:2]

        # Avoid excessive zoom-in by setting a minimum scale factor
        if drawing_width > 0 and drawing_height > 0:
            scale = min(canvas_size / (drawing_width * 2.5), canvas_size / (drawing_height * 2.5))
            scale = min(scale, 1.0)  # Limit the maximum zoom-in to 1.0 (no zoom)
            resized_drawing = cv2.resize(cropped_drawing, (int(drawing_width * scale), int(drawing_height * scale)))

            # Get coordinates to center the drawing on the canvas
            y_offset = (canvas_size - resized_drawing.shape[0]) // 2
            x_offset = (canvas_size - resized_drawing.shape[1]) // 2

            # Paste the resized drawing onto the canvas
            canvas[y_offset:y_offset + resized_drawing.shape[0], x_offset:x_offset + resized_drawing.shape[1]] = (
                resized_drawing)

            # Save the final centered drawing
            cv2.imwrite(output_path, canvas)
            log_message(f"Processed image saved at {output_path}", full_log=True)
        else:
            log_message("Drawing dimensions are invalid, skipping processing.", full_log=True)
    else:
        log_message(f"No blue drawing detected in {input_path}.", full_log=True)


# Function to process all images in the described folder structure

def process_folders(process_folders_main_folder_path):
    global count1
    for process_folders_child_folder in os.listdir(process_folders_main_folder_path):
        process_folders_child_folder_path = os.path.join(process_folders_main_folder_path, process_folders_child_folder)

        if os.path.isdir(process_folders_child_folder_path):
            log_message(f"Processing folder: {process_folders_child_folder_path}")
            simple_test_folder = os.path.join(process_folders_child_folder_path, "SimpleTest")

            if os.path.exists(simple_test_folder):
                extracted_images_folder = os.path.join(process_folders_child_folder_path, "Extracted images")
                os.makedirs(extracted_images_folder, exist_ok=True)
                log_message(f"Created folder: {extracted_images_folder}")

                for file_name in os.listdir(simple_test_folder):
                    if file_name.endswith(".png"):
                        input_image_path = os.path.join(simple_test_folder, file_name)
                        output_image_path = os.path.join(extracted_images_folder, file_name)

                        crop_and_center(input_image_path, output_image_path)
                        count1 += 1
                        log_message(f"Processed {file_name} and saved to {output_image_path}. "
                                    f"Total processed: {count1}", full_log=True)
            else:
                log_message(f"SimpleTest folder not found in {process_folders_child_folder_path}", full_log=True)
        else:
            log_message(f"{process_folders_child_folder_path} is not a folder.", full_log=True)


# Function to preprocess the image to a high-resolution MNIST-like format

def preprocess_image(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = 255 - gray
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((200, 200), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    padding = int(200 * 0.1)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(binary.shape[1] - x, w + 2 * padding)
    h = min(binary.shape[0] - y, h + 2 * padding)
    cropped = binary[y:y + h, x:x + w]
    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    start_x = (size - w) // 2
    start_y = (size - h) // 2
    square[start_y:start_y + h, start_x:start_x + w] = cropped
    resized = cv2.resize(square, (200, 200), interpolation=cv2.INTER_LANCZOS4)
    return resized


def process_and_create_mnist_images_with_same_names(process_and_create_main_folder_path):
    for process_and_create_child_folder in os.listdir(process_and_create_main_folder_path):
        process_and_create_child_folder_path = os.path.join(process_and_create_main_folder_path,
                                                            process_and_create_child_folder)

        if os.path.isdir(process_and_create_child_folder_path):
            extracted_images_folder = os.path.join(process_and_create_child_folder_path, "Extracted images")

            if os.path.exists(extracted_images_folder):
                log_message(f"Processing Extracted images folder: {extracted_images_folder}")
                for file_name in os.listdir(extracted_images_folder):
                    if file_name.endswith(".png") and "mnist" not in file_name:
                        input_image_path = os.path.join(extracted_images_folder, file_name)
                        base_name = file_name.split(".")[0]
                        output_image_path = os.path.join(
                            extracted_images_folder, f"{base_name}mnist.png"
                        )

                        image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
                        mnist_image = preprocess_image(image)
                        if mnist_image is not None:
                            cv2.imwrite(output_image_path, mnist_image)
                            log_message(f"Created MNIST image: {output_image_path}", full_log=True)
            else:
                log_message(f"Extracted images folder not found in "
                            f"{process_and_create_child_folder_path}", full_log=True)
        else:
            log_message(f"{process_and_create_child_folder_path} is not a folder. Skipping.", full_log=True)


# Function to collect MNIST images into the correct shape folders

def collect_mnist_images_to_shape_folders(collect_main_folder_path, collect_shapes_folder_path):
    global count2
    for collect_child_folder in os.listdir(collect_main_folder_path):
        collect_child_folder_path = os.path.join(collect_main_folder_path, collect_child_folder)

        if os.path.isdir(collect_child_folder_path):
            extracted_images_folder = os.path.join(collect_child_folder_path, "Extracted images")

            if os.path.exists(extracted_images_folder):
                log_message(f"Collecting MNIST images from folder: {extracted_images_folder}")
                for file_name in os.listdir(extracted_images_folder):
                    if file_name.endswith("mnist.png"):
                        base_name = file_name.split("mnist.png")[0]
                        shape_number = base_name.strip()
                        shape_folder_name = f"shape{shape_number}"
                        shape_folder = os.path.join(collect_shapes_folder_path, shape_folder_name)
                        os.makedirs(shape_folder, exist_ok=True)
                        log_message(f"Created folder: {shape_folder}")
                        source_image_path = os.path.join(extracted_images_folder, file_name)
                        destination_image_path = os.path.join(shape_folder, f"{collect_child_folder}_{file_name}")
                        shutil.copy2(source_image_path, destination_image_path)
                        count2 += 1
                        log_message(f"Copied {source_image_path} to {destination_image_path}. "
                                    f"Total copied: {count2}", full_log=True)
            else:
                log_message(f"Extracted images folder not found in {collect_child_folder_path}", full_log=True)
        else:
            log_message(f"{collect_child_folder_path} is not a folder. Skipping.", full_log=True)


# Define the main folder path on PC

main_folder_path = 'C:/Users/user/Documents/Sample Shapes/newshapes/All_The_Files'

# Run the processing functions
process_folders(main_folder_path)
process_and_create_mnist_images_with_same_names(main_folder_path)

# Define the shapes folder path

shapes_folder_path = 'C:/Users/user/Documents/Sample Shapes/shapes'

# Run the function to collect mnist images to shape folders
collect_mnist_images_to_shape_folders(main_folder_path, shapes_folder_path)

# Finally, move the processed child folders to the 'children' folder,
# and ensure they are erased from the original location

source_folder_path = 'C:/Users/user/Documents/Sample Shapes/newshapes/All_The_Files'
destination_folder_path = 'C:/Users/user/Documents/Sample Shapes/children'
count3 = 0
for child_folder in os.listdir(source_folder_path):
    child_folder_path = os.path.join(source_folder_path, child_folder)
    destination_path = os.path.join(destination_folder_path, child_folder)
    if os.path.isdir(child_folder_path):
        shutil.move(child_folder_path, destination_path)
        count3 += 1
        log_message(f"Moved {child_folder} to {destination_folder_path}. Total moved: {count3}")

# Log end time

with open(print_log_file, 'a') as f:
    f.write("End Time: " + datetime.utcnow().isoformat() + "\n")
with open(full_log_file, 'a') as f:
    f.write("End Time: " + datetime.utcnow().isoformat() + "\n")
