from sklearn.decomposition import PCA
import os
import cv2
import numpy as np
from datetime import datetime
from bokeh.plotting import figure, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.io import output_file
import base64

# Path to the main folder containing shape folders
main_folder_path = 'C:/Users/user/Documents/Sample Shapes/shapes'
output_folder_path = 'C:/Users/user/Documents/Sample Shapes/each shape'  # Folder to save PCA plots
log_folder_path = os.path.join(main_folder_path, "logs")
os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(log_folder_path, exist_ok=True)


def get_log_file_path(prefix):
    current_time = datetime.utcnow().strftime('%y-%m-%d-%H-%M')
    return os.path.join(log_folder_path, f"{prefix}_{current_time}.txt")


print_log_path = get_log_file_path("pca_print_log")
full_log_path = get_log_file_path("pca_full_log")


def log_message(message, *log_file_paths):
    for log_file_path in log_file_paths:
        with open(log_file_path, 'a') as log_file:
            log_file.write(message + "\n")
    print(message)


def load_images_from_folder(folder_path):
    """
    Load all images from a given folder as flattened grayscale arrays.
    """
    data = []
    labels = []
    image_paths = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):  # Process only PNG files
            file_path = os.path.join(folder_path, file_name)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                resized_image = cv2.resize(image, (32, 32))  # Downscale to 32x32 for faster computation
                data.append(resized_image.flatten())  # Flatten the image
                labels.append(file_name.split(".")[0])  # Use filename without extension as label
                image_paths.append(file_path)
    return np.array(data), labels, image_paths


def apply_pca(data, n_components=2):
    """
    Apply PCA for dimensionality reduction.
    """
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data


def convert_image_to_base64(image_path):
    """
    Convert an image to a base64 string.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded_string}"


def save_interactive_plot(reduced_data, labels, images, output_file_path, shape_name):
    """
    Save an interactive plot with PCA results using Bokeh with base64 encoded images.
    """
    # Convert all image paths to base64 strings
    images_base64 = [convert_image_to_base64(image) for image in images]

    # Prepare the Bokeh data source
    source = ColumnDataSource(data=dict(
        x=reduced_data[:, 0],
        y=reduced_data[:, 1],
        label=labels,
        image=images_base64
    ))

    # Create a Bokeh figure with 'wheel_zoom' tool added
    p = figure(
        title=f"PCA Visualization for {shape_name}",
        tools="hover,pan,wheel_zoom,reset,save",
        active_scroll='wheel_zoom',
        tooltips="""
            <div>
                <strong>Label:</strong> @label <br>
                <img src="@image" height="64" style="float: left; margin: 0px 15px 15px 0px;"/>
            </div>
        """
    )

    # Add scatter plot
    p.scatter('x', 'y', size=10, source=source)

    p.xaxis.axis_label = "PCA Dimension 1"
    p.yaxis.axis_label = "PCA Dimension 2"

    # Output to static HTML file
    output_file(output_file_path)

    # Save the figure
    save(p)
    log_message(f"Saved interactive PCA plot to {output_file_path}", print_log_path, full_log_path)


def process_folders_for_pca(main_folder_path, output_folder_path):
    """
    Process each shape folder, apply PCA, and save the plots.
    """
    start_time = datetime.utcnow()
    log_message(f"Process started at {start_time}", print_log_path, full_log_path)

    for shape_folder in os.listdir(main_folder_path):
        shape_folder_path = os.path.join(main_folder_path, shape_folder)

        if os.path.isdir(shape_folder_path) and shape_folder.startswith("shape"):
            log_message(f"Processing {shape_folder}...", print_log_path, full_log_path)

            data, labels, image_paths = load_images_from_folder(shape_folder_path)

            if data.shape[0] == 0:
                log_message(f"No images found in {shape_folder}. Skipping.", print_log_path, full_log_path)
                continue

            log_message(f"Loaded {data.shape[0]} images for {shape_folder}.", print_log_path, full_log_path)

            reduced_data = apply_pca(data)

            output_file_name = f"{shape_folder}_pca_plot.html"
            output_file_path = os.path.join(output_folder_path, output_file_name)
            save_interactive_plot(reduced_data, labels, image_paths, output_file_path, shape_folder)

    end_time = datetime.utcnow()
    log_message(f"Process ended at {end_time}", print_log_path, full_log_path)


# Main function
def main():
    process_folders_for_pca(main_folder_path, output_folder_path)


# Run the main function
main()
