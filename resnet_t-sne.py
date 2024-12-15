import os
import numpy as np
from sklearn.manifold import TSNE
from torchvision import models, transforms
from PIL import Image
import torch
from bokeh.plotting import figure, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.io import output_file
import base64
from datetime import datetime

# Path to the main folder containing shape folders
main_folder_path = 'C:/Users/user/Documents/Sample Shapes/shapes'

# Output folder to save t-SNE plots
output_folder_name = "ResNet + t-SNE for each shape"
output_folder_path = os.path.join('C:/Users/user/Documents/Sample Shapes', output_folder_name)

# Logs folder
logs_folder_path = 'C:/Users/user/Documents/Sample Shapes/logs'
os.makedirs(logs_folder_path, exist_ok=True)


def create_log_files():
    """
    Create log files with timestamps.
    """
    timestamp = datetime.utcnow().strftime('%y-%m-%d-%H-%M')
    print_log_path = os.path.join(logs_folder_path, f'resnet_t-sne_print_log_{timestamp}.txt')
    full_log_path = os.path.join(logs_folder_path, f'resnet_t-sne_full_log_{timestamp}.txt')
    return print_log_path, full_log_path


def log_message(log_file, message):
    """
    Write a message to the log file and print it.
    """
    with open(log_file, 'a') as f:
        f.write(f"{message}\n")
    print(message)


def load_and_process_images_with_resnet(folder_path, full_log):
    """
    Load images and extract features using a pre-trained ResNet model.
    """
    resnet = models.resnet18(pretrained=True)
    resnet.fc = torch.nn.Identity()  # Remove the classification layer
    resnet.eval()  # Set to evaluation mode

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalize with ImageNet mean and std
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data = []
    labels = []
    image_paths = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):
            file_path = os.path.join(folder_path, file_name)
            image = Image.open(file_path).convert('RGB')
            image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                features = resnet(image_tensor).numpy().flatten()
            data.append(features)
            labels.append(file_name.split(".")[0])  # Use filename without extension as label
            image_paths.append(file_path)
            log_message(full_log, f"Processed image: {file_path}")

    return np.array(data), labels, image_paths


def convert_image_to_base64(image_path):
    """
    Convert an image to a base64 string.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded_string}"


def apply_tsne(data):
    """
    Apply t-SNE for dimensionality reduction.
    """
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(data)
    return reduced_data


def save_interactive_plot(reduced_data, labels, images, output_file_path, shape_name, full_log):
    """
    Save an interactive plot with t-SNE results using Bokeh with base64 encoded images.
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
        title=f"t-SNE Visualization for {shape_name}",
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

    p.xaxis.axis_label = "t-SNE Dimension 1"
    p.yaxis.axis_label = "t-SNE Dimension 2"

    # Output to static HTML file
    output_file(output_file_path)

    # Save the figure
    save(p)
    log_message(full_log, f"Saved interactive t-SNE plot to {output_file_path}")


def process_folders_for_tsne(main_folder_path, output_folder_path, print_log, full_log):
    """
    Process each shape folder, apply t-SNE, and save the plots.
    """
    log_message(print_log, f"Process started at {datetime.utcnow().isoformat()} GMT")
    log_message(full_log, f"Process started at {datetime.utcnow().isoformat()} GMT")

    for shape_folder in os.listdir(main_folder_path):
        shape_folder_path = os.path.join(main_folder_path, shape_folder)

        # Check if it's a valid shape folder
        if os.path.isdir(shape_folder_path) and shape_folder.startswith("shape"):
            log_message(print_log, f"Processing {shape_folder}...")
            log_message(full_log, f"Processing folder: {shape_folder_path}")

            data, labels, image_paths = load_and_process_images_with_resnet(shape_folder_path, full_log)

            if data.shape[0] == 0:
                log_message(print_log, f"No images found in {shape_folder}. Skipping.")
                log_message(full_log, f"No images found in folder: {shape_folder_path}")
                continue

            log_message(print_log, f"Loaded and processed {data.shape[0]} images for {shape_folder}.")
            log_message(full_log, f"Loaded and processed {data.shape[0]} images for folder: {shape_folder_path}")

            reduced_data = apply_tsne(data)

            # Save the interactive plot
            output_file_name = f"{shape_folder}_tsne_plot.html"
            output_file_path = os.path.join(output_folder_path, output_file_name)
            save_interactive_plot(reduced_data, labels, image_paths, output_file_path, shape_folder, full_log)

    log_message(print_log, f"Process ended at {datetime.utcnow().isoformat()} GMT")
    log_message(full_log, f"Process ended at {datetime.utcnow().isoformat()} GMT")


# Main function
def main():
    print_log, full_log = create_log_files()
    # Create output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)
    process_folders_for_tsne(main_folder_path, output_folder_path, print_log, full_log)


# Run the main function
main()
