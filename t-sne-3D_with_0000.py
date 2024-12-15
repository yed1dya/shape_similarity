import os
import numpy as np
from sklearn.manifold import TSNE
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import torch
import plotly.graph_objects as go
import base64
from datetime import datetime

# Path to the main folder containing shape folders
main_folder_path = 'C:/Users/user/Documents/Sample Shapes/shapes'

# Output folder to save t-SNE plots
output_folder_name = "3_dimensions_with_0000"
output_folder_path = os.path.join('C:/Users/user/Documents/Sample Shapes/each shape', output_folder_name)

# Create the output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

# Path for log files
datetime_str = datetime.utcnow().strftime('%y-%m-%d-%H-%M')
print_log_path = f"C:/Users/user/Documents/Sample Shapes/logs/t-sne-3D_print_log_{datetime_str}.txt"
full_log_path = f"C:/Users/user/Documents/Sample Shapes/logs/t-sne-3D_full_log_{datetime_str}.txt"

# Open log files
print_log = open(print_log_path, 'w')
full_log = open(full_log_path, 'w')


def log(message):
    print(message)
    print_log.write(message + "\n")
    full_log.write(message + "\n")


log(f"Process started at: {datetime.utcnow().isoformat()} UTC")


def load_and_process_images_with_resnet(folder_path):
    """
    Load images and extract features using a pre-trained ResNet model.
    """
    resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
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
            full_log.write(f"Processed image: {file_path}\n")
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
    Apply t-SNE for dimensionality reduction to 3 dimensions.
    """
    tsne = TSNE(n_components=3, random_state=42)
    reduced_data = tsne.fit_transform(data)
    return reduced_data


def save_interactive_plot(reduced_data, labels, images, output_file_path, shape_name):
    """
    Save an interactive 3D plot with t-SNE results using Plotly with base64 encoded images.
    """
    # Adjust all points relative to the "ideal" point (0, 0, 0)
    reduced_data = reduced_data - reduced_data[labels.index("0000")]

    # Convert all image paths to base64 strings
    images_base64 = [convert_image_to_base64(image) for image in images]

    # Create a Plotly scatter 3D plot
    fig = go.Figure(data=[go.Scatter3d(
        x=reduced_data[:, 0],
        y=reduced_data[:, 1],
        z=reduced_data[:, 2],
        mode='markers',
        marker=dict(size=4, opacity=0.8),  # Smaller marker size
        text=[f"Label: {label}, Shape number: {index + 1}" for index, label in enumerate(labels)],
        hoverinfo='text'
    )])

    fig.update_layout(
        title=f't-SNE Visualization for {shape_name}',
        scene=dict(
            xaxis_title='t-SNE Dimension 1',
            yaxis_title='t-SNE Dimension 2',
            zaxis_title='t-SNE Dimension 3'
        )
    )

    # Save the plot to an HTML file
    fig.write_html(output_file_path)
    log(f"Saved interactive t-SNE plot to {output_file_path}")


def process_folders_for_tsne(main_folder_path, output_folder_path):
    """
    Process each shape folder, apply t-SNE, and save the plots.
    """
    for shape_folder in os.listdir(main_folder_path):
        shape_folder_path = os.path.join(main_folder_path, shape_folder)

        # Check if it's a valid shape folder
        if os.path.isdir(shape_folder_path) and shape_folder.startswith("shape"):
            log(f"Processing {shape_folder}...")

            data, labels, image_paths = load_and_process_images_with_resnet(shape_folder_path)

            if data.shape[0] == 0:
                log(f"No images found in {shape_folder}. Skipping.")
                continue

            log(f"Loaded and processed {data.shape[0]} images for {shape_folder}.")

            reduced_data = apply_tsne(data)

            # Save the interactive plot
            output_file_name = f"{shape_folder}_tsne_plot.html"
            output_file_path = os.path.join(output_folder_path, output_file_name)
            save_interactive_plot(reduced_data, labels, image_paths, output_file_path, shape_folder)


# Main function
def main():
    # Process each shape folder and generate t-SNE plots
    process_folders_for_tsne(main_folder_path, output_folder_path)
    log(f"Process completed at: {datetime.utcnow().isoformat()} UTC")
    print_log.close()
    full_log.close()


# Run the main function
main()
