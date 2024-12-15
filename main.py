import os
import pathlib
import stat
import time

# Start runtime counter
start_time = time.time()

# Define the directory containing the files
directory = pathlib.Path(r"C:\Users\user\Downloads\All_The_FIles")

problematic_count = 0

# Traverse all subdirectories and files within the given directory
for root, dirs, files in os.walk(directory):
    for filename in files:
        # Ignore CSV files with "Test Time Log"
        if filename.endswith(".csv") and "Test Time Log" in filename:
            continue
        # Remove file extension and check if the filename is not a number, ignoring "._" prefix
        name_without_extension = os.path.splitext(filename)[0]
        if name_without_extension.startswith("._"):
            continue
        if not name_without_extension.isdigit():
            print(f"Folder: {root}, Problematic file: {filename}")
            problematic_count += 1

hidden_count = 0
starts_with_dot_count = 0
for root, dirs, files in os.walk(directory):
    for filename in files:
        file_path = os.path.join(root, filename)
        try:
            # Check if file is hidden on Windows
            attributes = os.stat(file_path).st_file_attributes
            if attributes & stat.FILE_ATTRIBUTE_HIDDEN:
                hidden_count += 1
            # Check if the file starts with a "."
            if filename.startswith("."):
                starts_with_dot_count += 1
        except Exception as e:
            print(f"Error checking file: {file_path}, {e}")

# Calculate runtime
end_time = time.time()
runtime = end_time - start_time

print("Search complete.")
print(f"Total problematic files found: {problematic_count}")
print(f"Total hidden files found: {hidden_count}")
print(f"Total files starting with '.': {starts_with_dot_count}")
print(f"Runtime: {runtime:.2f} seconds")
