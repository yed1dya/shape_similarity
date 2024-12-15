import os
import pathlib
from datetime import datetime

# Define the directory to process
base_directory = pathlib.Path(r"C:\Users\user\Documents\Sample Shapes\newshapes\All_The_Files")

# Create a log file
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = pathlib.Path(r"C:\Users\user\Documents\Sample Shapes\newshapes") / f"final_check_anomalies_{current_time}.log"

with open(log_file_path, "w") as log_file:
    # Counter
    counters = {"folders_checked": 0, "anomalies_found": 0}

    # Traverse all items in the base directory
    for folder in base_directory.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            counters["folders_checked"] += 1
            anomalies = []

            # Check for exactly one subfolder named "SimpleTest"
            subfolders = [subfolder for subfolder in folder.iterdir() if subfolder.is_dir()]
            if len(subfolders) != 1 or subfolders[0].name != "SimpleTest":
                anomalies.append("Folder does not contain exactly one subfolder named 'SimpleTest'.")

            # Check contents of "SimpleTest"
            simple_test_folder = subfolders[0] if subfolders and subfolders[0].name == "SimpleTest" else None
            if simple_test_folder:
                png_files = [file for file in simple_test_folder.iterdir() if file.is_file() and file.suffix.lower() == ".png"]

                # Check for missing or extra files
                expected_files = {f"{i}" for i in range(1, 22)}  # Require 1-21
                actual_files = {file.stem for file in png_files}

                missing_files = expected_files - actual_files
                extra_files = actual_files - expected_files

                # Allow up to 14 missing files
                if len(missing_files) > 14:
                    anomalies.append(f"Missing {len(missing_files)} PNG files: {', '.join(sorted(missing_files))}.")
                if extra_files:
                    anomalies.append(f"Extra files in 'SimpleTest': {', '.join(sorted(extra_files))}.")

            # If "SimpleTest" folder is missing, log the anomaly
            if not simple_test_folder:
                anomalies.append("Missing 'SimpleTest' folder.")

            # Log anomalies if found
            if anomalies:
                counters["anomalies_found"] += 1
                log_file.write(f"Anomalies in folder: {folder}\n")
                for anomaly in anomalies:
                    log_file.write(f"  - {anomaly}\n")
                print(f"Anomalies in folder: {folder}")
                for anomaly in anomalies:
                    print(f"  - {anomaly}")
        else:
            # Log if a folder is not numbered
            counters["anomalies_found"] += 1
            log_file.write(f"Anomalies in folder: {folder}\n")
            log_file.write("  - Folder name is not a number.\n")
            print(f"Anomalies in folder: {folder}")
            print("  - Folder name is not a number.")

    # Write summary to log
    log_file.write("\nProcess Summary:\n")
    log_file.write(f"Total numbered folders checked: {counters['folders_checked']}\n")
    log_file.write(f"Total anomalies found: {counters['anomalies_found']}\n")

# Print summary to console
print("\nProcess Summary:")
print(f"Total numbered folders checked: {counters['folders_checked']}")
print(f"Total anomalies found: {counters['anomalies_found']}")