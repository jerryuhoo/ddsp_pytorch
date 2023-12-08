import os
import random
import shutil
import csv
import re


def custom_sort_key(item):
    """
    Custom sort key function to sort items based on all numeric parts after each '_'.
    """
    name, _ = item[0].split(".", 1)  # Split on the dot to isolate the extension
    parts = re.split("_(\d+)", name)  # Split on each underscore followed by digits
    sort_key = [parts[0]] + [int(num) if num.isdigit() else num for num in parts[1:]]
    return sort_key


def select_and_copy_samples_for_mos(root_dir, output_dir, samples_per_model=20):
    """
    Selects the same random samples from each subdirectory in the root directory,
    copies them into an output directory with renamed files based on a unique random index
    for each sample in each subdirectory. Records the mapping of new file names
    to their original subdirectory and generates a CSV file for scoring.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all subdirectories in the root directory
    subdirs = [
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    ]

    # Determine common files across all subdirectories
    common_files = set(os.listdir(os.path.join(root_dir, subdirs[0])))
    for subdir in subdirs[1:]:
        subdir_path = os.path.join(root_dir, subdir)
        common_files = common_files.intersection(set(os.listdir(subdir_path)))

    # Select random samples from common files
    selected_samples = random.sample(
        common_files, min(samples_per_model, len(common_files))
    )

    # Prepare for CSV and mapping file creation
    csv_rows = []
    mapping_entries = []

    for sample in selected_samples:
        used_indices = set()  # Track used indices to ensure uniqueness
        for subdir in subdirs:
            # Generate a unique random index for each file in each subdirectory
            index = random.randint(1, len(subdirs) * samples_per_model)
            while index in used_indices:
                index = random.randint(1, len(subdirs) * samples_per_model)
            used_indices.add(index)

            original_file_path = os.path.join(root_dir, subdir, sample)
            new_name = (
                f"{os.path.splitext(sample)[0]}_{index}{os.path.splitext(sample)[1]}"
            )

            # Copy file to output directory
            shutil.copy(original_file_path, os.path.join(output_dir, new_name))

            # Prepare mapping entry and CSV row
            mapping_entries.append(f"{new_name} -> {subdir}")
            csv_rows.append([new_name, ""])  # Adding row for scoring sheet

    # Sort the entries and rows with custom sort key
    mapping_entries.sort(key=lambda x: custom_sort_key(x.split(" -> ")))
    csv_rows.sort(key=custom_sort_key)

    # Write to mapping file
    mapping_file_path = os.path.join(output_dir, "mapping.txt")
    with open(mapping_file_path, "w", encoding="utf-8") as file:
        file.write("\n".join(mapping_entries))

    # Write to CSV file
    csv_file_path = os.path.join(output_dir, "scoring_sheet.csv")
    with open(csv_file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["File Name", "Score"])  # Writing header
        writer.writerows(csv_rows)  # Writing data rows

    print("Samples selection, copying, and CSV creation completed.")


# Define the root directory and output directory
root_directory = (
    "/home/yifeng/SVS/ddsp_pytorch/outputs"  # Change this to your root directory path
)
output_directory = "/home/yifeng/SVS/ddsp_pytorch/mos"  # Change this to your desired output directory path

# Call the function
select_and_copy_samples_for_mos(root_directory, output_directory)
