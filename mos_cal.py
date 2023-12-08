import csv
import os


def calculate_average_scores(output_dir):
    """
    Calculate the average score for each model based on the scores from the CSV file
    and the mapping from the mapping.txt file.
    """
    # Read scores from the CSV file
    scores = {}  # {file_name: score}
    csv_file_path = os.path.join(output_dir, "scoring_sheet.csv")
    with open(csv_file_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip header
        for row in reader:
            file_name, score = row
            scores[file_name] = float(score)

    # Read mapping from mapping.txt
    mapping = {}  # {file_name: model}
    mapping_file_path = os.path.join(output_dir, "mapping.txt")
    with open(mapping_file_path, "r", encoding="utf-8") as file:
        for line in file:
            new_name, subdir = line.strip().split(" -> ")
            mapping[new_name] = subdir

    # Calculate average scores per model
    model_scores = {}  # {model: [scores]}
    for file_name, model in mapping.items():
        if file_name in scores:
            if model not in model_scores:
                model_scores[model] = []
            model_scores[model].append(scores[file_name])

    # Calculate and print average
    for model, scores in model_scores.items():
        average_score = sum(scores) / len(scores)
        print(f"Average score for model {model}: {average_score:.2f}")


# Define the output directory
output_directory = (
    "/home/yifeng/SVS/ddsp_pytorch/mos"  # Change this to your output directory path
)

# Call the function
calculate_average_scores(output_directory)
