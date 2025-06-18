#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
import random


def read_random_lines(file_path, num_lines=10, seed=None):
    if seed is None:
        seed = random.randint(
            0, 2**32 - 1
        )  # Generate a random seed if none is provided
    random.seed(seed)  # Set the seed for reproducibility

    # Step 1: Count total lines in the file
    with open(file_path, "r") as file:
        total_lines = sum(1 for _ in file)

    if num_lines > total_lines:
        raise ValueError("Number of lines to read exceeds total lines in the file.")

    # Step 2: Randomly select line numbers
    selected_line_numbers = sorted(random.sample(range(total_lines), num_lines))

    # Step 3: Read only the selected lines
    random_lines = []
    with open(file_path, "r") as file:
        for current_line_number, line in enumerate(file):
            if current_line_number == selected_line_numbers[0]:
                random_lines.append(line.strip())
                selected_line_numbers.pop(0)  # Move to the next line to fetch
                if not selected_line_numbers:  # Stop if we've read all desired lines
                    break

    return random_lines, seed


# Example usage
### file_path = 'example.txt'  # Replace with your file path
### num_lines = 10
### random_lines, used_seed = read_random_lines_efficient(file_path, num_lines)
