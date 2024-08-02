
# Read data from "dataset.txt"
with open('./dataset.txt', 'r') as readfile:
    lines = readfile.readlines()

# Create a new file "newfile.txt" with the desired format
with open('./newfile.txt', "w") as output_file:
    for line in lines:

        # Split the line into columns using tab as the separator
        columns = line.strip().split("\t")
        
        c2 = float(columns[1])
        c3 = float(columns[2])
            
        # Calculate the average of c2 and c3
        average = (c2 + c3) / 2
            
        # Write the data to the new file in the desired format
        output_file.write(f"{c2}\t{c3}\t{average}\n")

print("Data has been processed and saved to newfile.txt.")



'''
import os

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file paths for input and output files
input_file_path = os.path.join(current_directory, "dataset.txt")
output_file_path = os.path.join(current_directory, "newfile.txt")

# Read data from "dataset.txt"
with open(input_file_path, "r") as input_file:
    lines = input_file.readlines()

# Create a new file "newfile.txt" with the desired format
with open(output_file_path, "w") as output_file:
    for line in lines:
        # Split the line into columns using tab as the separator
        columns = line.strip().split("\t")
        
        c2 = float(columns[1])
        c3 = float(columns[2])
            
        # Calculate the average of c2 and c3
        average = (c2 + c3) / 2
            
        # Write the data to the new file in the desired format
        output_file.write(f"{c2}\t{c3}\t{average}\n")

print("Data has been processed and saved to newfile.txt.")
'''
