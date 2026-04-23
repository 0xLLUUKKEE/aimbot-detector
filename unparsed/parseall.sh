#!/bin/bash

# Script to run DmoParser.py on every .dmo file in the current directory,
# generating a corresponding .json file for each.

# Enable globbing that expands to nothing if no .dmo files exist
shopt -s nullglob

# Array of .dmo files
files=( *.dmo )

# Check if there are any .dmo files
if [ ${#files[@]} -eq 0 ]; then
    echo "No .dmo files found in the current directory."
    exit 0
fi

# Process each file
for file in "${files[@]}"; do
    # Generate output filename by replacing .dmo with .json
    output="${file%.dmo}.json"
    echo "Processing '$file' -> '$output'"
    python3 DmoParser.py "$file" > "$output"
done

echo "Done."
