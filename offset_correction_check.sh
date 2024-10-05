#!/bin/bash

# Define the location of directories
location="/media/storage/raw_data"

check_not_none() {
    local value="$1"
    if [ "$value" != "None" ]; then
        echo "Value is not None: $value"
    else
        echo "Value is None"
    fi
}

# Iterate over each directory in the location
find "$location" -mindepth 2 -maxdepth 2 -type d -name "images" | while IFS= read -r dirname; do
    parent_dir="$(dirname "$dirname")"
    echo $parent_dir
    python_output=$(python image_seg_masking.py --img-dir "$parent_dir/images" -aw "$parent_dir/rec_parse-awdata.json")
    echo $python_output
done