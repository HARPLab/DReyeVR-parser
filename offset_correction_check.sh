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
    # Get the parent directory of the 'images' folder
    # parent_dir=$location"/cbdr1-11"
    # dirname=$parent_dir"/images"
    # echo $parent_dir $dirname    
    parent_dir="$(dirname "$dirname")"
    echo $parent_dir
    python_output=$(python image_seg_masking.py --img-dir "$parent_dir/images" -aw "$parent_dir/rec_parse-awdata.json")
    echo $python_output
    #check_not_none "$python_output"      
    #exit
done


# find "$location" -mindepth 2 -maxdepth 2 -type d -name "images" | while IFS= read -r dirname; do
#     # Get the parent directory of the 'images' folder
#     # parent_dir=$location"/cbdr1-11"
#     # dirname=$parent_dir"/images"
#     # echo $parent_dir $dirname    
#     parent_dir="$(dirname "$dirname")"        
    
#     # Check if 'offset.txt' exists in the 'images' subfolder
#     if [ -f "$dirname/offset.txt" ]; then
#         # Rename 'offset.txt' to 'offset-1.txt'
#         mv "$dirname/offset.txt" "$dirname/offset-sc.txt"
#         #mv "$dirname/offset.txt" "$dirname/offset-pc.txt"
#         # Run the python command
#         python image_seg_masking.py \
#             --img-dir $parent_dir/images \
#             -aw $parent_dir/rec_parse-awdata.json > /tmp/foo.txt
        
#         # Check if both offset files have the same content
#         if [ -z "$(diff -q "$dirname/offset-1.txt" "$dirname/offset.txt")" ]; then
#             echo "Offset same for: $parent_dir"
#         else
#             echo "Content in offset files is different for directory: $parent_dir"
#         fi    
#     fi    
#     #exit
# done


# Content in offset files is different for directory: /media/storage/raw_data/cbdr9-61
# Content in offset files is different for directory: /media/storage/raw_data/cbdr1-11
# Content in offset files is different for directory: /media/storage/raw_data/cbdr8-36
# Content in offset files is different for directory: /media/storage/raw_data/cbdr9-53
# Content in offset files is different for directory: /media/storage/raw_data/cbdr7-35
# Content in offset files is different for directory: /media/storage/raw_data/abd-21
# Content in offset files is different for directory: /media/storage/raw_data/cbdr6-35
# Content in offset files is different for directory: /media/storage/raw_data/cbdr7-61

# Second run
# Content in offset files is different for directory: /media/storage/raw_data/cbdr9-61
# Content in offset files is different for directory: /media/storage/raw_data/cbdr8-36
# Content in offset files is different for directory: /media/storage/raw_data/cbdr9-53
# Content in offset files is different for directory: /media/storage/raw_data/cbdr7-35
# Content in offset files is different for directory: /media/storage/raw_data/abd-21
# Content in offset files is different for directory: /media/storage/raw_data/cbdr6-35
# Content in offset files is different for directory: /media/storage/raw_data/cbdr7-61
# Content in offset files is different for directory: /media/storage/raw_data/akash-32

