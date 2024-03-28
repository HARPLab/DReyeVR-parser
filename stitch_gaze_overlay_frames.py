import cv2
import argparse
import os

# Parse command line arguments

argparser = argparse.ArgumentParser(
    description=__doc__)

argparser.add_argument(
    '-f', '--frames-dir',
    help = "path to the rgb frames"
)
argparser.add_argument(
    '-v', '--video-name',
    help = "path to the rgb frames"
)

args = argparser.parse_args()

# Directory containing your images
image_folder = args.frames_dir

# Video name and codec
video_name = args.video_name
codec = cv2.VideoWriter_fourcc(*'mp4v')

# Get the list of files in each folder
left_files = sorted(os.listdir(os.path.join(image_folder, 'left')), key = lambda k:int(k.split('.')[0]))
mid_files = sorted(os.listdir(os.path.join(image_folder, 'mid')), key = lambda k:int(k.split('.')[0]))
right_files = sorted(os.listdir(os.path.join(image_folder, 'right')), key = lambda k:int(k.split('.')[0]))

# Get dimensions of the first image from each folder
left_img = cv2.imread(os.path.join(image_folder, 'left', left_files[0]))
mid_img = cv2.imread(os.path.join(image_folder, 'mid', mid_files[0]))
right_img = cv2.imread(os.path.join(image_folder, 'right', right_files[0]))

height, width, _ = left_img.shape

# Create the video writer object
video = cv2.VideoWriter(os.path.join(image_folder, video_name), codec, 30, (width*3,height))

# Loop through each image and write to the video
for left_file, mid_file, right_file in zip(left_files, mid_files, right_files):
    left_frame = cv2.imread(os.path.join(image_folder, 'left', left_file))
    mid_frame = cv2.imread(os.path.join(image_folder, 'mid', mid_file))
    right_frame = cv2.imread(os.path.join(image_folder, 'right', right_file))

    # Concatenate frames horizontally
    horizontal_stacked = cv2.hconcat([left_frame, mid_frame, right_frame])

    # Write the stitched frame to the video
    video.write(horizontal_stacked)

# Release resources
cv2.destroyAllWindows()
video.release()