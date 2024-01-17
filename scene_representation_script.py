#Given a frame, 
# - get RGB image
# - get query instance segmentation
# - get gaze heatmap
# - get SA per object

#Step 2: call get_RGB on specific frame number to obtain that particular image
#Step 3: call get_instance_segm on specific frame number to obtain that particular image
import os

def get_RGB(frame_num):
    #write this code
    
def get_instance_segm(frame_num):
    #write this code

 
#Step 1: Run replay_instance_segm.py to get the rgb images and the instance segmentation images
recording_file = "/home/srkhuran-local/.config/Epic/CarlaUE4/Saved/exp_nik-pilot_12_05_2023_17_00_59.rec"
recorder_parse_file = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/exp_nik-pilot.txt"
os.system("python replay_instance_segm.py -f " + recording_file + " -parse " + recorder_parse_file)

#instance_segmentation_output and rgb_output directories will be in a directory by the name of the recording file in carla/PythonAPI/examples


