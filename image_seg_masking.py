import cv2
import os
import pandas as pd
import numpy as np
from recorder_info_extracter import get_data_dict
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import image


def get_label_mask(id, instance_mask_file, aw_visible, aw_answer, user_input, type_bit=16):
    id_idx = aw_visible.index(id)
    label = False
    if (user_input & type_bit == aw_answer[id_idx] & type_bit) and (user_input & aw_answer[id_idx]):
        label = True
    mask_img = Image.open(instance_mask_file)
    pixels = np.array(mask_img)
    white = (255, 255, 255, 255)
    
    white = np.array([255, 255, 255, 255], dtype=np.uint8)
    green= np.array([0, 255, 0, 255], dtype=np.uint8)
    red = np.array([255, 0, 0, 255], dtype=np.uint8)
    
    white_indices = np.all(pixels == white, axis=-1)

    if label == True:
        # Convert all white pixels to green.
        pixels[white_indices] = green
        
    else:
        pixels[white_indices] = red
    return pixels


def get_instance_seg_mask(inst_img_path, rgb_img_path, id):
    inst_img  = Image.open(inst_img_path) 
    rgb_img = Image.open(rgb_img_path)
    width, height = inst_img.size
    pixels_list = []
    vals = []
    mask = np.zeros_like(inst_img, dtype=np.uint8)
    #mask = np.zeros((width, height))
    raw_img = np.array(inst_img)
    b = raw_img[:, :, 2]
    g = raw_img[:, :, 1]
    # Calculate the sum of b*256 + g
    sum_bg = (b * 256) + g
    #print(sum_bg)
    # Create a mask where sum_bg is equal to target_value
    mask[sum_bg== id] =  [255, 255, 255, 255]
    mask[sum_bg != id] = [0, 0, 0, 255]
    #if pixels_list != []:
    #    coordinates = np.array(pixels_list)
    #    cv2.fillPoly(mask, [coordinates], color=(255, 255, 255))
    # rgb = np.array(rgb_img)
    # for point in pixels_list:
    #     cv2.circle(rgb, (point[0], point[1]), 5, (0, 0, 255), -1)

    # # Display the image
    # cv2.imshow("Image", rgb)
    # cv2.waitKey(0)
    
    return mask

def get_all_instance_segmentation_images(images_dir, awareness_df):
    for frame_num in range(len(awareness_df)):
        print(frame_num)
        id_list = awareness_df["AwarenessData_Visible"][frame_num]
        user_input = awareness_df["AwarenessData_UserInput"][frame_num]
        aw_answer = awareness_df["AwarenessData_Answer"][frame_num]
        inst_img = "%s/instance_segmentation_output/%.6d.png" % (images_dir, frame_num)
        rgb_img = "%s/rgb_output/%.6d.png" % (images_dir, frame_num)
        if not os.path.exists(inst_img) and not os.path.exists(rgb_img):
            continue
        
        for id in id_list:
            mask = get_instance_seg_mask(inst_img, rgb_img, id)
            mask_img = Image.fromarray(mask)
            mask_name = "%s/instance_masks/%.6d_%d.png" % (images_dir, frame_num, id)
            mask_img.save(mask_name)
            label_mask = get_label_mask(id, mask_name, id_list, aw_answer, user_input)
            label_mask_img = Image.fromarray(label_mask)
            label_name = "%s/label_masks/%.6d_%d.png" % (images_dir, frame_num, id)
            label_mask_img.save(label_name)


awareness_parse_file = "/home/srkhuran-local/CarlaDReyeVR/DReyeVR-parser/results/exp_sud_21_SA-awdata.json"
#awareness_parse_file = "/home/srkhuran-local/CarlaDReyeVR/DReyeVR-parser/results/exp_nik-pilot-awdata.json"
awareness_df = pd.read_json(awareness_parse_file, orient='index')
aw_visible = awareness_df["AwarenessData_Visible"]
images_dir = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/exp_sud_21_02_02_2024_09_33_57/images"
#images_dir = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/exp_nik-pilot_12_05_2023_17_00_59/images"
get_all_instance_segmentation_images(images_dir, awareness_df)
