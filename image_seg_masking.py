import cv2
import os
import pandas as pd
import numpy as np
import carla
import configparser
from recorder_info_extracter import get_data_dict
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import image
from recorder_info_extracter import get_data_dict
from scene_representation_script import ptsWorld2Cam, world2pixels

def camera_to_2d(sensor_config, frame_num, obj_loc, recording_data_dict):
    FOV = int(sensor_config['rgb']['fov'])
    w = int(sensor_config['rgb']['width'])
    h = int(sensor_config['rgb']['height'])
    F = w / (2 * np.tan(FOV * np.pi / 360))
    cam_info = {
        'F': F,
        'map_size' : 256,
        'pixels_per_world' : 5.5,
        'w' : w,
        'h' : h,
        'fy' : F,
        'fx' : 1.0 * F,
        'hack' : 0.4,
        'cam_height' : sensor_config['rgb']['z'],
    }
    K = np.array([
    [cam_info['fx'], 0, cam_info['w']/2],
    [0, cam_info['fy'], cam_info['h']/2],
    [0, 0, 1]])
    obj_loc_scaled = np.array(obj_loc.squeeze())/100
    loc = recording_data_dict[frame_num]["EgoVariables"]["VehicleLoc"]
    rot = recording_data_dict[frame_num]["EgoVariables"]["VehicleRot"]
    vehicle_loc = carla.Location(*(loc.squeeze()))/100
    vehicle_rot = carla.Rotation(*(rot.squeeze()))
    vehicle_transform = carla.Transform(location=vehicle_loc, rotation=vehicle_rot)

    pts2d_mid, pts2d_left, pts2d_right = world2pixels(obj_loc_scaled, vehicle_transform, K, sensor_config)
    return pts2d_mid, pts2d_left, pts2d_right


def check_frame_quality(frame_num, recording_data_dict, sensor_config):
    num_peds = 0
    peds_locs = []
    peds_dict = {}
    visible_dict = recording_data_dict[frame_num]["AwarenessData"]["Visible"]
    for id in visible_dict:
        answer = int(visible_dict[id]["Answer"])
        if answer <= 8 and answer >=1:
            location_dict = visible_dict[id]["Location"]
            obj_loc = np.asarray([location_dict["x"], location_dict["y"], location_dict["z"]])
            obj_loc_2d = camera_to_2d(sensor_config, frame_num, obj_loc, recording_data_dict)[0]
            if obj_loc_2d[0] > 0 and obj_loc_2d[1] > 0:
                peds_locs.append(obj_loc_2d)
                peds_dict[id] = visible_dict[id]
                num_peds += 1
    
    if num_peds >= 2:
        distances = []
        on_edge = False
        for i in range(len(peds_locs)):
            p1 = peds_locs[i]
            if p1[0] in range(0, 3) or p1[0] in range(797, 800) or p1[1] in range(0, 3) or p1[1] in range(597, 600):
                on_edge = True
            for j in range(i + 1, len(peds_locs)):
                p2 = peds_locs[j]
                distances.append(np.linalg.norm(np.array(p1) - np.array(p2)))
        min_dist = min(distances)
    
        if min_dist >= 30 and (not on_edge):
            return True, peds_dict
    return False, peds_dict
            
def get_instseg_id_offset(recording_data_dict, images_dir, sensor_config, awareness_df):
    f_num = 0
    for frame_num in range(1, len(awareness_df)):
        if os.path.exists("%s/instance_segmentation_output/%.6d.png" % (images_dir,  frame_num+2)):
            good, peds_dict = check_frame_quality(frame_num+2, recording_data_dict, sensor_config)
            if good:
                f_num = frame_num
                break
    print(f_num)
    # test, peds_dict = check_frame_quality(2380, recording_data_dict, sensor_config)
    # f_num = 2380
    # print(peds_dict)
        
    offset = -1

    id_keys = sorted([int(k) for k in peds_dict])
    inst_img_path = "%s/instance_segmentation_output/%.6d.png" % (images_dir,  f_num + 2)
    inst_img  = Image.open(inst_img_path)
    raw_img = np.array(inst_img)
    plt.imshow(raw_img)
    plt.show()
    for i in range(len(id_keys)):
        #get true id from dict keys 
        true_id = str(id_keys[i])

        #get location and convert from camera coordinates
        location_dict = peds_dict[true_id]["Location"]
        obj_loc = np.asarray([location_dict["x"], location_dict["y"], location_dict["z"]])
        pts2d_mid = camera_to_2d(sensor_config, f_num+2, obj_loc, recording_data_dict)[0]

        print(true_id)
        print(pts2d_mid)
        
        try:
            x, y =  pts2d_mid 
            x = max(min(x, 800 - 1), 0)
            y = max(min(y, 600 - 1), 0)
            region = raw_img[max(y - 5, 0):min(y + 5, 600),
                        max(x - 5, 0):min(x + 5, 800)]
            region_b = region[:, :, 2]
            region_g = region[:, :, 1]

            b_values = np.unique(region_b)
            g_values = np.unique(region_g)
            
            #b = raw_img[pts2d_mid[1], pts2d_mid[0], 2]
            #g = raw_img[pts2d_mid[1], pts2d_mid[0], 1]
            b = max(b_values)
            g = max(g_values)
            print(b)
            print(g)
        except:
            continue
        image = cv2.circle(raw_img, pts2d_mid, radius=10, color=(255, 0, 255), thickness=-1)
        plt.imshow(image)
        plt.show()
            
        #get rgb from that coordinate value in instance seg image
        # b = raw_img[pts2d_mid[0], pts2d_mid[1], 2]
        # g = raw_img[pts2d_mid[0], pts2d_mid[1], 1]

        #calculate 256*b+g to get run id
        if b != 0 and g != 0:
            run_id = 256*b + g

            #calculate offset
            diff = int(true_id) - run_id
            diff = abs(diff)

            #offset should be the same across all objects (and across run)
            if offset == -1:
                offset = diff
            else:
                assert(offset == diff)
    with open("%s/offset.txt" % images_dir, 'w') as file:
        file.write(str(offset))

    return offset


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

def get_all_instance_segmentation_images(rec_parse_file, images_dir, awareness_df, sensor_config_file):
    recording_data_dict = get_data_dict(rec_parse_file)
    sensor_config = configparser.ConfigParser()
    sensor_config.read(sensor_config_file)
    
    if os.path.exists("%s/offset.txt" % images_dir):
        with open("%s/offset.txt" % images_dir, 'r') as file:
            offset = int(file.read())
    else:
        offset = get_instseg_id_offset(recording_data_dict, images_dir, sensor_config, awareness_df)
    print(offset)

    # for frame_num in range(1, len(awareness_df)):
    #     print(frame_num)
    #     id_list = awareness_df["AwarenessData_Visible"][frame_num]
    #     user_input = awareness_df["AwarenessData_UserInput"][frame_num]
    #     aw_answer = awareness_df["AwarenessData_Answer"][frame_num]
    #     inst_img = "%s/instance_segmentation_output/%.6d.png" % (images_dir, frame_num)
    #     rgb_img = "%s/rgb_output/%.6d.png" % (images_dir, frame_num)
    #     if not os.path.exists(inst_img) and not os.path.exists(rgb_img):
    #         continue
        
        
        # for id in id_list:
        #     mask = get_instance_seg_mask(inst_img, rgb_img, id)
        #     mask_img = Image.fromarray(mask)
        #     mask_name = "%s/instance_masks/%.6d_%d.png" % (images_dir, frame_num, id)
        #     mask_img.save(mask_name)
        #     label_mask = get_label_mask(id, mask_name, id_list, aw_answer, user_input)
        #     label_mask_img = Image.fromarray(label_mask)
        #     label_name = "%s/label_masks/%.6d_%d.png" % (images_dir, frame_num, id)
        #     label_mask_img.save(label_name)


#awareness_parse_file = "/home/srkhuran-local/CarlaDReyeVR/DReyeVR-parser/results/exp_sud_21_SA-awdata.json"
awareness_parse_file = "/home/srkhuran-local/CarlaDReyeVR/DReyeVR-parser/results/ines_51-awdata.json"
rec_parse_file = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/exp_ines_51.txt"
#awareness_parse_file = "/home/srkhuran-local/CarlaDReyeVR/DReyeVR-parser/results/exp_nik-pilot-awdata.json"
awareness_df = pd.read_json(awareness_parse_file, orient='index')
aw_visible = awareness_df["AwarenessData_Visible"]
images_dir = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/exp_ines-51_02_13_2024_16_11_13/images"
sensor_config_file = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/sensor_config.ini"
#images_dir = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/exp_nik-pilot_12_05_2023_17_00_59/images"
get_all_instance_segmentation_images(rec_parse_file, images_dir, awareness_df, sensor_config_file)
