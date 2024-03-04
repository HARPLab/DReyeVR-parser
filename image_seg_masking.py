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

rgb_frame_delay = 2
_debug = False

def camera_to_2d(sensor_config, frame_num, obj_loc, awareness_df):
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
    location = awareness_df["EgoVariables_VehicleLoc"][frame_num-1]
    ego_loc = np.asarray([location[0], location[1], location[2]])
    rotation = awareness_df["EgoVariables_VehicleRot"][frame_num-1]
    ego_rot = np.asarray([rotation[0], rotation[1], rotation[2]])
    
    vehicle_loc = carla.Location(*(ego_loc.squeeze()))/100
    vehicle_rot = carla.Rotation(*(ego_rot.squeeze()))
    vehicle_transform = carla.Transform(location=vehicle_loc, rotation=vehicle_rot)

    pts2d_mid, pts2d_left, pts2d_right = world2pixels(obj_loc_scaled, vehicle_transform, K, sensor_config)
    return pts2d_mid, pts2d_left, pts2d_right


def check_edge(p, border, width=800, height=600):
    in_border = p[0] < border or p[0] > width - border or p[1] < border or p[1] > height - border    
    return in_border #or outside_edge

def vectorized_edge_check(peds_locs, border=5):
    return np.any([check_edge(p, border) for p in peds_locs])

def check_frame_quality_nonvec(frame_num, recording_data_dict, sensor_config, inst_img_path):
    num_peds = 0
    min_num_peds = 3
    peds_locs = []
    peds_dict = {}
    visible_dict = recording_data_dict[frame_num]["AwarenessData"]["Visible"]
    for id in visible_dict:
        answer = int(visible_dict[id]["Answer"])
        if answer <= 8 and answer >=1:
            location_dict = visible_dict[id]["Location"]
            obj_loc = np.asarray([location_dict["x"], location_dict["y"], location_dict["z"]])
            obj_loc_2d = camera_to_2d(sensor_config, frame_num, obj_loc, recording_data_dict, awareness_df)[0]
            if obj_loc_2d[0] > 0 and obj_loc_2d[1] > 0:
                peds_locs.append(obj_loc_2d)
                peds_dict[id] = visible_dict[id]
                num_peds += 1
    
    if num_peds >= min_num_peds:
        inst_img  = Image.open(inst_img_path)
        raw_img = np.array(inst_img)
        distances = []
        is_valid = False
        for i in range(len(peds_locs)):
            p1 = peds_locs[i]
            # check if any ped is on the edge of the frame
            if p1[0] in range(0, 3) or p1[0] in range(797, 800) or p1[1] in range(0, 3) or p1[1] in range(597, 600):
                is_valid = False
            for j in range(i + 1, len(peds_locs)):
                p2 = peds_locs[j]
                distances.append(np.linalg.norm(np.array(p1) - np.array(p2)))
            # check that the R value corresponds to pedestrians
            if raw_img[p1[1], p1[0]][0] != 4:
                is_valid=False
        min_dist = min(distances)        
    
        if min_dist >= 30 and is_valid:
            return True, peds_dict
    return False, peds_dict

def check_frame_quality(frame_num, sensor_config, inst_img_path, awareness_df):
    num_peds = 0
    min_num_peds = 2
    min_dist_bn_peds = 20
    peds_locs = []
    peds_dict = {}
    #visible_dict = recording_data_dict[frame_num]["AwarenessData"]["Visible"]
    visible_list = awareness_df["AwarenessData_Visible"][frame_num-1]
    #for id in visible_dict:
    for i in range(len(visible_list)):
        id = str(visible_list[i])
        #answer = int(visible_dict[id]["Answer"])
        answer = int(awareness_df["AwarenessData_Answer"][frame_num-1][i])
        if answer <= 8 and answer >=1:
            #location_dict = visible_dict[id]["Location"]
            #obj_loc = np.asarray([location_dict["x"], location_dict["y"], location_dict["z"]])
            location = awareness_df["AwarenessData_VisibleLocation"][frame_num-1][i]
            obj_loc = np.asarray([location[0], location[1], location[2]])
            
            obj_loc_2d = camera_to_2d(sensor_config, frame_num, obj_loc, awareness_df)[0]
            if obj_loc_2d[0] > 0 and obj_loc_2d[1] > 0:
                peds_locs.append(obj_loc_2d)
                #peds_dict[id] = visible_dict[id]
                peds_dict[id] = location
                num_peds += 1
    
    if num_peds >= min_num_peds:
        inst_img  = Image.open(inst_img_path)
        raw_img = np.array(inst_img)
        distances = []
        is_valid = True


        all_distances = np.linalg.norm(np.array(peds_locs)[:, np.newaxis, :] - np.array(peds_locs)[np.newaxis, :, :], axis=-1)
        distances = all_distances[np.triu_indices(num_peds, k=1)]
        
        for i, p1 in enumerate(peds_locs):
            # check if any of the locations are on the image edge
            if vectorized_edge_check([p1]):
                is_valid = False
                return is_valid, peds_dict
            # check that the red value of inst seg is 4 corresponding to peds
            if raw_img[p1[1], p1[0]][0] != 4:
                is_valid = False
                return is_valid, peds_dict

        min_dist = min(distances)
        if min_dist <= min_dist_bn_peds:
            is_valid = False
        
        return is_valid, peds_dict
    
    return None, None
        
            
def get_instseg_id_offset(images_dir, sensor_config, awareness_df):
    f_num = 0
    for frame_num in range(100, len(awareness_df)):
        inst_img_path = "%s/instance_segmentation_output/%.6d.png" % (images_dir,  frame_num + rgb_frame_delay)
        if os.path.exists(inst_img_path):
            # TODO: this input should be frame_num not +delay -- correct?
            good, peds_dict = check_frame_quality(frame_num, sensor_config, inst_img_path, awareness_df)
            if good:
                f_num = frame_num
                break
    print("Frame number used: ", f_num+rgb_frame_delay)
    # test, peds_dict = check_frame_quality(2380, recording_data_dict, sensor_config)
    # f_num = 2380
    # print(peds_dict)
        
    offset = -1
    print(peds_dict)

    id_keys = sorted([int(k) for k in peds_dict])
    inst_img_path = "%s/instance_segmentation_output/%.6d.png" % (images_dir,  f_num + rgb_frame_delay)
    inst_img  = Image.open(inst_img_path)
    raw_img = np.array(inst_img)

    for i in range(len(id_keys)):
        #get true id from dict keys 
        true_id = str(id_keys[i])

        #get location and convert from camera coordinates
        #location_dict = peds_dict[true_id]["Location"]
        #obj_loc = np.asarray([location_dict["x"], location_dict["y"], location_dict["z"]])
        location = peds_dict[true_id]
        obj_loc = np.asarray([location[0], location[1], location[2]])
        pts2d_mid = camera_to_2d(sensor_config, f_num+rgb_frame_delay, obj_loc, awareness_df)[0]

        
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
            # TODO: why max
            b = max(b_values)
            g = max(g_values)
            
        except:
            continue
        
        if _debug:
            
            print(b, g)
            print(true_id)
            print(pts2d_mid)
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
            #diff = abs(diff)

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

    # # Display the image
    # cv2.imshow("Image", rgb)
    # cv2.waitKey(0)
    
    return mask

def get_all_instance_segmentation_images(rec_parse_file, images_dir, awareness_df, sensor_config_file):
    
    if os.path.exists("%s/offset.txt" % images_dir):
        with open("%s/offset.txt" % images_dir, 'r') as file:
            offset = int(file.read())
    else:
        #recording_data_dict = get_data_dict(rec_parse_file)
        sensor_config = configparser.ConfigParser()
        sensor_config.read(sensor_config_file)

        offset = get_instseg_id_offset(images_dir, sensor_config, awareness_df)
    print("Offset value: ", offset)

    # for frame_num in range(1, len(awareness_df)):
    #     print(frame_num)
    #     id_list = awareness_df["AwarenessData_Visible"][frame_num]
    #     user_input = awareness_df["AwarenessData_UserInput"][frame_num]
    #     aw_answer = awareness_df["AwarenessData_Answer"][frame_num]
    #     inst_img = "%s/instance_segmentation_output/%.6d.png" % (images_dir, frame_num+rgb_frame_delay)
    #     rgb_img = "%s/rgb_output/%.6d.png" % (images_dir, frame_num+rgb_frame_delay)
    #     if not os.path.exists(inst_img) and not os.path.exists(rgb_img):
    #         continue
        
        
    #     for id in id_list:
    #         run_id = id - offset
    #         mask = get_instance_seg_mask(inst_img, rgb_img, run_id)
    #         mask_img = Image.fromarray(mask)
    #         mask_name = "%s/instance_masks/%.6d_%d.png" % (images_dir, frame_num+rgb_frame_delay, id)
    #         mask_img.save(mask_name)
    #         label_mask = get_label_mask(id, mask_name, id_list, aw_answer, user_input)
    #         label_mask_img = Image.fromarray(label_mask)
    #         label_name = "%s/label_masks/%.6d_%d.png" % (images_dir, frame_num+rgb_frame_delay, id)
    #         label_mask_img.save(label_name)

if __name__ =="__main__":
    # awareness_parse_file = "/home/srkhuran-local/CarlaDReyeVR/DReyeVR-parser/results/ines_51-awdata.json"
    # rec_parse_file = "/home/srkhuran-local/CarlaDReyeVR/DReyeVR-parser/recording_files/exp_ines_51.txt"
    # images_dir = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/exp_ines-51_02_13_2024_16_11_13/images"

    awareness_parse_file = "/home/srkhuran-local/CarlaDReyeVR/DReyeVR-parser/results/jd_51-awdata.json"
    rec_parse_file = "/home/srkhuran-local/CarlaDReyeVR/DReyeVR-parser/recording_files/jd_51.txt"
    images_dir = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/exp_jd_51_02_16_2024_14_21_48/images"

    sensor_config_file = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/sensor_config.ini"
    #images_dir = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/exp_nik-pilot_12_05_2023_17_00_59/images"
    
    awareness_df = pd.read_json(awareness_parse_file, orient='index')
    aw_visible = awareness_df["AwarenessData_Visible"]

    get_all_instance_segmentation_images(rec_parse_file, images_dir, awareness_df, sensor_config_file)
