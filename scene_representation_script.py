#Given a frame, 
# - get RGB image
# - get query instance segmentation
# - get gaze heatmap
# - get SA per object

#To do:
# - add in gaze heatmap
# - update script to have a main function
# - use an argument parser to get necessary file paths

import os
import cv2
import pandas as pd
import configparser
import numpy as np
from recorder_info_extracter import get_data_dict

CARLA_PATH = "/home/srkhuran-local/CarlaDReyeVR/carla"
import carla

def get_RGB(frame_num, recorder_filename):
    #write this code
    directory_name = recorder_filename.split("/")[-1]
    directory_name = directory_name.split(".rec")[0]
    image_filename = "%s/PythonAPI/examples/%s/images/rgb_output/%.6d.jpg" % (CARLA_PATH, directory_name, frame_num)
    #return filename, or the actual image .... ?
    rgb_image = cv2.imread(image_filename)
    return rgb_image
    
    
def get_instance_segm(frame_num, recorder_filename):
    #write this code
    directory_name = recorder_filename.split("/")[-1]
    directory_name = directory_name.split(".rec")[0]
    image_filename = "%s/PythonAPI/examples/%s/images/instance_segmentation_output/%.6d.jpg" % (CARLA_PATH, directory_name, frame_num)
    #return filename, or the actual image .... ?
    instance_segm_image = cv2.imread(image_filename)
    return instance_segm_image

# #instance_segmentation_output and rgb_output directories will be in a directory by the name of the recording file in carla/PythonAPI/examples

# #Step 2: call get_RGB on specific frame number to obtain that particular image
# rgb_img = get_RGB(frame_num, recording_file)
# # cv2.imshow("RGB", rgb_img)
# # cv2.waitKey(0) 
# print("Got RGB Image")

# #Step 3: call get_instance_segm on specific frame number to obtain that particular image
# instance_segm_img = get_instance_segm(frame_num, recording_file)
# # cv2.imshow("Instance Seg", instance_segm_img)
# # cv2.waitKey(0) 
# print("Got Instance Segmentation Image")
# #Step 4: Run awareness_parser
# #os.system("python awareness_parser.py -f " + recorder_parse_file)

# awareness_data_file = "results/exp_nik-pilot-awdata.json"
# awareness_df = pd.read_json(awareness_data_file, orient='index')
# #print(awareness_df.keys())

#create function from the parser to convert label into a T/F
def get_label(user_input, aw_visible, aw_answer, type_bit=16):
    actors_num = len(aw_visible)
    FoundCorrect = False
    for j in range(actors_num):
        if (user_input & type_bit == aw_answer[j] & type_bit) and (user_input & aw_answer[j]):
            FoundCorrect = True
            break
    return FoundCorrect

# #Step 5: get SA Label from awareness_frame
# aw_visible = awareness_df["AwarenessData_Visible"][frame_num]
# user_input = awareness_df["AwarenessData_UserInput"][frame_num]
# aw_answer = awareness_df["AwarenessData_Answer"][frame_num]
# if user_input == 0:
#     sa_label = None
# else:
#     sa_label = get_label(user_input, aw_visible, aw_answer)
# print(sa_label)


# #Step 6: CAMERA CONVERSION
# sensor_config = configparser.ConfigParser()
# sensor_config.read('sensor_config.ini')

# FOV = int(sensor_config['rgb']['fov'])
# w = int(sensor_config['rgb']['width'])
# h = int(sensor_config['rgb']['height'])
# F = w / (2 * np.tan(FOV * np.pi / 360))

# cam_info = {
#     'F': F,
#     'map_size' : 256,
#     'pixels_per_world' : 5.5,
#     'w' : w,
#     'h' : h,
#     'fy' : F,
#     'fx' : 1.0 * F,
#     'hack' : 0.4,
#     'cam_height' : sensor_config['rgb']['z'],
# }

# K = np.array([
# [cam_info['fx'], 0, cam_info['w']/2],
# [0, cam_info['fy'], cam_info['h']/2],
# [0, 0, 1]])

# #Construct dictionary containing necessary data per frame for the focus hit points and vehicle location/orientation
# recording_data_dict = get_data_dict(recorder_parse_file)

#Following functions are from gaze attention map test.ipynb
def ptsWorld2Cam(focus_hit_pt, world2camMatrix, K):
    tick_focus_hitpt_homog = np.hstack((focus_hit_pt,1))    
    sensor_points = np.dot(world2camMatrix, tick_focus_hitpt_homog)
    
    # Now we must change from UE4's coordinate system to an "standard" camera coordinate system

    # This can be achieved by multiplying by the following matrix:
    # [[ 0,  1,  0 ],
    #  [ 0,  0, -1 ],
    #  [ 1,  0,  0 ]]

    # Or, in this case, is the same as swapping:
    # (x, y ,z) -> (y, -z, x)
    point_in_camera_coords = np.array([
        sensor_points[1],
        sensor_points[2] * -1,
        sensor_points[0]])

    # Finally we can use our K matrix to do the actual 3D -> 2D.
    points_2d = np.dot(K, point_in_camera_coords)

    # Remember to normalize the x, y values by the 3rd value.
    points_2d /= points_2d[2]

    # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
    # contains all the y values of our points. In order to properly
    # visualize everything on a screen, the points that are out of the screen
    # must be discarted, the same with points behind the camera projection plane.
    # points_2d = points_2d.T

    # Extract the screen coords (uv) as integers.
    u_coord = points_2d[0].astype(np.int)
    v_coord = points_2d[1].astype(np.int)
    return (u_coord, v_coord)

def world2pixels(focus_hit_pt, vehicle_transform, K, sensor_config):
    '''
    takes in the dataframe row with all the information of where the world is currently 
    '''        
    vehicleP = vehicle_transform.get_matrix()
    
    # center image
    camera_loc_offset = carla.Location(x=float(sensor_config['rgb']['x']), y=float(sensor_config['rgb']['y']), z=float(sensor_config['rgb']['z']))    
    camera_rot_offset = carla.Rotation(pitch=float(sensor_config['rgb']['pitch']), yaw=float(sensor_config['rgb']['yaw']), roll=float(sensor_config['rgb']['roll']))
    cam_transform = carla.Transform(location=camera_loc_offset, rotation=camera_rot_offset)
    world2cam = np.matmul(cam_transform.get_inverse_matrix(), vehicle_transform.get_inverse_matrix())    
    
    u,v = ptsWorld2Cam(focus_hit_pt, world2cam, K)
    pts_mid = (u,v)
        
    # left image  
    camera_loc_offset = carla.Location(x=float(sensor_config['rgb_left']['x']), y=float(sensor_config['rgb_left']['y']), z=float(sensor_config['rgb_left']['z']))    
    camera_rot_offset = carla.Rotation(pitch=float(sensor_config['rgb_left']['pitch']), yaw=float(sensor_config['rgb_left']['yaw']), roll=float(sensor_config['rgb_left']['roll']))
    cam_transform = carla.Transform(location=camera_loc_offset, rotation=camera_rot_offset)    
    world2cam = np.matmul(cam_transform.get_inverse_matrix(), vehicle_transform.get_inverse_matrix())
        
    u,v = ptsWorld2Cam(focus_hit_pt, world2cam, K)
    pts_left = (u,v)
    
    # right image  
    camera_loc_offset = carla.Location(x=float(sensor_config['rgb_right']['x']), y=float(sensor_config['rgb_right']['y']), z=float(sensor_config['rgb_right']['z']))    
    camera_rot_offset = carla.Rotation(pitch=float(sensor_config['rgb_right']['pitch']), yaw=float(sensor_config['rgb_right']['yaw']), roll=float(sensor_config['rgb_right']['roll']))
    cam_transform = carla.Transform(location=camera_loc_offset, rotation=camera_rot_offset)    
    world2cam = np.matmul(cam_transform.get_inverse_matrix(), vehicle_transform.get_inverse_matrix())
        
    u,v = ptsWorld2Cam(focus_hit_pt, world2cam, K)
    pts_right = (u,v)    
    
    return pts_mid, pts_left, pts_right

# focus_hit_pt = recording_data_dict[frame_num]["FocusInfo"]["HitPoint"]
# loc = recording_data_dict[frame_num]["EgoVariables"]["VehicleLoc"]
# rot = recording_data_dict[frame_num]["EgoVariables"]["VehicleRot"]
# vehicle_loc = carla.Location(*(loc.squeeze()))/100
# vehicle_rot = carla.Rotation(*(rot.squeeze()))
# vehicle_transform = carla.Transform(location=vehicle_loc, rotation=vehicle_rot)

# pts2d_mid, pts2d_left, pts2d_right = world2pixels(focus_hit_pt, vehicle_transform, K, sensor_config)
# print(pts2d_mid)
# print(rgb_img.shape)

# #Plot the converted gaze coordinate onto the rgb image coordinate space
# image = cv2.circle(rgb_img, pts2d_mid, radius=10, color=(255, 0, 255), thickness=-1)
# cv2.imshow("RGB Current Frame Gaze", image)
# cv2.waitKey(0) 

# #Plot past 15 frames
# heatmap_points = []
# for i in range(1, 31):
#     frame = frame_num - i
#     focus_hit_pt_i = recording_data_dict[frame]["FocusInfo"]["HitPoint"]
#     loc = recording_data_dict[frame]["EgoVariables"]["VehicleLoc"]
#     rot = recording_data_dict[frame]["EgoVariables"]["VehicleRot"]
#     vehicle_loc_i = carla.Location(*(loc.squeeze()))/100
#     vehicle_rot_i = carla.Rotation(*(rot.squeeze()))
#     vehicle_transform = carla.Transform(location=vehicle_loc_i, rotation=vehicle_rot_i)

#     pts2d_mid, pts2d_left, pts2d_right = world2pixels(focus_hit_pt_i, vehicle_transform, K, sensor_config)
#     heatmap_points.append(pts2d_mid)
# print(heatmap_points)

# for p in heatmap_points:
#     cv2.circle(image, p, radius=3, color=(255, 0, 0), thickness=-1)
# cv2.imshow("RGB Gaze Heatmap",image)
# cv2.waitKey(0)

#Step 1: Run replay_instance_segm.py to get the rgb images and the instance segmentation images
recording_file = "/home/srkhuran-local/.config/Epic/CarlaUE4/Saved/exp_nik-pilot_12_05_2023_17_00_59.rec"
recorder_parse_file = "%s/PythonAPI/examples/exp_nik-pilot.txt" % CARLA_PATH
#os.system("python %s/PythonAPI/examples/replay_instance_segm.py -f %s -parse %s" %(CARLA_PATH, recording_file, recorder_parse_file))

frame_num = 5890 #update this

def get_image_inputs(frame_num, recording_file, recorder_parse_file):
    #call get_RGB on specific frame number to obtain that particular image
    rgb_img = get_RGB(frame_num, recording_file) 
    print("Got RGB Image.")

    #call get_instance_segm on specific frame number to obtain that particular image
    instance_segm_img = get_instance_segm(frame_num, recording_file)
    print("Got Instance Segmentation Image.")
    
    #Run awareness_parser
    #os.system("python awareness_parser.py -f " + recorder_parse_file)
    awareness_data_file = "results/exp_nik-pilot-awdata.json" #UPDATE THIS TO BE GENERAL    
    awareness_df = pd.read_json(awareness_data_file, orient='index')
    
    #get SA Label from awareness_frame
    aw_visible = awareness_df["AwarenessData_Visible"][frame_num]
    user_input = awareness_df["AwarenessData_UserInput"][frame_num]
    aw_answer = awareness_df["AwarenessData_Answer"][frame_num]
    if user_input == 0:
        sa_label = None
    else:
        sa_label = get_label(user_input, aw_visible, aw_answer)
    print("Situational Awareness Label: ", sa_label)
    
    sensor_config = configparser.ConfigParser()
    sensor_config.read('sensor_config.ini')

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

    #Construct dictionary containing necessary data per frame for the focus hit points and vehicle location/orientation
    recording_data_dict = get_data_dict(recorder_parse_file)
    
    focus_hit_pt = recording_data_dict[frame_num]["FocusInfo"]["HitPoint"]
    loc = recording_data_dict[frame_num]["EgoVariables"]["VehicleLoc"]
    rot = recording_data_dict[frame_num]["EgoVariables"]["VehicleRot"]
    vehicle_loc = carla.Location(*(loc.squeeze()))/100
    vehicle_rot = carla.Rotation(*(rot.squeeze()))
    vehicle_transform = carla.Transform(location=vehicle_loc, rotation=vehicle_rot)

    pts2d_mid, pts2d_left, pts2d_right = world2pixels(focus_hit_pt, vehicle_transform, K, sensor_config)
    print(pts2d_mid)
    print(rgb_img.shape)

    #Plot the converted gaze coordinate onto the rgb image coordinate space
    image = cv2.circle(rgb_img, pts2d_mid, radius=10, color=(255, 0, 255), thickness=-1)
    cv2.imshow("RGB Current Frame Gaze", image)
    cv2.waitKey(0) 

    #Plot past 15 frames
    heatmap_points = []
    for i in range(1, 31):
        frame = frame_num - i
        focus_hit_pt_i = recording_data_dict[frame]["FocusInfo"]["HitPoint"]
        loc = recording_data_dict[frame]["EgoVariables"]["VehicleLoc"]
        rot = recording_data_dict[frame]["EgoVariables"]["VehicleRot"]
        vehicle_loc_i = carla.Location(*(loc.squeeze()))/100
        vehicle_rot_i = carla.Rotation(*(rot.squeeze()))
        vehicle_transform = carla.Transform(location=vehicle_loc_i, rotation=vehicle_rot_i)

        pts2d_mid, pts2d_left, pts2d_right = world2pixels(focus_hit_pt_i, vehicle_transform, K, sensor_config)
        heatmap_points.append(pts2d_mid)
    print(heatmap_points)

    for p in heatmap_points:
        cv2.circle(image, p, radius=3, color=(255, 0, 0), thickness=-1)
    cv2.imshow("RGB Gaze Heatmap",image)
    cv2.waitKey(0)
    
    
get_image_inputs(frame_num, recording_file, recorder_parse_file)
    
    
    
    