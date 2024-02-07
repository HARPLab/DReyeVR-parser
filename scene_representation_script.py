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

import argparse

import carla
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from recorder_info_extracter import get_data_dict


def get_RGB(frame_num, images_dir):
    image_filename = "%s/rgb_output/%.6d.jpg" % (images_dir, frame_num)
    rgb_image = cv2.imread(image_filename)
    return rgb_image
    
    
def get_instance_segm(frame_num, images_dir):
    image_filename = "%s/instance_segmentation_output/%.6d.jpg" % (images_dir, frame_num)
    instance_segm_image = cv2.imread(image_filename)
    return instance_segm_image

#create function from the parser to convert label into a T/F
def get_label(user_input, aw_visible, aw_answer, type_bit=16):
    actors_num = len(aw_visible)
    FoundCorrect = False
    for j in range(actors_num):
        if (user_input & type_bit == aw_answer[j] & type_bit) and (user_input & aw_answer[j]):
            FoundCorrect = True
            break
    return FoundCorrect

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

def gaussian_contour_plot(gaze_image, fname, frame_num, gaze_points, sigma=1.0, contour_levels=3):
    # Create a grid of coordinates
    height, width = gaze_image.shape[:2]
    y, x = np.mgrid[0:height, 0:width]

    composite_gaussian = np.zeros((height, width), dtype=float)

    # Combine Gaussians centered at each point
    for center_pixel in gaze_points:
        mean = center_pixel
        covariance_matrix = np.eye(2) * (sigma**2)
        gaussian_distribution = multivariate_normal(mean=mean, cov=covariance_matrix)

        positions = np.column_stack((x.ravel(), y.ravel()))
        values = gaussian_distribution.pdf(positions)
        gaussian_image = values.reshape(height, width)

        composite_gaussian += gaussian_image

    # Plot the original image and overlay the composite Gaussian contour plot
    plt.imshow(gaze_image, cmap='gray')
    contours = plt.contourf(x, y, composite_gaussian, levels=contour_levels, cmap='Reds', alpha=0.7)
    
    if os.path.exists("gaze_heatmap/%s" % fname) is False:
        os.makedirs("gaze_heatmap/%s" % fname)
    output_file_name_heat = "gaze_heatmap/%s/%s.jpg" % (fname, str(frame_num))
    plt.savefig(output_file_name_heat)
    plt.clf()


def get_image_inputs(frame_num, recorder_parse_file, recording_data_dict, images_dir, awareness_parse_file, sensor_config):
    print("Frame Number: ", frame_num)
    #call get_RGB on specific frame number to obtain that particular image
    rgb_img = get_RGB(frame_num, images_dir) 
    print("Got RGB Image.")

    #call get_instance_segm on specific frame number to obtain that particular image
    instance_segm_img = get_instance_segm(frame_num, images_dir)

    print("Got Instance Segmentation Image.")
    
    fname = os.path.basename(recorder_parse_file)
    fname = os.path.splitext(fname)[0]

    awareness_df = pd.read_json(awareness_parse_file, orient='index')
    
    #get SA Label from awareness_frame
    aw_visible = awareness_df["AwarenessData_Visible"][frame_num]
    user_input = awareness_df["AwarenessData_UserInput"][frame_num]
    aw_answer = awareness_df["AwarenessData_Answer"][frame_num]
    if user_input == 0:
        sa_label = None
    else:
        sa_label = get_label(user_input, aw_visible, aw_answer)
    print("Situational Awareness Label: ", sa_label)

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
    
    focus_hit_pt = recording_data_dict[frame_num]["FocusInfo"]["HitPoint"]
    focus_hit_pt_scaled = np.array(focus_hit_pt.squeeze())/100
    loc = recording_data_dict[frame_num]["EgoVariables"]["VehicleLoc"]
    rot = recording_data_dict[frame_num]["EgoVariables"]["VehicleRot"]
    vehicle_loc = carla.Location(*(loc.squeeze()))/100
    vehicle_rot = carla.Rotation(*(rot.squeeze()))
    vehicle_transform = carla.Transform(location=vehicle_loc, rotation=vehicle_rot)

    pts2d_mid, pts2d_left, pts2d_right = world2pixels(focus_hit_pt_scaled, vehicle_transform, K, sensor_config)

    #Plot the converted gaze coordinate onto the rgb image coordinate space
    image = cv2.circle(rgb_img, pts2d_mid, radius=10, color=(255, 0, 255), thickness=-1)

    #Plot past 15 frames
    heat_points2 = [pts2d_mid]
    heatmap_points = []
    if frame_num <= 16:
        end_point = frame_num
    else:
        end_point = 16
    for i in range(1, end_point):
        frame = frame_num - i
        focus_hit_pt_i = recording_data_dict[frame]["FocusInfo"]["HitPoint"]
        focus_hit_pt_i_scaled = np.array(focus_hit_pt_i.squeeze())/100
        loc = recording_data_dict[frame]["EgoVariables"]["VehicleLoc"]
        rot = recording_data_dict[frame]["EgoVariables"]["VehicleRot"]
        vehicle_loc_i = carla.Location(*(loc.squeeze()))/100
        vehicle_rot_i = carla.Rotation(*(rot.squeeze()))
        vehicle_transform = carla.Transform(location=vehicle_loc_i, rotation=vehicle_rot_i)

        pts2d_mid, pts2d_left, pts2d_right = world2pixels(focus_hit_pt_i_scaled, vehicle_transform, K, sensor_config)
        heatmap_points.append(pts2d_mid)
        heat_points2.append(pts2d_mid)
    
    for p in heatmap_points:
        cv2.circle(image, p, radius=3, color=(255, 0, 0), thickness=-1)

    if os.path.exists("gaze_history/%s" % fname) is False:
        os.makedirs("gaze_history/%s" % fname)
    output_file_name = "gaze_history/%s/%s.jpg" % (fname, str(frame_num))
    cv2.imwrite(output_file_name, image)
    
    gaussian_contour_plot(image, fname, frame_num, heat_points2, sigma=40)
    
    return image, rgb_img, instance_segm_img
    

def get_all_images(recorder_parse_file, images_dir, awareness_parse_file, sensor_config):
    #Construct dictionary containing necessary data per frame for the focus hit points and vehicle location/orientation
    recording_data_dict = get_data_dict(recorder_parse_file)

    for f in recording_data_dict.keys():
        if f > 1:
            get_image_inputs(f, recorder_parse_file, recording_data_dict, images_dir, awareness_parse_file, sensor_config)
    

def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-a', '--aw-parse-file',
        metavar='A',
        help='parse json file outputted by awareness parser')
    argparser.add_argument(
        '-r', '--rec-parse-file',
        metavar='R',
        help='txt file from show recorder file nfo output')
    argparser.add_argument(
        '-i', '--images_dir',
        metavar='I',
        help='images directory, should have folder of rgb and instance segmentation images')
    argparser.add_argument(
        '-s', '--sensor-config',
        metavar='S',
        help='sensor config')
    args = argparser.parse_args()
    
    sensor_config = configparser.ConfigParser()
    sensor_config.read('sensor_config.ini')
    
    get_all_images(args.rec_parse_file, args.images_dir, args.aw_parse_file, sensor_config)
    

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')