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

import multiprocessing


def get_RGB(frame_num, images_dir):
    image_filename = "%s/rgb_output/%.6d.png" % (images_dir, frame_num)
    if not os.path.exists(image_filename):
        print(image_filename)
        raise FileNotFoundError
    
    rgb_image = cv2.imread(image_filename)
    image_filename_left = "%s/rgb_output_left/%.6d.png" % (images_dir, frame_num)
    if not os.path.exists(image_filename_left):
        print(image_filename_left)
        raise FileNotFoundError
    
    rgb_image_left = cv2.imread(image_filename_left)

    image_filename_right = "%s/rgb_output_right/%.6d.png" % (images_dir, frame_num)
    if not os.path.exists(image_filename_right):
        print(image_filename_right)
        raise FileNotFoundError
    
    rgb_image_right = cv2.imread(image_filename_right)
    
    return rgb_image, rgb_image_left, rgb_image_right
    
    
def get_instance_segm(frame_num, images_dir):
    image_filename = "%s/instance_segmentation_output/%.6d.png" % (images_dir, frame_num)
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

def gaussian_contour_plot(gaze_image, fname, frame_num, gaze_points, sigma=1.0, cam_dir='mid', contour_levels=3):
    # Create a grid of coordinates
    height, width = gaze_image.shape[:2]
    y, x = np.mgrid[0:height, 0:width]

    composite_gaussian = np.zeros((height, width), dtype=float)

    # Combine Gaussians centered at each point
    for center_pixel, _ in gaze_points:
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
    
    if os.path.exists("gaze_heatmap/%s/%s" % (fname, cam_dir)) is False:
        os.makedirs("gaze_heatmap/%s/%s" % (fname, cam_dir))
        
    output_file_name_heat = "gaze_heatmap/%s/%s/%s.jpg" % (fname, cam_dir, str(frame_num))
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
    # image = cv2.circle(rgb_img, pts2d_mid, radius=10, color=(255, 0, 255), thickness=-1)

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


def overlay_gaze_and_buttons(frame_num, recorder_parse_file, recording_data_dict, images_dir, awareness_parse_file, sensor_config, data_dir=None):
    rgb_frame_delay = 2 #TODO: Make Global Variable
    # rgb_frame_delay = 40
    print("Frame Number: ", frame_num)
    #call get_RGB on specific frame number to obtain that particular image
    try:
        rgb_img, rgb_img_left, rgb_img_right = get_RGB(frame_num+rgb_frame_delay, images_dir)         
        print("Got RGB Image.")
    except FileNotFoundError:
        return        

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
        # sa_label = get_label(user_input, aw_visible, aw_answer)
        # print("Situational Awareness Label: ", sa_label)    
            
        # Get and overlay button press
        if user_input // 16 == 1:
            color = (0, 255, 0)
            # Green is for vehicles
        else:
            color = (0, 0, 255)
            # red is for pedestrians

        if user_input%16 == 1:
            pt1 = (125, 100)
            pt2 = (100, 125)
            pt3 = (150, 125)
            
        elif user_input%16 == 2:
            pt1 = (125, 100)
            pt2 = (150, 125)
            pt3 = (125, 150)

        elif user_input%16 == 4:
            pt1 = (100, 125)
            pt2 = (150, 125)
            pt3 = (125, 150)

        elif user_input%16 == 8:
            pt1 = (125, 100)
            pt2 = (100, 125)
            pt3 = (125, 150)
        
        # print(color, user_input)
        triangle_cnt = np.array( [pt1, pt2, pt3] )
        cv2.drawContours(rgb_img, [triangle_cnt], 0, color, -1)    

    # get gaze 3d pt and map to pixels
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
    
    if (pts2d_mid[0] >= 0 and pts2d_mid[0] <= w) and (pts2d_mid[1] >= 0 and pts2d_mid[1] <= h):
        image_mid = cv2.circle(rgb_img, pts2d_mid, radius=10, color=(255, 0, 255), thickness=-1)
        heat_points2 = [[pts2d_mid, 'mid']]
        image_left = rgb_img_left.copy()
        image_right = rgb_img_right.copy()

    else:
        image_mid = rgb_img.copy()
        
        if (pts2d_left[0] >= 0 and pts2d_left[0] <= w) and (pts2d_left[1] >= 0 and pts2d_left[1] <= h):
            image_left = cv2.circle(rgb_img_left, pts2d_left, radius=10, color=(255, 0, 255), thickness=-1)
            heat_points2 = [[pts2d_left, 'left']]
        else:
            image_left = rgb_img_left.copy()
            
        if (pts2d_right[0] >= 0 and pts2d_right[0] <= w) and (pts2d_right[1] >= 0 and pts2d_right[1] <= h):
            image_right = cv2.circle(rgb_img_right, pts2d_right, radius=10, color=(255, 0, 255), thickness=-1)
            heat_points2 = [[pts2d_right, 'right']]
        else:
            image_right = rgb_img_right.copy()
        # Plot the converted gaze coordinate onto the rgb image coordinate space

    # Plot past 15 frames
    
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
        
        if (pts2d_mid[0] >= 0 and pts2d_mid[0] <= w) and (pts2d_mid[1] >= 0 and pts2d_mid[1] <= h):
            heatmap_points.append([pts2d_mid, 'mid'])
            heat_points2.append([pts2d_mid, 'mid'])
        
        elif (pts2d_left[0] >= 0 and pts2d_left[0] <= w) and (pts2d_left[1] >= 0 and pts2d_left[1] <= h):
            heatmap_points.append([pts2d_left, 'left'])
            heat_points2.append([pts2d_left, 'left'])
        
        elif (pts2d_right[0] >= 0 and pts2d_right[0] <= w) and (pts2d_right[1] >= 0 and pts2d_right[1] <= h):
            heatmap_points.append([pts2d_right, 'right'])
            heat_points2.append([pts2d_right, 'right'])
        else:
            pass
    
    for p in heatmap_points:
        if p[1] == 'mid':
            cv2.circle(image_mid, p[0], radius=3, color=(255, 0, 0), thickness=-1)
        elif p[1] == 'left':
            cv2.circle(image_left, p[0], radius=3, color=(255, 0, 0), thickness=-1)
        elif p[1] == 'right':
            cv2.circle(image_right, p[0], radius=3, color=(255, 0, 0), thickness=-1)
        else:
            pass

    if data_dir is None:
        overlay_out_dir = "gaze_button_overlay/%s" % fname
    else:
        overlay_out_dir = os.path.join(data_dir, "gaze_button_overlay")
        
    if os.path.exists(overlay_out_dir) is False:
        os.makedirs(overlay_out_dir)

    if os.path.exists(os.path.join(overlay_out_dir, 'mid')) is False:
        os.makedirs(os.path.join(overlay_out_dir, 'mid'))
    
    if os.path.exists(os.path.join(overlay_out_dir, 'left')) is False:
        os.makedirs(os.path.join(overlay_out_dir, 'left'))

    if os.path.exists(os.path.join(overlay_out_dir, 'right')) is False:
        os.makedirs(os.path.join(overlay_out_dir, 'right'))
        
    output_file_name = "{}/{}/{:06d}.jpg".format(overlay_out_dir, 'mid', frame_num)
    cv2.imwrite(output_file_name, image_mid)
    
    output_file_name = "{}/{}/{:06d}.jpg".format(overlay_out_dir, 'left', frame_num)
    cv2.imwrite(output_file_name, image_left)
    
    output_file_name = "{}/{}/{:06d}.jpg".format(overlay_out_dir, 'right', frame_num)
    cv2.imwrite(output_file_name, image_right)
    
    gaussian_contour_plot(image_mid, fname, frame_num, heat_points2, sigma=40, cam_dir='mid')
    gaussian_contour_plot(image_left, fname, frame_num, heat_points2, sigma=40, cam_dir='left')
    gaussian_contour_plot(image_right, fname, frame_num, heat_points2, sigma=40, cam_dir='right')
    
    return image_mid, rgb_img, image_left, rgb_img_left, image_right, rgb_img_right
    


def overlay_gaze_and_buttons_wrapper(args):
    f, recorder_parse_file, recording_data_dict, images_dir, awareness_parse_file, sensor_config, data_dir = args
    overlay_gaze_and_buttons(f, recorder_parse_file, recording_data_dict, images_dir, awareness_parse_file, sensor_config, data_dir)

def get_all_images(recorder_parse_file, images_dir, awareness_parse_file, sensor_config, data_dir=None):
    #Construct dictionary containing necessary data per frame for the focus hit points and vehicle location/orientation
    recording_data_dict = get_data_dict(recorder_parse_file)

    # for f in recording_data_dict.keys():
    #     if f > 1:
    #         # get_image_inputs(f, recorder_parse_file, recording_data_dict, images_dir, awareness_parse_file, sensor_config)
    #         overlay_gaze_and_buttons(f, recorder_parse_file, recording_data_dict, images_dir, awareness_parse_file, sensor_config)
    args_list = [(f, recorder_parse_file, recording_data_dict, images_dir, awareness_parse_file, sensor_config, data_dir) for f in range(1, max(recording_data_dict.keys()))]
    with multiprocessing.Pool(processes=10) as pool:
        pool.map(overlay_gaze_and_buttons_wrapper, args_list)
        
    

def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-data_dir', '--data-dir',
        help = "path to the participant data directory"
    )
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
    sensor_config.read(args.sensor_config)
    
    if args.data_dir is None:
        get_all_images(args.rec_parse_file, args.images_dir, args.aw_parse_file, sensor_config, None)
    else:
        is_frames_dir = os.path.join(args.data_dir, 'images')
        awareness_parse_file = os.path.join(args.data_dir, 'rec_parse-awdata.json') 
        rec_parse_file = os.path.join(args.data_dir, 'rec_parse.txt') 
        get_all_images(rec_parse_file, is_frames_dir, awareness_parse_file, sensor_config, args.data_dir)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')