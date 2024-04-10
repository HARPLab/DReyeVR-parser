import json
import numpy as np
import argparse
import os
import errno
import cv2
import pandas as pd
import subprocess
import signal
import re
from PIL import Image
import matplotlib as mpl
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)    
# log errors
# matplotlib


num2button = {
    1: 'Ped Forward',
    2: 'Ped Right',
    4: 'Ped Back',
    8: 'Ped Left',
    17: 'Veh Forward',
    18: 'Veh Right',
    20: 'Veh Back',
    24: 'Veh Left'
}


def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        f.close()
    return data

def get_objectid_offset(frames_dir):
    # line = '7'
    with open(os.path.join(frames_dir, 'offset.txt'), 'r') as f:
        line = f.readline().strip()
        f.close()
    # reads offset from the offset txt file placed inside the frames_dir
    return int(line)

def remove_common_visible(iv_vis, ip_vis, df_vis):
    fin_v_vis = []
    fin_p_vis = []
    for v in iv_vis:
        if v not in df_vis:
            fin_v_vis.append(v)
    
    for v in ip_vis:
        if v not in df_vis:
            fin_p_vis.append(v) 
    return fin_v_vis, fin_p_vis

def get_id_from_image(img, id_offset):
    u = img.reshape(-1, img.shape[-1])
    # vehicles have R = 10, pedestrians have R = 4 and 2 wheelers have R = 23 in the instance segmentation 
    # ATLEAST IN THE CURRENT VERSION OF THE INSTANCE SEGMENTATION
    # 2 Wheelers, person is 4 and object is 10
    v_mid = np.unique(u[u[:, 2] == 10], axis=0)
    p_mid = np.unique(u[u[:, 2] == 4], axis=0)
    w2_mid = np.unique(u[u[:, 2] == 23], axis=0)
    
    v_ids = v_mid[:, 0]*256 + v_mid[:, 1] + id_offset
    p_ids = p_mid[:, 0]*256 + p_mid[:, 1] + id_offset
    w2_ids = w2_mid[:, 0]*256 + w2_mid[:, 1] + id_offset
    
    # final_vids = []
    # for ids in v_ids:
    #     if int(ids) not in p_ids:
    #         final_vids.append(int(ids))
    p_ids = np.concatenate([p_ids, w2_ids]).astype(int)
    return v_ids, p_ids

def open_images(frames_dir, t):
    frame_delay = 2
    id_offset = get_objectid_offset(frames_dir)
    i = 0
    success = False
    while success == False:
        if os.path.exists(os.path.join(frames_dir, 'instance_segmentation_output', '{:06d}.png'.format(t+frame_delay + i))):
            success = True
        else:
            i += 1
        if i == 6:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.path.join(frames_dir, 'instance_segmentation_output', '{:06d}.png'.format(t+frame_delay)))
    
    
    im_mid = cv2.imread(os.path.join(frames_dir, 'instance_segmentation_output', '{:06d}.png'.format(t+frame_delay+i)))
    im_left = cv2.imread(os.path.join(frames_dir, 'instance_segmentation_output_left', '{:06d}.png'.format(t+frame_delay+i)))
    im_right = cv2.imread(os.path.join(frames_dir, 'instance_segmentation_output_right', '{:06d}.png'.format(t+frame_delay+i)))
    return im_left, im_mid, im_right, id_offset

def get_visible_vehicles(frames_dir, t):
    
    im_left, im_mid, im_right, id_offset = open_images(frames_dir, t)

    v_mid_ids, p_mid_ids = get_id_from_image(im_mid, id_offset)
    v_left_ids, p_left_ids = get_id_from_image(im_left, id_offset)
    v_right_ids, p_right_ids = get_id_from_image(im_right, id_offset)

    v_ids = np.unique(np.concatenate([v_mid_ids, v_left_ids, v_right_ids])).astype(int) # default data type for numpy arrays is float
    p_ids = np.unique(np.concatenate([p_mid_ids, p_left_ids, p_right_ids]))
    return v_ids, p_ids
    

def tag_object(userinput, vq, pq, vehicle_dict, is_frames_dir, rgb_frames_dir, t, row):
    # typ = userinput//16
    # if typ == 1:
    #     obj, error = vq.get_object(userinput, vq, pq, vehicle_dict, is_frames_dir, rgb_frames_dir, t)
    # else:
    #     obj, error = pq.get_object(userinput, vq, pq, vehicle_dict, is_frames_dir, rgb_frames_dir, t)
    obj, error = get_object(userinput, vq, pq, vehicle_dict, is_frames_dir, rgb_frames_dir, t, row)

    if obj == None:
        return error, None, vehicle_dict
    else:
        if obj.id in vehicle_dict:
            # vehicle_dict[obj.id].append(obj.curr_time)
            vehicle_dict[obj.id] = obj.curr_time
            return 'A3', obj, vehicle_dict
        else:
            vehicle_dict[obj.id] = obj.curr_time
            return error, obj, vehicle_dict

def correct_for_2_wheelers(vq, pq):
    # Lets hope the user does not click for the 2 wheeler when only the bike is visible
    delete_objs = []
    for obj in vq.object_dict:
        if obj in pq.object_dict:
            obj_obj = vq.object_dict[obj]
            qs = obj_obj.qs
            for q in qs:
                q.remove(obj_obj)
            delete_objs.append(obj)
    
    for obj in delete_objs:
        del vq.object_dict[obj]

def repeat_click_check(v, t, invisible_buffer, vehicle_dict, tagged_objs):
    #
    #invisible_buffer : the gap between the last visible frame and the current frame to be considered invisible
    if v not in vehicle_dict:
        return True
    else:
        if v in tagged_objs:
            obj = tagged_objs[v]
            t_clicked = obj.curr_time
            last_visible_time = obj.last_visible
            when_invisible = -1
            if t - last_visible_time > invisible_buffer:
                when_invisible = last_visible_time

            if v == 1149:
                obj.print_obj()
                print(when_invisible, t, last_visible_time, t_clicked)

            if when_invisible - t_clicked > invisible_buffer:
                return True
            else:
                return False
        return False


class object_data():
    def __init__(self, id, dist, time, l, answer, q):
        # q is a list of qs

        self.id = id
        self.dist = dist
        self.time_entered = time
        self.curr_time = time
        self.l1 = l
        self.compute_score()
        self.q_update_due_in = -1
        self.q_update_delay = 10
        self.qs = q
        self.curr_answer = answer    
        self.last_visible = time   
        self.status = 1
        
    def compute_score(self):
        # might need to scale the distance and time_entered by some value
        self.score = -(self.dist/50 + self.l1*(self.curr_time - self.time_entered))

    def update_object(self, dist, time):
        if self.id == 1149:
            print("update score", time, dist)
        self.dist = dist
        self.curr_time = time
        self.last_visible = time
        self.compute_score()
    
    def print_obj(self):
        print(self.id, self.curr_answer, self.curr_time, self.time_entered, self.score, self.q_update_due_in, self.last_visible, self.status)

    def update_object_qs(self, answer, qs):
        
        if self.curr_answer != answer:
            if self.q_update_due_in == 0:
                self.q_update_due_in = -1
                self.curr_answer = answer
                for q in self.qs:
                    q.remove(self)

                self.qs = qs
                for q in qs:
                    q.append(self)
            elif self.q_update_due_in == -1:
                self.q_update_due_in = self.q_update_delay
            else:
                self.q_update_due_in -= 1
        else:
            self.q_update_due_in = -1

def print_id_pixel_location_from_image(img, id_offset, image_typ):
    img_out = np.copy(img)
    u = img_out.reshape(-1, img.shape[-1])
    v_mid = np.unique(u[u[:, 2] == 10], axis=0)
    p_mid = np.unique(u[u[:, 2] == 4], axis=0)
    w2_mid = np.unique(u[u[:, 2] == 23], axis=0)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color in BGR
    line_thickness = 2
    text_position_offset = [0, 40]
    obj_ctr = 0
    # colors = mpl.cm.get_cmap('Set1', len(v_mid) + len(p_mid) + len(w2_mid))
    colors = mpl.cm.get_cmap('Set1', len(v_mid) + len(p_mid))

    # Put the text on the image
    for v in v_mid:
        obj_ctr += 1
        font_color = tuple((np.array(colors(obj_ctr)[:3])*255).tolist())
        indices = np.where(np.all(u == v, axis=(1)))
        id = v[0]*256 + v[1] + id_offset
        img[np.all(img == v, axis=-1)] = font_color
        print("Vehicle", id, image_typ, indices[0][0]//img.shape[1], indices[0][0]%img.shape[1])
        position = (text_position_offset[0] + indices[0][0]%img.shape[1], text_position_offset[1] + indices[0][0]//img.shape[1])
        text_position_offset[1] += 20 
        cv2.putText(img, str(id), position, font, font_scale, font_color, line_thickness)
        

    for v in p_mid:
        obj_ctr += 1
        font_color = tuple((np.array(colors(obj_ctr)[:3])*255).tolist())
        indices = np.where(np.all(u == v, axis=(1)))
        img[np.all(img == v, axis=-1)] = font_color
        id = v[0]*256 + v[1] + id_offset
        print("Pedestrian", id, image_typ, indices[0][0]//img.shape[1], indices[0][0]%img.shape[1])
        position = (text_position_offset[0] + indices[0][0]%img.shape[1], text_position_offset[1] + indices[0][0]//img.shape[1])
        text_position_offset[1] += 20 
        cv2.putText(img, str(id), position, font, font_scale, font_color, line_thickness)
        
    # for v in w2_mid:
    #     indices = np.where(np.all(u == v, axis=(1)))
    #     id = v[0]*256 + v[1] + id_offset
    #     print("2 Wheeler", id, image_typ, indices[0][0]//img.shape[1], indices[0][0]%img.shape[1])
    #     position = (text_position_offset[0] + indices[0][0]%img.shape[1], text_position_offset[1] + indices[0][0]//img.shape[1])
    #     cv2.putText(img, str(id), position, font, font_scale, font_color, line_thickness)
    print('\n')
    
    return img

def open_image_with_xdg(frames_dir, t, image_type):
    frame_delay = 2
    j = 1
    i = 0
    extension = ""
    if 'is' in image_type:
        dir_names = ['instance_segmentation_output', 'instance_segmentation_output_left', 'instance_segmentation_output_right']
        extension = ".png"
        search_len = 6
    else:
        dir_names = ['left', 'mid', 'right']
        extension = ".jpg"
        t = max(0, t - 20)
        search_len = 50
        
        
    success = False
    while success == False:
        
        if os.path.exists(os.path.join(frames_dir, dir_names[0], '{:06d}{}'.format(t+frame_delay + i, extension))):
            success = True
        elif os.path.exists(os.path.join(frames_dir, dir_names[0], '{:06d}{}'.format(t+frame_delay - i, extension))):
            success = True
            j = -1        
    # find the nearest subsequent image that exists
        else:
            i += 1
        if i == search_len:
            print("File Not Found", os.path.join(frames_dir, dir_names[0], '{:06d}{}'.format(t+frame_delay + i, extension)))
            break
            # raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.path.join(frames_dir, dir_names[0], '{}.png'.format(t+frame_delay)))
    
    

    try:
        p1 = subprocess.Popen(['eog', os.path.join(frames_dir, dir_names[0], '{:06d}{}'.format(t+frame_delay+i*j, extension))])
    except FileNotFoundError:
        print("Error: eog command not found.")
    
    try:
        p2 = subprocess.Popen(['eog', os.path.join(frames_dir, dir_names[1], '{:06d}{}'.format(t+frame_delay+i*j, extension))])
    except FileNotFoundError:
        print("Error: eog command not found.")
    
    try:
        p3 = subprocess.Popen(['eog', os.path.join(frames_dir, dir_names[2], '{:06d}{}'.format(t+frame_delay+i*j, extension))])
    except FileNotFoundError:
        print("Error: eog command not found.")
    
    return [p1, p2, p3], t+frame_delay+i*j


def correction(is_frames_dir, rgb_frames_dir, t, pq, vq, vehicle_dict, row):
    # read visible from is_frames
    im_left, im_mid, im_right, id_offset = open_images(is_frames_dir, t)

    # window = tk.Tk()
    # window.title("Image Display")
    
    # # Convert the PIL image to Tkinter PhotoImage
    # image_tk = ImageTk.PhotoImage(image_pil)

    # # Display the image on a Tkinter label
    # label = tk.Label(window, image=image_tk)
    # label.pack()
    
    # close_button = tk.Button(window, text="Close", command=close_window)
    # close_button.pack()

    # open the 3 is frames
    is_processes, f_index = open_image_with_xdg(is_frames_dir, t, 'is')
    
    # open the 3 rgb frames
    rgb_processes, _ = open_image_with_xdg(rgb_frames_dir, t, 'rgb')    
    
    print("Frame Index:", f_index)
    # print clickable object's data from queues
    v_mid_ids, p_mid_ids = get_id_from_image(im_mid, id_offset)
    v_left_ids, p_left_ids = get_id_from_image(im_left, id_offset)
    v_right_ids, p_right_ids = get_id_from_image(im_right, id_offset)

    v_ids = np.unique(np.concatenate([v_mid_ids, v_left_ids, v_right_ids])).astype(int) # default data type for numpy arrays is float
    p_ids = np.unique(np.concatenate([p_mid_ids, p_left_ids, p_right_ids]))
    candidate_ids = []
    for obj_id in vq.object_dict:
        obj = vq.object_dict[obj_id]
        v = obj.id
        clickable = repeat_click_check(v, t, vq.invisible_buffer, vehicle_dict, vq.tagged_objs) # Allowing repeated clicks after some delay
        if clickable:
            if obj != None and str(obj.id) in row['Actors_Location']:
                obj.print_obj()
                candidate_ids.append(int(obj.id))
            
    for obj_id in pq.object_dict:
        obj = pq.object_dict[obj_id]
        v = obj.id
        clickable = repeat_click_check(v, t, pq.invisible_buffer, vehicle_dict, pq.tagged_objs) # Allowing repeated clicks after some delay
        if clickable:
            if obj != None and str(obj.id) in row['Actors_Location']:
                obj.print_obj()
                candidate_ids.append(int(obj.id))

            
    # create frame with ids overlayed
    text_img_left = print_id_pixel_location_from_image(im_left, id_offset, 'left')
    text_img_mid = print_id_pixel_location_from_image(im_mid, id_offset, 'mid')
    text_img_right = print_id_pixel_location_from_image(im_right, id_offset, 'right')

    
    stacked_image = np.hstack((text_img_left, text_img_mid, text_img_right))
    image_pil = Image.fromarray(stacked_image[:,:,::-1])
    image_pil.show(title=str(f_index) + '.png')

    im_left, im_mid, im_right, id_offset = open_images(is_frames_dir, t-10)
    text_img_left = print_id_pixel_location_from_image(im_left, id_offset, 'left')
    text_img_mid = print_id_pixel_location_from_image(im_mid, id_offset, 'mid')
    text_img_right = print_id_pixel_location_from_image(im_right, id_offset, 'right')

    
    stacked_image = np.hstack((text_img_left, text_img_mid, text_img_right))
    image_pil = Image.fromarray(stacked_image[:,:,::-1])
    image_pil.show(title=str(f_index-10) + '.png')

    im_left, im_mid, im_right, id_offset = open_images(is_frames_dir, t+10)
    text_img_left = print_id_pixel_location_from_image(im_left, id_offset, 'left')
    text_img_mid = print_id_pixel_location_from_image(im_mid, id_offset, 'mid')
    text_img_right = print_id_pixel_location_from_image(im_right, id_offset, 'right')

    
    stacked_image = np.hstack((text_img_left, text_img_mid, text_img_right))
    image_pil = Image.fromarray(stacked_image[:,:,::-1])
    image_pil.show(title=str(f_index+10) + '.png')



    # read userinput from prompt (until valid input given)
    while True:
        try:
            correct_id = input('Correct Object Id (If it is an repeat/invalid button click type Inv:error message or Rep:error message):').strip()
                        
            # Type error messages, Invalid:         
            if re.match('^[0-9]*$', correct_id) == None:
                for p in is_processes + rgb_processes:
                    print("Process ID:", p.pid)
                    os.kill(p.pid, signal.SIGTERM)
                # window.destroy()       
                return None, correct_id
            if int(correct_id) in candidate_ids:
                if int(correct_id) in vq.object_dict:
                    # shutdown the images
                    for p in is_processes + rgb_processes:
                        print("Process ID:", p.pid)
                        os.kill(p.pid, signal.SIGTERM)     
                    # window.destroy()
                    
                    
                    with open(os.path.join(os.path.dirname(is_frames_dir), 'user_corrections.txt'), 'w+') as f:
                        f.write(str(t) + ' : ' + str(correct_id))
                        f.close()
                    return vq.find_object(int(correct_id)), None
                
                if int(correct_id) in pq.object_dict:
                    for p in is_processes + rgb_processes:
                        print("Process ID:", p.pid)
                        os.kill(p.pid, signal.SIGTERM)    
                    # window.destroy()       
                        
                    with open(os.path.join(os.path.dirname(is_frames_dir), 'user_corrections.txt'), 'w+') as f:
                        f.write(str(t) + ' : ' + str(correct_id))
                        f.close()

                    return pq.find_object(int(correct_id)), None
            else:
                print("Id is not in the list of clickable objects")
        except Exception as e:
            print(e, "Try again")
        
    # close images
    
    
    
    # return userid
    
def get_object(userinput, vq, pq, vehicle_dict, is_frames_dir, rgb_frames_dir, t, row):
    if userinput//16 == 0:
        q = pq.get_queue_from_answer(userinput)
    else:
        q = vq.get_queue_from_answer(userinput)
        
    error = None
    if len(q[0]) == 0:
        # trigger correction def correction(is_frames_dir, rgb_frames_dir, t, pq, vq, vehicle_dict): 
        print("USERINPUT:", num2button[userinput])
        output, er = correction(is_frames_dir, rgb_frames_dir, t, pq, vq, vehicle_dict, row)
        if output is None:
            return None, er
        max_obj = output
        error = 'A1'
    else:
        if len(q[0]) > 1:
            error = 'A2'
        max_score = -np.inf
        max_obj = None
        for obj in q[0]:
            score = obj.score
            if score>max_score:
                max_score = score
                max_obj = obj
        
        if error == 'A2':
            print("USERINPUT:", num2button[userinput])
            output, er = correction(is_frames_dir, rgb_frames_dir, t, pq, vq, vehicle_dict, row)
            if output is None:
                return None, er

            max_obj = output
            # trigger correction
        
    qs = max_obj.qs
    for q in qs:
        q.remove(max_obj)

    max_obj.status = 0
    if max_obj.id in pq.object_dict:
        del pq.object_dict[max_obj.id]
        pq.tagged_objs[max_obj.id] = max_obj

    if max_obj.id in vq.object_dict:
        del vq.object_dict[max_obj.id]
        vq.tagged_objs[max_obj.id] = max_obj
    
    return max_obj, error

class object_queues():
    def __init__(self, typ, l):
        
        self.typ = typ # 1 for vehicles 0 for pedestrians
        self.l = l # time weighting 
        # self.delay_repeat = 100*40 # time after which repeat clicks are allowed 100 seconds ~40 fps
        self.invisible_buffer = 10 # the gap between the last visible frame and the current frame to be considered invisible

        self.forward_angle_limit = 30
        self.lr_angle_limit = 60
        self.back_angle_limit = 120
        
        self.left_queue = []
        self.right_queue = []
        self.forward_queue = []
        self.back_queue = []

        self.object_dict = {}
        self.tagged_objs = {}


        
    def get_queue_from_answer(self, userinput):
        q = []
        if userinput & 1 != 0:
            q.append(self.forward_queue)
        if userinput & 2 != 0:
            q.append(self.right_queue)
        if userinput & 4 != 0:
            q.append(self.back_queue)
        if userinput & 8 != 0:
            q.append(self.left_queue)
        return q
    

    def find_object(self, o_id):
        if o_id in self.object_dict:
            return self.object_dict[o_id] 
        else:
            if o_id in self.tagged_objs:
                obj = self.tagged_objs[o_id]
                return obj
            
            return None


    def update_queue(self, row, t, vehicle_dict, is_frames_dir, visible_is_dict):
        if int(t) in visible_is_dict:
            iv_vis, ip_vis = visible_is_dict[int(t)]
        else:
            iv_vis, ip_vis = get_visible_vehicles(is_frames_dir, t)
        visible_from_IS = list(iv_vis) + list(ip_vis)
        # TODO: this is what to put in the dataframe
        visible = row['AwarenessData_Visible']
        
        v_locs = row['AwarenessData_VisibleLocation']
        v_typs = []
        # get types for objects already in visible
        for v_i in range(len(visible)):
            if visible[v_i] in iv_vis:
                v_typs.append(1)
            elif visible[v_i] in ip_vis:
                v_typs.append(0)
            else:
                v_typs.append(row['AwarenessData_Answer'][v_i]//16)
        
        # append to visible if in inst_seg but not in visible from df
        for v_i in range(len(iv_vis)):
            if iv_vis[v_i] not in visible:
                if str(iv_vis[v_i]) in row['Actors_Location']:
                    visible.append(iv_vis[v_i])
                    v_locs.append(row['Actors_Location'][str(iv_vis[v_i])])
                    v_typs.append(1)
        
        # same as above but for peds
        for v_i in range(len(ip_vis)):
            if ip_vis[v_i] not in visible:
                if str(ip_vis[v_i]) in row['Actors_Location']:
                    visible.append(ip_vis[v_i])
                    v_locs.append(row['Actors_Location'][str(ip_vis[v_i])])
                    v_typs.append(0)
        
        # if t > 11151:
        answers, angles = self.obtain_answers(visible, v_locs, v_typs, row['EgoVariables_VehicleLoc'], row['EgoVariables_VehicleRot'][1])
        print(t, visible, v_locs, angles, row['EgoVariables_VehicleLoc'], row['EgoVariables_VehicleRot'][1])

        
        # remove unclicked invisible objects in queues
        keys_to_delete = []
        for obj_id in self.object_dict.keys():
            obj = self.object_dict[obj_id]
            if obj.id not in visible:
                # print("Object not in visible has to be deleted,")
                # obj.print_obj()
                # print(obj.dist)
                
                if obj.q_update_due_in == -1:
                    obj.q_update_due_in = obj.q_update_delay
                elif obj.q_update_due_in == 0:
                    qs = obj.qs
                    print(obj.id, qs, self.object_dict)    
                    for q in qs:
                        q.remove(obj)
                        
                    keys_to_delete.append(obj.id)
                    
                    obj.status = 0
                    
                else:
                    obj.q_update_due_in -= 1
                    
        for key in keys_to_delete:
            del self.object_dict[key]                

        # update queues with clickable visible objects
        ego_loc = np.array(row['EgoVariables_VehicleLoc'])[:2]
        for v_i, v in enumerate(visible):
            if answers[v_i] // 16 == self.typ:
                update_q = repeat_click_check(v, t, self.invisible_buffer, vehicle_dict, self.tagged_objs) # Allowing repeated clicks after some delay
                if update_q: 
                # find if answer exists in the queue
                    obj_v = self.find_object(v)
                    v_loc = np.array(row['AwarenessData_VisibleLocation'][v_i])[:2]
                    dist = np.linalg.norm(ego_loc - v_loc)    
                    qs = self.get_queue_from_answer(answers[v_i])
                    if obj_v == None:
                        obj_v = object_data(v, dist, t, self.l, answers[v_i], qs)
                        self.object_dict[obj_v.id] = obj_v 
                        for q in qs:                        
                            q.append(obj_v)
                    else:
                        if obj_v.status == 0:
                            obj_v.time_entered = t
                            obj_v.q_update_due_in = -1
                            obj_v.qs = qs
                            for q in qs:
                                q.append(obj_v)
                            obj_v.curr_answer = answers[v_i]
                            obj_v.status = 1
                            self.object_dict[obj_v.id] = obj_v

                        obj_v.update_object(dist, t)
                        #change queues
                        obj_v.update_object_qs(answers[v_i], qs)
        
        # update the last visible time of tagged objects
        for obj_id in self.tagged_objs.keys():
            obj = self.tagged_objs[obj_id]
            if obj_id in visible:
                obj.last_visible = t
            else:
                if obj_id in vehicle_dict:
                     del vehicle_dict[obj_id]
        
        return visible_from_IS, visible

 
    def obtain_answers(self, visible_vehicles, v_locs, v_typs, ego_loc, ego_yaw):
        answers = []
        angles = []
        
        ego_vehicle_pos = np.array(ego_loc)[:2]
        ego_vehicle_yaw = ego_yaw
        ego_rot_matrix = np.array([[np.cos(np.radians(ego_vehicle_yaw)), -np.sin(np.radians(ego_vehicle_yaw))], 
                                [np.sin(np.radians(ego_vehicle_yaw)), np.cos(np.radians(ego_vehicle_yaw))]])
        driver_pos = ego_vehicle_pos +  ego_rot_matrix@np.array([20, -40]) # position of the driver inside the vehicle
        
        # visible_vehicles = row['AwarenessData_Visible']
        
        for v in range(len(visible_vehicles)):
            v_id = visible_vehicles[v]
            v_loc = np.array(v_locs[v])[:2]
            v_typ = v_typs[v]
            v_driver_fov = np.linalg.pinv(ego_rot_matrix)@(v_loc-driver_pos)
            angle = np.degrees(np.arctan2(v_driver_fov[1], v_driver_fov[0]))
            if angle > self.back_angle_limit or angle < -self.back_angle_limit: # back
                answer = 4 + (v_typ == 1)*16
            else:
                if angle < self.forward_angle_limit and angle > -self.forward_angle_limit: # front
                    answer = 1 + (v_typ == 1)*16
                elif angle > self.lr_angle_limit: # right
                    answer = 2 + (v_typ == 1)*16
                elif angle < -self.lr_angle_limit: #left
                    answer = 8 + (v_typ == 1)*16
                else: 
                    if angle > 0:
                        answer = 3 + (v_typ == 1)*16
                    else:
                        answer = 9 + (v_typ == 1)*16
            
            answers.append(answer)
            angles.append(angle)
            
        return answers, angles
    
    def print_q(self, q):
        for obj in q:
            obj.print_obj()
    
    def print_queues(self):
        print("Left")
        self.print_q(self.left_queue)
        print("Forward")
        self.print_q(self.forward_queue)
        print("Right")
        self.print_q(self.right_queue)
        print("Back")
        self.print_q(self.back_queue)

class error_stats():
    def __init__(self):
        self.count_tot = 0
        self.count_amb_none = 0
        self.count_amb_more = 0
        self.count_repeat = 0

    def update(self, error_typ):
        if error_typ == 'A1':
            self.count_amb_none += 1
        if error_typ == 'A2':
            self.count_amb_more += 1
        if error_typ == 'A3':
            self.count_repeat += 1
        self.count_tot += 1
    
    def print_stats(self):
        print("Total Button Presses:", self.count_tot)
        print("No corresponding objects:", self.count_amb_none)
        print("More than one corresponding objects:", self.count_amb_more)
        print("Repeat Button Presses:", self.count_repeat)

def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)

    argparser.add_argument(
        '-data_dir', '--data-dir',
        help = "path to the participant data directory"
    )

    argparser.add_argument(
        '-rgbf', '--rgbframes-dir',
        help = "path to the rgb frames"
    )
    
    argparser.add_argument(
        '-isf', '--isframes-dir',
        help = "path to the instance segmentation frames"
    )
    
    
    argparser.add_argument(
        '-aw', '--awareness-data',
        help = "awareness data frame in json form"
    )

    argparser.add_argument(
        '-l', '--time-weight',
        help = "time weight for the object queues",
        default=1
    )

    argparser.add_argument(
        '-d', '--debug',
        action = 'store_true',
        help = "debug mode to print object queues and error stats"
    )
    args = argparser.parse_args()

    if args.data_dir is None:
        is_frames_dir = args.isframes_dir        
        rgb_frames_dir = args.rgbframes_dir        
        awareness_parse_file = args.awareness_data
        visible_is_dict = np.load(os.path.join(args.awareness_data, 'visible_is.npy'), allow_pickle=True)[0]
        temp_visible_is_dict = {}
        for f_i, f in enumerate(visible_is_dict['frame_no']):
            temp_visible_is_dict[int(visible_is_dict['frame_no'][f_i])] = [visible_is_dict['iv_vis'][f_i], visible_is_dict['ip_vis'][f_i]]

    else:
        is_frames_dir = os.path.join(args.data_dir, 'images')
        rgb_frames_dir = os.path.join(args.data_dir, 'gaze_button_overlay')        
        awareness_parse_file = os.path.join(args.data_dir, 'rec_parse-awdata.json') 
        visible_is_dict = np.load(os.path.join(args.data_dir, 'visible_is.npy'), allow_pickle=True)[0]
        temp_visible_is_dict = {}
        for f_i, f in enumerate(visible_is_dict['frame_no']):
            temp_visible_is_dict[int(visible_is_dict['frame_no'][f_i])] = [visible_is_dict['iv_vis'][f_i], visible_is_dict['ip_vis'][f_i]]
    
    data = load_json(awareness_parse_file)
    es = error_stats()
    
    dense_label_df_dict = {'frame_no':[], 'visible_is':[], 'visible_total':[], 'awareness_label':[], 'error_msg':[]}
    
    
    vehicle_dict = {}
    final_vehicle_dict = {}
    vq = object_queues(1, float(args.time_weight))
    pq = object_queues(0, float(args.time_weight))
    listk = list(data.keys())
    prev_input = data[listk[0]]['AwarenessData_UserInput']
    max_consecutive_missing_frames = 6
    for i_k, k in enumerate(listk[1:-max_consecutive_missing_frames]):
        error_msg = None
    # for i_k, k in enumerate(listk[1:10]):
        _, _ = vq.update_queue(data[k], int(k), vehicle_dict, is_frames_dir, temp_visible_is_dict) # each vehicle in each direction
        visible_is, visible_total = pq.update_queue(data[k], int(k), vehicle_dict, is_frames_dir, temp_visible_is_dict) # each pedestrian in each direction
        correct_for_2_wheelers(vq, pq)
        # if int(k) >= 11276:
        print(k)
        print("Vehicles")
        vq.print_queues()
        print(vq.object_dict)
        print("Pedestrian")
        pq.print_queues()
        print(pq.object_dict)
        print('\n')
        print(vq.tagged_objs.keys())
        print(pq.tagged_objs.keys())
        print('\n')
        print(vehicle_dict)
        print('\n')

        userinput = data[k]['AwarenessData_UserInput']
        next_userinput = data[listk[i_k + 1]]['AwarenessData_UserInput']
        if (userinput != prev_input) and (userinput != 0):
            print(userinput, next_userinput)
        #  and (userinput!=next_userinput): #detect button click; remove instance where button clicks lasts only one frame
            error_typ, obj, vehicle_dict = tag_object(userinput, vq, pq, vehicle_dict, is_frames_dir, rgb_frames_dir, int(k), data[k])
            for v in vehicle_dict:
                if v not in final_vehicle_dict:
                    final_vehicle_dict[v] = [vehicle_dict[v]]
                else:
                    if vehicle_dict[v] != final_vehicle_dict[v][-1]:
                        final_vehicle_dict[v].append(vehicle_dict[v])
            es.update(error_typ)
            if args.debug:
                print("BUTTON PRESSES", k, error_typ, userinput, obj, final_vehicle_dict, vehicle_dict)
                vq.print_queues()
                pq.print_queues()
                es.print_stats()
                print('\n')
            error_msg = error_typ
        prev_input = userinput
        awareness_labels = []
        for obj_id in visible_total:
            if obj_id in vehicle_dict:
                awareness_labels.append(1)
            else:
                awareness_labels.append(0)
        print(visible_is, visible_total, awareness_labels)
        dense_label_df_dict['frame_no'].append(k)
        dense_label_df_dict['awareness_label'].append(awareness_labels)
        dense_label_df_dict['visible_is'].append(visible_is)
        dense_label_df_dict['visible_total'].append(visible_total)
        dense_label_df_dict['error_msg'].append(error_msg)
    es.print_stats()
    
    
    dense_label_df = pd.DataFrame.from_dict(dense_label_df_dict)
    if args.data_dir is not None:
        dense_label_df_fname = 'corrected-awlabels.csv'
        dense_label_df.to_csv(os.path.join(args.data_dir, dense_label_df_fname))
    else:
        dense_label_df_fname = os.path.basename(args.awareness_data).split('.')[0].replace('awdata', 'awlabels')+'.csv'
        dense_label_df.to_csv(dense_label_df_fname)
                
                
if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')