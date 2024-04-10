import json
import numpy as np
import argparse
import os
import errno
import cv2
import pandas as pd

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


def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)

    argparser.add_argument(
        '-data_dir', '--data-dir',
        help = "path to the participant data directory"
    )
    
    argparser.add_argument(
        '-isf', '--isframes-dir',
        help = "path to the instance segmentation frames"
    )
        
    argparser.add_argument(
        '-aw', '--awareness-data',
        help = "awareness data frame in json form"
    )

    args = argparser.parse_args()

    if args.data_dir is None:
        is_frames_dir = args.isframes_dir
        awareness_parse_file = args.awareness_data
    else:
        is_frames_dir = os.path.join(args.data_dir, 'images')
        awareness_parse_file = os.path.join(args.data_dir, 'rec_parse-awdata.json') 
    
    data = load_json(awareness_parse_file)
    listk = list(data.keys())
    dense_label_df_dict = {'frame_no':[], 'visible_is':[], 'iv_vis':[], 'ip_vis':[]}
    
    for i_k, k in enumerate(listk):
   # for i_k, k in enumerate(listk[1:10]):
        try:
            iv_vis, ip_vis = get_visible_vehicles(is_frames_dir, int(k))
        except Exception as e:
            print(e)
            iv_vis = prev_iv_vis
            ip_vis = prev_ip_vis
        visible_is = list(iv_vis) + list(ip_vis)        
        dense_label_df_dict['frame_no'].append(k)
        dense_label_df_dict['visible_is'].append(visible_is)
        dense_label_df_dict['iv_vis'].append(iv_vis)
        dense_label_df_dict['ip_vis'].append(ip_vis)
        prev_iv_vis = iv_vis
        prev_ip_vis = ip_vis
        print(k, visible_is)
    
    
    dense_label_df = np.array([dense_label_df_dict])
    if args.data_dir is not None:
        dense_label_df_fname = os.path.join(args.data_dir,  'visible_is.npy')
        np.save(dense_label_df_fname, dense_label_df)
    else:
        dense_label_df_fname = os.path.basename(args.awareness_data).split('.')[0].replace('awdata', 'visible_is')+'.npy'
        np.save(dense_label_df_fname, dense_label_df)
                
                
if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
