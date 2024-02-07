import cv2
import os
import argparse


def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-history', '--gaze-history-folder',
        metavar='G',
        help='directory with gaze history images')
    argparser.add_argument(
        '-heatmap', '--gaze-heatmap-folder',
        metavar='H',
        help='directory with gaze heatmap images')
    argparser.add_argument(
        '-hist-video', '--gaze-history-output',
        metavar='GO',
        help='filename for gaze history video output')
    argparser.add_argument(
        '-heatmap-video', '--gaze-heatmap-output',
        metavar='HO',
        help='filename for gazeheatmap video output')
    
    args = argparser.parse_args()
    
    image_folder = args.gaze_history_folder
    heat_image_folder = args.gaze_heatmap_folder
    #video1_name = 'exp_nik-pilot-video-gaze_history.avi'
    #video2_name = 'exp_nik-pilot-video-gaze_heatmap.avi'
    video1_name = args.gaze_history_output
    video2_name = args.gaze_heatmap_output

    images = sorted(os.listdir(image_folder), key=lambda x: int(x.split(".")[0]))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video1_name, 0, 20, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

    images_heat = sorted(os.listdir(heat_image_folder), key=lambda x: int(x.split(".")[0]))
    frame_heat = cv2.imread(os.path.join(heat_image_folder, images[0]))
    height2, width2, layers2 = frame_heat.shape

    video2 = cv2.VideoWriter(video2_name, 0, 20, (width2,height2))

    for image2 in images_heat:
        video2.write(cv2.imread(os.path.join(heat_image_folder, image2)))

    cv2.destroyAllWindows()
    video2.release()
    

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')