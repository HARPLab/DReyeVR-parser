import cv2
import os

image_folder = 'gaze_history/exp_nik-pilot'
heat_image_folder = 'gaze_heatmap/exp_nik-pilot'
video1_name = 'exp_nik-pilot-video-gaze_history.avi'
video2_name = 'exp_nik-pilot-video-gaze_heatmap.avi'

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