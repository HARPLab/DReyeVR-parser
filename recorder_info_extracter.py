import cv2
import numpy as np

#file_path = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/exp_nik-pilot.txt"
def get_data_dict(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    recording_info_dict = {}
    frame_num = 0
    prev_line = ""
    vehicleIds = []
    for l in range(len(lines)):
        line = lines[l]
        if "Frame " in line:
            subStr = line[6:]
            splitstr = subStr.split(" ")
            frame_num = int(splitstr[0])
            frame_dict = {}
        if "Create " and "dreyevr_vehicle" in line:
            splitstr = line.split(" ")
            vehicle_id = int(splitstr[2][:-1])
            vehicleIds.append(vehicle_id)
        if "Positions" in prev_line:
            splitstr = prev_line.split(" ")
            numPos = int(splitstr[2].strip())
            positions = {}
            for i in range(numPos):
                nLine = lines[l+i]
                nlinesplit = nLine.split(" ")
                idNum = int(nlinesplit[3])
                if idNum in vehicleIds:
                    location_x = float(nlinesplit[5][1:-1])
                    location_y = float(nlinesplit[6][:-1])
                    location_z = float(nlinesplit[7][:-1])
                    rotation = nlinesplit[9:]
                    positions[idNum] = (location_x, location_y, location_z)
            frame_dict["VehicleDict"] = positions
        if "FocusInfo" in line:
            splitstr = line.split(",")
            focus_dict = {}
            for i in splitstr:
                if "HitPoint" in i:
                    coordString = i.split(":")[1]
                    coords = coordString.split(" ")
                    location_x = float(coords[0].split("=")[1])
                    location_y = float(coords[1].split("=")[1])
                    location_z = float(coords[2].split("=")[1])
                    gaze_coords = (location_x, location_y, location_z)
                    focus_dict["HitPoint"] = gaze_coords
            frame_dict["FocusInfo"] = focus_dict
        if "EgoVariables" in line:
            splitstr = line.split(",")
            ego_dict = {}
            for i in splitstr:
                if "VehicleLoc" in i:
                    coordString = i.split(":")[2]
                    coords = coordString.split(" ")
                    location_x = float(coords[0].split("=")[1])
                    location_y = float(coords[1].split("=")[1])
                    location_z = float(coords[2].split("=")[1])
                    vehicle_coords = np.asarray([location_x, location_y, location_z])
                    ego_dict["VehicleLoc"] = vehicle_coords
                if "VehicleRot" in i:
                    coordString = i.split(":")[1]
                    coords = coordString.split(" ")
                    pitch = float(coords[0].split("=")[1])
                    yaw = float(coords[1].split("=")[1])
                    roll = float(coords[2].split("=")[1])
                    vehicle_rotation = np.asarray([pitch, yaw, roll])
                    ego_dict["VehicleRot"] = vehicle_rotation
            frame_dict["EgoVariables"] = ego_dict           
            recording_info_dict[frame_num] = frame_dict
        prev_line = line
    print("Recording Info Data Dictionary Created.")
    return recording_info_dict

#info_dict = get_data_dict(file_path)
#print(info_dict[1])
#Map vehicle position and ID on each frame
#for frame_num in vehicle_pos:
# info_tuple = vehicle_pos[521]
# id = info_tuple[0]
# red = [0, 0, 255]
# ##get frame image - frame_img
# fileName = 'results/%.6d' % (521)
# frame_img = cv2.imread(fileName)
# coord_tuple = info_tuple[1:3]
# frame_img[coord_tuple] = red
# #cv2.imwrite('results/labeled/%.6d.png' % frame_num, frame_img)
# cv2.imshow("image", frame_img)
# cv2.waitKey(0) 

"""
def format_number(num):
    res = str(num)
    while len(res) < 6:
        res = '0' + res
    return res

#Frome Awareness Image Parser run
all_vehicleId_pixels = {}
for frame_num in vehicle_pos:
    vehicle_list = vehicle_pos[frame_num]
    #get frame instance segmentation image
    frame_str = format_number(frame_num + 1)
    frame_img = cv2.imread("results/"+frame_str)

    vehicleId_pixels_dict = {}
    for vehicle in vehicle_list:
        v_id = vehicle[0]
        color_r = v_id/255.0
        color_g = ((v_id & 0x00ff) >> 0) / 255.0
        color_b = ((v_id & 0x00ff) >> 8) / 255.0
        color_vector = [color_r, color_g, color_b]
        pixels_list = []
        #psuedocode!!
        for i in range(frame_img.shape[0]):
            for j in range(frame_img.shape[1]):
                pixel = frame_img[i, j]
                print(pixel)
                if pixel == color_vector:
                    pixels_list.append((i, j))
        vehicleId_pixels_dict[v_id] = pixels_list
""" 




