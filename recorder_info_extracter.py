import numpy as np

file_path = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/exp_ines_51.txt"
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
        if "AwarenessData" in line:
            string = line.split("AwarenessData:")[1]
            parts = string.split(",")
            visible_total = parts[0].split(":")[1]
            visible_data_str = ",".join(parts[1:])
            visible_data_str = visible_data_str.split("Visible:{")[1]
            visible_data = visible_data_str.split("},")[:-1]
            visible_objects = {}
            for obj in visible_data:
                vel_loc_dict = {}
                id = 0
                for item in obj.split(","):
                    if item == "":
                        continue
                    key, value = item.split(":")
                    if key == "{Id":
                        id = value
                    if key in ["Location", "Velocity"]:
                        coords = value.split(" ")
                        x = float(coords[0].split("=")[1])
                        y = float(coords[1].split("=")[1])
                        z = float(coords[2].split("=")[1])
                        values = {"x":x, "y":y, "z":z}
                        vel_loc_dict[key] = values
                    if key == "Answer":
                        vel_loc_dict[key] = value
                visible_objects[id] = vel_loc_dict
            aw_dict = {"VisibleTotal": visible_total,
                    "Visible": visible_objects,
                }
            frame_dict["AwarenessData"] = aw_dict
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
                    gaze_coords = np.asarray([location_x, location_y, location_z])
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

# info_dict = get_data_dict(file_path)
# print(info_dict[1879])





