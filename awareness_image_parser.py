from typing import Dict, Optional, List
from src.parser import parse_file
from src.utils import get_good_idxs, fill_gaps, smooth_arr, compute_YP, convert_to_df, flatten_dict
import numpy as np
import argparse
from src.visualizer import (
    set_results_dir,
)
import os
import sys
from PIL import Image

def format_number(num):
    res = str(num)
    while len(res) < 6:
        res = '0' + res
    return res
    



def main(filename: str, images_dir: str, results_dir: str, vlines: Optional[List[float]] = None):
    set_results_dir(results_dir)

    """parse the file"""
    data: Dict[str, np.ndarray or dict] = parse_file(filename)
    # can also use data["TimestampCarla"] which is in simulator time
    # Set helper data structures
    t: np.ndarray = data["TimeElapsed"]
    AwarenessData = data["AwarenessData"]
    VisibleTotal = AwarenessData["VisibleTotal"]
    UserInput = AwarenessData["UserInput"]
    FramesNum = len(VisibleTotal)
    if (FramesNum == 0):
        print("0 length recording, nothing to parse")
        return

    VehicleType = 16
    TypeBit = 16
    Type = []
    Ids = []
    Answer = []
    Loc = []
    Vel = []
    idx = 0
    VisibleTotal[-1] = 0
    for num in VisibleTotal: 
        Ids.append(AwarenessData["Id"][idx : idx + num])
        Answer.append(AwarenessData["Answer"][idx : idx + num])
        Loc.append(AwarenessData["Location"][idx : idx + num])
        Vel.append(AwarenessData["Velocity"][idx : idx + num])
        idx += num
    Type = []
    TypeDict = {}
    for i in range (FramesNum):
        Type.append([])
        for j in range(VisibleTotal[i]):
            if Answer[i][j] & VehicleType:
                Type[i].append("vehicle")
                TypeDict[Ids[i][j]] = "vehicle"
            else:
                Type[i].append("walker")
                TypeDict[Ids[i][j]] = "walker"
    
    Noticed : Dict = {}
    EverNoticed: Dict = {}
    FirstAppeared : Dict = {}
    AllVisible : np.ndarray = []
    WasInputCorrect: np.ndarray = []

    ChangeToNoticed = [set() for i in range(FramesNum)]
    ChangeToUnnoticed = [set() for i in range(FramesNum)]

    # Parse the data
    for i in range (FramesNum):
        ActorsNum = VisibleTotal[i]
        for Id in Ids[i]:
            if i == 0 or not(Id in Ids[i - 1]):
                FirstAppeared[Id] = i + 1
            if not (Id in Noticed):
                AllVisible.append(Id)
                EverNoticed[Id] = False
                Noticed[Id] = False

        if UserInput[i] > 0:
            FoundCorrect = False
            for j in range(ActorsNum):
                if not(Noticed[Ids[i][j]]) and (UserInput[i] & TypeBit == Answer[i][j] & TypeBit) and (UserInput[i] & Answer[i][j]):
                    Noticed[Ids[i][j]] = True
                    #ChangedToNoticed[i].add(Ids[i][j])
                    EverNoticed[Ids[i][j]] = True
                    FoundCorrect = True
                    break
            WasInputCorrect.append(FoundCorrect)
        for Id in AllVisible:
            if EverNoticed[Id]:
                ChangeToNoticed[i].add(Id)
        if i == 0:
            continue

        '''for Id in Ids[i - 1]:
            if not(Id in Ids[i]):
                Noticed[id] = False'''
    print()
    print("All actors that were visible at some point:")
    for Id in AllVisible:
        print(Id, TypeDict[Id], end = '\n')
    print()
    print("Was actor ever noticed?")
    for Id in AllVisible:
        print(Id, ":", EverNoticed[Id])
    print("Correctness per input:")
    for el in WasInputCorrect:
        print(el, end = ' ')
    print()


    for frame in range(FramesNum):
        frame_str = format_number(frame + 1)
        instseg_image_path = images_dir + '/instance_segmentation_output/' + frame_str + ".jpg"
        rgb_image_path = images_dir + '/rgb_output/' + frame_str + ".jpg"

        #print(image_path)
        if not os.path.exists(instseg_image_path):
            print("no instance segmentation image for frame " + str(frame + 1))
            continue
        if not os.path.exists(rgb_image_path):
            print("no rgb image for frame " + str(frame + 1))
            continue
        inst_img  = Image.open(instseg_image_path) 
        rgb_img = Image.open(rgb_image_path)
        inst_pixels = inst_img.load()
        rgb_pixels = rgb_img.load()
        width, height = inst_img.size
        transp = 0.6
        for i in range(width):
            for j in range(height):
                ir, ig, ib = inst_img.getpixel((i, j))
                id = ib * 256 + ig

                rr, rg, rb = rgb_img.getpixel((i, j))
                res_r, res_g, res_b = (rr * transp, rg * transp, rb * transp)

                if id in TypeDict.keys():
                    if TypeDict[id] == "vehicle" or TypeDict[id] == "walker":
                        if id in ChangeToNoticed[frame]:
                            print(id)
                            res_r += (1 - transp) * 255
                            res_g += (1 - transp) * ig
                            res_b += (1 - transp) * ib
                        else:
                            res_g += (1 - transp) * ig
                            res_b += (1 - transp) * ib
                inst_pixels[i, j] = (int(res_r), int(res_b), int(res_g))
        inst_img.save(results_dir + '/' + frame_str, format="png")
   

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="DReyeVR recording parser")
    argparser.add_argument(
        "-f",
        "--file",
        metavar="P",
        type=str,
        help="path of the (human readable) recording file",
    )
    argparser.add_argument(
        "-i",
        "--images",
        default="not entered",
        type=str,
        help="path to the folder with images",
    )
    argparser.add_argument(
        "-o",
        "--out",
        default="results",
        type=str,
        help="path of the results folder",
    )

    
    args = argparser.parse_args()

    main(args.file, args.images, args.out)
