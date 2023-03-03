from typing import Dict, Optional, List
from src.parser import parse_file
from src.utils import get_good_idxs, fill_gaps, smooth_arr, compute_YP, convert_to_df, flatten_dict
import numpy as np
import argparse
from src.visualizer import (
    set_results_dir,
)


def main(filename: str, results_dir: str, vlines: Optional[List[float]] = None):
    set_results_dir(results_dir)

    """parse the file"""
    data: Dict[str, np.ndarray or dict] = parse_file(filename)
    # can also use data["TimestampCarla"] which is in simulator time
    # Set helper data structures
    t: np.ndarray = data["TimeElapsed"]
    AwarenessData = data["AwarenessData"]
    RenderedTotal = AwarenessData["RenderedTotal"]
    UserInput = AwarenessData["UserInput"]
    FramesNum = len(RenderedTotal)
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
    RenderedTotal[-1] = 0
    for num in RenderedTotal: 
        Ids.append(AwarenessData["Id"][idx : idx + num])
        Answer.append(AwarenessData["Answer"][idx : idx + num])
        Loc.append(AwarenessData["Location"][idx : idx + num])
        Vel.append(AwarenessData["Velocity"][idx : idx + num])
        idx += num
    Type = []
    TypeDict = {}
    for i in range (FramesNum):
        Type.append([])
        for j in range(RenderedTotal[i]):
            if Answer[i][j] & VehicleType:
                Type[i].append("vehicle")
                TypeDict[Ids[i][j]] = "vehicle"
            else:
                Type[i].append("walker")
                TypeDict[Ids[i][j]] = "walker"
    
    Noticed : Dict = {}
    EverNoticed: Dict = {}
    FirstAppeared : Dict = {}
    AllRendered : np.ndarray = []
    WasInputCorrect: np.ndarray = []

    # Parse the data
    for i in range (FramesNum):
        ActorsNum = RenderedTotal[i]
        for Id in Ids[i]:
            if i == 0 or not(Id in Ids[i - 1]):
                FirstAppeared[Id] = i + 1
            if not (Id in Noticed):
                AllRendered.append(Id)
                EverNoticed[Id] = False
                Noticed[Id] = False

        if UserInput[i] > 0:
            FoundCorrect = False
            for j in range(ActorsNum):
                if not(Noticed[Ids[i][j]]) and (UserInput[i] & TypeBit == Answer[i][j] & TypeBit) and (UserInput[i] & Answer[i][j]):
                    Noticed[Ids[i][j]] = True
                    EverNoticed[Ids[i][j]] = True
                    FoundCorrect = True
                    break
            WasInputCorrect.append(FoundCorrect)
    
        if i == 0:
            continue
        '''for Id in Ids[i - 1]:
            if not(Id in Ids[i]):
                Noticed[id] = False'''
    print()
    print("All actors that were rendered at some point:")
    for Id in AllRendered:
        print(Id, TypeDict[Id], end = '\n')
    print()
    print("Was actor ever noticed?")
    for Id in AllRendered:
        print(Id, ":", EverNoticed[Id])
    print("Correctness per input:")
    for el in WasInputCorrect:
        print(el, end = ' ')
    print()

    # Get rotation for rendered actors from the "Actors" field of the data dictionary
    Rot = []
    for i in range(FramesNum):
        Rot.append(np.array([data["Actors"]["Rotation"][i][id] for id in Ids[i]]))

    # Combine all data with awareness data
    datafinal = data.copy()

    AwData : Dict = {}
    AwData["Rendered"] = Ids
    AwData["Type"] = Type
    AwData["Answer"] = Answer
    AwData["UserInput"] = UserInput
    AwData["RenderedLocation"] = Loc
    AwData["RenderedVelocity"] = Vel
    AwData["RenderedRotation"] = Rot
    datafinal["AwarenessData"] = AwData

    # Convert to pandas dataframe
    import pandas as pd
    awareness_frame = convert_to_df(datafinal)
   
    # Display the frame
    from IPython.display import display
    from tabulate import tabulate
    print(awareness_frame.keys())
    #print(tabulate(awareness_frame, headers = 'keys', tablefmt = 'fancy_grid'))
    
   

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
        "-o",
        "--out",
        default="results",
        type=str,
        help="path of the results folder",
    )
    args = argparser.parse_args()

    main(args.file, args.out)
