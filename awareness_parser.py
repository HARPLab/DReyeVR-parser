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

    Ids = []
    Answer = []
    Loc = []
    Vel = []
    idx = 0
    for num in RenderedTotal: 
        Ids.append(AwarenessData["Id"][idx : idx + num])
        Answer.append(AwarenessData["Answer"][idx : idx + num])
        Loc.append(AwarenessData["Location"][idx : idx + num])
        Vel.append(AwarenessData["Velocity"][idx : idx + num])
        idx += num
    
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
                if not(Noticed[Ids[i][j]]) and (UserInput[i] & Answer[i][j]):
                    Noticed[Ids[i][j]] = True
                    EverNoticed[Ids[i][j]] = True
                    FoundCorrect = True
                    break
            WasInputCorrect.append(FoundCorrect)
    
        if i == 0:
            continue
        for Id in Ids[i - 1]:
            if not(Id in Ids[i]):
                Noticed[id] = False
    print()
    print("All actors that were rendered at some point:")
    for Id in AllRendered:
        print(Id, end = ' ')
    print()
    print("Was actor ever noticed?")
    for Id in AllRendered:
        print(Id, ":", EverNoticed[Id])
    print("Correctness per input:")
    for el in WasInputCorrect:
        print(el, end = ' ')
    print()

    # Get actor rotation from the "Actors" field of the data dictionary
    curr_idx : Dict = {}
    for key in data["Actors"]:
        curr_idx[key] = 0
    Rot = []

    for i in range(FramesNum):
        Rot.append(np.array([list(data["Actors"][id]["Rotation"][curr_idx[id]]) for id in Ids[i]]))
        for id in Ids[i]:
            curr_idx[id] += 1

    # Combine all data with awareness data
    datafinal = data.copy()
    del datafinal["Actors"]

    AwData : Dict = {}
    AwData["Rendered"] = Ids
    AwData["Answer"] = Answer
    AwData["UserInput"] = UserInput
    AwData["Location"] = Loc
    AwData["Velocity"] = Vel
    AwData["Rotation"] = Rot
    datafinal["AwarenessData"] = AwData

    # Convert to pandas dataframe
    import pandas as pd
    awareness_frame = convert_to_df(datafinal)
   
    # Display the frame
    from IPython.display import display
    from tabulate import tabulate
    # print(awareness_frame.keys())
    # print(tabulate(awareness_frame, headers = 'keys', tablefmt = 'fancy_grid'))
    
   

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
