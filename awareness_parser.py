from typing import Dict, Optional, List
from src.parser import parse_file
from src.utils import get_good_idxs, fill_gaps, smooth_arr, compute_YP
import numpy as np
import argparse
from src.visualizer import (
    set_results_dir,
)


def main(filename: str, results_dir: str, vlines: Optional[List[float]] = None):
    set_results_dir(results_dir)

    """parse the file"""
    data: Dict[str, np.ndarray or dict] = parse_file(filename)

    # """convert to pandas df"""
    # import pandas as pd
    # # need to split along groups so all data lengths are the same
    # data_groups = split_along_subgroup(data, ["CustomActor"])
    # data_groups_df: List[pd.DataFrame] = [convert_to_df(x) for x in data_groups]

    # can also use data["TimestampCarla"] which is in simulator time
    t: np.ndarray = data["TimeElapsed"]
    AwarenessData = data["AwarenessData"]
    '''print(d)
    print(len(d["RenderedTotal"]))
    print(sum(d["RenderedTotal"]))
    print(len(d["UserInput"]))
    print(len(d["Id"]))
    print(len(d["Answer"]))
    print(len(d["Velocity"]))
    print(t)'''

    RenderedTotal = AwarenessData["RenderedTotal"]
    UserInput = AwarenessData["UserInput"]
    FramesNum = len(RenderedTotal)
    if (FramesNum == 0):
        print("0 length recording, nothing to parse")
        return

    Ids : np.ndarray = []
    Answer : np.ndarray = []
    idx = 0
    for num in RenderedTotal: 
        Ids.append(AwarenessData["Id"][idx : idx + num])
        Answer.append(AwarenessData["Answer"][idx : idx + num])
        idx += num
    
    '''print(Ids)
    print()
    print(Answer)'''
    Noticed : Dict = {}
    FirstAppeared : Dict = {}
    for i in range (FramesNum):
        ActorsNum = RenderedTotal[i]
        for Id in Ids[i]:
            if i == 0 or not(Id in Ids[i - 1]):
                FirstAppeared[Id] = i + 1
            if not (Id in Noticed):
                #print(Id)
                Noticed[Id] = False

        if UserInput[i] > 0:
            for j in range(ActorsNum):
                if not(Noticed[Ids[i][j]]) and (UserInput[i] & Answer[i][j]):
                    Noticed[Ids[i][j]] = True
                    break
        
        print("Frame", i + 1)
        for Id in Ids[i]:
            print(Id, Noticed[Id])
        print()

        if i == 0:
            continue
        for Id in Ids[i - 1]:
            if not(Id in Ids[i]):
                Noticed[id] = False
   

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
