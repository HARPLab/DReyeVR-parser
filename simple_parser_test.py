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
    data.pop("AwarenessData")
    print(data.keys())
    # Convert to pandas dataframe
    import pandas as pd
    actors_data = {}
    actors_data["TimeElapsed"] = data["TimeElapsed"]
    actors_data["Actors"] = {}
    actors_data["Actors"]["Location"] = data["Actors"]["Location"]
  
    

    data_frame = convert_to_df(data)
    actors_df = convert_to_df(actors_data)
   
    # Display the frame
    from IPython.display import display
    from tabulate import tabulate
    print(data_frame.keys())
    print(tabulate(actors_df, headers = 'keys', tablefmt = 'fancy_grid'))
    
   

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
