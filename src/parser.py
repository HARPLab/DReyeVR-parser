import os
from typing import Dict, List, Any, Optional
import time
import sys
import pickle

# allow us to import from this current directory
# parser_dir: str = "/".join(__file__.split("/")[:-1])
# sys.path.insert(1, parser_dir)

# allow us to import from the directory this file is in
parser_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parser_dir)



from utils import (
    process_UE4_string_to_value,
    convert_to_np,
    convert_standalone_dict_to_list,
    get_filename_from_path,
    cleanup_data_line,
)
import numpy as np

# used as the dictionary key when the data has no explicit title (ie. included as raw array)
_no_title_key: str = "data_single"  # data with this key will be converted to a raw list
cache_dir: str = os.path.join(parser_dir, "cache")
os.makedirs(cache_dir, exist_ok=True)


def parse_data_line(data_line: str, t: float, working_map: dict) -> dict:
    if t != 0:
        if (
            "t" not in working_map
        ):  # in case we need to also link the time associated with this
            working_map["t"] = []
        working_map["t"].append(t)
    subtitle: str = ""  # subtitle for elements within the dictionary
    for element in data_line:

        # case when empty string occurs (from "{" or "}" splitting)
        if element == "":
            subtitle = ""  # reset subtitle name
            continue

        # case when only element (after title) is present (eg. TimestampCarla)
        if ":" not in element:
            if _no_title_key not in working_map:
                working_map[_no_title_key] = []
            working_map[_no_title_key].append(process_UE4_string_to_value(element))
            continue

        # common case
        key_value: List[str] = element.split(":")
        key_value: List[str] = [elem for elem in key_value if len(elem) > 0]
        if len(key_value) == 1:  # contains only sublabel, like "COMBINED"
            subtitle = key_value[0]
        elif len(key_value) == 2:  # typical key:value pair
            key, value = key_value
            key = f"{subtitle}{key}"
            value = process_UE4_string_to_value(value)  # evaluate what we think this is

            # add to the working map
            if key not in working_map:
                working_map[key] = []
            working_map[key].append(value)
        else:
            raise NotImplementedError
    return working_map


def parse_row(
    data: Dict[str, Any],
    data_line: str,
    title: Optional[str] = "",
    t: Optional[int] = 0,
) -> None:
    # NOTE: this is for DReyeVR specific recorder lines!!! Not Carla!

    # cleanup data line
    data_line: List[str] = cleanup_data_line(data_line)

    if title == "":  # find title with the data line
        title: str = data_line[0].split(":")[0]
        # remove title for first elem
        data_line[0] = data_line[0].replace(f"{title}:", "")
    working_map: Dict[str, Any] = {} if title not in data else data[title]
    data[title] = working_map  # ensure this working set contributes to the larger set

    working_map = parse_data_line(data_line, t, working_map)


def parse_custom_actor(
    data: Dict[str, Any],
    data_line: str,
    title: Optional[str] = "CustomActor",
    t: Optional[int] = 0,
):
    # cleanup data line
    data_line: List[str] = cleanup_data_line(data_line)
    name: str = data_line[0].replace("Name:", "")

    if title not in data:
        data[title] = {}

    working_map: Dict[str, Any] = {} if name not in data[title] else data[title][name]
    data[title][
        name
    ] = working_map  # ensure this working set contributes to the larger set

    working_map = parse_data_line(data_line, t, working_map)


def parse_actor_location_rotation(
    data: Dict[str, Any],
    data_line: str,
    actors_key: str,
    t: Optional[int] = 0,
):
    Id: int = int(data_line[: data_line.find("Location")])
    open_0: int = data_line.find("(")
    close_0: int = data_line.find(")") + 1
    open_1: int = close_0 + 2 + len("Rotation")  # always " Rotation "
    loc_data: tuple = eval(data_line[open_0:close_0])
    rot_data: tuple = eval(data_line[open_1:])
    if len(data[actors_key]["Id"]) <= t:
        data[actors_key]["Id"].append([])
        data[actors_key]["Location"].append({})
        data[actors_key]["Rotation"].append({})
    data[actors_key]["Id"][t].append(Id)
    data[actors_key]["Location"][t][Id] = loc_data
    data[actors_key]["Rotation"][t][Id] = rot_data
    

def validate(data: Dict[str, Any], L: Optional[int] = None) -> None:
    # verify the data structure is reasonable
    if L is None:
        L: int = len(data["TimeElapsed"])
    for k in data.keys():
        if k == "CustomActor" or k == "Actors":
            continue
        if isinstance(data[k], dict):
            validate(data[k], L)
        elif isinstance(data[k], list):
            assert len(data[k]) == L or len(data[k]) == L - 1
        else:
            raise NotImplementedError

    # ensure the custom actor data is also good
    if "CustomActor" in data:
        for name in data["CustomActor"]:
            CA_lens = [len(x) for x in data["CustomActor"][name].values()]
            assert min(CA_lens) == max(CA_lens)  # all same lens

    # ensure the Carla actor data is also good
    if "Actors" in data:
        for Id in data["Actors"].keys():
            # all the actors' data structures are consistent with each other
            assert all([len(x) for x in data["Actors"][Id].values()])


def parse_file_py(
    path: str, force_reload: Optional[bool] = False, debug: Optional[bool] = False
) -> Dict[str, np.ndarray or dict]:
    # NOTE: this function is a parser for data that is listened to from a PythonAPI client
    # rather than the raw simulator recording files

    if force_reload is False:
        """try to load cached data"""
        data = try_load_data(path)
        if data is not None:
            return data

    # this function reads in a DReyeVR recording file and parses every line to return
    # a dictionary following the parser structure depending on the group types

    assert os.path.exists(path)
    print(f"Reading DReyeVR (python) logfile: {path}")

    data: Dict[str, List[Any]] = {}

    def add_to_dict(data: Dict[str, List[Any]], incoming: Dict[str, List[Any]]) -> None:
        # this fn adds/updates data from incoming to data

        # add/ensure all existing keys exist
        for k in incoming.keys():
            if k not in data:
                data[k] = []

        for k in data.keys():
            data[k].append(incoming[k])

    with open(path, "r") as f:
        start_t: float = time.time()
        for i, line in enumerate(f.readlines()):
            # convert numpy prints to lists
            clean_line: str = line.replace("array(", "").replace("])", "]")
            if clean_line[-2:] == ",\n":
                clean_line = clean_line[:-2]  # remove the newline suffix
            try:
                line_dict: str = eval(clean_line)
            except Exception as e:
                print(f'Unable to read line {i} due to "{e}"')
                continue
            if isinstance(line_dict, dict):
                add_to_dict(data, line_dict)
            # TODO: add checks to ensure all lists are the same length

            # print status
            if i % 500 == 0:
                t: float = time.time() - start_t
                print(f"Lines read: {i} @ {t:.3f}s", end="\r", flush=True)

    n: int = len(data[list(data.keys())[0]])
    print(f"successfully read {n} frames in {time.time() - start_t:.3f}s")

    # TODO: do everything in np from the get-go rather than convert at the end
    data = convert_to_np(data)
    cache_data(data, path)
    return data


def parse_file(
    path: str, force_reload: Optional[bool] = False, debug: Optional[bool] = False
) -> Dict[str, np.ndarray or dict]:
    if force_reload is False:
        """try to load cached data"""
        data = try_load_data(path)
        if data is not None:
            return data

    # this function reads in a DReyeVR recording file and parses every line to return
    # a dictionary following the parser structure depending on the group types

    assert os.path.exists(path)
    print(f"Reading DReyeVR recording file: {path}")

    data: Dict[str, List[Any]] = {}
    data["TimeElapsed"] = []

    # these are the group types we are using for now
    TimeElapsed: str = "Frame "
    DReyeVR_core: str = "[DReyeVR]"
    DReyeVR_CA: str = "[DReyeVR_CA]"
    Carla_Actor: str = "Id: "

    actors_key: str = "Actors"
    data[actors_key] = {}
    data[actors_key]["Id"] = []
    data[actors_key]["Location"] = []
    data[actors_key]["Rotation"] = []

    with open(path, "r") as f:
        start_t: float = time.time()
        for i, line in enumerate(f.readlines()):
            # remove leading spaces
            line = line.strip(" ")
            # get wall-clock time elapsed
            if line[: len(TimeElapsed)] == TimeElapsed:
                # line is always in the form "Frame X at Y seconds\n"
                line_data = line[line.find("at") + 3 :].replace(" seconds\n", "")
                data["TimeElapsed"].append(float(line_data))

            # checking the line(s) for core DReyeVR data
            elif line[: len(DReyeVR_core)] == DReyeVR_core:
                data_line: str = line.strip(DReyeVR_core).strip("\n")
                parse_row(data, data_line)
                if debug:
                    validate(data)

            # checking the line(s) for DReyeVR custom actor data
            elif line[: len(DReyeVR_CA)] == DReyeVR_CA:
                data_line: str = line.strip(DReyeVR_CA).strip("\n")
                # can also use TimeElapsed here instead, but TimestampCarla is simulator based
                t = data["TimestampCarla"][_no_title_key][-1]  # get carla time
                parse_custom_actor(data, data_line, title="CustomActor", t=t)
                if debug:
                    validate(data)

            # checking the line(s) for DReyeVR custom actor data
            elif line[: len(Carla_Actor)] == Carla_Actor:
                data_line: str = line.strip(Carla_Actor).strip("\n")
                if "Location:" not in data_line or "Rotation" not in data_line:
                    continue  # don't care about state, light, animation, etc.
                
                t = len(data["TimeElapsed"]) - 1 # get carla time
                #print('!', t)
               
                parse_actor_location_rotation(data, data_line, actors_key, t=t)
               

            # print status
            if i % 500 == 0:
                t: float = time.time() - start_t
                print(f"Lines read: {i} @ {t:.3f}s", end="\r", flush=True)

    n: int = len(data["TimeElapsed"])
    print(f"successfully read {n} frames in {time.time() - start_t:.3f}s")

    data = convert_standalone_dict_to_list(data, _no_title_key)

    # TODO: do everything in np from the get-go rather than convert at the end
    data = convert_to_np(data)
    cache_data(data, path)
    return data


def try_load_data(filename: str) -> Optional[Dict[str, Any]]:
    actual_name: str = get_filename_from_path(filename)
    filename = f"{os.path.join(cache_dir, actual_name)}.pkl"
    data = None
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded data from {filename}")
    else:
        print(f"Did not find data at {filename}")
    return data


def cache_data(data: Dict[str, Any], filename: str) -> None:
    actual_name: str = get_filename_from_path(filename)
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"{os.path.join(cache_dir, actual_name)}.pkl"
    with open(filename, "wb") as filehandler:
        pickle.dump(data, filehandler)
    print(f"cached data to {filename}")
