from typing import Dict, Optional, List
from src.parser import parse_file
from src.utils import get_good_idxs, fill_gaps, smooth_arr, compute_YP
from src.visualizer import (
    plot_versus,
    plot_histogram2d,
    plot_vector_vs_time,
    plot_3Dt,
    set_results_dir,
)
import numpy as np
import argparse


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

    """visualize some interesting data!"""
    pupil_L = data["EyeTracker"]["LEFTPupilDiameter"]
    # drop invalid data
    good_pupil_diam = lambda x: x > 0  # need positive diameter
    good_idxs = get_good_idxs(pupil_L, good_pupil_diam)
    pupil_L = pupil_L[good_idxs]
    time_L = t[good_idxs]

    plot_versus(
        data_x=time_L,
        name_x="Time",
        data_y=pupil_L,
        name_y="Left pupil diameter",
        units_y="mm",
        units_x="s",
        lines=True,
        vlines=vlines,
    )

    pupil_R = data["EyeTracker"]["RIGHTPupilDiameter"]
    good_idxs = get_good_idxs(pupil_R, good_pupil_diam)
    pupil_R = pupil_R[good_idxs]
    time_R = t[good_idxs]
    plot_versus(
        data_x=time_R,
        name_x="Time",
        data_y=pupil_R,
        name_y="Right pupil diameter",
        units_y="mm",
        units_x="s",
        lines=True,
        vlines=vlines,
    )

    data["TimestampCarla"] = data["TimestampCarla"] / 1000  # to seconds
    t = data["TimestampCarla"]
    # now t is in seconds

    """visualize some interesting data!"""

    eye = data["EyeTracker"]
    all_valid = (
        eye["COMBINEDGazeValid"]
        & eye["LEFTGazeValid"]
        & eye["LEFTEyeOpennessValid"]
        & eye["LEFTPupilPositionValid"]
        & eye["RIGHTGazeValid"]
        & eye["RIGHTEyeOpennessValid"]
        & eye["RIGHTPupilPositionValid"]
    )
    all_valid_idxs = np.where(all_valid == 1)
    print(f"Total validity percentage: {100 * np.sum(all_valid) / len(all_valid):.3f}%")

    plot_versus(
        data_x=t,
        name_x="Time",
        data_y=all_valid,
        name_y="Confidence (validity)",
        units_x="s",
        lines=False,
        vlines=vlines,
    )

    pupil_mm_L = eye["LEFTPupilDiameter"]
    if (pupil_mm_L < 0).any():  # correct for negatives
        pupil_mm_L = fill_gaps(pupil_mm_L, lambda x: x < 0, mode="mean")
    plot_versus(
        data_x=t,
        name_x="Time",
        data_y=smooth_arr(pupil_mm_L, 100),
        name_y="Left pupil diameter",
        units_y="mm",
        units_x="s",
        valid_idxs=all_valid_idxs,
        lines=True,
        vlines=vlines,
    )

    pupil_mm_R = eye["RIGHTPupilDiameter"]
    if (pupil_mm_R < 0).any():  # correct for negatives
        pupil_mm_R = fill_gaps(pupil_mm_R, lambda x: x < 0, mode="mean")
    plot_versus(
        data_x=t,
        name_x="Time",
        data_y=smooth_arr(pupil_mm_R),
        name_y="Right pupil diameter",
        units_y="mm",
        units_x="s",
        valid_idxs=all_valid_idxs,
        vlines=vlines,
        lines=True,
    )

    gaze_dir_C = eye["COMBINEDGazeDir"][all_valid_idxs]
    gaze_yaw_C, gaze_pitch_C = compute_YP(gaze_dir_C)
    plot_histogram2d(
        data_x=gaze_yaw_C,
        data_y=gaze_pitch_C,
        name_x="yaw_C",
        name_y="pitch_C",
        units_x="deg",
        units_y="deg",
        bins=100,
    )
    gaze_dir_L = eye["LEFTGazeDir"][all_valid_idxs]
    gaze_yaw_L, gaze_pitch_L = compute_YP(gaze_dir_L)
    plot_histogram2d(
        data_x=gaze_yaw_L,
        data_y=gaze_pitch_L,
        name_x="yaw_L",
        name_y="pitch_L",
        units_x="deg",
        units_y="deg",
        bins=100,
    )

    gaze_dir_R = eye["RIGHTGazeDir"][all_valid_idxs]
    gaze_yaw_R, gaze_pitch_R = compute_YP(gaze_dir_R)
    plot_histogram2d(
        data_x=gaze_yaw_R,
        data_y=gaze_pitch_R,
        name_x="yaw_R",
        name_y="pitch_R",
        units_x="deg",
        units_y="deg",
        bins=100,
    )

    """plot 3D position over time"""
    pos3D = data["EgoVariables"]["VehicleLoc"]
    plot_vector_vs_time(pos3D, t, "EgoPos XYZ", vlines=vlines)

    if vlines is not None:  # if you want to zoom in to a particular point in time
        zoom_pt = vlines[0]  # TODO: generalize for all vlines
        width = 300  # number of ticks before/after the zoom point ("window size")
        print(f"Zooming in to time: {zoom_pt:.2f}s")
        # trimming off the ends
        omit_front: int = np.searchsorted(t, zoom_pt) - width
        omit_rear: int = len(t) - (np.searchsorted(t, zoom_pt) + width)

        # if using valid-ified t data
        valid_t: np.ndarray = t[all_valid_idxs]
        omit_front_valid: int = np.searchsorted(valid_t, zoom_pt) - width
        omit_rear_valid: int = len(valid_t) - np.searchsorted(valid_t, zoom_pt) + width
    else:
        omit_front: int = 300  # hardcoded number of ticks to ignore from the start
        omit_rear: int = 300  # hardcoded number of ticks to ignore from the end

        # if using valid-ified t data (unnecessary unless searching)
        omit_front_valid: int = omit_front
        omit_rear_valid: int = omit_rear

    plot_vector_vs_time(
        pos3D, t, "EgoPos XYZ", omit=(omit_front, omit_rear), vlines=vlines
    )

    rot3D = data["EgoVariables"]["VehicleRot"]
    plot_vector_vs_time(
        rot3D,
        t,
        "EgoRot PYR",
        ax_titles=["P", "Y", "R"],
        omit=(omit_front, omit_rear),
        vlines=vlines,
    )

    plot_3Dt(
        xyz=pos3D,
        t=t,
        title="Vehicle position over time",
        interactive=False,  # set to True to move it around
        omit=(omit_front, omit_rear),
    )

    """plot custom actor data"""
    if "CustomActor" in data:
        for name in data["CustomActor"].keys():
            CA_data: dict = data["CustomActor"][name]
            plot_3Dt(
                xyz=CA_data["Location"],
                t=CA_data["t"],
                title=f"CA {name} position",
                interactive=False,  # set to True to move it around
                # omit=(omit_front, omit_rear),
            )

    """plot pupil position"""
    pupil_pos_L = eye["LEFTPupilPosition"]
    plot_vector_vs_time(
        pupil_pos_L,
        t,
        "Left pupil position",
        valid_idxs=all_valid_idxs,
        omit=(omit_front_valid, omit_rear_valid),
        norm=True,
        vlines=vlines,
    )
    plot_histogram2d(
        data_x=pupil_pos_L[all_valid_idxs][:, 0],
        data_y=pupil_pos_L[all_valid_idxs][:, 1],
        name_x="LPupilX",
        name_y="LPupilY",
        bins=100,
    )

    pupil_pos_R = eye["RIGHTPupilPosition"]
    plot_vector_vs_time(
        pupil_pos_R,
        t,
        "Right pupil position",
        valid_idxs=all_valid_idxs,
        omit=(omit_front_valid, omit_rear_valid),
        norm=True,
        vlines=vlines,
    )
    plot_histogram2d(
        data_x=pupil_pos_R[all_valid_idxs][:, 0],
        data_y=pupil_pos_R[all_valid_idxs][:, 1],
        name_x="RPupilX",
        name_y="RPupilY",
        bins=100,
    )

    """plot eye vars"""
    plot_versus(
        data_x=t,
        name_x="Time",
        data_y=eye["LEFTEyeOpenness"],
        name_y="Left eye openness",
        units_y="mm",
        units_x="s",
        valid_idxs=all_valid_idxs,
        omit=(omit_front_valid, omit_rear_valid),
        vlines=vlines,
        lines=True,
    )

    plot_versus(
        data_x=t,
        name_x="Time",
        data_y=eye["RIGHTEyeOpenness"],
        name_y="Right eye openness",
        units_y="mm",
        units_x="s",
        lines=True,
        valid_idxs=all_valid_idxs,
        omit=(omit_front_valid, omit_rear_valid),
        vlines=vlines,
    )

    """compute intrinsic factors"""
    delta_ts = np.diff(t)  # t is in seconds
    n: int = len(delta_ts)
    assert delta_ts.min() > 0  # should always be monotonically increasing!
    ego_displacement = np.diff(data["EgoVariables"]["VehicleLoc"], axis=0)
    ego_velocity = (ego_displacement.T / delta_ts).T
    assert ego_velocity.shape == (n, 3)
    cmps2mph = 0.0223694  # cm/s to mph
    speed = np.linalg.norm(ego_velocity, axis=1)  # velocity (3D) to speed (1D)
    plot_versus(
        data_x=t[1:],
        name_x="Time",
        data_y=cmps2mph * speed,
        name_y="Ego Speed",
        units_y="mph",
        units_x="s",
        lines=True,
        vlines=vlines,
        omit=(omit_front, omit_rear),
    )

    ego_accel = (np.diff(ego_velocity, axis=0).T / delta_ts[1:]).T
    assert ego_accel.shape == (n - 1, 3)
    plot_versus(
        data_x=t[2:],
        name_x="Time",
        data_y=np.linalg.norm(ego_accel, axis=1),  # accel (3D) to speed (1D)
        name_y="Ego Accel",
        units_y="cm/s^2",
        units_x="s",
        vlines=vlines,
        lines=True,
        omit=(omit_front, omit_rear),
    )
    plot_versus(
        data_x=t[2:],
        name_x="Time",
        data_y=smooth_arr(
            np.linalg.norm(ego_accel, axis=1), kernel_size=20
        ),  # accel (3D) to speed (1D)
        name_y="Smooth Ego Accel",
        units_y="cm/s^2",
        units_x="s",
        vlines=vlines,
        lines=True,
        omit=(omit_front, omit_rear),
    )
    # jerk, snap, crackle, pop?

    angular_disp = np.diff(rot3D, axis=0)
    # fix rollovers for +360
    angular_disp[np.squeeze(np.where(np.abs(np.diff(rot3D[:, 1], axis=0)) > 359))] = 0
    # pos_roll_idxs = np.squeeze(np.where(np.diff(rot3D[:, 1], axis=0) > 359))
    # angular_disp[pos_roll_idxs][:, 1] = -1 * (360 - angular_disp[pos_roll_idxs][:, 1])
    # neg_roll_idxs = np.squeeze(np.where(np.diff(rot3D[:, 1], axis=0) < -359))
    # angular_disp[neg_roll_idxs][:, 1] = 360 + angular_disp[neg_roll_idxs][:, 1]
    angular_vel = (angular_disp.T / delta_ts).T
    plot_vector_vs_time(
        angular_vel,
        t[1:],
        "Delta EgoRot PYR",
        ax_titles=["P", "Y", "R"],
        omit=(omit_front, omit_rear),
        vlines=vlines,
    )

    # TODO: keep track of vehicles in the scene and track their positions (interpolated) over time

    """steering over time"""
    plot_versus(
        data_x=t,
        name_x="Time",
        data_y=data["UserInputs"]["Steering"],
        name_y="Steering",
        units_y="Deg",
        units_x="s",
        vlines=vlines,
        lines=True,
        omit=(omit_front, omit_rear),
    )

    """velocity over time"""

    plot_vector_vs_time(
        ego_velocity,
        t[1:],
        "EgoVel XYZ",
        omit=(omit_front, omit_rear),
        vlines=vlines,
    )
    plot_vector_vs_time(
        ego_accel, t[2:], "EgoAccel XYZ", omit=(omit_front, omit_rear), vlines=vlines
    )
    plot_vector_vs_time(
        data["EgoVariables"]["VehicleRot"],
        t,
        "EgoRot XYZ",
        omit=(omit_front, omit_rear),
        vlines=vlines,
    )

    """Stored velocity over time"""
    plot_versus(
        data_x=t,
        name_x="Time",
        data_y=data["EgoVariables"]["VehicleVel"],
        name_y="Velocity",
        units_y="m/s",
        vlines=vlines,
        units_x="s",
        lines=True,
        omit=(omit_front, omit_rear),
    )

    plot_versus(
        data_x=t,
        name_x="Time",
        data_y=data["UserInputs"]["Throttle"],
        name_y="Throttle",
        units_x="s",
        lines=True,
        vlines=vlines,
        omit=(omit_front, omit_rear),
    )

    plot_versus(
        data_x=t,
        name_x="Time",
        data_y=data["UserInputs"]["Brake"],
        name_y="Brake",
        units_x="s",
        lines=True,
        vlines=vlines,
        omit=(omit_front, omit_rear),
    )

    """plot relative camera things"""
    plot_vector_vs_time(
        data["EgoVariables"]["CameraLoc"],
        t,
        "CameraLoc",
        omit=(omit_front, omit_rear),
        vlines=vlines,
    )
    plot_vector_vs_time(
        data["EgoVariables"]["CameraRot"],
        t,
        "CameraRot",
        omit=(omit_front, omit_rear),
        vlines=vlines,
    )

    """plot gaze things"""
    plot_vector_vs_time(
        data["EyeTracker"]["COMBINEDGazeDir"],
        t,
        "CombinedGaze",
        valid_idxs=all_valid_idxs,
        omit=(omit_front_valid, omit_rear_valid),
        norm=True,
        vlines=vlines,
    )
    plot_vector_vs_time(
        data["EyeTracker"]["LEFTGazeDir"],
        t,
        "LeftGaze",
        valid_idxs=all_valid_idxs,
        norm=True,
        omit=(omit_front_valid, omit_rear_valid),
        vlines=vlines,
    )
    plot_vector_vs_time(
        data["EyeTracker"]["RIGHTGazeDir"],
        t,
        "RightGaze",
        valid_idxs=all_valid_idxs,
        omit=(omit_front_valid, omit_rear_valid),
        norm=True,
        vlines=vlines,
    )

    """plot actor things"""
    np.random.seed(2)
    for _ in range(10):  # plot 10 random actors
        Id: int = np.random.choice(np.array(list(data["Actors"].keys())))
        actor_data = data["Actors"][Id]

        def get_non_zero_idxs(arr: np.ndarray) -> np.ndarray:
            return np.array([i for i in range(len(arr)) if (arr[i] != 0).all()])

        non_zeros = get_non_zero_idxs(actor_data["Location"])
        pos3D = actor_data["Location"][non_zeros]
        _t = actor_data["Time"][non_zeros]
        plot_3Dt(
            xyz=pos3D,
            t=_t,
            title=f"Actor id {Id} position over time",
            interactive=False,  # set to True to move it around
        )

        # plot the distance to this actor (need to match their timestamps)
        haz_t = actor_data["Time"] / 1000  # to get in seconds
        t = t  # should be a superset of haz_t (contain everything + more)
        idxs = np.searchsorted(t, haz_t)  # find  where hazard's t is in world t
        new_t = t[idxs]

        ego_xyz = data["EgoVariables"]["VehicleLoc"][idxs]
        haz_xyz = actor_data["Location"]

        assert ego_xyz.shape == haz_xyz.shape
        assert len(ego_xyz) == len(new_t)
        dist = np.linalg.norm(ego_xyz - haz_xyz, axis=1) / 100
        print(f"Minimum distance: {np.min(dist):.2f}m")
        plot_versus(
            data_x=new_t,
            name_x="Time",
            data_y=dist,
            name_y=f"Distance to actor {Id}",
            units_y="m",
            units_x="s",
            lines=True,
            vlines=vlines,
            omit=(omit_front, omit_rear),
        )


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
