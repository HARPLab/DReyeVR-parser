#!/bin/bash

# init all relevant fixed paths
export carla_root=${HOME}/CarlaDReyeVR/carla
export sensor_config=${HOME}/CarlaDReyeVR/carla/PythonAPI/examples/sensor_config.ini
export out_dir=/media/storage/raw_data/
# export out_dir=${HOME}/raw_data/

export rec_file_dir=${HOME}/CBDR
# export awdata_file_dir=${HOME}/CarlaDReyeVR/DReyeVR-parser/results


# loop over and get all recording files

# read all .rec files in the recording_files directory
recording_files=$(ls ${rec_file_dir}/*.rec)

# loop over all recording files
for recording_file in ${recording_files}; do
    start=`date +%s`
    # recording_file=${HOME}/CarlaDReyeVR/DReyeVR-parser/recording_files/exp_abd-54_02_13_2024_10_31_55.rec

    # below this is in a loop:
    # get basename of the recording file
    recording_basename=$(basename ${recording_file} .rec) # exp_abd-54_02_13_2024_10_31_55
    # strip the 'exp' and the timestamp from the basename
    recording_basename=$(echo ${recording_basename} | cut -d'_' -f 2) # abd-54
    # get the participant name 
    participant_name=$(echo ${recording_basename} | cut -d'-' -f 1) # abd
    # get the route number
    route_number=$(echo ${recording_basename} | cut -d'-' -f 2) # 54
    if [ "$route_number" == "00" ]; then
        continue
    fi
    recording_out_dir=${out_dir}/${recording_basename}
    mkdir -p ${recording_out_dir}

    # construct dependent file paths
    # exp_abd_54.txt
    parse_file=${recording_out_dir}/rec_parse.txt
    # exp_abd_54-awdata.json
    awdata_file=${recording_out_dir}/rec_parse-awdata.json

    # produce replay parse_file
    python ${carla_root}/PythonAPI/examples/show_recorder_file_info.py -a -f ${recording_file} > ${parse_file}

    # do awareness parsing to get awdata json
    python awareness_parser.py -f  ${recording_out_dir}/rec_parse.txt -o ${recording_out_dir} -r True
    

    # do replay to get sensor data
    python ${carla_root}/PythonAPI/examples/replay_instance_segm_3_cameras.py \
    -f ${recording_file} \
    --sensor-config ${sensor_config} \
    -aw ${recording_out_dir}/rec_parse-awdata.json \
    --out-dir ${recording_out_dir}

    # calculate the offset
    python image_seg_masking.py \
     --img-dir ${recording_out_dir}/images \
     -aw ${recording_out_dir}/rec_parse-awdata.json 
    
    # TODO produce the gaze button overlay 
    # This will save the frames in the gaze_button_overlay dir in the folder where the script is called from
    python scene_representation_script.py \
    --data-dir ${recording_out_dir} \
    -s /home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/sensor_config.ini
    
    # POSSIBLy: do label correction?       
    end=`date +%s`

    runtime=$((end-start))
    echo $recording_basename runtime: $runtime >> runtimes.txt
done