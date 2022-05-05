# DReyeVR Recording Parser

Run an example of the DReyeVR parser with
```bash
python3 example.py --file /PATH/TO/FILE.txt
# this will cache the parsing and create a results directory in results/ with several visualizations
```

Where `/PATH/TO/FILE` points to a human readable DReyeVR (Carla) recording obtained as follows:
```bash
python show_recorder_file_info.py -a -f /PATH/TO/CARLA_LOG_REC > /PATH/TO/FILE.txt
```