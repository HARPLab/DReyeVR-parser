# DReyeVR Recording Awareness Parser

Run an example of the DReyeVR parser with
```bash
python3 example.py --file /PATH/TO/FILE.txt
# this will cache the parsing and create a results directory in results/ with several visualizations
```

Where `/PATH/TO/FILE` points to a human readable DReyeVR (Carla) recording obtained as follows:
```bash
python show_recorder_file_info.py -a -f /PATH/TO/CARLA_LOG_REC > /PATH/TO/FILE.txt
```

This is a version of the parser meant to be used for recordings in awareness mode (DReyeVR mode that tracks the driver's awareness of vehicles and walkers inside their visibility field). It analizes visible objects and user input to produce a pandas data frame and a report about the user's awareness of their surroundings.

## Usage
To use the parser, first, obtain a human readable version of the recording file: 
```bash
python show_recorder_file_info.py -a -f /PATH/TO/CARLA_LOG_REC > /PATH/TO/FILE.txt
```
Then, run the parser:
```bash
python3 example.py --file /PATH/TO/FILE.txt
```
It is going to produce a result table in a pandas data frame format.

## Data fields

By default, DReyeVR recording consists of several data fields describing the vehicle and the surroundings per frame. This data is contained inside the **data** dictionary, an output of the **parse_file** function from **src/parser.py**.
A recording in awareness mode contains an additional data field called **data["AwarenessData"]** (henceforth AwarenessData) with all the awareness information.
This field contains several subfields (per frame):
- **AwarenessData["VisibleTotal"]**: a number of vehicles and walkers (from now on - objetcs) visible
- **AwarenessData["Id"]**: a list of Carla ids of all the visible objects
- **AwarenessData["User Input"]**: a number encoding of the user input. More about encodings in a later section
- **AwarenessData["Answer"]**: a list of number encoding of the correct answers for each visible object
- **AwarenessData["Location"]**: current coordinates of all the visible actors
- **AwarenessData["Velocity"]**: current velocity of the visible actors
- **AwarenessData["Rotation"]**: current rotation of all the visible actors

## Resulting data fields
The result is first formatted as a dictionary **datafinal** containing all the initial fields from the **data** dictionary plus the "AwarenessData" field, also a dictionary. This field contains:
- **AwData["Visible"]**: IDs of all the visible actors per frame
 - **AwData["Type"]**: type per actor, "vehicle" or "walker"
 - **AwData["Answer"]**
 - **AwData["UserInput"]**
 - **AwData["VisibleLocation"]**
 - **AwData["VisibleVelocity"]** 
  - **AwData["VisibleRotation"]**
  
This dictionary is then be converted to pandas data frame.

## User input and answer: encoding and meaning

There are two versions of awareness mode: velocity and position. Awareness mode measures the degree us the user's awareness of their surroundings by asking the user to use the arrow keys to indicate either relative velocity or position (relative to the ego vehicle). For example: where is the car I see located relative to me - in front of me, on the left, on the right, of behind (position mode)? In addition, shift and ctrl keys are used to differentiate between vehicles and walkers. 

For velocity mode, "direction" refers to the the relative velocity direction. For position, it refers to the relative postition. In both cases, possible values are [left, right, forwward, backward]. 

Each frame, the user's attempt to identify the object direction is encoded in a **UserInput** field. If there was no keys pressed this frame, the field will be 0.
For each visible object, the correct answer (type + direction) is encoded in a similar fashion. 

**Encoding rules:**
-   For both user input and answer, 4 least significant bits represent the 4 directions:
-   Bit 0 for forward, bit 1 for right, bit 2 for backward, bit 3 for left
-   For user input, only the bit corresponding to the selected direction is set to 1. If there was no input, the value is 0
-   For the correct answer, all the possible correct direction bits are set to 1 (sometimes, direction is ambiguous)
-   Answer matches the input if there is at least 1 bit that is set for both numbers.
-   Thus, to check that for frame i, answer for jth actor in the list of the visible actors matches the input, *logical and* is used: **UserInput[i] & Answer[i][j]**

