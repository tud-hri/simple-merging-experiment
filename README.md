# Simple Merging Experiment
This project contains the code accompanying the paper **"Interactive Merging Behavior in a Coupled Driving Simulator: Experimental Framework and Case Study"** by Olger Siebinga, Arkady Zgonnikov, and David Abbink. This software was created to investigate interactive driving behavior between two human drivers in a highway merging scenario. It includes code to execute the experiments, playback the gathered data, and evaluate the data. The data that was gathered for the publication can be found [here](https://doi.org/10.4121/19550377). A detailed description of the experimental protocol can be found with the data.

Furthermore, this repository also contains the code accompanying the paper **"A model of dyadic merging interactions explains human drivers' behaviour from input signals to decisions"** by Olger Siebinga, Arkady Zgonnikov, and David Abbink. This part of the software contains the model implementation and the code to replicate the simulations. It can be found in the `cei_model` branch, it has it's own README file there. The data that was gathered for this publication can be found [here](https://doi.org/10.4121/d77ae5bd-cfd9-4c32-8f7a-c3565c2ccdd5). 

## Quick Start
The main project folder contains three run-scripts: `run_experiment.py`, `resume_experiment.py`, and `playback.py`. The `run-experiment.py` script will start the experiment as described in the paper, with one exception. By default, there are no human input devices connected to the vehicles. Code to connect Logitech Driving Force GT steering wheels is included in the script but commented out by default (lines 150 - 167). The experiment was conducted on a single computer with three screens. One for the experimenter to control the experiment, and one for each participant to show the vehicle they control. If the run script is started on a computer with three screens, the GUI dialogs are automatically placed on the three different screens. 

The experiment consists of 110 trials (11 conditions and 10 iterations per condition). 5 randomly selected iterations are used as training trials. Every trail is saved in separate data files (*.pkl, *.csv, and *.mat). The complete state of the experiment is also saved in an autosave file after every trial. This way, the experiment can be resumed in case something fails. This is done using the `resume_experiment.py` script.

The `playback.py` script can be used to replay a trial of the experiment. It also includes a live plot of the interaction. Select which trial to load by changing the file name in the script’s main block. The plotting module can be used to gain more insight into the recorded data. It contains the scripts to reproduce the figures from the paper and more.

## The modules
All modules will be separately discussed here

### Agents
The `agents` module contains classes that can be used to provide input to the vehicles. The name `agents` is derived from AI terminology, where every decision-maker is considered an agent. Such AI agents can be implemented in the same simulation (by extending the `Agent` class), but all included classes use human inputs. These human input agents can either provide continuous or discrete input to the vehicles. In the experiment, continuous input is used. The ps4 and steering-wheel agents also can provide vibration feedback to the user. 

### Controllable Objects
The `controllableobjects` module contains the dynamical models of controllable objects. In this experiment, only a point-mass model is used. This point-mass model can be controlled with discrete or continuous inputs. The point-mass object is subjected to a simplified resistance consisting of two parts, a constant term, and a velocity-dependent term. More information on this can be found in the comments in the class file. 

### Experiment
The `experiment` module contains all classes used to set up and run the experiment. Condition definitions define a single experimental condition. These can be put together in a condition list, where they are repeated in a randomized order. The `experiment_conditions.py` script can be used to generate the specific conditions used in the experiment as described in the paper. The `autosave.py` script handles the automatic saving of the experiment status after every trial.

### GUI
All files related to the graphical user interfaces are located in the GUI module. All GUI files are made with pyqt. There are 4 main GUIs in the project. The `experiment_control_gui` is the main window when the experiment is conducted. It is displayed on the center screen and shows an overview of the situation, the human inputs, the status of the cruise control, and the current experimental condition. The experimenter starts each trial individually from here. 

The participants see the simulation on their screens in the `participantdialog` GUI. Before the experiment starts, the `participantinfodialog` is used to gather some basic data from the participant. The data is saved with a (confidential) participant identifier. This identifier is shown only to the participant. They are asked to write it down and not share it with the experimenter. In case the participant wants to withdraw their data after the experiment was conducted, they can share their identifier so the experimenter knows which data to delete.

Finally, the `simulation_gui` is the user interface that is used when recorded data is played back. The GUI module also contains some widgets that are used to display specific information. They should be self-explanatory. 

### Plotting
The plotting module contains scripts that can provide insight into the recorded data. The file `compare_conditions.py` loads all experiment data and was used to create the CRT box plot in the paper. Besides that plot, it can also generate plots of other metrics on an individual (per driver) and global (per pair) level. Besides plotting metrics, the script can be used to plot raw signals. 

The file `conflict_signal.py` can reproduce the proposed visualization plot and the plot that shows the construction of the level of conflict signal. The `conflict signal GUI shows the same construction of the level of conflict signal but in an interactive way. The file `generate_headway_bounds` can be used to determine the bounds of the collision area in the proposed headway visualization. 

The conditions (as shown in the methods document that is stored with the dataset) can be visualized with the script in `plot_condition_plane.py`. The `trial_overview_plot` script will generate the multi-row figure that summarizes a single trial. All of these figures are pre-generated and available with the dataset. Finally, the `load_data_from_file` script is a support script that is used when loading the data.

### Simulation
The simulation module contains two classes that are used when running the simulation (experiment). The simmasters take care of the clock. These classes contain the loop in which all update functions are called. The normal SimMaster class runs the experiment while the PlaybackMaster is used when replaying a recorded trial. The simmasters are also used for saving the experiment data and status (in an autosave). The simmaster is configured to automatically reset the experiment after a trial is completed and to transition to the next experimental condition. To initialize all data storage, the simmasters have a maximum run time. If this maximum time is exceeded, the simulation will automatically stop. 

The `SimulationConstants` class can be used to create simple data objects that hold the parameters of the simulation. These are the dimensions of the track and the vehicles, and the timing parameters.

### Track Objects
The trackobjects module contains the definition of the experiment track. In the experiment, the tunnel track is used. This track consists of the three sections discussed in the paper (tunnel, approach, and car following). There is also a symmetric merging track available, this is a simplified version of the tunnel track that only contains an approach and a car-following section.

The `TrackSide` enum is used to distinguish the track’s left and right approach sides. It is used extensively throughout the code. The `surroundings` class is used to store the definition of the roadside markers, trees, and rocks. These objects serve to provide the participants with cues to perceive velocity. The used surroundings can be generated randomly, but a pickle file is included that contains the surroundings as used in the experiment.   
