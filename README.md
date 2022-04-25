# Simple Merging
This project contains a simulation environment for two vehicle in a simplified merging scenario. The scenario is symmetrical and the vehicles are 
modelled as point mass objects that can move in a single dimension. The acceleration of these vehicles can be controller either by computational agents or 
humans. This simulation environment was created to study the Communication-Enabled Interaction Model (CEI Model). 

It is also possible to run the simulation for just one vehicle, but this vehicle should be the left vehicle in the track. The right vehicle can be omitted. 

## CEI Model
The CEI model is a model of traffic interactions that explicitly accounts for the interaction between traffic participants. Besides that, it assumes that 
humans are not rational utility maximizers. Instead, it assumes that humans initialize behavior and keep on executing their current plan until their 
perceived risk either exceeds an upper threshold or is below a lower threshold for an extended amount of time.

## Status of the repository
This repository is still under heavy development. The documentation is very concise and the functionality may change heavily. 

## Run instructions
At the moment, there are three run scripts. `main.py` runs an online simulation. This works fine with simple agents, like the pd-agent that keeps a constant 
velocity, of with human input (e.g. mouse-agent or keyboard-agent). But the current implementation of the CEI model is too slow to run in real time. For 
that reason there are a `simulate_offline.py` and `playback.py` script to simulate and then playback the simulation results. Before running any offline 
simulations please make sure to create a folder called `data` in the main project folder.

All different agents can be found in the `agents` module. At this point the simulation environment only supports one track and one type of controllable object. 