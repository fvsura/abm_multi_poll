# abm_multi_poll
Collection of tools to create a multilayer network using an agent based model. Diffusion of ideas as well as complex interaction are implemented. Can be used to simulate voting scenarios and polls. 

Here you can find all the modules used in support of the paper "Systematic errors in polls: why we can't capture society's complexity yet".

All code is written in Python3, using Jupyter platform.

The following files are for support in the main simulations:

Node.py contains the class used in reating the agents

Networktool.py contains a number of useful tools, from checking if two nodes share common beliefs to drawing multi-layer netwrom in a fancy way

GenerativeTool.py contains the implementations of the algorithms used to create the multi-layer networks

VoteTool.py contains a number of useful function to implement complex voting procedures as well as some distortion effect in polls

The following files have been used to simulate and measure some of the results in the paper:

ConfDiffEffect.ipynb implent the study of conforming and difforming ideas in polls

BelDistAnalysis.ipynb implement the study of variation in beliefs due to ideas diffusion

ChiTest2.ipynb implment the measure of chisquared related to the diffusion of votes in a single layer and on the projection network

PollTry3.ipynb implement a simple polling 

ACB2-analysis-StepCreation-Copy2.ipynb implement the creation and analysis of a multi-layer network with an anglorithm that compare beliefs and create a new node at each step

PowerLawTest-ACB.ipynb, PowerLawTest-AFF.ipynb implement the analysis for power-law distribution in generating multi-layer networks with different algorithms
