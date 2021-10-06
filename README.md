# A GA-based Decision-making system for managing EV fleets

## About

This repository contains plenty of python code that implements a GA-based
closed-looop decision-making strategy to solve multiple E-VRP variants. The
code also implements a simple simulation environment to test the solutions,
considering both open and closed loop versions.
 
## Usage
### Before running the scripts

Check the following:

1. The python version is 3.7 (other versions haven't been tested).
2. If using Anaconda:
    * Check that the conda environment provides all packages listed in the 
      file *requirements.txt*. If it doesn't, you can install them by running 
      ``conda install  --file requirements.txt``

### The main.py file

The file *main.py* can be used to run the two operational stages independently
(you run the stages manually) or in a single execution 
(the script automatically runs pre-operation, and then, online). For the last
case, the script detects if the pre-operation was run in the past; if so,
the script doesn't runt it again.

The script runs as follows

``python main.py [optional arguments] STAGE OPT_METHOD TARGET_FOLDER``

where:

``STAGE`` sets the stage to execute. Select one of the following:

- 1: pre-operation
- 2: online simulations
- 3: pre-operation\*, followed by online simulations (* skips it if already
  executed in the past)

``OPT_METHOD`` sets the E-VRP variant the optimizer solves during the
pre-operation and online stages. Not considered when simulating the open loop
online stage.

- 1: deterministic
- 2: deterministic + waiting times
- 3: linear stochastic

``TARGET_FOLDER`` specifies the target folder.

#### The target folder

Contains the necessary data to execute any stage. The folder must contain one
or more instances defined as XML files. The script will perform the specified
stage over each of the instances in the target folder. 

The following directory structure shows how the results will be stored if 
the target folder contains a single instance:

```
target_folder
|   instance_name.xml
└───instance_name
    └───optimization_method_1
        └───pre_operation
            └───preop_results1
                |   preop result files
           ...
            └───preop_resultsN
                |   preop result files
        └───online_open_loop
            └───online_results1
                |   online result files
           ...
            └───online_resultsN
                |   online result files
        └───online_closed_loop
            └───online_results1
                |   online result files
           ...
            └───online_resultsN
                |   online result files
    └───optimization_method_2
            ...
   ...
    └───optimization_method_N
            ...
```

#### Relevant optional arguments
``--optimize`` if passed, it enables OnGA during the online stage.

``--preop_repetitions [N]`` number of pre-operation runs. Yields N
pre-operation solutions. Default: 5

``--online_simulations [N]`` number of online stage simulations. Default: 20

### The hyper_main.py file

## Examples
Folder *data/* contains a few instances to test the proper working of 
the code. We avoid storing all instances in this repo to lower the storing
space needed. The rest of the instances can be found at *(VRP REPO)*.

### Pre-operation stage

The folder ``batch1, batch2, batch3`` in ``data/instances/`` contains a set 
of ready-to-solve instances. Pick one of the three batch folders 
(e.g batch1) and do

``python main.py 1 1 data/instances/batch1/``

This code performs an offline optimization over all instances in 
``data/instances/batch1/`` considering a deterministic E-VRP.

If you want to consider a deterministic E-VRP with waiting times variant, 
modify the second mandatory argument:

``python main.py 1 2 data/instances/batch1/``

Similarly, to consider the linear stochastic E-VRP variant:

``python main.py 1 3 data/instances/batch1/``

### Online stage

### Full day

## Repository structure

 



