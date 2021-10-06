# A GA-based Decision-making system for managing EV fleets

## About

 
## Usage
### Before running the scripts

Check the following aspects:

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

``STAGE`` sets the operation to execute. Select one of the following:

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

``TARGET_FOLDER`` specifies the working folder.

#### The working folder

#### Relevant optional arguments
``--optimize`` if passed, it enables OnGA during the online stage.

``--pre_op_repetitions [N]`` number of pre-operation runs. Yields N
pre-operation solutions. Default: 5

``--num_of_simulations [N]`` number of online stage simulations. Default: 20

### The hyper_main.py file


## Examples
Folder *data/* contains a few instances to test the proper working of 
the code. We avoid storing all instances in this repo to lower the storing
space needed. The rest of the instances can be found at *(VRP REPO)*.

### Pre-operation stage


### Online stage

### Full day

## Repository structure

 



