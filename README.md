# A GA-based Decision-making system for managing EV fleets

## About

This repository contains plenty of python code that implements a GA-based
closed-looop decision-making strategy to solve multiple E-VRP variants. The
code also implements a simple simulation environment to test the solutions,
considering both open and closed loop versions.

 
## Code Usage
### Before running the scripts

Check the following:

1. The python version is 3.7 (other versions haven't been tested).
2. If using Anaconda:
    * Check that the conda environment provides all packages listed in the 
      file *requirements.txt*. If it doesn't, you can install them by running 
      ``conda install  --file requirements.txt``

### The ``main.py`` file

``main.py`` can be used to run the operational stages independently
(you run the stages manually) or in a single execution 
(the script automatically runs pre-operation, and then, online). For the last
case, the script detects if the pre-operation was run in the past. If so,
the script doesn't runt it again.

Run the code by executing

```commandline 
python main.py [optional arguments] STAGE OPT_METHOD TARGET_FOLDER
```

where:

``STAGE`` sets the stage to execute. Select one of the following:

- 1: pre-operation
- 2: online simulations
- 3: full day (pre-operation + online. Doesn't do pre-operation if it exists)

``OPT_METHOD`` sets the E-VRP variant the optimizer solves during the
pre-operation and online stages. Not considered when simulating the open loop
online stage.

- 1: deterministic
- 2: deterministic + waiting times
- 3: linear stochastic

``TARGET_FOLDER`` specifies the target folder. For pre-operation stage and full 
day, a folder containing a set of instances (XML files). For online stage, 
a folder containing the results from the pre-operation.

#### The target folder

Contains the necessary data to execute any stage. Essentially, each instance
is assigned a folder with its name. Inside each folder, you will find the
results for each stage/method you choose.

The folder structure for a folder containing one instance is as follows:

```
main_folder  <-- target folder for pre-operation
|   instance_name.xml
└───instance_name
    └───optimization_method_1
        └───pre_operation
            └───preop_results1  <-- could be source folder for online operation
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

Folders ``set1, set2, set3`` in ``data/instances/`` contain sets 
of ready-to-solve instances. If you want to solve a set of instances

```
python main.py 1 1 data/instances/set1/ <-- deterministic
python main.py 1 2 data/instances/set1/ <-- deterministic + waiting times
python main.py 1 3 data/instances/set1/ <-- linear stochastic
```

All results will be stored in folders with the same names of the
instances. For example, for instance ``c10_cs1_15x15km``, 
the folder ``data/instances/set1/c10_cs1_15x15km/`` will be created.

### Online stage

In this example, we solve and simulate the deterministic E-VRP variant.
Folder ``data/online/`` contains the ready-to-use instance 
``c20_cs2_15x15km``.

First, execute the pre-operation stage to 
generate the initial route candidates. To do this, run one of the
following:

```
python main.py 1 1 data/online/ <-- deterministic
python main.py 1 2 data/online/ <-- deterministic + waiting times
python main.py 1 3 data/online/ <-- linear stochastic
```

The directory ``data/online/c20_cs2_15x15km/`` will be created. Inside
it, you'll find the directory ``/pre-operation/[optimization method name]``.
Inside it, you will find several folders containing the
offline optimization results (5 by default), and the file 
``source_folder.txt`` that contains the folder with the best 
optimization result. Copy that directory and simulate the closed loop online
stage by executing:

```
python main.py --optimize 2 1 [best directory] <-- closed loop
```

To simulate the open loop online stage, remove the ``--optimize`` argument:

```
python main.py 2 1 [best directory] <-- open loop
```

In this case, the second argument (i.e., 1) doesn't do anything.

### Full day

If you want to avoid running both stages independently, you can use the third
STAGE option to run both stages sequentially. The following script will do this
automatically over instances at ``data/instances/set2/`` considering 
the deterministic E-VRP variant:

```
python main.py 3 1 data/instances/set2/ <-- open loop
python main.py --optimize 3 1 data/instances/set2/ <-- closed loop
```

The code detects if the pre-operation was run by checking if the directory 
``data/instances/set2/[instance_name]/pre_operation/`` exists and is not empty
for each instance in ``data/instances/set2/``.


## Repository structure

 



