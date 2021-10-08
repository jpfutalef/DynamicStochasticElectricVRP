# A GA-based Decision-making system for managing EV fleets

## About

This repository contains plenty of python code that implements a GA-based
decision-making strategy to manage EV fleets. The system consists
of: an offline and online optimizer to solve multiple E-VRP variants, and a 
simple simulator to test the strategy.The goal is to analyze the behavior of 
the EV routes during a single day via simulation.
 
A day is divided into two stages: **pre-operation** and **online**. In the 
**pre-operation** stage, an *offline solver* runs to calculate initial EV route
candidates. Then, in the **online** stage, a simulation is run to simulate 
semi-real-world environments. In this case, you can choose either to maintain the
initial routes throughout the stage (open-loop), or to update them using
an *online solver* (closed-loop).

### E-VRP variants

This repo implements three solvers, each considering a different E-VRP variants:
* Deterministic E-VRP
* Deterministic E-VRP with waiting times
* Linear stochastic E-VRP

## Installation

Clone the repo by executing

``git clone https://github.com/jpfutalef/DynamicStochasticElectricVRP.git``


## Code Usage
### Before running the scripts

Check the following:

1. The python version is 3.7 (other versions haven't been tested).
   
2. Update the repository in your computer by doing ``git fetch`` and then
``git pull``.

3. Check your Python executable provides all pacagesin *requirements.txt*.
    * If using Anaconda: you can install them by running 
      ``conda install  --file requirements.txt``
      

### The ``main.py`` file

``main.py`` can be used to run the operational stages. You can choose which 
operation you wish to run, or simulate a full day (run pre-operation and online
in a single execution).

The code syntax is as follows:

```commandline 
python main.py [optional arguments] STAGE OPT_METHOD TARGET_FOLDER
```

where:

``STAGE`` sets the stage to execute. Select one of the following:

- 1: pre-operation stage
- 2: online stage. It is open-loop by default. You can pass the ``--optimize``
  optional argument to activate the closed-loop strategy.
- 3: full day (pre-operation + online).

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

Under construction ....

## Examples

Folder *data/* contains a few instances to test the proper working of 
the code. We avoid storing all instances in this repo to lower the storing
space needed. The rest of the instances can be found at *(VRP REPO)*.

### Pre-operation stage

Folders ``set1, set2, set3`` in ``data/instances/`` contain sets 
of ready-to-solve instances. If you want to solve a set of instances, choose
one of the following:

```
python main.py 1 1 data/instances/set1/ <-- deterministic
python main.py 1 2 data/instances/set1/ <-- deterministic + waiting times
python main.py 1 3 data/instances/set1/ <-- linear stochastic
```

All results will be stored in folders with the same names of the
instances. For example, for instance ``c10_cs1_15x15km``, 
the folder ``data/instances/set1/c10_cs1_15x15km/`` will be created. See the
*Target folder* structure explained above to know how results are stored.

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

 ``./res`` contains all resource files. Here, you will find all methods and 
 classes necessary to run the system.

```./notebooks``` contains user-friendly Jupyter Notebooks to generate instances
and visualize results. Make sure to run ``jupyter notebook`` or ``jupyter lab``
in the main project folder.

``./data`` stores all data generated by running the system.

``./test`` contains experimental code and may be empty.

``./routines``, ``./simulations``, and ``./utility`` are deprecated
and will soon be removed.



