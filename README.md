# A GA-based Decision-making system for managing EV fleets

## About

This repository implements a GA-based
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

## Paper Reference
J.P. Futalef, D. Muñoz-Carpintero, H. Rozas, M. E. Orchard,
_An online decision-making strategy for routing of electric vehicle fleets_,
Information Sciences,
Volume 625,
2023,
Pages 715-737,
DOI: 10.1016/j.ins.2022.12.108.
[Link](https://www.sciencedirect.com/science/article/pii/S0020025522016036)

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

3. Check your Python executable provides all packages in the
   *requirements.txt* file.
    * If using Anaconda: you can install them by running 
      ``conda install  --file requirements.txt``
      

### The ``main.py`` file

``main.py`` can be used to run the operational stages. You can choose which 
operation you wish to run, or simulate a full day (run pre-operation and online
in a single execution).

The code syntax is as follows:

```commandline 
python main.py [optional arguments] STAGE TARGET_FOLDER
```

where:

``STAGE`` sets the stage to execute. Select one of the following:

- 1: pre-operation stage
- 2: online stage. It is open-loop by default. You can pass the ``--optimize``
  optional argument to activate the closed-loop strategy.
- 3: full day (pre-operation + online).

``TARGET_FOLDER`` specifies the target folder. For pre-operation stage and 
full-day simulation, the target folder must contain a set of instances
which will be solved (and simulated, for the full-day case). For the 
online stage, the target folder is such that it contains the results from
the pre-operation stage.

#### Folder structures

After running the pre-operation (that is, solving the offline E-VRP 
instances), each result is saved in a folder with the same name of the
instance. Inside each folder, you will find the results for the 
stage/method you choose.

The folder structure for a folder containing one instance is as follows:

```
folder_containing_several_instances  <-- target folder for pre-operation
|   instance_name.xml
└───instance_name
    └───evrp_variant_1
        └───pre_operation
            └───preop_results1 <-- can be source folder for online operation
                |   preop result files
           ...
            └───preop_resultsN <-- can be source folder for online operation too
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
    └───evrp_variant_2
            ...
   ...
    └───evrp_variant_N
            ...
```

#### Relevant optional arguments
``--variant`` sets the E-VRP variant the optimizer solves during the
pre-operation and online stages. Choose one of the following 
integers (Default: 1):

- 1: deterministic
- 2: deterministic + waiting times
- 3: linear stochastic

``--optimize`` if passed, it enables OnGA during the online stage.

``--preop_repetitions [N]`` number of pre-operation runs. Yields N
pre-operation solutions. Default: 5

``--online_simulations [N]`` number of online stage simulations. Default: 20

## Examples

Folder *data/instances/* contains a few instances to test the
decision-making system. We avoid storing all instances in this repo to 
lower the storing space needed for this repository. The rest of the 
instances can be found at *(VRP REPO, not ready yet...)*.

### Pre-operation stage

In this example, we will run the pre-operation stage, considering
instances at ``data/instances/``. To do this, run one of the following:

```
python main.py 1 data/instances/ <-- deterministic
python main.py --variant 2 1 data/instances/ <-- deterministic + waiting times
python main.py --variant 3 1 data/instances/ <-- linear stochastic
```

All results will be stored in folders with the same names of the
instances. For example, for instance ``c10_cs1_15x15km``, 
the folder ``data/instances/c10_cs1_15x15km/`` will be created. See 
the *Folder structures* section above to understand how results are stored.

### Online stage

In this example, we will simulate the online stage for instance 
```c20_cs2_15x15km```. This example considers you already ran the 
pre-operation from the previous example. 

Navigate to 
``data/instances/c20_cs2_15x15km/pre-operation/[EVRP variant name]``.
Inside it, you will find several folders containing the
offline optimization results (5 by default), and the file 
``source_folder.txt`` that contains the folder with the best 
optimization result. You can choose any folder or the one specified in
the ``source_folder.txt`` file to run the online stage.

Once you select a source folder, run the online stage as follows. Notice
that you can pass the ``--optimize`` argument to activate the closed-loop
strategy:

```
python main.py 2 [EVRP variant] [source folder] <-- open loop
python main.py --optimize 2 [EVRP variant] [source folder] <-- closed loop
```

Also, notice that, for the open loop case, the [EVRP variant] argument
does not alter the simulation. However, it is mandatory.

### Full day

If you want to avoid running both stages independently, you can use the third
STAGE option to run both stages sequentially. The following script will do
that over instances at ``data/instances/`` considering 
the deterministic E-VRP variant:

```
python main.py 3 1 data/instances/set2/
```

In this case, the script will run the pre-operation stage, followed by
the open-loop online stage, finishing with the closed-loop online stage.

## Repository organization

 ``./res`` contains all resource files. Here, you will find all methods and 
 classes necessary to run the system.

```./notebooks``` contains user-friendly Jupyter Notebooks to generate instances
and visualize results. Make sure to run ``jupyter notebook`` or ``jupyter lab``
in the main project folder.

``./data`` stores all data generated by the system.

``./test`` contains experimental code and may be empty.

``./docs`` contains documentation files.

``./routines`` and ``./utility`` are deprecated
and will soon be removed.



