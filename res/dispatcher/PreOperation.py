import os

from typing import List, Union

from res.models.Fleet import from_xml
from optimizer.GATools import HyperParameters, RouteDict
from optimizer.alphaGA import alphaGA
from optimizer.betaGA import betaGA


def pre_operation(filepath: Union[str, List[str, ...]] = None, folder: str = None) -> None:
    """
    Find initial routes for specified instances in XML format. There are three options: 1) specify a unique instance
    via filepath 2) specify a folder path and pass None to filepath, which optimizes all instances in the folder
    3) specify a folder, and a single or a collection of instance names in filepath, which will optimize only those
    instances in the folder. Saves results in a folder parallel to the XML file containing the instance.
    :param filepath: single or collection of filepath. If None passed, folder must not be None. Default: None
    :param folder: folder containing one or more instances. If None passed, it is ignored. Default: None.
    """
    if folder:
        a = 10
    return


def pre_operation_single_instance(filepath: str) -> RouteDict:
    return


# %% 1. Specify instances location
folder = 'data/instances/'
# instances = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
instances = ['c25cs3_10x10km.xml']

# %% 2. CS capacities and SOC policy
cs_capacity = 3
soc_policy = (20, 95)

# %% 3. Solve instances
for instance in instances:
    fleet = from_xml(folder + instance, False, False, True, False)

    fleet.modify_cs_capacities(cs_capacity)
    fleet.new_soc_policy(soc_policy[0], soc_policy[1])

    # %% 4. GA hyper-parameters
    num_individuals = int(len(fleet.network) * 1.5) + int(len(fleet) * 10) + 50
    K1 = 100. * len(fleet.network) + 1000 * len(fleet)
    hp_alpha = HyperParameters(num_individuals=num_individuals,
                               max_generations=num_individuals,
                               CXPB=0.65,
                               MUTPB=0.8,
                               weights=(0.1 / 2.218, 1. / 0.4364, 1. / 100, 1. / 500, 1. * 0.5),
                               K1=K1,
                               K2=K1 * 2.5,
                               keep_best=1,
                               tournament_size=3,
                               r=4,
                               alpha_up=soc_policy[1],
                               algorithm_name='alphaGA')

    num_individuals = int(len(fleet.network) * 1.5) + int(len(fleet) * 10) + 50
    K1 = 100. * len(fleet.network) + 1000 * len(fleet)
    hp_beta = HyperParameters(num_individuals=num_individuals,
                              max_generations=num_individuals * 3,
                              CXPB=0.7,
                              MUTPB=0.9,
                              weights=(0.1 / 2.218, 1. / 0.4364, 1. / 100, 1. / 500, 1. * 0.5),
                              K1=K1,
                              K2=K1 * 2.5,
                              keep_best=1,
                              tournament_size=3,
                              r=2,
                              alpha_up=soc_policy[1],
                              algorithm_name='betaGA',
                              crossover_repeat=2,
                              mutation_repeat=2)

    # %% 5. Specify data folder
    # Main instance folder
    instance_folder = instance[:-4] + '/'
    try:
        os.mkdir(instance_folder)
    except FileExistsError:
        pass

    # Main optimization folder
    opt_folders = [os.path.join(instance_folder, d) for d in os.listdir(instance_folder) if
                   os.path.isdir(os.path.join(instance_folder, d))]
    if opt_folders:
        opt_num = str(max([int(i.split('_')[-1]) for i in opt_folders if os.path.isdir(i)]) + 1)
    else:
        opt_num = '1'
    opt_folder = instance_folder + f'opt_{opt_num}/'
    try:
        os.mkdir(opt_folder)
    except FileExistsError:
        pass

    # %% 6. Run algorithm
    best_alpha = None
    best_beta = None
    mi = None
    # mi = int(sum([fleet.network.demand(i) for i in fleet.network.customers])/fleet.vehicles[0].max_payload) + 1
    for k in range(8):
        routes_alpha, opt_data_alpha, toolbox_alpha = alphaGA(fleet, hp_alpha, opt_folder,
                                                              best_ind=best_alpha,
                                                              savefig=True,
                                                              mi=mi,
                                                              plot_best_generation=False)
        routes_beta, opt_data_beta, toolbox_beta = betaGA(fleet, hp_beta, opt_folder,
                                                          best_ind=best_beta,
                                                          savefig=True)
        mi = opt_data_alpha.additional_info['mi']

        if not opt_data_alpha.acceptable and not opt_data_beta.acceptable:
            # Not feasible
            print('INCREASING FLEET SIZE BY 1...')
            mi += 1
            hp_alpha.num_individuals += 15
            hp_beta.num_individuals += 10
            hp_alpha.max_generations += 10
            hp_beta.max_generations += 10
        else:
            # At least one is acceptable
            break
