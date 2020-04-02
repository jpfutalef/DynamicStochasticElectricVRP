# %% 0. Imports

#  work with arguments and script paths

# scientific libraries and utilities
# Visualization tools
# GA library

# EV and network libraries
from models.Network import Network

# Initialize time


# %% 1. Specify real time instance folder
instance_name = 'test_instance'
folder_path = '../data/GA_implementation_xml/' + instance_name + '/'
file_path = folder_path + instance_name + '.xml'
file_path_realtime = folder_path + instance_name + '_realtime.xml'
print('OFFLINE file path:', file_path)
print(' ONLINE file path:', file_path_realtime)

# %% 2. Instance Network with initial values
network = Network()
network.from_xml(file_path)

# %% 3. Instantiate EVs
fleet = models.ElectricVehicle.from_xml()

# %% 4. GA hyperparameters


# %% 5. Setup GA


# %% 6. Run GA
while False:
    network.updateNetwork()




# %% 7. End GA: All vehicles have reached their assigned customers_per_vehicle
print("################  End of (successful) evolution  ################")

# %% 8. Show report
