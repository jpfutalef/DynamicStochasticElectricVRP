import models.Network as net

from numpy import array


if __name__ == '__main__':
    tech1 = {0.0: 0.0, 75.6: 85.0, 92.4: 95.0, 122.4: 100.0}  # slow
    tech2 = {0.0: 0.0, 37.2: 85.0, 46.2: 95.0, 60.6: 100.0}  # normal
    tech3 = {0.0: 0.0, 18.6: 85.0, 23.4: 95.0, 30.6: 100.0}  # fast
    n = net.from_xml('../data/instances/CS_capacity/c50cs5_30x30km.xml', True)
    for cs in n.charging_stations:
        n.nodes[cs].setTechnology(tech3, 3, 70.*2.5)
    n.write_xml('../data/instances/CS_capacity/c50cs5_30x30km_fast.xml', True)
