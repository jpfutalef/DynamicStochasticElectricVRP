import models.OnlineFleet as fleet
if __name__ == '__main__':
    path = '../data/instances/CS_capacity/c50cs5_30x30km.xml'
    f = fleet.from_xml(path, False, False, True, False)

    tech1 = {'description': {0.0: 0.0, 75.6: 85.0, 92.4: 95.0, 122.4: 100.0}, 'type': 1, 'price': 70.}  # slow
    tech2 = {'description': {0.0: 0.0, 37.2: 85.0, 46.2: 95.0, 60.6: 100.0}, 'type': 2, 'price': 70.*1.5}  # normal
    tech3 = {'description': {0.0: 0.0, 18.6: 85.0, 23.4: 95.0, 30.6: 100.0}, 'type': 3, 'price': 70.*2.5}  # fast

    techs = [tech1, tech2, tech3]
    for tech in techs:
        for cs in f.network.charging_stations:
            f.network.nodes[cs].setTechnology(tech)

        tech_name = 'slow' if tech['type']==1 else 'normal' if tech['type']==2 else 'fast'
        save_to = f'{path[:-4]}_{tech_name}.xml'
        f.write_xml(save_to, True, print_pretty=True)