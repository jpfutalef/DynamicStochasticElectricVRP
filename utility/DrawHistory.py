from pathlib import Path
import res.simulator.History as History
import res.models.Network as Network
import res.models.Fleet as Fleet

history_path = Path('../data/test/online/23-20-24_06-09-21/history.xml')
network_path = Path('../data/test/online/source/network.xml')
fleet_path = Path('../data/test/online/source/fleet.xml')
figsize = (16, 5)

if __name__ == '__main__':
    fleet = Fleet.from_xml(fleet_path)
    network = Network.from_xml(network_path)
    fleet_history = History.FleetHistory.from_xml(history_path)
    figs, info = fleet_history.draw_events(network, fleet, figsize=figsize)

    for f in figs:
        f.show()
    pass
