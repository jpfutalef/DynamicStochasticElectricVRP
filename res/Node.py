import numpy as np
import pandas as pd
import networkx as nx

"""
Classes
"""


class NetworkNode:
    """
    A general network node which has an associated Id
    """

    def __init__(self, node_id=0, spent_time=0, demand=0, pos=(0, 0), color='black', *args, **kwargs):
        """
        Simplest constructor with Id and a zero spent time
        :param node_id: an int number with an associated Id
        """
        self.id = node_id
        self.spent_time = spent_time
        self.demand = demand
        self.pos = pos
        self.color = color

    def spentTime(self, p, q):
        """
        A function all subclasses must implement in order to know how much time it was spent at the node
        :param q: battery SOC of a vehicle reaching the node
        :param p: battery SOC increment of the vehicle leaving the node
        :return: the spent time
        """
        return self.spent_time

    def requiredDemand(self):  # TODO add doc
        return self.demand

    def isChargeStation(self):  # TODO add doc
        return False

    def isDepot(self):  # TODO add doc
        return False

    def isCustomer(self):  # TODO add doc
        return False

    def getTypeAbbreviation(self):  # TODO add doc
        return 'NN'


class DepotNode(NetworkNode):  # TODO add documentation
    def __init__(self, node_id, *args, **kwargs):
        super().__init__(node_id, color='lightskyblue', *args, **kwargs)

    def isDepot(self):  # TODO add doc
        return True

    def getTypeAbbreviation(self):  # TODO add doc
        return 'DEPOT'


class CustomerNode(NetworkNode):  # TODO add documentation
    def __init__(self, node_id, spent_time, demand, time_window_up=None, time_window_down=None, *args, **kwargs):
        super().__init__(node_id, spent_time=spent_time, demand=demand, color='limegreen', *args, **kwargs)
        self.timeWindowDown = time_window_down
        self.timeWindowUp = time_window_up

    def spentTime(self, p, q):
        return self.spent_time

    def getTypeAbbreviation(self):  # FIXME maybe as an instance parameter?
        return 'C'

    def isCustomer(self):
        return True


class ChargeStationNode(NetworkNode):
    # TODO add documentation
    # TODO unit test
    def __init__(self, node_id, maximum_parallel_operations=4, time_points=(0.0, 120.0),
                 soc_points=(0.0, 100.0), *args, **kwargs):
        super().__init__(node_id, color='goldenrod', *args, **kwargs)
        self.maximumParallelOperations = maximum_parallel_operations
        self.timePoints = time_points
        self.socPoints = soc_points

    def calculateTimeSpent(self, init_soc, end_soc):
        # FIXME actually, verify it is working properly
        # TODO verify complexity
        # TODO add documentation
        doInit = True
        doEnd = False
        initIndex = 0
        endIndex = 0
        for i, ai in enumerate(self.socPoints):
            if doInit and ai >= init_soc:
                initIndex = i - 1
                doInit = False
                doEnd = True
            if doEnd and ai >= end_soc:
                endIndex = i - 1
                break
        # init time
        m = (self.socPoints[initIndex + 1] - self.socPoints[initIndex]) / (self.timePoints[initIndex + 1] -
                                                                           self.timePoints[initIndex])
        n = self.socPoints[initIndex] - m * self.timePoints[initIndex]
        initTime = (init_soc - n) / m

        # end time
        m = (self.socPoints[endIndex + 1] - self.socPoints[endIndex]) / (self.timePoints[endIndex + 1] -
                                                                         self.timePoints[endIndex])
        n = self.socPoints[endIndex] - m * self.timePoints[endIndex]
        endTime = (end_soc - n) / m
        return endTime - initTime

    def spentTime(self, init_soc, increment):
        return self.calculateTimeSpent(init_soc, init_soc + increment)

    def isChargeStation(self):
        return True

    def getTypeAbbreviation(self):
        return 'CS'