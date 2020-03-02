import unittest
import res.Network
import res.Node


class TestNetwork(unittest.TestCase):
    def setUp(self) -> None:
        depotNode = res.Node.DepotNode(0)

        self.network = res.Network
