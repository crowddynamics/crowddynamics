import unittest
from crowddynamics import main
from crowddynamics.config import Load


class MyTestCase(unittest.TestCase):
    load = Load().yaml("simulations")

    def test_attributes(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
