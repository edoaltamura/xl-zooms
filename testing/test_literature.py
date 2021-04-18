import unittest

from ..literature import *


class TestSum(unittest.TestCase):
    def run_test(self):
        Barnes2017()
        Bohringer2007()
        Budzynski2014()
        Eckert2016()
        Gonzalez2013()
        Kravtsov2018()
        Lin2012()
        Lovisari2015()
        PlanckSZ2015()
        Pratt2010()
        Sun2009()
        Vikhlinin2006()


if __name__ == '__main__':
    unittest.main()
