import unittest
from datetime import datetime, timedelta

from engine import HedgeEngine
from portfolio import Portfolio


class TestEngineMethods(unittest.TestCase):

    def test_step(self):
        engine = HedgeEngine(Portfolio())
        engine.set_t_start(datetime(2016, 12, 31))
        engine.step()
        t_curr = engine.get_t_curr()
        self.assertEqual(datetime(2017,1,1), t_curr)
