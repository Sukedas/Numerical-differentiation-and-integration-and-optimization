import sys
import os
import unittest
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.algorithms import ReferenceFunction, NumericalDifferentiation, NumericalIntegration, Optimization

class TestNumericalMethods(unittest.TestCase):
    def test_differentiation(self):
        # f(x) = x^2, f'(2) should be 4
        ref = ReferenceFunction("x**2")
        val, steps = NumericalDifferentiation.central_difference(ref.evaluate, 2.0, 0.001)
        print(f"Diff x^2 at 2: {val}")
        self.assertAlmostEqual(val, 4.0, places=3)

    def test_integration(self):
        # Integral of x from 0 to 1 is 0.5
        ref = ReferenceFunction("x")
        val, steps = NumericalIntegration.trapezoidal_rule(ref.evaluate, 0.0, 1.0, 100)
        print(f"Int x [0,1]: {val}")
        self.assertAlmostEqual(val, 0.5, places=3)

    def test_optimization(self):
        # Root of x^2 - 4 near 1 should be 2
        ref = ReferenceFunction("x**2 - 4")
        val, steps = Optimization.newton_raphson(ref, 1.0)
        print(f"Root x^2-4: {val}")
        self.assertAlmostEqual(val, 2.0, places=3)

if __name__ == '__main__':
    unittest.main()
