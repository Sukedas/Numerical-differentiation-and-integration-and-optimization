import sympy as sp
import numpy as np
import pandas as pd

class ReferenceFunction:
    """
    Handles parsing and evaluation of a user-defined function.
    """
    def __init__(self, func_str):
        self.func_str = func_str
        self.x = sp.symbols('x')
        try:
            self.expr = sp.sympify(func_str)
            self.f = sp.lambdify(self.x, self.expr, 'numpy')
        except Exception as e:
            raise ValueError(f"Invalid function string: {e}")

    def evaluate(self, x_val):
        return self.f(x_val)
    
    def get_symbolic_derivative(self):
        return sp.diff(self.expr, self.x)

class NumericalDifferentiation:
    """
    Provides methods for numerical differentiation.
    """
    @staticmethod
    def forward_difference(func, x, h):
        fx = func(x)
        fxh = func(x + h)
        derivative = (fxh - fx) / h
        steps = [
            f"f(x) = {fx}",
            f"f(x + h) = f({x} + {h}) = {fxh}",
            f"f'(x) ≈ (f(x+h) - f(x)) / h",
            f"f'(x) ≈ ({fxh} - {fx}) / {h}",
            f"f'(x) ≈ {derivative}"
        ]
        return derivative, steps

    @staticmethod
    def backward_difference(func, x, h):
        fx = func(x)
        fx_h = func(x - h)
        derivative = (fx - fx_h) / h
        steps = [
            f"f(x) = {fx}",
            f"f(x - h) = f({x} - {h}) = {fx_h}",
            f"f'(x) ≈ (f(x) - f(x-h)) / h",
            f"f'(x) ≈ ({fx} - {fx_h}) / {h}",
            f"f'(x) ≈ {derivative}"
        ]
        return derivative, steps

    @staticmethod
    def central_difference(func, x, h):
        fxh = func(x + h)
        fx_h = func(x - h)
        derivative = (fxh - fx_h) / (2 * h)
        steps = [
            f"f(x + h) = f({x} + {h}) = {fxh}",
            f"f(x - h) = f({x} - {h}) = {fx_h}",
            f"f'(x) ≈ (f(x+h) - f(x-h)) / 2h",
            f"f'(x) ≈ ({fxh} - {fx_h}) / {2 * h}",
            f"f'(x) ≈ {derivative}"
        ]
        return derivative, steps

class NumericalIntegration:
    """
    Provides methods for numerical integration.
    """
    @staticmethod
    def trapezoidal_rule(func, a, b, n):
        h = (b - a) / n
        x_vals = np.linspace(a, b, n+1)
        y_vals = func(x_vals)
        
        result = h * (0.5 * y_vals[0] + np.sum(y_vals[1:-1]) + 0.5 * y_vals[-1])
        
        steps = []
        steps.append(f"Interval [{a}, {b}], n = {n}, h = ({b} - {a}) / {n} = {h}")
        steps.append("x values: " + ", ".join([f"{val:.4f}" for val in x_vals]))
        steps.append("y values: " + ", ".join([f"{val:.4f}" for val in y_vals]))
        steps.append(f"Integral ≈ {h} * [0.5 * {y_vals[0]:.4f} + ({' + '.join([f'{y:.4f}' for y in y_vals[1:-1]])}) + 0.5 * {y_vals[-1]:.4f}]")
        steps.append(f"Integral ≈ {result}")
        
        return result, steps

    @staticmethod
    def simpsons_rule(func, a, b, n):
        if n % 2 != 0:
            raise ValueError("n must be even for Simpson's 1/3 rule")
        
        h = (b - a) / n
        x_vals = np.linspace(a, b, n+1)
        y_vals = func(x_vals)
        
        sum_odd = np.sum(y_vals[1:-1:2])
        sum_even = np.sum(y_vals[2:-2:2])
        
        result = (h / 3) * (y_vals[0] + 4 * sum_odd + 2 * sum_even + y_vals[-1])
        
        steps = []
        steps.append(f"Interval [{a}, {b}], n = {n} (even), h = {h}")
        steps.append(f"Sum of odd terms (x1, x3...): {sum_odd}")
        steps.append(f"Sum of even terms (x2, x4...): {sum_even}")
        steps.append(f"Integral ≈ ({h}/3) * [{y_vals[0]:.4f} + 4*({sum_odd:.4f}) + 2*({sum_even:.4f}) + {y_vals[-1]:.4f}]")
        steps.append(f"Integral ≈ {result}")
        
        return result, steps

class Optimization:
    """
    Provides methods for finding roots/optima.
    """
    @staticmethod
    def newton_raphson(func_obj, x0, tol=1e-6, max_iter=100):
        # Requires symbolic derivative for exact Newton-Raphson, 
        # or we could use numerical derivative. using symbolic for better precision here as we have sympy.
        
        deriv_expr = func_obj.get_symbolic_derivative()
        f_prime = sp.lambdify(func_obj.x, deriv_expr, 'numpy')
        
        steps = []
        xi = x0
        
        steps.append(f"Initial guess x0 = {xi}")
        steps.append(f"f'(x) = {deriv_expr}")
        
        for i in range(max_iter):
            f_val = func_obj.evaluate(xi)
            fp_val = f_prime(xi)
            
            if abs(fp_val) < 1e-10:
                steps.append("Derivative too close to zero. Divergence risk.")
                break
                
            x_next = xi - f_val / fp_val
            
            steps.append(f"Iter {i+1}: x_{i+1} = {xi:.6f} - ({f_val:.6f} / {fp_val:.6f}) = {x_next:.6f}")
            
            if abs(x_next - xi) < tol:
                xi = x_next
                steps.append(f"Converged to {xi} after {i+1} iterations")
                return xi, steps
            
            xi = x_next
            
        return xi, steps

    @staticmethod
    def gradient_descent(func_obj, x0, learning_rate=0.1, tol=1e-6, max_iter=100):
        # Finding MINIMUM
        deriv_expr = func_obj.get_symbolic_derivative()
        f_prime = sp.lambdify(func_obj.x, deriv_expr, 'numpy')
        
        steps = []
        xi = x0
        steps.append(f"Finding minimum. Initial guess x0 = {xi}, Learning R = {learning_rate}")
        
        for i in range(max_iter):
            grad = f_prime(xi)
            x_next = xi - learning_rate * grad
            
            steps.append(f"Iter {i+1}: x_{i+1} = {xi:.6f} - {learning_rate} * {grad:.6f} = {x_next:.6f}")
            
            if abs(x_next - xi) < tol:
                xi = x_next
                steps.append(f"Converged to minimum at {xi} after {i+1} iterations")
                return xi, steps
                
            xi = x_next
            
        return xi, steps
