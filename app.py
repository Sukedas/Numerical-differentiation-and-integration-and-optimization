import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from src.algorithms import ReferenceFunction, NumericalDifferentiation, NumericalIntegration, Optimization

# Page Config
st.set_page_config(page_title="Numerical Methods Solver", layout="wide")

st.title("🧮 Numerical Methods Solver")
st.markdown("""
This app solves **Differentiation**, **Integration**, and **Optimization** problems step-by-step using numerical methods.
Enter your function in terms of `x` (e.g., `x**2`, `sin(x)`, `exp(x)`).
""")

# Sidebar
category = st.sidebar.selectbox("Select Category", ["Differentiation", "Integration", "Optimization"])

# Common Input
st.sidebar.header("Function Input")
func_input = st.sidebar.text_input("Enter Function f(x)", value="x**2")

try:
    ref_func = ReferenceFunction(func_input)
    # Symbolic preview
    st.sidebar.latex(f"f(x) = {sp.latex(ref_func.expr)}")
except Exception as e:
    st.error(f"Error parsing function: {e}")
    st.stop()

# Handling Logic
if category == "Differentiation":
    st.header("Numerical Differentiation")
    
    diff_method = st.selectbox("Method", ["Forward Difference", "Backward Difference", "Central Difference"])
    
    col1, col2 = st.columns(2)
    with col1:
        x_val = st.number_input("Point (x)", value=2.0)
    with col2:
        h_val = st.number_input("Step size (h)", value=0.01, format="%.4f")

    if st.button("Calculate Derivative"):
        try:
            val = 0
            steps = []
            if diff_method == "Forward Difference":
                val, steps = NumericalDifferentiation.forward_difference(ref_func.evaluate, x_val, h_val)
            elif diff_method == "Backward Difference":
                val, steps = NumericalDifferentiation.backward_difference(ref_func.evaluate, x_val, h_val)
            elif diff_method == "Central Difference":
                val, steps = NumericalDifferentiation.central_difference(ref_func.evaluate, x_val, h_val)
            
            st.success(f"Result: {val:.6f}")
            
            with st.expander("Show Step-by-Step Calculation"):
                for s in steps:
                    st.write(s)
            
            # Plotting
            x_plot = np.linspace(x_val - 2, x_val + 2, 100)
            y_plot = ref_func.evaluate(x_plot)
            fig, ax = plt.subplots()
            ax.plot(x_plot, y_plot, label='f(x)')
            ax.scatter([x_val], [ref_func.evaluate(x_val)], color='red', label=f'Point ({x_val}, {ref_func.evaluate(x_val):.2f})')
            # Plot tangent line approximation
            tangent_y = ref_func.evaluate(x_val) + val * (x_plot - x_val)
            ax.plot(x_plot, tangent_y, '--', label='Tangent approx')
            ax.legend()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Calculation Error: {e}")

elif category == "Integration":
    st.header("Numerical Integration")
    
    int_method = st.selectbox("Method", ["Trapezoidal Rule", "Simpson's 1/3 Rule"])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        a_val = st.number_input("Lower Limit (a)", value=0.0)
    with col2:
        b_val = st.number_input("Upper Limit (b)", value=1.0)
    with col3:
        n_val = st.number_input("Segments (n)", value=10, step=1)

    if st.button("Calculate Integral"):
        try:
            val = 0
            steps = []
            if int_method == "Trapezoidal Rule":
                val, steps = NumericalIntegration.trapezoidal_rule(ref_func.evaluate, a_val, b_val, int(n_val))
            elif int_method == "Simpson's 1/3 Rule":
                if n_val % 2 != 0:
                    st.error("n must be even for Simpson's rule.")
                    st.stop()
                val, steps = NumericalIntegration.simpsons_rule(ref_func.evaluate, a_val, b_val, int(n_val))
            
            st.success(f"Result: {val:.6f}")
            
            with st.expander("Show Step-by-Step Calculation"):
                for s in steps:
                    st.write(s)

            # Plotting
            x_plot = np.linspace(a_val, b_val, 100)
            y_plot = ref_func.evaluate(x_plot)
            fig, ax = plt.subplots()
            ax.plot(x_plot, y_plot, label='f(x)')
            ax.fill_between(x_plot, y_plot, alpha=0.3, label='Area')
            ax.legend()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Calculation Error: {e}")

elif category == "Optimization":
    st.header("Optimization (Root Finding / Minimum)")
    
    opt_method = st.selectbox("Method", ["Newton-Raphson (Root)", "Gradient Descent (Minimum)"])
    
    col1, col2 = st.columns(2)
    with col1:
        x0_val = st.number_input("Initial Guess (x0)", value=1.0)
    with col2:
        if opt_method == "Gradient Descent (Minimum)":
            lr_val = st.number_input("Learning Rate", value=0.1, format="%.4f")
        else:
            lr_val = 0.1 # Unused

    if st.button("Locate"):
        try:
            val = 0
            steps = []
            if opt_method == "Newton-Raphson (Root)":
                val, steps = Optimization.newton_raphson(ref_func, x0_val)
            elif opt_method == "Gradient Descent (Minimum)":
                val, steps = Optimization.gradient_descent(ref_func, x0_val, learning_rate=lr_val)
            
            st.success(f"Converged Value: {val:.6f}")
            
            with st.expander("Show Iterations"):
                for s in steps:
                    st.write(s)
            
            # Plotting
            x_range = abs(val - x0_val) * 1.5 + 1
            x_plot = np.linspace(min(x0_val, val) - 1, max(x0_val, val) + 1, 100)
            y_plot = ref_func.evaluate(x_plot)
            fig, ax = plt.subplots()
            ax.plot(x_plot, y_plot, label='f(x)')
            ax.scatter([val], [ref_func.evaluate(val)], color='red', label=f'Found Point ({val:.4f})')
            ax.scatter([x0_val], [ref_func.evaluate(x0_val)], color='green', marker='x', label='Start')
            ax.legend()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Calculation Error: {e}")
