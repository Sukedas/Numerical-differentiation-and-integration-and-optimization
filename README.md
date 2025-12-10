# Numerical Methods Solver

This project is a Python-based application that allows users to perform Numerical Differentiation, Integration, and Optimization on custom mathematical functions. It provides step-by-step solutions to help understand the underlying algorithms.

## Features
<img src="./Diagramma de clases.png" width="550" alt="Diagramma de clases">
- **Numerical Differentiation**:
    - Forward Difference
    - Backward Difference
    - Central Difference
- **Numerical Integration**:
    - Trapezoidal Rule
    - Simpson's 1/3 Rule
- **Optimization**:
    - Newton-Raphson (Root Finding)
    - Gradient Descent (Minimum Finding)
- **Step-by-Step Display**: View the calculations for each iteration.
- **Visualization**: Interactive plots using Matplotlib.
## Prerequisites

- Python 3.9+
- Docker (optional)

## Installation & Running Locally

1.  **Clone the repository** (or navigate to the project folder).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```
4.  Open your browser at `http://localhost:8501`.

## Running with Docker

1.  **Build the Docker image**:
    ```bash
    docker build -t numerical-methods-app .
    ```
2.  **Run the container**:
    ```bash
    docker run -p 8501:8501 numerical-methods-app
    ```
3.  Open your browser at `http://localhost:8501`.

## Usage

1.  Select the category (Differentiation, Integration, Optimization) from the sidebar.
2.  Enter your function in Python syntax (e.g., `x**2`, `sin(x)`).
3.  Adjust parameters (start/end points, step size `h`, etc.).
4.  Click **Calculate** to see the result, steps, and plot.


