import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set page config
st.set_page_config(
    page_title="Insurgency Dynamics Model",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Palantir-like theme (dark green and black)
st.markdown("""
<style>
    .main {
        background-color: #0a0a0a;
        color: #e8f5e8;
    }
    .sidebar .sidebar-content {
        background-color: #1a1a1a;
        color: #e8f5e8;
    }
    .stTextInput, .stNumberInput, .stSelectbox, .stSlider {
        background-color: #2a2a2a;
        color: #e8f5e8;
        border: 1px solid #4a4a4a;
    }
    .stButton>button {
        background-color: #2d5016;
        color: #e8f5e8;
        border: 1px solid #4a7c2a;
        border-radius: 4px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #4a7c2a;
        border-color: #6a9c4a;
    }
    h1, h2, h3 {
        color: #4a7c2a;
    }
    .stMarkdown {
        color: #e8f5e8;
    }
    .css-1d391kg {
        background-color: #0a0a0a;
    }
    .css-12oz5g7 {
        background-color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class Parameters:
    r: float
    K: float
    alpha: float
    beta: float
    gamma: float
    mu_I: float
    delta: float
    phi: float
    mu_F: float
    sigma: float

def ode_system(t, y, params):
    C, I, F = y
    r, K, alpha, beta, gamma, mu_I, delta, phi, mu_F, sigma = (
        params.r, params.K, params.alpha, params.beta, params.gamma,
        params.mu_I, params.delta, params.phi, params.mu_F, params.sigma,
    )
    dCdt = r * C * (1 - C / K) - alpha * I * C - beta * F * C + gamma * F * C
    dIdt = alpha * I * C - mu_I * I - delta * F * I
    dFdt = phi - mu_F * F + sigma * I
    return [dCdt, dIdt, dFdt]

def compute_R0(params, C_free=None, F_free=None):
    if C_free is None:
        Fstar = params.phi / params.mu_F
        Cstar = params.K * max(0.0, 1 - (params.beta - params.gamma) * Fstar / params.r)
        C_free, F_free = Cstar, Fstar
    return (params.alpha * C_free) / (params.mu_I + params.delta * F_free)

def run_simulation(params, y0, t_span):
    sol = solve_ivp(
        lambda t, y: ode_system(t, y, params),
        t_span, y0, method="RK45", dense_output=True, max_step=0.5,
    )
    return sol

def plot_timeseries(sol, title):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    ax.plot(sol.t, sol.y[0], 'C0-', linewidth=2.5, label='Civilians (C)', color='#4a7c2a')
    ax.plot(sol.t, sol.y[1], 'C3-', linewidth=2.5, label='Insurgents (I)', color='#e74c3c')
    ax.plot(sol.t, sol.y[2], 'C2-', linewidth=2.5, label='COIN forces (F)', color='#3498db')
    ax.set_xlabel('Time (months)', fontsize=12, color='#e8f5e8')
    ax.set_ylabel('Population', fontsize=12, color='#e8f5e8')
    ax.set_title(title, fontsize=14, fontweight='bold', color='#4a7c2a')
    ax.legend(loc='best', framealpha=0.3, facecolor='#1a1a1a', edgecolor='#4a4a4a', labelcolor='#e8f5e8')
    ax.grid(True, alpha=0.3, linestyle='--', color='#4a4a4a')
    ax.tick_params(colors='#e8f5e8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#4a4a4a')
    return fig

def main():
    st.title("🎯 Insurgency Dynamics Model")
    st.markdown("---")

    # Sidebar for parameters
    st.sidebar.header("Model Parameters")

    # Default parameters
    default_params = {
        'r': 0.03,
        'K': 50000.0,
        'alpha': 0.0005,
        'beta': 0.0002,
        'gamma': 0.00015,
        'mu_I': 0.05,
        'delta': 0.005,
        'phi': 200.0,
        'mu_F': 0.02,
        'sigma': 0.003,
    }

    # Parameter inputs
    params = Parameters(
        r=st.sidebar.number_input("Growth rate (r)", value=default_params['r'], format="%.4f", step=0.001),
        K=st.sidebar.number_input("Carrying capacity (K)", value=default_params['K'], format="%.0f", step=1000.0),
        alpha=st.sidebar.slider("Recruitment rate (α)", 0.0001, 0.002, default_params['alpha'], format="%.5f"),
        beta=st.sidebar.slider("Harm rate (β)", 0.00005, 0.001, default_params['beta'], format="%.5f"),
        gamma=st.sidebar.slider("Legitimacy rate (γ)", 0.00005, 0.001, default_params['gamma'], format="%.5f"),
        mu_I=st.sidebar.number_input("Insurgent mortality (μ_I)", value=default_params['mu_I'], format="%.3f", step=0.001),
        delta=st.sidebar.number_input("COIN effectiveness (δ)", value=default_params['delta'], format="%.4f", step=0.0001),
        phi=st.sidebar.number_input("COIN recruitment (φ)", value=default_params['phi'], format="%.1f", step=10.0),
        mu_F=st.sidebar.number_input("COIN mortality (μ_F)", value=default_params['mu_F'], format="%.3f", step=0.001),
        sigma=st.sidebar.number_input("Conversion rate (σ)", value=default_params['sigma'], format="%.4f", step=0.0001),
    )

    # Initial conditions
    st.sidebar.header("Initial Conditions")
    C0 = st.sidebar.slider("Initial Civilians (C₀)", 0.1, 1.0, 0.8, format="%.2f") * params.K
    I0 = st.sidebar.slider("Initial Insurgents (I₀)", 0.0, 5000.0, 500.0, step=50.0)
    F0 = st.sidebar.slider("Initial COIN forces (F₀)", 1000.0, 5000.0, 2000.0, step=100.0)

    # Simulation time
    st.sidebar.header("Simulation")
    t_max = st.sidebar.slider("Simulation time (months)", 50, 300, 120, step=10)

    # Run simulation button
    if st.sidebar.button("Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            y0 = [C0, I0, F0]
            t_span = (0.0, t_max)
            sol = run_simulation(params, y0, t_span)
            R0 = compute_R0(params)

            # Store results in session state
            st.session_state['sol'] = sol
            st.session_state['R0'] = R0
            st.session_state['params'] = params

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Simulation Results")
        if 'sol' in st.session_state:
            sol = st.session_state['sol']
            R0 = st.session_state['R0']

            # R0 indicator
            if R0 < 1:
                st.success(f"**R₀ = {R0:.3f}** - Insurgency likely to collapse")
            else:
                st.error(f"**R₀ = {R0:.3f}** - Endemic insurgency expected")

            # Plot
            fig = plot_timeseries(sol, f"Population Dynamics (R₀ = {R0:.3f})")
            st.pyplot(fig)

            # Final values
            st.subheader("Final Populations")
            final_C = sol.y[0, -1]
            final_I = sol.y[1, -1]
            final_F = sol.y[2, -1]

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Final Civilians", f"{final_C:,.0f}")
            with col_b:
                st.metric("Final Insurgents", f"{final_I:,.0f}")
            with col_c:
                st.metric("Final COIN Forces", f"{final_F:,.0f}")

    with col2:
        st.header("Model Overview")
        st.markdown("""
        **Mathematical Model:**
        - **C**: Civilian population
        - **I**: Insurgent population
        - **F**: Counterinsurgency forces

        **Key Equation:**
        ```
        R₀ = αC*/(μ_I + δF*)
        ```

        **Threshold:**
        - R₀ < 1: Insurgency collapses
        - R₀ > 1: Endemic insurgency

        **Critical Parameters:**
        - α: Insurgent recruitment
        - β: COIN collateral damage
        - γ: Hearts-and-minds effect
        """)

        if 'R0' in st.session_state:
            st.markdown("---")
            st.subheader("Sensitivity Analysis")
            st.markdown("Adjust parameters in sidebar to see how R₀ changes")

if __name__ == "__main__":
    main()