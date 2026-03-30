import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# 1. Model Definition
# -----------------------------------------------------------------------------

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
        params.r,
        params.K,
        params.alpha,
        params.beta,
        params.gamma,
        params.mu_I,
        params.delta,
        params.phi,
        params.mu_F,
        params.sigma,
    )
    dCdt = r * C * (1 - C / K) - alpha * I * C - beta * F * C + gamma * F * C
    dIdt = alpha * I * C - mu_I * I - delta * F * I
    dFdt = phi - mu_F * F + sigma * I
    return [dCdt, dIdt, dFdt]


def dimensionless_system(tau, y, A, B, G, mI, D, Phi, mF, S):
    c, u, v = y
    dc = c * (1 - c) - A * u * c - B * v * c + G * v * c
    du = A * u * c - mI * u - D * v * u
    dv = Phi - mF * v + S * u
    return [dc, du, dv]


def nondimensionalize(params):
    A = params.alpha * params.K / params.r
    B = params.beta * params.K / params.r
    G = params.gamma * params.K / params.r
    mI = params.mu_I / params.r
    D = params.delta * params.K / params.r
    Phi = params.phi / (params.K * params.r)
    mF = params.mu_F / params.r
    S = params.sigma / params.r
    return A, B, G, mI, D, Phi, mF, S


def computing_equilibria(params):
    Fstar = params.phi / params.mu_F
    Cstar = params.K * max(0.0, 1 - (params.beta - params.gamma) * Fstar / params.r)
    E0 = (0.0, 0.0, Fstar)
    E1 = (Cstar, 0.0, Fstar)
    # Coexistence solved numerically:
    if Cstar <= 0:
        E2 = None
    else:
        # solve I* from R0 condition (approx)
        R0 = (params.alpha * Cstar) / (params.mu_I + params.delta * Fstar)
        if R0 <= 1:
            E2 = None
        else:
            I2 = params.mu_F * ((params.r * (1 - Cstar / params.K) + (params.gamma - params.beta) * Fstar) / params.alpha - 1) / params.sigma
            if I2 < 0:
                E2 = None
            else:
                E2 = (Cstar, I2, Fstar + params.sigma * I2 / params.mu_F)
    return E0, E1, E2


def jacobian(C, I, F, params):
    r, K, alpha, beta, gamma, mu_I, delta, phi, mu_F, sigma = (
        params.r,
        params.K,
        params.alpha,
        params.beta,
        params.gamma,
        params.mu_I,
        params.delta,
        params.phi,
        params.mu_F,
        params.sigma,
    )
    J = np.zeros((3, 3))
    J[0, 0] = r * (1 - 2 * C / K) - alpha * I - beta * F + gamma * F
    J[0, 1] = -alpha * C
    J[0, 2] = (-beta + gamma) * C
    J[1, 0] = alpha * I
    J[1, 1] = alpha * C - mu_I - delta * F
    J[1, 2] = -delta * I
    J[2, 0] = 0
    J[2, 1] = sigma
    J[2, 2] = -mu_F
    return J


def eigenvalues_at_equilibrium(eq, params):
    C, I, F = eq
    eigenvals = np.linalg.eigvals(jacobian(C, I, F, params))
    return eigenvals


def compute_R0(params, C_free=None, F_free=None):
    if C_free is None or F_free is None:
        _, E1, _ = computing_equilibria(params)
        if E1 is None:
            raise ValueError("No insurgency-free equilibrium computed for R0")
        C_free, _, F_free = E1
    numerator = params.alpha * C_free
    denominator = params.mu_I + params.delta * F_free
    return numerator / denominator


def run_simulation(params, y0, t_span):
    sol = solve_ivp(
        lambda t, y: ode_system(t, y, params),
        t_span,
        y0,
        method="RK45",
        dense_output=True,
        max_step=0.5,
    )
    return sol


def plot_timeseries(sol, ax, title):
    t = sol.t
    C = sol.y[0]
    I = sol.y[1]
    F = sol.y[2]
    ax.plot(t, C, label="C (civilians)")
    ax.plot(t, I, label="I (insurgents)")
    ax.plot(t, F, label="F (COIN)")
    ax.set_title(title)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Population")
    ax.legend(loc="upper right")
    ax.grid(True, ls='--', alpha=0.5)


def plot_phase_portrait(ax, params, c_range, i_range, resolution=16):
    c_vals = np.linspace(*c_range, resolution)
    i_vals = np.linspace(*i_range, resolution)
    Cg, Ig = np.meshgrid(c_vals, i_vals)
    Ft = params.phi / params.mu_F
    dC = params.r * Cg * (1 - Cg / params.K) - params.alpha * Ig * Cg - params.beta * Ft * Cg + params.gamma * Ft * Cg
    dI = params.alpha * Ig * Cg - params.mu_I * Ig - params.delta * Ft * Ig
    magnitude = np.sqrt(dC ** 2 + dI ** 2)
    ax.quiver(Cg, Ig, dC / (magnitude + 1e-8), dI / (magnitude + 1e-8), scale=30, alpha=0.7)
    ax.set_xlabel("C")
    ax.set_ylabel("I")
    ax.set_title("Phase portrait I vs C")


def sensitivity_sweep(param_name, values, base_params):
    I_final = []
    R0s = []
    for v in values:
        p = Parameters(**vars(base_params))
        setattr(p, param_name, float(v))
        E1 = computing_equilibria(p)[1]
        if E1 is None:
            R0 = np.nan
        else:
            R0 = compute_R0(p, C_free=E1[0], F_free=E1[2])
        sol = run_simulation(p, [0.8 * p.K, 500.0, 2000.0], (0.0, 200.0))
        I_final.append(sol.y[1, -1])
        R0s.append(R0)
    return np.array(R0s), np.array(I_final)


def plot_heatmap(beta_range, gamma_range, base_params):
    Istar = np.zeros((len(gamma_range), len(beta_range)))
    for j, gamma in enumerate(gamma_range):
        for i, beta in enumerate(beta_range):
            p = Parameters(**vars(base_params))
            p.beta = beta
            p.gamma = gamma
            sol = run_simulation(p, [0.8 * p.K, 500.0, 2000.0], (0.0, 200.0))
            Istar[j, i] = sol.y[1, -1]
    return Istar


def delayed_dynamics(t, y, Z, params, tau_d):
    C, I, F = y
    C_tau = Z(t - tau_d, 0)
    dI = params.alpha * I * C_tau - params.mu_I * I - params.delta * F * I
    dC = params.r * C * (1 - C / params.K) - params.alpha * I * C - params.beta * F * C + params.gamma * F * C
    dF = params.phi - params.mu_F * F + params.sigma * I
    return [dC, dI, dF]


def dde_simulation(params, y0, t_span, tau_d):
    # Simple Euler/Midpoint scheme approximate delay with interpolation
    t_eval = np.linspace(t_span[0], t_span[1], 1201)
    # history constant from initial y0
    sol = np.zeros((3, len(t_eval)))
    sol[:, 0] = y0
    for n in range(1, len(t_eval)):
        t = t_eval[n]
        dt = t_eval[n] - t_eval[n - 1]
        if t - tau_d <= t_eval[0]:
            C_tau = y0[0]
        else:
            fC = interp1d(t_eval[:n], sol[0, :n], kind='linear', fill_value='extrapolate')
            C_tau = fC(t - tau_d)
        dC = params.r * sol[0, n - 1] * (1 - sol[0, n - 1] / params.K) - params.alpha * sol[1, n - 1] * sol[0, n - 1] - params.beta * sol[2, n - 1] * sol[0, n - 1] + params.gamma * sol[2, n - 1] * sol[0, n - 1]
        dI = params.alpha * sol[1, n - 1] * C_tau - params.mu_I * sol[1, n - 1] - params.delta * sol[2, n - 1] * sol[1, n - 1]
        dF = params.phi - params.mu_F * sol[2, n - 1] + params.sigma * sol[1, n - 1]
        sol[0, n] = sol[0, n - 1] + dC * dt
        sol[1, n] = sol[1, n - 1] + dI * dt
        sol[2, n] = sol[2, n - 1] + dF * dt
    return t_eval, sol


def main():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'figure.dpi': 150,
    })

    base_params = Parameters(
        r=0.03,
        K=50000,
        alpha=0.0005,
        beta=0.0002,
        gamma=0.00015,
        mu_I=0.05,
        delta=0.005,
        phi=200.0,
        mu_F=0.02,
        sigma=0.003,
    )

    E0, E1, E2 = computing_equilibria(base_params)
    print("Equilibria:")
    print("E0:", E0)
    print("E1:", E1)
    print("E2:", E2)

    print("Eigenvalues at E0:", eigenvalues_at_equilibrium(E0, base_params))
    if E1 is not None:
        print("Eigenvalues at E1:", eigenvalues_at_equilibrium(E1, base_params))
    if E2 is not None:
        print("Eigenvalues at E2:", eigenvalues_at_equilibrium(E2, base_params))

    R0_base = compute_R0(base_params)
    print(f"R0 (base) = {R0_base:.4f}")

    # choose scenario parameters
    params_stable = Parameters(**vars(base_params))
    params_stable.alpha = 0.0002
    params_stable.beta = 0.00025
    params_stable.gamma = 0.00015

    params_unstable = Parameters(**vars(base_params))
    params_unstable.alpha = 0.0009
    params_unstable.beta = 0.0001
    params_unstable.gamma = 0.0002

    for tag, p in [('stable', params_stable), ('unstable', params_unstable)]:
        R0 = compute_R0(p)
        print(f"{tag} scenario R0 = {R0:.4f}")

    y0 = [0.8 * base_params.K, 500.0, 2000.0]
    t_span = (0.0, 120.0)
    sol_stable = run_simulation(params_stable, y0, t_span)
    sol_unstable = run_simulation(params_unstable, y0, t_span)

    # Figure 1 (2x2)
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_timeseries(sol_stable, ax1, "Scenario 1: R0 < 1 (insurgency collapses)")
    ax2 = fig.add_subplot(gs[0, 1])
    plot_timeseries(sol_unstable, ax2, "Scenario 2: R0 > 1 (insurgency persists)")

    ax3 = fig.add_subplot(gs[1, 0])
    # overlay phase path I vs C for both scenarios
    ax3.plot(sol_stable.y[0], sol_stable.y[1], label="stable scenario", color='tab:blue')
    ax3.plot(sol_unstable.y[0], sol_unstable.y[1], label="unstable scenario", color='tab:red')
    plot_phase_portrait(ax3, base_params, (0, base_params.K), (0, max(sol_stable.y[1].max(), sol_unstable.y[1].max())*1.1))
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(sol_stable.y[2], sol_stable.y[1], label="stable scenario", color='tab:blue')
    ax4.plot(sol_unstable.y[2], sol_unstable.y[1], label="unstable scenario", color='tab:red')
    ax4.set_xlabel("F")
    ax4.set_ylabel("I")
    ax4.set_title("Phase portrait I vs F")
    ax4.grid(True, ls='--', alpha=0.5)
    ax4.legend()

    fig.savefig("insurgency_dynamics.png", dpi=300, bbox_inches='tight')
    print("Saved insurgency_dynamics.png")

    # Sensitivity analysis alpha
    alphas = np.logspace(np.log10(0.00005), np.log10(0.002), 20)
    R0s, I_stars = sensitivity_sweep('alpha', alphas, base_params)

    fig2, ax21 = plt.subplots(figsize=(8, 5))
    ax21.plot(alphas, R0s, '-o', label='R0')
    ax21.axhline(1.0, color='k', linestyle='--', label='R0=1')
    ax21.set_xscale('log')
    ax21.set_xlabel('alpha')
    ax21.set_ylabel('R0')
    ax22 = ax21.twinx()
    ax22.plot(alphas, I_stars, '-s', color='tab:orange', label='I*')
    ax22.set_ylabel('I* (final insurgents)')
    ax21.legend(loc='upper left')
    ax22.legend(loc='upper right')
    fig2.savefig('alpha_sensitivity.png', dpi=300, bbox_inches='tight')
    print('Saved alpha_sensitivity.png')

    # Bifurcation diagram
    fig3, ax3b = plt.subplots(figsize=(8, 5))
    ax3b.plot(alphas, I_stars, '-o', color='tab:purple')
    ax3b.axvline(alphas[np.nanargmin(np.abs(R0s - 1))], color='k', linestyle='--', label='R0=1 threshold')
    ax3b.set_xscale('log')
    ax3b.set_xlabel('alpha')
    ax3b.set_ylabel('I* steady state')
    ax3b.set_title('Bifurcation diagram I* vs alpha')
    ax3b.legend()
    fig3.savefig('bifurcation_I_vs_alpha.png', dpi=300, bbox_inches='tight')
    print('Saved bifurcation_I_vs_alpha.png')

    # Heatmap beta vs gamma
    beta_range = np.linspace(0.00005, 0.001, 15)
    gamma_range = np.linspace(0.00005, 0.001, 15)
    I_heat = plot_heatmap(beta_range, gamma_range, base_params)

    fig4, ax4h = plt.subplots(figsize=(7, 6))
    im = ax4h.imshow(I_heat, origin='lower', aspect='auto', extent=[beta_range[0], beta_range[-1], gamma_range[0], gamma_range[-1]], cmap='viridis')
    cbar = fig4.colorbar(im, ax=ax4h)
    cbar.set_label('Steady-state I*')
    ax4h.set_xlabel('beta (harm)')
    ax4h.set_ylabel('gamma (legitimacy)')
    ax4h.set_title('Heatmap of Insurgent equilibrium I*')
    fig4.savefig('heatmap_beta_gamma.png', dpi=300, bbox_inches='tight')
    print('Saved heatmap_beta_gamma.png')

    # Optional delay extension
    delay_values = [0, 3, 6, 12]
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    for tau_d in delay_values:
        t_del, sol_del = dde_simulation(params_unstable, y0, (0, 120), tau_d)
        ax5.plot(t_del, sol_del[1], label=f'\u03C4={tau_d}')
    ax5.set_xlabel('Time (months)')
    ax5.set_ylabel('I(t)')
    ax5.set_title('Delay effect on insurgent dynamics')
    ax5.legend()
    fig5.savefig('delay_I_vs_t.png', dpi=300, bbox_inches='tight')
    print('Saved delay_I_vs_t.png')

    print('Simulation and plotting complete.')


if __name__ == '__main__':
    main()
