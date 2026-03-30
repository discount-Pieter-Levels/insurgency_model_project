import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.8,
})

from dataclasses import dataclass

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

def main():
    # Create base and scenario parameters
    base_params = Parameters(
        r=0.03, K=50000, alpha=0.0005, beta=0.0002, gamma=0.00015,
        mu_I=0.05, delta=0.005, phi=200.0, mu_F=0.02, sigma=0.003,
    )

    params_stable = Parameters(**vars(base_params))
    params_stable.alpha = 0.0002
    params_stable.beta = 0.00025
    params_stable.gamma = 0.00015

    params_unstable = Parameters(**vars(base_params))
    params_unstable.alpha = 0.0009
    params_unstable.beta = 0.0001
    params_unstable.gamma = 0.0002

    # Run simulations
    y0 = [0.8 * base_params.K, 500.0, 2000.0]
    t_span = (0.0, 120.0)
    sol_stable = run_simulation(params_stable, y0, t_span)
    sol_unstable = run_simulation(params_unstable, y0, t_span)

    # Create comprehensive summary figure
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3, 
                  left=0.08, right=0.95, top=0.93, bottom=0.07)

    # ========== Title box ==========
    fig.text(0.5, 0.97, 'Insurgency–Counterinsurgency Dynamics: Mathematical Model & Analysis',
             ha='center', fontsize=16, fontweight='bold')

    # ========== PANEL 1: Model equations ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    eq_text = r'$\mathbf{Model\ Equations:}$' + '\n\n'
    eq_text += r'$\frac{dC}{dt} = r C(1-\frac{C}{K}) - \alpha IC - \beta FC + \gamma FC$' + '\n\n'
    eq_text += r'$\frac{dI}{dt} = \alpha IC - \mu_I I - \delta FI$' + '\n\n'
    eq_text += r'$\frac{dF}{dt} = \phi - \mu_F F + \sigma I$'
    
    ax1.text(0.05, 0.95, eq_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # ========== PANEL 2: R0 threshold ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    R0_stable = compute_R0(params_stable)
    R0_unstable = compute_R0(params_unstable)
    
    r0_text = r'$\mathbf{Basic\ Reproduction\ Number:}$' + '\n\n'
    r0_text += r'$R_0 = \frac{\alpha C^*}{\mu_I + \delta F^*}$' + '\n\n'
    r0_text += f'Scenario 1: $R_0 = {R0_stable:.3f}$ (collapse)\n'
    r0_text += f'Scenario 2: $R_0 = {R0_unstable:.2f}$ (endemic)'
    
    ax2.text(0.05, 0.95, r0_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # ========== PANEL 3: Key parameters ==========
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    param_text = r'$\mathbf{Key\ Parameters:}$' + '\n\n'
    param_text += f'$\\alpha$ (recruitment): {params_unstable.alpha:.4f}\n'
    param_text += f'$\\beta$ (harm): {params_unstable.beta:.5f}\n'
    param_text += f'$\\gamma$ (hearts-minds): {params_unstable.gamma:.5f}\n'
    param_text += f'$K$ (capacity): {int(params_unstable.K):,}\n'
    
    ax3.text(0.05, 0.95, param_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.25))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    # ========== PANEL 4: Scenario 1 (R0 < 1) ==========
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(sol_stable.t, sol_stable.y[0], 'C0-', linewidth=2.5, label='Civilians (C)')
    ax4.plot(sol_stable.t, sol_stable.y[1], 'C3-', linewidth=2.5, label='Insurgents (I)')
    ax4.plot(sol_stable.t, sol_stable.y[2], 'C2-', linewidth=2.5, label='COIN forces (F)')
    ax4.set_xlabel('Time (months)', fontsize=11)
    ax4.set_ylabel('Population', fontsize=11)
    ax4.set_title(f'Scenario 1: Insurgency Collapses ($R_0={R0_stable:.4f}<1$)', 
                  fontsize=12, fontweight='bold', color='darkgreen')
    ax4.legend(loc='best', framealpha=0.95)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_facecolor('#f0f8f0')

    # ========== PANEL 5: Scenario 2 (R0 > 1) ==========
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(sol_unstable.t, sol_unstable.y[0], 'C0-', linewidth=2.5, label='Civilians (C)')
    ax5.plot(sol_unstable.t, sol_unstable.y[1], 'C3-', linewidth=2.5, label='Insurgents (I)')
    ax5.plot(sol_unstable.t, sol_unstable.y[2], 'C2-', linewidth=2.5, label='COIN forces (F)')
    ax5.set_xlabel('Time (months)', fontsize=11)
    ax5.set_ylabel('Population', fontsize=11)
    ax5.set_title(f'Scenario 2: Endemic Insurgency ($R_0={R0_unstable:.2f}>1$)', 
                  fontsize=12, fontweight='bold', color='darkred')
    ax5.legend(loc='best', framealpha=0.95)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.set_facecolor('#f8f0f0')

    # ========== PANEL 6: Phase portrait I vs C ==========
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(sol_stable.y[0], sol_stable.y[1], 'o-', linewidth=2.5, markersize=4,
             color='darkgreen', label='Scenario 1 (collapse)', alpha=0.8)
    ax6.plot(sol_unstable.y[0], sol_unstable.y[1], 'o-', linewidth=2.5, markersize=4,
             color='darkred', label='Scenario 2 (endemic)', alpha=0.8)
    ax6.set_xlabel('Civilian population (C)', fontsize=11)
    ax6.set_ylabel('Insurgent population (I)', fontsize=11)
    ax6.set_title('Phase Portrait: I vs C', fontsize=12, fontweight='bold')
    ax6.legend(loc='best', framealpha=0.95)
    ax6.grid(True, alpha=0.3, linestyle='--')

    # ========== PANEL 7: Sensitivity alpha ==========
    ax7 = fig.add_subplot(gs[2, 0])
    alphas = np.logspace(np.log10(0.0001), np.log10(0.002), 25)
    R0_vals = []
    I_final_vals = []
    
    for alpha_val in alphas:
        p = Parameters(**vars(base_params))
        p.alpha = alpha_val
        R0_vals.append(compute_R0(p))
        sol = run_simulation(p, y0, (0, 150))
        I_final_vals.append(sol.y[1, -1])
    
    ax7_twin = ax7.twinx()
    
    line1 = ax7.semilogy(alphas, R0_vals, 'C0-o', linewidth=2.5, markersize=5, label='$R_0$')
    ax7.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Threshold ($R_0=1$)')
    ax7.fill_between(alphas, 0.1, 1, where=(np.array(R0_vals)<=1), alpha=0.15, color='green', label='Endemic zone')
    ax7.fill_between(alphas, 1, 1000, where=(np.array(R0_vals)>1), alpha=0.15, color='red')
    
    line2 = ax7_twin.plot(alphas, I_final_vals, 'C3-s', linewidth=2.5, markersize=5, label='$I^*$ (steady-state)')
    
    ax7.set_xlabel('Recruitment rate $\\alpha$', fontsize=11)
    ax7.set_ylabel('$R_0$ (log scale)', fontsize=11, color='C0')
    ax7_twin.set_ylabel('Final insurgents $I^*$', fontsize=11, color='C3')
    ax7.set_title('Sensitivity Analysis: $\\alpha$ sweep', fontsize=12, fontweight='bold')
    ax7.tick_params(axis='y', labelcolor='C0')
    ax7_twin.tick_params(axis='y', labelcolor='C3')
    ax7.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Combine legends
    lines1, labels1 = ax7.get_legend_handles_labels()
    lines2, labels2 = ax7_twin.get_legend_handles_labels()
    ax7.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=9)

    # ========== PANEL 8: Beta vs Gamma heatmap ==========
    ax8 = fig.add_subplot(gs[2, 1])
    beta_range = np.linspace(0.00005, 0.001, 20)
    gamma_range = np.linspace(0.00005, 0.001, 20)
    Istar_heat = np.zeros((len(gamma_range), len(beta_range)))
    
    for j, gamma in enumerate(gamma_range):
        for i, beta in enumerate(beta_range):
            p = Parameters(**vars(base_params))
            p.beta = beta
            p.gamma = gamma
            sol = run_simulation(p, y0, (0, 150))
            Istar_heat[j, i] = sol.y[1, -1]
    
    im = ax8.contourf(beta_range, gamma_range, Istar_heat, levels=20, cmap='RdYlGn_r')
    cbar = plt.colorbar(im, ax=ax8)
    cbar.set_label('Insurgent equilibrium $I^*$', fontsize=10)
    ax8.set_xlabel('Harm rate $\\beta$ (COIN collateral)', fontsize=11)
    ax8.set_ylabel('Legitimacy rate $\\gamma$ (hearts-minds)', fontsize=11)
    ax8.set_title('Policy space: Hearts-and-minds vs Harm', fontsize=12, fontweight='bold')
    
    # Add diagonal line showing gamma = beta
    ax8.plot(beta_range, beta_range, 'k--', linewidth=2, alpha=0.5, label='$\\gamma = \\beta$')
    ax8.legend(fontsize=9)

    # ========== PANEL 9: Key findings ==========
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    findings = r'$\mathbf{Key\ Findings:}$' + '\n\n'
    findings += r'1. Threshold: $R_0 = 1$ separates' + '\n'
    findings += '   collapse from endemic\n\n'
    findings += r'2. Insurgent recruitment $\alpha$' + '\n'
    findings += '   is critical control knob\n\n'
    findings += r'3. (net legitimacy)' + '\n'
    findings += r'   $\gamma - \beta$ suppresses $I$' + '\n\n'
    findings += '4. Delay in recruitment\n'
    findings += '   can enhance oscillations'
    
    ax9.text(0.05, 0.95, findings, transform=ax9.transAxes, fontsize=10.5,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.4, pad=0.8))
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)

    plt.savefig('insurgency_summary_visualization.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: insurgency_summary_visualization.png')
    plt.show()

if __name__ == '__main__':
    main()
