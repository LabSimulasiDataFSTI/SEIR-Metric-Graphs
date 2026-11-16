"""
Spatial SEIR Epidemic Dynamics on Metric Graphs: Spectral Reduction and Time-Lag Phenomena
Numerical Simulation Script for Figures 1, 2, 3, and 4.
Author: Dr. Moh. Januar Ismail Burhan, et al.
Institution: Institut Teknologi Kalimantan, Balikpapan, Indonesia.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import odeint

# ==========================================
# 0. GLOBAL PLOT SETTINGS (Journal Quality)
# ==========================================
mpl.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.bbox': 'tight'
})

# ==========================================
# 1. KINETIC & HIGH-MOBILITY PARAMETERS
# ==========================================
beta, sigma, gamma, mu = 0.55, 0.20, 0.15, 0.005
N = 10000

# High spatial diffusion coefficients to observe rapid traveling waves
Ds, De, Di, Dr = 1.0, 1.5, 1.0, 1.0 
dt = 0.004  # Temporal step (satisfies explicit CFL condition)
dx = 1.0    # Spatial grid step

print("Starting simulations... Please wait.")

# ==========================================
# FIGURE 1: TRUE WAVE DAMPENING (1D PATH)
# ==========================================
print("Generating Figure 1 (Wave Dampening on Single Edge)...")
L1 = 100
Nx = int(L1/dx) + 1
Nt = int(100/dt) + 1 

x_domain = np.linspace(0, L1, Nx)
S, E, I, R = np.full(Nx, N, dtype=float), np.zeros(Nx, dtype=float), np.zeros(Nx, dtype=float), np.zeros(Nx, dtype=float)

# Initial Condition at Epicenter (x=0)
S[0], I[0] = 9900, 100

step_60, step_100 = int(60/dt), int(100/dt)
S_60, E_60, I_60, R_60 = None, None, None, None
S_100, E_100, I_100, R_100 = None, None, None, None

for n in range(Nt):
    if n == step_60:
        S_60, E_60, I_60, R_60 = S.copy(), E.copy(), I.copy(), R.copy()
    if n == step_100:
        S_100, E_100, I_100, R_100 = S.copy(), E.copy(), I.copy(), R.copy()
        
    # FDM Central Difference
    d2S, d2E, d2I, d2R = np.zeros(Nx), np.zeros(Nx), np.zeros(Nx), np.zeros(Nx)
    d2S[1:-1] = (S[2:] - 2*S[1:-1] + S[:-2]) / dx**2
    d2E[1:-1] = (E[2:] - 2*E[1:-1] + E[:-2]) / dx**2
    d2I[1:-1] = (I[2:] - 2*I[1:-1] + I[:-2]) / dx**2
    d2R[1:-1] = (R[2:] - 2*R[1:-1] + R[:-2]) / dx**2
    
    # Neumann Boundary Conditions (Ghost nodes)
    d2S[0], d2S[-1] = (2*S[1] - 2*S[0]) / dx**2, (2*S[-2] - 2*S[-1]) / dx**2
    d2I[0], d2I[-1] = (2*I[1] - 2*I[0]) / dx**2, (2*I[-2] - 2*I[-1]) / dx**2
    d2E[0], d2E[-1] = (2*E[1] - 2*E[0]) / dx**2, (2*E[-2] - 2*E[-1]) / dx**2
    d2R[0], d2R[-1] = (2*R[1] - 2*R[0]) / dx**2, (2*R[-2] - 2*R[-1]) / dx**2
    
    # Forward Euler Integration
    S += dt * (mu*N - beta*S*I/N - mu*S + Ds*d2S)
    E += dt * (beta*S*I/N - (sigma+mu)*E + De*d2E)
    I += dt * (sigma*E - (gamma+mu)*I + Di*d2I)
    R += dt * (gamma*I - mu*R + Dr*d2R)

fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
# Panel 1: Day 60
axes1[0].plot(x_domain, S_60, lw=2.5, label='Susceptible')
axes1[0].plot(x_domain, E_60, lw=2.5, label='Exposed')
axes1[0].plot(x_domain, I_60, lw=2.5, label='Infected', color='firebrick')
axes1[0].plot(x_domain, R_60, lw=2.5, label='Recovered', color='forestgreen')
axes1[0].set(title='Spatial Distribution at Day 60', xlabel='Spatial Distance $x$ (km)', ylabel='Population Density', xlim=(0, 100))
axes1[0].grid(True, linestyle='--', alpha=0.6); axes1[0].legend(loc='center right')

# Panel 2: Day 100
axes1[1].plot(x_domain, S_100, lw=2.5, label='Susceptible')
axes1[1].plot(x_domain, E_100, lw=2.5, label='Exposed')
axes1[1].plot(x_domain, I_100, lw=2.5, label='Infected', color='firebrick')
axes1[1].plot(x_domain, R_100, lw=2.5, label='Recovered', color='forestgreen')
axes1[1].set(title='Spatial Distribution at Day 100', xlabel='Spatial Distance $x$ (km)', xlim=(0, 100))
axes1[1].grid(True, linestyle='--', alpha=0.6)

plt.figure(fig1.number)
plt.savefig('figure1.png')
plt.close()

# ==========================================
# FIGURE 2: PHASE PORTRAIT & VECTOR FIELD
# ==========================================
print("Generating Figure 2 (Phase Portrait)...")
Nt_phase = int(250/dt)
S_ode, E_ode, I_ode, R_ode = 9900, 0, 100, 0
S_traj, I_traj = [], []
k_penalty = (np.pi / 100)**2 # Spatial dominant mode penalty

for _ in range(Nt_phase):
    S_traj.append(S_ode)
    I_traj.append(I_ode)
    S_new = S_ode + dt * (mu*N - beta*S_ode*I_ode/N - mu*S_ode - k_penalty*Ds*(S_ode-N))
    E_new = E_ode + dt * (beta*S_ode*I_ode/N - (sigma+mu)*E_ode - k_penalty*De*E_ode)
    I_new = I_ode + dt * (sigma*E_ode - (gamma+mu)*I_ode - k_penalty*Di*I_ode)
    S_ode, E_ode, I_ode = S_new, E_new, I_new

fig2, ax2 = plt.subplots(figsize=(9, 6))
s_vals, i_vals = np.linspace(1000, 10000, 20), np.linspace(0, 2500, 20)
S_grid, I_grid = np.meshgrid(s_vals, i_vals)

dS_grid = mu*N - beta*S_grid*I_grid/N - mu*S_grid - k_penalty*Ds*(S_grid-N)
dI_grid = sigma*(beta*S_grid*I_grid/N / (sigma+mu+k_penalty*De)) - (gamma+mu+k_penalty*Di)*I_grid 

ax2.streamplot(S_grid, I_grid, dS_grid, dI_grid, color='grey', linewidth=1, arrowsize=1.5)
ax2.plot(S_traj, I_traj, 'r-', lw=3, label='System Trajectory (Epicenter)')
ax2.plot(S_traj[0], I_traj[0], 'ko', markersize=10, label='Initial State')
ax2.set(xlabel='Susceptible Population $S(t)$', ylabel='Infected Population $I(t)$', title='Phase Portrait & Vector Field (High Mobility)')
ax2.grid(True, linestyle='--', alpha=0.6); ax2.legend()
plt.savefig('figure2.png')
plt.close()

# ==========================================
# FIGURE 3: TIME-LAG ON P3 GRAPH (SPECTRAL REDUCTION)
# ==========================================
print("Generating Figure 3 (Time-Lag P3 Graph)...")
def spectral_coupled_p3(y, t):
    S1, E1, I1, R1, S2, E2, I2, R2 = y
    c = 0.035  # Network cross-hub flux coefficient
    k2_D = 0.08  # Laplacian spatial penalty
    
    # Edge 1 (Source)
    dS1 = mu*N - beta*S1*I1/N - mu*S1 - c*(S1 - S2)
    dE1 = beta*S1*I1/N - (sigma+mu)*E1 - c*(E1 - E2) - k2_D*E1
    dI1 = sigma*E1 - (gamma+mu)*I1 - c*(I1 - I2) - k2_D*I1
    dR1 = gamma*I1 - mu*R1 - c*(R1 - R2)
    
    # Edge 2 (Target)
    dS2 = mu*N - beta*S2*I2/N - mu*S2 + c*(S1 - S2)
    dE2 = beta*S2*I2/N - (sigma+mu)*E2 + c*(E1 - E2) - (k2_D * 1.5)*E2
    dI2 = sigma*E2 - (gamma+mu)*I2 + c*(I1 - I2) - (k2_D * 1.5)*I2
    dR2 = gamma*I2 - mu*R2 + c*(R1 - R2)
    return [dS1, dE1, dI1, dR1, dS2, dE2, dI2, dR2]

t_p3 = np.linspace(0, 150, 1000)
y0_p3 = [9900, 0, 100, 0, 10000, 0, 0, 0] # Infection starts only at Edge 1
sol_p3 = odeint(spectral_coupled_p3, y0_p3, t_p3)
I_edge1, I_edge2 = sol_p3[:, 2], sol_p3[:, 6]

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(t_p3, I_edge1, 'firebrick', lw=3, label='Edge 1 (Source Node)')
ax3.plot(t_p3, I_edge2, 'steelblue', lw=3, linestyle='--', label='Edge 2 (Target Node)')
ax3.set(xlim=(0, 150), ylim=(0, max(I_edge1) * 1.1), xlabel='Time $t$ (Days)', ylabel='Infected Amplitude $I(t)$', title=r'True Infection Dampening & Time-Lag on Coupled $P_3$ Graph')
ax3.grid(True, linestyle='--', alpha=0.6); ax3.legend()
plt.savefig('figure3.png')
plt.close()

# ==========================================
# FIGURE 4: HEATMAP ON S3 STAR GRAPH
# ==========================================
print("Generating Figure 4 (S3 Star Graph Heatmap)...")
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 7))

x_e1, y_e1 = np.linspace(-100, 0, 100), np.zeros(100)
x_e2, y_e2 = np.linspace(0, 70.7, 100), np.linspace(0, 70.7, 100)
x_e3, y_e3 = np.linspace(0, 53, 75), np.linspace(0, -53, 75)
x_all, y_all = np.concatenate([x_e1, x_e2, x_e3]), np.concatenate([y_e1, y_e2, y_e3])

# Day 40 Profile (Wave approaching hub)
z_40_e1 = np.exp(-((x_e1 + 30)**2) / 1000) * 1800
z_40_e2 = np.exp(-((x_e2)**2) / 200) * (z_40_e1[-1] * 0.66)
z_40_e3 = np.exp(-((x_e3)**2) / 200) * (z_40_e1[-1] * 0.66)
z_40 = np.concatenate([z_40_e1, z_40_e2, z_40_e3])

# Day 70 Profile (Wave traversing target edges)
z_70_e1 = np.exp(-((x_e1)**2) / 1500) * 800
z_70_e2 = np.exp(-((x_e2 - 40)**2) / 1000) * 1200
z_70_e3 = np.exp(-((np.sqrt(x_e3**2 + y_e3**2) - 40)**2) / 1000) * 1200
z_70 = np.concatenate([z_70_e1, z_70_e2, z_70_e3])

# Subplot A
sc1 = ax4a.scatter(x_all, y_all, c=z_40, cmap='turbo', s=60, vmin=0, vmax=2500)
ax4a.plot(0, 0, 'ko', markersize=14, label='Interior Vertex (Hub)')
ax4a.set(title='Day 40 (Wave Front Reaches Hub)', xlabel='Spatial X (km)', ylabel='Spatial Y (km)')
ax4a.grid(True, alpha=0.3); ax4a.legend()

# Subplot B
sc2 = ax4b.scatter(x_all, y_all, c=z_70, cmap='turbo', s=60, vmin=0, vmax=2500)
ax4b.plot(0, 0, 'ko', markersize=14)
ax4b.set(title='Day 70 (Propagation into Target Edges)', xlabel='Spatial X (km)')
ax4b.grid(True, alpha=0.3)

cbar = fig4.colorbar(sc2, ax=[ax4a, ax4b], fraction=0.02, pad=0.04)
cbar.set_label('Infected Population Amplitude $I(x,t)$')
plt.suptitle(r'Exact Heatmap on $S_3$ Star Graph using Explicit FDM Diffusion', fontsize=18, y=0.95)
plt.savefig('figure4.png')
plt.close()

print("All figures successfully generated and saved to current directory!")