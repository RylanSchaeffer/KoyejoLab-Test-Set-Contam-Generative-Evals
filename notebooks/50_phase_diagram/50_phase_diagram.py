import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import root_scalar

import src.analyze
import src.plot

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)


# %% Helper function to solve for T


def solve_for_T(E: float, A: float, alpha: float, p_target: float) -> float:
    """
    Solves for T in the equation:
    ln(p) = -E(T-1) - Y(T^(1-alpha) - 1)

    where Y = A / (1 - alpha).

    Handles three regimes:
    - alpha < 1: Standard form, T^(1-alpha) grows with T
    - alpha = 1: Limiting form A*ln(T)
    - alpha > 1: T^(1-alpha) decays to 0 as T grows (may have lock-in)

    Args:
        E: Irreducible error rate
        A: Scaling coefficient
        alpha: Scaling exponent
        p_target: Target probability P(T)

    Returns:
        T: Solution length at which P(T) = p_target (np.inf for lock-in)
    """
    target = np.log(p_target)

    # Compute Y = A / (1 - alpha), handling alpha = 1
    if abs(1 - alpha) < 1e-10:
        Y = None  # Special case handled below
    else:
        Y = A / (1 - alpha)

    # Check for Lock-In condition (Regime III: alpha > 1, E ≈ 0)
    # For alpha > 1, as T → ∞: T^(1-alpha) → 0
    # The asymptotic log-probability is: lim_{T→∞} ln(P) = -E*T - Y*(0 - 1) = -E*T + Y
    # If E ≈ 0, then ln(P) → Y (which is negative since alpha > 1 means 1-alpha < 0)
    # Lock-in occurs if this asymptote > target, i.e., Y > target
    if alpha > 1 and E < 1e-9:
        # Y is negative for alpha > 1. Lock-in if Y > target (both negative, Y closer to 0)
        if Y > target:
            return np.inf

    def f(t):
        if t <= 1:
            return -target  # f(1) = -target > 0 for p_target < 1

        # Compute the penalty term based on alpha
        if Y is None:
            # α ≈ 1: Use limiting form A*ln(T)
            penalty = A * np.log(t)
        else:
            # General case: Y * (T^(1-α) - 1)
            try:
                penalty = Y * (t ** (1 - alpha) - 1)
            except OverflowError:
                return -np.inf

        return -E * (t - 1) - penalty - target

    # Bracket search for the root
    t_min = 1.0
    t_max = 10.0

    for _ in range(20):
        try:
            val = f(t_max)
        except OverflowError:
            val = -np.inf  # Blew up negatively, we passed the root

        if val < 0:  # We crossed the target probability
            break
        t_max *= 10
        if t_max > 1e15:  # Practical infinity
            return np.inf
    else:
        return np.inf  # Curve never drops below p_target

    try:
        sol = root_scalar(f, bracket=[t_min, t_max], method="brentq")
        return sol.root
    except Exception:
        return np.inf


# %% Phase Diagram: Solution Length Capacity (Two panels: alpha < 1 vs alpha > 1)

# --- Parameters ---
p_target = 0.01  # Probability threshold (1% survival)
alpha_values = [0.5, 1.5]  # Brittle Memorization vs potential Lock-In

# --- Grid Definition ---
resolution = 200
e_vals = np.logspace(-10, 0, resolution)  # E from 1e-10 to 1 (extended to capture lock-in)
a_vals = np.logspace(-3, 1, resolution)  # A from 1e-3 to 1e1

E_grid, A_grid = np.meshgrid(e_vals, a_vals)

# --- Compute T grids ---
all_T_grids = []
for alpha in alpha_values:
    T_grid = np.zeros_like(E_grid)
    for i in range(E_grid.shape[0]):
        for j in range(E_grid.shape[1]):
            e = E_grid[i, j]
            a = A_grid[i, j]
            t = solve_for_T(e, a, alpha, p_target)
            T_grid[i, j] = t
    all_T_grids.append(T_grid)

# Get global T range for finite values
finite_T_values = [T[np.isfinite(T)] for T in all_T_grids]
T_min = 1
T_max = max(np.max(t) for t in finite_T_values if len(t) > 0)

# Cap for display: values above this are shown as "lock-in"
T_cap = 1e6

# --- Plotting ---
plt.close()
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (alpha, T_grid) in enumerate(zip(alpha_values, all_T_grids)):
    ax = axes[idx]

    # Create display grid: cap infinite values for colormapping
    T_display = T_grid.copy()
    lock_in_mask = np.isinf(T_grid)
    T_display[lock_in_mask] = T_cap

    # Use LogNorm for color scaling
    pcm = ax.pcolormesh(
        E_grid,
        A_grid,
        T_display,
        norm=mcolors.LogNorm(vmin=T_min, vmax=T_cap),
        cmap="plasma",
        shading="nearest",
        rasterized=True,
    )

    # Overlay lock-in region with hatching or distinct marking
    if np.any(lock_in_mask):
        # Draw contour around the lock-in region
        ax.contour(
            E_grid,
            A_grid,
            lock_in_mask.astype(float),
            levels=[0.5],
            colors="black",
            linestyles="--",
            linewidths=2,
        )
        # Add label for lock-in region
        # Place label to the right of the lock-in boundary
        label_e = 3e-8  # Fixed position to the right of the lock-in boundary
        label_a = 3e-2  # Lower portion of the A range in log scale
        ax.text(
            label_e,
            label_a,
            r"\textbf{Lock-In}" + "\n" + r"($T = \infty$)",
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )

    # Set log scale for both axes
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Disable grid
    ax.grid(False)

    # Add contours for specific lengths (Iso-T lines)
    levels = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
    # Only use levels within the finite range
    valid_levels = [l for l in levels if T_min < l < np.nanmax(T_grid[np.isfinite(T_grid)])]

    if valid_levels:
        CS = ax.contour(
            E_grid,
            A_grid,
            T_grid,
            levels=valid_levels,
            colors="white",
            linestyles="solid",
            linewidths=0.8,
            alpha=0.8,
        )
        ax.clabel(CS, inline=True, fmt="%d")

    # Labels
    ax.set_xlabel(r"Irreducible Error ($E$)")
    ax.set_ylabel(r"Scaling Coefficient ($A$)")

    # Panel title showing alpha value
    ax.set_title(r"$\alpha_{\mathrm{eff}} = " + str(alpha) + r"$")

# Adjust layout first, then add colorbar
plt.tight_layout()
plt.subplots_adjust(right=0.88)

# Add single colorbar for all panels
cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.7])
cbar = fig.colorbar(pcm, cax=cbar_ax)
cbar.set_label(r"Solution Length $T$ where $P(T) = " + str(p_target) + r"$")

src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename=f"phase_diagram_y=A_x=E_hue=T_alpha=multi_p={p_target}",
    use_tight_layout=False,  # Already applied
)
plt.close()
