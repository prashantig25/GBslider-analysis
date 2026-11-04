"""
Runs the ORIGINAL gaborbandit_analysis-master code (GbTask, GbTaskVars,
GbAgentVars, gb_task_agent_int) with the analytical (polynomial) belief
update (eval_ana=1), once WITH integration (task_agent_analysis=True) and
once WITHOUT (task_agent_analysis=False), across 5 sigma values and 2
true-mu conditions (0.7, 0.9), for a single 40-trial block per simulation.

Python sibling of Agent/simulateAndPlot_analytical_intVsNoInt_longBlock.m
(the MATLAB Agent class version of this same comparison) -- same mu/sigma
levels, same trial count (40, capped for eval_ana=1's numerical
stability), same plot style (thin individual-simulation lines + mean+/-
SEM shaded band + true-mu dashed reference, mu x sigma subplot grid) as
Agent/simfigures/simulatedMu_withVsNoInt_evalAna1.pdf.

gaborbandit_analysis-master is left completely untouched. The only change
is importing GbAgent from Agent/gb_legacy_patch/GbAgent.py instead of the
original -- a patched copy with the compute_q/update_mu_from_q fix
applied to Agent.m earlier this session (only affects eval_ana=0; the
eval_ana=1 branch used here is unpatched original code, so no additional
patching is needed for this script).

Must be run with an older numpy/pandas stack (e.g. the "gb_legacy" conda
env created earlier this session: numpy 1.19.5, pandas 1.4.4, scipy
1.9.1) -- gaborbandit_analysis-master's unmodified code (np.int,
DataFrame.append) does not run on numpy>=1.24 / pandas>=2.0.
"""
import sys
import os

GB_LEGACY_PATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gb_legacy_patch')
GB_ORIGINAL_DIR = '/Users/prashantig/Brown Dropbox/Prashanti Ganesh/PhD/Semester 8/eLife_MS/gaborbandit_analysis-master'
sys.path.insert(0, GB_LEGACY_PATCH_DIR)  # patched GbAgent.py takes precedence
sys.path.insert(1, GB_ORIGINAL_DIR)      # everything else (GbTask, GbTaskVars, ...) comes from the untouched original

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from GbTaskVars import TaskVars
from GbAgentVars import AgentVars
from GbSimVars import SimVars
from gb_task_agent_int import gb_task_agent_int


def simulate(sigma, mu_true, n_trials, n_sims, task_agent_analysis):
    mu_est = np.zeros((n_sims, n_trials))

    task_vars = TaskVars()
    task_vars.T = n_trials
    task_vars.experiment = 3
    task_vars.mu = mu_true

    agent_vars = AgentVars()
    agent_vars.sigma = sigma
    agent_vars.beta = 100
    agent_vars.c0 = np.array([1])
    agent_vars.kappa = task_vars.kappa
    agent_vars.agent = 1
    agent_vars.lambda_param = np.nan
    agent_vars.task_agent_analysis = task_agent_analysis
    agent_vars.eval_ana = 1

    sim_vars = SimVars()
    sim_vars.take_pd = 0

    for sim in range(n_sims):
        sim_vars.block = sim
        df = gb_task_agent_int(task_vars, agent_vars, sim_vars)
        mu_est[sim, :] = df['e_mu_t'].to_numpy()

    return mu_est


def lighten(color_name, frac_to_white=0.75):
    r, g, b = mcolors.to_rgb(color_name)
    return (r + (1 - r) * frac_to_white, g + (1 - g) * frac_to_white, b + (1 - b) * frac_to_white)


def main():
    sigma_values = [0.02, 0.04, 0.06, 0.08, 0.1]
    mu_conditions = [0.7, 0.9]
    n_trials = 40  # capped for eval_ana=1's numerical stability -- see module docstring
    n_sims = 50

    no_int_color = 'tab:blue'
    no_int_color_light = lighten(no_int_color)
    int_color = 'tab:orange'
    int_color_light = lighten(int_color)

    fig, axes = plt.subplots(len(mu_conditions), len(sigma_values), figsize=(18, 6))
    trials = np.arange(1, n_trials + 1)

    for ci, mu_true in enumerate(mu_conditions):
        for si, sigma in enumerate(sigma_values):
            print(f'Simulating mu={mu_true}, sigma={sigma} ...')
            mu_est_noInt = simulate(sigma, mu_true, n_trials, n_sims, task_agent_analysis=False)
            mu_est_withInt = simulate(sigma, mu_true, n_trials, n_sims, task_agent_analysis=True)

            ax = axes[ci, si]

            # Individual-simulation lines drawn first so the mean+SEM overlays sit on top.
            for sim in range(n_sims):
                ax.plot(trials, mu_est_noInt[sim, :], color=no_int_color_light, linewidth=0.3)
                ax.plot(trials, mu_est_withInt[sim, :], color=int_color_light, linewidth=0.3)

            mean_noInt = mu_est_noInt.mean(axis=0)
            sem_noInt = mu_est_noInt.std(axis=0) / np.sqrt(n_sims)
            mean_withInt = mu_est_withInt.mean(axis=0)
            sem_withInt = mu_est_withInt.std(axis=0) / np.sqrt(n_sims)

            line_noInt, = ax.plot(trials, mean_noInt, color=no_int_color, linewidth=1.5)
            ax.fill_between(trials, mean_noInt - sem_noInt, mean_noInt + sem_noInt, color=no_int_color, alpha=0.3)
            line_withInt, = ax.plot(trials, mean_withInt, color=int_color, linewidth=1.5)
            ax.fill_between(trials, mean_withInt - sem_withInt, mean_withInt + sem_withInt, color=int_color, alpha=0.3)
            line_trueMu = ax.axhline(mu_true, linestyle='--', color='k', linewidth=1)

            ax.set_ylim(0, 1)
            ax.set_title(f'mu={mu_true}, sigma={sigma}', fontsize=9)
            if ci == len(mu_conditions) - 1:
                ax.set_xlabel('Trial')
            if si == 0:
                ax.set_ylabel('Simulated mu')

    fig.legend([line_noInt, line_withInt, line_trueMu], ['No Integration', 'Integration', 'True mu'],
               loc='center right', frameon=False)
    fig.suptitle('Simulated mu: gaborbandit_analysis-master, With vs Without Integration (eval_ana=1, o_t=c_t)', fontweight='bold')
    fig.tight_layout(rect=(0, 0, 0.93, 1))

    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(os.path.join(out_dir, 'gb_legacy_simulatedMu_analytical_withVsNoInt_longBlock40.png'), dpi=200)

    plt.show()


if __name__ == '__main__':
    main()
