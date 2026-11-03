"""
Runs the ORIGINAL gaborbandit_analysis-master code (GbTask, GbTaskVars,
GbAgentVars, gb_task_agent_int) with observation integration
(task_agent_analysis=True) and the numerical belief update (eval_ana=0),
across 5 sigma values and 2 true-mu conditions (0.7, 1.0), and plots
simulated mu over trials and economic performance.

gaborbandit_analysis-master is left completely untouched. The only change
is importing GbAgent from Agent/gb_legacy_patch/GbAgent.py instead of the
original -- a patched copy with the same compute_q/update_mu_from_q fix
applied to Agent.m this session (the unmodified original gets stuck at
exactly e_mu_t=0.5 forever for this agent/task_agent_analysis/eval_ana
combination -- see conversation). The original's o_t = c_t (exact,
noiseless observation) design is otherwise untouched, so this will still
show the overshoot-above-true-mu pattern found earlier, unlike
simulate_withInt_evalAna0.py's corrected design.

Must be run with an older numpy/pandas stack (e.g. the "gb_legacy" conda
env created this session: numpy 1.19.5, pandas 1.4.4, scipy 1.9.1) --
gaborbandit_analysis-master's unmodified code (np.int, DataFrame.append)
does not run on numpy>=1.24 / pandas>=2.0.
"""
import sys
import os

GB_LEGACY_PATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gb_legacy_patch')
GB_ORIGINAL_DIR = '/Users/prashantig/Brown Dropbox/Prashanti Ganesh/PhD/Semester 8/eLife_MS/gaborbandit_analysis-master'
sys.path.insert(0, GB_LEGACY_PATCH_DIR)  # patched GbAgent.py takes precedence
sys.path.insert(1, GB_ORIGINAL_DIR)      # everything else (GbTask, GbTaskVars, ...) comes from the untouched original

import numpy as np
import matplotlib.pyplot as plt
from GbTaskVars import TaskVars
from GbAgentVars import AgentVars
from GbSimVars import SimVars
from gb_task_agent_int import gb_task_agent_int


def simulate(sigma, mu_true, n_trials, n_sims):
    mu_est = np.zeros((n_sims, n_trials))
    correct = np.zeros((n_sims, n_trials))

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
    agent_vars.task_agent_analysis = True
    agent_vars.eval_ana = 0

    sim_vars = SimVars()
    sim_vars.take_pd = 0

    for sim in range(n_sims):
        sim_vars.block = sim
        df = gb_task_agent_int(task_vars, agent_vars, sim_vars)
        mu_est[sim, :] = df['e_mu_t'].to_numpy()
        correct[sim, :] = df['corr'].to_numpy()

    return mu_est, correct


def main():
    sigma_values = [0.02, 0.04, 0.06, 0.08, 0.1]
    mu_conditions = [0.7, 1.0]
    n_trials = 1000
    n_sims = 30
    nbins = 20

    fig_mu, axes_mu = plt.subplots(len(mu_conditions), len(sigma_values), figsize=(18, 6))
    fig_perf, axes_perf = plt.subplots(len(mu_conditions), len(sigma_values), figsize=(18, 6))

    for ci, mu_true in enumerate(mu_conditions):
        for si, sigma in enumerate(sigma_values):
            print(f'Simulating mu={mu_true}, sigma={sigma} ...')
            mu_est, correct = simulate(sigma, mu_true, n_trials, n_sims)
            trials = np.arange(1, n_trials + 1)

            ax = axes_mu[ci, si]
            mean_mu = mu_est.mean(axis=0)
            sem_mu = mu_est.std(axis=0) / np.sqrt(n_sims)
            ax.plot(trials, mean_mu, color='tab:blue')
            ax.fill_between(trials, mean_mu - sem_mu, mean_mu + sem_mu, color='tab:blue', alpha=0.3)
            ax.axhline(mu_true, linestyle='--', color='k', linewidth=1)
            ax.set_title(f'mu={mu_true}, sigma={sigma}', fontsize=9)
            if ci == len(mu_conditions) - 1:
                ax.set_xlabel('Trial')
            if si == 0:
                ax.set_ylabel('Mean simulated mu')

            ax2 = axes_perf[ci, si]
            bin_edges = np.linspace(0, n_trials, nbins + 1).astype(int)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mean_corr = np.zeros(nbins)
            sem_corr = np.zeros(nbins)
            for b in range(nbins):
                bin_vals = correct[:, bin_edges[b]:bin_edges[b + 1]]
                per_sim_mean = bin_vals.mean(axis=1)
                mean_corr[b] = per_sim_mean.mean()
                sem_corr[b] = per_sim_mean.std() / np.sqrt(n_sims)
            ax2.errorbar(bin_centers, mean_corr, yerr=sem_corr, marker='o', color='tab:blue', markersize=3)
            ax2.set_ylim(0, 1)
            ax2.set_title(f'mu={mu_true}, sigma={sigma}', fontsize=9)
            if ci == len(mu_conditions) - 1:
                ax2.set_xlabel('Trial')
            if si == 0:
                ax2.set_ylabel('Proportion correct')

    fig_mu.suptitle('Simulated mu: gaborbandit_analysis-master (patched stuck-at-0.5 fix only),\nWith Integration, eval_ana=0', fontweight='bold')
    fig_perf.suptitle('Economic performance: gaborbandit_analysis-master (patched stuck-at-0.5 fix only),\nWith Integration, eval_ana=0', fontweight='bold')
    fig_mu.tight_layout()
    fig_perf.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_mu.savefig(os.path.join(out_dir, 'gb_legacy_simulatedMu_withInt_evalAna0.png'))
    fig_perf.savefig(os.path.join(out_dir, 'gb_legacy_economicPerformance_withInt_evalAna0.png'))

    plt.show()


if __name__ == '__main__':
    main()
