"""
Simulates the normative Bayesian Agent with observation integration
(task_agent_analysis=1) and the numerical belief update (eval_ana=0),
matching Agent.m's current implementation -- across 5 sigma values and
2 true-mu conditions (0.7, 1.0) -- and plots simulated mu over trials and
economic performance, mirroring plotLongBlocks_integrationComparison.m's
figures (condition x sigma subplot grid, true-mu reference line).
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

KAPPA_MAX = 0.1
C = np.linspace(-KAPPA_MAX, KAPPA_MAX, 20)   # contrast-difference grid (taskvars.m)
SET_O = np.linspace(-0.25, 0.25, 20)         # observation grid (agentvars.m, widened)
P_MU = np.linspace(0, 1, 100)                # belief grid over mu (Agent.m)
THETA = 0.5                                  # p(state) (taskvars.m)
BETA = 100.0                                 # softmax inverse temperature

P_CS0 = (C < 0).astype(float); P_CS0 /= P_CS0.sum()
P_CS1 = (C >= 0).astype(float); P_CS1 /= P_CS1.sum()


def pi0_pi1(o, sigma):
    u = norm.cdf(0, o, sigma)
    v = norm.cdf(-KAPPA_MAX, o, sigma)
    w = norm.cdf(KAPPA_MAX, o, sigma)
    denom = w - v
    pi_0 = np.where(denom == 0, np.where(o < 0, 1.0, 0.0), (u - v) / np.where(denom == 0, 1, denom))
    pi_1 = np.where(denom == 0, np.where(o < 0, 0.0, 1.0), (w - u) / np.where(denom == 0, 1, denom))
    return pi_0, pi_1


def p_reward(a, s, mu):
    return mu if a == s else 1 - mu


def simulate(sigma, mu_true, n_trials, n_sims, rng):
    mu_est = np.zeros((n_sims, n_trials))
    correct = np.zeros((n_sims, n_trials))
    pi0_grid, pi1_grid = pi0_pi1(SET_O, sigma)

    for sim in range(n_sims):
        product = np.ones(100)  # uniform prior over P_MU
        for t in range(n_trials):
            s_t = rng.binomial(1, THETA)
            c_t = rng.choice(C, p=P_CS1 if s_t == 1 else P_CS0)
            o_t = rng.normal(c_t, sigma)  # observation_sample: noisy for task_agent_analysis=1 & eval_ana=0

            w = norm.pdf(SET_O, o_t, sigma)
            if w.sum() > 0:
                w = w / w.sum()
            else:
                w = np.zeros_like(SET_O)
                w[np.argmin(np.abs(SET_O - o_t))] = 1.0

            mu_for_ev = product / product.sum()
            e_mu_t = np.dot(mu_for_ev, P_MU)

            v_a0 = (pi0_grid - pi1_grid) * e_mu_t + pi1_grid
            v_a1 = (pi1_grid - pi0_grid) * e_mu_t + pi0_grid
            v_a0_bar = np.sum(v_a0 * w)
            v_a1_bar = np.sum(v_a1 * w)
            p_a1 = np.exp(v_a1_bar * BETA) / (np.exp(v_a0_bar * BETA) + np.exp(v_a1_bar * BETA))
            a_t = rng.binomial(1, p_a1)

            r_raw = rng.binomial(1, p_reward(a_t, s_t, mu_true))
            r_coded = r_raw + a_t * ((-1) ** (2 + r_raw))

            q0_o = (pi1_grid ** r_coded) * (pi0_grid ** (1 - r_coded))
            q1_o = ((-1) ** (r_coded + 1)) * (pi0_grid - pi1_grid)
            q0_bar = np.sum(q0_o * w)
            q1_bar = np.sum(q1_o * w)

            product = product * (q1_bar * P_MU + q0_bar)
            total = product.sum()
            product = product / total if total > 0 else np.ones(100) / 100

            mu_for_ev = product / product.sum()
            mu_est[sim, t] = np.dot(mu_for_ev, P_MU)
            correct[sim, t] = 1 if a_t == s_t else 0

    return mu_est, correct


def main():
    sigma_values = [0.02, 0.04, 0.06, 0.08, 0.1]
    mu_conditions = [0.7, 1.0]
    n_trials = 1000
    n_sims = 30
    nbins = 20
    rng = np.random.default_rng(0)

    fig_mu, axes_mu = plt.subplots(len(mu_conditions), len(sigma_values), figsize=(18, 6))
    fig_perf, axes_perf = plt.subplots(len(mu_conditions), len(sigma_values), figsize=(18, 6))

    for ci, mu_true in enumerate(mu_conditions):
        for si, sigma in enumerate(sigma_values):
            mu_est, correct = simulate(sigma, mu_true, n_trials, n_sims, rng)
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

    fig_mu.suptitle('Simulated mu: With Integration, eval_ana=0 (Python replica)', fontweight='bold')
    fig_perf.suptitle('Economic performance: With Integration, eval_ana=0 (Python replica)', fontweight='bold')
    fig_mu.tight_layout()
    fig_perf.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
