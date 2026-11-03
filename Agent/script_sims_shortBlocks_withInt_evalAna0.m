%% ========================================================================
%  Script: Simulate Short Blocks (25 Trials x 25 Blocks) -- With
%  Integration, eval_ana = 0
%  ------------------------------------------------------------------------
%  Runs the normative Bayesian Agent with observation integration
%  (task_agent_analysis = 1) and the numerical (fixed-size grid) belief
%  update (eval_ana = 0), at the task's actual block length (25 trials),
%  repeated across 25 independent blocks per simulation -- each block
%  resets the agent's belief to the uniform prior (task_agent_int.m
%  constructs a fresh Agent() per block call), so this gives 25 replicate
%  short learning curves per simulated "subject" rather than one
%  continuous long block.
%  ------------------------------------------------------------------------
%  This uses Agent.m's current with-integration implementation, which
%  this session fixed in two ways (see conversation for the full
%  derivation):
%    1) Agent.observation_sample now always draws a genuinely noisy
%       observation (normrnd(c_t,sigma)) regardless of
%       task_agent_analysis, instead of task_agent_analysis=1 previously
%       feeding the exact, noiseless c_t through as o_t. That exact-
%       observation design, combined with p_s_giv_o's sigma-based
%       confidence formula, manufactured artificial doubt on
%       small-contrast trials (where the true state was already known
%       for certain), which systematically biased the inferred mu
%       upward.
%    2) agentvars.m's set_o grid widened from +-0.1 to +-0.25 (2.5x
%       padding beyond kappa_max, matching GbAgentVars.py's own
%       set_o/kappa ratio), reducing the residual truncation bias left
%       from integrate_voi's second averaging pass around the
%       already-noisy observation.
%  Swept across the same sigma values and mu-conditions as
%  script_sims.m's long-block sections, for direct comparability.
%  contrast/congruence are not meaningful for this sigma-sweep (unlike
%  the fixed-condition blocks in script_sims.m's top section), so they
%  are left unset (NaN, task_agent_int.m's own default) here.
%  ========================================================================
clc
clearvars

% PATH STUFF
currentDir = cd;
reqPath = 'GBslider-analysis';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Data', 'agent simulations');

%% SIMULATE SHORT BLOCKS ACROSS SIGMA VALUES, WITH INTEGRATION, EVAL_ANA = 0
num_trials_short = 25; % matches taskvars.m's default block length (the task's real block length)
num_blocks_short = 1; % 25 independent blocks per simulation -- agent's belief resets each block
num_sims_short = 30;   % independent simulated "subjects" -- combined with num_blocks_short, gives 750 replicate blocks per (sigma, condition)
sigma_values_short = 0.06; %[0.02, 0.04, 0.06, 0.08, 0.1];
conditions_short = [1, 2]; % condition 1 -> true mu 0.7, condition 2 -> true mu 1 (rangeMu = 0, same mapping as script_sims.m)
beta_short = 100; % Agent's own default (agentvars.m) -- only sigma is swept here, beta held fixed
agent_short = 1; % normative agent
task_agent_analysis_short = 1; % with integration
eval_ana_short = 0; % numerical (mu_for_ev-based) belief update

simulations_short = [];
for sigma_short = sigma_values_short
    for condition_short = conditions_short
        sim_chunk = [];
        [sim_chunk] = run_simulations(num_sims_short,num_blocks_short,sim_chunk,condition_short,agent_short,0,task_agent_analysis_short,sigma_short,beta_short,num_trials_short,eval_ana_short); % num_trials_short/eval_ana_short threaded through task_agent_int.m's varargin{3}/varargin{4}
        sim_chunk.sigma = repmat(sigma_short,height(sim_chunk),1);
        sim_chunk.choice_cond = repmat(condition_short,height(sim_chunk),1);
        simulations_short = [simulations_short; sim_chunk];
    end
end
safe_saveall(fullfile(save_dir,'data_agent_shortBlocks_withInt_evalAna0.txt'),simulations_short);
