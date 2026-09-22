%% ========================================================================
%  Script: Verify recode_rewards_choice matches the old inline logic
%  ------------------------------------------------------------------------
%  Checks that fitSlider_ALLmodels.recode_rewards_choice produces exactly
%  the same output as (a) the old vectorized logic that used to live
%  inline in recovery_ReducedModelSpace.m, and (b) the old scalar if/else
%  logic that used to live inline in simulate_basicRL_integrated_choice /
%  simulate_RLsigma_integrated_choice, before both were replaced by calls
%  to the new shared function.
%  ========================================================================
clc; clearvars;
rng(1);

n = 10000;
reward = double(rand(n, 1) < 0.5);
choice = double(rand(n, 1) < 0.5);

%% Old vectorized logic (used to live inline in recovery_ReducedModelSpace.m)
recoded_old_vectorized = reward;
is_choice1 = choice == 1;
recoded_old_vectorized(is_choice1) = 1 - recoded_old_vectorized(is_choice1);

%% Old scalar if/else logic (used to live inline in the two simulate_* functions)
recoded_old_scalar = NaN(n, 1);
for k = 1:n
    if choice(k) == 0
        recoded_old_scalar(k) = reward(k);
    else
        recoded_old_scalar(k) = 1 - reward(k);
    end
end

%% New shared function
recoded_new = fitSlider_ALLmodels.recode_rewards_choice(reward, choice);

%% Compare
fprintf('n trials tested: %d\n', n);
fprintf('new vs old vectorized: %s\n', mat2str(isequal(recoded_new, recoded_old_vectorized)));
fprintf('new vs old scalar:     %s\n', mat2str(isequal(recoded_new, recoded_old_scalar)));
fprintf('old vectorized vs old scalar: %s\n', mat2str(isequal(recoded_old_vectorized, recoded_old_scalar)));

assert(isequal(recoded_new, recoded_old_vectorized), 'Mismatch: new function vs old vectorized logic');
assert(isequal(recoded_new, recoded_old_scalar), 'Mismatch: new function vs old scalar logic');
disp('PASS: recode_rewards_choice matches both old versions exactly.');
