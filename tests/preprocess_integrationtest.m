classdef preprocess_integrationtest < matlab.unittest.TestCase
    % PREPROCESS_INTEGRATIONTEST is an integration test for
    % various functions used for data pre-processing.
    
    methods(Test)
        
        function test_integration(obj)
            % Todo: documentaiton missing

            % INITIALIZE VARS
            preprocess_obj = preprocess_LR(); % class
            preprocess_obj.filename = "Data/descriptive data/main study/study2.txt"; % specify path to get the dataset
            preprocess_obj.initivaliseVars();
            num_trials = 20; % number of trials
            preprocess_obj.data = preprocess_obj.data(1:num_trials,:); % data table
            preprocess_obj.flipped_mu = NaN(num_trials,1); % congruence contingent flipped mu
            preprocess_obj.mu = linspace(0,1,num_trials); % generate mu
            preprocess_obj.data.congruence = [repelem(1,num_trials./2,1);repelem(0,num_trials./2,1)]; % set congruence
            preprocess_obj.data.contrast = [repelem(1,num_trials./2,1);repelem(0,num_trials./2,1)]; % set contrast
            rng(123)
            preprocess_obj.state = randi([0, 1],num_trials,1); % randomly generate state
            rng(23)
            preprocess_obj.action = randi([0, 1],num_trials,1); % randomly generate action 
            rng(31)
            preprocess_obj.obtained_reward = randi([0, 1],num_trials,1); % obtained reward
            preprocess_obj.removed_cond = 3; % which condition to be removed
            preprocess_obj.condition = preprocess_obj.data.choice_cond; % set condition
            preprocess_obj.data.trials = [1:num_trials].'; % trial number
            preprocess_obj.recoded_reward = NaN(num_trials,1); % array for recoded reward
            preprocess_obj.mu = rand(num_trials,1); % randomly generated mu
            preprocess_obj.mu_t = NaN(num_trials,1); % array for mu_t
            preprocess_obj.mu_t_1 = NaN(num_trials,1);

            % RUN THE PREPROCESSING PIPELINE
            preprocess_obj.flip_mu; % flip mu for congruence
            preprocess_obj.compute_action_dep_rew; % action dependent reward
            preprocess_obj.compute_mu; % compute mu for PE and UP
            preprocess_obj.compute_state_dep_pe; % compute PE and UP
            preprocess_obj.compute_confirm; % compute confirmation bias
            preprocess_obj.remove_conditions; % remove conditions
            preprocess_obj.compute_ru(); % reward uncertainty
            preprocess_obj.add_splithalf(); % split half variable
            preprocess_obj.add_saliencechoice(); % add salience choice variable
            norm_condiff = preprocess_obj.compute_normalise(linspace(0,0.1,num_trials).'); % normalised contrast difference
            preprocess_obj.add_vars(norm_condiff,{'norm_condiff'}); % add columns
            preprocess_obj.add_vars(preprocess_obj.data.ru,'reward_unc');
            preprocess_obj.add_vars(preprocess_obj.data.confirm_rew,'pe_sign');
            preprocess_obj.add_vars(preprocess_obj.flipped_mu,'flipped_mu');
            preprocess_obj.add_vars(preprocess_obj.mu_t,'mu_t');
            preprocess_obj.add_vars(preprocess_obj.mu_t_1,'mu_t_1');
            preprocess_obj.add_vars(preprocess_obj.recoded_reward,'recoded_reward');

            % EXPECTED
            expected_data = preprocess_obj.data;
            expected_data.flipped_mu = NaN(num_trials,1);
            expected_data.recoded_reward = NaN(num_trials,1);
            expected_data.mu_t = NaN(num_trials,1);
            expected_data.mu_t_1 = NaN(num_trials,1);
            expected_data.pe = NaN(num_trials,1);
            expected_data.up = NaN(num_trials,1);
            confirm_rew = NaN(num_trials,1);
            ru = NaN(num_trials,1);
            splithalf = NaN(num_trials,1);
            salience_choice = NaN(num_trials,1);

            % EXPECTED FLIPPED MU
            for i = 1:num_trials
                if preprocess_obj.data.congruence(i) == 0 % for incongruent blocks
                    expected_data.flipped_mu(i) = 1-preprocess_obj.mu(i);
                else
                    expected_data.flipped_mu(i) = preprocess_obj.mu(i);
                end
            end

            % EXPECTED RECODED REWARD
            for i = 1:num_trials
                expected_data.recoded_reward(i) = preprocess_obj.obtained_reward(i) + (preprocess_obj.action(i)*((-1) .^ ...
                    (2 + preprocess_obj.obtained_reward(i))));
            end

            % EXPECTED MU FOR CURRENT AND PREVIOUS TRIAL
            for i = 2:num_trials
                if preprocess_obj.data.contrast(i) == 1 % if actual mu < 0.5
                    expected_data.mu_t_1(i) = 1-expected_data.flipped_mu(i-1);
                    expected_data.mu_t(i) = 1-expected_data.flipped_mu(i);
                else
                    expected_data.mu_t_1(i) = expected_data.flipped_mu(i-1);
                    expected_data.mu_t(i) = expected_data.flipped_mu(i);
                end
            end

            % EXPECTED PE and UP
            for i = 2:num_trials
                if preprocess_obj.state(i) == 0
                    expected_data.pe(i) = expected_data.recoded_reward(i) - expected_data.mu_t_1(i);
                else
                    expected_data.pe(i) = (1-expected_data.recoded_reward(i))-expected_data.mu_t_1(i);
                end
                expected_data.up(i) = expected_data.mu_t(i) - expected_data.mu_t_1(i);
            end
            expected_data.pe(preprocess_obj.data.trials == 1,1) = 0;

            % EXPECTED VARIABLE FOR CHOICE CONFIRMATION
%             for i = 1:num_trials
%                 if preprocess_obj.data.contrast(i) == 1 % actual mu < 0.5
%                     if preprocess_obj.state(i) == preprocess_obj.action(i) % the less rewarding state and action combination
%                         confirm_rew(i) = 1-preprocess_obj.obtained_reward(i);
%                     else
%                         confirm_rew(i) = preprocess_obj.obtained_reward(i);
%                     end
%                 else
%                     if preprocess_obj.state(i) ~= preprocess_obj.action(i) % the less rewarding state and action combination
%                         confirm_rew(i) = 1-preprocess_obj.obtained_reward(i);
%                     else
%                         confirm_rew(i) = preprocess_obj.obtained_reward(i);
%                     end
%                 end
%             end

            % High contrast trials
            contrast_one_idx = preprocess_obj.data.contrast == 1;

            % actual mu < 0.5
            confirm_rew(contrast_one_idx & (preprocess_obj.state == preprocess_obj.action)) = 1 - preprocess_obj.obtained_reward(contrast_one_idx & (preprocess_obj.state == preprocess_obj.action)); % less rewarding state-action
            confirm_rew(contrast_one_idx & (preprocess_obj.state ~= preprocess_obj.action)) = preprocess_obj.obtained_reward(contrast_one_idx & (preprocess_obj.state ~= preprocess_obj.action));

            % actual mu > 0.5
            confirm_rew(~contrast_one_idx & (preprocess_obj.state ~= preprocess_obj.action)) = 1 - preprocess_obj.obtained_reward(~contrast_one_idx & (preprocess_obj.state ~= preprocess_obj.action)); % less rewarding state-action
            confirm_rew(~contrast_one_idx & (preprocess_obj.state == preprocess_obj.action)) = preprocess_obj.obtained_reward(~contrast_one_idx & (preprocess_obj.state == preprocess_obj.action));

            % EXPECTED REWARD UNCERTAINTY VARIABLE
            ru = preprocess_obj.condition ~= 1; % Set ru to 0 where condition is 1, and to 1 otherwise            

            % EXPECTED SPLITHALF VARIABLE
            even_trials_idx = mod(preprocess_obj.data.trials, 2) == 0; % Create a logical array where the trial numbers are even
            splithalf = even_trials_idx; % Set splithalf to 1 where trials are even, and 0 otherwise

            % EXPECTED SALIENCE CHOICE
            left_greater_idx = find(preprocess_obj.data.contrast_left > preprocess_obj.data.contrast_right); % contrast left is greater than contrast right
            right_greater_idx = find(preprocess_obj.data.contrast_left <= preprocess_obj.data.contrast_right); % contrast right is greater than contrast left

            % Set salience_choice to 1 or 0 based on choice for these indices
            salience_choice(left_greater_idx) = preprocess_obj.data.choice(left_greater_idx) == 0;
            salience_choice(right_greater_idx) = preprocess_obj.data.choice(right_greater_idx) == 1;
            
            % EXPECTED NORMALISATION
            norm_data = NaN(height(linspace(0,0.1,num_trials).'),1);
            normalised = normalise_zero_one(linspace(0,0.1,num_trials).',norm_data);

            % ADD VARIABLES
            expected_data.norm_condiff = normalised;
            expected_data.reward_unc = ru;
            expected_data.pe_sign = confirm_rew;
            expected_data.splithalf = splithalf;
            expected_data.salience_choice = salience_choice;

            % REMOVE PE = 0 FROM EXPECTED
            expected_data = expected_data(expected_data.pe ~= 0,:);
            preprocess_obj.remove_zero_pe(); % remove trials with PE = 0

            % TEST
            assert(isequaln(preprocess_obj.data, expected_data), 'Data does not match.');
        end
    end
end