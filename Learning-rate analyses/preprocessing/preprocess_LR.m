classdef preprocess_LR < preprocess_vars
    % PREPROCESS_LR initialises, computes and preprocesses required regressors for
    % model based analyses.

    methods

        function initivaliseVars(obj)
            % function initivaliseVars initializes all the required
            % variables for the preprocessing of data for LR analyses.
            %
            % INPUTS:
            %   obj: current object

            obj.data = readtable(obj.filename); % load the data
            obj.mu = obj.data.mu; % mu for further analyses
            obj.obtained_reward = obj.data.correct; % reward obtained by participant
            obj.flipped_mu = NaN(height(obj.data),1); % mu flipped for congruence
            if obj.online == 1 % depends on which dataset the analysis is being performed
                obj.condition = obj.data.choice_cond;
                obj.action = obj.data.choice;
            elseif obj.agent == 1
                obj.condition = obj.data.choice_cond;
                obj.action = obj.data.action;
                obj.obtained_reward = obj.data.reward;
                obj.flipped_mu = obj.data.mu;
            end
            obj.state = obj.data.state; % trial state
            obj.recoded_reward = NaN(height(obj.data),1); % variable to store recoded reward
            obj.mu_t = NaN(height(obj.data),1); % variable to store mu for that trial
            obj.mu_t_1 = NaN(height(obj.data),1); % variable to store mu for previous trial
            if ~ismember('trials',obj.data.Properties.VariableNames) % trial number
                obj.data.trials = obj.data.trial;
            end
        end
        function flip_mu(obj)
            % function flip_mu computes the reported contingency parameter, after
            % correcting for incongruent blocks (eq. 16).
            %
            % INPUTS:
            %   obj: current object

            incongruent_idx = obj.data.congruence == 0; % incongruent trials index
            obj.flipped_mu(incongruent_idx) = 1 - obj.mu(incongruent_idx); % for incongruent
            obj.flipped_mu(~incongruent_idx) = obj.mu(~incongruent_idx); % for congruent
        end

        function compute_action_dep_rew(obj)
            % function compute_action_dep_rew recodes task generated reward
            % contingent on action (obj.recoded_reward).
            %
            % INPUT:
            %   obj: current object

            recoding = obj.action .* ((-1) .^ (2 + obj.obtained_reward)); % recoding for action = 0
            obj.recoded_reward = obj.obtained_reward + recoding; % storing recoded values
        end

        function compute_mu(obj)
            % function compute_mu computes the reported contingency parameter,
            % depending on if actual mu < 0.5 for current (obj.mu_t)
            % and previous trial (obj.mu_t_1).
            %
            % INPUTS:
            %   obj: current object

            for i = 2:height(obj.data)
                if obj.data.contrast(i) == 1 % if actual mu < 0.5
                    obj.mu_t_1(i) = 1-obj.flipped_mu(i-1);
                    obj.mu_t(i) = 1-obj.flipped_mu(i);
                else
                    obj.mu_t_1(i) = obj.flipped_mu(i-1);
                    obj.mu_t(i) = obj.flipped_mu(i);
                end
            end
        end

        function [pe,up] = compute_state_dep_pe(obj)
            % function compute_state_dep_pe computes prediction error, using recoded
            % reward and contingent on state
            %
            % INPUTS:
            %   obj: current object
            %
            % OUTPUT:
            %   pe: prediciton error
            %   up: update

            state_zero_idx = obj.state == 0; % index for rows where state is 0

            % COMPUTE PE
            obj.data.pe(state_zero_idx) = obj.recoded_reward(state_zero_idx) - obj.mu_t_1(state_zero_idx); % state = 0
            obj.data.pe(~state_zero_idx) = (1 - obj.recoded_reward(~state_zero_idx)) - obj.mu_t_1(~state_zero_idx); % state = 1

            % COMPUTE UP
            obj.data.up = NaN(height(obj.data),1);
            obj.data.up(2:end) = obj.mu_t(2:height(obj.data)) - obj.mu_t_1(2:height(obj.data));
            obj.data.pe(obj.data.trials == 1,1) = 0;
        end

        function compute_confirm(obj)
            % function compute_confirm checks whether the outcome confirms the
            % choice (obj.confirm_rew).
            %
            % INPUT:
            %   obj: current object

            % High contrast trials
            contrast_one_idx = obj.data.contrast == 1;

            % actual mu < 0.5
            obj.data.confirm_rew(contrast_one_idx & (obj.state == obj.action)) = 1 - obj.obtained_reward(contrast_one_idx & (obj.state == obj.action)); % less rewarding state-action
            obj.data.confirm_rew(contrast_one_idx & (obj.state ~= obj.action)) = obj.obtained_reward(contrast_one_idx & (obj.state ~= obj.action));

            % actual mu > 0.5
            obj.data.confirm_rew(~contrast_one_idx & (obj.state ~= obj.action)) = 1 - obj.obtained_reward(~contrast_one_idx & (obj.state ~= obj.action)); % less rewarding state-action
            obj.data.confirm_rew(~contrast_one_idx & (obj.state == obj.action)) = obj.obtained_reward(~contrast_one_idx & (obj.state == obj.action));
        end

        function remove_conditions(obj)
            % function remove_conditions removes conditions that are not wanted for
            % further analysis.
            %
            % INPUTS:
            %   obj: current object

            obj.data = obj.data(obj.condition ~= obj.removed_cond,:);
            obj.condition = obj.data.choice_cond;
        end

        function zscored = compute_nanzscore(var_zscore)
            % function compute_nanzscore computes the z-score for a given
            % variable.
            %
            % INPUTS:
            %   var_zscore: variable that needs to be z-scored
            %
            % OUTPUTS:
            %   zscored: z-scored variable

            zscored = nanzscore(var_zscore);
        end

        function normalised = compute_normalise(~,var_normalise)
            % function compute_normalise normalises a given variable.
            %
            % INPUT:
            %   var_normalise: variable that needs to be normalised
            %
            % OUTPUT:
            %   normalised: normalised variable

            norm_data = NaN(height(var_normalise),1);
            normalised = normalise_zero_one(var_normalise,norm_data);
        end

        function compute_ru(obj)
            % function compute_ru checks if reward uncertainty is high or low, given
            % the experimental condition.
            %
            % INPUT:
            %   obj: current object

            obj.data.ru = obj.condition ~= 1; % Set ru to 0 where condition is 1, and to 1 otherwise
        end

        function add_vars(obj,var,varname)
            % function add_vars adds array as table columns.
            %
            % INPUTS:
            %   obj: current object
            %   var: array to be added
            %   varname: table column name to be used

            obj.data = addvars(obj.data,var,'NewVariableNames',varname);
        end

        function remove_zero_pe(obj)
            % function remove_zero_pe gets rid of trials with PE = 0
            %
            % INPUTS:
            %   obj: current object

            obj.data = obj.data(obj.data.pe ~= 0,:);
        end

        function add_splithalf(obj)
            % function add_splithalf splits and groups alternating trials into
            % different groups.
            %
            % INPUTS:
            %   obj: current object
            
            even_trials_idx = mod(obj.data.trials, 2) == 0; % Create a logical array where the trial numbers are even
            obj.data.splithalf = even_trials_idx; % Set splithalf to 1 where trials are even, and 0 otherwise
        end

        function add_saliencechoice(obj)
            % function add_saliencechoice adds a categorical variable for whether
            % the more salient choice was made on a given trial.
            %
            % INPUTS:
            %   obj: current object

            left_greater_idx = find(obj.data.contrast_left > obj.data.contrast_right); % contrast left is greater than contrast right
            right_greater_idx = find(obj.data.contrast_left <= obj.data.contrast_right); % contrast right is greater than contrast left

            % Set salience_choice to 1 or 0 based on choice for these indices
            obj.data.salience_choice(left_greater_idx) = obj.data.choice(left_greater_idx) == 0;
            obj.data.salience_choice(right_greater_idx) = obj.data.choice(right_greater_idx) == 1;
        end

    end
end