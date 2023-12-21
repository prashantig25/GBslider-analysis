classdef preprocess < preproc_vars
% PREPROCESS initialises, computes and preprocesses required regressors for
% model based analyses.
        methods
            function obj = preprocess()
            % The contructor methods initialises all other properties of
            % the class that are computed based on exisitng static properties of
            % the class.
            obj.data = readtable(obj.filename);
            obj.mu = obj.data.mu;
            obj.flipped_mu = obj.mu;
            obj.incorr_mu = obj.mu;
            obj.correct_choice = obj.data.choice_corr;
            obj.mu_pe = obj.mu;
            obj.obtained_reward = obj.data.correct;
            obj.pe = zeros(height(obj.data),1);
            obj.up = zeros(height(obj.data),1);
            obj.expected_reward = zeros(height(obj.data),1);
            obj.condition = obj.data.choice_cond;
            obj.subjest = zeros(height(obj.data),1);
            obj.ru = NaN(height(obj.data),1);
            obj.pe_sign = NaN(height(obj.data),1);
        end

        function flip_mu(obj)
            % flip_mu computes the reported contingency parameter, after
            % correcting for incongruent blocks. 

            % INPUTS:
                % obj: current object

            % OUTPUT:
                % obj.flipped_mu: congruence corrected reported
                % contingency parameter
            for i = 1:height(obj.data)
                if obj.data.congruence(i) == 0 % for incongruent blocks
                    obj.flipped_mu(i) = 1-obj.mu(i);
                else
                    obj.flipped_mu(i) = obj.mu(i);
                end
           end
        end

        function compute_incorr_mu(obj)
            % compute_incorr_mu computes the reported contingency parameter
            % for the less rewarding option.

            % INPUTS:
                % obj: current object

            % OUTPUT:
                % obj.incorr_mu: reported contingency parameter for the
                % less rewarding option
            for i = 1:height(obj.data) % for less rewarding option
                    if obj.data.congruence(i) == 0
                        obj.incorr_mu(i) = obj.mu(i);
                    else
                        obj.incorr_mu(i) = 1-obj.mu(i);
                    end
             end
        end

        function compute_mu_pe(obj)
            % compute_mu_pe computes the reported contingency parameter,
            % given the rewards obtained by the participant.

            % INPUTS:
                % obj: current object
               
            % OUTPUT:
                % obj.mu_pe: reported contingency parameter for
                % predicition errors
            for i = 2:height(obj.data)
                 if obj.correct_choice(i) == 1 % using observed rewards
                    obj.mu_pe(i-1) = obj.flipped_mu(i-1);
                 elseif obj.correct_choice(i) == 0 
                    obj.mu_pe(i-1) = 1-obj.flipped_mu(i-1);
                 end
            end
        end

        function get_rew_mu(obj)
            % get_rew_mu recodes reported vontingency parameter (mu) for 
            % incongruent blocks, incorrect option and mu to compute 
            % prediction error.

            % INPUTS:
                % obj: current obj
            if obj.agent == 1 % if recoding for agent
                obj.flipped_mu = obj.mu;
                obj.incorr_mu = 1-obj.mu;
            else
                obj.flip_mu();
                obj.compute_incorr_mu();
            end
            obj.compute_mu_pe();
        end

        function [pe,up] = get_pe_up(obj)
            % get_pe_up computes action contingent prediction errors and updates.

            % INPUT:
                % obj: current object

            % OUTPUT:
                % obj.pe: prediciton error
                % obj.up: update
            % COMPUTE PE, UP FOR SINGLE TRIALS
            obj.expected_reward = obj.mu_pe;
            for t = 2:length(obj.obtained_reward)
                obj.pe(t) = obj.obtained_reward(t) - obj.expected_reward(t-1);
                if obj.correct_choice(t) == 1
                    obj.up(t) = obj.flipped_mu(t)-obj.flipped_mu(t-1);
                else
                    obj.up(t) = obj.incorr_mu(t)-obj.incorr_mu(t-1);
                end
            end
        end

        function compute_subjest(obj)
            % compute_subjest computes single trial subjective estimation uncertainty.

            % INPUT:
                % obj: current object

            % OUTPUT:
                % obj.subj_est: subjective estimation uncertainty for each trial
            obj.subjest = obj.mu.*(1-obj.mu);
        end

        function remove_conditions(obj)
            % remove_conditions removes conditions that are not wanted for
            % further analysis.

            % INPUTS:
                % obj: current object

            % OUTPUT:
                % obj.condition: reduced condition array
            obj.data = obj.data(obj.condition ~= obj.removed_cond,:);
            obj.condition = obj.data.condition;
        end

        function zscored = compute_nanzscore(var_zscore)
            % COMPUTE_NANZSCORE computes the z-score for a given variable.

            % INPUTS:
                % var_zscore: variable that needs to be z-scored

            % OUTPUTS:
                % zscored: z-scored variable
            zscored = nanzscore(var_zscore);
        end

        function normalised = compute_normalise(~,var_normalise)
            % compute_normalise normalises a given variable.

            % INPUTS:
                % var_normalise: variable that needs to be normalised
            % OUTPUTS:
                % normalised: normalised variable
            norm_data = NaN(height(var_normalise),1);
            normalised = normalise_zero_one(var_normalise,norm_data);
        end

        function compute_ru(obj)
            % compute_ru checks if reward uncertainty is high or low, given
            % the experimental condition.

            % INPUTS:
                % obj: current object

            % OUTPUTS:
                % obj.ru: reward uncertainty
           for i = 1:height(obj.data)
                if obj.condition(i) == 1
                    obj.ru(i) = 0;
                else
                    obj.ru(i) = 1;
                end
           end
        end

        function compute_pesign(obj)
            % compute_pesign checks if prediction error is positive or
            % negative.

            % INPUTS:
                % obj: current object

            % OUTPUTS:
                % obj.pe_sign: sign of prediction error
            for i = 1:height(obj.data)
                if obj.pe(i) > 0
                    obj.pe_sign(i) = 1;
                else
                    obj.pe_sign(i) = 0;
                end
            end
        end

        function add_vars(obj,var,varname)
            % add_vars adds array as table columns.

            % INPUTS:
                % obj: current object
                % var: array to be added
                % varname: table column name to be used

            % OUTPUT:
                % obj.data: table with added column
            obj.data = addvars(obj.data,var,'NewVariableNames',varname);
        end

        function remove_zero_pe(obj)
            % remove_zero_pe gets rid of trials with PE = 0

            % INPUTS:
                % obj: current object

            % OUTPUT:
                % obj.data: table without PE = 0
            obj.data = obj.data(obj.data.pe ~= 0,:);
        end
    end
end