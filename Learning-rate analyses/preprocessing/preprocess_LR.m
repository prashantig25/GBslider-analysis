classdef preprocess_LR < preprocess_vars
% PREPROCESS_LR initialises, computes and preprocesses required regressors for
% model based analyses.
        methods
            function obj = preprocess_LR()

            % The contructor methods initialises all other properties of
            % the class that are computed based on exisitng static properties of
            % the class.

            obj.data = readtable(obj.filename);
            obj.mu = obj.data.mu;
            obj.obtained_reward = obj.data.correct;
            if obj.online == 1 % depends on which dataset the analysis is being performed
                obj.condition = obj.data.choice_cond;
                obj.action = obj.data.choice;
            elseif obj.agent == 1
                obj.condition = obj.data.condition;
                obj.action = obj.data.action;
            elseif obj.pupil == 1
                obj.condition = obj.data.condition;
                obj.action = obj.data.choice;
            elseif obj.space == 1
                obj.condition = obj.data.condition;
                obj.action = obj.data.choice;
            end
            obj.state = obj.data.state;
            obj.flipped_mu = NaN(height(obj.data),1);
            obj.recoded_reward = NaN(height(obj.data),1);
            obj.mu_t = NaN(height(obj.data),1);
            obj.mu_t_1 = NaN(height(obj.data),1);
            if ~ismember('trials',obj.data.Properties.VariableNames)
                obj.data.trials = obj.data.trial;
            end
        end

        function flip_mu(obj)
            
            % function flip_mu computes the reported contingency parameter, after
            % correcting for incongruent blocks (eq. 16).
            %
            % INPUTS:
            %   obj: current object
            %
            % OUTPUT:
            %   obj.flipped_mu: congruence corrected reported
            %   contingency parameter
            
            for i = 1:height(obj.data)
                if obj.data.congruence(i) == 0 % for incongruent blocks
                    obj.flipped_mu(i) = 1-obj.mu(i);
                else
                    obj.flipped_mu(i) = obj.mu(i);
                end
           end
        end

        function compute_action_dep_rew(obj) 
            
            % function compute_action_dep_rew recodes task generated reward
            % contingent on action.
            %
            % INPUT:
            %   obj: current object
            %
            % OUTPUT:
            %   obj.recoded_reward: recoded reward for a = 0
            
            for i = 1:height(obj.data)
                obj.recoded_reward(i) = obj.obtained_reward(i) + (obj.action(i)*((-1) .^ ...
                    (2 + obj.obtained_reward(i))));
            end
        end
        
        function compute_mu(obj)
            
            % function compute_mu computes the reported contingency parameter,
            % depending on if actual mu < 0.5.
            %
            % INPUTS:
            %   obj: current object
            %
            % OUTPUT:
            %   obj.mu_t: reported contingency parameter for
            %   current trial
            %   obj.mu_t_1: reported contingency parameter for
            %   previous trial
            
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
            %   obj.pe: prediciton error
            %   obj.up: update
            
            for i = 2:height(obj.data)
                if obj.state(i) == 0
                    obj.data.pe(i) = obj.recoded_reward(i) - obj.mu_t_1(i);
                else
                    obj.data.pe(i) = (1-obj.recoded_reward(i))-obj.mu_t_1(i);
                end
                obj.data.up(i) = obj.mu_t(i) - obj.mu_t_1(i);
            end
            obj.data.pe(obj.data.trials == 1,1) = 0;
            if obj.absolute_lr == 1 % for absolute LR analysis
                obj.data.pe = abs(obj.data.pe);
                obj.data.up = abs(obj.data.up);
            end
        end

        function compute_confirm(obj)
            
            % function compute_confirm checks whether the outcome confirms the
            % choice.
            %
            % INPUT:
            %   obj: current object
            %
            % OUTPUT:
            %   obj.confirm_rew: if the outcome was confirming
            
            for i = 1:height(obj.data)
                if obj.data.contrast(i) == 1 % actual mu < 0.5
                    if obj.state(i) == obj.action(i) % the less rewarding state and action combination
                        obj.data.confirm_rew(i) = 1-obj.obtained_reward(i);
                    else
                        obj.data.confirm_rew(i) = obj.obtained_reward(i);
                    end
                else
                    if obj.state(i) ~= obj.action(i) % the less rewarding state and action combination
                        obj.data.confirm_rew(i) = 1-obj.obtained_reward(i);
                    else
                        obj.data.confirm_rew(i) = obj.obtained_reward(i);
                    end
                end
            end
        end

        function remove_conditions(obj)
            
            % function remove_conditions removes conditions that are not wanted for
            % further analysis.
            %
            % INPUTS:
            %   obj: current object
            %
            % OUTPUT:
            %   obj.condition: reduced condition array
            
            obj.data = obj.data(obj.condition ~= obj.removed_cond,:);
            if obj.agent == 0
                obj.condition = obj.data.choice_cond;
            else
                obj.condition = obj.data.condition;
            end
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
            % INPUTS:
            %   var_normalise: variable that needs to be normalised
            %
            % OUTPUTS:
            %   normalised: normalised variable
            
            norm_data = NaN(height(var_normalise),1);
            normalised = normalise_zero_one(var_normalise,norm_data);
        end

        function compute_ru(obj)
            
            % function compute_ru checks if reward uncertainty is high or low, given
            % the experimental condition.
            %
            % INPUTS:
            %   obj: current object
            %
            % OUTPUTS:
            %   obj.ru: reward uncertainty
            
           for i = 1:height(obj.data)
                if obj.condition(i) == 1
                    obj.data.ru(i) = 0;
                else
                    obj.data.ru(i) = 1;
                end
           end
        end

        function add_vars(obj,var,varname)
            
            % function add_vars adds array as table columns.
            %
            % INPUTS:
            %   obj: current object
            %   var: array to be added
            %   varname: table column name to be used
            %
            % OUTPUT:
            %   obj.data: table with added column
            
            obj.data = addvars(obj.data,var,'NewVariableNames',varname);
        end

        function remove_zero_pe(obj)
            
            % function remove_zero_pe gets rid of trials with PE = 0
            %
            % INPUTS:
            %   obj: current object
            %
            % OUTPUT:
            %   obj.data: table without PE = 0
            
            obj.data = obj.data(obj.data.pe ~= 0,:);
        end

        function add_splithalf(obj)
            
            % function add_splithalf splits and groups alternating trials into
            % different groups.
            %
            % INPUTS:
            %   obj: current object
            %
            % OUTPUT:
            %   obj.data.splithalf: split half variable for that trial
            
            for h = 1:height(obj.data)
                if mod(obj.data.trials(h),2) == 0
                    obj.data.splithalf(h) = 1;
                else
                    obj.data.splithalf(h) = 0;
                end
            end
        end

        function add_saliencechoice(obj)
            
            % function add_saliencechoice adds a categorical variable for whether
            % the more salient choice was made on a given trial.
            %
            % INPUTS:
            %   obj: current object
            %
            % OUTPUT:
            %   obj.data.saliencechoice: variable regarding the salient
            %   choice
            
            for i = 1:height(obj.data)
                if obj.data.contrast_left(i) > obj.data.contrast_right(i)
                    if obj.data.choice(i) == 0
                        obj.data.salience_choice(i) = 1;
                    else
                        obj.data.salience_choice(i) = 0;
                    end
                else
                    if obj.data.choice(i) == 1
                        obj.data.salience_choice(i) = 1;
                    else
                        obj.data.salience_choice(i) = 0;
                    end
                end
            end
        end

    end
end