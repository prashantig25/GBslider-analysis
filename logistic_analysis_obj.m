classdef logistic_analysis_obj < logistic_vars
% LOGISTIC_ANALYSIS_OBJ initialises, computes required regressors to fit a 
% logistic regression model to choice behaviour.

    methods
        % The contructor methods initialises all other properties of
        % the class that are computed based on exisitng static properties of
        % the class.
        function obj = logistic_analysis_obj
            obj.data = readtable(obj.filename);
            obj.con_diff = NaN(height(obj.data),1);
            obj.reward = obj.data.correct;
            obj.trials = obj.data.trials;
            obj.prev_rew = NaN(height(obj.data),1);
            obj.condition = obj.data.condition_int;
            obj.ru = NaN(height(obj.data),1);
            obj.cat_choice = NaN(height(obj.data),1);
            obj.h_vals = nan(1,obj.num_vars);
            obj.p_vals = nan(1,obj.num_vars); 
            obj.t_vals = nan(1,obj.num_vars);
        end
        
        function [categorical_var] = compute_cat(~,var)

            % COMPUTE_CAT converts a variable into categorical variable.
            % INPUTS:
                % obj =  current object
                % var = variable to be made cateogircal
            % OUTPUT:
                % categorical_var = categorical variable

            categorical_var = categorical(var);
        end

        function compute_condiff(obj)

            % COMPUTE_CONDIFF computes contrast differences.
            % INPUTS:
                % obj = current object
            % OUTPUT:
                % obj.con_diff = computed contrast difference

            obj.con_diff = obj.data.contrast_right - obj.data.contrast_left;
        end

        function [abs_var] = compute_abs(~,var)

            % COMPUTE_ABS converts a variable into absolute form.
            % INPUTS:
                % obj =  current object
                % var = input variable
            % OUTPUT:
                % abs_var = absolute variable

            abs_var = abs(var);
        end

        function compute_prevrew(obj)

            % COMPUTE_PREVREW computes previous reward for current trial.
            % INPUTS:
                % obj = current object
            % OUTPUT:
                % obj.prev_rew = previous trial's reward

            for i = 1:height(obj.data)-1
                obj.prev_rew(i+1) = obj.reward(i); % previous reward
            end
            obj.prev_rew(obj.trials == 1) = 0;
        end

        function remove_conditions(obj)

            % REMOVE_CONDITIONS removes conditions that are not wanted for
            % further analysis.
            % INPUTS:
                % obj = current object
            % OUTPUT:
                % obj.condition = reduced condition array

            obj.data = obj.data(obj.condition ~= obj.removed_cond,:);
            obj.condition = obj.data.condition;
        end

        function zscored = compute_nanzscore(~,var_zscore)

            % COMPUTE_NANZSCORE computes the z-score for a given variable.
            % INPUTS:
                % var_zscore = variable that needs to be z-scored
            % OUTPUTS:
                % zscored = z-scored variable

            zscored = nanzscore(var_zscore);
        end

        function normalised = compute_normalise(~,var_normalise)

            % COMPUTE_NORMALISE normalises a given variable.
            % INPUTS:
                % var_normalise = variable that needs to be normalised
            % OUTPUTS:
                % normalised = normalised variable

            norm_data = NaN(height(var_normalise),1);
            normalised = normalise_zero_one(var_normalise,norm_data);
        end

        function compute_ru(obj)

            % COMPUTE_RU checks if reward uncertainty is high or low, given
            % the experimental condition.
            % INPUTS:
                % obj = current object
            % OUTPUTS:
                % obj.ru = reward uncertainty

           for i = 1:height(obj.data)
                if obj.condition(i) == 1
                    obj.ru(i) = 0;
                else
                    obj.ru(i) = 1;
                end
           end
        end

        function add_vars(obj,var,varname)

            % ADD_VARS adds array as table columns.
            % INPUTS:
                % obj = current object
                % var = array to be added
                % varname = table column name to be used
            % OUTPUT:
                % obj.data = table with added column

            obj.data = addvars(obj.data,var,'NewVariableNames',varname);
        end

        function [betas,rsquared,residuals,coeffs_name,lm] = logistic_fit(obj,data)

            % LINEAR_FIT fits a logistic regression model to choices as a
            % function of task and computational
            % variables.
            % INPUT:
                % obj = current object
                % tbl = table with predictor vars data 
            % OUTPUT:
                % betas = array containing beta value for each predictor by fitlm
                % rsquared = rsquared after fitting mdl to the data
                % residuals  = residuals after fitting mdl to the data
                % coeffs_name = cell array containing name of all regressors
                % lm = fitted model

            lm = fitglm(data,obj.mdl,'Distribution',obj.distribution,'PredictorVars', ...
            obj.pred_vars,'ResponseVar',obj.resp_var,'CategoricalVars',obj.cat_vars);

            betas = nan(1,obj.num_vars+1);
            for b = 1:obj.num_vars+1
                betas(1,b) = lm.Coefficients.Estimate(b);
            end
            coeffs_name = lm.CoefficientNames;
            rsquared = lm.Rsquared.Ordinary;
            residuals = lm.Residuals;
        end

        function [betas_all,rsquared_all,h,p,t,coeffs_name] = get_coeffs(obj)

            % GET_COEFFS fits the logistic regression model to get the beta coefficients along with
            % significance testing for the betas.
            % INPUT:
                % obj = current object
            % OUTPUT:
                % betas_all = betas for all regressors
                % rsqaured_full = r-squared values for each participant
                % residuals_reg = residuals from fitting the model
                % coeffs_name = cell array with the model generated coefficients
                % name
                % h_vals = h-values from significance testing
                % p_vals = p-values from significance testing
                % posterior_up_subjs = posterior predicted updates by model

            % INITIALIZE VARS
            betas_all = NaN(length(obj.num_subjs),obj.num_vars+1);
            rsquared_all = NaN(length(obj.num_subjs),1);
            id_subjs = unique(obj.data.id);
            
            % FIT MODEL FOR EACH SUBJECT
            for i = 1:obj.num_subjs
                data_subject = obj.data(obj.data.id == id_subjs(i),:);
                choice = data_subject.correct_cat;
                condiff = data_subject.abs_diff;
                cond = data_subject.condition_int;
                contrast = data_subject.contrast;
                prevrew = data_subject.prev_rew;
                rew_unc = data_subject.reward_unc;
                tbl = table(choice, condiff, cond,contrast,prevrew,rew_unc,'VariableNames', ...
                    {'choice','con_diff','condition','contrast','prev_rew','reward_unc'});
                [betas,rsquared,~,coeffs_name,~] = obj.logistic_fit(tbl);
                betas_all(i,:) = betas(1:end);
                rsquared_all(i,:) = rsquared;
            end
            
            % significance testing
            
            % INITIALISE ARRAYS TO STORE h AND p VALUES
            h = nan(1,obj.num_vars);
            p = nan(1,obj.num_vars); 
            t = nan(1,obj.num_vars);
            
            % RUN TTESTS
            for i = 1:obj.num_vars
                [h(1,i),p(1,i),~,stats] = ttest(betas_all(:,i));
                t(1,i) = stats.tstat;
            end
            obj.h_vals = h;
            obj.p_vals = p;
            obj.t_vals = t;
        end
    end
end