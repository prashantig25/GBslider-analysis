classdef lr_analysis_obj < lr_vars
% LR_ANALYSIS_OBJ initialises, computes required regressors to fit a linear
% regression model to single trial updates.
    methods
        function obj = lr_analysis_obj()
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
            obj.h_vals = nan(1,obj.num_vars);
            obj.p_vals = nan(1,obj.num_vars); 
            obj.t_vals = nan(1,obj.num_vars);
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

        function [betas,rsquared,residuals,coeffs_name,lm] = linear_fit(obj,tbl,varargin)
            % linear_fit fits a linear regression model to the updates as a
            % function of prediction error and other task based computational
            % variables.

            % INPUT:
                % obj: current object
                % tbl: table with predictor vars data 
                % varargin{1}: weights

            % OUTPUT:
                % betas: array containing beta value for each predictor by fitlm
                % rsquared: rsquared after fitting mdl to the data
                % residuals: residuals after fitting mdl to the data
                % coeffs_name: cell array containing name of all regressors
                % lm: fitted model
            % FIT THE MODEL USING WEIGHTED/NON-WEIGHTED REGRESSION
            if obj.weight_y_n == 1
                lm = fitlm(tbl,obj.mdl,'ResponseVar',obj.resp_var,'PredictorVars',obj.pred_vars, ...
                    'CategoricalVars',obj.cat_vars,'Weights',varargin{1});
            else
                lm = fitlm(tbl,obj.mdl,'ResponseVar',obj.resp_var,'PredictorVars',obj.pred_vars, ...
                    'CategoricalVars',obj.cat_vars);
            end
        
            % SAVE R-SQUARED, RESIDUALS AND BETA VALUES
            rsquared = lm.Rsquared.Adjusted;
            residuals = lm.Residuals.Raw;
            betas = nan(1,obj.num_vars+1);
            for b = 1:obj.num_vars+1
                betas(1,b) = lm.Coefficients.Estimate(b);
            end
            coeffs_name = lm.CoefficientNames;
        end

        function [betas_all,rsquared_full,residuals_reg,coeffs_name,posterior_up_subjs] = get_coeffs(obj)
            % get_coeffs fits the linear regression model by running non-weighted
            % and weighted regressions to get the beta coefficients along with
            % significance testing for the betas.

            % INPUT:
                % obj: current object

            % OUTPUT:
                % betas_all: betas for all regressors
                % rsqaured_full: r-squared values for each participant
                % residuals_reg: residuals from fitting the model
                % coeffs_name: cell array with the model generated coefficients
                % name
                % h_vals: h-values from significance testing
                % p_vals: p-values from significance testing
                % posterior_up_subjs: posterior predicted updates by model
            % SET VARIABLES TO RUN THE FUNCTION
            id_subjs = unique(obj.data.run_id);
            
            % INITIALISE VARIABLES
            betas_all = NaN(length(obj.num_subjs),obj.num_vars);
            rsquared_full = NaN(length(obj.num_subjs),1);
            posterior_up_subjs = [];
            
            % FIT THE MODEL TO GET RESIDUALS 
            for i = 1:obj.num_subjs
                obj.weight_y_n = 0;
                data_subject = obj.data(obj.data.run_id == id_subjs(i),:);
                tbl = table(data_subject.pe,data_subject.up, round(data_subject.norm_condiff,2), data_subject.contrast,...
                    data_subject.condition,data_subject.congruence,data_subject.reward_unc,data_subject.norm_subjest,data_subject.pe_sign,...
                    'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
                    ,'reward_unc','subj_est_unc','pe_sign'});
                [betas,rsquared,residuals_reg,coeffs_name,lm] = obj.linear_fit(tbl);
                obj.res_subjs = [obj.res_subjs; residuals_reg, repelem(id_subjs(i),length(residuals_reg)).'];
            end
            
            % WEIGHTED REGRESSION USING RESIDUALS
            if obj.weighted == 1
                [wt_subjs] = weights(obj.data, obj.res_subjs);
                wt_subjs(:,2) = obj.res_subjs(:,2);
                for i = 1:obj.num_subjs
                    weights_subj = wt_subjs(wt_subjs(:,2) == id_subjs(i));
                    data_subject = obj.data(obj.data.run_id == id_subjs(i),:);
                    tbl = table(data_subject.pe,data_subject.up, round(data_subject.norm_condiff,2), data_subject.contrast,...
                    data_subject.condition,data_subject.congruence,data_subject.reward_unc,data_subject.norm_subjest,data_subject.pe_sign,...
                    'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
                    ,'reward_unc','subj_est_unc','pe_sign'}); 
                    [betas,rsquared,residuals_reg,coeffs_name,lm] = obj.linear_fit(tbl,weights_subj);
                    betas_all(i,:) = betas(2:end);
                    rsquared_full(i,1) = rsquared;
                    [post_up] = obj.posterior_up(tbl,betas);
                    posterior_update = post_up;
                    posterior_up_subjs = [posterior_up_subjs; posterior_update];
                end
            end
            
            if obj.weighted == 1
                h = nan(1,obj.num_vars);
                p = nan(1,obj.num_vars); 
                t = nan(1,obj.num_vars);
                for i = 1:obj.num_vars
                    [h(1,i),p(1,i),~,stats] = ttest(betas_all(:,i));
                    t(1,i) = stats.tstat;
                end
                obj.h_vals = h;
                obj.p_vals = p;
                obj.t_vals = t;
            end
        end

        function [post_up] = posterior_up(obj,tbl,betas)
            % posterior_up calculates the posterior updated predicted by the model
            % given the pe and other task/computational vars. 

            % INPUT:
                % obj: current object
                % tbl: table contatining all vars including pe, task/computational
                % vars such as contrast difference and so on
                % betas: beta values by the model

            % OUTPUT:
                % post_up: posterior update predicted by the model
            % INITIALISE OUTPUT AND OTHER VARS
            post_up = zeros(height(tbl),1);
            var_array = NaN(height(tbl),length(obj.var_names));
        
            % GET X VARS FOR Y_POST = X.*BETA + ERROR
            for v = 1:length(obj.var_names)
                var_array(:,v) = tbl.(obj.var_names{v});
            end
        
            % CALCULATE Y_POST
            post_up(:,1) = post_up(:,1) + betas(1);
            for b = 2:length(betas)
                    if b == 2
                        post_up(:,1) = post_up(:,1) + betas(b).*var_array(:,b-1);
                    else
                        post_up(:,1) = post_up(:,1) + betas(b).*var_array(:,1).*var_array(:,b-1);
                    end
            end
        end
    end
end