classdef lr_analysis_obj < lr_vars
% LR_ANALYSIS_OBJ fits a linear regression model to single trial updates.
    methods
        function obj = lr_analysis_obj()
            
            % The contructor methods initialises all other properties of
            % the class that are computed based on exisitng static properties of
            % the class.
            
            obj.data = readtable(obj.filename);
        end

        function [betas,rsquared,residuals,coeffs_name,lm] = linear_fit(obj,tbl,fit_fn,varargin)
            
            % function linear_fit fits a linear regression model to the updates as a
            % function of prediction error and other task based computational
            % variables.
            %
            % INPUT:
            %   obj: current object
            %   tbl: table with predictor vars data
            %   varargin{1}: weights
            %   fit_fn: function to be used, adjust if using mock fitlm
            %   for unit testing
            %
            % OUTPUT:
            %   betas: array containing beta value for each predictor by fitlm
            %   rsquared: rsquared after fitting mdl to the data
            %   residuals: residuals after fitting mdl to the data
            %   coeffs_name: cell array containing name of all regressors
            %   lm: fitted model
            
            % FIT THE MODEL USING WEIGHTED/NON-WEIGHTED REGRESSION
            if obj.weight_y_n == 1
                lm = fit_fn(tbl,obj.mdl,'ResponseVar',obj.resp_var,'PredictorVars',obj.pred_vars, ...
                    'CategoricalVars',obj.cat_vars,'Weights',varargin{1});
            else
                lm = fit_fn(tbl,obj.mdl,'ResponseVar',obj.resp_var,'PredictorVars',obj.pred_vars, ...
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

        function [betas_all,rsquared_full,residuals_reg,coeffs_name,posterior_up_subjs] = get_coeffs(obj,fit_fn)
            
            % function get_coeffs fits the linear regression model by running non-weighted
            % and weighted regressions to get the beta coefficients across
            % subjects
            %
            % INPUT:
            %   obj: current object
            %
            % OUTPUT:
            %   betas_all: betas for all regressors
            %   rsqaured_full: r-squared values for each participant
            %   residuals_reg: residuals from fitting the model
            %   coeffs_name: cell array with the model generated coefficients
            %   name
            %   posterior_up_subjs: posterior predicted updates by model
            %   fit_fn: function to be used, adjust if using mock fitlm
            
            % SET VARIABLES TO RUN THE FUNCTION
            id_subjs = unique(obj.data.ID);
            
            % INITIALISE VARIABLES
            betas_all = NaN(length(obj.num_subjs),obj.num_vars);
            rsquared_full = NaN(length(obj.num_subjs),1);
            posterior_up_subjs = [];

            % CHECK IF ANALYSIS NEEDS TO BE RUN FOR ABSOLUTE OR SIGNED LRs
            if obj.absolute_analysis == 1
                obj.data.pe = abs(obj.data.pe);
                obj.data.up = abs(obj.data.up);
            end
            
            % FIT THE MODEL TO GET RESIDUALS (non-weighted)
            for i = 1:obj.num_subjs
                obj.weight_y_n = 0; % non-weighted
                data_subject = obj.data(obj.data.ID == id_subjs(i),:); % single-subject data
                tbl = table(data_subject.pe,data_subject.up, round(data_subject.norm_condiff,2), data_subject.contrast,...
                    data_subject.choice_cond,data_subject.congruence,data_subject.reward_unc,data_subject.pe_sign,data_subject.salience_choice,...
                    'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
                    ,'reward_unc','pe_sign','salience_choice'});
                if obj.pupil == 1
                    tbl = table(data_subject.pe,data_subject.up, round(data_subject.norm_condiff,2), data_subject.contrast,...
                    data_subject.condition,data_subject.congruence,data_subject.reward_unc,data_subject.pe_sign, ...
                    data_subject.salience_choice,normalise_zero_one(data_subject.patch_phasic).',...
                    normalise_zero_one(data_subject.patch_tonic).',normalise_zero_one(data_subject.fb_phasic).',...
                    normalise_zero_one(data_subject.fb_tonic).',...
                    'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
                    ,'reward_unc','pe_sign','salience_choice','patch_phasic','patch_tonic','fb_phasic','fb_tonic'});
                end
                [betas,rsquared,residuals_reg,coeffs_name,lm] = obj.linear_fit(tbl,fit_fn);
                obj.res_subjs = [obj.res_subjs; residuals_reg, repelem(id_subjs(i),length(residuals_reg)).'];
            end
            
            % WEIGHTED REGRESSION USING RESIDUALS
            if obj.weighted == 1
                obj.weight_y_n = 1;
                [wt_subjs] = weights_general(obj.data, obj.res_subjs); % get weights
                wt_subjs(:,2) = obj.res_subjs(:,2);
                for i = 1:obj.num_subjs
                    weights_subj = wt_subjs(wt_subjs(:,2) == id_subjs(i));
                    data_subject = obj.data(obj.data.ID == id_subjs(i),:);
                    tbl = table(data_subject.pe,data_subject.up, round(data_subject.norm_condiff,2), data_subject.contrast,...
                    data_subject.choice_cond,data_subject.congruence,data_subject.reward_unc,data_subject.pe_sign,data_subject.salience_choice,...
                    'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
                    ,'reward_unc','pe_sign','salience_choice'}); 
                    if obj.pupil == 1
                        tbl = table(data_subject.pe,data_subject.up, round(data_subject.norm_condiff,2), data_subject.contrast,...
                            data_subject.condition,data_subject.congruence,data_subject.reward_unc,data_subject.pe_sign, ...
                            data_subject.salience_choice,normalise_zero_one(data_subject.patch_phasic).',...
                            normalise_zero_one(data_subject.patch_tonic).',normalise_zero_one(data_subject.fb_phasic).',...
                            normalise_zero_one(data_subject.fb_tonic).',...
                            'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
                            ,'reward_unc','pe_sign','salience_choice','patch_phasic','patch_tonic','fb_phasic','fb_tonic'});
                    end
                    [betas,rsquared,residuals_reg,coeffs_name,lm] = obj.linear_fit(tbl,fit_fn,weights_subj);
                    betas_all(i,:) = betas(2:end);
                    rsquared_full(i,1) = rsquared;
                    [post_up] = obj.posterior_up(tbl,betas);
                    posterior_update = post_up;
                    posterior_up_subjs = [posterior_up_subjs; posterior_update];
                end
            end            
        end

        function [post_up] = posterior_up(obj,tbl,betas)
            
            % function posterior_up calculates the posterior updated predicted by the model
            % given the pe and other task/computational vars.
            %
            % INPUT:
            %   obj: current object
            %   tbl: table contatining all vars including pe, task/computational
            %   vars such as contrast difference and so on
            %   betas: beta values by the model
            %
            % OUTPUT:
            %   post_up: posterior update predicted by the model
            
            % INITIALISE OUTPUT AND OTHER VARS
            post_up = zeros(height(tbl),1);
            var_array = NaN(height(tbl),length(obj.var_names));
        
            % GET TRIAL-BASED VALUES FOR PREDICTOR VARS FOR Y_POST = X.*BETA + ERROR
            for v = 1:length(obj.var_names)
                var_array(:,v) = tbl.(obj.var_names{v});
            end
        
            % CALCULATE Y_POST
            post_up(:,1) = post_up(:,1) + betas(1); % add intercept
            for b = 2:length(betas)
                if b == 2
                    post_up(:,1) = post_up(:,1) + betas(b).*var_array(:,b-1); % main effect of PE
                else
                    post_up(:,1) = post_up(:,1) + betas(b).*var_array(:,1).*var_array(:,b-1); % interaction effect
                end
            end
        end
    end
end