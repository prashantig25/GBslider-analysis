classdef lr_analysis_obj < lr_vars
% LR_ANALYSIS_OBJ fits a linear regression model to single trial updates.
    methods
        function obj = lr_analysis_obj()
            % The contructor methods initialises all other properties of
            % the class that are computed based on exisitng static properties of
            % the class.
            obj.data = readtable(obj.filename);
            obj.h_vals = nan(1,obj.num_vars);
            obj.p_vals = nan(1,obj.num_vars); 
            obj.t_vals = nan(1,obj.num_vars);
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

            % CHECK IF ANALYSIS NEEDS TO BE RUN FOR ABSOLUTE OR SIGNED LRs
            if obj.absolute_analysis == 1
                obj.data.pe = abs(obj.data.pe);
                obj.data.up = abs(obj.data.up);
            end
            
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
                [wt_subjs] = weights_general(obj.data, obj.res_subjs);
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