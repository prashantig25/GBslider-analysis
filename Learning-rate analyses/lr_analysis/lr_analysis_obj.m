classdef lr_analysis_obj < lr_vars
    % LR_ANALYSIS_OBJ fits a linear regression model to single trial updates.
    
    methods

        function initialiseVars(obj)
            % function initivaliseVars initializes all the required
            % variables for the preprocessing of data for LR analyses.
            %
            % INPUTS:
            %   obj: current object

            obj.data = importdata(obj.filename);
        end

        function compute_numvars(obj)
            % function compute_numvars computes the number of variables in
            % the model definition.
            %
            % INPUT:
            %   obj: current object

            % SPLIT THE FORMULA
            rhs = split(obj.mdl, '~');
            rhs = strtrim(rhs{2}); % trim any leading/trailing whitespace

            % GET NUM_VARS
            terms = strtrim(split(rhs, '+'));
            unique_terms = unique(terms);
            obj.num_vars = length(unique_terms);
        end

        function model_definition(obj,varargin)
            % function model_definition defines the regression model
            % equation for the desired analysis.
            %
            % INPUT:
            %   obj: current object

            if obj.lr_mdl == 1
                obj.mdl = 'up ~ pe + pe:salience + pe:congruence + pe:pe_sign + pe:contrast_diff';
                obj.compute_numvars;
            elseif obj.risk_mdl == 1
                obj.mdl = 'up ~ pe + pe:salience + pe:congruence + pe:pe_sign + pe:contrast_diff + pe:reward_unc';
                obj.compute_numvars;
            elseif obj.saliencechoice_mdl == 1
                obj.mdl = 'up ~ pe + pe:contrast_diff + pe:salience_choice + pe:congruence + pe:pe_sign ';
                obj.compute_numvars;
            elseif obj.EEanalysis == 1
                obj.mdl = varargin{1};
                obj.compute_numvars;
            end

            if obj.agent == 1
                obj.pred_vars = {'pe','salience','contrast_diff','congruence','reward_unc','reward','mu','pe_sign'}; % cell array with names of predictor variables
                obj.cat_vars = {'salience','congruence','condition','reward_unc','pe_sign'}; % cell array with names of categorical variables
                obj.resp_var = 'up';
            elseif obj.online == 1
                obj.pred_vars = {'pe','salience','contrast_diff','congruence','reward_unc','reward','mu','pe_sign','salience_choice'}; % cell array with names of predictor variables
                obj.cat_vars = {'salience','congruence','condition','reward_unc','pe_sign','salience_choice'}; % cell array with names of categorical variables
                obj.resp_var = 'up';
            elseif obj.EEanalysis == 1
                obj.pred_vars = {'pe','pe__condiff','pe__salience','pe__congruence','pe__pesign'}; % variable names
                obj.cat_vars = '';
                obj.resp_var = 'perf';
            end
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

        function [betas_all,rsquared_full,residuals_reg,coeffs_name,posterior_all] = get_coeffs(obj,fit_fn,predict_fn)
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
            posterior_all = cell(length(obj.num_subjs),1);
            obj.res_subjs = [];

            % CHECK IF ANALYSIS NEEDS TO BE RUN FOR ABSOLUTE OR SIGNED LRs
            if obj.absolute_analysis == 1
                obj.data.pe = abs(obj.data.pe);
                obj.data.up = abs(obj.data.up);
            end

            % FIT THE MODEL TO GET RESIDUALS (non-weighted)
            for i = 1:obj.num_subjs
                obj.weight_y_n = 0; % non-weighted
                data_subject = obj.data(obj.data.ID == id_subjs(i),:); % single-subject data
                if obj.online == 1
                    tbl = table(data_subject.pe,data_subject.up, round(data_subject.norm_condiff,2), data_subject.contrast,...
                        data_subject.choice_cond,data_subject.congruence,data_subject.reward_unc,data_subject.pe_sign,data_subject.salience_choice,...
                        'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
                        ,'reward_unc','pe_sign','salience_choice'});
                elseif obj.agent == 1
                    tbl = table(data_subject.pe,data_subject.up, round(data_subject.norm_condiff,2), data_subject.contrast,...
                        data_subject.choice_cond,data_subject.congruence,data_subject.reward_unc,data_subject.pe_sign,...
                        'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
                        ,'reward_unc','pe_sign'});
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
                    data_subject = obj.data(obj.data.ID == id_subjs(i),:); % single-subject data
                    weights_subj = wt_subjs(wt_subjs(:,2) == id_subjs(i));
                    if obj.online == 1
                        tbl = table(data_subject.pe,data_subject.up, round(data_subject.norm_condiff,2), data_subject.contrast,...
                            data_subject.choice_cond,data_subject.congruence,data_subject.reward_unc,data_subject.pe_sign,data_subject.salience_choice,...
                            'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
                            ,'reward_unc','pe_sign','salience_choice'});
                    elseif obj.agent == 1
                        tbl = table(data_subject.pe,data_subject.up, round(data_subject.norm_condiff,2), data_subject.contrast,...
                            data_subject.choice_cond,data_subject.congruence,data_subject.reward_unc,data_subject.pe_sign,...
                            'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
                            ,'reward_unc','pe_sign'});
                    end
                    [betas,rsquared,residuals_reg,coeffs_name,lm] = obj.linear_fit(tbl,fit_fn,weights_subj);
                    betas_all(i,:) = betas(2:end);
                    rsquared_full(i,1) = rsquared;
                    [post_up] = predict_fn(lm,tbl);
                    posterior_all{i,1} = post_up;
                end
            end
        end
    end
end