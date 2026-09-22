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

            % VALIDATE MODEL-SELECTION FLAGS ARE MUTUALLY EXCLUSIVE
            model_flags = [obj.lr_mdl, obj.risk_mdl, obj.saliencechoice_mdl, obj.EEanalysis];
            if sum(model_flags) ~= 1
                error('model_definition:ambiguousModel', ...
                    ['Exactly one of lr_mdl, risk_mdl, saliencechoice_mdl, EEanalysis ' ...
                    'must be set to 1 (got %d set).'], sum(model_flags));
            end

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

            % VALIDATE PREDICTOR-SET FLAGS ARE MUTUALLY EXCLUSIVE
            predictor_flags = [obj.agent, obj.online, obj.EEanalysis];
            if sum(predictor_flags) ~= 1
                error('model_definition:ambiguousPredictorSet', ...
                    'Exactly one of agent, online, EEanalysis must be set to 1 (got %d set).', ...
                    sum(predictor_flags));
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
            %   fit_fn: function to be used, adjust if using mock fitlm
            %   for unit testing
            %   varargin{1}: weights (only used when obj.weight_y_n == 1)
            %   varargin{end}: OPTIONAL reference coefficient names (cell
            %   array) -- the canonical name/order to match this fit's
            %   coefficients against (see get_coeffs). When given, betas
            %   are assigned by matching lm.CoefficientNames against
            %   reference_names, filling NaN for any expected term fitlm
            %   dropped for this particular fit (e.g. a categorical
            %   predictor with only one level present in a single
            %   subject's trials -- fitlm silently omits that term rather
            %   than erroring, which would otherwise shift every
            %   subsequent term into the wrong column under positional
            %   assignment). When omitted, falls back to the original
            %   positional assignment, so existing callers that don't
            %   need this (e.g. esterror_analysis.m, LR_unittests.m)
            %   are unaffected.
            %
            % OUTPUT:
            %   betas: array containing beta value for each predictor by fitlm
            %   rsquared: rsquared after fitting mdl to the data
            %   residuals: residuals after fitting mdl to the data
            %   coeffs_name: cell array containing name of all regressors
            %   lm: fitted model

            % FIT THE MODEL USING WEIGHTED/NON-WEIGHTED REGRESSION
            reference_names = {};
            if obj.weight_y_n == 1
                lm = fit_fn(tbl,obj.mdl,'ResponseVar',obj.resp_var,'PredictorVars',obj.pred_vars, ...
                    'CategoricalVars',obj.cat_vars,'Weights',varargin{1});
                if length(varargin) >= 2
                    reference_names = varargin{2};
                end
            else
                lm = fit_fn(tbl,obj.mdl,'ResponseVar',obj.resp_var,'PredictorVars',obj.pred_vars, ...
                    'CategoricalVars',obj.cat_vars);
                if length(varargin) >= 1
                    reference_names = varargin{1};
                end
            end

            % SAVE R-SQUARED, RESIDUALS AND BETA VALUES
            rsquared = lm.Rsquared.Adjusted;
            residuals = lm.Residuals.Raw;
            coeffs_name = lm.CoefficientNames;

            if isempty(reference_names)
                % No reference given -- original positional assignment.
                betas = nan(1,obj.num_vars+1);
                for b = 1:obj.num_vars+1
                    betas(1,b) = lm.Coefficients.Estimate(b);
                end
            else
                % Match each expected term by name; NaN if fitlm dropped
                % it for this particular fit.
                betas = nan(1, length(reference_names));
                for b = 1:length(reference_names)
                    match_idx = find(strcmp(lm.CoefficientNames, reference_names{b}), 1);
                    if ~isempty(match_idx)
                        betas(1,b) = lm.Coefficients.Estimate(match_idx);
                    end
                end
            end
        end

        function tbl = build_subject_table(obj,data_subject)
            % function build_subject_table builds the predictor table for
            % a single subject's data, used to fit the linear regression
            % model.
            %
            % INPUT:
            %   obj: current object
            %   data_subject: table with single-subject data
            %
            % OUTPUT:
            %   tbl: table with predictor vars data, formatted for
            %   linear_fit

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

            % REFERENCE COEFFICIENT NAMES: fit once on the full
            % (all-subjects pooled) dataset, which is virtually
            % guaranteed to have every categorical term's levels present,
            % to get the canonical name/order for this formula. Every
            % per-subject fit below is then matched against these names
            % (see linear_fit) rather than assumed to land at a fixed
            % position -- a subject with sparse (e.g. condition-filtered)
            % data can have fitlm silently drop a term whose categorical
            % predictor has only one level in their trials, which would
            % otherwise shift every subsequent term into the wrong
            % column under positional assignment.
            obj.weight_y_n = 0;
            reference_tbl = obj.build_subject_table(obj.data);
            [~,~,~,reference_names,~] = obj.linear_fit(reference_tbl,fit_fn);

            % FIT THE MODEL TO GET RESIDUALS (non-weighted)
            for i = 1:obj.num_subjs
                obj.weight_y_n = 0; % non-weighted
                data_subject = obj.data(obj.data.ID == id_subjs(i),:); % single-subject data
                tbl = obj.build_subject_table(data_subject);
                [betas,rsquared,residuals_reg,coeffs_name,lm] = obj.linear_fit(tbl,fit_fn,reference_names);
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
                    tbl = obj.build_subject_table(data_subject);
                    [betas,rsquared,residuals_reg,coeffs_name,lm] = obj.linear_fit(tbl,fit_fn,weights_subj,reference_names);
                    betas_all(i,:) = betas(2:end);
                    rsquared_full(i,1) = rsquared;
                    [post_up] = predict_fn(lm,tbl);
                    posterior_all{i,1} = post_up;
                end
            end
        end
    end
end