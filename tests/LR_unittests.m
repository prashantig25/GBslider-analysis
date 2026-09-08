classdef LR_unittests < matlab.unittest.TestCase
    % LR_UNITTESTS is a collection of functions to run unit tests on
    % various functions used to fit a linear regression model.
    
    methods(Test)

        function test_linearfit(obj)
            % test_linearfit runs a unit test on linear_fit within object
            % lr_analysis_obj().

            % INITIALIZE VARS
            LR_obj = lr_analysis_obj(); % object
            LR_obj.filename = "Data/LR analyses/preprocessed_data.mat"; % specify path to get the dataset
            LR_obj.lr_mdl = 1; % run best behavioral model
            LR_obj.risk_mdl = 0; % run model including risk regressor
            LR_obj.saliencechoice_mdl = 0; % run model including salience choice regressor
            LR_obj.num_subjs = 98; % number of subjects
            LR_obj.absolute_analysis = 1; % pre-process data for absolute LR analysis
            LR_obj.grouped = 0; % set to 1 if regression model needs to be fit separately for different groups of trials
            LR_obj.num_groups = 2; % number of groups for grouped regression
            LR_obj.agent = 0; % fit model to agent simulations
            LR_obj.online = 1; % fit model to online dataset
            LR_obj.weighted = 1;
            LR_obj.initialiseVars();
            LR_obj.model_definition();
            tbl = table; % empty table for regressors
            tbl.up = rand(100,1); % random update
            tbl.pe = rand(100,1); % pe
            tbl.contrast_diff = rand(100,1); % contrast difference
            tbl.congruence = randi([0, 1],100,1); % congruence
            tbl.salience = randi([0, 1],100,1); % contrast
            tbl.pe_sign = randi([0, 1],100,1); % pe sign
            num_vars = 5; % number of variables
            LR_obj.weight_y_n = 0; % non-weighted
            [betas,rsquared,residuals,coeffs_name,~] = LR_obj.linear_fit(tbl, ...
                @fitlm_mock); % fit the model

            % EXPECTED
            % CHANGED: fitlm_mock now returns fixed values (tests/fitlm_mock.m),
            % so they're asserted directly here instead of calling fitlm_mock
            % a second time to "discover" the same values it would return.
            expected_rsquared = 0.5; % expected r-squared
            expected_residuals = (1:height(tbl)).'; % expected residuals
            expected_betas = 1:num_vars+1; % expected betas
            expected_coeffs_name = {'Intercept','pe','pe:contrast_diff','pe:congruence','pe:salience','pe:pe_sign_1'}; % expected coeffs name

            % RUN TESTS
            obj.verifyEqual(betas,expected_betas,'LM generated betas do not match.')
            obj.verifyEqual(rsquared,expected_rsquared,'LM generated r-squared do not match.')
            obj.verifyEqual(residuals,expected_residuals,'LM generated residuals do not match.')
            obj.verifyEqual(coeffs_name,expected_coeffs_name,'LM generated coefficient names do not match.')

        end

        function test_get_coeffs(obj)
            % test_get_coeffs runs a unit test on get_coeffs within object
            % lr_analysis_obj().

            % INITIALIZE VARS
            LR_obj = lr_analysis_obj(); % object
            LR_obj.filename = "Data/LR analyses/preprocessed_data.mat"; % specify path to get the dataset
            LR_obj.lr_mdl = 1; % run best behavioral model
            LR_obj.risk_mdl = 0; % run model including risk regressor
            LR_obj.saliencechoice_mdl = 0; % run model including salience choice regressor
            LR_obj.num_subjs = 98; % number of subjects
            LR_obj.absolute_analysis = 1; % pre-process data for absolute LR analysis
            LR_obj.grouped = 0; % set to 1 if regression model needs to be fit separately for different groups of trials
            LR_obj.num_groups = 2; % number of groups for grouped regression
            LR_obj.agent = 0; % fit model to agent simulations
            LR_obj.online = 1; % fit model to online dataset
            LR_obj.weighted = 1;
            LR_obj.initialiseVars();
            LR_obj.model_definition();
            LR_obj.absolute_analysis = 0; % whether the test should be run on absolute or relative analysis
            num_trials = 1000; % number of trials for randomly generated regressors data
            LR_obj.num_subjs = 2; % number of subjects for the test
            LR_obj.data = table;
            LR_obj.data.up = rand(num_trials*LR_obj.num_subjs,1);
            LR_obj.data.pe = rand(num_trials*LR_obj.num_subjs,1);
            LR_obj.data.norm_condiff = rand(num_trials*LR_obj.num_subjs,1);
            LR_obj.data.congruence = randi([0, 1],num_trials*LR_obj.num_subjs,1);
            LR_obj.data.contrast = randi([0, 1],num_trials*LR_obj.num_subjs,1);
            LR_obj.data.pe_sign = randi([0, 1],num_trials*LR_obj.num_subjs,1);
            LR_obj.data.choice_cond = randi([0, 1],num_trials*LR_obj.num_subjs,1);
            LR_obj.data.reward_unc = randi([0, 1],num_trials*LR_obj.num_subjs,1);
            LR_obj.data.ID = [repelem(1,num_trials,1);repelem(2,num_trials,1)];
            LR_obj.data.salience_choice = randi([0, 1],num_trials*LR_obj.num_subjs,1);
            % Replaced @predict_mock (tests/predict_mock.m, now deleted) with
            % this inline mock: it only ever needs to return a deterministic
            % column vector of the right height, so an anonymous function
            % avoids a whole file plus the RNG seeding predict_mock used.
            [betas_all,rsquared_full,residuals_reg,coeffs_name,posterior_up_subjs] = LR_obj.get_coeffs(@fitlm_mock,@(lm,tbl) (1:height(tbl)).');

            % EXPECTED
            expected_id_subjs = unique(LR_obj.data.ID);
            expected_betas_all = NaN(length(LR_obj.num_subjs),LR_obj.num_vars);
            expected_rsquared_full = NaN(length(LR_obj.num_subjs),1);
            expected_posterior_up_subjs =  cell(length(LR_obj.num_subjs),1);
            expected_res_subjs = [];
            if LR_obj.absolute_analysis == 1
                LR_obj.data.pe = abs(LR_obj.data.pe);
                LR_obj.data.up = abs(LR_obj.data.up);
            end

            % CHANGED: fitlm_mock now returns fixed values regardless of tbl
            % content (tests/fitlm_mock.m), so its Residuals.Raw for a given
            % subject is just (1:height(data_subject)).' -- computed directly
            % below instead of building tbl and calling linear_fit/fitlm_mock
            % again to "discover" the same value.
            for i = 1:LR_obj.num_subjs
                LR_obj.weight_y_n = 0;
                data_subject = LR_obj.data(LR_obj.data.ID == expected_id_subjs(i),:);
                expected_residuals_reg = (1:height(data_subject)).';
                expected_res_subjs = [expected_res_subjs; expected_residuals_reg, repelem(expected_id_subjs(i),length(expected_residuals_reg)).'];
            end

            if LR_obj.weighted == 1
                % CHANGED: weights_general is still exercised by the actual
                % get_coeffs call above; it isn't recomputed here since
                % fitlm_mock ignores weights entirely, so no weights_subj
                % value could change the (now-fixed) expected betas/rsquared/
                % coefficient names below.
                expected_coeffs_name = {'Intercept','pe','pe:contrast_diff','pe:congruence','pe:salience','pe:pe_sign_1'}; % fixed, from fitlm_mock
                for i = 1:LR_obj.num_subjs
                    data_subject = LR_obj.data(LR_obj.data.ID == expected_id_subjs(i),:);
                    % CHANGED: fitlm_mock's Coefficients.Estimate is fixed at
                    % (1:6).' and Rsquared.Adjusted at 0.5; betas_all keeps
                    % Estimate(2:end) (drops the intercept term).
                    expected_betas_all(i,:) = 2:6;
                    expected_rsquared_full(i,1) = 0.5;
                    expected_post_up = (1:height(data_subject)).'; % matches the mock predict_fn passed to get_coeffs
                    expected_posterior_up_subjs{i,1} = expected_post_up;
                end
            end

            % RUN TESTS
            obj.verifyEqual(betas_all,expected_betas_all,'LM generated betas do not match.')
            obj.verifyEqual(rsquared_full,expected_rsquared_full,'LM generated r-squared do not match.')
            obj.verifyEqual(residuals_reg,expected_residuals_reg,'LM generated residuals do not match.')
            obj.verifyEqual(coeffs_name,expected_coeffs_name,'LM generated coefficient names do not match.')
            obj.verifyEqual(posterior_up_subjs,expected_posterior_up_subjs,'Posterior updates do not match.')
        end

    end
end