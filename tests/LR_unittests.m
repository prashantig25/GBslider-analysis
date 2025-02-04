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
            rng(123) % seed
            lm_mock = fitlm_mock(tbl,''); % run mock fitlm
            expected_rsquared = lm_mock.Rsquared.Adjusted; % expected r-squared
            expected_residuals = lm_mock.Residuals.Raw; % expected residuals
            expected_betas = nan(1,num_vars+1); % expected betas
            for b = 1:num_vars+1
                expected_betas(1,b) = lm_mock.Coefficients.Estimate(b);
            end
            expected_coeffs_name = lm_mock.CoefficientNames; % expected coeffs name

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
            [betas_all,rsquared_full,residuals_reg,coeffs_name,posterior_up_subjs] = LR_obj.get_coeffs(@fitlm_mock,@predict_mock);

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

            for i = 1:LR_obj.num_subjs
                LR_obj.weight_y_n = 0;
                data_subject = LR_obj.data(LR_obj.data.ID == expected_id_subjs(i),:);
                tbl = table(data_subject.pe,data_subject.up, round(data_subject.norm_condiff,2), data_subject.contrast,...
                    data_subject.choice_cond,data_subject.congruence,data_subject.reward_unc,data_subject.pe_sign,data_subject.salience_choice,...
                    'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
                    ,'reward_unc','pe_sign','salience_choice'});
                [~,~,expected_residuals_reg,~,~] = LR_obj.linear_fit(tbl,@fitlm_mock);
                expected_res_subjs = [expected_res_subjs; expected_residuals_reg, repelem(expected_id_subjs(i),length(expected_residuals_reg)).'];
            end

            if LR_obj.weighted == 1
                LR_obj.weight_y_n = 1;
                [wt_subjs] = weights_general(LR_obj.data, expected_res_subjs);
                wt_subjs(:,2) = expected_res_subjs(:,2);
                for i = 1:LR_obj.num_subjs
                    weights_subj = wt_subjs(wt_subjs(:,2) == expected_id_subjs(i));
                    data_subject = LR_obj.data(LR_obj.data.ID == expected_id_subjs(i),:);
                    tbl = table(data_subject.pe,data_subject.up, round(data_subject.norm_condiff,2), data_subject.contrast,...
                        data_subject.choice_cond,data_subject.congruence,data_subject.reward_unc,data_subject.pe_sign,data_subject.salience_choice,...
                        'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
                        ,'reward_unc','pe_sign','salience_choice'});
                    [expected_betas,expected_rsquared,expected_residuals_reg,expected_coeffs_name,lm] = LR_obj.linear_fit(tbl,@fitlm_mock,weights_subj);
                    expected_betas_all(i,:) = expected_betas(2:end);
                    expected_rsquared_full(i,1) = expected_rsquared;
                    [expected_post_up] = predict_mock(lm,tbl);
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