classdef ParameterRecovery < handle
    % ParameterRecovery - A class for conducting parameter recovery analysis
    % across multiple reinforcement learning models
    
    properties
        n_parameters = 100;
        n_startingPoints = 30;
        n_trials = 500;
        options;
        results;
    end
    
    methods
        function obj = ParameterRecovery(varargin)
            % Constructor with optional parameter specification
            if nargin > 0
                obj.n_parameters = varargin{1};
            end
            if nargin > 1
                obj.n_startingPoints = varargin{2};
            end
            if nargin > 2
                obj.n_trials = varargin{3};
            end
            
            % Set optimization options
            obj.options = optimoptions('fmincon', ...
                'Display', 'off', ...
                'Algorithm', 'active-set', ...
                'FiniteDifferenceType', 'central', ...
                'MaxIterations', 500, ...
                'FunctionTolerance', 1e-4);
        end
        
        function results = run_basicRL_recovery(obj)
            % Parameter recovery for Basic RL model
            fprintf('Running Basic RL parameter recovery...\n');
            
            % Define true parameters
            true_alpha = unifrnd(0, 0.3, [obj.n_parameters, 1]);
            true_kappa = unifrnd(1, 20, [obj.n_parameters, 1]);
            
            % Initialize arrays
            [best_recovered_alphas, best_recovered_kappas, best_nlls] = ...
                obj.initialize_arrays(2);
            [recovered_alphas, recovered_kappas, nllAll] = ...
                obj.initialize_recovery_arrays(2);
            
            % Parameter bounds
            lb = [0, 1];
            ub = [0.3, 100];
            
            for p = 1:obj.n_parameters
                % Generate simulated data
                rewards = rand(1, obj.n_trials) < 0.7;
                state = rand(1, obj.n_trials) < 0.5;
                blocks_all = obj.generate_blocks();
                
                % Simulate data using predict_allModels
                [~, mu_hat] = predict_allModels.predict_basicRL(...
                    [true_alpha(p), true_kappa(p)], blocks_all, rewards, state);
                
                for sp = 1:obj.n_startingPoints
                    % Initial parameter guesses
                    init_params = [unifrnd(0, 0.3), unifrnd(0, 20)];
                    
                    % Optimize
                    [recovered_params, nll] = fmincon(...
                        @(params) fitSlider_ALLmodels.nll_basicRL(params, mu_hat, blocks_all, state, rewards), ...
                        init_params, [], [], [], [], lb, ub, [], obj.options);
                    
                    % Store results
                    recovered_alphas(sp, p) = recovered_params(1);
                    recovered_kappas(sp, p) = recovered_params(2);
                    nllAll(sp, p) = nll;
                    
                    % Update best if better
                    if nll < best_nlls(p)
                        best_nlls(p) = nll;
                        best_recovered_alphas(p) = recovered_params(1);
                        best_recovered_kappas(p) = recovered_params(2);
                    end
                end
                
                if mod(p, 10) == 0
                    fprintf('Completed %d/%d parameter sets\n', p, obj.n_parameters);
                end
            end
            
            % Store results
            results.model_name = 'Basic RL';
            results.true_params = [true_alpha, true_kappa];
            results.best_recovered = [best_recovered_alphas, best_recovered_kappas];
            results.all_recovered = {recovered_alphas, recovered_kappas};
            results.param_names = {'Alpha', 'Kappa'};
            results.nll = best_nlls;
            
            obj.results.basicRL = results;
        end
        
        function results = run_RLsigma_recovery(obj)
            % Parameter recovery for RL + Sigma model
            fprintf('Running RL+Sigma parameter recovery...\n');
            
            % Define true parameters
            true_alpha = unifrnd(0, 0.3, [obj.n_parameters, 1]);
            true_kappa = unifrnd(1, 20, [obj.n_parameters, 1]);
            true_sigma = unifrnd(0, 0.1, [obj.n_parameters, 1]);
            
            % Initialize arrays
            [best_recovered_alphas, best_recovered_kappas, best_recovered_sigmas, best_nlls] = ...
                obj.initialize_arrays(3);
            [recovered_alphas, recovered_kappas, recovered_sigmas, nllAll] = ...
                obj.initialize_recovery_arrays(3);
            
            % Parameter bounds
            lb = [0, 1, 0];
            ub = [0.3, 100, 0.1];
            
            for p = 1:obj.n_parameters
                % Generate simulated data
                rewards = [double(rand(1, obj.n_trials/2) < 0.7), ...
                          double(rand(1, obj.n_trials/2) < 0.9)];
                state = double(rand(1, obj.n_trials) < 0.5);
                condiff = obj.generate_condiff(state);
                blocks_all = obj.generate_blocks();
                
                % Simulate data
                [~, mu_hat] = predict_allModels.predict_RLsigma(...
                    [true_alpha(p), true_kappa(p), true_sigma(p)], ...
                    blocks_all, rewards, state, condiff);
                
                for sp = 1:obj.n_startingPoints
                    % Initial parameter guesses
                    init_params = [unifrnd(0, 0.3), unifrnd(0, 20), unifrnd(0, 0.1)];
                    
                    % Optimize
                    [recovered_params, nll] = fmincon(...
                        @(params) fitSlider_ALLmodels.nll_RLsigma(params, mu_hat, blocks_all, rewards, condiff), ...
                        init_params, [], [], [], [], lb, ub, [], obj.options);
                    
                    % Store results
                    recovered_alphas(sp, p) = recovered_params(1);
                    recovered_kappas(sp, p) = recovered_params(2);
                    recovered_sigmas(sp, p) = recovered_params(3);
                    nllAll(sp, p) = nll;
                    
                    % Update best if better
                    if nll < best_nlls(p)
                        best_nlls(p) = nll;
                        best_recovered_alphas(p) = recovered_params(1);
                        best_recovered_kappas(p) = recovered_params(2);
                        best_recovered_sigmas(p) = recovered_params(3);
                    end
                end
                
                if mod(p, 10) == 0
                    fprintf('Completed %d/%d parameter sets\n', p, obj.n_parameters);
                end
            end
            
            % Store results
            results.model_name = 'RL + Sigma';
            results.true_params = [true_alpha, true_kappa, true_sigma];
            results.best_recovered = [best_recovered_alphas, best_recovered_kappas, best_recovered_sigmas];
            results.all_recovered = {recovered_alphas, recovered_kappas, recovered_sigmas};
            results.param_names = {'Alpha', 'Kappa', 'Sigma'};
            results.nll = best_nlls;
            
            obj.results.RLsigma = results;
        end
        
        function results = run_PWRL_recovery(obj)
            % Parameter recovery for Posterior-Weighted RL model
            fprintf('Running PWRL parameter recovery...\n');
            
            % Define true parameters
            true_alpha = unifrnd(0, 0.3, [obj.n_parameters, 1]);
            true_kappa = unifrnd(1, 50, [obj.n_parameters, 1]);
            true_sigma = unifrnd(0, 0.1, [obj.n_parameters, 1]);
            
            % Initialize arrays
            [best_recovered_alphas, best_recovered_kappas, best_recovered_sigmas, best_nlls] = ...
                obj.initialize_arrays(3);
            [recovered_alphas, recovered_kappas, recovered_sigmas, nllAll] = ...
                obj.initialize_recovery_arrays(3);
            
            % Parameter bounds
            lb = [0, 1, 0];
            ub = [1, 100, 0.1];
            
            for p = 1:obj.n_parameters
                % Generate simulated data
                state = double(rand(1, obj.n_trials) < 0.5);
                condiff = obj.generate_condiff(state);
                blocks_all = obj.generate_blocks();
                condition = [repelem(1, obj.n_trials/2), repelem(2, obj.n_trials/2)];
                
                % Generate synthetic choices and rewards for PWRL
                choices = double(rand(1, obj.n_trials) < 0.5);
                rewards = double(rand(1, obj.n_trials) < 0.7);
                recoded_rewards = rewards;
                contrast = double(rand(1, obj.n_trials) < 0.5);
                
                % Simulate data
                [~, mu_hat] = predict_allModels.predict_PWRL(...
                    [true_alpha(p), true_kappa(p), true_sigma(p)], ...
                    blocks_all, rewards, condiff, choices, recoded_rewards, state, 'mean', contrast);
                
                for sp = 1:obj.n_startingPoints
                    % Initial parameter guesses
                    init_params = [unifrnd(0, 1), unifrnd(0, 30), unifrnd(0, 0.1)];
                    
                    % Optimize
                    [recovered_params, nll] = fmincon(...
                        @(params) fitSlider_ALLmodels.nll_PWRL(params, mu_hat, blocks_all, rewards, condiff, choices, recoded_rewards, contrast), ...
                        init_params, [], [], [], [], lb, ub, [], obj.options);
                    
                    % Store results
                    recovered_alphas(sp, p) = recovered_params(1);
                    recovered_kappas(sp, p) = recovered_params(2);
                    recovered_sigmas(sp, p) = recovered_params(3);
                    nllAll(sp, p) = nll;
                    
                    % Update best if better
                    if nll < best_nlls(p)
                        best_nlls(p) = nll;
                        best_recovered_alphas(p) = recovered_params(1);
                        best_recovered_kappas(p) = recovered_params(2);
                        best_recovered_sigmas(p) = recovered_params(3);
                    end
                end
                
                if mod(p, 10) == 0
                    fprintf('Completed %d/%d parameter sets\n', p, obj.n_parameters);
                end
            end
            
            % Store results
            results.model_name = 'PWRL';
            results.true_params = [true_alpha, true_kappa, true_sigma];
            results.best_recovered = [best_recovered_alphas, best_recovered_kappas, best_recovered_sigmas];
            results.all_recovered = {recovered_alphas, recovered_kappas, recovered_sigmas};
            results.param_names = {'Alpha', 'Kappa', 'Sigma'};
            results.nll = best_nlls;
            
            obj.results.PWRL = results;
        end
        
        function results = run_bayesianAgent_recovery(obj)
            % Parameter recovery for Bayesian Agent model
            fprintf('Running Bayesian Agent parameter recovery...\n');
            
            % Define true parameters
            true_kappa = unifrnd(1, 50, [obj.n_parameters, 1]);
            true_sigma = unifrnd(0, 0.1, [obj.n_parameters, 1]);
            
            % Initialize arrays
            [best_recovered_kappas, best_recovered_sigmas, best_nlls] = ...
                obj.initialize_arrays(2);
            [recovered_kappas, recovered_sigmas, nllAll] = ...
                obj.initialize_recovery_arrays(2);
            
            % Parameter bounds
            lb = [1, 0];
            ub = [100, 0.1];
            
            for p = 1:obj.n_parameters
                % Generate simulated data
                state = double(rand(1, obj.n_trials) < 0.5);
                condiff = obj.generate_condiff(state);
                blocks_all = obj.generate_blocks();
                condition = [repelem(1, obj.n_trials/50), repelem(2, obj.n_trials/50)];
                
                % Generate synthetic data for Bayesian agent
                choices = double(rand(1, obj.n_trials) < 0.5);
                rewards = double(rand(1, obj.n_trials) < 0.7);
                
                % Create data table
                data = table(blocks_all', choices', condiff, rewards', ...
                    'VariableNames', {'blocks', 'choice', 'condiff_relative', 'rewards'});
                
                % Simulate data
                [~, mu_hat] = predict_allModels.predict_bayesianAgent(...
                    [true_kappa(p), true_sigma(p)], blocks_all, rewards, condiff, choices, condition,'mean');
                
                for sp = 1:obj.n_startingPoints
                    % Initial parameter guesses
                    init_params = [unifrnd(0, 30), unifrnd(0, 0.1)];
                    
                    % Optimize
                    [recovered_params, nll] = fmincon(...
                        @(params) fitSlider_ALLmodels.nll_bayesianAgent(params, mu_hat, data, length(unique(blocks_all)), obj.n_trials, blocks_all, rewards), ...
                        init_params, [], [], [], [], lb, ub, [], obj.options);
                    
                    % Store results
                    recovered_kappas(sp, p) = recovered_params(1);
                    recovered_sigmas(sp, p) = recovered_params(2);
                    nllAll(sp, p) = nll;
                    
                    % Update best if better
                    if nll < best_nlls(p)
                        best_nlls(p) = nll;
                        best_recovered_kappas(p) = recovered_params(1);
                        best_recovered_sigmas(p) = recovered_params(2);
                    end
                end
                
                if mod(p, 10) == 0
                    fprintf('Completed %d/%d parameter sets\n', p, obj.n_parameters);
                end
            end
            
            % Store results
            results.model_name = 'Bayesian Agent';
            results.true_params = [true_kappa, true_sigma];
            results.best_recovered = [best_recovered_kappas, best_recovered_sigmas];
            results.all_recovered = {recovered_kappas, recovered_sigmas};
            results.param_names = {'Kappa', 'Sigma'};
            results.nll = best_nlls;
            
            obj.results.bayesianAgent = results;
        end
        
        function plot_recovery_results(obj, model_name)
            % Plot parameter recovery results for a specific model
            if ~isfield(obj.results, model_name)
                error('Model %s not found in results', model_name);
            end
            
            result = obj.results.(model_name);
            n_params = length(result.param_names);
            
            figure('Position', [100, 100, 300*n_params, 250]);
            
            for i = 1:n_params
                subplot(1, n_params, i);
                
                % Create scatter plot
                scatter(result.true_params(:, i), result.best_recovered(:, i), ...
                    50, 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b', ...
                    'MarkerFaceAlpha', 0.6);
                
                hold on;
                
                % Add identity line
                min_val = min([result.true_params(:, i); result.best_recovered(:, i)]);
                max_val = max([result.true_params(:, i); result.best_recovered(:, i)]);
                plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);
                
                % Add regression line
                lsline;
                
                % Calculate correlation
                r = corr(result.true_params(:, i), result.best_recovered(:, i), 'rows', 'complete');
                
                xlabel(sprintf('True %s', result.param_names{i}));
                ylabel(sprintf('Recovered %s', result.param_names{i}));
                title(sprintf('%s: r = %.3f', result.param_names{i}, r));
                grid on;
                axis equal;
                xlim([min_val, max_val]);
                ylim([min_val, max_val]);
            end
            
            sgtitle(sprintf('%s Parameter Recovery', result.model_name));
        end
        
        function plot_all_models(obj)
            % Plot parameter recovery for all models
            model_names = fieldnames(obj.results);
            
            for i = 1:length(model_names)
                obj.plot_recovery_results(model_names{i});
            end
        end
        
        function summary_stats = compute_summary_statistics(obj)
            % Compute summary statistics for all models
            model_names = fieldnames(obj.results);
            summary_stats = struct();
            
            for i = 1:length(model_names)
                model = model_names{i};
                result = obj.results.(model);
                
                for j = 1:length(result.param_names)
                    param_name = result.param_names{j};
                    true_vals = result.true_params(:, j);
                    recovered_vals = result.best_recovered(:, j);
                    
                    % Remove NaN values
                    valid_idx = ~isnan(true_vals) & ~isnan(recovered_vals);
                    true_vals = true_vals(valid_idx);
                    recovered_vals = recovered_vals(valid_idx);
                    
                    % Compute statistics
                    correlation = corr(true_vals, recovered_vals);
                    rmse = sqrt(mean((true_vals - recovered_vals).^2));
                    bias = mean(recovered_vals - true_vals);
                    
                    summary_stats.(model).(param_name) = struct(...
                        'correlation', correlation, ...
                        'rmse', rmse, ...
                        'bias', bias, ...
                        'n_valid', length(true_vals));
                end
            end
        end
    end
    
    methods (Access = private)
        function blocks_all = generate_blocks(obj)
            % Generate block structure
            blocks_all = [];
            for nb = 1:(obj.n_trials/25)
                blocks_all = [repelem(nb, 25), blocks_all];
            end
        end
        
        function condiff = generate_condiff(obj, state)
            % Generate contrast differences based on state
            condiff = state';
            for s = 1:length(state)
                if state(s) == 0
                    condiff(s) = unifrnd(-0.08, 0, 1);
                else
                    condiff(s) = unifrnd(0, 0.08, 1);
                end
            end
        end
        
        function varargout = initialize_arrays(obj, n_params)
            % Initialize arrays for storing best recovered parameters
            varargout = cell(1, n_params + 1);
            for i = 1:n_params
                varargout{i} = NaN(obj.n_parameters, 1);
            end
            varargout{n_params + 1} = Inf(obj.n_parameters, 1); % NLL array
        end
        
        function varargout = initialize_recovery_arrays(obj, n_params)
            % Initialize arrays for storing all recovered parameters
            varargout = cell(1, n_params + 1);
            for i = 1:n_params
                varargout{i} = NaN(obj.n_startingPoints, obj.n_parameters);
            end
            varargout{n_params + 1} = NaN(obj.n_startingPoints, obj.n_parameters); % NLL array
        end
    end
end
