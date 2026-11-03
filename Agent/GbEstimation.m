classdef GbEstimation < EstVars
    %     % This class specifies instance variables and methods of the estimation object for parameter estimation
    %
    properties
        % This function defines the instance variables unique to each instance

        % Estimation variables from GbEstVars
        sigma = 0.04;
        agentNum = 1;
    end
    %
    methods
        function perc_model(obj,exp_data)
            %         """ This function estimates the perceptual choice model
            %
            %         :param exp_data: Experimental data
            %         :return results_list: Results of evaluation and/or estimation procedure
            %         """
            %
            %           Control random number generator for reproducible results
            rng(123);
            %
            %           Agent parameters in GbAgentVars.py object
            agent_vars = agentvars();
            %
            % Add trial as index
            %             exp_data.loc[:, 'trial'] = linspace(0, len(exp_data) - 1, len(exp_data));
            %             exp_data = exp_data.set_index('trial');
            % Assuming exp_data is a table in MATLAB

            % Create a new column 'trial' with linearly spaced values
            exp_data.trial = linspace(0, height(exp_data) - 1, height(exp_data))';

            % Set 'trial' as the row names (index) of the table
            exp_data.Properties.RowNames = string(exp_data.trial);

            % Optionally, remove the 'trial' column if you only want it as an index
            exp_data.trial = [];

            %           Initialize values
            min_llh = 1e10;  % unrealistically high likelihood
            min_x = NaN;  % no parameter estimate
            %
            %         Cycle over starting points
            for r = [0, obj.n_sp]

                %             Starting point
                if obj.rand_sp
                    x0 = unifrnd(obj.s_bnds(1), obj.s_bnds(2),1,1);
                else
                    x0 = [obj.s_fixedsp];
                end

                %             Parameter boundary
                bnds = [obj.s_bnds];

                %             Run the estimation
                lowerBound = bnds(1);  % Assuming bnds is a 1x2 vector [lowerBound, upperBound]
                upperBound = bnds(2);

                % Call fminbnd to find the minimum
                res = fminbnd(obj.d_llh(exp_data), lowerBound, upperBound);

                %             Extract optimization output maximum likelihood
                x = res.x; % parameter estimate
                llh = res.fun;  % minimized negative log likelihood

                %             Check if cumulated negative log likelihood is lower than the previous
                %             one and select the lowest
                if llh < min_llh
                    min_llh = llh;
                    min_x = x;
                end
            end

            %         Add ID to results to check if participant order is maintained during parallelization
            obj.id = exp_data.id;

            %         Compute BIC with free parameter
            bic_d = obj.compute_bic(min_llh, 1);

            %         Initialize and fill list with results
            min_x = min_x.tolist();
            results_list = [];
            results_list = [results_list,min_llh];
            results_list = [results_list,bic_d];
            results_list = [results_list,obj.id];
            results_list = [results_list,min_llh];
            results_list = results_list + min_x;
            %
        end

        function llh_d = d_llh(obj,df)
            %         """ This function computes the likelihood of perceptual decisions
            %
            %             Used to estimate the sigma parameter and to evaluate the likelihood with
            %             a fixed sigma parameter.
            %
            %         :param sigma: Perceptual sensitivity parameter
            %         :param df: Data frame that contains current data set
            %         :param agent_vars: Variables to initialize the agent object
            %         :return: llh: Cumulated negative log likelihood
            %         """
            %
            %         Number of trials
            n_trials = height(df);
            %
            %         Agent object initialization
            agent = Agent();
            %
            %         Set sensory noise parameter
            agent.sigma = obj.sigma;
            %
            %         Initialize likelihood array
            llh_d = NaN(n_trials, 1);
            %
            %         Cycle over trials
            for t = [1:n_trials]

                %             Set current observation
                agent.o_t = df.condiff(t);

                %             Evaluate probability of current perceptual decision
                agent.decide_p();

                %             Compute log likelihood of perceptual decision
                llh_d(t) = log(agent.p_d_t(int32(df.d_t(t))));
            end

            %         Sum negative log likelihoods
            llh = -1 * sum(llh_d);
            %
        end

        function edm_l_model(obj, exp_data, est_vars, est_params)
            %         """ This function computes the likelihood of the models that include economic decision making
            %
            %             Used for both, evaluation with fixed parameters and estimation with
            %             free parameters.
            %
            %         :param exp_data: Data of current experiment
            %         :param est_vars: Variables that are used for estimation and evaluation
            %         :param sub_params: Participants' parameter estimates
            %         :param est_params: Indicates if we evaluate model or additionally estimate a free parameter
            %         :param save_plot: Indicates if we save plot of model evaluation
            %         :return: results: Results of evaluation and estimation procedure
            %         """

            %         Control random number generator for reproducible results
            rng(123);

            %         Agent parameters in GbAgentVars.py object
            agent_vars = agentvars();

            %         Select current agent
            agent_vars.agent = est_vars.agent;
            obj.agentNum = est_vars.agent;

            %         eventuell reset_index() anstatt andere Lösung.
            %         --> https://cmdlinetips.com/2018/04/how-to-reset-index-in-pandas-dataframe/
            %         Reset index for perceptual parameter estimation
            %             exp_data.loc[:, 'trial'] = linspace(0, length(exp_data)-1, length(exp_data));
            %             exp_data = exp_data.set_index('trial');

            % Create a new column 'trial' with linearly spaced values
            exp_data.trial = linspace(0, height(exp_data) - 1, height(exp_data))';

            % Set 'trial' as the row names (similar to setting an index)
            exp_data.Properties.RowNames = string(exp_data.trial);
            exp_data.trial = []; % Optionally remove the 'trial' column if you only want it as an index

            %         Evaluate likelihood of perceptual decisions
%             f_llh_d = obj.d_llh();
            %             (sub_params['sigma'], exp_data, agent_vars);

            %         Reset index for additional parameter estimation
            %             exp_data.loc[:, 'trial'] = np.tile(np.linspace(0, obj.T-1, obj.T), obj.B);
            %             exp_data = exp_data.set_index('trial');
            exp_data.trial = linspace(0, height(exp_data) - 1, height(exp_data))';

            % Set 'trial' as the row names (similar to setting an index)
            exp_data.Properties.RowNames = string(exp_data.trial);
            exp_data.trial = []; % Optionally remove the 'trial' column if you only want it as an index

            %         Add ID to results to check if participant order is maintained during parallelization
            obj.id = unique(exp_data.ID);
            %         Estimate free parameters
            if est_params == 1

                %             Initialize values
                min_llh = 1e10; % unrealistically high likelihood
                min_x = NaN; % no parameter estimate

                %             Cycle over starting points
                for r = [1, obj.n_sp]

                    %                 Estimate parameters
                    if obj.agentNum == 1 || obj.agentNum == 2
                        if obj.rand_sp
                            x0 = unifrnd(obj.b_bnds(1), obj.b_bnds(2))
                        else
                            x0 = obj.b_fixedsp
                        end

                        %                     Parameter boundary
                        boundary = [obj.b_bnds]

                    elseif obj.agentNum == 3

                        %                     Starting point
                        if obj.rand_sp
                            x0 = unifrnd(obj.l_bnds(1), obj.l_bnds(2)),
                            unifrnd(obj.b_bnds(1), obj.b_bnds(2))
                        else
                            x0 = [obj.l_fixedsp, obj.b_fixedsp]
                        end

                        %                     Parameter boundary
                        boundary = [obj.l_bnds, obj.b_bnds]

                    elseif obj.agentNum == 4 || obj.agentNum == 5
                        if obj.rand_sp
                            x0 = unifrnd(obj.b_bnds(1), obj.b_bnds(2)),
                            unifrnd(obj.a_bnds(1), obj.a_bnds(2))
                        else
                            x0 = [obj.b_fixedsp, obj.a_fixedsp]
                        end

                        %                     Parameter boundary
                        boundary = [obj.b_bnds, obj.a_bnds]
                    else
                        if obj.rand_sp
                            x0 = [unifrnd(obj.l_bnds(1), obj.l_bnds(2)),...
                                unifrnd(obj.a_bnds(1), obj.a_bnds(2)),...
                                unifrnd(obj.b_bnds(1), obj.b_bnds(2))]
                        else
                            x0 = [obj.l_fixedsp, obj.a_fixedsp, obj.b_fixedsp]
                        end
                    end
                end

                %                     Parameter boundary
                boundary = [obj.l_bnds, obj.a_bnds, obj.b_bnds];

                %                 Run the estimation
                %                     res = fminbnd(obj.a_llh, x0, args=(exp_data, sub_params, agent_vars, est_params), method='L-BFGS-B',
                %                     bounds=bounds)
                res = fminbnd(@(x) obj.a_llh(x, exp_data, est_params), boundary(1), boundary(end));


                %                 Extract maximum likelihood parameter estimate
                x = res.x;

                %                 Extract minimized negative log likelihood
                llh = res.fun;

                %                 Check if cumulated negative log likelihood is lower than the previous
                %                 one and select the lowest
                if llh < min_llh
                    min_llh = llh;
                    min_x = x;
                end

                %             Compute BIC for economic decision
                if obj.agentNum == 1 || obj.agentNum == 2
                    bic_a = obj.compute_bic(min_llh, 1);
                elseif obj.agentNum == 3 || obj.agentNum == 4 || obj.agentNum == 5
                    bic_a = obj.compute_bic(min_llh, 2);
                else
                    bic_a = obj.compute_bic(min_llh, 3);
                end

            else

                %             Evaluate the model
                if obj.agentNum == 1
                    min_llh = obj.a_llh(obj, x, exp_data, sub_params.beta_A1, est_params);
                    %(sub_params['beta_A1'], exp_data, sub_params, agent_vars, est_params,
                    %save_plot=save_plot)
                elseif obj.agentNum == 2
                    %                         min_llh = obj.a_llh(sub_params['beta_A2'], exp_data, sub_params, agent_vars, est_params,
                    %                         save_plot=save_plot)
                    min_llh = obj.a_llh(obj, x, exp_data, sub_params.beta_A2, est_params);
                elseif obj.agentNum == 3
                    %                         min_llh = obj.a_llh([sub_params['lambda_A3'], sub_params['beta_A3']], exp_data, sub_params,
                    %                         agent_vars, est_params, save_plot=save_plot)
                    min_llh = obj.a_llh(obj, x, exp_data, [sub_params.beta_A3, sub_params.lambda_A3], est_params);
                elseif obj.agentNum == 4
                    %                         min_llh = obj.a_llh([sub_params['beta_A4'], sub_params['alpha_A4']], exp_data, sub_params,
                    %                         agent_vars, est_params, save_plot=save_plot)
                    min_llh = obj.a_llh(obj, x, exp_data, [sub_params.beta_A4, sub_params.alpha_A4], est_params);
                elseif obj.agentNum == 5
                    %                         min_llh = obj.a_llh([sub_params['beta_A5'], sub_params['alpha_A5']], exp_data, sub_params,
                    %                         agent_vars, est_params, save_plot=save_plot)
                    min_llh = obj.a_llh(obj, x, exp_data, [sub_params.beta_A5, sub_params.alpha_A5], est_params);
                elseif obj.agentNum == 6
                    %                         min_llh = obj.a_llh([sub_params['lambda_A6'], sub_params['alpha_A6'], sub_params['beta_A6']],
                    %                         exp_data, sub_params, agent_vars, est_params, save_plot=save_plot)
                    min_llh = obj.a_llh(obj, x, exp_data, [sub_params.lambda_A6, sub_params.alpha_A6], est_params);
                end
                %                     else
                %                         %                             A0
                %                         min_llh = obj.a_llh(np.nan, exp_data, sub_params, agent_vars, est_params, save_plot=save_plot)
                %                     end
            end


            %             Compute BIC
            bic_a = obj.compute_bic(min_llh, 0);

            %         Compute BIC for perceptual decision without free parameters
            bic_d = obj.compute_bic(f_llh_d, 0);

            %         Initialize and fill list with results
            results_list = list();
            results_list.append(np.float(min_llh))
            results_list.append(np.float(bic_d))
            results_list.append(np.float(bic_a))
            results_list.append(np.float(obj.id))
            results_list.append(np.float(obj.agentNum))


            if est_params == 1
                min_x = min_x.tolist();
                results_list = results_list + min_x;
            end
        end

        %%%%%%%%%%% ???
        function llh_a = a_llh(obj, x, exp_data, sub_params, est_params)
            %         """ This function computes the likelihood of economic decisions
            %
            %             Used to estimate the beta parameter of the softmax choice rule and
            %             the lambda parameter. The function is also used to evaluate
            %             the likelihood of the economic decisions based on fixed parameters
            %
            %         :param x: Free parameter
            %         :param exp_data: Data frame that contains current data set
            %         :param sub_params:
            %         :param agent_vars: Variables to initialize the agent object
            %         :param est_params:
            %         :param save_plot:
            %         :return: llh: Cumulated negative log likelihood
            %         """

            %         Initialize likelihood array
            %         Create a matrix filled with NaN values
            llh_a = nan(obj.T, obj.B);

            %         Initialize figure, if requested
            if est_params == 0
                %         Cycle over blocks
                for b = 1:obj.B

                    % Filter the table for the specified block number
                    filteredData = exp_data(exp_data.blockNumber == b, :);

                    % Extract the relevant columns
                    u = filteredData.u_t;
                    d = filteredData.d_t;
                    a = filteredData.a_t - 1;  % Rescale action to {0, 1}
                    r = filteredData.r_t;
                    corr = filteredData.decision2_corr;  % Adjust for MATLAB's naming convention

                    %             Agent object initialization
                    agent = Agent();

                    % Assign the sigma values to agent.sigma
                    agent.sigma = sub_params.sigma;

                    %             Assign current beta and (if required) lambda parameter
                    if obj.agentNum == 0 || obj.agentNum == 1 || obj.agentNum == 2
                        agent.beta = x;
                    elseif obj.agentNum == 3
                        agent.lambda_param = x(1);
                        agent.beta = x(2);
                    elseif obj.agentNum == 4 || obj.agentNum == 5
                        agent.beta = x(1);
                        agent.alpha = x(2);
                    else
                        % Agent 6
                        agent.lambda_param = x(1);
                        agent.alpha = x(2);
                        agent.beta = x(3);

                        %             Initialize block specific variables
                        cp = NaN(obj.T,1); % choice probability
                        ev = NaN(obj.T,1);  % expected value
                        cond_ev = NaN(obj.T,2);  % conditional expected value

                        %             Cycle over trials
                        for t = [1:obj.T]

                            %                 Set current observation
                            agent.o_t = u(t);

                            %                 Set perceptual decision
                            agent.d_t = d(t);

                            %                 Evaluate probability of economic decisions
                            agent.decide_e();

                            %                 Set economic decision
                            agent.a_t = a(t);

                            %                 Extract probability of current economic decision
                            cp(t) = agent.p_a_t(a(t));

                            %                 Compute log likelihood of economic decision
                            llh_a(t, b) = log(agent.p_a_t(int32(a(t))));

                            %                 Agent contingency parameter update
                            agent.learn(int32(r(t)));

                            %                 Extract expected value
                            ev(t) = agent.E_mu_t;           % Assign E_mu_t to ev at index t
                            cond_ev(t, :) = agent.v_a_t;    % Assign v_a_t to the t-th row of cond_ev
                            rel_r_t(t) = agent.r_t;         % Assign r_t to rel_r_t at index t


                            %         Sum negative log likelihoods
                            llh = -1 * nansum(llh_a(:));  % Sum all elements in llh_a and multiply by -1
                        end
                    end
                end
            end
        end
        %         if est_params == 0
        %             return cp, ev, cond_ev
        %
        %         else:
        %             return llh

        function bic = compute_bic(obj, llh, n_params)
            %         """ This function compute the Bayesian information criterion (BIC)
            %
            %             See Stephan et al. (2009). Bayesian model selection for group studies. NeuroImage
            %
            %         :param llh: Negative log likelihood
            %         :param n_params: Number of free parameters
            %         :return: BIC
            %         """

            %         Eq. 42
            bic = (-1 * llh) - (n_params / 2) * log(obj.T * obj.B);
        end
    end
end
