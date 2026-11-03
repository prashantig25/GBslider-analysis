function parallelest(exp_data, est_vars)
%     """ This function implements the parallel parameter estimation
%
%     :param exp_data: Data of experimental data that are used for parameter estimation
%     :param est_vars: Estimation variables object instance
%     :param kwargs: optionally, fixed parameters sigma and beta
%     :return: results_df: Data frame with participants IDs, estimated parameters, maximum likelihood and BIC
%     """

%     # If provided, get fixed parameters: sigma and beta
%     fixed_params = kwargs.get('fixed_params', None)

% Initialize estimation object
estimation = GbEstimation();

% Inform user
if est_vars.est_sigma
    disp('Perceptual parameter estimation...')
else
    disp('Economic choice/learning parameter estimation...')
end

% Initialize progress bar
n = length(unique(exp_data.ID)); % Count unique values in the 'ID' column

% estimation.perc_model(exp_data);
estimation.edm_l_model(exp_data, EstVars, 1);
%     # Pool object instance for parallel processing
%     pool = Pool(processes=est_vars.n_ker)
%
%     # Estimate parameters
%     if est_vars.est_sigma:
%         results = [pool.apply_async(estimation.perc_model, args=(exp_data[(exp_data['id'] == i)].copy(),),
%                                     callback=callback) for i in range(0, n)]
%     else:
%         results = [pool.apply_async(estimation.edm_l_model, args=(exp_data[(exp_data['id'] == i)].copy(),
%                                                                   est_vars, fixed_params.loc[i, :]),
%                                     callback=callback) for i in range(0, n)]
%     output = [p.get() for p in results]
%     pool.close()
%     pool.join()

%     # Put estimated parameters in data frame
%     if est_vars.est_sigma:
%         results_df = pd.DataFrame(output, columns=['llh', 'd_BIC', 'id', 'minimum'])
%     else:
%         if est_vars.agent == 1 or est_vars.agent == 2:
%             results_df = pd.DataFrame(output, columns=['llh', 'd_BIC', 'a_BIC', 'id', 'agent',
%                                                        'minimum_beta'])
%         elif est_vars.agent == 3:
%             results_df = pd.DataFrame(output, columns=['llh', 'd_BIC', 'a_BIC', 'id', 'agent',
%                                                        'minimum_lambda', 'minimum_beta'])
%         elif est_vars.agent == 4 or est_vars.agent == 5:
%             results_df = pd.DataFrame(output, columns=['llh', 'd_BIC', 'a_BIC', 'id', 'agent',
%                                                        'minimum_beta', 'minimum_alpha'])
%         else:
%             results_df = pd.DataFrame(output, columns=['llh', 'd_BIC', 'a_BIC', 'id', 'agent',
%                                                        'minimum_lambda', 'minimum_alpha', 'minimum_beta'])

%     # Make sure that we keep the same order of participants
%     results_df = results_df.sort_values(by=['id'])
%
%     pbar.close()
%
%     return results_df

end