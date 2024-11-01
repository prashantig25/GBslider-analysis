function [betas_all,rsquared_full,residuals_reg,coeffs_name,h_vals,p_vals,t_vals,posterior_up_subjs] = get_coeffs(data,mdl,pred_vars, ...
    resp_var,cat_vars,num_vars,res_subjs,num_subjs,weighted)

    % GET_COEFFS fits the linear regression model by running non-weighted
    % and weighted regressions to get the beta coefficients along with
    % significance testing for the betas.
    % INPUT:
        % data = data table
        % mdl = formula of the model to be fit to the data
        % pred_vars = cell array containing names of all regressors
        % resp_var = cell array with response names
        % cat_vars = cell array with names of regressors that are
        % categorical
        % num_vars = number of coefficients expected from the regression
        % model
        % res_subjs = empty array to store the residuals
        % weighted = whether weighted regression to be run or not
        % num_subjs = number of subjects
    % OUTPUT:
        % betas_all = betas for all regressors
        % rsqaured_full = r-squared values for each participant
        % residuals_reg = residuals from fitting the model
        % coeffs_name = cell array with the model generated coefficients
        % name
        % h_vals = h-values from significance testing
        % p_vals = p-values from significance testing
        % posterior_up_subjs = posterior predicted updates by model

    % SET VARIABLES TO RUN THE FUNCTION
    id_subjs = unique(data.ID);
    var_names = {'pe','contrast_diff','salience','congruence','pe_sign'};
    
    % INITIALISE VARIABLES
    betas_all = NaN(length(num_subjs),num_vars);
    rsquared_full = NaN(length(num_subjs),1);
    posterior_up_subjs = [];
    data(data.pe == 0,:) = [];
    % FIT THE MODEL TO GET RESIDUALS 
    for i = num_subjs
        weight_y_n = 0;
        data_subject = data(data.ID == id_subjs(i),:);
        tbl = table(data_subject.pe,data_subject.up, round(data_subject.norm_condiff,2), data_subject.contrast,...
            data_subject.choice_cond,data_subject.congruence,data_subject.reward_unc,data_subject.pe_sign,...
            'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
            ,'reward_unc','pe_sign'});
        [betas,~,residuals_reg,coeffs_name,lm] = linear_fit(tbl,mdl,pred_vars,resp_var, ...
            cat_vars,num_vars,weight_y_n);
        res_subjs = [res_subjs; residuals_reg, repelem(id_subjs(i),length(residuals_reg)).'];
        post_up = predict(lm,tbl);
        posterior_update = post_up;
        posterior_up_subjs = [posterior_up_subjs; posterior_update];
    end
    
    % WEIGHTED REGRESSION USING RESIDUALS
    if weighted == 1
        data_weighted = [];
        for i = 1:length(num_subjs)
            data_weighted = [data_weighted; data(data.id == id_subjs(num_subjs(i)),:)];
        end
        [wt_subjs] = weights(data_weighted, res_subjs);
        wt_subjs(:,2) = res_subjs(:,2);
        posterior_up_subjs = [];
        for i = num_subjs
            weights_subj = wt_subjs(wt_subjs(:,2) == id_subjs(i));
            data_subject = data(data.ID == id_subjs(i),:);
            tbl = table(data_subject.pe,data_subject.up, round(data_subject.norm_condiff,2), data_subject.contrast,...
            data_subject.condition,data_subject.congruence,data_subject.reward_unc,data_subject.pe_sign,...
            'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
            ,'reward_unc','pe_sign'}); 
            [betas,rsquared,residuals_reg,coeffs_name,lm] = linear_fit(tbl,mdl,pred_vars, ...
                resp_var,cat_vars,num_vars,weighted,weights_subj);
            betas_all(i,:) = betas(2:end);
            rsquared_full(i,1) = rsquared;
        end
    end
    
    % INITIALISE ARRAYS TO STORE h AND p VALUES
    h_vals = nan(1,num_vars);
    p_vals = nan(1,num_vars); 
    t_vals = nan(1,num_vars);
    
    if weighted == 1
        for i = 1:num_vars
            [h_vals(1,i),p_vals(1,i),~,stats] = ttest(betas_all(:,i));
            t_vals(1,i) = stats.tstat;
        end
    end
end
