function [avg_ydata,avg_binneddata,sem_ydata] = binned_analysis(binned_data,y_data,run_id,valid_trials,id_subjs,num_subjs,nbins)
    % function binned_analysis sorts trials into percentile bins of a
    % variable and computes the per-subject mean of a dependent variable
    % within each bin, then averages those per-subject means across
    % subjects. Used to plot e.g. mean LR or mean absolute UP against
    % binned contrast difference/PE.
    %
    % INPUTS:
    %   binned_data: variable to bin (e.g. absolute contrast difference
    %   or absolute PE), one row per trial, before trial filtering
    %   y_data: dependent variable averaged within each bin (e.g. LR or
    %   absolute UP), one row per trial, before trial filtering
    %   run_id: subject ID per trial, before trial filtering
    %   valid_trials: logical selection vector marking which trials to
    %   keep (e.g. excludes PE = 0 and outlier-LR trials)
    %   id_subjs: unique subject IDs
    %   num_subjs: number of subjects
    %   nbins: number of percentile bins
    %
    % OUTPUTS:
    %   avg_ydata: mean of y_data per bin, averaged across subjects
    %   avg_binneddata: mean of binned_data per bin, averaged across
    %   subjects (bin centres, used to correlate against avg_ydata)
    %   sem_ydata: SEM of y_data per bin, across subjects

    % BIN EDGES ARE COMPUTED ON ALL TRIALS, BEFORE FILTERING
    bin_edges = prctile(binned_data, 0:100/nbins:100); % percentile edges
    bins = discretize(binned_data, bin_edges); % assign each trial to a bin

    % APPLY TRIAL FILTER
    run_id = run_id(valid_trials);
    y_data = y_data(valid_trials);
    bins = bins(valid_trials);
    binned_data = binned_data(valid_trials);

    % MEAN y_data AND binned_data PER BIN, PER SUBJECT
    avg_ydata_bins = NaN(nbins,num_subjs);
    avg_behv_bins = NaN(nbins,num_subjs);
    for b = 1:nbins % run for each bin
        for n = 1:num_subjs % run for each subject
            bins_subj = bins(run_id == id_subjs(n));
            y_data_subj = y_data(run_id == id_subjs(n)); % data from that bin for that subject
            binned_data_subj = binned_data(run_id == id_subjs(n)); % binned data for that subject
            avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b));
            avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
        end
    end

    % AVERAGE PER-SUBJECT BIN MEANS ACROSS SUBJECTS
    avg_ydata = nanmean(avg_ydata_bins,2);
    avg_binneddata = nanmean(avg_behv_bins,2);
    sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(num_subjs);
end
