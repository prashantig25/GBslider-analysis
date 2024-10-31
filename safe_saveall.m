function safe_saveall(filename, newData)

    % function SAFE_SAVEALL is a custom replacement for MATLAB's save function
    % to ensure that data does not get saved unexpectedly.
    % INPUTS:
    %   filename: input string with filename
    %   newData: input data that needs to be saved

    % Check if file already exists
    if isfile(filename)
        % Load the existing data
        if contains(filename, '.mat')
            oldData = importdata(filename);
        elseif contains(filename, '.csv') || contains(filename, '.txt') || contains(filename, '.xlsx')
            oldData = readtable(filename);
        else
            error('Unsupported file type.');
        end
        
        % Compare the new data with the old data
        if isa(newData,"double")
            equality_check = isequaln(round(newData,10),round(oldData,10));
        elseif strcmp(string(class(newData)),"LinearModel") == 1
            equality_check = compare_LinearModels(newData,oldData);
        else
            equality_check = isequaln(newData,oldData);
        end

        if equality_check == 1
            disp('Data is consistent. No need to save.');
        else
            % For different data, create a new filename
            [path, name, ext] = fileparts(filename);
            timestamp = datestr(now, 'yymmdd');
            newFilename = strcat(name, "_", timestamp, ext);
            
            % Save with new filename
            if contains(ext, '.mat')
                save(fullfile(path, newFilename), 'newData');
            elseif contains(ext, '.csv') || contains(ext, '.txt') || contains(ext, '.xlsx')
                writetable(newData, fullfile(path, newFilename));
            end
            disp(['Data is different. Saved with a new filename: ' newFilename]);
        end
    else
        % Save the new data
        if contains(filename, '.mat')
            save(filename, 'newData');
        elseif contains(filename, '.csv') || contains(filename, '.txt') || contains(filename, '.xlsx')
            writetable(newData, filename);
        else
            error('Unsupported file type.');
        end
        disp(['File does not exist. Saved data to: ' filename]);
    end
end
