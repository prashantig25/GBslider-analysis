function desiredPath = createSavePaths(fullPath,reqPath)

% Step 1: Use fileparts to get the directory of the full path
[folderPath, ~, ~] = fileparts(fullPath);

% Step 2: Split the folder path into parts
pathParts = strsplit(folderPath, filesep);

% Step 3: Find the index of the desired folder
desiredFolderIndex = find(contains(pathParts, reqPath), 1);

% Step 4: Construct the desired path
if ~isempty(desiredFolderIndex)
    desiredPath = strjoin(pathParts(1:desiredFolderIndex), filesep);
else
    error('Desired folder not found in the path. Please ensure that your current directory contains the "Reward-learning-analysis (code_review)" folder');
end
end