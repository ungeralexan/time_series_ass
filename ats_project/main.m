
% This is the base path that directs to the folder 
basePath = fileparts(mfilename(' /Users/alexung/Desktop/ats_project'));

% Add subfolders to the MATLAB path
addpath(fullfile(basePath, 'task1'));
addpath(fullfile(basePath, 'Task2_UnitRoot'));
addpath(fullfile(basePath, 'Task3_LogLikelihood'));
addpath(fullfile(basePath, 'Task4_Estimation'));
addpath(fullfile(basePath, 'Shared'));