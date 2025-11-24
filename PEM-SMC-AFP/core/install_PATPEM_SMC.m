%% 
%
% /PATPEM_SMC/
% ├─ core/                      % core implementation
% │   ├─ PATPEM_SMC.m          % main SMC routine (public API)
% │   ├─ ResampSys.m            % systematic resampling
% │   ├─ randw.m                % random-walk proposal
% │   ├─ Generatep_fold.m       % DE-MH mutation with fold-style bounds
% │   ├─ sequencesGen.m         % exponential schedule for beta
% │   └─ install_PATPEM_SMC.m  % this installation script
% ├─ example_1/                 % case study 1: 2-D 20-mode Gaussian mixture
% │   ├─ example_1.m            % driver script (user edits Np, S, bounds, etc.)
% │   └─ target.m               % log target density for this example
% ├─ example_2/                 % case study 2: 100-D bimodal Gaussian
% │   ├─ example_2.m
% │   └─ target.m
% ├─ example_3/                 % case study 3: synthetic CoLM calibration
% │   ├─ example_3.m
% │   └─ target.m
% ├─ example_4/                 % case study 4: real-world CoLM calibration
% │   ├─ example_4.m
% │   └─ target.m
% ├─ postprocessing/            % plotting / diagnostics / utilities for examples
% └─ README.md                  % quick usage & tips
%
%% install_PATPEM_SMC.m
% One-shot script to add the PATPEM_SMC core to the MATLAB path.
% This script adds the 'core' folder (and optionally 'postprocessing') to
% the MATLAB search path. It does NOT add any example_* folders in order to
% avoid name collisions between different target.m files.
%
% Usage:
%   1) From the project root, run:
%        >> cd path/to/PATPEM_SMC
%        >> core/install_PATPEM_SMC
%   2) To run a specific example:
%        >> cd example_1
%        >> example_1
%      To switch examples, simply cd to another example_k folder and run
%      its script (e.g., example_2.m).
%
% Persisting the path:
%   - If you want these paths to persist across MATLAB sessions, call:
%        >> savepath
%
% Notes:
%   - Core API files (PATPEM_SMC.m, ResampSys.m, randw.m, Generatep_fold.m,
%     sequencesGen.m, etc.) live in the 'core' folder.
%   - Each example folder (example_1/, example_2/, ...) contains its own
%     example_k.m and target.m. Keeping examples off the global path avoids
%     accidental shadowing between different target.m files.
%   - The 'postprocessing' folder provides common plotting/diagnostic
%     utilities shared by the examples.

clc;clear;
%% 1) Locate project root and core folder
thisFile = mfilename('fullpath');   % absolute path to this script
coreDir  = fileparts(thisFile);     % .../PATPEM_SMC/core
rootDir  = fileparts(coreDir);      % .../PATPEM_SMC  (project root)

%% 2) Collect core directories to add
% We add the core directory itself. Optionally also add diagnostics (if any)
% and the postprocessing folder under the project root.
coreDirs = {coreDir};

% --- postprocessing under project root (sibling of core/) ---
postDir = fullfile(rootDir, 'postprocessing');
if exist(postDir, 'dir')
    coreDirs{end+1} = postDir;
    % If you want to also add subfolders:
    % coreDirs{end+1} = genpath(postDir);
end

%% 3) Add core directories to the MATLAB path (idempotent)
for i = 1:numel(coreDirs)
    d = coreDirs{i};
    if ~contains([path pathsep], [d pathsep])
        addpath(d);
        fprintf('[PATPEM_SMC] Added to path: %s\n', d);
    else
        fprintf('[PATPEM_SMC] Already on path: %s\n', d);
    end
end

%% 4) Friendly tip to the user
fprintf('\n[PATPEM_SMC] Core installed.\n');
fprintf('To run an example:\n');
fprintf('  >> cd example_1\n');
fprintf('  >> example_1\n');
fprintf('Switch examples by cd-ing into another example_k folder and running its script.\n');
fprintf('Use "savepath" if you want these paths to persist across sessions.\n\n')
