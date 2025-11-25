clc; clear; 


%===========================
% Example 4: Real-Observation CoLM parameter (6) Calibration
%===========================

% (Customizable) list of RNG seeds:
% To customize, just change the line below.
seeds=1234; % default
% seeds = [1, 7, 13, 23, 37, 41, 59, 83, 101, 211];

%---------------------------
% Common settings
%---------------------------
% Problem size
Np   = 400;                 % number of particles
S    = 1000;                  % number of Beta stages (fixed schedule)
bound=[10,  0.25,  2.5,   0.05,    2.5,   -500; 
       200, 0.75,  7.5,   0.08,    7.5,   -50]; 

d = size(bound,2);

% Target: use the local example_1/target.m
logpdf = @(theta,data) target_new(theta,data);   % returns log density

% Options (tune as needed)
opts = struct();

% 1) Beta:small step early, relax later
opts.Beta.Adaptive         = true;          % true => adaptive; false => fixed exponential (scalar=1e-6)
opts.Beta.TargetBand       = [0.7, 0.85];    % keep rCESS high early:small dBeta
opts.Beta.MinDeltaBeta     = 1e-4;          % absolute lower bound on step size
opts.Beta.MaxDeltaBeta     = 0.1;           % absolute upper bound on step size
opts.Beta.GrowthFactor     = 1.1;          % cap: dBeta_next <= 1.25 * dBeta_prev
opts.Beta.WarmupNoResample = 20;             % forbid resampling for the first K stages
opts.Beta.BandDecay=struct(...
    'Enabled',true,...                          % once beta is large, decay to a looser band
    'BetaThresh',0.9,...                        % start decaying when beta>0.9
    'Floor',[0.45,0.55]...                      % late stages: allow larger steps for efficiency
    );

% 2) Resampling
%  'ESS'  : trigger when ESS < ESSalpha * Np (post-reweight)
%  'Periodic': every K stages (K>=0; 0=never; 1=every stage)
opts.Resample.Policy    = 'ESS';
opts.Resample.Method    = 'residual';  % 'systematic' (default) | 'residual'
opts.Resample.ESSalpha  = 0.5;           % used as factor: thr = ESSalpha * min(current band)

% 3) Moves (any subset & order: 'AWM' | 'XOVER' | 'DEMH')
opts.Moves.Enable = {'AWM','XOVER','DEMH'}; % 

% 4) AWM (adaptive random-walk MH)
opts.Moves.AWM.Cov       = 1e-3 * eye(d);   % initial; re-estimated each stage as (2.38^2/d)*SigmaW + jitter*I
opts.Moves.AWM.FoldType  = 'fold';       % 'fold' | 'reflect'
opts.Moves.AWM.Jitter    = 1e-6;            % diagonal jitter to keep covariance PD

% 5) Crossover (single-point, tempered two-body MH)
opts.Moves.Crossover.pc  = 0.70;            % per-pair crossover probability

% 6) DE–MH (differential evolution MH)
opts.Moves.DEMH.Gamma    = 2.38 / sqrt(2*d);
opts.Moves.DEMH.NoiseSD  = 5e-5;
opts.Moves.DEMH.FoldType = 'fold';       % 'fold' | 'reflect'

% 7) Parallel
opts.Parallel.Enabled     = true;    % set true to use parfor
opts.Parallel.NumWorkers  = [];       % [] lets MATLAB decide

% 8) Thinning / printing
opts.Thinning     = 1;                % store/print every 5th stage (and always first/last)
opts.PrintConfig  = true;             % print the effective configuration once

% Data for target.m calculation
Data=struct();
Data.observed_LE=readmatrix('./LE_verify.txt');
Data.observed_NEE=readmatrix('./NEE_verify.txt');
Data.observed_RSM=readmatrix('./RSM_verify.txt');

% LE: Yeo–Johnson (robust to negatives)
Data.LE.distribution  = 'studentt'; Data.LE.nu = 5; Data.LE.phi = 0.4;
Data.LE.heteroA       = 0.005;      Data.LE.inflation = 1.4;
Data.LE.transform     = 'yeojohnson'; Data.LE.lambda = 0.2; Data.LE.addJacobian = true;

% NEE: Yeo–Johnson
Data.NEE.distribution = 'studentt'; Data.NEE.nu = 5; Data.NEE.phi = 0.3;
Data.NEE.heteroA      = 0.03;       Data.NEE.inflation = 1.3;
Data.NEE.transform    = 'yeojohnson'; Data.NEE.lambda = 0.2; Data.NEE.addJacobian = true;

% RSM: standard Box–Cox
Data.RSM.distribution = 'studentt'; Data.RSM.nu = 5; Data.RSM.phi = 0.2;
Data.RSM.heteroA      = 0.2;        Data.RSM.inflation = 1.5;
Data.RSM.transform    = 'boxcox';   Data.RSM.lambda = 0.15; Data.RSM.addJacobian = true;



% file path for target.m CoLM model run
% CoLM model file path
old_path  = '/data/groups/lzu_public/home/u120220909911/lustre_data/Arou/PATPEM_SMC/WMO';
baseDir='/data/groups/lzu_public/home/u120220909911/lustre_data/Arou/PATPEM_SMC';
Data.old_path=old_path;
Data.baseDir=baseDir;
opts.Data=Data;

% Run PEM-SMC-FP algorithm
[parameter_iteration, out] = PEM_SMC_AFP(Np, S, bound, logpdf, opts);

% Plot & Statistics
 plot_pem_smc_diagnostics(out, opts, Np)

% save results
save parameter_iteration.mat parameter_iteration
save out.mat out
save bound.mat bound
save opts.mat opts
