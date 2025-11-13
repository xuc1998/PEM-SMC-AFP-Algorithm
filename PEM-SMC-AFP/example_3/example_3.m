clc; clear; 


%===========================
% Example 3: Synthetic CoLM Benchmark for High-Dimensional Bayesian Calibration
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
S    = 1000;                  % number of tempering stages (fixed schedule)
bound=[10,  0.25,  2.5,   0.05,    2.5,   -500; 
       200, 0.75,  7.5,   0.08,    7.5,   -50];

d = size(bound,2);

% Target: use the local example_1/target.m
logpdf = @(theta,data) target(theta,data);   % returns log density

% Options (tune as needed)
opts = struct();

% 1) Tempering:small step early, relax later
opts.Tempering.Adaptive         = true;          % true => adaptive; false => fixed exponential (scalar=1e-6)
opts.Tempering.TargetBand       = [0.7, 0.85];    % keep rCESS high early:small dBeta
opts.Tempering.MinDeltaBeta     = 1e-4;          % absolute lower bound on step size
opts.Tempering.MaxDeltaBeta     = 0.1;           % absolute upper bound on step size
opts.Tempering.GrowthFactor     = 1.1;          % cap: dBeta_next <= 1.25 * dBeta_prev
opts.Tempering.WarmupNoResample = 20;             % forbid resampling for the first K stages
opts.Tempering.BandDecay=struct(...
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
opts.Moves.Sequence = {'AWM','XOVER','DEMH'}; % 

% 4) AWM (adaptive random-walk MH)
opts.Moves.AWM.Cov       = 1e-3 * eye(d);   % initial; re-estimated each stage as (2.38^2/d)*SigmaW + jitter*I
opts.Moves.AWM.Jitter    = 1e-6;            % diagonal jitter to keep covariance PD

% 5) Crossover (single-point, tempered two-body MH)
opts.Moves.Crossover.pc  = 0.70;            % per-pair crossover probability

% 6) DE–MH (differential evolution MH)
opts.Moves.DEMH.Gamma    = 2.38 / sqrt(2*d);
opts.Moves.DEMH.NoiseSD  = 5e-5;


% 7) Parallel
opts.Parallel.Enabled     = true;    % set true to use parfor
opts.Parallel.NumWorkers  = [];       % [] lets MATLAB decide

% 8) Thinning / printing
opts.Thinning     = 1;                % store/print every 5th stage (and always first/last)
opts.PrintConfig  = true;             % print the effective configuration once

% Data for target.m calculation
Data=struct();
Data.observed_LE=readmatrix('./obs_LE.txt');
Data.observed_NEE=readmatrix('./obs_NEE.txt');
Data.observed_RSM=readmatrix('obs_RSM.txt');
Data.sigma_LE=readmatrix('./sigma_LE.txt');
Data.sigma_NEE=readmatrix('./sigma_NEE.txt');
Data.sigma_RSM=readmatrix('./sigma_RSM.txt');

% file path for target.m CoLM model run
% CoLM model file path
old_path  = '/data/groups/lzu_public/home/u120220909911/lustre_data/Arou/PEM-SMC-AFP/WMO';
baseDir='/data/groups/lzu_public/home/u120220909911/lustre_data/Arou/PEM-SMC-AFP';
Data.old_path=old_path;
Data.baseDir=baseDir;
opts.Data=Data;

% Run PEM-SMC-FP algorithm
[parameter_iteration, out] = PEM_SMC_AFP(Np, S, bound, logpdf, opts);

% Plot & Statistics
plot_pem_smc_diagnostics(out, opts, Np)
plot_pem_smc_posteriors(parameter_iteration)
% save results
save parameter_iteration.mat parameter_iteration
save out.mat out
save bound.mat bound
save opts.mat opts
