clc; clear;

%===========================
% Example 1: 2d 20-mode target
%===========================

% (Customizable) list of RNG seeds:
% To customize, just change the line below.
seeds=1; % default
% seeds = [1, 7, 13, 23, 37, 41, 59, 83, 101, 211];

%---------------------------
% Common settings 
%---------------------------
% Problem size
Np   = 1000;                 % number of particles
S    = 800;                  % max tempering stages if adaptive tempering
bound = [-1 -1;              % lower bounds (row 1)
          10 10];            % upper bounds (row 2)
d = size(bound,2);

% Target: use the local example_1/target.m
logpdf = @(th) target(th);   % returns log density

% Options (tune as needed)
opts = struct();

% 1) Tempering:small step early, relax later
opts.Tempering.Adaptive         = true;          % true => adaptive; false => fixed exponential (scalar=1e-6)
opts.Tempering.TargetBand       = [0.75, 0.85];    % keep rCESS high early:small dBeta
opts.Tempering.MinDeltaBeta     = 1e-3;          % absolute lower bound on step size
opts.Tempering.MaxDeltaBeta     = 0.1;           % absolute upper bound on step size
opts.Tempering.GrowthFactor     = 1.1;          % cap: dBeta_next <= 1.25 * dBeta_prev
opts.Tempering.WarmupNoResample = 5;             % forbid resampling for the first K stages
opts.Tempering.BandDecay=struct(...
    'Enabled',false,...                          % once beta is large, decay to a looser band
    'BetaThresh',0.9,...                        % start decaying when beta>0.9
    'Floor',[0.45,0.55]...                      % late stages: allow larger steps for efficiency
    );

% 2) Resampling
%  'ESS'  : trigger when ESS < ESSalpha * Np (post-reweight)
%  'Periodic': every K stages (K>=0; 0=never; 1=every stage)
opts.Resample.Policy    = 'ESS';
opts.Resample.Method    = 'residual';  % 'systematic' (default) | 'residual'
opts.Resample.ESSalpha  = 0.9;           % used as factor: thr = ESSalpha * min(current band)

% 3) Moves (any subset & order: 'ARM' | 'XOVER' | 'DEMH')
opts.Moves.Sequence = {'ARM','XOVER','DEMH'}; % 

% 4) ARM (adaptive random-walk MH)
opts.Moves.ARM.Cov       = 1e-3 * eye(d);   % initial; re-estimated each stage as (2.38^2/d)*SigmaW + jitter*I
opts.Moves.ARM.Jitter    = 1e-6;            % diagonal jitter to keep covariance PD

% 5) Crossover (single-point, tempered two-body MH)
opts.Moves.Crossover.pc  = 0.70;            % per-pair crossover probability

% 6) DE–MH (differential evolution MH)
opts.Moves.DEMH.Gamma    = 2.38 / sqrt(2*d);  
opts.Moves.DEMH.NoiseSD  = 1e-4;


% 7) Parallel
opts.Parallel.Enabled     = false;    % set true to use parfor
opts.Parallel.NumWorkers  = [];       % [] lets MATLAB decide

% 8) Thinning / printing
opts.Thinning     = 1;                % store/print every 5th stage (and always first/last)
opts.PrintConfig  = true;             % print the effective configuration once


%---------------------------
% Evaluation references
%---------------------------
Nm    = 20;
truth = [2.18 5.76; 8.67 9.59; 4.24 8.48; 8.41 1.68; 3.93 7.82; 3.25 3.47; 1.70 0.50;
         4.59 5.60; 6.91 5.81; 6.87 5.40; 5.41 2.65; 2.70 7.88; 4.98 3.70; 1.14 2.39;
         8.33 9.50; 4.93 1.50; 1.83 0.09; 2.26 0.31; 5.54 6.86; 1.69 8.11];
sigma = ones(Nm, d) * (0.1^2);

%---------------------------
% Postprocessing functions' Path handling
%---------------------------
postproc_rel = fullfile('..','postprocessing');
if exist(postproc_rel,'dir')
    addpath(postproc_rel);
end

%---------------------------
% Storage for metrics
%---------------------------
nS = numel(seeds);
seedsCell = cell(nS,1);          % posterior samples per seed (N x d)
Ds_vals  = nan(nS,1);
W2_vals  = nan(nS,1);



%---------------------------
% Batch loop: run PEM-SMC-AFP with different random seed
%---------------------------

for i = 1:nS
    seed_i = seeds(i);
    fprintf('=== Running seed %d of %d: %d ===\n', i, nS, seed_i);
    rng(seed_i, 'twister');

    % Run PEM-SMC-FP
    [parameter_iteration, out] = PATPEM_SMC(Np, S, bound, logpdf, opts);
    
    % Final-stage particles
    S_K = out.S_K;
    theta_final = squeeze(parameter_iteration(:,:,S_K));  % [Np x d]
    
    % ---- store posterior samples for this seed ----
    seedsCell{i} = theta_final;

    % Compute metrics only (no figures)
    try
        metrics = pem_smc_eval_posterior_vs_reference( ...
            theta_final, truth, ...
            'SigmaRef', sigma, ...
            'Dims', 1:d, ...
            'MakePlots',false,...
            'W2Method','empirical');    
    catch ME
        warning('pem_smc_eval_posterior_vs_reference failed (seed=%d): %s', seed_i, ME.message);
        metrics = struct();
    end

    % Collect target fields
    if isfield(metrics,'Ds_mean')
        Ds_vals(i) = metrics.Ds_mean;
    else
        warning('metrics is missing Ds_mean (seed=%d).', seed_i);
    end
    if isfield(metrics,'W2_marginal_mean')
        W2_vals(i) = metrics.W2_marginal_mean;
    else
        warning('metrics is missing W2_marginal_mean (seed=%d).', seed_i);
    end
end

%---------------------------
% Aggregate stats across seeds (ignore NaNs)
%---------------------------
Ds_mean_over_seeds = mean(Ds_vals, 'omitnan');
Ds_std_over_seeds  = std(Ds_vals,  'omitnan');

W2_mean_over_seeds = mean(W2_vals, 'omitnan');
W2_std_over_seeds  = std(W2_vals,  'omitnan');

%---------------------------
% Print results
%---------------------------
fprintf('\n===== Summary over %d seeds =====\n', nS);
fprintf('Ds_mean:            mean = %.6g, std = %.6g\n', Ds_mean_over_seeds, Ds_std_over_seeds);
fprintf('W2_marginal_mean:   mean = %.6g, std = %.6g\n', W2_mean_over_seeds, W2_std_over_seeds);

% Display: table with per-seed results
T = table(seeds(:), Ds_vals, W2_vals, ...
    'VariableNames', {'Seed','Ds_mean_per','W2_marginal_mean'});
disp(T);


%% Figures 

% ---------------------------
% After batch: randomly pick one or the first seed to visualize Diagnostics
%---------------------------
% Restore figure visibility for plotting
% idx_plot = randi(nS);               % randomly choose one index
idx_plot = 1;                         % the first seed for Reproduction
seed_plot = seeds(idx_plot);
fprintf('\n>>> Plotting full figures for randomly chosen seed: %d <<<\n', seed_plot);
% % 
% rng(seed_plot, 'twister');
[parameter_iteration, out] = PATPEM_SMC(Np, S, bound, logpdf, opts);

% Now draw  figures (trajectories, diagnostics, posteriors, scatter2D, eval)
[figs, metrics_plot] = plot_pem_smc_all(parameter_iteration, out, bound, opts, ...
    'NumTraj', 30, 'Seed', seeds, ...
    'Truth', truth, ...
    'SigmaRef', sigma, ...
    'W2Method','bimodal-ref','W2RefM',1e5, ...
    'Plots', {'trajectories','diagnostics','posteriors','Distance_eval','scatter2d'});

%% ---------------------------
% Draw 1D marginals – truth vs estimates: single seed or multiple seeds
%---------------------------
w      = ones(20,1) * (1/20);     % mixture weights (0.05 each)
samples_rep = seedsCell{1};      % representative seed: Np x 2 samples

fig1 = plot_20mode_marginals(samples_rep, truth, 0.1, w, ...
      'KDEBandwidth', 0.08, ...  % e.g., 0.06–0.10; if omitted, uses default
      'ShowSeedBand', false, ...
      'FigureTitle','1D marginals — truth vs posterior (random seed = 1)');

% When multiple seeds are provided, the function will instead plot:
%   truth as a line + posterior mean line + ±SD shaded band.
fig2 = plot_20mode_marginals(samples_rep, truth, 0.1, w, ...
      'SeedSamples', seedsCell, ...
      'KDEBandwidth', 0.07, ...
      'FigureTitle','1D marginals — truth vs posterior (repeated seeds: mean ± SD)');

