clc; clear;

% ===========================================================
% Example 2: 100-D bimodal Gaussian
%   pi(x) = (1/3) N(-5*1_d, I_d) + (2/3) N( 5*1_d, I_d)
% ===========================================================

% RNG seeds (customize as needed)
seeds = 1;
% seeds = [1 7 13 23 37]; % 41 59 83 101 211

% ---------------------------
% Common settings
% ---------------------------
Np    = 1000;               % number of particles
S     = 1000;               % stage budget (upper bound; loop stops earlier if beta hits 1)
d     = 100;                % dimensionality
bound = [ -10*ones(1,d) ;   % lower bounds
           10*ones(1,d) ];  % upper bounds

% Target log-density handle (1-arg or 2-arg both supported by PEM_SMC_AFP)
logpdf = @(th) target(th);

% ---------------------------
% Options (new schedule interface)
% ---------------------------
opts = struct();

% 1) Tempering (Adaptive rCESS-band schedule)
opts.Tempering.Adaptive         = true;          % true => adaptive; false => fixed exponential (scalar=1e-6)
opts.Tempering.TargetBand       = [0.8, 0.9];    % per-stage rCESS target band
opts.Tempering.MinDeltaBeta     = 1e-6;          % absolute lower bound on step size
opts.Tempering.MaxDeltaBeta     = 0.1;           % absolute upper bound on step size
opts.Tempering.GrowthFactor     = 1.1;          % cap: dBeta_next <= 1.25 * dBeta_prev
opts.Tempering.WarmupNoResample = 2;             % forbid resampling for the first K stages

% Optional: gradually relax the band in early/mid stages to speed up later steps
opts.Tempering.BandDecay = struct( ...
    'Enabled',    false, ...     % turn band decay on/off
    'BetaThresh', 0.9,  ...     % start decaying once current beta exceeds this threshold
    'Floor',      [0.6 0.85] ... % late-stage target rCESS band (ABSOLUTE); lower/wider -> larger Δβ
);

% 2) Resampling
%  'ESS'  : trigger when ESS < ESSalpha * Np (post-reweight)
%  'Periodic': every K stages (K>=0; 0=never; 1=every stage)
opts.Resample.Policy    = 'ESS';
opts.Resample.Method    = 'systematic';  % 'systematic' (default) | 'residual'
opts.Resample.ESSalpha  = 0.9;           % used as factor: thr = ESSalpha * min(current band)

% 3) Move operators (any subset/order: 'ARM' | 'XOVER' | 'DEMH')
opts.Moves.Sequence = {'ARM','XOVER','DEMH','DEMH'};

% 4) ARM (Adaptive Random-Walk MH)
opts.Moves.ARM.Cov       = 1e-3 * eye(d);   % initial; re-estimated per stage as (2.38^2/d)*SigmaW + jitter*I
opts.Moves.ARM.Jitter    = 1e-6;            % diagonal jitter to keep covariance PD

% 5) Crossover (single-point, tempered two-body MH)
opts.Moves.Crossover.pc  = 0.70;            % per-pair crossover probability

% 6) DE–MH (Differential Evolution MH)
opts.Moves.DEMH.Gamma    = 2.38 / sqrt(2*d);
opts.Moves.DEMH.NoiseSD  = 1e-4;


% 7) Parallel (enable if you have PCT; otherwise keep false)
opts.Parallel.Enabled     = false;          % set true to use parfor
opts.Parallel.NumWorkers  = [];             % [] lets MATLAB decide

% 8) Thinning / printing
opts.Thinning     = 1;                      % store/print every stage (and always first/last)
opts.PrintConfig  = true;                   % print effective configuration once

% ---------------------------
% Reference mixture for evaluation
% ---------------------------
truth = [-5*ones(1,d); 5*ones(1,d)];
sigma = ones(2,d);

% ---------------------------
% Postprocessing path (if you have helper scripts)
% ---------------------------
postproc_rel = fullfile('..','postprocessing');
if exist(postproc_rel,'dir'), addpath(postproc_rel); end

% ---------------------------
% Storage for per-seed metrics
% ---------------------------
nS = numel(seeds);
Ds_vals   = nan(nS,1);
W2_vals   = nan(nS,1);
Time_vals = nan(nS,1);  % wall-clock time (seconds)

% ---------------------------
% Batch loop over seeds
% ---------------------------
for i = 1:nS
    seed_i = seeds(i);
    fprintf('=== Running seed %d of %d: %d ===\n', i, nS, seed_i);
    rng(seed_i, 'twister');

    % Run PEM-SMC-AFP
    [parameter_iteration, out] = PEM_SMC_AFP(Np, S, bound, logpdf, opts);

    % Record wall time (seconds)
    if isfield(out,'timing') && isfield(out.timing,'elapsed_sec') && ~isempty(out.timing.elapsed_sec)
        Time_vals(i) = out.timing.elapsed_sec;
    else
        Time_vals(i) = NaN;
    end

    % Final-stage particles
    S_K = out.S_K;
    theta_final = squeeze(parameter_iteration(:,:,S_K));  % [Np x d]

    % Compute metrics (no plots)
    try
        metrics = pem_smc_eval_posterior_vs_reference( ...
            theta_final, truth, ...
            'SigmaRef', sigma, ...
            'Dims', 1:d, ...
            'MakePlots', false, ...
            'W2Method', 'bimodal-ref', ...
            'W2RefM', 5e5);
    catch ME
        warning('pem_smc_eval_posterior_vs_reference failed (seed=%d): %s', seed_i, ME.message);
        metrics = struct();
    end

    % Collect key fields
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

% ---------------------------
% Aggregate stats across seeds (ignore NaNs)
% ---------------------------
Ds_mean_over_seeds  = mean(Ds_vals,  'omitnan');
Ds_std_over_seeds   = std(Ds_vals,   'omitnan');

W2_mean_over_seeds  = mean(W2_vals,  'omitnan');
W2_std_over_seeds   = std(W2_vals,   'omitnan');

Time_mean_over_seeds = mean(Time_vals, 'omitnan');
Time_std_over_seeds  = std(Time_vals,  'omitnan');

% ---------------------------
% Print results
% ---------------------------
fprintf('\n===== Summary over %d seeds =====\n', nS);
fprintf('Ds_mean:            mean = %.6g, std = %.6g\n', Ds_mean_over_seeds, Ds_std_over_seeds);
fprintf('W2_marginal_mean:   mean = %.6g, std = %.6g\n', W2_mean_over_seeds, W2_std_over_seeds);
fprintf('Wall time (sec):    mean = %.6g, std = %.6g\n', Time_mean_over_seeds, Time_std_over_seeds);

% Per-seed table (now includes time)
T = table(seeds(:), Ds_vals, W2_vals, Time_vals, ...
    'VariableNames', {'Seed','Ds_mean','W2_marginal_mean','Time_sec'});
disp(T);

%% Figures
% ---------------------------
% Full-figure diagnostics 
% ---------------------------
% PT overlays the analytic 1D marginals for the bimodal normal in posterior plots
PT = struct( ...
    'Enable',  true, ...
    'Type',    'bimodal-normal', ...
    'Weights', [1/3 2/3], ...
    'MuVals',  [-5 5], ...
    'Sigma',   1, ...
    'Bounds',  [-10*ones(1,d); 10*ones(1,d)], ...
    'NumGrid', 50, ...
    'LineSpec','-');

[figs, metrics_plot] = plot_pem_smc_all(parameter_iteration, out, bound, opts, ...
    'NumTraj', 20, 'Seed', seeds, ...
    'Truth', truth, ...
    'SigmaRef', sigma, ...
    'W2Method','bimodal-ref','W2RefM',1e5, ...
    'Plots', {'trajectories','diagnostics','posteriors','Distance_eval'}, ... % 'trajectories','diagnostics','posteriors','Distance_eval'
    'PosteriorTruth', PT);
