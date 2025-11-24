function [figs,metrics] = plot_pem_smc_all(parameter_iteration, out, bound, opts, varargin)
%PLOT_PEM_SMC_ALL   (trajectories + diagnostics + posteriors).
%
% Usage:
%   figs = plot_pem_smc_all(parameter_iteration, out, bound, opts, ... 'Truth', truthPoints)
%
% Name-Value options:
%   'NumTraj'    (default: 20)
%   'Seed'       (default: 42)
%   'Truth'      (default: [])   % K x d0 matrix of reference points for d >= 2
%   'SigmaRef'   (default: [])   % K x d0 reference std per mode (optional, for Ds)
%   'EvalDims'   (default: [])   % vector of column indices of X to compare; [] -> 1:d0 (width of Truth)
%   'W2Method'     (default: 'empirical')  % 'empirical' or 'bimodal-ref'
%   'W2RefM'       (default: 1e5)          % reference sample size when W2Method='bimodal-ref'
%   'Plots'      (default: 'all')  % 'all' or cellstr of plot names to draw.
%                                   Valid names:
%                                   'trajectories',
%                                   'diagnostics',
%                                   'posteriors',
%                                   'scatter2d',
%                                   'Distance_eval'
%   'PosteriorTruth' (default: struct('Enable',false))
%       Struct controlling overlay of the analytic truth curve in posterior plots.
%       Fields (all optional; shown with defaults):
%         .Enable   : false            % whether to overlay the truth curve
%         .Type     : 'bimodal-normal' % or 'custom'
%         .Weights  : [1/3 2/3]
%         .MuVals   : [-5 5]
%         .Sigma    : 1
%         .Bounds   : []               % [2 x d]; if empty, plotting code picks a range
%         .TruthPDF : []               % @(x,j) -> pdf values (used when Type='custom')
%         .NumGrid  : 400              % x-grid resolution for the curve
%         .LineSpec : '-'              % line style for the curve



% ---------- Parse name-value options ----------
args = inputParser;
args.addParameter('NumTraj', 20);
args.addParameter('Num2D',   10);
args.addParameter('Seed',    42);
args.addParameter('Truth',   []);
args.addParameter('SigmaRef',[]);
args.addParameter('EvalDims', []);
args.addParameter('MakeEvalPlots', true);
args.addParameter('W2Method','empirical', @(s)ischar(s) || isstring(s));
args.addParameter('W2RefM', 1e5, @(x)isnumeric(x) && x>1000);
args.addParameter('Plots','all', @(p)ischar(p) || isstring(p) || iscellstr(p));
args.addParameter('PosteriorTruth', struct('Enable', false), @(s) isstruct(s) || isempty(s));
args.parse(varargin{:});
prm = args.Results;

% ---------- Basic sizes ----------
[~, d, ~] = size(parameter_iteration);
S_K = out.S_K;

% ---------- 1) Trajectories ----------
% Figure: per-dimension particle trajectories across tempering stages
if want('trajectories', prm.Plots)
    figs.traj = plot_pem_smc_trajectories(parameter_iteration, opts, ...
    'NumTraj', prm.NumTraj, 'Seed', 42, 'NumDims', 3);
else
    figs.traj = [];
end



% ---------- 2) Diagnostics ----------
% Figure: ESS, CESS, acceptance rates, temperature schedule, etc.
if want('diagnostics', prm.Plots)
    figs.diagnostics = plot_pem_smc_diagnostics(out, opts, size(parameter_iteration,1));
else
    figs.diagnostics = [];
end

% ---------- 3) Posteriors ----------
% Figure: marginal posteriors at the final stage (one panel per parameter)
if want('posteriors', prm.Plots)
    TO = merge_truth_defaults(prm.PosteriorTruth, d);   % fill defaults, validate
    figs.marginals = plot_pem_smc_posteriors(parameter_iteration, ...
        'OverlayTruth', TO.Enable, ...
        'TruthType',    TO.Type, ...
        'Weights',      TO.Weights, ...
        'MuVals',       TO.MuVals, ...
        'Sigma',        TO.Sigma, ...
        'Bounds',       TO.Bounds, ...
        'NumGrid',      TO.NumGrid, ...
        'LineSpec',     TO.LineSpec, ...
        'TruthPDF',     TO.TruthPDF);
else
    figs.marginals = [];
end


% ---------- 4) 'scatter2d' ----------
% Figure: final particles in 2D (dims 1 & 2), with optional truth overlay
theta_final = squeeze(parameter_iteration(:,:,S_K));  % [Np x d]
if d >= 2 && want('scatter2d', prm.Plots)
    figs.scatter2D = plot_pem_smc_scatter2D_final(theta_final, bound, prm.Truth);
else
    figs.scatter2D = [];
end

% ---------- 5) 'Distance_eval'  ----------
% Posterior vs Reference: metrics 
% Metrics: Ds (per mode + summary), marginal W2 (per-dim + mean), 
if want('Distance_eval', prm.Plots) && d >= 1 && ~isempty(prm.Truth)
    [metrics,figs.Distance_eval] = pem_smc_eval_posterior_vs_reference( ...
        theta_final, prm.Truth, ...
        'SigmaRef', prm.SigmaRef, ...
        'Dims', prm.EvalDims, ...
        'MakePlots',prm.MakeEvalPlots,....
        'W2Method', prm.W2Method, ...
        'W2RefM',   prm.W2RefM);
    
else
    metrics   = [];
    figs.Distance_eval=[];
end

end


function tf = want(plotName, Plots)
    if ischar(Plots) || isstring(Plots)
        tf = strcmpi(Plots,'all') || strcmpi(Plots, plotName);
    else
        tf = any(strcmpi(Plots, plotName));
    end
end

function T = merge_truth_defaults(T, d)
    % defaults
    def.Enable   = false;
    def.Type     = 'bimodal-normal';
    def.Weights  = [1/3 2/3];
    def.MuVals   = [-5 5];
    def.Sigma    = 1;
    def.Bounds   = [];           % [2 x d]; optional
    def.TruthPDF = [];
    def.NumGrid  = 400;
    def.LineSpec = '-';
    if nargin < 1 || isempty(T), T = struct; end
    f = fieldnames(def);
    for i = 1:numel(f)
        k = f{i};
        if ~isfield(T,k) || isempty(T.(k)), T.(k) = def.(k); end
    end
    % light validation
    if numel(T.Weights)~=2, error('PosteriorTruth.Weights must be length-2.'); end
    if numel(T.MuVals) ~=2, error('PosteriorTruth.MuVals must be length-2.'); end
    if ~isempty(T.Bounds) && (~ismatrix(T.Bounds) || size(T.Bounds,1)~=2 || size(T.Bounds,2)~=d)
        error('PosteriorTruth.Bounds must be [2 x d] or empty.');
    end
end

