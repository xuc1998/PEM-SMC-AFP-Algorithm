function [metrics, figs] = pem_smc_eval_posterior_vs_reference(X, truth, varargin)
%PEM_SMC_EVAL_POSTERIOR_VS_REFERENCE  Compare posterior samples to reference modes.
% Inputs:
%   X      : [N x d] final particles (rows = samples, cols = parameters)
%   truth  : [K x d0] reference samples/points (treated as samples from the reference posterior)
%
% Name-Value:
%   'SigmaRef' : [K x d0] reference std per mode (optional; [] -> use cluster std as proxy)
%   'Dims'     : vector of column indices of X to compare against truth (default: 1:d0)
%   'MakePlots': logical, make helper plots (default true)
%   'W2Method' : char/string selector for marginal W2 computation (default: 'empirical').
%                Options:
%                  - 'empirical'  : compute W2 between X and the provided truth samples Y.
%                                   Use this when 'truth' really contains samples from the
%                                   reference distribution.
%                  - 'bimodal-ref': compute W2 between X and the *analytic* bimodal reference
%                                   (1/3 N(-5·1_d, I_d) + 2/3 N(5·1_d, I_d)). Internally draws
%                                   M synthetic reference samples and then calls marginal_w2.
%   'W2RefM'   : positive integer, number of synthetic reference samples to draw when
%                'W2Method' = 'bimodal-ref' (default: 1e5). Ignored otherwise.
%
% Outputs:
%   metrics : struct with Ds_per_mode, Ds_mean/median, W2_marginal_per_dim, W2_marginal_mean, desc
%   figs    : struct of figure handles 
% -------------------- Parse args --------------------
args = inputParser;
args.addParameter('SigmaRef', [], @(x) isempty(x) || isnumeric(x));
args.addParameter('Dims', [], @(v) isempty(v) || (isvector(v) && isnumeric(v)));
args.addParameter('MakePlots', true, @islogical);
args.addParameter('W2Method', 'empirical', @(s)ischar(s) || isstring(s)); % 'empirical' or 'bimodal-ref'
args.addParameter('W2RefM', 1e5, @(x)isnumeric(x) && x>1000);             % reference sample size
args.parse(varargin{:});
prm = args.Results;

% -------------------- Basic sizes & dims --------------------
K  = size(truth,1);
d0 = size(truth,2);

dims = prm.Dims;
if isempty(dims)
    dims = 1:d0;                     % default: match truth's dimensionality
end

% Defensive checks
if max(dims) > size(X,2)
    error('Some indices in ''Dims'' exceed the number of columns in X.');
end
if numel(dims) ~= d0
    if size(truth,2) ~= numel(dims)
        error(['Size mismatch: width of truth (%d) must equal numel(Dims) (%d).'], ...
              size(truth,2), numel(dims));
    end
end

Xcmp = X(:, dims);     % columns of X to compare against truth

% -------------------- Cluster-to-truth stats for Ds --------------------
[mu_est, sigma_est, idx] = cluster_stats_to_truth(X, truth);

% Reference std per mode (if absent, approximate by cluster std)
sigma_ref = prm.SigmaRef;
if isempty(sigma_ref)
    warning('SigmaRef not provided; using cluster std as proxy in Ds (approximate).');
    sigma_ref = sigma_est;
end

% Ds per mode
Ds_per_mode = nan(K,1);
for m = 1:K
    Ds_per_mode(m) = Ds_metric(truth(m,:), sigma_ref(m,:), mu_est(m,:), sigma_est(m,:));
end

% -------------------- Marginal W2 (per-dimension + scalar mean) --------
opts = struct;
opts.normalize = 'zscore'; % optional: 'minmax','zscore','range'
switch lower(prm.W2Method)
    case 'empirical'
        % X vs. truth (truth must be samples from the reference)
        [W2_per_dim, W2_mean] = marginal_w2(Xcmp, truth, 512, opts);

    case 'bimodal-ref'
        % X vs. analytic bimodal reference (internally draws samples)
        opts.M = prm.W2RefM;
        [W2_per_dim, W2_mean] = marginal_w2_bimodal_ref(Xcmp, 512, opts);

    otherwise
        error('Unknown W2Method: %s', prm.W2Method);
end

% -------------------- Package metrics ----------------------------------
metrics = struct();

% Ds metrics (per-mode + summaries)
metrics.Ds_per_mode = Ds_per_mode;
metrics.Ds_mean     = mean(Ds_per_mode, 'omitnan');
metrics.Ds_std   = std(Ds_per_mode, 'omitnan');

% Marginal W2
metrics.W2_marginal_per_dim = W2_per_dim;   % [numel(dims) x 1]
metrics.W2_marginal_mean    = W2_mean;      % scalar summary

% Human-readable descriptions
metrics.desc = struct( ...
  'Ds_per_mode', ['Kx1 Ds distance per mode; lower = better; 0 = perfect match of per-mode ' ...
                  'marginal mean and std to reference.'], ...
  'Ds_mean',     ['Mean Ds across modes (scalar summary); lower = better; summarizes overall ' ...
                  'closeness across modes.'], ...
  'Ds_mstd',   ['Std Ds across modes (robust summary); lower = better; less sensitive ' ...
                  'to outlier modes.'], ...
  'W2_marginal_per_dim', ['Marginal (axis-aligned) 2-Wasserstein per dimension between X(:,Dims) ' ...
                  'and truth; lower = better; 0 = identical marginals.'], ...
  'W2_marginal_mean', ['Mean of marginal W2 across dimensions (scalar summary); ' ...
                  'lower = better; same units as the parameter.'] ...
);
% --------------------  plot ------------------------------
figs = struct();
if prm.MakePlots
    % --- bar chart of Ds per mode ---
    figs.DsBar = figure('Name','Ds per mode','Color','w');
    K = numel(Ds_per_mode);
    bar(1:K, Ds_per_mode, 'FaceColor',[0.2 0.5 0.9]);
    grid on; box on;
    xlabel('Mode'); ylabel('Ds');
    title('Ds per mode');
    xlim([0.5, K+0.5]); xticks(1:K);
end

end
