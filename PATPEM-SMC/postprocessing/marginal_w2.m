function [W2_per_dim, W2_mean] = marginal_w2(X, Y, k, opts)
%MARGINAL_W2  Marginal (per-dimension) 2-Wasserstein distance with optional normalization.
%
%   [W2_per_dim, W2_mean] = MARGINAL_W2(X, Y)
%   [W2_per_dim, W2_mean] = MARGINAL_W2(X, Y, k)
%   [W2_per_dim, W2_mean] = MARGINAL_W2(X, Y, k, opts)
%
% DESCRIPTION
%   Computes per-dimension 2-Wasserstein distances (W2) between two
%   multivariate samples X and Y. For each coordinate j, the 1D W2 is
%   approximated by matching empirical quantiles on a common grid and
%   taking the root-mean-squared difference between the quantile curves
%   (equivalently, the L2 distance between inverse CDFs).
%
%   This implementation optionally **normalizes** X and Y before computing W2
%   to mitigate scale dominance when parameters have heterogeneous ranges.
%   The recommended option for Bayesian calibration with known prior bounds
%   is 'minmax' using prior [a_i, b_i] (scale-invariant and reproducible).
%
% INPUTS
%   X : [N1 x d] numeric matrix, rows are samples and columns are dimensions.
%   Y : [N2 x d] numeric matrix, rows are samples and columns are dimensions.
%   k : (optional) integer > 1, number of quantile points for the common
%       grid. Default = 512. Larger k increases accuracy and cost O(d*k).
%   opts : (optional) struct controlling normalization; fields:
%       opts.normalize : 'none' (default) | 'minmax' | 'zscore' | 'range'
%           'none'   : no scaling (results are scale-dependent).
%           'minmax' : linear min–max using prior bounds:  z = (x-a)./(b-a)
%                      Requires opts.a and opts.b (1xd or dx1). Preferred.
%           'zscore' : standard score using mean/s.d.: z = (x-mu)./sigma
%                      Uses opts.mu and opts.sigma if provided; otherwise
%                      computed from the pooled data [X;Y].
%           'range'  : min–max using pooled sample range of [X;Y].
%                      Not recommended for benchmarking (data-dependent).
%       opts.a, opts.b   : prior lower/upper bounds for 'minmax' (1xd or dx1).
%       opts.mu, opts.sigma : mean and std for 'zscore' (1xd or dx1).
%
% OUTPUTS
%   W2_per_dim : [d x 1] vector of marginal W2 distances for each dimension.
%   W2_mean    : scalar, mean(W2_per_dim) with 'omitnan' to ignore NaN dims.
%
% NOTES
%   • X and Y must have the same number of columns d.
%   • NaNs are removed independently per dimension before quantiles.
%   • This is an axis-aligned **marginal** metric (not sliced or full multivariate W2).
%   • Normalization guidance:
%       - For fair, scale-invariant comparisons across parameters, use
%         opts.normalize = 'minmax' with prior bounds (Table S1).
%       - The linear min–max map has a constant Jacobian; if priors are
%         uniform on [a_i,b_i], it does not alter posterior comparisons.
%       - If bounds are unknown, 'zscore' is a reasonable fallback.
%
% EXAMPLES
%   % 1) No normalization (scale-sensitive)
%   [w2_j, w2_mean] = marginal_w2(X, Y);
%
%   % 2) Prior-bounds min–max normalization (recommended)
%   opts.normalize = 'minmax'; opts.a = a; opts.b = b;   % 1xd bounds
%   [w2_j, w2_mean] = marginal_w2(X, Y, 512, opts);
%
%   % 3) Z-score normalization using provided moments
%   opts.normalize = 'zscore'; opts.mu = mu; opts.sigma = sigma;
%   [w2_j, w2_mean] = marginal_w2(X, Y, 1024, opts);
%
% SEE ALSO: quantile
%
% License: Provided as-is without warranty. Adapt comments to your style guide.

    % -------------------- Input validation --------------------
    if nargin < 3 || isempty(k), k = 512; end
    if nargin < 4 || isempty(opts), opts = struct; end
    if ~isfield(opts,'normalize'), opts.normalize = 'none'; end

    if ~isnumeric(X) || ~isnumeric(Y)
        error('X and Y must be numeric matrices.');
    end
    if size(X,2) ~= size(Y,2)
        error('X and Y must have the same number of columns (dimensions).');
    end
    if ~isscalar(k) || k <= 1
        error('k must be a scalar integer > 1.');
    end

    % -------------------- Optional normalization --------------------
    [~, d] = size(X);
    switch lower(opts.normalize)
        case 'none'
            % No scaling; distances are computed in native units.
        case 'minmax'
            if ~isfield(opts,'a') || ~isfield(opts,'b')
                error('opts.a and opts.b (prior bounds) are required for ''minmax''.');
            end
            a = opts.a(:)';  b = opts.b(:)';          % ensure row vectors
            if numel(a) ~= d || numel(b) ~= d
                error('opts.a and opts.b must have length d.');
            end
            s = b - a;  s(s == 0) = 1;                % avoid zero range
            X = (X - a) ./ s;
            Y = (Y - a) ./ s;
        case 'zscore'
            if isfield(opts,'mu') && isfield(opts,'sigma')
                mu = opts.mu(:)';  sig = opts.sigma(:)';
            else
                P = [X; Y];
                mu = nanmean(P, 1);
                sig = nanstd(P, 0, 1);
            end
            sig(sig == 0) = 1;                        % avoid divide-by-zero
            X = (X - mu) ./ sig;
            Y = (Y - mu) ./ sig;
        case 'range'
            % Data-dependent min–max (use only when prior bounds are unavailable).
            Pmin = nanmin([X; Y], [], 1);
            Pmax = nanmax([X; Y], [], 1);
            s = Pmax - Pmin;  s(s == 0) = 1;
            X = (X - Pmin) ./ s;
            Y = (Y - Pmin) ./ s;
        otherwise
            error('Unknown opts.normalize option: %s', opts.normalize);
    end

    % -------------------- Quantile grid --------------------
    q = linspace(0, 1, k);                % common grid, includes 0 and 1
    W2_per_dim = nan(d, 1);               % preallocate per-dimension output

    % -------------------- Per-dimension W2 --------------------
    for j = 1:d
        % Extract j-th coordinate and drop NaNs independently in X and Y
        xj = X(:, j);  xj = xj(~isnan(xj));
        yj = Y(:, j);  yj = yj(~isnan(yj));

        % Guard: need at least two points per side for stable quantiles
        if numel(xj) < 2 || numel(yj) < 2
            W2_per_dim(j) = NaN;
            continue;
        end

        % Empirical quantiles on the common grid (linear interpolation)
        xs = quantile(xj, q);
        ys = quantile(yj, q);

        % 1D W2 approximation: RMS distance between inverse CDFs (quantiles)
        W2_per_dim(j) = sqrt(mean((xs - ys).^2));
    end

    % -------------------- Scalar summary --------------------
    W2_mean = mean(W2_per_dim, 'omitnan');  % ignore NaN dimensions
end
