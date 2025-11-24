function [W2_per_dim, W2_mean] = marginal_w2_bimodal_ref(X, k, opts)
%MARGINAL_W2_BIMODAL_REF  W2 between X and an analytic bimodal reference.
%   Reference: (1/3) N(-5*1_d, I_d) + (2/3) N( 5*1_d, I_d)
%
% INPUT
%   X    : [N x d] samples (rows)
%   k    : quantile grid size (default 512); same semantics as marginal_w2
%   opts : struct with fields (all optional)
%       .M        : number of reference samples to draw (default 1e5)
%       .w        : [1x2] mixture weights (default [1/3 2/3])
%       .mu_vals  : [1x2] component means per dimension (default [-5 5])
%       .sigma    : scalar std for each dimension in each component (default 1)
%       .normalize / .a / .b / .mu / .sigma_z : passed through to marginal_w2
%
% OUTPUT
%   W2_per_dim, W2_mean : same outputs as marginal_w2

    if nargin < 2 || isempty(k),   k = 512; end
    if nargin < 3, opts = struct; end

    [~, d] = size(X);

    % Defaults for the analytic reference
    M   = getfield_with_default(opts, 'M', 1e5);
    w   = getfield_with_default(opts, 'w', [1/3, 2/3]);
    muv = getfield_with_default(opts, 'mu_vals', [-5, 5]);  % [-5, +5]
    sig = getfield_with_default(opts, 'sigma', 1);          % std

    % Draw M i.i.d. samples from the reference mixture
    Y = sample_bimodal_normal(d, M, w, muv, sig);

    % Reuse the original marginal_w2 implementation (normalization unchanged)
    [W2_per_dim, W2_mean] = marginal_w2(X, Y, k, opts);
end



function Z = sample_bimodal_normal(d, M, w, mu_vals, sigma)
%SAMPLE_BIMODAL_NORMAL  Draw M samples in R^d from the bimodal reference.
%   Mixture: w(1)*N(mu_vals(1)*1_d, I_d*sigma^2) + w(2)*N(mu_vals(2)*1_d, I_d*sigma^2)

    if numel(w) ~= 2, error('w must be length 2.'); end
    if numel(mu_vals) ~= 2, error('mu_vals must be length 2.'); end

    comp1 = rand(M,1) < w(1);           % logical index for component 1
    Z = zeros(M, d);
    Z(comp1,:)  = mu_vals(1) + sigma * randn(sum(comp1), d);
    Z(~comp1,:) = mu_vals(2) + sigma * randn(sum(~comp1), d);
end

function v = getfield_with_default(s, name, default)
%GETFIELD_WITH_DEFAULT  Return s.(name) if present and non-empty; otherwise default.
    if isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = default;
    end
end

