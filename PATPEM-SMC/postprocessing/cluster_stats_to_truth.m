function [mu_est, sigma_est, idx] = cluster_stats_to_truth(X, truth)
%cluster_stats_to_truth  K-means initialized at truth; return per-cluster mean/std.
% Inputs:
%   X      : [N x d] posterior samples
%   truth  : [K x d0] reference centers (e.g., true mode locations)
% Outputs:
%   mu_est    : [K x d0] cluster means
%   sigma_est : [K x d0] cluster std (per dimension)
%   idx       : [N x 1] cluster assignment (based on first d0 dims)

    [~, d]   = size(X);
    [K, d0]  = size(truth);
    Xc       = X(:, 1:d0);  % cluster in the same dims as 'truth'

    opts = statset('MaxIter',200,'Display','off');
    [idx, C] = kmeans(Xc, K, ...
        'Start', truth, ...
        'MaxIter', 200, ...
        'Options', opts, ...
        'Replicates', 1, ...
        'EmptyAction','singleton', ...  % avoid empty clusters (n=0)
        'OnlinePhase','on');

    mu_est    = C;                  % [K x d0]
    sigma_est = nan(K, d0);

    for k = 1:K
        Xk = Xc(idx==k, :);
        nk = size(Xk,1);
        if nk == 0
            % Extremely rare due to 'singleton'; keep NaN to signal empty.
            sigma_est(k,:) = NaN(1,d0);
        elseif nk == 1
            % Unbiased std at n=1 is NaN; set to 0 (or use population std below).
            sigma_est(k,:) = zeros(1,d0);
        else
            sigma_est(k,:) = std(Xk, 0, 1);  % unbiased std per dimension
        end
    end
end
