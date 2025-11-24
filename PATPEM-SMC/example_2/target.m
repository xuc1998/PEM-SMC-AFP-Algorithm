function L = target(x)
%TARGET  Log target density for a d-D bimodal Gaussian mixture.
%   INPUT:
%     x : 1 x d row vector (point in R^d)
%   OUTPUT:
%     L : scalar log-density log pi(x)
%
%   pi(x) = (1/3) N(-5*1_d, I_d) + (2/3) N( 5*1_d, I_d)

    d = numel(x);

    % mixture weights and means
    w1 = 1/3;    w2 = 2/3;
    mu1 = -5*ones(1,d);
    mu2 =  5*ones(1,d);

    % log-density of N(mu, I_d): c0 - 0.5 * ||x-mu||^2
    c0 = -0.5 * d * log(2*pi);

    % log component densities (scalars)
    diff1 = x - mu1;
    l1 = log(w1) + c0 - 0.5 * (diff1 * diff1');  % 1xd * dx1 -> scalar

    diff2 = x - mu2;
    l2 = log(w2) + c0 - 0.5 * (diff2 * diff2');

    % log-sum-exp for numerical stability
    m = max(l1, l2);
    L = m + log(exp(l1 - m) + exp(l2 - m));
end
