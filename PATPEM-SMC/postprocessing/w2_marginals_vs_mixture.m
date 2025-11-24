function [W2_1, W2_2] = w2_marginals_vs_mixture(samples, mu20, sigma, w)
    Mref = max(5*size(samples,1), 20000);
    idx  = randsample(20, Mref, true, w);
    x1_ref = mu20(idx,1) + sigma*randn(Mref,1);
    x2_ref = mu20(idx,2) + sigma*randn(Mref,1);
    W2_1 = w2_1d(samples(:,1), x1_ref);
    W2_2 = w2_1d(samples(:,2), x2_ref);
end