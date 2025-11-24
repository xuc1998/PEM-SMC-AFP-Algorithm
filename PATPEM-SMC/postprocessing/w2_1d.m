function W2 = w2_1d(a, b)
    a = a(:); b = b(:);
    a = a(isfinite(a)); b = b(isfinite(b));
    K = min(max(numel(a), numel(b)), 100000);
    a = sort(randsample(a, K, true));
    b = sort(randsample(b, K, true));
    W2 = sqrt(mean((a-b).^2));
end