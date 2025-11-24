function [mmd2, bw] = mmd_rbf(X, Y, bw)
%mmd_rbf  Unbiased MMD^2 with RBF kernel; bandwidth via median heuristic if empty.
    if nargin < 3 || isempty(bw)
        Z = [X; Y];
        m = min(size(Z,1), 2000);
        D = pdist(Z(randperm(size(Z,1), m),:), 'euclidean');
        bw = median(D); if ~isfinite(bw) || bw<=0, bw = 1.0; end
    end
    Kxx = rbfK(X, X, bw);  Kyy = rbfK(Y, Y, bw);  Kxy = rbfK(X, Y, bw);
    n = size(X,1); m = size(Y,1);
    mmd2 = (sum(Kxx(:))-sum(diag(Kxx))) / (n*(n-1)) ...
         + (sum(Kyy(:))-sum(diag(Kyy))) / (m*(m-1)) ...
         - 2*sum(Kxy(:)) / (n*m);
end

function K = rbfK(A, B, bw)
%rbfK  RBF kernel matrix with bandwidth bw.
    Aa = sum(A.^2,2); Bb = sum(B.^2,2);
    D2 = bsxfun(@plus, Aa, Bb') - 2*(A*B');
    K  = exp(-D2 / (2*bw^2));
end
