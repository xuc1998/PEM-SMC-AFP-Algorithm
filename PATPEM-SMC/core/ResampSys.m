function outIndex = ResampSys(w, N, method)
%RESAMPSYS  Low-variance resampling with two methods: 'systematic' (default) or 'residual'.
%
% USAGE (backward compatible):
%   idx = ResampSys(w)                      % N = numel(w), method='systematic'
%   idx = ResampSys(w, N)                   % method='systematic'
%   idx = ResampSys(w, N, 'residual')       % explicit N and method
%   idx = ResampSys(w, 'residual')          % N = numel(w), method='residual'
%
% INPUT
%   w      : weight vector (row/col ok). Should be nonnegative; will be normalized internally.
%   N      : number of offspring to draw (default: numel(w)).
%   method : 'systematic' | 'residual' (default: 'systematic')
%
% OUTPUT
%   outIndex : 1 x N integer indices in 1..numel(w).
%
% METHODS & WHEN TO USE
%   - 'systematic' (default): classic low-variance resampling, O(N).
%       Good general-purpose choice; simple and fast.
%   - 'residual': deterministic floor(N*w_i) allocation + low-variance draw
%       on residuals. Often preserves minority components better when
%       weights are highly unbalanced (e.g., imbalanced multi-modal targets).
%
% NOTE
%   We normalize w with a small numerical guard and enforce CDF(end)=1 to
%   avoid boundary issues.

    % ---------- overloads / defaults ----------
    if nargin < 2 || isempty(N); N = numel(w); end
    if nargin == 2 && (ischar(N) || (isstring(N) && isscalar(N)))
        method = N;  % ResampSys(w,'method')
        N = numel(w);
    end
    if nargin < 3 || isempty(method); method = 'systematic'; end
    method = lower(string(method));

    w = w(:);                 % column vector
    M = numel(w);
    if M == 0, error('ResampSys:EmptyWeights','Empty weight vector.'); end
    if N <= 0, outIndex = zeros(1,0); return; end

    % ---------- normalize & CDF ----------
    s = sum(w);
    if ~isfinite(s) || s <= 0
        error('ResampSys:BadWeights','Weights must be positive and finite.');
    end
    w = w / s;
    F = cumsum(w);
    F(end) = 1.0;             % numeric guard

    switch method
        case "systematic"
            outIndex = systematic_from_cdf(F, N);

        case "residual"
            % deterministic integer allocation
            a = floor(N * w);              % floor shares
            R = N - sum(a);                % remaining draws
            outIndex = zeros(N,1);
            c = 0;
            for i = 1:M
                if a(i) > 0
                    outIndex(c+1:c+a(i)) = i;
                    c = c + a(i);
                end
            end
            if R > 0
                r = N*w - a;               % residual weights (nonnegative)
                rsum = sum(r);
                if rsum > 0
                    r = r / rsum;
                    Fr = cumsum(r); Fr(end) = 1.0;
                    outIndex(c+1:end) = systematic_from_cdf(Fr, R); % use low-variance draw
                else
                    % All mass allocated deterministically; fill any slack with a fallback
                    outIndex(c+1:end) = systematic_from_cdf(F, R);
                end
            end
            outIndex = outIndex.';         % row vector

        otherwise
            error('ResampSys:UnknownMethod','Unknown method "%s". Use ''systematic'' or ''residual''.', method);
    end
end

% ===== helpers =====
function idx = systematic_from_cdf(F, N)
    % F: cumulative weights with F(end)=1
    u0 = rand / N;
    u  = u0 + (0:(N-1))' / N;
    idx = zeros(N,1);
    j = 1;
    for i = 1:N
        while u(i) > F(j)
            j = j + 1;
        end
        idx(i) = j;
    end
    idx = idx.'; % row
end
