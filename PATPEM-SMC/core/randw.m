function theta_new = randw(theta_old, LB, UB, RWM)
%RANDW  Box-constrained RWM proposal using a single Gaussian draw + reflection.
%
% Input
%   theta_old : 1 x d current state
%   LB, UB    : 1 x d lower/upper bounds
%   RWM       : struct with fields:
%                 .Cov      : d x d proposal covariance (default 1e-4*I)
%                 .Jitter   : scalar diagonal jitter (optional; default 1e-6)
%
% Output
%   theta_new : 1 x d proposed state after reflecting into [LB,UB]
%
% Notes
% - Robust to near-singular/unsymmetric covariance by enforcing symmetry and
%   increasing diagonal jitter until Cholesky succeeds.
% - Boundaries are handled by triangle-wave reflection per dimension.

    d = numel(theta_old);

    % ---- Defaults ----
    if ~isfield(RWM,'Cov')    || isempty(RWM.Cov),    RWM.Cov    = 1e-4*eye(d); end
    if ~isfield(RWM,'Jitter') || isempty(RWM.Jitter), RWM.Jitter = 1e-6;        end

    % ---- Ensure symmetry & positive definiteness (Cholesky-safe) ----
    Sigma = RWM.Cov;
    % Guard wrong size or NaN
    if ~ismatrix(Sigma) || any(size(Sigma) ~= d) || any(~isfinite(Sigma(:)))
        Sigma = 1e-4*eye(d);
    end

    % Exact symmetrization
    Sigma = 0.5 * (Sigma + Sigma.');

    % Base jitter: relative to scale of Sigma, with a tiny absolute floor
    baseScale = max(1e-10, mean(abs(diag(Sigma))));   % scale proxy
    jit = max(RWM.Jitter, 1e-8);
    jit = max(jit, 1e-6 * baseScale);

    % Try Cholesky; if it fails, escalate jitter
    [R,p] = cholcov(Sigma + jit*eye(d), 0);   % R'*R = Sigma+jit*I, p=0 if OK
    tries = 0;
    while p ~= 0 && tries < 6
        jit = jit * 10;                        % escalate aggressively
        [R,p] = cholcov(Sigma + jit*eye(d), 0);
        tries = tries + 1;
    end

    if p ~= 0
        % Final fallback: use only diagonal variances (very conservative)
        diagVar = abs(diag(Sigma));
        if all(diagVar <= 0), diagVar = ones(d,1)*1e-4; end
        [R,~] = chol(diag(diagVar) + (jit+1e-6)*eye(d));
    end

    % ---- Draw proposal: theta_old + N(0, Sigma_PD) ----
    delta = randn(1,d) * R;           % R is upper-tri: N(0, R'*R)
    z = theta_old + delta;

    % ---- Reflect back into the box (triangle-wave reflection) ----
    theta_new = reflect_into_box(z, LB, UB);
end

% --------- local reflection utility ---------
function y = reflect_into_box(z, LB, UB)
%REFLECT_INTO_BOX  Triangle-wave reflection to enforce bounds.
% Applies per-dimension reflection into [LB,UB].

    L = UB - LB;
    y = (z - LB) ./ L;        % scale to unit
    y = abs(mod(y,2) - 1);    % triangle wave in [0,1]
    y = LB + y .* L;          % scale back
end
