function log_L = target_new(x, data)
% CoLM (Enhanced, robust likelihood with Box-Cox-Shift and Yeo–Johnson)
% Computes the joint log-likelihood for LE/NEE/RSM given parameter vector x.
%
% Workflow:
%   1) Write x to a unique run folder cloned from a template.
%   2) Run CoLM externally.
%   3) Read model outputs and evaluate a per-series robust likelihood with
%      configurable transform (none/boxcox/boxcox_shift/yeojohnson),
%      AR(1), Student-t, heteroskedasticity, and scale inflation.
%
% Backward compatibility:
%   - If no robust fields are provided in `data`, behavior reduces to the
%     original concentrated Gaussian likelihood (independent series).
%
% Optional per-series fields in `data` for each S in {LE, NEE, RSM}:
%   data.S.transform     : 'none' (default) | 'boxcox' | 'boxcox_shift' | 'yeojohnson'
%   data.S.lambda        : scalar lambda for boxcox/yeojohnson (required when applicable)
%   data.S.addJacobian   : logical, include Jacobian (default true for transforms, else false)
%   data.S.shift         : for 'boxcox_shift': numeric c>0 or 'auto' (default: 'auto')
%   data.S.phi           : AR(1) coefficient in [-1,1] (default 0)
%   data.S.includeInit   : include stationary initial-state term (default false)
%   data.S.distribution  : 'gaussian' (default) | 'studentt'
%   data.S.nu            : degrees of freedom for t (default 5 if studentt)
%   data.S.heteroA       : a >= 0 for s_t^2 = sigma^2 (1 + a*|Y_sim(t)|)^2 (default 0)
%   data.S.inflation     : tau >= 1 scale inflation (default 1)
%
% Notes:
%   - All helper routines are pure and parfor-safe.
%   - Concentrated scale s^2 is estimated from AR(1) innovations with optional hetero weights.

    % ---------------- Observations ----------------
    y_LE  = data.observed_LE(:);
    y_NEE = data.observed_NEE(:);
    y_RSM = data.observed_RSM(:);

    % ---------------- Prepare unique run dir ----------------
    old_path = data.old_path;                          % template directory
    baseDir  = data.baseDir;                           % parent directory for runs
    new_path = copyToUniqueRunDir(old_path, baseDir);  % collision-free copy
    cleanupObj = onCleanup(@() safeRmdir(new_path));   % ensure cleanup

    % ---------------- Write parameter vector ----------------
    fp = fullfile(new_path, 'Arou', 'input_step.txt');
    fid = fopen(fp,'w');
    if fid == -1
        warning('CoLM:IO', 'Could not open parameter file for writing: %s', fp);
        error('CoLM:Abort', 'Stopping due to I/O error while opening parameter file.');
    end
    x = x(:);  % columnize
    for i = 1:numel(x)
        fprintf(fid, '%24.16e\n', x(i));
    end
    fclose(fid);

    % ---------------- Run external model ----------------
    status = system(fullfile(new_path, 'run'));
    if status ~= 0
        warning('CoLM:RunFailed', 'External model failed (status=%d) in run folder: %s', status, new_path);
        error('CoLM:Abort', 'Stopping due to external model failure.');
    end

    % ---------------- Read model outputs -------------------
    fLE  = fullfile(new_path, 'Arou', 'output_LE.txt');
    fNEE = fullfile(new_path, 'Arou', 'output_NEE.txt');
    fRSM = fullfile(new_path, 'Arou', 'output_RSM.txt');

    mLE  = read_vector_robust(fLE);
    mNEE = read_vector_robust(fNEE);
    mRSM = read_vector_robust(fRSM);

    % ---------------- Early rejects ------------------------
    if isempty(mLE) || isempty(mNEE) || isempty(mRSM)
        warning('CoLM:EmptyOutput', 'One or more model outputs are empty.');
        error('CoLM:Abort', 'Stopping due to empty model output.');
    end
    if any(~isfinite(mLE)) || any(~isfinite(mNEE)) || any(~isfinite(mRSM))
        warning('CoLM:NonFiniteOutput', 'Model output contains NaN/Inf.');
        error('CoLM:Abort', 'Stopping due to non-finite values in model output.');
    end
    if ~isequal(numel(mLE), numel(y_LE)) || ~isequal(numel(mNEE), numel(y_NEE)) || ~isequal(numel(mRSM), numel(y_RSM))
        warning('CoLM:LengthMismatch', 'Model output length mismatch.');
        error('CoLM:Abort', 'Stopping due to output length mismatch.');
    end

    % ---------------- Build per-series options -------------
    optLE  = build_series_opts(data, 'LE');
    optNEE = build_series_opts(data, 'NEE');
    optRSM = build_series_opts(data, 'RSM');

    % ---------------- Per-series robust log-likelihood -----
    logL_LE  = loglik_series(y_LE,  mLE,  optLE);
    logL_NEE = loglik_series(y_NEE, mNEE, optNEE);
    logL_RSM = loglik_series(y_RSM, mRSM, optRSM);

    % ---------------- Sum joint log-likelihood -------------
    log_L = logL_LE + logL_NEE + logL_RSM;
end

% ======================================================================
% Helpers (pure / parfor-safe)
% ======================================================================

function opt = build_series_opts(data, tag)
% Assemble options for a series tag ('LE'/'NEE'/'RSM') with safe defaults.
    if isfield(data, tag), S = data.(tag); else, S = struct(); end

    % Transform
    if ~isfield(S,'transform'),  S.transform = 'none'; end
    switch lower(S.transform)
        case 'none'
            S.addJacobian = falseIfMissing(S, 'addJacobian', false);
        case 'boxcox'
            requireField(S, 'lambda', ['lambda required for ', tag, ' boxcox']);
            S.addJacobian = falseIfMissing(S, 'addJacobian', true);
        case 'boxcox_shift'
            requireField(S, 'lambda', ['lambda required for ', tag, ' boxcox_shift']);
            if ~isfield(S, 'shift'), S.shift = 'auto'; end % 'auto' or numeric
            S.addJacobian = falseIfMissing(S, 'addJacobian', true);
        case 'yeojohnson'
            requireField(S, 'lambda', ['lambda required for ', tag, ' yeojohnson']);
            S.addJacobian = falseIfMissing(S, 'addJacobian', true);
        otherwise
            error('Unknown transform type for %s: %s', tag, S.transform);
    end

    % AR(1)
    if ~isfield(S,'phi'), S.phi = 0.0; end
    if ~isfield(S,'includeInit'), S.includeInit = false; end

    % Distribution
    if ~isfield(S,'distribution'), S.distribution = 'gaussian'; end
    if strcmpi(S.distribution,'studentt') && ~isfield(S,'nu'), S.nu = 5; end

    % Heteroskedasticity and inflation
    if ~isfield(S,'heteroA'),  S.heteroA  = 0.0; end
    if ~isfield(S,'inflation'), S.inflation = 1.0; end
    S.inflation = max(S.inflation, 1.0);

    opt = S;
end

function requireField(S, fname, msg)
    if ~isfield(S, fname)
        error('%s', msg);
    end
end

function val = falseIfMissing(S, fname, defaultVal)
    if isfield(S, fname), val = S.(fname); else, val = defaultVal; end
end

function logL = loglik_series(y_obs, y_sim, optS)
% Robust per-series likelihood with transform (none/boxcox/boxcox_shift/yeojohnson),
% AR(1), heteroskedasticity, and Gaussian/Student-t innovations (concentrated scale).

    N = numel(y_obs);
    if N < 2, logL = -Inf; return; end

    % ---------- Transform & Jacobian (on observations only) ----------
    switch lower(optS.transform)
        case 'none'
            Z_obs = y_obs;  Z_sim = y_sim;  jac_term = 0;

        case 'boxcox'
            % requires positivity
            if any(y_obs <= 0) || any(y_sim <= 0), logL = -Inf; return; end
            [Z_obs, Z_sim, jac_term] = transform_boxcox(y_obs, y_sim, optS.lambda, optS.addJacobian);

        case 'boxcox_shift'
            % ensure positivity via shift c > 0 on both obs and sim
            [Z_obs, Z_sim, jac_term] = transform_boxcox_shift(y_obs, y_sim, optS.lambda, optS);

        case 'yeojohnson'
            [Z_obs, Z_sim, jac_term] = transform_yeojohnson(y_obs, y_sim, optS.lambda, optS.addJacobian);

        otherwise
            error('Unknown transform: %s', optS.transform);
    end

    % ---------- AR(1) innovations ----------
    phi = optS.phi;
    eta = Z_obs - Z_sim;                   % residuals
    W   = eta(2:end) - phi*eta(1:end-1);   % innovations, length N-1

    % ---------- Heteroskedastic factor and concentrated base variance ----------
    a = max(optS.heteroA, 0);
    if a > 0
        g_full = (1 + a*abs(y_sim)).^2;    % use ORIGINAL scale by design
    else
        g_full = ones(N,1);
    end
    g = g_full(2:end);                      % align with innovations

    % Concentrated/weighted scale estimate
    s2 = mean( (W.^2) ./ max(g, realmin) );
    s2 = max(s2, realmin) * (optS.inflation^2);

    % ---------- Initial-state term (optional, Gaussian stationary) ----------
    if optS.includeInit && abs(phi) < 1
        var_eta1 = s2 / max(1 - phi^2, realmin);
        init_term = -0.5*( log(2*pi*var_eta1) + (eta(1)^2)/var_eta1 );
    else
        init_term = 0;
    end

    % ---------- Innovations likelihood ----------
    switch lower(optS.distribution)
        case 'gaussian'
            S2 = s2 * g;
            term = -0.5*sum( log(2*pi*S2) + (W.^2)./S2 );

        case 'studentt'
            nu = optS.nu;
            S2 = s2 * g;
            term = -0.5*(nu+1)*sum( log1p( (W.^2) ./ max(nu*S2, realmin) ) ) ...
                   -0.5*sum( log( max(S2, realmin) ) );

        otherwise
            error('Unknown distribution: %s', optS.distribution);
    end

    % ---------- Combine ----------
    logL = init_term + term + jac_term;
end

% ------------------ Transforms (+ Jacobians) ------------------

function [Z_obs, Z_sim, jac_term] = transform_boxcox(y_obs, y_sim, lambda, addJac)
% Box-Cox for strictly positive data; Jacobian = sum((lambda-1) * log(y_obs))
    Z_obs = boxcox_apply(y_obs, lambda);
    Z_sim = boxcox_apply(y_sim, lambda);
    if addJac
        jac_term = sum( (lambda - 1) * log(y_obs) );
    else
        jac_term = 0;
    end
end

function [Z_obs, Z_sim, jac_term] = transform_boxcox_shift(y_obs, y_sim, lambda, optS)
% Shifted Box-Cox: ensure y + c > 0 on both obs and sim, then Box-Cox.
% If optS.shift == 'auto', choose c = max(eps, -(min(obs,sim)) + eps).
    if ischar(optS.shift) && strcmpi(optS.shift,'auto')
        y_min = min( min(y_obs), min(y_sim) );
        % small safety margin based on dynamic range
        dr = max(1.0, max([abs(y_obs); abs(y_sim)]));
        c  = max(1e-6*dr, -y_min + 1e-6*dr);
    else
        c = optS.shift;
        if ~(isscalar(c) && isfinite(c) && c > 0)
            error('boxcox_shift requires a positive scalar "shift".');
        end
    end
    yo = y_obs + c; ys = y_sim + c;
    if any(yo <= 0) || any(ys <= 0)
        % should not happen; guard anyway
        jac_term = -Inf; Z_obs = yo; Z_sim = ys; return;
    end
    Z_obs = boxcox_apply(yo, lambda);
    Z_sim = boxcox_apply(ys, lambda);
    if optS.addJacobian
        jac_term = sum( (lambda - 1) * log(yo) );  % Jacobian on observations only
    else
        jac_term = 0;
    end
end

function [Z_obs, Z_sim, jac_term] = transform_yeojohnson(y_obs, y_sim, lambda, addJac)
% Yeo–Johnson for real-valued data; Jacobian is piecewise:
%  dz/dy = (1+y)^(lambda-1) for y>=0; (1-y)^(1-lambda) for y<0
    Z_obs = yeojohnson_apply(y_obs, lambda);
    Z_sim = yeojohnson_apply(y_sim, lambda);
    if addJac
        mask_pos = (y_obs >= 0);
        mask_neg = ~mask_pos;
        jac_term = 0;
        if any(mask_pos)
            jac_term = jac_term + sum( (lambda - 1) * log(1 + y_obs(mask_pos)) );
        end
        if any(mask_neg)
            jac_term = jac_term + sum( (1 - lambda) * log(1 - y_obs(mask_neg)) );
        end
    else
        jac_term = 0;
    end
end

function z = boxcox_apply(y, lambda)
% Strictly positive y expected.
    if lambda ~= 0
        z = ((y.^lambda) - 1) / lambda;
    else
        z = log(y);
    end
end

function z = yeojohnson_apply(y, lambda)
% Vectorized Yeo–Johnson transform for real-valued y.
    z = zeros(size(y));
    pos = (y >= 0);
    neg = ~pos;

    if any(pos)
        yp = y(pos);
        if lambda ~= 0
            z(pos) = ((yp + 1).^lambda - 1) / lambda;
        else
            z(pos) = log(yp + 1);
        end
    end
    if any(neg)
        yn = y(neg);
        if lambda ~= 2
            z(neg) = - ( (1 - yn).^(2 - lambda) - 1 ) / (2 - lambda);
        else
            z(neg) = - log(1 - yn);
        end
    end
end

% ------------------ IO helpers ------------------

function v = read_vector_robust(p)
% Read a text file as a single numeric column (NaN/Inf tokens preserved).
    if ~isfile(p), v = []; return; end
    lines   = readlines(p);
    trimmed = strtrim(lines);
    emptyMask = (strlength(trimmed) == 0);
    trimmed   = trimmed(~emptyMask);
    v = str2double(trimmed);
    v = v(:);
end

function safeRmdir(p)
% Remove a directory tree safely, suppressing warnings if already gone.
    if exist(p,'dir')
        try, rmdir(p,'s'); catch, end
    end
end
