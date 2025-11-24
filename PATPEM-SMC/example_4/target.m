function log_L = target(x, data)
% CoLM
% Compute the joint log-likelihood for LE/NEE/RSM given parameter vector x.
% Workflow:
%   1) Write x to a unique run folder cloned from a template.
%   2) Run CoLM externally.
%   3) Read model outputs and evaluate a Gaussian *concentrated* likelihood
%      (per variable v, plug in sig2_v = SSE_v / T at each x).
%
% Robustness:
%   - If any model output is empty, contains NaN/Inf, or length-mismatched
%     with observations, this function emits a WARNING and then throws an
%     ERROR (intentional hard stop).
%
% Inputs
%   x       : [1xP] or [Px1] parameter vector
%   data    : struct with fields
%               observed_LE, observed_NEE, observed_RSM   (column vectors)
%               old_path (template directory), baseDir (parent for unique runs)
%
% Output
%   log_L   : scalar joint log-likelihood
%
% Notes
%   - Concentrated likelihood includes constants: for variable v,
%       logL_v = -0.5*T*(log(2*pi) + log(SSE_v/T) + 1)
%     Constants cancel in MH/SMC ratios within the same model/length T,
%     but matter for absolute evidence comparisons across models/lengths.

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
    fp = fullfile(new_path, 'Arou', 'input_step_S1.txt');
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

    % ---------------- Read model outputs (robust single-column read) -----
    fLE  = fullfile(new_path, 'Arou', 'output_LE.txt');
    fNEE = fullfile(new_path, 'Arou', 'output_NEE.txt');   % case-correct path
    fRSM = fullfile(new_path, 'Arou', 'output_RSM.txt');

    mLE  = read_vector_robust(fLE);
    mNEE = read_vector_robust(fNEE);
    mRSM = read_vector_robust(fRSM);

    % ---------------- Early rejects: empty / NaN / length mismatch -------
    if isempty(mLE) || isempty(mNEE) || isempty(mRSM)
        warning('CoLM:EmptyOutput', ...
            'One or more model outputs are empty. Files: LE=%s | NEE=%s | RSM=%s', fLE, fNEE, fRSM);
        error('CoLM:Abort', 'Stopping due to empty model output.');
    end

    if any(~isfinite(mLE)) || any(~isfinite(mNEE)) || any(~isfinite(mRSM))
        % Any NaN/Inf present in model outputs -> abort
        warning('CoLM:NonFiniteOutput', ...
            'Model output contains NaN/Inf (LE/NEE/RSM). Files: LE=%s | NEE=%s | RSM=%s', fLE, fNEE, fRSM);
        error('CoLM:Abort', 'Stopping due to non-finite values in model output.');
    end

    if numel(mLE) ~= numel(y_LE) || numel(mNEE) ~= numel(y_NEE) || numel(mRSM) ~= numel(y_RSM)
        % Require exact length match (no truncation/alignment)
        warning('CoLM:LengthMismatch', ...
            'Model output length mismatch. LE %d!=%d | NEE %d!=%d | RSM %d!=%d', ...
            numel(mLE), numel(y_LE), numel(mNEE), numel(y_NEE), numel(mRSM), numel(y_RSM));
        error('CoLM:Abort', 'Stopping due to output length mismatch.');
    end

    % ---------------- Concentrated Gaussian log-likelihood ----------------
    % Use concentrated likelihood: at each x set sig2_v = SSE_v / T and plug into Gaussian.
    % Numerical safety: floor sig2_v at realmin to avoid log(0) or division by zero.
    T = numel(y_LE);  % lengths already checked equal for LE/NEE/RSM

    % LE
    SSE_LE   = sum((mLE  - y_LE ).^2);
    sig2_LE  = max(SSE_LE / T, realmin);
    logL_LE  = -0.5*T*log(2*pi*sig2_LE) - SSE_LE/(2*sig2_LE);

    % NEE
    SSE_NEE  = sum((mNEE - y_NEE).^2);
    sig2_NEE = max(SSE_NEE / T, realmin);
    logL_NEE = -0.5*T*log(2*pi*sig2_NEE) - SSE_NEE/(2*sig2_NEE);

    % RSM
    SSE_RSM  = sum((mRSM - y_RSM).^2);
    sig2_RSM = max(SSE_RSM / T, realmin);
    logL_RSM = -0.5*T*log(2*pi*sig2_RSM) - SSE_RSM/(2*sig2_RSM);

    % ---------------- Sum joint log-likelihood ----------------
    log_L = logL_LE + logL_NEE + logL_RSM;
end


function v = read_vector_robust(p)
% Read a text file as a single numeric column:
% - Trims leading/trailing whitespace/tabs.
% - Keeps genuine NaN/Inf tokens (e.g., "NaN", "Inf").
% - Drops empty lines (including trailing newline-only last line).
% - Returns a column vector (double). If file missing, returns [].
    if ~isfile(p), v = []; return; end
    lines   = readlines(p);
    trimmed = strtrim(lines);

    % Empty lines (length==0) are artifacts; drop them.
    emptyMask = (strlength(trimmed) == 0);
    trimmed   = trimmed(~emptyMask);

    % Convert; "NaN"/"Inf" remain as NaN/Inf.
    v = str2double(trimmed);
    v = v(:);
end


function safeRmdir(p)
% Remove a directory tree safely, suppressing warnings if already gone.
    if exist(p,'dir')
        try, rmdir(p,'s'); catch, end
    end
end
