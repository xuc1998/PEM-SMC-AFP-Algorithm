function log_L = target(x, data)
% CoLM
% Compute the joint log-likelihood for LE/NEE/RSM given parameter vector x.
% - Writes x to a unique run folder cloned from a template, runs CoLM,
%   reads model outputs, and evaluates the Gaussian log-likelihood with
%   FIXED (known) noise std devs from plugin.sigma_*.
% - Robust I/O: if any model output contains NaN, is empty, or lengths do
%   not match observations, returns log_L = -Inf.  (Modified per request:
%   now emits a WARNING and then throws an ERROR to stop the program.)
%
% Inputs
%   x       : [1xP] or [Px1] parameter vector
%   data    : struct with fields
%               observed_LE, observed_NEE, observed_RSM   (column vectors)
%               sigma_LE,    sigma_NEE,    sigma_RSM      (scalars, fixed)
%               old_path (template dir), baseDir (parent for unique runs)
%
% Output
%   log_L   : scalar joint log-likelihood

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
    fNEE = fullfile(new_path, 'Arou', 'output_NEE.txt');
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
        % Require exact length match (no truncation/alignment as requested)
        warning('CoLM:LengthMismatch', ...
            'Model output length mismatch. LE %d!=%d | NEE %d!=%d | RSM %d!=%d', ...
            numel(mLE), numel(y_LE), numel(mNEE), numel(y_NEE), numel(mRSM), numel(y_RSM));
        error('CoLM:Abort', 'Stopping due to output length mismatch.');
    end

    % ---------------- Fixed-sigma Gaussian log-likelihood ----------------
    sLE  = data.sigma_LE;    % fixed std dev (same as used to create synthetic obs)
    sNEE = data.sigma_NEE;
    sRSM = data.sigma_RSM;

    % LE
    resLE   = mLE  - y_LE;
    TLE     = numel(resLE);
    logL_LE = -0.5*TLE*log(2*pi*sLE^2) - (resLE.'*resLE)/(2*sLE^2);

    % NEE
    resNEE   = mNEE - y_NEE;
    TNEE     = numel(resNEE);
    logL_NEE = -0.5*TNEE*log(2*pi*sNEE^2) - (resNEE.'*resNEE)/(2*sNEE^2);

    % RSM  (note: uses observed_RSM, i.e., column 3)
    resRSM   = mRSM - y_RSM;
    TRSM     = numel(resRSM);
    logL_RSM = -0.5*TRSM*log(2*pi*sRSM^2) - (resRSM.'*resRSM)/(2*sRSM^2);

    % ---------------- Sum joint log-likelihood ----------------
    log_L = logL_LE + logL_NEE + logL_RSM;
end


function v = read_vector_robust(p)
% Read a text file as a single numeric column:
% - trims leading/trailing whitespace/tabs
% - keeps genuine NaN/Inf tokens (e.g., "NaN", "Inf")
% - drops empty lines (including trailing newline-only last line)
% - returns a column vector
    if ~isfile(p), v = []; return; end
    lines   = readlines(p);
    trimmed = strtrim(lines);

    % empty lines (length==0) are artifacts; drop them
    emptyMask = (strlength(trimmed) == 0);
    trimmed   = trimmed(~emptyMask);

    % convert; "NaN"/"Inf" remain as NaN/Inf
    v = str2double(trimmed);
    v = v(:);
end


function safeRmdir(p)
% Remove a directory tree safely, suppressing warnings if already gone.
    if exist(p,'dir')
        try, rmdir(p,'s'); catch, end
    end
end
