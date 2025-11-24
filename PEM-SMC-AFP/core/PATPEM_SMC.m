function [parameter_iteration, out] = PATPEM_SMC(Np, S, bound, logpdf_handle, options)
%PATPEM_SMC  Tempered SMC with (i) a robust rCESS-band adaptive schedule
%             or (ii) a fixed exponential schedule (no hyperparams exposed),
%             configurable resampling, and pluggable move operators
%             (ARM / Crossover / DE–MH), with parallelization support.
%
% Stage routine:
%   reweight (log-domain) -> (policy-triggered) resample -> MCMC moves
% Beta lives in [0,1]. For Adaptive=true, we choose dBeta so that per-stage
% rCESS falls inside a TargetBand via variance-prediction + bracketing.
% For Adaptive=false, a fixed exponential schedule with scalar=1e-6 is used.
% The model evidence log Z is accumulated using the annealed-SMC estimator.
%
% -------------------------------------------------------------------------
% Syntax
%   parameter_iteration = PATPEM_SMC(Np, S, bound)
%   [parameter_iteration, out] = PATPEM_SMC(Np, S, bound)
%   [parameter_iteration, out] = PATPEM_SMC(Np, S, bound, @logpdf, options)
%
% Inputs
%   Np            : integer, number of particles.
%   S             : integer, stage budget (upper bound if beta reaches 1 earlier).
%   bound         : 2 x d array, [LB; UB] box constraints.
%   logpdf_handle : (optional) handle to log target density log f(x).
%                   Supported signatures:
%                       logpdf(x)
%                       logpdf(x, data)
%   options       : (optional) struct of hyperparameters (missing fields get defaults).
%
% Options (relevant fields; see default_options() for full defaults)
%   options.Tempering
%     .Adaptive         logical  true => rCESS-band adaptive; false => fixed exponential
%     % --- (Adaptive path) ---
%     .TargetBand       [1x2]    rCESS target band per step, e.g., [0.70 0.85]
%     .MinDeltaBeta     scalar   absolute lower bound on dBeta (e.g., 1e-3)
%     .MaxDeltaBeta     scalar   absolute upper bound on dBeta (e.g., 0.25)
%     .GrowthFactor     scalar   cap on next/prev step: dBeta_{t+1} <= g * dBeta_t
%     .WarmupNoResample integer  forbid resampling for the first K stages
%     .BandDecay        struct   optional beta-threshold decay of TargetBand:
%                               .Enabled (bool), .BetaThresh (scalar), .Floor [low high]
%
%   options.Resample
%     .Policy     char     'ESS' | 'Periodic'                                   (default: 'ESS')
%     .ESSalpha   scalar   ESS/N threshold for 'ESS'.
%                 If empty -> auto = 0.9 * min(current band, respecting BandDecay)
%                 If provided -> treated as a factor multiplied by min(current band)
%     .Method     char     'systematic' | 'residual'                             (default: 'systematic')
%     .PeriodK    integer  for 'Periodic': resample every K stages               (default: 1)
%
%   options.Moves
%     .Sequence     cellstr enabled move names in the order to run each stage.
%                   Valid names: 'ARM', 'XOVER', 'DEMH'. Examples:
%                     {'ARM','XOVER','DEMH'}   % run all, in this order (default)
%                     {'DEMH','ARM'}           % only DE–MH then ARM
%                     {'XOVER'}                % only crossover
%                 - ARM       : Adaptive Random-Walk Metropolis
%                               with reflective box boundaries.
%                     .ARM.Cov       [d x d] ...
%                     .ARM.Jitter    ...
%                 - DEMH      : Differential-Evolution Metropolis–Hastings
%                               with reflective box boundaries.
%                     .DEMH.Gamma    ...
%                     .DEMH.NoiseSD  ...
%                 - Crossover : Single-point two-parent crossover
%                     .Crossover.pc  scalar   per-pair crossover probability
%
%   options.Parallel
%     .Enabled    logical use parfor if true                                     (default: true)
%     .NumWorkers scalar  [] lets MATLAB decide                                  (default: [])
%
%   options.Thinning : integer >=1. Thinning factor for storage/printing         (default: 1)
%   options.Data     : (optional) struct passed to logpdf_handle as 2nd arg.
%   options.PrintConfig : logical, print effective configuration once            (default: true)
%
% Outputs
%   parameter_iteration : [Np x d x K] particle states at thinned stages (includes beta=0)
%   out                 : diagnostics struct (thinned to match storage)
%       .beta        : [1 x K] schedule (beta_1 = 0)
%       .ESS         : [1 x K] ESS after reweight (pre-resample)
%       .CESS        : [1 x K] rCESS for chosen dBeta
%       .dBeta       : [1 x K] step sizes used
%       .resampled   : [1 x K] logical flags if resampling happened
%       .acc_arm     : [1 x K] acceptance rate of ARM
%       .acc_xover   : [1 x K] acceptance rate of crossover (per-individual)
%       .acc_de      : [1 x K] acceptance rate of DE–MH
%       .logZ_cum    : [1 x K] cumulative log-evidence estimate
%       .S_K         : scalar   number of thinned stages actually output (K)
%       .S_full      : total number of true stages executed
%       .timing      : struct   .elapsed_sec total wall time (seconds)
%
% Notes
% - Proposal boundary handling uses reflective mappings (external helpers).
% - Evidence increment per stage uses the standard annealed-SMC estimator in log-domain.
% - Resampling is governed by Policy ('ESS' or 'Periodic') and Method ('systematic'|'residual').
% - Move operators and their hyperparameters live under options.Moves.* and run in the order given.
%
% External helpers required on path:
%   target (optional default logpdf), ResampSys, randw, Generatep_reflect.
%
% -------------------------------------------------------------------------

t0 = tic;

%% ------------------- Defaults & inputs -------------------
d  = size(bound,2);
LB = bound(1,:);
UB = bound(2,:);

if nargin < 4 || isempty(logpdf_handle)
    logpdf_handle = @target; % fallback to a function on path
end
if nargin < 5, options = struct(); end

opts = default_options(options, d, S);

% Open/resize pool per opts.Parallel.*
[opts, poolInfo] = ensure_parallel_pool(opts);

% Stage budget
S_max = S;

% Thinning factor (for storage & progress printing)
thin = max(1, round(opts.Thinning));

% ---- Print effective hyper-parameters (once) ----
if isfield(opts,'PrintConfig') && opts.PrintConfig
    print_pem_smc_config(opts, Np, d, S, logpdf_handle);
end

% Pre-allocation for thinned storage only
Kmax = ceil(S_max / thin) + 2;
parameter_iteration = nan(Np, d, Kmax);
k_store = 1; % number of stored frames (1st = beta 0)

%% ------------------- Initialize particles & weights -------------------
theta = zeros(Np, d);
w     = (1/Np) * ones(Np,1);
for i=1:Np
    theta(i,:) = LB + (UB - LB) .* rand(1,d); % uniform in the box
end
parameter_iteration(:,:,k_store) = theta; % store beta=0 frame

% Diagnostics (kept per-stage; thinned later)
beta_vec    = zeros(1, S_max);  beta_vec(1) = 0;
ESS_vec     = nan(1, S_max);
CESS_vec    = nan(1, S_max);
dBeta_rec   = nan(1, S_max);
resampled_f = false(1, S_max);
acc_arm_v   = nan(1, S_max);
acc_xo_v    = nan(1, S_max);
acc_de_v    = nan(1, S_max);
logZ_cum_v  = nan(1, S_max);
logZ        = 0;

%% ------------------- SMC evolution -------------------
s = 1;
beta_raw = 0;                 % beta in [0,1]
beta_vec(1) = 0;

% persistent prev_dBeta_for_growth
% if isempty(prev_dBeta_for_growth), prev_dBeta_for_growth = []; end
prev_dBeta_for_growth = [];   % reset per run to ensure reproducibility


while true
    if s >= S_max, break; end

    % Log-likelihoods at current particles
    ll = zeros(Np,1);
    if opts.Parallel.Enabled
        parfor i = 1:Np
            ll(i) = call_logpdf(logpdf_handle, theta(i,:), opts.Data);
        end
    else
        for i = 1:Np
            ll(i) = call_logpdf(logpdf_handle, theta(i,:), opts.Data);
        end
    end

    
    if s == 1 && isempty(prev_dBeta_for_growth)
        % current weights are uniform at beta=0; use unweighted var as fallback
        w = (1/ Np) * ones(Np,1);
        mu = sum(w .* ll);
        v  = sum(w .* (ll - mu).^2) / max(eps, 1 - sum(w.^2));
        if ~isfinite(v) || v <= 0, v = var(ll); end
    
        % geometric mean target inside your band
        tau_mid   = sqrt(prod(sort(opts.Tempering.TargetBand(:)).'));
        dPred     = sqrt( max(0, -log(tau_mid)) / max(v, eps) );
    
        % conservative seed for the first-step cap
        dSafe = min(opts.Tempering.MaxDeltaBeta, max(opts.Tempering.MinDeltaBeta, dPred));
        prev_dBeta_for_growth = 0.8 * dSafe;   % 0.7~0.9 are fine
    end

    % ---- Choose next beta (Adaptive vs Fixed) ----
    if opts.Tempering.Adaptive
        % Adaptive: rCESS-band (variance-predicted + bracketing + guards)
        [beta_new_raw, stepinfo] = choose_beta_by_rCESS_band( ...
            w, ll, beta_raw, 1.0, opts.Tempering, prev_dBeta_for_growth, s+1);
        dBeta_raw = beta_new_raw - beta_raw;
        prev_dBeta_for_growth = dBeta_raw;
    else
        % Fixed: exponential schedule with scalar = 1e-6 
        M = 1e-6;
        beta_new_raw = 1 - (1 - M) ^ ((s+1)/S_max);
        beta_new_raw = min(1.0, max(beta_new_raw, beta_raw + 1e-12));
        dBeta_raw    = max(1e-12, beta_new_raw - beta_raw);
        % fabricate stepinfo for printing consistency (optional)
        stepinfo = struct('dBeta_used', dBeta_raw, 'rCESS_used', NaN, ...
                          'dBeta_pred', NaN, 'band', [NaN NaN]);
    end

    % ---- Evidence increment & reweight (log-domain, no scaling) ----
    w_old       = w;
    logw_propto = log(w_old) + dBeta_raw * ll;
    log_incr    = logsumexp(logw_propto);
    logZ        = logZ + log_incr;
    w           = exp(logw_propto - log_incr);

    % ---- Diagnostics: ESS / rCESS ----
    ESS_vec(s+1)  = 1 / sum(w.^2);
    log_num       = logsumexp(log(w_old) + dBeta_raw * ll);
    log_den       = logsumexp(2*log(w_old) + 2*dBeta_raw * ll);
    CESS_vec(s+1) = exp(2*log_num - log_den);
    logZ_cum_v(s+1) = logZ;
    dBeta_rec(s+1)  = dBeta_raw;

    % ---------- Resampling policy ----------
    doResample = false;
    pol = lower(char(opts.Resample.Policy));
    switch pol
        case 'periodic'
            % K=1 => every stage; K=0 => never.
            K = max(0, round(opts.Resample.PeriodK));
            if K > 0 && mod(s+1, K) == 0
                doResample = true;
            end
        case 'ess'
            % compute the *current* rCESS band (respecting BandDecay)
            band_now = sort(opts.Tempering.TargetBand(:)).';
            if isfield(opts.Tempering,'BandDecay') && isfield(opts.Tempering.BandDecay,'Enabled') && opts.Tempering.BandDecay.Enabled
                bt = 0.95;
                if isfield(opts.Tempering.BandDecay,'BetaThresh') && ~isempty(opts.Tempering.BandDecay.BetaThresh)
                    bt = double(opts.Tempering.BandDecay.BetaThresh);
                end
                floorBand = sort(opts.Tempering.BandDecay.Floor(:)).';
                if beta_raw > bt
                    tdec = min(1, (beta_raw - bt) / max(1e-12, 1 - bt));
                    band_now = (1 - tdec) * band_now + tdec * floorBand;
                end
            end
            tau_low_now = max(0.05, band_now(1));
            
            % if user provides ESSalpha, treat it as a multiplier of min(current band)
            if isempty(opts.Resample.ESSalpha)
                thr = 0.9 * tau_low_now;                   % auto mode: 0.9 × min(current band)
            else
                thr = double(opts.Resample.ESSalpha) * tau_low_now;  % user factor × min(current band)
            end
            % clamp thr to a sensible range
            thr = min(0.999, max(0.05, thr));
            if ESS_vec(s+1) < thr * Np
                doResample = true;
            end
        otherwise
            error('Unknown Resample.Policy: %s (use ''Periodic'' or ''ESS'')', opts.Resample.Policy);
    end
    % Warm-up: forbid resampling for the first K stages (helps early stability; harmless for fixed)
    if (s+1) <= max(0, round(opts.Tempering.WarmupNoResample))
        doResample = false;
    end

    if doResample
        method = 'systematic';
        if isfield(opts.Resample,'Method') && ~isempty(opts.Resample.Method)
            method = char(opts.Resample.Method);
        end
        idx   = ResampSys(w, Np, method);
        theta = theta(idx,:);
        w     = (1/Np) * ones(Np,1);
        ll    = ll(idx);
        resampled_f(s+1) = true;
    end

    % ---------- ARM covariance (weighted) ----------
    mu  = sum(theta .* w, 1);
    Xm  = theta - mu;
    den = 1 - sum(w.^2); den = max(den, eps);
    SigmaW = (Xm' * (Xm .* w)) / den;
    SigmaW = 0.5 * (SigmaW + SigmaW.');

    jit = max(opts.Moves.ARM.Jitter, 1e-8);
    Cov  = (2.38^2 / d) * (SigmaW + jit * eye(d));
    [~,p] = cholcov(Cov,0);
    tries = 0;
    while p ~= 0 && tries < 5
        jit   = jit * 10;
        Cov   = (2.38^2 / d) * (SigmaW + jit * eye(d));
        [~,p] = cholcov(Cov,0);
        tries = tries + 1;
    end
    opts.Moves.ARM.Cov = Cov;

    % ---------- Move operators (order defined by options.Moves.Enable) ----------
    for mvi = 1:numel(opts.Moves.Sequence)
        mv = upper(string(opts.Moves.Sequence{mvi}));

        switch mv
            case "ARM"
                % Adaptive random-walk Metropolis (fold/reflect boundary per opts.Moves.ARM)
                acc_flags_ARM = false(Np,1);

                if opts.Parallel.Enabled
                    theta_new = theta;           % cached
                    ll_newarr = ll;              % cached
                    parfor i = 1:Np
                        prop    = randw(theta(i,:), LB, UB, opts.Moves.ARM);
                        ll_prop = call_logpdf(logpdf_handle, prop, opts.Data);
                        ratio = min(1, exp(beta_new_raw * (ll_prop - ll(i))));
                        if rand < ratio
                            theta_new(i,:)   = prop;
                            ll_newarr(i)     = ll_prop;   % update cache
                            acc_flags_ARM(i) = true;
                        end
                    end
                    theta = theta_new;  ll = ll_newarr;
                else
                    for i = 1:Np
                        prop    = randw(theta(i,:), LB, UB, opts.Moves.ARM);
                        ll_prop = call_logpdf(logpdf_handle, prop, opts.Data);
                        ratio = min(1, exp(beta_new_raw * (ll_prop - ll(i))));
                        if rand < ratio
                            theta(i,:)       = prop;
                            ll(i)            = ll_prop;   % update cache
                            acc_flags_ARM(i) = true;
                        end
                    end
                end
                acc_arm_v(s+1) = mean(acc_flags_ARM);

            case "XOVER"
                % Single-point crossover with tempered two-body MH acceptance.
                [theta_prop, pair_idx, will_xo] = crossover_propose(theta, opts.Moves.Crossover.pc);
                P = size(pair_idx,1);
                acc2 = false(P,1);

                if P > 0
                    if opts.Parallel.Enabled
                        ll_child1 = nan(P,1); ll_child2 = nan(P,1);
                        parfor k = 1:P
                            if ~will_xo(k)
                                acc2(k) = false;
                            else
                                a = pair_idx(k,1); b = pair_idx(k,2);
                                ll_old_sum = ll(a) + ll(b);               % cached
                                ch1 = theta_prop(2*k-1,:); ch2 = theta_prop(2*k,:);
                                l1  = call_logpdf(logpdf_handle, ch1, opts.Data);
                                l2  = call_logpdf(logpdf_handle, ch2, opts.Data);
                                ll_child1(k) = l1; ll_child2(k) = l2;
                                r = min(1, exp(beta_new_raw * ((l1 + l2) - ll_old_sum)));
                                acc2(k) = (rand < r);
                            end
                        end
                        for k = 1:P
                            if acc2(k)
                                a = pair_idx(k,1); b = pair_idx(k,2);
                                theta(a,:) = theta_prop(2*k-1,:);  ll(a) = ll_child1(k);
                                theta(b,:) = theta_prop(2*k,  :);  ll(b) = ll_child2(k);
                            end
                        end
                    else
                        for k = 1:P
                            if ~will_xo(k), acc2(k)=false; continue; end
                            a = pair_idx(k,1); b = pair_idx(k,2);
                            ll_old_sum = ll(a) + ll(b);
                            ch1 = theta_prop(2*k-1,:); ch2 = theta_prop(2*k,:);
                            l1  = call_logpdf(logpdf_handle, ch1, opts.Data);
                            l2  = call_logpdf(logpdf_handle, ch2, opts.Data);
                            r = min(1, exp(beta_new_raw * ((l1 + l2) - ll_old_sum)));
                            if rand < r
                                theta(a,:) = ch1; ll(a) = l1;
                                theta(b,:) = ch2; ll(b) = l2;
                                acc2(k)    = true;
                            else
                                acc2(k)    = false;
                            end
                        end
                    end
                    acc_xo_v(s+1) = (2*sum(acc2)) / Np; % per-individual acceptance
                else
                    acc_xo_v(s+1) = 0;
                end

            case "DEMH"
                % Differential-Evolution MH
                acc_flags_de = false(Np,1);
                old_theta  = theta;
                old_ll     = ll;                     % cached
                theta_new  = old_theta;
                ll_newarr  = old_ll;

                if opts.Parallel.Enabled
                    parfor k = 1:Np
                        propDE  = Generatep_reflect(old_theta, k, LB, UB, opts.Moves.DEMH);
                        ll_prop = call_logpdf(logpdf_handle, propDE, opts.Data);
                        ratio = min(1, exp(beta_new_raw * (ll_prop - old_ll(k))));
                        if rand() < ratio
                            theta_new(k,:)  = propDE;
                            ll_newarr(k)    = ll_prop;
                            acc_flags_de(k) = true;
                        end
                    end
                else
                    for k = 1:Np
                        propDE  = Generatep_reflect(old_theta, k, LB, UB, opts.Moves.DEMH);
                        ll_prop = call_logpdf(logpdf_handle, propDE, opts.Data);
                        ratio = min(1, exp(beta_new_raw * (ll_prop - old_ll(k))));
                        if rand() < ratio
                            theta_new(k,:)  = propDE;
                            ll_newarr(k)    = ll_prop;
                            acc_flags_de(k) = true;
                        end
                    end
                end

                theta = theta_new;
                ll    = ll_newarr;
                acc_de_v(s+1) = mean(acc_flags_de);

            otherwise
                error('Unknown move "%s". Use ARM | XOVER | DEMH.', mv);
        end
    end

    % ---------- Save & progress (thinned) ----------
    beta_raw = beta_new_raw;                  % advance beta
    beta_vec(s+1) = beta_raw;                 % beta in [0,1]

    store_now = (mod(s+1, thin) == 0) || (beta_raw >= 1 - 1e-12);
    if store_now
        if k_store >= size(parameter_iteration,3)
            parameter_iteration(:,:,end+10) = nan; %#ok<AGROW>
        end
        k_store = k_store + 1;
        parameter_iteration(:,:,k_store) = theta;
    end

    if mod(s+1, thin) == 0 || beta_raw >= 1 - 1e-12
        acc_arm = acc_arm_v(s+1); if isnan(acc_arm), acc_arm = 0; end
        acc_xov = acc_xo_v(s+1);  if isnan(acc_xov), acc_xov = 0; end
        acc_de  = acc_de_v(s+1);  if isnan(acc_de),  acc_de  = 0; end
        rs_tag  = ternary(resampled_f(s+1),'Yes','No');
        fprintf('SMC stage %4d | beta=%.6f | dBeta=%.4g | rCESS=%.3f | ESS=%.1f | acc[ARM]=%.2f acc[XOVER]=%.2f acc[DE-MH]=%.2f | Resample=%s\n', ...
            s+1, beta_vec(s+1), dBeta_rec(s+1), CESS_vec(s+1), ESS_vec(s+1), acc_arm, acc_xov, acc_de, rs_tag);
    end

    if beta_raw >= 1 - 1e-12
        break;
    end

    s = s + 1;
end

% Budget exhaustion guard
hit_budget = (s >= S_max);
last_idx   = min(s, S_max);
beta_last  = beta_vec(max(1, last_idx));
if hit_budget && (beta_last < 1 - 1e-12)
    error('PATPEM_SMC:MaxStagesTooSmall', ...
        ['[PEM-SMC-AFP] Stage budget (MaxStages = %d) exhausted before reaching beta = 1. ', ...
         'Current last beta = %.6f.\n\n', ...
         'Suggestions:\n  • Increase S (stage budget), or\n', ...
         '  • For Adaptive: lower TargetBand slightly (e.g., [0.65,0.80]) or increase MinDeltaBeta.\n'], ...
         S_max, beta_last);
end

% Thinned outputs
S_used_full = min(s+1, S_max);
all_idx = 1:S_used_full;
keep_idx = all_idx( mod(all_idx-1, thin)==0 | all_idx==1 | all_idx==S_used_full );
keep_idx = unique(keep_idx);

parameter_iteration = parameter_iteration(:,:,1:k_store);

out.beta        = beta_vec(keep_idx);
out.ESS         = ESS_vec(keep_idx);
out.CESS        = CESS_vec(keep_idx);
out.dBeta       = dBeta_rec(keep_idx);
out.resampled   = resampled_f(keep_idx);
out.acc_arm     = acc_arm_v(keep_idx);
out.acc_xover   = acc_xo_v(keep_idx);
out.acc_de      = acc_de_v(keep_idx);
out.logZ_cum    = logZ_cum_v(keep_idx);
out.S_K         = numel(keep_idx);
out.S_full      = S_used_full;
out.timing.elapsed_sec = toc(t0);

close_parallel_pool_if_opened(poolInfo);
end

% =================== Defaults & merging ===================
function opts = default_options(user, d, S)
%DEFAULT_OPTIONS Build default options and merge user overrides (shallow for substructs).
opts = struct();

% Tempering (adaptive knobs; fixed schedule has no exposed knobs)
opts.Tempering = struct();
opts.Tempering.Adaptive         = true;          % false => fixed exponential with scalar=1e-6
% Adaptive path
opts.Tempering.TargetBand       = [0.70, 0.85];
opts.Tempering.MinDeltaBeta     = 1e-3;
opts.Tempering.MaxDeltaBeta     = 0.1;
opts.Tempering.GrowthFactor     = 2.0;
opts.Tempering.WarmupNoResample = 2;
opts.Tempering.BandDecay = struct( ...
    'Enabled',    false, ...      % turn band decay on/off
    'BetaThresh', 0.95,  ...      % start decaying once beta exceeds this threshold
    'Floor',      [0.70, 0.85] ...% target band to decay toward (wider -> larger Δβ)
);


% Moves: kernel set & hyperparameters
opts.Moves = struct();
opts.Moves.Sequence = {'ARM','XOVER','DEMH'};              % default order
opts.Moves.ARM = struct('Cov', 1e-3*eye(d), ...
                        'Jitter', 1e-6);
opts.Moves.DEMH = struct('Gamma',   2.38 / sqrt(2*d), ...
                         'NoiseSD', 1e-4);
opts.Moves.Crossover = struct('pc', 0.7);

% Resampling
opts.Resample = struct();
opts.Resample.Policy   = 'ESS';           % recommend ESS
opts.Resample.ESSalpha = [];              % [] => auto from TargetBand
opts.Resample.Method   = 'systematic';    % 'systematic' | 'residual'
opts.Resample.PeriodK  = 1;               % unused for ESS

% Parallel
opts.Parallel = struct();
opts.Parallel.Enabled    = true;
opts.Parallel.NumWorkers = [];

% Thinning
opts.Thinning = 1;

% Data & printing
opts.Data = struct();
opts.PrintConfig = true;

% Merge user overrides (shallow)
if nargin>=1 && ~isempty(user)
    opts = merge_options(opts, user);
end

% Sanity clamps
opts.Tempering.MinDeltaBeta = max(1e-6, double(opts.Tempering.MinDeltaBeta));
opts.Tempering.MaxDeltaBeta = max(opts.Tempering.MinDeltaBeta, double(opts.Tempering.MaxDeltaBeta));
opts.Thinning               = max(1, round(opts.Thinning));
end

function opts = merge_options(opts, user)
%MERGE_OPTIONS Shallow-merge fields; nested structs merged one level deep.
fn = fieldnames(user);
for i=1:numel(fn)
    f = fn{i};
    if isstruct(user.(f)) && isfield(opts,f) && isstruct(opts.(f))
        sfn = fieldnames(user.(f));
        for j=1:numel(sfn)
            opts.(f).(sfn{j}) = user.(f).(sfn{j});
        end
    else
        opts.(f) = user.(f);
    end
end
end

% =================== Moves helpers ===================
function [new_gen, pair_idx, will_xo] = crossover_propose(theta, pc)
%CROSSOVER_PROPOSE Single-point crossover proposals on random pairs.
% Pairs are formed by a random permutation; per pair, with prob pc we swap
% the suffix after a random crossover point. If N is odd, the last one is unchanged.
    [N, d] = size(theta);
    perm = randperm(N);
    P = floor(N/2);
    pair_idx = [perm(1:2:P*2-1).', perm(2:2:P*2).'];
    will_xo  = rand(P,1) < pc;

    new_gen = zeros(2*P, d);
    for k = 1:P
        a = pair_idx(k,1); b = pair_idx(k,2);
        if ~will_xo(k)
            new_gen(2*k-1,:) = theta(a,:);
            new_gen(2*k,  :) = theta(b,:);
        else
            cpoint = randi(d);
            new_gen(2*k-1,:) = [theta(a,1:cpoint), theta(b,cpoint+1:end)];
            new_gen(2*k,  :) = [theta(b,1:cpoint), theta(a,cpoint+1:end)];
        end
    end
end

% =================== Adaptive step chooser (rCESS band) ===================
function [beta_new, info] = choose_beta_by_rCESS_band( ...
        w, ll, beta_old, beta_max, tempopts, prev_dBeta, stage_idx)
% Choose dBeta so that per-stage rCESS falls within a target band [tau_low,tau_high].
% 1) Predict dBeta via Var_w(ll): rCESS(dBeta) ≈ exp(-Var_w(ll)*dBeta^2) ⇒
%       dBeta_pred = sqrt(-log(sqrt(tau_low*tau_high)) / Var_w(ll)).
% 2) Clamp by guards: MinDeltaBeta ≤ dBeta ≤ MaxDeltaBeta, remaining-span,
%    and dBeta ≤ GrowthFactor * previous dBeta; optional band decay over stages.
% 3) If rCESS(dBeta_pred) not in band, use bisection on [dBeta_min,dBeta_max]
%    to find the largest dBeta with rCESS ≥ tau_low, then nudge toward band.
% Output: beta_new = beta_old + dBeta; info returns dBeta_used / rCESS_used.

    
    % -------- Band (beta-threshold-only decay) --------
    band = sort(tempopts.TargetBand(:)).';   % [low, high]
    if isfield(tempopts,'BandDecay') && tempopts.BandDecay.Enabled
        % read threshold and floor band
        bt = 0.95;
        if isfield(tempopts.BandDecay,'BetaThresh') && ~isempty(tempopts.BandDecay.BetaThresh)
            bt = double(tempopts.BandDecay.BetaThresh);
        end
        floorBand = sort(tempopts.BandDecay.Floor(:)).';
    
        % decay progress t ∈ [0,1]:
        % - no decay while beta_old ≤ bt
        % - once beta_old > bt, linearly interpolate TargetBand -> Floor as beta approaches 1
        if beta_old > bt
            t = min(1, (beta_old - bt) / max(1e-12, 1 - bt));
            band = (1 - t) * band + t * floorBand;
        end
    end

    tau_low  = max(0.05, band(1));
    tau_high = min(0.999, band(2));
    tau_mid  = sqrt(tau_low * tau_high);

    % Bounds on dBeta (dynamic with remaining span)
    span      = beta_max - beta_old;
    dmin_user = max(1e-6, tempopts.MinDeltaBeta);
    dmax_user = max(dmin_user, tempopts.MaxDeltaBeta);
    dBeta_min = max(dmin_user, min(0.05, 0.5*span/10)); % tiny floor
    dBeta_max = max(dBeta_min, min(dmax_user, 0.5*span));
    if ~isempty(prev_dBeta) && isfinite(prev_dBeta) && prev_dBeta > 0
        dBeta_max = min(dBeta_max, tempopts.GrowthFactor * prev_dBeta);
    end

    % Variance-based predictor
    w = w(:); w = w / max(eps, sum(w));
    l = ll(:);
    mu = sum(w .* l);
    v  = sum(w .* (l - mu).^2) / max(eps, 1 - sum(w.^2));
    if ~isfinite(v) || v <= 0, v = var(l); end
    if ~isfinite(v) || v <= 0, v = 1; end
    dBeta_pred = sqrt( max(0, -log(tau_mid)) / max(v, eps) );
    dBeta_pred = min(dBeta_max, max(dBeta_min, dBeta_pred));

    % Accept if in-band; else bracket to maximal feasible in-band
    r_pred = rcess_value(w, l, dBeta_pred);
    if r_pred >= tau_low && r_pred <= tau_high
        dBeta = dBeta_pred;
        r_use = r_pred;
    else
        L = dBeta_min; R = dBeta_max;
        % ensure R is infeasible for tau_low
        rR = rcess_value(w,l,R);
        while rR >= tau_low && (R - L) > 1e-6*max(1,R)
            R  = 0.9*R + 0.1*L;
            rR = rcess_value(w,l,R);
        end
        % bisection to the largest dBeta with rCESS >= tau_low
        for it=1:40
            M  = 0.5*(L+R);
            rM = rcess_value(w,l,M);
            if rM >= tau_low, L = M; else, R = M; end
            if (R-L) < 1e-5*max(1.0,L), break; end
        end
        dBeta = L; r_use = rcess_value(w,l,dBeta);
        % nudge toward inside band (not exceeding R)
        if r_use > tau_high
            scale = sqrt(tau_high / max(eps, r_use));
            dBeta = min(R, max(dBeta_min, dBeta * scale));
            r_use = rcess_value(w,l,dBeta);
        end
    end

    beta_new = min(beta_max, beta_old + dBeta);
    info = struct('dBeta_used', dBeta, 'rCESS_used', r_use, ...
                  'dBeta_pred', dBeta_pred, 'band', [tau_low, tau_high]);
end

function r = rcess_value(w, l, dBeta)
% rCESS = (sum w_i e^{dBeta*l_i})^2 / sum w_i^2 e^{2 dBeta*l_i}
    a = logsumexp( log(w)   + dBeta * l );
    b = logsumexp( 2*log(w) + 2*dBeta * l );
    r = exp(2*a - b);
    if ~isfinite(r), r = 0; end
end

% =================== Utilities ===================
function y = logsumexp(x)
    m = max(x);
    if ~isfinite(m); y = m; return; end
    y = m + log(sum(exp(x - m)));
end

function lp = call_logpdf(h, x, userData)
%CALL_LOGPDF  Call logpdf handle with 1 or 2 args depending on its signature.
    nh = NaN;
    try
        nh = nargin(h);
    catch
    end
    if ~isnan(nh) && nh == 1
        lp = h(x);
    else
        lp = h(x, userData);
    end
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end

% =================== Pretty-print effective configuration ===================
function print_pem_smc_config(opts, Np, d, S_in, logpdf_handle)
    fstr = '<unavailable>';
    try
        fstr = func2str(logpdf_handle);
    catch
    end
    nin_str = '<unknown>';
    try
        nin = nargin(logpdf_handle);
        if ~isnan(nin), nin_str = num2str(nin); end
    catch
    end
    tf = @(b) char(string(logical(b)));

    fprintf('\n========== [PEM-SMC-AFP] Configuration ==========\n');
    fprintf(' Problem size   : Np=%d | d=%d | S(budget)=%d | Thinning=%d\n', ...
            Np, d, S_in, max(1, round(opts.Thinning)));

    if opts.Tempering.Adaptive
        fprintf(' Tempering      : Adaptive=TRUE | TargetBand=[%.3f, %.3f] | MinDeltaBeta=%.3g | MaxDeltaBeta=%.3g | GrowthFactor=%.3g | WarmupNoResample=%d\n', ...
            opts.Tempering.TargetBand(1), opts.Tempering.TargetBand(2), ...
            opts.Tempering.MinDeltaBeta, opts.Tempering.MaxDeltaBeta, opts.Tempering.GrowthFactor, ...
            max(0, round(opts.Tempering.WarmupNoResample)));
        if isfield(opts.Tempering,'BandDecay') && isfield(opts.Tempering.BandDecay,'Enabled') && opts.Tempering.BandDecay.Enabled
            fprintf('                : BandDecay(beta-threshold) -> BetaThresh=%.3f, Floor=[%.3f, %.3f]\n', ...
                opts.Tempering.BandDecay.BetaThresh, ...
                opts.Tempering.BandDecay.Floor(1), opts.Tempering.BandDecay.Floor(2));
        end

    else
        fprintf(' Tempering      : Adaptive=FALSE | Fixed=exponential(scalar=1e-6)\n');
    end

    fprintf(' Resampling     : Policy=%s', char(opts.Resample.Policy));
    if strcmpi(opts.Resample.Policy,'ESS')
        if isempty(opts.Resample.ESSalpha)
            fprintf(' | ESSalpha=auto(0.9*min(current band))');
        else
            fprintf(' | ESSalpha=%.3f*min(current band)', opts.Resample.ESSalpha);
        end
    elseif strcmpi(opts.Resample.Policy,'Periodic')
        Kshow = opts.Resample.PeriodK;
        fprintf(' | PeriodK=%d', Kshow);
    end
    fprintf(' | Method=%s\n', char(opts.Resample.Method));

    fprintf(' Moves (order)  : ');
    for i=1:numel(opts.Moves.Sequence)
        fprintf('%s', upper(char(opts.Moves.Sequence{i})));
        if i<numel(opts.Moves.Sequence), fprintf(' -> '); end
    end
    fprintf('\n');

    fprintf(' ARM kernel     : Boundary=reflect | Jitter=%.6g | Cov=adaptive per stage (2.38^2/d)\n', ...
        opts.Moves.ARM.Jitter);

    fprintf(' DE–MH kernel   : Boundary=reflect | Gamma=%.6g | NoiseSD=%.6g\n', ...
        opts.Moves.DEMH.Gamma, opts.Moves.DEMH.NoiseSD);


    fprintf(' XOVER kernel   : pc=%.3f (single-point)\n', opts.Moves.Crossover.pc);

    nw_str = '[]';
    if isfield(opts,'Parallel') && isfield(opts.Parallel,'NumWorkers') && ~isempty(opts.Parallel.NumWorkers)
        if isnumeric(opts.Parallel.NumWorkers) && isscalar(opts.Parallel.NumWorkers)
            nw_str = num2str(opts.Parallel.NumWorkers);
        else
            nw_str = '<var>';
        end
    end
    fprintf(' Parallel       : Enabled=%s | NumWorkers=%s\n', ...
            tf(opts.Parallel.Enabled), nw_str);

    hasData = isfield(opts,'Data') && ~isempty(opts.Data);
    fprintf(' Logpdf         : handle=%s | nargin=%s | Data provided=%s\n', ...
            fstr, nin_str, tf(hasData));
    fprintf('=================================================\n\n');
end

% =================== Parallel pool helpers ===================
function [opts, poolInfo] = ensure_parallel_pool(opts)
%ENSURE_PARALLEL_POOL  Ensure a parallel pool that matches opts.Parallel.* settings.
    poolInfo = struct('enabled', false, 'want', 0, 'maxAllowed', 0, ...
                      'openedByThisCall', false, 'profile', 'local');

    if ~isfield(opts,'Parallel') || ~isfield(opts.Parallel,'Enabled') || ~opts.Parallel.Enabled
        return;
    end

    try
        c = parcluster('local');              % requires Parallel Computing Toolbox
        poolInfo.maxAllowed = c.NumWorkers;
        poolInfo.profile    = 'local';

        if ~isfield(opts.Parallel,'NumWorkers') || isempty(opts.Parallel.NumWorkers)
            want = poolInfo.maxAllowed;       % [] → use maximum available
        else
            want = max(1, round(double(opts.Parallel.NumWorkers)));
            want = min(want, poolInfo.maxAllowed);
        end
        poolInfo.want = want;

        p = gcp('nocreate');
        if isempty(p)
            parpool(c, want);
            poolInfo.openedByThisCall = true;
        elseif p.NumWorkers ~= want
            delete(p);
            parpool(c, want);
            poolInfo.openedByThisCall = true;
        end

        poolInfo.enabled = true;

    catch ME
        warning('[ensure_parallel_pool] Parallel pool unavailable. Falling back to serial.');
        if isfield(opts,'Parallel')
            opts.Parallel.Enabled = false;
        end
        poolInfo.enabled = false;
    end
end

function close_parallel_pool_if_opened(poolInfo)
%CLOSE_PARALLEL_POOL_IF_OPENED  Close the pool only if this call opened/resized it.
    try
        if isstruct(poolInfo) && isfield(poolInfo,'openedByThisCall') && poolInfo.openedByThisCall
            p = gcp('nocreate');
            if ~isempty(p)
                delete(p);
            end
        end
    catch ME
        warning('[close_parallel_pool_if_opened] Failed to close pool');
    end
end
