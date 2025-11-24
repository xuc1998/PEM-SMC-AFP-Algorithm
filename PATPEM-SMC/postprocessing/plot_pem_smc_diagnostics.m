function fig = plot_pem_smc_diagnostics(out, opts, Np)
%PLOT_PEM_SMC_DIAGNOSTICS Diagnostic plots:
%   (1) beta schedule
%   (2) ESS with (possibly dynamic) threshold
%   (3) rCESS (= CESS / Np)
%   (4) MH acceptance rates for enabled moves (ARM / XOVER / DEMH)
%   (5) resampling events
%   (6) cumulative log-evidence
%
% Usage:
%   fig = plot_pem_smc_diagnostics(out, opts, Np)
%
% Notes:
%   - If opts or fields are missing, sensible defaults are used.
%   - rCESS is computed as out.CESS / Np (if CESS is present).
%   - Acceptance panel will only plot curves that exist and are not all-NaN.
%   - X-axes use thinned stage indices (1..K), then labeled with thinning.

if nargin < 2 || isempty(opts), opts = struct(); end
if nargin < 3 || isempty(Np),   Np   = [];      end

% ESS threshold (only meaningful if Policy='ESS', but plotting it is harmless)
if ~isfield(opts,'Resample') || ~isfield(opts.Resample,'ESSalpha')
    essThr = 0.7;
else
    essThr = opts.Resample.ESSalpha;
end

% Thinning (for x-label only) and true total stages (if available)
thin  = 1;
if isfield(out,'Thinning') && ~isempty(out.Thinning)
    thin = out.Thinning;
elseif isfield(opts,'Thinning') && ~isempty(opts.Thinning)
    thin = opts.Thinning;
end
Sfull = [];
if isfield(out,'S_full') && ~isempty(out.S_full)
    Sfull = out.S_full;
end
xLabelStr = sprintf('Stage s (thinning=%d)', thin);

% --- Decide how many thinned frames to PLOT ---
K_all  = numel(out.beta);      % stored
K_plot = K_all;                % default: plot all
% If the final "beta=1" frame was added only for completeness (i.e., S_full not on stride),
% drop it for plotting: show only strict stride frames.
if ~isempty(Sfull) && thin >= 1
    if mod(Sfull-1, thin) ~= 0 && K_all >= 2
        K_plot = K_all - 1;    % drop tail frame
    end
end

% Stages (thinned indices, 1..K_plot)
stages = 1:K_plot;

% Pull diagnostics (truncate to K_plot)
beta     = safe_row(out,'beta',K_plot);
ESS      = safe_row(out,'ESS',K_plot);
CESS     = safe_row(out,'CESS',K_plot);
rCESS    = CESS / max(1,Np); rCESS(~isfinite(rCESS)) = NaN;

acc_arm   = safe_row(out,'acc_arm',   K_plot);
acc_xover = safe_row(out,'acc_xover', K_plot);
acc_de    = safe_row(out,'acc_de',    K_plot);

resamp  = safe_row(out,'resampled', K_plot);
logZc   = safe_row(out,'logZ_cum',  K_plot);

% ---- plotting
fig = figure('Name','Diagnostics','Color','w','Position',[100,100,800,700]);
tiledlayout(3,2,'TileSpacing','compact','Padding','compact');
set(fig, 'DefaultAxesFontSize', 14, ...
         'DefaultTextFontSize', 14, ...
         'DefaultAxesTitleFontSizeMultiplier', 1.2, ...
         'DefaultAxesLabelFontSizeMultiplier', 1);

% (1) beta schedule
nexttile;
plot(stages, beta, 'LineWidth', 1.5);
ylabel('\beta','Interpreter','tex'); xlabel(xLabelStr);
title('\beta schedule','Interpreter','tex'); grid on;
xlim([1,K_plot]); ylim([0,1]);

% (2) ESS with threshold (dynamic if Policy='ESS' and Tempering provided)
nexttile;
plot(stages, ESS, '-', 'LineWidth', 1.5); hold on; grid on;

% Only draw threshold if we know Np
drawThr = ~isempty(Np);

thrSeries = [];          % ESS threshold per stage (absolute, i.e., in ESS units)
legText   = '';          % legend text to describe the threshold

if drawThr && isfield(opts,'Resample') && isfield(opts.Resample,'Policy') ...
        && strcmpi(opts.Resample.Policy,'ESS')

    % -------- alpha factor: auto=0.90 if ESSalpha is [], else user-provided --------
    if ~isfield(opts.Resample,'ESSalpha') || isempty(opts.Resample.ESSalpha)
        alpha = 0.90;  % matches main algorithm's auto mode
    else
        alpha = double(opts.Resample.ESSalpha);
    end
    alpha = min(0.999, max(0.05, alpha));   % clamp for safety

    % -------- base target band (fallback if missing) --------
    band0 = [0.70 0.85];
    if isfield(opts,'Tempering') && isfield(opts.Tempering,'TargetBand') ...
            && ~isempty(opts.Tempering.TargetBand)
        band0 = sort(opts.Tempering.TargetBand(:)).';
    end

    % -------- build min(current band) for each plotted stage (respect BandDecay) --------
    useDecay = isfield(opts,'Tempering') && isfield(opts.Tempering,'BandDecay') ...
               && isfield(opts.Tempering.BandDecay,'Enabled') ...
               && opts.Tempering.BandDecay.Enabled;

    % defaults for BandDecay
    bt = 0.95;                           % BetaThresh default
    floorBand = [0.70 0.85];             % Floor default
    if useDecay
        if isfield(opts.Tempering.BandDecay,'BetaThresh') && ~isempty(opts.Tempering.BandDecay.BetaThresh)
            bt = double(opts.Tempering.BandDecay.BetaThresh);
        end
        if isfield(opts.Tempering.BandDecay,'Floor') && ~isempty(opts.Tempering.BandDecay.Floor)
            floorBand = sort(opts.Tempering.BandDecay.Floor(:)).';
        end
    end

    % compute tau_low_now(k) = min(current band at stage k)
    band_low = zeros(1, K_plot);
    beta_plot = beta(1:K_plot);
    for k = 1:K_plot
        b = band0;
        if useDecay && (beta_plot(k) > bt)
            tdec = min(1, (beta_plot(k) - bt) / max(1e-12, 1 - bt));
            b = (1 - tdec) * b + tdec * floorBand;   % linear interp TargetBand -> Floor
        end
        band_low(k) = max(0.05, b(1));
    end

    % -------- ESS threshold series: thr_s = alpha * min(current band_s) * Np --------
    thrSeries = alpha * band_low * Np;

    % draw the series and prepare numeric legend text
    plot(stages, thrSeries, '--', 'LineWidth', 1.2);

    coeff_series = thrSeries / max(1, Np);     % per-stage coefficient
    cmin = min(coeff_series);  cmax = max(coeff_series);
    if (cmax - cmin) < 1e-3
        legText = sprintf('thr = %.2f \\times N_p', mean(coeff_series));
    else
        legText = sprintf('thr = [%.2f–%.2f] \\times N_p', cmin, cmax);
    end

    legend({'ESS', legText}, 'Location','northeast','Interpreter','tex');

elseif drawThr && isfield(opts,'Resample') && isfield(opts.Resample,'ESSalpha') ...
        && ~isempty(opts.Resample.ESSalpha)
    % Fallback: if Policy is not 'ESS' or band info is unavailable,
    % show a constant line at alpha * Np (legacy behavior).
%     yline(opts.Resample.ESSalpha * Np, '--');
%     legend('ESS', sprintf('%.0f%% of N_p', 100*opts.Resample.ESSalpha), 'Location','best');
end

xlabel(xLabelStr); ylabel('ESS'); title('ESS (pre-resample)');
xlim([1, K_plot]); if ~isempty(Np), ylim([1, Np]); end


% (3) rCESS
nexttile;
plot(stages, rCESS, '-', 'LineWidth', 1.5); grid on;
xlabel(xLabelStr); ylabel('rCESS'); title('Relative CESS = CESS/N_p');
xlim([1,K_plot]); ylim([0,1]);

% (4) acceptance (ARM / XOVER / DEMH) — only plot those with data
nexttile;
hold on; grid on;
legend_entries = {};
h = [];

if any(isfinite(acc_arm))
    h(end+1) = plot(stages, acc_arm, '-', 'LineWidth', 1.2);
    legend_entries{end+1} = 'ARM';
end
if any(isfinite(acc_xover))
    h(end+1) = plot(stages, acc_xover, '-', 'LineWidth', 1.2);
    legend_entries{end+1} = 'XOVER';
end
if any(isfinite(acc_de))
    h(end+1) = plot(stages, acc_de, '-', 'LineWidth', 1.2);
    legend_entries{end+1} = 'DE-MH';
end

xlabel(xLabelStr); ylabel('acceptance');
title('MH acceptance rates');
if ~isempty(h)
    legend(h, legend_entries, 'Location','best','Orientation','horizontal');
end
xlim([1,K_plot]); ylim([0,1]);

% (5) resampling events
nexttile;
stem(stages, double(resamp), 'filled'); grid on;
xlabel(xLabelStr); ylabel('resampled?');
title('Resampling (1 = yes)'); ylim([0,1.1]); xlim([1,K_plot]);

% (6) logZ cumulative
nexttile;
plot(stages, logZc, '-', 'LineWidth', 1.5); grid on;
xlabel(xLabelStr); ylabel('log Z (cum)'); title('Cumulative log-evidence');
xlim([1,K_plot]);

end

function v = safe_row(out, field, K_plot)
    if isfield(out, field) && ~isempty(out.(field))
        v = out.(field)(1:K_plot);
        v = v(:).';
    else
        v = nan(1,K_plot);
    end
end
