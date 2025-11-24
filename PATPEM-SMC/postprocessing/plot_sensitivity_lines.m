function fig = plot_sensitivity_lines(Np_vals, S_vals, cess_targets, stats, default_cfg)
%PLOT_SENSITIVITY_LINES  Line charts with shaded bands (mean ± SD).
%
% PURPOSE
%   Visualize how the discrepancy D_s and the 2-Wasserstein distance W_2 vary
%   with particle counts N_p across several adaptive stage counts S.
%   Each S is shown as a mean curve with a semi-transparent band = mean ± SD.
%   (Compared to error bars, shaded bands avoid visual clutter and do not
%   require horizontal offsets when curves share identical x-locations.)
%
% USAGE
%   fig = plot_sensitivity_lines(Np_vals, S_vals, cess_targets, stats, default_cfg)
%
% INPUTS
%   Np_vals      [1 x nX] numeric   : particle counts (ascending).
%   S_vals       [1 x nY] numeric   : adaptive stage counts (ascending).
%   cess_targets [1 x nY] numeric   : target CESS values aligned with S_vals.
%   stats  struct with fields (nY x nX matrices):
%          - Ds_mean, Ds_sd, W2_mean, W2_sd (mean/SD across seeds).
%   default_cfg struct (optional) used to mark a reference point:
%          - .Np, .S (values highlighted with a star and legend label 'Default').
%
% OUTPUT
%   fig : figure handle.

    nX = numel(Np_vals);
    nY = numel(S_vals);

    % ----- Figure & layout -----
    fig = figure('Color','w','Name','PEM-SMC-FP Sensitivity (lines)', ...
                 'Units','pixels','Position',[100 100 1200 600]);
    tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
    set(fig,'DefaultAxesFontSize',12,'DefaultTextFontSize',12);

    % Distinct colors for each S; markers for the mean curve
    cmap = lines(nY);
    mrks = {'o','s','^','d','v','>','<'};

    % Legend entries show S and its associated CESS target
    lgdStr = arrayfun(@(s,c) sprintf('S=%d (CESS=%.2f)', s, c), ...
                      S_vals(:), cess_targets(:), 'UniformOutput', false);

    % ===================== (a) D_s =====================
    ax1 = nexttile; hold(ax1,'on'); grid(ax1,'on'); box(ax1,'on');

    % Plot each S as a shaded band (mean±SD) plus a mean line on top
    for iy = 1:nY
        x  = Np_vals;                        % no horizontal dodging needed
        mu = stats.Ds_mean(iy,:);
        sd = stats.Ds_sd(iy,:);
        shaded_mean_sd(ax1, x, mu, sd, cmap(iy,:), lgdStr{iy}, mrks{mod(iy-1,numel(mrks))+1});
    end

    xlabel(ax1,'$N_p$','Interpret','latex');
    ylabel(ax1,'$D_s$','Interpret','latex');
    title(ax1,'$D_s$ across [$N_p$, S]','Interpret','latex');
    xticks(ax1, Np_vals);

    % Small padding on x for aesthetics
    xpad = 0.02*max(range(Np_vals),1);
    xlim(ax1, [min(Np_vals)-xpad, max(Np_vals)+xpad]);
    text(ax1, 0.02, 0.97, '(a)', 'Units','normalized', 'FontWeight','bold');

    % Mark the default configuration (if provided) with a star
    if ~isempty(default_cfg) && isfield(default_cfg,'Np') && isfield(default_cfg,'S')
        [~, jx] = min(abs(Np_vals - default_cfg.Np));
        [~, jy] = min(abs(S_vals  - default_cfg.S));
        plot(ax1, Np_vals(jx), stats.Ds_mean(jy,jx), ...
            'p', 'MarkerSize', 8, 'MarkerFaceColor','k', 'Color','k', ...
            'LineWidth',0.5, 'DisplayName','Default');
    end
    leg1 = legend(ax1,'Location','northeast'); leg1.Box = 'off';

    % Y-limits based on (mean ± SD) across all series, with small padding
    Ds_hi = stats.Ds_mean + stats.Ds_sd;
    Ds_lo = stats.Ds_mean - stats.Ds_sd;
    lo = min(Ds_lo(:), [], 'omitnan'); hi = max(Ds_hi(:), [], 'omitnan');
    pad = 0.06 * max(hi-lo, eps);
    ylim(ax1, [lo-pad, hi+pad]);

    % ===================== (b) W_2 =====================
    ax2 = nexttile; hold(ax2,'on'); grid(ax2,'on'); box(ax2,'on');

    for iy = 1:nY
        x  = Np_vals;
        mu = stats.W2_mean(iy,:);
        sd = stats.W2_sd(iy,:);
        shaded_mean_sd(ax2, x, mu, sd, cmap(iy,:), lgdStr{iy}, mrks{mod(iy-1,numel(mrks))+1});
    end

    xlabel(ax2,'$N_p$','Interpret','latex');
    ylabel(ax2,'$W_2$','Interpret','latex');
    title(ax2,'$W_2$ across [$N_p$, S]','Interpret','latex');
    xticks(ax2, Np_vals);
    xlim(ax2, [min(Np_vals)-xpad, max(Np_vals)+xpad]);
    text(ax2, 0.02, 0.97, '(b)', 'Units','normalized', 'FontWeight','bold');

    % Default star on W2 panel (same coordinates in (Np,S) space)
    if exist('jx','var') && exist('jy','var')
        plot(ax2, Np_vals(jx), stats.W2_mean(jy,jx), ...
            'p', 'MarkerSize', 8, 'MarkerFaceColor','k', 'Color','k', ...
            'LineWidth',0.5, 'DisplayName','Default');
    end
    leg2 = legend(ax2,'Location','northeast'); leg2.Box = 'off';

    % Y-limits based on (mean ± SD)
    W2_hi = stats.W2_mean + stats.W2_sd;
    W2_lo = stats.W2_mean - stats.W2_sd;
    lo = min(W2_lo(:), [], 'omitnan'); hi = max(W2_hi(:), [], 'omitnan');
    pad = 0.06 * max(hi-lo, eps);
    ylim(ax2, [lo-pad, hi+pad]);
end

% -------------------------------------------------------------------------
% Helper: draw a semi-transparent band for mean±SD and a mean line on top.
% - The band is created with PATCH for speed and control; it is excluded
%   from the legend (HandleVisibility='off') so only the mean curve appears
%   in the legend for each S.
% - FaceAlpha controls the transparency of the band. Increase it if you
%   want a more opaque interval; decrease it if curves overlap heavily.
% - Markers are used on the mean line only for readability at sparse x-axes.
% -------------------------------------------------------------------------
function hLine = shaded_mean_sd(ax, x, mu, sd, color, label, marker)
    up = mu + sd;
    lo = mu - sd;
    X  = [x, fliplr(x)];             % polygon x-coordinates (upper then lower)
    Y  = [up, fliplr(lo)];           % polygon y-coordinates

    % Draw the shaded interval. It does not show in the legend.
    patch('Parent',ax, 'XData',X, 'YData',Y, ...
          'FaceColor',color, 'FaceAlpha',0.28, ...
          'EdgeColor','none', 'HandleVisibility','off');

    % Mean line on top of the band (legend item)
    hLine = plot(ax, x, mu, '-', ...
                 'Color', color, 'LineWidth', 1.8, ...
                 'Marker', marker, 'MarkerFaceColor','w', ...
                 'MarkerSize', 5, 'DisplayName', label);
end
