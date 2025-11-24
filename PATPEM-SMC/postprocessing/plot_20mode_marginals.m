function fig = plot_20mode_marginals(samples, mu20, sigma, weights, varargin)
%PLOT_20MODE_MARGINALS  Plot 1D marginals: true mixture vs posterior KDE.
%
%   PURPOSE
%     Visualize how well posterior samples approximate the 2-D 20-mode target
%     by comparing the analytic 1D mixture marginals with KDE from (i) a
%     representative seed (solid line + filled area) and optionally
%     (ii) mean±SD band across multiple seeds. Corner text reports marginal
%     D_s (mean/scale alignment) and 1D W2 distances; both metrics are based
%     directly on samples (independent of the KDE bandwidth).
%
%   BASIC SYNTAX
%     fig = plot_20mode_marginals(S, MU20, SIGMA, W)
%
%       S     : N-by-2 posterior samples for a representative seed.
%       MU20  : 20-by-2 matrix of true mode means (Liang & Wong benchmark).
%       SIGMA : scalar component std (e.g., 0.1).
%       W     : 20-by-1 weights (sum to 1). For the benchmark: ones(20,1)/20.
%
%   NAME–VALUE OPTIONS
%     'GridN'        : integer, number of x-grid points per marginal (default 2000).
%     'KDEBandwidth' : numeric bandwidth for ksdensity; if not numeric, uses
%                      MATLAB default. Set to ~0.06–0.10 for σ=0.1 to retain peaks.
%     'FigureTitle'  : title string.
%     'SeedSamples'  : cell array {S1, S2, ...}, each N-by-2 samples for a seed.
%     'ShowSeedBand' : logical, show mean±SD band over SeedSamples (default true).
%     'Colors'       : struct with fields 'true','rep','band' (RGB triples).
%     'BandAlpha'    : transparency of the mean±SD band (default 0.15).
%
%   OUTPUT
%     fig : figure handle.
%
%   NOTES
%     • Both the true marginal curve and the KDE curve are area-normalized to 1.
%     • D_s and W2 printed in the panels are computed from the samples, not from KDE.
%     • KDE is for visualization only; its shape depends on bandwidth
%       (bias–variance tradeoff). Choose a bandwidth comparable to the true σ
%       or use cross-validated selection if desired.
%
%   EXAMPLES
%     % --- 1) Minimal: single (representative) seed ---
%     w = ones(20,1)/20;
%     fig = plot_20mode_marginals(S_rep, MU20, 0.1, w, ...
%             'KDEBandwidth', 0.07, ...
%             'FigureTitle','1D marginals — true vs posterior (rep seed)', ...
%             'ShowSeedBand', false);
%
%     % --- 2) With multiple seeds: show mean±SD band and annotate mean±SD of metrics ---
%     seedsCell = {S1,S2,S3,S4,S5,S6,S7,S8,S9,S10};  % each N-by-2
%     fig = plot_20mode_marginals(S_rep, MU20, 0.1, w, ...
%             'SeedSamples', seedsCell, ...
%             'KDEBandwidth', 0.07, ...
%             'FigureTitle','1D marginals — true vs posterior (rep + seeds)');
%
%     % --- 3) How to pick a “representative” seed from many (median W2) ---
%     w = ones(20,1)/20;
%     W2vals = zeros(numel(seedsCell),1);
%     for s = 1:numel(seedsCell)
%         W2vals(s) = local_w2_sum(seedsCell{s}, MU20, 0.1, w); % helper below
%     end
%     [~,mid] = min(abs(W2vals - median(W2vals)));
%     S_rep = seedsCell{mid};
%     fig = plot_20mode_marginals(S_rep, MU20, 0.1, w, ...
%             'SeedSamples', seedsCell, 'KDEBandwidth', 0.07);
%
%   % Helper (place in your script if you use Example 3)
%   function v = local_w2_sum(S, MU20, sigma, w)
%       [w21,w22] = w2_marginals_vs_mixture(S, MU20, sigma, w);
%       v = w21 + w22;  % or max([w21,w22])
%   end
%
% Xucong helper, 2025-11-01 (fixed kde1d outputs; added doc & examples)


% ---- options ----
ip = inputParser;
ip.addParameter('GridN', 2000, @(x)isscalar(x)&&x>=400);
ip.addParameter('KDEBandwidth','default');   % numeric to force bandwidth, otherwise default
ip.addParameter('FigureTitle','True mixture vs Posterior KDE');
ip.addParameter('SeedSamples', {}, @(c)iscell(c));  % cell of N-by-2
ip.addParameter('ShowSeedBand', true, @(x)islogical(x)&&isscalar(x));
ip.addParameter('Colors', struct('true',[0.1 0.1 0.1],'rep',[0.00 0.45 0.74], ...
                                 'band',[0.00 0.45 0.74]));
ip.addParameter('BandAlpha', 0.15, @(x)isnumeric(x)&&x>=0&&x<=1);
ip.parse(varargin{:});
opt = ip.Results;

% ---- checks ----
assert(size(samples,2)==2,'samples must be N-by-2.');
assert(isequal(size(mu20),[20,2]),'mu20 must be 20-by-2.');
if nargin<4 || isempty(weights), weights = ones(20,1)/20; end
weights = weights(:)/sum(weights);
assert(isfinite(sigma)&&sigma>0);

% ---- grids ----
pad = 0.5;
x1 = linspace(min(samples(:,1))-pad, max(samples(:,1))+pad, opt.GridN);
x2 = linspace(min(samples(:,2))-pad, max(samples(:,2))+pad, opt.GridN);

% ---- true 1D mixture marginals ----
p1_true = mixpdf_1d(x1, mu20(:,1), sigma, weights);
p2_true = mixpdf_1d(x2, mu20(:,2), sigma, weights);
p1_true = p1_true / trapz(x1, p1_true);
p2_true = p2_true / trapz(x2, p2_true);

% ---- representative seed KDE (use x1/x2 as grids) ----
p1_kde_rep = kde1d(samples(:,1), x1, opt.KDEBandwidth);  x1_kde = x1;
p2_kde_rep = kde1d(samples(:,2), x2, opt.KDEBandwidth);  x2_kde = x2;
p1_kde_rep = p1_kde_rep / trapz(x1_kde, p1_kde_rep);
p2_kde_rep = p2_kde_rep / trapz(x2_kde, p2_kde_rep);

% ---- optional mean±SD band across seeds ----
haveBand = opt.ShowSeedBand && ~isempty(opt.SeedSamples);
if haveBand
    nS = numel(opt.SeedSamples);
    P1 = zeros(numel(x1), nS); P2 = zeros(numel(x2), nS);
    W2_1_all = zeros(nS,1); W2_2_all = zeros(nS,1);
    Ds1_all = zeros(nS,1);  Ds2_all = zeros(nS,1);

    [m1_true, s1_true] = mix_mean_std(mu20(:,1), sigma, weights);
    [m2_true, s2_true] = mix_mean_std(mu20(:,2), sigma, weights);

    for s = 1:nS
        Si = opt.SeedSamples{s};
        assert(size(Si,2)==2,'Each cell in SeedSamples must be N-by-2.');
        Si = Si(all(isfinite(Si),2),:);

        P1(:,s) = kde1d(Si(:,1), x1, opt.KDEBandwidth); P1(:,s)=P1(:,s)/trapz(x1,P1(:,s));
        P2(:,s) = kde1d(Si(:,2), x2, opt.KDEBandwidth); P2(:,s)=P2(:,s)/trapz(x2,P2(:,s));

        m1 = mean(Si(:,1)); s1 = std(Si(:,1));
        m2 = mean(Si(:,2)); s2 = std(Si(:,2));
        Ds1_all(s) = sqrt(0.5*(( (m1_true-m1)/s1_true )^2 + ((s1_true-s1)/s1_true)^2));
        Ds2_all(s) = sqrt(0.5*(( (m2_true-m2)/s2_true )^2 + ((s2_true-s2)/s2_true)^2));

        [W2_1_all(s), W2_2_all(s)] = w2_marginals_vs_mixture(Si, mu20, sigma, weights);
    end
    p1_mu = mean(P1,2); p1_sd = std(P1,0,2);
    p2_mu = mean(P2,2); p2_sd = std(P2,0,2);
    Ds1_mean = mean(Ds1_all); Ds1_sd = std(Ds1_all);
    Ds2_mean = mean(Ds2_all); Ds2_sd = std(Ds2_all);
    W21_mean = mean(W2_1_all); W21_sd = std(W2_1_all);
    W22_mean = mean(W2_2_all); W22_sd = std(W2_2_all);
else
    [m1_true, s1_true] = mix_mean_std(mu20(:,1), sigma, weights);
    [m2_true, s2_true] = mix_mean_std(mu20(:,2), sigma, weights);
    Ds1_mean = sqrt(0.5*(( (m1_true-mean(samples(:,1)))/s1_true )^2 + ...
                         ((s1_true-std(samples(:,1)))/s1_true)^2));
    Ds2_mean = sqrt(0.5*(( (m2_true-mean(samples(:,2)))/s2_true )^2 + ...
                         ((s2_true-std(samples(:,2)))/s2_true)^2));
    [W21_mean, W22_mean] = w2_marginals_vs_mixture(samples, mu20, sigma, weights);
    Ds1_sd=NaN; Ds2_sd=NaN; W21_sd=NaN; W22_sd=NaN;
end

% ---------- plot ----------
fig = figure('Color','w','Name','1D marginals (20-mode)','Position',[100 100 1200 700]);
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

% styling
fillAlphaTrue_single = 0.20;   % 仅单种子时使用
fillAlphaRep_single  = 0.30;   % 仅单种子时使用
bandAlpha_multi      = 0.48;   % 多种子时 posterior 的均值±SD 阴影透明度

% ================= X1 =================
ax1 = nexttile; 
hold(ax1,'on');
ax1.FontSize = 12;           % 刻度数字字号

if haveBand
    % ---- 多种子：posterior 画 mean±SD 带 + 均值线；true 只画线 ----
    patch([x1, fliplr(x1)], [p1_mu-p1_sd; flipud(p1_mu+p1_sd)]', ...
          opt.Colors.rep, 'FaceAlpha', bandAlpha_multi, 'EdgeColor','none', ...
          'HandleVisibility','off');                          % band (no legend)
    h_rep_mu  = plot(x1, p1_mu, '-', 'Color', opt.Colors.rep, ...
                     'LineWidth', 1.8, 'DisplayName','Posterior (mean)');
    h_true_ln = plot(x1, p1_true, '-', 'Color', opt.Colors.true, ...
                     'LineWidth', 1.8, 'DisplayName','True mixture');
else
    % ---- 单种子：保持原样，填充对比更直观 ----
    area(x1, p1_true,    'FaceColor', opt.Colors.true, 'FaceAlpha', fillAlphaTrue_single, ...
         'EdgeAlpha',0, 'HandleVisibility','off');
    area(x1, p1_kde_rep, 'FaceColor', opt.Colors.rep,  'FaceAlpha', fillAlphaRep_single,  ...
         'EdgeAlpha',0, 'HandleVisibility','off');
    h_true_ln = plot(x1, p1_true,    '-', 'Color', opt.Colors.true, 'LineWidth',1.8, 'DisplayName','True mixture');
    h_rep_mu  = plot(x1, p1_kde_rep, '--','Color', opt.Colors.rep,  'LineWidth',1.8, 'DisplayName','Posterior KDE');
end

xlabel('X_1','FontSize',12); ylabel('Density (area = 1)','FontSize',12); box on; grid on;
legend([h_true_ln,h_rep_mu],'Location','northeast','FontSize',12);
text(0.02,0.9, annotate_line(Ds1_mean, Ds1_sd, W21_mean, W21_sd), ...
     'Units','normalized','HorizontalAlignment','left','FontSize',12);
xlim([-1,10]);ylim([0, max(ylim)]); 

% ================= X2 =================
ax2 = nexttile; 
hold(ax2,'on');
ax2.FontSize = 12;           % 刻度数字字号

if haveBand
    patch([x2, fliplr(x2)], [p2_mu-p2_sd; flipud(p2_mu+p2_sd)]', ...
          opt.Colors.rep, 'FaceAlpha', bandAlpha_multi, 'EdgeColor','none', ...
          'HandleVisibility','off');
    h_rep_mu2  = plot(x2, p2_mu, '-', 'Color', opt.Colors.rep, ...
                      'LineWidth', 1.8, 'DisplayName','Posterior (mean)');
    h_true_ln2 = plot(x2, p2_true, '-', 'Color', opt.Colors.true, ...
                      'LineWidth', 1.8, 'DisplayName','True mixture');
else
    area(x2, p2_true,    'FaceColor', opt.Colors.true, 'FaceAlpha', fillAlphaTrue_single, ...
         'EdgeAlpha',0, 'HandleVisibility','off');
    area(x2, p2_kde_rep, 'FaceColor', opt.Colors.rep,  'FaceAlpha', fillAlphaRep_single,  ...
         'EdgeAlpha',0, 'HandleVisibility','off');
    h_true_ln2 = plot(x2, p2_true,    '-', 'Color', opt.Colors.true, 'LineWidth',1.8, 'DisplayName','True mixture');
    h_rep_mu2  = plot(x2, p2_kde_rep, '--','Color', opt.Colors.rep,  'LineWidth',1.8, 'DisplayName','Posterior KDE');
end

xlabel('X_2','FontSize',12); ylabel('Density (area = 1)','FontSize',12); box on; grid on;
legend([h_true_ln2,h_rep_mu2],'Location','northeast','FontSize',12);
text(0.02,0.9, annotate_line(Ds2_mean, Ds2_sd, W22_mean, W22_sd), ...
     'Units','normalized','HorizontalAlignment','left','FontSize',12);
xlim([-1,10]);ylim([0, max(ylim)]); 

sgtitle(opt.FigureTitle, 'FontWeight','bold');


end % ===== main =====

% ----------------- helpers -----------------
function p = mixpdf_1d(x, mu, sigma, w)
    x = x(:);
    p = zeros(numel(x),1);
    for j=1:numel(w)
        p = p + w(j).*normpdf(x, mu(j), sigma);
    end
end

function [m,s] = mix_mean_std(mu, sigma, w)
    m = sum(w.*mu);
    v = sum(w.*(sigma^2 + mu.^2)) - m.^2;
    s = sqrt(max(v, eps));
end

function y = kde1d(data, gridx, bw)
% Only one output; grid is always the input grid.
    data = data(:); data = data(isfinite(data));
    if isempty(data), y = zeros(size(gridx)); return; end
    if isnumeric(bw)
        y = ksdensity(data, gridx, 'Bandwidth', bw);
    else
        y = ksdensity(data, gridx); % default bandwidth
    end
end



function s = annotate_line(DsMean, DsSD, W2Mean, W2SD)
    if ~isnan(DsSD) && ~isnan(W2SD)
        s = sprintf('D_s=%.3g \\pm %.3g   W_2=%.3g \\pm %.3g', DsMean, DsSD, W2Mean, W2SD);
    else
        s = sprintf('D_s=%.3g   W_2=%.3g', DsMean, W2Mean);
    end
end
