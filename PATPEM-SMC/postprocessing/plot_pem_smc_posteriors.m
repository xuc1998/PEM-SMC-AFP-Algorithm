function fig = plot_pem_smc_posteriors(parameter_iteration, varargin)
%PLOT_PEM_SMC_POSTERIORS  Final posterior marginals (histograms per dimension),
% with optional overlay of the true marginal pdf curve. All dimensions are
% plotted. If d > 20, figures are split into batches of 20 dimensions per figure.
% No legends are shown.
%
% Usage:
%   fig = plot_pem_smc_posteriors(parameter_iteration)
%   fig = plot_pem_smc_posteriors(parameter_iteration, 'OverlayTruth', true, ...)
%
% Inputs
%   parameter_iteration : [Np x d x S_K]  (thinned along 3rd dim; last page is final)
%
% Name-Value (optional)
%   % truth overlay (optional)
%   OverlayTruth : logical, whether to draw the true marginal pdf (default: false)
%   TruthType    : 'bimodal-normal' | 'custom' (default: 'bimodal-normal')
%   Weights      : [1x2] mixture weights for bimodal-normal (default: [1/3 2/3])
%   MuVals       : [1x2] component means per dimension for bimodal-normal (default: [-5 5])
%   Sigma        : scalar std for each component in bimodal-normal (default: 1)
%   TruthPDF     : function handle @(x,j) -> pdf values (used when TruthType='custom')
%
%   % plotting range & grid for overlay
%   Bounds    : [2 x d] plotting bounds per dim; if empty, use data range with padding (default: [])
%   NumGrid   : number of x-grid points for truth curve (default: 400)
%   LineSpec  : line specification for truth curve (default: '-', linewidth=1.5)
%
% Behavior
%   • Plots all dimensions. If d > 20, splits across multiple figures with 20 dims per figure.
%   • Uses a compact tiled layout (up to 4x5 per figure).
%   • No legends are displayed.

    % -------- Parse options --------
    args = inputParser;
    % truth overlay
    args.addParameter('OverlayTruth', false, @islogical);
    args.addParameter('TruthType', 'bimodal-normal', @(s)ischar(s) || isstring(s));
    args.addParameter('Weights', [1/3 2/3], @(v)isnumeric(v) && numel(v)==2);
    args.addParameter('MuVals',  [-5 5], @(v)isnumeric(v) && numel(v)==2);
    args.addParameter('Sigma',   1, @(x)isnumeric(x) && isscalar(x) && x>0);
    args.addParameter('TruthPDF', [], @(f) isempty(f) || isa(f,'function_handle'));
    % plotting range & grid
    args.addParameter('Bounds', [], @(B) isempty(B) || (ismatrix(B) && size(B,1)==2));
    args.addParameter('NumGrid', 400, @(x)isnumeric(x) && isscalar(x) && x>=50);
    args.addParameter('LineSpec', '-', @(s)ischar(s) || isstring(s));
    args.parse(varargin{:});
    prm = args.Results;

    % -------- Data --------
    [~, d, S_K] = size(parameter_iteration);
    theta_final = squeeze(parameter_iteration(:,:,S_K));

    % -------- Figure batching (20 dims per figure) --------
    perFig = 20;
    nFigs  = ceil(d / perFig);
    fig    = gobjects(1, nFigs);  % figure handles

    for f = 1:nFigs
        idxStart = (f-1)*perFig + 1;
        idxEnd   = min(f*perFig, d);
        dims_f   = idxStart:idxEnd;
        nThis    = numel(dims_f);

        % Determine a compact grid up to 4x5 (rows x cols)
        % Balanced grid: at most 5 columns, at least 2, close to square
        maxCols = 5;
        cols = max(2, min(maxCols, ceil(sqrt(nThis))));
        rows = ceil(nThis / cols);


        fig(f) = figure('Name', sprintf('Final posterior marginals (%d/%d)', f, nFigs), ...
                        'Color','w');
        % ---- Auto-maximize when many panels ----
        if d >= 5 || nThis > 10
            try
                % R2018a+ (recommended)
                fig(f).WindowState = 'maximized';
            catch
                % Fallback for older MATLAB
                set(fig(f), 'Units','normalized', 'OuterPosition',[0 0 1 1]);
            end
        end
        tiledlayout(rows, cols, 'TileSpacing','compact','Padding','compact');

        for t = 1:nThis
            j = dims_f(t);
            nexttile; hold on; grid on; box on;

            % histogram (empirical posterior marginal)
            histogram(theta_final(:,j), 60, 'Normalization','pdf', 'EdgeColor','k');

            xlabel(sprintf('\\theta_{%d}', j),'FontSize',20);
            ylabel('pdf','FontSize',20);
%             title(sprintf('Marginal of \\theta_{%d} (final)', j));

            % ---- optional truth overlay (no legend) ----
            if prm.OverlayTruth
                % build x-grid
                if ~isempty(prm.Bounds) && size(prm.Bounds,2) >= j && all(isfinite(prm.Bounds(:,j)))
                    xmin = prm.Bounds(1,j); xmax = prm.Bounds(2,j);
                else
                    xmin = min(theta_final(:,j)); 
                    xmax = max(theta_final(:,j));
                    if ~isfinite(xmin) || ~isfinite(xmax)
                        xmin = -1; xmax = 1; % guard
                    end
                    if strcmpi(prm.TruthType,'bimodal-normal')
                        pad = 3*prm.Sigma;
                        if isfinite(pad)
                            xmin = min(xmin, min(prm.MuVals) - pad);
                            xmax = max(xmax, max(prm.MuVals) + pad);
                        end
                    end
                end
                if xmin==xmax, xmin=xmin-1; xmax=xmax+1; end
                xs = linspace(xmin, xmax, prm.NumGrid);

                % evaluate truth pdf per dim
                ys = [];
                switch lower(prm.TruthType)
                    case 'bimodal-normal'
                        w  = prm.Weights(:)';      % [w1 w2]
                        mu = prm.MuVals(:)';       % [m1 m2]
                        s  = prm.Sigma;
                        ys = w(1)*normpdf(xs, mu(1), s) + w(2)*normpdf(xs, mu(2), s);
                    case 'custom'
                        if ~isempty(prm.TruthPDF)
                            ys = prm.TruthPDF(xs, j);
                            if ~isvector(ys) || numel(ys)~=numel(xs)
                                warning('TruthPDF output size mismatch; skipping overlay for dim %d.', j);
                                ys = [];
                            end
                        end
                    otherwise
                        warning('Unknown TruthType "%s"; skipping overlay.', prm.TruthType);
                end

                % plot truth curve if available (no legend)
                if ~isempty(ys)
                    plot(xs, ys, prm.LineSpec, 'LineWidth', 3);
                    % adjust y-lim to fit both histogram and curve
                    curYLim = ylim;
                    ymax = max([curYLim(2), max(ys)]);
                    ylim([0, ymax*1.05]);
                end
            end
        end
    end
end
