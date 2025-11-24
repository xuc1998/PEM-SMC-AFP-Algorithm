function hfig = plot_ds_bars3(truth, Ds, varargin)
%PLOT_DS_BARS3  3D bar chart of Ds per mode at the true (X1,X2) locations.
%
% Usage:
%   hfig = plot_ds_bars3(truth, Ds, 'BarWidth',0.35, 'CMap',parula, 'ShowValues',true)
%
% Inputs
%   truth : [K x 2] matrix of reference coordinates (X1, X2) for each mode
%   Ds    : [K x 1] vector of Ds values per mode (NaN entries are skipped)
%
% Name-Value options (all optional)
%   'BarWidth'   (default: 0.35)  Square base width of each bar (in data units)
%   'FaceAlpha'  (default: 0.95)  Face transparency of bars
%   'EdgeColor'  (default: [0.15 0.15 0.15]) Edge color of bars
%   'CMap'       (default: parula) Colormap function handle or Kx3 colors
%   'FixedColor' (default: [])     If set to [r g b], use one fixed color
%   'ShowValues' (default: false)  If true, print Ds value on top of each bar
%
% Output
%   hfig : figure handle

% ---------- parse options ----------
p = inputParser;
p.addParameter('BarWidth', 0.35, @(x) isscalar(x) && x>0);
p.addParameter('FaceAlpha', 0.95, @(x) isscalar(x) && x>=0 && x<=1);
p.addParameter('EdgeColor', [0.15 0.15 0.15]);
p.addParameter('CMap', @parula);
p.addParameter('FixedColor', []);
p.addParameter('ShowValues', false, @islogical);
p.parse(varargin{:});
opt = p.Results;

% ---------- basic checks ----------
truth = double(truth);
Ds    = double(Ds(:));
K = size(truth,1);
assert(size(truth,2)==2, 'truth must be Kx2.');
assert(numel(Ds)==K, 'Ds must have K elements.');

% ---------- color assignment ----------
valid = ~isnan(Ds);
Ds_valid = Ds(valid);
if isempty(opt.FixedColor)
    % map height to colormap
    if isa(opt.CMap,'function_handle')
        cmap = opt.CMap(256);
    else
        cmap = opt.CMap;
    end
    % normalize Ds to [0,1] for color indexing (robust to outliers)
    lo = prctile(Ds_valid, 5); hi = prctile(Ds_valid, 95);
    if hi<=lo, lo=min(Ds_valid); hi=max(Ds_valid)+eps; end
    cidx = min(max( round( 1 + (size(cmap,1)-1) * (Ds - lo) / (hi - lo) ), 1), size(cmap,1));
    colors = cmap(cidx, :);
else
    colors = repmat(opt.FixedColor, K, 1);
end

% ---------- figure ----------
hfig = figure('Name','Ds per-mode (3D bars)','Color','w');
hold on;

% ---------- draw bars as prisms ----------
w = opt.BarWidth;
for k = 1:K
    if ~valid(k), continue; end
    x = truth(k,1); y = truth(k,2); h = Ds(k);
    if h<0, continue; end   % Ds should be >= 0

    % square base corners (counter-clockwise)
    x0 = x - w/2; x1 = x + w/2;
    y0 = y - w/2; y1 = y + w/2;

    % 5 faces (4 sides + top). Bottom is at z=0.
    % top face
    patch('XData',[x0 x1 x1 x0], 'YData',[y0 y0 y1 y1], 'ZData',[h h h h], ...
          'FaceColor', colors(k,:), 'FaceAlpha', opt.FaceAlpha, ...
          'EdgeColor', opt.EdgeColor);

    % sides
    % front (y=y0)
    patch([x0 x1 x1 x0],[y0 y0 y0 y0],[0 0 h h], ...
        colors(k,:), 'FaceAlpha', opt.FaceAlpha, 'EdgeColor', opt.EdgeColor);
    % back (y=y1)
    patch([x0 x1 x1 x0],[y1 y1 y1 y1],[0 0 h h], ...
        colors(k,:), 'FaceAlpha', opt.FaceAlpha, 'EdgeColor', opt.EdgeColor);
    % left (x=x0)
    patch([x0 x0 x0 x0],[y0 y1 y1 y0],[0 0 h h], ...
        colors(k,:), 'FaceAlpha', opt.FaceAlpha, 'EdgeColor', opt.EdgeColor);
    % right (x=x1)
    patch([x1 x1 x1 x1],[y0 y1 y1 y0],[0 0 h h], ...
        colors(k,:), 'FaceAlpha', opt.FaceAlpha, 'EdgeColor', opt.EdgeColor);

    % optional value label
    if opt.ShowValues
        text(x, y, h, sprintf(' %.2f', h), 'HorizontalAlignment','left', ...
             'VerticalAlignment','bottom', 'FontSize',9, 'Color',[0.1 0.1 0.1]);
    end
end

% ---------- axes & view ----------
% ---------- axes & view (robust limits + clipping) ----------
xlabel('$X_1$','Interpreter','latex');
ylabel('$X_2$','Interpreter','latex');
zlabel('$D_s$','Interpreter','latex');

% Compute padding and limits
valid = ~isnan(Ds);
xmin = min(truth(valid,1)); xmax = max(truth(valid,1));
ymin = min(truth(valid,2)); ymax = max(truth(valid,2));
zmax = max(Ds(valid));

padXY = max(opt.BarWidth, 0.4);   % XY padding (data units)
padZ  = 0.08*zmax;                % Z padding (8%)

xlim([xmin - padXY, xmax + padXY]);
ylim([ymin - padXY, ymax + padXY]);
zlim([0, zmax + padZ]);

grid on; box on;
set(gca,'Clipping','on');         % clip everything to axes box

% Keep a pleasant aspect (square XY, slightly compressed Z)
pbaspect([1 1 0.8]);              % XY 1:1, Z 压缩一点
set(gca,'Projection','perspective');
view(45, 25);

% Colorbar
if isempty(opt.FixedColor)
    cb = colorbar('Location','eastoutside');
    cb.Label.String = 'D_s (lower is better)';
end

end
