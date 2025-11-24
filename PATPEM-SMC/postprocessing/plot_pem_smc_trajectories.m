function figs = plot_pem_smc_trajectories(parameter_iteration, opts, varargin)
%PLOT_PEM_SMC_TRAJECTORIES Particle trajectories across stages (1D only).
%
% Usage:
%   figs = plot_pem_smc_trajectories(parameter_iteration, opts, ...
%             'NumTraj',20,'Seed',42,'Marker','.', 'MarkerSize',8, ...
%             'NumDims', [])   % [] or omit => randomly sample up to 4 dims; otherwise pick that many at random (capped to 4)
%
% Inputs
%   parameter_iteration : [Np x d x S_K]  (already thinned)
%   opts                : struct with at least .Thinning (integer >=1)
%
% Name-Value (all optional)
%   NumTraj      how many particle trajectories per dimension (default 20)
%   Seed         RNG seed for reproducible selection (default 42)
%   Marker       marker symbol for points (default '.')
%   MarkerSize   marker size (default 8)
%   NumDims      number of parameter dimensions to plot (randomly chosen if provided).
%                If omitted or empty, randomly sample up to 4 dims (reproducible with Seed).
%
% Behavior:
%   • Plots point trajectories per selected dimensions (no continuous lines).
%   • Dimension selection is always random (reproducible via 'Seed'), capped at 4.

    if nargin < 2 || isempty(opts), opts = struct(); end
    if ~isfield(opts,'Thinning') || isempty(opts.Thinning)
        thin = 1;
    else
        thin = max(1, round(opts.Thinning));
    end

    % -------- Parse options --------
    p = inputParser;
    p.addParameter('NumTraj', 20);
    p.addParameter('Seed', 1234);
    p.addParameter('Marker', '.');
    p.addParameter('MarkerSize', 8);
    p.addParameter('NumDims', []);     % number of dims to draw (random), [] => random up to 4
    p.parse(varargin{:});
    prm = p.Results;

    [Np, d, S_K] = size(parameter_iteration);

    % Thinned stage indices (1..S_K)
    stages = 1:S_K;

    % Select particles to plot
    K = min(prm.NumTraj, Np);
    rng(prm.Seed);
    selParticles = randperm(Np, K);

    % -------- Dimension selection (random, capped at 4) --------
    if isempty(prm.NumDims)
        nShow = min(d, 4);
        dimsToShow = sort(randperm(d, nShow));
        if d > 4
            fprintf('[plot_pem_smc_trajectories] d=%d; randomly sampled %d dims (Seed=%d): [%s]\n', ...
                d, nShow, prm.Seed, num2str(dimsToShow));
        end
    else
        nWant = max(1, round(prm.NumDims));
        nShow = min([nWant, d, 4]);
        dimsToShow = sort(randperm(d, nShow));
        fprintf('[plot_pem_smc_trajectories] randomly selected %d/%d dims (Seed=%d): [%s]\n', ...
            nShow, d, prm.Seed, num2str(dimsToShow));
    end

    % -------- Plot: per-dimension trajectories (points only) --------
    figs.traj = figure('Name','Trajectories per dimension','Color','w');
%     try
%         % R2018a+ (recommended)
%         figs.traj.WindowState = 'maximized';
%     catch
%         % Fallback for older MATLAB
%         set(figs.traj, 'Units','normalized', 'OuterPosition',[0 0 1 1]);
%     end
    set(figs.traj, 'DefaultAxesFontSize', 14, ...
             'DefaultTextFontSize', 14, ...
             'DefaultAxesTitleFontSizeMultiplier', 1.2, ...
             'DefaultAxesLabelFontSizeMultiplier', 1);
    
    tiledlayout(nShow, 1, 'TileSpacing','compact','Padding','compact');

    xLabelStr = sprintf('Stage s (thinning=%d)', thin);

    for ii = 1:nShow
        j = dimsToShow(ii);
        ax = nexttile; hold(ax,'on'); grid(ax,'on'); box(ax,'on');

        Y = squeeze(parameter_iteration(selParticles, j, stages));  % [K x S_K]
        for k = 1:K
            plot(ax, stages, Y(k,:), 'LineStyle','none', ...
                'Marker', prm.Marker, 'MarkerSize', prm.MarkerSize);
        end

        % Only the last subplot shows the x-label
        if ii == nShow
            xlabel(ax, xLabelStr,'FontSize',11);
        else
            xlabel(ax, '');
        end

        ylabel(ax, sprintf('\\theta_{%d}', j),'FontSize',11);
        title(ax, sprintf('Particle trajectories — \\theta_{%d}', j),'FontSize',14);
        xlim(ax, [stages(1), stages(end)]);

    end
end
