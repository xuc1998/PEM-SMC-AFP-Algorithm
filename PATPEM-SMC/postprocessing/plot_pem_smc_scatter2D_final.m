function hfig = plot_pem_smc_scatter2D_final(theta_final, bound, truth)
%PLOT_PEM_SMC_SCATTER2D_FINAL  Final-stage particles in 2D with optional truth overlay.
% Inputs:
%   theta_final : [Np x d] final particles (d >= 2)
%   bound       : [2 x d] box bounds (optional; can be [])
%   truth       : [K x 2] reference points to overlay (optional; [])
%
% Output:
%   hfig : figure handle

hfig = figure('Name','Final posterior scatter 2D','Color','w');
scatter(theta_final(:,1), theta_final(:,2), 8, 'k', 'filled'); 
hold on;

% Overlay truth as magenta 'x'
if ~isempty(truth)
    if size(truth,2) ~= 2
        warning('Truth must be Kx2 for 2D overlay. Ignoring truth.');
    else
        plot(truth(:,1), truth(:,2), 'x', 'Color', [1 0 1], ...
            'LineWidth', 1.8, 'MarkerSize', 9);
        legend({'Particles','Truth'}, 'Location','best');
    end
end

% Axes appearance
if ~isempty(bound) && size(bound,2) >= 2
    xlim([bound(1,1), bound(2,1)]);
    ylim([bound(1,2), bound(2,2)]);
end
xlim([-1 10]);
ylim([-1 10]);
xlabel('$X_1$','Interpreter','latex'); 
ylabel('$X_2$','Interpreter','latex'); 
title('Final particles Distribution');
grid on; box on;
ax=gca;
ax.FontSize=12;
axis tight;           % square plot box
end
