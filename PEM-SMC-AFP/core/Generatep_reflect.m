function theta_new = Generatep_reflect(particles, k, LB, UB, DEMH)
%GENERATEP_REFLECT  DE–MH proposal with reflective box boundaries.
%
% Input
%   particles : N x d matrix of current particles
%   k         : index of the particle to mutate (1..N)
%   LB, UB    : 1 x d lower/upper bounds
%   DEMH      : struct with fields:
%                 .Gamma    : DE jump rate (default 2.38/sqrt(2*d))
%                 .NoiseSD  : Gaussian jitter std (default 1e-4)
%
% Output
%   theta_new : 1 x d DE proposal after reflection into [LB,UB]
%
% Notes
% - Parents (r1,r2) are drawn as an ordered pair uniformly (r1~=r2, r1/r2~=k).
% - With symmetric jitter and ordered-pair symmetry, the reflective proposal is
%   symmetric; MH acceptance uses the standard ratio with the tempered target.

    [N, d] = size(particles);

    if ~isfield(DEMH,'Gamma')   || isempty(DEMH.Gamma),   DEMH.Gamma   = 2.38 / sqrt(2*d); end
    if ~isfield(DEMH,'NoiseSD') || isempty(DEMH.NoiseSD), DEMH.NoiseSD = 1e-4;             end

    % choose two distinct parents excluding k
    B = setdiff(1:N, k);
    R = B(randperm(N-1, 2));

    % base DE proposal + Gaussian jitter
    prop = particles(k,:) ...
         + DEMH.Gamma   .* (particles(R(1),:) - particles(R(2),:)) ...
         + DEMH.NoiseSD .* randn(1,d);

    % reflect back into the box
    theta_new = reflect_into_box(prop, LB, UB);
end

% --------- local reflection utility ---------
function y = reflect_into_box(z, LB, UB)
%REFLECT_INTO_BOX  Triangle-wave reflection to enforce bounds.
% Applies per-dimension reflection into [LB,UB].

    L = UB - LB;
    y = (z - LB) ./ L;
    y = abs(mod(y,2) - 1);
    y = LB + y .* L;
end
