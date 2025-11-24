function betas = sequencesGen(S, m, x)
%SEQUENCESGEN  Exponential tempering schedule {beta_s}.
%
% Input
%   S : number of stages; returns (S+1)-vector beta(0..S)
%   m : small scalar (e.g., 7e-16)
%   x : exponent chosen so that log(1/m)/log(S) in your original code
%
% Output
%   betas : column vector of length S+1
%           betas(1)   = beta_0 = 0
%           betas(S+1) = beta_S ≈ 1 (tail is forced to 1 upstream)
%
% Original relation: betas = m * s.^x, with s = 0:S.
s = (0:S).';
betas = m * s.^x;
end
