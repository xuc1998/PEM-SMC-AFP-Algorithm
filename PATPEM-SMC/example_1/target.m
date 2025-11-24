function L = target(x)
%TARGET  Log target density for a 2D, 20-mode Gaussian mixture (example).
%
% INPUT
%   x : 1 x 2 vector (point in R^2)
%
% OUTPUT
%   L : scalar log-density log f(x)
%
% Note: This is your test target; the sampler only needs target(x) to
% return a log-density up to a constant.

Nm    = 20;                       % number of modes
sigma = ones(1, Nm) * (0.1^2);    % variance per mode
mu = [2.18 5.76; 8.67 9.59; 4.24 8.48; 8.41 1.68; 3.93 7.82; 3.25 3.47; 1.70 0.50;
      4.59 5.60; 6.91 5.81; 6.87 5.40; 5.41 2.65; 2.70 7.88; 4.98 3.70; 1.14 2.39;
      8.33 9.50; 4.93 1.50; 1.83 0.09; 2.26 0.31; 5.54 6.86; 1.69 8.11];
w  = ones(1, Nm) * 0.05;          % equal weights (sum to 1)

f = 0;
for i = 1:Nm
    diff = x - mu(i,:);
    f = f + w(i) / (2*pi*sigma(i)) * exp( - (diff*diff') / (2*sigma(i)) );
end
L = log(f);
end
