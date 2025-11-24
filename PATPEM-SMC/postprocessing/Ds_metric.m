function Ds = Ds_metric(mu_ref, sigma_ref, mu_est, sigma_est)
%Ds_metric  Scalar distance combining standardized mean error and relative std error.
% Inputs: 1xd row vectors: mu_ref, sigma_ref, mu_est, sigma_est
    mu_ref   = mu_ref(:)';   sigma_ref = sigma_ref(:)';
    mu_est   = mu_est(:)';   sigma_est = sigma_est(:)';
    d = numel(mu_ref);
    dm = (mu_ref - mu_est) ./ sigma_ref;        % standardized mean error
    ds = (sigma_ref - sigma_est) ./ sigma_ref;  % relative std error
    Ds = sqrt( (sum(dm.^2 + ds.^2)) / (2*d) );
end
