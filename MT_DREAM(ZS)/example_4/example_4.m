%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
%%                                                                                    %%
%% DDDDD   RRRR    EEEEE    AA    MM   MM      SSSSSS  UU   UU   II   TTTTTTTT  EEEEE %%
%% DDDDDD  RRRR    EEEEE   AAAA   MM   MM      SSSSS   UU   UU   II   TTTTTTTT  EEEEE %%
%% DD  DD  RR RR   EE     AA  AA  MMM MMM      SS      UU   UU   II      TT     EE    %%
%% DD  DD  RR RR   EEE    AA  AA  MMMMMMM ---- SS      UU   UU   II      TT     EEE   %%
%% DD  DD  RRRRR   EEE    AAAAAA  MMM MMM ---- SSSSSS  UU   UU   II      TT     EEE   %%
%% DD  DD  RR RR   EE     AAAAAA  MM   MM          SS  UU   UU   II      TT     EE    %%
%% DDDDDD  RR  RR  EEEEE  AA  AA  MM   MM       SSSSS  UUUUUUU   II      TT     EEEEE %%
%% DDDDD   RR  RR  EEEEE  AA  AA  MM   MM      SSSSSS  UUUUUUU   II      TT     EEEEE %%
%%                                                                                    %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
%%                                                                                    %%
%% Example 3: Synthetic CoLM Benchmark for High-Dimensional Bayesian Calibration
%%                                                                                    %%
%% Check the following papers                                                         %%
%%  Vrugt, J.A., C.J.F. ter Braak, C.G.H. Diks, D. Higdon, B.A. Robinson, and J.M.    %%
%%      Hyman (2009), Accelerating Markov chain Monte Carlo simulation by             %%
%%      differential evolution with self-adaptive randomized subspace sampling,       %%
%%      International Journal of Nonlinear Sciences and Numerical Simulation, 10(3),  %%
%%      271-288.                                                                      %%
%%   Vrugt, J.A., H.V. Gupta, W. Bouten and S. Sorooshian (2003), A Shuffled Complex  %%
%%      Evolution Metropolis algorithm for optimization and uncertainty assessment of %%
%%      hydrologic model parameters, Water Resour. Res., 39 (8), 1201,                %%
%%      doi:10.1029/2002WR001642.                                                     %%
%%                                                                                    %%
%% ---------------------------------------------------------------------------------- %%
rng(1234,'twister');
%% Problem settings defined by user
DREAMPar.d = 6;                        % Dimension of the problem
DREAMPar.lik = 2;                      % Model output is log-likelihood

%% Provide information parameter space and initial sampling
Par_info.initial = 'uniform';           % N2(µ,Σ) initial distribution
% Par_info.mu = zeros(1,DREAMPar.d);     % if 'normal', µ-mean of distribution
% Par_info.cov = 10 * eye(DREAMPar.d);   % if 'normal', Σ-covariance matrix
% bounds=[0.005,0.015;0.25,0.75;0.25,0.75;2.5,7.5;2.5,7.5;0.001,1;0.001,1;
%         -500,-50;-500,-50;0.15,0.45;0.01,0.1;5 15;0.002,0.006;
%         0.0012,0.0036;0.17 0.51;0.25 0.5;0.05,0.15;0.05,0.3;-0.3,0.1;
%         0.07,0.105;0.16,0.36;0.35,0.58;0.39,0.58;0.04,0.08;0.1,0.3;
%         0.1,0.3;0.3,0.5;0.1,0.3;278,288;0.15,0.45;305,315;2.5,7.5;0.05,0.08;
%         10,200;4,9;0.01,0.04;0.5,0.75;-2e5,-1e5;-1e8,-9e7;1e-4,100e-4];
bounds=[10,  0.25,  2.5,   0.05,    2.5,   -500; 
       200, 0.75,  7.5,   0.08,    7.5,   -50]; 
Par_info.min = bounds(1,:);                % 1xd-vector of min parameter values    
Par_info.max=bounds(2,:);                  % 1xd-vector of max parameter values    
Par_info.boundhandling='fold';            % reject of boudn proposals

%% Define name of function (.m file) for posterior exploration
Func_name = 'target';
observed_LE=importdata('./LE_verify.txt');
observed_NEE=importdata('./NEE_verify.txt');
observed_RSM=importdata('./RSM_verify.txt');
plugin.observed_LE=observed_LE;
plugin.observed_NEE=observed_NEE;
plugin.observed_RSM=observed_RSM;


% CoLM model file path
old_path  = '/data/groups/lzu_public/home/u120220909911/lustre_data/Arou/PEM-SMC/WMO';
baseDir='/data/groups/lzu_public/home/u120220909911/lustre_data/Arou/PEM-SMC';
plugin.old_path=old_path;
plugin.baseDir=baseDir;

%% Define method to use {'dream','dream_zs','dream_d','dream_dzs','mtdream_zs'}
method = 'mtdream_zs';

switch method
    case {'dream','dream_d'}
        DREAMPar.N = 10;                        % # Markov chains
        DREAMPar.T = 100000;                    % # generations
    case {'dream_zs','dream_dzs','mtdream_zs'}
        DREAMPar.N = 5;                        % # Markov chains
        DREAMPar.T = 10000;                      % # generations
end

if strcmp(method,'dream_d') || strcmp(method,'dream_dzs')
    Par_info.min = -100 * ones(1,DREAMPar.d);   % Min value for discrete sampling
    Par_info.max = 100* ones(1,DREAMPar.d);    % Max value for discrete sampling
    Par_info.steps = 2001*ones(1,DREAMPar.d);   % # discrete steps
end

DREAMPar.mt=5;       % Number of multi-try proposals
DREAMPar.thinning=1; 
%% Optional Settings
options.modout='no';   % Return model simulations samples?
options.parallel='yes';  %Run each chain on a different core
options.IO='no';
options.save = 'no';                 % Save workspace DREAM during run

%% Run DREAM-Suite package
tic
[chain,output,FX,Z,logL] = DREAM_Suite(method,Func_name,DREAMPar,...
    Par_info,[],options,[],plugin);
toc
save chain chain
save output output
save DREAMPar.mat DREAMPar
%% Statistics and Plot
saveAllFigures('./graphs','Format','png');