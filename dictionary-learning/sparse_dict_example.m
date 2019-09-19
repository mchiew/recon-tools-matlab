% Example sparse patch-based dictionary reconstruction of undersampled data
% Mark Chiew (mark.chiew@ndcn.ox.ac.uk)
%
% Solve the following sparse dictionary learning problem:
% min_{X,D,a} ||E*X - d||_2 + lamdba*||R*X-D*a||_2
% s.t. ||a||_0 <= k
%
% where X is the image, D is the over-complete dictionary, a is sparse code
% E is the encoding operator, d is the sampled k-space data
% lambda is the sparse regularisation factor, R is the patch operator
% and k is the sparsity constraint on the sparse coding of the patches
%
% Solved using alternating minimisation:
% Subproblem 1 : % min_{D,a} ||R*X-D*a||_2 s.t. ||a||_0 <= k
% Subproblem 2 : % min_{X} ||E*X - d||_2 + lamdba*||R*X-D*a||_2
%
% Subproblem 1 is solved using k-SVD and Orthogonal Matching Pursuit
% Subproblem 2 is linear and solved using conjugate gradient descent
% 
% See Aharon, Elad, and Bruckstein. IEEE-TSP 2006, 54, no. 11 : 4311?22. 
% https://doi.org/10.1109/TSP.2006.881199
% and
% Ravishankar and Bresler. IEEE-TMI 2011, 30, no. 5 : 1028?41. 
% https://doi.org/10.1109/TMI.2010.2090538
%
% Depends on the MR encoding operators that can be downloaded at:
% https://users.fmrib.ox.ac.uk/~mchiew/tools.html

% Initialise parameters
lambda = 0.1;                       % sparse regularisation factor         
k = 1;                              % patch dictionary sparsity
patch_size = [6,6];                 % patch size
patch_gap  = 2;                     % distance between patches
dict_train = 0.25;                  % percentage of patches for training
dict_size  = prod(patch_size)*4;    % dictionary size
image_size = [96,96];               % image size
outer_iters = 25;                   % num alternations
inner_iters = 10;                   % num iters for pcg

% Test image
X0 = phantom(image_size(1));

% Under-sampling operator
mask = poisson_disc(image_size, 0.9, 32);       % R ~ 3 under-sampling
E = xfm_FFT([image_size 1 1],[],[],'mask',mask);% Encoding operator

% Sample data and add noise
d = E*X0 + 1E-1*(randn(E.dsize)+1j*randn(E.dsize))/sqrt(2);

% Zero-filled recon
X = reshape(E'*d,E.Nd);

% Get helper operator (M = R'R)
p       =   patch_get(X,patch_size,patch_gap);
[~,M]   =   patch_adj(p, image_size, patch_gap);

% Plot initial estimate and ground truth
figure(101);clf();
splt(2,2,1);show(abs(X),[0 1]);title('Zero Filled');
splt(2,2,4);show(abs(X0),[0 1]);title('Ground Truth');

% Solve via alternating minimisation
for i = 1:outer_iters
    % Learn dictionary and sparse code
    % (Find D, a, given X)    
    p       =   patch_get(X,patch_size,patch_gap);
    Np      =   size(p,2);
    D       =   ksvd(p(:,randperm(Np,round(Np*dict_train))), dict_size, k);
    a       =   omp(D,p,k);
    
    % Plot current dictionary estimate (sorted by variance)
    v = var(D,[],1); [~,ii] = sort(v,'Descend');
    splt(2,2,2);mcat(padarray(abs(reshape(D(:,ii),[patch_size size(D,2)])),[1 1 0]));title('Current dictionary');
    
    % Estimate image
    % (Find X, give D, a)    
    [X,~] = pcg(@(x)pcg_fn(x,E,M,lambda), E'*d + reshape(lambda*patch_adj(D*a, image_size, patch_gap),[],1), 1E-4, inner_iters);    
    X = reshape(X, E.Nd);
    
    % Plot current image estimate
    splt(2,2,3);show(abs(X),[0 1]);title('Current estimate');
    
    % Disp normalised RMSE
    fprintf(1,'Iter: %02d  NRMSE: %4.3g\n',i,norm(X0(:)-X(:))/norm(X(:)));
end

%% Helper function for pcg
function x = pcg_fn(x,E,M,lambda)
    x = reshape(x,E.Nd);
    x = E.mtimes2(x) + reshape(lambda*(M.*x),[],1);
end
