% Example sparse patch-based dictionary reconstruction of undersampled data
% Mark Chiew (mark.chiew@ndcn.ox.ac.uk)

% Solve the following sparse dictionary learning problem:
% min_{X,D,a} ||E*X - d||_2 + lamdba*||R*X-D*a||_2
% s.t. ||a||_0 ? k
%
% where X is the image, D is the over-complete dictionary, a is sparse code
% E is the encoding operator, d is the sampled k-space data
% lambda is the sparse regularisation factor, R is the patch operator
% and k is the sparsity constraint on the sparse coding of the patches

% Initialise parameters
lambda = 1;
k = 3;
patch_size = [6,6];
patch_gap  = 2;
dict_train = 0.5;   % percentage of patches to use for learning
dict_size  = prod(patch_size)*6;
image_size = [96,96];
outer_iters = 10;
inner_iters = 10;

% Test image
X0 = phantom(image_size(1));

% Under-sampling operator
mask = poisson_disc(image_size, 0.9, 32); % R ~ 3 under-sampling
E = xfm_FFT([image_size 1 1],[],[],'mask',mask);

% Sample data and add noise
d = E*X0 + 1E-1*(randn(E.dsize)+1j*randn(E.dsize))/sqrt(2);

% Zero-filled recon
X = reshape(E'*d,E.Nd);

% Get helper operator
p       =   patch_get(X,patch_size,patch_gap);
[~,M]   =   patch_adj(p, image_size, patch_gap);

figure(101);clf();
splt(2,2,1);show(abs(X),[0 1]);title('Zero Filled');
splt(2,2,4);show(abs(X0),[0 1]);title('Ground Truth');

% Solve via alternating minimisation
for i = 1:outer_iters
    % Learn dictionary and sparse code
    % (Find D, a, given X)
    disp('Learning dictionary')
    p       =   patch_get(X,patch_size,patch_gap);
    Np      =   size(p,2);
    D       =   ksvd(p(:,randperm(Np,round(Np*dict_train))), dict_size, k);
    a       =   omp(D,p,k);
    %[D,a]   =   ksvd(p, dict_size, k);
    splt(2,2,2);mcat(padarray(abs(reshape(D,[patch_size size(D,2)])),[1 1 0]));title('Current dictionary');
    
    % Estimate image
    % (Find X, give D, a)
    disp('Estimating image')    
    X = pcg(@(x)pcg_fn(x,E,M,lambda), E'*d + reshape(sqrt(lambda)*patch_adj(D*a, image_size, patch_gap),[],1), 1E-4, inner_iters);
    X = reshape(X, E.Nd);
    splt(2,2,3);show(abs(X),[0 1]);title('Current estimate');
end

%% Helper function for pcg
function x = pcg_fn(x,E,M,lambda)
    x = reshape(x,E.Nd);
    x = E.mtimes2(x) + reshape(lambda*(M.*x),[],1);
end