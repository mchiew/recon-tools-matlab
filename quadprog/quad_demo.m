%   Load Data
load('Data_for_Harry');
load('Truth_for_Harry');

%   Prepare Data by Spatial Fourier Transform
d   =   reshape(fftdim(reshape(U_truth,64,64,[]),1:2),64*64,[])*V_truth';

%   Initialise U, V and other parameters
U   =   [];
V   =   [];
R       =   0.2;    %   Undersampling factor
niters  =   100;     %   Alternating iterations per rank
rank    =   10;     %   Final desired rank
fprintf('Undersampling Factor: %1.2f    Final Rank: %i    Iters per Rank: %i\n', R, rank, niters);

%   Prepare sampling mask
mask=   false(size(d));
idx =   randperm(prod(size(d)), round(prod(size(d))*R));
mask(idx)   =   true;

%   Main Recon Loop
%   Loop over rank, going from 1 to r
fprintf('%3s\t%-10s\t%-10s\n','Rank','Err_samp','Err_full');
for r = 1:rank
    [U, V]  =   quad_iter(d, r, niters, mask, U, V);
    est     =   U*V';
    err_samp=   norm(est(mask)-d(mask))/norm(d(mask));
    err_full=   norm(est(:)-d(:))/norm(d(:));
    fprintf('%-3i\t%-10.4f\t%-10.4f\n',r,err_samp,err_full);
end
