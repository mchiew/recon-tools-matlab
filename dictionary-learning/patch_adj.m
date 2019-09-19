function [I M] = patch_adj(p, N, d)

% Patch operator for putting image patches back (adjoint of patch_get)
% Mark Chiew (mark.chiew@ndcn.ox.ac.uk)
%
% I = patch_put(p, N, d)
%
% inputs:   p   - array of linearised patches
%           N   - dimension of image [Nx, Ny]
%           d   - delta between patch centres
% outputs:  I   - output image
%           M   - normalisation needed for inversion

M = false(N);
M(1:d:end,1:d:end)=true;

[ii,jj] = ind2sub(N,find(M));

idx_i = 0:sqrt(size(p,1))-1;
idx_j = 0:sqrt(size(p,1))-1;

I   = zeros(N);
M   = zeros(N);

for k = 1:size(p,2)
    i = mod(ii(k)+idx_i-1,N(1))+1;
    j = mod(jj(k)+idx_j-1,N(2))+1;
    I(i,j) = I(i,j) + reshape(p(:,k),sqrt(size(p,1)),[]);
    M(i,j) = M(i,j) + 1;   
end