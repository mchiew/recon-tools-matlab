function I = patch_put(p, N, d)

% Patch operator for putting image patches back (inverse of patch_get)
% Mark Chiew (mark.chiew@ndcn.ox.ac.uk)
%
% I = patch_put(p, N, d)
%
% inputs:   p   - array of linearised patches
%           N   - dimension of image [Nx, Ny]
%           d   - delta between patch centres
% outputs:  I   - output image

[I, M]  = patch_adj(p, N, d);
I       = I./M;