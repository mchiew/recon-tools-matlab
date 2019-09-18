function p = patch_get(I, N, d)

% Patch operator for getting image patches
% Mark Chiew (mark.chiew@ndcn.ox.ac.uk)
%
% p = patch_get(I, N, d)
%
% inputs:   I   - input image
%           N   - dimension of patch [Nx Ny]
%           d   - delta between patch corners
% outputs:  p   - array of linearised patches

M = false(size(I));
M(1:d:end,1:d:end)=true;

[ii,jj] = ind2sub(size(I),find(M));

idx_i = 0:N(1)-1;
idx_j = 0:N(2)-1;

p   = zeros(prod(N), length(ii));

for k = 1:length(ii)
    i = mod(ii(k)+idx_i-1, size(I,1)) + 1;
    j = mod(jj(k)+idx_j-1, size(I,2)) + 1;
    p(:,k) = reshape(I(i,j),[],1);
end
