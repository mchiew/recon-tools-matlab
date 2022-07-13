function out = lLORA(input, r1, r2, patch)

% tensor LOw RAnk Denoising for MRSI data
%
%   inputs:
%           input   - space-time MRSI data
%           r1      - rank threshold for locally low-rank constraint
%           r2      - rank threshold for Hankel filtering constraint
%           patch   - [patch_size, patch_gap]
%                      patch_size is a scalar, indicating width of patch
%                      box in every dimension (default 3 -> [3,3,3] patch)
%                      patch_gap is a scalar, indicating distance between
%                      patches (default 1)
%   output:
%           out     - space-time denoised MRSI data

if nargin < 4
    % Default to 3x3x3 patches, with patch gap 1
    patch = [3,1];
end

input   = reshape(input,size(input,1),size(input,2),[],size(input,ndims(input)));
dims    = size(input);
out     = zeros(dims);

% Extract 4D-patches
M = false(dims(1:3));
M(1:patch(2):end,1:patch(2):end)=true;
[ii,jj,kk] = ind2sub(size(M),find(M));

idx_i = 0:min(patch(1),dims(1))-1;
idx_j = 0:min(patch(1),dims(2))-1;
idx_k = 0:min(patch(1),dims(3))-1;

% Loop over patches
M = zeros(dims(1:3));
for p = 1:length(ii)
    i = mod(ii(p)+idx_i-1, dims(1)) + 1;
    j = mod(jj(p)+idx_j-1, dims(2)) + 1;
    k = mod(kk(p)+idx_k-1, dims(3)) + 1;
   
    z = LORA(input(i,j,k,:), r1, r2);
    
    out(i,j,k,:)  = out(i,j,k,:) + reshape(z,length(idx_i),length(idx_j),length(idx_k),[]);
    M(i,j,k)    = M(i,j,k) + 1;
end

out = reshape(out./M, dims);

