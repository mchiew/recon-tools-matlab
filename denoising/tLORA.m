function out = tLORA(input, r)

% tensor LOw RAnk Denoising for MRSI data
%
%   inputs:
%           input   - space-time MRSI data
%           r1      - rank threshold for tensor LR constraint
%
%   output:
%           out     - space-time denoised MRSI data

dims = size(input);
out  = reshape(input,[],size(input,ndims(input)));

% Construct Low-rank tensor from Hankel matrices
W = size(out,2)/2;
K = size(out,2)-W;
H = zeros(size(out,1),W,W+1);
% Loop over voxels
for x = 1:size(out,1)
    for i = 1:W
        H(x,i,:) = out(x,i:i+K);
    end
end

% Truncate unfolded tensor
H = svd_trunc(H,r);

% Extract output signal
for x = 1:size(out,1)
    out(x,:) = [reshape(H(x,1,:),1,[]) reshape(H(x,2:end,end),1,[])]; 
end

out = reshape(out, dims);
