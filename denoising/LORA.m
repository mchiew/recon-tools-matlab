function out = LORA(input, r1, r2)

% LOw RAnk Denoising for MRSI data
%
%   inputs:
%           input   - space-time MRSI data
%           r1      - rank threshold for space-time LR constraint
%           r2      - rank threshold for Hankel filtering constraint
%
%   output:
%           out     - space-time denoised MRSI data

dims = size(input);
out  = reshape(input,[],size(input,ndims(input)));

% Space-time Low Rank enforcement
out = svd_trunc(out, r1);

% Hankel matrix Low-Rank enforcement
W = size(out,2)/2;
K = size(out,2)-W;
H = zeros(W,W+1);
% Loop over voxels
for x = 1:size(out,1)
    for i = 1:W
        H(i,:) = out(x,i:i+K);
    end
    H = svd_trunc(H, r2);
    out(x,:) = [H(1,:) H(2:end,end).'];
end

out = reshape(out, dims);