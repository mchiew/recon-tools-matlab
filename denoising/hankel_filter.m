function y = hankel_filter(x, r, W, iters)

% y = hankel_filter(x, W)
%
% Mark Chiew (mark.chiew@ndcn.ox.ac.uk)
% Oct 2019
%
% Input:
%       x - Nx1 vector input signal
%       r - rank of output 
%       W - Window length, s.t. 2 < W < length(x)/2
%       iters - # iterations of Cadzow's algorithm
%
% Output:
%       y - Nx1 denoised vector output signal

if nargin < 3
    W = floor(length(x)/2);
end
if nargin < 4
    iters = 1;
end

i = 0;
y = x;
K = length(x)-W;
while i < iters
    % Form Hankel matrix
    H = zeros(W,K+1);    
    for j = 1:W
        H(j,:) = y(j:j+K);
    end

    % Truncate Hankel matrix
    [u,s,v] = svd(H);
    H = u(:,1:r)*s(1:r,1:r)*v(:,1:r)';

    % Form denoised estimate by averaging anti-diagonals
    z = zeros(W,length(x));
    for j = 1:W
        z(j,j:j+K) = H(j,:);
    end
    y = reshape(sum(z,1)./sum(z~=0,1),[],1);

    % Increment counter
    i = i + 1; 
end
