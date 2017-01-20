function u = spBg_TV_L1(d, xfm, W, im_size, niter, mu, lambda, gamma)

%   Mark Chiew  
%   Jan 2017
%
%   Implementation of the Split-Bregman iteration scheme for L1-Regularised problems
%   Adapted from Goldstein & Osher, 2008 J Sci Am Imaging
%
%   This will solve a problem of the form:
%
%   min{u} lambda*TV(u) + gamma*|W*u|_1 + (mu/2)|xfm*u - d|_2
%
%   where u is the estimate, TV is the isotropic total variation norm, W is 
%   a Wavelet transform, mu is a Lagrange multiplier, xfm is the measurement operator, 
%   and d is the measured raw k-space data
%
%   Seems to work best when gamma < lambda ~ mu


niter_outer  =   niter;
niter_inner  =   50;

if nargin < 6
mu      =   1;
end
if nargin < 7
lambda  =   mu;
end
if nargin < 8
gamma   =   mu/10;
end

%   Normalise mu, lambda and gamma relative to norm(d)
mu      =   mu*norm(d);
lambda  =   lambda*norm(d);
gamma   =   gamma*norm(d);

%   Initialise variables
u   =   zeros(im_size);
dx  =   g(u,-1,1);
dy  =   g(u,-1,2);
w   =   W*u;
bx  =   0;
by  =   0;
bw  =   w; 
f   =   d;

%   Pre-compute helper matrix to solve the inner u-subproblem
%   Note that this is approximate in some cases because we're using the FFT to 
%   speed up the solution of the Afun-system, but it is only exact when the 
%   data are sampled on a Cartesian grid (i.e. FFT-mode in xfm)
%   For non-uniform sampling (i.e. NUFFT-mode in xfm), this fast exact
%   solution isn't practical, but we just use an approximate form 
%   Alternatively, for precision, you can use the symmlq solver commented
%   out below
A   =   zeros(im_size);
A(1,1)  =   4;
A(1,2)  =   -1;
A(2,1)  =   -1;
A(end,1)=   -1;
A(1,end)=   -1;
if strcmp(xfm.mode, 'NUFFT')
    A   =  mu + lambda*fftshift(fftshift(fft2(A),1),2) + gamma;
else
    A   =  lambda*fftshift(fftshift(fft2(A),1),2) + gamma;
    A(xfm.mask) = A(xfm.mask) + mu; 
end

%   Main loops
for i  = 1:niter_outer
for ii = 1:niter_inner
   
    %   Solve u-subproblem
    %   Can use symmlq because Afun is symmetric
    %   This is slow because it's indirect, but it's technically more precise
    %[u,~,relres]=  reshape(symmlq(@(x,mode) Afun(reshape(x, im_size), xfm, mu, lambda, gamma),...
    %               reshape(mu*(xfm'.*f) + lambda*(g(dx-bx,1,1) + g(dy-by,1,2)) + gamma*(W'*(w-bw)),[],1),...
    %               1E-9, 500, [], [], u(:)), im_size);

    %   Solve u-subproblem
    u   =   ifft2c(fft2c(mu*(xfm'.*f) + lambda*(g(dx-bx,1,1) + g(dy-by,1,2)) + gamma*(W'*(w-bw)))./A);

    %   Update auxilliary variables
    s   =   sqrt(abs(g(u,-1,1) + bx).^2 + abs(g(u,-1,2) + by).^2);
    dx  =   max(s - 1/lambda,0).*((g(u,-1,1)+bx)./(s+(s<1/lambda)));
    dy  =   max(s - 1/lambda,0).*((g(u,-1,2)+by)./(s+(s<1/lambda)));
    w   =   shrink(W*u + bw, 1/gamma); 
    bx  =   bx + (g(u,-1,1) - dx);
    by  =   by + (g(u,-1,2) - dy);
    bw  =   bw + (W*u - w);
end
    %   Update fidelity term for outer loop
    f   =   f + d - xfm*u;
    figure(100);show(abs(u),[]);
    fprintf(1, 'Iter: %03d, Objective: %f, DataConsistency: %f\n', i, objective(f, u, xfm, W, mu, lambda, gamma)/norm(d).^2, norm(reshape(xfm*u-d,[],1)));
end

%   The main objective function
function res = objective(d, x, xfm, W, mu, lambda, gamma)
    res =   lambda*(sqrt(L1(g(x,-1,1)).^2 + L1(g(x,-1,2)).^2)) + gamma*L1(W*x) + (mu/2)*sum(reshape(abs(xfm*x-d).^2,[],1));

%   Matrix for solving the u-subproblem
function res = Afun(x, E, mu, lambda, gamma)
    res =   mu*(E'.*(E*x)) + lambda*(g(g(x,-1,1),1,1) + g(g(x,-1,2),1,2)) + gamma*x;
    res =   reshape(res,[],1);

%   L1-norm
function res = L1(input)
    res =   sum(abs(reshape(input, [], 1)));

%   Shrinkage operator for complex input
function res = shrink(input, thresh)
    res =   exp(1j*angle(input)).*max(abs(input)-thresh, 0);

%   Finite difference operator
%   dir = -1, forward
%   dir = 1, adjoint
function res = g(x, dir, dim)
    res =   x - circshift(x,dir,dim);
