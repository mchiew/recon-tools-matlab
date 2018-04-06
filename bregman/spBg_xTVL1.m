function u = spBg_xTVL1(d, xfm, W, niter, mu, lambda, gamma)

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
niter_inner  =   5;

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
%mu      =   mu*norm(d);
%lambda  =   lambda*norm(d);
%gamma   =   gamma*norm(d);

%   Initialise variables
u   =   zeros([xfm.Nd(1:3) xfm.Nt]);
dx  =   g(u,-1,1);
dy  =   g(u,-1,2);
w   =   W*u;
bx  =   0;
by  =   0;
bw  =   w; 
f   =   d;

%   Main loops
for i  = 1:niter_outer
for ii = 1:niter_inner
   
    %   Solve u-subproblem
    [u,~,relres]=  minres(@(x) reshape(mu*mtimes2(xfm,x),[],1) + reshape(lambda*(g(g(reshape(x,[xfm.Nd xfm.Nt]),-1,1),1,1) + g(g(reshape(x,[xfm.Nd xfm.Nt]),-1,2),1,2)),[],1) + gamma*x,...
                   reshape(mu*(xfm'.*f) + lambda*(g(dx-bx,1,1) + g(dy-by,1,2)) + gamma*(W'*(w-bw)),[],1),...
                   1E-6, 10, [], [], u(:));
    u           =   reshape(u, [xfm.Nd(1:3) xfm.Nt]);


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
    fprintf(1, 'Iter: %03d, Objective: %f, DataConsistency: %f\n', i, objective(d, u, xfm, W, mu, lambda, gamma), (mu/2)*sum(reshape(abs(xfm*u-d).^2,[],1)));
end

%   The main objective function
function res = objective(d, x, xfm, W, mu, lambda, gamma)
    res =   lambda*(sqrt(L1(g(x,-1,1)).^2 + L1(g(x,-1,2)).^2)) + gamma*L1(W*x) + (mu/2)*sum(reshape(abs(xfm*x-d).^2,[],1));

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
