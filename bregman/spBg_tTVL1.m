function u = spBg_tTVL1(d, xfm, W, niter, mu, lambda, gamma)

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
%{
mu      =   mu*norm(d);
lambda  =   lambda*norm(d);
gamma   =   gamma*norm(d);
%}

%   Initialise variables
u   =   zeros(xfm.msize);
dt  =   g(u,-1,2);
w   =   W*u;
bt  =   0;
bw  =   w; 
f   =   d;


%   Main loops
for i  = 1:niter_outer
    dd  =   xfm'*f;
for ii = 1:niter_inner
   
    %   Solve u-subproblem
    [u,~,relres]=   minres(@(x) Afun(reshape(x, xfm.msize), xfm, mu, lambda, gamma),...
                    reshape(mu*dd + lambda*g(dt-bt,1,2) + gamma*(W'*(w-bw)),[],1),...
                    1E-6, 50, [], [], u(:));
    u           =   reshape(u, xfm.msize);

    %   Update auxilliary variables
    s   =   abs(g(u,-1,2) + bt);
    dt  =   max(s - 1/lambda,0).*((g(u,-1,2)+bt)./(s+(s<1/lambda)));
    w   =   shrink(W*u + bw, 1/gamma); 
    bt  =   bt + (g(u,-1,2) - dt);
    bw  =   bw + (W*u - w);
end
    %   Update fidelity term for outer loop
    f   =   f + d - xfm*u;
    fprintf(1, 'Iter: %03d, Objective: %f, DataConsistency: %f\n', i, objective(d, u, xfm, W, mu, lambda, gamma), (mu/2)*sum(reshape(abs(xfm*u-d).^2,[],1)));
end

%   The main objective function
function res = objective(d, x, xfm, W, mu, lambda, gamma)
    res =   lambda*L1(g(x,-1,2)) + gamma*L1(W*x) + (mu/2)*sum(reshape(abs(xfm*x-d).^2,[],1));

%   Matrix for solving the u-subproblem
function res = Afun(x, E, mu, lambda, gamma)
    res =   reshape(mu*mtimes2(E,x) + lambda*g(g(x,-1,2),1,2) + gamma*x,[],1);

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
