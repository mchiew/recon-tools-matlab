function u = spBg_xtTVL1(d, xfm, W, niter, mu, lambda, gamma)

%   Mark Chiew  
%   Jan 2017
%   Modified Apr 2018
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
%
%   lambda = [Lx, Ly, Lz, Lt]


niter_outer =   niter;
niter_inner =   5;
niter_ls    =   10;

if nargin < 6
mu      =   1;
end
if nargin < 7
lambda  =   [0 0];
end
if nargin < 8
gamma   =   mu/10;
end

%   Initialise variables
u   =   zeros([xfm.Nd xfm.Nt]);

dx  =   g(u,-1,1);
dy  =   g(u,-1,2);
dz  =   g(u,-1,3);
dt  =   g(u,-1,4);

bx  =   0;
by  =   0;
bz  =   0;
bt  =   0;

w   =   W*u;
bw  =   w; 

f   =   d;
    
Lx  =   lambda(1);
Ly  =   lambda(2);
Lz  =   lambda(3);
Lt  =   lambda(4);

%   Main loops
for i  = 1:niter_outer
    dd  =   xfm'.*f;
for ii = 1:niter_inner
   
    %   Solve u-subproblem
    [u,~,relres]=   minres(@(x) Afun(reshape(x, [xfm.Nd xfm.Nt]), xfm, mu, [Lx Ly Lz Lt], gamma),...
                    reshape(mu*dd + Lx*g(dx-bx,1,1) + Ly*g(dy-by,1,2) + Lz*g(dz-bz,1,3) + Lt*g(dt-bt,1,4) + gamma*(W'*(w-bw)),[],1),...
                    1E-6, niter_ls, [], [], u(:));
    u           =   reshape(u, [xfm.Nd xfm.Nt]);

    %   Update auxilliary variables
    s   =   sqrt(abs(Lx*g(u,-1,1)+bx).^2 + abs(Ly*g(u,-1,2)+by).^2 + abs(Lz*g(u,-1,3)+bz).^2 + abs(Lt*g(u,-1,4)+bt).^2);
    dx  =   max(s - 1,0).*((g(u,-1,1)+bx)./(s+(s<1)));
    dy  =   max(s - 1,0).*((g(u,-1,2)+by)./(s+(s<1)));
    dz  =   max(s - 1,0).*((g(u,-1,3)+bz)./(s+(s<1)));
    dt  =   max(s - 1,0).*((g(u,-1,4)+bt)./(s+(s<1)));
    w   =   shrink(W*u + bw, 1/gamma); 
    bx  =   bx + (g(u,-1,1) - dx);
    by  =   by + (g(u,-1,2) - dy);
    bz  =   bz + (g(u,-1,3) - dz);
    bt  =   bt + (g(u,-1,4) - dt);
    bw  =   bw + (W*u - w);
end
    %   Update fidelity term for outer loop
    f   =   f + d - xfm*u;
    fprintf(1, 'Iter: %03d, Objective: %f, DataConsistency: %f\n', i, objective(d, u, xfm, W, mu, [Lx,Ly,Lz,Lt], gamma), (mu/2)*sum(reshape(abs(xfm*u-d).^2,[],1)));
end

%   The main objective function
function res = objective(d, x, xfm, W, mu, L, gamma)
    res =   sqrt(L1(L(1)*g(x,-1,1)).^2 + L1(L(2)*g(x,-1,2)).^2 + L1(L(3)*g(x,-1,3)).^2 + L1(L(4)*g(x,-1,4)).^2) + gamma*L1(W*x) + (mu/2)*sum(reshape(abs(xfm*x-d).^2,[],1));

%   Matrix for solving the u-subproblem
function res = Afun(x, E, mu, L, gamma)
    res =   reshape(mu*mtimes2(E,x) + L(1)*g(g(x,-1,1),1,1) + L(2)*g(g(x,-1,2),1,2) + L(3)*g(g(x,-1,3),1,3) + L(4)*g(g(x,-1,4),1,4) + gamma*x,[],1);

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
