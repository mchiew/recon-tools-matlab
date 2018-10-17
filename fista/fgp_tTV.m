function est = fgp_tTV(xfm, samp, lambda, step, maxIter, tol)
%
%   est = fgp_tTV(xfm, samp, lambda, [step], [maxIter], [tol])
%
%   fgp_tTV 
%   Temporal Total Variation Constrained Fast Gradient Projection
%   From Beck & Teboulle, 2009 IEEE Trans Im Proc
%
%   Note that this involves solving nested optimisation problems
%   Each iteration of the algorithm requires solving the TV
%   denoising problem, which itself requires an iterative solution

%   Each iteration must solve the following TV-denoising problem:
%   Denoise(X - (2/L)*xfm'*(xfm*X - d), 2*lambda/L)
%   where L is an upper bound of the Lipschitz constant of 
%   the gradient of the data consistency term 
%===========================================================
%   Initialization
%===========================================================
if nargin < 6
    tol =   1E-4;
end
if nargin < 5
    maxIter = 100;
end
if nargin < 4
    step = 1;
end

tv_iters    =   50;
tv_tol      =   1E-4;

msize   =   xfm.msize;
est     =   zeros(msize);
est0    =   zeros(msize);
y       =   zeros(msize);

L       =   1/step;
iter    =   1;
t1      =   1;
update  =   inf;

%===========================================================
%   Main Iteration Loop
%===========================================================
fprintf(1, '%-5s %-16s %-16s %-16s %-16s\n', 'Iter','Update','L2','L1','Cost');
while iter <= maxIter && update > tol

    est     =   denoise(y - (2/L)*(xfm'*(xfm*y - samp)), 2*lambda/L, tv_iters, tv_tol);

    %   Accleration
    t2      =   (1+sqrt(1+4*t1^2))/2;
    y       =   est + ((t1-1)/t2)*(est - est0);

    %   Update and iteration counter
    update  =   norm(est(:)-est0(:))/norm(est0(:));
    iter    =   iter + 1;
    est0    =   est;
    t1      =   t2;

    %   Display iteration summary data
    fprintf(1, '%-5d %-16s %-16s %-16s %-16s\n', iter-1, update, '-', '-', '-');
end


end

%   Denoising sub-problem
function y = denoise(x, lambda, iters, tol)
    p   =   zeros(size(x,1), size(x,2)-1);
    p0  =   zeros(size(x,1), size(x,2)-1);
    r   =   zeros(size(x,1), size(x,2)-1);

    iter    =   1;
    p0      =   p;
    t1      =   1;
    update  =   inf;

    while iter <= iters && update > tol
        p   =   proj_TV(r + (1/(8*lambda))*Dadj(x - lambda*Dfwd(r)));
        t2  =   (1+sqrt(1+4*t1^2))/2;
        r   =   p + ((t1-1)/t2)*(p - p0);
        
        update  =   norm(p(:)-p0(:))/norm(p0(:));
        iter    =   iter + 1;
        p0      =   p;
        t1      =   t2;
    end
    
    y   =  x - lambda*Dfwd(p);
end

%   Finite difference and adjoint as defined in Beck & Teboulle
function y = Dfwd(x)
    y   =   -[x(:,1) diff(x,1,2) -x(:,end)];
end
function y = Dadj(x)
    y   =   diff(x,1,2);
end

%   TV Projection
function x = proj_TV(x)
    x = x./max(1,abs(x)); 
end
