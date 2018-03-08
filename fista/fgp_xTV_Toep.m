function est = fgp_xTV_Toep(xfm, T, samp, lambda, maxIter, tol)
%
%   est = fgp_xTV(xfm, T, samp, lambda, [maxIter], [tol])
%
%   fgp_xTV 
%   Spatial Total Variation Constrained Fast Gradient Projection
%   From Beck & Teboulle, 2009 IEEE Trans Im Proc
%
%   Uses isotropic TV by default
%   For anisotropic, use the prox_TVL1 subfunction instead
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

tv_iters    =   100;
tv_tol      =   1E-6;

est     =   zeros([xfm.Nd(1:2) xfm.Nt]);
est0    =   est;
y       =   zeros(xfm.msize);

L       =   abs(T(1));
step    =   1/L;
iter    =   1;
t1      =   1;
update  =   inf;

d       =   xfm'*samp;

%===========================================================
%   Main Iteration Loop
%===========================================================
fprintf(1, '%-5s %-16s %-16s %-16s\n', 'Iter','L2','TV','Cost');
while iter <= maxIter && update > tol

    est     =   denoise(reshape(y - (2/L)*(mtimes_Toeplitz(xfm,T,y) - d), [xfm.Nd(1:2), xfm.Nt]), 2*lambda/L, tv_iters, tv_tol);

    %   Accleration
    t2      =   (1+sqrt(1+4*t1^2))/2;
    y       =   reshape(est + ((t1-1)/t2)*(est - est0), xfm.msize);

    %   Display iteration summary data
    fprintf(1, '%-5d %-16s %-16s %-16s\n', iter, '-', isoTV(est), '-');

    %   Update and iteration counter
    update  =   norm(est(:)-est0(:))/norm(est0(:));
    iter    =   iter + 1;
    est0    =   est;
    t1      =   t2;
end


end

%   Denoising sub-problem
function y = denoise(x, lambda, iters, tol)
    p   =   zeros(size(x,1), size(x,2), size(x,3), 2);
    p0  =   p;
    r   =   p;

    iter    =   1;
    t1      =   1;
    update  =   inf;

    while iter <= iters && update > tol
        p   =   proj_isoTV(r + (1/(8*lambda))*Dadj(x - lambda*Dfwd(r)));
        t2  =   (1+sqrt(1+4*t1^2))/2;
        r   =   p + ((t1-1)/t2)*(p - p0);
        
        update  =   norm(p(:)-p0(:))/norm(p0(:));
        iter    =   iter + 1;
        p0      =   p;
        t1      =   t2;
    end
    
    y   =  x - lambda*Dfwd(p);
end

%   Finite difference and adjoint
%   Using periodic boundary conditions
function y = Dfwd(x)
%   2->1
    y   =   zeros(size(x(:,:,:,1)));
    y   =   y - cat(1,diff(x(:,:,:,1),1,1),x(1,:,:,1)-x(end,:,:,1));
    y   =   y - cat(2,diff(x(:,:,:,2),1,2),x(:,1,:,2)-x(:,end,:,2));
end
function y = Dadj(x)
%   1->2
    y           =   zeros([size(x,1) size(x,2) size(x,3) 2]);
    y(:,:,:,1)  =   diff(cat(1,x(end,:,:),x),1,1);
    y(:,:,:,2)  =   diff(cat(2,x(:,end,:),x),1,2);
end

%   Isotropic TV Projection
function x = proj_isoTV(x)
    x = bsxfun(@rdivide,x,max(1,sqrt(sum(abs(x).^2,4))));
end
%   Anisotropic TV-L1 Projection
function x = proj_TVL1(x)
    x = x./max(1,abs(x));
end

%   Iso TV norm
function y = isoTV(x)
    y = sum(sqrt(sum(reshape(abs(Dadj(x)).^2,[],2),2)));
end
