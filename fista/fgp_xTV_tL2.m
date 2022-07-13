function est = fgp_xTV_tL2(E, dd, lambda_x, lambda_t, maxIter, tol)
%
%   est = fgp_xTV_tL2(E, dd, lambda_x, lambda_t, [maxIter], [tol])
%
%   fgp_xTV_tL2
%   Spatial Total Variation Constrained Fast Gradient Projection
%   From Beck & Teboulle, 2009 IEEE Trans Im Proc
%
%   With an extra L2 constraint on temporal smoothness
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

%   
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

est     =   zeros([E.Nd E.Nt]);
est0    =   est;
y       =   zeros(E.msize);

%   Use the power method to find the Lipschitz smoothness
q   =   randn(E.msize);
N   =   0;
ii  =   0;
while abs(norm(q(:)) - N)/N > 1E-4 && ii < 100
    N   =   norm(q(:)); 
    if nargout == 0
        disp(1./N);
    end
    q   =   E.mtimes2(q/N) + reshape(E.R1(reshape(q/N,[E.Nd E.Nt]), [0,0,0, lambda_t]), E.msize);
    ii  =   ii+1;
end
L   =   1.1*norm(q(:));


iter    =   1;
t1      =   1;
update  =   inf;


% Assumes dd = E'*d;

%===========================================================
%   Main Iteration Loop
%===========================================================
fprintf(1, '%-5s %-16s %-16s %-16s\n', 'Iter','L2','TV','Cost');
while iter <= maxIter && update > tol

    est     =   denoise(reshape(y - (1/L)*(E.mtimes2(y) + reshape(E.R1(reshape(y,[E.Nd E.Nt]),[0 0 0 lambda_t]), E.msize) - dd), [E.Nd, E.Nt]), lambda_x/L, tv_iters, tv_tol);

    %   Accleration
    t2      =   (1+sqrt(1+4*t1^2))/2;
    y       =   reshape(est + ((t1-1)/t2)*(est - est0), E.msize);

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
    p   =   zeros([size(x) 3]);
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
    y   =   zeros(size(x(:,:,:,:,1)));
    y   =   y - cat(1,diff(x(:,:,:,:,1),1,1),x(1,:,:,:,1)-x(end,:,:,:,1));
    y   =   y - cat(2,diff(x(:,:,:,:,2),1,2),x(:,1,:,:,2)-x(:,end,:,:,2));
    y   =   y - cat(3,diff(x(:,:,:,:,3),1,3),x(:,:,1,:,3)-x(:,:,end,:,3));
end
function y = Dadj(x)
%   1->2
    y               =   zeros([size(x) 3]);
    y(:,:,:,:,1)    =   diff(cat(1,x(end,:,:,:),x),1,1);
    y(:,:,:,:,2)    =   diff(cat(2,x(:,end,:,:),x),1,2);
    y(:,:,:,:,3)    =   diff(cat(3,x(:,:,end,:),x),1,3);
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
