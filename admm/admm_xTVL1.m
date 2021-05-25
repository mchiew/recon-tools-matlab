function x = admm_xTVL1(E, dd, lambda, maxIter, tol)
%
%   x = admm_xTVL1(E, dd, lambda, [maxIter], [tol])
%
%   ADMM to solve the anisotropic TV
%===========================================================
%   Initialization
%===========================================================
if nargin < 4
    maxIter = 100;
end
if nargin < 5
    tol =   1E-3;
end

p   =   1;
t   =   2;
a   =   1;
i   =   0;

x   =   zeros([E.Nd E.Nt]);
z   =   zeros([E.Nd E.Nt, 3]);
u   =   zeros([E.Nd E.Nt, 3]);

% Assumes dd = E'*d;
dd  =   dd(:);

%===========================================================
%   Main Iteration Loop
%===========================================================
fprintf(1, '%-5s %-16s\n', 'Iter','Primal+Dual Residual');
while i < maxIter

    %   x-update
    [x,~]   =   pcg(@(x,mode) reshape(E.mtimes2(reshape(x,[E.Nd E.Nt])) + E.R1(reshape(x,[E.Nd E.Nt]),[p,p,p,0]),[],1), dd + reshape(p*fd_adj(z-u),[],1), 1E-4, 100, [], []);
    x       =   reshape(x, [E.Nd E.Nt]);

    %   z-update
    z0  =   z;
    z   =   shrink(a*fd_fwd(x) + (1-a)*(z0) + u, lambda/p);

    %   u-update
    u   =   u + a*fd_fwd(x) + (1-a)*(z0) - z;

    %   Update and iteration counter
    i   =   i + 1;
    r   =   (fd_fwd(x) - z)/max(norm(reshape(fd_fwd(x),[],1)), norm(z(:)));
    s   =   (p*fd_adj(z-z0))/norm(reshape(fd_adj(u),[],1));
    if norm(r(:)) > 10*norm(s(:))
        p   =   p*t;
        u   =   u/t;
    elseif norm(s(:)) > 10*norm(r(:))
        p   =   p/t;
        u   =   u*t;
    end
    
    %   Display iteration summary data
    relres  =   norm(r(:))^2 + norm(s(:))^2;
    fprintf(1, '%-5d %-16f\n', i, relres);
    
    if relres < tol
        break;
    end
    
end


end

%   Finite Difference Operator
function y = fd_fwd(x)
    y   =   zeros([size(x), 3]);
    y(:,:,:,:,1)  =   cat(1, diff(x,1,1), x(1,:,:,:)-x(end,:,:,:));
    y(:,:,:,:,2)  =   cat(2, diff(x,1,2), x(:,1,:,:)-x(:,end,:,:));
    y(:,:,:,:,3)  =   cat(3, diff(x,1,3), x(:,:,1,:)-x(:,:,end,:));
end

%   Finite Difference Adjoint
function x = fd_adj(y)
    x   =   zeros(size(y(:,:,:,:,1)));
    x   =   x + cat(1, y(end,:,:,:,1)-y(1,:,:,:,1), -1*diff(y(:,:,:,:,1),1,1));
    x   =   x + cat(2, y(:,end,:,:,2)-y(:,1,:,:,2), -1*diff(y(:,:,:,:,2),1,2));
    x   =   x + cat(3, y(:,:,end,:,3)-y(:,:,1,:,3), -1*diff(y(:,:,:,:,3),1,3));
end

%   Isotropic TV Projection
function x = proj_isoTV(x)
    x = bsxfun(@rdivide,x,max(1,sqrt(sum(abs(x).^2,4))));
end
%   Anisotropic TV-L1 Projection
function x = proj_TVL1(x)
    x = x./max(1,abs(x));
end

%   Isotropic TV shrinkage
function z = shrink(z, lambda)
    s   =   sqrt(sum(abs(z).^2,5));
    z   =   (max(abs(s)-lambda,0)./(s+eps)).*z;
end
