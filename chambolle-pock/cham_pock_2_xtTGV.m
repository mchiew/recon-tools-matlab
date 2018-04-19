function res = cham_pock_2_xtTGV(d, xfm, niter, lambda_x, lambda_t, tol, step, plt_fn)

%   Mark Chiew
%   Feb 2018
%
%   Updated to 4D (3D+Time)
%   Apr 2018
%
%   Implementation of the Chambolle-Pock Primal Dual First Order Method
%   for total generalised variation
%   From Chambole & Pock, 2011 Journal of Mathematical Imaging and Vision
%   Knoll, Bredies, Pock & Stollberger, 2011 MRM
%
%   This corresponds to Algorithm 2 in Chambolle & Pock
%   This has O(1/N^2) convergence compared to Algorithm 1
%
%   Note lambda inversely weights the data consistency term here
%   i.e., bigger lambda means more relative TGV weighting
%
%   This reformulates the algorithm to take advantage of Toeplitz embedding
%   for evaluation of xfm'*xfm

%   Initialise
if nargin < 6
tol =   1E-4;
end
if nargin < 7
step = 1;
end
if nargin < 8
plt_fn = [];
end

u   =   zeros([xfm.Nd(1:3) xfm.Nt],'single');
uu  =   zeros([xfm.Nd(1:3) xfm.Nt],'single');
v   =   zeros([xfm.Nd(1:3) xfm.Nt 4],'single');
vv  =   zeros([xfm.Nd(1:3) xfm.Nt 4],'single');
p   =   zeros([xfm.Nd(1:3) xfm.Nt 4],'single');
q   =   zeros([xfm.Nd(1:3) xfm.Nt 10],'single');
r2  =   zeros([xfm.Nd(1:3) xfm.Nt],'single');
d   =   reshape(d, [xfm.Nd(1:3) xfm.Nt]);

h   =   step;                  % this is theta in Chambolle & Pock
t   =   h/sqrt(8);          % this is tau in Chambolle & Pock
s   =   h/sqrt(8);          % this is sigma in Chambolle & Pock
g   =   1;                  % this is gamma in Chambolle & Pock
a1  =   1;
a0  =   2;

iter    =   0;
update  =   inf;

lambda      =   max(lambda_x, lambda_t);

fprintf(1, '%-5s %-16s\n', 'iter','rel. update');
while iter < niter && update > tol
    p   =   proj(p + s*(grad(u+h*(u-uu),lambda_x,lambda_t) - (v+h*(v-vv))), a1);
    q   =   proj(q + s*symgrad(v+h*(v-vv),lambda_x,lambda_t), a0);
    r2  =   prox(r2 - s*d + s*reshape(mtimes2(xfm, u+h*(u-uu)),[xfm.Nd(1:3) xfm.Nt]), s, lambda_t); 

    uu  =   u;
    u   =   u + t*(div(p,lambda_x,lambda_t)-r2);
    vv  =   v;
    v   =   v + t*(p + symdiv(q,lambda_x,lambda_t));

    %   Adaptive steps
    h   =   1/sqrt(1+2*g*t);
    t   =   t*h;
    s   =   s/h;

    if isa(plt_fn, 'function_handle')
        plt_fn(u);
    end

    update  =   norm(u(:)-uu(:))/norm(uu(:));
    iter    =   iter+1;
    fprintf(1, '%-5d %-16G\n', iter, update);
end

res.out =   u;
res.L_t =   lambda_t;
res.L_x =   lambda_x;

end

function x = proj(x,z)
    x = x./max(1,abs(x)/z);
end

function x = pgrad(x, dim, Lx, Lt)
%   Use the definition in Bredies et al.
%
%   Example 4x4 forward difference operator:
%   -1  1  0  0
%    0 -1  1  0
%    0  0 -1  1
%    0  0  0  0
    if size(x,dim) > 1
        switch dim
        case 1
            x = cat(1,diff(x,1,1),zeros(1,size(x,2),size(x,3),size(x,4)))*(Lx/Lt);
        case 2
            x = cat(2,diff(x,1,2),zeros(size(x,1),1,size(x,3),size(x,4)))*(Lx/Lt);
        case 3
            x = cat(3,diff(x,1,3),zeros(size(x,1),size(x,2),1,size(x,4)))*(Lx/Lt);
        case 4
            x = cat(4,diff(x,1,4),zeros(size(x,1),size(x,2),size(x,3),1));
        end
    else
        x   =   zeros(size(x));
    end

%{
%   Cyclic boundary conditions instead
%    1  0  0 -1
%   -1  1  0  0
%    0 -1  1  0
%    0  0 -1  1
    switch dim
    case 1
        x = diff(cat(1,x(end,:,:,:,:),x),1,1)*(Lx/Lt);
    case 2
        x = diff(cat(2,x(:,end,:,:,:),x),1,2)*(Lx/Lt);
    case 3
        x = diff(cat(3,x(:,:,end,:,:),x),1,3)*(Lx/Lt);
    case 4
        x = diff(cat(4,x(:,:,:,end,:),x),1,4);
    end
%}
end

function x = ngrad(x,dim, Lx, Lt)
%   Use the definition in Bredies et al.
%   This is also the -adjoint of the pgrad operator
%   Example 4x4 reverse difference operator:
%    1  0  0  0
%   -1  1  0  0
%    0 -1  1  0
%    0  0 -1  0
    if size(x,dim) > 1
        switch dim
        case 1
            x = cat(1,x(1,:,:,:),diff(x(1:end-1,:,:,:),1,1),-x(end-1,:,:,:))*(Lx/Lt);
        case 2
            x = cat(2,x(:,1,:,:),diff(x(:,1:end-1,:,:),1,2),-x(:,end-1,:,:))*(Lx/Lt);
        case 3
            x = cat(3,x(:,:,1,:),diff(x(:,:,1:end-1,:),1,3),-x(:,:,end-1,:))*(Lx/Lt);
        case 4
            x = cat(4,x(:,:,:,1),diff(x(:,:,:,1:end-1),1,4),-x(:,:,:,end-1));
        end
    else
        x   =   zeros(size(x));
    end

    
%{
%   Cyclic adjoint
%   -1  1  0  0
%    0 -1  1  0
%    0  0 -1  1
%    1  0  0 -1
    switch dim
    case 1
        x = cat(1,diff(x,1,1),x(1,:,:,:,:)-x(end,:,:,:,:))*(Lx/Lt);
    case 2
        x = cat(2,diff(x,1,2),x(:,1,:,:,:)-x(:,end,:,:,:))*(Lx/Lt);
    case 3
        x = cat(3,diff(x,1,3),x(:,:,1,:,:)-x(:,:,end,:,:))*(Lx/Lt);
    case 4
        x = cat(4,diff(x,1,4),x(:,:,:,1,:)-x(:,:,:,end,:));
    end
%}
end

function y = grad(x,Lx,Lt)
%   1->4
    y = cat(5,pgrad(x,1,Lx,Lt),pgrad(x,2,Lx,Lt),pgrad(x,3,Lx,Lt),pgrad(x,4,Lx,Lt));
end

function y = symgrad(x,Lx,Lt)
%   Defined as 0.5*(grad(x) + grad(x)')
%
%   Takes 4x1 gradient field and turns it into a 4x4 symmetric gradient tensor
%   Symgrad(v)  = 0.5*( [Gx(v1) Gx(v2) Gx(v3) Gx(v4)] + [Gx(v1) Gy(v1) Gz(v1) Gt(v1)] )
%                       [Gy(v1) Gy(v2) Gy(v3) Gy(v4)]   [Gx(v2) Gy(v2) Gz(v2) Gt(v2)]
%                       [Gz(v1) Gz(v2) Gz(v3) Gz(v4)]   [Gx(v3) Gy(v3) Gz(v3) Gt(v3)]
%                       [Gt(v1) Gt(v2) Gt(v3) Gt(v4)]   [Gx(v4) Gy(v4) Gz(v4) Gt(v4)]
%
%               = 0.5*( [Gx(v1)*2       Gx(v2)+Gy(v1)  Gx(v3)+Gz(v1)  Gx(v4)+Gt(v1)]
%                       [Gy(v1)+Gx(v2)  Gy(v2)*2       Gy(v3)+Gz(v2)  Gy(v4)+Gt(v2)]
%                       [Gz(v1)+Gx(v3)  Gz(v2)+Gy(v3)  Gz(v3)*2       Gz(v4)+Gt(v3)]
%                       [Gt(v1)+Gx(v4)  Gt(v2)+Gy(v4)  Gt(v3)+Gz(v4)  Gt(v4)*2     ]
%
%   Since this is symmetric, we store the upper trigangular entries only
%
%   So Symgrad([v1;v2;v3;v4])= 0.5*[Gx(v1)*2,           :1      01 05 06 07
%                                   Gy(v2)*2,           :2         02 08 09
%                                   Gz(v3)*2,           :3            03 10
%                                   Gt(v4)*2,           :4               04
%                                   Gx(v2) + Gy(v1),    :5
%                                   Gx(v3) + Gz(v1),    :6
%                                   Gx(v4) + Gt(v1),    :7
%                                   Gy(v3) + Gz(v2),    :8
%                                   Gy(v4) + Gt(v2),    :9
%                                   Gz(v4) + Gt(v3)]    :10
%
%   Symgrad([v1;v2;v3;v4]) = [Gx 0  0  0 ] * [v1]
%                            [0  Gy 0  0 ]   [v2]
%                            [0  0  Gz 0 ]   [v3]
%                            [0  0  0  Gt]   [v4]
%                            [Gy Gx 0  0 ]
%                            [Gz 0  Gx 0 ]
%                            [Gt 0  0  Gx]
%                            [0  Gz Gy 0 ]
%                            [0  Gt 0  Gy]
%                            [0  0  Gt Gz]
%   
%   4->10
    y   =   zeros([size(x,1) size(x,2) size(x,3) size(x,4) 10]);
    
    y(:,:,:,:,1) =      pgrad(x(:,:,:,:,1),1,Lx,Lt);
    y(:,:,:,:,2) =                                  pgrad(x(:,:,:,:,2),2,Lx,Lt);
    y(:,:,:,:,3) =                                                               pgrad(x(:,:,:,:,3),3,Lx,Lt);
    y(:,:,:,:,4) =                                                                                            pgrad(x(:,:,:,:,4),4,Lx,Lt);
    y(:,:,:,:,5) = 0.5*(pgrad(x(:,:,:,:,1),2,Lx,Lt)+pgrad(x(:,:,:,:,2),1,Lx,Lt));
    y(:,:,:,:,6) = 0.5*(pgrad(x(:,:,:,:,1),3,Lx,Lt)                             +pgrad(x(:,:,:,:,3),1,Lx,Lt));
    y(:,:,:,:,7) = 0.5*(pgrad(x(:,:,:,:,1),4,Lx,Lt)                                                          +pgrad(x(:,:,:,:,4),1,Lx,Lt));
    y(:,:,:,:,8) = 0.5*(                            pgrad(x(:,:,:,:,2),3,Lx,Lt) +pgrad(x(:,:,:,:,3),2,Lx,Lt));
    y(:,:,:,:,9) = 0.5*(                            pgrad(x(:,:,:,:,2),4,Lx,Lt)                              +pgrad(x(:,:,:,:,4),2,Lx,Lt));
    y(:,:,:,:,10)= 0.5*(                                                         pgrad(x(:,:,:,:,3),4,Lx,Lt) +pgrad(x(:,:,:,:,4),3,Lx,Lt));
end

function x = prox(x,s,lambda) 
    x = x./(1 + s*lambda);
end

function y = div(x,Lx,Lt)
%   Adjoint of grad
%   4->1
    y = ngrad(x(:,:,:,:,1),1,Lx,Lt) ...
      + ngrad(x(:,:,:,:,2),2,Lx,Lt) ...
      + ngrad(x(:,:,:,:,3),3,Lx,Lt) ...
      + ngrad(x(:,:,:,:,4),4,Lx,Lt);
end

function y = symdiv(x,Lx,Lt)
%   Adjoint of Symgrad
%   10->4
%
%   Symdiv([x]) = [Gx 0  0  0  Gy Gz Gt 0  0  0 ] * [x1 ]
%                 [0  Gy 0  0  Gx 0  0  Gz Gt 0 ]   [ . ]
%                 [0  0  Gz 0  0  Gx 0  Gy 0  Gt]   [ . ]
%                 [0  0  0  Gt 0  0  Gx 0  Gy Gz]   [ . ]
%                                                   [x10]
%
%   The reason the some terms might seem to be off by a factor of 0.5 compared
%   to symgrad is due to the definition of the inner product in the symgrad domain
%   See Bredies et al. for details
%
    y   =   zeros([size(x,1) size(x,2) size(x,3) size(x,4) 4]);

    y(:,:,:,:,1) = ngrad(x(:,:,:,:,1),1,Lx,Lt) + ngrad(x(:,:,:,:,5),2,Lx,Lt) + ngrad(x(:,:,:,:,6),3,Lx,Lt) + ngrad(x(:,:,:,:,7),4,Lx,Lt);
    y(:,:,:,:,2) = ngrad(x(:,:,:,:,2),2,Lx,Lt) + ngrad(x(:,:,:,:,5),1,Lx,Lt) + ngrad(x(:,:,:,:,8),3,Lx,Lt) + ngrad(x(:,:,:,:,9),4,Lx,Lt);
    y(:,:,:,:,3) = ngrad(x(:,:,:,:,3),3,Lx,Lt) + ngrad(x(:,:,:,:,6),1,Lx,Lt) + ngrad(x(:,:,:,:,8),2,Lx,Lt) + ngrad(x(:,:,:,:,10),4,Lx,Lt);
    y(:,:,:,:,4) = ngrad(x(:,:,:,:,4),4,Lx,Lt) + ngrad(x(:,:,:,:,7),1,Lx,Lt) + ngrad(x(:,:,:,:,9),3,Lx,Lt) + ngrad(x(:,:,:,:,10),4,Lx,Lt);
end
