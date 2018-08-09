function res = cham_pock_1_xtTGV(d, xfm, niter, lambda_x, lambda_t, tol, plt_fn)

%   Mark Chiew
%   Feb 2018
%
%   Implementation of the Chambolle-Pock Primal Dual First Order Method
%   for total generalised variation
%   From Chambole & Pock, 2011 Journal of Mathematical Imaging and Vision
%   Knoll, Bredies, Pock & Stollberger, 2011 MRM
%
%   This corresponds to Algorithm 1 in Chambolle & Pock

%   Note lambda inversely weights the data consistency term here
%   i.e., bigger lambda means more relative TGV weighting

%   This reformulates the algorithm to take advantage of Toeplitz embedding
%   for evaluation of xfm'*xfm

%   Initialise
if nargin < 7
tol =   1E-4;
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

t   =   1/abs(T(1));
s   =   1/abs(T(1));
a1  =   1;
a0  =   2;

lambda      =   max(lambda_x, lambda_t);

fprintf(1, '%-5s %-16s\n', 'iter','rel. update');
for i = 1:niter
    p   =   proj(p + s*(grad(2*u-uu,lambda_x,lambda_t) - (2*v-vv)), a1);
    q   =   proj(q + s*symgrad(2*v-vv,lambda_x,lambda_t), a0);
    r2  =   prox(r2 - s*d + s*reshape(mtimes_Toeplitz(xfm, T, 2*u-uu),[xfm.Nd(1:2) xfm.Nt]), s, lambda_t); 

    uu  =   u;
    u   =   u + t*(div(p,lambda_x,lambda_t)-r2);
    vv  =   v;
    v   =   v + t*(p + symdiv(q,lambda_x,lambda_t));

    if isa(plt_fn, 'function_handle')
        plt_fn(u);
    end

    update = norm(u(:)-uu(:))/norm(uu(:));
    fprintf(1, '%-5d %-16G\n', i, update);
    if update < tol
        break;
    end
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
%{
    if dim == 1
        x = cat(1,diff(x,1,1),zeros(1,size(x,2),size(x,3),size(x,4)))*(Lx/Lt);
    elseif dim == 2 
        x = cat(2,diff(x,1,2),zeros(size(x,1),1,size(x,3),size(x,4)))*(Lx/Lt);
    elseif dim == 3 
        x = cat(3,diff(x,1,3),zeros(size(x,1),size(x,2),1,size(x,4)));
    end
%}

%   Cyclic boundary conditions instead
%    1  0  0 -1
%   -1  1  0  0
%    0 -1  1  0
%    0  0 -1  1
    if dim == 1
        x = diff(cat(1,x(end,:,:,:),x),1,1)*(Lx/Lt);
    elseif dim == 2 
        x = diff(cat(2,x(:,end,:,:),x),1,2)*(Lx/Lt);
    elseif dim == 3 
        x = diff(cat(3,x(:,:,end,:),x),1,3);
    end
end

function x = ngrad(x,dim, Lx, Lt)
%   Use the definition in Bredies et al.
%   This is also the -adjoint of the pgrad operator
%   Example 4x4 reverse difference operator:
%    1  0  0  0
%   -1  1  0  0
%    0 -1  1  0
%    0  0 -1  0
%{
    if dim == 1
        x = cat(1,x(1,:,:,:),diff(x(1:end-1,:,:,:),1,1),-x(end-1,:,:,:))*(Lx/Lt);
    elseif dim == 2
        x = cat(2,x(:,1,:,:),diff(x(:,1:end-1,:,:),1,2),-x(:,end-1,:,:))*(Lx/Lt);
    elseif dim == 3
        x = cat(3,x(:,:,1,:),diff(x(:,:,1:end-1,:),1,3),-x(:,:,end-1,:));
    end
%}
    
%   Cyclic adjoint
%   -1  1  0  0
%    0 -1  1  0
%    0  0 -1  1
%    1  0  0 -1
    if dim == 1
        x = cat(1,diff(x,1,1),x(1,:,:,:)-x(end,:,:,:))*(Lx/Lt);
    elseif dim == 2 
        x = cat(2,diff(x,1,2),x(:,1,:,:)-x(:,end,:,:))*(Lx/Lt);
    elseif dim == 3 
        x = cat(3,diff(x,1,3),x(:,:,1,:)-x(:,:,end,:));
    end
end

function y = grad(x,Lx,Lt)
%   1->3
    y = cat(4,pgrad(x,1,Lx,Lt),pgrad(x,2,Lx,Lt),pgrad(x,3,Lx,Lt));
end

function y = symgrad(x,Lx,Lt)
%   Defined as 0.5*(grad(x) + grad(x)')
%
%   Takes 3x1 gradient field and turns it into a 3x3 symmetric gradient tensor
%   Symgrad(v)  = 0.5*( [Gx(v1) Gx(v2)] + [Gx(v1) Gy(v1)] )
%                       [Gy(v1) Gy(v2)]   [Gx(v2) Gy(v2)]
%
%               = [Gx(v1)         Gx(v2)+Gy(v1)]
%                 [Gy(v1)+Gx(v2)  Gy(v2)       ]
%
%   Since this is symmetric, we store the upper trigangular entries only
%
%   So Symgrad([v1,v2]) = [Gx(v1),
%                          Gy(v2),
%                          Gx(v2)+Gy(v1)]
%
%   1 4 5
%     2 6
%       3
%   3->6
    y   =   zeros([size(x,1) size(x,2) size(x,3) 6]);
    
    y(:,:,:,1) =        pgrad(x(:,:,:,1),1,Lx,Lt);
    y(:,:,:,2) =                            pgrad(x(:,:,:,2),2,Lx,Lt);
    y(:,:,:,3) =                                                pgrad(x(:,:,:,3),3,Lx,Lt);
    y(:,:,:,4) = 0.5*(  pgrad(x(:,:,:,1),2,Lx,Lt)+pgrad(x(:,:,:,2),1,Lx,Lt));
    y(:,:,:,5) = 0.5*(  pgrad(x(:,:,:,1),3,Lx,Lt)+                    pgrad(x(:,:,:,3),1,Lx,Lt));
    y(:,:,:,6) = 0.5*(                      pgrad(x(:,:,:,2),3,Lx,Lt)+pgrad(x(:,:,:,3),2,Lx,Lt));
end

function x = prox(x,s,lambda) 
    x = x./(1 + s*lambda);
end

function y = div(x,Lx,Lt)
%   Adjoint of grad
%   3->1
    y = ngrad(x(:,:,:,1),1,Lx,Lt) + ngrad(x(:,:,:,2),2,Lx,Lt) + ngrad(x(:,:,:,3),3,Lx,Lt);
end

function y = symdiv(x,Lx,Lt)
%   Adjoint of Symgrad
%   6->3
    y   =   zeros([size(x,1) size(x,2) size(x,3) 3]);

    y(:,:,:,1) = ngrad(x(:,:,:,1),1,Lx,Lt) + ngrad(x(:,:,:,4),2,Lx,Lt) + ngrad(x(:,:,:,5),3,Lx,Lt);
    y(:,:,:,2) = ngrad(x(:,:,:,2),2,Lx,Lt) + ngrad(x(:,:,:,4),1,Lx,Lt) + ngrad(x(:,:,:,6),3,Lx,Lt);
    y(:,:,:,3) = ngrad(x(:,:,:,3),3,Lx,Lt) + ngrad(x(:,:,:,5),1,Lx,Lt) + ngrad(x(:,:,:,6),2,Lx,Lt);
end
