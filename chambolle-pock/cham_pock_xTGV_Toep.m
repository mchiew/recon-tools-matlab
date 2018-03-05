function res = cham_pock_xTGV_Toep(d, xfm, T, niter, lambda, plt_fn, u)

%   Mark Chiew
%   Feb 2018
%
%   Implementation of the Chambolle-Pock Primal Dual First Order Method
%   for total generalised variation
%   From Chambole & Pock, 2011 Journal of Mathematical Imaging and Vision
%   Knoll, Bredies, Pock & Stollberger, 2011 MRM

%   Note lambda inversely weights the data consistency term here
%   i.e., bigger lambda means more relative TGV weighting

%   This reformulates the algorithm to take advantage of Toeplitz embedding
%   for evaluation of xfm'*xfm

%   Initialise
if nargin < 6
plt_fn = [];
end
if nargin < 7
u   =   zeros([xfm.Nd(1:2) xfm.Nt],'single');
uu  =   zeros([xfm.Nd(1:2) xfm.Nt],'single');
else
uu  =   u;
end
v   =   zeros([xfm.Nd(1:2) xfm.Nt 2],'single');
vv  =   zeros([xfm.Nd(1:2) xfm.Nt 2],'single');
p   =   zeros([xfm.Nd(1:2) xfm.Nt 2],'single');
q   =   zeros([xfm.Nd(1:2) xfm.Nt 3],'single');
r2  =   zeros([xfm.Nd(1:2) xfm.Nt],'single');
d   =   reshape(xfm'*d, [xfm.Nd(1:2) xfm.Nt]);

t   =   1/sqrt(12);
s   =   1/sqrt(12);
a1  =   1;
a0  =   2;

min_update  =   1E-4;

fprintf(1, '%-5s %-16s\n', 'iter','rel. update');
for i = 1:niter
    p   =   proj(p + s*(grad(2*u-uu) - (2*v-vv)), a1);
    q   =   proj(q + s*symgrad(2*v-vv), a0);
    r2  =   prox(r2 - s*d + s*reshape(mtimes_Toeplitz(xfm, T, 2*u-uu),[xfm.Nd(1:2) xfm.Nt]), s, lambda); 

    uu  =   u;
    u   =   u + t*(div(p)-r2);
    vv  =   v;
    v   =   v + t*(p + symdiv(q));

    if isa(plt_fn, 'function_handle')
        plt_fn(u);
    end

    update = norm(u(:)-uu(:))/norm(uu(:));
    fprintf(1, '%-5d %-16G\n', i, update);
    if update < min_update
        break;
    end
end

res.out =   u;
res.L   =   lambda;

end

function x = proj(x,z)
    x = x./max(1,abs(x)/z);
end

function x = pgrad(x, dim)
%   Use the definition in Bredies et al.
%
%   Example 4x4 forward difference operator:
%   -1  1  0  0
%    0 -1  1  0
%    0  0 -1  1
%    0  0  0  0
    if dim == 1
        x = cat(1,diff(x,1,1),zeros(1,size(x,2),size(x,3),size(x,4)));
    elseif dim == 2 
        x = cat(2,diff(x,1,2),zeros(size(x,1),1,size(x,3),size(x,4)));
    end
end

function x = ngrad(x,dim)
%   Use the definition in Bredies et al.
%   This is also the -adjoint of the pgrad operator
%   Example 4x4 reverse difference operator:
%    1  0  0  0
%   -1  1  0  0
%    0 -1  1  0
%    0  0 -1  0
    if dim == 1
        x = cat(1,x(1,:,:,:),diff(x(1:end-1,:,:,:),1,1),-x(end-1,:,:,:));
    elseif dim == 2
        x = cat(2,x(:,1,:,:),diff(x(:,1:end-1,:,:),1,2),-x(:,end-1,:,:));
    end
end

function y = grad(x)
%   1->2
    y = cat(4,pgrad(x,1),pgrad(x,2));
end

function y = symgrad(x)
%   Defined as 0.5*(grad(x) + grad(x)')
%
%   Takes 2x1 gradient field and turns it into a 4x4 symmetric gradient tensor
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
%   2->3
    y   =   zeros([size(x,1) size(x,2) size(x,3) 3]);
    
    y(:,:,:,1)  =   pgrad(x(:,:,:,1),1);
    y(:,:,:,2)  =   pgrad(x(:,:,:,2),2);
    y(:,:,:,3)  =   0.5*(pgrad(x(:,:,:,1),2)+pgrad(x(:,:,:,2),1));
end

function x = prox(x,s,lambda) 
    x = x./(1 + s*lambda);
end

function y = div(x)
%   Adjoint of grad
%   2->1
    y = ngrad(x(:,:,:,1),1) + ngrad(x(:,:,:,2),2);
end

function y = symdiv(x)
%   Adjoint of Symgrad
%   3->2
    y   =   zeros([size(x,1) size(x,2) size(x,3) 2]);

    y(:,:,:,1)  =   ngrad(x(:,:,:,1),1) + ngrad(x(:,:,:,3),2);
    y(:,:,:,2)  =   ngrad(x(:,:,:,3),1) + ngrad(x(:,:,:,2),2);
end
