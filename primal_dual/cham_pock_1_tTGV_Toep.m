function res = cham_pock_1_tTGV_Toep(d, xfm, T, niter, lambda, tol, plt_fn, u)

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
tol =   1E-4;
end
if nargin < 7
plt_fn = [];
end
if nargin < 8
u   =   zeros(xfm.msize,'single');
uu  =   zeros(xfm.msize,'single');
else
uu  =   u;
end
v   =   zeros(xfm.msize,'single');
vv  =   zeros(xfm.msize,'single');
p   =   zeros(xfm.msize,'single');
q   =   zeros(xfm.msize,'single');
r2  =   zeros(xfm.msize,'single');
d   =   xfm'*d;

t   =   1/abs(T(1));
s   =   1/abs(T(1));
a1  =   1;
a0  =   2;

fprintf(1, '%-5s %-16s\n', 'iter','rel. update');
for i = 1:niter
    p   =   proj(p + s*(pgrad(2*u-uu) - (2*v-vv)), a1);
    q   =   proj(q + s*symgrad(2*v-vv), a0);
    r2  =   prox(r2 - s*d + s*mtimes_Toeplitz(xfm, T, 2*u-uu), s, lambda); 

    uu  =   u;
    u   =   u + t*(div(p)-r2);
    vv  =   v;
    v   =   v + t*(p + symdiv(q));

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
res.L   =   lambda;

end

function x = proj(x,z)
    x = x./max(1,abs(x)/z);
end

function x = pgrad(x)
%   Use the definition in Bredies et al.
%
%   Example 4x4 forward difference operator:
%   -1  1  0  0
%    0 -1  1  0
%    0  0 -1  1
%    0  0  0  0
    %x = [diff(x,1,2) zeros(size(x,1),1)];

%   Different boundary conditions
%    1  0  0 -1
%   -1  1  0  0
%    0 -1  1  0
%    0  0 -1  1
    x = diff([x(:,end) x],1,2);
end

function x = ngrad(x)
%   Use the definition in Bredies et al.
%   This is also the -adjoint of the pgrad operator
%   Example 4x4 reverse difference operator:
%    1  0  0  0
%   -1  1  0  0
%    0 -1  1  0
%    0  0 -1  0
    %x = [x(:,1) diff(x(:,1:end-1),1,2) -x(:,end-1)];

%   -1  1  0  0
%    0 -1  1  0
%    0  0 -1  1
%    1  0  0 -1
    x = [diff(x,1,2) x(:,1)-x(:,end)];
end

function x = symgrad(x)
    %x = 0.5*pgrad(x) + 0.5*ngrad(x);
    x = pgrad(x);
end

function x = prox(x,s,lambda) 
    x = x./(1 + s*lambda);
end

function x = div(x)
    x = ngrad(x);
end

function x = symdiv(x)
    %x   =   symgrad(x);
    x = ngrad(x);
end
