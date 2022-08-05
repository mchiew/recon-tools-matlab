function u = cham_pock_1_tTGV(d, xfm, niter, lambda, tol, u)

%   Mark Chiew
%   Feb 2018
%
%   Implementation of the Chambolle-Pock Primal Dual First Order Method
%   for total generalised variation
%   From Chambole & Pock, 2011 Journal of Mathematical Imaging and Vision
%   Knoll, Bredies, Pock & Stollberger, 2011 MRM

%   Note lambda inversely weights the data consistency term here
%   i.e., bigger lambda means more relative TGV weighting

%   Initialise
if nargin < 5
tol =   1E-4;
end
if nargin < 6
u   =   zeros(xfm.msize,'single');
uu  =   zeros(xfm.msize,'single');
else
uu  =   u;
end
v   =   zeros(xfm.msize,'single');
vv  =   zeros(xfm.msize,'single');
p   =   zeros(xfm.msize,'single');
q   =   zeros(xfm.msize,'single');
r   =   zeros(xfm.dsize,'single');

t   =   1/sqrt(12);
s   =   1/sqrt(12);
a1  =   1;
a0  =   2;

fprintf(1, '%-5s %-16s\n', 'iter','rel. update');
for i = 1:niter
    p   =   proj(p + s*(grad(2*u-uu) - (2*v-vv)), a1);
    q   =   proj(q + s*symgrad(2*v-vv), a0);
    r   =   prox(r + s*(xfm*(2*u-uu) - d), s, lambda); 

    uu  =   u;
    u   =   u + t*(div(p)-xfm'*r);
    vv  =   v;
    v   =   v + t*(p + symdiv(q));

    update = norm(u(:)-uu(:))/norm(uu(:));
    fprintf(1, '%-5d %-16G\n', i, update);
    if update < tol
        break;
    end
end

end

function x = proj(x,z)
    x = x./max(1,abs(x)/z);
end

function x = grad(x)
    x = diff([x(:,end) x],1,2);
end

function x = symgrad(x)
    x = 0.5*diff([x(:,end) x],1,2) + 0.5*diff([x x(:,1)],1,2);
end

function x = prox(x,s,lambda) 
    x = x./(1 + s*lambda);
end

function x = div(x)
    x = diff([x x(:,1)],1,2);
end

function x = symdiv(x)
    x   =   symgrad(x);
end
