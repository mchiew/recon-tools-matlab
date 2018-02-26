function res = cham_pock_TGV_Toep(d, xfm, T, niter, lambda)

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
u   =   zeros(xfm.msize,'single');
uu  =   zeros(xfm.msize,'single');
v   =   zeros(xfm.msize,'single');
vv  =   zeros(xfm.msize,'single');
p   =   zeros(xfm.msize,'single');
q   =   zeros(xfm.msize,'single');
r2  =   zeros(xfm.msize,'single');
d   =   xfm'*d;

t   =   1/sqrt(12);
s   =   1/sqrt(12);
a1  =   1;
a0  =   2;

min_update  =   1E-4;

fprintf(1, '%-5s %-16s\n', 'iter','rel. update');
for i = 1:niter
    p   =   proj(p + s*(grad(2*u-uu) - (2*v-vv)), a1);
    q   =   proj(q + s*symgrad(2*v-vv), a0);
    r2  =   prox(r2 - s*d + s*mtimes_Toeplitz(xfm, T, 2*u-uu), s, lambda); 

    uu  =   u;
    u   =   u + t*(div(p)-r2);
    vv  =   v;
    v   =   v + t*(p + symdiv(q));

    f(100);
    show(abs(u(:,1)),[0 1E-4],'colormap',gray);
    update = norm(u-uu)/norm(uu);
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
