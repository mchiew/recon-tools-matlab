function est = lsqr_rough(xfm, d, Lx, Lt, tol, iters)

if nargin < 4 
    tol = 1E-4;
end
if nargin < 5
    iters = 100;
end

[est, flag, relres] = lsqr(@(x,mode) afun(x, mode, xfm, Lx, Lt), [reshape(d,[],1);zeros(3*prod(xfm.msize),1)], tol, iters);
 
est =   reshape(est, xfm.msize);

end

function y = afun(x, mode, xfm, Lx, Lt)
if strcmp(mode, 'transp')
    x1  =   reshape(x(1:prod(xfm.dsize)), xfm.dsize);
    d2x =   reshape(x(prod(xfm.dsize)+1:prod(xfm.dsize)+prod(xfm.msize)), [xfm.Nd xfm.Nt]);
    d2y =   reshape(x(prod(xfm.dsize)+prod(xfm.msize)+1:prod(xfm.dsize)+2*prod(xfm.msize)), [xfm.Nd xfm.Nt]);
    d2t =   reshape(x(prod(xfm.dsize)+2*prod(xfm.msize)+1:prod(xfm.dsize)+3*prod(xfm.msize)), [xfm.Nd xfm.Nt]);
    y   =   reshape(xfm'.*x1 + D2adj(d2x, d2y, d2t, sqrt(Lx), sqrt(Lt)),[],1);
else
    x   =   reshape(x, [xfm.Nd xfm.Nt]);
    [d2x, d2y, d2t] = D2(x, sqrt(Lx), sqrt(Lt));
    y   =   [reshape(xfm*x,[],1);d2x(:);d2y(:);d2t(:)];
end
end

function [d2x d2y d2t] = D2(x, Lx, Lt)
%   Use symmetric definition
%   cyclic boundary condition
    d2t =   Lt*(circshift(x,1,4) - 2*x + circshift(x,-1,4));
    d2x =   Lx*(circshift(x,1,1) - 2*x + circshift(x,-1,1));
    d2y =   Lx*(circshift(x,1,2) - 2*x + circshift(x,-1,2));
end

function x = D2adj(d2x, d2y, d2t, Lx, Lt)
%   Use symmetric definition
%   cyclic boundary condition
    x = Lt*(circshift(d2t,1,4) - 2*d2t + circshift(d2t,-1,4)) + ...
        Lx*(circshift(d2x,1,1) - 2*d2x + circshift(d2x,-1,1)) + ...
        Lx*(circshift(d2y,1,2) - 2*d2y + circshift(d2y,-1,2));
end
