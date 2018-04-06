function est = symm_rough(xfm, d, Lx, Lt, tol, iters)

if nargin < 4 
    tol = 1E-4;
end
if nargin < 5
    iters = 100;
end

[est, flag, relres] = minres(@(x) afun(x, xfm, Lx, Lt), reshape(xfm'*d,[],1), tol, iters);
 
est =   reshape(est, xfm.msize);

end

function x = afun(x, xfm, Lx, Lt)
    x   =   reshape(x, [xfm.Nd xfm.Nt]);
    x   =   reshape(mtimes2(xfm,x),[],1) + reshape(R(x,Lx,Lt),[],1);
end

function x = R(x, Lx, Lt)
    %   Use symmetric definition
    %   R = D2'*D2
    x = Lt*(1*circshift(x,-2,4) - 4*circshift(x,-1,4) + 6*x - 4*circshift(x,1,4) + 1*circshift(x,2,4)) + ...
        Lx*(1*circshift(x,-2,1) - 4*circshift(x,-1,1) + 6*x - 4*circshift(x,1,1) + 1*circshift(x,2,1)) + ...
        Lx*(1*circshift(x,-2,2) - 4*circshift(x,-1,2) + 6*x - 4*circshift(x,1,2) + 1*circshift(x,2,2));
end
