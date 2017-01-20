function x = cg_lsqr(xfm, d, tol, iters, x0)

%   Performs iterative conjugate gradient optimisation using built-in lsqr

    [x, flag]   =   lsqr(@(x, mode) afun(x, mode, xfm), d, tol, iters, [], [], x0);

function y = afun(x, mode, xfm)
    if strcmp(mode, 'transp')
        y   =   xfm'*x;
    else
        y   =   xfm*x;
    end
