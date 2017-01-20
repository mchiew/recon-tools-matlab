function x = cg(xfm, d, tol, iters, x0)

%   Conjugate Gradient

    r   =   xfm'*(d-xfm*x0);
    p   =   r;

    k   =   0;
    n   =   1./norm(r);

    while k < iters

        q   =   xfm'*(xfm*p);
        x   =   x + p*(r'*r)/(p'*q);
        r2  =   r - q*(r'*r)/(p'*q);
        p   =   r2 + p*(r2'*r2)/(r'*r);
        r   =   r2;
        k   =   k+1;

        g   =   norm(r)*n;
        fprintf(1, '    %04d    %1.9f\n', k, g);

        if g < tol
            break;
        end
    end
