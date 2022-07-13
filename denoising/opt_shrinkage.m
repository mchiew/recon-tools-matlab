function S = opt_shrinkage(S, sigma, m, n)
    b = min(m,n)/max(m,n);
    S = S/(sigma*sqrt(max(m,n)));
    i = S<(1+sqrt(b));
    S = (1./S).*sqrt((S.^2-b-1).^2-4*b);
    S(i) = 0;
    S = S*sigma*sqrt(max(m,n));
end