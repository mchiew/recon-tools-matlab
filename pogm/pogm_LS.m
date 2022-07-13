function [L, S] = pogm_LS(E, dd, lambda_L, lambda_S, im_size, niter)

%   Mark Chiew  
%   May 2021
%
%   L+S Reconstruction using POGM
%   (p1306, Taylor et al., 2017)
%   (Lin & Fessler, 2019)

%   Initialise
    L   =   zeros(im_size);
    S   =   zeros(im_size);
    L0  =   zeros(im_size);
    S0  =   zeros(im_size);
    L1  =   zeros(im_size);
    S1  =   zeros(im_size);
    
    dd  =   reshape(dd, im_size);

    t   =   E.max_step(100)/2;

    a   =   1;  % theta in algorithm
    b   =   1;  % gamma in algorithm

%   Main loop
fprintf(1, '%-5s %-16s\n', 'Iter','Cost');
for iter = 1:niter


    g   =   t*(E.mtimes2(L+S) - dd);
    L0p =   L0;
    L0  =   L - g;
    S0p =   S0;
    S0  =   S - g;

    %   a-update (theta)
    a0  =   a;
    if iter < niter
        a = (1+sqrt(4*a^2+1))/2;
    else
        a = (1+sqrt(8*a^2+1))/2;
    end

    L1  =   L0 + ((a0-1)/a)*(L0-L0p) + (a0/a)*(L0-L) + ((a0-1)/(b*a))*t*(L1-L);
    S1  =   S0 + ((a0-1)/a)*(S0-S0p) + (a0/a)*(S0-S) + ((a0-1)/(b*a))*t*(S1-S);

    %   b-update (gamma)
    b   =   t*(1 + (a0-1)/a + a0/a);


    L   =   svt(L1, lambda_L);
    S   =   thresh(S1, lambda_S);
    
    %   Display iteration summary data
    fprintf(1, '%-5d -\n', iter);
end

end


function y = svt(x, lambda)
    [S,V]   =   half_SVD(reshape(x,[],size(x,4)));
    y       =   reshape(reshape(x,[],size(x,4))*V*diag(1./S)*diag(max(S-lambda,0))*V', size(x));
end

function y = thresh(x, lambda)
    y   =   fft(x,[],4);
    y   =   exp(1j*angle(y)).*max(abs(y)-lambda,0);
    y   =   ifft(y,[],4);
end

function [S,V] = half_SVD(X)

    [V,D]   =   eig(X'*X,'vector');

    [~,ii]  =   sort(D,'descend');

    S       =   sqrt(D(ii));
    V       =   V(:,ii);
end
