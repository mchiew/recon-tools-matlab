function x = pogm_LLR(E, dd, lambda, patch_size, im_size, niter)

%   Mark Chiew  
%   May 2021
%
%   Locally-low-rank constrained reconstruction using POGM
%   (p1306, Taylor et al., 2017)

%   Initialise
    x   =   zeros(im_size);
    y   =   zeros(im_size);
    z   =   zeros(im_size);
    y0  =   zeros(im_size);
    
    dd  =   reshape(dd, im_size);

    p   =   patch_size;
    L   =   1/E.max_step(10);

    a   =   1;  % theta in algorithm
    b   =   1;  % gamma in algorithm

%   Main loop
fprintf(1, '%-5s %-16s\n', 'Iter','Cost');
for iter = 1:niter

    %   y-update
    y0  =   y;
    y   =   x - (1/L)*(E.mtimes2(x)-dd);

    %   a-update (theta)
    a0  =   a;
    if iter < niter
        a = (1+sqrt(4*a^2+1))/2;
    else
        a = (1+sqrt(8*a^2+1))/2;
    end

    %   z-update
    z   =   y + ((a0-1)/a)*(y-y0) + (a0/a)*(y-x) + ((a0-1)/(L*b*a))*(z - x);

    %   b-update (gamma)
    b0  =   b;
    b   =   (2*a0+a-1)/(L*a);

    %   x-update
    [ii,jj,kk]  =   meshgrid(randperm(p(1),1)-(p(1)-1)/2:p(1):im_size(1),randperm(p(2),1)-(p(2)-1)/2:p(2):im_size(2),randperm(p(3),1)-(p(3)-1)/2:p(3):im_size(3));
   
    for idx = 1:length(ii(:))
        q   =   get_patch(z, ii(idx), jj(idx), kk(idx), p);
        [u,s,v]     =   svd(reshape(q,[],im_size(4)),'econ');
        s   =   shrink(s, lambda*b); 
        q   =   reshape(u*s*v', size(q));
        x   =   put_patch(x, q, ii(idx), jj(idx), kk(idx), p);
    end
    
    
    %   Display iteration summary data
    fprintf(1, '%-5d -\n', iter);
end

end

function q = get_patch(X, i, j, k, p)

    [sx,sy,sz,st]   =   size(X);
    q               =   X(max(i-(p(1)-1)/2,1):min(i+(p(1)-1)/2,sx),max(j-(p(2)-1)/2,1):min(j+(p(2)-1)/2,sy), max(k-(p(3)-1)/2,1):min(k+(p(3)-1)/2,sz),:);
    
end

function X = put_patch(X, q, i, j, k, p)
    [sx,sy,sz,st]   =   size(X);
    X(max(i-(p(1)-1)/2,1):min(i+(p(1)-1)/2,sx),max(j-(p(2)-1)/2,1):min(j+(p(2)-1)/2,sy), max(k-(p(3)-1)/2,1):min(k+(p(3)-1)/2,sz),:) = q;
end

function y = shrink(x, thresh)
    y = diag(max(diag(x)-thresh,0));
end
