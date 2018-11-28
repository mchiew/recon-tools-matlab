function [u,cost] = fista_xf(E, dd, lambda, im_size, niter, step)

%   Mark Chiew  
%   Nov 2018
%
%   x-f version of FISTA

%   Initialise
    u   =   zeros(im_size);
    v   =   u;
    t   =   1;
    up  =   u;

%   Main loop
fprintf(1, '%-5s %-16s %-16s %-16s\n', 'Iter','L2','L1','Cost');
for iter = 1:niter

    %   Data consistency
    u   =   v + step*(dd - E.mtimes2(v));

    %   Solve proximal sub-problem
    u   =   ifftdim(shrink(fftdim(u,2), lambda*step),2);

    %   Compute momentum parameter
    t2  =   (1+sqrt(1+4*t^2))/2;

    %   Compute momentum update
    v   =   u + ((t-1)/t2)*(u-up);
    
    %   Update variables
    t   =   t2;
    up  =   u;

    %   Error terms and cost function
    err1(iter)  =   0;
    err2(iter)  =   lambda*norm(u(:),1);
    cost(iter)  =   0.5*err1(iter).^2 + err2(iter);

    %   Display iteration summary data
    fprintf(1, '%-5d %-16G %-16G %-16G\n', iter, err1(iter), err2(iter), cost(iter));
end

end


function y = shrink(x, thresh)
    y = exp(1j*angle(x)).*max(abs(x)-thresh,0);
end
