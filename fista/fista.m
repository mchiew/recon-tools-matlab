function [u,cost] = fista(xfm, d, W, lambda, im_size, niter, step)

%   Mark Chiew  
%   May 2017
%
%   Implementation of the FISTA iteration scheme for L1-Regularised problems
%   Adapted from Beck and Teboulle, SIAM J Im Sci 2009
%
%   This will solve a problem of the form:
%
%   min{u} 0.5*|xfm*u - d|_2^2 + lambda*|W*u|_1
%
%   where u is the estimate,  W is an invertible  transform (often Wavelet), 
%   or identity transform
%   xfm is the measurement operator, lambda is the L1-weighting
%   and d is the measured raw k-space data

%   To make things easier, we work with our estimate u in the main loop
%   in the sparse domain, so that our measurement operator is actually 
%   effectively xfm*W'
%   Then we simply transform by W' as a final step
%   
%   Another way of viewing this is that we perform a change of variables
%   such that u_new = W*u

%   Constant stepsize variant

%   Initialise
    u   =   W*zeros(im_size);
    v   =   u;
    t   =   1;
    up  =   u;

%   Main loop
fprintf(1, '%-5s %-16s %-16s %-16s\n', 'Iter','L2','L1','Cost');
for iter = 1:niter

    %   Data consistency
    tmp =   d - xfm*(W'*v);
    u   =   v + step*(W*(xfm'.*tmp));

    %   Solve proximal sub-problem
    u   =   shrink(u, lambda*step);

    %   Compute momentum parameter
    t2  =   (1+sqrt(1+4*t^2))/2;

    %   Compute momentum update
    v   =   u + ((t-1)/t2)*(u-up);
    
    %   Update variables
    t   =   t2;
    up  =   u;

    %   Error terms and cost function
    err1(iter)  =   norm(tmp(:),2);
    err2(iter)  =   lambda*norm(u(:),1);
    cost(iter)  =   0.5*err1(iter).^2 + err2(iter);

    %   Display iteration summary data
    fprintf(1, '%-5d %-16G %-16G %-16G\n', iter, err1(iter), err2(iter), cost(iter));
end

u   =   W'*u;
end


function y = shrink(x, thresh)
    y = exp(1j*angle(x)).*max(abs(x)-thresh,0);
end
