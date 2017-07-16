function [u,M] = fista(d, xfm, W, lambda, im_size, niter, step)

%   Mark Chiew  
%   May 2017
%
%   Implementation of the FISTA iteration scheme for L1-Regularised problems
%   Adapted from Beck and Teboulle, SIAM J Im Sci 2009
%
%   This will solve a problem of the form:
%
%   min{u} |xfm*u - d|_2 + lambda*|W*u|_1
%
%   where u is the estimate,  W is an invertible  transform (often Wavelet), 
%   xfm is the measurement operator, lambda is the L1-weighting
%   and d is the measured raw k-space data

%   Constant stepsize variant

%   Initialise
    u   =   W*zeros(im_size);
    v   =   u;
    t   =   1;
    up  =   u;

%   Main loop
for ii = 1:niter
    %   Solve proximal sub-problem
    u   =   shrink(v - (1./(1+(ii*step).^.25))*(W*(xfm'.*(xfm*(W'*v)-d))),lambda*step);    
    %u   =   shrink(v - step*(W*(xfm'.*(xfm*(W'*v)-d))),lambda*step);    

    %   Compute momentum parameter
    t2  =   (1+sqrt(1+4*t^2))/2;

    %   Compute momentum update
    v   =   u + ((t-1)/t2)*(u-up);
    
    %   Update variables
    t   =   t2;
    up  =   u;

    show(abs(W'*u)',[]);
    text(2,6,sprintf('%03d',ii),'color','w','FontSize',64);
    M(ii)=getframe;
end

u   =   W'*u;
end


function y = shrink(x, thresh)
    y = exp(1j*angle(x)).*max(abs(x)-thresh,0);
end
