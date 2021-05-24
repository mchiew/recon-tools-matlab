function x = pogm_Huber(E, dd, lambda, h, im_size, niter)

%   Mark Chiew  
%   May 2021
%
%   Huber-loss constrained reconstruction using POGM
%   (p1306, Taylor et al., 2017)

%   Initialise
    x   =   zeros(im_size);
    y   =   zeros(im_size);
    z   =   zeros(im_size);
    y0  =   zeros(im_size);
    
    dd  =   reshape(dd, im_size);

    L   =   1/E.max_step(100);

    a   =   1;  % theta in algorithm
    b   =   1;  % gamma in algorithm

%   Main loop
fprintf(1, '%-5s %-16s\n', 'Iter','Cost');
for iter = 1:niter

    %   y-update
    y0  =   y;
    y   =   x - (1/L)*(E.mtimes2(x)-dd);

    %   a-update
    a0  =   a;
    if iter < niter
        a = (1+sqrt(4*a0^2+1))/2;
    else
        a = (1+sqrt(8*a0^2+1))/2;
    end

    %   z-update
    z   =   y + ((a0-1)/a)*(y-y0) + (a0/a)*(y-x) + ((a0-1)/(L*b*a))*(z - x);

    %   b-update
    b0  =   b;
    b   =   (2*a0+a-1)/(L*a);

    %   x-update
    x   =   h*huber_proximal(z/h, lambda*b);
    
    %   Display iteration summary data
    fprintf(1, '%-5d -\n', iter);
end

end

function y = huber_proximal(x, sigma)
    y = x - (sigma*x)/max(abs(x),sigma+1);
end
