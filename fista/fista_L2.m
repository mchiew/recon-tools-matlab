function [u,cost] = fista_L2(E, dd, Lx, Lt, im_size, niter, step)

%   Mark Chiew  
%   Nov 2018
%
%   x-f version of FISTA

%   Initialise
    u   =   zeros(im_size);
    v   =   u;
    t   =   1;
    up  =   u;
    
    z   =   u;

%   Main loop
fprintf(1, '%-5s %-16s %-16s %-16s %-16s\n', 'Iter','DataCon','L1','L2','Cost');
for iter = 1:niter

    %   Data consistency
    u   =   v + step*(dd - z - Lt*R1(v));

    %   Solve proximal sub-problem
    u   =   shrink(u, Lx*step);

    %   Compute momentum parameter
    t2  =   (1+sqrt(1+4*t^2))/2;

    %   Compute momentum update
    v   =   u + ((t-1)/t2)*(u-up);
    
    %   Update variables
    t   =   t2;
    up  =   u;
    z   =   E.mtimes2(u);

    %   Error terms and cost function
    err1(iter)  =   u(:)'*(z(:)-2*dd(:));
    err2(iter)  =   Lx*norm(u(:),1);
    err3(iter)  =   Lt*sum(abs(reshape(R1(u),[],1)).^2);
    cost(iter)  =   0.5*err1(iter) + err2(iter) + 0.5*err3(iter);

    %   Display iteration summary data
    fprintf(1, '%-5d %-16G %-16G %-16G %-16G\n', iter, err1(iter), err2(iter), err3(iter), cost(iter));
    
    if iter > 1
        if abs((cost(iter)-cost(iter-1))/cost(iter-1)) <  1E-6
            break;
        end
    end
end

end


function y = shrink(x, thresh)
    y = exp(1j*angle(x)).*max(abs(x)-thresh,0);
end

function x = R1(x)
    %x =  -1*circshift(x,-1,2) + 2*x - 1*circshift(x,1,2);
    x = [x(:,1)-x(:,2), -1*x(:,1:end-2) + 2*x(:,2:end-1) - x(:,3:end), -1*x(:,end-1) + x(:,end)];
end
