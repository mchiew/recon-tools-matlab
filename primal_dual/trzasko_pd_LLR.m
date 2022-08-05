function x = trzasko_pd_LLR(E, dd, lambda, patch_size, im_size, niter)

%   Mark Chiew  
%   Aug 2022 
%
%   Locally-low-rank constrained reconstruction using Trzasko Stochastic Primal Dual Method
%   ISMRM 2017 #0455
%   https://cds.ismrm.org/protected/17MProceedings/PDFfiles/0455.html

%   Initialise
    % x is output dataset
    x   =   zeros(im_size);
    x0  =   x;

    % y is a collection of all possible patches
    y   =   zeros([patch_size prod(im_size - patch_size + 1)]);

    dd  =   reshape(dd, im_size);

    L   =   1/E.max_step(10);

    t   =   1;  % tau
    s   =   1;  % sigma

    % z is pre-computed tau*E'*dd
    z   =   t*(E'*dd);
    
    % define patch positions
    [ii,jj,kk]  =   meshgrid(1:im_size(1)-patch_size(1)+1, 1:im_size(2)-patch_size(2)+1, 1:im_size(3)-patch_size(3)+1);

    % get normalization factor
    N   =   zeros(im_size(1:3));
    for i = 1:length(ii)
        N = put_patch(N, ones(patch_size), ii(i), jj(i), kk(i), patch_size);
    end

    % proportion of blocks to update
    p   =   round(0.1*length(ii));
    

%   Main loop
    fprintf(1, '%-5s %-16s\n', 'Iter','Cost');
    for iter = 1:niter

        %   x-update
        x   =   x(:) + z(:);
        for i = 1:length(ii)
            x = put_patch(x, -1*y(:,:,:,i), ii(i), jj(i), kk(i), patch_size);
        end
        x   =   pcg(@(x)Afun(x, E, t), x(:), 1E-6, 10);

        %   y-update
        idx = randperm(length(ii), p);
        for i = idx
            y(:,:,:,i) = singular_value_clipping(y(:,:,:,i) + s*get_patch(2*x-x0, ii(i),jj(i),kk(i), patch_size) , lambda/2);
        end

        x0  =   x;
        
        %   Display iteration summary data
        fprintf(1, '%-5d -\n', iter);
    end

end

function q = get_patch(X, i, j, k, p)

    [sx,sy,sz,st]   =   size(X);
    x_idx           =   max(i-(p(1)-1)/2,1):min(i+(p(1)-1)/2,sx);
    y_idx           =   max(j-(p(2)-1)/2,1):min(j+(p(2)-1)/2,sy);
    z_idx           =   max(k-(p(3)-1)/2,1):min(k+(p(3)-1)/2,sz);
    q               =   X(x_idx, y_idx, z_idx, :);
    
end

function X = put_patch(X, q, i, j, k, p)

    [sx,sy,sz,st]   =   size(X);
    x_idx           =   max(i-(p(1)-1)/2,1):min(i+(p(1)-1)/2,sx);
    y_idx           =   max(j-(p(2)-1)/2,1):min(j+(p(2)-1)/2,sy);
    z_idx           =   max(k-(p(3)-1)/2,1):min(k+(p(3)-1)/2,sz);
    X(x_idx, y_idx, z_idx, :) = X(x_idx, y_idx, z_idx, :) + q;
end


function q = singular_value_clipping(q, lambda)
    [u,s,v] =   svd(q, 'econ');
    s       =   diag(min(diag(s), lambda));
    q       =   u*s*v';
end

function y = Afun(x, E, tau)
    y = tau*reshape(E.mtimes2(reshape(x,E.msize)),[],1) + x;
end
