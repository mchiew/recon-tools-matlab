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
    y   =   zeros([patch_size prod(im_size(1:3) - patch_size(1:3) + 1)]);

    dd  =   reshape(dd, im_size);

    t   =   .5;  % tau
    s   =   .5;  % sigma

    % z is pre-computed tau*E'*d
    z   =   t*dd;
    
    % define patch positions
    [ii,jj,kk]  =   meshgrid(1:im_size(1)-patch_size(1)+1, 1:im_size(2)-patch_size(2)+1, 1:im_size(3)-patch_size(3)+1);

    % proportion of blocks to update
    p   =   round(0.05*size(y,5));
    

%   Main loop
    fprintf(1, '%-5s %-16s\n', 'Iter','Cost');
    for iter = 1:niter

        %   x-update
        
        x   =   x + z;
        for i = 1:size(y,5)
            x = put_patch(x, -1*y(:,:,:,:,i), ii(i), jj(i), kk(i), patch_size);
        end
        
        [x,~] =   pcg(@(x)Afun(x, E, t), x(:), 1E-6, 5, [], [], x(:));
        x   =   reshape(x, im_size);
        
        %   y-update
        idx = randperm(size(y,5), p);
        for i = idx
            y(:,:,:,:,i) = singular_value_clipping(y(:,:,:,:,i) + s*get_patch(2*x-x0, ii(i),jj(i),kk(i), patch_size) , lambda);
        end
        
        x0  =   x;
        
        %   Display iteration summary data
        fprintf(1, '%-5d -\n', iter);
    end

end

function q = get_patch(X, i, j, k, p)

    x_idx           =   i:i+p(1)-1;
    y_idx           =   j:j+p(2)-1;
    z_idx           =   k:k+p(3)-1;
    q               =   X(x_idx, y_idx, z_idx, :);
    q               =   padarray(q, [p(1)-length(x_idx),p(2)-length(y_idx),p(3)-length(z_idx),0], 'post');
    
end

function X = put_patch(X, q, i, j, k, p)

    x_idx           =   i:i+p(1)-1;
    y_idx           =   j:j+p(2)-1;
    z_idx           =   k:k+p(3)-1;
    X(x_idx, y_idx, z_idx, :) = X(x_idx, y_idx, z_idx, :) + q(1:length(x_idx),1:length(y_idx),1:length(z_idx),:);
end


function q = singular_value_clipping(q, lambda)
    sz      =   size(q);
    [u,s,v] =   svd(reshape(q,[],sz(4)), 'econ');
    s       =   diag(min(diag(s), lambda));
    q       =   reshape(u*s*v',sz);
end

function y = Afun(x, E, tau)
    y = tau*reshape(E.mtimes2(reshape(x,E.msize)),[],1) + x;
end
