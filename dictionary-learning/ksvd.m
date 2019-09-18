function [D, x] = ksvd(y,K,s)

% k-SVD algorithm for designing sparse, overcomplete dictionaries
% Mark Chiew (mark.chiew@ndcn.ox.ac.uk)
% From Aharon, Elad and Bruckstein, IEEE-TSP 2006
%
% D = ksvd(y,k)
%
% inputs:   y   - input training signals
%           K   - number of atoms
%           s   - sparsity of dictionary
% outputs:  D   - over-complete dictionary

max_iter = 100;
iter     = 0;
tol      = 1E-4;

% Initialise D with normalised columns
D = randn(size(y,1), K);
D = D./sqrt(sum(abs(D).^2,1));

x = sparse(K,size(y,2));
approx_err = 1;
update = inf;

while iter < max_iter && update > tol
    % sparse coding
    for i = 1:size(y,2)
        x(:,i) = sparse(omp(D, y(:,i), s));        
    end
    
    % codebook update
    for i = 1:K
        w = find(x(i,:));
        E = y - D(:,[1:i-1 i+1:K])*x([1:i-1 i+1:K],:);
        E = E(:,w);
        
        [u,s,v] = svd(E,0);
        
        D(:,i) = u(:,1);
        x(i,w) = s(1)*v(:,1).';
    end
    
    % iteration update    
    iter = iter + 1;
    iter_err = norm(y-D*x,'fro');
    update = abs(iter_err-approx_err)/approx_err;
    approx_err = iter_err;
end

