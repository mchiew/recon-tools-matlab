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
    %for i = 1:size(y,2)
    %    x(:,i) = sparse(omp(D, y(:,i), s));        
    %end
    x = sparse(omp(D, y, s));
    
    % codebook update
    for i = 1:K
        w = find(x(i,:));
        if ~isempty(w)
            E = y - D(:,[1:i-1 i+1:K])*x([1:i-1 i+1:K],:);
            E = E(:,w);

            [u,sig,v] = svd(E,0);

            % prune atom if too similar to existing atom
            m = max(abs(u(:,1)'*D));
            if m < 0.9
                D(:,i) = u(:,1);
                x(i,w) = sig(1)*v(:,1).';
            else
                tmp     = sum(abs(y - D*x).^2,1);
                [~, ii] = max(tmp);
                D(:,i)  = y(:,ii)/norm(y(:,ii));
                x(:,ii)  = 0;
                x(i,ii) = norm(y(:,ii));
            end            

        else
            % prune atom if unused
            tmp     = sum(abs(y - D*x).^2,1);
            [~, ii] = max(tmp);
            D(:,i)  = y(:,ii)/norm(y(:,ii));
            x(:,ii)  = 0;
            x(i,ii) = norm(y(:,ii));
        end
    end
    
    % iteration update    
    iter = iter + 1;
    iter_err = norm(y-D*x,'fro');
    update = abs(iter_err-approx_err)/approx_err;
    approx_err = iter_err;
end

