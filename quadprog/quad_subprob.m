function [Y] = quad_subprob(X,b,mask)

%   To solve problems of the form ||UV'-b||
%
%   For solving the subproblem for V (assuming U fixed)
%       use V=quad_subprob(U,b)
%   For solving the subproblem for U (assuming V fixed)
%       use U=quad_subprob(V,b')

for i = 1:size(mask,2)
    B{i} = sparse(X(find(mask(:,i)),:));
end
A=blkdiag(B{:});

H   =   2*A'*A;

f   =   -2*b(mask)'*A;

c   =   norm(b(mask)).^2;

Y   =   quadprog_admm(H,f',c,1,1);
Y   =   reshape(Y,[],size(b,2))';
