function c = omp(D,x,k)

% Orthogonal matching pursuit algorithm
% Mark Chiew (mark.chiew@ndcn.ox.ac.uk)
%
% c = mp(D,x)
%
% inputs:   D   - dictionary (normalised columns)
%           x   - input signal(s)
%           k   - number of coefficients
% outputs:  c   - dictionary coefficients

c = zeros(size(D,2),size(x,2));

for i = 1:size(x,2)
    R = x(:,i);
    for n = 1:k
        a = D'*R;
        [~,ii] = max(abs(a));
        c(ii,i) = a(ii);
        DD = D(:,c(:,i)~=0);
        P = (DD/(DD'*DD))*DD';
        R = R - P*R;
    end
end