function c = omp(D,x,k)

% Orthogonal matching pursuit algorithm
% Mark Chiew (mark.chiew@ndcn.ox.ac.uk)
%
% c = mp(D,x)
%
% inputs:   D   - dictionary (normalised columns)
%           x   - input signal
%           k   - number of coefficients
% outputs:  c   - dictionary coefficients

R = x;
c = zeros(size(D,2),1);

for n = 1:k
    a = D'*R;
    [~,ii] = max(abs(a));
    c(ii) = a(ii);
    DD = D(:,c~=0);
    P = (DD/(DD'*DD))*DD';
    R = R - P*R;
end

