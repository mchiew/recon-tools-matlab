function [u,v] = quad_iter(d, rank, iters, mask, u0, v0)

if ~isempty(u0) && size(u0,2) <= rank
    u   =   padarray(u0,[0 rank-size(u0,2)],eps,'post');
else
    u   =   zeros(size(d,1),rank)+eps;
end
if ~isempty(v0) && size(v0,2) <= rank
    v   =   padarray(v0,[0 rank-size(v0,2)],eps,'post');
else
    v   =   zeros(size(d,2),rank)+eps;
end

iter=   0;

while iter<iters
    u   =   quad_subprob(v,d',mask');
    v   =   quad_subprob(u,d,mask);
    iter=   iter+1;
end
