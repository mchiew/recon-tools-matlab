classdef (Abstract) xfm
properties (SetAccess = protected, GetAccess = public)
    Nd      =   [];
    Nt      =   0;
    msize   =   []; 
    adjoint =   0;
    dsize   =   [];
    S       =   [];
    Nc      =   0;
    M;
end

methods
function res = xfm(dims, coils, fieldmap_struct)
    %   Initialise dimensions 
    res.Nd      =   dims(1:3);
    res.Nt      =   dims(4);
    res.msize   =   [prod(res.Nd) res.Nt];
    res.adjoint =   0;

    %   Initialise sensitivity encoding operator
    if isempty(coils)
        coils   =   ones([res.Nd 1]);
    end
        res.S   =   sensEncodingMatrix(coils);
        res.Nc  =   res.S.Nc;

    %   Initialise multi-frequency interpolation off-resonance correction operator
    if isempty(fieldmap_struct)
        res.M   =   mfi();
    else
        for i = 1:length(fieldmap_struct)
            M(i)    =   mfi(fieldmap_struct(i).field, ...
                            fieldmap_struct(i).t, ...
                            fieldmap_struct(i).L, ...
                            fieldmap_struct(i).idx);
        end
        res.M   =   M;
    end
end

function a = ctranspose(a)   
    a.adjoint   = xor(a.adjoint,1);    
end

function step = max_step(xfm,iters)
    if nargin < 2
        iters = inf;
    end
    %   Use the power method to find the max eigenvalue of E'E
    y   =   randn(xfm.msize);
    N   =   0;
    ii  =   0;
    while abs(norm(y(:)) - N)/N > 1E-4 && ii < iters
        N   =   norm(y(:)); 
        if nargout == 0
            disp(1./N);
        end
        y   =   xfm.mtimes2(y/N);
        ii  =   ii+1;
    end
    step    =   1./norm(y(:));
end

function est = iter(xfm, d, optfn, tol, iters, L)

    %   Performs symmetric iterative recon using built-ins
    %   Input d should be shaped like the output of mtimes
    %   Solves normal equation

    if nargin < 3
        optfn   =   @minres;
    end
    if nargin < 4
        tol     =   [];
    end
    if nargin < 5
        iters   =   100;
    end
    if nargin < 6
        L       =   [0,0,0,0];
    end

    if isequal(size(d), xfm.dsize) || (numel(d) == prod(xfm.dsize))
        d = xfm'*d;
    elseif ~isequal(size(d), xfm.msize) 
        error('Wrong input dimensions');
    end
    
    [est, flag, relres, iter] =   optfn(@(x,mode) reshape(mtimes2(xfm, reshape(x,[xfm.Nd xfm.Nt])),[],1) + reshape(xfm.R1(reshape(x,[xfm.Nd xfm.Nt]),L),[],1), reshape(d,[],1), tol, iters, [], []);

    est =   reshape(est, xfm.msize);
    
    fprintf(1, 'Exit after %i iterations, residual: %G\n', iter, relres);
end

function x = cg(xfm, y, tol, iters, L)
    if nargin < 5
        L       =   [0,0,0,0];
    end
    r = y;
    d = 0;
    x = zeros(xfm.msize);
    for i = 1:iters
        g = xfm'*r - reshape(xfm.R1(reshape(x,[xfm.Nd xfm.Nt]),L),[],1);
        if i == 1
            gamma = 0;
        else
            gamma = (g(:)'*g(:))/(g0(:)'*g0(:));
        end
        d = g + gamma*d;
        q = xfm*d;
        a = (d(:)'*g(:))/(q(:)'*q(:)) + d(:)'*reshape(xfm.R1(reshape(d,[xfm.Nd xfm.Nt]),L),[],1);
        x = x + a*d;
        r = r - a*q;
        g0 = g;
    end
end

function res = times(a,b)
    if a.adjoint
        res =   reshape(mtimes(a,b), [a.Nd(1:2) a.Nd(3), a.Nt]);
    else
        res =   mtimes(a,b);
    end
end

end

methods (Static)
function b = fftfn(b,dims)
    for i = dims
        b   =   fftshift(fft(ifftshift(b, i), [], i), i)/sqrt(size(b,i));
    end
end
function b = fftfn_ns(b,dims)
    for i = dims
        b   =   fft(b, [], i)/sqrt(size(b,i));
    end
end

function b = ifftfn(b,dims)
    for i = dims
        b   =   fftshift(ifft(ifftshift(b, i), [], i), i)*sqrt(size(b,i));
    end
end
function b = ifftfn_ns(b,dims)
    for i = dims
        b   =   ifft(b, [], i)*sqrt(size(b,i));
    end
end
function x = size(b)
    [x(1) x(2) x(3) x(4)]   =   size(b);
end
function x = R1(x, L)    
    % cyclic spatial boundary conditions
    % non-cyclic boundary conditionsfor time dimension
    if size(x,4) > 3
    x = L(1)*(-1*circshift(x,-1,1) + 2*x -1*circshift(x,1,1)) + ...
        L(2)*(-1*circshift(x,-1,2) + 2*x -1*circshift(x,1,2)) + ...
        L(3)*(-1*circshift(x,-1,3) + 2*x -1*circshift(x,1,3)) + ...
        L(4)*cat(4, x(:,:,:,1) - x(:,:,:,2),...
                    -1*x(:,:,:,1:end-2) + 2*x(:,:,:,2:end-1) -1*x(:,:,:,3:end),...
                    -1*x(:,:,:,end-1) + x(:,:,:,end));
    else
    x = L(1)*(-1*circshift(x,-1,1) + 2*x -1*circshift(x,1,1)) + ...
        L(2)*(-1*circshift(x,-1,2) + 2*x -1*circshift(x,1,2)) + ...
        L(3)*(-1*circshift(x,-1,3) + 2*x -1*circshift(x,1,3)) + ...
        L(4)*(-1*circshift(x,-1,4) + 2*x -1*circshift(x,1,4));
    end
end
end

end
