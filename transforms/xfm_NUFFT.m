classdef xfm_NUFFT < xfm
%   NUFFT Linear Operator
%   Forward transforms images to multi-coil, arbitrary non-Cartesian k-space
%   along with the adjoint (multi-coil k-space to combined image)
%
%   Mark Chiew
%   mchiew@fmrib.ox.ac.uk
%   July 2015
%
%   NB: Requires the nufft portion of Jeff Fessler's irt toolbox
%       See http://web.eecs.umich.edu/~fessler/irt/fessler.tgz
%
%   Required Inputs:
%       dims    =   [Nx, Ny, Nz, Nt] 4D vector of image dimensions
%       k       =   [Nsamp, Nt, 2] (2D) or [Nsamp, Nt, 3] (3D) 
%                   array of sampled k-space positions
%                   (in radians, normalised range -pi to pi)
%
%   Optional Inputs:
%       coils   =   [Nx, Ny, Nz, Nc] array of coil sensitivities 
%                   Defaults to single channel i.e. ones(Nx,Ny,Nz)
%       wi      =   Density compensation weights
%       Jd      =   Size of interpolation kernel
%                   Defaults to [6,6]
%       Kd      =   Size of upsampled grid
%                   Defaults to 200% of image size
%       shift   =   NUFFT shift factor
%                   Defaults to 50% of image size
%
%   Usage:
%           Forward transforms can be applied using the "*" or ".*" operators
%           or equivalently the "mtimes" or "times" functions
%           The input can be either the "Casorati" matrix (Nx*Ny*Nz, Nt) or
%           an n-d image array
%           The adjoint transform can be accessed by transposing the transform
%           object 
%           The difference between "*" and ".*" is only significant for the 
%           adjoint transform, and changes  the shape of the image ouptut:
%               "*" produces data that is in matrix form (Nx*Ny*Nz, Nt) whereas
%               ".*" produces data in n-d  array form (Nx, Ny, Nz, Nt) 

properties (SetAccess = protected, GetAccess = public)
    k       =   [];
    w       =   [];
    norm    =   1;
    Jd      =   [6,6];
    Kd      =   [];
    shift   =   [];
    st;
end

methods
function res = xfm_NUFFT(dims, coils, fieldmap_struct, k, varargin)

    %   Base class constructor
    res =   res@xfm(dims, coils, fieldmap_struct);
    
    %   Parse remaining inputs
    p   =   inputParser;

    %   Input validation functions
    lengthValidator =   @(x) length(x) == 2 || length(x) == 3;

    %   Input options
    p.addParamValue('wi',       [],                     @(x) size(x,2) == dims(4)||isscalar(x));
    p.addParamValue('Jd',       [6,6,6],                lengthValidator);
    p.addParamValue('Kd',       2*dims(1:3),            lengthValidator);
    p.addParamValue('shift',    floor(dims(1:3)/2),     lengthValidator);
    p.addParamValue('mean',     true,                   @islogical);

    p.parse(varargin{:});
    p   =   p.Results;

    res.Jd      =   p.Jd;
    res.Kd      =   p.Kd;
    res.shift   =   p.shift;

    res.k       =   k;
    res.dsize   =   [size(k,1) res.Nt res.Nc];

    disp('Initialising NUFFT(s)');
    nd  =   (res.Nd(3) > 1) + 2;
    for t = res.Nt:-1:1
        st(t)   =   nufft_init(squeeze(k(:, t, 1:nd)),...
                               res.Nd(1:nd),...
                               p.Jd(1:nd),...
                               p.Kd(1:nd),...
                               p.shift(1:nd));
    end
    res.st  =   st;
    if isempty(p.wi)
    disp('Generating Density Compensation Weights');
    %   Use (Pipe 1999) fixed point method
        for t = 1:res.Nt
            res.w(:,t)  =   ones(size(k,1),1);
            for ii = 1:5
                res.w(:,t)  =   res.w(:,t)./real(res.st(t).p*(res.st(t).p'*res.w(:,t)));
            end
        end
    elseif p.wi == 0
        w   =   ones(size(k,1),1);
        for ii = 1:5
            w  =   w./real(res.st(t).p*(res.st(t).p'*w));
        end
        res.w = repmat(w,1,res.Nt);
    elseif isscalar(p.wi)
        res.w   =   repmat(p.wi, 1, res.Nt);
    else
        res.w   =   reshape(p.wi, [], res.Nt);
    end
    res.w       =   sqrt(res.w);
    res.norm    =   sqrt(res.st(1).sn(ceil(end/2),ceil(end/2),ceil(end/2))^(-2)/prod(res.st(1).Kd));


end

function res = calcToeplitzEmbedding(a,idx)
    %   Right now assumes square 2D matrix
    %   Will generalise later....
    %   Computes first column of the block-circulant embedding for the block toeplitz A'A
    %   Need to use NUFFT twice to get 2 columns of A'A to do this
    %   A'A is conjugate symmetric, but within toeplitz blocks are NOT symmetric
    %
    %   First compute circulant embeddings for each Toeplitz block
    %   Then compute block-circulant embedding across blocks
    %
    %   Here's a 2D, 2x2 example
    %
    %   Block Toeplitz T=A'A 4x4 matrix (2x2 blocks of 2x2):
    %
    %                               a b'| c'd'
    %                               b a | e'c'
    %                               --- + ---
    %                               c e | a b' 
    %                               d c | b a
    %
    %   Note this matrix is globally conjugate symmetric, and the block structure is also,
    %   but within each block they are not (at least not for the off-diagonal blocks)
    %   
    %   To get all the degrees of freedom, we need the first column and Nth column (where N
    %   is one of the dimensions. i.e., we need the first and last column of a column-block
    %
    %   This are easily estimated from A'A*[1;0;0;0] and A'A*[0;1;0;0]
    %
    %   Now we want to construct the first column of the block circulant embedding:
    %   The block circulant embedding C is constructed by first embedding each block within
    %   its circulant embedding (we use 0 for the arbitrary point):
    %
    %                               a b'0 b | c'd'0 e'
    %                               b a b'0 | e'c'd'0
    %                               0 b a b'| 0 e'c'd'
    %                               b'0 b a | d'0 e'c'
    %                               ------- + -------
    %                               c e 0 d | a b'0 b
    %                               d c e 0 | b a b'0
    %                               0 d c e | 0 b a b'
    %                               e 0 d c | b'0 b a
    %                               
    %   and then constructing the block-level embedding:
    %
    %                               a b'0 b | c'd'0 e'| 0 0 0 0 | c e 0 d
    %                               b a b'0 | e'c'd'0 | 0 0 0 0 | d c e 0
    %                               0 b a b'| 0 e'c'd'| 0 0 0 0 | 0 d c e
    %                               b'0 b a | d'0 e'c'| 0 0 0 0 | e 0 d c
    %                               ------- + ------- + ------- + -------
    %                               c e 0 d | a b'0 b | c'd'0 e | 0 0 0 0
    %                               d c e 0 | b a b'0 | e'c'd'0 | 0 0 0 0
    %                               0 d c e | 0 b a b'| 0 e'c'd | 0 0 0 0
    %                               e 0 d c | b'0 b a | d'0 e'c | 0 0 0 0
    %                               ------- + ------- + ------- + -------
    %                               0 0 0 0 | c e 0 d | a b'0 b | c'd'0 e
    %                               0 0 0 0 | d c e 0 | b a b'0 | e'c'd'0
    %                               0 0 0 0 | 0 d c e | 0 b a b'| 0 e'c'd
    %                               0 0 0 0 | e 0 d c | b'0 b a | d'0 e'c
    %                               ------- + ------- + ------- + -------
    %                               c'd'0 e'| 0 0 0 0 | c e 0 d | a b'0 b 
    %                               e'c'd'0 | 0 0 0 0 | d c e 0 | b a b'0 
    %                               0 e'c'd'| 0 0 0 0 | 0 d c e | 0 b a b'
    %                               d'0 e'c'| 0 0 0 0 | e 0 d c | b'0 b a 
    %
    %   In general, for an NxN image, we get an N^2 x N^2 Block-Toeplitz matrix (NxN blocks of NxN),
    %   and the circulant embedding is (2^d)N^2 x (2^d)N^2, where d=dimension (2 in this case)
    %   so that the total dimension is 4N^2 x 4N^2
    %
    %   Multiplication by this block-circulant matrix is completely characterised by its first column
    %   That is, the diagonalisation C = F*DF, where F are DFTs (where we can use FFT for O(N log N) 
    %   multiplication rather than O(N^2), and diag(D) = FFT(C(:,1))
    %   Intuitively, consider that circulant matrices perform circular convolutions, so that appealing
    %   to the Fourier convolution theorem, we can simply perform FFTs, point-wise multiply, and iFFT back
    %
    %   Because the upper left 
    %
    %   Practically, the first column of C is completely determined by the two columns of A'A we extracted
    %   Then C(:,1) should be reshaped into an 2Nx2N PSF tensor:
    %
    %                               a 
    %                               b
    %                               0
    %                               b
    %                               c   
    %                               d     a c 0 c'
    %                               0     b d 0 e'
    %                               e  =  0 0 0 0
    %                               0     b'e 0 d'
    %                               0    
    %                               0    
    %                               0
    %                               c
    %                               e
    %                               0
    %                               d
    %
    %   While we don't FFTshift in practice, shifting this 2D tensor makes its nature as a PSF more evident:
    %   
    %                                   0 0 0 0 
    %                                   0 d'b'e
    %                                   0 c'a c
    %                                   0'e'b d
    %                           
    %   Once constructed, then A'A*x can be computed via iFFT(FFT(PSF).*FFT(padarray(x)))
    %   This greatly speeds up computation of A'Ax, from O(N^2) to O(N log N)

    disp('Computing Toeplitz Embedding')
    if nargin < 2
        idx =   1:a.Nt;
    end
    N   =   sqrt(a.msize(1));
    Nt  =   length(idx);
    x   =   zeros(2*N,N,Nt);
    tmp =   zeros(N);
    tmp(1,1)    =   1;
    st  =   a.st(idx);
    w   =   a.w.^2;
    for t = 1:Nt
        x(1:N,:,t)  =   nufft_adj(w(:,t).*nufft(tmp, st(t)), st(t));
    end
    tmp([1 N],1)    =   [0;1];
    for t = 1:Nt
        x([N+2:end N+1],:,t)  =   nufft_adj(w(:,t).*nufft(tmp, st(t)), st(t));
    end
    x(N+1,:,:)  =   0;
    clear tmp;
    
    res =   zeros(4*N^2,Nt);
    res(1:2*N^2,:)  =   reshape(x,[],Nt);
    res(2*N^2+2*N+1:end,:)  =   conj(reshape(x([1 end:-1:2],end:-1:2,:),[],Nt));

    res     =   fft2(reshape(res,2*N,2*N,[]))*a.norm^2;
end

function b = mtimes_Toeplitz(a,T,b)
    Nt  =   size(T,3);
    Nd  =   a.Nd(1:2);
    S   =   a.S;
    b   =   reshape(b,[],Nt);
    tmp =   zeros(2*Nd(1),2*Nd(2),1,1,a.Nc);
    tmp2=   zeros(2*Nd(1),2*Nd(2),1,1,a.Nc);
    for t = 1:Nt
        tmp(1:Nd(1),1:Nd(2),1,1,:)  =  S*b(:,t); 
        tmp2    =   ifft2(T(:,:,t).*fft2(tmp)); 
        b(:,t)  =   reshape(S'*tmp2(1:Nd(1),1:Nd(2),1,1,:),[],1);
    end
end

function res = mtimes(a,b,idx)
    if nargin < 3
        idx =   1:a.Nt;
    end
    %   Property access in MATLAB classes is very slow, so any property accessed
    %   more than once is copied into the local scope first
    nt  =   length(idx);
    st  =   a.st(idx);

    if a.adjoint
    %   Adjoint NUFFT and coil transform
        res =   zeros([a.Nd, nt a.Nc]);
        b   =   bsxfun(@times, b, a.w(:,idx));
        for t = 1:nt
            res(:,:,:,t,:)  =   nufft_adj(squeeze(b(:,t,:)), st(t));
        end
        res =   reshape(a.norm*(a.S'*res), [], nt);
    else
    %   Forward NUFFT and coil transform
        res =   zeros([a.dsize(1) nt a.dsize(3)]);
        tmp =   a.norm*(a.S*b);
        for t = 1:nt
            res(:,t,:)  =   nufft(squeeze(tmp(:,:,:,t,:)), st(t));
        end
        res =   bsxfun(@times, res, a.w(:,idx));
    end

end

function res = mean(a,b)
    nd  =   (a.Nd(3) > 1) + 2;
    st  =   nufft_init(reshape(a.k,[],nd),...
                       a.Nd(1:nd),...
                       a.Jd(1:nd),...
                       a.Kd(1:nd),...
                       a.shift(1:nd));
    %   Use (Pipe 1999) fixed point method
    w   =   ones(numel(a.k)/nd,1);
    for ii = 1:20
        tmp =   st.p*(st.p'*w);
        w   =   w./real(tmp);
    end
    w   =   w*sqrt(st.sn(ceil(end/2),ceil(end/2))^(-2)/prod(st.Kd));
    res =   a.S'*(nufft_adj(bsxfun(@times, reshape(b,[],a.Nc), w), st));
end

end
end