classdef xfm_DFT < xfm
%   DFT Linear Operator
%   Forward transforms images to multi-coil, arbitrary non-Cartesian k-space
%   along with the adjoint (multi-coil k-space to combined image)
%   NB. Does not depend on Fessler's IRT or NUFFT
%
%   Mark Chiew
%   mchiew@fmrib.ox.ac.uk
%   Apr 2019
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
%       shift   =   DFT shift factor
%                   Defaults to 50% of image size
%       DFT     =   Pre-calculated DFT matrix
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
    shift   =   [];
    PSF     =   []; % Eigenvalues of circulant embedding 
    E       =   [];
end

methods
function res = xfm_DFT(dims, coils, fieldmap_struct, k, varargin)

    %   Base class constructor
    res =   res@xfm(dims, coils, fieldmap_struct);
    
    %   Parse remaining inputs
    p   =   inputParser;

    %   Input validation functions
    lengthValidator =   @(x) length(x) == 2 || length(x) == 3;

    %   Input options
    p.addParamValue('shift',    floor(dims(1:3)/2),     lengthValidator);
    p.addParamValue('PSF',      []);
    p.addParamValue('DFT',      []);

    p.parse(varargin{:});
    p   =   p.Results;

    res.k       =   k;
    res.shift   =   p.shift;
    res.dsize   =   [size(k,1) res.Nt res.Nc];

    res.E       =   p.DFT;
    if isempty(res.E)
        [x,y,z] =   meshgrid((0:res.Nd(1)-1) - p.shift(1), (0:res.Nd(2)-1) - p.shift(2), (0:res.Nd(3)-1) - p.shift(3));
        res.E   =   zeros(size(k,1),numel(x),res.Nt);
        for t = 1:res.Nt
            if res.Nd(3) > 1    % 3D
                res.E(:,:,t)    =   exp(-1j*(k(:,t,1)*x(:)' + k(:,t,2)*y(:)' + k(:,t,3)*z(:)'));

            else
                res.E(:,:,t)    =   exp(-1j*(k(:,t,1)*x(:)' + k(:,t,2)*y(:)'));
            end
        end
    end

    res.PSF     =   p.PSF;
    if isempty(res.PSF)
        res.PSF =   res.calcToeplitzEmbedding();
    end
end

function T = calcToeplitzEmbedding(a)
    %   
    %   Should work on arbitrary shaped 3D [Nx, Ny, Nz] problems
    %   
    %   Computes first column of the block-circulant embedding for the block toeplitz A'A
    %   Need to use DFT twice to get 2 columns of A'A to do this
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

    %       Explicit 3D example (2x2x2) 
    %       Symmetric 3-level 8x8 block Toeplitz A'A matrix
    %       
    %       a b'| c'd'| f'j'| l'n'
    %       b a | e'c'| g'f'| m'l'
    %       --- + --- + --- + ---
    %       c e | a b'| h'k'| f'j'
    %       d c | b a | i'h'| g'f'
    %       --- + --- + --- + ---
    %       f g | h i | a b'| c'd'
    %       j f | k h | b a | e'c'
    %       --- + --- + --- + ---
    %       l m | f g | c e | a b'
    %       n l | j f | d c | b a
    %
    %       Step 1 of circulant embedding:
    %       Circulant embed each of the lowest scale 2x2 Toeplitz blocks 
    %       (2x8) x (2x8)
    %
    %       a b'0 b | c'd'0 e'| f'j'0 g'| l'n'0 m'
    %       b a b'0 | e'c'd'0 | g'f'j'0 | m'l'n'0
    %       0 b a b'| 0 e'c'd'| 0 g'f'j'| 0 m'l'n'
    %       b'0 b a | d'0 e'c'| j'0 g'f'| n'0 m'l'
    %       ------- + ------- + ------- + -------
    %       c e 0 d | a b'0 b | h'k'0 i'| f'j'0 g'
    %       d c e 0 | b a b'0 | i'h'k'0 | g'f'j'0 
    %       0 d c e | 0 b a b'| 0 i'h'k'| 0 g'f'j'
    %       e 0 d c | b'0 b a | k'0 i'h'| j'0 g'f'
    %       ------- + ------- + ------- + -------
    %       f g 0 j | h i 0 k | a b'0 b | c'd'0 e'
    %       j f g 0 | k h i 0 | b a b'0 | e'c'd'0 
    %       0 j f g | 0 k h i | 0 b a b'| 0 e'c'd'
    %       g 0 j f | i 0 k h | b'0 b a | d'0 e'c'
    %       ------- + ------- + ------- + ------- 
    %       l m 0 n | f g 0 j | c e 0 d | a b'0 b 
    %       n l m 0 | j f g 0 | d c e 0 | b a b'0 
    %       0 n l m | 0 j f g | 0 d c e | 0 b a b'
    %       m 0 n l | g 0 j f | e 0 d c | b'0 b a 
    %
    %       Step 2 of circulant embedding:
    %       Circulant embed each of the second level 8x8 block Toeplitz blocks
    %       (4x8) x (4x8)
    %
    %       a b'0 b   c'd'0 e'  0 0 0 0   c e 0 d | f'j'0 g'  l'n'0 m'  0 0 0 0   h'k'0 i'
    %       b a b'0   e'c'd'0   0 0 0 0   d c e 0 | g'f'j'0   m'l'n'0   0 0 0 0   i'h'k'0 
    %       0 b a b'  0 e'c'd'  0 0 0 0   0 d c e | 0 g'f'j'  0 m'l'n'  0 0 0 0   0 i'h'k'
    %       b'0 b a   d'0 e'c'  0 0 0 0   e 0 d c | j'0 g'f'  n'0 m'l'  0 0 0 0   k'0 i'h'
    %                                             |                                      
    %       c e 0 d   a b'0 b   c'd'0 e   0 0 0 0 | h'k'0 i'  f'j'0 g'  l'n'0 m'  0 0 0 0
    %       d c e 0   b a b'0   e'c'd'0   0 0 0 0 | i'h'k'0   g'f'j'0   m'l'n'0   0 0 0 0
    %       0 d c e   0 b a b'  0 e'c'd   0 0 0 0 | 0 i'h'k'  0 g'f'j'  0 m'l'n'  0 0 0 0
    %       e 0 d c   b'0 b a   d'0 e'c   0 0 0 0 | k'0 i'h'  j'0 g'f'  n'0 m'l'  0 0 0 0
    %                                             |                                       
    %       0 0 0 0   c e 0 d   a b'0 b   c'd'0 e | f'j'0 g'  l'n'0 m'  0 0 0 0   h'k'0 i'
    %       0 0 0 0   d c e 0   b a b'0   e'c'd'0 | g'f'j'0   m'l'n'0   0 0 0 0   i'h'k'0 
    %       0 0 0 0   0 d c e   0 b a b'  0 e'c'd | 0 g'f'j'  0 m'l'n'  0 0 0 0   0 i'h'k'
    %       0 0 0 0   e 0 d c   b'0 b a   d'0 e'c | j'0 g'f'  n'0 m'l'  0 0 0 0   k'0 i'h'
    %                                             |                                        
    %       c'd'0 e'  0 0 0 0   c e 0 d   a b'0 b | l'n'0 m'  0 0 0 0   h'k'0 i'  f'j'0 g'
    %       e'c'd'0   0 0 0 0   d c e 0   b a b'0 | m'l'n'0   0 0 0 0   i'h'k'0   g'f'j'0 
    %       0 e'c'd'  0 0 0 0   0 d c e   0 b a b'| 0 m'l'n'  0 0 0 0   0 i'h'k'  0 g'f'j'
    %       d'0 e'c'  0 0 0 0   e 0 d c   b'0 b a | n'0 m'l'  0 0 0 0   k'0 i'h'  j'0 g'f'
    %       ------------------------------------- + -------------------------------------  
    %       f g 0 j   h i 0 k   0 0 0 0   l m 0 n | a b'0 b   c'd'0 e'  0 0 0 0   c e 0 d
    %       j f g 0   k h i 0   0 0 0 0   n l m 0 | b a b'0   e'c'd'0   0 0 0 0   d c e 0
    %       0 j f g   0 k h i   0 0 0 0   0 n l m | 0 b a b'  0 e'c'd'  0 0 0 0   0 d c e
    %       g 0 j f   i 0 k h   0 0 0 0   m 0 n l | b'0 b a   d'0 e'c'  0 0 0 0   e 0 d c
    %                                             |                                        
    %       l m 0 n   f g 0 j   h i 0 k   0 0 0 0 | c e 0 d   a b'0 b   c'd'0 e   0 0 0 0
    %       n l m 0   j f g 0   k h i 0   0 0 0 0 | d c e 0   b a b'0   e'c'd'0   0 0 0 0
    %       0 n l m   0 j f g   0 k h i   0 0 0 0 | 0 d c e   0 b a b'  0 e'c'd   0 0 0 0
    %       m 0 n l   g 0 j f   i 0 k h   0 0 0 0 | e 0 d c   b'0 b a   d'0 e'c   0 0 0 0
    %                                             |                                        
    %       0 0 0 0   l m 0 n   f g 0 j   h i 0 k | 0 0 0 0   c e 0 d   a b'0 b   c'd'0 e
    %       0 0 0 0   n l m 0   j f g 0   k h i 0 | 0 0 0 0   d c e 0   b a b'0   e'c'd'0
    %       0 0 0 0   0 n l m   0 j f g   0 k h i | 0 0 0 0   0 d c e   0 b a b'  0 e'c'd
    %       0 0 0 0   m 0 n l   g 0 j f   i 0 k h | 0 0 0 0   e 0 d c   b'0 b a   d'0 e'c
    %                                             |                                        
    %       h i 0 k   0 0 0 0   l m 0 n   f g 0 j | c'd'0 e'  0 0 0 0   c e 0 d   a b'0 b
    %       k h i 0   0 0 0 0   n l m 0   j f g 0 | e'c'd'0   0 0 0 0   d c e 0   b a b'0
    %       0 k h i   0 0 0 0   0 n l m   0 j f g | 0 e'c'd'  0 0 0 0   0 d c e   0 b a b
    %       i 0 k h   0 0 0 0   m 0 n l   g 0 j f | d'0 e'c'  0 0 0 0   e 0 d c   b'0 b a
    %
    %       Step 3 of circulant embedding:
    %       Circulant embed the third level 16x16 block Toeplitz blocks
    %       (8x8) x (8x8)
    %
    %       a b'0 b   c'd'0 e'  0 0 0 0   c e 0 d | f'j'0 g'  l'n'0 m'  0 0 0 0   h'k'0 i'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | f g 0 j   h i 0 k   0 0 0 0   l m 0 n
    %       b a b'0   e'c'd'0   0 0 0 0   d c e 0 | g'f'j'0   m'l'n'0   0 0 0 0   i'h'k'0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | j f g 0   k h i 0   0 0 0 0   n l m 0
    %       0 b a b'  0 e'c'd'  0 0 0 0   0 d c e | 0 g'f'j'  0 m'l'n'  0 0 0 0   0 i'h'k'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 j f g   0 k h i   0 0 0 0   0 n l m
    %       b'0 b a   d'0 e'c'  0 0 0 0   e 0 d c | j'0 g'f'  n'0 m'l'  0 0 0 0   k'0 i'h'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | g 0 j f   i 0 k h   0 0 0 0   m 0 n l
    %                                             |                                       |                                       |                                      
    %       c e 0 d   a b'0 b   c'd'0 e   0 0 0 0 | h'k'0 i'  f'j'0 g'  l'n'0 m'  0 0 0 0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | l m 0 n   f g 0 j   h i 0 k   0 0 0 0
    %       d c e 0   b a b'0   e'c'd'0   0 0 0 0 | i'h'k'0   g'f'j'0   m'l'n'0   0 0 0 0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | n l m 0   j f g 0   k h i 0   0 0 0 0
    %       0 d c e   0 b a b'  0 e'c'd   0 0 0 0 | 0 i'h'k'  0 g'f'j'  0 m'l'n'  0 0 0 0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 n l m   0 j f g   0 k h i   0 0 0 0
    %       e 0 d c   b'0 b a   d'0 e'c   0 0 0 0 | k'0 i'h'  j'0 g'f'  n'0 m'l'  0 0 0 0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | m 0 n l   g 0 j f   i 0 k h   0 0 0 0
    %                                             |                                       |                                       |                                      
    %       0 0 0 0   c e 0 d   a b'0 b   c'd'0 e | f'j'0 g'  l'n'0 m'  0 0 0 0   h'k'0 i'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 0 0 0   l m 0 n   f g 0 j   h i 0 k
    %       0 0 0 0   d c e 0   b a b'0   e'c'd'0 | g'f'j'0   m'l'n'0   0 0 0 0   i'h'k'0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 0 0 0   n l m 0   j f g 0   k h i 0
    %       0 0 0 0   0 d c e   0 b a b'  0 e'c'd | 0 g'f'j'  0 m'l'n'  0 0 0 0   0 i'h'k'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 0 0 0   0 n l m   0 j f g   0 k h i
    %       0 0 0 0   e 0 d c   b'0 b a   d'0 e'c | j'0 g'f'  n'0 m'l'  0 0 0 0   k'0 i'h'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 0 0 0   m 0 n l   g 0 j f   i 0 k h
    %                                             |                                       |                                       |                                      
    %       c'd'0 e'  0 0 0 0   c e 0 d   a b'0 b | l'n'0 m'  0 0 0 0   h'k'0 i'  f'j'0 g'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | h i 0 k   0 0 0 0   l m 0 n   f g 0 j
    %       e'c'd'0   0 0 0 0   d c e 0   b a b'0 | m'l'n'0   0 0 0 0   i'h'k'0   g'f'j'0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | k h i 0   0 0 0 0   n l m 0   j f g 0
    %       0 e'c'd'  0 0 0 0   0 d c e   0 b a b'| 0 m'l'n'  0 0 0 0   0 i'h'k'  0 g'f'j'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 k h i   0 0 0 0   0 n l m   0 j f g
    %       d'0 e'c'  0 0 0 0   e 0 d c   b'0 b a | n'0 m'l'  0 0 0 0   k'0 i'h'  j'0 g'f'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | i 0 k h   0 0 0 0   m 0 n l   g 0 j f
    %       ------------------------------------- + ------------------------------------- + ------------------------------------- + -------------------------------------
    %       f g 0 j   h i 0 k   0 0 0 0   l m 0 n | a b'0 b   c'd'0 e'  0 0 0 0   c e 0 d | f'j'0 g'  l'n'0 m'  0 0 0 0   h'k'0 i'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0
    %       j f g 0   k h i 0   0 0 0 0   n l m 0 | b a b'0   e'c'd'0   0 0 0 0   d c e 0 | g'f'j'0   m'l'n'0   0 0 0 0   i'h'k'0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0
    %       0 j f g   0 k h i   0 0 0 0   0 n l m | 0 b a b'  0 e'c'd'  0 0 0 0   0 d c e | 0 g'f'j'  0 m'l'n'  0 0 0 0   0 i'h'k'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0
    %       g 0 j f   i 0 k h   0 0 0 0   m 0 n l | b'0 b a   d'0 e'c'  0 0 0 0   e 0 d c | j'0 g'f'  n'0 m'l'  0 0 0 0   k'0 i'h'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0
    %                                             |                                       |                                       |                                      
    %       l m 0 n   f g 0 j   h i 0 k   0 0 0 0 | c e 0 d   a b'0 b   c'd'0 e   0 0 0 0 | h'k'0 i'  f'j'0 g'  l'n'0 m'  0 0 0 0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0
    %       n l m 0   j f g 0   k h i 0   0 0 0 0 | d c e 0   b a b'0   e'c'd'0   0 0 0 0 | i'h'k'0   g'f'j'0   m'l'n'0   0 0 0 0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0
    %       0 n l m   0 j f g   0 k h i   0 0 0 0 | 0 d c e   0 b a b'  0 e'c'd   0 0 0 0 | 0 i'h'k'  0 g'f'j'  0 m'l'n'  0 0 0 0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0
    %       m 0 n l   g 0 j f   i 0 k h   0 0 0 0 | e 0 d c   b'0 b a   d'0 e'c   0 0 0 0 | k'0 i'h'  j'0 g'f'  n'0 m'l'  0 0 0 0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0
    %                                             |                                       |                                       |                                      
    %       0 0 0 0   l m 0 n   f g 0 j   h i 0 k | 0 0 0 0   c e 0 d   a b'0 b   c'd'0 e | f'j'0 g'  l'n'0 m'  0 0 0 0   h'k'0 i'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0
    %       0 0 0 0   n l m 0   j f g 0   k h i 0 | 0 0 0 0   d c e 0   b a b'0   e'c'd'0 | g'f'j'0   m'l'n'0   0 0 0 0   i'h'k'0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0
    %       0 0 0 0   0 n l m   0 j f g   0 k h i | 0 0 0 0   0 d c e   0 b a b'  0 e'c'd | 0 g'f'j'  0 m'l'n'  0 0 0 0   0 i'h'k'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0
    %       0 0 0 0   m 0 n l   g 0 j f   i 0 k h | 0 0 0 0   e 0 d c   b'0 b a   d'0 e'c | j'0 g'f'  n'0 m'l'  0 0 0 0   k'0 i'h'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0
    %                                             |                                       |                                       |                                      
    %       h i 0 k   0 0 0 0   l m 0 n   f g 0 j | c'd'0 e'  0 0 0 0   c e 0 d   a b'0 b | l'n'0 m'  0 0 0 0   h'k'0 i'  f'j'0 g'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0
    %       k h i 0   0 0 0 0   n l m 0   j f g 0 | e'c'd'0   0 0 0 0   d c e 0   b a b'0 | m'l'n'0   0 0 0 0   i'h'k'0   g'f'j'0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0
    %       0 k h i   0 0 0 0   0 n l m   0 j f g | 0 e'c'd'  0 0 0 0   0 d c e   0 b a b | 0 m'l'n'  0 0 0 0   0 i'h'k'  0 g'f'j'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0
    %       i 0 k h   0 0 0 0   m 0 n l   g 0 j f | d'0 e'c'  0 0 0 0   e 0 d c   b'0 b a | n'0 m'l'  0 0 0 0   k'0 i'h'  j'0 g'f'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0
    %       ------------------------------------- + ------------------------------------- + ------------------------------------- + -------------------------------------
    %       0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | f g 0 j   h i 0 k   0 0 0 0   l m 0 n | a b'0 b   c'd'0 e'  0 0 0 0   c e 0 d | f'j'0 g'  l'n'0 m'  0 0 0 0   h'k'0 i'
    %       0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | j f g 0   k h i 0   0 0 0 0   n l m 0 | b a b'0   e'c'd'0   0 0 0 0   d c e 0 | g'f'j'0   m'l'n'0   0 0 0 0   i'h'k'0 
    %       0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 j f g   0 k h i   0 0 0 0   0 n l m | 0 b a b'  0 e'c'd'  0 0 0 0   0 d c e | 0 g'f'j'  0 m'l'n'  0 0 0 0   0 i'h'k'
    %       0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | g 0 j f   i 0 k h   0 0 0 0   m 0 n l | b'0 b a   d'0 e'c'  0 0 0 0   e 0 d c | j'0 g'f'  n'0 m'l'  0 0 0 0   k'0 i'h'
    %                                             |                                       |                                       |                                       
    %       0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | l m 0 n   f g 0 j   h i 0 k   0 0 0 0 | c e 0 d   a b'0 b   c'd'0 e   0 0 0 0 | h'k'0 i'  f'j'0 g'  l'n'0 m'  0 0 0 0 
    %       0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | n l m 0   j f g 0   k h i 0   0 0 0 0 | d c e 0   b a b'0   e'c'd'0   0 0 0 0 | i'h'k'0   g'f'j'0   m'l'n'0   0 0 0 0 
    %       0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 n l m   0 j f g   0 k h i   0 0 0 0 | 0 d c e   0 b a b'  0 e'c'd   0 0 0 0 | 0 i'h'k'  0 g'f'j'  0 m'l'n'  0 0 0 0 
    %       0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | m 0 n l   g 0 j f   i 0 k h   0 0 0 0 | e 0 d c   b'0 b a   d'0 e'c   0 0 0 0 | k'0 i'h'  j'0 g'f'  n'0 m'l'  0 0 0 0 
    %                                             |                                       |                                       |                                       
    %       0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 0 0 0   l m 0 n   f g 0 j   h i 0 k | 0 0 0 0   c e 0 d   a b'0 b   c'd'0 e | 0 0 0 0   h'k'0 i'  f'j'0 g'  l'n'0 m'
    %       0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 0 0 0   n l m 0   j f g 0   k h i 0 | 0 0 0 0   d c e 0   b a b'0   e'c'd'0 | 0 0 0 0   i'h'k'0   g'f'j'0   m'l'n'0 
    %       0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 0 0 0   0 n l m   0 j f g   0 k h i | 0 0 0 0   0 d c e   0 b a b'  0 e'c'd | 0 0 0 0   0 i'h'k'  0 g'f'j'  0 m'l'n'
    %       0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 0 0 0   m 0 n l   g 0 j f   i 0 k h | 0 0 0 0   e 0 d c   b'0 b a   d'0 e'c | 0 0 0 0   k'0 i'h'  j'0 g'f'  n'0 m'l'
    %                                             |                                       |                                       |                                       
    %       0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | h i 0 k   0 0 0 0   l m 0 n   f g 0 j | c'd'0 e'  0 0 0 0   c e 0 d   a b'0 b | l'n'0 m'  0 0 0 0   h'k'0 i'  f'j'0 g'
    %       0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | k h i 0   0 0 0 0   n l m 0   j f g 0 | e'c'd'0   0 0 0 0   d c e 0   b a b'0 | m'l'n'0   0 0 0 0   i'h'k'0   g'f'j'0 
    %       0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 k h i   0 0 0 0   0 n l m   0 j f g | 0 e'c'd'  0 0 0 0   0 d c e   0 b a b | 0 m'l'n'  0 0 0 0   0 i'h'k'  0 g'f'j'
    %       0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | i 0 k h   0 0 0 0   m 0 n l   g 0 j f | d'0 e'c'  0 0 0 0   e 0 d c   b'0 b a | n'0 m'l'  0 0 0 0   k'0 i'h'  j'0 g'f'
    %       ------------------------------------- + ------------------------------------- + ------------------------------------- + -------------------------------------
    %       f'j'0 g'  l'n'0 m'  0 0 0 0   h'k'0 i'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | f g 0 j   h i 0 k   0 0 0 0   l m 0 n | a b'0 b   c'd'0 e'  0 0 0 0   c e 0 d
    %       g'f'j'0   m'l'n'0   0 0 0 0   i'h'k'0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | j f g 0   k h i 0   0 0 0 0   n l m 0 | b a b'0   e'c'd'0   0 0 0 0   d c e 0
    %       0 g'f'j'  0 m'l'n'  0 0 0 0   0 i'h'k'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 j f g   0 k h i   0 0 0 0   0 n l m | 0 b a b'  0 e'c'd'  0 0 0 0   0 d c e
    %       j'0 g'f'  n'0 m'l'  0 0 0 0   k'0 i'h'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | g 0 j f   i 0 k h   0 0 0 0   m 0 n l | b'0 b a   d'0 e'c'  0 0 0 0   e 0 d c
    %                                             |                                       |                                       |                                      
    %       h'k'0 i'  f'j'0 g'  l'n'0 m'  0 0 0 0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | l m 0 n   f g 0 j   h i 0 k   0 0 0 0 | c e 0 d   a b'0 b   c'd'0 e   0 0 0 0
    %       i'h'k'0   g'f'j'0   m'l'n'0   0 0 0 0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | n l m 0   j f g 0   k h i 0   0 0 0 0 | d c e 0   b a b'0   e'c'd'0   0 0 0 0
    %       0 i'h'k'  0 g'f'j'  0 m'l'n'  0 0 0 0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 n l m   0 j f g   0 k h i   0 0 0 0 | 0 d c e   0 b a b'  0 e'c'd   0 0 0 0
    %       k'0 i'h'  j'0 g'f'  n'0 m'l'  0 0 0 0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | m 0 n l   g 0 j f   i 0 k h   0 0 0 0 | e 0 d c   b'0 b a   d'0 e'c   0 0 0 0
    %                                             |                                       |                                       |                                      
    %       0 0 0 0   h'k'0 i'  f'j'0 g'  l'n'0 m'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 0 0 0   l m 0 n   f g 0 j   h i 0 k | 0 0 0 0   c e 0 d   a b'0 b   c'd'0 e
    %       0 0 0 0   i'h'k'0   g'f'j'0   m'l'n'0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 0 0 0   n l m 0   j f g 0   k h i 0 | 0 0 0 0   d c e 0   b a b'0   e'c'd'0
    %       0 0 0 0   0 i'h'k'  0 g'f'j'  0 m'l'n'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 0 0 0   0 n l m   0 j f g   0 k h i | 0 0 0 0   0 d c e   0 b a b'  0 e'c'd
    %       0 0 0 0   k'0 i'h'  j'0 g'f'  n'0 m'l'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 0 0 0   m 0 n l   g 0 j f   i 0 k h | 0 0 0 0   e 0 d c   b'0 b a   d'0 e'c
    %                                             |                                       |                                       |                                      
    %       l'n'0 m'  0 0 0 0   h'k'0 i'  f'j'0 g'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | h i 0 k   0 0 0 0   l m 0 n   f g 0 j | c'd'0 e'  0 0 0 0   c e 0 d   a b'0 b
    %       m'l'n'0   0 0 0 0   i'h'k'0   g'f'j'0 | 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | k h i 0   0 0 0 0   n l m 0   j f g 0 | e'c'd'0   0 0 0 0   d c e 0   b a b'0
    %       0 m'l'n'  0 0 0 0   0 i'h'k'  0 g'f'j'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | 0 k h i   0 0 0 0   0 n l m   0 j f g | 0 e'c'd'  0 0 0 0   0 d c e   0 b a b
    %       n'0 m'l'  0 0 0 0   k'0 i'h'  j'0 g'f'| 0 0 0 0   0 0 0 0   0 0 0 0   0 0 0 0 | i 0 k h   0 0 0 0   m 0 n l   g 0 j f | d'0 e'c'  0 0 0 0   e 0 d c   b'0 b a

    disp('Computing Toeplitz Embedding')
    Nd  =   a.Nd;
    Nt  =   a.Nt;
    E   =   a.E;
    EE  =   zeros(size(E,2),size(E,2),size(E,3));

    %   Need 2^(d-1) columns of A'A
    %   4 columns for 3D problems
    x1  =   zeros([Nd(1) prod(Nd(2:3)) Nt],'single');
    x2  =   zeros([Nd(1) prod(Nd(2:3)) Nt],'single');
    x3  =   zeros([Nd(1) prod(Nd(2:3)) Nt],'single');
    x4  =   zeros([Nd(1) prod(Nd(2:3)) Nt],'single');

    T   =   zeros(8*prod(Nd), Nt,'single');

    %   First column
    tmp =   zeros(Nd,'single');
    tmp(1,1,1)  =   1;
    for t = 1:Nt
        EE(:,:,t)   =   E(:,:,t)'*E(:,:,t);
        x1(:,:,t)   =   reshape(EE(:,:,t)*tmp(:), Nd(1), []);
    end

    %   Second column
    tmp =   zeros(Nd,'single');
    tmp(end,1,1)    =   1;
    for t = 1:Nt
        x2(:,:,t)   =   reshape(EE(:,:,t)*tmp(:), Nd(1), []);
        x2(end,:,t) =   0;
    end

    %   Third column
    tmp =   zeros(Nd,'single');
    tmp(1,end,1)    =   1;
    for t = 1:Nt
        x3(:,:,t)   =   reshape(EE(:,:,t)*tmp(:), Nd(1), []);
    end

    %   Fourth column
    tmp =   zeros(Nd,'single');
    tmp(end,end,1)  =   1;
    for t = 1:Nt
        x4(:,:,t)   =   reshape(EE(:,:,t)*tmp(:), Nd(1), []);
        x4(end,:,t) =   0;
    end

    %   Perform first level embedding
    M1  =   cat(1, x1, circshift(x2,1,1));
    clear x1 x2;
    M2  =   cat(1, x3, circshift(x4,1,1));
    clear x3 x4;


    %   Perform second level embedding
    M2  =   reshape(M2, [2*Nd(1) Nd(2:3) Nt]);
    M2(:,end,:,:)   =   0;
    M1  =   reshape(M1, [], Nd(3), Nt);
    M2  =   reshape(M2, [], Nd(3), Nt);
    M3  =   cat(1, M1,  circshift(M2,2*Nd(1),1));
    
    clear M1 M2;

    %   Perform third (final) level embedding
    M3  =   reshape(M3, 2*Nd(1), 2*Nd(2), Nd(3), Nt);

    T(1:4*prod(Nd),:) = reshape(M3, [], Nt);

    M3  =   circshift(flipdim(M3,3),1,3);
    M3  =   circshift(flipdim(M3,2),1,2);
    M3  =   circshift(flipdim(M3,1),1,1);

    for i = 1
        T(4*prod(Nd)+4*(i-1)*prod(Nd(1:2))+1:4*prod(Nd)+4*i*prod(Nd(1:2)),:)    =   0;
    end
    for i = 2:Nd(3)
        T(4*prod(Nd)+4*(i-1)*prod(Nd(1:2))+1:4*prod(Nd)+4*i*prod(Nd(1:2)),:)    =   conj(reshape(M3(:,:,i,:),[],Nt));
    end

    T   =   a.fftfn_ns(reshape(T,[2*Nd Nt]), 1:3);

end


function b = mtimes2(a,b)
    %   If mtimes(A,b) = A*b, mtimes2(A,b) = A'A*b
    %   If Toeplitz embedding is available, uses that
    %   otherwise computes by mtimes(A',mtimes(A,b))
    PSF =   a.PSF;
    Nt  =   a.Nt;
    Nd  =   a.Nd;
    Nc  =   a.Nc;
    S   =   a.S;
    dim =   size(b);
    b   =   reshape(b,[],Nt);
    
    if Nd(3) == 1
    %   2D mode
        tmp =   zeros(2*Nd(1),2*Nd(2),1,1,Nc);
        tmp2=   zeros(2*Nd(1),2*Nd(2),1,1,Nc);
        for t = 1:Nt
            tmp(1:Nd(1),1:Nd(2),1,1,:)  =  S*b(:,t); 
            tmp2(:,:,1,1,:) =   ifft2(PSF(:,:,1,t).*fft2(tmp)); 
            b(:,t)  =   reshape(S'*tmp2(1:Nd(1),1:Nd(2),1,1,:),[],1);
        end
    else
    %   3D mode, break out coil loop for reduced memory footprint
        for t = 1:Nt
            out =   zeros(size(b,1),1);
            for c = 1:Nc
                tmp =   zeros(2*Nd(1),2*Nd(2),2*Nd(3));
                tmp(1:Nd(1),1:Nd(2),1:Nd(3))  =  mtimes(S,b(:,t),c); 
                tmp =   ifftn(PSF(:,:,:,t).*fftn(tmp)); 
                out =   out + reshape(mtimes(S',tmp(1:Nd(1),1:Nd(2),1:Nd(3)),c),[],1);
            end
            b(:,t)  =   out;
        end
    end

    %   Return b in the same shape it came in
    b   =   reshape(b, dim);    
end

function res = mtimes(a,b,idx)
    if nargin < 3
        idx =   1:a.Nt;
    end
    %   Property access in MATLAB classes is very slow, so any property accessed
    %   more than once is copied into the local scope first
    nt  =   length(idx);
    nd  =   a.Nd;
    nc  =   a.Nc;
    ds  =   a.dsize;
    E   =   a.E(:,:,idx);

    if a.adjoint
    %   Adjoint DFT and coil transform
        res =   zeros([nd, nt nc]);
        for t = 1:nt
            res(:,:,:,t,:)  =   reshape(E(:,:,t)'*squeeze(b(:,t,:)),[nd,1,nc]);
        end
        res =   reshape(a.S'*res, [], nt);
    else
    %   Forward DFT and coil transform
        res =   zeros([ds(1) nt ds(3)]);
        tmp =   a.S*b;
        for t = 1:nt
            res(:,t,:)  =   reshape(E(:,:,t)*reshape(tmp(:,:,:,t,:),[],ds(3)),[],1,ds(3));
        end
    end

end


end
end
