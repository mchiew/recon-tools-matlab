classdef xfm_MB < xfm
%   MB-FFT Linear Operator
%   Forward transforms images to multi-coil, multi-band Cartesian k-space
%   along with the adjoint (multi-coil k-space to combined image)
%
%   Mark Chiew
%   mchiew@fmrib.ox.ac.uk
%   Dec 2016
%
%   Usage:  
%       xfm     =   xfm_MB(dims, coils, field, ['mask', mask, 'phs', phs]);
%               
%
%
%   Required Inputs:
%       dims    -   [Nx, Ny, Nz, Nt] 4D vector of image dimensions
%       coils   -   [Nx, Ny, Nz, Nc] array of coil sensitivities 
%                   Pass empty array to default to single channel 
%                   uniform sensitivity i.e. ones(Nx,Ny,Nz)
%       field   -   [Ns, 1] array of structs containing fieldmap information
%                   (one for each MB set)
%                   .field: [Nx, Ny, Nz] field map in Hz
%                   .t:     [Kx, Ky, Kz, n] time of acquisition in s
%                   .L:     scalar value for number of interpolants
%                   .idx:   [Nt, 1] vector indicating the sequence
%                           of different trajectory timings, indexing
%                           the "n" different timing arrays in .t
%
%   Optional Inputs:
%       mask    -   [Nx, Ny, Nz, Nt] logical sampling mask
%       phs     -   [Nx, Ny, Nz, Nt] array of phase data (inhomogeneity + rf phases)
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
    mask    =   [];
    phs     =   [];
    Ns      =   1;
    MB      =   1;
end

methods
function res = xfm_MB(dims, coils, fieldmap_struct, varargin)

    %   Base class constructor
    res =   res@xfm(dims, coils, fieldmap_struct);
    
    %   Parse remaining inputs
    p   =   inputParser;

    %   Input options
    p.addParameter('mask',     true(dims));
    p.addParameter('phs',   1         );

    p.parse(varargin{:});
    p   =   p.Results;

    res.mask    =   p.mask;
    res.Ns      =   size(p.mask, 5);
    res.MB      =   dims(3)/res.Ns;
    res.phs     =   p.phs;
    res.dsize   =   [nnz(p.mask(:,:,:,:,1)), res.Nc, res.Ns];

    if length(res.M) < res.Ns
        res.M   =   repmat(res.M,1,res.Ns);
    end

end


function res = mtimes(a,b)

    m   =   a.mask;
    phs =   a.phs;
    if a.adjoint
    %   Inverse FFT
        res =   zeros([a.Nd a.Nt]);
        b   =   reshape(b, [], a.Nt, a.Nc, a.Ns);
        for t = 1:a.Nt
            d   =   zeros([a.Nd 1 a.Nc]);
        for c = 1:a.Nc
        for s = 1:a.Ns
            tmp =   zeros([a.Nd(1:2) a.Nd(3)/a.Ns]);
            tmp(m(:,:,:,t,s))   =   b(:,t,c,s);
            d(:,:,s:a.Ns:end,1,c) =   mtimes(a.M(s)', tmp, @xfm.ifftfn_ns, 2:3, t);
        end
        end
            res(:,:,:,t)    =   (a.S'*d).*conj(phs(:,:,:,t));
        end
        res =   reshape(res, [], a.Nt);
    else
    %   Forward FFT and sampling
        res =   zeros([nnz(m(:,:,:,1,1)), a.Nt, a.Nc a.Ns]);
        b   =   (a.S*b).*phs;   
        for c = 1:a.Nc
        for t = 1:a.Nt
        for s = 1:a.Ns
            tmp =   mtimes(a.M(s), b(:,:,s:a.Ns:end,t,c), @xfm.fftfn_ns, 2:3, t);
            res(:,t,c,s)  =   tmp(m(:,:,:,t,s));
        end
        end
        end
        res =   reshape(res, a.dsize);
    end

end

function res = mean(a,b)
    res =   zeros([prod(a.Nd)*a.Nt a.Nc]);
    res(a.mask,:)   =   b;
    res =   reshape(res, [a.Nd a.Nt a.Nc]);
    N   =   sum(a.mask,4); 
    res =   bsxfun(@rdivide, sum(res,4), N+eps);
    res =   reshape(a.S'*ifftfn(a, res, 1:3), [], 1);
end

end
end
