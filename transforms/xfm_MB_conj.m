classdef xfm_MB_conj < xfm_MB
%   MB-FFT Linear Operator
%   Forward transforms images to multi-coil, multi-band Cartesian k-space
%   along with the adjoint (multi-coil k-space to combined image)
%
%   Mark Chiew
%   mchiew@fmrib.ox.ac.uk
%   Dec 2016
%
%   Usage:  
%       xfm     =   xfm_MB(dims, coils, field, ['mask', mask, 'rf_phs', rf_phs]);
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

properties (SetAccess = private, GetAccess = public)
   S2   =   []; 
   mask2=   [];
end

methods
function res = xfm_MB_conj(dims, coils, fieldmap_struct, varargin)

    %   Base class constructor
    res =   res@xfm_MB(dims, [], fieldmap_struct, varargin{:});

    %   Conjugate coils
    res.S2  =   sensEncodingMatrix(cat(4,coils,conj(coils)));
    res.Nc  =   res.S2.Nc;
    res.dsize(2)    =   res.Nc;

    %   Flip sampling mask
    res.mask2   =   cat(6, res.mask, circshift(flip(circshift(flip(res.mask,2),1,2),3),1,3));

    %
    if isscalar(res.phs)
        res.phs =   repmat(res.phs,1,1,1,res.Nt);
    end

end


function res = mtimes(a,b)

    m   =   a.mask2;
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
            tmp(m(:,:,:,t,s,(c>a.Nc/2)+1))   =   b(:,t,c,s);
            d(:,:,s:a.Ns:end,1,c) =   mtimes(a.M(s)', tmp, @xfm.ifftfn_ns, 2:3, t);
        end
        end
            d(:,:,:,:,1:a.Nc/2) =   d(:,:,:,:,1:a.Nc/2).*conj(phs(:,:,:,t));
            d(:,:,:,:,a.Nc/2+1:a.Nc)    =   d(:,:,:,:,a.Nc/2+1:a.Nc).*phs(:,:,:,t);
            res(:,:,:,t)    =   a.S2'*d;
        end
        res =   reshape(res, [], a.Nt);
    else
    %   Forward FFT and sampling
        res =   zeros([nnz(m(:,:,:,1,1,1)), a.Nt, a.Nc a.Ns]);
        b   =   reshape(b, [], a.Nt);
        for t = 1:a.Nt
            bb  =   a.S2*b(:,t);
            bb(:,:,:,1,1:a.Nc/2) =   bb(:,:,:,1,1:a.Nc/2).*phs(:,:,:,t);
            bb(:,:,:,1,a.Nc/2+1:a.Nc) =   bb(:,:,:,1,a.Nc/2+1:a.Nc).*conj(phs(:,:,:,t));
        for c = 1:a.Nc
        for s = 1:a.Ns
            tmp =   mtimes(a.M(s), bb(:,:,s:a.Ns:end,1,c), @xfm.fftfn_ns, 2:3, t);
            res(:,t,c,s)  =   tmp(m(:,:,:,t,s,(c>a.Nc/2)+1));
        end
        end
        end
        res =   reshape(res, a.dsize);
    end

end

function res = flip(a,b)
    b   =   reshape(b, [], a.Nt, a.Nc/2, a.Ns);
    res =   zeros(size(b));
    m   =   a.mask(:,:,:,:,:);
    m2  =   a.mask2(:,:,:,:,:,2);
    tmp =   zeros([numel(m(:,:,:,1,1)), a.Nc/2, a.Ns]);
    for t = 1:a.Nt
    for s = 1:a.Ns
        tmp(m(:,:,:,t,s),:,s) =   b(:,t,:,s);
    end
        tmp =   reshape(tmp,[size(m(:,:,:,1,1)), a.Nc/2, a.Ns]);
        tmp =   conj(circshift(flip(circshift(flip(tmp,2),1,2),3),1,3));
        tmp =   reshape(tmp, [], a.Nc/2, a.Ns);
    for s = 1:a.Ns
        res(:,t,:,s)  =   tmp(m2(:,:,:,t,s),:,s);
    end
    end
    res =   reshape(cat(3, b, res), [], a.Nc, a.Ns);
end

end
end
