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
    res.S2  =   sensEncodingMatrix(cat(4,coils,conj(coils))/sqrt(2));
    res.Nc  =   res.S2.Nc;
    res.dsize(3)    =   res.Nc;

    %   Flip sampling mask
    res.mask2   =   cat(6, res.mask, circshift(flip(circshift(flip(res.mask,2),1,2),3),1,3));

    %
    if isscalar(res.phs)
        res.phs =   repmat(res.phs,1,1,1,res.Nt);
    end

    res.w   =   ones(res.Nc,1);

end


function res = mtimes(a,b)

    m   =   a.mask2;
    phs =   a.phs;
    w   =   sqrt(a.w);

    if a.adjoint
    %   Inverse FFT
        res =   zeros([a.Nd a.Nt]);
        b   =   reshape(b, [], a.Nt, a.Nc, a.Ns);
        for t = 1:a.Nt
            d   =   zeros([a.Nd 1 a.Nc]);
        for c = 1:a.Nc
        for s = 1:a.Ns
            tmp =   zeros([a.Nd(1:2) a.Nd(3)/a.Ns]);
            tmp(m(:,:,:,t,s,(c>a.Nc/2)+1))   =   w(c)*b(:,t,c,s);
            %d(:,:,s:a.Ns:end,1,c) =   mtimes(a.M(s)', tmp, @xfm.ifftfn_ns, 2:3, t);
            d(:,:,s:a.Ns:end,1,c) =   a.ifftfn_ns(tmp, 2:3);
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
            %tmp =   mtimes(a.M(s), bb(:,:,s:a.Ns:end,1,c), @xfm.fftfn_ns, 2:3, t);
            tmp =   a.fftfn_ns(bb(:,:,s:a.Ns:end,1,c), 2:3);
            res(:,t,c,s)  =   w(c)*tmp(m(:,:,:,t,s,(c>a.Nc/2)+1));
        end
        end
        end
        %res =   reshape(res, a.dsize);
    end

end

function b = mtimes_Toeplitz(a,T,b,w)
    m   =   a.mask2;
    phs =   a.phs;
    if nargin < 4
        w   =   ones(a.Nc,1);
    end
    %   Forward FFT and sampling
        b   =   reshape(b, [], a.Nt);
        for t = 1:a.Nt
            bb  =   a.S2*b(:,t);
            bb(:,:,:,1,1:a.Nc/2) =   bb(:,:,:,1,1:a.Nc/2).*phs(:,:,:,t);
            bb(:,:,:,1,a.Nc/2+1:a.Nc) =   bb(:,:,:,1,a.Nc/2+1:a.Nc).*conj(phs(:,:,:,t));
        for c = 1:a.Nc
        for s = 1:a.Ns
            bb(:,:,s:a.Ns:end,1,c)  =   w(c)*a.ifftfn_ns(a.fftfn_ns(bb(:,:,s:a.Ns:end,1,c),2:3).*m(:,:,:,t,s,(c>a.Nc/2)+1),2:3);
        end
        end
            bb(:,:,:,1,1:a.Nc/2) =   bb(:,:,:,1,1:a.Nc/2).*conj(phs(:,:,:,t));
            bb(:,:,:,1,a.Nc/2+1:a.Nc) =   bb(:,:,:,1,a.Nc/2+1:a.Nc).*phs(:,:,:,t);
            b(:,t)  =   reshape(a.S2'*bb,[],1); 
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
        tmp =   reshape(tmp,[size(m,1), size(m,2), size(m,3) a.Nc/2, a.Ns]);
        tmp =   conj(circshift(flip(circshift(flip(tmp,2),1,2),3),1,3));
        tmp =   reshape(tmp, [], a.Nc/2, a.Ns);
    for s = 1:a.Ns
        res(:,t,:,s)  =   tmp(m2(:,:,:,t,s),:,s);
    end
    end
    %res =   reshape(cat(3, b, res), [], a.Nc, a.Ns);
    res =   cat(3, b, res);
end

function g = gfactor(a,t)
    if nargin < 2
        t = 1;
    end

    m1      =   fftshift(fftshift(a.mask2(:,:,:,t,1,1),2),3);
    m2      =   fftshift(fftshift(a.mask2(:,:,:,t,1,2),2),3);
    sens    =   reshape(a.S2.coils,[a.Nd(1:3) a.Nc]);
    c1      =   1:a.Nc/2;
    c2      =   a.Nc/2+1:a.Nc;
    sens(:,:,:,c1) = sens(:,:,:,c1).*a.phs(:,:,:,t);
    sens(:,:,:,c2) = sens(:,:,:,c2).*conj(a.phs(:,:,:,t));
    g       =   zeros(a.Nd(1:3));

    for i = 1:a.Nd(1)
        idx =   find(sens(i,:,:,1));
        tmp =   zeros(length(idx));
        [uu1 vv1] = ind2sub(a.Nd(2:3),find(m1(i,:,:)));
        [uu2 vv2] = ind2sub(a.Nd(2:3),find(m2(i,:,:)));
        [ii jj] = ind2sub(a.Nd(2:3),idx);
        F1  =   exp(-1j*2*pi*((ii-1)'.*(uu1-1)/a.Nd(2)+(jj-1)'.*(vv1-1)/a.Nd(3)))/prod(a.Nd(2:3));
        F2  =   exp(-1j*2*pi*((ii-1)'.*(uu2-1)/a.Nd(2)+(jj-1)'.*(vv2-1)/a.Nd(3)))/prod(a.Nd(2:3));
        s   =   reshape(sens(i,:,:,c1),[],a.Nc/2);
        s   =   s(idx,:).';
        sd  =   zeros(length(idx));
        for c = 1:a.Nc/2
            sd  = sd + s(c,:)'*s(c,:);
        end
        tmp =   (F1'*F1 + F2'*F2).*sd;
        g(i,idx)    =   real(sqrt(diag(inv(tmp)).*diag(tmp)));
    end

    
end

end
end
