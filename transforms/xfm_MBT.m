classdef xfm_MBT < xfm_MB
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
    win  =   [];
    dt   =   [];
end

methods
function res = xfm_MBT(dims, coils, fieldmap_struct, win, varargin)

    %   Base class constructor
    res =   res@xfm_MB(dims, coils, fieldmap_struct, varargin{:});

    %   Sliding window properties
    res.win =   win/norm(win);
    res.dt  =   (length(win)-1)/2;

    res.dsize(5)    =   length(win);
end


function res = mtimes(a,b)

    m   =   a.mask;
    phs =   a.phs;
    dt  =   a.dt;
    win =   a.win;
    Nd  =   a.Nd;
    Nt  =   a.Nt;
    Ns  =   a.Ns;
    Nc  =   a.Nc;

    if a.adjoint
    %   Inverse FFT
        res =   zeros([Nd Nt]);
        b   =   reshape(b, [], Nt, Nc, Ns, length(win));
        d   =   zeros([Nd 1 Nc]);
        for t = 1:Nt
        for c = 1:Nc
        for s = 1:Ns
            tmp =   zeros([Nd(1:2) Nd(3)/Ns]);
        for tt = -dt:dt
            tmp(m(:,:,:,mod(t+tt-1,Nt)+1,s))   =    tmp(m(:,:,:,mod(t+tt-1,Nt)+1,s)) + win(tt+dt+1)*b(:,t,c,s,tt+dt+1);
        end
        d(:,:,s:Ns:end,1,c) =   a.ifftfn_ns(tmp, 2:3);
        end
        end
            res(:,:,:,t)    =   (a.S'*d).*conj(phs(:,:,:,t));
        end
        res =   reshape(res, [], Nt);
    else
    %   Forward FFT and sampling
        res =   zeros(a.dsize);
        b   =   (a.S*b).*phs;   
        for c = 1:Nc
        for t = 1:Nt
        for s = 1:Ns
            tmp =   a.fftfn_ns(b(:,:,s:Ns:end,t,c), 2:3);
        for tt = -dt:dt
            res(:,t,c,s,tt+dt+1)  =   win(tt+dt+1)*tmp(m(:,:,:,mod(t+tt-1,Nt)+1,s));
        end
        end
        end
        end
    end

end

function b = mtimes2(a,T,b)
    m   =   a.mask;
    phs =   a.phs;
    Ns  =   a.Ns;
    Nt  =   a.Nt;
    Nc  =   a.Nc;
    dt  =   a.dt;
    S   =   a.S;

    w   =   reshape(a.win,1,1,1,[]).^2;
    b   =   reshape(b, [], Nt);

    for t = 1:Nt
        bb  =   (S*b(:,t)).*phs(:,:,:,t);
        mm  =   sum(m(:,:,:,mod(t+(-dt:dt)-1,Nt)+1,1).*w,4);
    for c = 1:Nc
    for s = 1:Ns
        bb(:,:,s:Ns:end,1,c)  =  a.ifftfn_ns(a.fftfn_ns(bb(:,:,s:Ns:end,1,c),2:3).*mm,2:3);
    end
    end
        b(:,t)  =   reshape(S'*(bb.*conj(phs(:,:,:,t))),[],1); 
    end
     
end

function g = gfactor(a,t)
    if nargin < 2
        t = 1;
    end
    dt  =   a.dt;
    w   =   a.win;
    Nt  =   a.Nt;
    
    m       =   fftshift(fftshift(a.mask(:,:,:,mod(t+(-dt:dt)-1,Nt)+1,1),2),3);
    sens    =   reshape(a.S.coils,[a.Nd(1:3) a.Nc]).*a.phs(:,:,:,t);
    g       =   zeros(a.Nd(1:3));

    for i = 1:a.Nd(1)
        idx =   find(sens(i,:,:,1));
        tmp =   zeros(length(idx));
        s   =   reshape(sens(i,:,:,:),[],a.Nc);
        s   =   s(idx,:).';
        sd  =   zeros(length(idx));
        for c = 1:a.Nc
            sd  = sd + s(c,:)'*s(c,:);
        end
        [ii jj] = ind2sub(a.Nd(2:3),idx);
        for tt = 1:length(w)
            [uu vv] = ind2sub(a.Nd(2:3),find(m(i,:,:,tt)));
            F       =   w(tt)*exp(-1j*2*pi*((ii-1)'.*(uu-1)/a.Nd(2)+(jj-1)'.*(vv-1)/a.Nd(3)))/prod(a.Nd(2:3));
            tmp     =   tmp + F'*F;
        end
        tmp         =   tmp.*sd;
        g(i,idx)    =   real(sqrt(diag(inv(tmp)).*diag(tmp)));
    end
end

function dd = prep(a,d)
    w   =   a.win;
    dd  =   repmat(d,1,1,1,1,length(w)); 
    tt  =   -a.dt:a.dt;
    for i = 1:length(tt)
        dd(:,:,:,:,i)   =   w(i)*circshift(d,-tt(i),2);
    end
end

end
end
