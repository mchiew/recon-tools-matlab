classdef xfm_TURBINE < xfm
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
    PAT     =   [];
    k       =   [];
    Jd      =   [6,6];
    Kd      =   [];
    shift   =   [];
    PSF     =   []; % Eigenvalues of circulant embedding 
    st      =   [];
end

methods
function res = xfm_TURBINE(dims, coils, fieldmap_struct, k, varargin)

    %   Base class constructor
    res =   res@xfm(dims, coils, fieldmap_struct);
    
    %   Parse remaining inputs
    p   =   inputParser;

    %   Input validation functions
    lengthValidator =   @(x) length(x) == 2 || length(x) == 3;

    %   Input options
    p.addParamValue('Jd',       [6,6],                  lengthValidator);
    p.addParamValue('Kd',       2*dims(1:2),            lengthValidator);
    p.addParamValue('shift',    floor(dims(1:2)/2),     lengthValidator);
    p.addParamValue('PSF',      []);
    p.addParamValue('st',       []);
    p.addParamValue('PAT',      1,                      @isscalar);

    p.parse(varargin{:});
    p   =   p.Results;

    res.Jd      =   p.Jd;
    res.Kd      =   p.Kd;
    res.shift   =   p.shift;

    res.PAT     =   p.PAT;

    res.k       =   k;
    res.dsize   =   [size(k,1)*size(k,2)/res.PAT dims(3) res.Nt res.Nc];

    res.PSF     =   p.PSF;
    res.st      =   p.st;


    if isempty(res.st)
        disp('Initialising NUFFT(s)');
        for p = 1:res.PAT
            for t = res.Nt:-1:1
                st(t)   =   nufft_init(reshape(k(:,p:res.PAT:end, t, 1:2),[],2),...
                                       res.Nd(1:2),...
                                       res.Jd,...
                                       res.Kd,...
                                       res.shift);
            end
        res.st{p}   =   st;
        end
    end

    if isempty(res.PSF)
        res.PSF =   res.calcToeplitzEmbedding();
    end

end

function TT = calcToeplitzEmbedding(a)
%   See xfm_NUFFT.m for details
    disp('Computing Toeplitz Embedding')
    Nd  =   a.Nd;
    Nt  =   a.Nt;

    for p = 1:a.PAT
        st  =   a.st{p};

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
            x1(:,:,t)   =   reshape(nufft_adj(nufft(tmp, st(t)), st(t)), Nd(1), []);
        end

        %   Second column
        tmp =   zeros(Nd,'single');
        tmp(end,1,1)    =   1;
        for t = 1:Nt
            x2(:,:,t)   =   reshape(nufft_adj(nufft(tmp, st(t)), st(t)), Nd(1), []);
            x2(end,:,t) =   0;
        end

        %   Third column
        tmp =   zeros(Nd,'single');
        tmp(1,end,1)    =   1;
        for t = 1:Nt
            x3(:,:,t)   =   reshape(nufft_adj(nufft(tmp, st(t)), st(t)), Nd(1), []);
        end

        %   Fourth column
        tmp =   zeros(Nd,'single');
        tmp(end,end,1)  =   1;
        for t = 1:Nt
            x4(:,:,t)   =   reshape(nufft_adj(nufft(tmp, st(t)), st(t)), Nd(1), []);
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

        TT{p}   =   prod(sqrt(2*Nd))*a.fftfn_ns(reshape(T,[2*Nd Nt]), 1:3);
    end

end

function res = mtimes(a,b)
    %   Property access in MATLAB classes is very slow, so any property accessed
    %   more than once is copied into the local scope first
    nt  =   a.Nt;
    nz  =   a.Nd(3);
    st  =   a.st;
    PAT =   a.PAT;
    M   =   a.M;

    if a.adjoint
    %   Adjoint NUFFT and coil transform
        res =   zeros([a.Nd, nt a.Nc]);
        for t = 1:nt
            for z = 1:nz
                p   =   mod(z-1,PAT)+1;
                res(:,:,z,t,:)  =   nufft_adj(squeeze(b(:,z,t,:)), st{p}(t));
            end
            res(:,:,:,t,:)  =   mtimes(M', res(:,:,:,t,:), @xfm.ifftfn, 3);
        end
        res =   reshape(a.S'*res, [], nt);
    else
    %   Forward NUFFT and coil transform
        res =   zeros(a.dsize);
        tmp =   a.S*b;
        for t = 1:nt
            tmp(:,:,:,t,:)  =   mtimes(M, tmp(:,:,:,t,:), @xfm.fftfn, 3);
            for z = 1:nz
                p   =   mod(z-1,PAT)+1;
                res(:,z,t,:)  =   nufft(squeeze(tmp(:,:,z,t,:)), st{p}(t));
            end
        end
    end

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
    PAT =   a.PAT;
    M   =   a.M;
    dim =   size(b);
    b   =   reshape(b,[Nd Nt]);
    
    tmp =   zeros(2*Nd(1),2*Nd(2),1,1,Nc);
    tmp2=   zeros(2*Nd(1),2*Nd(2),1,1,Nc);
    for t = 1:Nt
        tmpz    =   mtimes(M, S*b(:,:,:,t), @xfm.fftfn, 3);
        for z = 1:Nd(3)
            p   =   mod(z-1,PAT)+1;
            tmp(1:Nd(1),1:Nd(2),1,1,:)  =  tmpz(:,:,z,1,:); 
            tmp2(:,:,1,1,:) =   ifft2(PSF{p}(:,:,1,t).*fft2(tmp)); 
            tmpz(:,:,z,1,:) =   tmp2(1:Nd(1),1:Nd(2),1,1,:);
        end
        b(:,:,:,t)  =   S'*mtimes(M', tmpz, @xfm.ifftfn, 3);
    end

    %   Return b in the same shape it came in
    b   =   reshape(b, dim);    
end

end
end

