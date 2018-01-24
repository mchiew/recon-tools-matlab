classdef xfm_NUFFT_conj < xfm_NUFFT
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

properties (SetAccess = private, GetAccess = public)
    S2  =   [];
    k2  =   [];
    st2 =   [];
    w2  =   [];
    phs =   [];
end

methods
function res = xfm_NUFFT_conj(dims, coils, fieldmap_struct, k, phs, varargin)

    %   Base class constructor
    res =   res@xfm_NUFFT(dims, coils, fieldmap_struct, k, varargin{:});

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

    %   Phase information
    if isscalar(phs)
        phs =   repmat(phs,1,1,1,res.Nt);
    end
    res.phs =   phs; 

    %   Conjugate coils
    res.S2  =   sensEncodingMatrix(cat(4,coils,conj(coils)));
    res.Nc  =   res.S2.Nc;
    res.dsize(3)    =   res.Nc;

    disp('Initialising conjugate NUFFT(s)');
    nd  =   (res.Nd(3) > 1) + 2;
    for t = res.Nt:-1:1
        st2(t)   =   nufft_init(-1*squeeze(k(:, t, 1:nd)),...
                               res.Nd(1:nd),...
                               p.Jd(1:nd),...
                               p.Kd(1:nd),...
                               p.shift(1:nd));
    end
    res.st2  =   st2;
    if isempty(p.wi)
    disp('Generating Density Compensation Weights');
    %   Use (Pipe 1999) fixed point method
        for t = 1:res.Nt
            res.w2(:,t)  =   ones(size(k,1),1);
            for ii = 1:10
                tmp =   res.st2(t).p*(res.st2(t).p'*res.w2(:,t));
                res.w2(:,t)  =   res.w2(:,t)./real(tmp);
            end
        end
    elseif isscalar(p.wi)
        res.w2  =   repmat(p.wi, 1, res.Nt);
    else
        res.w2  =   reshape(p.wi, [], res.Nt);
    end
    res.w2      =   sqrt(res.w2);

end


function res = mtimes(a,b,idx)
    if nargin < 3
        idx =   1:a.Nt;
    end
    %   Property access in MATLAB classes is very slow, so any property accessed
    %   more than once is copied into the local scope first
    nt  =   length(idx);
    st  =   a.st(idx);
    st2 =   a.st2(idx);
    phs =   a.phs;
    cc  =   reshape(1:a.Nc,[],2);

    if a.adjoint
    %   Adjoint NUFFT and coil transform
        res =   zeros([a.Nd, nt a.Nc]);
        b(:,:,cc(:,1))  =   bsxfun(@times, b(:,:,cc(:,1)), a.w(:,idx));
        b(:,:,cc(:,2))  =   bsxfun(@times, b(:,:,cc(:,2)), a.w2(:,idx));
        for t = 1:nt
            res(:,:,:,t,cc(:,1))  =   nufft_adj(squeeze(b(:,t,cc(:,1))), st(t)).*conj(phs(:,:,:,t));
            res(:,:,:,t,cc(:,2))  =   nufft_adj(squeeze(b(:,t,cc(:,2))), st(t)).*phs(:,:,:,t);
        end
        res =   reshape(a.norm*(a.S2'*res), [], nt);
    else
    %   Forward NUFFT and coil transform
        res =   zeros([a.dsize(1) nt a.dsize(3)]);
        tmp =   a.norm*(a.S2*b);
        for t = 1:nt
            tmp(:,:,:,t,cc(:,1))    =   tmp(:,:,:,t,cc(:,1)).*phs(:,:,:,t);
            res(:,t,cc(:,1))        =   nufft(squeeze(tmp(:,:,:,t,cc(:,1))), st(t));
            tmp(:,:,:,t,cc(:,2))    =   tmp(:,:,:,t,cc(:,2)).*conj(phs(:,:,:,t));
            res(:,t,cc(:,2))        =   nufft(squeeze(tmp(:,:,:,t,cc(:,2))), st2(t));
        end
        res(:,:,cc(:,1))=   bsxfun(@times, res(:,:,cc(:,1)), a.w(:,idx));
        res(:,:,cc(:,2))=   bsxfun(@times, res(:,:,cc(:,2)), a.w2(:,idx));
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
