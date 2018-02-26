classdef xfm_GA< xfm
%   NUFFT Linear Operator for Golden-Angle specific radial trajectories
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
    k       =   [];
    w       =   [];
    norm    =   1;
    Jd      =   [6,6];
    Kd      =   [];
    shift   =   [];
    st;
    pshift  =   [];
end

methods
function res = xfm_GA(dims, coils, fieldmap_struct, k, varargin)

    %   Base class constructor
    res =   res@xfm(dims, coils, fieldmap_struct);
    
    %   Parse remaining inputs
    p   =   inputParser;

    %   Input validation functions
    toggleValidator =   @(x) x == 0 || x == 1;
    lengthValidator =   @(x) length(x) == 2 || length(x) == 3;

    %   Input options
    p.addParamValue('wi',       [],                     @(x) isscalar(x) || size(x,1) == size(k,1));
    p.addParamValue('Jd',       [6,6,6],                lengthValidator);
    p.addParamValue('Kd',       2*dims(1:3),            lengthValidator);
    p.addParamValue('shift',    floor(dims(1:3)/2),     lengthValidator);
    p.addParamValue('mean',     true,                   @islogical);

    p.parse(varargin{:});
    p   =   p.Results;

    res.Jd      =   p.Jd;
    res.Kd      =   p.Kd;
    res.shift   =   p.shift;

    res.k       =   permute(k,[1,3,2]);
    res.dsize   =   [size(k,1) res.Nt res.Nc];

    disp('Initialising NUFFT(s)');
    nd  =   (res.Nd(3) > 1) + 2;
    for t = 1:res.Nt
        st      =   nufft_init(squeeze(k(:, t, 1:nd)),...
                               res.Nd(1:nd),...
                               p.Jd(1:nd),...
                               p.Kd(1:nd),...
                               p.shift(1:nd),...
                               'table',2^11,'minmax:kb');
        res.pshift(:,t) =   st.phase_shift;
    end
    res.st  =   st;
    if isempty(p.wi)
    disp('Generating Density Compensation Weights');
    %   Use (Pipe 1999) fixed point method
        tmp_st   =   nufft_init(squeeze(k(:,1,1:nd)),...
                                res.Nd(1:nd),...
                                p.Jd(1:nd),...
                                p.Kd(1:nd),...
                                p.shift(1:nd));
        res.w   =   ones(size(k,1),1);
        for ii = 1:20
            tmp =   tmp_st.p*(tmp_st.p'*res.w);
            res.w  =   res.w./real(tmp);
        end
    elseif ~isscalar(p.wi)
        res.w   =   reshape(p.wi, [], 1);
    else
        res.w   =   p.wi;
    end
    res.w       =   sqrt(res.w);
    res.norm    =   sqrt(res.st(1).sn(ceil(end/2),ceil(end/2))^(-2)/prod(res.st(1).Kd));

end


function res = mtimes(a,b,idx)
    if nargin < 3
        idx =   1:a.Nt;
    end
    %   Property access in MATLAB classes is very slow, so any property accessed
    %   more than once is copied into the local scope first
    nt  =   length(idx);
    st  =   a.st;
    k   =   a.k(:,:,idx);
    ps  =   a.pshift(:,idx);

    if a.adjoint
    %   Adjoint NUFFT and coil transform
        res =   zeros([a.Nd, nt a.Nc]);
        b   =   bsxfun(@times, b, a.w);
        for t = 1:nt
            st.om           =   k(:,:,t);
            st.phase_shift  =   ps(:,t);
            res(:,:,:,t,:)  =   nufft_adj(squeeze(b(:,t,:)), st);
        end
        res =   reshape(a.norm*(a.S'*res), [], nt);
    else
    %   Forward NUFFT and coil transform
        res =   zeros([a.dsize(1) nt a.dsize(3)]);
        tmp =   a.norm*(a.S*b);
        for t = 1:nt
            st.om           =   k(:,:,t);
            st.phase_shift  =   ps(:,t);
            res(:,t,:)  =   nufft(squeeze(tmp(:,:,:,t,:)), st);
        end
        res =   bsxfun(@times, res, a.w);
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