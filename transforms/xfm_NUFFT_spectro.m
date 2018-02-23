classdef xfm_NUFFT_spectro < xfm
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
function res = xfm_NUFFT_spectro(dims, coils, fieldmap_struct, k, kf, varargin)

    %   Base class constructor
    res =   res@xfm(dims, coils, fieldmap_struct);
    
    %   Parse remaining inputs
    p   =   inputParser;

    %   Input validation functions
    lengthValidator =   @(x) length(x) == 2 || length(x) == 3;

    %   Input options
    p.addParamValue('wi',       [],                     @(x) size(x,2) == dims(4)||isscalar(x));
    p.addParamValue('Jd',       [6,6,6,6],              lengthValidator);
    p.addParamValue('Kd',       2*dims(1:4),            lengthValidator);
    p.addParamValue('shift',    floor(dims(1:4)/2),     lengthValidator);

    p.parse(varargin{:});
    p   =   p.Results;

    res.Jd      =   p.Jd;
    res.Kd      =   p.Kd;
    res.shift   =   p.shift;

    k       =   repmat(reshape(k,[],1,3),1,length(kf),1);
    k(:,:,4)=   repmat(reshape(kf,1,[]), size(k,1),1);
    res.k   =   reshape(k,[],4);

    res.dsize   =   [size(res.k,1) res.Nc];

    disp('Initialising NUFFT(s)');
    res.st   =   nufft_init(res.k,...
                            dims,...
                            p.Jd,...
                            p.Kd,...
                            p.shift);
    if isempty(p.wi)
    disp('Generating Density Compensation Weights');
    %   Use (Pipe 1999) fixed point method
        w   =   ones(size(k,1),1);
        for ii = 1:20
            w   =   w./real(res.st.p*(res.st.p'*w));
        end
        res.w   =   w;
    else
        res.w   =   p.wi;
    end
    res.w       =   sqrt(res.w);
    res.norm    =   sqrt(res.st.sn(ceil(end/2),ceil(end/2),ceil(end/2),ceil(end/2))^(-2)/prod(res.st.Kd));


end

function res = mtimes(a,b)
    %   Property access in MATLAB classes is very slow, so any property accessed
    %   more than once is copied into the local scope first

    st  =   a.st;

    if a.adjoint
    %   Adjoint NUFFT and coil transform
        res =   zeros([a.Nd a.Nt a.Nc]);
        b   =   bsxfun(@times, b, a.w);
        for c = 1:a.Nc
            res(:,:,:,:,c)  =   nufft_adj(b(:,c), st);
        end
        res =   a.norm*(a.S'*res);
    else
    %   Forward NUFFT and coil transform
        res =   zeros(a.dsize);
        tmp =   a.norm*(a.S*b);
        for c = 1:a.Nc
            res(:,c)  =   nufft(tmp(:,:,:,:,c), st);
        end
        res =   bsxfun(@times, res, a.w);
    end

end

end
end
