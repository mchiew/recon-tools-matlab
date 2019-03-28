classdef xfm_FFT < xfm
%   FFT Linear Operator
%   Forward transforms images to multi-coil, Cartesian k-space
%   along with the adjoint (multi-coil k-space to combined image)
%
%   Mark Chiew
%   mchiew@fmrib.ox.ac.uk
%   July 2015
%
%   Required Inputs:
%       dims    =   [Nx, Ny, Nz, Nt] 4D vector of image dimensions
%
%   Optional Inputs:
%       mask    -   [Nx, Ny, Nz, Nt] logical sampling mask
%       coils   -   [Nx, Ny, Nz, Nc] array of coil sensitivities 
%                   Defaults to single channel i.e. ones(Nx,Ny,Nz)
%       fieldmap_struct - struct(
%                   'field',    [field map in Hz],
%                   't',        [time (in seconds) of every k-sample],
%                   'L',        number of interpolation bins (recommend 20),
%                   'idx',      [vector specifying which "t" to use at each time-point])
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
    mask    =   [];
end

methods
function res = xfm_FFT(dims, coils, fieldmap_struct, varargin)

    %   Base class constructor
    res =   res@xfm(dims, coils, fieldmap_struct);
    
    %   Parse remaining inputs
    p   =   inputParser;

    %   Input options
    p.addParamValue('mask',     true(dims), @(x) isequal(xfm.size(x),dims));

    p.parse(varargin{:});
    p   =   p.Results;

    res.mask    =   p.mask;
    res.dsize   =   [nnz(p.mask), res.Nc];

end


function res = mtimes(a,b)

    if a.adjoint
    %   Inverse FFT
        res =   zeros([prod(a.Nd)*a.Nt a.Nc]);
        res(a.mask,:)   =   b;
        res =   reshape(res, [a.Nd a.Nt a.Nc]);
        res =   reshape(a.S'*mtimes(a.M', res, @xfm.ifftfn), [], a.Nt);
    else
    %   Forward FFT and sampling
        res =   reshape(mtimes(a.M, a.S*b, @xfm.fftfn), [], a.Nc);
        res =   res(a.mask,:);
    end

end

function res = mtimes2(a,b)
    res = mtimes(a',mtimes(a,b));
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
