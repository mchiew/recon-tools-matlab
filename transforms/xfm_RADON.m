classdef xfm_RADON < xfm
%   Radon Transform Operator
%   Forward transforms images to multi-coil, radial k-space
%   along with the adjoint (multi-coil k-space to combined image)
%
%   Mark Chiew
%   mchiew@fmrib.ox.ac.uk
%   July 2015
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
    theta   =   [];
end

methods
function res = xfm_RADON(dims, coils, fieldmap_struct, theta)

    %   Base class constructor
    res =   res@xfm(dims, coils, fieldmap_struct);

    res.theta   =   theta;
    res.dsize   =   [numel(radon(zeros(dims(1:2)),0))*size(theta,1) size(theta,2) res.Nc];
end


function res = mtimes(a,b,idx)
    if nargin < 3
        idx =   1:a.Nt;
    end
    %   Property access in MATLAB classes is very slow, so any property accessed
    %   more than once is copied into the local scope first
    nt      =   length(idx);
    theta   =   a.theta;

    if a.adjoint
    %   Adjoint NUFFT and coil transform
        res =   zeros([a.Nd, nt a.Nc]);
        for t = 1:nt
        for c = 1:a.Nc
            res(:,:,1,t,c)  =   iradon(reshape(real(b(:,t,c)),[],size(theta,1)), theta(:,t), a.Nd(1), 'linear', 'Ram-Lak') + 1j*iradon(reshape(imag(b(:,t,c)),[],size(theta,1)), theta(:,t), a.Nd(1), 'linear', 'Ram-Lak');
        end
        end
        res =   reshape(a.S'*res, [], nt);
    else
    %   Forward NUFFT and coil transform
        tmp =   (a.S*b);
        res =   zeros([size(radon(tmp(:,:,1,1,1),0),1)*size(theta,1), size(theta,2) a.Nc]);
        for t = 1:nt
        for c = 1:a.Nc
            res(:,t,c)  =   reshape(radon(squeeze(tmp(:,:,1,t,c)), theta(:,t)),[],1);
        end
        end
    end

end

end
end

