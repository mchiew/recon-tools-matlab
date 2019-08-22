classdef mfi

%   Multi-Frequency Interpolation class for Off-Resonance Correction
%   Based on Man et al., MRM 37:785-792 (1997)
%
%   Last Modified:
%   Mark Chiew
%   July 2017
%
%   Inputs:
%           dw  -   spatial off-resonance (in Hz) map
%           t   -   time (in s) of each k-space sample
%           L   -   number of frequency interpolation bins
%           idx -   which t-index to use for each time-point

properties (SetAccess = private, GetAccess = public)

    dw;
    t;
    rdim;
    kdim;
    L;
    c;
    adjoint=0;
    D;
    M;
    E;
    d;
    Nt;
    idx;
    passthrough=0;    
end

methods
%   Constructor
function res = mfi(dw, t, L, idx)

    if nargin == 0
    res.passthrough =   1;
    else

    res.dw  =   dw;
    res.rdim(1) = size(dw,1);
    res.rdim(2) = size(dw,2);
    res.rdim(3) = size(dw,3);
    
    res.t   =   t;
    res.kdim(1) =   size(t,1);
    res.kdim(2) =   size(t,2);
    res.kdim(3) =   size(t,3);
    tidx    =   size(t,4);
    t       =   reshape(t,[],tidx);
    
    res.idx =   idx;
    res.Nt  =   length(idx);

    res.c   =   zeros([res.rdim tidx 1 L],'single');
    res.L   =   zeros([res.rdim tidx 1 L],'single');
    res.d   =   zeros([res.kdim tidx 1 L],'single');
    res.M   =   zeros([res.kdim tidx 1 L],'single');
    for tt = 1:tidx
        
        %   0.1 Hz and 0.1 ms interpolation resolution
        tmpW    =   linspace(min(dw(:)), max(dw(:)), round(1E1*(max(dw(:))-min(dw(:)))));
        tmpT    =   linspace(min(t(:,tt)), max(t(:,tt)), round(1E4*(max(t(:,tt))-min(t(:,tt)))));

        %   MFI Interpolation
        %   We use the regularised pseudo-inverse to perform the LS coefficient fits
        %   Because combination weights are a smooth function of off-resonance,
        %   we only fit a reduced set of coefficients, and use spline interpolation
        %   to fill in the rest

        LL      =   linspace(min(dw(:)), max(dw(:)), L);
        D       =   pinv(exp(-1j*2*pi*bsxfun(@times, LL, tmpT')));
        c       =   D*exp(-1j*2*pi*bsxfun(@times, tmpW, tmpT'));
        c       =   interp1(tmpW, c.', dw(:), 'spline');
       
        res.c(:,:,:,tt,:,:)   =   reshape(c, [res.rdim 1 1 L]);
        res.L(:,:,:,tt,:,:)   =   exp(-1j*2*pi*bsxfun(@times, reshape(LL,1,1,1,1,1,[]), res.t(:,:,:,tt)));

        %   Reverse MFI Interpolation (i.e. time-interpolation for fwd transform)
        %   We use the regularised pseudo-inverse to perform the LS coefficient fits
        %   Because combination weights are a smooth function of time,
        %   we only fit a reduced set of coefficients, and use spline interpolation
        %   to fill in the rest    
        MM      =   linspace(min(t(:,tt)), max(t(:,tt)), L);
        E       =   pinv(exp(1j*2*pi*dw(:).*MM));
        d       =   E*exp(1j*2*pi*dw(:).*tmpT);
        d       =   interp1(tmpT, d.', t(:,tt), 'spline');
                
        res.d(:,:,:,tt,:,:)   =   reshape(d, [res.kdim 1 1 L]);
        res.M(:,:,:,tt,:,:)   =   exp(1j*2*pi*bsxfun(@times, res.dw, reshape(MM,1,1,1,1,1,[])));
    end
    end
end

%   Overload ctranspose to denote fwd and rev transforms
function res = ctranspose(a)

    a.adjoint   = xor(a.adjoint,1);
    res         = a;

end

%   Interpolated fwd and rev transforms
function res = mtimes(a,b, fftfn, dims, tt)

    if nargin < 5
        tt      =   1:a.Nt;
    end
    nt  =   length(tt);
    if nargin < 4
        dims    =   1:3;
    end
    if a.passthrough
        res =   fftfn(b, dims);
    else
    %   Adj
    if a.adjoint
        if nargin < 3
            fftfn = @xfm.ifftfn;
        end

        
        %{ 
        %   Estimate the value of each voxel as some linear combination of the frequency-bin images
        %   This can be one step, but is broken out into a loop for memory reasons
        %   res =   sum(bsxfun(@times, tmp, a.c), 6);
        res =   zeros([a.rdim nt size(b,5)]);
        c   =   a.c;
        L   =   a.L;
        idx =   a.idx;
        for d = 1:size(b,5)
        for t = 1:nt
            %   Compute demodulated images at the frequency interpolation bins
            %tmp =   fftfn(bsxfun(@times, b(:,:,:,t,d), L(:,:,:,idx(t),:,:)), dims);
            tmp =   fftfn(b(:,:,:,t,d).*L(:,:,:,idx(tt(t)),:,:), dims);

            %   Compute the interpolation using pre-calculated weights
            res(:,:,:,t,d)  =   sum(tmp.*c, 6);
        end
        end
        %}

        res =   sum(fftfn(b.*a.L(:,:,:,a.idx(tt),:,:), dims).*a.c(:,:,:,a.idx(tt),:,:), 6);

    %   Fwd
    else
        if nargin < 3
            fftfn = @xfm.fftfn;
        end

        %{
        %   Estimate the value of each k-point as some linear combination of the time-binned k-spaces
        %   This can be one step, but is broken out into a loop for memory reasons
        %   res =   sum(bsxfun(@times, tmp, a.d(:,:,:,a.idx,:,:)), 6);
        res =   zeros([a.kdim nt size(b,5)]);
        d   =   a.d;
        idx =   a.idx;
        M   =   a.M;
        for c = 1:size(b,5)
        for t = 1:nt
            %   Compute off-resonance k-spaces at the time interpolation bins
            %tmp =   fftfn(bsxfun(@times, b(:,:,:,t,:), M), dims);
            tmp =   fftfn(b(:,:,:,t,c).*M, dims);

            %   Compute the interpolation using pre-calculated weights
            %res(:,:,:,t,c)  =   sum(bsxfun(@times, tmp, d(:,:,:,idx(t),:,:)), 6);
            res(:,:,:,t,c)  =   sum(tmp.*d(:,:,:,idx(tt(t)),:,:), 6);
        end
        end
        %tmp =   fftfn(b.*a.M, dims);
        %res =   sum(tmp.*a.d(:,:,:,a.idx(tt),:,:), 6);
        %}

        res =   sum(fftfn(b.*a.M(:,:,:,a.idx(tt),:,:), dims).*a.d(:,:,:,a.idx(tt),:,:), 6);
    end
    end

end

%   Exact fwd transform
function res = fwd(a,b,fftfn,dims)
    res =   zeros([prod(a.kdim) a.Nt size(b,5)]);
    for tt = 1:a.Nt
    t   =   a.t(:,:,:,tt);
    dw  =   a.dw;
    for ii = 1:numel(t)
        tmp =   reshape(fftfn(bsxfun(@times, b(:,:,:,tt,:), exp(1j*2*pi*dw*t(ii))), dims),[],1,size(b,5));
        res(ii,tt,:) =   tmp(ii,1,:);
    end
    end
    res =   reshape(res,[a.kdim a.Nt size(b,5)]);
end

%   Exact, voxel-by-voxel rev transform (conj. phase reconstruction)
function res = rev(a,b,fftfn,dims)
    res =   zeros([prod(a.rdim) a.Nt size(b,5)]);
    for tt = 1:a.Nt
    t   =   a.t(:,:,:,tt);
    dw  =   a.dw;
    for ii = 1:numel(dw)
        tmp =   reshape(fftfn(bsxfun(@times, b(:,:,:,tt,:), exp(-1j*2*pi*dw(ii)*t)), dims),[],1,size(b,5));
        res(ii,tt,:) =   tmp(ii,1,:);
    end
    end
    res =   reshape(res,[a.rdim a.Nt size(b,5)]);
end
end
end
