classdef xfm_MB < xfm
%   MB-FFT Linear Operator
%   Forward transforms images to multi-coil, multi-band Cartesian k-space
%   along with the adjoint (multi-coil k-space to combined image)
%
%   Mark Chiew
%   mchiew@fmrib.ox.ac.uk
%   Dec 2016
%
%   Usage:  
%       xfm     =   xfm_MB(dims, coils, field, ['mask', mask, 'phs', phs]);
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

properties (SetAccess = protected, GetAccess = public)
    mask    =   [];
    phs     =   [];
    Ns      =   1;
    MB      =   1;
    w       =   [];
end

methods
function res = xfm_MB(dims, coils, fieldmap_struct, varargin)

    %   Base class constructor
    res =   res@xfm(dims, coils, fieldmap_struct);
    
    %   Parse remaining inputs
    p   =   inputParser;

    %   Input options
    p.addParameter('mask',  true(dims));
    p.addParameter('phs',   ones(1,1,1,res.Nt));
    p.addParameter('w',     ones(res.Nc,1));

    p.parse(varargin{:});
    p   =   p.Results;

    res.mask    =   p.mask;
    res.Ns      =   size(p.mask, 5);
    res.MB      =   dims(3)/res.Ns;
    res.phs     =   p.phs;
    res.dsize   =   [nnz(p.mask(:,:,:,1,1)), res.Nt, res.Nc, res.Ns];
    res.w       =   p.w;

    if length(res.M) < res.Ns
        res.M   =   repmat(res.M,1,res.Ns);
    end

end


function res = mtimes(a,b)

    m   =   a.mask;
    phs =   a.phs;
    if a.adjoint
    %   Inverse FFT
        res =   zeros([a.Nd a.Nt]);
        b   =   reshape(b, [], a.Nt, a.Nc, a.Ns);
        d   =   zeros([a.Nd 1 a.Nc]);
        for t = 1:a.Nt
        for c = 1:a.Nc
        for s = 1:a.Ns
            tmp =   zeros([a.Nd(1:2) a.Nd(3)/a.Ns]);
            tmp(m(:,:,:,t,s))   =   b(:,t,c,s);
            %d(:,:,s:a.Ns:end,1,c) =   mtimes(a.M(s)', tmp, @xfm.ifftfn_ns, 2:3, t);
            d(:,:,s:a.Ns:end,1,c) =   a.ifftfn_ns(tmp, 2:3);
        end
        end
            res(:,:,:,t)    =   (a.S'*d).*conj(phs(:,:,:,t));
        end
        res =   reshape(res, [], a.Nt);
    else
    %   Forward FFT and sampling
        res =   zeros([nnz(m(:,:,:,1,1)), a.Nt, a.Nc a.Ns]);
        b   =   (a.S*b).*phs;   
        for c = 1:a.Nc
        for t = 1:a.Nt
        for s = 1:a.Ns
            %tmp =   mtimes(a.M(s), b(:,:,s:a.Ns:end,t,c), @xfm.fftfn_ns, 2:3, t);
            tmp =   a.fftfn_ns(b(:,:,s:a.Ns:end,t,c), 2:3);
            res(:,t,c,s)  =   tmp(m(:,:,:,t,s));
        end
        end
        end
        %res =   reshape(res, a.dsize);
    end

end

function b = mtimes2(a,b,nt,nx)
    if nargin < 3
        nt = 1:a.Nt;
    end
    if nargin < 3
        nx = 1:a.Nd(1);
    end

    m   =   a.mask(nx,:,:,:);
    dim =   size(b);
    %   Forward FFT and sampling
        b   =   reshape(b, [], length(nt));
        for t = nt 
            bb  =   mtimes(a.S,b(:,t),1:a.Nc,nx);
        for c = 1:a.Nc
        for s = 1:a.Ns
            %bb(:,:,s:a.Ns:end,1,c)  =   a.ifftfn_ns(a.fftfn_ns(bb(:,:,s:a.Ns:end,1,c),2:3).*m(:,:,:,t,s),2:3);
            bb(:,:,s:a.Ns:end,1,c)  =   ifft(ifft(fft(fft(bb(:,:,s:a.Ns:end,1,c),[],2),[],3).*m(:,:,:,t,s),[],2),[],3);
        end
        end
            b(:,t)  =   reshape(mtimes(a.S',bb,1:a.Nc,nx),[],1); 
        end
    b   =   reshape(b, dim); 
end


function res = mean(a,b)
    res =   zeros([prod(a.Nd)*a.Nt a.Nc]);
    res(a.mask,:)   =   b;
    res =   reshape(res, [a.Nd a.Nt a.Nc]);
    N   =   sum(a.mask,4); 
    res =   bsxfun(@rdivide, sum(res,4), N+eps);
    res =   reshape(a.S'*ifftfn(a, res, 1:3), [], 1);
end

function g = gfactor(a,t)
    if nargin < 2
        t = 1;
    end
    
    m1      =   a.mask(:,:,:,t);
    sens    =   reshape(a.S.coils,[a.Nd(1:3) a.Nc]).*a.phs(:,:,:,t);
    g       =   zeros(a.Nd(1:3));

    for i = 1:a.Nd(1)
        idx =   find(sens(i,:,:,1));
        [uu vv] = ind2sub(a.Nd(2:3),find(m1(i,:,:)));
        [ii jj] = ind2sub(a.Nd(2:3),idx);
        F   =   exp(-1j*2*pi*((ii(:)-1)'.*(uu(:)-1)/a.Nd(2)+(jj(:)-1)'.*(vv(:)-1)/a.Nd(3)))/sqrt(prod(a.Nd(2:3)));
        s   =   reshape(sens(i,:,:,:),[],a.Nc);
        s   =   s(idx,:).';

        sd  =   zeros(length(idx));
        for c = 1:a.Nc
            sd  = sd + s(c,:)'*s(c,:);
        end
        tmp = (F'*F).*sd;

        g(i,idx)    =   real(sqrt(diag(inv(tmp)).*diag(tmp)));
    end
end

function b = inv(a,b)
    Nt      = a.Nt;

    sens    =   reshape(a.S.coils,[a.Nd(1:3) a.Nc]);
    b       =   reshape(a'*b, a.Nd(1), [], Nt);
    m       =   a.mask(:,:,:,:,1);

    for i = 1:a.Nd(1)
        idx =   reshape(find(sens(i,:,:,1)),[],1);
        if ~isempty(idx)
            s   =   reshape(sens(i,:,:,:),[],a.Nc);
            s   =   s(idx,:).';
            sd  =   zeros(length(idx));
            for c = 1:a.Nc
                sd  = sd + s(c,:)'*s(c,:);
            end
            [ii jj] = ind2sub(a.Nd(2:3),idx);
            for t = 1:Nt
                [uu vv] =   ind2sub(a.Nd(2:3),reshape(find(m(i,:,:,t)),[],1));
                F       =   exp(-1j*2*pi*((ii-1)'.*(uu-1)/a.Nd(2)+(jj-1)'.*(vv-1)/a.Nd(3)))/sqrt(prod(a.Nd(2:3)));

                b(i, idx, t)=   inv((F'*F).*sd)*shiftdim(b(i,idx,t),1);
            end
        end
    end
    b   =   reshape(b,[a.Nd(1:3) Nt]);
end


function [d, g] = bias_var(a,L,shift)

    nt  =   lcm(shift,a.Nd(3))/shift;
    nyz =   prod(a.Nd(2:3));
    d   =   zeros(nt*nyz, a.Nd(1));
    g   =   zeros(nt*nyz, a.Nd(1));        
    r2  =   reshape(L*[6 -4 1 zeros(1,nt-5) 1 -4],1,1,1,nt);
    
    phs =   repmat(exp(1j*reshape(2*pi*(0:a.Nd(3)-1)/a.Nd(3),1,1,a.Nd(3))),nt,a.Nd(3),1);
    for i = 1:nt
        for j = 1:a.Nd(3)
            phs(i,j,:)  =   circshift(phs(i,j,:).^((i-1)*shift),j-1,3);
        end
    end

    
    for x = 1:a.Nd(1)
        
        ii  =   [];
        jj  =   [];
        dd  =   [];
        
        uu  =   [];
        vv  =   [];
        bb  =   [];
        
        for k = 1:a.Nd(3)
        for j = 1:a.Nd(2)
            tmp         = zeros(1,a.Nd(2),a.Nd(3));
            tmp(1,j,k)  = 1;
            tmp2        = reshape(a.mtimes2(tmp,1,x).*phs(:,k,:),nt,[]);
            tmp3        = tmp.*r2;
            if nnz(tmp2)
                for t = 1:nt
                    [idx,jdx] = find(abs(tmp2(t,:)).'>eps);
                    ii = [ii;idx+(t-1)*nyz];                    
                    jj = [jj;jdx*((k-1)*a.Nd(2)+j+(t-1)*nyz)];
                    dd = [dd;tmp2(t,idx).'];
                end
            end
            if nnz(tmp3)
                for t = 1:nt
                    [idx,jdx,bdx] = find(reshape(circshift(tmp3,t-1,4),[],1));
                    uu = [uu;idx];
                    vv = [vv;jdx*((k-1)*a.Nd(2)+j+(t-1)*nyz)];
                    bb = [bb;bdx];
                end
            end
        end
        end
        A   =   sparse(ii,jj,dd,nyz*nt,nyz*nt);
        B   =   sparse([ii;uu],[jj;vv],[dd;bb],nyz*nt,nyz*nt);    
        idx         =   find(diag(A));
        As          =   A(idx,idx);
        Bs          =   inv(B(idx,idx));        
        C           =   Bs*As;    
        d(idx,x)    =   diag(C);
        C           =   C*Bs;
        g(idx,x)    =   sqrt(diag(C).*diag(As));
    end
    
    d   =   permute(reshape(d,a.Nd(2),a.Nd(3),nt,a.Nd(1)),[4,1,2,3]);
    g   =   permute(reshape(g,a.Nd(2),a.Nd(3),nt,a.Nd(1)),[4,1,2,3]);
    d   =   real(d(:,:,:,1));
    g   =   real(g(:,:,:,1));

end


end
end
