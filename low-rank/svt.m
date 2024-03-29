function est = svt(E, d, L, varargin)
%function [est, err, fErr] = svt(E, data, L, step, maxIter, tol)
%
%   Last Modified by Mark Chiew
%   Oct 2016
%
%   Created by Mark Chiew (mark.chiew@ndcn.ox.ac.uk)
%   Jun 2014
%
%   Outputs:    est         is the final matrix estimate 
%
%   Inputs:     E           the nufft/fft-based measurement model transform
%                           generated by the transform class
%               data        sampled data points, the size/shape of this should
%                           match the output of E*est
%               L           nuclear norm (low rank) regularization parameter
%               step        scales the size of the changes to the iteratively
%                           updated estimate matrix
%               maxIter     the maximum number of iterations to be performed
%               tol         the error tolerance, if error is below errTol, 
%                           iteration stops (error in sampled entries only)

%===========================================================
%   Script Setup and Parse Inputs
%===========================================================
p       =   inputParser;
msize   =   E.msize;

p.addRequired('E');
p.addRequired('d');
p.addRequired('L');

p.addParamValue('step',       1,      @isscalar);
p.addParamValue('maxIter',    100,    @isscalar);
p.addParamValue('tol',        1E-4,      @isscalar);
p.addParamValue('verbose',    false,  @islogical);

p.parse(E, d, L, varargin{:});
step        =   p.Results.step;
maxIter     =   p.Results.maxIter;
tol         =   p.Results.tol;
verbose     =   p.Results.verbose;

clear p;

%===========================================================
%   Initialization
%===========================================================
%   Initialize some iteration parameters
est     =   zeros(msize);
iter    =   1;
est0    =   est;
y       =   est;
update  =   inf;
t1      =   1;
d       =   E'*d;

%===========================================================
%   Main Iteration Loop
%===========================================================
if verbose
    fprintf(1, '%-5s %-16s %-16s %-16s %-16s\n', 'Iter','Update','DataCon','Nuc.Norm.','Cost');
end
while iter <= maxIter && update > tol

    %   Data Consistency and Roughness Penalisation  
    est = y + step*(d - E.mtimes2(y) - 1*R1(E,y));
    %tmp     =   E*y;
    %tmp(~E.mask) = 0;
    %est = y + step*(d - E'*tmp);

    %   Singular Value Thresholding
    [U, Sig, V] =   lsvd(est);
    Sig2        =   diag(max(diag(Sig) - step*L, 0));
    est         =   U*Sig2*V';

    %   Update the error and change metrics
    %err1(iter)  =   0.5*norm(samp(:)-k_est(:)).^2;
    %err3(iter)  =   L*sum(abs(diag(Sig2)));

    %   Update the error and change metrics
    update      =   norm(est(:)-est0(:))/norm(est(:));

    %   Compute cost
    %cost        =   err1(iter) + err3(iter);

    %   Display iteration summary data
    if verbose
        fprintf(1, '%-5d %-16G %-16s %-16s %-16s\n', iter, update, '-', '-', '-');
    end
    
    %   Acceleration
    t2  =   (1+sqrt(1+4*t1^2))/2;
    y   =   est + ((t1-1)/t2)*(est - est0);
    t1  =   t2;

    %   Update iteration counter
    est0    =   est;
    iter    =   iter + 1;

end
disp('Finished'); 
end

function x = R1(E,x)    
    x = reshape(x,[E.Nd E.Nt]);
    x = -1*circshift(x,-1,1) + 2*x -1*circshift(x,1,1) + ...
        -1*circshift(x,-1,2) + 2*x -1*circshift(x,1,2) + ...
        -1*circshift(x,-1,3) + 2*x -1*circshift(x,1,3);
    x = reshape(x,E.msize);
end
