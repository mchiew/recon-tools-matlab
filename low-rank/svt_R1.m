function est = svt_R1(xfm, d, varargin)
%function [est, err, fErr] = iht_ms(xfm, samp, rank, step, shrink, maxIter,...
%                                   errTol, minUpdate, truth, est)
%
%   Last Modified by Mark Chiew
%   Oct 2016
%   -   Switched the order of hard thresholding/shrink, and data consistency
%       Ending on the data consistency step seems to work a bit better
%   -   Added temporal constraints
%
%   Modified by Mark Chiew
%   Jun 2014
%
%   Outputs:    est         is the final matrix estimate 
%               err         is the error of the estimated values at the sampled
%                           k-space locations only
%               fErr        is the total error of the entire estimated matrix
%                           (only computed if ground truth is provided)
%
%   Inpute:     xfm         the nufft/fft-based measurement model transform
%                           generated by the transform class
%               samp        sampled data points, the size/shape of this should
%                           match the output of xfm*est
%               step        scales the size of the changes to the iteratively
%                           updated estimate matrix
%               shrink      scales the shrinkage parameter (0 < shrink < 1)
%               maxIter     the maximum number of iterations to be performed
%               errTol      the error tolerance, if error is below errTol, 
%                           iteration stops (error in sampled entries only)
%               minUpdate   minimum update tolerance, if the change in the
%                           estimate matrix is below minUpdate, iteration stops
%               truth       the "true" full data, used to calculate the full
%                           error
%               est         initial matrix estimate (optional)

%===========================================================
%   Script Setup and Parse Inputs
%===========================================================
p       =   inputParser;
msize   =   xfm.msize;

p.addRequired('xfm');
p.addRequired('d');

p.addParamValue('step',       1,      @isscalar);
p.addParamValue('maxIter',    100,    @isscalar);
p.addParamValue('tol',        0,      @isscalar);
p.addParamValue('Lr',         0,      @isscalar);
p.addParamValue('Lt',         0,      @isscalar);

p.parse(xfm, d, varargin{:});
step        =   p.Results.step;
Lr          =   p.Results.Lr;
maxIter     =   p.Results.maxIter;
tol         =   p.Results.tol;
Lt          =   p.Results.Lt;

clear p;

%===========================================================
%   Initialization
%===========================================================
%   Initialize some iteration parameters
est =   zeros(msize);
iter    =   1;
est0    =   est;
y       =   est;
update  =   inf;
t1      =   1;

%===========================================================
%   Main Iteration Loop
%===========================================================
fprintf(1, '%-5s %-16s %-16s %-16s %-16s %-16s\n', 'Iter','Update','DataCon','Rough','Nuc.Norm.','Cost');
while iter <= maxIter && update > tol

    %   Data Consistency and Roughness Penalisation  
    est = y + step*(d - xfm.mtimes2(y) - Lt*R1(y));

    %   Singular Value Thresholding
    [U, Sig, V] =   lsvd(est);
    Sig2        =   diag(max(diag(Sig) - step*Lr, 0));
    est         =   U*Sig2*V';

    %   Update the error and change metrics
    %err1(iter)  =   0.5*norm(samp(:)-k_est(:)).^2;
    %err3(iter)  =   L*sum(abs(diag(Sig2)));

    %   Update the error and change metrics
    update      =   norm(est(:)-est0(:))/norm(est(:));

    %   Compute cost
    %cost        =   err1(iter) + err3(iter);

    %   Display iteration summary data
    fprintf(1, '%-5d %-16G %-16s %-16s %-16s %-16s\n', iter, update, '-', '-', '-', '-');

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
   
function x = R1(x)
    x =  -1*circshift(x,-1,2) + 2*x - 1*circshift(x,1,2);
end
