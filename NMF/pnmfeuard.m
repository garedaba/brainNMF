function [W wnorm sigma] = pnmfeuard(X, Winit, max_iter, tol)
%
% Automatic Rank Determination PNMF (ARDPNMF) based on Euclidean distance
% input:
%   X          nonnegative data input (m times n)
%   Winit      initial W
%
%   max_iter   maximum number of iterations (defaut 5000)
%   tol        convergence tolerance (default 1e-5)
%
% output:
%   W          the factorizing matrix (m times rinit)
%   wnorm      norm of each column of W
%
%  Zhanxing Zhu and Zhirong Yang, September, 2013
%
if ~exist('max_iter', 'var') || isempty(max_iter)
    max_iter = 5000;
end
if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-5;
end
check_step = 100;

W = Winit;

XX = X * X';
for iter=1:max_iter
    W_old = W;
    if mod(iter,check_step)==0
        fprintf('iter=% 5d ', iter);
    end
    sigma = sum(bsxfun(@times, W, W));
    
    W = W .* ((XX*W)*diag(sigma) ./ ((W*(W'*XX*W) + XX*W*(W'*W))*diag(sigma) + W + eps));
    W = W ./ norm(W);
    
    diffW = norm(W_old-W, 'fro') / norm(W_old, 'fro');
    if diffW<tol
        fprintf('converged after %d steps.\n', iter);
        break;
    end
    
    if mod(iter,check_step)==0
        fprintf('diff=%.10f, ', diffW);
        fprintf('obj=%.10f', norm(X-W*(W'*X), 'fro'));
        fprintf('\n');
        fprintf('max sigma: %f    min sigma: %f\n', max(sigma), min(sigma));
    end
end

wnorm = [];
rinit = size(W,2);
for i = 1:rinit
    wnorm = [wnorm norm(W(:,i))];
end
