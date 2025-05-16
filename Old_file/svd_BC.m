function [U,S,V] = svd_BC(A, tol, maxit)
% SVD_BC  SVD di A tramite problema autovalori + QR-shift
  if nargin<2, tol=1e-3; end
  if nargin<3, maxit=200; end

  fprintf('   [svd_BC] costruisco ATA...\n');
  ATA = A' * A;

  fprintf('   [svd_BC] chiamo qr_eig su %dx%d (tol=%.1e,maxit=%d)...\n', ...
          size(ATA,1), size(ATA,2), tol, maxit);
  t0 = tic;
  [lambda, V] = qr_eig(ATA, tol, maxit);
  t1 = toc(t0);
  fprintf('   [svd_BC] qr_eig completata in %.2f s\n', t1);

  idx   = lambda > tol;
  sigma = sqrt(lambda(idx));
  V     = V(:, idx);

  fprintf('   [svd_BC] costruisco S e U...\n');
  [m,n] = size(A);
  S = zeros(m,n);
  S(1:sum(idx),1:sum(idx)) = diag(sigma);
  U = A * V * diag(1./sigma);
end
