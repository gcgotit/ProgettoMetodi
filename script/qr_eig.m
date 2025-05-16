function [eigvals, V] = qr_eig(A, tol, maxit)
% QR_EIG  Autovalori e autovettori di matrice simmetrica A
%         tramite riduzione a tridiagonale + QR‐iteration deflazionata.
 [H, Qh] = houshess(A);
  n      = size(H,1);
  V      = Qh;
  eigvals = zeros(n,1);

  for k = n:-1:2
    fprintf('         [qr_eig] deflazione k=%d ... ', k);
    t_k = tic;
    iter = 0; 
    I_k  = eye(k);

    while abs(H(k,k-1)) > tol*(abs(H(k,k))+abs(H(k-1,k-1)))
      iter = iter + 1;
      if iter > maxit
        warning('qr_eig:NoConv','k=%d non conv in %d iter', k, maxit);
        break;
      end

      % Wilkinson double‐shift
      d  = (H(k-1,k-1)-H(k,k))/2;
      mu = H(k,k) - sign(d)*(H(k,k-1)^2)/(abs(d)+sqrt(d^2+H(k,k-1)^2));

      % Applico QR ottimizzato sulla sub-matrice tridiagonale H(1:k,1:k)
      [Qk, Rk] = qr_tridiag(H(1:k,1:k) - mu*I_k);

      % Ricostruisco H e accumulo Q
      H(1:k,1:k) = Rk*Qk + mu*I_k;
      V(:,1:k)    = V(:,1:k) * Qk;
    end

    eigvals(k)   = H(k,k);
    H(k,k-1)     = 0;
    H(k-1,k)     = 0;
    fprintf('iter=%d time=%.2f s\n', iter, toc(t_k));
  end

  eigvals(1) = H(1,1);
  fprintf('      [qr_eig] tutti autovalori estratti.\n');
end