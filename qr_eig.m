function [eigvals, V] = qr_eig(A, tol, maxit)
% QR_EIG  Autovalori e autovettori di matrice simmetrica A
%         tramite riduzione a tridiagonale + QR‐iteration deflazionata.
  n = size(A,1);
  fprintf('      [qr_eig] riduzione a tridiagonale %dx%d...\n',n,n);
  t0 = tic;
  [H, Qh] = houshess(A);
  fprintf('      [qr_eig] tridiagonale in %.2f s\n', toc(t0));
  V = Qh;
  eigvals = zeros(n,1);

  for k = n:-1:2
    fprintf('         [qr_eig] deflazione k=%d ... ', k);
    t_k = tic;
    iter = 0; I = eye(k);
    while abs(H(k,k-1)) > tol*(abs(H(k,k))+abs(H(k-1,k-1)))
      iter = iter + 1;
      if iter > maxit
        warning('qr_eig:NoConv','k=%d non conv in %d iter',k,maxit);
        break;
      end
      d  = (H(k-1,k-1)-H(k,k))/2;
      mu = H(k,k) - sign(d)*(H(k,k-1)^2)/(abs(d)+sqrt(d^2+H(k,k-1)^2));
      [Qk,Rk] = qr_hessenberg(H(1:k,1:k)-mu*I);
      H(1:k,1:k) = Rk*Qk + mu*I;
      V(:,1:k)    = V(:,1:k)*Qk;
    end
    eigvals(k) = H(k,k);
    H(k,k-1)   = 0; H(k-1,k) = 0;
    fprintf('iter=%d, time=%.2f s\n', iter, toc(t_k));
  end
  eigvals(1) = H(1,1);
  fprintf('      [qr_eig] tutti autovalori estratti.\n');
end


function [H, Q] = houshess(A)
% HOUSHESS  Riduce A simmetrica a tridiagonale
  n = size(A,1);
  H = A; Q = eye(n);
  for k = 1:n-2
    x = H(k+1:n,k);
    [v,beta] = vhouse(x);
    H(k+1:n,k:n)   = H(k+1:n,k:n)   - beta*v*(v'*H(k+1:n,k:n));
    H(1:n,k+1:n)   = H(1:n,k+1:n)   - beta*(H(1:n,k+1:n)*v)*v';
    Q(:,k+1:n)     = Q(:,k+1:n)     - beta*(Q(:,k+1:n)*v)*v';
  end
end


function [v, beta] = vhouse(x)
% VHOUSE   Genera vettore e coefficiente per riflessione di Householder
  n = length(x);
  sigma = x(2:end)'*x(2:end);
  v = [1; x(2:end)];
  if sigma==0
    beta = 0;
  else
    mu = sqrt(x(1)^2 + sigma);
    if x(1)<=0
      v(1) = x(1)-mu;
    else
      v(1) = -sigma/(x(1)+mu);
    end
    beta = 2*v(1)^2/(sigma+v(1)^2);
    v = v/v(1);
  end
end


function [Q,R] = qr_hessenberg(A)
% QR_HESSENBERG  QR tramite rotazioni di Givens su matrice quasi tridiagonale
  n = size(A,1);
  Q = eye(n); R = A;
  for j = 1:n-1
    for i = n:-1:j+1
      a = R(i-1,j); b = R(i,j);
      if b~=0
        r = hypot(a,b);
        c = a/r; s = -b/r;
        G = eye(n);
        G([i-1,i],[i-1,i]) = [c -s; s c];
        R = G'*R;
        Q = Q*G;
      end
    end
  end
end
