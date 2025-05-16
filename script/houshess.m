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
