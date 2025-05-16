function [Q,R] = qr_tridiag(T)
% QR_TRIDIAG fattorizzazione QR di una matrice tridiagonale T via Givens
% Input:  T (k×k) tridiagonale, solo tre diagonali non nulle
% Output: Q (k×k) ortogonale, R (k×k) triangolare superiore

  k = size(T,1);
  R = T;
  Q = eye(k);

  for i = 1:k-1
    a = R(i,i);
    b = R(i+1,i);
    if b == 0
      continue;
    end
    r = hypot(a,b);
    c = a/r;
    s = -b/r;
    % costruisco le 2×2 di rotazione
    G2 = [c  s;
         -s  c];    % questa fa [r;0]' = G2 * [a;b]
    % applico solo sul blocco di due righe di R
    rows = [i, i+1];
    R(rows, :) = G2 * R(rows, :);
    % accumulo su Q sulle stesse due colonne
    Q(:, rows) = Q(:, rows) * G2';  % G2' per accumulare correttamente
  end
end