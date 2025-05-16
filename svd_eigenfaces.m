clear; clc; close all;

% Carichiamo A e labels da fase precedente 
load('volti_dataset.mat'); 

% Dimensioni
[m, n] = size(A); % m = numero pixel, n = numero immagini

% 1. Calcoliamo il volto medio
mean_face = mean(A, 2);

% Salviamo il volto medio
save('mean_face.mat', 'mean_face')

% 2. Centriamo i dati (sottraiamo volto medio a ogni colonna)
A_centered = A - mean_face;

% 3. Calcoliamo la SVD usando la nostra funzione personalizzata
[U, S, V] = svd_BC(A_centered);

% Salviamo i risultati della SVD
save('svd_data.mat', 'U', 'S', 'V')

% 4. Visualizziamo il volto medio
figure;
imshow(reshape(mean_face, 112, 92), []);
title('Volto medio');

% 5. Visualizziamo le prime 9 eigenfaces
figure;
for i = 1:9
    subplot(3,3,i);
    eigenface = U(:,i);
    imshow(reshape(eigenface, 112, 92), []);
    title(['Eigenface ', num2str(i)]);
end

disp('Calcolo eigenfaces completato con SVD_BC.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SVD_BC: implementazione della SVD di A tramite riduzione a problemi di autovalori
function [U,S,V] = svd_BC(A)
  % A: matrice di dati (es. colonne = campioni, righe = feature)
  tol   = 1e-8;    % tolleranza per convergenza QR‐shift
  maxit = 500;     % numero massimo di iterazioni QR

  % Step 0: formiamo la matrice simmetrica ATA per ricondurci a un problema
  % di autovalori in ℝ^n invece che di valori singolari in ℝ^{m×n}
  ATA = A' * A;
  
  % Step 1: calcolo di autovalori e autovettori di ATA tramite iterazioni QR‐shift
  % T diventerà tridiagonale (o quasi) con autovalori sulla diagonale,
  % M accumula le riflessioni in modo da formare la matrice degli autovettori
  [T, M] = qrshift(ATA, tol, maxit);
  
  % Estraiamo gli autovalori da T e ordiniamo in ordine decrescente
  lambda = diag(T);
  [lambda, idx] = sort(lambda, 'descend');
  
  % Determiniamo p = numero di autovalori significativamente > 0
  p = sum(lambda > tol);
  lambda = lambda(1:p);
  
  % V è la matrice dei vettori singolari destri (autovettori di ATA),
  % presi nelle prime p colonne ordinate
  V = M(:, idx(1:p));
  
  % Step 2: costruiamo la matrice S (sigma) e U (vettori singolari sinistri)
  % I valori singolari sono le radici degli autovalori positivi
  sigma = sqrt(lambda);
  S = zeros(size(A));
  S(1:p,1:p) = diag(sigma);      % posizioniamo sigma sulla pseudo-diagonale
  
  % U = A * V * Sigma^{-1}, da definizione di SVD: A V = U Sigma
  U = A * V * diag(1 ./ sigma);
end


% MY_QR: fattorizzazione QR via riflessioni di Householder
function [Q,R] = my_qr(A)
  [m,n] = size(A);
  Q = eye(m);    % accumulatore delle riflessioni
  R = A;         % matrice su cui riduciamo a triangolare superiore

  % Per ogni colonna j fino a min(m-1,n)
  for j = 1 : min(m-1,n)
    % x è il vettore da annullare sotto la diagonale su R(j:m,j)
    x = R(j:m, j);
    % e1 è il primo vettore base di dimensione length(x)
    e1 = zeros(length(x),1); e1(1) = 1;

    % calcolo del vettore Householder v
    % sign(x1)*||x|| e1 + x crea riflessione che azzera tutti i componenti
    v = sign(x(1))*norm(x)*e1 + x;
    v = v / norm(v);  % normalizzazione per ottenere riflessione ortogonale

    % costruiamo la riflessione Hi a blocchi
    Hi = eye(m);
    Hi(j:m, j:m) = eye(length(v)) - 2*(v*v');  % H = I - 2 v v^T

    % Applichiamo H su R e accumuliamo in Q
    R = Hi * R;       % annulliamo sotto-diagonale
    Q = Q * Hi';      % Q = H1' * H2' * ... accumula riflessioni
  end
end


% QRSHIFT: iterazione QR con shift di Wilkinson per autovalori
function [T, M] = qrshift(A, tol, maxit)
  % A deve essere quadrata e simmetrica per garantire convergenza
  [n,n2] = size(A);
  assert(n==n2, 'qrshift richiede matrice quadrata');

  % 1) riduzione a forma Hessenberg (simmetrica → tridiagonale)
  H = A;
  M = eye(n);   % accumula le riflessioni Householder
  for k = 1 : n-2
    % estraiamo il sotto-vettore da eliminare
    x = H(k+1:n, k);
    e1 = zeros(length(x),1); e1(1)=1;
    v = sign(x(1))*norm(x)*e1 + x; 
    v = v / norm(v);
    Hi = eye(n);
    Hi(k+1:n, k+1:n) = eye(length(v)) - 2*(v*v');
    H = Hi * H * Hi;    % riduciamo H a tridiagonale
    M = M * Hi;         % accumuliamo le riflessioni in M
  end

  % 2) ciclo di QR-iteration con shift di Wilkinson
  for it = 1 : maxit
    mu = H(n,n);               % shift = ultimo elemento diagonale
    [Q,R] = my_qr(H - mu*eye(n));  % fattorizzazione QR su H - mu I
    H = R * Q + mu * eye(n);      % ricostruzione con shift
    M = M * Q;                    % accumulo trasformazioni sulle direzioni

    % criterio di stoppaggio: ogni sub-diagonale è quasi nulla
    if norm(tril(H,-1), 'fro') < tol
      break;
    end
  end

  T = H;  % T ora è quasi triangolare con autovalori sulla diagonale
end










%{
%% Funzione svd_BC ORIGINALE CON EIG()
function [U, S, V] = svd_BC(A)
    % Calcolo della SVD tramite decomposizione di A^T A
    % Input:  A (matrice centrata)
    % Output: U, S, V tali che A = U*S*V'

    % Step 1: Calcolo di A^T * A
    ATA = A' * A;

    % Step 2: Autovalori e autovettori di A^T * A
    [V_raw, D_raw] = eig(ATA); % D_raw è diagonale con autovalori, V_raw con autovettori

    % Step 3: Ordiniamo gli autovalori in ordine decrescente
    [eigenvalues, idx] = sort(diag(D_raw), 'descend');
    eigenvalues = eigenvalues(eigenvalues > 1e-10); % scarta autovalori ~0
    V = V_raw(:, idx);
    V = V(:, 1:length(eigenvalues)); % taglia colonne nulle

    % Step 4: Calcolo dei valori singolari (radice degli autovalori)
    singular_values = sqrt(eigenvalues);
    S = diag(singular_values);

    % Step 5: Calcolo di U = A * V * S^-1
    U = A * V * diag(1 ./ singular_values);
end
%}