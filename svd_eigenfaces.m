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

%% Funzione svd_BC 
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