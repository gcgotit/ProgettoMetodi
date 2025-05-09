clear; clc; close all;

% Carica A e labels da fase 1 (se già salvati)
% load('volti_dataset.mat'); % se hai usato save in fase 1

% Oppure esegui la FASE 1 qui (solo per test)

% Dimensioni
[m, n] = size(A); % m = numero pixel, n = numero immagini

% 1. Calcolo del volto medio
mean_face = mean(A, 2);

% 2. Centra i dati (sottrai volto medio a ogni colonna)
A_centered = A - mean_face;

% 3. Calcola la SVD
[U, S, V] = svd(A_centered, 'econ'); % 'econ' per efficienza

% 4. Visualizza il volto medio
figure;
imshow(reshape(mean_face, 112, 92), []);
title('Volto medio');

% 5. Visualizza le prime 9 eigenfaces
figure;
for i = 1:9
    subplot(3,3,i);
    eigenface = U(:,i);
    imshow(reshape(eigenface, 112, 92), []);
    title(['Eigenface ', num2str(i)]);
end

disp('Calcolo eigenfaces completato.');
