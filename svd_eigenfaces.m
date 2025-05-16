%% svd_eigenfaces.m
clear; clc; close all;

%% 0) Caricamento dati
fprintf('>>> Caricamento dataset...\n');
tic;
load('volti_dataset.mat');  % A (m×n), labels...
fprintf('   fatto in %.2f s\n', toc);

%% 1) Parametri
[m, n] = size(A);
tol   = 1e-3;
maxit = 200;

%% 2) Calcolo volto medio
fprintf('>>> Calcolo volto medio...\n');
tic;
mean_face = mean(A, 2);
save('mean_face.mat', 'mean_face');
fprintf('   fatto in %.2f s\n', toc);

%% 3) Centratura
fprintf('>>> Centratura dati...\n');
tic;
A_centered = A - mean_face;
fprintf('   fatto in %.2f s\n', toc);

%% 4) SVD con svd_BC
fprintf('>>> Lancio svd_BC (tol=%.1e,maxit=%d)...\n', tol, maxit);
tic;
[U, S, V] = svd_BC(A_centered, tol, maxit);
t_svd = toc;
fprintf('   svd_BC completata in %.2f s\n', t_svd);
save('svd_data.mat', 'U', 'S', 'V');

%% 5) Visualizzazione
fprintf('>>> Visualizzazione risultati...\n');
% Volto medio
figure('Name','Volto Medio','NumberTitle','off');
imshow(reshape(mean_face,112,92),[]);
title('Volto Medio');

% Prime 9 Eigenfaces
figure('Name','Prime 9 Eigenfaces','NumberTitle','off');
for i = 1:9
    subplot(3,3,i);
    imshow(reshape(U(:,i),112,92),[]);
    title(sprintf('Eigenface %d', i));
end

fprintf('>>> Fine script (totale %.2f s)\n', sum([toc-t_svd, t_svd]));  % approssimazione
