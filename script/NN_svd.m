%% nn_svd.m (versione con mlp_gradient_descent)
clear; clc; close all;

%% 0) Caricamento e preparazione dati
load('dataset/volti_dataset.mat');    % A, labels
load('results/mean_face.mat');        % mean_face
load('results/svd_data.mat');         % U,S,V

% Parametri SVD
k = 100;
U_k = U(:,1:k);

A_centered = A - mean_face;
projections= U_k' * A_centered;  % [k x n]
    
% Split train/test
num_subjects = max(labels);
train_idx = []; test_idx = [];
for s=1:num_subjects
  idx = find(labels==s);
  idx = idx(randperm(length(idx)));
  train_idx = [train_idx, idx(1:7)];
  test_idx  = [test_idx,  idx(8:end)];
end

X_train = projections(:,train_idx)';   % [n_train x k]
y_train = labels(train_idx)';
X_test  = projections(:,test_idx)';    % [n_test  x k]
y_test  = labels(test_idx)';

% One‐hot encoding
num_classes = num_subjects;
Y_train = zeros(num_classes, length(y_train));
for i=1:length(y_train)
  Y_train(y_train(i),i) = 1;
end

%% 1) Iperparametri MLP
hidden_size   = 128;
epochs        = 100;
learning_rate = 0.005;
dropout_rate  = 0.3;

%% 2) Addestramento tramite gradient descent “artigianale”
fprintf('--- Training MLP via gradient descent ---\n');
tic;
[W1,b1,W2,b2,loss_hist] = mlp_gradient_descent( ...
    X_train, Y_train, hidden_size, epochs, learning_rate, dropout_rate);
fprintf('Training completato in %.2f s\n', toc);

%% 3) Valutazione su test set
X_test_T = X_test';
Z1 = W1 * X_test_T + b1;
A1 = max(0, Z1);
Z2 = W2 * A1 + b2;
Z2 = Z2 - max(Z2,[],1);
expZ = exp(Z2);
A2 = expZ ./ sum(expZ,1);
[~, y_pred] = max(A2,[],1);
acc = mean(y_pred'==y_test)*100;
fprintf('Accuracy on test set: %.2f%%\n', acc);

%% 4) plot della loss
figure; plot(1:epochs, loss_hist, 'LineWidth',1.5);
xlabel('Epoca'); ylabel('Cross-Entropy Loss'); grid on;
title('Andamento della Loss in Training');
