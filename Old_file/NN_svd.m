clear; clc; close all;

% Carica dati già calcolati
load('volti_dataset.mat');     % A, labels
load('mean_face.mat');         % mean_face
load('svd_data.mat');          % U, S, V

[m, n] = size(A);

% Parametri
k = 100;                     % numero di eigenfaces da usare
U_k = U(:, 1:k);             % prime k eigenfaces

% Centra i dati e proiettali nel sottospazio ridotto
A_centered = A - mean_face;
projections = U_k' * A_centered; % matrice k x n

% Divisione training/test
num_subjects = max(labels);
imgs_per_subject = n / num_subjects;
train_idx = [];
test_idx = [];
for s = 1:num_subjects
    idx = find(labels == s);
    idx = idx(randperm(length(idx)));
    train_idx = [train_idx, idx(1:7)];
    test_idx = [test_idx, idx(8:end)];
end

% Training e test set
X_train = projections(:, train_idx)';
y_train = labels(train_idx)';
X_test = projections(:, test_idx)';
y_test = labels(test_idx)';

% Parametri rete neurale
input_size = size(X_train, 2);
hidden_size = 128;
num_classes = max(y_train);
epochs = 100;
learning_rate = 0.005;
dropout_rate = 0.3;

% Inizializza pesi
W1 = randn(hidden_size, input_size) * sqrt(2 / input_size); 
b1 = zeros(hidden_size, 1);
W2 = randn(num_classes, hidden_size) * sqrt(2 / hidden_size);
b2 = zeros(num_classes, 1);

% One-hot encoding
Y_train_onehot = zeros(num_classes, length(y_train));
for i = 1:length(y_train)
    Y_train_onehot(y_train(i), i) = 1;
end

% Transponi input
X_train_T = X_train';

% -------------------------
% Addestramento
% -------------------------
for epoch = 1:epochs
    % Forward
    Z1 = W1 * X_train_T + b1;
    A1 = max(0, Z1);  % ReLU
    
    % Dropout solo nel training
    A1_dropout = apply_dropout(A1, dropout_rate);

    Z2 = W2 * A1_dropout + b2;

    % Softmax
    expZ = exp(Z2 - max(Z2, [], 1));
    A2 = expZ ./ sum(expZ, 1);

    % Loss
    epsilon = 1e-10;
    loss = -sum(log(sum(A2 .* Y_train_onehot, 1) + epsilon)) / length(y_train);
    if mod(epoch,10) == 0
        fprintf('Epoca %d - Loss: %.4f\n', epoch, loss);
    end

    % Backpropagation
    dZ2 = A2 - Y_train_onehot;
    dW2 = dZ2 * A1_dropout' / length(y_train);
    db2 = sum(dZ2, 2) / length(y_train);

    dA1 = W2' * dZ2;
    dZ1 = dA1 .* (Z1 > 0); % ReLU derivative
    dW1 = dZ1 * X_train_T' / length(y_train);
    db1 = sum(dZ1, 2) / length(y_train);

    % Update pesi
    W1 = W1 - learning_rate * dW1;
    b1 = b1 - learning_rate * db1;
    W2 = W2 - learning_rate * dW2;
    b2 = b2 - learning_rate * db2;
end

% -------------------------
% Test
% -------------------------
X_test_T = X_test';
Z1_test = W1 * X_test_T + b1;
A1_test = max(0, Z1_test);  % NO dropout nel test
Z2_test = W2 * A1_test + b2;

expZ = exp(Z2_test - max(Z2_test, [], 1));
A2_test = expZ ./ sum(expZ, 1);

[~, y_pred] = max(A2_test, [], 1);
y_pred = y_pred';

accuracy = sum(y_pred == y_test) / length(y_test);
fprintf('Accuratezza Rete Neurale con SVD: %.2f%%\n', accuracy * 100);

% Funzione Dropout
function A_dropout = apply_dropout(A, rate)
    mask = rand(size(A)) > rate;
    A_dropout = A .* mask;
end

% Output
%Epoca 10 - Loss: 0.9882
%Epoca 20 - Loss: 0.4178
%Epoca 30 - Loss: 0.2407
%Epoca 40 - Loss: 0.3442
%Epoca 50 - Loss: 0.3209
%Epoca 60 - Loss: 0.2377
%Epoca 70 - Loss: 0.1880
%Epoca 80 - Loss: 0.2407
%Epoca 90 - Loss: 0.3209
%Epoca 100 - Loss: 0.2407
%Accuratezza Rete Neurale: 96.75%