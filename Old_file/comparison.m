%% compare_svd_vs_custom_svd_manual_mlp.m
% Confronto tra SVD artigianale vs built-in + MLP manuale via gradient descent
clear; clc; close all;

%% 0) Caricamento dataset
fprintf('>>> Caricamento dataset...\n');
load('dataset/volti_dataset.mat');  % variabili: A (m×n), labels (1×n)
[m, n] = size(A);

%% 1) Pre‐processing comune
mean_face   = mean(A,2);
A_centered  = A - mean_face;

% split train/test coerente per entrambi i metodi
num_subjects = max(labels);
train_idx = []; test_idx = [];
for s = 1:num_subjects
    idx = find(labels==s);
    idx = idx(randperm(numel(idx)));
    train_idx = [train_idx, idx(1:7)];
    test_idx  = [test_idx,  idx(8:end)];
end

y_train = labels(train_idx)';
y_test  = labels(test_idx)';

% one-hot encoding per gradient descent
Y_train = zeros(num_subjects, numel(y_train));
for i = 1:numel(y_train)
    Y_train(y_train(i), i) = 1;
end

%% 2) Parametri
k              = 100;     % numero di eigenfaces
hidden_size    = 128;
epochs         = 100;
learning_rate  = 0.005;
dropout_rate   = 0.3;

%% 3) SVD ARTIGIANALE
fprintf('\n=== SVD ARTIGIANALE ===\n');
tic;
[U_c,S_c,V_c] = svd_BC(A_centered, 1e-4, 500);
t_svd_custom = toc;
fprintf('svd_BC completata in %.3f s\n', t_svd_custom);

U_c_k = U_c(:,1:k);
proj_custom = U_c_k' * A_centered;  % [k x n]

%% 4) SVD BUILT-IN
fprintf('\n=== SVD BUILT-IN ===\n');
tic;
[U_b,~,~] = svd(A_centered,'econ');
t_svd_builtin = toc;
fprintf('svd built-in completata in %.3f s\n', t_svd_builtin);

U_b_k = U_b(:,1:k);
proj_builtin = U_b_k' * A_centered;  % [k x n]

%% 5) Estrazione feature train/test
X_train_c = proj_custom(:,train_idx)';   % [n_train x k]
X_test_c  = proj_custom(:,test_idx)';

X_train_b = proj_builtin(:,train_idx)';  % [n_train x k]
X_test_b  = proj_builtin(:,test_idx)';

%% 6A) MLP via gradient descent su SVD artigianale
fprintf('\n=== TRAIN CUSTOM MLP su SVD ARTIGIANALE ===\n');
tic;
[W1_c,b1_c,W2_c,b2_c,loss_c] = mlp_gradient_descent( ...
    X_train_c, Y_train, hidden_size, epochs, learning_rate, dropout_rate);
t_mlp_custom = toc;
fprintf('Custom MLP addestrata in %.3f s\n', t_mlp_custom);

% predizione
X_test_T = X_test_c';
Z1 = W1_c*X_test_T + b1_c;   A1 = max(0,Z1);
Z2 = W2_c*A1 + b2_c;         Z2 = Z2 - max(Z2,[],1);
A2 = exp(Z2)./sum(exp(Z2),1);
[~, y_pred_c] = max(A2,[],1);
acc_custom = mean(y_pred_c'==y_test)*100;
fprintf('Accuracy custom MLP su SVD artigianale: %.2f%%\n', acc_custom);

%% 6B) MLP via gradient descent su SVD built-in
fprintf('\n=== TRAIN CUSTOM MLP su SVD BUILT-IN ===\n');
tic;
[W1_b,b1_b,W2_b,b2_b,loss_b] = mlp_gradient_descent( ...
    X_train_b, Y_train, hidden_size, epochs, learning_rate, dropout_rate);
t_mlp_builtin_mlp = toc;
fprintf('Custom MLP addestrata in %.3f s\n', t_mlp_builtin_mlp);

% predizione
X_test_Tb = X_test_b';
Z1b = W1_b*X_test_Tb + b1_b; A1b = max(0,Z1b);
Z2b = W2_b*A1b + b2_b;       Z2b = Z2b - max(Z2b,[],1);
A2b = exp(Z2b)./sum(exp(Z2b),1);
[~, y_pred_b] = max(A2b,[],1);
acc_builtin_mlp = mean(y_pred_b'==y_test)*100;
fprintf('Accuracy custom MLP su SVD built-in: %.2f%%\n', acc_builtin_mlp);

%% 7) Riepilogo
fprintf('\n=== RIEPILOGO COMPLETO ===\n');
fprintf('SVD custom:   %.3f s | SVD built-in:   %.3f s\n', t_svd_custom, t_svd_builtin);
fprintf('MLP@customSVD: %.3f s | MLP@builtSVD: %.3f s\n', t_mlp_custom, t_mlp_builtin_mlp);
fprintf('Acc@customSVD: %.2f%% | Acc@builtSVD: %.2f%%\n', acc_custom, acc_builtin_mlp);
