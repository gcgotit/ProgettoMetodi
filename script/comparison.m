%% compare_builtin_vs_custom.m
% Confronto tra implementazioni “artigianali” e built-in di MATLAB
clear; clc; close all;

%% 0) Caricamento dati
fprintf('>>> Caricamento dataset...\n');
load('dataset/volti_dataset.mat');  % A (m×n), labels(n)

[m, n] = size(A);

%% 1) Calcolo volto medio e centratura (stessa per entrambi)
mean_face = mean(A,2);
A_centered = A - mean_face;

%% Pre-split train/test (stesso per entrambi)
num_subjects = max(labels);
train_idx = []; test_idx = [];
for s = 1:num_subjects
    idx = find(labels==s);
    idx = idx(randperm(length(idx)));
    train_idx = [train_idx, idx(1:7)];
    test_idx  = [test_idx,  idx(8:end)];
end

%% Estrazione features 
k = 100;                    % numero di eigenfaces
% — Artigianale: U calcolata a monte
load('results/svd_data.mat','U');
U_k_custom = U(:,1:k);
proj_custom = U_k_custom' * A_centered;  % [k x n]

% — Built-in SVD:
fprintf('\n=== BUILT-IN SVD ===\n');
tic;
[U_b,S_b,V_b] = svd(A_centered,'econ');  % matlab built-in
t_svd_builtin = toc;
fprintf('Built-in svd() in %.3f s\n', t_svd_builtin);
U_k_builtin = U_b(:,1:k);
proj_builtin = U_k_builtin' * A_centered;  % [k x n]

%% Prepara train/test set per entrambe
X_train_c = proj_custom(:,train_idx)';   y_train = labels(train_idx)';
X_test_c  = proj_custom(:,test_idx)';    y_test  = labels(test_idx)';

X_train_b = proj_builtin(:,train_idx)';  % same indices
X_test_b  = proj_builtin(:,test_idx)';

% One-hot encoding
num_classes = num_subjects;
Y_train = zeros(num_classes, length(y_train));
for i=1:length(y_train)
    Y_train(y_train(i),i) = 1;
end

%% Iperparametri MLP
hidden_size   = 128;
epochs        = 100;
learning_rate = 0.005;
dropout_rate  = 0.3;

%% 2A) Addestramento CUSTOM via mlp_gradient_descent
fprintf('\n=== CUSTOM MLP TRAINING ===\n');
tic;
[W1_c,b1_c,W2_c,b2_c,loss_c] = mlp_gradient_descent( ...
    X_train_c, Y_train, hidden_size, epochs, learning_rate, dropout_rate);
t_mlp_custom = toc;
fprintf('Custom MLP addestrata in %.3f s\n', t_mlp_custom);

% predizione custom
X_test_T = X_test_c';
Z1 = W1_c*X_test_T + b1_c; A1 = max(0,Z1);
Z2 = W2_c*A1 + b2_c; Z2 = Z2 - max(Z2,[],1);
A2 = exp(Z2)./sum(exp(Z2),1);
[~, y_pred_c] = max(A2,[],1);
acc_custom = mean(y_pred_c'==y_test)*100;
fprintf('Accuracy custom MLP: %.2f%%\n', acc_custom);

%% 2B) Addestramento BUILT-IN MLP con fitcnet
fprintf('\n=== BUILT-IN MLP (fitcnet) ===\n');
opts.Standardize = true;
opts.LayerSizes  = hidden_size;
opts.Activation  = "relu";
opts.Lambda      = 0;
opts.Loss        = "crossentropy";
opts.Options     = trainingOptions("sgdm", ...
    "MaxEpochs",epochs, "Verbose",false);

tic;
mdl = fitcnet( X_train_b, y_train, ...
    "Standardize",opts.Standardize, ...
    "LayerSizes", opts.LayerSizes, ...
    "Activation", opts.Activation, ...
    "Lambda", opts.Lambda, ...
    "Loss", opts.Loss, ...
    "Options", opts.Options );
t_mlp_builtin = toc;
fprintf('Built-in MLP addestrata in %.3f s\n', t_mlp_builtin);

% predizione built-in
y_pred_b = predict(mdl, X_test_b);
acc_builtin = mean(y_pred_b == y_test)*100;
fprintf('Accuracy built-in MLP: %.2f%%\n', acc_builtin);


%% 3) Riepilogo tempi e accuracy
fprintf('\n=== RIEPILOGO ===\n');
fprintf('SVD custom:    %.3f s | SVD built-in:    %.3f s\n', t_svd_custom, t_svd_builtin);
fprintf('MLP custom:    %.3f s | MLP built-in:    %.3f s\n', t_mlp_custom, t_mlp_builtin);
fprintf('Accuracy custom: %.2f%% | Accuracy built-in: %.2f%%\n', acc_custom, acc_builtin);

%% (Opzionale) plot confronto loss
figure;
plot(1:epochs, loss_c, 'LineWidth',1.5);
xlabel('Epoca'); ylabel('Loss custom');
title('Training Loss (custom MLP)');
grid on;
