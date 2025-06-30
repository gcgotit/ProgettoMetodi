function [W, b, loss_history, acc_history] = slp_gradient_descent(X, Y_onehot, epochs, lr)
% SLP_GRADIENT_DESCENT  Addestra una Single Layer Perceptron (softmax)
%
%   X:           [n_samples × input_size]
%   Y_onehot:    [n_classes × n_samples]
%   epochs:      numero epoche
%   lr:          learning rate
%
% Ritorna: W, b, loss_history [1×epochs], acc_history [1×epochs]

[n_samples, input_size] = size(X);
[n_classes, ~]       = size(Y_onehot);

% Inizializzazione pesi (Xavier)
W = randn(n_classes, input_size) * sqrt(2/input_size);
b = zeros(n_classes, 1);

X_T = X';  % [input_size × n_samples]
loss_history = zeros(1, epochs);
acc_history  = zeros(1, epochs);

for epoch = 1:epochs
    % --- Forward pass ---
    Z = W * X_T + b;           % [n_classes × n_samples]
    Z = Z - max(Z,[],1);       % per stabilita numerica
    expZ = exp(Z);
    A = expZ ./ sum(expZ,1);   % softmax output

    % --- Loss ---
    eps = 1e-12;
    L = -mean(log(sum(A .* Y_onehot,1) + eps));
    loss_history(epoch) = L;

    % --- Accuracy ---
    [~, y_pred] = max(A,[],1);
    [~, y_true] = max(Y_onehot,[],1);
    acc = mean(y_pred == y_true) * 100;
    acc_history(epoch) = acc;

    if mod(epoch,10)==0 || epoch==1
        fprintf('[SLP] Epoca %3d/%3d - Loss: %.4f - Acc: %.2f%%\n', epoch, epochs, L, acc);
    end

    % --- Backward pass ---
    dZ = A - Y_onehot;                   % [C × N]
    dW = (dZ * X_T') / n_samples;
    db = sum(dZ,2) / n_samples;

    % --- Update ---
    W = W - lr * dW;
    b = b - lr * db;
end
end
