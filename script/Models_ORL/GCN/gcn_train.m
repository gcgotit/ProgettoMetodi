function [W, loss_history, acc_history] = gcn_train(X, Y_onehot, A_hat, epochs, lr)
% GCN_TRAIN  Addestra una GCN (1 layer) con gradient descent
%
%   X           [n_samples × input_dim]
%   Y_onehot    [n_classes × n_samples]
%   A_hat       [n_samples × n_samples] mat. di adiacenza normalizzata
%   epochs      numero epoche
%   lr          learning rate
%
% Ritorna:
%   W           [input_dim × n_classes]  – pesi
%   loss_history [1 × epochs]
%   acc_history  [1 × epochs]

    [n_samples, input_dim] = size(X);
    n_classes = size(Y_onehot, 1);

    % Initialization
    W = randn(input_dim, n_classes) * sqrt(2/input_dim);
    loss_history = zeros(1, epochs);
    acc_history  = zeros(1, epochs);

    for epoch = 1:epochs
        % --- Forward ---
        H   = A_hat * X * W;   % [n_samples × n_classes]
        Z   = H';              % [n_classes × n_samples]
        Z   = Z - max(Z,[],1);
        expZ= exp(Z);
        A2  = expZ ./ sum(expZ,1);

        % --- Loss (cross-entropy) ---
        eps = 1e-12;
        L = -sum(log(sum(A2 .* Y_onehot,1) + eps)) / n_samples;
        loss_history(epoch) = L;

        % --- Accuracy sul training set ---
        [~, y_pred] = max(A2,[],1);
        [~, y_true] = max(Y_onehot,[],1);
        acc_history(epoch) = mean(y_pred == y_true) * 100;

        % --- Log ogni 10 epoche ---
        if mod(epoch,10)==0 || epoch==1
            fprintf('[GCN] Epoca %3d/%3d — Loss: %.4f — Acc: %.2f%%\n', ...
                     epoch, epochs, L, acc_history(epoch));
        end

        % --- Backprop & update ---
        dZ = A2 - Y_onehot;                          % [C × N]
        dW = (X' * A_hat' * dZ') / n_samples;        % [input_dim × C]
        W  = W - lr * dW;
    end
end
