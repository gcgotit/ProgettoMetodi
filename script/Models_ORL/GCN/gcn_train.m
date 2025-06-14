function [W, loss_history] = gcn_train(X, Y_onehot, A_hat, epochs, lr)
% GCN_TRAIN  Addestra una Graph Convolutional Network artigianale con 1 layer
% Input:
%   X        - [n_samples x input_dim]  (features dei nodi)
%   Y_onehot - [n_classes x n_samples]  (label in one-hot)
%   A_hat    - [n_samples x n_samples]  (adjacency matrix normalizzata)
%   epochs   - epoche di training
%   lr       - learning rate
% Output:
%   W        - pesi GCN
%   loss_history - MSE/cross-entropy loss durante il training

[n_samples, input_dim] = size(X);
n_classes = size(Y_onehot, 1);

rng(0);
W = randn(input_dim, n_classes) * sqrt(2/input_dim);  % [input_dim x n_classes]
loss_history = zeros(1, epochs);

for epoch = 1:epochs
    % --- Forward: propagation + linear classifier ---
    H = A_hat * X * W;  % [n_samples x n_classes]
    Z = H';             % [n_classes x n_samples]
    Z = Z - max(Z,[],1);   % stabilità numerica
    expZ = exp(Z);
    A2 = expZ ./ sum(expZ,1); % softmax

    % --- Loss ---
    eps = 1e-12;
    L = -sum(log(sum(A2 .* Y_onehot,1) + eps))/n_samples;
    loss_history(epoch) = L;

    if mod(epoch,10)==0
        fprintf('[GCN] Epoch %3d/%3d - Loss: %.4f\n', epoch, epochs, L);
    end

    % --- Backprop: gradiente ---
    dZ = A2 - Y_onehot;                        % [C x N]
    dW = (X' * A_hat' * dZ') / n_samples;      % [input_dim x C]

    W = W - lr * dW;
end

end
