function [W1,b1,W2,b2,loss_history] = ...
    mlp_gradient_descent(X, Y_onehot, hidden_size, epochs, lr, dropout_rate)

% MLP_GRADIENT_DESCENT Addestra una MLP a un hidden layer con discesa
% del gradiente esplicita.
% 
% Input:
%   X            dimensioni [n_samples x input_size]
%   Y_onehot     dimensioni [n_classes x n_samples]
%   hidden_size  numero di neuroni nel layer nascosto
%   epochs       numero di epoche
%   lr           learning rate
%   dropout_rate tasso di dropout in training
%
% Output:
%   W1, b1       pesi e bias del primo layer
%   W2, b2       pesi e bias del secondo layer
%   loss_history vettore [1 x epochs] con il valore di loss ad ogni epoca

    [n_samples, input_size] = size(X);
    n_classes = size(Y_onehot,1);

    % He initialization
    rng(0);
    W1 = randn(hidden_size, input_size) * sqrt(2/input_size);
    b1 = zeros(hidden_size,1);
    W2 = randn(n_classes, hidden_size)* sqrt(2/hidden_size);
    b2 = zeros(n_classes,1);

    X_T = X';       % [input_size x n_samples]
    loss_history = zeros(1,epochs);

    for epoch = 1:epochs
        % ---- Forward pass ----
        Z1 = W1*X_T + b1;               
        A1 = max(0, Z1);                % ReLU

        % ---- Dropout (inverted) ----
        M = (rand(size(A1)) > dropout_rate) / (1-dropout_rate);
        A1_drop = A1 .* M;             

        Z2 = W2*A1_drop + b2;           
        % Softmax (stable)
        Z2 = Z2 - max(Z2,[],1);
        expZ = exp(Z2);
        A2 = expZ ./ sum(expZ,1);      % [n_classes x n_samples]

        % ---- Compute cross-entropy loss ----
        eps = 1e-12;
        L = -sum(log(sum(A2 .* Y_onehot,1) + eps))/n_samples;
        loss_history(epoch) = L;
        if mod(epoch,10)==0
            fprintf('[Epoch %3d/%3d] Loss = %.4f\n', epoch, epochs, L);
        end

        % ---- Backward pass ----
        dZ2 = A2 - Y_onehot;                         % [C x N]
        dW2 = (dZ2 * A1_drop') / n_samples;          % [C x H]
        db2 = sum(dZ2,2) / n_samples;                % [C x 1]

        dA1 = W2' * dZ2;                             % [H x N]
        dZ1 = dA1 .* (Z1>0);                         % ReLU'
        dW1 = (dZ1 * X_T') / n_samples;              % [H x I]
        db1 = sum(dZ1,2) / n_samples;                % [H x 1]

        % ---- Gradient update ----
        W2 = W2 - lr * dW2;
        b2 = b2 - lr * db2;
        W1 = W1 - lr * dW1;
        b1 = b1 - lr * db1;
    end
end
