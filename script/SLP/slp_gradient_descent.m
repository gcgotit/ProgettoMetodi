% == SLP_GradientDescent.m ==
% Addestramento e valutazione di una Single Layer Perceptron artigianale

function [W, b, loss_history] = slp_gradient_descent(X, Y_onehot, epochs, lr)
    % X: [n_samples x input_size]
    % Y_onehot: [n_classes x n_samples]
    [n_samples, input_size] = size(X);
    n_classes = size(Y_onehot,1);

    rng(0);
    W = randn(n_classes, input_size) * sqrt(2/input_size);
    b = zeros(n_classes, 1);

    X_T = X';       % [input_size x n_samples]
    loss_history = zeros(1, epochs);

    for epoch = 1:epochs
        % Forward
        Z = W * X_T + b;
        % Softmax (stabile)
        Z = Z - max(Z,[],1);     % stabilità numerica
        expZ = exp(Z);
        A = expZ ./ sum(expZ, 1); % [n_classes x n_samples]
        
        % Loss (cross-entropy)
        eps = 1e-12;
        L = -sum(log(sum(A .* Y_onehot, 1) + eps)) / n_samples;
        loss_history(epoch) = L;
        if mod(epoch, 10) == 0
            fprintf('[SLP] Epoch %3d/%3d - Loss: %.4f\n', epoch, epochs, L);
        end
        
        % Backward (gradients)
        dZ = A - Y_onehot;
        dW = (dZ * X_T') / n_samples;  % [C x I]
        db = sum(dZ, 2) / n_samples;   % [C x 1]
        
        % Aggiornamento pesi
        W = W - lr * dW;
        b = b - lr * db;
    end
end
