function [W1,b1,W2,b2,loss_history, acc_history] = ...
    mlp_gradient_descent(X, Y_onehot, hidden_size, epochs, lr, dropout_rate)
% MLP_GRADIENT_DESCENT  Addestra una MLP (1 hidden layer) con gradiente esplicito
%
%   X            [n_samples × input_size]
%   Y_onehot     [n_classes × n_samples]  -- one-hot
%   hidden_size  neuroni nel layer nascosto
%   epochs       epoche di training
%   lr           learning-rate
%   dropout_rate frazione di neuroni “spenti” (inverted dropout)
%
%   Ritorna: pesi, bias, loss_history e acc_history

    %% --- Init ---
    [n_samples, input_size] = size(X);
    n_classes = size(Y_onehot,1);

    W1 = randn(hidden_size, input_size) * sqrt(2/input_size);
    b1 = zeros(hidden_size,1);
    W2 = randn(n_classes, hidden_size) * sqrt(2/hidden_size);
    b2 = zeros(n_classes,1);

    X_T = X';                 % [I × N]
    loss_history = zeros(1,epochs);
    acc_history  = zeros(1,epochs);  % initialize accuracy history

    %% --- Training loop ---
    for epoch = 1:epochs
        % ---------- Forward ----------
        Z1 = W1 * X_T + b1;          % [H × N]
        A1 = max(0, Z1);             % ReLU

        M  = (rand(size(A1)) > dropout_rate) / (1-dropout_rate);
        A1_drop = A1 .* M;           % inverted dropout

        Z2 = W2 * A1_drop + b2;      % [C × N]
        Z2 = Z2 - max(Z2,[],1);      % stabilita numerica
        expZ = exp(Z2);
        A2 = expZ ./ sum(expZ,1);    % soft-max -> prob.

        % ---------- Loss ----------
        eps = 1e-12;
        L = -mean(log(sum(A2 .* Y_onehot,1) + eps));  % cross-entropy
        loss_history(epoch) = L;

        % ---------- Accuracy ----------
        [~, y_pred] = max(A2,[],1);
        [~, y_true] = max(Y_onehot,[],1);
        acc = mean(y_pred == y_true) * 100;
        acc_history(epoch) = acc;    % store training accuracy

        if mod(epoch,10)==0 || epoch==1
            fprintf('[Epoch %3d/%3d]  Loss = %.4f   Acc = %.4f\n', ...
                     epoch, epochs, L, acc);
        end

        % ---------- Back-prop ----------
        dZ2 = A2 - Y_onehot;                 % [C × N]
        dW2 = (dZ2 * A1_drop') / n_samples;
        db2 = sum(dZ2,2) / n_samples;

        dA1 = W2' * dZ2;                     % [H × N]
        dZ1 = dA1 .* (Z1 > 0);               % ReLU'
        dW1 = (dZ1 * X_T') / n_samples;
        db1 = sum(dZ1,2) / n_samples;

        % ---------- Update ----------
        W2 = W2 - lr * dW2;
        b2 = b2 - lr * db2;
        W1 = W1 - lr * dW1;
        b1 = b1 - lr * db1;
    end
end
