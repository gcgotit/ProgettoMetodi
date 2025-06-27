function [W, b, loss_history, acc_history] = ...
          slp_gradient_descent(X, Y_onehot, epochs, lr)
    % X: [n_samples × input_size]
    % Y_onehot: [n_classes × n_samples]

    [n_samples, input_size] = size(X);
    n_classes               = size(Y_onehot,1);

    % ---------- init ----------
    W = randn(n_classes, input_size) * sqrt(2/input_size);
    b = zeros(n_classes, 1);

    X_T          = X';                       % [I × N]
    loss_history = zeros(1, epochs);
    acc_history  = zeros(1, epochs);

    % ---------- training loop ----------
    for epoch = 1:epochs
        % ---- forward ----
        Z = W * X_T + b;                     % [C × N]
        Z = Z - max(Z,[],1);                 % stabilità numerica
        A = softmax(Z);                      % [C × N]

        % ---- loss ----
        eps  = 1e-12;
        L    = -sum(log(sum(A .* Y_onehot,1) + eps)) / n_samples;
        loss_history(epoch) = L;

        % ---- accuracy ----
        [~, y_pred] = max(A,        [], 1);
        [~, y_true] = max(Y_onehot, [], 1);
        acc         = mean(y_pred == y_true) * 100;
        acc_history(epoch) = acc;

        if mod(epoch,10) == 0
            fprintf('[SLP] Epoch %3d/%3d | Loss: %.4f | Acc: %.2f%%\n', ...
                    epoch, epochs, L, acc);
        end

        % ---- back-prop ----
        dZ = A - Y_onehot;                   % [C × N]
        dW = (dZ * X_T') / n_samples;        % [C × I]
        db = sum(dZ,2)  / n_samples;         % [C × 1]

        % ---- update ----
        W = W - lr * dW;
        b = b - lr * db;
    end
end

% --- helper ---
function A = softmax(Z)
    expZ = exp(Z);
    A    = expZ ./ sum(expZ,1);
end
