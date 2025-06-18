function [W_enc, b_enc, W_dec, b_dec, loss_history, test_loss_history] = ...
    autoencoder_train_2(X_train, X_test, mean_face, bottleneck_size, epochs, lr)
% AUTOENCODER_TRAIN   Autoencoder training with per-epoch test-loss
%
% [W_enc,b_enc,W_dec,b_dec,loss_history,test_loss_history] =
%   autoencoder_train(X_train, X_test, mean_face, bottleneck_size, epochs, lr)
%
%   X_train:    [n_train x input_dim]
%   X_test:     [n_test  x input_dim]
%   mean_face:  [input_dim x 1]
%   bottleneck_size, epochs, lr: come prima

    [n_train, input_dim] = size(X_train);
    n_test  = size(X_test,1);

    rng(0);
    W_enc = randn(bottleneck_size, input_dim) * sqrt(2/input_dim);
    b_enc = zeros(bottleneck_size,1);
    W_dec = randn(input_dim, bottleneck_size) * sqrt(2/bottleneck_size);
    b_dec = zeros(input_dim,1);

    Xtr = X_train';    % [input_dim x n_train]
    Xte = X_test';     % [input_dim x n_test]

    loss_history      = zeros(1, epochs);
    test_loss_history = zeros(1, epochs);

    for epoch = 1:epochs
        total_loss = 0;

        % --- TRAIN LOOP ---
        for i = 1:n_train
            x = Xtr(:,i);
            % forward
            h     = max(0, W_enc * x + b_enc);
            x_rec = W_dec * h + b_dec;
            % loss
            err = x_rec - x;
            total_loss = total_loss + mean(err.^2);
            % backprop
            dL_dxrec  = 2 * err / input_dim;
            dW_dec    = dL_dxrec * h';
            db_dec    = dL_dxrec;
            dL_dh     = W_dec' * dL_dxrec;
            dh_dz     = (W_enc * x + b_enc) > 0;
            dL_dz_enc = dL_dh .* dh_dz;
            dW_enc    = dL_dz_enc * x';
            db_enc    = dL_dz_enc;
            % update
            W_dec = W_dec - lr * dW_dec;
            b_dec = b_dec - lr * db_dec;
            W_enc = W_enc - lr * dW_enc;
            b_enc = b_enc - lr * db_enc;
        end

        loss_history(epoch) = total_loss / n_train;

        % --- TEST EVALUATION per epoca ---
        H_test  = max(0, W_enc * Xte + b_enc);  % [k×n_test]
        Xhat_te = W_dec * H_test + b_dec;       % [k×n_test]
        % MSE nello spazio PCA normalizzato:
        Err_te  = Xte - Xhat_te;                % [k×n_test]
        test_loss_history(epoch) = mean(Err_te(:).^2);

        if mod(epoch,10)==0
            fprintf('[AE] Epoca %3d/%3d: Train MSE=%.5f, Test MSE=%.5f\n', ...
                    epoch, epochs, loss_history(epoch), test_loss_history(epoch));
        end
    end
end
