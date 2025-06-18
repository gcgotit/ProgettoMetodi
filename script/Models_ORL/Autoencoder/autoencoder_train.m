%% Funzione di training autoencoder
function [W_enc, b_enc, W_dec, b_dec, loss_history] = ...
    autoencoder_train(X, bottleneck_size, epochs, lr)
% X: [n_samples x input_dim], bottleneck_size: dimensione codifica
% Ritorna pesi encoder/decoder, bias e loss storico

[n_samples, input_dim] = size(X);

rng(0);
W_enc = randn(bottleneck_size, input_dim) * sqrt(2/input_dim);
b_enc = zeros(bottleneck_size,1);
W_dec = randn(input_dim, bottleneck_size) * sqrt(2/bottleneck_size);
b_dec = zeros(input_dim,1);

loss_history = zeros(1, epochs);

for epoch = 1:epochs
    total_loss = 0;
    % Stochastic Gradient Descent (puoi renderlo mini-batch per ottimizzare)
    for i = 1:n_samples
        x = X(i,:)'; % input colonna

        % --- FORWARD ---
        h = max(0, W_enc * x + b_enc); % ReLU
        x_rec = W_dec * h + b_dec;     % lineare

        % --- LOSS (MSE) ---
        err = x_rec - x;
        loss = mean(err.^2);
        total_loss = total_loss + loss;

        % --- BACKPROP ---
        dL_dxrec = 2 * (x_rec - x) / input_dim; % derivata MSE
        dW_dec = dL_dxrec * h';
        db_dec = dL_dxrec;
        dL_dh = W_dec' * dL_dxrec;

        dh_dz = (W_enc * x + b_enc) > 0; % ReLU derivata
        dL_dz_enc = dL_dh .* dh_dz;

        dW_enc = dL_dz_enc * x';
        db_enc = dL_dz_enc;

        % --- AGGIORNA PESI ---
        W_dec = W_dec - lr * dW_dec;
        b_dec = b_dec - lr * db_dec;
        W_enc = W_enc - lr * dW_enc;
        b_enc = b_enc - lr * db_enc;
    end
    loss_history(epoch) = total_loss / n_samples;
    if mod(epoch,10)==0
        fprintf('[AE] Epoca %d/%d, MSE=%.5f\n', epoch, epochs, loss_history(epoch));
    end
end
end
