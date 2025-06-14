function [params, loss_history, acc_history, val_loss_history, val_acc_history] = ...
    cnn_train_full(imgs_train, Y_train, imgs_val, Y_val, ...
                   f1, n_f1, f2, n_f2, f3, n_f3, pool, ...
                   epochs, base_lr, lambda, dropout_rate, batch_size, patience)
% cnn_train_full   Allena una CNN 3×conv + 3×pool + FC + softmax con:
%                  • dropout (dropout_rate)
%                  • weight decay (lambda)
%                  • mini‐batch (batch_size)
%                  • LR scheduling (decay a epoca 26)
%                  • early stopping su validation (patience)
%
%  Input:
%   • imgs_train:  [img_h × img_w × N_train] tensore di immagini normalizzate 0–1
%   • Y_train:     [num_classes × N_train]  matrice one‐hot delle etichette training
%   • imgs_val:    [img_h × img_w × N_val]  tensore di validazione (same dims)
%   • Y_val:       [num_classes × N_val]
%   • f1, n_f1:    filtro/lunghezza primo conv layer
%   • f2, n_f2:    filtro/lunghezza secondo conv layer
%   • f3, n_f3:    filtro/lunghezza terzo conv layer
%   • pool:        dimensione pool (es. 2)
%   • epochs:      numero di epoche
%   • base_lr:     learning rate iniziale
%   • lambda:      L2 weight decay
%   • dropout_rate: dropout rate (es. 0.2)
%   • batch_size:  dimensione mini‐batch (es. 32)
%   • patience:    epoche di “fermo” per early stopping
%
%  Output:
%   • params: struct contenente i pesi finali {filters1,bias1,filters2,bias2,filters3,bias3,W_fc,b_fc,pool}
%   • loss_history: [1×num_epoche_effettive] loss di training
%   • acc_history:  [1×num_epoche_effettive] accuracy training (%)
%   • val_loss_history: [1×num_epoche_effettive] loss su validation
%   • val_acc_history:  [1×num_epoche_effettive] accuracy su validation (%)
%
%  Autore: [Tuo Nome]
%  Data:   [gg/mm/aaaa]

[img_h, img_w, num_train] = size(imgs_train);
num_classes = size(Y_train, 1);

% —————————————
% 1) Calcolo dimensioni intermedie e FC_input_size
% —————————————
conv1_h = img_h - f1 + 1;
conv1_w = img_w - f1 + 1;
pool1_h = floor(conv1_h / pool);
pool1_w = floor(conv1_w / pool);

conv2_h = pool1_h - f2 + 1;
conv2_w = pool1_w - f2 + 1;
pool2_h = floor(conv2_h / pool);
pool2_w = floor(conv2_w / pool);

conv3_h = pool2_h - f3 + 1;
conv3_w = pool2_w - f3 + 1;
pool3_h = floor(conv3_h / pool);
pool3_w = floor(conv3_w / pool);

fc_input_size = pool3_h * pool3_w * n_f3;  % es. 12×9×8 = 864

% —————————————
% 2) Inizializzazione pesi (He init)
% —————————————
rng(0);
filters1 = randn(f1, f1, n_f1) * sqrt(2/(f1*f1*n_f1)); bias1 = zeros(n_f1, 1);
filters2 = randn(f2, f2, n_f2, n_f1) * sqrt(2/(f2*f2*n_f2)); bias2 = zeros(n_f2, 1);
filters3 = randn(f3, f3, n_f3, n_f2) * sqrt(2/(f3*f3*n_f3)); bias3 = zeros(n_f3, 1);

W_fc = randn(num_classes, fc_input_size) * sqrt(2 / fc_input_size);
b_fc = zeros(num_classes, 1);

% —————————————
% 3) Inizializzo storici
% —————————————
loss_history     = zeros(1, epochs);
acc_history      = zeros(1, epochs);
val_loss_history = zeros(1, epochs);
val_acc_history  = zeros(1, epochs);

best_val_acc       = 0;
no_improve_counter = 0;
best_params        = struct();

num_train_eff = num_train;
num_batches   = floor(num_train_eff / batch_size);

% —————————————
% 4) Loop sulle epoche
% —————————————
for epoch = 1:epochs
    % 4.1) Aggiorno learning rate (decay a epoca 26)
    if epoch == 26
        lr = base_lr * 0.1;
    else
        lr = base_lr;
    end

    % 4.2) Shuffle mini‐batch
    perm = randperm(num_train_eff);
    total_loss = 0;
    correct    = 0;

    % 4.3) Loop sui mini‐batch
    for b = 1:num_batches
        idx_b = perm((b-1)*batch_size + (1:batch_size));
        Xb = imgs_train(:, :, idx_b);   % [img_h×img_w×batch_size]
        Yb = Y_train(:, idx_b);         % [num_classes×batch_size]
        Nb = size(Xb, 3);

        % Inizializzo gradienti accumulati
        dW1  = zeros(size(filters1)); db1  = zeros(size(bias1));
        dW2  = zeros(size(filters2)); db2  = zeros(size(bias2));
        dW3  = zeros(size(filters3)); db3  = zeros(size(bias3));
        dWfc = zeros(size(W_fc));      dbfc = zeros(size(b_fc));

        % — Forwards + Backwards su ogni esempio del mini‐batch —
        for i_b = 1:Nb
            x      = Xb(:, :, i_b);     % [img_h×img_w]
            y_true = Yb(:, i_b);        % [num_classes×1]

            % == Forward pass ==
            [c1, cache1] = conv2d_forward(x, filters1, bias1);
            r1 = max(c1, 0);  % ReLU
            [p1, mask1] = maxpool2d_forward(r1, pool);

            [c2, cache2] = conv2d_forward_multi(p1, filters2, bias2);
            r2 = max(c2, 0);
            [p2, mask2] = maxpool2d_forward(r2, pool);

            [c3, cache3] = conv2d_forward_multi(p2, filters3, bias3);
            r3 = max(c3, 0);
            [p3, mask3] = maxpool2d_forward(r3, pool);

            x_fc = reshape(p3, [], 1);                  % [fc_input_size×1]
            dropout_mask = (rand(size(x_fc)) > dropout_rate);
            x_fc_do = x_fc .* dropout_mask / (1 - dropout_rate);

            z_fc = W_fc * x_fc_do + b_fc;                % [num_classes×1]
            z_fc = z_fc - max(z_fc);
            expz = exp(z_fc);
            y_pred = expz / sum(expz);                  % softmax

            loss = -sum(y_true .* log(y_pred + 1e-12));
            total_loss = total_loss + loss;
            [~, pl] = max(y_pred); [~, tl] = max(y_true);
            if pl == tl, correct = correct + 1; end

            % == Backward pass ==
            dz_fc  = y_pred - y_true;                    % [num_classes×1]
            dWfc_b = dz_fc * x_fc_do';                   % [num_classes×fc_input_size]
            dbfc_b = dz_fc;                              % [num_classes×1]

            dx_fc_do = W_fc' * dz_fc;                    % [fc_input_size×1]
            dx_fc    = dx_fc_do .* dropout_mask / (1 - dropout_rate);

            dp3 = reshape(dx_fc, size(p3));              % [pool3_h×pool3_w×n_f3]
            dr3 = maxpool2d_backward(dp3, mask3);
            dc3 = dr3 .* (c3 > 0);
            [dp2, dW3_b, db3_b] = conv2d_backward_multi(p2, filters3, dc3, cache3);

            dr2 = maxpool2d_backward(dp2, mask2);
            dc2 = dr2 .* (c2 > 0);
            [dp1, dW2_b, db2_b] = conv2d_backward_multi(p1, filters2, dc2, cache2);

            dr1 = maxpool2d_backward(dp1, mask1);
            dc1 = dr1 .* (c1 > 0);
            [~, dW1_b, db1_b] = conv2d_backward(x, filters1, dc1, cache1);

            % Accumulo gradienti
            dWfc = dWfc + dWfc_b;        dbfc = dbfc + dbfc_b;
            dW3  = dW3  + dW3_b;         db3  = db3  + db3_b;
            dW2  = dW2  + dW2_b;         db2  = db2  + db2_b;
            dW1  = dW1  + dW1_b;         db1  = db1  + db1_b;
        end  % for i_b

        % == Update dei pesi (mini‐batch) ==
        filters1 = filters1 - lr * (dW1 / Nb + lambda * filters1);
        bias1    = bias1    - lr * (db1  / Nb);
        filters2 = filters2 - lr * (dW2 / Nb + lambda * filters2);
        bias2    = bias2    - lr * (db2  / Nb);
        filters3 = filters3 - lr * (dW3 / Nb + lambda * filters3);
        bias3    = bias3    - lr * (db3  / Nb);

        W_fc = W_fc - lr * (dWfc / Nb + lambda * W_fc);
        b_fc = b_fc - lr * (dbfc / Nb);
    end  % for b

    % 4.4) Salvo training loss / acc
    loss_history(epoch) = total_loss / num_train_eff;
    acc_history(epoch)  = correct / num_train_eff * 100;

    % 4.5) Validation (solo forward)
    N_val = size(imgs_val, 3);
    correct_val = 0;
    val_loss = 0;
    for j = 1:N_val
        x      = imgs_val(:, :, j);
        y_true = Y_val(:, j);

        [c1, ~] = conv2d_forward(x, filters1, bias1); r1 = max(c1,0);
        [p1, ~] = maxpool2d_forward(r1, pool);
        [c2, ~] = conv2d_forward_multi(p1, filters2, bias2); r2 = max(c2,0);
        [p2, ~] = maxpool2d_forward(r2, pool);
        [c3, ~] = conv2d_forward_multi(p2, filters3, bias3); r3 = max(c3,0);
        [p3, ~] = maxpool2d_forward(r3, pool);

        x_fc = reshape(p3, [], 1);
        z_fc = W_fc * x_fc + b_fc;
        z_fc = z_fc - max(z_fc);
        expz = exp(z_fc);
        y_pred = expz / sum(expz);

        loss_v = -sum(y_true .* log(y_pred + 1e-12));
        val_loss = val_loss + loss_v;
        [~, pl] = max(y_pred); [~, tl] = max(y_true);
        if pl == tl, correct_val = correct_val + 1; end
    end

    val_loss_history(epoch) = val_loss / N_val;
    val_acc_history(epoch)  = correct_val / N_val * 100;

    fprintf('[EPOCA %2d] Train Loss=%.4f Acc=%.2f%% | Val Loss=%.4f Acc=%.2f%%\n', ...
            epoch, loss_history(epoch), acc_history(epoch), ...
            val_loss_history(epoch), val_acc_history(epoch));

    % 4.6) Early stopping
    if val_acc_history(epoch) > best_val_acc
        best_val_acc = val_acc_history(epoch);
        best_params.filters1 = filters1; best_params.bias1    = bias1;
        best_params.filters2 = filters2; best_params.bias2    = bias2;
        best_params.filters3 = filters3; best_params.bias3    = bias3;
        best_params.W_fc     = W_fc;     best_params.b_fc     = b_fc;
        best_params.pool     = pool;
        no_improve_counter   = 0;
    else
        no_improve_counter = no_improve_counter + 1;
        if no_improve_counter >= patience
            fprintf('Early stopping: non migliora validazione da %d epoche!\n', patience);
            break;
        end
    end
end  % for epoch

% Se ci siamo fermati in anticipo, riduco gli array storici
actual_epochs = epoch;
loss_history     = loss_history(1:actual_epochs);
acc_history      = acc_history(1:actual_epochs);
val_loss_history = val_loss_history(1:actual_epochs);
val_acc_history  = val_acc_history(1:actual_epochs);

% 5) Output dei pesi migliori (early stopping)
params = best_params;

end
