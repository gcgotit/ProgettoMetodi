function [params, loss_history, acc_history] = cnn2_train_full(imgs, Y_train, epochs, lr)
% imgs: [H x W x N] (immagini), Y_train: [num_class x N]

[img_h, img_w, num_train] = size(imgs);
num_classes = size(Y_train,1);

% Parametri architettura
f1 = 5; n_f1 = 8;
f2 = 3; n_f2 = 16;
pool = 2;
fc_input_size = (((img_h-f1+1)/pool - f2+1)/pool) * (((img_w-f1+1)/pool - f2+1)/pool) * n_f2;
fc_input_size = floor(fc_input_size);

% Inizializzazione pesi
rng(0);
filters1 = randn(f1, f1, n_f1) * 0.1;   bias1 = zeros(n_f1,1);
filters2 = randn(f2, f2, n_f2, n_f1) * 0.1; bias2 = zeros(n_f2,1);
W_fc = randn(num_classes, fc_input_size) * sqrt(2/fc_input_size);
b_fc = zeros(num_classes,1);

loss_history = zeros(1, epochs);
acc_history  = zeros(1, epochs);

for epoch = 1:epochs
    total_loss = 0;
    correct = 0;
    for i = 1:num_train
        % === FORWARD ===
        x = imgs(:,:,i);
        y_true = Y_train(:,i);

        [c1, cache1] = conv2d_forward(x, filters1, bias1);     % Conv1 + ReLU
        [p1, p1_mask] = maxpool2d_forward(c1, pool);           % Pool1
        [c2, cache2] = conv2d_forward_multi(p1, filters2, bias2); % Conv2 + ReLU
        [p2, p2_mask] = maxpool2d_forward(c2, pool);           % Pool2

        x_fc = reshape(p2, [], 1);  % flatten
        
        z_fc = W_fc * x_fc + b_fc;  % FC
        z_fc = z_fc - max(z_fc);    % stabilize
        expz = exp(z_fc);           % softmax
        y_pred = expz / sum(expz);

        loss = -sum(y_true .* log(y_pred + 1e-10));
        total_loss = total_loss + loss;

        % === Accuracy (predizione) ===
        [~, pred_label] = max(y_pred);
        [~, true_label] = max(y_true);
        if pred_label == true_label
            correct = correct + 1;
        end

        % === BACKPROP ===
        dz_fc = y_pred - y_true;
        dW_fc = dz_fc * x_fc';    db_fc = dz_fc;
        dx_fc = W_fc' * dz_fc;    % grad per flatten

        % Backprop pool2
        dp2 = reshape(dx_fc, size(p2));
        dc2 = maxpool2d_backward(dp2, p2_mask);

        % Backprop conv2
        [dp1, dfilters2, dbias2] = conv2d_backward_multi(p1, filters2, dc2, cache2);

        % Backprop pool1
        dc1 = maxpool2d_backward(dp1, p1_mask);

        % Backprop conv1
        [~, dfilters1, dbias1] = conv2d_backward(x, filters1, dc1, cache1);

        % === GRADIENT UPDATE ===
        W_fc     = W_fc - lr * dW_fc;
        b_fc     = b_fc - lr * db_fc;
        filters2 = filters2 - lr * dfilters2;
        bias2    = bias2    - lr * dbias2;
        filters1 = filters1 - lr * dfilters1;
        bias1    = bias1    - lr * dbias1;
    end
    loss_history(epoch) = total_loss / num_train;
    acc_history(epoch)  = correct / num_train * 100;
    if mod(epoch,2)==0
        fprintf('[CNN] Epoca %d/%d, Loss=%.4f, Acc=%.2f%%\n', epoch, epochs, ...
                loss_history(epoch), acc_history(epoch));
    end
end

params = struct('filters1',filters1,'bias1',bias1, ...
                'filters2',filters2,'bias2',bias2, ...
                'W_fc',W_fc,'b_fc',b_fc, ...
                'pool',pool);

end
