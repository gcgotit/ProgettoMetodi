function [filters1, bias1, filters2, bias2, W_fc, b_fc, loss_history] = ...
    cnn_two_layers(imgs_train, Y_train, epochs, learning_rate)
% imgs_train: [img_h x img_w x Ntrain]
% Y_train:    [num_classes x Ntrain]
% Returns: filtri, bias, pesi FC e storia della loss

img_h = size(imgs_train,1); img_w = size(imgs_train,2);
num_train = size(imgs_train,3);
num_classes = size(Y_train,1);

filter_size1 = 5; num_filters1 = 8;
filter_size2 = 3; num_filters2 = 16;
pool_size = 2;

% Layer 1
conv1_h = img_h - filter_size1 + 1;
conv1_w = img_w - filter_size1 + 1;
pool1_h = floor(conv1_h/pool_size);
pool1_w = floor(conv1_w/pool_size);

% Layer 2
conv2_h = pool1_h - filter_size2 + 1;
conv2_w = pool1_w - filter_size2 + 1;
pool2_h = floor(conv2_h/pool_size);
pool2_w = floor(conv2_w/pool_size);

fc_input_size = pool2_h * pool2_w * num_filters2;

% Inizializzazione pesi
rng(0);
filters1 = randn(filter_size1, filter_size1, num_filters1) * 0.1;
bias1 = zeros(num_filters1,1);
filters2 = randn(filter_size2, filter_size2, num_filters2, num_filters1) * 0.1;
bias2 = zeros(num_filters2,1);
W_fc = randn(num_classes, fc_input_size) * sqrt(2/fc_input_size);
b_fc = zeros(num_classes,1);

loss_history = zeros(1, epochs);

for epoch = 1:epochs
    total_loss = 0;
    for i = 1:num_train
        x_img = imgs_train(:,:,i);
        y_true = Y_train(:,i);

        % ==== FORWARD ====
        % Conv1 + ReLU
        conv1 = zeros(conv1_h, conv1_w, num_filters1);
        for f = 1:num_filters1
            conv1(:,:,f) = conv2(x_img, rot90(filters1(:,:,f),2), 'valid') + bias1(f);
            conv1(:,:,f) = max(0, conv1(:,:,f));
        end
        pool1 = zeros(pool1_h, pool1_w, num_filters1);
        for f = 1:num_filters1
            pool1(:,:,f) = maxpool2d(conv1(:,:,f), pool_size);
        end

        % Conv2 + ReLU (su ogni feature map di pool1)
        conv2_out = zeros(conv2_h, conv2_w, num_filters2);
        for f2 = 1:num_filters2
            acc = zeros(conv2_h, conv2_w);
            for f1 = 1:num_filters1
                acc = acc + conv2(pool1(:,:,f1), rot90(filters2(:,:,f2,f1),2), 'valid');
            end
            conv2_out(:,:,f2) = max(0, acc + bias2(f2));
        end
        pool2 = zeros(pool2_h, pool2_w, num_filters2);
        for f = 1:num_filters2
            pool2(:,:,f) = maxpool2d(conv2_out(:,:,f), pool_size);
        end

        % Flatten
        x_fc = reshape(pool2, [], 1);

        % FC
        z_fc = W_fc * x_fc + b_fc;
        z_fc = z_fc - max(z_fc);
        expz = exp(z_fc);
        y_pred = expz / sum(expz);

        % ==== LOSS ====
        eps = 1e-10;
        loss = -sum(y_true .* log(y_pred + eps));
        total_loss = total_loss + loss;

        % ==== BACKPROP SOLO FC ====
        dz_fc = y_pred - y_true;
        dW_fc = dz_fc * x_fc';
        db_fc = dz_fc;
        W_fc = W_fc - learning_rate * dW_fc;
        b_fc = b_fc - learning_rate * db_fc;
        % (non aggiorniamo i filtri per semplicità)

    end
    loss_history(epoch) = total_loss / num_train;
    if mod(epoch,5)==0
        fprintf('[CNN] Epoch %d/%d, Loss = %.4f\n', epoch, epochs, loss_history(epoch));
    end
end

end

% --- Maxpool2d come prima ---
function out = maxpool2d(mat, pool_size)
    s = size(mat);
    out = zeros(floor(s(1)/pool_size), floor(s(2)/pool_size));
    for i = 1:size(out,1)
        for j = 1:size(out,2)
            region = mat( (i-1)*pool_size+1:i*pool_size, (j-1)*pool_size+1:j*pool_size );
            out(i,j) = max(region(:));
        end
    end
end
