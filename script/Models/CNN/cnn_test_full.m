function acc = cnn_test_full(imgs_test, y_test, params)
% cnn_test_full   Valuta una CNN (3×conv+pool+FC+softmax) SENZA dropout e senza L2
%
%  Accetta:
%     • imgs_test: [img_h × img_w × N_test] di immagini di test
%     • y_test:    [1 × N_test] etichette (1..num_classes)
%     • params:    struct contenente i pesi {filters1,bias1,filters2,bias2,filters3,bias3,W_fc,b_fc,pool}
%  Restituisce:
%     • acc:       accuracy (%) sul test set
%
%  Autore: [Tuo Nome]
%  Data:   [gg/mm/aaaa]

img_h       = size(imgs_test, 1);
img_w       = size(imgs_test, 2);
num_test    = size(imgs_test, 3);
num_classes = length(params.b_fc);
pool        = params.pool;

f1 = size(params.filters1, 1);
f2 = size(params.filters2, 1);
f3 = size(params.filters3, 1);

% Calcolo dimensioni feature map per FC
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

fc_input_size = pool3_h * pool3_w * size(params.filters3, 3);

correct = 0;
for i = 1:num_test
    x = imgs_test(:, :, i);

    [c1, ~] = conv2d_forward(x, params.filters1, params.bias1);
    r1 = max(c1, 0);
    [p1, ~] = maxpool2d_forward(r1, pool);

    [c2, ~] = conv2d_forward_multi(p1, params.filters2, params.bias2);
    r2 = max(c2, 0);
    [p2, ~] = maxpool2d_forward(r2, pool);

    [c3, ~] = conv2d_forward_multi(p2, params.filters3, params.bias3);
    r3 = max(c3, 0);
    [p3, ~] = maxpool2d_forward(r3, pool);

    x_fc = reshape(p3, [], 1);  % [fc_input_size×1]
    z_fc = params.W_fc * x_fc + params.b_fc;  % [num_classes×1]
    z_fc = z_fc - max(z_fc);
    expz = exp(z_fc);
    y_pred = expz / sum(expz);

    [~, pl] = max(y_pred);
    if pl == y_test(i), correct = correct + 1; end
end

acc = correct / num_test * 100;

end
