% --------- cnn2_test_full.m ---------
function acc = cnn2_test_full(imgs, y_test, params)
img_h = size(imgs,1); img_w = size(imgs,2);
num_classes = length(params.b_fc);
pool = params.pool;
num_test = size(imgs,3);

f1 = size(params.filters1,1); f2 = size(params.filters2,1);

conv1_h = img_h - f1 + 1;
conv1_w = img_w - f1 + 1;
pool1_h = floor(conv1_h/pool); pool1_w = floor(conv1_w/pool);
conv2_h = pool1_h - f2 + 1; conv2_w = pool1_w - f2 + 1;
pool2_h = floor(conv2_h/pool); pool2_w = floor(conv2_w/pool);

fc_input_size = pool2_h * pool2_w * size(params.filters2,3);

correct = 0;
for i = 1:num_test
    x = imgs(:,:,i);

    % Forward
    [c1, ~] = conv2d_forward(x, params.filters1, params.bias1);
    [p1, ~] = maxpool2d_forward(c1, pool);
    [c2, ~] = conv2d_forward_multi(p1, params.filters2, params.bias2);
    [p2, ~] = maxpool2d_forward(c2, pool);
    x_fc = reshape(p2, [], 1);
    disp(size(x_fc));  % metti sia nel train che nel test!

    z_fc = params.W_fc * x_fc + params.b_fc;
    z_fc = z_fc - max(z_fc);
    expz = exp(z_fc);
    y_pred = expz / sum(expz);
    [~, pred_label] = max(y_pred);
    if pred_label == y_test(i)
        correct = correct + 1;
    end
    disp([pred_label, y_test(i)]);

end
acc = correct / num_test * 100;
end
