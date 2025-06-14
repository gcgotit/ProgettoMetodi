function [out, cache] = conv2d_forward(X, filters, bias)
% X: [H x W], filters: [f x f x F], bias: [F x 1]
f = size(filters,1); F = size(filters,3);
H = size(X,1); W = size(X,2);
out = zeros(H-f+1, W-f+1, F);
cache = cell(1,F);
for i=1:F
    out(:,:,i) = max(0, conv2(X, rot90(filters(:,:,i),2), 'valid') + bias(i));
    cache{i} = (out(:,:,i) > 0);
end
end