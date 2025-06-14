function [out, cache] = conv2d_forward_multi(X, filters, bias)
% X: [H x W x F1], filters: [f x f x F2 x F1], bias: [F2 x 1]
f = size(filters,1); F2 = size(filters,3); F1 = size(filters,4);
H = size(X,1); W = size(X,2);
out = zeros(H-f+1, W-f+1, F2);
cache = cell(1,F2);
for i=1:F2
    acc = zeros(H-f+1, W-f+1);
    for j=1:F1
        acc = acc + conv2(X(:,:,j), rot90(filters(:,:,i,j),2), 'valid');
    end
    out(:,:,i) = max(0, acc + bias(i));
    cache{i} = (out(:,:,i) > 0);
end
end
