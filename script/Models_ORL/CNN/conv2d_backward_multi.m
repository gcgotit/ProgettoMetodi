function [dx, dfilters, dbias] = conv2d_backward_multi(X, filters, dout, mask)
f = size(filters,1); F2 = size(filters,3); F1 = size(filters,4);
[H, W, ~] = size(X);
dx = zeros(size(X));
dfilters = zeros(size(filters));
dbias = zeros(F2,1);
for i=1:F2
    dZ = dout(:,:,i) .* mask{i};  % relu
    dbias(i) = sum(dZ(:));
    for j=1:F1
        dfilters(:,:,i,j) = conv2(X(:,:,j), rot90(dZ,2), 'valid');
        dx(:,:,j) = dx(:,:,j) + conv2(dZ, filters(:,:,i,j), 'full');
    end
end
end
