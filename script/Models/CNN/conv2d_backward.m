function [dx, dfilters, dbias] = conv2d_backward(X, filters, dout, mask)
% Backward di un layer conv + relu
f = size(filters,1); F = size(filters,3);
dx = zeros(size(X));
dfilters = zeros(size(filters));
dbias = zeros(F,1);
for i=1:F
    dZ = dout(:,:,i) .* mask{i};  % relu
    dbias(i) = sum(dZ(:));
    dfilters(:,:,i) = conv2(X, rot90(dZ,2), 'valid');
    dx = dx + conv2(dZ, filters(:,:,i), 'full');
end
end