function [out, mask] = maxpool2d_forward(X, pool)
[H, W, F] = size(X);
out = zeros(floor(H/pool), floor(W/pool), F);
mask = cell(1,F);
for f=1:F
    [out(:,:,f), mask{f}] = maxpool_single(X(:,:,f), pool);
end
end