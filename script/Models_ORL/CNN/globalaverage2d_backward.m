function dX = globalaverage2d_backward(dout, cache)
% dout: [F x 1], cache = [H, W]
H = cache(1); W = cache(2);
F = numel(dout);
dX = zeros(H, W, F);
for f = 1:F
    dX(:,:,f) = dout(f) / (H*W);
end