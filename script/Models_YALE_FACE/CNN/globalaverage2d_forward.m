function [out, cache] = globalaverage2d_forward(X)
% X: [H x W x F]
[H, W, F] = size(X);
out = squeeze(mean(mean(X,1),2));  % [F x 1]
cache = [H, W];
end