function [out, idx] = maxpool_single(mat, pool)
s = size(mat);
out = zeros(floor(s(1)/pool), floor(s(2)/pool));
idx = zeros(size(mat));
for i = 1:size(out,1)
    for j = 1:size(out,2)
        region = mat((i-1)*pool+1:i*pool, (j-1)*pool+1:j*pool);
        [maxval, ind] = max(region(:));
        out(i,j) = maxval;
        [a,b] = ind2sub([pool,pool], ind);
        idx((i-1)*pool+a, (j-1)*pool+b) = 1;
    end
end
end