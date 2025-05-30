function dx = maxpool2d_backward(dout, mask)
    % dout: [pooled_h x pooled_w x F] -- gradienti dalla layer successiva
    % mask: cell array di maschere (ognuna size [H x W]), una per ogni feature map
    [H, W, F] = size(mask{1});    % size originale della feature map
    dx = zeros(H, W, numel(mask)); % output: stesso size dell'input a maxpool

    for f = 1:numel(mask)
        msk = mask{f}; % [H x W]
        d = dout(:,:,f); % [pooled_h x pooled_w]
        % Per ogni regione di pooling, metti il gradiente d(i,j) nel punto max
        pool = sqrt(numel(msk)/numel(d)); % pool size dedotto (di solito 2)
        for i = 1:size(d,1)
            for j = 1:size(d,2)
                % Trova il blocco nella maschera
                block = msk((i-1)*pool+1:i*pool, (j-1)*pool+1:j*pool);
                % "Spargi" il gradiente solo dove la mask è 1
                dx((i-1)*pool+1:i*pool, (j-1)*pool+1:j*pool, f) = block * d(i,j);
            end
        end
    end
end

