function dx = maxpool2d_backward(dout, mask)
    % In questa versione calcoliamo pool_h e pool_w con floor(H/H_p) e floor(W/W_p),
    % in modo da non incorrere in dimensioni non intere dovute al floor nella fase forward.

    F = numel(mask);
    [H, W]   = size(mask{1});        % dimensione originale di ogni feature map
    [H_p, W_p] = size(dout(:,:,1));   % dimensione della feature map dopo pool

    % Calcolo pool size come floor della divisione
    pool_h = floor(H / H_p);
    pool_w = floor(W / W_p);

    % Inizializzo uscita con zeri
    dx = zeros(H, W, F);

    for f = 1:F
        msk = mask{f};        % [H x W] = maschera binaria
        d   = dout(:,:,f);    % [H_p x W_p] = gradiente sul pool

        for i = 1:H_p
            for j = 1:W_p
                row_start = (i-1)*pool_h + 1;
                row_end   = i * pool_h;
                col_start = (j-1)*pool_w + 1;
                col_end   = j * pool_w;

                block = msk(row_start:row_end, col_start:col_end);
                dx(row_start:row_end, col_start:col_end, f) = block * d(i,j);
            end
        end
    end
end
