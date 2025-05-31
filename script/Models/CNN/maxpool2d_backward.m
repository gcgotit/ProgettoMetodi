function dx = maxpool2d_backward(dout, mask)
% maxpool2d_backward   Backprop del max‐pool 2D senza errori di indici non interi
%
%  Il forward è:
%     [pool_out, mask] = maxpool2d_forward(input, pool)
%  dove:
%     – input: [H × W × F]
%     – pool: dimensione (es. 2) → pool_out: [H_p × W_p × F]
%     – mask: cell array di dimensione {1×F}, ciascuna maschera [H × W]
%          con 1 nel pixel del max per ogni patch pool×pool + 0 altrove.
%
%  Questa backward prende:
%     • dout: [H_p × W_p × F] = gradienti del loss su pool_out
%     • mask: cell array {1×F} di maschere [H × W]
%  Restituisce:
%     • dx: [H × W × F], gradienti su input del maxpool
%
%  Calcola pool_h = floor(H/H_p), pool_w = floor(W/W_p) per evitare
%  frazioni. Suppone che H_p = floor(H/pool) e W_p = floor(W/pool).

F = numel(mask);
[H, W]   = size(mask{1});        % dimensione originale di ogni feature map
[H_p, W_p] = size(dout(:,:,1));   % dimensione dopo pooling

pool_h = floor(H / H_p);
pool_w = floor(W / W_p);

dx = zeros(H, W, F);

for f = 1:F
    msk = mask{f};        % [H × W], maschera binaria
    d   = dout(:,:,f);    % [H_p × W_p], gradienti sul pool out

    for i = 1:H_p
        for j = 1:W_p
            row_start = (i-1)*pool_h + 1;
            row_end   = i * pool_h;
            col_start = (j-1)*pool_w + 1;
            col_end   = j * pool_w;

            block = msk(row_start:row_end, col_start:col_end);
            % “Spargi” d(i,j) solo nella posizione del max (block==1)
            dx(row_start:row_end, col_start:col_end, f) = block * d(i,j);
        end
    end
end

end
