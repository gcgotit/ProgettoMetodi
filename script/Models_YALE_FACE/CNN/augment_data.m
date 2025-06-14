function [imgs_aug, y_aug] = augment_data(imgs_base, y_base)
% augment_data   Esegue data augmentation “×12” su un set di immagini grayscale.
%
%   [imgs_aug, y_aug] = augment_data(imgs_base, y_base)
%
%   Input:
%     • imgs_base: array [H × W × N_base] di immagini originali (0–1)
%     • y_base:    vettore [1 × N_base] di etichette corrispondenti
%
%   Output:
%     • imgs_aug: array [H × W × (12·N_base)] con immagini originali +
%                 traslation, rotazioni, zoom, flip  (totale 12 per immagine)
%     • y_aug:    vettore [1 × (12·N_base)] di etichette replicate
%
%   La routine usa:
%     1) Originale
%     2) Traslazioni ±2 px orizzontali, ±1 px verticali  (4 var.)
%     3) Rotazioni ±5°, ±10°                            (4 var.)
%     4) Zoom in/out ±10%                               (2 var.)
%     5) Flip orizzontale                               (1 var.)
%
%   Autore: [Tuo Nome]
%   Data:   [gg/mm/aaaa]

[H, W, N_base] = size(imgs_base);
aug_per_image  = 12;
N_aug = N_base * aug_per_image;

imgs_aug = zeros(H, W, N_aug);
y_aug    = zeros(1, N_aug);
cnt = 0;

for i = 1:N_base
    x   = imgs_base(:, :, i);
    lbl = y_base(i);

    % 1) Originale
    cnt = cnt + 1;
    imgs_aug(:, :, cnt) = x;
    y_aug(cnt) = lbl;

    % 2) Traslazioni ±2 orizzontali, ±1 verticali (4)
    x_sh1 = imtranslate(x, [2,  0], 'FillValues', 0);
    x_sh2 = imtranslate(x, [-2, 0], 'FillValues', 0);
    x_sh3 = imtranslate(x, [0,  1], 'FillValues', 0);
    x_sh4 = imtranslate(x, [0, -1], 'FillValues', 0);
    for xx = {x_sh1, x_sh2, x_sh3, x_sh4}
        cnt = cnt + 1;
        imgs_aug(:, :, cnt) = xx{1};
        y_aug(cnt) = lbl;
    end

    % 3) Rotazioni ±5°, ±10° (4)
    x_r5p  = imrotate(x,  5,  'bilinear', 'crop');
    x_r5n  = imrotate(x, -5,  'bilinear', 'crop');
    x_r10p = imrotate(x, 10, 'bilinear', 'crop');
    x_r10n = imrotate(x, -10, 'bilinear', 'crop');
    for xx = {x_r5p, x_r5n, x_r10p, x_r10n}
        cnt = cnt + 1;
        imgs_aug(:, :, cnt) = xx{1};
        y_aug(cnt) = lbl;
    end

    % 4) Zoom in +10%, Zoom out -10% (2)
    tmp_in  = imresize(x, 1.10);   x_zi = imresize(tmp_in, [H, W]);
    tmp_out = imresize(x, 0.90);   x_zo = imresize(tmp_out, [H, W]);
    for xx = {x_zi, x_zo}
        cnt = cnt + 1;
        imgs_aug(:, :, cnt) = xx{1};
        y_aug(cnt) = lbl;
    end

    % 5) Flip orizzontale (1)
    x_f = fliplr(x);
    cnt = cnt + 1;
    imgs_aug(:, :, cnt) = x_f;
    y_aug(cnt) = lbl;
end

assert(cnt == N_aug, 'augment_data: il numero di esempi augmentato non corrisponde.');

end
