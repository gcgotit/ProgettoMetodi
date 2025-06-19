function [imgs_aug, y_aug] = augment_data(imgs_base, y_base)
% augment_data   Data augmentation “×15” su immagini grayscale (Image Processing Toolbox)
%   [imgs_aug, y_aug] = augment_data(imgs_base, y_base)
%
%   Per ogni immagine: originale, 4 traslazioni, 4 rotazioni, 2 zoom, flip, 2 luminosità, 1 rumore


[H, W, N_base] = size(imgs_base);
aug_per_image  = 15;
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

    % 2) Traslazioni ±2 px orizz, ±1 px vert (4 var.)
    tforms = [2,0; -2,0; 0,1; 0,-1];
    for t = 1:size(tforms,1)
        shifted = imtranslate(x, tforms(t,:), 'FillValues', 0);
        cnt = cnt + 1;
        imgs_aug(:,:,cnt) = shifted;
        y_aug(cnt) = lbl;
    end

    % 3) Rotazioni ±5°, ±10° (4 var.)
    angles = [5, -5, 10, -10];
    for ang = angles
        rotated = imrotate(x, ang, 'bilinear', 'crop');
        cnt = cnt + 1;
        imgs_aug(:,:,cnt) = rotated;
        y_aug(cnt) = lbl;
    end

    % 4) Zoom in +10%, Zoom out -10% (2 var.)
    z_factors = [1.10, 0.90];
    for z = z_factors
        zoomed = imresize(x, z, 'bilinear');
        % Crop or pad to restore size
        if z > 1
            % Crop the central part
            start_row = floor((size(zoomed,1)-H)/2)+1;
            start_col = floor((size(zoomed,2)-W)/2)+1;
            zoomed = zoomed(start_row:start_row+H-1, start_col:start_col+W-1);
        else
            % Pad to original size
            zoomed = padarray(zoomed, [ceil((H-size(zoomed,1))/2), ceil((W-size(zoomed,2))/2)], 0, 'both');
            zoomed = zoomed(1:H,1:W); % safety crop
        end
        cnt = cnt + 1;
        imgs_aug(:,:,cnt) = zoomed;
        y_aug(cnt) = lbl;
    end

    % 5) Flip orizzontale
    flipped = fliplr(x);
    cnt = cnt + 1;
    imgs_aug(:,:,cnt) = flipped;
    y_aug(cnt) = lbl;

    % 6) Luminosità ±15% (2 var.)
    x_bright = min(x + 0.15, 1);
    x_dark   = max(x - 0.15, 0);
    cnt = cnt + 1; imgs_aug(:,:,cnt) = x_bright; y_aug(cnt) = lbl;
    cnt = cnt + 1; imgs_aug(:,:,cnt) = x_dark;   y_aug(cnt) = lbl;

    % 7) Rumore gaussiano (1 var.)
    x_noise = x + 0.05*randn(size(x));
    x_noise = min(max(x_noise,0),1);
    cnt = cnt + 1; imgs_aug(:,:,cnt) = x_noise; y_aug(cnt) = lbl;
end

assert(cnt == N_aug, 'augment_data: il numero di esempi augmentato non corrisponde.');
end
