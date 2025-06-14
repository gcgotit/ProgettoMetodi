clear; clc; close all;

% Parametri immagine
img_rows = 112;
img_cols = 92;

% Percorso alla cartella del dataset
dataset_path = '../ORL_dataset';

% Lista dei file .jpg
file_list = dir(fullfile(dataset_path, '*.jpg'));
total_imgs = length(file_list);

% Ordiniamo per nome (es: 1_1.jpg, 2_1.jpg, ..., 410_41.jpg)
[~, idx] = sort({file_list.name});
file_list = file_list(idx);

% Inizializziamo matrice A e labels
A = zeros(img_rows * img_cols, total_imgs);
labels = zeros(1, total_imgs);

for i = 1:total_imgs
    filename = file_list(i).name;
    full_path = fullfile(dataset_path, filename);

    % Estraiamo il numero del soggetto dal nome del file
    parts = split(filename, {'_', '.'});  % es: '21_3.jpg' → {'21','3','jpg'}
    subject_num = str2double(parts{2});   % seconda parte è il soggetto

    % Leggiamo l'immagine
    I = imread(full_path);

    % Se RGB → scala di grigi
    if size(I, 3) == 3
        I = rgb2gray(I);
    end

    % Ridimensioniamo 
    I = imresize(I, [img_rows, img_cols]);

    % Vettoriziamo e memorizza
    A(:, i) = double(I(:));
    labels(i) = subject_num;
end

save('volti_dataset_ORL.mat', 'A', 'labels');

% Visualiziamo 9 immagini per controllo
figure;
for i = 1:9
    subplot(3,3,i);
    imshow(reshape(A(:,i), img_rows, img_cols), []);
    title(['Soggetto ', num2str(labels(i))]);
end

disp(['Caricate ', num2str(total_imgs), ' immagini con successo.']);
