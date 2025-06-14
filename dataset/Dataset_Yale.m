%% ===========================================
%% 0) PREPARAZIONE YALE_FACE_DATASET → volti_dataset.mat
%% ===========================================
clear; clc; close all;

% Parametri immagine (come in ORL)
img_rows = 112;  
img_cols = 92;

% Cartella del dataset Yale (modifica se serve)
dataset_path = '../Yale_face_dataset';

% Lista di tutti i file “subjectXX.*” (esclude Readme e cartelle)
d = dir(fullfile(dataset_path,'subject*.*'));
d = d(~[d.isdir]);  
d = d(~endsWith({d.name},'.txt'));  

total_imgs = numel(d);
A      = zeros(img_rows*img_cols, total_imgs);
labels = zeros(1, total_imgs);

for i = 1:total_imgs
    fname = d(i).name;
    fullpath = fullfile(d(i).folder, fname);
    
    % Estrai il numero di soggetto: 'subject01.…' → 1
    tok = regexp(fname, 'subject(\d+)', 'tokens', 'once');
    subj = str2double(tok{1});            
    
    % Leggi e porta in grayscale
    I = imread(fullpath);
    if size(I,3)==3
        I = rgb2gray(I);
    end
    
    % Ridimensiona, vettorializza
    I = imresize(I, [img_rows, img_cols]);
    A(:,i)     = double(I(:));  
    labels(i)  = subj;
end

% Normalizza (facoltativo: in ORL dividevi per 255 più avanti)
A = A / 255;

% Salva nello stesso formato di ORL
save('volti_dataset_Yale.mat','A','labels');
fprintf('Caricate %d immagini di %d soggetti da Yale.\n', total_imgs, max(labels));

% Controllo rapido
figure;
for i=1:9
    subplot(3,3,i);
    imshow(reshape(A(:,i), img_rows, img_cols), []);
    title(sprintf('Subj %d', labels(i)));
end
