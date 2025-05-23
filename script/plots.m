%% Assumiamo che in workspace ci siano:
% S (m×n) con valori singolari in S(1:k,1:k)
% loss_hist (1×epochs)
% k_list = [10,20,40,60,80,100];
% accuracies = zeros(size(k_list));
% times_total = zeros(size(k_list));

% 1) Scree plot
singulars = diag(S);
figure;
semilogy(singulars(1:50), 'o-','LineWidth',1.5);
xlabel('Index i'); ylabel('\sigma_i (log scale)');
title('Scree Plot: Prime 50 Valori Singolari');
grid on;

% 2) Cumulative explained variance
explained = cumsum(singulars.^2) / sum(singulars.^2);
figure;
plot(explained(1:100), 's-','LineWidth',1.5);
xlabel('Numero componenti k'); ylabel('Varianza spiegata cumulativa');
title('Varianza spiegata in funzione di k');
grid on;
ylim([0 1]); yticks(0:0.1:1);

% 3–4) Time & Accuracy vs k
for idx=1:length(k_list)
    k = k_list(idx);
    tic;
    % ricomputate U_k e projections
    U_k = U(:,1:k);
    proj = U_k'*(A - mean_face);
    % split, train e test MLP come in nn_svd.m
    % (riutilizzare mlp_gradient_descent per training e predizione)
    % [W1,b1,W2,b2,~] = mlp_gradient_descent(...);
    % y_pred = predict_mlp(W1,b1,W2,b2, proj, train_idx, test_idx);
    % accuracies(idx) = mean(y_pred==y_test)*100;
    times_total(idx) = toc;
end

figure;
yyaxis left
plot(k_list, times_total, 'o-','LineWidth',1.5);
ylabel('Tempo totale (s)');
yyaxis right
plot(k_list, accuracies, 's--','LineWidth',1.5);
ylabel('Accuracy (%)');
xlabel('Numero di Eigenfaces k');
title('Tempo e Accuracy vs k');
legend('Tempo','Accuracy','Location','best');
grid on;

% 5) Loss curve
figure;
plot(1:length(loss_hist), loss_hist, 'LineWidth',1.5);
xlabel('Epoca'); ylabel('Cross-Entropy Loss');
title('Training Loss Curve MLP');
grid on;

% 6) Confusion matrix
% Calcola y_pred una volta per il k ottimale (es. k=80)
k_opt = 80;
U_k = U(:,1:k_opt);
proj = U_k'*(A - mean_face);
% [W1,b1,W2,b2,~] = mlp_gradient_descent(X_train, Y_train, ...);
% [~, y_pred] = mlp_predict(W1,b1,W2,b2, proj, test_idx);
C = confusionmat(y_test, y_pred);
figure;
imagesc(C);
colormap(flipud(gray));
colorbar;
xlabel('Predicted Class'); ylabel('True Class');
title(sprintf('Confusion Matrix (k = %d): Accuracy = %.2f%%', k_opt, mean(y_pred==y_test)*100));