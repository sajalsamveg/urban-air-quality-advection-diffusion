clc; clear; close all;

% ── Headless rendering: painters renderer, no Java client ────────────────
set(0, 'DefaultFigureVisible',       'off');
set(0, 'DefaultFigureRenderer',      'painters');
set(0, 'DefaultFigurePaperUnits',    'inches');
set(0, 'DefaultFigurePaperPosition', [0 0 14 5]);

% Helper: save figure without touching the Java graphics client
    function saveFigHeadless(fig, filename, widthIn, heightIn)
        set(fig, 'PaperUnits',    'inches', ...
                 'PaperSize',     [widthIn heightIn], ...
                 'PaperPosition', [0 0 widthIn heightIn], ...
                 'Renderer',      'painters');
        drawnow('expose');                          % flush offscreen queue
        print(fig, filename, '-dpng', '-r220');     % painters — no Java
        close(fig);
        fprintf('  Saved: %s.png\n', filename);
    end

% ── 1. LOAD & FILTER ─────────────────────────────────────────────────────
opts = detectImportOptions('delhi_ncr_hourly_aqi.csv');
opts.VariableNamingRule = 'preserve';
T    = readtable('delhi_ncr_hourly_aqi.csv', opts);

T    = T(strcmp(T.location_name, 'Delhi'), :);
T    = sortrows(T, 'timestamp');

pm25 = double(T.pm25);
N    = numel(pm25);
fprintf('Delhi PM2.5 series — %d hourly samples (%.1f years)\n', N, N/8760);

% ── 2. NaN / NEGATIVE REPAIR ─────────────────────────────────────────────
nanMask = isnan(pm25) | pm25 < 0;
if any(nanMask)
    idx   = (1:N)';
    valid = ~nanMask;
    pm25  = interp1(idx(valid), pm25(valid), idx, 'previous', 'extrap');
    pm25  = max(pm25, 0);
    fprintf('Repaired %d bad samples.\n', sum(nanMask));
else
    fprintf('No missing values detected.\n');
end

% ── 3. Z-SCORE NORMALISATION ─────────────────────────────────────────────
mu     = mean(pm25);
sigma  = std(pm25);
pm25_z = (pm25 - mu) / sigma;
fprintf('PM2.5 — Mean: %.2f  Std: %.2f  Min: %.2f  Max: %.2f µg/m³\n', ...
        mu, sigma, min(pm25), max(pm25));

% ── 4. SLIDING-WINDOW SEQUENCES (48-hour lookback) ───────────────────────
seqLen = 48;
n_seq  = N - seqLen;

X_all  = cell(n_seq, 1);
Y_all  = zeros(n_seq, 1);        % MUST be (n × 1) column vector

for i = 1:n_seq
    X_all{i} = pm25_z(i : i+seqLen-1)';
    Y_all(i)  = pm25_z(i + seqLen);
end

train_n = floor(0.80 * n_seq);
X_train = X_all(1:train_n);         Y_train = Y_all(1:train_n);
X_test  = X_all(train_n+1:end);     Y_test  = Y_all(train_n+1:end);
fprintf('Train: %d sequences  |  Test: %d sequences\n', ...
        train_n, n_seq - train_n);

% ── 5. LSTM ARCHITECTURE ─────────────────────────────────────────────────
layers = [
    sequenceInputLayer(1,  'Name','input',  'Normalization','none')
    lstmLayer(100,         'Name','lstm',   'OutputMode','last')
    dropoutLayer(0.20,     'Name','drop')
    fullyConnectedLayer(1, 'Name','fc')
    regressionLayer(       'Name','output')
];

tr_opts = trainingOptions('adam',                      ...
    'MaxEpochs',             50,                      ...
    'MiniBatchSize',         128,                     ...
    'InitialLearnRate',      1e-3,                    ...
    'LearnRateSchedule',    'piecewise',              ...
    'LearnRateDropFactor',   0.50,                    ...
    'LearnRateDropPeriod',   20,                      ...
    'GradientThreshold',     1,                       ...
    'Shuffle',              'every-epoch',            ...
    'ValidationData',       {X_test, Y_test},         ...
    'ValidationFrequency',   100,                     ...
    'ExecutionEnvironment', 'auto',                   ...
    'Verbose',               true,                    ...
    'Plots',                'none');          % no live-plot window

fprintf('\nTraining LSTM (100 units, Adam, 50 epochs)...\n');
tic;
net = trainNetwork(X_train, Y_train, layers, tr_opts);
fprintf('Training complete in %.1f s.\n\n', toc);

% ── 6. PREDICTION & DE-NORMALISE ─────────────────────────────────────────
Y_pred_z = predict(net, X_test);
Y_actual = Y_test(:)  * sigma + mu;
Y_pred   = max(Y_pred_z(:) * sigma + mu, 0);

% ── 7. METRICS ───────────────────────────────────────────────────────────
errors = Y_actual - Y_pred;
RMSE   = sqrt(mean(errors.^2));
MAE    = mean(abs(errors));
R2     = 1 - sum(errors.^2) / sum((Y_actual - mean(Y_actual)).^2);

% MAPE: exclude near-zero actuals (< 10 µg/m³) — standard air-quality practice
mape_mask = Y_actual >= 10;
MAPE      = mean(abs(errors(mape_mask) ./ Y_actual(mape_mask))) * 100;

fprintf('%s\n  LSTM Benchmarking Results\n', repmat('=',1,48));
fprintf('  RMSE : %7.3f  µg/m³\n',  RMSE);
fprintf('  MAE  : %7.3f  µg/m³\n',  MAE);
fprintf('  MAPE : %7.3f  %%  (samples with actual >= 10 µg/m³)\n', MAPE);
fprintf('  R²   : %7.4f\n%s\n\n', R2, repmat('=',1,48));

% ── 8. COLOUR PALETTE ────────────────────────────────────────────────────
cAct  = [0.122  0.471  0.706];
cPred = [0.839  0.153  0.157];
cSctt = [0.576  0.094  0.639];
cHist = [0.933  0.507  0.193];

t_idx = (1:numel(Y_actual))';

% ─────────────────────────────────────────────────────────────────────────
% FIGURE A — Time-series: Actual vs Predicted
% ─────────────────────────────────────────────────────────────────────────
fA = figure('Color','w','Position',[0 0 1400 420]);
fill([t_idx; flipud(t_idx)], ...
     [Y_actual+RMSE; flipud(Y_actual-RMSE)], ...
     cAct, 'FaceAlpha',0.07,'EdgeColor','none','HandleVisibility','off');
hold on;
plot(t_idx, Y_actual, 'Color',[cAct,0.55], 'LineWidth',0.8, ...
     'DisplayName','Actual PM_{2.5}');
plot(t_idx, Y_pred,   'Color',cPred,       'LineWidth',1.2, ...
     'DisplayName','LSTM Predicted');
yline(mean(Y_actual),'--','Color',[0.5 0.5 0.5],'LineWidth',1.0,'Alpha',0.8,...
      'Label','Mean','LabelHorizontalAlignment','left');
ylabel('PM_{2.5}  (\mug m^{-3})','FontSize',11);
xlabel('Test Sample Index  (hourly)','FontSize',11);
title(sprintf('\\bfLSTM: Actual vs Predicted  —  RMSE=%.2f µg/m³   MAPE=%.2f%%   R²=%.4f',...
      RMSE,MAPE,R2),'FontSize',12);
legend('Location','northeast','FontSize',10,'Box','off');
grid on; box on;
saveFigHeadless(fA, 'figa_TimeSeries', 14, 4.5);

% ─────────────────────────────────────────────────────────────────────────
% FIGURE B — Regression scatter
% ─────────────────────────────────────────────────────────────────────────
fB = figure('Color','w','Position',[0 0 620 560]);
idx_s = randperm(numel(Y_actual), min(3000, numel(Y_actual)));
scatter(Y_actual(idx_s), Y_pred(idx_s), 6, cSctt, 'filled', ...
        'MarkerFaceAlpha',0.18,'MarkerEdgeColor','none', ...
        'DisplayName','Test samples');
hold on;
lims = [min([Y_actual;Y_pred]), max([Y_actual;Y_pred])];
plot(lims, lims, 'k-', 'LineWidth',2.0, 'DisplayName','Ideal  y = x');
p  = polyfit(Y_actual, Y_pred, 1);
xl = linspace(lims(1), lims(2), 200);
plot(xl, polyval(p,xl), '--', 'Color',cPred, 'LineWidth',1.5, ...
     'DisplayName', sprintf('Fit: y=%.3fx%+.2f', p(1), p(2)));
xlabel('Actual PM_{2.5}  (\mug m^{-3})','FontSize',11);
ylabel('Predicted PM_{2.5}  (\mug m^{-3})','FontSize',11);
title('\bfRegression Scatter — Actual vs. Predicted','FontSize',12);
text(0.05,0.91,sprintf('R^{2} = %.4f',R2),'Units','normalized', ...
     'FontSize',12,'FontWeight','bold','Color',[0.12 0.12 0.12]);
legend('Location','southeast','FontSize',9,'Box','off');
grid on; box on; axis equal tight;
saveFigHeadless(fB, 'figb_Scatter', 6.5, 5.8);

% ─────────────────────────────────────────────────────────────────────────
% FIGURE C — Error distribution histogram
% ─────────────────────────────────────────────────────────────────────────
fC = figure('Color','w','Position',[0 0 760 500]);
hh = histogram(errors, 'NumBins',70, 'FaceColor',cHist, ...
               'EdgeColor','none', 'Normalization','probability', ...
               'DisplayName','Prediction errors');
hold on;
pd    = fitdist(errors, 'Normal');
e_lin = linspace(min(errors)-5, max(errors)+5, 600);
bw    = mean(diff(hh.BinEdges));
plot(e_lin, pd.pdf(e_lin)*bw, 'Color',cAct, 'LineWidth',2.5, ...
     'DisplayName',sprintf('Normal fit  (\\mu=%.2f, \\sigma=%.2f)', pd.mu, pd.sigma));
xline(0,    '-',  'Color',[0.15 0.15 0.15],'LineWidth',1.8, ...
      'Label','Zero','LabelVerticalAlignment','bottom');
xline( RMSE,'--', 'Color',cPred,'LineWidth',1.2,'Alpha',0.8, ...
      'Label','+RMSE','LabelVerticalAlignment','bottom');
xline(-RMSE,'--', 'Color',cPred,'LineWidth',1.2,'Alpha',0.8, ...
      'Label','-RMSE','LabelVerticalAlignment','bottom');
xlabel('Prediction Error  \epsilon = Actual - Predicted  (\mug m^{-3})','FontSize',11);
ylabel('Relative Frequency','FontSize',11);
title('\bfError Distribution — AI Model Statistical Reliability Benchmark','FontSize',12);
text(0.02, 0.91, ...
     sprintf('Skewness  = %.3f\nKurtosis  = %.3f\nRMSE      = %.2f µg/m³', ...
             skewness(errors), kurtosis(errors), RMSE), ...
     'Units','normalized','FontSize',9,'Color',[0.12 0.12 0.12], ...
     'BackgroundColor',[0.96 0.96 0.96],'EdgeColor',[0.80 0.80 0.80]);
legend('Location','northeast','FontSize',9,'Box','off');
grid on; box on;
saveFigHeadless(fC, 'figc_ErrorDist', 8, 5.2);

fprintf('complete — 3 figures saved (no graphics client used).\n');