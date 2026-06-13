clc; clear; close all;

% ── 1. LOAD & FILTER ──────────────────────────────────────────────────────
opts             = detectImportOptions('delhi-weather-aqi-2025.csv');
opts.VariableNamingRule = 'preserve';
T                = readtable('delhi-weather-aqi-2025.csv', opts);

% Use Anand Vihar — highest-traffic station in the dataset
loc              = 'Anand Vihar';
mask             = strcmp(T.location, loc);
T                = T(mask, :);
T                = sortrows(T, {'date_ist','time_ist'});

pm25_raw         = double(T.pm2_5);
no2_raw          = double(T.no2);
N                = numel(pm25_raw);
t_vec            = (0 : N-1)';          % integer-hour index

fprintf('Station : %s\n', loc);
fprintf('Samples : %d  (%.0f days)\n', N, N/24);

% ── 2. SYNTHETIC NaN INJECTION (5 % random gaps) ─────────────────────────
%    Real dataset has no missing values; we demonstrate C² spline
%    imputation on 5% randomly removed observations — standard academic
%    methodology for robustness testing.
rng(2025);
miss_frac         = 0.05;
nan_pm25          = sort(randperm(N, round(miss_frac * N)));
nan_no2           = sort(randperm(N, round(miss_frac * N)));

pm25_miss         = pm25_raw;   pm25_miss(nan_pm25) = NaN;
no2_miss          = no2_raw;    no2_miss(nan_no2)   = NaN;

fprintf('Missing values injected : %d in PM2.5,  %d in NO2\n', ...
        numel(nan_pm25), numel(nan_no2));

% ── 3. PIECEWISE CUBIC SPLINE IMPUTATION (C² Continuity) ──────────────────
%    spline() fits a not-a-knot cubic spline across known nodes, guaranteeing
%    C² continuity (continuous 2nd derivatives) at every interior knot.
ok_pm25  = ~isnan(pm25_miss);
ok_no2   = ~isnan(no2_miss);

pp_pm25  = spline(t_vec(ok_pm25), pm25_miss(ok_pm25));
pp_no2   = spline(t_vec(ok_no2),  no2_miss(ok_no2));

pm25_sp  = max(ppval(pp_pm25, t_vec), 0);   % enforce non-negativity
no2_sp   = max(ppval(pp_no2,  t_vec), 0);

impute_err_pm25 = rms(pm25_sp(nan_pm25) - pm25_raw(nan_pm25));
impute_err_no2  = rms(no2_sp(nan_no2)   - no2_raw(nan_no2));
fprintf('Imputation RMSE — PM2.5 : %.3f  |  NO2 : %.3f  µg/m³\n', ...
        impute_err_pm25, impute_err_no2);

% ── 4. DIURNAL TRAFFIC MULTIPLIER (dual-peak: 09:00 AM & 18:00 PM) ────────
%    Two-peak traffic day modelled as a sum of narrow Gaussians.
%    Normalised to M ∈ [1, 2] so the baseline emission is never suppressed.
hod       = mod(t_vec, 24);                 % hour-of-day [0–23]
sigma_t   = 1.5;                            % peak width  [hours]
M_AM      = exp(-((hod -  9).^2) / (2*sigma_t^2));
M_PM      = exp(-((hod - 18).^2) / (2*sigma_t^2));
M_raw     = 0.25 + 0.80*M_AM + 1.00*M_PM;  % PM peak slightly stronger
M_traffic = 1 + (M_raw - min(M_raw)) ./ (max(M_raw) - min(M_raw));  % ∈[1,2]

% ── 5. DYNAMIC SOURCE VECTOR  S(t) = NO2_splined × M_traffic ─────────────
S_dyn     = no2_sp .* M_traffic;

% ── 6. EXPORT for Script 2 ───────────────────────────────────────────────
u_mean_ms = mean(T.windspeed_kph) * (1000/3600);   % km/h → m/s
S_mean    = mean(S_dyn);
save('data_outputs.mat', ...
     'pm25_sp','no2_sp','M_traffic','S_dyn','u_mean_ms','S_mean','t_vec','N');
fprintf('Mean wind speed  u  = %.3f m/s\n', u_mean_ms);
fprintf('Mean source term S  = %.2f  µg/m³\n', S_mean);

% ── 7. PUBLICATION-READY FIGURE ──────────────────────────────────────────
days  = t_vec / 24;

% Colour palette (ColorBrewer-safe)
cPM   = [0.122  0.471  0.706];   % blue
cNO2  = [0.200  0.627  0.173];   % green
cImp  = [0.890  0.102  0.110];   % red
cMult = [0.576  0.094  0.639];   % purple
cSrc  = [0.933  0.420  0.149];   % orange

fig = figure('Color','w','Position',[60 60 1440 900]);
tl  = tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

% ── Panel A : PM2.5 Raw vs Spline ─────────────────────────────────────────
ax1 = nexttile;
fill([days; flipud(days)], ...
     [pm25_sp + 5; flipud(pm25_sp - 5)], ...
     cPM, 'FaceAlpha',0.12,'EdgeColor','none','HandleVisibility','off');
hold on;
plot(days, pm25_raw, 'Color',[0.72 0.72 0.72], 'LineWidth',0.7, ...
     'DisplayName','Raw PM_{2.5}');
plot(days, pm25_sp,  'Color',cPM, 'LineWidth',1.7, ...
     'DisplayName','Spline Reconstructed (C^{2})');
scatter(days(nan_pm25), pm25_sp(nan_pm25), 16, cImp, 'filled', ...
        'MarkerEdgeColor','none', 'DisplayName', ...
        sprintf('Imputed (%.0f%% gaps, RMS=%.2f µg/m³)',miss_frac*100,impute_err_pm25));
ylabel('PM_{2.5}  (µg m^{-3})','FontSize',11);
title('\bfPM_{2.5} — Raw vs. Piecewise Cubic Spline Reconstruction  (C^{2} Continuity)', ...
      'FontSize',12);
legend('Location','northeast','FontSize',9,'Box','off');
grid on; box on; xlim([0 days(end)]); set(ax1,'XTickLabel',[]);

% ── Panel B : NO2 Raw vs Spline ───────────────────────────────────────────
ax2 = nexttile;
fill([days; flipud(days)], ...
     [no2_sp + 2; flipud(no2_sp - 2)], ...
     cNO2, 'FaceAlpha',0.12,'EdgeColor','none','HandleVisibility','off');
hold on;
plot(days, no2_raw, 'Color',[0.72 0.72 0.72], 'LineWidth',0.7, ...
     'DisplayName','Raw NO_{2}');
plot(days, no2_sp,  'Color',cNO2, 'LineWidth',1.7, ...
     'DisplayName','Spline Reconstructed (C^{2})');
scatter(days(nan_no2), no2_sp(nan_no2), 16, cImp, 'filled', ...
        'MarkerEdgeColor','none', 'DisplayName', ...
        sprintf('Imputed (%.0f%% gaps, RMS=%.2f µg/m³)',miss_frac*100,impute_err_no2));
ylabel('NO_{2}  (µg m^{-3})','FontSize',11);
title('\bfNO_{2} — Raw vs. Piecewise Cubic Spline Reconstruction', ...
      'FontSize',12);
legend('Location','northeast','FontSize',9,'Box','off');
grid on; box on; xlim([0 days(end)]); set(ax2,'XTickLabel',[]);

% ── Panel C : Dynamic Source S(t)  —  first 14 days ──────────────────────
ax3 = nexttile;
nd  = min(14*24, N);
yyaxis left
  area(days(1:nd), S_dyn(1:nd), ...
       'FaceColor',cSrc,'FaceAlpha',0.30,'EdgeColor',cSrc,'LineWidth',1.3, ...
       'DisplayName','S(t) = NO_{2,spline} \times M_{traffic}');
  ylabel('Source S(t)  (µg m^{-3} h^{-1})','FontSize',11);
yyaxis right
  plot(days(1:nd), M_traffic(1:nd), 'Color',cMult, 'LineWidth',1.8, ...
       'DisplayName','M_{traffic}(t)  ∈ [1,2]');
  ylabel('Traffic Multiplier M(t)','FontSize',11);
  ylim([0.8 2.4]);
xlabel('Day of Year 2025','FontSize',11);
title('\bfDynamic Source Term S(t) — First 14 Days  (Dual-Peak Diurnal Traffic Modulation)', ...
      'FontSize',12);
legend('Location','northeast','FontSize',9,'Box','off');
grid on; box on; xlim([0 14]);

linkaxes([ax1, ax2], 'x');
title(tl, ...
    {'PM_{2.5} & NO_{2} Preprocessing: Piecewise Cubic Spline Imputation & Diurnal Source Model'; ...
     ['\fontsize{11}Station: ', loc, '  |  Delhi 2025  |  C^{2} Continuity Guaranteed']}, ...
    'FontSize',14,'FontWeight','bold','Color',[0.15 0.15 0.15]);

drawnow('expose');
print(fig, 'data_Preprocessing', '-dpng', '-r220');
fprintf('\n✓  complete  →  data_Preprocessing.png saved.\n');