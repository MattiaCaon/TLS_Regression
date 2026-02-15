clear; clc; close all;

% LOAD THE DATASET
load("../res/dataset.mat")

% CONFIGURATION
RANGES = [0; 0.28; 0.78; 1.0];
N_blocks = length(RANGES)-1;
N_total = length(use_data_soc_meas);
s_x = NOISE_STD_DEV_SOC; 
s_y = NOISE_STD_DEV_R0; 
N = floor(length(use_data_soc_meas)/N_blocks); % Data for each block

a_svd_w_blocks = zeros(N_blocks, 1);
b_svd_w_blocks = zeros(N_blocks, 1);
a_svd_u_blocks = zeros(N_blocks, 1);
b_svd_u_blocks = zeros(N_blocks, 1);
a_true_blocks = zeros(N_blocks, 1);
b_true_blocks = zeros(N_blocks, 1);

fprintf('\nDataset data:\n\tstd_dev_x: %.8f\n', s_x);
fprintf('\nDataset data:\n\tstd_dev_y: %.8f\n', s_y);

% PLOT INIT
figure('Position', [100, 600, 1000, 800]);
grid on;
colors = ['r', 'g', 'b'];

% BLOCKS 
for block_idx = 1:N_blocks
    fprintf('\n\n%s\n', repmat('-', 1, 46));
    fprintf('%s Block  %d %s\n', repmat('-', 1, 18), block_idx, repmat('-', 1, 18));
    fprintf('%s\n', repmat('-', 1, 46));
    

    %%%%%%%%%%%%%%%%%%%% 1) Get raw data partion %%%%%%%%%%%%%%%%%%%%

    % Partitionate data
    current_range = floor(RANGES(block_idx)*N_total)+1 : floor(RANGES(block_idx+1)*N_total);
    N = length(current_range);

    x_raw = use_data_soc_meas(current_range);
    y_raw = use_data_r0_meas(current_range);

    %-- true estimation
    x_raw_true = use_data_soc_true(current_range);
    y_raw_true = use_data_r0_true(current_range);
    

    %%%%%%%%%%%%%%%%%%%%% 2) Mean centering %%%%%%%%%%%%%%%%%%%%

    % Shift the data so its center is at (0,0).
    % This allows y=ax solver to find the correct slope.
    x_mean = mean(x_raw);
    y_mean = mean(y_raw);
    x = x_raw - x_mean;
    y = y_raw - y_mean;

    % True estimation (no x noise)
    x_true_mean = mean(x_raw_true);
    y_true_mean = mean(y_raw_true);
    x_true = x_raw_true - x_true_mean;
    y_true = y_raw_true - y_true_mean;

    % True estimation
    a_true = x_true \ y_true;
    b_true = y_true_mean - a_true * x_true_mean;
    %--save
    a_true_blocks(block_idx) = a_true;
    b_true_blocks(block_idx) = b_true;


    %%%%%%%%%%%%%%%%%%%% 3) Actual calculation %%%%%%%%%%%%%%%%%%%%
    
    % 1. WEIGHTED svd_w EVALUATION (The "Honest" Solver)
    % Whitening
    Z_w = [x./s_x, y./s_y];
    [U_w, S_w, V_w] = svd(Z_w, 0);
    
    % Slope Calculation (De-whitened)
    v_min_w = V_w(:, end); % The singular vector for the smallest singular value
    a_weighted = -v_min_w(1) / v_min_w(2) * (s_y/s_x);
    b_weighted = y_mean - a_weighted * x_mean;
    %--save
    a_svd_w_blocks(block_idx) = a_weighted;
    b_svd_w_blocks(block_idx) = b_weighted;
    
    % Reconstruct the "Clean" Data (Rank-1 Approximation)
    % This removes the noise component defined by the smallest singular value
    Z_clean_w = U_w(:,1) * S_w(1,1) * V_w(:,1)'; 
    
    % Map back to the physical space
    x_fit_w = Z_clean_w(:,1) * s_x; 
    y_fit_w = Z_clean_w(:,2) * s_y;
    
    % Calculate Residuals
    dx_w = x - x_fit_w;
    dy_w = y - y_fit_w;
    
    % --- METRICS ---
    % 1. Statistical Cost (Chi-Squared): Weighted svd_w minimizes this.
    cost_stat_w = mean( (dx_w./s_x).^2 + (dy_w./s_y).^2 ); 
    
    % 2. Geometric Cost (Euclidean): Weighted svd_w ignores this.
    cost_geom_w = mean( dx_w.^2 + dy_w.^2 );


    % 2. UNWEIGHTED svd_w EVALUATION
    % Raw Data
    Z_u = [x, y];
    [U_u, S_u, V_u] = svd(Z_u, 0);
    
    % Slope
    v_min_u = V_u(:, end);
    a_unweighted = -v_min_u(1) / v_min_u(2); 
    b_unweighted = y_mean - a_unweighted * x_mean;
    %--save
    a_svd_u_blocks(block_idx) = a_unweighted;
    b_svd_u_blocks(block_idx) = b_unweighted;
    
    % Reconstruction
    Z_clean_u = U_u(:,1) * S_u(1,1) * V_u(:,1)';
    x_fit_u = Z_clean_u(:,1);
    y_fit_u = Z_clean_u(:,2);
    
    % Residuals
    dx_u = x - x_fit_u;
    dy_u = y - y_fit_u;
    
    % --- METRICS ---
    % Statistical Cost (Chi-Squared)
    cost_stat_u = mean( (dx_u./s_x).^2 + (dy_u./s_y).^2 );
    
    % Geometric Cost (Euclidean): Unweighted svd_w minimizes this.
    cost_geom_u = mean( dx_u.^2 + dy_u.^2 );

    fprintf('TRUE   Model: y = %.8fx + %.8f \n', a_true, b_true);
    fprintf('SVD UW Model : y = %.8fx + %.8f \t Delta slope: %.5f \n', a_unweighted, b_unweighted, abs(a_true-a_unweighted));
    fprintf('SVD W  Model : y = %.8fx + %.8f \t Delta slope: %.5f \n', a_weighted, b_weighted, abs(a_true-a_weighted));
    
    % --- FINAL COMPARISON PRINT ---
    fprintf('\n');
    fprintf('1. STATISTICAL COST \n');
    fprintf('\t Weighted svd_w:   %.7f\n', cost_stat_w);
    fprintf('\t Unweighted svd_w: %.7f\n', cost_stat_u);
    
    fprintf('%s\n', repmat('-', 1, 46));
    fprintf('2. GEOMETRIC COST\n');
    fprintf('\t Weighted svd_w:   %.7f\n', cost_geom_w);
    fprintf('\t Unweighted svd_w: %.7f\n', cost_geom_u);
    

    %%%%%%%%%%%%%%%%%%%% 6) Data plotting %%%%%%%%%%%%%%%%%%%%

    % Reconstruct the index range for this block to plot only where data is valid
    x_seg = use_data_soc_meas(current_range);

    % Plot SVD (ok)
    y_fit = a_weighted * x_seg + b_weighted;
    plot(x_seg, y_fit, [colors(2)], LineWidth=2, DisplayName=sprintf('Fit Block %d (a=%.4f)', block_idx, a_weighted)); hold on;

    % Plot SVD (wrong one)
    y_fit = a_unweighted * x_seg + b_unweighted;
    plot(x_seg, y_fit, [colors(1) '--'], LineWidth=2, DisplayName=sprintf('Fit Block %d (a=%.4f)', block_idx, a_unweighted)); hold on;
    
    % Info printing
    fprintf('Final svd_w SVD Model: y = %.4fx + %.4f\n', a_unweighted, b_unweighted);

end

% Plot data points
plot(use_data_soc_meas, use_data_r0_meas, 'ko', 'DisplayName', 'Data', 'MarkerFaceColor', 'k', 'MarkerSize', 3);

% Finishing up the plot
legend show;
title('TLS fitting with with SVD method');
ylabel('R0 [Ohm]');
xlabel('SOC [%]');




% Plot all interpolation approachs comparing them with the true best linear approximations
figure('Position', [100, 600, 1000, 800]);

% TRUE FUNCTION
plot(use_data_soc_true, use_data_r0_true, DisplayName='REAL', Color='black', LineStyle='--', LineWidth=1); hold on;

% TRUE APPROX
R0_approx_true_simple = simpleComputePiecewiseApprox(b_true_blocks, a_true_blocks, use_data_soc_true, RANGES);
R0_approx_true_adv    = computePiecewiseApprox(b_true_blocks, a_true_blocks, use_data_soc_true);
plot(use_data_soc_true, R0_approx_true_simple, DisplayName='TRUE (simple)', Color='black', LineStyle='--', LineWidth=1); hold on;
plot(use_data_soc_true, R0_approx_true_adv,    DisplayName='TRUE (adv)',    Color='black', LineStyle='-',  LineWidth=1); hold on;

% SVD NOT WHITENED
R0_approx_svd_u_simple = simpleComputePiecewiseApprox(b_svd_u_blocks, a_svd_u_blocks, use_data_soc_true, RANGES);
R0_approx_svd_u_adv    = computePiecewiseApprox(b_svd_u_blocks, a_svd_u_blocks, use_data_soc_true);
plot(use_data_soc_true, R0_approx_svd_u_simple, DisplayName='SVD not whitened (simple)', Color='red', LineStyle='--', LineWidth=1); hold on;
plot(use_data_soc_true, R0_approx_svd_u_adv,    DisplayName='SVD not whitened (adv)',    Color='red', LineStyle='-',  LineWidth=1); hold on;

% SVD WHITENED
R0_approx_svd_w_simple = simpleComputePiecewiseApprox(b_svd_w_blocks, a_svd_w_blocks, use_data_soc_true, RANGES);
R0_approx_svd_w_adv    = computePiecewiseApprox(b_svd_w_blocks, a_svd_w_blocks, use_data_soc_true);
plot(use_data_soc_true, R0_approx_svd_w_simple, DisplayName='SVD whitened (simple)', Color='#00F000', LineStyle='--', LineWidth=1); hold on;
plot(use_data_soc_true, R0_approx_svd_w_adv,    DisplayName='SVD whitened (adv)',    Color='#00F000', LineStyle='-',  LineWidth=1); hold on;

legend show;
ylabel('R0 [Ohm]');
xlabel('SOC [%]');

%%%%%%%%%%%%%%%%%%%% 7) Final results %%%%%%%%%%%%%%%%%%%%

fprintf("\n----------------------------------------------------------------------\n")
fprintf("\nFINAL COMPARISON")
fprintf("\n----------------------------------------------------------------------\n")


% Residuals calculation from the true best interpolation
final_diff_svd_u = sqrt(mean((R0_approx_true_simple(1:N_total) - R0_approx_svd_u_simple).^2));
final_diff_svd_w = sqrt(mean((R0_approx_true_simple(1:N_total) - R0_approx_svd_w_simple).^2));
delta = (final_diff_svd_u - final_diff_svd_w);

fprintf("SIMPLE\n")
fprintf("svd_u mean residual: %.10f\n", final_diff_svd_u);
fprintf("svd_w mean residual: %.10f\n", final_diff_svd_w);
fprintf("Delta in approaches: %.10f\n", delta);
fprintf("Improvements: %.3f%%", (delta/final_diff_svd_u) * 100.0)

fprintf("\n----------------------------------------------------------------------\n")
fprintf("ADVANCED\n")
final_diff_svd_u = sqrt(mean((R0_approx_true_adv(1:N_total) - R0_approx_svd_u_adv).^2));
final_diff_svd_w = sqrt(mean((R0_approx_true_adv(1:N_total) - R0_approx_svd_w_adv).^2));
delta = (final_diff_svd_u - final_diff_svd_w);

fprintf("svd_u mean residual: %.10f\n", final_diff_svd_u);
fprintf("svd_w mean residual: %.10f\n", final_diff_svd_w);
fprintf("Delta in approaches: %.10f\n", delta);
fprintf("Improvements: %.3f%%\n\n", (delta/final_diff_svd_u) * 100.0)
