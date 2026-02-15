clear; clc; close all;

% LOAD THE DATASET
load("../res/dataset.mat")

s_x = NOISE_STD_DEV_SOC; 
s_y = NOISE_STD_DEV_R0;

% CONFIGURATION
RANGES = [0; 0.28; 0.78; 1.0];
N_blocks = length(RANGES)-1;
N_total = length(use_data_soc_meas);
N_iter = 12;
DAMPING_FACTOR = 0.7; % Default: 1.0

weighted_sq_err_history = zeros(N_blocks, N_iter);

% Initization of lines parameters vector for all the blocks
a_tls_blocks = zeros(N_blocks, 1);
b_tls_blocks = zeros(N_blocks, 1);
a_ols_blocks = zeros(N_blocks, 1);
b_ols_blocks = zeros(N_blocks, 1);
a_true_blocks = zeros(N_blocks, 1);
b_true_blocks = zeros(N_blocks, 1);

% PLOT INIT
figure('Position', [100, 400, 1000, 800]);
grid on;


% BLOCKS 
for block_idx = 1:N_blocks
    fprintf('\n\n%s\n', repmat('-', 1, 46)); % Print a separator line
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

    
    %%%%%%%%%%%%%%%%%%%% 3) Actual calculation %%%%%%%%%%%%%%%%%%%%
    
    fprintf('%4s | %11s | %13s | %8s\n', 'Iter', 'Slope', 'Cost (Chi2)', 'Norm');
    fprintf('%s\n', repmat('-', 1, 46));

    % Initial guess using centered data
    a_ols = x \ y;
    b_ols = y_mean - a_ols * x_mean;
    %--save
    a_ols_blocks(block_idx) = a_ols;
    b_ols_blocks(block_idx) = b_ols;

    % True estimation
    a_true = x_true \ y_true;
    b_true = y_true_mean - a_true * x_true_mean;
    %--save
    a_true_blocks(block_idx) = a_true;
    b_true_blocks(block_idx) = b_true;


    %%%%%%%%%%%%%%%%%%%% 3) Iterative process %%%%%%%%%%%%%%%%%%%%

    % Init
    as = a_ols;
    xs = x;
    
    % Temporary vector containing progression of estimations (for visualization)
    as_log_block = zeros(N_iter, 1);
    bs_log_block = zeros(N_iter, 1);

    % Construct weight coefficients matrix (std)
    si = diag([1/s_x 1/s_y]);
    W = kron(si, eye(N));
    
    % Loop
    for iter = 1:N_iter
        % Obtain array of residuals
        ys = as * xs;
        dx = x - xs;       
        dy = y - ys;      
        delta_d = [dx; dy];      
        
        % Construct Jacobian matrix
        Gx = [eye(N), zeros(N,1)]; 
        Gy = [as*eye(N), xs];       
        G = [Gx; Gy];
        
        % Apply weights
        WG = W * G;
        Wd = W * delta_d;
        
        % Calculate inverse (best practice operator)
        delta_m = (WG \ Wd) * DAMPING_FACTOR; % By modifying it we can see how big of a step each iteration takes

        % Update estimation for next iteration
        xs = xs + delta_m(1:N);      
        as = as + delta_m(N+1);  

        % Update history of a_tls for iteration visualization
        as_log_block(iter) = as;
        bs_log_block(iter) = y_mean - as * x_mean;
        
        % This is what the math is actually minimizing:
        weighted_sq_err = mean( (dx/s_x).^2 + (dy/s_y).^2 );
        weighted_sq_err_history(block_idx, iter) = weighted_sq_err;
        
        % This is just for curiosity (will likely go up)
        euclidean_norm = sqrt(mean(dx.^2 + dy.^2));
    
        fprintf('%4d | %11.6f | %13.6f | %8.6f\n', iter, as, weighted_sq_err, euclidean_norm);
        
    end
    
    %%%%%%%%%%%%%%%%%%%% 5) Finalizing the results %%%%%%%%%%%%%%%%%%%%
    
    % The slope 'a' is correct.
    % Calculate 'b' (intercept) to map back to original coordinates.
    % y = a*x + b  =>  mean_y = a*mean_x + b  =>  b = mean_y - a*mean_x
    a_tls = as;
    b_tls = y_mean - a_tls * x_mean;
    %--save
    a_tls_blocks(block_idx) = a_tls;
    b_tls_blocks(block_idx) = b_tls;

    % Reconstruct the index for this block to plot the line segment only where data exists
    idx_range = floor(RANGES(block_idx)*N_total)+1 : floor(RANGES(block_idx+1)*N_total);
    x_block_meas = use_data_soc_meas(idx_range);

    % Show starting point (OLS)
    plot(x_block_meas, (x_block_meas * a_ols + b_ols), Color='#FF0000', LineWidth=1, LineStyle='--', DisplayName=sprintf('OLS (a=%.6f)', a_ols));  hold on;

    % Show iterations evolution (first one is OLS, thus skipping it)
    for i = 2:N_iter    
        plot(x_block_meas, x_block_meas * as_log_block(i) + bs_log_block(i), ...
            Color='#00FF00', LineWidth=1, LineStyle='--', DisplayName=sprintf('Iter %d (a=%.4f)', i, as_log_block(i)));
        hold on;
    end   

    % Calculate and plot the final fit
    plot(x_block_meas, (x_block_meas * a_tls + b_tls), Color='#0000FF', LineWidth=1, DisplayName=sprintf('Final Fit %d (a=%.6f)', i, a_tls));  hold on;
    
    % Info printing
    fprintf('\n');
    fprintf('TRUE Model: y = %.8fx + %.8f \n', a_true, b_true);
    fprintf('OLS Model : y = %.8fx + %.8f \t Delta slope: %.5f \n', a_ols, b_ols, abs(a_true-a_ols));
    fprintf('TLS Model : y = %.8fx + %.8f \t Delta slope: %.5f \n', a_tls, b_tls, abs(a_true-a_tls));
    
end

% Plot data points
plot(use_data_soc_meas, use_data_r0_meas, 'ko', 'DisplayName', 'Data', 'MarkerFaceColor', 'k', 'MarkerSize', 3); hold on;
ylim([0 0.100])
legend show;
title('TLS fitting with iterative approach');
ylabel('R0 [Ohm]');
xlabel('SOC [%]');


%%%%%%%%%%%%%%%%%%%% 6) Plotting residuals over iterations %%%%%%%%%%%%%%%%%%%%

% Plot residual decrease for each block
figure('Position', [100, 500, 1000, 800]);
ax = zeros(N_blocks, 1);
for block_idx = 1:N_blocks
    ax(block_idx) = subplot(N_blocks, 1, block_idx);
    plot(weighted_sq_err_history(block_idx,:),LineWidth=0.5, DisplayName=sprintf('Block %d', block_idx)); hold on;
    title('Statistical residual over iterations');
    legend show;
end
linkaxes(ax, 'x');
ylabel('RMS');
xlabel('Iteration');




%%%%%%%%%%%%%%%%%%%% 7) Plotting interpolation comparisons %%%%%%%%%%%%%%%%%%%%

% Plot all interpolation approachs comparing them with the true best linear approximations
figure('Position', [100, 600, 1000, 800]);

% TRUE FUNCTION
plot(use_data_soc_true, use_data_r0_true, DisplayName='REAL', Color='black', LineStyle='--', LineWidth=1); hold on;

% TRUE APPROX
R0_approx_true_simple = simpleComputePiecewiseApprox(b_true_blocks, a_true_blocks, use_data_soc_true, RANGES);
R0_approx_true_adv    = computePiecewiseApprox(b_true_blocks, a_true_blocks, use_data_soc_true);
plot(use_data_soc_true, R0_approx_true_simple, DisplayName='TRUE simple', Color='black', LineStyle='--', LineWidth=1); hold on;
plot(use_data_soc_true, R0_approx_true_adv,    DisplayName='TRUE adv',    Color='black', LineStyle='-',  LineWidth=1); hold on;

% OLS
R0_approx_ols_simple = simpleComputePiecewiseApprox(b_ols_blocks, a_ols_blocks, use_data_soc_true, RANGES);
R0_approx_ols_adv    = computePiecewiseApprox(b_ols_blocks, a_ols_blocks, use_data_soc_true);
plot(use_data_soc_true, R0_approx_ols_simple, DisplayName='OLS simple', Color='red', LineStyle='--', LineWidth=1); hold on;
plot(use_data_soc_true, R0_approx_ols_adv,    DisplayName='OLS adv',    Color='red', LineStyle='-',  LineWidth=1); hold on;

% TLS
R0_approx_tls_simple = simpleComputePiecewiseApprox(b_tls_blocks, a_tls_blocks, use_data_soc_true, RANGES);
R0_approx_tls_adv    = computePiecewiseApprox(b_tls_blocks, a_tls_blocks, use_data_soc_true);
plot(use_data_soc_true, R0_approx_tls_simple, DisplayName='TLS simple', Color='#00F000', LineStyle='--', LineWidth=1); hold on;
plot(use_data_soc_true, R0_approx_tls_adv,    DisplayName='TLS adv',    Color='#00F000', LineStyle='-',  LineWidth=1); hold on;

legend show;
ylabel('R0 [Ohm]');
xlabel('SOC [%]');


%%%%%%%%%%%%%%%%%%%% 7) Final results %%%%%%%%%%%%%%%%%%%%
fprintf("\n----------------------------------------------------------------------\n")
fprintf("\nFINAL COMPARISON")
fprintf("\n----------------------------------------------------------------------\n")


% Residuals calculation from the true best interpolation

fprintf("SIMPLE\n")
final_diff_ols = sqrt(mean((R0_approx_true_simple(1:N_total) - R0_approx_ols_simple).^2));
final_diff_tls = sqrt(mean((R0_approx_true_simple(1:N_total) - R0_approx_tls_simple).^2));
delta = (final_diff_ols - final_diff_tls);

fprintf("OLS mean residual: %.10f\n", final_diff_ols);
fprintf("TLS mean residual: %.10f\n", final_diff_tls);
fprintf("Delta in approaches: %.10f\n", delta);
fprintf("Improvements: %.3f%%", (delta/final_diff_ols) * 100.0)

fprintf("\n----------------------------------------------------------------------\n")

fprintf("ADVANCED\n")
final_diff_ols = sqrt(mean((R0_approx_true_adv(1:N_total) - R0_approx_ols_adv).^2));
final_diff_tls = sqrt(mean((R0_approx_true_adv(1:N_total) - R0_approx_tls_adv).^2));
delta = (final_diff_ols - final_diff_tls);

fprintf("OLS mean residual: %.10f\n", final_diff_ols);
fprintf("TLS mean residual: %.10f\n", final_diff_tls);
fprintf("Delta in approaches: %.10f\n", delta);
fprintf("Improvements: %.3f%%\n\n", (delta/final_diff_ols) * 100.0)
