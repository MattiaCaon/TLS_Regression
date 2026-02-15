clear; clc; close all;

% LOAD THE DATASET
load("../res/dataset.mat")

% CONFIGURATION
RANGES = [0; 0.28; 0.78; 1.0];
N_blocks = length(RANGES)-1;
sx = NOISE_STD_DEV_SOC; 
sy = NOISE_STD_DEV_R0; 
N = floor(length(use_data_soc_meas)/N_blocks);
N_total = length(use_data_soc_meas);
N_iter = 10;
weighted_sq_err_history = zeros(N_blocks, N_iter);
final_r0_approx_tls = [];
final_r0_approx_ols = [];
final_r0_approx_true = [];

% PLOT INIT
figure('Position', [100, 400, 1000, 800]);
grid on;
colors = ['r', 'g', 'b'];


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
    mean_x = mean(x_raw);
    mean_y = mean(y_raw);
    x = x_raw - mean_x;
    y = y_raw - mean_y;

    %-- true estimation
    mean_x_true = mean(x_raw_true);
    mean_y_true = mean(y_raw_true);
    x_true = x_raw_true - mean_x_true;
    y_true = y_raw_true - mean_y_true;

    
    %%%%%%%%%%%%%%%%%%%% 3) Actual calculation %%%%%%%%%%%%%%%%%%%%
    
    fprintf('%4s | %11s | %13s | %8s\n', 'Iter', 'Slope', 'Cost (Chi2)', 'Norm');
    fprintf('%s\n', repmat('-', 1, 46)); % Print a separator line

    % Construct weight coefficients matrix (std)
    si = diag([1/sx 1/sy]);
    W = kron(si, eye(N));
    
    % Initial Guess using centered data
    a_ols = x \ y;
    b_ols = mean_y - a_ols * mean_x;

    %-- true estimation
    a_true = x_true \ y_true;
    b_true = mean_y_true - a_true * mean_x_true;
    final_r0_approx_true = [final_r0_approx_true; a_true * x_raw_true + b_true];


    %%%%%%%%%%%%%%%%%%%% 3) Iterative process %%%%%%%%%%%%%%%%%%%%

    % Init
    as = a_ols;
    xs = x;
    as_log_block = zeros(N_iter, 1);
    bs_log_block = zeros(N_iter, 1);
    
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
        delta_m = WG \ Wd;

        % Update estimation for next iteration
        xs = xs + delta_m(1:N);      
        as = as + delta_m(N+1);  

        % Update history of a_tls
        as_log_block(iter) = as;
        bs_log_block(iter) = mean_y - as * mean_x;
        
        % Calculate the COST FUNCTION properly
        % This is what the math is actually minimizing:
        weighted_sq_err = mean( (dx/sx).^2 + (dy/sy).^2 );
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
    b_tls = mean_y - a_tls * mean_x;

    % Reconstruct the index for this block to plot the line segment only where data exists
    idx_range = floor(RANGES(block_idx)*N_total)+1 : floor(RANGES(block_idx+1)*N_total);
    x_seg_meas = use_data_soc_meas(idx_range);
    x_seg_true = use_data_soc_true(idx_range);

    % Calculate y using y = ax + b (for every iteration)
    for i = 1:N_iter    
        y_fit = x_seg_meas * as_log_block(i) + bs_log_block(i);
        plot(x_seg_meas, y_fit, [colors(1) '--'], 'LineWidth', 2, 'DisplayName', sprintf('Fit Block %d (a=%.4f)', block_idx, as_log_block(i)));
        hold on;
    end    

    % Calculate and plot the final fit
    y_fit = x_seg_meas * a_tls + b_tls;
    plot(x_seg_meas, y_fit, [colors(2)], 'LineWidth', 2, 'DisplayName', sprintf('Final Fit %d (a=%.4f)', i, a_tls));  hold on;
    
    % For final comparison graph
    final_r0_approx_ols = [final_r0_approx_ols; x_seg_true * a_ols + b_ols];
    final_r0_approx_tls = [final_r0_approx_tls; x_seg_true * a_tls + b_tls];
    
    % Info printing
    fprintf('\nFinal    TLS Model: y = %.8fx + %.8f\n', a_tls, b_tls);
    fprintf('Original OLS Model: y = %.8fx + %.8f\n', a_ols, b_ols);
end


%%%%%%%%%%%%%%%%%%%% 6) Plotting comparisons %%%%%%%%%%%%%%%%%%%%

% Plot data points
plot(use_data_soc_meas, use_data_r0_meas, 'ko', 'DisplayName', 'Data', 'MarkerFaceColor', 'k', 'MarkerSize', 3); hold on;
ylim([0 0.100])
legend show;
title('TLS Fitting with iterative approach');
xlabel('SOC'); ylabel('R0');

% Plot residual decrease for each block
figure('Position', [100, 500, 1000, 800]);
ax = zeros(N_blocks, 1);
for block_idx = 1:N_blocks
    ax(block_idx) = subplot(3, 1, block_idx);
    plot(weighted_sq_err_history(block_idx,:), 'DisplayName', sprintf('Block %d', block_idx)); hold on;
    title('Statistical residual over iterations');
    legend show;
end
linkaxes(ax, 'x');

% Plot all interpolation approachs comparing them with the true best linear approximations
figure('Position', [100, 600, 1000, 800]);

plot(use_data_soc_true, final_r0_approx_ols, 'DisplayName', 'OLS'); hold on;
plot(use_data_soc_true, final_r0_approx_tls, 'DisplayName', 'TLS'); hold on;
plot(use_data_soc_true, use_data_r0_true, 'DisplayName', 'REAL'); hold on;
plot(use_data_soc_true, final_r0_approx_true, 'DisplayName', 'TRUE_Approx'); hold on;
legend show;

% Residuals calculation from the true best interpolation
final_diff_ols = sqrt(mean((final_r0_approx_true(1:N_total) - final_r0_approx_ols).^2));
final_diff_tls = sqrt(mean((final_r0_approx_true(1:N_total) - final_r0_approx_tls).^2));

fprintf("\n----------------------------------------------------------------------\n")
fprintf("OLS mean residual: %.10f\n", final_diff_ols);
fprintf("TLS mean residual: %.10f\n", final_diff_tls);