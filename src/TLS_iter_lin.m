clear; clc; close all;

% Generates a piecewise linear approximation based on intersection points of sequential line segments.
%
% Inputs:
%   b_blocks      - Vector of Y-intercepts (b) for each block
%   a_blocks      - Vector of Slopes (a) for each block
%   x_data        - Vector of X-axis data (SOC)
%
% Output:
%   y_line - The resulting approximated vector
function y_line = computePiecewiseApprox(b_blocks, a_blocks, x_data)
    % Determine sizes
    N_blocks = length(a_blocks);
    N_total  = length(x_data);
    
    % Initialize intercepts vector
    % Boundaries: 0, cut_1, cut_2, ..., cut_(N-1), N_total
    block_intercepts = zeros(1, N_blocks + 1);
    block_intercepts(1) = 0;

    % Calculate transition indices
    for k = 1:N_blocks
        if k < N_blocks
            % Calculate where line K and line K+1 intersect: 
            % a1*x + b1 = a2*x + b2  =>  x = (b2 - b1) / (a1 - a2)
            x_intersect = (b_blocks(k+1) - b_blocks(k)) / ...
                          (a_blocks(k) - a_blocks(k+1));
            
            % Find the index in data closest to this x-value
            [~, index] = min(abs(x_data - x_intersect));
            
            block_intercepts(k+1) = index;
        end
    end
    
    % Set the final boundary to the end of the data
    block_intercepts(end) = N_total;

    % Construct the result vector 
    % (Pre-allocate output to match the size of input data)
    y_line = zeros(N_total, 1); 

    for k = 1:N_blocks
        % Define the range for this block
        idx_start = block_intercepts(k) + 1;
        idx_end   = block_intercepts(k+1);
        
        % Safety check: ensure start is before end
        if idx_start <= idx_end
            idx_range = idx_start : idx_end;
            
            % Extract x segment
            x_block = x_data(idx_range);
            
            % Ensure column vector for calculation consistency
            if isrow(x_block), x_block = x_block'; end
            
            % Calculate y = ax + b
            segment_vals = x_block * a_blocks(k) + b_blocks(k);
            
            % Store in the final vector
            y_line(idx_range) = segment_vals;
        end
    end
end


function y_line = simpleComputePiecewiseApprox(b_blocks, a_blocks, x_data, block_ranges)
    N_total  = length(x_data);
    N_blocks = length(block_ranges)-1;

    y_line = zeros(N_total, 1);

    for block = 1:N_blocks
        idx_range = floor(block_ranges(block)*N_total)+1 : floor(block_ranges(block+1)*N_total);
        x_block = x_data(idx_range);
        y_line(idx_range) = x_block * a_blocks(block) + b_blocks(block);
    end 
end




% LOAD THE DATASET
load("../res/dataset.mat")

% CONFIGURATION
RANGES = [0; 0.28; 0.78; 1.0];
N_blocks = length(RANGES)-1;
sx = NOISE_STD_DEV_SOC; 
sy = NOISE_STD_DEV_R0; 
N_total = length(use_data_soc_meas);
N_iter = 10;
weighted_sq_err_history = zeros(N_blocks, N_iter);

x_intercepts = zeros(N_blocks-1, 1);
block_intercepts = zeros(N_blocks+1, 1);
a_tls_blocks = zeros(N_blocks, 1);
b_tls_blocks = zeros(N_blocks, 1);
a_ols_blocks = zeros(N_blocks, 1);
b_ols_blocks = zeros(N_blocks, 1);
a_true_blocks = zeros(N_blocks, 1);
b_true_blocks = zeros(N_blocks, 1);

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
    fprintf('%s\n', repmat('-', 1, 46));

    % Initial guess using centered data
    a_ols = x \ y;
    b_ols = mean_y - a_ols * mean_x;
    %--save
    a_ols_blocks(block_idx) = a_ols;
    b_ols_blocks(block_idx) = b_ols;

    % True estimation
    a_true = x_true \ y_true;
    b_true = mean_y_true - a_true * mean_x_true;
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
    si = diag([1/sx 1/sy]);
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
        delta_m = WG \ Wd;

        % Update estimation for next iteration
        xs = xs + delta_m(1:N);      
        as = as + delta_m(N+1);  

        % Update history of a_tls
        as_log_block(iter) = as;
        bs_log_block(iter) = mean_y - as * mean_x;
        
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
    %--save
    a_tls_blocks(block_idx) = a_tls;
    b_tls_blocks(block_idx) = b_tls;

    % Reconstruct the index for this block to plot the line segment only where data exists
    idx_range = floor(RANGES(block_idx)*N_total)+1 : floor(RANGES(block_idx+1)*N_total);
    x_seg_meas = use_data_soc_meas(idx_range);

    % Calculate y using y = ax + b (for every iteration)
    for i = 1:N_iter    
        y_fit = x_seg_meas * as_log_block(i) + bs_log_block(i);
        plot(x_seg_meas, y_fit, [colors(1) '--'], 'LineWidth', 2, 'DisplayName', sprintf('Fit Block %d (a=%.4f)', block_idx, as_log_block(i)));
        hold on;
    end   

    % Calculate and plot the final fit
    plot(x_seg_meas, (x_seg_meas * a_tls + b_tls), [colors(2)], 'LineWidth', 2, 'DisplayName', sprintf('Final Fit %d (a=%.4f)', i, a_tls));  hold on;
    
    % Info printing
    fprintf('\nFinal    TLS Model: y = %.8fx + %.8f\n', a_tls, b_tls);
    fprintf('Original OLS Model: y = %.8fx + %.8f\n\n', a_ols, b_ols);
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
% TRUE FUNCTION
plot(use_data_soc_true, use_data_r0_true, 'DisplayName', 'REAL'); hold on;
% OLS
R0_approx_ols_simple = simpleComputePiecewiseApprox(b_ols_blocks, a_ols_blocks, use_data_soc_true, RANGES);
plot(use_data_soc_true, R0_approx_ols_simple, 'DisplayName', 'OLS'); hold on;
% TLS
R0_approx_tls_simple = simpleComputePiecewiseApprox(b_tls_blocks, a_tls_blocks, use_data_soc_true, RANGES);
plot(use_data_soc_true, R0_approx_tls_simple, 'DisplayName', 'TLS'); hold on;
% TRUE APPROX
R0_approx_true_simple = simpleComputePiecewiseApprox(b_true_blocks, a_true_blocks, use_data_soc_true, RANGES);
plot(use_data_soc_true, R0_approx_true_simple, 'DisplayName', 'TRUE_Approx'); hold on;
legend show;

% Residuals calculation from the true best interpolation
final_diff_ols = sqrt(mean((R0_approx_true_simple(1:N_total) - R0_approx_ols_simple).^2));
final_diff_tls = sqrt(mean((R0_approx_true_simple(1:N_total) - R0_approx_tls_simple).^2));

fprintf("\n----------------------------------------------------------------------\n")
fprintf("OLS mean residual: %.10f\n", final_diff_ols);
fprintf("TLS mean residual: %.10f\n", final_diff_tls);