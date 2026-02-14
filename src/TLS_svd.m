clear; clc; close all;

% LOAD THE DATASET
load("../res/dataset.mat")

% Sort the data based on SOC, thus avoiding overlaps due to unordered data
% ArrayA and capture the indices
[use_data_soc_meas, sortIdx] = sort(use_data_soc_meas);
% Apply the same indices to ArrayB
use_data_r0_meas = use_data_r0_meas(sortIdx);

% CONFIGURATION
N_blocks = 3;
sx = NOISE_STD_DEV_SOC; 
sy = NOISE_STD_DEV_R0; 
N = floor(length(use_data_soc_meas)/N_blocks); % Data for each block

fprintf('\nDataset data:\n\tstd_dev_x: %.8f\n\tstd_dev_y: %.8f', sx, sy);



% PLOT INIT
figure;
grid on;
colors = ['r', 'g', 'b'];

% BLOCKS 
for block_idx = 1:N_blocks
    fprintf('\n\n%s\n', repmat('-', 1, 46)); % Print a separator line
    fprintf('%s Block  %d %s\n', repmat('-', 1, 18), block_idx, repmat('-', 1, 18));
    fprintf('%s\n', repmat('-', 1, 46));
    

    %%%%%%%%%%%%%%%%%%%% 1) Get raw data partion %%%%%%%%%%%%%%%%%%%%

    % Partitionate data
    x_raw = use_data_soc_meas( (block_idx-1)*N+1 : block_idx*N );
    y_raw = use_data_r0_meas( (block_idx-1)*N+1 : block_idx*N ); 
    

    %%%%%%%%%%%%%%%%%%%%% 2) Mean centering %%%%%%%%%%%%%%%%%%%%

    % Shift the data so its center is at (0,0).
    % This allows y=ax solver to find the correct slope.
    mx = mean(x_raw);
    my = mean(y_raw);
    x = x_raw - mx;
    y = y_raw - my;


    %%%%%%%%%%%%%%%%%%%% 3) Actual calculation %%%%%%%%%%%%%%%%%%%%
    
    % 1. WEIGHTED TLS EVALUATION (The "Honest" Solver)
    % A. Whitening
    Z_w = [x./sx, y./sy];
    [U_w, S_w, V_w] = svd(Z_w, 0);
    
    % B. Slope Calculation (De-whitened)
    v_min_w = V_w(:, end); % The singular vector for the smallest singular value
    a_weighted = -v_min_w(1) / v_min_w(2) * (sy/sx);
    b_weighted = my - a_weighted * mx;
    
    % C. Reconstruct the "Clean" Data (Rank-1 Approximation)
    % This removes the noise component defined by the smallest singular value
    Z_clean_w = U_w(:,1) * S_w(1,1) * V_w(:,1)'; 
    
    % D. Map back to Physical Space
    x_fit_w = Z_clean_w(:,1) * sx; 
    y_fit_w = Z_clean_w(:,2) * sy;
    
    % E. Calculate Residuals
    dx_w = x - x_fit_w;
    dy_w = y - y_fit_w;
    
    % --- METRICS ---
    % 1. Statistical Cost (Chi-Squared): Weighted TLS minimizes this.
    cost_stat_w = mean( (dx_w./sx).^2 + (dy_w./sy).^2 ); 
    
    % 2. Geometric Cost (Euclidean): Weighted TLS ignores this.
    cost_geom_w = mean( dx_w.^2 + dy_w.^2 );



    % 2. UNWEIGHTED TLS EVALUATION (The "Overfitting" Solver)
    % A. Raw Data
    Z_u = [x, y];
    [U_u, S_u, V_u] = svd(Z_u, 0);
    
    % B. Slope
    v_min_u = V_u(:, end);
    a_unweighted = -v_min_u(1) / v_min_u(2); 
    b_unweighted = my - a_unweighted * mx;
    
    % C. Reconstruction
    Z_clean_u = U_u(:,1) * S_u(1,1) * V_u(:,1)';
    x_fit_u = Z_clean_u(:,1);
    y_fit_u = Z_clean_u(:,2);
    
    % D. Residuals
    dx_u = x - x_fit_u;
    dy_u = y - y_fit_u;
    
    % --- METRICS ---
    % 1. Statistical Cost (Chi-Squared)
    cost_stat_u = mean( (dx_u./sx).^2 + (dy_u./sy).^2 );
    
    % 2. Geometric Cost (Euclidean): Unweighted TLS minimizes this.
    cost_geom_u = mean( dx_u.^2 + dy_u.^2 );
    
    % 3. FINAL COMPARISON PRINT
    fprintf('1. STATISTICAL COST \n');
    fprintf('%s\n', repmat('-', 1, 46));
    fprintf('   Weighted TLS:   %.7f\n', cost_stat_w);
    fprintf('   Unweighted TLS: %.7f\n', cost_stat_u);
    
    fprintf('%s\n', repmat('-', 1, 46));
    fprintf('2. GEOMETRIC COST\n');
    fprintf('%s\n', repmat('-', 1, 46));
    fprintf('   Weighted TLS:   %.7f\n', cost_geom_w);
    fprintf('   Unweighted TLS: %.7f\n', cost_geom_u);
    

    %%%%%%%%%%%%%%%%%%%% 6) Data plotting %%%%%%%%%%%%%%%%%%%%

    % Reconstruct the index range for this block to plot only where data is valid
    idx_range = (block_idx-1)*N+1 : block_idx*N;
    x_seg = use_data_soc_meas(idx_range);

    % Plot SVD (ok)
    y_fit = a_weighted * x_seg + b_weighted;
    plot(x_seg, y_fit, [colors(2)], 'LineWidth', 2, 'DisplayName', sprintf('Fit Block %d (a=%.4f)', block_idx, a_weighted));
    hold on;

    % Plot SVD (wrong one)
    y_fit = a_unweighted * x_seg + b_unweighted;
    plot(x_seg, y_fit, [colors(1) '--'], 'LineWidth', 2, 'DisplayName', sprintf('Fit Block %d (a=%.4f)', block_idx, a_unweighted));
    hold on;
    
    % Info printing
    fprintf('Final TLS SVD Model: y = %.4fx + %.4f\n', a_unweighted, b_unweighted);
    %fprintf('RMS whitened:     %.10f\n',rms_x_svd);
    %fprintf('RMS not whitened: %.10f\n',rms_x_svd_not_whitened);

end

% Plot data points
plot(use_data_soc_meas, use_data_r0_meas, 'ko', 'DisplayName', 'Data', 'MarkerFaceColor', 'k', 'MarkerSize', 3);

% Finishing up the plot
legend show;
title('TLS Fitting with with SVD method');
xlabel('SOC'); ylabel('R0');