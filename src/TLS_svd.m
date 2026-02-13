clear; clc; close all;

% LOAD THE DATASET
load("../res/dataset.mat")

% CONFIGURATION
N_blocks = 3;
sx = NOISE_STD_DEV_SOC; 
sy = NOISE_STD_DEV_R0; 
N = floor(length(use_data_soc_meas)/N_blocks); % Data for each block

fprintf('\nDataset data:\n\tstd_dev_x: %.5f\n\tstd_dev_x: %.5f', sx, sy);



% PLOT INIT
figure;
grid on;
colors = ['r', 'g', 'b'];

% BLOCKS 
for block_idx = 1:N_blocks
    fprintf('\n--- Block %d ---\n', block_idx);
    

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
    
    % -- SVD calculation --
    Z = [x./sx y./sy];
    [U,S,V] = svd(Z);
    v2 = V(:,2);
    as = -v2(1)/v2(2)*sy/sx; % Restore the original variable space (note the sigma coefficient)
    
    % From the best approximation of Z (v1) we extract the RMS
    Z1 = U(:,1)*S(1,1)*V(:,1)';
    xs = Z1(:,1)*sx;
    Jx = mean((x - xs).^2); 
    rms_x_svd = sqrt(Jx);

    % -- SVD not normalized calculation (just for comparison) --
    Z = [x y];
    [U,S,V] = svd(Z);
    v2 = V(:,2);
    as_SVD_not_whitened = -v2(1)/v2(2);
    Z1 = U(:,1)*S(1,1)*V(:,1)';
    xs = Z1(:,1);
    Jx = mean((x - xs).^2); 
    rms_x_svd_not_whitened = sqrt(Jx);


    %%%%%%%%%%%%%%%%%%%% 5) Finalizing the results %%%%%%%%%%%%%%%%%%%%

    % The slope 'a' is correct.
    % Calculate 'b' (intercept) to map back to original coordinates.
    % y = a*x + b  =>  mean_y = a*mean_x + b  =>  b = mean_y - a*mean_x
    a_tls = as;
    b_tls = my - a_tls * mx;


    %%%%%%%%%%%%%%%%%%%% 6) Data plotting %%%%%%%%%%%%%%%%%%%%

    % Reconstruct the index range for this block to plot only where data is valid
    idx_range = (block_idx-1)*N+1 : block_idx*N;
    x_seg = use_data_soc_meas(idx_range);

    % Plot SVD (ok)
    y_fit = a_tls * x_seg + b_tls;
    plot(x_seg, y_fit, [colors(block_idx)], 'LineWidth', 2, 'DisplayName', sprintf('Fit Block %d (a=%.4f)', block_idx, a_tls));
    hold on;

    % Plot SVD (wrong one)
    y_fit = as_SVD_not_whitened * x_seg + b_tls;
    plot(x_seg, y_fit, [colors(block_idx) '--'], 'LineWidth', 2, 'DisplayName', sprintf('Fit Block %d (a=%.4f)', block_idx, a_tls));
    hold on;
    
    % Info printing
    fprintf('Final TLS SVD Model: y = %.4fx + %.4f\n', a_tls, b_tls);
    fprintf('RMS whitened:     %.4f\n',rms_x_svd);
    fprintf('RMS not whitened: %.4f\n',rms_x_svd_not_whitened);

end

% Plot data points
plot(use_data_soc_meas, use_data_r0_meas, 'ko', 'DisplayName', 'Data', 'MarkerFaceColor', 'k', 'MarkerSize', 3);

% Finishing up the plot
legend show;
title('TLS Fitting with Intercept Correction');
xlabel('SOC'); ylabel('R0');