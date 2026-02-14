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
N = floor(length(use_data_soc_meas)/N_blocks);

% PLOT INIT
figure;
grid on;
colors = ['r', 'g', 'b'];

% BLOCKS 
for block_idx = 1:N_blocks
    fprintf('\n--- Block %d ---\n', block_idx);
    

    %%%%%%%%%%%%%%%%%%%% 1) Get raw data partion %%%%%%%%%%%%%%%%%%%%

    % Partitionate data
    x_raw = use_data_soc_meas( (block_idx-1)*N +1 : block_idx*N );
    y_raw = use_data_r0_meas( (block_idx-1)*N +1 : block_idx*N ); 
    

     %%%%%%%%%%%%%%%%%%%%% 2) Mean centering %%%%%%%%%%%%%%%%%%%%

    % Shift the data so its center is at (0,0).
    % This allows y=ax solver to find the correct slope.
    mean_x = mean(x_raw);
    mean_y = mean(y_raw);
    
    x = x_raw - mean_x;
    y = y_raw - mean_y;

    
    %%%%%%%%%%%%%%%%%%%% 3) Actual calculation %%%%%%%%%%%%%%%%%%%%
    
    fprintf('     | Slope:    | Cost (Chi2) Desc) | Norm: (Incr)\n');

    % Construct weight coefficients matrix (std)
    si = diag([1/sx 1/sy]);
    W = kron(si, eye(N));
    
    % Initial Guess using centered data
    as = x \ y;     
    xs = x;         
    
    % Init
    N_iter = 13; 
    as_log_block = zeros(N_iter, 1);
    bs_log_block = zeros(N_iter, 1);
    

    %%%%%%%%%%%%%%%%%%%% 3) Iterative process %%%%%%%%%%%%%%%%%%%%
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
        weighted_sq_err = sum( (dx/sx).^2 + (dy/sy).^2 );
        
        % This is just for your curiosity (will likely go UP)
        euclidean_norm = sqrt(sum(dx.^2 + dy.^2));
    
        fprintf('%4d | %.6f |  %.4f | %.4f \n', iter, as, weighted_sq_err, euclidean_norm);
        
    end
    
    %%%%%%%%%%%%%%%%%%%% 5) Finalizing the results %%%%%%%%%%%%%%%%%%%%
    
    % The slope 'a' is correct.
    % Calculate 'b' (intercept) to map back to original coordinates.
    % y = a*x + b  =>  mean_y = a*mean_x + b  =>  b = mean_y - a*mean_x
    a_tls = as;
    b_tls = mean_y - a_tls * mean_x;

    % Reconstruct the index for this block to plot the line segment only where data exists
    idx_range = (block_idx-1)*N +1 : block_idx*N;
    x_seg = use_data_soc_meas(idx_range);

    % Calculate y using y = ax + b (for every iteration)
    for i = 1:N_iter    
        y_fit = x_seg * as_log_block(i) + bs_log_block(i);
        plot(x_seg, y_fit, [colors(1) '--'], 'LineWidth', 2, 'DisplayName', sprintf('Fit Block %d (a=%.4f)', block_idx, as_log_block(i)));
        hold on;
    end    

    % Calculate and plot the final fit
    y_fit = a_tls * x_seg + b_tls;
    plot(x_seg, y_fit, [colors(2)], 'LineWidth', 2, 'DisplayName', sprintf('Final Fit %d (a=%.4f)', i, a_tls));  hold on;
    
    % Info printing
    fprintf('Final TLS Model: y = %.4fx + %.4f\n', a_tls, b_tls);
end

% --- PLOTTING ---

% Plot data points
plot(use_data_soc_meas, use_data_r0_meas, 'ko', 'DisplayName', 'Data', 'MarkerFaceColor', 'k', 'MarkerSize', 3); 

% Finishing up the plot
ylim([0 0.1])
legend show;
title('TLS Fitting with Intercept Correction');
xlabel('SOC'); ylabel('R0');