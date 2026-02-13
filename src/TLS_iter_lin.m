clear; clc; close all;

% LOAD THE DATASET
load("../res/dataset.mat")

% CONFIGURATION
N_blocks = 3;
sx = NOISE_STD_DEV_SOC; 
sy = NOISE_STD_DEV_R0; 
N = floor(length(use_data_soc_meas)/N_blocks);

% INIT HISTORY VECTORS
as_log = []; 
bs_log = [];

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
    y_raw = use_data_r0_meas( (block_idx-1)* N+1 : block_idx*N ); 
    

     %%%%%%%%%%%%%%%%%%%%% 2) Mean centering %%%%%%%%%%%%%%%%%%%%

    % Shift the data so its center is at (0,0).
    % This allows y=ax solver to find the correct slope.
    mx = mean(x_raw);
    my = mean(y_raw);
    
    x = x_raw - mx;
    y = y_raw - my;

    
    %%%%%%%%%%%%%%%%%%%% 3) Actual calculation %%%%%%%%%%%%%%%%%%%%
    
    fprintf('Iter |  Slope (a)  | Residual Norm\n');
    fprintf('-----------------------------------\n');

    % Construct weight coefficients matrix (std)
    si = diag([1/sx 1/sy]);
    W = kron(si, eye(N));
    
    % Initial Guess using centered data
    as = x \ y;     
    xs = x;         
    
    % Init
    N_iter = 4; 
    as_log_block = zeros(N_iter);
    

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
        
        res_norm = norm(Wd);
        res_norm_expl = sqrt(sum(Wd.^2));
        fprintf('%4d | % .6f | %.5f (%.5f)\n', iter, as, res_norm, res_norm_expl);
        
    end
    
    %%%%%%%%%%%%%%%%%%%% 5) Finalizing the results %%%%%%%%%%%%%%%%%%%%
    
    % The slope 'a' is correct.
    % Calculate 'b' (intercept) to map back to original coordinates.
    % y = a*x + b  =>  mean_y = a*mean_x + b  =>  b = mean_y - a*mean_x
    a_tls = as;
    b_tls = my - a_tls * mx;
    
    as_log = [as_log; a_tls];
    bs_log = [bs_log; b_tls];

    % Reconstruct the index for this block to plot the line segment only where data exists
    idx_range = (block_idx-1)*N+1 : block_idx*N;
    x_seg = use_data_soc_meas(idx_range);

    % Calculate y using y = ax + b (for every iteration)
    for i = 1:N_iter    
        y_fit = as_log_block(i) * x_seg + bs_log(block_idx);
        plot(x_seg, y_fit, [colors(block_idx) '--'], 'LineWidth', 2, 'DisplayName', sprintf('Fit Block %d (a=%.4f)', block_idx, as_log_block(i)));
        hold on;
    end    

    % Calculate and plot the final fit
    y_fit = a_tls * x_seg + b_tls;
    plot(x_seg, y_fit, [colors(block_idx)], 'LineWidth', 2, 'DisplayName', sprintf('Final Fit %d (a=%.4f)', i, a_tls));  hold on;
    
    % Info printing
    fprintf('Final TLS Model: y = %.4fx + %.4f\n', a_tls, b_tls);
end

% --- PLOTTING ---

% Plot data points
plot(use_data_soc_meas, use_data_r0_meas, 'ko', 'DisplayName', 'Data', 'MarkerFaceColor', 'k', 'MarkerSize', 3); 

% Finishing up the plot
legend show;
title('TLS Fitting with Intercept Correction');
xlabel('SOC'); ylabel('R0');