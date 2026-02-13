%{
clear, clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Z = [x./sx y./sy];
[U,S,V] = svd(Z);
v2 = V(:,2);
as = -v2(1)/v2(2)*sy/sx; % riportiamo le misure al di fuori dello spazio normalizzato
a_stim_SVD = as;

% Prendiamo la migliore approssimazione (v1) e otteniamo cosi' lo
% scarto quadratico medio
Z1 = U(:,1)*S(1,1)*V(:,1)';
xs = Z1(:,1)*sx;
Jx = mean((x - xs).^2); 

rms_x_svd = sqrt(Jx);
T{4,1} = 'TLS SVD';
T{4,2} = mean(a_stim_SVD - at);
T{4,3} = std(a_stim_SVD - at);
T{4,4} = sqrt(mean((a_stim_SVD - at).^2));
T
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

clear; clc; close all;

% --- 1. SETUP DATA ---
load("../res/dataset.mat")

% --- CONFIGURATION ---
N_blocks = 3;
sx = NOISE_STD_DEV_I; 
sy = NOISE_STD_DEV_V; 
N = floor(length(use_data_soc_meas)/N_blocks);

% Initialize log
as_log = []; 
bs_log = []; % We need to store intercepts now

colors = ['r', 'g', 'b', 'p', 'o']; % Colors for the blocks

figure;
grid on;

for block_idx = 1:N_blocks
    fprintf('\n--- Block %d ---\n', block_idx);
    

    %%%%%%%%%%%%%%%%%%%% 1) Get Raw Data %%%%%%%%%%%%%%%%%%%%

    % Partitionate data
    x_raw = use_data_soc_meas( (block_idx-1)*N+1 : block_idx*N );
    y_raw = use_data_r0_meas( (block_idx-1)*N+1 : block_idx*N ); 
    

    %%%%%%%%%%%%%%%%%%%%% 2) Mean centering %%%%%%%%%%%%%%%%%%%%

    % We shift the data so its center is at (0,0).
    % This allows your y=ax solver to find the correct rotation (slope).
    mx = mean(x_raw);
    my = mean(y_raw);
    
    x = x_raw - mx;
    y = y_raw - my;


    %%%%%%%%%%%%%%%%%%%% 3) ACTUAL CALCULATION %%%%%%%%%%%%%%%%%%%%
    
    % -- SVD calculation --
    Z = [x./sx y./sy];
    [U,S,V] = svd(Z);
    v2 = V(:,2);
    as = -v2(1)/v2(2)*sy/sx; % riportiamo le misure al di fuori dello spazio normalizzato
    
    % Prendiamo la migliore approssimazione (v1) e otteniamo cosi' lo scarto quadratico medio
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


    %%%%%%%%%%%%%%%%%%%% 5) RESULTS RECOVERY %%%%%%%%%%%%%%%%%%%%

    % The slope 'a' is correct.
    % We must calculate 'b' (intercept) to map back to original coordinates.
    % y = a*x + b  =>  mean_y = a*mean_x + b  =>  b = mean_y - a*mean_x
    a_tls = as;
    b_tls = my - a_tls * mx;
    
    as_log = [as_log; a_tls];
    bs_log = [bs_log; b_tls];


    %%%%%%%%%%%%%%%%%%%% 6) PLOTTING %%%%%%%%%%%%%%%%%%%%

    % Reconstruct the index for this block to plot the line segment only where data exists
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