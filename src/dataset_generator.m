clear; clc; close all;

% ==========================================
% 1. USER CONFIGURATION
% ==========================================

% Electrical Parameters
CAPACITY_AH = 15.0;      % Nominal Capacity
V_MAX = 4.2;             % Max Voltage (Full)
V_MIN = 2.8;             % Cutoff Voltage (Empty)
I_PULSE = 15.0;          % Discharge Current (A)
DT = 0.5;                % Sampling step (s)

% Timing Parameters
TIME_PULSE = 5;          % Discharge duration (s)
TIME_REST = 20;          % Rest duration (s)

% --- NOISE CONFIGURATION ---
NOISE_VARIANCE = 2.0e-4;
NOISE_VARIANCE_I = 5.0e-2;
NOISE_MEAN = 0.0;

% Calculate Standard Deviation (Sigma)
NOISE_STD_DEV = sqrt(NOISE_VARIANCE);
NOISE_STD_DEV_I = sqrt(NOISE_VARIANCE_I);

% ==========================================
% 2. PHYSICAL MODELING (LOOKUP & FIT)
% ==========================================

% OCV Fit Points (Typical NMC Li-Ion)
soc_meas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0];
ocv_meas = [2.80, 3.15, 3.30, 3.50, 3.60, 3.68, 3.76, 3.85, 3.94, 4.03, 4.11, 4.16, 4.20];

% Variable R0 function (Linear Interpolation points)
r0_soc_x = [0.0,   0.1,   0.2,   0.4,   0.6,   0.8,   1.0];
r0_val_y = [0.060, 0.038, 0.028, 0.018, 0.022, 0.020, 0.015]; % Ohm

% Polynomial Fit (Degree 5)
% MATLAB returns coefficients [p1, p2... pn], similar to numpy
poly_coeffs_ocv = polyfit(soc_meas, ocv_meas, 5);
poly_coeffs_r0 = polyfit(r0_soc_x, r0_val_y, 5);

% ==========================================
% 3. SIMULATION ENGINE
% ==========================================

% Initialize Simulation Variables
current_soc = 1.0;
current_soc_meas = 1.0;
t = 0.0;
last_meas_V = 4.2;
last_meas_I = 0;

% Pre-allocate data array (estimating size to improve speed, or growing dynamically)
% Here we initialize empty for direct translation logic
data_time = [];
data_current = [];
data_v_meas = [];
data_v_clean = [];
data_soc = [];
data_ocv = [];
data_r0_true = [];
data_i_meas = [];
data_r0_meas = [];

use_data_r0_meas = [];
use_data_soc_meas = [];

fprintf('Starting simulation with Noise Variance: %e (StdDev: %.4f V)\n', NOISE_VARIANCE, NOISE_STD_DEV);

while current_soc > 0
    
    % --- DISCHARGE PHASE (PULSE) ---
    current = I_PULSE;
    steps = floor(TIME_PULSE / DT);
    first_entrance = true;
    
    for k = 1:steps
        % Get True Params
        soc_clipped = max(0.0, min(1.0, current_soc));
        ocv = polyval(poly_coeffs_ocv, soc_clipped);
        r0 = polyval(poly_coeffs_r0, soc_clipped);
        
        % Ideal Voltage
        v_clean = ocv - (r0 * current);
        
        % Generate Noise
        noise_sample = NOISE_MEAN + NOISE_STD_DEV * randn();
        noise_sample_i = NOISE_MEAN + NOISE_STD_DEV_I * randn();
        
        % Measured Values
        v_noisy = v_clean + noise_sample;
        i_noisy = current + noise_sample_i;

        current_soc_meas = current_soc_meas - (i_noisy * DT) / (CAPACITY_AH * 3600);
        
        % Calculate R0 (on step response)
        if first_entrance
            % Avoid division by zero if noise is perfectly zero (unlikely but safe to handle)
            denom = i_noisy - last_meas_I;
            if abs(denom) < 1e-9
                r0_meas = 0; 
            else
                r0_meas = abs((last_meas_V - v_noisy) / denom);
            end
            first_entrance = false;

            % Save data that will be used for TLS
            use_data_r0_meas = [use_data_r0_meas; r0_meas];
            use_data_soc_meas = [use_data_soc_meas; current_soc_meas];
        end
        
        % Store Data
        data_time = [data_time; t];
        data_current = [data_current; current];
        data_v_meas = [data_v_meas; v_noisy];
        data_v_clean = [data_v_clean; v_clean];
        data_soc = [data_soc; current_soc];
        data_ocv = [data_ocv; ocv];
        data_r0_true = [data_r0_true; r0];
        data_i_meas = [data_i_meas; i_noisy];
        data_r0_meas = [data_r0_meas; r0_meas];
        
        % Coulomb Counting
        current_soc = current_soc - (current * DT) / (CAPACITY_AH * 3600);
        t = t + DT;
        
        last_meas_V = v_noisy;
        last_meas_I = i_noisy;
        
        if ocv <= V_MIN || current_soc <= 0
            break;
        end
    end
    
    if ocv <= V_MIN || current_soc <= 0
        break;
    end
    
    % --- REST PHASE (REST) ---
    current = 0.0;
    steps = floor(TIME_REST / DT);
    first_entrance = true;
    
    for k = 1:steps
        % Get True Params
        soc_clipped = max(0.0, min(1.0, current_soc));
        ocv = polyval(poly_coeffs_ocv, soc_clipped);
        r0 = polyval(poly_coeffs_r0, soc_clipped);
        
        v_clean = ocv; % At rest V = OCV
        
        noise_sample = NOISE_MEAN + NOISE_STD_DEV * randn();
        noise_sample_i = NOISE_MEAN + NOISE_STD_DEV_I * randn();
        
        v_noisy = v_clean + noise_sample;
        % Note: Python script had a likely bug using 'noise_sample' (voltage noise) 
        % for current here. We use 'noise_sample_i' for correctness.
        i_noisy = current + noise_sample_i; 
        
        if first_entrance
            denom = i_noisy - last_meas_I;
            if abs(denom) < 1e-9
                r0_meas = 0; 
            else
                r0_meas = abs((last_meas_V - v_noisy) / denom);
            end
            first_entrance = false;

            % Save data that will be used for TLS
            use_data_r0_meas = [use_data_r0_meas; r0_meas];
            use_data_soc_meas = [use_data_soc_meas; current_soc_meas];
        end
        
        % Store Data
        data_time = [data_time; t];
        data_current = [data_current; current];
        data_v_meas = [data_v_meas; v_noisy];
        data_v_clean = [data_v_clean; v_clean];
        data_soc = [data_soc; current_soc];
        data_ocv = [data_ocv; ocv];
        data_r0_true = [data_r0_true; r0];
        data_i_meas = [data_i_meas; i_noisy];
        data_r0_meas = [data_r0_meas; r0_meas];
        
        last_meas_V = v_noisy;
        last_meas_I = i_noisy;
        
        t = t + DT; % SoC does not change at rest
    end
end

% Create Table (DataFrame equivalent)
df = table(data_time, data_current, data_v_meas, data_v_clean, ...
           data_soc, data_ocv, data_r0_true, data_i_meas, data_r0_meas, ...
           'VariableNames', {'Time_s', 'Current_A', 'Voltage_Measured_V', ...
                             'Voltage_Clean_V', 'SoC', 'OCV_True_V', ...
                             'R0_True_Ohm', 'Current_Meas', 'R0_Meas'});

% ==========================================
% 4. SAVING AND VISUALIZATION
% ==========================================

% Saving
%filename = 'dataset_batteria_noisy.csv';
%writetable(df, filename);
%fprintf('Dataset saved: %s\n', filename);

% Plot
figure('Position', [100, 100, 1000, 800]);

% Subplot 1: Voltage
ax1 = subplot(3, 1, 1);
hold on;
plot(df.Time_s, df.Voltage_Measured_V, 'b--', 'LineWidth', 0.5, 'DisplayName', 'Tensione Misurata (Noisy)');
plot(df.Time_s, df.Voltage_Clean_V, 'b', 'LineWidth', 1.5, 'DisplayName', 'Tensione Reale (Clean)');
% Transparency (alpha) is harder in standard MATLAB plots without using 'patch', 
% so we stick to line styles for clarity.
hold off;
title(sprintf('Profilo di Scarica con Rumore Gaussiano (Var=%.1e)', NOISE_VARIANCE));
ylabel('Voltage [V]');
legend('show', 'Location', 'northeast');
grid on;

% Subplot 2: Current
ax2 = subplot(3, 1, 2);
hold on;
plot(df.Time_s, df.Current_Meas, 'r--', 'DisplayName', 'Current Noise (A)');
plot(df.Time_s, df.Current_A, 'r', 'DisplayName', 'Current (A)');
hold off;
ylabel('Current [A]');
ylim([-5, 20]);
% Set y-axis text color to red
ax2.YColor = 'r';
legend('show', 'Location', 'northeast');
grid on;

% Subplot 3: Resistance
ax3 = subplot(3, 1, 3);
hold on;
plot(df.Time_s, df.R0_True_Ohm * 1000, 'g', 'DisplayName', 'True R0 (mOhm)');
plot(df.Time_s, df.R0_Meas * 1000, 'g--', 'DisplayName', 'Meas R0 (mOhm)');
hold off;
title(sprintf('Resistenza (Var=%.1e)', NOISE_VARIANCE));
ylabel('Resistance [Ohm]');
xlabel('Time [s]');
legend('show', 'Location', 'northeast');
grid on;

linkaxes([ax1, ax2, ax3], 'x');


% 2. Plot the graph
figure;               % Opens a new figure window
plot(use_data_soc_meas, use_data_r0_meas, 'b--');    % Plots x vs y. 'b-o' means blue line with circle markers

% 3. Add labels and title
title('R0 vs SOC plot');
xlabel('SOC');
ylabel('R0 ');

% 4. Add a grid
grid on;

save("dataset.mat", "use_data_r0_meas", "use_data_soc_meas")