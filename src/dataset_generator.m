clear; clc; close all;


%%%%%%%%%%%%%%%%%%%% 1) CONFIGURATION %%%%%%%%%%%%%%%%%%%%

% Simulation parameters
CAPACITY_AH = 15.0;      % Nominal Capacity
V_MAX = 4.2;             % Max Voltage (Full)
V_MIN = 2.8;             % Cutoff Voltage (Empty)
I_PULSE = 15.0;          % Discharge Current (A)
DT = 0.5;                % Sampling step (s)

% Timing parameters
TIME_PULSE = 20;          % Discharge duration (s)
TIME_REST = 20;          % Rest duration (s)

% Noise
NOISE_VARIANCE_V = 1.0e-3;
NOISE_VARIANCE_I = 5.0e-1;
NOISE_MEAN = 0.0;

% Calculate Standard Deviation
NOISE_STD_DEV_V = sqrt(NOISE_VARIANCE_V);
NOISE_STD_DEV_I = sqrt(NOISE_VARIANCE_I);


%%%%%%%%%%%%%%%%%%%% 2) PHYSICAL MODELING %%%%%%%%%%%%%%%%%%%%

% OCV Fit Points (Li-Ion)
soc_meas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0];
ocv_meas = [2.80, 3.15, 3.30, 3.50, 3.60, 3.68, 3.76, 3.85, 3.94, 4.03, 4.11, 4.16, 4.20];

% Variable R0 function (Linear Interpolation points)
r0_soc_x = [0.0,   0.1,   0.2,   0.4,   0.6,   0.8,   1.0];
r0_val_y = [0.080, 0.048, 0.028, 0.020, 0.032, 0.025, 0.020]; % Ohm

% Polynomial Fit (MATLAB returns coefficients [p1, p2... pn])
poly_coeffs_ocv = polyfit(soc_meas, ocv_meas, 4);
poly_coeffs_r0 = polyfit(r0_soc_x, r0_val_y, 4);


%%%%%%%%%%%%%%%%%%%% 3) SIMULATION INIT %%%%%%%%%%%%%%%%%%%%

% Initialize simulation variables
current_soc = 1.0;
current_soc_meas = 1.0;
current_soc_true = 1.0;
t = 0.0;
last_meas_V = 4.2;
last_meas_I = 0;

% Data arrays
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
use_data_r0_true = [];
use_data_soc_meas = [];
use_data_soc_true = [];

fprintf('Starting simulation with Noise Variance: %e (StdDev: %.4f V)\n', NOISE_VARIANCE_V, NOISE_STD_DEV_V);

% Main loop
while current_soc > 0

    %%%%%%%%%%%%%%%%%%%% 4) DISCHARGE PHASE (PULSE) %%%%%%%%%%%%%%%%%%%%

    i_true = I_PULSE;
    N_steps = floor(TIME_PULSE / DT);
    first_entrance = true;

    for k = 1:N_steps
        % Get true params
        soc_clipped = max(0.0, min(1.0, current_soc));
        ocv = polyval(poly_coeffs_ocv, soc_clipped);
        r0 = polyval(poly_coeffs_r0, soc_clipped);

        % Ideal voltage
        v_true = ocv - (r0 * i_true);

        % Generate noise
        noise_sample_v = NOISE_MEAN + NOISE_STD_DEV_V * randn();
        noise_sample_i = NOISE_MEAN + NOISE_STD_DEV_I * randn();

        % Measured values
        v_meas = v_true + noise_sample_v;
        i_meas = i_true + noise_sample_i;
        current_soc_true = current_soc_true - (i_true * DT) / (CAPACITY_AH * 3600);

        % SOC imprecision
        v_sensor_reading = ocv + NOISE_STD_DEV_V * randn();
        current_soc_meas = interp1(ocv_meas, soc_meas, v_sensor_reading, 'linear', 'extrap'); % Interp OCV -> SOC
        
        % 3. Clip it to stay realistic (0 to 1)
        current_soc_meas = max(0.0, min(1.0, current_soc_meas));

        % Calculate R0 (on step response)
        if first_entrance
            % Avoid division by zero if noise is perfectly zero
            denom = i_meas - last_meas_I;
            if abs(denom) < 1e-9
                r0_meas = 0;
            else
                r0_meas = abs((last_meas_V - v_meas) / denom);
            end

            % Save data that will be used for TLS
            use_data_r0_meas = [use_data_r0_meas; r0_meas];
            use_data_r0_true = [use_data_r0_true; r0];

            use_data_soc_meas = [use_data_soc_meas; current_soc_meas];
            use_data_soc_true = [use_data_soc_true; current_soc_true];

            first_entrance = false;
        end

        % Store Data
        data_time = [data_time; t];
        data_current = [data_current; i_true];
        data_v_meas = [data_v_meas; v_meas];
        data_v_clean = [data_v_clean; v_true];
        data_soc = [data_soc; current_soc];
        data_ocv = [data_ocv; ocv];
        data_r0_true = [data_r0_true; r0];
        data_i_meas = [data_i_meas; i_meas];
        data_r0_meas = [data_r0_meas; r0_meas];

        % Coulomb Counting
        current_soc = current_soc - (i_true * DT) / (CAPACITY_AH * 3600);
        t = t + DT;

        % Save for next rising/falling edge (delta calculation)
        last_meas_V = v_meas;
        last_meas_I = i_meas;

        % Exit condition
        if ocv <= V_MIN || current_soc <= 0
            break;
        end
    end

    % Exit condition
    if ocv <= V_MIN || current_soc <= 0
        break;
    end

    %%%%%%%%%%%%%%%%%%%% 5) REST PHASE (REST) %%%%%%%%%%%%%%%%%%%%

    i_true = 0.0;
    N_steps = floor(TIME_REST / DT);
    first_entrance = true;

    for k = 1:N_steps
        % Get true params
        soc_clipped = max(0.0, min(1.0, current_soc));
        ocv = polyval(poly_coeffs_ocv, soc_clipped);
        r0 = polyval(poly_coeffs_r0, soc_clipped);

        v_true = ocv; % At rest V = OCV (no current)

        noise_sample_v = NOISE_MEAN + NOISE_STD_DEV_V * randn();
        noise_sample_i = NOISE_MEAN + NOISE_STD_DEV_I * randn();

        v_meas = v_true + noise_sample_v;
        i_meas = i_true + noise_sample_i;

        if first_entrance
            denom = i_meas - last_meas_I;
            if abs(denom) < 1e-9
                r0_meas = 0;
            else
                r0_meas = abs((last_meas_V - v_meas) / denom);
            end
            first_entrance = false;

            % Save data that will be used for TLS
            use_data_r0_meas = [use_data_r0_meas; r0_meas];
            use_data_r0_true = [use_data_r0_true; r0];

            use_data_soc_meas = [use_data_soc_meas; current_soc_meas];
            use_data_soc_true = [use_data_soc_true; current_soc_true];
        end

        % Store Data
        data_time = [data_time; t];
        data_current = [data_current; i_true];
        data_v_meas = [data_v_meas; v_meas];
        data_v_clean = [data_v_clean; v_true];
        data_soc = [data_soc; current_soc];
        data_ocv = [data_ocv; ocv];
        data_r0_true = [data_r0_true; r0];
        data_i_meas = [data_i_meas; i_meas];
        data_r0_meas = [data_r0_meas; r0_meas];

        last_meas_V = v_meas;
        last_meas_I = i_meas;

        t = t + DT; % SoC does not change at rest
    end
end

% Create Table (DataFrame equivalent)
df = table(data_time, data_current, data_v_meas, data_v_clean, ...
           data_soc, data_ocv, data_r0_true, data_i_meas, data_r0_meas, ...
           'VariableNames', {'Time_s', 'Current_A', 'Voltage_Measured_V', ...
                             'Voltage_Clean_V', 'SoC', 'OCV_True_V', ...
                             'R0_True_Ohm', 'Current_Meas', 'R0_Meas'});


%%%%%%%%%%%%%%%%%%%% 6) VISUALIZATION %%%%%%%%%%%%%%%%%%%%

% Plot
figure('Position', [100, 100, 1000, 800]);

% Voltage
ax1 = subplot(3, 1, 1);
hold on;
plot(df.Time_s, df.Voltage_Measured_V, 'b--', 'LineWidth', 0.5, 'DisplayName', 'Tensione Misurata (Noisy)');
plot(df.Time_s, df.Voltage_Clean_V, 'b', 'LineWidth', 1.5, 'DisplayName', 'Tensione Reale (Clean)');
hold off;
title(sprintf('Profilo di Scarica con Rumore Gaussiano (Var=%.1e)', NOISE_VARIANCE_V));
ylabel('Voltage [V]');
legend('show', 'Location', 'northeast');
grid on;

% Current
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

% R0
ax3 = subplot(3, 1, 3);
hold on;
plot(df.Time_s, df.R0_True_Ohm * 1000, 'g', 'DisplayName', 'True R0 (mOhm)');
plot(df.Time_s, df.R0_Meas * 1000, 'g--', 'DisplayName', 'Meas R0 (mOhm)');
hold off;
title(sprintf('Resistenza (Var=%.1e)', NOISE_VARIANCE_V));
ylabel('Resistance [Ohm]');
xlabel('Time [s]');
legend('show', 'Location', 'northeast');
grid on;

linkaxes([ax1, ax2, ax3], 'x');

% Std deviation calculation (thanks to the know value)
diff_soc = use_data_soc_true - use_data_soc_meas;
diff_squared_soc = diff_soc.^2;
NOISE_STD_DEV_SOC = sqrt(mean(diff_squared_soc));

diff_r0 = use_data_r0_true - use_data_r0_meas;
diff_squared_r0 = diff_r0.^2;
NOISE_STD_DEV_R0 = sqrt(mean(diff_squared_r0));

figure('Position', [100, 200, 1000, 800]);
ax1 = subplot(2, 1, 1);
plot(diff_soc, 'magenta'); hold on;
xlim([0, length(use_data_soc_meas)]);
ylabel('SOC residual (meas - true)');

a2 = subplot(2, 1, 2);
plot(diff_r0, 'green');
xlim([0, length(use_data_soc_meas)]);
ylabel('R0 residual (meas - true)');


% Save to .mat in order to load it later
save("../res/dataset.mat", "use_data_r0_meas", "use_data_soc_meas", "NOISE_STD_DEV_V", "NOISE_STD_DEV_I", "NOISE_STD_DEV_SOC", "NOISE_STD_DEV_R0", "use_data_r0_true", "use_data_soc_true")
