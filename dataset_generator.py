import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURAZIONE UTENTE
# ==========================================

# Parametri Elettrici
CAPACITY_AH = 15.0       # Capacità nominale
V_MAX = 4.2              # Tensione massima (Full)
V_MIN = 2.8              # Tensione di cutoff (Empty)
I_PULSE = 15.0           # Corrente di scarica (A)
DT = 0.1                 # Passo di campionamento (s) - Alta frequenza per vedere bene il rumore

# Parametri Temporali
TIME_PULSE = 6          # Durata scarica (s)
TIME_REST = 12          # Durata riposo (s)

# --- CONFIGURAZIONE RUMORE ---
# Qui imposti la varianza desiderata per il rumore di misura
# Esempio: 1e-4 V^2 corrisponde a una deviazione standard di 0.01 V (10mV)
NOISE_VARIANCE = 2.0e-4
NOISE_MEAN = 0.0

# Calcolo Deviazione Standard (Sigma) necessaria per np.random.normal
NOISE_STD_DEV = np.sqrt(NOISE_VARIANCE)

# ==========================================
# 2. MODELLAZIONE FISICA (LOOKUP & FIT)
# ==========================================

# Punti per fit OCV (NMC Li-Ion Tipica)
soc_meas = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
ocv_meas = np.array([2.80, 3.15, 3.30, 3.50, 3.60, 3.68, 3.76, 3.85, 3.94, 4.03, 4.11, 4.16, 4.20])

# Fit Polinomiale (Grado 5 per smooth curve senza oscillazioni eccessive)
poly_coeffs = np.polyfit(soc_meas, ocv_meas, 5)
ocv_poly_func = np.poly1d(poly_coeffs)

# Funzione R0 variabile (interpolazione lineare)
# R0 sale quando la batteria è scarica (SoC < 0.2) o piena (SoC > 0.9)
r0_soc_x = np.array([0.0,   0.1,   0.2,   0.8,   1.0])
r0_val_y = np.array([0.025, 0.012, 0.008, 0.008, 0.010]) # Ohm
#r0_soc_x = np.array([0.0,     0.1,   0.2,   0.4,      0.6,    0.8,   1.0])
#r0_val_y = np.array([0.025, 0.012, 0.020, 0.015,    0.025,   0.40,  0.60]) # Ohm

def get_true_params(soc):
    """Restituisce OCV e R0 ideali per un dato SoC"""
    # Clip SoC per sicurezza numerica nei polinomi
    soc = np.clip(soc, 0.0, 1.0)

    ocv = ocv_poly_func(soc)
    r0 = np.interp(soc, r0_soc_x, r0_val_y)
    return ocv, r0

# ==========================================
# 3. MOTORE DI SIMULAZIONE
# ==========================================

def run_simulation_with_noise():
    # Inizializzazione
    current_soc = 1.0
    t = 0.0
    data = []

    print(f"Avvio simulazione con Varianza Rumore: {NOISE_VARIANCE} (StdDev: {NOISE_STD_DEV:.4f} V)")

    while current_soc > 0:

        # --- FASE SCARICA (PULSE) ---
        current = I_PULSE
        steps = int(TIME_PULSE / DT)

        for _ in range(steps):
            ocv, r0 = get_true_params(current_soc)

            # Tensione ideale (Modello resistivo puro)
            v_clean = ocv - (r0 * current)

            # Generazione Rumore (Gaussian White Noise)
            noise_sample = np.random.normal(NOISE_MEAN, NOISE_STD_DEV)

            # Tensione "Misurata" (con rumore)
            v_noisy = v_clean + noise_sample

            data.append([t, current, v_noisy, v_clean, current_soc, ocv, r0])

            # Integrazione SoC (Coulomb Counting)
            # dSoC = - I * dt / Q_tot (As)
            current_soc -= (current * DT) / (CAPACITY_AH * 3600)
            t += DT

            if ocv <= V_MIN or current_soc <= 0: break

        if ocv <= V_MIN or current_soc <= 0: break

        # --- FASE RIPOSO (REST) ---
        current = 0.0
        steps = int(TIME_REST / DT)

        for _ in range(steps):
            ocv, r0 = get_true_params(current_soc)

            v_clean = ocv # A riposo V = OCV
            noise_sample = np.random.normal(NOISE_MEAN, NOISE_STD_DEV)
            v_noisy = v_clean + noise_sample

            data.append([t, current, v_noisy, v_clean, current_soc, ocv, r0])

            t += DT # SoC non cambia a riposo

    columns = ['Time_s', 'Current_A', 'Voltage_Measured_V', 'Voltage_Clean_V', 'SoC', 'OCV_True_V', 'R0_True_Ohm']
    return pd.DataFrame(data, columns=columns)

# ==========================================
# 4. SALVATAGGIO E VISUALIZZAZIONE
# ==========================================

df = run_simulation_with_noise()

# Salvataggio
filename = 'dataset_batteria_noisy.csv'
df.to_csv(filename, index=False)
print(f"Dataset salvato: {filename}")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Grafico 1: Tensione con focus sul rumore
ax1.set_title(f'Profilo di Scarica con Rumore Gaussiano (Var={NOISE_VARIANCE})')
ax1.plot(df['Time_s'], df['Voltage_Measured_V'], color='gray', alpha=0.6, label='Tensione Misurata (Noisy)', linewidth=0.5)
ax1.plot(df['Time_s'], df['Voltage_Clean_V'], color='blue', label='Tensione Reale (Clean)', linewidth=1.5)
ax1.set_ylabel('Voltage [V]')
ax1.legend(loc='upper right')
ax1.grid(True)

# Zoom su un singolo gradino per vedere bene il rumore
# Prendiamo un intervallo arbitrario a metà simulazione se esiste
mid_idx = len(df) // 2
zoom_range = 200 # campioni
if len(df) > zoom_range:
    # Inseriamo un inset plot (piccolo grafico dentro il grafico)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax1, width="30%", height="30%", loc='lower left', borderpad=4)

    start = mid_idx
    end = mid_idx + int(60/DT) # 60 secondi
    axins.plot(df['Time_s'][start:end], df['Voltage_Measured_V'][start:end], 'gray', alpha=0.7)
    axins.plot(df['Time_s'][start:end], df['Voltage_Clean_V'][start:end], 'b')
    axins.set_title("Zoom su un gradino")
    axins.grid(True)

# Grafico 2: Corrente e R0
ax2.plot(df['Time_s'], df['Current_A'], 'r', label='Current (A)')
ax2.set_ylabel('Current [A]', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.grid(True, alpha=0.3)

# Secondo asse Y per R0
ax2_r = ax2.twinx()
ax2_r.plot(df['Time_s'], df['R0_True_Ohm']*1000, 'g--', label='True R0 (mOhm)')
ax2_r.set_ylabel('R0 [mΩ]', color='g')
ax2_r.tick_params(axis='y', labelcolor='g')
ax2.set_xlabel('Time [s]')

plt.tight_layout()
plt.show()
