import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson, norm, uniform, gaussian_kde

# Set random seed for reproducibility
np.random.seed(42)
n_trials = 10000

# DataFrame for storing failure times
df = pd.DataFrame(columns=['motor', 'sensor', 'control', 'vehicle_failure'])

# Simulation loop
for _ in range(n_trials):
    motor_time = np.random.poisson(5000)
    sensor_time = np.random.normal(6000, 100)
    control_time = np.random.uniform(4000, 8000)
    
    # Sensor failure reduces motor lifetime
    if sensor_time < motor_time:
        motor_time -= 1000
    
    # Vehicle failure occurs when either motor or control unit fails
    vehicle_failure = min(motor_time, control_time)
    
    # Append data to DataFrame
    df.loc[len(df)] = [motor_time, sensor_time, control_time, vehicle_failure]

# Convert columns to numeric
df = df.apply(pd.to_numeric)

# Histogram of vehicle failure times
plt.figure(figsize=(8, 5))
plt.hist(df['vehicle_failure'], bins=50, density=True, color='skyblue', alpha=0.7)
plt.xlabel('Ausfallzeit des Fahrzeugs (Stunden)')
plt.ylabel('Dichte')
plt.title('Verteilung der Ausfallzeiten (Fahrzeug)')
plt.show()

# Probability of component failures given vehicle failure before 5000 hours
filtered_df = df[df['vehicle_failure'] < 5000]
p_motor_given_vehicle = (filtered_df['motor'] < 5000).mean()
p_sensor_given_vehicle = (filtered_df['sensor'] < 5000).mean()
p_control_given_vehicle = (filtered_df['control'] < 5000).mean()

print("Bedingte Wahrscheinlichkeiten:")
print(f"P(Motor fällt aus | Fahrzeug fällt vor 5000h aus) = {p_motor_given_vehicle:.4f}")
print(f"P(Sensor fällt aus | Fahrzeug fällt vor 5000h aus) = {p_sensor_given_vehicle:.4f}")
print(f"P(Steuereinheit fällt aus | Fahrzeug fällt vor 5000h aus) = {p_control_given_vehicle:.4f}")

# Correlation analysis
correlation_matrix = df.corr()
print("Korrelationsmatrix:")
print(correlation_matrix)

# Scatter plot matrix
sns.pairplot(df, diag_kind='kde')
plt.show()

# Conditional probability: Control unit failure given vehicle failure before 4000h
filtered_4000_df = df[df['vehicle_failure'] < 4000]
p_control_given_4000 = (filtered_4000_df['control'] < 4000).mean()
print(f"P(Steuereinheit fällt aus | Fahrzeug fällt vor 4000h aus) = {p_control_given_4000:.4f}")

# Fit a distribution to vehicle failure times
from scipy.stats import expon

params_expon = expon.fit(df['vehicle_failure'])
x = np.linspace(df['vehicle_failure'].min(), df['vehicle_failure'].max(), 1000)
pdf_expon_fitted = expon.pdf(x, *params_expon)

plt.figure(figsize=(8, 5))
plt.hist(df['vehicle_failure'], bins=50, density=True, alpha=0.7, label='Simulation')
plt.plot(x, pdf_expon_fitted, 'r-', label='Exponential Fit')
plt.xlabel('Ausfallzeit des Fahrzeugs (Stunden)')
plt.ylabel('Dichte')
plt.title('Anpassung der Verteilung')
plt.legend()
plt.show()

print(f"Angepasste Parameter der Exponentialverteilung: {params_expon}")

# Fit a normal distribution to vehicle failure times
params_norm = norm.fit(df['vehicle_failure'])
pdf_norm_fitted = norm.pdf(x, *params_norm)

plt.figure(figsize=(8, 5))
plt.hist(df['vehicle_failure'], bins=50, density=True, alpha=0.7, label='Simulation')
plt.plot(x, pdf_expon_fitted, 'r-', label='Exponential Fit')
plt.plot(x, pdf_norm_fitted, 'g-', label='Normal Fit')
plt.xlabel('Ausfallzeit des Fahrzeugs (Stunden)')
plt.ylabel('Dichte')
plt.title('Anpassung der Verteilung')
plt.legend()
plt.show()

print(f"Angepasste Parameter der Normalverteilung: {params_norm}")