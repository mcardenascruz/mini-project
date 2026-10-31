import numpy as np
import matplotlib.pyplot as plt

# Parameters
rho = 1230        
cp = 1700         
k = 0.18        
a = 0.7e-3        
h_air = 27        
h_plate = 230       
T_nozzle = 215    
T_plate = 53      
T_air = 24         
v = 0.06        
L = 0.15        

# Discretization constants
N = 100                 
ds = L / N              
dt = 0.001              
t_print = L / v         

T = np.zeros(N)         
printed = 0             

# loop settings
t = 0
times = []
T_first = []            
T_last = []             

while True:
    # New filament
    if t < t_print:
        new_printed = int(v * t / ds) + 1
        for i in range(printed, new_printed):
            T[i] = T_nozzle
        printed = new_printed

    T_new = T.copy()

    for i in range(printed):
        # left
        if i > 0:
            T_left = T[i - 1]
        elif t < t_print:
            T_left = T[i]           
        else:
            T_left = T[N - 1]       
        # right
        if i < printed - 1:
            T_right = T[i + 1]
        elif t < t_print:
            T_right = T_nozzle      
        else:
            T_right = T[0]          

        # PDE
        conduction = k * (T_left - 2 * T[i] + T_right) / ds**2
        loss = 3 * h_air / a * (T[i] - T_air) + h_plate / a * (T[i] - T_plate)
        T_new[i] = T[i] + dt * (conduction - loss) / (rho * cp)

    if t < t_print:
        T_new[printed - 1] = T_nozzle

    T = T_new
    t = t + dt

    times.append(t)
    T_first.append(T[0])
    T_last.append(T[printed - 1])

    if t > t_print and max(T) - min(T) < 0.001:
        break

print("Printing takes", round(t_print, 2), "s")
print("Temperature is uniform after", round(t, 2), "s")
print("Final temperature is about", round(T.mean(), 1), "C")

# plot
plt.plot(times, T_first, label="first piece printed")
plt.plot(times, T_last, label="last piece printed")
plt.xlabel("time [s]")
plt.ylabel("temperature [C]")
plt.legend()
plt.show()