import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lsim

# parameters 
r_m  = 0.005  
J_m  = 5.0e-6    
K_m  = 0.05     
R_m  = 1.5       
L_m  = 1.2e-3    
m    = 0.30      
mu_r = 1e-5       
mu_x = 0.4        
mu_y = 0.4        

dt = 1e-4        

# state-space matrices (Q1.2)
D  = J_m + 0.5 * m * r_m**2
a1 = -(3 * mu_r + r_m**2 / 4 * (mu_x + mu_y)) / D
a2 = -(mu_r + r_m**2 / 4 * (mu_x - mu_y)) / D

A = np.array([
    [0, 0, 1,            0,            0,            0           ],
    [0, 0, 0,            1,            0,            0           ],
    [0, 0, a1,           a2,           K_m / D,      0           ],
    [0, 0, a2,           a1,           0,            K_m / D     ],
    [0, 0, -K_m / L_m,   0,            -R_m / L_m,   0           ],
    [0, 0, 0,            -K_m / L_m,   0,            -R_m / L_m  ],
])

B = np.array([
    [0,       0      ],
    [0,       0      ],
    [0,       0      ],
    [0,       0      ],
    [1 / L_m, 0      ],
    [0,       1 / L_m],
])

C = np.array([
    [r_m / 2,  r_m / 2, 0, 0, 0, 0],
    [r_m / 2, -r_m / 2, 0, 0, 0, 0],
])


def d(f):
    return np.gradient(f, dt, edge_order=2)                     


def run(t, x_d, y_d, name):
    th_l = (x_d + y_d) / r_m
    th_r = (x_d - y_d) / r_m

    w_l = d(th_l)
    w_r = d(th_r)

    acc = np.array([d(w_l), d(w_r)])

    A1 = A[2:4, 2:4]
    A2 = A[2:4, 4:6]

    w = np.array([w_l, w_r])
    i = np.linalg.solve(A2, acc - np.dot(A1, w))

    X = np.array([th_l, th_r, w_l, w_r, i[0], i[1]])
    X_dot = np.array([d(row) for row in X])

    u = np.linalg.pinv(B) @ (X_dot - np.dot(A, X))

    notused, out, notused = lsim(
        (A, B, C, np.zeros((2, 2))),
        u.T, t, X0=X[:, 0]
    )

    x = out[:, 0]
    y = out[:, 1]

    error = np.sqrt((x - x_d)**2 + (y - y_d)**2)

    plt.figure()
    plt.plot(x_d * 1000, y_d * 1000, '--', label='wanted path')
    plt.plot(x * 1000, y * 1000, label='actual path')
    plt.title(name + 'path')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.legend()
    plt.grid()
    plt.savefig(name + '.png')

    plt.figure()
    plt.plot(t, x * 1000, label='x')
    plt.plot(t, y * 1000, label='y')
    plt.title(name + ' position vs time')
    plt.xlabel('time (s)')
    plt.ylabel('position (mm)')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(t, u[0], label='left motor')
    plt.plot(t, u[1], label='right motor')
    plt.title(name + ' - motor voltages')
    plt.xlabel('time (s)')
    plt.ylabel('voltage (V)')
    plt.legend()
    plt.grid()


    plt.figure()
    plt.plot(t, error * 1000000)
    plt.title(name + 'tracking error')
    plt.xlabel('time (s)')
    plt.ylabel('error (micrometres)')
    plt.grid()





# ellipse
T = 2
t = np.arange(0, 2*T, dt)

x_d = 0.25 + 0.12 * np.cos(2*np.pi*t/T)
y_d = 0.25 + 0.15 * np.sin(2*np.pi*t/T)

run(t, x_d, y_d, 'ellipse')


# triangle
corners = [(0.12, 0.28), (0.22, 0.23), (0.28, 0.03)]
T_edge = 1.5

tau = np.arange(0, 1, dt/T_edge)
s = tau - np.sin(2*np.pi*tau)/(2*np.pi)

x_list = []
y_list = []

for lap in range(2):
    for k in range(3):
        x0, y0 = corners[k]
        x1, y1 = corners[(k+1) % 3]

        x_list.append(x0 + s*(x1-x0))
        y_list.append(y0 + s*(y1-y0))

x_d = np.concatenate(x_list)
y_d = np.concatenate(y_list)
t = np.arange(len(x_d)) * dt

run(t, x_d, y_d, 'triangle')

plt.show()
