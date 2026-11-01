import numpy as np
import matplotlib.pyplot as plt


# square, only draws the square
# full, filament replacement first then draws the square
SCENARIO = 'square'

# parameters
r_m = 0.005
J_m = 5.0e-6
K_m = 0.05
R_m = 1.5
L_m = 1.2e-3
m = 0.30
mu_r = 1e-5
mu_x = 0.4
mu_y = 0.4

dt = 1e-4

T_save = 0.2
n_w = 3
T_w = 0.5
T_load = 2.0
T_low = 180
T_high = 215
T_air = 24
T_max = 260

rho = 1230
cp = 1700
h_air = 27
a = 0.7e-3
q_heat = 4 * h_air / a * (T_max - T_air)

L_start = 0.007
L_new = 1.0
e_ratio = 0.033
tol = 0.5e-3
T_move = 1.0
T_edge = 1.5

#tolerances for the corner guard 
tol_theta = 0.01     
tol_omega = 0.2      

# motor model
D = J_m + 0.5 * m * r_m**2
a1 = -(3 * mu_r + r_m**2 / 4 * (mu_x + mu_y)) / D
a2 = -(mu_r + r_m**2 / 4 * (mu_x - mu_y)) / D

A = np.array([
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, a1, a2, K_m/D, 0],
    [0, 0, a2, a1, 0, K_m/D],
    [0, 0, -K_m/L_m, 0, -R_m/L_m, 0],
    [0, 0, 0, -K_m/L_m, 0, -R_m/L_m]
])

B = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [1/L_m, 0],
    [0, 1/L_m]
])

C = np.array([
    [r_m/2, r_m/2, 0, 0, 0, 0],
    [r_m/2, -r_m/2, 0, 0, 0, 0]
])

square_modes = ['square1', 'square2', 'square3', 'square4']


def derivative(f):
    return np.gradient(f, dt, edge_order=2)


# motor angles needed to be at point (x, y)
def xy_to_theta(point):
    x, y = point
    theta_l = (x + y) / r_m
    theta_r = (x - y) / r_m
    return np.array([theta_l, theta_r])


#calculating the motor voltages
def feedforward(x, y):
    theta_l = (x + y) / r_m
    theta_r = (x - y) / r_m

    w_l = derivative(theta_l)
    w_r = derivative(theta_r)

    acc_l = derivative(w_l)
    acc_r = derivative(w_r)

    A1 = A[2:4, 2:4]
    A2 = A[2:4, 4:6]

    w = np.array([w_l, w_r])
    acc = np.array([acc_l, acc_r])

    current = np.linalg.solve(A2, acc - A1 @ w)

    X = np.array([theta_l, theta_r, w_l, w_r,
                  current[0], current[1]])

    X_dot = np.array([derivative(row) for row in X])
    U = np.linalg.pinv(B) @ (X_dot - A @ X)

    return X, U


# move
def moveto(points, T_edge):
    t = np.linspace(0, 1, int(T_edge / dt) + 1)

   
    s = 3 * t**2 - 2 * t**3

    x = []
    y = []

    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i+1]

        x.extend(x0 + s * (x1 - x0))
        y.extend(y0 + s * (y1 - y0))

    return feedforward(np.array(x), np.array(y))


# corners of the square
square_corners = [(0.095, 0.095), (0.195, 0.095), (0.195, 0.195), (0.095, 0.195)]


# plan a new movement
def plan(h, points, T_edge):
    X, h['U'] = moveto(points, T_edge)
    h['k'] = 0


# choose motor voltage
def inputs(q, h):
    if q == 'print' or q == 'H' or q == 'V':
        if h['k'] < h['U'].shape[1]:
            return h['U'][:, h['k']]

    if q == 'wipe' or q == 'return':
        if h['k'] < h['U'].shape[1]:
            return h['U'][:, h['k']]

    if q == 'approach' or q in square_modes:
        if h['k'] < h['U'].shape[1]:
            return h['U'][:, h['k']]

    return np.zeros(2)


# updating the system
def f(q, X, u, h):
    X = X + dt * (A @ X + B @ u)
    h['tau'] = h['tau'] + dt
    h['k'] = h['k'] + 1

    if q == 'print' or q in square_modes:
        velocity = C[:, 0:2] @ X[2:4]
        h['lf'] = h['lf'] - dt * e_ratio * np.linalg.norm(velocity)

    loss = 4 * h_air / a * (h['Tn'] - T_air)

    if q == 'cool':
        h['Tn'] = h['Tn'] - dt * loss / (rho * cp)

    if q == 'heat':
        h['Tn'] = h['Tn'] + dt * (q_heat - loss) / (rho * cp)

    return X


# corner guard
def corner_reached(X, h):
    theta_l, theta_r, w_l, w_r = X[0], X[1], X[2], X[3]
    target_l, target_r = h['theta_target']

    angle_ok = abs(theta_l - target_l) <= tol_theta and abs(theta_r - target_r) <= tol_theta
    stopped = abs(w_l) <= tol_omega and abs(w_r) <= tol_omega

    return angle_ok and stopped


# check when to change mode
def event(q, X, h):
    x, y = C @ X

    if q == 'approach' or q in square_modes:
        return corner_reached(X, h)

    if q == 'print':
        return h['lf'] <= 0

    if q == 'save':
        return h['tau'] >= T_save

    if q == 'H':
        return x <= 0

    if q == 'V':
        return y <= 0

    if q == 'wipe':
        return h['tau'] >= n_w * T_w

    if q == 'cool':
        return h['Tn'] <= T_low

    if q == 'load':
        return h['tau'] >= T_load

    if q == 'heat':
        return h['Tn'] >= T_high

    if q == 'return':
       
        return np.sqrt((x-h['xs'])**2 + (y-h['ys'])**2) <= tol and h['k'] >= h['U'].shape[1]

    return False


# change mode and reset values
def jump(q, X, h):
    X = X.copy()
    h['tau'] = 0

    x, y = C @ X

    # square: approach, square1, square2, square3, square4, done
    if q == 'approach' or q in square_modes:
        h['hits'].append((x, y))     

        if q == 'square4':
            return 'done', X

        if q == 'approach':
            n = 1
        else:
            n = int(q[-1]) + 1        

        
        target = square_corners[n]
        h['theta_target'] = xy_to_theta(target)
        plan(h, [(x, y), target], T_edge)
        return 'square' + str(n), X

    if q == 'print':
        h['xs'] = x
        h['ys'] = y
        return 'save', X

    if q == 'save':
        plan(h, [(x, y), (-0.002, y)], T_move)
        return 'H', X

    if q == 'H':
        w = (X[2] - X[3]) / 2
        i = (X[4] - X[5]) / 2

        X[2] = w
        X[3] = -w
        X[4] = i
        X[5] = -i

        plan(h, [(x, y), (x, -0.002)], T_move)
        return 'V', X

    if q == 'V':
        w = (X[2] + X[3]) / 2
        i = (X[4] + X[5]) / 2

        X[2] = w
        X[3] = w
        X[4] = i
        X[5] = i

        points = [(x, y), (x + 0.02, y)] * n_w
        points.append((x, y))

        plan(h, points, T_w / 2)
        return 'wipe', X

    if q == 'wipe':
        return 'cool', X

    if q == 'cool':
        return 'load', X

    if q == 'load':
        h['lf'] = L_new
        return 'heat', X

    if q == 'heat':
        plan(h, [(x, y), (h['xs'], h['ys'])], T_move)
        return 'return', X

    if q == 'return':
        h['theta_target'] = xy_to_theta(square_corners[0])
        plan(h, [(x, y), square_corners[0]], T_move)
        return 'approach', X

    return q, X


def simulate(q, X, h, t_end=60):
    t = 0

    times = []
    modes = []
    xs = []
    ys = []
    th_l = []
    th_r = []

    while t < t_end:
        if q == 'done':
            break

        u = inputs(q, h)
        X = f(q, X, u, h)
        t = t + dt

        if event(q, X, h):
            old_q = q
            q, X = jump(q, X, h)
            print(t, old_q, '->', q)

        x, y = C @ X

        times.append(t)
        modes.append(q)
        xs.append(x)
        ys.append(y)
        th_l.append(X[0])
        th_r.append(X[1])

    return (np.array(times), np.array(xs), np.array(ys),
            np.array(th_l), np.array(th_r), modes)


# initial values
h = {
    'tau': 0,
    'lf': L_start,
    'Tn': T_high,
    'xs': 0,
    'ys': 0,
    'k': 0,
    'U': np.zeros((2, 0)),
    'theta_target': xy_to_theta(square_corners[0]),
    'hits': []
}

if SCENARIO == 'full':
    corners = [(0.12, 0.28), (0.22, 0.23), (0.28, 0.03)]
    X_tri, h['U'] = moveto(corners + [corners[0]], T_edge)
    q = 'print'
    X = X_tri[:, 0]
else:
    q = 'approach'
    X = np.zeros(6)
    X[0:2] = xy_to_theta(square_corners[0])
    h['lf'] = L_new

t, x, y, th_l, th_r, modes = simulate(q, X, h)


# check the square's edge lengths and corner angles
P = np.array(h['hits'])     

edges = []
for i in range(4):
    edge = P[i+1] - P[i]
    edges.append(edge)
    print('edge', i+1, 'length:', round(np.linalg.norm(edge) * 1000, 3), 'mm')

for i in range(4):
    e1 = edges[i-1]
    e2 = edges[i]
    cos_angle = np.dot(-e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    angle = np.degrees(np.arccos(cos_angle))
    print('corner', i+1, 'angle:', round(angle, 3), 'deg')


# square graph
plt.figure()
plt.plot(x * 1000, y * 1000)
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('Square path')
plt.axis('equal')
plt.grid()
plt.show()


# motor angle graph
plt.figure()
plt.plot(t, th_l, label='theta_l')
plt.plot(t, th_r, label='theta_r')
plt.xlabel('time (s)')
plt.ylabel('motor angle (rad)')
plt.title('Motor angles')
plt.legend()
plt.grid()
plt.show()


# mode graph
mode_names = ['print', 'save', 'H', 'V', 'wipe', 'cool', 'load', 'heat', 'return',
              'approach', 'square1', 'square2', 'square3', 'square4', 'done']

mode_numbers = []

for q in modes:
    mode_numbers.append(mode_names.index(q))

plt.figure()
plt.step(t, mode_numbers)
plt.yticks(range(len(mode_names)), mode_names)
plt.xlabel('time (s)')
plt.ylabel('mode')
plt.title('Mode vs time')
plt.grid()
plt.show()
