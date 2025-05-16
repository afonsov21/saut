import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi, log, exp, atan2, sqrt
import random

def move_robot_random(robot, grid):
    # Tenta ir para frente
    new_x = robot.x + cos(robot.theta)
    new_y = robot.y + sin(robot.theta)

    if (0 <= int(new_x) < grid.shape[0] and
        0 <= int(new_y) < grid.shape[1] and
        grid[int(new_x), int(new_y)] == 0):
        robot.x = new_x
        robot.y = new_y
    else:
        # Se bateu, muda para uma direção aleatória
        robot.theta = random.uniform(0, 2 * pi)


# Ambiente simulado
def create_environment(size=20):
    grid = np.zeros((size, size))
    grid[5:15, 10] = 1
    grid[15, 5:15] = 1
    return grid

# Robô
class Robot:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

# Sensor (simples)
def inverse_range_sensor_model(robot, angle, dist, max_range, cell, l_occ=0.85, l_free=-0.4):
    # Posição esperada da célula
    cell_x, cell_y = cell
    dx = cell_x - robot.x
    dy = cell_y - robot.y
    r = sqrt((dx**2 + dy**2)) # Distância da célula ao robô
    phi = atan2(dy, dx) - robot.theta

    # Verifica se a célula está dentro do campo de visão do sensor
    if abs(phi - angle) > pi / 4:
        return 0  # Fora do campo de visão

    # Modelo de probabilidade
    if r < dist - 1:  # Região livre
        return l_free
    elif dist - 1 <= r <= dist + 1:  # Região ocupada
        return l_occ
    else:  # Fora do alcance do sensor
        return 0

# Log-odds functions
def prob_to_logodds(p):
    return log(p / (1 - p))

def logodds_to_prob(l):
    return 1 - 1 / (1 + exp(l))

# Atualiza mapa com log-odds
def update_map_logodds(logodds_map, robot, readings, max_range=10, l_occ=0.85, l_free=-0.4):
    for angle, dist in readings:
        dx = cos(robot.theta + angle)
        dy = sin(robot.theta + angle)

        # Atualiza células livres no caminho do sensor
        for r in range(1, dist):
            x = int(round(robot.x + dx * r))
            y = int(round(robot.y + dy * r))
            if 0 <= x < logodds_map.shape[0] and 0 <= y < logodds_map.shape[1]:
                logodds_map[x, y] += l_free

        # Atualiza célula ocupada (se dentro do alcance do sensor)
        if dist < max_range:
            x = int(round(robot.x + dx * dist))
            y = int(round(robot.y + dy * dist))
            if 0 <= x < logodds_map.shape[0] and 0 <= y < logodds_map.shape[1]:
                logodds_map[x, y] += l_occ
# Movimento do robô
def move_robot(robot, grid):
    new_x = robot.x + cos(robot.theta)
    new_y = robot.y + sin(robot.theta)
    if (0 <= int(new_x) < grid.shape[0] and 
        0 <= int(new_y) < grid.shape[1] and 
        grid[int(new_x), int(new_y)] == 0):
        robot.x = new_x
        robot.y = new_y
    else:
        robot.theta += pi / 2

# Visualização
def plot_maps(env, logodds_map, robot, step):
    prob_map = np.vectorize(logodds_to_prob)(logodds_map)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(env, cmap='Greys', origin='lower')
    axs[0].set_title("Mapa Real")
    axs[0].plot(robot.y, robot.x, 'ro')

    axs[1].imshow(prob_map, cmap='gray', vmin=0, vmax=1, origin='lower')
    axs[1].set_title(f"Mapa OGM (Passo {step})")
    axs[1].plot(robot.y, robot.x, 'ro')
    plt.pause(0.3)
    plt.clf()

# Simulação principal
def simulate():
    env = create_environment()
    robot = Robot(2, 2, 0)
    logodds_map = np.zeros(env.shape)

    # Criar visualização uma vez
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    img_real = axs[0].imshow(env, cmap='Greys', origin='lower')
    axs[0].set_title("Mapa Real")
    robot_dot_real, = axs[0].plot(robot.y, robot.x, 'ro')

    prob_map = np.vectorize(logodds_to_prob)(logodds_map)
    img_map = axs[1].imshow(prob_map, cmap='gray', vmin=0, vmax=1, origin='lower')
    axs[1].set_title("Mapa OGM")
    robot_dot_map, = axs[1].plot(robot.y, robot.x, 'ro')

    plt.ion()
    plt.show()

    sensor_angles = np.linspace(-pi / 4, pi / 4, 5)  # Ângulos do sensor

    for step in range(1000):
        readings = []
        for angle in sensor_angles:
            # Simula uma leitura de sensor (distância até o obstáculo ou max_range)
            for dist in range(1, 6):  # Distância máxima de 5
                x = int(round(robot.x + cos(robot.theta + angle) * dist))
                y = int(round(robot.y + sin(robot.theta + angle) * dist))
                if 0 <= x < env.shape[0] and 0 <= y < env.shape[1]:
                    if env[x, y] == 1:  # Obstáculo encontrado
                        readings.append((angle, dist))
                        break
            else:
                readings.append((angle, 5))  # Sem obstáculo no alcance máximo

        # Atualiza o mapa com base nas leituras do sensor
        update_map_logodds(logodds_map, robot, readings)

        # Move o robô aleatoriamente
        move_robot_random(robot, env)

        # Atualizar mapa com novas probabilidades
        prob_map = np.vectorize(logodds_to_prob)(logodds_map)
        img_map.set_data(prob_map)
        axs[1].set_title(f"Mapa OGM (Passo {step})")

        # Atualizar posição do robô
        robot_dot_real.set_data(robot.y, robot.x)
        robot_dot_map.set_data(robot.y, robot.x)

        plt.draw()
        plt.pause(0.1)

    plt.ioff()
    plt.show()

simulate()