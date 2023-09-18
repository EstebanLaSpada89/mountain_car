import gym
import numpy as np
import pickle

# Constantes
NUM_BINS_POSICION = 20
NUM_BINS_VELOCIDAD = 20
TASA_APRENDIZAJE_ALPHA = 0.9
FACTOR_DESCUENTO_GAMMA = 0.9


class Acciones:
    IZQUIERDA = 0
    NEUTRAL = 1
    DERECHA = 2


def ejecutar(episodios, entrenamiento=True, renderizar=False):
    # Define EPSILON_DECAY_RATE aquí basado en episodios
    EPSILON_DECAY_RATE = 2 / episodios

    # Crea el entorno MountainCar
    env = gym.make('MountainCar-v0',
                   render_mode='human' if renderizar else None)

    # Divide el espacio de observación en segmentos para discretizarlo
    espacios_posicion = np.linspace(
        env.observation_space.low[0], env.observation_space.high[0], NUM_BINS_POSICION)
    espacios_velocidad = np.linspace(
        env.observation_space.low[1], env.observation_space.high[1], NUM_BINS_VELOCIDAD)

    if entrenamiento:
        # Inicializa la tabla Q si estamos en modo de entrenamiento
        q = np.zeros((len(espacios_posicion), len(
            espacios_velocidad), env.action_space.n))
    else:
        # Carga la tabla Q desde un archivo si no estamos en modo de entrenamiento
        with open('mountain_car.pkl', 'rb') as f:
            q = pickle.load(f)

    epsilon = 1
    rng = np.random.default_rng()

    for i in range(episodios):
        # Reinicia el entorno para un nuevo episodio
        estado = env.reset()[0]
        estado_p = np.digitize(estado[0], espacios_posicion)
        estado_v = np.digitize(estado[1], espacios_velocidad)
        terminado = False

        while not terminado:
            if entrenamiento and rng.random() < epsilon:
                # Elije una acción aleatoria en modo de entrenamiento con probabilidad epsilon
                accion = env.action_space.sample()
            else:
                # Elije la acción con el valor Q máximo en el estado actual
                accion = np.argmax(q[estado_p, estado_v, :])

            nuevo_estado, recompensa, terminado, _, _ = env.step(accion)
            nuevo_estado_p = np.digitize(nuevo_estado[0], espacios_posicion)
            nuevo_estado_v = np.digitize(nuevo_estado[1], espacios_velocidad)

            if entrenamiento:
                # Actualiza la tabla Q en modo de entrenamiento
                q[estado_p, estado_v, accion] += TASA_APRENDIZAJE_ALPHA * (
                    recompensa + FACTOR_DESCUENTO_GAMMA *
                    np.max(q[nuevo_estado_p, nuevo_estado_v, :]) -
                    q[estado_p, estado_v, accion]
                )

            estado = nuevo_estado
            estado_p = nuevo_estado_p
            estado_v = nuevo_estado_v

        epsilon = max(epsilon - EPSILON_DECAY_RATE, 0)

    env.close()

    if entrenamiento:
        # Guarda la tabla Q en un archivo si estamos en modo de entrenamiento
        with open('mountain_car.pkl', 'wb') as f:
            pickle.dump(q, f)


if __name__ == '__main__':
    # Ejecuta el entorno MountainCar con 10 episodios en modo de prueba
    ejecutar(10, entrenamiento=False, renderizar=True)

    # Ejemplo de entrenamiento con 500 episodios (comentado para evitar la ejecución)
    # ejecutar(500, entrenamiento=True, renderizar=False)
