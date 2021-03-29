import cv2
import random
import numpy as np

class PSO:

    def __init__(self, ref_image: np.array,landscape: np.array,
                 W: float, C1 :float, C2: float,
                 nb_of_particles: int,
                 max_iter_before_explosion:int, max_iter_without_gbest_update: int,max_iter_before_termination: int):

        self.gbest_value = 0
        self.gbest_position = 0
        self.W = W
        self.c1 = C1
        self.c2 = C2
        self.nb_of_particles = nb_of_particles
        self.ref_image = ref_image
        self.landscape = landscape
        self.nb_of_channels = ref_image.shape[-1]
        self.max_x, self.max_y = landscape.shape[:2]
        self.min_x, self.min_y = [0,0]
        self.min_s = 0.5
        self.max_s = 2
        self.min_r = -360
        self.max_r = 360
        self.particles = self.init_with_random_values()
        self.iteration = 0
        self.iteration_since_gbest_update = 0
        self.max_iter_without_gbest_update = 0
        self.max_iter_before_termination = max_iter_before_termination
        self.max_iter_before_explosion = max_iter_before_explosion

    def __call__(self) -> None:
        """
        Things that will be returned after calling on object ()
        :return:
        """
        pass


    def init_with_random_values(self) -> list:
        """
        Create particles with random values
        :return: List of particles
        """
        particles = []

        for i in range(0, self.nb_of_particles):
            x = random.randint(self.min_x, self.max_x)
            y = random.randint(self.min_y, self.max_y)
            s = random.randint(self.min_s, self.max_s)
            r = random.randint(self.min_r, self.max_r)

            particles.append(Particle(x, y, s, r))

        return particles

    def update_particles_velocity(self) -> None:
        """
        Updating particles velocity
        :return:
        """
        pass

    def update_particles_position(self) -> None:
        """
        Updating particles position
        :return:
        """
        pass

    def evaluate_fitness(self) -> None:
        """
        Calculating fitnness function and choosing best
        :return:
        """
        pass

    def explode(self) -> None:
        """
        Moving particles to random position
        :return:
        """
        pass

    def should_terminate(self) -> bool:
        """
        Checking if termination requirements are met
        :return:
        """
        pass









