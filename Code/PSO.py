import cv2
import random
import numpy as np
import copy
import math
import time

class PSO:

    def __init__(self, ref_image: np.array,landscape: np.array,
                 W: float, C1 :float, C2: float,
                 nb_of_particles: int,
                 max_iter_before_explosion:int, max_iter_without_gbest_update: int,max_iter_before_termination: int):

        self.gbest_value = 0
        self.gbest_position = [0, 0]
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
        self.iteration_since_explosion = 0
        self.max_iter_without_gbest_update = max_iter_without_gbest_update
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
        r1 = random.uniform(0, 1, 1)
        r2 = random.uniform(0, 1, 1)
        for particle in self.particles:
            particle.velocity = (self.W * particle.velocity) + (self.c1 * r1 * (particle.pbest_position - particle.position)) + (self.c2 * r2 * (self.gbest_position - particle.position))

    def update_particles_position(self) -> None:
        for particle in self.particles:
            particle.position = particle.position + particle.velocity

    def evaluate_fitness(self) -> None:
        """
        Calculating fitnness function and choosing best
        :return: None
        """
        ## Updating nb of iterations
        self.iteration += 1

        for k in range(len(self.particles)):
            particle = copy.deepcopy(self.particles[k]) ## Creating local copy of particle
            x, y, s, tau = particle.position
            Pinv = 0 ## ??????????????????????
            nbits = 8
            m, n = self.max_x, self.max_y
            error_calc = 0
            for i in range (0, n):
                for j in range (0, m):
                    ddX = j - m/2
                    ddY = i - n/2
                    I = int(y + s * (ddX * math.sin(-tau) + ddY * math.cos(tau)))
                    J = int(x + s * (ddX * math.cos(-tau) + ddY * math.sin(tau)))

                    ## Calculating error calc and checking if indexes works well
                    try:
                        error_calc += abs(self.ref_image[j, i] - self.landscape[J, I])
                    except IndexError:
                        error_calc += 255
                        print(f"IndexError in iteration{self.iteration} for particle {k} for position {x}, {y}"
                              f"for i = {i}, j = {j},"
                              f"J = {J}, I = {I}")

            err_max = (2 ** nbits) * ((m * n) - Pinv)

            ## Calculating error and checking if err_max is not 0
            try:
                error = (err_max - error_calc) / err_max
            except ZeroDivisionError:
                error = 69
                print(f"ZeroDivisionError for iteration{self.iteration} for particle {k} for position {x}, {y}")

            ## Updating gbest, pbest if needed
            if error < self.gbest_value:
                self.gbest_value = error
                self.gbest_position = [x, y]
                self.iteration_since_gbest_update = 0 ## Zeroing iterations since gbest update


            if error < particle.pbest_value:
                particle.pbest_value = error
                particle.pbest_position = [x, y]
                self.particles[k] = particle ## Updating particle if sth changed





    def explode(self) -> None:
        """
        Moving particles to random position # wszystkie czy wybrane?
        :return:
        """
        for particle in self.particles:

            x = random.randint(0, self.landscape.shape[1] - self.ref_image.shape[1])
            y = random.randint(0, self.landscape.shape[0] - self.ref_image.shape[0])
            s = random.randrange(0, 2)
            tau = 0 # for now
            particle.position = [x, y, s, tau * (math.pi / 180)]
            # velocity tez zmieniamy?


    def should_terminate(self) -> bool:
        """
        Checking if termination requirements are met
        :return:
        True if should terminate
        False if not
        """
        if self.iteration >= self.max_iter_before_termination \
                or self.iteration_since_gbest_update >= self.max_iter_without_gbest_update:
            return True
        else:
            return False

class Particle:


    def __init__(self, x: int, y: int, s: float, tau: float):

        tau = 0
        self.position = [x, y, s, tau*(math.pi/180)]
        self.velocity = [random.uniform(-1, 1) for i in range(0, 4)]
        self.pbest_position = copy.deepcopy(self.position)
        self.pbest_value = 0


def main():
    ################Parametry do zmiany####################
    W = 1.0
    C1 = 1.0
    C2 = 1.0
    nb_of_particles = 5
    max_iter_before_explosion = 15
    max_iter_without_gbest_update = 20
    max_iter_before_termination = 50
    reference_image_path = ""
    landscape_image_path = ""
    #######################################################
    refImage = cv2.imread(reference_image_path, cv2.COLOR_BGR2GRAY)
    landImage = cv2.imread(landscape_image_path, cv2.COLOR_BGR2GRAY)

    pso = PSO(refImage, landImage,W, C1, C2, nb_of_particles,
                 max_iter_before_explosion, max_iter_without_gbest_update, max_iter_before_termination)

    start_time = time.time()

    while not pso.should_terminate():# ??kolejność plus co gdzie powinno byc wywoływane?
        pso.evaluate_fitness()
        pso.update_particles_velocity()
        pso.update_particles_position()

    end_time = time.time()

    #draw rectangle # TODO: dodac skalowanie prostokata
    start_point = (int(pso.gbest_position[0]) - 30, int(pso.gbest_position[1]) - 30)
    end_point = (int(pso.gbest_position[0]) + 30, int(pso.gbest_position[1]) + 30)
    color = (255, 0, 0)
    thickness = 3

    cv2.rectangle(landImage, start_point, end_point, color, thickness)

    cv2.imshow('landscape', landImage)
    cv2.imshow("refImage", refImage)
    cv2.waitKey(0)

    print(f"elapsed time: {end_time - start_time} s")
    print(f"gbest value : {pso.gbest_value}")



if __name__ == "__main__":
    main()






