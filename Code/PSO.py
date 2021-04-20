import cv2
import random
import numpy as np
import copy
import math
import sys
import time

class PSO:

    def __init__(self, ref_image: np.array,landscape: np.array,
                 W: float, C1 :float, C2: float,
                 nb_of_particles: int,
                 max_iter_before_explosion:int, max_iter_without_gbest_update: int,max_iter_before_termination: int):

        self.gbest_value = 1000000
        self.gbest_position = [0, 0]
        self.W = W
        self.c1 = C1
        self.c2 = C2
        self.nb_of_particles = nb_of_particles
        self.ref_image = ref_image
        self.landscape = landscape
        self.nb_of_channels = ref_image.shape[-1]
        self.max_y, self.max_x = landscape.shape[:2]
        self.min_x, self.min_y = [0,0]
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
        min_x = int(self.min_x + (self.ref_image.shape[1]/2)+1)
        min_y = int(self.min_y + (self.ref_image.shape[0]/2)+1)
        max_x = int(self.max_x - (self.ref_image.shape[1]/2)-1)
        max_y = int(self.max_y - (self.ref_image.shape[0]/2)-1)
        print(min_x, min_y, max_x, max_y)
        for i in range(0, self.nb_of_particles):
            x = random.randint(int(min_x), int(max_x))
            y = random.randint(int(min_y), int(max_y))
            particles.append(Particle(x, y))
        return particles

    def update_particles_velocity(self) -> None:
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        for particle in self.particles:
            velocity = (self.W * particle.velocity) + (self.c1 * r1 * (particle.pbest_position - particle.position)) + (self.c2 * r2 * (self.gbest_position - particle.position))
            particle.update_velocity(velocity)

    def update_particles_position(self) -> None:
        min_x = int(self.min_x + (self.ref_image.shape[1]/2)+1)
        min_y = int(self.min_y + (self.ref_image.shape[0]/2)+1)
        max_x = int(self.max_x - (self.ref_image.shape[1]/2)-1)
        max_y = int(self.max_y - (self.ref_image.shape[0]/2)-1)
        for particle in self.particles:
            position = np.zeros(2)
            if min_x < int(particle.position[0] + particle.velocity[0]) < max_x:
                position[0] = int(particle.position[0] + particle.velocity[0])
            else:
                position[0] = random.randint(int(min_x), int(max_x))
            if min_y < int(particle.position[1] + particle.velocity[1]) < max_y:
                position[1] = int(particle.position[1] + particle.velocity[1])
            else:
                position[1] = random.randint(int(min_y), int(max_y))

            particle.update_position(position)

    def evaluate_fitness(self) -> None:
        """
        Calculating fitnness function and choosing best
        :return: None
        """
        ## Updating nb of iterations
        self.iteration += 1

        for particle in self.particles:
            x, y = particle.position
            Pinv = 0 ## ??????????????????????
            nbits = 8
            n, m = self.ref_image.shape[:2]
            error_calc = 0
            for i in range(0,n):
                for j in range(0,m):
                    # ddX = j - m/2
                    # ddY = i - n/2
                    # I = int(y + s * (ddX * math.sin(-tau * (math.pi / 180)) + ddY * math.cos(tau * (math.pi / 180))))
                    # J = int(x + s * (ddX * math.cos(-tau * (math.pi / 180)) + ddY * math.sin(tau * (math.pi / 180))))
                    I = int(y + i - n/2)
                    J = int(x + j - m/2)
                    ## Calculating error calc and checking if indexes works well
                    try:
                        error_calc += abs(self.ref_image[i, j] - self.landscape[I, J])
                    except IndexError:
                        error_calc += 255
                        print(f"IndexError in iteration{self.iteration} for position {x}, {y}"
                               f"for i = {j}, j = {i},"
                               f"for I = {J}, J = {I}")
                        sys.exit(7)

            err_max = (2 ** nbits) * ((m * n) - Pinv)

            ## Calculating error and checking if err_max is not 0
            try:
                error = (err_max - error_calc) / err_max
            except ZeroDivisionError:
                error = 69
                print(f"ZeroDivisionError for iteration{self.iteration}  for position {x}, {y}")

            ## Updating gbest, pbest if needed
            if error < self.gbest_value:
                self.gbest_value = error
                self.gbest_position = [x, y]
                self.iteration_since_gbest_update = 0 ## Zeroing iterations since gbest update
            else:
                self.iteration_since_gbest_update = self.iteration_since_gbest_update + 1

            if error < particle.pbest_value:
                particle.update_pbest(error, [x, y])


    def explode(self) -> None:
        """
        Moving particles to random position # wszystkie czy wybrane?
        :return:
        """
        for particle in self.particles:
            position = np.zeros(2)
            min_x = int(self.min_x + (self.ref_image.shape[1]/2) + 1)
            min_y = int(self.min_y + (self.ref_image.shape[0]/2) + 1)
            max_x = int(self.max_x - (self.ref_image.shape[1]/2) - 1)
            max_y = int(self.max_y - (self.ref_image.shape[0]/2) - 1)
            position[0] = random.randint(int(min_x), int(max_x))
            position[1] = random.randint(int(min_y), int(max_y))
            particle.update_position(position)
            particle.update_pbest(0, [0, 0])
        print("EXPLODE")


    def should_terminate(self) -> bool:
        """
        Checking if termination requirements are met
        :return:
        True if should terminate
        False if not
        """
        if self.iteration >= self.max_iter_before_termination:
            print("MAX ITER REACHED, ACHIEVED")
            return True
        # TODO: Konniec jak za długo nie zmienia się gbest
        elif self.iteration_since_gbest_update >= self.max_iter_before_explosion:
            self.explode()
            self.iteration_since_gbest_update = 0 # potem to wywalić
            return False
        else:
            return False

class Particle:

    # TO DO : Update_position, update_velocity, update_pbest etc

    def __init__(self, x: int, y: int):

        tau = 0
        self.position = np.array([x, y])
        self.velocity = np.array([random.uniform(-0.1, 0.1) for i in range(0, 2)])
        self.pbest_position = copy.deepcopy(self.position)
        self.pbest_value = 1000000

    def update_position(self, position: np.array):
        self.position = position


    def update_velocity(self, velocity):
        self.velocity = velocity

    def update_pbest(self,pbest_value, pbest_position):
        self.pbest_position = pbest_position
        self.pbest_value = pbest_value


def main():
    ################Parametry do zmiany####################
    W = 0.5
    C1 = 2
    C2 = 2
    nb_of_particles = 20
    max_iter_before_explosion = 10*nb_of_particles
    max_iter_without_gbest_update = 20
    max_iter_before_termination = 50
    reference_image_path = "shapes_pat.png"
    landscape_image_path = "shapes.jpg"
    #######################################################
    refImage = cv2.imread(reference_image_path,0)
    landImage = cv2.imread(landscape_image_path,0)
    print(f"refImage size is {refImage.shape}")
    print(f"LandImage size is {landImage.shape}")

    pso = PSO(refImage, landImage,W, C1, C2, nb_of_particles,
                 max_iter_before_explosion, max_iter_without_gbest_update, max_iter_before_termination)

    start_time = time.time()
    count = 0
    show_me = landImage.copy()
    while not pso.should_terminate():# ??kolejność plus co gdzie powinno byc wywoływane?
        for particle in pso.particles:
            cv2.circle(show_me, (int(particle.position[0]), int(particle.position[1])), radius=0, color=(0, 0, 0), thickness=6)
        # cv2.imshow("Image", show_me)
        # cv2.waitKey()
        pso.evaluate_fitness()
        pso.update_particles_velocity()
        pso.update_particles_position()
        count = count + 1
        print(count)

    end_time = time.time()

    start_point = (int(pso.gbest_position[0] - (refImage.shape[1])/2), int(pso.gbest_position[1] - (refImage.shape[0])/2))
    end_point = (int(pso.gbest_position[0] + (refImage.shape[1])/2), int(pso.gbest_position[1] + (refImage.shape[0])/2))
    color = (68, 104, 124)
    thickness = 3

    cv2.rectangle(show_me, start_point, end_point, color, thickness)

    cv2.imshow('landscape', show_me)
    cv2.imshow("refImage", refImage)

    print(f"elapsed time: {end_time - start_time} s")
    print(f"gbest value : {pso.gbest_value}")
    print(f"gbest position : {pso.gbest_position}")

    cv2.waitKey()

if __name__ == "__main__":
    main()






