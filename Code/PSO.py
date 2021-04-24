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

        self.gbest_value = 0
        self.gbest_position = [0, 0]
        self.W = W
        self.c1 = C1
        self.c2 = C2
        self.nb_of_particles = nb_of_particles
        self.ref_image = ref_image
        self.landscape = landscape
        self.nb_of_channels = ref_image.shape[-1]
        self.max_y = int(landscape.shape[0] - (ref_image.shape[0]/2) - 1)
        self.max_x = int(landscape.shape[1] - (ref_image.shape[1]/2) - 1)
        self.min_x = int((ref_image.shape[1]/2)+1)
        self.min_y = int((ref_image.shape[0]/2)+1)
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
            particles.append(Particle(x, y))
        return particles

    def update_particles_velocity(self) -> None:
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        for particle in self.particles:
            velocity = (self.W * particle.velocity) + (self.c1 * r1 * (particle.pbest_position - particle.position)) + (self.c2 * r2 * (self.gbest_position - particle.position))
            particle.update_velocity(velocity)

    def update_particles_position(self) -> None:
        for particle in self.particles:
            position = np.zeros(2)
            if self.min_x < int(particle.position[0] + particle.velocity[0]) < self.max_x:
                position[0] = int(particle.position[0] + particle.velocity[0])
            else:
                position[0] = random.randint(self.min_x, self.max_x)
            if self.min_y < int(particle.position[1] + particle.velocity[1]) < self.max_y:
                position[1] = int(particle.position[1] + particle.velocity[1])
            else:
                position[1] = random.randint(self.min_y, self.max_y)

            particle.update_position(position)

    def evaluate_fitness(self) -> None:
        """
        Calculating fitnness function and choosing best
        :return: None
        """
        ## Updating nb of iterations
        self.iteration += 1
        was_update = False

        for particle in self.particles:
            x, y = particle.position
            Pinv = 0
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
            if error > self.gbest_value:
                self.gbest_value = error
                self.gbest_position = [x, y]
                self.iteration_since_gbest_update = 0 ## Zeroing iterations since gbest update
                self.iteration_since_explosion = 0
                was_update = True

            if error > particle.pbest_value:
                particle.update_pbest(error, [x, y])

        if not was_update:
            self.iteration_since_gbest_update = self.iteration_since_gbest_update + 1
            self.iteration_since_explosion = self.iteration_since_explosion + 1


    def evaluate_fitness2(self) -> None:

        ## Updating nb of iterations
        self.iteration += 1
        was_update = False
        for particle in self.particles:
            x, y = particle.position
            n, m = self.ref_image.shape[:2]
            landcape_sample = copy.deepcopy(self.ref_image)

            for i in range(0, n):
                for j in range(0, m):

                    I = int(y + i - n / 2)
                    J = int(x + j - m / 2)
                    try:
                        landcape_sample[i, j] = self.landscape[I, J]
                    except IndexError:
                        landcape_sample[i, j] = 400

            hgram, x_edges, y_edges = np.histogram2d(
                landcape_sample.ravel(),
                self.ref_image.ravel(),
                bins=20)

            pxy = hgram / float(np.sum(hgram))
            px = np.sum(pxy, axis=1)  # marginal for x over y
            py = np.sum(pxy, axis=0)  # marginal for y over x
            px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
            # Now we can do the calculation using the pxy, px_py 2D arrays
            nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
            result = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
            ## Updating gbest, pbest if needed
            if result > self.gbest_value:
                self.gbest_value = result
                self.gbest_position = [x, y]
                self.iteration_since_gbest_update = 0  ## Zeroing iterations since gbest update
                self.iteration_since_explosion = 0
                was_update = True

            if result > particle.pbest_value:
                particle.update_pbest(result, [x, y])

        if not was_update:
            self.iteration_since_gbest_update = self.iteration_since_gbest_update + 1
            self.iteration_since_explosion = self.iteration_since_explosion + 1

    def explode(self) -> None:
        """
        Moving particles to random position # wszystkie czy wybrane?
        :return:
        """
        for particle in self.particles:
            position = np.zeros(2)
            position[0] = random.randint(self.min_x, self.max_x)
            position[1] = random.randint(self.min_y, self.max_y)
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
            print("MAX ITER REACHED")
            return True
        elif self.iteration_since_gbest_update >= self.max_iter_without_gbest_update:
            print("MAX ITER WITHOUT GBEST UPDATE REACHED")
            return True
        else:
            return False

class Particle:

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
    max_iter_before_explosion = 9
    max_iter_without_gbest_update = 20
    max_iter_before_termination = 50
    reference_image_path = "pattern.jpg"
    landscape_image_path = "druzyna_AGH_01.jpg"
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
    while not pso.should_terminate():
        for particle in pso.particles:
            cv2.circle(show_me, (int(particle.position[0]), int(particle.position[1])), radius=0, color=(0, 0, 0), thickness=6)
        # cv2.imshow("Image", show_me)
        # cv2.waitKey()
        pso.evaluate_fitness2()
        pso.update_particles_velocity()
        pso.update_particles_position()
        count = count + 1
        if pso.iteration_since_explosion > pso.max_iter_before_explosion:
            pso.iteration_since_explosion = 0
            pso.explode()
        print("Iteration nr " + str(count) + ", since gbest_update: " + str(pso.iteration_since_gbest_update) + ", since explode: " + str(pso.iteration_since_explosion))

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



def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        reference_image_path = "shapes_pat.png"
        landscape_image_path = "shapes.jpg"
        ref_image = cv2.imread(reference_image_path, 0)
        land_image = cv2.imread(landscape_image_path, 0)
        print(f"check for x = {x} , y = {y}")
        Pinv = 0
        nbits = 8
        n, m = ref_image.shape[:2]
        error_calc = 0
        for i in range(0, n):
            for j in range(0, m):
                # ddX = j - m/2
                # ddY = i - n/2
                # I = int(y + s * (ddX * math.sin(-tau * (math.pi / 180)) + ddY * math.cos(tau * (math.pi / 180))))
                # J = int(x + s * (ddX * math.cos(-tau * (math.pi / 180)) + ddY * math.sin(tau * (math.pi / 180))))
                I = int(y + i - n / 2)
                J = int(x + j - m / 2)
                ## Calculating error calc and checking if indexes works well
                try:
                    error_calc += abs(ref_image[i, j] - land_image[I, J])
                except IndexError:
                    error_calc += 255


        err_max = (2 ** nbits) * ((m * n) - Pinv)

        ## Calculating error and checking if err_max is not 0
        try:
            error = (err_max - error_calc) / err_max
        except ZeroDivisionError:
            error = 69
            print(f"ZeroDivisionError")
        print(f"ERROR = {error}")


def check_function():
    ################Parametry do zmiany####################
    reference_image_path = "shapes_pat.png"
    landscape_image_path = "shapes.jpg"
    #######################################################

    landImage = cv2.imread(landscape_image_path, 0)
    cv2.imshow('image', landImage)

    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    #check_function()

