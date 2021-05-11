import cv2
import random
import numpy as np
import copy
import math
import sys
import time
#from hausdorff import hausdorff_distance
from scipy.spatial.distance import directed_hausdorff


class PSO:

    def __init__(self, ref_image: np.array,landscape: np.array,
                 W: float, C1 :float, C2: float,
                 nb_of_particles: int,
                 max_iter_before_explosion:int, max_iter_without_gbest_update: int,max_iter_before_termination: int, RGB: bool = False,choosen_function: int = 1 ):

        if choosen_function == 3:
            self.gbest_value = 1000000000000
        else:
            self.gbest_value = 0
        self.gbest_position = [0, 0, 1, 0]
        self.W = W
        self.c1 = C1
        self.c2 = C2
        self.nb_of_particles = nb_of_particles
        self.ref_image = ref_image if RGB else cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        self.landscape = landscape if RGB else cv2.cvtColor(landscape, cv2.COLOR_BGR2GRAY)
        self.nb_of_channels = ref_image.shape[-1]
        self.max_y = int(landscape.shape[0] - max(ref_image.shape))
        self.max_x = int(landscape.shape[1] - max(ref_image.shape))
        self.min_x = int(max(ref_image.shape))
        self.min_y = int(max(ref_image.shape))
        self.min_s = 0.75
        self.max_s = 1.5
        self.min_r = -math.pi/2
        self.max_r = math.pi/2
        self.particles = self.init_with_random_values()
        self.iteration = 0
        self.iteration_since_gbest_update = 0
        self.iteration_since_explosion = 0
        self.max_iter_without_gbest_update = max_iter_without_gbest_update
        self.max_iter_before_termination = max_iter_before_termination
        self.max_iter_before_explosion = max_iter_before_explosion

        self.choosen_function = choosen_function
        self.RGB = RGB

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
            s = random.uniform(self.min_s, self.max_s)
            r = random.uniform(self.min_r, self.max_r)
            particles.append(Particle(x, y, s, r))
        return particles

    def update_particles_velocity(self) -> None:
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        for particle in self.particles:
            velocity = (self.W * particle.velocity) + (self.c1 * r1 * (particle.pbest_position - particle.position)) + (self.c2 * r2 * (self.gbest_position - particle.position))
            particle.update_velocity(velocity)

    def update_particles_position(self) -> None:
        for particle in self.particles:
            position = np.zeros(4)
            if self.min_x < int(particle.position[0] + particle.velocity[0]) < self.max_x:
                position[0] = int(particle.position[0] + particle.velocity[0])
            else:
                position[0] = random.randint(self.min_x, self.max_x)
            if self.min_y < int(particle.position[1] + particle.velocity[1]) < self.max_y:
                position[1] = int(particle.position[1] + particle.velocity[1])
            else:
                position[1] = random.randint(self.min_y, self.max_y)
            if self.min_s < int(particle.position[2] + particle.velocity[2]) < self.max_s:
                position[2] = int(particle.position[2] + particle.velocity[2])
            else:
                position[2] = random.uniform(self.min_s, self.max_s)
            if self.min_r < int(particle.position[3] + particle.velocity[3]) < self.max_r:
                position[3] = int(particle.position[3] + particle.velocity[3])
            else:
                position[3] = random.uniform(self.min_r, self.max_r)
            particle.update_position(position)

    def evaluate_fitness1(self) -> None:
        """
        Calculating fitnness function and choosing best
        :return: None
        """
        ## Updating nb of iterations
        self.iteration += 1
        was_update = False

        for particle in self.particles:
            x, y, s, r = particle.position
            Pinv = 0
            nbits = 8
            n, m = self.ref_image.shape[:2]
            error_calc = 0

            if n % 2 == 0:
                ddN = n/2
            else:
                ddN = (n-1)/2
            if m % 2 == 0:
                ddM = m/2
            else:
                ddM = (m-1)/2

            for i in range(0,n):
                for j in range(0,m):
                    ddX = j - ddM
                    ddY = i - ddN
                    I = int(y + s * (ddX * math.sin(-r) + ddY * math.cos(r)))
                    J = int(x + s * (ddX * math.cos(-r) + ddY * math.sin(r)))
                    # I = int(y + i - ddN)
                    # J = int(x + j - ddM)
                    ## Calculating error calc and checking if indexes works well
                    try:
                        if self.RGB:
                            for k in range(0,3):
                                error_calc += abs(self.ref_image[i, j, k] - self.landscape[I, J, k])
                        else:
                            error_calc = abs(self.ref_image[i, j] - self.landscape[I, J])
                    except IndexError:
                        print(f"IndexError in iteration{self.iteration} for position {x}, {y}, {s}, {r}"
                               f"for i = {i}, j = {j},"
                               f"for I = {I}, J = {J}")
                        sys.exit(7)

            err_max = (2 ** nbits) * ((m * n) - Pinv)

            ## Calculating error and checking if err_max is not 0
            try:
                if self. RGB:
                     error = (3* err_max - error_calc) / err_max
                else:
                    error = (err_max - error_calc) / err_max
            except ZeroDivisionError:
                error = 69
                print(f"ZeroDivisionError for iteration{self.iteration}  for position {x}, {y}")

            ## Updating gbest, pbest if needed
            if error > self.gbest_value:
                self.gbest_value = error
                self.gbest_position = [x, y, s, r]
                self.iteration_since_gbest_update = 0 ## Zeroing iterations since gbest update
                self.iteration_since_explosion = 0
                was_update = True

            if error > particle.pbest_value:
                particle.update_pbest(error, [x, y, s, r])

        if not was_update:
            self.iteration_since_gbest_update = self.iteration_since_gbest_update + 1
            self.iteration_since_explosion = self.iteration_since_explosion + 1

    def evaluate_fitness2(self) -> None:

        ## Updating nb of iterations
        self.iteration += 1
        was_update = False
        for particle in self.particles:
            x, y, s, r = particle.position
            n, m = self.ref_image.shape[:2]
            landcape_sample = copy.deepcopy(self.ref_image)

            for i in range(0, n):
                for j in range(0, m):
                    ddX = j - m/2
                    ddY = i - n/2
                    I = int(y + s * (ddX * math.sin(-r) + ddY * math.cos(r)))
                    J = int(x + s * (ddX * math.cos(-r) + ddY * math.sin(r)))
                    # I = int(y + i - n/2)
                    # J = int(x + j - m/2)
                    try:
                        if self.RGB:
                            landcape_sample[i, j, :] = self.landscape[I, J, :]
                        else:
                            landcape_sample[i, j] = self.landscape[I, J]
                    except IndexError:
                        sys.exit(7)
            result = 0

            if self.RGB:
                for k in range(0, 3):
                    hgram, x_edges, y_edges = np.histogram2d( landcape_sample[:,:,k].ravel(),self.ref_image[:,:,k].ravel(),bins=20)
                    pxy = hgram / float(np.sum(hgram))
                    px = np.sum(pxy, axis=1)  # marginal for x over y
                    py = np.sum(pxy, axis=0)  # marginal for y over x
                    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
                    # Now we can do the calculation using the pxy, px_py 2D arrays
                    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
                    result += np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
            else:
                hgram, x_edges, y_edges = np.histogram2d(landcape_sample[:, :].ravel(),self.ref_image[:, :].ravel(), bins=20)
                pxy = hgram / float(np.sum(hgram))
                px = np.sum(pxy, axis=1)
                py = np.sum(pxy, axis=0)
                px_py = px[:, None] * py[None, :]
                nzs = pxy > 0
                result = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

            ## Updating gbest, pbest if needed
            if result > self.gbest_value:
                self.gbest_value = result
                self.gbest_position = [x, y, s, r]
                self.iteration_since_gbest_update = 0  ## Zeroing iterations since gbest update
                self.iteration_since_explosion = 0
                was_update = True

            if result > particle.pbest_value:
                particle.update_pbest(result, [x, y, s, r])

        if not was_update:
            self.iteration_since_gbest_update = self.iteration_since_gbest_update + 1
            self.iteration_since_explosion = self.iteration_since_explosion + 1



    def evaluate_fitness_hausdorf(self) -> None:

        ## Updating nb of iterations
        self.iteration += 1
        was_update = False
        for particle in self.particles:
            x, y, s, r = particle.position
            n, m = self.ref_image.shape[:2]
            landcape_sample = copy.deepcopy(self.ref_image)

            for i in range(0, n):
                for j in range(0, m):

                    ddX = j - m/2
                    ddY = i - n/2
                    I = int(y + s * (ddX * math.sin(-r) + ddY * math.cos(r)))
                    J = int(x + s * (ddX * math.cos(-r) + ddY * math.sin(r)))
                    # I = int(y + i - n/2)
                    # J = int(x + j - m/2)
                    try:
                        if self.RGB:
                            landcape_sample[i, j, :] = self.landscape[I, J, :]
                        else:
                            landcape_sample[i, j] = self.landscape[I, J]
                    except IndexError:
                        sys.exit(7)
            result = 0

            if self.RGB:
                for k in range(0, 3):

                    # result += hausdorff_distance(landcape_sample[:, :, k], self.ref_image[:, :, k])

                    result += max(directed_hausdorff(landcape_sample[:, :, k], self.ref_image[:, :, k])[0], directed_hausdorff(self.ref_image[:, :, k], landcape_sample[:, :, k])[0])
            else:
                result =max(directed_hausdorff(landcape_sample[:, :, k], self.ref_image[:, :, k])[0], directed_hausdorff(self.ref_image[:, :, k], landcape_sample[:, :, k])[0])
                # result = hausdorff_distance(landcape_sample, self.ref_image)

            ## Updating gbest, pbest if needed
            if result < self.gbest_value:
                self.gbest_value = result
                self.gbest_position = [x, y, s, r]
                self.iteration_since_gbest_update = 0  ## Zeroing iterations since gbest update
                self.iteration_since_explosion = 0
                was_update = True

            if result < particle.pbest_value:
                particle.update_pbest(result, [x, y, s, r])

        if not was_update:
            self.iteration_since_gbest_update = self.iteration_since_gbest_update + 1
            self.iteration_since_explosion = self.iteration_since_explosion + 1


    def evaluate(self):
        if self.choosen_function == 1:
            self.evaluate_fitness1()
        elif self.choosen_function == 2:
            self.evaluate_fitness2()
        elif self.choosen_function == 3:
            self.evaluate_fitness_hausdorf()
        else:
            pass

    def explode(self) -> None:
        """
        Moving particles to random position # wszystkie czy wybrane?
        :return:
        """
        for particle in self.particles:
            position = np.zeros(4)
            position[0] = random.randint(self.min_x, self.max_x)
            position[1] = random.randint(self.min_y, self.max_y)
            position[2] = random.uniform(self.min_s, self.max_s)
            position[3] = random.uniform(self.min_r, self.max_r)
            particle.update_position(position)
            particle.update_pbest(0, [0, 0, 1, 0])
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

    def __init__(self, x: int, y: int, s: float, r: float):

        tau = 0
        self.position = np.array([x, y, s, r])
        self.velocity = np.array([random.uniform(-0.1, 0.1) for i in range(0, 4)])
        self.pbest_position = copy.deepcopy(self.position)
        self.pbest_value = 0

    def update_position(self, position: np.array):
        self.position = position


    def update_velocity(self, velocity):
        self.velocity = velocity

    def update_pbest(self,pbest_value, pbest_position):
        self.pbest_position = pbest_position
        self.pbest_value = pbest_value




def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        reference_image_path = "shapes_pat.png"
        landscape_image_path = "shapes.jpg"
        ref_image = cv2.imread(reference_image_path,0)
        landscape_image = cv2.imread(landscape_image_path,0)
        print(f"check for x = {x} , y = {y}")
        landcape_sample = copy.deepcopy(ref_image)
        n, m = ref_image.shape[:2]

        RGB = False
        for i in range(0, n):
            for j in range(0, m):

                I = int(y + i - n / 2)
                J = int(x + j - m / 2)
                try:
                    if RGB:
                        landcape_sample[i, j, :] = landscape_image[I, J, :]
                    else:
                        landcape_sample[i, j] = landscape_image[I, J]
                except IndexError:
                    sys.exit(7)
        result = 0

        if RGB:
            for k in range(0, 3):
                # result += hausdorff_distance(landcape_sample[:, :, k], self.ref_image[:, :, k])

                result += max(directed_hausdorff(landcape_sample[:, :, k], ref_image[:, :, k])[0],
                              directed_hausdorff(ref_image[:, :, k], landcape_sample[:, :, k])[0])
        else:
            result = max(directed_hausdorff(landcape_sample, ref_image)[0],
                         directed_hausdorff(ref_image, landcape_sample)[0])
            # result = hausdorff_distance(landcape_sample, self.ref_image)

        print(f"RESULT = {result}")

def check_function():
    ################Parametry do zmiany####################
    reference_image_path = "Wally_ref.jpg"
    landscape_image_path = "Wally_landscape.jpeg"
    #######################################################

    landImage = cv2.imread(landscape_image_path, 0)
    cv2.imshow('image', landImage)

    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    ################Parametry do zmiany####################
    W = 0.5
    C1 = 2
    C2 = 2
    nb_of_particles = 20
    max_iter_before_explosion = 49
    max_iter_without_gbest_update = 300
    max_iter_before_termination = 800
    reference_image_path = "pattern_flip.jpg"
    landscape_image_path = "druzyna_AGH_01.jpg"
    RGB = True
    choosen_function = 3
    # 1 - Count difference measure
    # 2 - Mutual information measure
    # 3 - Hausdorff distance
    #######################################################

    refImage = cv2.imread(reference_image_path)
    landImage = cv2.imread(landscape_image_path)

    print(f"refImage size is {refImage.shape}")
    print(f"LandImage size is {landImage.shape}")


    pso = PSO(refImage, landImage,W, C1, C2, nb_of_particles,
                 max_iter_before_explosion, max_iter_without_gbest_update, max_iter_before_termination, RGB= RGB, choosen_function= choosen_function)

    if not RGB:
        refImage = cv2.cvtColor(refImage, cv2.COLOR_BGR2GRAY)
        landImage = cv2.cvtColor(landImage, cv2.COLOR_BGR2GRAY)

    start_time = time.time()
    count = 0
    show_me = landImage.copy()
    while not pso.should_terminate():
        for particle in pso.particles:
            cv2.circle(show_me, (int(particle.position[0]), int(particle.position[1])), radius=0, color=(0, 0, 0), thickness=6)
        pso.evaluate()
        pso.update_particles_velocity()
        pso.update_particles_position()
        count = count + 1
        if pso.iteration_since_explosion > pso.max_iter_before_explosion:
            pso.iteration_since_explosion = 0
            pso.explode()
        print("Iteration nr " + str(count) + ", since gbest_update: " + str(pso.iteration_since_gbest_update) + ", since explode: " + str(pso.iteration_since_explosion))

    end_time = time.time()

    start_point = (int(pso.gbest_position[0] - (refImage.shape[1])/2), int(pso.gbest_position[1] - (refImage.shape[0])/2))
    end_point = (int((pso.gbest_position[0] + (refImage.shape[1])/2)), int((pso.gbest_position[1] + (refImage.shape[0])/2)))
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
    #check_function()

