


class PSO(object):

    def __init__(self, ref_image, landscape_image, c1=2, c2=2, W=0.5, n_particles=20, n_iter=200, epislon=3):
        self.gbest_position = [0, 0, 0, 0]
        self.gbest_value = 0
        self.c1 = c1
        self.c2 = c2
        self.W = W
        self.n_particles = n_particles
        self.ref_image = ref_image
        self.landscape_image = landscape_image
        self.particles = self.init_particles()
        self.epsilon = epislon
        self.n_iter = n_iter

    def init_particles(self):
        """
        Initialize particles with random values
        """
        particles = []
        for i in range(0, self.n_particles):
            m, n = self.landscape_image.shape
            x = random.randint(0, n)
            y = random.randint(0, m)
            s = random.uniform(0, 2)
            tau = random.randint(-360, 360)
            particles.append(Particle(x, y, s, tau, self.ref_image, self.landscape_image, self.c1, self.c2, self.W))

        return particles

    def evaluate(self):
        for p in self.particles:

            #self.print_current_sol(p)
            p_fitness = p.count_mutual_information()

            # update g_best
            if p_fitness > self.gbest_value:
                print(p_fitness)
                self.gbest_value = p_fitness
                self.gbest_position = copy.deepcopy(p.position)
                print(p.position)

            p.update_pbest(p_fitness)

    def update_velocity(self):
        for p in self.particles:
            p.compute_velocity(self.gbest_position)
            p.move()

    def print_solution(self):
        m, n = self.ref_image.shape
        self.gbest_position[2] = 1

        for i in range(0, n):
            for j in range(0, m):
                cv2.putText(self.landscape_image, ".", (int(self.gbest_position[0]+self.gbest_position[2]*i), int(self.gbest_position[1]+self.gbest_position[2]*j)),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))
        cv2.imshow('image', self.landscape_image)
        cv2.waitKey(0)

    def print_solution_borders(self):
        m, n = self.ref_image.shape
        self.gbest_position[2] = 1

        cv2.rectangle(self.landscape_image, (int(self.gbest_position[0])-5, int(self.gbest_position[1])-5),
                      (int(self.gbest_position[0]+self.gbest_position[2]*n)+5, int(self.gbest_position[1]+self.gbest_position[2]*m)+5),
                      (255, 0, 0), 3)

        cv2.imshow('image', self.landscape_image)
        cv2.waitKey(0)

    def print_current_sol(self, particle):
        cv2.putText(self.landscape_image, ".", (int(particle.position[0]),
                                                int(particle.position[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))

    def run(self):
        k = 0
        while self.gbest_value < self.epsilon and k < self.n_iter:
            self.evaluate()
            self.update_velocity()
            k = k+1
        print(self.gbest_value)
        print(self.gbest_position)
        print(k)
        self.print_solution_borders()
        #self.print_solution()


class Particle(object):

    def __init__(self, x, y, s, tau, ref_image, landscape_image, c1, c2, W):

        self.ref_image = ref_image
        self.landscape_image = landscape_image
        tau = 0
        self.position = [x, y, s, tau*(math.pi/180)]

        self.velocity = [random.uniform(-1, 1) for i in range(0, 4)]
        self.pbest_position = copy.deepcopy(self.position)
        self.pbest_value = 0

        self.change_it = 1
        self.iteration = 0

        self.c1 = c1
        self.c2 = c2
        self.W = W

    def go_to_random(self):
        x = random.randint(0, self.landscape_image.shape[1]-self.ref_image.shape[1])
        y = random.randint(0, self.landscape_image.shape[0]-self.ref_image.shape[0])
        s = random.randrange(0, 2)
        tau = 0
        self.position = [x, y, s, tau * (math.pi / 180)]

    def update_pbest(self, fitness):
        self.iteration = self.iteration+1
        if fitness > self.pbest_value:
            self.change_it = self.iteration
            self.pbest_value = fitness
            self.pbest_position = copy.deepcopy(self.position)

        # go to random position if algorithms has stacked
        if self.iteration - self.change_it > 10:
            self.change_it = self.iteration
            self.go_to_random()

    def count_mutual_information(self):
        x = self.position[0]
        y = self.position[1]
        s = self.position[2]
        tau = self.position[3]

        m, n = self.ref_image.shape
        landcape_sample = copy.deepcopy(self.ref_image)
        err_sol = 0
        #for i in range(-10, n-10):
        for i in range(0, n):
            for j in range(0, m):
                s=1
                I = int(x+s*i)
                J = int(y+s*j)
                try:
                    landcape_sample[j, i] = self.landscape_image[J, I]
                except IndexError:
                    landcape_sample[j, i] = 400

                '''
                cv2.putText(self.ref_image, ".", (i, j), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))
                cv2.putText(self.landscape_image, ".", (i, j), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))
                cv2.putText(self.landscape_image, ".", (I, J), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))
                print(abs(self.ref_image[j, i] - self.landscape_image[I, J]))
                '''
        #print(landcape_sample.shape)
        #print(self.ref_image.shape)
        import numpy as np
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
        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


    def count_difference_measure(self, nbits=8):
        """
        Count difference between images in gray scale
        #TODO catch zero devision error
        """
        x = self.position[0]
        y = self.position[1]
        s = self.position[2]
        tau = self.position[3]
        Pinv = 0
        m, n = self.ref_image.shape
        err_sol = 0
        for i in range(-10, n-10):
        #for i in range(0, n):
            for j in range(0, m):
                s=1
                I = int(x+s*i)
                J = int(y+s*j)

                '''
                cv2.putText(self.ref_image, ".", (i, j), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))
                cv2.putText(self.landscape_image, ".", (i, j), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))
                cv2.putText(self.landscape_image, ".", (I, J), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))
                print(abs(self.ref_image[j, i] - self.landscape_image[I, J]))
                '''

                try:
                    err_sol = err_sol + abs(self.ref_image[j, i] - self.landscape_image[J, I])
                except IndexError:
                    # Pinv = Pinv+1
                    err_sol = err_sol+255

        err_max = (2 ** nbits) * ((m * n) - Pinv)
        try:
            eval_sol = (err_max-err_sol)/err_max
        except ZeroDivisionError:
            eval_sol = 0

        return eval_sol

    def compute_velocity(self, gbest_position):

        local_factor = [self.c1 * random.uniform(0, 10)*(self.pbest_position[i] - self.position[i]) for i, _ in
                        enumerate(self.position)]
        global_factor = [(random.uniform(0, 10)*self.c2)*(gbest_position[i] - self.position[i]) for i, _ in
                         enumerate(self.position)]

        self.velocity = [self.W * self.velocity[i] + local_factor[i] + global_factor[i] for i in range(0, 4)]
        self.velocity[3] = 0

    def move(self):
        temp = [self.position[i]+self.velocity[i] for i in range(0, 4)]

        if temp[0] < 0:
            temp[0] = -temp[0]
        if temp[1] < 0:
            temp[1] = -temp[1]

        if temp[0]+self.ref_image.shape[1] > self.landscape_image.shape[1]:
            #temp[0] = self.landscape_image.shape[1]-self.ref_image.shape[1]*random.uniform(1, 5)
            temp[0] = random.randint(0, self.landscape_image.shape[1] - self.ref_image.shape[1])

        if temp[1]+self.ref_image.shape[0] > self.landscape_image.shape[0]:
            #temp[1] = self.landscape_image.shape[0]-self.ref_image.shape[0]*random.uniform(1, 5)
            temp[1] = random.randint(0, self.landscape_image.shape[0] - self.ref_image.shape[0])

        if temp[2] < 0.5:
            temp[2] = 0.5
        if temp[2] > 2:
            temp[2] = 2

        self.position = copy.deepcopy(temp)


refimage = cv2.imread('data/ref2.png')

#refimage = cv2.imread('data/garlic.png')
#refimage = cv2.imread('data/onion.png')
refimage = cv2.imread('data/peper.png')

#lanscape_image = cv2.imread('data/landscape.bmp')
lanscape_image = cv2.imread('data/mono.bmp')

#rozmycie gaussa
#landscape_image = cv2.blur(lanscape_image,(5,5))
#refimage = cv2.blur(refimage,(5,5))


refimage = cv2.cvtColor(refimage, cv2.COLOR_BGR2GRAY)
lanscape_image = cv2.cvtColor(lanscape_image, cv2.COLOR_BGR2GRAY)

p = Particle(0, 0, 2, 60, refimage, refimage, 1, 1, 1)
diff = p.count_mutual_information()
print('difference')
print(diff)


PSO = PSO(refimage, lanscape_image)
PSO.run()

'''
#p = Particle(400, 240, 2, 60, refimage, lanscape_image, 1, 1, 1)
p = Particle(410, 230, 2, 60, refimage, lanscape_image, 1, 1, 1)

p = Particle(33, 53, 2, 60, refimage, lanscape_image, 1, 1, 1)
diff = p.count_difference_measure()
print('difference')
print(diff)
cv2.imshow('image', lanscape_image)
cv2.waitKey(0)

cv2.imshow('ref', refimage)
cv2.waitKey(0)
'''


