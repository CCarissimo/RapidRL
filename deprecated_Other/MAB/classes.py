import numpy as np


class distribution:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        return None


class Uniform(distribution):

    def sample(self):
        return np.random.uniform(low=self.mu-self.sigma/2, high=self.mu+self.sigma/2)


class Normal(distribution):

    def sample(self):
        return np.random.normal(self.mu, self.sigma)


class colored_MAB:
    def __init__(self, n_colors, sigma_A, sigma_C, initial_mu_C=None):
        self.n_colors = n_colors
        self.n_arms_per_color = {i:0 for i in range(n_colors)}
        self.reservoir_distribution = Uniform(0, 1)
        if initial_mu_C is not None:
            self.color_distributions = {i: Uniform(mu, sigma_C) for i, mu in enumerate(initial_mu_C)}
        else:
            self.color_distributions = {i: Uniform(mu, sigma_C) for i, mu in
                                        enumerate([self.reservoir_distribution.sample() for j in range(self.n_colors)])}
        self.arm_distributions = {}
        self.sigma_A = sigma_A
        self.sigma_C = sigma_C

    def sample(self, arm):
        return self.arm_distributions[arm].sample()

    def new_color(self):
        mu = self.reservoir_distribution.sample()
        color_distribution = Uniform(mu, self.sigma_C)
        self.color_distributions[self.n_colors] = color_distribution
        self.n_arms_per_color[self.n_colors] = 0
        self.n_colors += 1
        return self.n_colors - 1

    def new_arm(self, color):
        mu = self.color_distributions[color].sample()
        arm = (color, self.n_arms_per_color[color])
        self.arm_distributions[arm] = Normal(mu, self.sigma_A)
        self.n_arms_per_color[color] += 1
        return arm


class Agent:
    def __init__(self, delta, epsilon, theta, env):
        self.Q_A = {}
        self.Q_C = {}
        self.N_A = {}
        self.N_C = {}
        self.UCB_A = {}
        self.UCB_C = {}
        self.delta = delta
        self.epsilon = epsilon
        self.theta = theta
        self.env = env
        self.sumX = 0

    def ucb(self, mean, n):
        return mean + np.sqrt(2 * np.log(1 / self.delta) / n)

    def __init__run__(self):
        self.results = []
        # start by sampling the first arm
        for color in self.env.color_distributions.keys():
            arm = self.env.new_arm(color)
            self.result = self.env.sample(arm)
            self.sumX += self.result
            self.results.append(self.result)
            self.Q_A[(color, 0)] = self.result
            self.Q_C[color] = self.result
            self.N_A[(color, 0)] = 1
            self.N_C[color] = 1
            first_ucb = self.ucb(self.result, 1)
            self.UCB_A[(color, 0)] = first_ucb
            self.UCB_C[color] = first_ucb

    def run(self, iterations):
        # now we use a probability epsilon to determine whether we request a new arm
        for i in range(iterations):
            if np.random.random() >= 1 - self.epsilon:
                if np.random.random() >= 1 - self.theta:
                    color = self.env.new_color()
                    arm = self.env.new_arm(color)
                    self.result = self.env.sample(arm)
                    self.results.append(self.result)
                    self.N_A[arm] = 1
                    self.N_C[color] = 1
                    self.Q_A[arm] = self.result
                    self.Q_C[color] = self.result
                    first_ucb = self.ucb(self.result, 1)
                    self.UCB_A[arm] = first_ucb
                    self.UCB_C[color] = first_ucb
                else:
                    color = max(self.Q_C, key=self.Q_C.get)
                    arm = self.env.new_arm(color)
                    self.result = self.env.sample(arm)
                    self.results.append(self.result)
                    self.N_A[arm] = 1
                    self.N_C[color] += 1
                    self.Q_A[arm] = self.result
                    self.Q_C[color] = self.Q_C[color] * (self.N_C[color] - 1) / self.N_C[color] + self.result / self.N_C[color]
                    self.UCB_A[arm] = self.ucb(self.Q_A[arm], 1)
                    self.UCB_C[color] = self.ucb(self.Q_C[color], self.N_C[color])
            else:
                arm = max(self.UCB_A, key=self.UCB_A.get)
                color = arm[0]
                self.result = self.env.sample(arm)
                self.results.append(self.result)
                self.N_A[arm] += 1
                self.N_C[color] += 1
                self.Q_A[arm] = self.Q_A[arm] * (self.N_A[arm] - 1) / self.N_A[arm] + self.result / self.N_A[arm]
                self.Q_C[color] = self.Q_C[color] * (self.N_C[color] - 1) / self.N_C[color] + self.result / self.N_C[color]
                self.UCB_A[arm] = self.ucb(self.Q_A[arm], self.N_A[arm])
                self.UCB_C[color] = self.ucb(self.Q_C[color], self.N_C[color])

            self.sumX += self.result

    def recommend(self, ):
        return max(self.Q_A, key=self.Q_A.get)
