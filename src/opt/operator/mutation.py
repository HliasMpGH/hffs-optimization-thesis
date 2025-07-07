import numpy as np

class Mutation:

    def __init__(self, prob=0.5):
        # the probability of executing a mutation on an offspring
        self.prob = prob

    def local_swap(self, offspring, ga_instance):

        def mutate(offspring, ga_instance):
            mutated_offspring = offspring.copy()

            k = np.random.randint(offspring.shape[0])

            for i in range(1, 4):
                if k - i > 0:
                    temp_offspring = swap(mutated_offspring, k, k - i)

                    if ga_instance.fitness_func(ga_instance, temp_offspring, "") > ga_instance.fitness_func(ga_instance, mutated_offspring, ""):
                        mutated_offspring = temp_offspring

                if k + i < offspring.shape[0]:
                    temp_offspring = swap(mutated_offspring, k, k - i)
                    if ga_instance.fitness_func(ga_instance, temp_offspring, "") > ga_instance.fitness_func(ga_instance, mutated_offspring, ""):
                        mutated_offspring = temp_offspring

            return mutated_offspring

        def swap(offspring, i, j):
            # dont manipulate the offspring in place
            temp_offspring = offspring.copy()

            temp_offspring[i], temp_offspring[j] = temp_offspring[j], temp_offspring[i]

            return temp_offspring


        if np.random.random() > self.prob:
            return offspring

        if np.random.random() < 0.5:
            mutated1 = mutate(offspring[0], ga_instance)
            return np.array([mutated1, offspring[1]])
        else:
            mutated2 = mutate(offspring[1], ga_instance)
            return np.array([mutated2, offspring[0]])
