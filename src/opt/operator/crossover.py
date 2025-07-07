import numpy as np
import copy
import random

from util.objectives import Objective

class Crossover:

    def __init__(self, problem, best_two_rule=False):
        self.problem = problem
        # if True, after crossover, return the best 2/4 solutions
        self.best_two_rule = best_two_rule

    '''
    Return an operator callable, matched by its name.
    '''
    def get_by_name(self, operator_name):
        match operator_name:
            case "rmpx": return self.random_maximal_preservative
            case "armpx": return self.antagonist_random_maximal_preservative
            case "ljmpx": return self.jobs_maximal_preservative
            case "mpobx": return self.preservative_order_block
            case "limpx_mpobx_combo": return self.limpx_mpobx_combo
            case "param_uniform": return self.parameterized_uniform
            case "swap": return self.swap
            case _: raise ValueError(f"Not recognized crossover operator '{operator_name}'")

    '''
    Use limpx and mpobx with 50% probability each.
    '''
    def limpx_mpobx_combo(self, parents, offspring_size, ga_instance):
        if np.random.random() < 0.5:
            print(f"executing jobs_maximal_preservative")
            return self.jobs_maximal_preservative(parents, offspring_size, ga_instance)
        else:
            print(f"executing preservative_order_block")
            return self.preservative_order_block(parents, offspring_size, ga_instance)

    def random_maximal_preservative(self, parents, offspring_size, ga_instance):
        def rmpx_cross(first_parent, second_parent, c1, c2, p1, offspring_genes_length):
            # define the offspring
            offspring = np.full(offspring_genes_length, -1)

            p1_segment = first_parent[c1:(c2 + 1)]
            print(f"parent segment = {p1_segment}")
            print(offspring)
            print(f"offspring segment = {offspring[p1:(p1 + c2 - c1 + 1)]}")

            offspring[p1:(p1 + c2 - c1 + 1)] = p1_segment
            print(f"offspring = {offspring}")

            # the rest of the offspring is completed from the other parent, in the order of appearance from its first position.
            placed_orders = set(p1_segment)
            print(f"placed = {placed_orders}")

            self.__place_orders_by_appearance(offspring, second_parent, placed_orders, offspring_genes_length)

            print(f"Offspring = {offspring}")
            return offspring

        n = offspring_size[1]
        c1, c2 = self.__get_random_crossover_points(n, 2)
        print(f"C1,C2 = [{c1}, {c2}]")

        # (ii) an insertion point pi is then chosen in the offspring O1, pi being a random number in the interval [1, n - ( C2 - C1)]
        p1 = np.random.randint(0, n - (c2 - c1)) # +1 to make it inclusive maybe?
        print(f"p1 = {p1}")

        o1 = rmpx_cross(parents[0], parents[1], c1, c2, p1, offspring_size[1])
        o2 = rmpx_cross(parents[1], parents[0], c1, c2, p1, offspring_size[1])

        print(f"CHILD ({offspring_size}): {o1, o2}")
        return np.array([o1, o2])

    def antagonist_random_maximal_preservative(self, parents, offspring_size, ga_instance):
        def armpx_cross(first_parent, second_parent, c1, c2, offspring_genes_length):
            # define the offspring
            offspring = np.full(offspring_genes_length, -1)

            offspring[:c1] = first_parent[:c1]
            offspring[(c2 + 1):] = first_parent[(c2 + 1):]

            print(f"offspring = {offspring}")

            # the rest of the offspring is completed from the other parent, in the order of appearance from its first position.
            placed_orders = set(offspring[:c1]) | set(offspring[(c2 + 1):])
            print(f"placed = {placed_orders}")

            self.__place_orders_by_appearance(offspring, second_parent, placed_orders, offspring_genes_length)

            print(f"Offspring = {offspring}")
            return offspring

        c1, c2 = self.__get_random_crossover_points(offspring_size[1], 2)

        print(f"C1,C2 = [{c1}, {c2}]")

        o1 = armpx_cross(parents[0], parents[1], c1, c2, offspring_size[1])
        o2 = armpx_cross(parents[1], parents[0], c1, c2, offspring_size[1])

        print(f"CHILD ({offspring_size}): {o1, o2}")
        return np.array([o1, o2])

    def __place_orders_by_appearance(self, offspring, parent, placed_orders, offspring_genes_length):
        for i in range(offspring_genes_length):
            # skip placed positions
            if offspring[i] != -1: continue

            # fill the offspring from the other parent based on order of appearance
            for order in parent:
                if order in placed_orders:
                    # skip placed orders
                    continue
                else:
                    # place the first valid order from parent
                    offspring[i] = order
                    placed_orders.add(order)
                    break

    def jobs_maximal_preservative(self, parents, offspring_size, ga_instance):

        def ljmpx_cross(first_parent, second_parent, c1, c2, p1, offspring_genes_length):
            # define the offspring
            offspring = np.full(offspring_genes_length, -1)

            p1_segment = first_parent[c1:(c2 + 1)]
            offspring[p1:(p1 + c2 - c1 + 1)] = p1_segment
            # the rest of the offspring O1 is completed from the other parent in the order of appearance from its first position.

            ll_size = p1
            rl_size = offspring_genes_length - (p1 + c2 - c1 + 1)

            left_list = []
            right_list = []

            placed_orders = set(p1_segment)

            for order in second_parent:
                if order in placed_orders: continue

                if len(left_list) < ll_size:
                    left_list.append(order)
                elif len(right_list) < rl_size:
                    right_list.append(order)

            for i in range(ll_size):
                if len(left_list) == 1:
                    # if there is only one order left, place it in the position
                    offspring[p1 - i - 1] = left_list[0]
                    break

                best_fitness = -1
                best_order = None
                for order in left_list:

                    temp_sequence = offspring.copy()

                    # try to place the order in the position
                    temp_sequence[p1 - i - 1] = order

                    unplaced_left_orders = list(set(left_list) - set([order]))

                    # place the rest of LL and RL randomly in the empty positions
                    for j in range(len(unplaced_left_orders)):
                        temp_sequence[j] = unplaced_left_orders[j]
                    for j in range(len(right_list)):
                        temp_sequence[p1 + c2 - c1 + j + 1] = right_list[j]

                    updated_processing_times = self.__update_processing_times(set(left_list + right_list) - set([order]))

                    # calculate fitness based on the updated processing times
                    makespan_fitness = Objective(self.problem | updated_processing_times).makespan_fitness(ga_instance, temp_sequence, "")

                    # record the best order for this position
                    if makespan_fitness > best_fitness:
                        best_fitness = makespan_fitness
                        best_order = order

                # place the best order in this position
                offspring[p1 - i - 1] = best_order

                # dont consider this order for the next iterations
                left_list.remove(best_order)

            for i in range(rl_size):

                if len(right_list) == 1:
                    # if there is only one order left, place it in the position
                    offspring[p1 + c2 - c1 + 1 + i] = right_list[0]
                    break

                best_fitness = -1
                best_order = None
                for order in right_list:
                    temp_sequence = offspring.copy()

                    # try to place the order in the position
                    temp_sequence[p1 + c2 - c1 + 1 + i] = order

                    unplaced_right_orders = list(set(right_list) - set([order]))

                    # place the rest of LL and RL randomly in the empty positions
                    for j in range(len(unplaced_right_orders)):
                        temp_sequence[p1 + c2 - c1 + 2 + i + j] = unplaced_right_orders[j]

                    updated_processing_times = self.__update_processing_times(set(left_list + right_list) - set([order]))

                    # calculate fitness based on the updated processing times
                    makespan_fitness = Objective(self.problem | updated_processing_times).makespan_fitness(ga_instance, temp_sequence, "")

                    # record the best order for this position
                    if makespan_fitness > best_fitness:
                        best_fitness = makespan_fitness
                        best_order = order

                # place the best order in this position
                offspring[p1 + c2 - c1 + 1 + i] = best_order
                # dont consider this order for the next iterations
                right_list.remove(best_order)

            return offspring

        c1, c2 = self.__get_random_crossover_points(offspring_size[1], 2)

        p1 = np.random.randint(0, offspring_size[1] - (c2 - c1)) # +1 to make it inclusive maybe?

        o1 = ljmpx_cross(parents[0], parents[1], c1, c2, p1, offspring_size[1])
        o2 = ljmpx_cross(parents[1], parents[0], c1, c2, p1, offspring_size[1])

        return np.array([o1, o2])

    def preservative_order_block(self, parents, offspring_size, ga_instance):
        '''Preservative Order Block Crossover (MPOBX) which works as follows.
        First, from the two parents, we insert the longest job blocks at the
        same positions, using four crossover points. After that, as in the LJMPX
        crossover, we calculate an approximate value Cmax using the pij and sijk
        for the unscheduled job positions. Then we place the remaining unscheduled
        jobs as in the LJMPX crossover, from a single job list. As shown in Figure 3,
        we place the block f3, 1, 4g from P1 and block f9, 7, 2g from P2. The unscheduled
        job list contains jobs 5, 6 and 8. These jobs will be placed one by one using the Cmax value.'''

        def mpobx_cross(first_parent, second_parent, largest_blocks, offspring_genes_length):
            # define the offspring
            offspring = np.full(offspring_genes_length, -1)

            placed_orders = set()
            available_positions = set(range(offspring_genes_length))

            current_parent = first_parent
            for block in largest_blocks:
                parent_block = current_parent[block[0]:block[1]]
                offspring[block[0]:block[1]] = parent_block

                available_positions -= set(range(block[0], block[1]))
                for pos in range(block[0], block[1]):
                    # check for any duplicate orders in the offspring
                    if offspring[pos] in placed_orders:
                        offspring[pos] = -1
                        available_positions.add(pos)
                # add the orders from the block to the placed orders
                for order in parent_block:
                    placed_orders.add(order)
                # switch parent for the next block
                current_parent = second_parent if current_parent is first_parent else first_parent

            # find the remaining orders to place

            available_orders = set(parents[0]) - placed_orders
            placed_positions = set()
            for position in available_positions:
                if len(available_orders) == 1:
                    # if there is only one order left, place it in the position
                    offspring[position] = available_orders.pop()
                    break

                best_fitness = -1
                best_order = None
                for order in available_orders:
                    temp_sequence = offspring.copy()
                    temp_sequence[position] = order

                    unplaced_orders = list(available_orders - set([order]))

                    # place the rest of LL and RL randomly in the empty positions
                    for j in available_positions - placed_positions - set([position]):
                        temp_sequence[j] = unplaced_orders.pop(0)

                    updated_processing_times = self.__update_processing_times(available_orders - set([order]))
                    # calculate fitness based on the updated processing times
                    makespan_fitness = Objective(self.problem | updated_processing_times).makespan_fitness(ga_instance, temp_sequence, "")

                    # record the best order for this position
                    if makespan_fitness > best_fitness:
                        best_fitness = makespan_fitness
                        best_order = order

                # place the best order in this position
                offspring[position] = best_order
                available_orders.remove(best_order)
                placed_positions.add(position)

            return offspring

        cut_points = self.__get_random_crossover_points(offspring_size[1], 4)

        boundaries = [0] + list(cut_points) + [offspring_size[1]]

        # Calculate block sizes
        blocks = []
        max_size_block = (0, 0, -1)
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            size = end - start
            blocks.append((start, end, size))
            if size > max_size_block[2]:
                max_size_block = (start, end, size)

        # Get all blocks with maximum size
        largest_blocks = [(start, end) for start, end, size in blocks if size == max_size_block[2]]

        o1 = mpobx_cross(parents[0], parents[1], largest_blocks, offspring_size[1])
        o2 = mpobx_cross(parents[1], parents[0], largest_blocks, offspring_size[1])

        return np.array([o1, o2])

    def __get_random_crossover_points(self, chromosome_size, n):
        return tuple(sorted(random.sample(range(chromosome_size), n)))

    def __update_processing_times(self, orders):
        updated_processing_times = copy.deepcopy(self.problem["process"])

        # for order in orders:
        for stage in self.problem["stages"]:

            avg_pi = np.mean([
                self.problem["process"][stage][job]
                for order in orders
                for job in self.problem["orders"][f"Order-{int(order)}"]
            ])

            for order in orders:
                for job in self.problem["orders"][f"Order-{int(order)}"]:
                    updated_processing_times[stage][job] = avg_pi
        # return the updated processing times
        return updated_processing_times


    def parameterized_uniform(self, parents, offspring_size, ga_instance):

        def fix_offspring(offspring):
            o1_placed_orders = set(offspring[0]) - set([-1])
            o2_placed_orders = set(offspring[1]) - set([-1])
            if np.random.random() < 0.7:
                if -1 in offspring[0]:
                    self.__place_orders_by_appearance(offspring[0], parents[0], o1_placed_orders, offspring_size[1])
                if -1 in offspring[1]:
                    self.__place_orders_by_appearance(offspring[1], parents[1], o2_placed_orders, offspring_size[1])
            else:
                if -1 in offspring[0]:
                    self.__place_orders_by_appearance(offspring[0], parents[1], o1_placed_orders, offspring_size[1])
                if -1 in offspring[1]:
                    self.__place_orders_by_appearance(offspring[1], parents[0], o2_placed_orders, offspring_size[1])

            for i in range(offspring_size[1]):
                if offspring[0, i] == -1:
                    # find the first available order from the parents
                    for j in range(offspring_size[1]):
                        if parents[0, j] not in offspring[0]:
                            offspring[0, i] = parents[0, j]
                            break
                if offspring[1, i] == -1:
                    # find the first available order from the parents
                    for j in range(offspring_size[1]):
                        if parents[1, j] not in offspring[1]:
                            offspring[1, i] = parents[1, j]
                            break

        # define the offspring
        offspring = np.full(offspring_size, -1)

        for i in range(offspring_size[1]):
            if np.random.random() < 0.7:
                # ensure no duplicate orders in the offsprings
                if parents[0, i] not in offspring[0]:
                    offspring[0, i] = parents[0, i]
                if parents[1, i] not in offspring[1]:
                    offspring[1, i] = parents[1, i]
            else:
                # ensure no duplicate orders in the offsprings
                if parents[1, i] not in offspring[0]:
                    offspring[0, i] = parents[1, i]
                if parents[0, i] not in offspring[1]:
                    offspring[1, i] = parents[0, i]

        # fill any missing orders in the offspring
        fix_offspring(offspring)

        return offspring


    def swap(self, parents, offspring_size, ga_instance):

        def swap_cross(offspring, parent, other_parent_fitness):
            for i in range(offspring_size[1]):
                # if the order is the same as the parent, skip it
                if offspring[i] == parent[i]:
                    continue

                # find the index of the order that differs
                j = np.where(offspring == parent[i])[0][0]


                # swap the orders to be the same as the parent
                offspring[i], offspring[j] = offspring[j], offspring[i]

                if ga_instance.fitness_func(ga_instance, offspring, "") > other_parent_fitness:
                    # print("New offspring is better than old parent")
                    return offspring

            return offspring

        p1 = parents[0].copy()
        p2 = parents[1].copy()
        o1 = swap_cross(p1, p2, ga_instance.fitness_func(ga_instance, p1, ""))
        # print("generating second offspring")

        p1 = parents[0].copy()
        p2 = parents[1].copy()
        o2 = swap_cross(p2, p1, ga_instance.fitness_func(ga_instance, p2, ""))

        # print(f"CHILD ({offspring_size}): {offspring}")
        return np.array([o1, o2])
