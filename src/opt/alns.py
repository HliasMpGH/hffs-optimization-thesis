from util.objectives import Objective

import random
import numpy as np
import copy
from scipy.spatial import distance
import time
import math


class DestroyOperators:

    def __init__(self, problem_attributes):
        self.problem_attributes = problem_attributes

        self.order_dissimilarity_table = self.compute_order_dissimilarity_table()

        # initialize the weights of all available operators
        self.operators_weights = {
            member: (1, 0) # (score, number of uses)
            for member in dir(self)
            if callable(getattr(self, member)) and member.startswith("destroy_")
        }

    '''
    Computes and returns a |orders| x |orders| dissimilarity table,
    based on the euclidean distance between the average processing
    times of the orders' jobs on each stage.
    '''
    def compute_order_dissimilarity_table(self):
        orders = self.problem_attributes["orders"]
        processing_times = self.problem_attributes["process"]

        # print(processing_times, end="\n\n\n")
        order_process_times = dict()
        for order_name, order_jobs in orders.items():
            order_process_times[order_name] = []
            # compute the order processing time per stage, as the average time of the orders' jobs in that stage
            for stage in processing_times.keys():

                # dont consider the jobs that are not present in the stage
                job_times = [processing_times[stage][job] for job in order_jobs if processing_times[stage][job] > 0]

                # add the average processing time of the jobs in the stage (or 0 if no jobs in the stage)
                order_process_times[order_name].append(np.mean(job_times) if job_times else 0)

        # print(order_process_times)

        # create the dissimilarity table
        order_dissimilarity_table = dict()
        for order1 in orders.keys():
            for order2 in orders.keys():

                # dont repeat information
                if (order2, order1) in order_dissimilarity_table.keys():
                    continue

                # compute the dissimilarity between the two orders
                d = distance.euclidean(order_process_times[order1], order_process_times[order2])
                # print(f"Distance between {order1} and {order2}: {d}")
                # save the dissimilarity in the table
                order_dissimilarity_table[(order1, order2)] = d

        # print(f"DISSIMILARITY TABLE: {order_dissimilarity_table}")
        return order_dissimilarity_table

    def get_by_roulette(self):
        print("roulette on destroy")
        total_weight_sum = sum(op_weight[0] for op_weight in self.operators_weights.values())

        selection_probs_destroy = [
            operator_weight[0] / total_weight_sum
            for operator_weight in self.operators_weights.values()
        ]

        # print(f"total probabilities: {selection_probs_destroy}")

        chosen_destroy_operator = np.random.choice(
            list(self.operators_weights.keys()),
            p=selection_probs_destroy
        )

        return chosen_destroy_operator

    '''
    Update the weights of the last destroy operator used.
    Larger weight means that the operator wields good results
    and has a largest probability of being chosen in the future.
    '''
    # r=0.9 based on (Shao et al, 2024)
    def update_weight(self, destroy_operator, recreated_sequence_cost, r=0.9):
        # update destroy operator
        d_op = self.operators_weights[destroy_operator]
        d_op_score = d_op[0]
        d_op_calls = d_op[1]

        # w_{i, j+1} = w_{i, j} * (1 - r) + r * (p_i / u_i)
        # based on (Ropke et al) and (Shao et al, 2024)
        # reverse the makespan to achieve a larger score when the cost is smaller
        d_new_score = d_op_score * (1 - r) + r * ((1 / recreated_sequence_cost) / (d_op_calls + 1))

        self.operators_weights[destroy_operator] = (d_new_score, d_op_calls + 1)

    '''
    Remove a random set of orders from a given sequence
    and return them, along their positions.
    '''
    # n=4 based on (Shao et al, 2024)
    def destroy_random_orders(self, seq, n=4):
        print(f"random destroying {seq}.")
        # pick *n* random position
        random_positions = random.sample(range(len(seq)), n)
        # return the orders in the positions
        # print(f"chose: {random_positions} with jobs: {[seq[p] for p in random_positions]}")
        return [seq[p] for p in random_positions], random_positions

    '''
    Using a roulette wheel, remove the most dissimilar orders,
    based on a dissimilarity table.
    Return the orders, along with their positions.
    '''
    def destroy_dissimilar_orders(self, seq, n_range=(1, 4)):
        '''
        Return exactly *n* unique dissimilar orders from the sequence,
        using a roulette wheel.
        '''
        def get_dissimilar_orders(n):
            # orders_pairs_to_remove = n // 2 # even number of orders to remove

            order_dissimilarity_sum = sum(distance for distance in self.order_dissimilarity_table.values())
            # print(f"MATRIX:")
            # for order, score in self.order_dissimilarity_table.items():
            #     print(f"{order}-> {score}")
            # probabilities of the order pairs (higher score means higher probability)
            selection_probs_orders = [
                distance / order_dissimilarity_sum
                for distance in self.order_dissimilarity_table.values()
            ]

            # print(f"Dissimilarity probs: {selection_probs_orders}. Sum: {sum(selection_probs_orders)})")

            # pick the order pairs using the probabilities


            # create indices for the order pairs to sample from them
            order_pairs = list(self.order_dissimilarity_table.keys())
            order_pairs_indices = np.arange(len(order_pairs))

            chosen_orders = set()

            # pick orders pairs until we reach the number of orders to remove
            while len(chosen_orders) < n:
                chosen_orders_pair_index = np.random.choice(
                    order_pairs_indices, # choice() cant take list of tuples
                    p=selection_probs_orders,
                    # size=orders_pairs_to_remove
                )

                # map the index back to the order pair
                chosen_order_pair = order_pairs[chosen_orders_pair_index]
                # print(f"chosen pair: {chosen_order_pair}")
                # add the two orders to the chosen set
                chosen_orders.add(chosen_order_pair[0])
                chosen_orders.add(chosen_order_pair[1])

                # if a point is reached where only one order is left to be removed (not at least 2),
                # pick a random order that is not already chosen
                if len(chosen_orders) + 1 == n:
                    chosen_orders.add(
                        random.choice(
                            list(set(seq).difference(chosen_orders))
                        )
                    )
                    # this solves the following problems:
                    # i) if n is odd, we wouldnt be able to reach it by removing pairs of orders
                    # ii) if we picked two times the same order (through 2 pairs), we would still be left with one order to remove
                    break

            return list(chosen_orders)

        print(f"dissimilarity destroying {seq}.")
        # order_dissimilarity_table = order_dissimilarity_table["dissimilarity_table"]
        # the number of orders to remove is random, between the given range
        n = random.randint(n_range[0], n_range[1])
        # print(f"removing {n} orders")
        # get the dissimilar orders to remove
        order_to_remove = get_dissimilar_orders(n)

        # return the orders in the positions
        # print(f"chose: {[seq.index(order) for order in order_to_remove]} with orders: {order_to_remove}")
        return order_to_remove, [seq.index(order) for order in order_to_remove]

    def destroy_block_orders(self, seq):
        print(f"random block destroying {seq}.")
        r1 = np.random.randint(len(seq))
        r2 = np.random.randint(len(seq))
        c1, c2 = sorted([r1, r2])

        # print(f"block range = {c1, c2}")

        order_block = seq[c1:(c2 + 1)]

        # print(f"block = {order_block}")

        # print(f"chose: {[pos for pos in range(c1, c2 + 1)]} with orders: {order_block}")
        return order_block, [pos for pos in range(c1, c2 + 1)]

    def destroy_shortest_greedy(self, seq):
        print(f"shortest greedy destroying {seq}.")
        n = np.random.randint(2, len(seq))
        # print(f"size = {n}")

        # compute a random number for each order
        r = {
            str(order): np.random.random()
            for order in seq
        }
        # print(f"\n\n\n{r}\n\n\n")

        # compute the attractiveness for each order
        attractiveness = {
            order: self.get_order_attractiveness(order, r, a_constant=0.5)
            for order in seq
        }
        # for o, a in attractiveness.items():
        #     print(f"{o}: {a}")

        # print(f"SUM={sum(attractiveness.values())}")

        removed_orders = np.random.choice(list(attractiveness.keys()), p=list(attractiveness.values()), size=n, replace=False)
        # removed_orders = self.get_orders_by_attractiveness(attractiveness, size=n)

        # print(f"chose: {[seq.index(order) for order in removed_orders]} with orders: {removed_orders}")
        return removed_orders, [seq.index(order) for order in removed_orders]

    def destroy_longest_greedy(self, seq):
        print(f"longest greedy destroying {seq}.")
        n = np.random.randint(2, len(seq))
        # print(f"size = {n}")

        # compute a random number for each order
        r = {
            str(order): np.random.random()
            for order in seq
        }

        # compute the attractiveness for each order
        attractiveness = {
            order: self.get_order_attractiveness(order, r, a_constant=0.5, total_processing_time_criterion="max")
            for order in seq
        }
        # for o, a in attractiveness.items():
            # print(f"{o}: {a}")

        # print(f"SUM={sum(attractiveness.values())}")

        removed_orders = np.random.choice(list(attractiveness.keys()), p=list(attractiveness.values()), size=n, replace=False)
        # removed_orders = self.get_orders_by_attractiveness(attractiveness, size=n)

        # print(f"chose: {[seq.index(order) for order in removed_orders]} with orders: {removed_orders}")
        return removed_orders, [seq.index(order) for order in removed_orders]

    def get_order_attractiveness(self, order, r, a_constant, total_processing_time_criterion="min"):

        if total_processing_time_criterion == "min":
            numerator = r[order] * (a_constant / min(stage[job] for job in self.problem_attributes["orders"][order] for stage in self.problem_attributes["process"].values() if stage[job] > 0))
            denominator = sum(
                (r * a_constant / min(stage[job]
                # for order in total_orders
                for job in self.problem_attributes["orders"][o]
                for stage in self.problem_attributes["process"].values() if stage[job] > 0))
                for o, r in r.items()
            )
        else:
            numerator = r[order] * max(stage[job] for job in self.problem_attributes["orders"][order] for stage in self.problem_attributes["process"].values())
            denominator = sum(
                (r * max(
                        stage[job]
                        # for order in total_orders
                        for job in self.problem_attributes["orders"][o]
                        for stage in self.problem_attributes["process"].values()
                    )
                ) for o, r in r.items()
            )
        return numerator / denominator

    def get_orders_by_attractiveness(self, attractiveness, size):
        # compute total attractiveness
        sum_attractiveness = sum(attractiveness.values())

        # compute the score for each order (probability)
        scores = {
            order: attr / sum_attractiveness
            for order, attr in attractiveness.items()
        }

        # remove orders based on probability
        return np.random.choice(list(scores.keys()), p=list(scores.values()), size=size, replace=False)

    def destroy_two_tails(self, seq):
        print(f"two-tails destroying {seq}.")
        n = np.random.randint(1, len(seq))
        # print(f"size = {n}")

        # compute a random number for each order
        r = {
            str(order): np.random.random()
            for order in seq
        }

        # compute the attractiveness for each order
        attractiveness = {
            order : self.get_order_attractiveness(order, r, a_constant=0.5)
            for order in seq
        }
        # for o, a in attractiveness.items():
        #     print(f"{o}: {a}")

        # print(f"SUM={sum(attractiveness.values())}")
        removed_orders = np.random.choice(list(attractiveness.keys()), p=list(attractiveness.values()), size=n, replace=False)
        # removed_orders = self.get_orders_by_attractiveness(attractiveness, size=n)
        # print(f"removed orders: {removed_orders}")

        L = []
        left = 0
        right = len(removed_orders) - 1

        while left <= right:
            L.append(removed_orders[left])
            left += 1

            if left <= right:
                L.append(removed_orders[right])
                right -= 1

        # print(f"chose: {[seq.index(order) for order in L]} with orders: {L}")
        return L, [seq.index(order) for order in L]


class RecreateOperators:

    def __init__(self, problem_attributes, calculate_cost):
        self.problem_attributes = problem_attributes
        self.calculate_cost = calculate_cost

        # initialize the weights of all available operators
        self.operators_weights = {
            member: (1, 0) # (score, number of uses)
            for member in dir(self)
            if callable(getattr(self, member)) and member.startswith("recreate_")
        }
        # self.operators_weights["recreate_greedily"] = (100000, 0)
        print(self.operators_weights)

    def get_by_roulette(self):
        print("roulette on recreate")
        total_weight_sum = sum(op_weight[0] for op_weight in self.operators_weights.values())

        selection_probs_recreate = [
            operator_weight[0] / total_weight_sum
            for operator_weight in self.operators_weights.values()
        ]

        # print(f"total probabilities: {selection_probs_recreate}")

        chosen_recreate_operator = np.random.choice(
            list(self.operators_weights.keys()),
            p=selection_probs_recreate
        )

        return chosen_recreate_operator

    def update_weight(self, recreate_operator, recreated_sequence_cost, r=0.9):
        # update recreation operator
        r_op = self.operators_weights[recreate_operator]
        r_op_score = r_op[0]
        r_op_calls = r_op[1]

        r_new_score = r_op_score * (1 - r) + r * (1 / recreated_sequence_cost / (r_op_calls + 1))

        self.operators_weights[recreate_operator] = (r_new_score, r_op_calls + 1)

    '''
    Given a sequence, some removed orders and some positions,
    allocate randomly the orders in the positions, inplace in
    the sequence.
    '''
    def recreate_randomly(self, seq, removed_orders, removed_positions):
        print("random recreation.")
        # randomize both lists
        random.shuffle(removed_orders)
        random.shuffle(removed_positions)
        # print(f"shuffled: {removed_orders} and {removed_positions}")

        # assign all orders on random valid positions
        for position, order in enumerate(removed_orders):
            random_position = removed_positions[position]
            seq[random_position] = order

    # '''
    # Given a sequence, some removed orders, some positions, and
    # a cost function, allocate the orders in their optimal positions, inplace in
    # the sequence.
    # '''
    # def recreate_optimally(seq, removed_orders, removed_positions, data_for_cost_calculation):
    #     print("optimal recreation.")
    #     best_sequence = None
    #     best_cost = float('inf')
    #     # best_solution = []

    #     # try all different order permutations
    #     for order_permutation in permutations(removed_orders):
    #         # Create a copy to test this permutation
    #         # temp_sequence = copy.deepcopy(seq)
    #         for position, order in zip(removed_positions, order_permutation):
    #             seq[position] = order

    #         current_cost, _, _, _ = cost(seq, data_for_cost_calculation)
    #         if current_cost < best_cost:
    #             best_cost = current_cost
    #             best_sequence = seq

    #     # update the input sequence in place
    #     for pos in range(len(seq)):
    #         seq[pos] = best_sequence[pos]

    '''
    Given a sequence, some removed orders, some positions, and
    a cost function, allocate the orders greedily, by performing
    the best swap for each order and locking the swapped orders
    for the next iterations, inplace in the sequence.
    '''
    def recreate_greedily(self, seq, removed_orders, removed_positions):
        print("greedy recreation.")
        placed_orders = set()
        for order in removed_orders:
            if order in placed_orders:
                # this order is already placed in the sequence
                continue

            if len(removed_positions) == 1:
                # place the order in the only available position
                print(f"placing {order} in the only available position {removed_positions[0]}")
                seq[removed_positions[0]] = order
                break
            elif len(removed_positions) == 2:
                # place the order in the only valid position
                print(f"placing {order} in the only valid position {removed_positions[1]}")
                seq[removed_positions[1]] = order
                removed_positions.remove(removed_positions[1])
                continue

            print(f"FIND BEST FOR ORDER {order}")
            # find the best position for the order in the sequence
            best_position = -1
            best_cost = float('inf')

            # the order that will be swapped with the current order
            best_swapped_order = None

            # keep the current position of the order in the sequence
            current_order_position = seq.index(order)
            c_o_p = removed_positions[0]

            if c_o_p != current_order_position:
                raise ValueError(f"Current order position {current_order_position} does not match the expected position {c_o_p} in the removed positions {removed_positions}")

            # try all positions where the order can be placed
            for position in removed_positions[1:]: # dont try the first position, as it is already occupied by the order
                print(f"trying position {position}")
                # create a copy of the sequence to test this order in this position
                temp_sequence = copy.deepcopy(seq)

                # swap the order with the one in the position
                order_to_swap = temp_sequence[position]
                temp_sequence[position] = order
                temp_sequence[current_order_position] = order_to_swap
                print(f"result of try: {temp_sequence}")
                current_cost = self.calculate_cost(temp_sequence)

                # record the best swap
                if current_cost < best_cost:
                    print(f"new best!")
                    best_cost = current_cost
                    best_position = position
                    best_swapped_order = order_to_swap

            if best_position != -1:
                # perform the best swap in the original sequence
                if best_position == current_order_position:
                    raise ValueError(f"Best position {best_position} is the same as the current order position {current_order_position}. This should not happen.")

                print(f"PLACED ORDER: {order} in position {best_position} and swapped with {best_swapped_order} on position {current_order_position}")
                seq[best_position] = order
                seq[current_order_position] = best_swapped_order
                # print(f"swapped {best_position} with {current_order_position}: {seq}")
                # the two positions are no longer available for other orders
                print(f"REMOVING POSITIONS: {best_position} and {current_order_position}")
                removed_positions.remove(best_position)

                # if best_position != current_order_position:
                #     removed_positions.remove(current_order_position)
                removed_positions.remove(current_order_position)


                # dont try to remove the order that got swapped
                # its position is already defined by the swap with the current order
                placed_orders.add(best_swapped_order)
                placed_orders.add(order)

        # if best_cost != float('inf'):
        #     # also return the cost of the sequence (last bests_cost) to store for later use
        #     return best_cost

    def recreate_block(self, seq, removed_orders, removed_positions):
        removed_orders = list(removed_orders)
        print("block recreation.")
        partial_sequence = [order for order in seq if order not in removed_orders]
        print(removed_positions)
        print(partial_sequence)
        # block_length = len(removed_positions)
        best_cost = float('inf')
        best_sequence = seq
        for pos in range(len(partial_sequence) + 1):
            if pos == removed_positions[0]:
                # print("skipping same position")
                continue
            # print(f"placing block on pos = {pos}")
            # if pos in removed_positions: continue

            # dont modify the original sequence
            temp_seq = (partial_sequence[:pos] + removed_orders + partial_sequence[pos:])

            # place the block in each position
            # temp_seq[pos:(pos + block_length)] = removed_orders
            # print(f"temp_seq = {temp_seq}")

            # dont try and place the block in the destroyed positions
            # if temp_seq == seq:
            #     print(f"same as old seq")
            #     continue

            current_cost = self.calculate_cost(temp_seq)
            # print(f"CURRENT COST = {current_cost}, BEST COST = {best_cost}")
            if current_cost < best_cost:
                # print("found better position")
                best_cost = current_cost
                best_sequence = temp_seq

        # print(f"FINAL BEST = {best_sequence}")
        # update the input sequence in place
        for pos in range(len(seq)):
            seq[pos] = best_sequence[pos]

        if best_cost != float('inf'):
            # also return the cost of the sequence to store for later use
            return best_cost

    def recreate_cyclic(self, seq, removed_orders, removed_positions):
        print("cyclic recreation")
        new_positions = [removed_positions[(i + 1) % len(removed_positions)]
                        for i in range(len(removed_positions))]

        # print(f"removed orders: {removed_orders}")
        # print(f"removed positions: {removed_positions}")
        # print(f"shifted positions: {new_positions}")
        for i, pos in enumerate(new_positions):
            seq[pos] = removed_orders[i]

        # print(f"new seq: {seq}")


class ALNS:
    def __init__(self, problem_attributes, starting_solution=None):
        self.problem_attributes = problem_attributes

        # define cost function & operator sets
        self.calculate_cost = Objective(self.problem_attributes).makespan_cost
        self.destroy_operators = DestroyOperators(self.problem_attributes)
        self.recreate_operators = RecreateOperators(self.problem_attributes, self.calculate_cost)

        if starting_solution is None:
            # do an ALNS move to produce it?
            pass
        self.current_solution = starting_solution


    def __choose_operators(self):
        print(f"choosing operators")
        # pick destroy operator
        destroy_op_name = self.destroy_operators.get_by_roulette()
        self.destroy_operator = getattr(self.destroy_operators, destroy_op_name)

        # allow only block recreation on block destroy
        if destroy_op_name == "destroy_block_orders":
            self.recreate_operator = self.recreate_operators.recreate_block
        else:
            # pick recreate operator
            recreate_op_name = self.recreate_operators.get_by_roulette()
            self.recreate_operator = getattr(self.recreate_operators, recreate_op_name)


    def __destroy(self):
        print(f"now destroying {self.current_solution}")
        self.destroyed_orders, self.destroyed_positions = self.destroy_operator(self.current_solution)
        print(f"destruction result : removed_orders = {self.destroyed_orders} and positions= {self.destroyed_positions}")


    def __recreate(self):
        print(f"now recreating {self.current_solution}")
        self.recreation_result = self.current_solution.copy()
        self.current_cost = self.recreate_operator(self.recreation_result, self.destroyed_orders, self.destroyed_positions)
        print(f"recreation result is temp with {self.recreation_result} but current is {self.current_solution}")

        # validity check - costs should match
        if self.current_cost is not None:
            if self.current_cost != self.calculate_cost(self.recreation_result):
                raise ValueError(f"Costs do not match: ({self.current_cost, self.calculate_cost(self.recreation_result)})")


    def __update_operator_weights(self):
        print(f"updating weights")
        # if the creation method returned a cost, dont recalculate
        if self.current_cost is None:
            self.current_cost = self.calculate_cost(self.recreation_result)

        self.destroy_operators.update_weight(self.destroy_operator.__name__, self.current_cost)
        self.recreate_operators.update_weight(self.recreate_operator.__name__, self.current_cost)

        # print(f"new weights: {self.destroy_operators.operators_weights}")
        # print(f"new weights: {self.recreate_operators.operators_weights}")


    def __update_best(self):
        current_time = time.time()
        if self.current_cost < self.best_cost:
            print(f"new global best = {self.current_cost} (old best={self.best_cost})")
            self.best_cost = self.current_cost
            self.best_solution = self.recreation_result
            self.iteration_found = self.iterations_completed
            self.time_found = current_time - self.start_time


    def __accept_criteria(self):
        current_time_passed = time.time() - self.start_time
        # if self.old_cost is None:
        #     self.old_cost = self.calculate_cost(self.current_solution)

        # validity check - costs should match
        if self.old_cost != self.calculate_cost(self.current_solution):
            raise ValueError(f"Old Costs dont match: {self.old_cost, self.calculate_cost(self.current_solution)}")

        delta = self.current_cost - self.old_cost
        print(f"new cost: {self.current_cost} old cost: {self.old_cost} ({delta})")
        if delta < 0:
            print(f"accepted better")
            self.current_solution = self.recreation_result
            self.old_cost = self.current_cost

            # update global best if needed
            self.__update_best()
        else:
            p = 1 - math.exp(
                -((delta / self.old_cost)
                + (current_time_passed - self.time_limit / 2) ** 2)
            )
            print(f"delta/old_cost={delta/self.old_cost}")
            print(f"(current_time - self.time_limit / 2) ** 2={(current_time_passed - self.time_limit / 2) ** 2}")
            print(f"sum={(delta / self.old_cost) + (current_time_passed - self.time_limit / 2) ** 2}")
            print(f"accept prob={p}")
            if np.random.random() < p:
                print(f"accepted worse")
                self.current_solution = self.recreation_result
                self.old_cost = self.current_cost


    def run(self, starting_solution, time_limit=10):
        self.best_cost = self.calculate_cost(starting_solution)
        self.best_solution = starting_solution
        self.old_cost = self.best_cost
        self.iteration_found = -1
        self.total_time = -1
        self.time_found = -1
        self.time_limit = time_limit

        self.current_solution = starting_solution

        self.iterations_completed = 0
        self.start_time = time.time()

        # run while on time limit
        while time.time() - self.start_time < time_limit:
            self.__choose_operators()

            self.__destroy()

            self.__recreate()

            self.__update_operator_weights()

            self.iterations_completed += 1

            self.__accept_criteria()
        self.total_time = time.time() - self.start_time
