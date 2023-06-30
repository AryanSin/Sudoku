import numpy as np
import random
import operator
from past.builtins import range
import sys
import time
random.seed()

Nd = 9  

class Population(object):

    def __init__(self):
        self.candidates = []
        return

    def seed(self, Nc, given):
        self.candidates = []

        helper = Candidate()
        helper.values = [[[] for i in range(Nd)] for j in range(Nd)]
        for row in range(0, Nd):
            for column in range(0, Nd):
                for value in range(1, 10):
                    avail = given.values[row][column] == 0 and not (given.is_column_duplicate(column, value) or given.is_block_duplicate(row, column, value) or given.is_row_duplicate(row, value))
                    if (avail):
                        helper.values[row][column].append(value)
                    elif given.values[row][column] != 0:
                        helper.values[row][column].append(given.values[row][column])
                        break

        for p in range(0, Nc):
            g = Candidate()
            for i in range(0, Nd):
                row = np.zeros(Nd)

                for j in range(0, Nd):  

                    # if given.values[i][j] != 0:
                    #     row[j] = given.values[i][j]
                    # else: 
                    #     row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)]
                    row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)] if given.values[i][j] == 0 else given.values[i][j]

                ii = 0
                while len(list(set(row))) != Nd:
                    ii += 1
                    if(ii<=500000):
                        for j in range(0, Nd):
                            if given.values[i][j] == 0:
                                row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)]
                    else:
                        return 0

                g.values[i] = row
            self.candidates.append(g)
        self.update_fitness()
        return 1

    def update_fitness(self):
        for i in range(len(self.candidates)):
            self.candidates[i].update_fitness()
        return


    def sort(self):
        self.candidates.sort(key=lambda x: x.fitness, reverse=True)
        return

class Candidate(object):
    
    def __init__(self):
        self.values = [[0 for i in range(Nd)] for j in range(Nd)]
        self.values = np.reshape(self.values, (Nd, Nd))
        self.fitness = None
        return

    def update_fitness(self):

        column_count, block_count , column_sum, block_sum = np.zeros(Nd), np.zeros(Nd), 0, 0


        self.values = self.values.astype(int)
        for j in range(0, Nd):
            for i in range(0, Nd):
                column_count[self.values[i][j] - 1] += 1
        
            for k in range(len(column_count)):
                if column_count[k] == 1:
                    column_sum += (1/Nd)/Nd
            column_count = np.zeros(Nd)

        for i in range(0, Nd, 3):
            for j in range(0, Nd, 3):
                dir = [0 ,1 ,2]
                for k in dir:
                    for l in dir:
                        block_count[self.values[i + k][j + l] - 1] += 1
                for k in range(len(block_count)):
                    if block_count[k] == 1:
                        block_sum += (1/Nd)/Nd
                block_count = np.zeros(Nd)

        fitness = 1.0 if int(column_sum) == 1 and int(block_sum) == 1 else column_sum * block_sum

        self.fitness = fitness
        return

    def mutate(self, mutation_rate, given):
        """ Mutate a candidate by picking a row, and then picking two values within that row to swap. """

        r = random.uniform(0, 1.1)
        while r > 1:  
            r = random.uniform(0, 1.1)

        success = False
        if r < mutation_rate: 
            while not success:
                row1 = random.randint(0, 8)
                row2 = random.randint(0, 8)
                row2 = row1

                from_column = random.randint(0, 8)
                to_column = random.randint(0, 8)
                while from_column == to_column:
                    from_column = random.randint(0, 8)
                    to_column = random.randint(0, 8)

                if given.values[row1][from_column] == 0 and given.values[row1][to_column] == 0:
                    if not given.is_column_duplicate(to_column, self.values[row1][from_column]) and not given.is_column_duplicate(from_column, self.values[row2][to_column]) and not given.is_block_duplicate(row2, to_column, self.values[row1][from_column]) and not given.is_block_duplicate(row1, from_column, self.values[row2][to_column]):
                        temp = self.values[row2][to_column]
                        self.values[row2][to_column] = self.values[row1][from_column]
                        self.values[row1][from_column] = temp
                        success = True

        return success


class Fixed(Candidate):
    """ fixed/given values. """

    def __init__(self, values):
        self.values = values
        return

    def is_row_duplicate(self, row, value):
        """ Check duplicate in a row. """
        return value in self.values[row, :]

    def is_column_duplicate(self, column, value):
        """ Check duplicate in a column. """
        return value in self.values[:, column]

    def is_block_duplicate(self, row, column, value):
        """ Check duplicate in a 3 x 3 block. """
        i = 3 * (int(row / 3))
        j = 3 * (int(column / 3))

        dir = [0 , 1 , 2]
        for x in dir:
            for y in dir:
                if self.values[i+x][j+y] == value:
                    return True
        return False

    def make_index(self, v):
        return int(v/3)*3

    def no_duplicates(self):
        for row in range(0, Nd):
            for col in range(0, Nd):
                if self.values[row][col] != 0:

                    cnt1 = list(self.values[row]).count(self.values[row][col])
                    cnt2 = list(self.values[:,col]).count(self.values[row][col])

                    block_values = [y[self.make_index(col):self.make_index(col)+3] for y in
                                    self.values[self.make_index(row):self.make_index(row)+3]]
                    block_values_ = [int(x) for y in block_values for x in y]
                    cnt3 = block_values_.count(self.values[row][col])

                    if cnt1 > 1 or cnt2 > 1 or cnt3 > 1:
                        return False
        return True

class Tournament(object):
    """ The crossover function requires two parents to be selected from the population pool. The Tournament class is used to do this.

    Two individuals are selected from the population pool and a random number in [0, 1] is chosen. If this number is less than the 'selection rate' (e.g. 0.85), then the fitter individual is selected; otherwise, the weaker one is selected.
    """

    def __init__(self):
        return

    def compete(self, candidates):
        """ Pick 2 random candidates from the population and get them to compete against each other. """
        c1, c2 = random.sample(candidates, 2)
        f1, f2 = c1.fitness, c2.fitness

        fittest, weakest = (c1, c2) if f1 > f2 else (c2, c1)
        selection_rate = 0.80
        r = 1.1
        while (r > 1): 
            r = random.uniform(0, 1.1)
        return fittest if r < selection_rate else weakest


class CycleCrossover(object):
    """ Crossover relates to the analogy of genes within each parent candidate
    mixing together in the hopes of creating a fitter child candidate.
    Cycle crossover is used here (see e.g. A. E. Eiben, J. E. Smith.
    Introduction to Evolutionary Computing. Springer, 2007). """

    def __init__(self):
        return

    def crossover(self, parent1, parent2, crossover_rate):
        """ Create two new child candidates by crossing over parent genes. """
        childs = [ Candidate(), Candidate() ]
        childs[0].values , childs[1].values = np.copy(parent1.values), np.copy(parent2.values)

        r = 1.1
        while r > 1:
            r = random.uniform(0, 1.1)


        if (r < crossover_rate):
            crossover_point1 = 0
            crossover_point2 = 0
            while (crossover_point1 == crossover_point2):
                limit = 8
                crossover_point1 = random.randint(limit - 8 , limit)
                limit = limit + 1
                crossover_point2 = random.randint(limit - 8 , limit)

            if (crossover_point1 > crossover_point2):
                crossover_point1, crossover_point2 = crossover_point2, crossover_point1
                

            for i in range(crossover_point1, crossover_point2):
                childs[0].values[i], childs[1].values[i] = self.crossover_rows(childs[0].values[i], childs[1].values[i])

        return childs[0], childs[1]

    def crossover_rows(self, row1, row2):
        child_rows = [np.zeros(Nd), np.zeros(Nd)]

        remaining = range(1, Nd + 1)
        cycle = 0

        while ((0 in child_rows[0]) and (0 in child_rows[1])):  # While child rows not complete...
            index = self.find_unused(row1, remaining)
            start = row1[index]
            remaining.remove(start)
            child_rows[(cycle % 2)][index] , child_rows[1 - (cycle % 2)][index] = start, row2[index]
            next = child_rows[1 - (cycle % 2)][index]

            while (next != start):  
                index = self.find_value(row1, next)
                child_rows[1 - (cycle % 2)][index] = row2[index]
                remaining.remove(row1[index])
                child_rows[(cycle % 2)][index] = row1[index]
                next = row2[index]

            cycle += 1

        return child_rows[0], child_rows[1]

    def find_unused(self, parent_row, remaining):
        return [i for i in range(0, len(parent_row)) if parent_row[i] in remaining][0]

    def find_value(self, parent_row, value):
        return [i for i in range(0, len(parent_row)) if parent_row[i] == value][0]


class Sudoku(object):
    """ Solves a given Sudoku puzzle using a genetic algorithm. """

    def __init__(self):
        self.given = None
        return

    def load(self, p):
        self.given = Fixed(p)

    def solve(self):
        Nc, Ne, Ng, Nm = 1000, 50, 2000, 5

        phi , sigma, mutation_rate = 0, 1, 0.06
        if self.given.no_duplicates() == False:
            return (-1, 1)

        self.population = Population()
        print("create an initial population.")
        if self.population.seed(Nc, self.given) ==  1:
            pass
        else:
            return (-1, 1)

        stale = 0
        for generation in range(0, Ng):

            best_fitness = 0.0
            for c in range(0, Nc):
                fitness = self.population.candidates[c].fitness
                if (fitness == 1):
                    print("Solution found at generation %d!" % generation)
                    return (generation, self.population.candidates[c])

                best_fitness = max(best_fitness, fitness)
            print("Generation:", generation, " Best fitness:", best_fitness)

            next_population = []

            self.population.sort()
            elites = []
            for e in range(0, Ne):
                elite = Candidate()
                elite.values = np.copy(self.population.candidates[e].values)
                elites.append(elite)

            for count in range(Ne, Nc, 2):
                t = Tournament()
                parent1 = t.compete(self.population.candidates)
                parent2 = t.compete(self.population.candidates)

                cc = CycleCrossover()
                child1, child2 = cc.crossover(parent1, parent2, crossover_rate=1.0)

                childs = [child1, child2]
                for child in childs:
                    child.update_fitness()
                    old_fitness = child.fitness
                    success = child.mutate(mutation_rate, self.given)
                    child.update_fitness()
                    if (success):
                        Nm += 1
                        if (child.fitness > old_fitness):
                            phi = phi + 1
                
                next_population.append(childs[0])
                next_population.append(childs[1])

            next_population.extend(elites)

            self.population.candidates = next_population
            self.population.update_fitness()

            phi = (phi / Nm) if Nm != 0 else 0

            sigma = (sigma / 0.998) if (phi > 0.2) else (sigma * 0.998)

            mutation_rate = abs(np.random.normal(loc=0.0, scale=sigma, size=None))

            self.population.sort()
            stale = 0 if (self.population.candidates[0].fitness != self.population.candidates[1].fitness) else (stale + 1)

            if (stale >= 100):
                print("The population has gone stale. Re-seeding...")
                self.population.seed(Nc, self.given)
                stale = 0
                sigma = 1
                phi = 0
                mutation_rate = 0.06

        print("No solution found.")
        return (-2, 1)
    


class Start(object):
    """ Starts the program. """

    def __init__(self, csv_file):
        self.grid = np.loadtxt(csv_file, delimiter=",")
        self.grid = np.nan_to_num(self.grid)
        self.grid = self.grid.astype(int)
        self.grid = np.array(self.grid)
        self.sudoku = Sudoku()
        return

    def run(self):
        self.sudoku.load(self.grid)
        start_time = time.time()
        generation, solution = self.sudoku.solve()
        time_taken = time.time() - start_time
        if time_taken < 60:
            print("Time taken:", round(time_taken,4), "seconds")
        else:
            print("Time taken:", (time_taken)//60, "minutes " + str(round((time_taken)%60,4)) + " seconds")
        if(solution):
            if generation >= 0:
                self.grid_2 = solution.values
                str_print = ""
                for i in range(9):
                    for j in range(9):
                        str_print += str(self.grid_2[i][j]) + " "
                    str_print += "\n"
                print(str_print)
        return

s = Start(sys.argv[1])
s.run()
