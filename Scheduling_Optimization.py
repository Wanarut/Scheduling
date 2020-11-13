import pandas as pd
import numpy as np
import random as rn

# Set general parameters
starting_population_size = 10
maximum_generation = 20
minimum_population_size = 10
maximum_population_size = 10
number_of_objectives = 3

Start_Date = pd.to_datetime('October 17, 2018 5:00 PM', format='%B %d, %Y %I:%M %p')
Finish_Date = pd.to_datetime('October 5, 2020 5:00 PM', format='%B %d, %Y %I:%M %p')


def main():
    tasks = pd.read_excel('Optimizing-CMU.xlsm', sheet_name='Task_Table')

    tasks = PDM_calculation(tasks=tasks)
    print(tasks)
    # Create starting population
    chromosome_length = len(tasks)
    mutation_probability = 1.0/chromosome_length
    population = create_population(starting_population_size, chromosome_length, tasks['Total_Float'])
    print(population)

    # Loop through the generations of genetic algorithm

    for generation in range(maximum_generation):
        if generation % 10 == 0:
            print('Generation (out of %i): %i ' % (maximum_generation, generation))

        # Breed
        population = breed_population(population)
        population = randomly_mutate_population(population, mutation_probability, tasks['Total_Float'])

        tasks = PDM_calculation(tasks=tasks, population=population)

        # Score population
        scores = score_population(tasks, population)

        # # Build pareto front
        # population = build_pareto_population(
        #     population, scores, minimum_population_size, maximum_population_size)


def create_population(individuals_size, chromosome_length, constraints):
    """
    Create random population with given number of individuals and chroosome length.
    """
    # Set up an initial array of all zeros
    population = np.zeros((individuals_size, chromosome_length))
    # Loop through each row (individual)
    for i in range(individuals_size):
        # Loop through each task (chromosome)
        for j in range(chromosome_length):
            # zero day for summary job
            if constraints[j].days < 0 :
                continue
            # random number of shift day
            shiftday = rn.randint(0, constraints[j].days)
            population[i, j] = shiftday

    return population


def breed_population(population):
    """
    Create child population by repetedly calling breeding function (two parents
    producing two children), applying genetic mutation to the child population,
    combining parent and child population, and removing duplicatee chromosomes.
    """
    # Create an empty list for new population
    new_population = []
    population_size = population.shape[0]
    # Create new popualtion generating two children at a time
    for _ in range(int(population_size/2)):
        parent_1 = population[rn.randint(0, population_size-1)]
        parent_2 = population[rn.randint(0, population_size-1)]
        child_1, child_2 = breed_by_crossover(parent_1, parent_2)
        new_population.append(child_1)
        new_population.append(child_2)

    # Add the child population to the parent population
    # In this method we allow parents and children to compete to be kept
    population = np.vstack((np.array(new_population), population))
    population = np.unique(population, axis=0)

    return population


def breed_by_crossover(parent_1, parent_2):
    """
    Combine two parent chromsomes by crossover to produce two children.
    """
    # Get length of chromosome
    chromosome_length = len(parent_1)

    # Pick crossover point, avoding ends of chromsome
    crossover_point = rn.randint(1, chromosome_length-1)

    # Create children. np.hstack joins two arrays
    child_1 = np.hstack((parent_1[0:crossover_point],
                         parent_2[crossover_point:]))

    child_2 = np.hstack((parent_2[0:crossover_point],
                         parent_1[crossover_point:]))

    # Return children
    return child_1, child_2


def randomly_mutate_population(population, mutation_probability, constraints):
    """
    Randomly mutate population with a given individual gene mutation probability.
    """
    population_size = population.shape[0]
    # Apply random mutation through each row (individual)
    for i in range(int(population_size/2)):
        # Loop through each task (chromosome)
        for j in range(len(population[i])):
            # zero day for summary job
            if constraints[j].days < 0 :
                continue
            # check constraint before mutate
            if population[i, j] > constraints[j].days:
                shiftday = rn.randint(0, constraints[j].days)
                population[i, j] = shiftday
            # mutation prob
            if rn.uniform(0, 1) <= mutation_probability:
                # random number of shift day
                shiftday = rn.randint(0, constraints[j].days)
                population[i, j] = shiftday

    # Return mutation population
    return population


def score_population(tasks, population):
    """
    Loop through all objectives and request score/fitness of population.
    """
    scores = np.zeros((population.shape[0], number_of_objectives))
    # for i in range(number_of_objectives):
    #     scores[:, i] = calculate_fitness(tasks, population)
    scores[:, i] = calculate_fitness(tasks, population)

    return scores


def calculate_fitness(tasks, population):
    """
    Calculate fitness scores in each solution.
    """
    # Create an array of True/False compared to reference
    identical_to_reference = population == reference
    # Sum number of genes that are identical to the reference
    fitness_scores = identical_to_reference.sum(axis=1)

    return fitness_scores


def PDM_calculation(tasks, population=None):
    for _ in range(4):
        # Forward calculate (Early_Start, Early_Finish)
        for i in range(len(tasks)):
            predecessors = tasks.at[i, 'Predecessors']

            # No relationship
            if pd.isnull(predecessors):
                tasks.at[i, 'Early_Start'], tasks.at[i, 'Early_Finish'] = NO_calculation(Di=tasks.at[i, 'Duration'], forward=True)

                # print('i', i + 1, '\thes', '-1', '\t', tasks.at[i, 'Early_Start'], '\thef', '-1', '\t', tasks.at[i, 'Early_Finish'])
            else:
                predecessors = str(predecessors).split(',')
                for predecessor in predecessors:
                    lag_loc = max(str(predecessor).find('+'), str(predecessor).find('-'))
                    if lag_loc != -1:
                        lag_time = str(predecessor)[lag_loc:]
                    else :
                        lag_time = '+0 days'
                    
                    FS_loc = str(predecessor).find('FS')
                    FF_loc = str(predecessor).find('FF')
                    SF_loc = str(predecessor).find('SF')
                    SS_loc = str(predecessor).find('SS')
                    
                    early_set = []
                    # FS relationship
                    if FS_loc != -1:
                        h_set = str(predecessor[:FS_loc]).split(',')
                        h_set = list(map(int, h_set))
                        for h in h_set:
                            h = h - 1
                            early_set.append(FS_calculation(EFh=tasks.at[h, 'Early_Finish'], Di=tasks.at[i, 'Duration'], lag=lag_time, forward=True))

                    # FF relationship
                    elif FF_loc != -1:
                        h_set = str(predecessor[:FF_loc]).split(',')
                        h_set = list(map(int, h_set))
                        for h in h_set:
                            h = h - 1
                            early_set.append(FF_calculation(EFh=tasks.at[h, 'Early_Finish'], Di=tasks.at[i, 'Duration'], lag=lag_time, forward=True))

                    # SF relationship
                    elif SF_loc != -1:
                        h_set = str(predecessor[:SF_loc]).split(',')
                        h_set = list(map(int, h_set))
                        for h in h_set:
                            h = h - 1
                            early_set.append(SF_calculation(ESh=tasks.at[h, 'Early_Start'], Di=tasks.at[i, 'Duration'], lag=lag_time, forward=True))

                    # SS relationship
                    elif SS_loc != -1:
                        h_set = str(predecessor[:SS_loc]).split(',')
                        h_set = list(map(int, h_set))
                        for h in h_set:
                            h = h - 1
                            early_set.append(SS_calculation(ESh=tasks.at[h, 'Early_Start'], Di=tasks.at[i, 'Duration'], lag=lag_time, forward=True))
                    
                    # FS relationship
                    else:
                        h_set = str(predecessor).split(',')
                        h_set = list(map(int, h_set))
                        for h in h_set:
                            h = h - 1
                            early_set.append(FS_calculation(EFh=tasks.at[h, 'Early_Finish'], Di=tasks.at[i, 'Duration'], lag=lag_time, forward=True))
                    
                    early_set = np.array(early_set).T.tolist()
                    max_es = max(early_set[0])
                    max_ef = max(early_set[1])
                    h_es = h_set[early_set[0].index(max_es)]
                    h_ef = h_set[early_set[1].index(max_ef)]
                    tasks.at[i, 'Early_Start'] = max_es
                    tasks.at[i, 'Early_Finish'] = max_ef

                    # print('i', i + 1, '\thes', h_es, '\t', tasks.at[i, 'Early_Start'], '\thef', h_ef, '\t', tasks.at[i, 'Early_Finish'])
    
    # return 0
    # for _ in range(2):
        
        # Backward calculate (Late_Start, Late_Finish)
        for i in range(len(tasks)-1, -1, -1):
            successors = tasks.at[i, 'Successors']

            # No relationship
            if pd.isnull(successors):
                tasks.at[i, 'Late_Start'], tasks.at[i, 'Late_Finish'] = NO_calculation(EFh=max(tasks['Early_Finish']), Di=tasks.at[i, 'Duration'], forward=False)

                # print('i', i + 1, '\tjls', '-1', '\t', tasks.at[i, 'Late_Start'], '\tjlf', '-1', '\t', tasks.at[i, 'Late_Finish'])
            else:
                successors = str(successors).split(',')
                for successor in successors:
                    lag_loc = max(str(successor).find('+'), str(successor).find('-'))
                    if lag_loc != -1:
                        lag_time = str(successor)[lag_loc:]
                    else :
                        lag_time = '+0 days'
                    
                    FS_loc = str(successor).find('FS')
                    FF_loc = str(successor).find('FF')
                    SF_loc = str(successor).find('SF')
                    SS_loc = str(successor).find('SS')
                    
                    late_set = []
                    # FS relationship
                    if FS_loc != -1:
                        j_set = str(successor[:FS_loc]).split(',')
                        j_set = list(map(int, j_set))
                        for j in j_set:
                            j = j - 1
                            late_set.append(FS_calculation(LSj=tasks.at[j, 'Late_Start'], Di=tasks.at[i, 'Duration'], lag=lag_time, forward=False))

                    # FF relationship
                    elif FF_loc != -1:
                        j_set = str(successor[:FF_loc]).split(',')
                        j_set = list(map(int, j_set))
                        for j in j_set:
                            j = j - 1
                            late_set.append(FF_calculation(LFj=tasks.at[j, 'Late_Finish'], Di=tasks.at[i, 'Duration'], lag=lag_time, forward=False))

                    # SF relationship
                    elif SF_loc != -1:
                        j_set = str(successor[:SF_loc]).split(',')
                        j_set = list(map(int, j_set))
                        for j in j_set:
                            j = j - 1
                            late_set.append(SF_calculation(LFj=tasks.at[j, 'Late_Finish'], Di=tasks.at[i, 'Duration'], lag=lag_time, forward=False))

                    # SS relationship
                    elif SS_loc != -1:
                        j_set = str(successor[:SS_loc]).split(',')
                        j_set = list(map(int, j_set))
                        for j in j_set:
                            j = j - 1
                            late_set.append(SS_calculation(LSj=tasks.at[j, 'Late_Start'], Di=tasks.at[i, 'Duration'], lag=lag_time, forward=False))
                    
                    # FS relationship
                    else:
                        j_set = str(successor).split(',')
                        j_set = list(map(int, j_set))
                        for j in j_set:
                            j = j - 1
                            late_set.append(FS_calculation(LSj=tasks.at[j, 'Late_Start'], Di=tasks.at[i, 'Duration'], lag=lag_time, forward=False))
                    
                    late_set = np.array(late_set).T.tolist()
                    min_ls = min(late_set[0])
                    min_lf = min(late_set[1])
                    h_ls = j_set[late_set[0].index(min_ls)]
                    h_lf = j_set[late_set[1].index(min_lf)]
                    tasks.at[i, 'Late_Start'] = min_ls
                    tasks.at[i, 'Late_Finish'] = min_lf

                    # print('i', i + 1, '\tjls', h_ls, '\t', tasks.at[i, 'Late_Start'], '\tjlf', h_lf, '\t', tasks.at[i, 'Late_Finish'])
        
        # print(tasks[['Early_Start', 'Early_Finish', 'Late_Start', 'Late_Finish']])
        # print('###########################################################################################')
    # Filter
    # tasks = tasks[tasks['Summary']=='No']
    # print(tasks)
    # tasks = tasks.reset_index()
    # print(tasks)
    #TF Constraint
    # print(tasks[['Early_Start', 'Early_Finish', 'Late_Start', 'Late_Finish']])
    tasks['Total_Float'] = tasks['Late_Finish'] - tasks['Early_Finish'] - pd.to_timedelta(tasks['Duration'])
    # for i in range(len(tasks)):
    #     print(i, '\t', tasks.at[i, 'Early_Finish'], '\t', tasks.at[i, 'Late_Finish'], '\t', tasks.at[i, 'Duration'], '\t', tasks.at[i, 'Total_Float'])
    # print(tasks[['Early_Start', 'Early_Finish', 'Late_Start', 'Late_Finish', 'Total_Float']])
    return tasks

    
def NO_calculation( ESh=None, EFh=None, LSj=None, LFj=None,
                    Si=0, Di=None, lag=None, exc=0, forward=None):
    if forward :
        ESi = Start_Date + pd.to_timedelta(Si + exc, unit='d')
        EFi = ESi + pd.to_timedelta(Di) + pd.to_timedelta(-1 + exc, unit='d')
        return ESi, EFi
    else :
        if pd.notnull(Finish_Date) :
            LFi = Finish_Date
        else :
            LFi = EFh
        LSi = LFi - pd.to_timedelta(Di) + pd.to_timedelta(1 - exc, unit='d')
        return LSi, LFi


def FS_calculation( ESh=None, EFh=None, LSj=None, LFj=None,
                    Si=0, Di=None, lag=None, exc=0, forward=None):
    if forward :
        ESi = EFh + pd.to_timedelta(Si, unit='d') + pd.to_timedelta(lag) + pd.to_timedelta(1 + exc, unit='d')
        EFi = ESi + pd.to_timedelta(Di) + pd.to_timedelta(exc - 1, unit='d')
        return ESi, EFi
    else :
        LFi = LSj - pd.to_timedelta(lag) + pd.to_timedelta(- 1 - exc, unit='d')
        LSi = LFi - pd.to_timedelta(Di) + pd.to_timedelta(1 - exc, unit='d')
        return LSi, LFi


def FF_calculation( ESh=None, EFh=None, LSj=None, LFj=None,
                    Si=0, Di=None, lag=None, exc=0, forward=None):
    if forward :
        EFi = EFh + pd.to_timedelta(Si, unit='d') + pd.to_timedelta(lag)
        ESi = EFi - pd.to_timedelta(Di) + pd.to_timedelta(1 - exc, unit='d')
        return ESi, EFi
    else :
        LFi = LFj - pd.to_timedelta(lag)
        LSi = LFi - pd.to_timedelta(Di) + pd.to_timedelta(1 - exc, unit='d')
        return LSi, LFi


def SF_calculation( ESh=None, EFh=None, LSj=None, LFj=None,
                    Si=0, Di=None, lag=None, exc=0, forward=None):
    if forward :
        EFi = ESh + pd.to_timedelta(Si, unit='d') + pd.to_timedelta(lag) + pd.to_timedelta(-1 - exc, unit='d')
        ESi = EFi - pd.to_timedelta(Di) + pd.to_timedelta(1 - exc, unit='d')
        return ESi, EFi
    else :
        LSi = LFj - pd.to_timedelta(lag) + pd.to_timedelta(1 + exc, unit='d')
        LFi = LSi + pd.to_timedelta(Di) + pd.to_timedelta(-1 + exc, unit='d')
        return LSi, LFi


def SS_calculation( ESh=None, EFh=None, LSj=None, LFj=None,
                    Si=0, Di=None, lag=None, exc=0, forward=None):
    if forward :
        ESi = ESh + pd.to_timedelta(Si, unit='d') + pd.to_timedelta(lag)
        EFi = ESi + pd.to_timedelta(Di) + pd.to_timedelta(-1 + exc, unit='d')
        return ESi, EFi
    else :
        LSi = LSj - pd.to_timedelta(lag)
        LFi = LSi + pd.to_timedelta(Di) + pd.to_timedelta(-1 + exc, unit='d')
        return LSi, LFi


main()
