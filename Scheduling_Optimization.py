import pandas as pd
import numpy as np
import random as rn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer
from scipy import interpolate

# Set general parameters
starting_population_size = 50
maximum_generation = 2
minimum_population_size = 30
maximum_population_size = 50
print_interval = 1

Start_Date = pd.to_datetime('October 17, 2018 5:00 PM', format='%B %d, %Y %I:%M %p')
Finish_Date = pd.to_datetime('October 5, 2020 5:00 PM', format='%B %d, %Y %I:%M %p')
max_project_duration = 780


def main():
    tasks = pd.read_excel('Optimizing-CMU.xlsm', sheet_name='Task_Table')
    costs = pd.read_excel('สรุป Cost BOQ ตาม Activity ID RV.1.xlsx', sheet_name='BOQ Activity')
    
    chromosome_length = len(tasks)
    #all shift day equal zero
    individual_0 = np.zeros((chromosome_length, 2), int)

    start = timer()
    for _ in range(4):
        tasks_0 = PDM_calculation(tasks, individual_0)
    
    cost_0 = calculate_cost_fitness(tasks_0, costs)
    time_0 = calculate_time_fitness(tasks_0)
    mx_0 = calculate_mx_fitness(tasks_0, costs)
    # time per individual
    tpi = timer()-start

    # No Optimization
    print('No Optimization')
    print('Total Cost', cost_0, 'Baht')
    print('Project Duration', time_0, 'Days')
    print('Mx', mx_0, 'man^2')
    print('> use' , tpi, 'sec/individual -> estimate time left', pd.to_timedelta(starting_population_size*maximum_generation*tpi, unit='s'))
    # return 0

    print('Start Optimization')
    TF_constraints = tasks_0['Late_Finish'] - tasks_0['Early_Finish'] - pd.to_timedelta(tasks_0['Duration'])

    # Create starting population
    population = create_population(starting_population_size, chromosome_length, TF_constraints)
    # print(population)

    # return 0
    mutation_probability = 1.0/chromosome_length
    # Loop through the generations of genetic algorithm
    for generation in range(maximum_generation):
        start = timer()

        # Breed
        population = breed_population(population)
        population = randomly_mutate_population(population, mutation_probability, TF_constraints)

        # Score population
        if generation % print_interval == 0:
            print('Generation (out of %i): gen %i' % (maximum_generation, generation + 1), end='', flush=True)
            scores = score_population(tasks_0, costs, population, True)
        else :
            scores = score_population(tasks_0, costs, population)

        # Build pareto front
        population = build_pareto_population(population, scores, minimum_population_size, maximum_population_size)
        # time per population
        tpp = timer()-start
        if generation % print_interval == 0:
            print('> use' , pd.to_timedelta(tpp, unit='s'), '/pop -> estimate time left', pd.to_timedelta((maximum_generation-generation)*tpp, unit='s'))
    
    # Get final pareto front
    scores = score_population(tasks_0, costs, population, True)
    population_ids = np.arange(population.shape[0]).astype(int)
    pareto_front = identify_pareto(scores, population_ids)
    population = population[pareto_front, :]
    scores = -scores[pareto_front]
    
    order = np.argsort(scores[:, 0])
    population = population[order]
    scores = scores[order]
    print(population)
    print(scores)

    # Plot Pareto front
    x = scores[:, 0]
    y = scores[:, 1]
    z = scores[:, 2]

    tck, u = interpolate.splprep([x,y,z], s=2)
    u_fine = np.linspace(0,1,200)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color='b', marker='o')
    ax.plot(x_fine, y_fine, z_fine, 'b')

    ax.set_xlabel('cost (Baht)')
    ax.set_ylabel('time (days)')
    ax.set_zlabel('Mx^2 (man^2)')
    plt.savefig('pareto.png')

    plt.show()


def create_population(individuals_size, chromosome_length, constraints):
    """
    Create random population with given number of individuals and chroosome length.
    """
    # Set up an initial array of all zeros
    population = np.zeros((individuals_size, chromosome_length, 2), int)

    # Loop through each row (individual)
    for i in range(individuals_size):
        # Loop through each task (chromosome)
        for j in range(chromosome_length):
            constraint = constraints[j].days
            # zero day for summary job
            if constraint < 0 :
                continue
            # random number of shift day
            # population[i, j, 0] = rn.randint(0, 5)
            # population[i, j, 1] = rn.randint(0, 1)
            population[i, j, 0] = round(rn.uniform(0, 5))
            population[i, j, 1] = round(rn.uniform(0, 1))

    return population

# Crossover
def breed_population(population):
    """
    Create child population by repetedly calling breeding function (two parents
    producing two children), applying genetic mutation to the child population,
    combining parent and child population, and removing duplicatee chromosomes.
    """
    # Create an empty list for new population
    new_population = []
    population_size = population.shape[0]
    # Create new population generating two children at a time
    for _ in range(int(population_size/2)):
        parent_1_loc = rn.randint(0, population_size-1)
        parent_2_loc = rn.randint(0, population_size-1)
        child_1, child_2 = breed_by_crossover(population, parent_1_loc, parent_2_loc)
        new_population.append(child_1)
        new_population.append(child_2)

    # Add the child population to the parent population
    # In this method we allow parents and children to compete to be kept
    population = np.vstack((population, np.array(new_population)))
    population = np.unique(population, axis=0)

    return population


def breed_by_crossover(population, parent_1_loc, parent_2_loc):
    """
    Combine two parent chromsomes by crossover to produce two children.
    """
    parent_1 = population[parent_1_loc]
    parent_2 = population[parent_2_loc]
    # Get length of chromosome
    # chromosome_length = len(parent_1)

    # # Pick crossover point, avoding ends of chromsome
    # crossover_point = rn.randint(1, chromosome_length-2)

    # # Create children. np.hstack joins two arrays
    # child_1 = np.vstack((parent_1[0:crossover_point],
    #                      parent_2[crossover_point:]))

    # child_2 = np.vstack((parent_2[0:crossover_point],
    #                      parent_1[crossover_point:]))

    center = (parent_1 + parent_2)/2
    diff = abs(parent_2-parent_1)/2
    child_1 = center - parent_1_loc/(parent_1_loc+parent_2_loc)*diff
    child_2 = center + parent_2_loc/(parent_1_loc+parent_2_loc)*diff
    
    # Return children
    return child_1, child_2

# Mutation
def randomly_mutate_population(population, mutation_probability, constraints):
    """
    Randomly mutate population with a given individual gene mutation probability.
    """
    # x=random.randint(100, size=(5))
    population_size = population.shape[0]
    chromosome_length = population.shape[1]
    # Apply random mutation through each row (individual)
    for i in range(int(population_size/2)):
        j = round(rn.uniform(0, chromosome_length-1))
        constraint = constraints[j].days
        if constraint < 0 :
                continue
        shiftday = population[i, j, 0]
        shiftday = shiftday + round(rn.uniform(-5, 5))
        if shiftday < 0 or shiftday > constraint:
            shiftday = round(rn.uniform(0, 5))
        population[i, j, 0] = shiftday
        population[i, j, 1] = round(rn.uniform(0, 1))

    # Return mutation population
    return population

# Fitness score
def score_population(tasks, costs, population, display=False):
    """
    Loop through all objectives and request score/fitness of population.
    """
    population_size = population.shape[0]
    scores = np.zeros((population_size, 3), int)
    show_interval = int(population_size/50) + 1

    for i in range(population_size):
        if display and i % show_interval == 0 :
            print('.', end='', flush=True)
        shift_tasks = PDM_calculation(tasks, population[i])
        scores[i, 0] = -calculate_cost_fitness(shift_tasks, costs)
        scores[i, 1] = -calculate_time_fitness(shift_tasks)
        scores[i, 2] = -calculate_mx_fitness(shift_tasks, costs)
    return scores


def calculate_cost_fitness(tasks, costs):
    """
    Calculate fitness scores in each solution.
    """
    T = max(tasks['Early_Finish'])

    MC = (costs['ค่าวัสดุต่อวัน\n(บาท/วัน)'][:-13]) * (pd.to_timedelta(tasks['Duration']).dt.days)
    LC = (costs['ค่าแรงงานต่อวัน\n(บาท/วัน)'][:-13]) * (pd.to_timedelta(tasks['Duration']).dt.days)
    DC = sum(MC) + sum(LC)
    
    Daily_indirect_cost = costs.at[256, 'ค่าวัสดุรวม\n(บาท)']
    IC = Daily_indirect_cost * (T-Start_Date).days
    
    Daily_penalty_cost = costs.at[258, 'ค่าวัสดุรวม\n(บาท)']
    if (T-Finish_Date).days > 0:
        PC = Daily_penalty_cost * (T-Finish_Date).days
    else:
        PC = 0
    if PC > 0.1*(DC + IC):
        return DC + IC
        
    Total_cost = int(DC + IC + PC)

    return Total_cost


def calculate_time_fitness(tasks):
    """
    Calculate fitness scores in each solution.
    """
    T = max(tasks['Early_Finish'])
    Project_duration = (T-Start_Date).days + 1
    if Project_duration > max_project_duration:
        Project_duration = max_project_duration*2

    return Project_duration


def calculate_mx_fitness(tasks, costs):
    """
    Calculate fitness scores in each solution.
    """
    Early_Start = tasks['Early_Start']
    Early_Finish = tasks['Early_Finish']
    T = max(Early_Finish)
    Project_duration = (T-Start_Date).days + 1
    labour_resource = costs['Resource\n(คน)'][:-13]

    Mx = 0
    for i in range(Project_duration):
        cur_day = Start_Date + pd.to_timedelta(i, unit='d')
        cur_job = labour_resource[(cur_day >= Early_Start) & (cur_day <= Early_Finish)]
        cur_job = cur_job[pd.notnull(cur_job)]
        Mx = Mx + sum(cur_job)**2
    return Mx

# Pareto front
def build_pareto_population(population, scores, minimum_population_size, maximum_population_size):
    """
    As necessary repeats Pareto front selection to build a population within
    defined size limits. Will reduce a Pareto front by applying crowding 
    selection as necessary.    
    """
    unselected_population_ids = np.arange(population.shape[0])
    all_population_ids = np.arange(population.shape[0])
    pareto_front = []
    while len(pareto_front) < minimum_population_size:
        temp_pareto_front = identify_pareto(
            scores[unselected_population_ids, :], unselected_population_ids)

        # Check size of total parteo front.
        # If larger than maximum size reduce new pareto front by crowding
        combined_pareto_size = len(pareto_front) + len(temp_pareto_front)
        if combined_pareto_size > maximum_population_size:
            number_to_select = combined_pareto_size - maximum_population_size
            selected_individuals = (reduce_by_crowding(
                scores[temp_pareto_front], number_to_select))
            temp_pareto_front = temp_pareto_front[selected_individuals]

        # Add latest pareto front to full Pareto front
        pareto_front = np.hstack((pareto_front, temp_pareto_front))

        # Update unselected population ID by using sets to find IDs in all
        # ids that are not in the selected front
        unselected_set = set(all_population_ids) - set(pareto_front)
        unselected_population_ids = np.array(list(unselected_set))

    population = population[pareto_front.astype(int)]
    return population


def identify_pareto(scores, population_ids):
    """
    Identifies a single Pareto front, and returns the population IDs of
    the selected solutions.
    """
    population_size = scores.shape[0]
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]


def reduce_by_crowding(scores, number_to_select):
    """
    This function selects a number of solutions based on tournament of
    crowding distances. Two members of the population are picked at
    random. The one with the higher croding dostance is always picked
    """
    population_ids = np.arange(scores.shape[0])

    crowding_distances = calculate_crowding(scores)

    picked_population_ids = np.zeros((number_to_select))

    picked_scores = np.zeros((number_to_select, len(scores[0, :])))

    for i in range(number_to_select):

        population_size = population_ids.shape[0]

        fighter1ID = rn.randint(0, population_size - 1)

        fighter2ID = rn.randint(0, population_size - 1)

        # If fighter # 1 is better
        if crowding_distances[fighter1ID] >= crowding_distances[fighter2ID]:

            # add solution to picked solutions array
            picked_population_ids[i] = population_ids[fighter1ID]

            # Add score to picked scores array
            picked_scores[i, :] = scores[fighter1ID, :]

            # remove selected solution from available solutions
            population_ids = np.delete(population_ids, (fighter1ID), axis=0)

            scores = np.delete(scores, (fighter1ID), axis=0)

            crowding_distances = np.delete(crowding_distances, (fighter1ID), axis=0)
        else:
            picked_population_ids[i] = population_ids[fighter2ID]

            picked_scores[i, :] = scores[fighter2ID, :]

            population_ids = np.delete(population_ids, (fighter2ID), axis=0)

            scores = np.delete(scores, (fighter2ID), axis=0)

            crowding_distances = np.delete(
                crowding_distances, (fighter2ID), axis=0)

    # Convert to integer
    picked_population_ids = np.asarray(picked_population_ids, dtype=int)

    return (picked_population_ids)


def calculate_crowding(scores):
    """
    Crowding is based on a vector for each individual
    All scores are normalised between low and high. For any one score, all
    solutions are sorted in order low to high. Crowding for chromsome x
    for that score is the difference between the next highest and next
    lowest score. Total crowding value sums all crowding for all scores
    """

    population_size = len(scores[:, 0])
    number_of_scores = len(scores[0, :])

    # create crowding matrix of population (row) and score (column)
    crowding_matrix = np.zeros((population_size, number_of_scores))

    # normalise scores (ptp is max-min)
    normed_scores = (scores - scores.min(0)) / 1+scores.ptp(0)

    # calculate crowding distance for each score in turn
    for col in range(number_of_scores):
        crowding = np.zeros(population_size)

        # end points have maximum crowding
        crowding[0] = 1
        crowding[population_size - 1] = 1

        # Sort each score (to calculate crowding between adjacent scores)
        sorted_scores = np.sort(normed_scores[:, col])

        sorted_scores_index = np.argsort(normed_scores[:, col])

        # Calculate crowding distance for each individual
        crowding[1:population_size - 1] = (sorted_scores[2:population_size] - sorted_scores[0:population_size - 2])

        # resort to orginal order (two steps)
        re_sort_order = np.argsort(sorted_scores_index)
        sorted_crowding = crowding[re_sort_order]

        # Record crowding distances
        crowding_matrix[:, col] = sorted_crowding

    # Sum crowding distances of each score
    crowding_distances = np.sum(crowding_matrix, axis=1)

    return crowding_distances

# PDM network
def PDM_calculation(tasks, individual):
    # for _ in range(4):
    # Forward calculate (Early_Start, Early_Finish)
    tasks_length = tasks.shape[0]
    for i in range(tasks_length):
        duration = tasks.at[i, 'Duration']
        shiftday = individual[i][0]
        option = individual[i][1]

        if option == 0:
            predecessors = tasks.at[i, 'Predecessors']
        else:
            predecessors = tasks.at[i, 'Predecessors2']

        # No relationship
        if pd.isnull(predecessors):
            tasks.at[i, 'Early_Start'], tasks.at[i, 'Early_Finish'] = NO_calculation(Di=duration, Si=shiftday, forward=True)
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
                        early_set.append(FS_calculation(EFh=tasks.at[h, 'Early_Finish'], Di=duration, Si=shiftday, lag=lag_time, forward=True))

                # FF relationship
                elif FF_loc != -1:
                    h_set = str(predecessor[:FF_loc]).split(',')
                    h_set = list(map(int, h_set))
                    for h in h_set:
                        h = h - 1
                        early_set.append(FF_calculation(EFh=tasks.at[h, 'Early_Finish'], Di=duration, Si=shiftday, lag=lag_time, forward=True))

                # SF relationship
                elif SF_loc != -1:
                    h_set = str(predecessor[:SF_loc]).split(',')
                    h_set = list(map(int, h_set))
                    for h in h_set:
                        h = h - 1
                        early_set.append(SF_calculation(ESh=tasks.at[h, 'Early_Start'], Di=duration, Si=shiftday, lag=lag_time, forward=True))

                # SS relationship
                elif SS_loc != -1:
                    h_set = str(predecessor[:SS_loc]).split(',')
                    h_set = list(map(int, h_set))
                    for h in h_set:
                        h = h - 1
                        early_set.append(SS_calculation(ESh=tasks.at[h, 'Early_Start'], Di=duration, Si=shiftday, lag=lag_time, forward=True))
                
                # FS relationship
                else:
                    h_set = str(predecessor).split(',')
                    h_set = list(map(int, h_set))
                    for h in h_set:
                        h = h - 1
                        early_set.append(FS_calculation(EFh=tasks.at[h, 'Early_Finish'], Di=duration, Si=shiftday, lag=lag_time, forward=True))
                
                early_set = np.array(early_set).T.tolist()
                max_es = max(early_set[0])
                max_ef = max(early_set[1])
                # h_es = h_set[early_set[0].index(max_es)]
                # h_ef = h_set[early_set[1].index(max_ef)]
                tasks.at[i, 'Early_Start'] = max_es
                tasks.at[i, 'Early_Finish'] = max_ef
    
    # Backward calculate (Late_Start, Late_Finish)
    for i in range(tasks_length-1, -1, -1):
        successors = tasks.at[i, 'Successors']
        duration = tasks.at[i, 'Duration']

        # No relationship
        if pd.isnull(successors):
            tasks.at[i, 'Late_Start'], tasks.at[i, 'Late_Finish'] = NO_calculation(EFh=max(tasks['Early_Finish']), Di=duration, forward=False)
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
                        late_set.append(FS_calculation(LSj=tasks.at[j, 'Late_Start'], Di=duration, lag=lag_time, forward=False))

                # FF relationship
                elif FF_loc != -1:
                    j_set = str(successor[:FF_loc]).split(',')
                    j_set = list(map(int, j_set))
                    for j in j_set:
                        j = j - 1
                        late_set.append(FF_calculation(LFj=tasks.at[j, 'Late_Finish'], Di=duration, lag=lag_time, forward=False))

                # SF relationship
                elif SF_loc != -1:
                    j_set = str(successor[:SF_loc]).split(',')
                    j_set = list(map(int, j_set))
                    for j in j_set:
                        j = j - 1
                        late_set.append(SF_calculation(LFj=tasks.at[j, 'Late_Finish'], Di=duration, lag=lag_time, forward=False))

                # SS relationship
                elif SS_loc != -1:
                    j_set = str(successor[:SS_loc]).split(',')
                    j_set = list(map(int, j_set))
                    for j in j_set:
                        j = j - 1
                        late_set.append(SS_calculation(LSj=tasks.at[j, 'Late_Start'], Di=duration, lag=lag_time, forward=False))
                
                # FS relationship
                else:
                    j_set = str(successor).split(',')
                    j_set = list(map(int, j_set))
                    for j in j_set:
                        j = j - 1
                        late_set.append(FS_calculation(LSj=tasks.at[j, 'Late_Start'], Di=duration, lag=lag_time, forward=False))
                
                late_set = np.array(late_set).T.tolist()
                min_ls = min(late_set[0])
                min_lf = min(late_set[1])
                # h_ls = j_set[late_set[0].index(min_ls)]
                # h_lf = j_set[late_set[1].index(min_lf)]
                tasks.at[i, 'Late_Start'] = min_ls
                tasks.at[i, 'Late_Finish'] = min_lf

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
