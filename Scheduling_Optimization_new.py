import pandas as pd
import numpy as np
import random as rn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer
from scipy import interpolate

# Set general parameters
starting_population_size = 500
maximum_generation = 100
minimum_population_size = 400
maximum_population_size = 500
print_interval = 10
fitness_based = False

Start_Date = pd.to_datetime('October 17, 2018 5:00 PM', format='%B %d, %Y %I:%M %p')
Finish_Date = pd.to_datetime('October 5, 2020 5:00 PM', format='%B %d, %Y %I:%M %p')
Start_Days = (Start_Date-Start_Date).days
Finish_Days = (Finish_Date-Start_Date).days
max_project_duration = 780

def main():
    tasks = pd.read_excel('Optimizing-CMU.xlsm', sheet_name='Task_Table')
    costs = pd.read_excel('สรุป Cost BOQ ตาม Activity ID RV.1.xlsx', sheet_name='BOQ Activity')
    # Filter reduce data load
    tasks = tasks[['Duration','Predecessors','Predecessors2','Successors','Early_Start','Early_Finish','Late_Start','Late_Finish']]
    costs = costs[['Resource\n(คน)','ค่าวัสดุรวม\n(บาท)','ค่าวัสดุต่อวัน\n(บาท/วัน)','ค่าแรงงานต่อวัน\n(บาท/วัน)']]
    tasks['Duration'] = pd.to_timedelta(tasks['Duration']).dt.days
    
    start = timer()
    chromosome_length = len(tasks)
    #all shift day equal zero
    individual_0 = np.zeros((chromosome_length, 2), int)

    tasks_0 = tasks.copy()
    PDM_calculation(tasks_0, individual_0)
    # print(tasks)
    # print(tasks_0)
    # return 0

    cost_0 = calculate_cost_fitness(tasks_0, costs)
    time_0 = calculate_time_fitness(tasks_0)
    mx_0 = calculate_mx_fitness(tasks_0, costs)
    
    # Create starting population
    TF_constraints = tasks_0['Late_Finish'] - tasks_0['Early_Finish'] - tasks_0['Duration']
    population = create_population(starting_population_size, chromosome_length, TF_constraints)
    # print(population)

    # time per individual
    tpp = timer()-start

    # No Optimization
    print('No Optimization')
    print('Total Cost', cost_0, 'Baht')
    print('Project Duration', time_0, 'Days')
    print('Mx', mx_0, 'man^2')
    print('> use' , tpp, 'sec/sol -> estimated time left', pd.to_timedelta(maximum_population_size*maximum_generation*tpp, unit='s'))
    # return 0

    print('Start Optimization')
    mutation_probability = 1.0/chromosome_length
    # Loop through the generations of genetic algorithm
    with pd.ExcelWriter('scores_log.xlsx') as writer:
        for generation in range(maximum_generation):
            start = timer()

            # Breed
            population = breed_population(population)
            population = randomly_mutate_population(population, mutation_probability, TF_constraints)

            # Score population
            if generation % print_interval == 0:
                print('Generation (out of %i): gen %i' % (maximum_generation, generation + 1), end='', flush=True)
                scores = score_population(tasks, costs, population, True)
            else :
                scores = score_population(tasks, costs, population)

            # Build pareto front
            population, scores = build_pareto_population(population, scores, minimum_population_size, maximum_population_size)
            # order = np.argsort(scores[:, 2])
            # population = population[order]
            # scores = scores[order]

            if generation % print_interval == 0:
                # Save
                scores_df = pd.DataFrame(-scores)
                scores_df.to_excel(writer, sheet_name='gen_' + str(generation+1), index=False, header=False)

            # time per population
            tpg = timer()-start
            if generation % print_interval == 0:
                print('> use' , pd.to_timedelta(tpg, unit='s'), '/gen -> estimated time left', pd.to_timedelta((maximum_generation-generation)*tpg, unit='s'))
    
    # Get final pareto front
    # print('Final Pareto Generation', end='', flush=True)
    # scores = score_population(tasks, costs, population, True)

    population_ids = np.arange(population.shape[0]).astype(int)
    pareto_front = identify_pareto(scores, population_ids)
    population = population[pareto_front, :]
    scores = scores[pareto_front]
    
    order = np.argsort(scores[:, 2])
    population = population[order]
    scores = -scores[order]

    for score in scores:
        print(score)
        
    np.savetxt('shiftdays.csv', population[:,:,0].T, delimiter=',', fmt='% 4d')
    np.savetxt('options.csv', population[:,:,1].T, delimiter=',', fmt='% 4d')
    np.savetxt('scores.csv', scores, delimiter=',', fmt='% 8d')

    # Plot Pareto front
    scores = np.unique(scores, axis=0)
    x = scores[:,0]
    y = scores[:,1]
    z = scores[:,2]

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
            constraint = constraints[j]
            # zero day for summary job
            if constraint < 0 :
                continue
            # random number of shift day
            # population[i, j, 0] = rn.randint(0, 5)
            # population[i, j, 1] = rn.randint(0, 1)
            population[i, j, 0] = round(rn.uniform(0, 10))
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
        if fitness_based:
            child_1, child_2 = breed_by_fitnessbased_crossover(population, parent_1_loc, parent_2_loc)
        else:
            child_1, child_2 = breed_by_traditional_crossover(population, parent_1_loc, parent_2_loc)
        new_population.append(child_1)
        new_population.append(child_2)

    # Add the child population to the parent population
    # In this method we allow parents and children to compete to be kept
    population = np.vstack((population, np.array(new_population)))
    population = np.unique(population, axis=0)

    return population


def breed_by_traditional_crossover(population, parent_1_loc, parent_2_loc):
    """
    Combine two parent chromsomes by crossover to produce two children.
    """
    parent_1 = population[parent_1_loc]
    parent_2 = population[parent_2_loc]
    
    # Traditional crossover
    # Get length of chromosome
    chromosome_length = len(parent_1)

    # Pick crossover point, avoding ends of chromsome
    crossover_point = rn.randint(1, chromosome_length-2)

    # Create children. np.vstack joins two arrays
    child_1 = np.vstack((parent_1[0:crossover_point], parent_2[crossover_point:]))
    child_2 = np.vstack((parent_2[0:crossover_point], parent_1[crossover_point:]))
    
    # Return children
    return child_1, child_2


def breed_by_fitnessbased_crossover(population, fitness_1, fitness_2):
    """
    Combine two parent chromsomes by crossover to produce two children.
    """
    parent_1 = population[fitness_1]
    parent_2 = population[fitness_2]

    #Fitness based crossover
    center = (parent_1 + parent_2)/2
    diff = abs(parent_2 - parent_1)/2
    child_1 = center - ((fitness_1+1)/(fitness_1+fitness_2+1))*diff
    child_2 = center + ((fitness_2+1)/(fitness_1+fitness_2+1))*diff
    child_1 = np.rint(child_1)
    child_2 = np.rint(child_2)
    
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
        constraint = constraints[j]
        if constraint < 0 :
                continue
        shiftday = population[i, j, 0]
        shiftday = shiftday + round(rn.uniform(-10, 10))
        if shiftday < 0 or shiftday > constraint:
            shiftday = round(rn.uniform(0, 10))
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
            print('-', end='', flush=True)
        shift_tasks = tasks.copy()
        PDM_calculation(shift_tasks, population[i])
        scores[i, 0] = -calculate_cost_fitness(shift_tasks, costs)
        scores[i, 1] = -calculate_time_fitness(shift_tasks)
        scores[i, 2] = -calculate_mx_fitness(shift_tasks, costs)
    return scores


def calculate_cost_fitness(tasks, costs):
    """
    Calculate fitness scores in each solution.
    """
    T = max(tasks['Early_Finish'])

    MC = costs['ค่าวัสดุต่อวัน\n(บาท/วัน)'][:-13] * tasks['Duration']
    LC = costs['ค่าแรงงานต่อวัน\n(บาท/วัน)'][:-13] * tasks['Duration']
    DC = sum(MC) + sum(LC)
    
    Daily_indirect_cost = costs.at[256, 'ค่าวัสดุรวม\n(บาท)']
    IC = Daily_indirect_cost * (T-Start_Days)
    
    Daily_penalty_cost = costs.at[258, 'ค่าวัสดุรวม\n(บาท)']
    if (T-Finish_Days) > 0:
        PC = Daily_penalty_cost * (T-Finish_Days)
    else:
        PC = 0
    if PC > 0.1*(DC + IC):
        PC = 0.1*(DC + IC)
        
    Total_cost = DC + IC + PC

    return Total_cost


def calculate_time_fitness(tasks):
    """
    Calculate fitness scores in each solution.
    """
    T = max(tasks['Early_Finish'])
    Project_duration = T-Start_Days + 1
    if Project_duration > max_project_duration:
        Project_duration = max_project_duration

    return Project_duration


def calculate_mx_fitness(tasks, costs):
    """
    Calculate fitness scores in each solution.
    """
    Early_Start = tasks['Early_Start']
    Early_Finish = tasks['Early_Finish']
    T = max(Early_Finish)
    Project_duration = int(T - Start_Days + 1)
    labour_resource = costs['Resource\n(คน)'][:-13]

    Mx = 0
    for i in range(Project_duration):
        cur_day = Start_Days + i
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

    index = pareto_front.astype(int)
    population = population[index]
    scores = scores[index]
    return population, scores


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
def PDM_Forward(i, tasks, individual):
    # Calculated
    if pd.notnull(tasks.at[i, 'Early_Start']):
        return
    
    shiftday = individual[i, 0]
    option = individual[i, 1]
    
    # Forward calculate (Early_Start, Early_Finish)
    if option == 0:
        predecessors = tasks.at[i, 'Predecessors']
    else:
        predecessors = tasks.at[i, 'Predecessors2']
        if pd.isnull(predecessors):
            predecessors = tasks.at[i, 'Predecessors']
            individual[i, 1] = 0
    duration = tasks.at[i, 'Duration']

    # No relationship
    if pd.isnull(predecessors):
        tasks.at[i, 'Early_Start'], tasks.at[i, 'Early_Finish'] = NO_calculation(Di=duration, Si=shiftday, forward=True)
    else:
        early_set = []
        predecessors = str(predecessors).split(',')
        for predecessor in predecessors:
            lag_loc = max(predecessor.find('+'), predecessor.find('-'))
            if lag_loc != -1:
                lag_time = predecessor[lag_loc:]
                lag_time = pd.to_timedelta(lag_time).days
            else :
                lag_time = 0
            
            FS_loc = predecessor.find('FS')
            FF_loc = predecessor.find('FF')
            SF_loc = predecessor.find('SF')
            SS_loc = predecessor.find('SS')
            
            # FS relationship
            if FS_loc != -1:
                h = int(predecessor[:FS_loc])
                h = h - 1
                PDM_Forward(h, tasks, individual)
                early_set.append(FS_calculation(EFh=tasks.at[h, 'Early_Finish'], Di=duration, Si=shiftday, lag=lag_time, forward=True))

            # FF relationship
            elif FF_loc != -1:
                h = int(predecessor[:FF_loc])
                h = h - 1
                PDM_Forward(h, tasks, individual)
                early_set.append(FF_calculation(EFh=tasks.at[h, 'Early_Finish'], Di=duration, Si=shiftday, lag=lag_time, forward=True))

            # SF relationship
            elif SF_loc != -1:
                h = int(predecessor[:SF_loc])
                h = h - 1
                PDM_Forward(h, tasks, individual)
                early_set.append(SF_calculation(ESh=tasks.at[h, 'Early_Start'], Di=duration, Si=shiftday, lag=lag_time, forward=True))

            # SS relationship
            elif SS_loc != -1:
                h = int(predecessor[:SS_loc])
                h = h - 1
                PDM_Forward(h, tasks, individual)
                early_set.append(SS_calculation(ESh=tasks.at[h, 'Early_Start'], Di=duration, Si=shiftday, lag=lag_time, forward=True))
            
            # FS relationship
            else:
                h = int(predecessor)
                h = h - 1
                PDM_Forward(h, tasks, individual)
                early_set.append(FS_calculation(EFh=tasks.at[h, 'Early_Finish'], Di=duration, Si=shiftday, lag=lag_time, forward=True))
            
        early_set = np.array(early_set)
        max_es = max(early_set[:,0])
        max_ef = max(early_set[:,1])
        # h_es = h_set[early_set[0].index(max_es)]
        # h_ef = h_set[early_set[1].index(max_ef)]
        tasks.at[i, 'Early_Start'] = max_es
        tasks.at[i, 'Early_Finish'] = max_ef


def PDM_Backward(i, tasks, individual):
    # Backward calculate (Late_Start, Late_Finish)
    if pd.notnull(tasks.at[i, 'Late_Start']):
        return

    successors = tasks.at[i, 'Successors']
    duration = tasks.at[i, 'Duration']

    # No relationship
    if pd.isnull(successors):
        tasks.at[i, 'Late_Start'], tasks.at[i, 'Late_Finish'] = NO_calculation(EFh=max(tasks['Early_Finish']), Di=duration, forward=False)
    else:
        late_set = []
        successors = str(successors).split(',')
        for successor in successors:
            lag_loc = max(successor.find('+'), successor.find('-'))
            if lag_loc != -1:
                lag_time = successor[lag_loc:]
                lag_time = pd.to_timedelta(lag_time).days
            else :
                lag_time = 0
            
            FS_loc = successor.find('FS')
            FF_loc = successor.find('FF')
            SF_loc = successor.find('SF')
            SS_loc = successor.find('SS')
            
            # FS relationship
            if FS_loc != -1:
                j = int(successor[:FS_loc])
                j = j - 1
                PDM_Backward(j, tasks, individual)
                late_set.append(FS_calculation(LSj=tasks.at[j, 'Late_Start'], Di=duration, lag=lag_time, forward=False))

            # FF relationship
            elif FF_loc != -1:
                j = int(successor[:FF_loc])
                j = j - 1
                PDM_Backward(j, tasks, individual)
                late_set.append(FF_calculation(LFj=tasks.at[j, 'Late_Finish'], Di=duration, lag=lag_time, forward=False))

            # SF relationship
            elif SF_loc != -1:
                j = int(successor[:SF_loc])
                j = j - 1
                PDM_Backward(j, tasks, individual)
                late_set.append(SF_calculation(LFj=tasks.at[j, 'Late_Finish'], Di=duration, lag=lag_time, forward=False))

            # SS relationship
            elif SS_loc != -1:
                j = int(successor[:SS_loc])
                j = j - 1
                PDM_Backward(j, tasks, individual)
                late_set.append(SS_calculation(LSj=tasks.at[j, 'Late_Start'], Di=duration, lag=lag_time, forward=False))
            
            # FS relationship
            else:
                j = int(successor[:FS_loc])
                j = j - 1
                PDM_Backward(j, tasks, individual)
                late_set.append(FS_calculation(LSj=tasks.at[j, 'Late_Start'], Di=duration, lag=lag_time, forward=False))

        late_set = np.array(late_set)
        min_ls = min(late_set[:,0])
        min_lf = min(late_set[:,1])
        # h_ls = j_set[late_set[0].index(min_ls)]
        # h_lf = j_set[late_set[1].index(min_lf)]
        tasks.at[i, 'Late_Start'] = min_ls
        tasks.at[i, 'Late_Finish'] = min_lf


def PDM_calculation(tasks, individual):
    tasks_length = tasks.shape[0]
    for i in range(tasks_length):
        PDM_Forward(i, tasks, individual)
        PDM_Backward(tasks_length - i - 1, tasks, individual)


def NO_calculation( ESh=None, EFh=None, LSj=None, LFj=None,
                    Si=0, Di=None, lag=None, exc=0, forward=None):
    if forward :
        ESi = Start_Days + Si + exc
        EFi = ESi + Di - 1 + exc
        return ESi, EFi
    else :
        if pd.notnull(Finish_Days) :
            LFi = Finish_Days
        else :
            LFi = EFh
        LSi = LFi - Di + 1 - exc
        return LSi, LFi


def FS_calculation( ESh=None, EFh=None, LSj=None, LFj=None,
                    Si=0, Di=None, lag=None, exc=0, forward=None):
    if forward :
        ESi = EFh + Si + 1 + lag + exc
        EFi = ESi + Di - 1 + exc
        return ESi, EFi
    else :
        LFi = LSj - 1 - lag - exc
        LSi = LFi - Di + 1 - exc
        return LSi, LFi


def FF_calculation( ESh=None, EFh=None, LSj=None, LFj=None,
                    Si=0, Di=None, lag=None, exc=0, forward=None):
    if forward :
        EFi = EFh + Si + lag
        ESi = EFi - Di + 1 - exc
        return ESi, EFi
    else :
        LFi = LFj - lag
        LSi = LFi - Di + 1 - exc
        return LSi, LFi


def SF_calculation( ESh=None, EFh=None, LSj=None, LFj=None,
                    Si=0, Di=None, lag=None, exc=0, forward=None):
    if forward :
        EFi = ESh + Si - 1 + lag - exc
        ESi = EFi - Di + 1 - exc
        return ESi, EFi
    else :
        LSi = LFj - lag + 1 + exc
        LFi = LSi + Di - 1 + exc
        return LSi, LFi


def SS_calculation( ESh=None, EFh=None, LSj=None, LFj=None,
                    Si=0, Di=None, lag=None, exc=0, forward=None):
    if forward :
        ESi = ESh + Si + lag
        EFi = ESi + Di - 1 + exc
        return ESi, EFi
    else :
        LSi = LSj - lag
        LFi = LSi + Di - 1 + exc
        return LSi, LFi


main()
