import pandas as pd
import numpy as np
import random as rn

# Set general parameters
starting_population_size = 10
maximum_generation = 20
minimum_population_size = 10
maximum_population_size = 10

Project_Start = pd.to_datetime(
    'October 17, 2018 5:00 PM', format='%B %d, %Y %I:%M %p')
Project_Finish = pd.to_datetime(
    'October 5, 2020 5:00 PM', format='%B %d, %Y %I:%M %p')


def main():
    tasks = pd.read_excel('Optimizing-CMU.xlsm', sheet_name='Task_Table')
    # tasks = tasks[tasks['Summary']=='No']

    # Create starting population
    population = create_population(starting_population_size, len(tasks))

    TF = create_constraints(tasks)


def create_population(individuals, chromosome_length):
    """
    Create random population with given number of individuals and chroosome length.
    """
    # Set up an initial array of all zeros
    population = np.zeros((individuals, chromosome_length))

    return population


def create_constraints(tasks):
    # Forward calculate
    for _ in range(2):
        for i in range(len(tasks)):
            predecessor = tasks.at[i, 'Predecessors']
            # successor = tasks.at[i, 'Successors']
            shiftday = 0
            exception = 0

            # No relationship
            if pd.isnull(predecessor):
                tasks.at[i, 'Early_Start'] = Project_Start + pd.to_timedelta(shiftday, unit='d') + pd.to_timedelta(exception, unit='d')
                tasks.at[i, 'Early_Finish'] = tasks.at[i, 'Early_Start'] + pd.to_timedelta(tasks.at[i, 'Duration']) + pd.to_timedelta(exception - 1, unit='d')

                # print(i + 1, tasks.at[i, 'Early_Start'])
            else:
                lag_loc = max(str(predecessor).find('+'), str(predecessor).find('-'))
                if lag_loc != -1:
                    lag_time = str(predecessor)[lag_loc:]
                else :
                    lag_time = '+0 days'
                
                FS_loc = str(predecessor).find('FS')
                FF_loc = str(predecessor).find('FF')
                SF_loc = str(predecessor).find('SF')
                SS_loc = str(predecessor).find('SS')

                # FS relationship
                if FS_loc != -1:
                    h = int(str(predecessor)[0:FS_loc])-1
                    tasks.at[i, 'Early_Start'] = tasks.at[h, 'Early_Finish'] + pd.to_timedelta(shiftday, unit='d') + pd.to_timedelta(lag_time) + pd.to_timedelta(1 + exception, unit='d')
                    tasks.at[i, 'Early_Finish'] = tasks.at[i, 'Early_Start'] + pd.to_timedelta(tasks.at[i, 'Duration']) + pd.to_timedelta(exception - 1, unit='d')
                # FF relationship
                elif FF_loc != -1:
                    h = int(str(predecessor)[0:FF_loc])-1
                    tasks.at[i, 'Early_Finish'] = tasks.at[h, 'Early_Finish'] + pd.to_timedelta(shiftday, unit='d') + pd.to_timedelta(lag_time)
                    tasks.at[i, 'Early_Start'] = tasks.at[i, 'Early_Finish'] - pd.to_timedelta(tasks.at[i, 'Duration']) + pd.to_timedelta(1 + exception, unit='d')
                # SF relationship
                elif SF_loc != -1:
                    h = int(str(predecessor)[0:SF_loc])-1
                    tasks.at[i, 'Early_Finish'] = tasks.at[h, 'Early_Start'] + pd.to_timedelta(shiftday, unit='d') + pd.to_timedelta(lag_time) + pd.to_timedelta(- 1 - exception, unit='d')
                    tasks.at[i, 'Early_Start'] = tasks.at[i, 'Early_Finish'] - pd.to_timedelta(tasks.at[i, 'Duration']) + pd.to_timedelta(1 - exception, unit='d')
                # SS relationship
                elif SS_loc != -1:
                    h = int(str(predecessor)[0:SS_loc])-1
                    tasks.at[i, 'Early_Start'] = tasks.at[h, 'Early_Start'] + pd.to_timedelta(shiftday, unit='d') + pd.to_timedelta(lag_time)
                    tasks.at[i, 'Early_Finish'] = tasks.at[i, 'Early_Start'] + pd.to_timedelta(tasks.at[i, 'Duration']) + pd.to_timedelta(exception - 1, unit='d')
                else:
                    h_set = str(predecessor).split(',')
                    h_set = list(map(int, h_set))
                    h_es = []
                    for h in h_set:
                        h = h - 1
                        h_es.append(tasks.at[h, 'Early_Finish'] + pd.to_timedelta(shiftday, unit='d') + pd.to_timedelta(lag_time) + pd.to_timedelta(1 + exception, unit='d'))

                    # print(h_es)
                    max_es = max(h_es)
                    h_loc = h_es.index(max_es)
                    h = h_set[h_loc] - 1
                    tasks.at[i, 'Early_Start'] = max_es
                    tasks.at[i, 'Early_Finish'] = tasks.at[i, 'Early_Start'] + pd.to_timedelta(tasks.at[i, 'Duration']) + pd.to_timedelta(exception - 1, unit='d')

                print(i + 1, h + 1, tasks.at[i, 'Early_Start'])
        print()

    # #Backward calculate
    return 0


main()
