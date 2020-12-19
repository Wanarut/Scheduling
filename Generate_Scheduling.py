import pandas as pd
import numpy as np
import random as rn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer
from scipy import interpolate

Start_Date = pd.to_datetime('October 17, 2018 5:00 PM', format='%B %d, %Y %I:%M %p')
Finish_Date = pd.to_datetime('October 5, 2020 5:00 PM', format='%B %d, %Y %I:%M %p')
Start_Days = (Start_Date-Start_Date).days
Finish_Days = (Finish_Date-Start_Date).days
max_project_duration = 780

def main():
    tasks = pd.read_excel('Optimizing-CMU.xlsm', sheet_name='Task_Table')
    costs = pd.read_excel('สรุป Cost BOQ ตาม Activity ID RV.1.xlsx', sheet_name='BOQ Activity')
    # Filter reduce data load
    filter_tasks = tasks[['Duration','Predecessors','Predecessors2','Successors','Early_Start','Early_Finish','Late_Start','Late_Finish']]
    costs = costs[['Resource\n(คน)','ค่าวัสดุรวม\n(บาท)','ค่าวัสดุต่อวัน\n(บาท/วัน)','ค่าแรงงานต่อวัน\n(บาท/วัน)']]  
    filter_tasks['Duration'] = pd.to_timedelta(filter_tasks['Duration']).dt.days

    shiftdays = pd.read_csv('shiftdays.csv', header=None)
    options = pd.read_csv('options.csv', header=None)

    chromosome_length = len(tasks)
    
    score_length = len(pd.read_csv('scores.csv', header=None))
    with pd.ExcelWriter('output_tasks.xlsx') as writer:
        for i in range(score_length):
            individual = np.zeros((chromosome_length, 2), int)
            shiftday = shiftdays[i].T
            option = options[i].T
            individual[:,0] = shiftday
            individual[:,1] = option
            # print(individual)

            shift_tasks = filter_tasks.copy()
            save_tasks = tasks.copy()
            PDM_calculation(shift_tasks, individual)
            # print(tasks)

            save_tasks['Early_Start'] = Start_Date + pd.to_timedelta(shift_tasks['Early_Start'], unit='d')
            save_tasks['Early_Finish'] = Start_Date + pd.to_timedelta(shift_tasks['Early_Finish'], unit='d')
            save_tasks['Late_Start'] = Start_Date + pd.to_timedelta(shift_tasks['Late_Start'], unit='d')
            save_tasks['Late_Finish'] = Start_Date + pd.to_timedelta(shift_tasks['Late_Finish'], unit='d')

            save_tasks['Early_Start'] = save_tasks['Early_Start'].dt.strftime('%B %d, %Y %I:%M %p')
            save_tasks['Early_Finish'] = save_tasks['Early_Finish'].dt.strftime('%B %d, %Y %I:%M %p')
            save_tasks['Late_Start'] = save_tasks['Late_Start'].dt.strftime('%B %d, %Y %I:%M %p')
            save_tasks['Late_Finish'] = save_tasks['Late_Finish'].dt.strftime('%B %d, %Y %I:%M %p')

            cost_0 = calculate_cost_fitness(shift_tasks, costs)
            time_0 = calculate_time_fitness(shift_tasks)
            mx_0 = calculate_mx_fitness(shift_tasks, costs)

            print('=====Solution', i+1, '=====')
            print('Total Cost', cost_0, 'Baht')
            print('Project Duration', time_0, 'Days')
            print('Mx', mx_0, 'man^2')

            save_tasks.to_excel(writer, sheet_name='solutoion_' + str(i+1), index=False)


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
