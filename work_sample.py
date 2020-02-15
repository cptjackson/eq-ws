import pandas as pd
import numpy as np


###
# Problem 1: Cleanup
##
df = pd.read_csv('data/DataSample.csv')

# Drop duplicates, ignoring ID
print('Problem 1: dropped duplicates')
print('Duplicates dropped: {}'.format(df.duplicated(subset=[' TimeSt',
                                                            'Latitude',
                                                            'Longitude'])
                                                            .sum()),'\n')
df.drop_duplicates(subset=[' TimeSt', 'Latitude','Longitude'],inplace=True)
df.reset_index(drop=True,inplace=True)


###
# Problem 2: Label
##
df_pois = pd.read_csv('data/POIList.csv')

# Remove identical POIs (POI1 and POI2 are the same in the sample)
df_pois.drop_duplicates(subset=[' Latitude','Longitude'],inplace=True)
df_pois.reset_index(drop=True,inplace=True)

# Find the Haversine distance between two lat/lon points
# Optimized to use NumPy arrays for speed
def distance(lat1, lon1, lat2, lon2):
    p = np.pi/180
    a = 0.5 - np.cos((lat2-lat1)*p)/2 + np.cos(lat1*p)*np.cos(lat2*p) *\
        (1-np.cos((lon2-lon1)*p)) / 2
    return 12742 * np.arcsin(np.sqrt(a))

# Convert to NumPy and calculate all Haversine distances to all POIs for each
# request - Pandas is too slow to scale this effectively
lat1 = df['Latitude'].values
lat2 = df_pois[' Latitude'].values.reshape(-1,1)
lon1 = df['Longitude'].values
lon2 = df_pois['Longitude'].values.reshape(-1,1)
distances = distance(lat1, lon1, lat2, lon2)
df['Closest POI'] = pd.Series(np.argmin(distances,
                                        axis=0)).replace(df_pois['POIID'])
df['POI Distance'] = np.min(distances,axis=0)


###
# Problem 3: Analysis
##

# 1. Average and standard deviation of POI to assigned requests
df_pois['Average Distance'] =\
    df.groupby('Closest POI')['POI Distance'].mean().values
df_pois['Std Distance'] =\
    df.groupby('Closest POI')['POI Distance'].std().values

print('Problems 2 and 3: average and std distances, radius and density')
print('Average distance (km): ')
print(df_pois[['POIID','Average Distance']],'\n')
print('Std distance (km): ')
print(df_pois[['POIID','Std Distance']],'\n')

# 2. Radius and density of a circle with the POI at its centre
df_pois['Radius'] = df.groupby('Closest POI')['POI Distance'].max().values
df_pois['Area'] = np.pi * np.power(df_pois['Radius'],2)
df_pois['Requests'] = df['Closest POI'].value_counts().values
df_pois['Density'] = df_pois['Requests'] / df_pois['Area']

print('Radius (km): ')
print(df_pois[['POIID','Radius']],'\n')
print('Density (requests/km): ')
print(df_pois[['POIID','Density']],'\n')


###
# Problem 4a: Model
##

# Check file bonus.txt for assumptions and hypotheses
# Model assumes that distance and density are both important
#
#  popularity = sum(1/(request distance)^2)^4 * density

def map_raw_scores(raw_scores):

    op_start = -10
    op_end = 10
    ip_start = np.min(raw_scores)
    ip_end = np.max(raw_scores)

    output = op_start + ((op_end - op_start) / (ip_end - ip_start)) \
             * (raw_scores - ip_start)

    return output.round()


def score_pois(df, df_pois):

    total = np.power(df.groupby('Closest POI')['POI Distance']
                     .apply(lambda x: np.sum(np.reciprocal(x))),4) *\
                     df_pois['Density'].values
    return total

popularity = map_raw_scores(score_pois(df,df_pois))

print('Problem 4a: POI Popularity')
print(popularity,'\n')


###
# Problem 4b: Pipeline Dependency
##

# Read in tasks and dependencies
tasks = pd.read_csv('data/task_ids.txt',header=None).values[0]
dep_arr = pd.read_csv('data/relations.txt',header=None,sep='->',
                   engine='python').values
start_task = 73
goal_task = 36

def get_tasks(dep_arr,tasks,tasks_running):

    # Deal with scalars
    if isinstance(tasks,int):
        tasks = [tasks]

    for task in tasks:
        if task not in tasks_running:

            # If it has dependencies, call the function again
            deps = list(dep_arr[np.where(dep_arr[:,1] == task),0][0])
            if deps:
                tasks_running = get_tasks(dep_arr,deps,tasks_running)

            # Append the task to the list
            tasks_running.append(task)

    return tasks_running


def get_task_order(start_task,end_task,dep_arr):

    tasks_running = []

    # Start initial tasks running (make sure we make a copy)
    initial_tasks = list(get_tasks(dep_arr,start_task,tasks_running))

    # Remove starting task for the initial run
    initial_tasks.pop(-1)

    # Final task list
    task_order = get_tasks(dep_arr,end_task,tasks_running)

    # Remove initial tasks from task order list
    [task_order.remove(task) for task in initial_tasks]

    return task_order


print('Problem 4b: Pipeline dependency task order')
print(get_task_order(start_task,goal_task,dep_arr))
