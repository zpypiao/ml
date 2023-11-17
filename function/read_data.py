import numpy as np

# this is a function which define to read the data
def read_data():

    # creat the new list to store the raw data
    raw_data = []

    # import data
    with open('./data/abalone.data', 'r') as file:
        for line in file:
            raw_data.append(line.strip().split(','))

    # creat the list to process the raw data
    processed_data = []

    # if the first parameter is 'M', then replace it by '1';
    # if the first parameter is 'F', then replace it by '2';
    # if the first parameter is 'I', then replace it by '3'.
    for each in raw_data:
        if each[0] == 'M':
            each[0] = 1
        elif each[0] == 'F':
            each[0] = 2
        else:
            each[0] = 3
        
        processed_data.append(each)

    # transfer the string list to a numpy array
    data = np.asarray(processed_data, dtype=float)

    return data