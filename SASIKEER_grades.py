import numpy as np
import csv


def gradeInfo(filename, numExams, hwWeight):

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        rows = []
        for row in reader:
            if ("%" in row[0]):
                continue
            rows.append(row)
 
    rows = np.asarray(rows)
    rows = np.delete(rows, -1, axis=1)  # remove last column that contains comment
    data = rows[4:] 
    data[data==''] = '0'
    data = data.astype(int)

    hw_1 = data[:,1]*10 
    hw1_average = hw_1.mean()

    sort1 = data[:, [0, 2]]
    hw2_sorted=sort1[sort1[:, 1].argsort()]

    temp = data[:, [0,1,3]]
    mask = (temp[:,[1]] > 0.1*89) & (temp[:,[2]] > 0.5*89)
    mask=(np.tile(mask,(1,3)))
    Y = temp[mask]
    Y = np.reshape(Y, (-1, 3))
    score_above_90 = Y[:,0]

    temp = data[:, [0,1,2]]
    mask = (temp[:,[1]] < 0.1*81) & (temp[:,[2]] > 0.1*89)
    mask=(np.tile(mask,(1,3)))
    d2 = temp[mask]
    d2 = np.reshape(d2,(-1,3))
    no_of_students, columns = d2.shape

    max_hw, max_exam = rows[2, 1:-1] , rows[2, -1]
    hw1_scaled = data[:,1] / int(max_hw[0])
    hw2_scaled = data[:,2] / int(max_hw[1])
    hw3_scaled = data[:,3] / int(max_hw[2])
    hw4_scaled = data[:,4] / int(max_hw[3])
    hw_scaled = ((hw1_scaled + hw2_scaled + hw3_scaled + hw4_scaled) / 4) * hwWeight
    exam_scaled = ((data[:,5] / int(max_exam)) / numExams) * (1 - hwWeight)
    weighted_score = np.around((exam_scaled + hw_scaled) * 100 , 1)
    weighted_score_list = np.hstack((data[:,0].reshape(-1,1), weighted_score.reshape(-1,1)))

    return (hw1_average, hw2_sorted, score_above_90, no_of_students, weighted_score_list)
