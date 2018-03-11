import numpy as np
from skimage import morphology

def validate(answer, truth):
    # labeling
    answer_labels = morphology.label(answer, connectivity=1)
    truth_labels = morphology.label(truth, connectivity=1)

    # nucleus number
    answer_num = np.max(answer_labels)
    truth_num = np.max(truth_labels)

    # separate nucleus
    answer_individual = np.array([np.where(answer_labels == i+1, 1, 0) for i in range(answer_num)])
    truth_individual = np.array([np.where(truth_labels == i+1, 1, 0) for i in range(truth_num)])

    # validation for each threshold
    #validations = np.array([])
    validations = np.empty((10,), float)

    # IoU table
    #IoU_table = np.empty((0,truth_num), float)
    IoU_table = np.empty((answer_num, truth_num), float)

    # thresholds
    threshold = np.linspace(0.5, 0.95, 10)


    # create IoU table
    for i in range(answer_num):
        #IoU_line = np.array([])
        IoU_line = np.empty((truth_num,), float)
        for j in range(truth_num):
            #import pdb; pdb.set_trace()
            label = answer_individual[i] + truth_individual[j]
            IoU = len(np.where(label==2)[0]) / len(np.where(label>=1)[0])
            #IoU_line = np.append(IoU_line, IoU)
            IoU_line[j] = IoU

        #IoU_table = np.append(IoU_table, np.array([IoU_line]), axis=0)
        IoU_table[i] = IoU_line

    # calculate validation for each threshold
    for i in range(len(threshold)):
        TP = len(np.where(IoU_table>=threshold[i])[0])
        FP = answer_num - TP
        FN = truth_num - TP
        #validations = np.append(validations, TP / (TP + FP + FN))
        validations[i] = TP / (TP + FP + FN)
    # calculate overall validation
    validation = np.mean(validations)
    print(validation, validations)
    return validation, validations, IoU_table
