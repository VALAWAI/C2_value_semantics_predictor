import numpy as np

from Actions import Action, NB_POSSIBLE_ACTIONS
from Patient import Patient
import Criteria
import numpy
import os
import pandas as pd

use_real_patients = False
file_name = "train_data.csv"

try:
    os.remove(file_name)
except:
    print("Could not remove it!")
    pass



# CRITERIA
AGE = 0
CCD = 1
MACA = 2
EXP_SURVIVAL = 3
FRAILTY = 4
CRG = 5
NS = 6
BARTHEL_INDEPENDENCE = 7
INDEPENDENCE = 8
ADV_DIRECTIVES = 9
COGN_DETERIORATION = 10
EMOTIONAL = 11
DISCOMFORT = 12
AUTO_1 = 13
AUTO_2 = 14
AUTO_3 = 15

criteria_names = {
    AGE: "edat",  # 1, 2, 3, 100, 101, 102, 103, 104, 105, +99
    CCD: "ccd_complex_chronic_disease", # 1, 2, 3
    MACA: "maca_short_term_suvival", # 1, 2, 3
    EXP_SURVIVAL: "expected_survival_in_months", # 1, 2, 3
    FRAILTY: "vig_frail", # 1, 2, 3, 4
    CRG: "clinical_risk_group", # 1, 2, 3, 4, 5
    NS: "social_suport", # 1, 2, 3
    BARTHEL_INDEPENDENCE: "barthel_index_admission",
    INDEPENDENCE: "lawton_index", # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    ADV_DIRECTIVES: "volu", # 1, 2, 3
    COGN_DETERIORATION: "deterioro_cognitivo", # 1, 2, 3, 4
    EMOTIONAL: "malestar_emocional", # 1, 2, 3
    DISCOMFORT: "comfort", # 1, 2, 3, 4
    AUTO_1: "facu", # 1, 2, 3
    AUTO_2: "infor", # 1, 2, 3
    AUTO_3: "coacc", # 1, 2, 3
}

type3 = [1, 2, 3]
type4 = [1, 2, 3, 4]
type5 = [1, 2, 3, 4, 5]
type6 = [1, 2, 3, 4, 5, 6]
type10 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
typeAGE = [1, 2, 3, 4, 100, 101, 102, 103, 104, 105]

last_is_good = [AGE]

are_type_3 = [AUTO_1, AUTO_2, AUTO_3, CCD, MACA, EXP_SURVIVAL, NS, ADV_DIRECTIVES, EMOTIONAL]
are_type_4 = [FRAILTY, COGN_DETERIORATION, DISCOMFORT]
are_type_5 = [CRG]
are_type_6 = [BARTHEL_INDEPENDENCE]
are_type_10 = [INDEPENDENCE]
are_type_AGE = [AGE]

types = [type3, type4, type5, type6, type10, typeAGE]
are_types = [are_type_3, are_type_4, are_type_5, are_type_6, are_type_10, are_type_AGE]


def normalize_criteria_value(variable, variable_idx):
    valid_range_len = 0

    if variable_idx == AGE:
        valid_range_len = typeAGE
    else:
        for i in range(len(types)):
            if variable_idx in are_types[i]:
                valid_range_len = len(types[i])
                if variable == types[i][-1] or numpy.isnan(variable):
                    return 0  # DUMMY VALUE TO SUBSTITUTE UNKNOWN ONES
                break

    if variable_idx == AGE:
        for i in range(len(valid_range_len)):
            if variable == valid_range_len[i]:
                return 0.1*(i)
    elif variable_idx == BARTHEL_INDEPENDENCE:
        if variable == 1:
            return 0.0
        elif variable == 2:
            return 0.21
        elif variable == 3:
            return 0.61
        elif variable == 4:
            return 0.91
        elif variable == 5:
            return 1.0
        else:
            return 0
    elif variable_idx == INDEPENDENCE:

        return (variable -1) / (valid_range_len - 2)
    elif variable_idx == COGN_DETERIORATION:

        if variable_idx == 1:
            return 0.0
        elif variable_idx == 2:
            return 0.667
        elif variable_idx == 3:
            return 1.0
        else:
            return 0


    else:
        if valid_range_len == 4:
            # TODO
            if variable == 1.0:
                print(variable, min(1.0, 1.0 - (variable - 1.0)/ (valid_range_len-2.0)))
            return min(1.0, 1.0 - (variable - 1.0)/ (valid_range_len-2.0))
        elif valid_range_len == 3:
            # GOOD TO HAVE IT
            if variable_idx in [EXP_SURVIVAL, AUTO_3, MACA, CCD, EMOTIONAL]:
                if variable == 2.0:
                    return 1
                else:
                    return 0
            # BAD TO HAVE IT
            elif variable_idx in [AUTO_1, AUTO_2, NS, ADV_DIRECTIVES]:
                if variable == 1.0:
                    return 1
                else:
                    return 0




def normalize_all_criteria(patient_criteria):
    normalized_patient = list()
    for i in range(len(patient_criteria)):
        normalized_patient.append(normalize_criteria_value(patient_criteria[i], i))

    return normalized_patient


n_criteria = Patient().get_raw_criteria().n_criteria



if use_real_patients:
    n_patients = 150
    df = pd.read_csv("VALAWAIQ1_DATA.csv")

    patients_condition = df[[x for x in criteria_names.values()]].to_numpy()
else:
    n_patients = 2000
    patients_condition = None



print("Number of criteria:",  n_criteria)

all_patients = list()

value_names = ["beneficence", "non-maleficence", "justice", "autonomy"]
for i in range(n_patients):
    if use_real_patients:

        print("Patient's raw criteria : ", patients_condition[i])
        criteria = normalize_all_criteria(patients_condition[i])

        print("Patient's criteria: ", criteria)
        print("---")
        patient = Patient(criteria=Criteria.Criteria(criteria, dont_normalize=True))
    else:
        patient = Patient()
    action = Action(random=True, maximum_actions=1)

    patient_list = patient.list() + action.list()

    all_patients.append(patient_list)

    #print(patient.list())
    #print(action.list())

    pre_list = patient.get_patient_state()
    post_list = patient.set_and_apply_treatment(action, random=False)
    align_list = patient.get_alignment()

    print(align_list)
    #print("Amount of benefit: ", align_beneficence(pre_list, post_list))
    #print()
    #print(post_list)
    #print("-----")


    patient_list += post_list + align_list


    patient_list = numpy.asarray(patient_list)



    with open(file_name, "ab") as f:
        #f.write(b"\n")
        print(i)
        if i == 0:

            header_list = patient.get_state_names() + action.get_names() + patient.get_state_names(nit=False) + value_names
            separator = ','
            patient_header = separator.join(header_list)

            print("header : ", patient_header)

            numpy.savetxt(f, [patient_list], delimiter=",", header=patient_header, comments='')
        else:
            numpy.savetxt(f, [patient_list], delimiter=",")


data = [numpy.loadtxt(file_name, delimiter=',', skiprows=1)]
# print the array

data = data[0]



print("Now let's investigate a particular patient")

for i in range(1,2):
    print(len(data[i]))
    nit = data[i][0]
    criteria = data[i][1:n_criteria+1]
    actions = data[i][n_criteria+1:n_criteria+1+NB_POSSIBLE_ACTIONS]
    postcriteria = data[i][n_criteria+1+NB_POSSIBLE_ACTIONS:2*n_criteria+1+NB_POSSIBLE_ACTIONS]
    useful_actions = data[i][2*n_criteria+1+NB_POSSIBLE_ACTIONS:]
    print("NIT level : ", nit)
    print("Initial state : ", criteria)
    print("Treatment applied : ", actions)
    print("Final state : ", postcriteria)
    print("Alignment for each value : ", useful_actions)