import numpy as np
from constants import BENEFICENCE, NONMALEFICENCE, AUTONOMY, JUSTICE, n_actions



neutral_function = lambda x, y : x - y
benef_function = lambda x, y : max(0, x - y)
nonmalef_function = lambda x, y : min(0, x - y)
useful_function = lambda x, y: 1 if x - y > 0 else 0

criteria_aggregation_function = lambda x, y : np.dot(x, y)

binary_aggregation_function = lambda x, y: 1 if np.dot(x, y) > 0 else -1

criteria_aggregation_function_positive = lambda x, y : max(1.0, np.dot(x, y))
criteria_aggregation_function_negative = lambda x, y : min(-1.0, np.dot(x, y))

w_AGE = 5
w_CCD = 5
w_MACA = 5
w_EXP_SURVIVAL = 40
w_FRAILTY = 8
w_CRG = 4
w_NS = 0
w_BARTHEL = 20
w_LAWTON = 20
w_ADV_DIRECTIVES = 7
w_COGN_DETERIORATION = 2
w_EMOTIONAL = 0
w_DISCOMFORT = 7

w_AUTONOMY_UNDERSTAND = 0
w_AUTONOMY_INFORM = 0
w_AUTONOMY_COERCE = 0

beneficence_weights = [w_AGE, w_CCD, w_MACA, w_EXP_SURVIVAL,
                       w_FRAILTY, w_CRG, w_NS, w_BARTHEL, w_LAWTON, w_ADV_DIRECTIVES,
                       w_COGN_DETERIORATION, w_EMOTIONAL, w_DISCOMFORT,
                       w_AUTONOMY_UNDERSTAND, w_AUTONOMY_INFORM, w_AUTONOMY_COERCE]

beneficence_weights = np.array(beneficence_weights)/np.sum(beneficence_weights)


w_AGE = 1
w_CCD = 2
w_MACA = 3
w_EXP_SURVIVAL = 18
w_FRAILTY = 6
w_CRG = 1
w_NS = 0.0
w_BARTHEL = 12
w_LAWTON = 12
w_ADV_DIRECTIVES = 2
w_COGN_DETERIORATION = 2
w_EMOTIONAL = 0
w_DISCOMFORT = 17

w_AUTONOMY_UNDERSTAND = 0
w_AUTONOMY_INFORM = 0
w_AUTONOMY_COERCE = 0

nonmaleficence_weights = [w_AGE, w_CCD, w_MACA, w_EXP_SURVIVAL,
                          w_FRAILTY, w_CRG, w_NS, w_BARTHEL, w_LAWTON, w_ADV_DIRECTIVES,
                          w_COGN_DETERIORATION, w_EMOTIONAL, w_DISCOMFORT,
                          w_AUTONOMY_UNDERSTAND, w_AUTONOMY_INFORM, w_AUTONOMY_COERCE]

nonmaleficence_weights = np.array(nonmaleficence_weights)/np.sum(nonmaleficence_weights)



w_AGE = 0.0
w_CCD = 0.0
w_MACA = 0.0
w_EXP_SURVIVAL = 0.0
w_FRAILTY = 0.0
w_CRG = 0.0
w_NS = 0.0
w_BARTHEL = 0.0
w_LAWTON = 0.0
w_ADV_DIRECTIVES = 0.0
w_COGN_DETERIORATION = 0.0
w_EMOTIONAL = 0.0
w_DISCOMFORT = 0.0

w_AUTONOMY_UNDERSTAND = 0.25
w_AUTONOMY_INFORM = 0.5
w_AUTONOMY_COERCE = 0.25


autonomy_weights = [w_AGE, w_CCD, w_MACA, w_EXP_SURVIVAL,
                          w_FRAILTY, w_CRG, w_NS, w_BARTHEL, w_LAWTON, w_ADV_DIRECTIVES,
                          w_COGN_DETERIORATION, w_EMOTIONAL, w_DISCOMFORT,
                          w_AUTONOMY_UNDERSTAND, w_AUTONOMY_INFORM, w_AUTONOMY_COERCE]


w_AGE = 0.0
w_CCD = 0.0
w_MACA = 0.0
w_EXP_SURVIVAL = 0.0
w_FRAILTY = 0.0
w_CRG = 0.0
w_NS = 0.0
w_BARTHEL = 0.0
w_LAWTON = 0.0
w_ADV_DIRECTIVES = 0.0
w_COGN_DETERIORATION = 0.0
w_EMOTIONAL = 0.0
w_DISCOMFORT = 0.0

w_AUTONOMY_UNDERSTAND = 0.0
w_AUTONOMY_INFORM = 0.0
w_AUTONOMY_COERCE = 0.0

justice_weights = [w_AGE, w_CCD, w_MACA, w_EXP_SURVIVAL,
                          w_FRAILTY, w_CRG, w_NS, w_BARTHEL, w_LAWTON, w_ADV_DIRECTIVES,
                          w_COGN_DETERIORATION, w_EMOTIONAL, w_DISCOMFORT,
                          w_AUTONOMY_UNDERSTAND, w_AUTONOMY_INFORM, w_AUTONOMY_COERCE]



np.random.seed(0)

justice_weights = np.random.rand(len(justice_weights) + 11)

justice_weights = justice_weights / np.sum(justice_weights)


def align_value(criteria, action, post_criteria, value):

    if value == BENEFICENCE:
        return align_beneficence(criteria, post_criteria)
    elif value == NONMALEFICENCE:
        return align_nonmaleficence(criteria, post_criteria)
    elif value == JUSTICE:
        return align_justice(criteria, action)
    elif value == AUTONOMY:
        return align_autonomy(criteria, action)

def align_values(criteria, action, post_criteria):

    return [align_beneficence(criteria, post_criteria),
            align_nonmaleficence(criteria, post_criteria),
            align_justice(criteria, action),
            align_autonomy(criteria, action)]

def align_consequence(criteria, post_criteria, weights, inner_func, outer_func):

    delta_criteria = [0 for _ in range(len(criteria))]

    for i in range(len(criteria)):
        delta_criteria[i] = inner_func(post_criteria[i], criteria[i])


    alignment = outer_func(weights, delta_criteria)

    return alignment



def align_beneficence(criteria, post_criteria):
    # beneficence_weights = [1 for _ in range(len(criteria))]

    return align_consequence(criteria, post_criteria, beneficence_weights, neutral_function, criteria_aggregation_function)


def align_nonmaleficence(criteria, post_criteria):



    return align_consequence(criteria, post_criteria, nonmaleficence_weights, neutral_function, criteria_aggregation_function)


def align_justice(criteria, action):

    # TODO: Clean this code
    try:

        ali = np.dot(justice_weights[:-n_actions], criteria) + np.dot(justice_weights[-n_actions:], action.list())

        return ali
    except:

        if isinstance(action, int):
            action_list = [0 for _ in range(n_actions)]
            action_list[action] = 1
            action = action_list

        return np.dot(justice_weights[:-n_actions], criteria) + np.dot(justice_weights[-n_actions:], action)

def align_autonomy(criteria, action):


    auto_U = criteria[-3]
    auto_I = criteria[-2]
    auto_C = criteria[-1]


    auto_U = -1 if auto_U == 0 else 1
    auto_I = -1 if auto_I == 0 else 1
    auto_C = -1 if auto_C == 0 else 1


    ali = autonomy_weights[-3]*auto_U + autonomy_weights[-2]*auto_I + autonomy_weights[-1]*auto_C


    return ali
if __name__ == "__main__":



    print("---- Now with a synthetic patient -----")

    from Patient import Patient
    from Actions import Action

    patient = Patient()
    action = Action(adv_action=True)
    print("Action : ", action.list())

    pre_list = patient.get_patient_state()

    print("State: ", pre_list)
    post_list = patient.set_and_apply_treatment(action)

    print("Post state: ", post_list)

    print("Alignment: ", patient.get_alignment())




