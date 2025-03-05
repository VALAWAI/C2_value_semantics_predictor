import numpy as np
import Actions
import Criteria
import Patient
from Alignment import align_value
from constants import BENEFICENCE, NONMALEFICENCE, JUSTICE, AUTONOMY, n_consequence_values, n_values
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from Alignment import beneficence_weights, nonmaleficence_weights, justice_weights, autonomy_weights
from sklearn.model_selection import train_test_split
import pickle

n_criteria = Patient.NB_CRITERIA
n_actions = Actions.NB_POSSIBLE_ACTIONS

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

PREDICT_SA_nextstate = 0
PREDICT_SS_alignment = 1
PREDICT_SA_alignment = 2
PREDICT_SA_alignment_multi_action = 3

class cheat_model:



    def __init__(self, mode, value=None, action=None, post_criterion=None):

        self.__mode = mode
        self.__value = value
        self.__action = action
        self.__post_criterion = post_criterion

        if self.__mode < 0 or self.__mode > 3:
            print("Warning!! Incompatible mode!!")

    def predict(self, data):
        output_list = list()

        if self.__mode == PREDICT_SA_nextstate:
            for i in range(len(data)):
                criteria = data[i][:-1]

                #print("Patient's age before:", data[i][0])


                aux_patient = Patient.Patient(criteria=Criteria.Criteria(criteria, dont_normalize=True))


                postcriteria = aux_patient.set_and_apply_treatment(action=self.__action, random=False)


                post_state = Criteria.Criteria(postcriteria).get_criteria_normalized()
                #print("Patient's age after: ", post_state[0])
                #print()

                output_list.append(post_state[self.__post_criterion])

        elif self.__mode == PREDICT_SS_alignment:

            dummy_action = 0
            for i in range(len(data)):

                post_criteria = data[i]
                criteria = np.zeros(len(data[i]))

                alignment = align_value(criteria, dummy_action, post_criteria, self.__value)
                output_list.append(alignment)

        elif self.__mode == PREDICT_SA_alignment:
            for i in range(len(data)):
                criteria = data[i][:-1]

                dummy_post_criteria = criteria

                alignment = align_value(criteria, self.__action, dummy_post_criteria, self.__value)

                output_list.append(alignment)

        elif self.__mode == PREDICT_SA_alignment_multi_action:
            for i in range(len(data)):

                criteria = data[i][:-n_actions]

                action = data[i][-n_actions:]

                dummy_post_criteria = criteria

                alignment = align_value(criteria, action, dummy_post_criteria, self.__value)

                output_list.append(alignment)

        return output_list

def preprocess_actions(raw_dataset, value):
    """

    This method is the preprocess for later computing Align(S,a,v)
    for a particular value v and a particular action a

    :param raw_dataset:
    :return:
    """

    X_y = list()

    for i in range(len(raw_dataset)):

        precriteria = raw_dataset[i][1:n_criteria + 1]
        applied_actions = raw_dataset[i][n_criteria + 1:n_criteria + 1 + n_actions]

        reward = raw_dataset[i][2 * n_criteria + 1 + n_actions:][value]

        state = Criteria.Criteria(precriteria).get_criteria_normalized()

        state_action = np.concatenate((state, applied_actions))

        xy = np.concatenate((state_action, [reward]))

        X_y.append(xy)

    X_y = np.array(X_y)

    #return X_y
    "remove repeated patients and return"

    return np.unique(X_y, axis=0)

def preprocess_action(raw_dataset, action, value):
    """

    This method is the preprocess for later computing Align(S,a,v)
    for a particular value v and a particular action a

    :param raw_dataset:
    :return:
    """

    X_y = list()

    for i in range(len(raw_dataset)):

        precriteria = raw_dataset[i][1:n_criteria + 1]
        applied_actions = raw_dataset[i][n_criteria + 1:n_criteria + 1 + n_actions][action]

        if applied_actions == 1:
            #postcriteria = data[i][n_criteria + 1 + n_actions:2 * n_criteria + 1 + n_actions]

            reward = raw_dataset[i][2 * n_criteria + 1 + n_actions:][value]

            state = Criteria.Criteria(precriteria).get_criteria_normalized()

            state_action = np.concatenate((state, [applied_actions]))

            xy = np.concatenate((state_action, [reward]))

            X_y.append(xy)

    X_y = np.array(X_y)

    #return X_y
    "remove repeated patients and return"

    return np.unique(X_y, axis=0)


def preprocess_consequentialist(raw_dataset, value, all_values=False):
    """

    This method is the preprocess for later computing Align(S,S')

    :param raw_dataset:
    :return:
    """

    X_y = list()

    for i in range(len(raw_dataset)):

        precriteria = raw_dataset[i][1:n_criteria + 1]
        #applied_actions = raw_dataset[i][n_criteria + 1:n_criteria + 1 + n_actions]
        if all_values:
            reward = raw_dataset[i][2 * n_criteria + 1 + n_actions:]
        else:
            reward = raw_dataset[i][2 * n_criteria + 1 + n_actions:][value]
        postcriteria = raw_dataset[i][n_criteria + 1 + n_actions:2 * n_criteria + 1 + n_actions]

        state = Criteria.Criteria(precriteria).get_criteria_normalized()

        post_state = Criteria.Criteria(postcriteria).get_criteria_normalized()

        delta = np.array(post_state) - np.array(state)

        #print(i, "Delta : ", delta,  "reward : ", reward)
        if all_values:
            xy = np.concatenate((delta, np.array(reward)))
        else:
            xy = np.concatenate((delta, np.array([reward])))

        X_y.append(xy)

    X_y = np.array(X_y)

    #return X_y
    "remove repeated patients and return"

    return np.unique(X_y, axis=0)


def preprocess_next_state(raw_dataset, action, post_criterion):
    """

    This method is the preprocess for later computing Succ(S,a) = s'
    for a particular criterion s' and a particular action a

    :param raw_dataset:
    :return:
    """

    X_y = list()

    for i in range(len(raw_dataset)):

        precriteria = raw_dataset[i][1:n_criteria + 1]
        applied_actions = raw_dataset[i][n_criteria + 1:n_criteria + 1 + n_actions][action]

        if applied_actions == 1:
            postcriteria = raw_dataset[i][n_criteria + 1 + n_actions:2 * n_criteria + 1 + n_actions]

            state = Criteria.Criteria(precriteria).get_criteria_normalized()

            post_state = Criteria.Criteria(postcriteria).get_criteria_normalized()

            s_prima = post_state[post_criterion] # We only care about one criterion

            state_action = np.concatenate((state, [applied_actions]))


            #print(i, "Delta : ", delta,  "reward : ", reward)
            xy = np.concatenate((state_action, [s_prima]))

            X_y.append(xy)

    X_y = np.array(X_y)

    #return X_y
    "remove repeated patients and return"

    return np.unique(X_y, axis=0)


def preprocess_everything(raw_dataset):
    """

    This method is the preprocess for later computing Align(S,a,S',V)
    for a particular criterion s' and a particular action a

    :param raw_dataset:
    :return:
    """

    X_1 = list()
    X_2= list()
    X_3 = list()
    Y = list()



    for i in range(len(raw_dataset)):

        precriteria = raw_dataset[i][1:n_criteria + 1]
        applied_actions = raw_dataset[i][n_criteria + 1:n_criteria + 1 + n_actions]

        for j in range(len(applied_actions)):
            if applied_actions[j] == 1:
                applied_actions = j
                break

        state = Criteria.Criteria(precriteria).get_criteria_normalized()
        postcriteria = raw_dataset[i][n_criteria + 1 + n_actions:2 * n_criteria + 1 + n_actions]
        post_state = Criteria.Criteria(postcriteria).get_criteria_normalized()


        reward = raw_dataset[i][2 * n_criteria + 1 + n_actions:]

        X_1.append(np.concatenate((state, [applied_actions])))
        X_2.append(np.array(state))
        X_3.append(np.array(post_state))
        Y.append(reward)


    #return X_y
    "remove repeated patients and return"

    return np.array(X_1), np.array(X_2), np.array(X_3), np.array(Y)



def regression_post_state(data, cheat=True):

    """
    Computes Succ(S,A)

    :param data:
    :return:
    """


    predict_weights_values = list()
    r2_values = list()
    models = list()

    for a in range(n_actions):

        predict_weights_values.append(list())
        r2_values.append(list())
        models.append(list())

        for post_c in range(n_criteria):

            X_y = preprocess_next_state(data, a, post_c)

            X = X_y[:, :-1]
            Y = np.array(X_y[:, -1])




            " Now we will split the dataset between training and testint"
            X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

            #print("Shape of train test : ", X_train.shape, X_test.shape)

            #print("---")
            #print("Shape of test set : ", y_train.shape, y_test.shape)

            if cheat:
                model = cheat_model(mode=PREDICT_SA_nextstate, action=a, post_criterion=post_c)
            else:
                model = linear_model.LinearRegression()
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            if cheat:
                predicted_weights = None
            else:
                predicted_weights = model.coef_
                #mse = mean_squared_error(true_weights[value], predicted_weights)

            predict_weights_values[a].append(predicted_weights)
            #mse_values.append(mse)
            r2_values[a].append(r2)
            models[a].append(model)

    return predict_weights_values, r2_values, models


def regression_alignmentSA_multi_actions(train_data, test_data, cheat=False, pretrained=False):

    """
    Computes Align(S,A)

    :param data:
    :return:
    """


    predict_weights_values = list()
    r2_values = list()
    models = list()


    for v in [JUSTICE, AUTONOMY]:

        X_y = preprocess_actions(train_data, v)


        X = X_y[:, :-1]
        Y = np.array(X_y[:, -1])

        if test_data is not None:
            X_y_test = preprocess_actions(test_data, v)

            X_train = X
            y_train = Y

            X_test = X_y_test[:, :-1]
            y_test = np.array(X_y_test[:, -1])

        else:
            X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

        " Now we will split the dataset between training and testing"

        #print("Shape of train test : ", X_train.shape, X_test.shape)

        #print("---")
        #print("Shape of test set : ", y_train.shape, y_test.shape)

        if pretrained:
            with open('model_SA_' + str(v) + '.pkl', 'rb') as f:
                model = pickle.load(f)
        else:
            if cheat:

                model = cheat_model(mode=PREDICT_SA_alignment_multi_action, value=v)
            else:
                model = linear_model.LinearRegression()
                model.fit(X_train, y_train)

                with open('model_SA_' + str(v) + '.pkl', 'wb') as f:
                    pickle.dump(model, f)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        if cheat:
            predicted_weights = None
        else:
            predicted_weights = model.coef_
        #print("The predicted weights : ", model.coef_)

        #mse = mean_squared_error(true_weights[value], predicted_weights)

        predict_weights_values.append(predicted_weights)
        #mse_values.append(mse)
        r2_values.append(r2)
        models.append(model)

    return predict_weights_values, np.array(r2_values), models


def regression_alignmentSA(train_data, test_data, cheat=False, pretrained=False):

    """
    Computes Align(S,A)

    :param data:
    :return:
    """


    predict_weights_values = list()
    r2_values = list()
    models = list()

    for a in range(n_actions):

        predict_weights_values.append(list())
        r2_values.append(list())
        models.append(list())

        for v in range(n_values):

            X_y = preprocess_action(train_data, a, v)

            if len(X_y) == 0:
                print("Not enough data for action ", a)

            X = X_y[:, :-1]
            Y = np.array(X_y[:, -1])

            if test_data is not None:
                X_y_test = preprocess_action(test_data, a, v)

                X_train = X
                y_train = Y

                X_test = X_y_test[:, :-1]
                y_test = np.array(X_y_test[:, -1])

            else:
                X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

            " Now we will split the dataset between training and testing"

            #print("Shape of train test : ", X_train.shape, X_test.shape)

            #print("---")
            #print("Shape of test set : ", y_train.shape, y_test.shape)

            if pretrained:
                with open('model_SA' + str(v) + '.pkl', 'rb') as f:
                    model = pickle.load(f)

            else:
                if cheat:

                    one_hot_action = [0 for _ in range(n_actions)]
                    one_hot_action[a] = 1

                    model = cheat_model(mode=PREDICT_SA_alignment, value=v, action=one_hot_action)
                else:
                    model = linear_model.LinearRegression()
                    model.fit(X_train, y_train)

                    with open('model_SA' + str(v) + '.pkl', 'wb') as f:
                        pickle.dump(model, f)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            if cheat:
                predicted_weights = None
            else:
                predicted_weights = model.coef_
            #mse = mean_squared_error(true_weights[value], predicted_weights)

            predict_weights_values[a].append(predicted_weights)
            #mse_values.append(mse)
            r2_values[a].append(r2)
            models[a].append(model)

    return predict_weights_values, np.array(r2_values), models




def regression_alignmentSS(train_data, test_data, cheat=False, pretrained=False):
    """
     Computes Align(S,S')

     :param data:
     :return:
     """

    #true_weights = [beneficence_weights, nonmaleficence_weights, autonomy_weights, justice_weights]

    predict_weights_values = list()
    #mse_values = list()
    r2_values = list()
    models = list()

    X_y = (preprocess_consequentialist(train_data, None, all_values=True))

    if test_data is not None:
        X_y_test = (preprocess_consequentialist(test_data, None, all_values=True))

    for value in range(n_consequence_values):
        X = X_y[:, :-n_values]
        Y = np.array(X_y[:, -(n_values-value)])


        " Now we will split the dataset between training and testint"
        if test_data is not None:
            X_train = X
            y_train = Y

            X_test = X_y_test[:, :-n_values]
            y_test = np.array(X_y_test[:, -(n_values - value)])

        else:
            X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

        #print("Shape of train test : ", X_train.shape, X_test.shape)

        #print("---")
        #print("Shape of test set : ", y_train.shape, y_test.shape)

        if pretrained:
            with open('model_SS' + str(value) + '.pkl', 'rb') as f:
                model = pickle.load(f)
        else:
            if cheat:
                model = cheat_model(mode=PREDICT_SS_alignment,value=value,action=None)
            else:
                model = linear_model.LinearRegression()
                model.fit(X_train, y_train)

                with open('model_SS' + str(value) + '.pkl', 'wb') as f:
                    pickle.dump(model, f)


        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)

        if cheat:
            predicted_weights = None
        else:
            predicted_weights = model.coef_
            #print("The predicted weights : ", model.coef_)
        #mse = mean_squared_error(true_weights[value], predicted_weights)

        predict_weights_values.append(predicted_weights)
        #mse_values.append(mse)
        r2_values.append(r2)
        models.append(model)

    return predict_weights_values, r2_values, models


def regression_everything(data, cheat=False, verbose=False, pretrained=False):

    """

    This function tries to estimate Align(S,A) by first estimating Succ(S,A) to later estimate Align(S,S').

    :param data:
    :return:
    """

    state_actions, state, state_prima, Y = preprocess_everything(data)

    #for i in range(10):
    #    print(data[i])
    #    print("Age before: ", state_actions[i][0], "Age after:", state_prima[i][0])
    #    print()

    if verbose:
        print("Computing Align(S,S)!")
    _, r_2v, alignSS, = regression_alignmentSS(data, None, cheat=cheat, pretrained=pretrained)

    if verbose:
        print("AlignSS R2 for each value was : ", r_2v)

        print("Computing Succ(S,A)!")
    _, r2_c, succSA = regression_post_state(data, cheat=cheat)

    if verbose:
        for a in range(n_actions):
            print("Succ R2 at Action ", a, " was :", r2_c[a][1:-3], np.mean(r2_c[a][1:-3]))

    r2_values = list()


    "Load data in correct format:"


    X_prima_test = list()


    "We create Succesor State "
    for i in range(len(state_actions)):

        X_prima_test.append(list())

        "We look for which action was applied, otherwise succesor to compare won't make sense"
        act = state_actions[i][-1]

        "We change the format of state-action"
        state_actions[i][-1] = 1

        for c in range(n_criteria):
            #print(int(act), c, state_actions[i])
            c_prima = succSA[int(act)][c].predict([state_actions[i]])

            X_prima_test[i].append(c_prima[0])

            #if c == 0:
            #    print("Original:", state_actions[i][0], " Predicted : ", c_prima[0], "Real : ", state_prima[i][0])
        #print("----", X_prima_test[i][:5], state_prima[i][:5], "ooooo")

    r2 = r2_score(X_prima_test, state_prima)

    " Now we compute the delta between states"
    delta_hat = X_prima_test - state
    if verbose:
        print("How good it was at predicting next state : ", r2)
    for v in range(n_consequence_values):

        "And feed this delta to compute the alignment"
        y_pred = alignSS[v].predict(delta_hat)

        r2 = r2_score(Y[:,v], y_pred)

        #mse_values.append(mse)
        r2_values.append(r2)

    return r2_values


def evaluate_learned_consequence_based_values(train_data, test_data, stored_results, cheat=False, verbose=True, pretrained=False):

    predicted_v, r2_v, _ = regression_alignmentSS(train_data, test_data, cheat=cheat, pretrained=pretrained)

    if verbose:
        print("Beneficence weights : ", predicted_v[BENEFICENCE])
        print("Real weights : ", beneficence_weights)
        print("R2 benef alignSS: ", r2_v[BENEFICENCE])

    stored_results[BENEFICENCE].append(r2_v[BENEFICENCE])

    if verbose:
        print("Non-maleficence weights : ", predicted_v[NONMALEFICENCE])
        print("Real weights : ", nonmaleficence_weights)
        print("R2 nonmalef alignSS : ", r2_v[NONMALEFICENCE])
    stored_results[NONMALEFICENCE].append(r2_v[NONMALEFICENCE])


def evaluate_learned_duty_based_values(train_data, test_data, stored_results, cheat=False, verbose=True, pretrained=False):

    predicted_v, r2_v, _ = regression_alignmentSA_multi_actions(train_data, test_data, cheat=cheat, pretrained=pretrained)


    if verbose:
        print("Justice weights : ", predicted_v[0])
        print("Real weights : ", justice_weights)
        print("R2 justice alignSA : ", r2_v[0])

    stored_results[JUSTICE].append(r2_v[0])

    if verbose:
        print("Autonomy weights : ", predicted_v[1])
        print("Real weights : ", autonomy_weights)
        print("R2 autonomy alignSA: ", r2_v[1])
    stored_results[AUTONOMY].append(r2_v[1])


def evaluate_learned_values_SAS(verbose=False, pretrained=False):
    dataset = [np.loadtxt('train_data.csv', delimiter=',', skiprows=1)][0]
    test_set = [np.loadtxt('test_data.csv', delimiter=',', skiprows=1)][0]
    all_r2 = [list() for _ in range(n_values)]

    v_names = ["Beneficence", "Non-maleficence", "Autonomy", "Justice"]
    if verbose:
        print("Number of train patients : ", len(dataset))
        print("Number of test patients : ", len(test_set))

    if verbose:
        print("---Let's predict now Align(S,A,S')=Align(S,S') for consequence based values!---")
    else:
        print("---Let's evaluate how good is our model at predicting Align(S,A,S') for all four values!---")

    evaluate_learned_consequence_based_values(dataset, test_set, all_r2, verbose=verbose, pretrained=pretrained)

    if verbose:
        print("---Let's predict now Align(S,A,S')=Align(S,A) for duty based values!---")
    evaluate_learned_duty_based_values(dataset, test_set, all_r2, verbose=verbose, pretrained=pretrained)

    print("---R2 results for all four values (1 best, 0 worst)---")
    for i in range(len(all_r2)):
        print("Value ", v_names[i], " : ", np.mean(all_r2[i]), " + ", 3 * np.std(all_r2[i]))


def evaluate_learned_values_SA(verbose=False):
    dataset = [np.loadtxt('train_data.csv', delimiter=',', skiprows=1)][0]
    test_set = [np.loadtxt('test_data.csv', delimiter=',', skiprows=1)][0]
    v_names = ["Beneficence", "Non-maleficence", "Autonomy", "Justice"]

    print("Let's predict Align(S,A) = Align(S,Succ(S,A)) for all values!")

    r2_v = regression_everything(dataset, cheat=False)

    if verbose:
        print(" R2 at predicting Align(S,A) with succ in the middle :", r2_v)

        print("---Let's predict now Align(S,A) for all values!---")

    predicted_v, r2_v, _ = regression_alignmentSA(dataset, test_set, cheat=False)

    r2_v = np.array(r2_v)

    print("---R2 results for all four values (1 best, 0 worst)---")

    if verbose:
        for a in range(n_actions):
            print("Action ", a, ". R2 :", r2_v[a])

    for v in range(n_values):
        print("Value ", v_names[v], " : ", np.mean(r2_v[:,v]), " + ", 3 * np.std(r2_v[:,v]))


def example_prediction(verbose=False):

    models = list()
    v_names = ["Beneficence", "Non-maleficence", "Autonomy", "Justice"]


    if verbose:
        print("---LOADING MODELS---")

    for value in [BENEFICENCE, NONMALEFICENCE]:
        with open('model_SS' + str(value) + '.pkl', 'rb') as f:
            models.append(pickle.load(f))

    for value in [JUSTICE, AUTONOMY]:
        with open('model_SA_' + str(value) + '.pkl', 'rb') as f:
            models.append(pickle.load(f))

    test_set = [np.loadtxt('test_data.csv', delimiter=',', skiprows=1)][0]


    print("Next, we select an example patient.")
    patient_n = 35

    nit = test_set[patient_n][0]
    criteria = test_set[patient_n][1:n_criteria + 1]
    actions = test_set[patient_n][n_criteria + 1:n_criteria + 1 + n_actions]
    postcriteria = test_set[patient_n][n_criteria + 1 + n_actions:2 * n_criteria + 1 + n_actions]
    useful_actions = test_set[patient_n][2 * n_criteria + 1 + n_actions:]

    if verbose:
        print("--We briefly describe them.--")
        print("NIT level : ", nit)
        print("Initial state : ", criteria)
        print("Was curative surgery applied : ", actions[Actions.curative_surgery])
        print("Final state : ", postcriteria)

    for v in range(n_values):
        print(v_names[v] , " Alignment of treatment received by this patient: ", useful_actions[v])

    print()

    print("--Now, the predicted alignment of the treatment received according to our learned model.")

    preds = [0 for _ in range(n_values)]

    test_data = [test_set[patient_n]]
    for v in range(n_values):

        if v < n_consequence_values:
            X_y_test = (preprocess_consequentialist(test_data, None, all_values=True))
            X_test = X_y_test[:, :-n_values]
        else:
            X_y_test = preprocess_actions(test_data, v)
            X_test = X_y_test[:, :-1]

        preds[v] = models[v].predict(X_test)

    for v in range(n_values):
        print("Predicted alignment for value ", v_names[v], " : ", preds[v])


if __name__ == "__main__":

    """
    
    The following code evaluates the learned value semantics against a test set of 2000 synthetically created patients.
    
    Value semantics were learned from a training data set of other 2000 synthetic patients
    
    For the training data set, four placeholder value semantics were used as the ground truth (they are provisional)
    
    If you wish to re-trained the models, set "pretrained=False" in the next line
    
    """

    evaluate_learned_values_SAS(pretrained=True)

    """

    The following code illustrates how the learned value semantics provide feedback for a synthetically created patient.

    """

    print()

    example_prediction()