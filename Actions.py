import random as rd

rcp = 0
transplant = 1
icu = 2
vmni = 3
dva = 4
dialysis = 5
analysis = 6
mild_action = 7
adv_action = 8
palliative_surgery = 9
curative_surgery = 10

NB_POSSIBLE_ACTIONS = len([rcp, transplant, icu, vmni, dva, dialysis, analysis, mild_action, adv_action, palliative_surgery, curative_surgery])

class Action:
    minimum_actions = 1


    def __init__(self, *args, random=False, maximum_actions=4, rcp=False, transplant=False, icu=False, vmni=False, dva=False, dialysis=False,
                 analysis=False, mild_action=False, adv_action=False, palliative_surgery=False,
                 curative_surgery=False):

        if random:
            nb_applied_actions = rd.randint(Action.minimum_actions, maximum_actions)
            applied_actions = [0 for _ in range(NB_POSSIBLE_ACTIONS)]

            while sum(applied_actions) < nb_applied_actions:
                idx = rd.randint(0, 10)
                applied_actions[idx] = 1

            self.__cardiopulmonary_resuscitation = applied_actions[0]  # NIT 1
            self.__transplant = applied_actions[1]  # NIT 3
            self.__intensive_care_unit = applied_actions[2]  # NIT 2A
            self.__non_invasive_respiratory_support = applied_actions[3]  # NIT 5
            self.__vasoactive_drugs = applied_actions[4]  # NIT 5

            self.__dialysis = applied_actions[5]  # NIT 5
            self.__analysis = applied_actions[6]  # NIT 4
            self.__mild_action = applied_actions[7]  # NIT 5
            self.__adv_action = applied_actions[8]  # NIT 3
            self.__palliative_surgery = applied_actions[9]  # NIT 5

            self.__curative_surgery = applied_actions[10]  # NIT 3
        elif len(args) > 0:

            applied_actions = args[0]

            self.__cardiopulmonary_resuscitation = applied_actions[0]  # NIT 1
            self.__transplant = applied_actions[1]  # NIT 3
            self.__intensive_care_unit = applied_actions[2]  # NIT 2A
            self.__non_invasive_respiratory_support = applied_actions[3]  # NIT 5
            self.__vasoactive_drugs = applied_actions[4]  # NIT 5

            self.__dialysis = applied_actions[5]  # NIT 5
            self.__analysis = applied_actions[6]  # NIT 4
            self.__mild_action = applied_actions[7]  # NIT 5
            self.__adv_action = applied_actions[8]  # NIT 3
            self.__palliative_surgery = applied_actions[9]  # NIT 5

            self.__curative_surgery = applied_actions[10]  # NIT 3

        else:
            # All actions are boolean, either taken or not taken
            # No action for NIT 2B!!
            self.__cardiopulmonary_resuscitation = rcp  # NIT 1
            self.__transplant = transplant  # NIT 3
            self.__intensive_care_unit = icu  # NIT 2A
            self.__non_invasive_respiratory_support = vmni  # NIT 5
            self.__vasoactive_drugs = dva  # NIT 5
            self.__dialysis = dialysis  # NIT 5
            self.__analysis = analysis  # NIT 4
            self.__mild_action = mild_action  # NIT 5
            self.__adv_action = adv_action  # NIT 3
            self.__palliative_surgery = palliative_surgery  # NIT 5
            self.__curative_surgery = curative_surgery  # NIT 3

    def partial_list(self, rcp=False, transplant=False, icu=False, vmni=False, dva=False,
                     dialysis=False, analysis=False, mild_action=False, adv_action=False,
                     palliative_surgery=False, curative_surgery=False):

        action_list = []

        if rcp:
            action_list.append(self.__cardiopulmonary_resuscitation)
        if transplant:
            action_list.append(self.__transplant)
        if icu:
            action_list.append(self.__intensive_care_unit)
        if vmni:
            action_list.append(self.__non_invasive_respiratory_support)
        if dva:
            action_list.append(self.__vasoactive_drugs)
        if dialysis:
            action_list.append(self.__dialysis)
        if analysis:
            action_list.append(self.__analysis)
        if mild_action:
            action_list.append(self.__mild_action)
        if adv_action:
            action_list.append(self.__adv_action)
        if curative_surgery:
            action_list.append(self.__curative_surgery)
        if palliative_surgery:
            action_list.append(self.__palliative_surgery)

        if len(action_list) == 0:
            return "Error!! Empty list!!"
        elif len(action_list) == 1:
            return action_list[0]
        else:
            return action_list

    def compute_nit(self):

        if self.__cardiopulmonary_resuscitation:
            nit = 1
        elif self.__intensive_care_unit:
            nit = 2
        elif self.__transplant or self.__adv_action or self.__curative_surgery:
            nit = 3
        elif self.__analysis:
            nit = 4
        else:
            nit = 5

        return nit

    def list(self):
        return [self.__cardiopulmonary_resuscitation,
                self.__transplant,
                self.__intensive_care_unit,
                self.__non_invasive_respiratory_support,
                self.__vasoactive_drugs,
                self.__dialysis,
                self.__analysis,
                self.__mild_action,
                self.__adv_action,
                self.__palliative_surgery,
                self.__curative_surgery
                ]

    def get_names(self, useful=False):

        action_names = ["cardiopulmonary_resuscitation",
                "transplant",
                "intensive_care_unit",
                "non_invasive_respiratory_support",
                "vasoactive_drugs",
                "dialysis",
                "analysis",
                "mild_action",
                "adv_action",
                "palliative_surgery",
                "curative_surgery"
                ]

        if useful:
            for i in range(len(action_names)):
                action_names[i] = "useful " + action_names[i]

        return action_names


if __name__ == "__main__":

    for _ in range(10):
        a = Action(random=True)

        print(a.list())
