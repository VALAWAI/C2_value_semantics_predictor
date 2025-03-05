from Actions import Action
import Actions
from Alignment import align_values
from Criteria import Criteria, improve_criteria, worsen_criteria

import random as rd

class Patient:

    def __init__(self, criteria=None, nit=None):

        if criteria is not None:
            self.__conditions = criteria
        else:
            self.__conditions = Criteria(random=True)
        if nit is not None:
            self.__nit = nit
        else:
            self.__nit = rd.randint(1, 5)

        self.__initial_state = self.__conditions.list()
        self.__normalized_initial_state = self.__conditions.get_criteria_normalized()
        self.__treatment = None
        self.__alignment = [0.0, 0.0, 0.0, 0.0]

    def list(self):
        return [self.__nit] + self.__conditions.list()

    def set_treatment(self, action):
        self.__treatment = action

    def get_raw_criteria(self):
        return self.__conditions
    def get_patient_state(self):
        return self.__conditions.list()

    def get_normalized_state(self):
        return self.__conditions.get_criteria_normalized()

    def get_alignment(self):
        return self.__alignment

    def get_state_names(self,nit=True):

        if nit:
            return ["NIT"] + self.__conditions.get_criteria_names()
        else:
            return self.__conditions.get_criteria_names()


    def check_nit_coherence(self):
        action_nit = self.__treatment.compute_nit()

        if action_nit < self.__nit:
            return False
        else:
            return True

    def apply_rcp(self, random=True):

        age, ccd, maca, exp_survival, frailty, crg, ns, barthel, lawton, adv_directives, cogn_deterioration, \
            emotional, discomfort, auto1, auto2, auto3 = self.__conditions.list()

        if random:
            if rd.randint(0, 100) < 90:
                maca = improve_criteria(maca=maca)
            if rd.randint(0, 100) < 85:
                frailty = worsen_criteria(frailty=frailty)
            if rd.randint(0, 100) < 80:
                crg = improve_criteria(crg=crg)

            discomfort = worsen_criteria(discomfort=discomfort)
            if age > 80:
                if rd.randint(0, 100) < 90:
                    discomfort = worsen_criteria(discomfort=discomfort)

            if cogn_deterioration < 3:

                if rd.randint(0, 100) < 70:
                    barthel = improve_criteria(barthel=barthel)
                else:
                    barthel = worsen_criteria(barthel=barthel)
                if rd.randint(0, 100) < 70:
                    lawton = improve_criteria(lawton=lawton)


        else:
            maca = improve_criteria(maca=maca)
            frailty = worsen_criteria(frailty=frailty)
            crg = improve_criteria(crg=crg)

            if age > 80:
                discomfort = worsen_criteria(discomfort=discomfort)
                discomfort = worsen_criteria(discomfort=discomfort)
            else:
                discomfort = worsen_criteria(discomfort=discomfort)

            if cogn_deterioration < 3:
                barthel = improve_criteria(barthel=barthel)
                lawton = improve_criteria(lawton=lawton)

        self.__conditions.modify_criteria(age=age, ccd=ccd, maca=maca, exp_survival=exp_survival,
                                          frailty=frailty, crg=crg, ns=ns, barthel=barthel, lawton=lawton,
                                          adv_directives=adv_directives, cogn_deterioration=cogn_deterioration,
                                          emotional=emotional, discomfort=discomfort)

    def apply_transplant(self, random=True):

        age, ccd, maca, exp_survival, frailty, crg, ns, barthel, lawton, adv_directives, cogn_deterioration, \
            emotional, discomfort, auto1, auto2, auto3 = self.__conditions.list()


        if random:
            if rd.randint(0, 100) < 95:
                exp_survival = improve_criteria(exp_survival=exp_survival)
            if rd.randint(0, 100) < 70:
                frailty = worsen_criteria(frailty=frailty)
            if rd.randint(0, 100) < 75:
                crg = worsen_criteria(crg)

            discomfort = worsen_criteria(discomfort=discomfort)
            if age > 70:
                if rd.randint(0, 100) < 90:
                    discomfort = worsen_criteria(discomfort=discomfort)

            if cogn_deterioration < 3:

                if rd.randint(0, 100) < 80:
                    barthel = improve_criteria(barthel=barthel)
                else:
                    barthel = worsen_criteria(barthel=barthel)
                if rd.randint(0, 100) < 80:
                    lawton = improve_criteria(lawton=lawton)


        else:
            exp_survival = improve_criteria(exp_survival=exp_survival)
            frailty = worsen_criteria(frailty=frailty)
            crg = worsen_criteria(crg)

            if age > 70:
                discomfort = worsen_criteria(discomfort=discomfort)
                discomfort = worsen_criteria(discomfort=discomfort)
            else:
                discomfort = worsen_criteria(discomfort=discomfort)

            if cogn_deterioration < 3:
                barthel = improve_criteria(barthel=barthel)
                lawton = improve_criteria(lawton=lawton)

        self.__conditions.modify_criteria(age=age, ccd=ccd, maca=maca, exp_survival=exp_survival,
                                          frailty=frailty, crg=crg, ns=ns, barthel=barthel, lawton=lawton,
                                          adv_directives=adv_directives, cogn_deterioration=cogn_deterioration,
                                          emotional=emotional, discomfort=discomfort)

        """TO DO: Right now we are comparing initial state instead of future state without actions"""


    def apply_icu(self, random=True):

        age, ccd, maca, exp_survival, frailty, crg, ns, barthel, lawton, adv_directives, cogn_deterioration, \
            emotional, discomfort, auto1, auto2, auto3 = self.__conditions.list()



        if random:

            if rd.randint(0, 100) < 85:
                exp_survival = improve_criteria(exp_survival=exp_survival)
                frailty = worsen_criteria(frailty=frailty)

            if age < 60:
                if rd.randint(0, 100) < 60:
                    discomfort = worsen_criteria(discomfort=discomfort)
            elif age >= 60:
                if rd.randint(0, 100) < 90:
                    discomfort = worsen_criteria(discomfort=discomfort)

            if rd.randint(0, 100) < 40:
                cogn_deterioration = worsen_criteria(cogn_deterioration=cogn_deterioration)

            if cogn_deterioration < 3:

                if rd.randint(0, 100) < 40:
                    barthel = worsen_criteria(barthel=barthel)
                if rd.randint(0, 100) < 70:
                    lawton = improve_criteria(lawton=lawton)

        else:

            exp_survival = improve_criteria(exp_survival=exp_survival)

            frailty = worsen_criteria(frailty=frailty)

            discomfort = worsen_criteria(discomfort=discomfort)

            if cogn_deterioration < 3:
                barthel = worsen_criteria(barthel=barthel)
                lawton = improve_criteria(lawton=lawton)
        self.__conditions.modify_criteria(age=age, ccd=ccd, maca=maca, exp_survival=exp_survival,
                                          frailty=frailty, crg=crg, ns=ns, barthel=barthel, lawton=lawton,
                                          adv_directives=adv_directives, cogn_deterioration=cogn_deterioration,
                                          emotional=emotional, discomfort=discomfort)

        """TO DO: Right now we are comparing initial state instead of future state without actions"""

    def apply_vmni(self, random=True):

        age, ccd, maca, exp_survival, frailty, crg, ns, barthel, lawton, adv_directives, cogn_deterioration, \
            emotional, discomfort, auto1, auto2, auto3 = self.__conditions.list()
        if random:
            if rd.randint(0, 100) < 60:
                exp_survival = improve_criteria(exp_survival=exp_survival)

            if cogn_deterioration < 3:
                if rd.randint(0, 100) < 75:
                    discomfort = worsen_criteria(discomfort=discomfort)

            if rd.randint(0, 100) < 70:
                frailty = worsen_criteria(frailty=frailty)


        else:
            exp_survival = improve_criteria(exp_survival=exp_survival)
            if cogn_deterioration < 3:
                discomfort = worsen_criteria(discomfort=discomfort)
            frailty = worsen_criteria(frailty=frailty)

        self.__conditions.modify_criteria(age=age, ccd=ccd, maca=maca, exp_survival=exp_survival,
                                          frailty=frailty, crg=crg, ns=ns, barthel=barthel, lawton=lawton,
                                          adv_directives=adv_directives, cogn_deterioration=cogn_deterioration,
                                          emotional=emotional, discomfort=discomfort)



    def apply_dva(self, random=True):

        age, ccd, maca, exp_survival, frailty, crg, ns, barthel, lawton, adv_directives, cogn_deterioration, \
            emotional, discomfort, auto1, auto2, auto3 = self.__conditions.list()


        if random:

            if rd.randint(0, 100) < 55:
                exp_survival = improve_criteria(exp_survival=exp_survival)

            if rd.randint(0, 100) < 30:
                frailty = worsen_criteria(frailty=frailty)

            if age < 60:
                if rd.randint(0, 100) < 75:
                    discomfort = worsen_criteria(discomfort=discomfort)
            elif age >= 60:
                if rd.randint(0, 100) < 95:
                    discomfort = worsen_criteria(discomfort=discomfort)

        else:
            exp_survival = improve_criteria(exp_survival=exp_survival)

            discomfort = worsen_criteria(discomfort=discomfort)

        self.__conditions.modify_criteria(age=age, ccd=ccd, maca=maca, exp_survival=exp_survival,
                                          frailty=frailty, crg=crg, ns=ns, barthel=barthel, lawton=lawton,
                                          adv_directives=adv_directives, cogn_deterioration=cogn_deterioration,
                                          emotional=emotional, discomfort=discomfort)

        """TO DO: Right now we are comparing initial state instead of future state without actions"""


    def apply_dialysis(self, random=True):

        age, ccd, maca, exp_survival, frailty, crg, ns, barthel, lawton, adv_directives, cogn_deterioration, \
            emotional, discomfort, auto1, auto2, auto3 = self.__conditions.list()

        if random:

            if rd.randint(0, 100) < 60:
                crg = improve_criteria(crg=crg)

            if age < 70:
                if rd.randint(0, 100) < 65:
                    discomfort = worsen_criteria(discomfort=discomfort)
            elif age >= 70:
                if rd.randint(0, 100) < 85:
                    discomfort = worsen_criteria(discomfort=discomfort)

        else:
            crg = improve_criteria(crg=crg)

            discomfort = worsen_criteria(discomfort=discomfort)

        self.__conditions.modify_criteria(age=age, ccd=ccd, maca=maca, exp_survival=exp_survival,
                                          frailty=frailty, crg=crg, ns=ns, barthel=barthel, lawton=lawton,
                                          adv_directives=adv_directives, cogn_deterioration=cogn_deterioration,
                                          emotional=emotional, discomfort=discomfort)


    def apply_analysis(self, random=True):

        age, ccd, maca, exp_survival, frailty, crg, ns, barthel, lawton, adv_directives, cogn_deterioration, \
            emotional, discomfort, auto1, auto2, auto3 = self.__conditions.list()

        if random:

            if rd.randint(0, 100) < 60:
                lawton = improve_criteria(lawton=lawton)
                barthel = improve_criteria(barthel=barthel)

            if rd.randint(0, 100) < 30:
                    discomfort = worsen_criteria(discomfort=discomfort)


        else:
            lawton = improve_criteria(lawton=lawton)
            barthel = improve_criteria(barthel=barthel)

        self.__conditions.modify_criteria(age=age, ccd=ccd, maca=maca, exp_survival=exp_survival,
                                          frailty=frailty, crg=crg, ns=ns, barthel=barthel, lawton=lawton,
                                          adv_directives=adv_directives, cogn_deterioration=cogn_deterioration,
                                          emotional=emotional, discomfort=discomfort)

    def apply_mild_action(self, random=True):

        age, ccd, maca, exp_survival, frailty, crg, ns, barthel, lawton, adv_directives, cogn_deterioration, \
            emotional, discomfort, auto1, auto2, auto3 = self.__conditions.list()
        if random:

            if cogn_deterioration < 3:
                if rd.randint(0, 100) < 70:
                    lawton = improve_criteria(lawton=lawton)
                    barthel = improve_criteria(barthel=barthel)

            if rd.randint(0, 100) < 55:
                discomfort = worsen_criteria(discomfort=discomfort)


        else:

            if cogn_deterioration < 3:
                lawton = improve_criteria(lawton=lawton)
                barthel = improve_criteria(barthel=barthel)

            discomfort = worsen_criteria(discomfort=discomfort)

        self.__conditions.modify_criteria(age=age, ccd=ccd, maca=maca, exp_survival=exp_survival,
                                          frailty=frailty, crg=crg, ns=ns, barthel=barthel, lawton=lawton,
                                          adv_directives=adv_directives, cogn_deterioration=cogn_deterioration,
                                          emotional=emotional, discomfort=discomfort)


    def apply_adv_action(self, random=True):

        age, ccd, maca, exp_survival, frailty, crg, ns, barthel, lawton, adv_directives, cogn_deterioration, \
            emotional, discomfort, auto1, auto2, auto3 = self.__conditions.list()

        if random:

            if age < 80 and frailty < 2:
                if rd.randint(0, 100) < 90:
                    exp_survival = improve_criteria(exp_survival=exp_survival)

            if cogn_deterioration < 3:
                if rd.randint(0, 100) < 90:
                    lawton = improve_criteria(lawton=lawton)
                    barthel = improve_criteria(barthel=barthel)

            if rd.randint(0, 100) < 85:
                discomfort = worsen_criteria(discomfort=discomfort)
            if rd.randint(0, 100) < 55:
                discomfort = worsen_criteria(discomfort=discomfort)

            if rd.randint(0, 100) < 85:
                emotional = worsen_criteria(emotional=emotional)



        else:

            if age < 80 and frailty < 2:
                exp_survival = improve_criteria(exp_survival=exp_survival)

            if cogn_deterioration < 3:
                lawton = improve_criteria(lawton=lawton)
                barthel = improve_criteria(barthel=barthel)

            discomfort = worsen_criteria(discomfort=discomfort)
            discomfort = worsen_criteria(discomfort=discomfort)

            emotional = worsen_criteria(emotional=emotional)

        self.__conditions.modify_criteria(age=age, ccd=ccd, maca=maca, exp_survival=exp_survival,
                                          frailty=frailty, crg=crg, ns=ns, barthel=barthel, lawton=lawton,
                                          adv_directives=adv_directives, cogn_deterioration=cogn_deterioration,
                                          emotional=emotional, discomfort=discomfort)




    def apply_palliative_surgery(self, random=True):

        age, ccd, maca, exp_survival, frailty, crg, ns, barthel, lawton, adv_directives, cogn_deterioration, \
            emotional, discomfort, auto1, auto2, auto3 = self.__conditions.list()

        if random:

            if cogn_deterioration < 3:
                if rd.randint(0, 100) < 10:
                    lawton = worsen_criteria(lawton=lawton)
                    barthel = worsen_criteria(barthel=barthel)

            if rd.randint(0, 100) < 85:
                discomfort = improve_criteria(discomfort=discomfort)

            if rd.randint(0, 100) < 80:
                emotional = improve_criteria(emotional=emotional)


        else:

            discomfort = improve_criteria(discomfort=discomfort)
            emotional = improve_criteria(emotional=emotional)

        self.__conditions.modify_criteria(age=age, ccd=ccd, maca=maca, exp_survival=exp_survival,
                                          frailty=frailty, crg=crg, ns=ns, barthel=barthel, lawton=lawton,
                                          adv_directives=adv_directives, cogn_deterioration=cogn_deterioration,
                                          emotional=emotional, discomfort=discomfort)

        """TO DO: Right now we are comparing initial state instead of future state without actions"""


    def apply_curative_surgery(self, random=True):

        age, ccd, maca, exp_survival, frailty, crg, ns, barthel, lawton, adv_directives, cogn_deterioration, \
            emotional, discomfort, auto1, auto2, auto3 = self.__conditions.list()
        age, ccd, maca, exp_survival, frailty, crg, ns, barthel, lawton, adv_directives, cogn_deterioration, \
            emotional, discomfort, auto1, auto2, auto3 = self.__conditions.list()

        if random:

            if age < 80:
                if cogn_deterioration < 3:
                    if rd.randint(0, 100) < 80:
                        lawton = improve_criteria(lawton=lawton)
                        barthel = improve_criteria(barthel=barthel)

                if rd.randint(0, 100) < 70:
                    frailty = improve_criteria(frailty=frailty)

                if rd.randint(0, 100) < 75:
                    crg = improve_criteria(crg=crg)
            else:
                if rd.randint(0, 100) < 30:
                    frailty = improve_criteria(frailty=frailty)

                if rd.randint(0, 100) < 55:
                    crg = improve_criteria(crg=crg)

            if age < 60:
                if rd.randint(0, 100) < 70:
                    discomfort = worsen_criteria(discomfort=discomfort)
            else:
                if rd.randint(0, 100) < 90:
                    discomfort = worsen_criteria(discomfort=discomfort)
                    emotional = worsen_criteria(emotional=emotional)



        else:

            if age < 80:
                if cogn_deterioration < 3:
                    lawton = improve_criteria(lawton=lawton)
                    barthel = improve_criteria(barthel=barthel)

                frailty = improve_criteria(frailty=frailty)

                crg = improve_criteria(crg=crg)
            else:
                crg = improve_criteria(crg=crg)

            if age < 60:
                discomfort = worsen_criteria(discomfort=discomfort)
            else:
                discomfort = worsen_criteria(discomfort=discomfort)
                emotional = worsen_criteria(emotional=emotional)

        self.__conditions.modify_criteria(age=age, ccd=ccd, maca=maca, exp_survival=exp_survival,
                                          frailty=frailty, crg=crg, ns=ns, barthel=barthel, lawton=lawton,
                                          adv_directives=adv_directives, cogn_deterioration=cogn_deterioration,
                                          emotional=emotional, discomfort=discomfort)

        """TO DO: Right now we are comparing initial state instead of future state without actions"""


    def apply_treatment(self, random=True):

        if isinstance(self.__treatment, int):
            rcp = self.__treatment == Actions.rcp
            transplant = self.__treatment == Actions.transplant
            icu = self.__treatment == Actions.icu
            vmni = self.__treatment == Actions.vmni
            dva = self.__treatment == Actions.dva
            dialysis = self.__treatment == Actions.dialysis
            analysis = self.__treatment == Actions.analysis
            mild_action = self.__treatment == Actions.mild_action
            adv_action = self.__treatment == Actions.adv_action
            palliative_surgery = self.__treatment == Actions.palliative_surgery
            curative_surgery = self.__treatment == Actions.curative_surgery

        else:
            rcp, transplant, icu, vmni, dva, dialysis, analysis, mild_action, adv_action, palliative_surgery, curative_surgery = self.__treatment.list()

        if rcp:
            self.apply_rcp(random=random)
        if transplant:
            self.apply_transplant(random=random)
        if icu:
            self.apply_icu(random=random)
        if vmni:
            self.apply_vmni(random=random)
        if dva:
            self.apply_dva(random=random)
        if dialysis:
            self.apply_dialysis(random=random)
        if analysis:
            self.apply_analysis(random=random)
        if mild_action:
            self.apply_mild_action(random=random)
        if adv_action:
            self.apply_adv_action(random=random)
        if palliative_surgery:
            self.apply_palliative_surgery(random=random)
        if curative_surgery:
            self.apply_curative_surgery(random=random)

        self.__treatment = None

    def set_and_apply_treatment(self, action, random=False):
        self.set_treatment(action)

        #print("NIT respected : ", self.check_nit_coherence())

        self.apply_treatment(random=random)


        self.__alignment = align_values(self.__normalized_initial_state, action, self.__conditions.get_criteria_normalized())

        return self.get_patient_state()


NB_CRITERIA = Patient().get_raw_criteria().n_criteria

if __name__ == "__main__":

    """
    patient1 = Patient()
    action1 = Action(icu=True)
    patient_state = patient1.get_patient_state()
    print(patient_state)

    patient1.set_and_apply_treatment(action1, random=False)
    patient_state = patient1.get_patient_state()

    print(patient_state)
    
    """



