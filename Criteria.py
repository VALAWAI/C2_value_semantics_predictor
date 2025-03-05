import random as rd
import numpy as np


def distance_patients(patient1, patient2):

    v1 = np.array(patient1.get_criteria_normalized())
    v2 = np.array(patient2.get_criteria_normalized())

    return np.linalg.norm(v1-v2)


def create_similar_patient(patient, changes=0, strength_changes=1, mod_autonomy=False):

    modify_criteria = list()

    criteria_counter = patient.n_criteria if mod_autonomy else patient.n_criteria - patient.n_autonomy_criteria

    if changes > criteria_counter:
        changes = criteria_counter

    for _ in range(criteria_counter):
        modify_criteria.append(None)

    increase_criteria = modify_criteria.copy()
    decrease_criteria = modify_criteria.copy()

    for i in range(changes):
        modify_criteria[i] = True

    rd.shuffle(modify_criteria)

    #print("Mod : ", modify_criteria)

    for i in range(len(modify_criteria)):
        if modify_criteria[i] is not None:
            if rd.randint(0, 1):
                increase_criteria[i] = True
            else:
                decrease_criteria[i] = True

    if not mod_autonomy:
        for _ in range(patient.n_autonomy_criteria):
            increase_criteria.append(None)
            decrease_criteria.append(None)

    #print("Mod increase : ", increase_criteria)
    #print("Mod decrease : ", decrease_criteria)

    for _ in range(strength_changes):
        patient.increase_criteria_from_list(increase_criteria)
        patient.decrease_criteria_from_list(decrease_criteria)


def force_different_patient(patient, changes=0, strength_changes=1, mod_autonomy=False):

    aux = patient.copy()

    while distance_patients(patient, aux) <= 0:
        create_similar_patient(patient, changes, strength_changes, mod_autonomy)


class Criteria:

    AGE = 0
    CCD = 1
    MACA = 2
    EXP_SURVIVAL = 3
    FRAILTY = 4
    CRG = 5
    NS = 6
    BARTHEL = 7
    LAWTON = 8
    ADV_DIRECTIVES = 9
    COGN_DETERIORATION = 10
    EMOTIONAL = 11
    DISCOMFORT = 12

    AUTONOMY_UNDERSTAND = 13
    AUTONOMY_INFORM = 14
    AUTONOMY_COERCE = 15

    n_criteria = 16

    n_autonomy_criteria = 3

    max_age = 100
    max_frailty = max_discomfort = 2
    max_crg = max_cogn = 3
    max_barthel = 100
    max_lawton = 8
    max_ccd = max_maca = max_ns = max_adv_directives = max_emotional = max_exp_survival = max_auto1 = max_auto2 = max_auto3 = 1

    min_age = 1
    min_exp_survival = min_ccd = min_maca = min_ns = min_adv_directives = min_frailty = min_crg = min_barthel = \
        min_cogn = min_discomfort = min_lawton = min_emotional = min_auto1 = min_auto2 = min_auto3 = 0

    def __init__(self, *args, random=False, age=70, ccd=False, maca=False, exp_survival=False, frailty="Low", crg=0, ns=0,
                 barthel=100, lawton=5, adv_directives=True, sci_support=True, cogn_deterioration="Absent", emotional_distress=True,
                 discomfort="High", auto1=True, auto2=True, auto3=True, dont_normalize=False):

        # Criteria that do not change with actions
        self.__age = 70 # integer, years, between 1 and 140
        self.__complex_chronic_disease = False  # boolean
        self.__expected_survival = False  # boolean, True: >12 months, False: <12 months
        self.__short_term_survival = False  # boolean
        self.__frailty = "Moderate"  # Low, Moderate, or High

        self.__clinical_risk_group = 2  # integer between 0 and 3, both included
        self.__social_support = False  # boolean
        self.__functional_independence = 40  # integer between 0 and 100
        self.__instrumental_independence = 8  # integer between 0 and 8
        self.__advanced_directives = False  # boolean

        self.__cognitive_deterioration = "Absent"  # Absent, Slight, Moderate, Severe
        self.__emotional_distress = True # boolean
        self.__discomfort = "High"  # integer between 0 and 2

        self.__autonomy_understand = True # boolean
        self.__autonomy_inform = True # boolean
        self.__autonomy_coerce = False # boolean


        #self.__decision_taken_with_scientific_support = False   # boolean

        if random and isinstance(random, bool):
            self.random_init()
        elif len(args) == 0:
            self.normal_init(age, ccd, maca, exp_survival, frailty, crg, ns, barthel, lawton, adv_directives,
                        sci_support, cogn_deterioration, emotional_distress, discomfort, auto1, auto2, auto3)
        else:
            if len(args) == 1:
                args = args[0]

            self.list_init(args, dont_normalize)


    def random_init(self):

        # Criteria that do not change with actions
        self.__age = 65 + rd.randint(0, 35)  # integer, years, between 0 and 150
        self.__complex_chronic_disease = rd.randint(Criteria.min_ccd, Criteria.max_ccd)  # boolean
        self.__expected_survival = rd.randint(0, 1)  # boolean
        self.__clinical_risk_group = rd.randint(Criteria.min_crg, Criteria.max_crg)  # integer between 0 and 3, both included
        self.__social_support = rd.randint(0, 1)  # boolean
        self.__advanced_directives = rd.randint(0, 1)  # boolean
        self.__cognitive_deterioration = rd.randint(Criteria.min_cogn, Criteria.max_cogn)  # Absent, Slight, Moderate, Severe

        self.__short_term_survival = rd.randint(0, 1)  # boolean
        self.__frailty = rd.randint(Criteria.min_frailty, Criteria.max_frailty)  # Low, Moderate, or High
        self.__functional_independence = rd.randint(Criteria.min_barthel, Criteria.max_barthel)  # integer between 0 and 100
        self.__instrumental_independence = rd.randint(Criteria.min_lawton, Criteria.max_lawton) # integer between 0 and 8
        #self.__decision_taken_with_scientific_support = rd.randint(0, 1)   # boolean
        self.__emotional_distress = rd.randint(0, 1) # boolean
        self.__discomfort = rd.randint(Criteria.min_discomfort, Criteria.max_discomfort)  # integer between 0 and 100

        self.__autonomy_understand = rd.randint(0, 1) # boolean
        self.__autonomy_inform = rd.randint(0, 1)  # boolean
        self.__autonomy_coerce = rd.randint(0, 1)  # boolean

        if self.there_are_inconsistencies():
            print("Something's wrong with this patient!")


    def list_init(self, args, dont_normalize):
        # Criteria that do not change with actions


        self.__age = args[Criteria.AGE]  # integer, years, between 0 and 150
        self.__complex_chronic_disease = args[Criteria.CCD]  # boolean
        self.__short_term_survival = args[Criteria.MACA]  # boolean
        self.__expected_survival = args[Criteria.EXP_SURVIVAL]  # boolean
        self.__frailty = args[Criteria.FRAILTY]  # Low, Moderate, or High

        self.__clinical_risk_group = args[Criteria.CRG]  # integer between 0 and 3, both included
        self.__social_support = args[Criteria.NS]  # boolean
        self.__functional_independence = args[Criteria.BARTHEL]  # integer between 0 and 100
        self.__instrumental_independence = args[Criteria.LAWTON]  # integer between 0 and 8
        self.__advanced_directives = args[Criteria.ADV_DIRECTIVES]  # boolean

        self.__cognitive_deterioration = args[Criteria.COGN_DETERIORATION]  # Absent, Slight, Moderate, Severe
        self.__emotional_distress = args[Criteria.EMOTIONAL] # boolean
        self.__discomfort = args[Criteria.DISCOMFORT]  # integer between 0 and 100

        self.__autonomy_understand = args[Criteria.AUTONOMY_UNDERSTAND] # boolean
        self.__autonomy_inform = args[Criteria.AUTONOMY_INFORM] # boolean
        self.__autonomy_coerce = args[Criteria.AUTONOMY_COERCE] # boolean


        #self.__decision_taken_with_scientific_support = args[10]   # boolean

        if dont_normalize:
            self.un_normalize_criteria()
        else:
            self.transform_to_numbers_all_criteria()

        if self.there_are_inconsistencies():
            print("Something's wrong with this patient!")

    def normal_init(self, age=70, ccd=False, maca=False, exp_survival=False, frailty="Low", crg=0, ns=0,
                 barthel=100, lawton=0, adv_directives=True, sci_support=True, cogn_deterioration="Absent",
                 emotional_distress=True, discomfort="High", auto1=True, auto2=True, auto3=False):

        # Criteria that do not change with actions
        self.__age = age  # integer, years, between 0 and 150
        self.__complex_chronic_disease = ccd  # boolean
        self.__expected_survival = exp_survival  # integer, months: 6<, 6, 6-12, 12, >12

        self.__clinical_risk_group = crg  # integer between 0 and 3, both included
        self.__social_support = ns  # boolean
        self.__advanced_directives = adv_directives  # boolean
        self.__cognitive_deterioration = cogn_deterioration  # Absent, Slight, Moderate, Severe

        # Criteria that change with actions
        self.__short_term_survival = maca  # boolean
        self.__frailty = frailty  # Low, Moderate, or High
        self.__functional_independence = barthel  # integer between 0 and 100

        self.__instrumental_independence = lawton  # integer between 0 and 8
        self.__emotional_distress = emotional_distress # boolean
        self.__discomfort = discomfort  # Low, Moderate, High

        self.__autonomy_understand = auto1  # boolean
        self.__autonomy_inform = auto2  # boolean
        self.__autonomy_coerce = auto3  # boolean




        if self.there_are_inconsistencies():
            print("Something's wrong with this patient!")

        self.transform_to_numbers_all_criteria()

    def get_criteria_names(self):
        return ["age", "ccd", "maca", "exp_survival", "frailty", "crg", "ns", \
            "barthel", "lawton", "adv_directives", "cogn_deterioration", "emotional", "discomfort", "understand", "informed", "coerced"]


    def modify_criteria(self, age=None, ccd=None, maca=None, exp_survival=None, frailty=None, crg=None, ns=None,
                 barthel=None, lawton=None, adv_directives=None, cogn_deterioration=None, emotional=None,
                 discomfort=None, auto1=None, auto2=None, auto3=None):

        # Criteria that do not change with actions
        if age is not None:
            self.__age = age  # integer, years, between 0 and 150
        if ccd is not None:
            self.__complex_chronic_disease = ccd  # boolean
        if maca is not None:
            self.__short_term_survival = maca  # boolean
        if exp_survival is not None:
            self.__expected_survival = exp_survival  # integer, months: 6<, 6, 6-12, 12, >12
        if frailty is not None:
            self.__frailty = frailty  # Low, Moderate, or High
        if crg is not None:
            self.__clinical_risk_group = crg  # integer between 0 and 3, both included
        if ns is not None:
            self.__social_support = ns  # boolean
        if adv_directives is not None:
            self.__advanced_directives = adv_directives  # boolean
        if cogn_deterioration is not None:
            self.__cognitive_deterioration = cogn_deterioration  # Absent, Slight, Moderate, Severe
        if barthel is not None:
            self.__functional_independence = barthel  # integer between 0 and 100
        if lawton is not None:
            self.__instrumental_independence = lawton  # integer between 0 and 8
        #self.__decision_taken_with_scientific_support = sci_support   # boolean
        if emotional is not None:
            self.__emotional_distress = emotional # boolean
        if discomfort is not None:
            self.__discomfort = discomfort  # Low, Moderate, High
        if auto1 is not None:
            self.__autonomy_understand = auto1 # boolean
        if auto2 is not None:
            self.__autonomy_inform = auto2 # boolean
        if auto3 is not None:
            self.__autonomy_coerce = auto3 # boolean

        self.correct_inconsistencies()

    def increase_criteria_from_list(self, list_criteria):

        (age, ccd, maca, exp_survival, frailty, crg, ns, barthel, lawton, adv_directives, cogn_deterioration, emotional,
         discomfort, auto1, auto2, auto3) = list_criteria

        self.increase_criteria(age, ccd, maca, exp_survival, frailty, crg, ns, barthel, lawton, adv_directives, cogn_deterioration, emotional,
         discomfort, auto1, auto2, auto3)

    def increase_criteria(self, age=None, ccd=None, maca=None, exp_survival=None, frailty=None, crg=None, ns=None,
                 barthel=None, lawton=None, adv_directives=None, cogn_deterioration=None, emotional=None,
                 discomfort=None, auto1=None, auto2=None, auto3=None):

        # Criteria that do not change with actions
        if age is not None:
            self.__age += 10  # integer, years, between 0 and 150
        if ccd is not None:
            self.__complex_chronic_disease = True  # boolean
        if maca is not None:
            self.__short_term_survival = True  # boolean
        if exp_survival is not None:
            self.__expected_survival = True  # boolean
        if frailty is not None:
            self.__frailty += 1  # Low, Moderate, or High
        if crg is not None:
            self.__clinical_risk_group += 1  # integer between 0 and 3, both included
        if ns is not None:
            self.__social_support = True  # boolean
        if adv_directives is not None:
            self.__advanced_directives = True  # boolean
        if cogn_deterioration is not None:
            self.__cognitive_deterioration += 1  # Absent, Slight, Moderate, Severe
        if barthel is not None:

            if self.__functional_independence < 21:
                self.__functional_independence = 40
            elif self.__functional_independence < 61:
                self.__functional_independence = 75
            elif self.__functional_independence < 91:
                self.__functional_independence = 95
            else:
                self.__functional_independence = 100

        if lawton is not None:
            self.__instrumental_independence += 1  # integer between 0 and 8
        #self.__decision_taken_with_scientific_support = sci_support   # boolean
        if emotional is not None:
            self.__emotional_distress = True # boolean
        if discomfort is not None:
            self.__discomfort += 1  # Low, Moderate, High
        if auto1 is not None:
            self.__autonomy_understand = True # boolean
        if auto2 is not None:
            self.__autonomy_inform = True # boolean
        if auto3 is not None:
            self.__autonomy_coerce = True # boolean

        self.correct_inconsistencies()

    def decrease_criteria_from_list(self, list_criteria):

        (age, ccd, maca, exp_survival, frailty, crg, ns, barthel, lawton, adv_directives, cogn_deterioration, emotional,
         discomfort, auto1, auto2, auto3) = list_criteria

        self.decrease_criteria(age, ccd, maca, exp_survival, frailty, crg, ns, barthel, lawton, adv_directives,
                               cogn_deterioration, emotional,
                               discomfort, auto1, auto2, auto3)

    def decrease_criteria(self, age=None, ccd=None, maca=None, exp_survival=None, frailty=None, crg=None, ns=None,
                 barthel=None, lawton=None, adv_directives=None, cogn_deterioration=None, emotional=None,
                 discomfort=None, auto1=None, auto2=None, auto3=None):

        # Criteria that do not change with actions
        if age is not None:
            self.__age -= 10  # integer, years, between 0 and 150
        if ccd is not None:
            self.__complex_chronic_disease = False  # boolean
        if maca is not None:
            self.__short_term_survival = False  # boolean
        if exp_survival is not None:
            self.__expected_survival = False  # boolean
        if frailty is not None:
            self.__frailty -= 1  # Low, Moderate, or High
        if crg is not None:
            self.__clinical_risk_group -= 1  # integer between 0 and 3, both included
        if ns is not None:
            self.__social_support = False  # boolean
        if adv_directives is not None:
            self.__advanced_directives = False  # boolean
        if cogn_deterioration is not None:
            self.__cognitive_deterioration -= 1  # Absent, Slight, Moderate, Severe
        if barthel is not None:

            if self.__functional_independence == 100:
                self.__functional_independence = 95
            elif self.__functional_independence > 90:
                self.__functional_independence = 75
            elif self.__functional_independence > 60:
                self.__functional_independence = 40
            else:
                self.__functional_independence = 10

        if lawton is not None:
            self.__instrumental_independence -= 1  # integer between 0 and 8
        #self.__decision_taken_with_scientific_support = sci_support   # boolean
        if emotional is not None:
            self.__emotional_distress = False # boolean
        if discomfort is not None:
            self.__discomfort -= 1  # Low, Moderate, High
        if auto1 is not None:
            self.__autonomy_understand = False # boolean
        if auto2 is not None:
            self.__autonomy_inform = False # boolean
        if auto3 is not None:
            self.__autonomy_coerce = False # boolean

        self.correct_inconsistencies()



    def transform_to_numbers_all_criteria(self):
        """Maybe we could normalize all criteria"""
        self.__short_term_survival = int(self.__short_term_survival)
        self.__complex_chronic_disease = int(self.__complex_chronic_disease)
        self.__social_support = int(self.__social_support)
        self.__advanced_directives = int(self.__advanced_directives)
        self.__emotional_distress = int(self.__emotional_distress)

        self.__autonomy_understand = int(self.__autonomy_understand)
        self.__autonomy_inform = int(self.__autonomy_inform)
        self.__autonomy_coerce = int(self.__autonomy_coerce)

        #self.__decision_taken_with_scientific_support = int(self.__decision_taken_with_scientific_support)

        if self.__discomfort == "Low":
            self.__discomfort = Criteria.min_discomfort
        elif self.__discomfort == "Moderate":
            self.__discomfort = Criteria.min_discomfort + 1
        elif self.__discomfort == "High":
            self.__discomfort = Criteria.max_discomfort

        if Criteria.min_discomfort + 2 != Criteria.max_discomfort:
            print("Discomfort ranges are not consistent!!!")

        if self.__frailty == "Low":
            self.__frailty = Criteria.min_frailty
        elif self.__frailty == "Moderate":
            self.__frailty = Criteria.min_frailty + 1
        elif self.__frailty == "High":
            self.__frailty = Criteria.max_frailty
        #else:
        #    print("Unknown value for frailty:", self.__frailty)

        if Criteria.min_frailty + 2 != Criteria.max_frailty:
            print("Frailty ranges are not consistent!!!")

        if self.__cognitive_deterioration == "Absent":
            self.__cognitive_deterioration = Criteria.min_cogn
        elif self.__cognitive_deterioration == "Slight":
            self.__cognitive_deterioration = Criteria.min_cogn + 1
        elif self.__cognitive_deterioration == "Moderate":
            self.__cognitive_deterioration = Criteria.min_cogn + 2
        elif self.__cognitive_deterioration == "Severe":
            self.__cognitive_deterioration = Criteria.max_cogn

        if Criteria.min_cogn + 3 != Criteria.max_cogn:
            print("Cognitive deterioration ranges are not consistent!!!")


    def un_normalize_criteria(self):
        """

        - All criteria are integers from 0 to 1
        - 1 denotes the best situation
        - 0 denotes the worst situation

        """

        age, ccd, maca, exp_survival, frailty, crg, ns, barthel, lawton, adv_directives, cogn_deterioration, emotional, discomfort, auto1, auto2, auto3 = \
            self.list()

        self.__age = 100.0*age
        #print("Age normalized!", age, self.__age)
        self.__complex_chronic_disease = int(not ccd)
        self.__short_term_survival = int(not maca)
        self.__emotional_distress = int(not emotional)
        self.__autonomy_coerce = int(not auto3)

        if frailty == 1.0:
            self.__frailty = 0
        elif frailty == 0.5:
            self.__frailty = 1
        elif frailty == 0.0:
            self.__frailty = 2
        else:
            print("Problematic frailty value : ", frailty)

        self.__functional_independence = barthel*100
        self.__instrumental_independence = lawton*Criteria.max_lawton

        if cogn_deterioration >= 1.0:
            self.__cognitive_deterioration = 0
        elif cogn_deterioration >= 0.66:
            self.__cognitive_deterioration = 1
        elif cogn_deterioration >= 0.33:
            self.__cognitive_deterioration = 2
        elif cogn_deterioration == 0.0:
            self.__cognitive_deterioration = 3
        else:
            print("Problem cogn :", cogn_deterioration)

        if discomfort == 1.0:
            self.__discomfort = 0
        elif discomfort == 0.5:
            self.__discomfort = 1
        elif discomfort == 0.0:
            self.__discomfort = 2
        else:
            print("Problem discomfort : ", discomfort)


    def get_criteria_normalized(self):
        """

        - All criteria are integers from 0 to 1
        - 1 denotes the best situation
        - 0 denotes the worst situation

        """

        age, ccd, maca, exp_survival, frailty, crg, ns, barthel, lawton, adv_directives, cogn_deterioration, emotional, discomfort, auto1, auto2, auto3 = \
            self.list()

        age2 = min(100.0, age)/100.0

        #print("Age really normalized!", age, age2)
        age = age2
        ccd = int(not ccd)
        maca = int(not maca)
        exp_survival = int(exp_survival)
        emotional = int(not emotional)

        auto1 = int(auto1)
        auto2 = int(auto2)
        auto3 = int(not auto3)

        if frailty in ("Low", 0):
            frailty = 1.0
        elif frailty in ("Moderate", 1):
            frailty = 0.5
        elif frailty in ("High", 2):
            frailty = 0.0
        ns = int(ns)
        barthel = barthel/100.
        lawton = (1./Criteria.max_lawton)*lawton
        adv_directives = int(adv_directives)
        if self.__cognitive_deterioration in ("Absent", 0):
            cogn_deterioration = 1.0
        elif self.__cognitive_deterioration in ("Slight", 1):
            cogn_deterioration = 0.667
        elif self.__cognitive_deterioration in ("Moderate", 2):
            cogn_deterioration = 0.333
        elif self.__cognitive_deterioration in ("Severe", 3):
            cogn_deterioration = 0.0

        if self.__discomfort in ("Low", 0):
            discomfort = 1.0
        elif self.__discomfort in ("Moderate", 1):
            discomfort = 0.5
        elif self.__discomfort in ("High", 2):
            discomfort = 0.0

        return [age, ccd, maca, exp_survival, frailty, crg, ns, barthel,
            lawton, adv_directives, cogn_deterioration, emotional, discomfort, auto1, auto2, auto3]

    def there_are_inconsistencies(self):

        if not Criteria.min_age <= self.__age <= Criteria.max_age:
            print("thiss age???", self.__age)
            return True

        if not Criteria.min_ccd <= self.__complex_chronic_disease <= Criteria.max_ccd:
            print("hmmm")
            return True

        if not Criteria.min_exp_survival <= self.__expected_survival <= Criteria.max_exp_survival:
            print("ha!")
            return True

        if not Criteria.min_crg <= self.__clinical_risk_group <= Criteria.max_crg:
            print("he!")
            return True

        if not Criteria.min_ns <= self.__social_support <= Criteria.max_ns:
            print("oh oh")
            return True

        if not Criteria.min_adv_directives <= self.__advanced_directives <= Criteria.max_adv_directives:
            print("le lelee")
            return True

        if not Criteria.min_cogn <= self.__cognitive_deterioration <= Criteria.max_cogn:
            print("hu!", self.__cognitive_deterioration)
            return True

        if not Criteria.min_frailty <= self.__frailty <= Criteria.max_frailty:
            print("ho!")
            return True

        if not Criteria.min_maca <= self.__short_term_survival <= Criteria.max_maca:
            print("shortie!")
            return True

        if not Criteria.min_barthel <= self.__functional_independence <= Criteria.max_barthel:
            print("indepe!")
            return True

        if not Criteria.min_lawton <= self.__instrumental_independence <= Criteria.max_lawton:
            print("wrong lawton!!", self.__functional_independence)
            return True

        if not Criteria.min_emotional <= self.__emotional_distress <= Criteria.max_emotional:
            print("emotional!")
            return True

        if not Criteria.min_discomfort <= self.__discomfort <= Criteria.max_discomfort:
            print("comfort!")
            return True

        return False

    def correct_inconsistencies(self):

        self.__age = max(Criteria.min_age, min(Criteria.max_age, self.__age))
        self.__complex_chronic_disease = max(Criteria.min_ccd, min(Criteria.max_ccd, self.__complex_chronic_disease))
        self.__expected_survival = max(Criteria.min_exp_survival, min(Criteria.max_exp_survival, self.__expected_survival))
        self.__clinical_risk_group = max(Criteria.min_crg, min(Criteria.max_crg, self.__clinical_risk_group))
        self.__social_support = max(Criteria.min_ns, min(Criteria.max_ns, self.__social_support))

        self.__advanced_directives = max(Criteria.min_adv_directives, min(Criteria.max_adv_directives, self.__advanced_directives))
        self.__cognitive_deterioration = max(Criteria.min_cogn, min(Criteria.max_cogn, self.__cognitive_deterioration))
        self.__frailty = max(Criteria.min_frailty, min(Criteria.max_frailty, self.__frailty))
        self.__short_term_survival = max(Criteria.min_maca, min(Criteria.max_maca, self.__short_term_survival))
        self.__functional_independence = max(Criteria.min_barthel, min(Criteria.max_barthel, self.__functional_independence))

        self.__instrumental_independence = max(Criteria.min_lawton, min(Criteria.max_lawton, self.__instrumental_independence))
        self.__emotional_distress = max(Criteria.min_emotional, min(Criteria.max_emotional, self.__emotional_distress))
        self.__discomfort = max(Criteria.min_discomfort, min(Criteria.max_discomfort, self.__discomfort))



    def partial_list(self, age=False, ccd=False, maca=False, exp_survival=False, frailty=False,
                     crg=False, ns=False, barthel=False, lawton=False, adv_directives=False, sci_support=False,
                     cogn_deterioration=False, emotional=False, discomfort=False, auto1=False, auto2=False, auto3=False):

        criteria_list = []

        if age:
            criteria_list.append(self.__age)
        if ccd:
            criteria_list.append(self.__complex_chronic_disease)
        if maca:
            criteria_list.append(self.__short_term_survival)
        if exp_survival:
            criteria_list.append(self.__expected_survival)
        if frailty:
            criteria_list.append(self.__frailty)
        if crg:
            criteria_list.append(self.__clinical_risk_group)
        if ns:
            criteria_list.append(self.__social_support)
        if barthel:
            criteria_list.append(self.__functional_independence)
        if lawton:
            criteria_list.append(self.__instrumental_independence)
        if adv_directives:
            criteria_list.append(self.__advanced_directives)
        if sci_support:
            criteria_list.append(self.__decision_taken_with_scientific_support)
        if cogn_deterioration:
            criteria_list.append(self.__cognitive_deterioration)
        if emotional:
            criteria_list.append(self.__emotional_distress)
        if discomfort:
            criteria_list.append(self.__discomfort)
        if auto1:
            criteria_list.append(self.__autonomy_understand)
        if auto2:
            criteria_list.append(self.__autonomy_inform)
        if auto3:
            criteria_list.append(self.__autonomy_coerce)

        if len(criteria_list) == 0:
            return "Error!! Empty list!!"
        elif len(criteria_list) == 1:
            return criteria_list[0]
        else:
            return criteria_list

    def list(self):
        return [self.__age,
                self.__complex_chronic_disease,
                self.__short_term_survival,
                self.__expected_survival,
                self.__frailty,
                self.__clinical_risk_group,
                self.__social_support,
                self.__functional_independence,
                self.__instrumental_independence,
                self.__advanced_directives,
                #self.__decision_taken_with_scientific_support,
                self.__cognitive_deterioration,
                self.__emotional_distress,
                self.__discomfort,
                self.__autonomy_understand,
                self.__autonomy_inform,
                self.__autonomy_coerce
                ]

    def copy(self):
        return Criteria(age=self.__age, ccd=self.__complex_chronic_disease, maca=self.__short_term_survival,
                                    exp_survival=self.__expected_survival, frailty=self.__frailty,
                                    crg=self.__clinical_risk_group, ns=self.__social_support,
                                    barthel=self.__functional_independence, lawton=self.__instrumental_independence,
                                    adv_directives=self.__advanced_directives,
                                    cogn_deterioration=self.__cognitive_deterioration,
                                    emotional_distress=self.__emotional_distress, discomfort=self.__discomfort,
                                    auto1=self.__autonomy_understand, auto2=self.__autonomy_inform,
                                    auto3=self.__autonomy_coerce)

def improve_criteria(ccd=None, maca=None, exp_survival=None, frailty=None, crg=None, ns=None,
                          barthel=None, lawton=None, cogn_deterioration=None, emotional=None,
                          discomfort=None):

        if ccd is not None:
            ccd = False
            return ccd
        if maca is not None:
            maca = False
            return maca
        if exp_survival is not None:
            exp_survival = True  # boolean
            return exp_survival
        if frailty is not None:
            frailty -= 1  # Low, Moderate, or High
            return frailty
        if crg is not None:
            crg += 1  # integer between 0 and 3, both included
            return crg
        if ns is not None:
            ns = True  # boolean
            return ns
        if cogn_deterioration is not None:
            cogn_deterioration -= 1
            return cogn_deterioration
        if barthel is not None:
            if barthel < 40:
                return 40
            elif barthel < 60:
                return 60
            elif barthel < 90:
                return 75
            elif barthel < 95:
                return 90
            else:
                return 100

        if lawton is not None:
            lawton += 1  # integer between 0 and 8
            return lawton
        if emotional is not None:
            emotional = False
            return emotional
        if discomfort is not None:
            discomfort -= 1
            return discomfort



def worsen_criteria(ccd=None, maca=None, exp_survival=None, frailty=None, crg=None, ns=None,
                 barthel=None, lawton=None, adv_directives=None, cogn_deterioration=None, emotional=None,
                 discomfort=None):


        if ccd is not None:
            ccd = True  # boolean
            return ccd
        if maca is not None:
            maca = True
            return maca
        if exp_survival is not None:
            exp_survival = False  # boolean
            return exp_survival
        if frailty is not None:
            frailty += 1  # Low, Moderate, or High
            return frailty
        if crg is not None:
            crg -= 1  # integer between 0 and 3, both included
            return crg
        if ns is not None:
            ns = False  # boolean
            return ns
        if adv_directives is not None:
            adv_directives = False
            return adv_directives
        if cogn_deterioration is not None:
            cogn_deterioration += 1
            return cogn_deterioration
        if barthel is not None:

            if barthel == 100:
                barthel = 95
            elif barthel > 90:
                barthel = 75
            elif barthel > 60:
                barthel = 40
            else:
                barthel = 10
            return barthel

        if lawton is not None:
            lawton -= 1  # integer between 0 and 8
            return lawton
        if emotional is not None:
            emotional = True
            return emotional
        if discomfort is not None:
            discomfort += 1
            return discomfort


if __name__ == "__main__":

    #for _ in range(1):
    patient1 = Criteria(random=True)

    print(patient1.get_criteria_names())
    print(patient1.list())
    print(patient1.get_criteria_normalized())
    print("---")

    patient2 = patient1.copy()

    force_different_patient(patient2, 1)

    print(patient2.get_criteria_names())
    print(patient2.list())
    print(patient2.get_criteria_normalized())
    print("---")

    print(distance_patients(patient1, patient2))