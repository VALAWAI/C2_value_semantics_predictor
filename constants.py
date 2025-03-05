test_percentage = 0.1

BENEFICENCE = 0
NONMALEFICENCE = 1
JUSTICE = 2
AUTONOMY = 3

n_values = len([BENEFICENCE, NONMALEFICENCE, JUSTICE, AUTONOMY])

n_consequence_values = len([BENEFICENCE, NONMALEFICENCE])
n_duty_values = len([JUSTICE, AUTONOMY])

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

n_actions = 11

max_age = 100
max_frailty = max_discomfort = 2
max_crg = max_cogn = 3
max_barthel = 8
max_lawton = 100
max_ccd = max_maca = max_ns = max_adv_directives = max_emotional = max_exp_survival = max_auto1 = max_auto2 = max_auto3 = 1

min_age = 1
min_exp_survival = min_ccd = min_maca = min_ns = min_adv_directives = min_frailty = min_crg = min_barthel = \
    min_cogn = min_discomfort = min_lawton = min_emotional = min_auto1 = min_auto2 = min_auto3 = 0
