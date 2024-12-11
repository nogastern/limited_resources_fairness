MIMIC = 'mimic'
KAGGLE = 'kaggle'
XGBOOST = 'xgboost'
LR = 'logistic regression'
LGBM = 'lgbm'
MY_XGBOOST = 'my_xgboost'
MY_LR = 'my_logistic_regression'
SENSITIVITY = 'Sensitivity'
PPV = 'PPV'
NPV = 'NPV'
SPECIFICITY = 'Specificity'
LIFT = 'Lift'
AGE_GROUPS = 'age_groups'
RACE = 'race'
AGE_GROUPS_RACE = AGE_GROUPS + '_' + RACE
ETHNICITY = 'ethnicity'
AGE_GROUPS_ETHNICITY = AGE_GROUPS + '_' + ETHNICITY
GENDER = 'gender'
MARITAL_STATUS = 'marital_status'
ICU_TYPE = 'icu_type'
AGE_GROUPS_ICU_TYPE = AGE_GROUPS + '_' + ICU_TYPE
DIABETES = 'diabetes_mellitus'
APACHE_3 = 'apache_3j_bodysystem'
AGE_GROUPS_APACHE_3 = AGE_GROUPS + '_' + APACHE_3
AGE_GROUPS_GENDER = AGE_GROUPS + '_' + GENDER
PERCENT_CUTOFF = 'Percent Cutoff'
OVERALL = 'overall'
CHECK_CONVERGENCE = 'check_convergence'
BY_NUM_OF_ITERS = 'Iterative'
BRUTE_FORCE = "Branch and Bound"
BUCKETS = 'Buckets'
BINARY_SEARCH = "binary_search"
PRE = 'Pre-process'
IN = 'In-process'
POST = 'Post-process'
ORIGINAL = 'Original'
FAIRNESS = 'max-min'
COLOR_MAPPING = \
    {'Original': '#1f77b4',
     'Pre-process':'#ff7f0e',
     'In-process': '#2ca02c',
     BUCKETS: '#d62728',
     f'{POST} ({BUCKETS})': 'red',
     BY_NUM_OF_ITERS: 'purple',
     f'{POST} ({BY_NUM_OF_ITERS})': 'purple',
     f'{POST} ({BRUTE_FORCE})': 'pink'
     }
STD = 'standard deviation'

