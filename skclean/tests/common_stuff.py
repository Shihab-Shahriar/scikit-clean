from itertools import product

import pandas as pd
from sklearn import clone

from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import skclean
from skclean.utils import load_data
from skclean.detectors import KDN
from skclean.pipeline import make_pipeline
from skclean.simulate_noise import UniformNoise


SEED = 39
dummy = DecisionTreeClassifier(max_depth=4, random_state=SEED)



NOISE_SIMULATORS = [
    skclean.simulate_noise.UniformNoise(.2),
    skclean.simulate_noise.CCNoise(.3),
    skclean.simulate_noise.BCNoise(KNeighborsClassifier(), .2),
]

NOISE_DETECTORS = [
    skclean.detectors.ForestKDN(),
    skclean.detectors.InstanceHardness(),
    skclean.detectors.KDN(),
    skclean.detectors.RkDN(),
    # skclean.detectors.HybridKDN(dummy),
    skclean.detectors.MCS(),
    skclean.detectors.PartitioningDetector(),
    skclean.detectors.RandomForestDetector(),
]

NOISE_HANDLERS = [
    skclean.handlers.CLNI(dummy, KDN()),
    skclean.handlers.Filter(dummy, KDN()),
    skclean.handlers.IPF(dummy, KDN()),
    skclean.handlers.SampleWeight(dummy, KDN()),
    skclean.handlers.WeightedBagging(detector=KDN()),
    skclean.handlers.Costing(detector=KDN()),
    skclean.handlers.FilterCV(dummy, KDN(), thresholds=[.2, .3, .4], cv=3)
]

ROBUST_MODELS = [
    skclean.models.Centroid(),
    skclean.models.RobustForest(),
    skclean.models.RobustLR(),
]

PIPELINE = skclean.pipeline.Pipeline([
    ('a', skclean.simulate_noise.UniformNoise(.2, random_state=SEED)),
    ('b', StandardScaler()),
    ('c', VarianceThreshold(.2)),
    ('d', KDN()),
    ('e', skclean.handlers.SampleWeight(dummy))
])


# Inside Pipeline
tmp_Handlers = []
for h in NOISE_HANDLERS:
    if h.iterative:  # Exclude iterative handlers
        continue
    ch = clone(h)
    ch.detector = None
    tmp_Handlers.append(ch)
preli_steps = [UniformNoise(.2, random_state=SEED), StandardScaler()]
all_comb = product(NOISE_DETECTORS, tmp_Handlers)
INSIDE_PIPE = [make_pipeline(*preli_steps + list(comb)) for comb in all_comb]

# Outside Pipe
OUTSIDE_PIPE = []
for h in NOISE_HANDLERS:
    for d in NOISE_DETECTORS:
        ch, d = clone(h), clone(d)
        ch.detector = d
        if 'random_state' in ch.get_params():  # trying to avoid flaky tests
            ch.set_params(random_state=42)
        OUTSIDE_PIPE.append(ch)

ALL_COMBS = INSIDE_PIPE + OUTSIDE_PIPE

ALL_ESTIMATORS = NOISE_SIMULATORS + NOISE_DETECTORS + ALL_COMBS + [PIPELINE]

Xiris, yiris = load_data('iris')
X, y = make_classification(100, 4)

df = pd.DataFrame(X)
df['y'] = y

# TODO: Add imbalanced datasets
y_imb = yiris.copy()
y_imb[yiris == 1] = 0

DATASETS = [
    (X, y),
    (Xiris, yiris),
    #(Xiris, y_imb),
    (X.tolist(), y.tolist()),
    (df.drop('y', axis=1), df['y'])
]
