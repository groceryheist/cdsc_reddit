from sklearn import metrics
from sklearn.cluster import AffinityPropagation
from functools import partial
# sillouette is the only one that doesn't need the feature matrix. So it's probably the only one that's worth trying. 
