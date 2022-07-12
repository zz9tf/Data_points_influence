https://www.kaggle.com/datasets/jeffheaton/tabular-feature-engineering-dataset

About Dataset
Machine learning models, such as neural networks, decision trees, random forests, and gradient boosting machines, accept a feature vector, and provide a prediction. These models learn in a supervised fashion where we provide feature vectors with the expected output. It is common practice to engineer new features from the provided feature set. Such engineered features will either augment or replace portions of the existing feature vector. These engineered features are essentially calculated fields based on the values of the other features.

Engineering such features is primarily a manual, time-consuming task. Additionally, each type of model will respond differently to different kinds of engineered features. This paper reports empirical research to demonstrate what kinds of engineered features are best suited to various machine learning model types. We provide this recommendation by generating several datasets that we designed to benefit from a particular type of engineered feature. The experiment demonstrates to what degree the machine learning model can synthesize the needed feature on its own. If a model can synthesize a planned feature, it is not necessary to provide that feature. The research demonstrated that the studied models do indeed perform differently with various types of engineered features.

We generated this dataset for the following paper:

Heaton, J. (2016, April). An Empirical Analysis of Feature Engineering for Predictive Modeling. In SoutheastCon 2016 (pp. 1-6). IEEE.