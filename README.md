# SurvLIMEpy

<p align="center">
    <img src="https://github.com/imatge-upc/SurvLIMEpy/blob/main/logo.png?raw=true" width="256" height="256">
</p>


**SurvLIMEpy** implements SurvLIME algorithm (**Survival Local Interpretable Model-agnostic Explanation**), a local interpretable algorithm for Survival Analysis, which was proposed in [this paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705120304044).

The publication in which we introduce this package will soon be available.

## Install
**SurvLIMEpy** can be installed from PyPI:

```
pip install survlimepy
```

## How to use
```python
from survlimepy import SurvLimeExplainer
from survlimepy.load_datasets import Loader
from sksurv.linear_model import CoxPHSurvivalAnalysis

# Load UDCA dataset
loader = Loader(dataset_name='udca')
X, events, times = loader.load_data()

# Train a model
train, test = loader.preprocess_datasets(X, events, times)
model = CoxPHSurvivalAnalysis()
model.fit(train[0], train[1])

# Use SurvLimeExplainer class to find the feature importance
training_features = train[0]
training_events = [event for event, _ in train[1]]
training_times = [time for _, time in train[1]]

explainer = SurvLimeExplainer(
    training_features=training_features,
    training_events=training_events,
    training_times=training_times,
    model_output_times=model.event_times_,
)

# explanation variable will have the computed SurvLIME values
explanation = explainer.explain_instance(
    data_row=test[0].iloc[0],
    predict_fn=model.predict_cumulative_hazard_function,
    num_samples=1000,
)
print(explanation)

# Display the weights
explainer.plot_weights()
```

## Model compatibility
Our package can manage multiple types of survival models as long as the functionality that predicts is implemented as a function that takes a vector of size $p$ (the number of features) and outputs a vector of size $q \leq m$ (where $m$ is the number of unique times to event). Most of the packages are compliant with this rule. Therefore, apart from the **Cox Proportional Hazards Model**, which is implemented in **sksurv** library, **SurvLIMEpy** also manages more recent algorithms such as **Random Survival Forest**, implemented in **sksurv** library, **Survival regression with accelerated failure time model in XGBoost**, implemented in **xgbse** library, **DeepHit** and **DeepSurv**, both implemented in **pycox** library.

We choose to ensure the integration of these algorithms with **SurvLIMEpy** as they are the most predominant in the field. Note that if a new survival package is developed, **SurvLIMEpy** will support it as long as the output provided by the predict function is a vector of length $q \leq m$. In [this notebook](https://github.com/imatge-upc/SurvLIME-experiments/blob/main/notebooks/multiple_models.ipynb) there are several examples with different models.

## Citations
Please if you use this package, do not forget to cite us. citation data available soon.

## References

### Algorithms
*SurvLIME*: Maxim S. Kovalev, Lev V. Utkin, & Ernest M. Kasimov (2020). SurvLIME: A method for explaining machine learning survival models. Knowledge-Based Systems, 203, 106164.

*Cox Proportional Hazards Model*: Cox, D. R. (1972). Regression models and life-tables. Journal of the Royal Statistical Society: Series B (Methodological), 34(2), 187–202.

*Random Survival Forest*: Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S. (2008). Random survival forests. The Annals of Applied Statistics, 2(3), 841–860. doi:10.1214/08-AOAS169

*Survival regression with accelerated failure time model in XGBoost*: Barnwal, A., Cho, H., & Hocking, T. (2022). Survival Regression with Accelerated Failure Time Model in XGBoost. Journal of Computational and Graphical Statistics, 0(0), 1–11. doi:10.1080/10618600.2022.2067548

*DeepHit*: Lee, C., Zame, W., Yoon, J., & van der Schaar, M. (2018). DeepHit: A Deep Learning Approach to Survival Analysis With Competing Risks. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1). doi:10.1609/aaai.v32i1.11842

*DeepSurv*: Katzman, J., Shaham, U., Cloninger, A., Bates, J., Jiang, T., & Kluger, Y. (02 2018). DeepSurv: Personalized treatment recommender system using a Cox proportional hazards deep neural network. BMC Medical Research Methodology, 18. doi:10.1186/s12874-018-0482-1

### Datasets
*PBC*: Therneau, T.M., Grambsch, P.M. (2000). Expected Survival. In: Modeling Survival Data: Extending the Cox Model. Statistics for Biology and Health. Springer, New York, NY. https://doi.org/10.1007/978-1-4757-3294-8_10

*Lung*: Loprinzi CL. Laurie JA. Wieand HS. Krook JE. Novotny PJ. Kugler JW. Bartel J. Law M. Bateman M. Klatt NE. et al. Prospective evaluation of prognostic variables from patient-completed questionnaires. North Central Cancer Treatment Group. Journal of Clinical Oncology. 12(3):601-7, 1994.

*UDCA*: (1) T. M. Therneau and P. M. Grambsch, Modeling survival data: extending the Cox model. Springer, 2000; (2) K. D. Lindor, E. R. Dickson, W. P Baldus, R.A. Jorgensen, J. Ludwig, P. A. Murtaugh, J. M. Harrison, R. H. Weisner, M. L. Anderson, S. M. Lange, G. LeSage, S. S. Rossi and A. F. Hofman. Ursodeoxycholic acid in the treatment of primary biliary cirrhosis. Gastroenterology, 106:1284-1290, 1994. 

*Veterans*: D Kalbfleisch and RL Prentice (1980), The Statistical Analysis of Failure Time Data. Wiley, New York. 

### Libraries
*sksurv*: Sebastian Polsterl (2020). scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn. Journal of Machine Learning Research, 21(212), 1-6.

*xgbse*: Davi Vieira, Gabriel Gimenez, Guilherme Marmerola, & Vitor Estima. (2020). XGBoost Survival Embeddings: improving statistical properties of XGBoost survival analysis implementation.

*pycox*: Håvard Kvamme, Ørnulf Borgan, and Ida Scheel. Time-to-event prediction with neural networks and Cox regression. Journal of Machine Learning Research, 20(129):1–30, 2019.
