# SurvLIME

<p align="center">
    <img src="logo.png" width="256" height="256">
</p>


SurvLIME (**Survival Local Interpretable Model-agnostic Explanation**) is a local interpretable algorithm for Survival Analysis models. This implements the method proposed in the [original paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705120304044).

The publication in which we introduce this package will soon be available.

## Install
SurvLIME can be installed from PyPI:

```
pip install survlime
```

## How to use
```python
from survlime import survlime_explainer
from survlime.load_datasets import Loader
from sksurv.linear_model import CoxPHSurvivalAnalysis
from functools import partial

# Load UDCA dataset
loader = Loader(dataset_name='udca')
x, events, times = loader.load_data()

# Train a model
train, val, test = loader.preprocess_datasets(x, events, times)
model = CoxPHSurvivalAnalysis()
model.fit(train[0], train[1])

# Use SurvLIME explainer
explainer = survlime_explainer.SurvLimeExplainer(train[0], train[1], model_output_times=model.event_times_)
pred_func = partial(model.predict_cumulative_hazard_function, return_array=True)
 
# explanation variable will have the computed SurvLIME values
explanation = explainer.explain_instance(test[0].iloc[0], pred_func, num_samples=1000)
```

## Citations
Please if you use this package, do not forget to cite us. citation data available soon.

## References:

### Datasets

PBC:
Therneau, T.M., Grambsch, P.M. (2000). Expected Survival. In: Modeling Survival Data: Extending the Cox Model. Statistics for Biology and Health. Springer, New York, NY. https://doi.org/10.1007/978-1-4757-3294-8_10

Lung:
Loprinzi CL. Laurie JA. Wieand HS. Krook JE. Novotny PJ. Kugler JW. Bartel J. Law M. Bateman M. Klatt NE. et al. Prospective evaluation of prognostic variables from patient-completed questionnaires. North Central Cancer Treatment Group. Journal of Clinical Oncology. 12(3):601-7, 1994.

UDCA:
T. M. Therneau and P. M. Grambsch, Modeling survival data: extending the Cox model. Springer, 2000.

K. D. Lindor, E. R. Dickson, W. P Baldus, R.A. Jorgensen, J. Ludwig, P. A. Murtaugh, J. M. Harrison, R. H. Weisner, M. L. Anderson, S. M. Lange, G. LeSage, S. S. Rossi and A. F. Hofman. Ursodeoxycholic acid in the treatment of primary biliary cirrhosis. Gastroenterology, 106:1284-1290, 1994. 

Veterans:
D Kalbfleisch and RL Prentice (1980), The Statistical Analysis of Failure Time Data. Wiley, New York. 
