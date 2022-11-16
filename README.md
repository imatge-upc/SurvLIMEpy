# SurvLIME

<div style="text-align:center"><img src="logo.png" /></div>


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
from survlime.datasets.load_datasets import Loader
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
explanation, min_value = explainer.explain_instance(test[0].iloc[0], pred_func, num_samples=1000)

print(explanation)
```

## Citations
Please if you use this package, do not forget to cite us. citation data available soon.
