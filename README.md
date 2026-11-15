# Predictive maintenance on the AI4I 2020 dataset

Predicting machine failure from sensor readings. I come from a mechanical background, so this is a
problem I find genuinely interesting: given a few live readings off a machine (temperatures, rotational
speed, torque, tool wear), can you flag that it is about to fail before it does?

The data is the AI4I 2020 predictive maintenance dataset, 10,000 synthetic machining runs. It is
deliberately imbalanced, only 3.4% of runs actually fail, which is realistic and also the whole
challenge. A model that just says "no failure" every time is right 96.6% of the time and completely
useless. So accuracy is the wrong thing to chase here. What matters is recall on the failure class,
catching the failures, because a missed failure is the one that costs you.

## What's in here

- `analysis.ipynb` is the full walkthrough with the code, charts and commentary.
- `figures/` has the saved plots.
- `data/ai4i2020.csv` is the dataset.

## Approach

A few decisions worth calling out:

- **Dropped the leakage columns.** The dataset ships with five columns (TWF, HDF, PWF, OSF, RNF) that
  label the specific failure mode. That is just the answer broken down by cause, so training on them
  would be cheating. They come out before modelling.
- **Added two features from the physics.** Mechanical power is roughly torque times angular speed, so I
  built a `power` feature, it maps onto the power failure mode. The process-minus-air temperature gap
  maps onto the heat failure mode, so that went in too. Both ended up near the top of the importance
  ranking, so they earned their place.
- **Balanced class weights.** With a 3.4% positive rate the model has to be told not to ignore the
  failure class, so both models use balanced weights.
- **Stratified split** so the small failure class is represented in train and test.

## Results

Random forest, on the held-out test set:

| metric | score |
|---|---|
| recall (failure) | 0.81 |
| precision (failure) | 0.85 |
| F1 (failure) | 0.83 |
| ROC-AUC | 0.97 |

The random forest clearly beat the logistic regression baseline. The failures cluster at the extremes of
torque and speed rather than along a straight line, which is exactly the kind of structure a tree model
handles better than a linear one.

The most useful features were tool wear, rotational speed, torque and the engineered power feature, which
lines up with the physical failure modes in the data.

One more thing the notebook covers: the default 0.5 threshold is not fixed. Because a missed failure is
worse than a false alarm, it is worth moving the threshold to push recall higher and accepting a bit more
precision loss. For a real maintenance setup that trade is usually the right call.

## Honest limitations

This is synthetic data, so it is cleaner than real sensor logs would be, and the scores flatter the model
a little because of that. On real equipment I would expect noisier readings and drift over time, and I
would validate on a time-based split rather than a random one so the test set is genuinely "the future".

## Live demo

`app.py` is a small Streamlit app that trains the model on load (same choices as the
notebook) and lets you move the sensor readings to see the failure probability. Run it with
`streamlit run app.py`, or deploy it free on Streamlit Community Cloud straight from this repo.

## Running it

```
pip install -r requirements.txt
jupyter notebook analysis.ipynb   # the full analysis
streamlit run app.py              # the interactive demo
```

Data source: AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository.
