# JDSA-APiES
Repository for the Aggregated Prediction in Event Streams framework, for the paper "Aggregated Prediction in Event Streams of Shopper behaviour" for the International Journal of Data Science and Analytics, Special Issue on Domain-Driven Data Mining. 

To run the code, first create a virtual environment (Python 3.8) and install the requirements

```pip install -r requirements.txt```

From there you can recreate the results for the shoppers with:

```python APiES/experiments/consumers/results.py``` 

For the BPIC'19 set with

```python APiES/experiments/bpic2019/results.py```

And rerun the bpic2019 experiments using

```python APiES/experiments/bpic2019/experiment.py```

The shopper experiment contains private data which cannot be shared at the moment.

A standalone python package will be made available in the future.