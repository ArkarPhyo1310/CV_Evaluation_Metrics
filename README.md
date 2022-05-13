# <div align="center">CV Evaluation Metrics [All in One] </div>

This project is inspired due to the lack of various evaluation metrics of computer vision in one place.

## <div align="center"> Features </div>

<details open>
<summary> Classification  </summary>

- Accuracy :ballot_box_with_check:
- Precision :ballot_box_with_check:
- Recall :ballot_box_with_check:
- F1-Score :ballot_box_with_check:
- Confusion Matrix :ballot_box_with_check:
  
</details>

<details open>
<summary> Detection </summary>

- Mean Average Precision/Mean Average Recall :ballot_box_with_check:
  
</details>

<details>
<summary> Segmentation </summary>

- Coming Soon...
  
</details>
<details open>
<summary> Tracking </summary>

- CLEAR metrics :ballot_box_with_check:
- HOTA metrics :ballot_box_with_check:
- Identity metrics :ballot_box_with_check:
  
</details>
<details>
<summary> Pose Estimation </summary>

- Coming Soon...
  
</details>
<details>
<summary> Image Super Resolution </summary>

- Coming Soon...
  
</details>

>Incompleted features will be implemented in near future.

## <div align="center"> Installation </div>

### Requirements

- Python 3.6+
- numpy
- pandas
- scipy
- tabulate

```bash
git clone https://github.com/ArkarPhyo1310/CV_Evaluation_Metrics.git
cd CV_Evaluation_Metrics
python setup.py develop
```

## <div align="center"> Usage/Examples </div>

- Classification

1. Create Classfication Object for prediction and target. (Classification Object)
2. Create Classification Metric Config.
3. Update the Metric Config with Classification Object.
4. Create Evaluator.
5. Then evaluate the result.

```python
pred = [[0.75, 0.15, 0.10, 0.0],
        [0.15, 0.75, 0.0, 0.10],
        [0.15, 0.10, 0.75, 0.0],
        [0.0, 0.10, 0.15, 0.75]]
target = [0, 1, 3, 2]

gt = ClassificationObject(labels=target)
pred = ClassificationObject(labels=pred)

cls_cfg = CMetricConfig(
    classes=[i for i in range(4)], top_k=2, average="macro"
)

cls_cfg.update(gt, pred)

metric_list = ["Accuracy", "Precision", "Recall", "F1-score", "Confusion Matrix"]
evaluator = MetricEvaluator(specific_metric_fields=metric_list, evaluation_task="classification")
evaluator.evaluate(cls_cfg)
evaluator.render_result(model_name="Custom", show_overall=False)
```

Result will be like this:

```bash
+------------------+-----+-----+-----+-----+
| Confusion Matrix |  0  |  1  |  2  |  3  |
+------------------+-----+-----+-----+-----+
|        0         | 1.0 | 0.0 | 0.0 | 0.0 |
|        1         | 0.0 | 1.0 | 0.0 | 0.0 |
|        2         | 0.0 | 0.0 | 0.0 | 1.0 |
|        3         | 0.0 | 0.0 | 1.0 | 0.0 |
+------------------+-----+-----+-----+-----+
+--------------------+----------+-----------+--------+----------+
| MODEL NAME: Custom | Accuracy | Precision | Recall | F1-Score |
+--------------------+----------+-----------+--------+----------+
|        N/A         |    75    |   33.33   |   75   |  45.83   |
+--------------------+----------+-----------+--------+----------+
```

You can check the running scripts in [here](tools).

```bash
python tools/run_classification.py
python tools/run_tracking.py
python tools/run_detection.py
```

>NOTE: More evaluation metrics for different task such as detection, segmentation, etc. will be updated soon. I have also provided data required to test for tracking in the ***data*** folder. Original data can be found [here](https://github.com/JonathonLuiten/TrackEval#quickly-evaluate-on-supported-benchmarks).

## <div align="center"> References </div>

- [TrackEval](https://github.com/JonathonLuiten/TrackEval)
- [TorchMetrics](https://torchmetrics.rtfd.io/en/latest)
- [Pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI)
