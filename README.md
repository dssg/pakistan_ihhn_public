
<p align="center">
  <img src="https://user-images.githubusercontent.com/36204574/185437078-e7790525-4140-475d-9b1f-8cab051faa61.png">
</p>

# Analyzing Emergency Room triage notes to support better treatment and improve health outcomes in Pakistan

To view the code produced for this project by the 2022 Data Science for Social Good (DSSG) fellows, click [here](https://github.com/dssg/pakistan_ihhn).

## The Policy Problem:

The Indus Hospital and Health Network (IHHN) is a non-profit healthcare provider in Pakistan serving more than 5.4 million patients a year. Our partner is a team of doctors and researchers with the Emergency Department (ED) of IHHN’s flagship hospital in Korangi, Karachi. IHHN’s hospitals are the preferred destination for many patients because its services are provided free of charge. However, patients regularly face long wait times because the demand for seats outstrips the available staff. 

## The Machine Learning Problem: 

### Objective
Due to high patient volume and limited hospital resources, the hospital is also constrained in its ability to triage and diagnose patients efficiently and effectively. Additionally, IHHN lacks structured workflows and follow-up planning because the intake process relies on unstructured text data (nurse and physician notes) and is subjective (varies across physicians), variable (varies for each physician over time), inefficient (takes too long to do manually), and error-prone (due to human involvement). Our work with IHHN focuses on building a system to convert nurse and physician notes into structured International Classification of Disease (ICD-10) codes that IHHN can use to create workflows and tools for physicians, nurses, and hospital staff as they work to improve patient care and outcomes. 

### Unit of Analysis + Temporal Cross-Validation
Predictions are made at the patient-visit level (eg. as of January 1, 2019, what are the top 10 predicted ICD-10 categories that a given diagnosis is associated with?). We used data from 2019-2021 for this analysis. We filter for non-pregnant adults. We used a temporal cross-validation approach to train the models to ensure that our final model will generalize effectively to new data. As illustrated by Figure 1 below, this temporal cross-validation approach splits the data by time - training each individual model on increasing amounts of data 

![Temporal Cross-Validation](https://user-images.githubusercontent.com/36204574/185440407-35d870e2-21f0-48e5-a6c6-f4acfe7c4ca4.png)

We have two types of cohorts: training and validation cohorts. The training cohort refers to all data points before the start of the validation cohort. Validation cohorts are three months of data. The first validation cohort begins at the tail end of our data set (2021-09-01 to 2021-12-30). The training cohort for this validation set begins 2019-01-01 and ends 2021-08-31. The validation window is then sequentially moved back to cover the previous 3 months preceeding the current validation cohort start date. The training cohort then becomes every data point preceeding the start date of the current validation cohort.

### Feature Generation

We generated several features 


| Feature Name | Data Source(s) | Description |
| ----------- | ----------- | ----------- |
| **Text Features** | | |
| Triage_complaint | HMIS data | Categorical, short text notes, assigned to patients at the first point of contact |
| HOPI | HMIS data | Clinician note |
| ED_DX | HMIS data | Provisional diagnosis |
| **Categorical Features** | | |
| Acuity | HMIS data | Severity of a patient’s medical condition | 
| Gender | HMIS data | Sex assigned at birth | 
| Investigations completed during past visits | Investigations data | Refers to how many medical investigations have been carried out on the patient prior to the current visit 
| **Continuous Features** | 
| Age | HMIS data | The age of the patient |
| Number of previous visits | HMIS data | The number of times a patient has been to the hospital in the past | 
| Number of notes completed during current visit | HMIS data | How many notes did the physician | write for the current patient visit | 
|Length of doctor's note in words or characters  | HMIS data | Possible proxy for case complexity | Number of question marks | Possible proxy for level of certainty about provisional diagnosis | 
| Pulse | 
| Temperature | Temperature in Fahrenheit | 
| Breathing rate  | 
| **Temporal Features** | 
| Season | Season corresponding to `triage_datetime` | Season of the year (e.g., fall, winter, spring, summer) |


## Pipeline Configuration File

The ETL and machine learning pipeline can be configured using the `config` file [`config/model_settings.py`](https://github.com/dssg/pakistan_ihhn/blob/develop/config/model_settings.py). 

Please reference the [readme_config.yaml](https://github.com/dssg/El_Salvador_mined_education/blob/master/experiments/readme_config.yaml) file for the set of configuration parameters that build all of the final models discussed in this document.

We take advantage of some secondary data sources provided
Original files: https://www.cms.gov/medicare/icd-10/2022-icd-10-cm
Google Drive: https://drive.google.com/drive/folders/1Y1rAYzsHBddZIdinMYMqTcIYPJs0Fp8f?usp=sharing

These are required to be uploaded to the directory where all the other raw files from 
### ETL
To run the full ETL and machine learning pipeline complete the following steps:

```
bash run_full_pipeline.sh /path/to/raw/files
```
This will complete 5 tasks:
1. converts all raw `xls`, `xlsx` and `text` files to `csv`
2. converts "raw" `csv` files  into a "processed" csv directory
3. From the "processed" directory, we write the csvs to a `raw` schema
4. From the `raw` schema, we clean data and return to `processed` schema
5. We build our training set from this and return to `model_output` schema.
6. Running the full machine learning pipeline on the `dev` schema (limiting training data to 1000 rows for testing purposes).

to run only ETL comment out the final line such that:
```
# pakistan-ihhn run-pipeline --schema_type dev
```

### Machine Learning Pipeline
With the settings in the `config`, the machine learning pipeline can be run from the command line using the CLI we set up:
```pakistan-ihhn run-pipeline --schema_type prod```

For testing changes to the pipeline using small amounts of data:
```pakistan-ihhn run-pipeline --schema_type dev```
* This will create a new schema called `{SCHEMA_NAME}_dev` if it does not already exist and will save all output there. It will also append `dev` to the model objects written to the server.
* Setting `--schema_type` to `prod` to run on full dataset.

## Results
For our analysis, we have configured our pipeline to use random forest classifiers, decision tree classifiers, multinomial logistic regression, multinomial naive bayes, and XGBoost models<sup>2</sup>. However, for our current model output, we prioritized running random forest classifiers, decision tree classifiers, and multinomial naive bayes. We evaluated or models based on the following metrics at various constraints:

- Average recall at 5,10,15,20 ICD-10 categories
- Average precision at 5,10,15,20 ICD-10 categories
- Average accuracy at 5,10,15, 20 ICD-10 categories

We compared our results of our three models against two baselines: 1) we compared a given patient diagnosis to similar cases which have passed through the ED in the past and 2)  treated the provisional diagnoses as “assumed to be correct”, and then predicted the code with maximum similarity to the provisional diagnosis.
