from dataclasses import field
from typing import Any, Dict, List, Sequence

from pydantic.dataclasses import dataclass


@dataclass
class ProcessXlsxConfig:
    NON_XLSX_FILETYPES = [".csv", ".xls", ".json", ".pdf", ".txt"]
    SHEET_NAME: str = "Sheet 1"

    ICD_COLUMNS_SHEET_MAPPING = [
        {
            "AGE_YEARS": "age_years",
            "Unnamed: 1": "triagecomplaint",
            "HOPI_": "hopi_",
            "ED_DX": "ed_dx",
            "Code": "code",
            "HOPI_.1": "h",
        },
        {
            "Unnamed: 5": "code_using_hopi_",
            "h": "additional_hopi_",
            "Unnamed: 7": "coders_comments_1",
            "Unnamed: 8": "coders_comments_2",
        },
    ]


@dataclass
class ProcessOfficialCodesConfig:
    TEXT_FILETYPES = [".txt"]
    OFFICIAL_CODES_TABLE_ROOT: str = "official_codes"


@dataclass
class ProcessedColumnsConfig:
    DECISIONS_COLUMNS_TO_KEEP = [
        "new_mr",
        "ADMISSION_NO",
        "DIAGNOSIS_CODE",
        "DIAGNOSIS",
        "DIAGNOSIS_TYPE",
        "DOCTOR_CODE",
    ]
    PATIENT_COLUMNS_TO_KEEP = [
        "new_mr",
        "PATIENT_AGE",
        "PATIENT_TYPE",
        "AREA_CODE",
        "DIST_CODE",
        "DISCTRICT",
        "CITY_CODE",
    ]
    PROCEDURE_COLUMNS_TO_KEEP = [
        "new_mr",
        "PROCEDURE_SEQUENCE",
        "PROCEDURE_CODE",
        "PROCEDURE",
        "PROCEDUREDATE",
        "DOCTOR_CODE",
        "PROC_DOCTOR_CODE",
    ]
    ADMISSIONS_COLUMNS_FROM_DIAG_TO_KEEP = [
        "new_mr",
        "ADMISSION_NO",
        "ADMISSION_SOURCE",
        "ADMISSION_DATE",
        "DISCHARGE_DATE",
        "STAY_DURATION",
        "ADMISSION_STATUS",
        "DOCTOR_CODE",
        "PATIENT_TYPE",
        "PATIENT_AGE",
        "GENDER",
        "DEATH_REASON_CODE",
        "DEATH_REASON",
    ]
    ADMISSIONS_COLUMNS_FROM_PROC_TO_KEEP = [
        "new_mr",
        "ADMISSION_NO",
        "ADMISSION_DIAGNOSIS_CODE",
        "ADMISSION_DIAGNOSIS",
        "PRINCIPAL_DIAGNOSIS_CODE",
        "PRINCIPAL_DIAGNOSIS",
        "ASSOCIATED_DISEASE_CODE",
        "ASSOCIATED_DISEASE",
    ]
    TRANSACTIONS_COLUMNS_TO_KEEP = [
        "new_er",
        "new_mr",
        "GENDER",
        "CITY",
        "AREA",
        "AGE_YEARS",
        "TRIAGE_DATETIME",
        "TRIAGECOMPLAINT",
        "BP",
        "TR_PULSE",
        "TR_TEMP",
        "TR_RESP",
        "ACUITY",
        "VISIT_DATETIME",
        "SYSTOLIC",
        "DIASTOLIC",
        "TEMPERATURE",
        "WEIGHT",
        "O2SAT",
        "NURSE_ID",
        "DISPOSITION",
        "DISPOSITION_TIME",
        "HOPI_",
        "ED_DX",
        "DOCTOR_ID",
        "SPECIALTY",
        "ADMISSION_DATE",
        "ADMISSION_WARD",
        "DISCHARGE_WARD",
        "DISCHARGE_DATETIME",
    ]
    INVESTIGATIONS_COLUMNS_TO_KEEP = [
        "new_mr",
        "new_er",
        "LAB_ORDER_DATE",
        "LAB_RESULT_DATE",
        "LAB_TEST",
    ]
    DOCTORS_COLUMNS_FROM_HMIS_TO_KEEP = [
        "SPECIALTY",
        "DOCTOR_ID",
        "new_mr",
    ]
    DOCTORS_COLUMNS_FROM_DIAG_TO_KEEP = ["DOCTOR_CODE", "new_mr"]
    NURSE_COLUMNS_TO_KEEP = [
        "NURSE_ID",
    ]
    SUPPLIES_COLUMNS_TO_KEEP = [
        "CONSUMPTION_ID",
        "ITEM",
        "ITEM_CATEGORY",
        "new_er",
        "new_mr",
    ]

    ICD_COLUMNS_TO_KEEP = [
        "age_years",
        "triagecomplaint",
        "hopi_",
        "ed_dx",
        "code",
        "code_using_hopi_",
        "additional_hopi_",
        "coders_comments_1",
        "coders_comments_2",
        "sheet",
    ]

    ABBREV_COLUMNS_TO_KEEP = [
        "ID",
        "Abbreviation",
        "Term",
    ]
    PRIORTY_COLUMNS_TO_KEEP = [
        "icd_10_cm",
        "count",
        "description_long",
        "ed_dx_json",
    ]


@dataclass
class ProcessCsvConfig(ProcessedColumnsConfig):
    NON_CSV_FILETYPES: Sequence[str] = field(
        default_factory=lambda: [
            ".xlsx",
            ".xls",
            ".json",
        ]
    )
    ADMISSIONS_COLUMNS_TO_JOIN = ["new_mr", "ADMISSION_NO"]
    DOCTORS_COLUMNS_TO_JOIN = ["new_mr"]
    FILESTEMS = {
        "diag": {
            "decisions": ProcessedColumnsConfig.DECISIONS_COLUMNS_TO_KEEP,
            "patients": ProcessedColumnsConfig.PATIENT_COLUMNS_TO_KEEP,
            "admissions": ProcessedColumnsConfig.ADMISSIONS_COLUMNS_FROM_DIAG_TO_KEEP,
            "doctors": ProcessedColumnsConfig.DOCTORS_COLUMNS_FROM_DIAG_TO_KEEP,
        },
        "invs-": {
            "investigations": ProcessedColumnsConfig.INVESTIGATIONS_COLUMNS_TO_KEEP
        },
        "med-": {"supplies": ProcessedColumnsConfig.SUPPLIES_COLUMNS_TO_KEEP},
        "HMIS_visit": {
            "transactions": ProcessedColumnsConfig.TRANSACTIONS_COLUMNS_TO_KEEP,
            "nurses": ProcessedColumnsConfig.NURSE_COLUMNS_TO_KEEP,
            "doctors": ProcessedColumnsConfig.DOCTORS_COLUMNS_FROM_HMIS_TO_KEEP,
        },
        "procedure": {
            "procedures": ProcessedColumnsConfig.PROCEDURE_COLUMNS_TO_KEEP,
            "admissions": ProcessedColumnsConfig.ADMISSIONS_COLUMNS_FROM_PROC_TO_KEEP,
        },
        "Coded_ICD": {
            "codes": ProcessedColumnsConfig.ICD_COLUMNS_TO_KEEP,
        },
        "medicalTermsDictionary": {
            "abbreviations": ProcessedColumnsConfig.ABBREV_COLUMNS_TO_KEEP,
        },
        "official_codes_with_ed_dx": {
            "priority_codes": ProcessedColumnsConfig.PRIORTY_COLUMNS_TO_KEEP,
        },
    }


@dataclass
class ProcessCodesConfig:
    TABLE_NAME = "codes"
    SCHEMA_NAME = "processed"
    COLUMNS_TO_KEEP: Sequence[str] = field(
        default_factory=lambda: [
            "ed_dx",
            "age_years",
            "triagecomplaint",
            "hopi",
            "code",
            "category",
            "ed_dx_hash",
        ]
    )


@dataclass
class ProcessTransactionsConfig:
    TABLE_NAME = "transactions"
    SCHEMA_NAME = "processed"
    COLUMNS_TO_KEEP: Sequence[str] = field(
        default_factory=lambda: [
            "new_er",
            "new_mr",
            "gender",
            "city",
            "age_years",
            "triage_datetime",
            "triagecomplaint",
            "bp",
            "tr_pulse",
            "tr_temp",
            "tr_resp",
            "acuity",
            "visit_datetime",
            "disposition",
            "disposition_time",
            "doctor_id",
            "specialty",
            "admission_date",
            "admission_ward",
            "discharge_ward",
            "discharge_datetime",
            "ed_dx",
            "hopi",
            "ed_dx_hash",
            "systolic_agg",
            "diastolic_agg",
            "temperature_agg",
            "weight_agg",
            "o2sat_agg",
            "nurse_id_agg",
            "hopi_agg",
        ]
    )
    COLUMNS_TO_GROUP: Sequence[str] = field(
        default_factory=lambda: [
            "new_er",
            "new_mr",
            "ed_dx",
            "hopi",
            "gender",
            "age_years",
            "city",
            "triage_datetime",
            "triagecomplaint",
            "bp",
            "tr_pulse",
            "tr_temp",
            "tr_resp",
            "acuity",
            "visit_datetime",
            "disposition",
            "disposition_time",
            "doctor_id",
            "specialty",
            "admission_date",
            "admission_ward",
            "discharge_ward",
            "discharge_datetime",
        ]
    )


@dataclass
class ProcessAdmissionsConfig:
    TABLE_NAME = "admissions"
    SCHEMA_NAME = "processed"

    COLUMNS_TO_KEEP: Sequence[str] = field(
        default_factory=lambda: [
            "new_mr",
            "admission_no",
            "gender",
            "patient_age",
            "patient_type",
            "admission_source",
            "admission_date",
            "discharge_date",
            "stay_duration",
        ]
    )
    DIAGNOSIS_STEMS: Sequence[str] = field(
        default_factory=lambda: [
            "admission_diagnosis",
            "principal_diagnosis",
            "associated_disease",
        ]
    )


@dataclass
class BaselineConfig:
    CONSTRAINT = range(1, 10)
    CATEGORY_COL = "category"
    SCHEMA_NAME = "model_output"
    TEXT_COLS = ["ed_dx"]


@dataclass
class TimeSplitterConfig:
    DATE_COL: str = "triage_datetime"
    TIME_WINDOW_LENGTH: int = 3
    WITHIN_WINDOW_SAMPLER: int = 3
    WINDOW_COUNT: int = 12  # this will increase for more than one split
    SCHEMA_NAME: str = "model_output"
    TABLE_NAME: str = "train"
    TRAIN_VALIDATION_DICT: Dict[str, List[Any]] = field(
        default_factory=lambda: dict(
            validation=[],
            training=[],
        )
    )


@dataclass
class RetrainSplitterConfig:
    DATE_COL: str = "triage_datetime"
    VALIDATION_START: str = "2021-01-01"
    VALIDATION_END: str = "2021-03-31"
    SCHEMA_NAME: str = "model_output"
    TABLE_NAME: str = "retraining"
    TRAIN_VALIDATION_DICT: Dict[str, List[Any]] = field(
        default_factory=lambda: dict(
            validation=[],
            training=[],
        )
    )


@dataclass
class CohortBuilderConfig:
    ENTITY_ID_COLS: Sequence[str] = field(default_factory=lambda: ["unique_id"])
    DATE_COL: str = "triage_datetime"
    TABLE_NAME: str = "train"
    SCHEMA_NAME: str = "model_output"
    FILTER_DICT: Dict[str, Any] = field(
        default_factory=lambda: dict(
            filter_pregnancies=["triagecomplaint"],
            filter_children=["triagecomplaint", "age_years"],
            # filter_non_standard_codes=["category"],
            filter_priority_categories=["category"],
        ),
    )
    PRIORITY_SCHEMA_NAME = "raw"
    PRIORITY_TABLE_NAME = "priority_codes"
    NO_OF_OCCURENCES = 500


@dataclass
class MatrixGeneratorConfig:
    ALGORITHM: str = "sklearn"
    SCHEMA_NAME: str = "model_output"
    COHORT_SPLITS_LIST: Sequence[int] = field(default_factory=lambda: [])


@dataclass
class FeatureGeneratorConfig:
    ID_COLUMN_LIST: Sequence[str] = field(
        default_factory=lambda: [
            "cohort",
            "cohort_type",
            "train_validation_set",
            "unique_id",
        ]
    )
    TEXT_COLUMN_LIST: Sequence[str] = field(
        default_factory=lambda: [
            "hopi",
            "triagecomplaint",
            "ed_dx",
        ]  # "triagecomplaint", "hopi_agg", "ed_dx"
    )
    CAT_COLUMN_LIST: Sequence[str] = field(default_factory=lambda: ["acuity", "gender"])
    CONT_COLUMN_LIST: Sequence[str] = field(
        default_factory=lambda: ["age_years", "tr_pulse", "tr_resp", "tr_temp"]
    )
    SPECIAL_COLUMN_LIST: Sequence[str] = field(
        default_factory=lambda: [
            "past_visit_count",
            "season",
        ]
    )
    EXCLUDE_COLUMN_LIST: Sequence[str] = field(default_factory=lambda: [])

    IGNORE_PUNCTUATION: Dict[str, List[Any]] = field(
        default_factory=lambda: dict(ed_dx=[])
    )

    PROCESS_FUNCTIONS: Dict[str, List[Any]] = field(
        default_factory=lambda: dict(
            ed_dx=[
                "remove_numbers",
                "remove_punctuation",
            ],
            triagecomplaint=[
                "remove_numbers",
                "remove_punctuation",
            ],
            hopi=[
                "remove_numbers",
                "remove_punctuation",
            ],
        )
    )

    TABLE_NAME: str = "train"
    SCHEMA_NAME: str = "model_output"
    ID_VAR: str = "unique_id"

    @property
    def ALL_MODEL_FEATURES(self) -> List[str]:
        """Return all features to be fed into the model"""
        return list(
            set(self.TEXT_COLUMN_LIST + self.CAT_COLUMN_LIST + self.CONT_COLUMN_LIST)
            - set(self.EXCLUDE_COLUMN_LIST)
        )


@dataclass
class LabelGeneratorConfig:
    TARGET_COL: str = "category"
    TRAIN_TABLE_NAME: str = "train"
    TABLE_NAME: str = "labels"
    SCHEMA_NAME: str = "model_output"
    ID_VAR: str = "unique_id"


@dataclass
class ModelTrainerConfig:
    SCHEMA_NAME = "model_output"
    MODEL_NAMES_LIST = ["RFC"]  # "DTC", "MNB", "RFC", "MLR"
    ID_COLS_TO_REMOVE = [
        "unique_id",
        "cohort",
        "cohort_type",
    ]
    RANDOM_STATE = 99
    SCHEMA_NAME: str = "model_output"


@dataclass
class FeatureImportanceConfig:
    NUM_RECORDS: int = 5
    TABLE_NAME: str = "feature_importance"
    SCHEMA_NAME: str = "model_output"


@dataclass
class ModelScorerConfig:
    TABLE_NAME: str = "scores"
    SEED: int = 99


@dataclass
class ModelRankerConfig:
    TABLE_NAME: str = "rank"


@dataclass
class ModelEvaluatorConfig:
    METRICS: Sequence[str] = field(
        default_factory=lambda: ["recall", "precision", "accuracy"]
    )
    CONSTRAINT: Sequence[int] = field(
        default_factory=lambda: [5, 10, 15, 20]  # 3, 5, 8, 10, 12, 15, 18,
    )
    SUMMARY_METHOD = "summary"
    VALID_MODELS: Sequence[str] = field(
        default_factory=lambda: ["DTC", "RFC", "XGB", "MNB", "MLR"]
    )

    TABLE_NAME_PRIORITY_CAT: str = "results_priority_cat_rev"
    PRIORITY_CATEGORIES: Sequence[str] = field(default_factory=lambda: ["a09", "a90"])

    VALIDATION_TABLE_NAME: str = "valid"
    TABLE_NAME: str = "results_rev"
    MICRO_PR_TABLE_NAME: str = "micro_pr_rev"

    SCHEMA_NAME: str = "model_output"
    ID_VAR: str = "unique_id"


@dataclass
class ModelVisualizerConfig:
    PLOT: bool = True
    PLOT_METRICS: Sequence[str] = field(default_factory=lambda: ["mean"])

    PLOTS_TABLE_NAME: str = "plots"
    PLOTS_SCHEMA_NAME: str = "model_output"


@dataclass
class HyperparamConfig:
    MODEL_TYPES = ["DTC", "MNB", "RFC", "XGB"]
    MODEL_HYPERPARAMS = {
        "DTC": {
            "max_depth": [5, 10, 20, 30, 40]
        },  # 5, 50, 500, 10000 50, 100, 200, 300
        "RFC": {
            "n_estimators": [500, 800],  # 100, 500, 800, 1000
            "max_depth": [10, 50, 70],  # 5, 50, 80, 500, 10000  100, 200, 300
        },
        "XGB": {"max_depth": [5, 150, 200, 250, 300], "learning_rate": [0.1, 0.5, 1]},
        "MNB": {"alpha": [0, 0.05]},  # 0.1, 0.5, 0.8, 1
        "MLR": {
            "penalty": ["l2"],
            "C": [1, 0.1, 0.01],
            "solver": ["saga"],
            "max_iter": [2000],
        },
    }


@dataclass
class RetrainingConfig:
    BEST_MODEL: str = "RFC"
    BEST_MODEL_HYPERPARAMS: tuple = (800, 300)
    CONSTRAINT: int = 10
    PREDICTIONS_TABLE_NAME: str = "predictions"
    SCHEMA_NAME: str = "model_output"
