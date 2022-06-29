from typing import List

import mlflow.pyfunc

import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression


class WorkshopModel(mlflow.pyfunc.PythonModel):
    def __init__(self, fit_intercept: bool, model_features: List):
        self.model_threshold = 0.5
        self.model_features = model_features
        self.model = LogisticRegression(fit_intercept=fit_intercept)

    @staticmethod
    def preprocess(data: pd.DataFrame, model_features: List) -> pd.DataFrame:
        data ['avg_likes'] = data['avg_likes'].astype(int)
        data["approval_time"] = (data["approved_at"] - data["created_at"])

        data["approval_time_mins"] = data["approval_time"].astype('timedelta64[m]')

        data["6h_approval"] = np.where(data["approval_time_mins"]<=360.0,1,0)

        data = data.assign(is_referred=data['referral_code'].notnull().astype(int))

        top5_states = ['California','New York','Florida','Gerogia','Texas']
        data["region_clean"] = np.where(data['region'].isin(top5_states), data['region'], 'Other')
        top2_states = ["California","New York"]
        data["region_clean"] = np.where(data['region_clean'].isin(top2_states), data['region_clean'], 'Other')
        region_cat = pd.get_dummies(data["region_clean"], drop_first=True).add_prefix('region_clean_')
        region_cat.columns = [c.replace(' ', '_') for c in region_cat.columns]
        data = pd.concat([data, region_cat], axis=1)

        tmz = "US/Central"
        data = data.assign(partner_created_hour_us_central=data["created_at"].dt.tz_localize(None).dt.tz_localize("utc").dt.tz_convert(tmz).dt.hour)
        data = data.assign(partner_approved_hour_us_central=data["approved_at"].dt.tz_localize(None).dt.tz_localize("utc").dt.tz_convert(tmz).dt.hour)

        high_hours = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        data["day_approval"] = np.where(data["partner_approved_hour_us_central"].isin(high_hours),1,0)

        data["day_approved_at"] = data['approved_at'].dt.tz_localize(None).dt.tz_localize("utc").dt.tz_convert(tmz).dt.dayofweek
        thu_fri = [3.0, 4.0]
        data["thu_fri_approval"] = np.where(data['day_approved_at'].isin(thu_fri), 1, 0)

        data["followers_count"] = pd.to_numeric(data["followers_count"], downcast="float")
        data["posts_count"] = pd.to_numeric(data["posts_count"], downcast="float")

        data["ln_followers_count"] = np.log(data["followers_count"].replace(0, 1))
        data["ln_posts_count"] = np.log(data["posts_count"].replace(0, 1))

        return data[model_features]

    def load_context(self, context):
        # We're required to override method, but we don't use context in this model so noop
        pass

    def predict(self, context, features: pd.DataFrame) -> pd.DataFrame:
        features_preprocessed = self.preprocess(features, self.model_features)
        result = self.predict_preprocessed(features_preprocessed)

        return result

    def predict_preprocessed(self, features_preprocessed: pd.DataFrame) -> pd.DataFrame:
        proba = self.model.predict_proba(features_preprocessed)
        result = pd.DataFrame({"probability": proba[:, 1]}, index=features_preprocessed.index)
        result['decision'] = result.apply(lambda p: p['probability'] > self.model_threshold, axis=1)

        return result

    def fit(self, train_x: pd.DataFrame, train_y: pd.Series) -> ClassifierMixin:       
        train_x_preprocessed = self.preprocess(train_x, self.model_features)
        return self.fit_preprocessed(train_x_preprocessed, train_y)

    def fit_preprocessed(self, train_x: pd.DataFrame, train_y: pd.Series) -> ClassifierMixin:
        return self.model.fit(train_x, train_y)

    def get_classifier(self) -> ClassifierMixin:
        return self.model