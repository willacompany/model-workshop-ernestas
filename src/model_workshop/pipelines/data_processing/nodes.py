import pandas as pd
import numpy as np
from typing import Dict, Any

def preprocess(data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """ Node for basic feature engineering and normalization """

    data['avg_likes'] = data['avg_likes'].astype(int)
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
    thu_fri = [3.0,4.0]
    data["thu_fri_approval"] = np.where(data['day_approved_at'].isin(thu_fri), 1, 0)

    data["followers_count"] = pd.to_numeric(data["followers_count"], downcast="float")
    data["posts_count"] = pd.to_numeric(data["posts_count"], downcast="float")

    data["ln_followers_count"] = np.log(data["followers_count"].replace(0, 1))
    data["ln_posts_count"] = np.log(data["posts_count"].replace(0, 1))

    data = data[data['created_at'] >= parameters['partner_created_at_min']]

    target = parameters['target']
    data[target] = data[target].astype(int)

    return data
