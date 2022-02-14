import datetime
from pathlib import Path

import pandas as pd

from functions.progress import ProgressShower
from ... import constants as sc

fd = Path('results/gcs_datasets/bpic2019')
fn_in = fd / 'raw.csv'
fn_out = fd / 'full.csv'
fn_out_events = fd / 'events.csv'
fn_out_cases = fd / 'cases.csv'

# Import data, rename columns, remove unused columns
rem_cols = ['(case) Item',
            '(case) Purchasing Document',
            '(case) Name',
            '(case) Vendor',
            '(case) Sub spend area text',
            '(case) Source',
            '(case) Purch. Doc. Category name']

df = pd \
    .read_csv(fn_in) \
    .rename(columns={'Activity': sc.ACTIVITY, 'Case ID': sc.CASE, 'Complete Timestamp': sc.TIMESTAMP}) \
    .drop(columns=['Resource', 'Variant', 'Variant index', 'Cumulative net worth (EUR)', 'User']) \
    .drop(columns=rem_cols)


# Clean column names ===================================================================================================
def clean(x):
    if x.startswith('(case) '):
        return x[7:]
    else:
        assert x in [sc.ACTIVITY, sc.CASE, sc.TIMESTAMP]
        return x


df.columns = map(clean, df.columns)

# Convert timestamps ===================================================================================================
df[sc.TIMESTAMP] = pd.to_datetime(df[sc.TIMESTAMP])

# Remove cases that are (partly) outside of the frame ==================================================================
start_date = datetime.date(2018, 1, 1)
end_date = datetime.date(2019, 1, 1)
first_activity = 'Vendor creates invoice'
second_activity = 'Record Invoice Receipt'


# TODO can probably be more efficient, but at least it is correct
def is_valid(x):
    case_df = x[1]
    if (case_df[sc.TIMESTAMP].dt.date < start_date).any():
        return False

    if (case_df[sc.TIMESTAMP].dt.date >= end_date).any():
        return False

    if (case_df[sc.ACTIVITY] == first_activity).sum() != 1:
        return False

    if (case_df[sc.ACTIVITY] == second_activity).sum() != 1:
        return False

    # We can take the min, as we are sure the required activities occur exactly once
    gb = case_df.groupby(sc.ACTIVITY)[sc.TIMESTAMP].min()
    if gb[first_activity] > gb[second_activity]:
        return False

    return True


valid_cases = list(map(lambda x: x[0], (filter(is_valid, ProgressShower(df.groupby(sc.CASE), 'Filtering cases')))))

df = df[df[sc.CASE].isin(valid_cases)]

fn_out.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(fn_out, index=False)
df.loc[:, [sc.CASE, sc.TIMESTAMP, sc.ACTIVITY]].to_csv(fn_out_events, index=False)
df.drop(columns=[sc.TIMESTAMP, sc.ACTIVITY]).drop_duplicates().set_index(sc.CASE).fillna('Empty').to_csv(fn_out_cases)
