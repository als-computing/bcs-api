#!/usr/bin/env python

"""bcs_events.py

    These helper functions were extracted from 'xmcd_in.py', a script that
    reads data that was saved during BCS Trajectory Scans, then generates
    documents in a format compatible with the databroker-bluesky Data/Event 
    model.
    
    An example under development to ingest XMCD or XMLD data into a MongoDB 
    server, using a format compatible with the databroker-bluesky Data/Event 
    model.
    
    TODO: These functions could/should be split from the 'xmcd_in.py' script
    and made available for general utility.

    XMCD Ingestor example for flying BCS Trajectory Scans (array events):
    https://github.com/als-computing/xmcd_ingestor/blob/4f9150984666492e2b14a49339a5257305110eca/ingestors/xmcd_in.py#L503

    XMCD Ingestor example for BCS Trajectory Scans (1 event per reading):
    https://github.com/als-computing/xmcd_ingestor/blob/614980e14267d8cd27ecdf8de04fb4860d25f3aa/ingestors/xmcd_in.py#L369
    ...works for flying or stepped scans
    
    
    https://nsls-ii.github.io/architecture-overview.html
    https://docs.google.com/document/d/1vC-EPNhYojh2k2WwXxBTlSlKiUsYOrOSrM4WDlHDCds/edit#
"""

import logging

logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

# from dotenv import load_dotenv
# import sys
# import os
# import argparse

# from pymongo import MongoClient
# import urllib.parse

# from suitcase.mongo_normalized import Serializer as MongoS
import event_model

from datetime import datetime, date, time, timedelta
from dateutil import relativedelta as rel_date
import pytz

from numpy import array
import numpy as np
# import glob
import pandas as pd

# import uuid

# from bcs_data import get_data_file_numbers, read_data_file
# from bcs_find import find_data_files_in_date_range, replace_subpath
# from bcs_scans import get_scan_file_path, import_scan_file

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_descriptor_keys(data_df, sanitize_event_data_keys, data_src="BCS"):
    """Create Databroker-style Event descriptor keys from BCS data file.
        
        data_df: PANDAS dataframe, extracted from BCS data file
        sanitize_event_data_keys: Translate BCS data keys to 
            databroker-compatible event data keys; dict-like
        data_src: Source of data; str 
         
        RETURN: Databroker-style event descriptor; dict-like
    """
    default_signal = {
        'dtype': 'number',
        # 'source': data_df.iloc[0]["filename"],  # Assume all rows from same file
        'source': data_src, 
        'shape': [],
        'units': '',
    }
    
    descriptor_keys = {signal_name : default_signal.copy() 
        for signal_name in data_df.columns[1:].values}
    
    for key in descriptor_keys.keys():
        
        if (
           (key == "I0 BL") or (key == "I0 ES") or
           (key == "EY") or (key == "LY") or (key == "FY") or 
           (key == "EY SCVM") or (key == "LY SCVM") or 
           (key == "LY SCVM (Original)")
           ):
            
            descriptor_keys[key]["units"] = "counts / sec"
            # continue
        
        # if (key == "Clock") or (key == "Original Clock"):
        if ("Clock" in key):

            descriptor_keys[key]["units"] = "counts"
            # continue
        
        # if key.contains("Energy"):
        if ("Energy" in key):
            
            descriptor_keys[key]["units"] = "eV"
            # continue
        
        # if key.contains("Grating") or key.contains("Premirror") or key.contains("Jack"):
        if ("Grating" in key) or ("Premirror" in key) or ("Jack" in key):
            
            descriptor_keys[key]["units"] = "um"
            # continue
        
        # if key.contains("Gap") or key.contains("EPU Z") or key.contains("Phase"):
        # if key.contains("EPU A") or key.contains("EPU B"):
        if (
           ("Gap" in key) or ("EPU Z" in key) or ("Phase" in key) or
           ("EPU A" in key) or ("EPU B" in key)
           ):
            
            descriptor_keys[key]["units"] = "mm"
            # continue
        
        if (
           (key == "Hx") or (key == "Hy") or (key == "Hz") or
           (key == "XMLD H") or (key == "XMLD Theta") or (key == "XMLD Phi") or 
           (key == "XMCD H") or (key == "XMCD Phi")
           ):
            
            descriptor_keys[key]["units"] = "T"
            # continue
        
        # if key.contains("Temp"):
        if ("Temp" in key):
            
            descriptor_keys[key]["units"] = "K"
            # continue
        
        if (key == "X") or (key == "Y") or (key == "Z"):
            
            descriptor_keys[key]["units"] = "mm"
            # continue
        
        if (key == "Theta") or (key == "Azimuth"):
            
            descriptor_keys[key]["units"] = "degrees"
            # continue
        
        # if key.contains("Pitch") or key.contains("Roll") or key.contains("Yaw"):
        if ("Pitch" in key) or ("Roll" in key) or ("Yaw" in key):
            
            descriptor_keys[key]["units"] = "mm"
            # continue
        
        # if key.contains("Amp (nA)"):
        if ("Amp (nA)" in key):
            
            descriptor_keys[key]["units"] = "nA / V"
            # continue
        
        # if key.contains("Slit"):
        if ("Slit" in key):
            
            descriptor_keys[key]["units"] = "um"
            # continue
        
        # if key.contains("Current"):
        if ("Current" in key):
            
            descriptor_keys[key]["units"] = "mA"
            # continue
        
        # if key.contains("Velocity"):
        if ("Velocity" in key):
            
            descriptor_keys[key]["units"] = "mm / sec"
            # continue
        
        # if key.contains("Stepper") or key.contains("Encoder"):
        if ("Stepper" in key) or ("Encoder" in key):
            
            descriptor_keys[key]["units"] = "counts"
            # continue
        
        # if key.contains("in position"):
        if ("in position" in key):
            
            # Theoretically a boolean,  
            #   but can be non-integer (indicating error in data collection)
            # descriptor_keys[key]["dtype"] = "boolean"
            descriptor_keys[key]["units"] = ''
            # continue
        
        # if key.contains("filename"):
        if ("filename" in key):
            
            descriptor_keys[key]["dtype"] = "string"
            descriptor_keys[key]["units"] = ''
            # continue
        
        # if key.contains("Time (s)"):
        if ("Time (s)" in key):
            
            descriptor_keys[key]["units"] = "sec"
            # continue
        
    # descriptor_keys = {sanitize_key(key):value for (key, value) in descriptor_keys.items()}
    descriptor_keys = {sanitize_event_data_keys[key]:value 
        for (key, value) in descriptor_keys.items()}
    
    return descriptor_keys

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_event_data_keys(data_row, sanitize_event_data_keys):
    """Create Databroker-style Event descriptor keys from BCS data file.
        
        data_row: PANDAS dataframe row, extracted from BCS data file
        sanitize_event_data_keys: Translate BCS data keys to 
            databroker-compatible event data keys; dict-like
        
        RETURN: Databroker-style event data keys; dict-like
    """
    row = data_row
    
    event_keys = {sanitize_event_data_keys[signal]: row[signal] 
        for signal in row.index[1:].values}
    
    return event_keys

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_array_event_data_keys(data_df, timestamp, sanitize_event_data_keys):
    """Create Databroker-style Event descriptor keys from BCS data file.
        
        data_df: PANDAS dataframe, extracted from BCS data file
        timestamp: PANDAS dataframe, extracted from BCS data file
        sanitize_event_data_keys: Translate BCS data keys to 
            databroker-compatible event data keys; dict-like
        
        RETURN: Databroker-style event data keys; dict-like
    """
    df = data_df
    
    event_keys = {
        sanitize_event_data_keys[signal]: df[signal].values 
        for signal in df.columns.values
        }
    
    return event_keys

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def sanitize_key(data_name):
    """Replace illegal characters in Databroker-style Event descriptor key 
        from BCS data file.
        
        data_name: Data column name, extracted from BCS data file
        
        RETURN: Databroker-compatible event data key
    """
    
    # Rename descriptors that have invalid keys
    return data_name.replace(
        '/', " div ").replace('.', " dot ").replace('^', " hat ")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def add_timestamps(
        df, 
        data_date, 
        time_col="Time of Day", 
        output_col="timestamp", 
        inplace=True):
    """Create timestamps from BCS data file imported by PANDAS.
        
        df: PANDAS dataframe
        data_date: start date of BCS data file; datetime.date object
        time_col: name of column in df with BCS time of day
        output_col: name of [new] column to store timestamps
        inplace: True == modify df; False == return a modified copy of df
        
        RETURN: PANDAS dataframe, updated with timestamps in output_col 
    """
    if not inplace:
        output_df = df.copy()
    else:
        output_df = df
        
    # Check for new day (rollover) during data aquistion in file
    times = array([datetime.strptime(time_val, "%H:%M:%S").time() 
        for time_val in df[time_col]])
    times_padded = np.empty(len(times) + 1, dtype="object")
    times_padded[0] = time(0)
    times_padded[1:] = times
    # time_diffs = np.diff(times_padded)
    # time_diff_signs = np.sign(time_diffs)
    new_day_events = np.where(times < times_padded[:-1])
    # same_day_events = np.where(times >= times_padded[:-1])
    new_days = np.zeros(len(times), dtype="float")
    new_days[new_day_events] = 1
    extra_days = np.cumsum(new_days)
    
    dates = np.full(len(times), data_date)
    dates = [date_val + timedelta(days=num_days) 
        for (date_val, num_days) in zip(dates, extra_days)]
    
    # Combine dates & times; add TZ awareness
    timezone = pytz.timezone('America/Los_Angeles')
    datetimes = [
        timezone.localize(datetime.combine(date_val, time_val)).timestamp() 
        for (date_val, time_val) in zip(dates, times)
        ]

    assert(len(extra_days) == len(times))
    assert(len(dates) == len(times))
    assert(len(datetimes) == len(times))
    
    output_df[output_col] = datetimes
    
    return output_df

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_timestamps(
        df, 
        data_date, 
        time_col="Time of Day", 
        output_col="timestamp", 
        inplace=True):
    """Create timestamps from BCS data file imported by PANDAS.
        
        df: PANDAS dataframe
        data_date: start date of BCS data file; datetime.date object
        time_col: name of column in df with BCS time of day
        output_col: name of [new] column to store timestamps
        inplace: True == modify df; False == return a modified copy of df
        
        RETURN: Databroker-style event; dict-like
    """
    if not inplace:
        output_df = df.copy()
    else:
        output_df = df
        
    # Check for new day (rollover) during data aquistion in file
    times = array([datetime.strptime(time_val, "%H:%M:%S").time() 
        for time_val in df[time_col]])
    times_padded = np.empty(len(times) + 1, dtype="object")
    times_padded[0] = time(0)
    times_padded[1:] = times
    # time_diffs = np.diff(times_padded)
    # time_diff_signs = np.sign(time_diffs)
    new_day_events = np.where(times < times_padded[:-1])
    # same_day_events = np.where(times >= times_padded[:-1])
    new_days = np.zeros(len(times), dtype="float")
    new_days[new_day_events] = 1
    extra_days = np.cumsum(new_days)
    
    dates = np.full(len(times), data_date)
    dates = [date_val + timedelta(days=num_days) 
        for (date_val, num_days) in zip(dates, extra_days)]
    
    # Combine dates & times; add TZ awareness
    timezone = pytz.timezone('America/Los_Angeles')
    datetimes = [
        timezone.localize(datetime.combine(date_val, time_val)).timestamp() 
        for (date_val, time_val) in zip(dates, times)
        ]

    assert(len(extra_days) == len(times))
    assert(len(dates) == len(times))
    assert(len(datetimes) == len(times))
    
    output_df[output_col] = datetimes
    
    return output_df

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_event(
        data_row, 
        sanitize_event_data_keys, 
        stream_bundle, 
        timestamp_col="timestamp"):
    """Create Databroker-style Event from BCS data file row.
        
        data_row: PANDAS dataframe row, extracted from BCS data file
        sanitize_event_data_keys: Translate BCS data keys to 
            databroker-compatible event data keys; dict-like
        stream_bundle: event_model stream bundle to which the event 
            will be sent
        timestamp_col: name of column in data_row with event timestamp
        
        RETURN: Databroker-style event; dict-like
    """
    timestamp_val = data_row[timestamp_col]
    row = data_row.drop(timestamp_col)
    
    data_dict = get_event_data_keys(row, sanitize_event_data_keys)
    timestamp_dict = {key: timestamp_val for key in data_dict.keys()}

    event = stream_bundle.compose_event(
        data = data_dict,
        timestamps = timestamp_dict,
        )
    
    return event

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_array_event(
        data_df, 
        timestamp,
        sanitize_event_data_keys, 
        stream_bundle, 
        **kwargs):
    """Create Databroker-style Event from BCS data file row.
        
        data_df: PANDAS dataframe, extracted from BCS data file
        timestamp: One value, applies to each row in data_df; float64
        sanitize_event_data_keys: Translate BCS data keys to 
            databroker-compatible event data keys; dict-like
        stream_bundle: event_model stream bundle to which the event 
            will be sent
        
        RETURN: Databroker-style event; dict-like
    """
    timestamp_val = timestamp
    
    data_dict = get_array_event_data_keys(
        data_df, timestamp_val, sanitize_event_data_keys)
    timestamp_dict = {key: timestamp_val for key in data_dict.keys()}

    event = stream_bundle.compose_event(
        data = data_dict,
        timestamps = timestamp_dict,
        )
    
    return event

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def prepend_dimension_to_descriptor_key(
        key, 
        size,
        dim_name=None,
        inplace=False,
        **kwargs):
    """Prepend an extra dimension of known size to event descriptor key
        
        key: existing data key properties from an event descriptor
        size: size of the new dimension
        dim_name: name of the new dimension
        inplace: True == Update the key provided; False == create new key
        
        RETURN: Updated descriptor key
    """
    if inplace:
        new_key = key
    else:
        new_key = key.copy()
    
    old_size = key.get("shape", [])
    new_size = [size] + old_size
    new_key["shape"] = new_size

    if dim_name:
        old_dims = key.get("dims", [])
        new_dims = [dim_name] + old_dims
        new_key["dims"] = new_dims     
    
    return new_key

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_empty_config_key(
        object_name,
        key_names,  # list[dict[str, str]]
        metadata={},
        **kwargs):
    """Prepend an extra dimension of known size to event descriptor key
        
        object_name: name of configuration object key; string
        key_names: list of dictionaries, each containing:
            name: name of configuration data key; string
            dtype: valid datatype for event model data key; string
                   OPTIONAL
        metadata: metadata provided by ingestor
        
        RETURN: configuration key for event descriptor; dict-like
    """
    config_key = {
        f"{object_name}": {
            "data": {f"{key['name']}": "" for key in key_names},
            "data_keys": {
                f"{key['name']}": {
                    "dtype": f"{key.get('dtype', 'string')}",
                    "shape": [],
                    "source": metadata["ingestor"],
                    "units": "",
                    } for key in key_names
                },
            "timestamps": {f"{key['name']}": 0 for key in key_names}
            }
        }     
    
    return config_key

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import types
__all__ = [name for name, thing in globals().items()
            if not (
              name.startswith('_') or 
              isinstance(thing, types.ModuleType) or 
              # isinstance(thing, types.FunctionType) or 
              isinstance(thing, type) or  # Class type
              isinstance(thing, dict) 
              )
            ]
del types
