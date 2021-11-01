#!/usr/bin/env python

"""bcs_api.py

    These are a collection of helper functions and classes that support
    connecting the LabView Beamline Control System API (BCS-API) 
    to the `bluesky` family of experiment orchestration tools.

    Each BCS-API scan is wrapped as as an `ophyd` device that implements
    the "Fly-able Interface" (e.g., FlyerInterface); it is a "flyer".
    
    The bluesky fly plan will cede control to an ophyd flyer to collect 
    data asynchronously, and then return data as "event" documents.
    
    References:
    * http://nsls-ii.github.io/ophyd/architecture.html#fly-able-interface
    * https://github.com/bluesky/ophyd/blob/dd4b3e389a0202ecacce39fc3965d703c616b0d4/ophyd/flyers.py#L17
    * https://blueskyproject.io/event-model/data-model.html#event-document

    Credits:
    This module builds heavily upon two recently published works, 
    as well as the many developers and contributors to the bluesky 
    and BCS projects.

    * BCS-API: ZeroMQ library and python bindings (BCSz.py) developed 
      by Damon English for interfacing with the Advanced Light Source's 
      BCS control system.
    * sscan as 1D Flyer example published by Pete Jemian to demonstrate 
      how the bluesky fly plan can be used to initiate scans and report 
      data from external control systems (in that case, the EPICS sscan 
      record).
    * References:
        * http://bcsapi.als.lbl.gov:3080/
        * https://github.com/daenglis
        * https://github.com/BCDA-APS/bluesky_training/blob/31-sscan-1D-as-flyer/sscan_1d_flyer.ipynb
        * https://github.com/prjemian
"""

import logging

logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

from warnings import warn

# These are necessary to use the API
import asyncio
# import BCSz
import BCSz_sync as BCSz

# The rest of the imports are for the Example program
import time
import random

# for working with images
from PIL import Image
import io
from IPython.display import display # to display images

# for plotting and working with files
import pandas as pd
import matplotlib.pyplot as plt
import os

from numpy import nan
from numpy import cumsum, roll
from numpy import array

import itertools as it

from ophyd import Device, Component, Signal
from ophyd.status import DeviceStatus, SubscriptionStatus
from ophyd.flyers import FlyerInterface

# Helper functions for converting BCS data to Bluesky event model data
from bcs_events import *

from datetime import datetime
import event_model

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def find_bcs_data(data):
    """
    Keep trying to import the data into pandas until success
    The older BCS scans (so called 'Beamline Scans') have a variably sized header.
    """
    for skrows in range(30):
        data.seek(0)
        try:
            df = pd.read_csv(data, sep='\t', skiprows=skrows, nrows=10)
            df['Time (s)']
            break
        except (pd.errors.ParserError, KeyError):
            pass # try again with skrows +=1

    data.seek(0)
    return skrows

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_bcs_file(file_text='', plot_title='', x='Time (s)', y=None):
    """Display example plot for API testing"""
    data = io.StringIO(file_text)
    skip_rows = find_bcs_data(data)
    df = pd.read_csv(data, sep='\t', skiprows=skip_rows)
    print(df.columns)
    df.drop(0, inplace=True)
    print(df.head())

    if not y:
        y = random.choice(df.columns)

    df.plot(kind='scatter', x=x, y=y)
    plt.title(plot_title)
    plt.show()
    return df

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CLASSES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class BcsSingleMotorScan(Device):
    """
    BCS Single Motor Scan (Move-Delay-Acquire) as an Ophyd device
    """
    
    # Signals to enable subscriptions for scan setup
    motor_name = Component(Signal, kind="config", value='')
    first_value = Component(Signal, kind="config", value=nan)
    last_value = Component(Signal, kind="config", value=nan)
    step_value = Component(Signal, kind="config", value=nan)
    delay_sec = Component(Signal, kind="config", value=0.)
    count_sec = Component(Signal, kind="config", value=1.)
    num_scans = Component(Signal, kind="config", value=1)
    bidirectional = Component(Signal, kind="config", value=False)
    final_move = Component(Signal, kind="config", value="Stay")
    memo = Component(Signal, kind="config", value='')
    filename_pattern = Component(Signal,kind="config", value="*.txt")
    
    # Signals to enable subscriptions/polling for scan status
    ready = Component(Signal, kind="omitted", value=False)
    busy = Component(Signal, kind="omitted", value=False)
    done = Component(Signal, kind="omitted", value=True)
    execute_scan = Component(Signal, kind="omitted", value=True)
    data_path = Component(Signal, kind="normal", value='')


    def scan_setup(
            self, *, 
            # server: str, port: int, 
            bcs_server: BCSz.BCSServer, 
            motor: str, start: float, stop: float, step: float, 
            delay: float = 0., count: float = 1., 
            num_scans: int = 1, bidirect: bool = False, 
            final: str = "Stay",
            memo: str = '',
            filename_pattern: str = "*.txt", 
            **kwargs):
        """Configure a BCS Single Motor Scan"""
        
        # TODO: Check that bcs_server is connected
        self._bcs = bcs_server
        
        self.motor_name.put(motor)
        self.first_value.put(start)
        self.last_value.put(stop)
        self.step_value.put(step)
        self.delay_sec.put(delay)
        self.count_sec.put(count)
        self.num_scans.put(num_scans)
        self.bidirectional.put(bidirect)
        self.final_move.put(final)
        self.memo.put(memo)
        self.filename_pattern.put(filename_pattern)
        
        # Scan is configured and ready to execute
        self.ready.put(True)


    def set(self, value, **kwargs):
        """interface to use bps.mv()"""
        if value != 1:
            return
        if self.ready.get():
            bcs_st = self._bcs.scan_status()
            self.busy.put(bcs_st["running_scan"])
        if (not self.ready.get()) or (self.busy.get()):
            # TODO: Raise Warning that scan is not ready
            return

        async def check_for_acquire_done(self):
            bcs_st = self._bcs.scan_status()
            while bcs_st["running_scan"]:
                await asyncio.sleep(0.1)
                bcs_st = self._bcs.scan_status()
            else:
                # bcs_st = self._bcs.scan_status()
                print("Scan is finshed!")
                # print(f"{bcs_st =}...")
                self.data_path.put(os.path.join(bcs_st["Log Directory"], bcs_st["Last Filename"]))
                # print(f"{self.data_path.get() =}")
                self.done.put(True)
                self.busy.put(False)
                
        def check_value(*, old_value, value, **kwargs):
            "Return True when the acquisition is complete, False otherwise."
            return (value and not old_value)
        
        status = SubscriptionStatus(self.done, check_value)
        
        self.execute_scan.put(True)
        call_status = self._bcs.sc_single_motor_scan(
            x_motor=self.motor_name.get(), 
            start=self.first_value.get(), 
            stop=self.last_value.get(), 
            increment=self.step_value.get(), 
            delay_after_move_s=self.delay_sec.get(), 
            count_time_s=self.count_sec.get(), 
            number_of_scans=self.num_scans.get(),
            bidirect=self.bidirectional.get(),
            at_end_of_scan=self.final_move.get(), 
            description=self.memo.get(),
            file_pattern=self.filename_pattern.get(),
            )
        if not call_status["success"]:
            print(f"{call_status = }")
            # TODO: Raise exception; reset state?
        self.busy.put(True)
        self.done.put(False)
        # Give the scanner time to start
        time.sleep(1)  # TODO: Is this needed?
        self.execute_scan.put(False)
        acquire_task = asyncio.create_task(check_for_acquire_done(self))
        # acquire_task.add_done_callback()
        
        return status

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class BcsSigScanFlyer(FlyerInterface, BcsSingleMotorScan):
    '''Example of BCS Single Motor Scan accessed through Bluesky'''
    
    yield_array_events = Component(Signal, kind="config", value=False)

    def __init__(self, *args, **kwargs):
        self._acquiring = False
        self._paused = False

        super().__init__(*args, **kwargs)
        
    def stage(self):
        super().stage()
        # self.select_channels()

    def unstage(self):
        super().unstage()
        # self.select_channels()

    def read_configuration(self):
        return {}

    def describe_configuration(self):
        return {}

    def kickoff(self):
        """Start the scan."""
        self.stage()
        time.sleep(0.1)

        # Check for currently running scan
        bcs_st = self._bcs.scan_status()
        if bcs_st["running_scan"]:
            self.unstage()
            raise RuntimeError("Cannot start scan. Another scan is already running.")
        
        self.set(1)
        self._acquiring = True
        self._paused = False

        status = DeviceStatus(self)
        status.set_finished()  # means that kickoff was successful
        return status

    def complete(self):
        """Wait for sscan to complete."""
        logging.info("complete() starting")
        if not self._acquiring:
            raise RuntimeError("Not acquiring")

        # status = DeviceStatus(self)
        # cb_started = False

        def is_scan_complete(*, old_value, value, **kwargs):
            "Return True when the acquisition is complete, False otherwise."
            # value = bool(value)
            if self._acquiring and value and not old_value:
                logging.info("complete() ending")
                self.unstage()
                self._acquiring = False
                return True
            return False

        status = SubscriptionStatus(self.done, is_scan_complete)
        
        return status

    def describe_collect(self):
        """
        Provide schema & meta-data from collect().
        
        http://nsls-ii.github.io/ophyd/generated/ophyd.flyers.FlyerInterface.describe_collect.html
        """
        # TODO: Add hinted signals
        
        ai_names = self._bcs.list_ais()['names']
        
        scan_motor = self.motor_name.get()
        scan_channels = [
            "Time of Day", 
            "Time (s)", 
            f"{scan_motor} Goal", 
            f"{scan_motor} Actual", 
            ] + ai_names
      
        # DataFrame enables convenience functions from 'bcs_events'
        data_df = pd.DataFrame(columns=scan_channels)

        sanitize_event_data_keys = {col: sanitize_key(col) 
            for col in data_df.columns[1:].values}

        descriptor_keys = get_descriptor_keys(
            data_df, 
            sanitize_event_data_keys,
            data_src="Inferred from AI List")
        
        event_stream_name = "primary"
        
        return {event_stream_name: descriptor_keys}

    def collect(self):
        """
        Retrieve all collected data (after complete()).
        
        Retrieve data from the flyer as proto-events.
        http://nsls-ii.github.io/ophyd/generated/ophyd.flyers.FlyerInterface.collect.html
        """
        if self._acquiring:
            raise RuntimeError("Acquisition still in progress. Call complete() first.")
        
        def get_data_from_scan(self):
            """Extract data from scan output file; return as PANDAS DataFrame."""
            file_path = self.data_path.get()
            if not file_path:
                raise RuntimeError("There is no scan data. Call kickoff() first.")
            
            file_text = self._bcs.get_text_file(file_path)['text']

            # Get date from BCS data file header
            with io.StringIO(file_text) as data_file:
                data_file_date_str = data_file.readline().strip().split("Date: ", 1)[1]
                data_date = datetime.strptime(data_file_date_str, "%m/%d/%Y").date()
                self._data_date = data_date
                
            data = io.StringIO(file_text)
            skip_rows = find_bcs_data(data)
            df = pd.read_csv(data, sep='\t', skiprows=skip_rows)
            return df

        def generate_scan_data_events(self):
            """Get the entire scan data and yield bluesky events."""
            data_df = get_data_from_scan(self)
            num_points = len(data_df)
            
            if self.yield_array_events.get():
                raise NotImplementedError("Array events not currently supported")
                # TODO: Implement array event generation

            # Run info will not be used by fly() plan
            # ...enables convenience functions from 'bcs_events'
            run_bundle = event_model.compose_run()
            event_stream_name = 'primary'

            sanitize_event_data_keys = {col: sanitize_key(col) 
                for col in data_df.columns[1:].values}

            descriptor_keys = get_descriptor_keys(
                data_df, 
                sanitize_event_data_keys,
                data_src=self.data_path.get())
    
            stream_descriptor = run_bundle.compose_descriptor(
                data_keys=descriptor_keys,
                name=event_stream_name,
                )

            # Get date from BCS data file header
#             with io.StringIO(self.data_path.get()) as data_file:
#                 data_file_date_str = data_file.readline().strip().split("Date: ", 1)[1]
#                 data_date = datetime.strptime(data_file_date_str, "%m/%d/%Y").date()
            data_date = self._data_date

            # This is only for array events
#             data_df = add_timestamps(data_df, data_date, inplace=True)
#             data_df.drop('Time of Day', axis=1, inplace=True)  # Redundant; have timestamps
#             timestamp_col="timestamp"
    
            data_df = get_timestamps(data_df, data_date, inplace=True)

            def get_bundled_event(data_row):
                return get_event(
                    data_row, sanitize_event_data_keys, stream_descriptor)

            # Pack events into an event_page
            events = data_df.apply(get_bundled_event, axis='columns').values

            # yield 'event_page', event_model.pack_event_page(*events)
            
            for event in events:
                # yield 'event', event
                yield dict(
                    seq_num=event["seq_num"],
                    time=event["time"],
                    data=event["data"],
                    timestamps=event["timestamps"],
                )

        yield from generate_scan_data_events(self)
        self.unstage()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class BcsSingleMotorFlyingScan(Device):
    """
    BCS Single Motor Flying Scan (continuous acquisition) as an Ophyd device
    """
    
    # Signals to enable subscriptions for scan setup
    motor_name = Component(Signal, kind="config", value='')
    first_value = Component(Signal, kind="config", value=nan)
    last_value = Component(Signal, kind="config", value=nan)
    step_value = Component(Signal, kind="config", value=nan)
    velocity = Component(Signal, kind="config", value=0.)
    num_scans = Component(Signal, kind="config", value=1)
    bidirectional = Component(Signal, kind="config", value=False)
    final_move = Component(Signal, kind="config", value="Stay")
    memo = Component(Signal, kind="config", value='')
    filename_pattern = Component(Signal,kind="config", value="*.txt")
    
    # Signals to enable subscriptions/polling for scan status
    ready = Component(Signal, kind="omitted", value=False)
    busy = Component(Signal, kind="omitted", value=False)
    done = Component(Signal, kind="omitted", value=True)
    execute_scan = Component(Signal, kind="omitted", value=True)
    data_path = Component(Signal, kind="normal", value='')


    def scan_setup(
            self, *, 
            # server: str, port: int, 
            bcs_server: BCSz.BCSServer, 
            motor: str, start: float, stop: float, step: float, 
            velocity: float = 0., 
            num_scans: int = 1, bidirect: bool = False, 
            final: str = "Stay",
            memo: str = '',
            filename_pattern: str = "*.txt", 
            **kwargs):
        """Configure a BCS Single Motor Flying Scan"""
        
        # TODO: Check that bcs_server is connected
        self._bcs = bcs_server
        
        self.motor_name.put(motor)
        self.first_value.put(start)
        self.last_value.put(stop)
        self.step_value.put(step)
        self.velocity.put(velocity)
        self.num_scans.put(num_scans)
        self.bidirectional.put(bidirect)
        self.final_move.put(final)
        self.memo.put(memo)
        self.filename_pattern.put(filename_pattern)
        
        # Scan is configured and ready to execute
        self.ready.put(True)


    def set(self, value, **kwargs):
        """interface to use bps.mv()"""
        if value != 1:
            return
        if self.ready.get():
            bcs_st = self._bcs.scan_status()
            self.busy.put(bcs_st["running_scan"])
        if (not self.ready.get()) or (self.busy.get()):
            # TODO: Raise Warning that scan is not ready
            return
        
        # Workaround until implemented in BCS Flying Scan API
        velocity_value = self.velocity.get()
        if (velocity_value == 0) and (self.motor_name.get() == "Beamline Energy"):
            velocity_value = 0.01  # mm/sec for EPU Gap

        async def check_for_acquire_done(self):
            bcs_st = self._bcs.scan_status()
            while bcs_st["running_scan"]:
                await asyncio.sleep(0.1)
                bcs_st = self._bcs.scan_status()
            else:
                # bcs_st = self._bcs.scan_status()
                print("Scan is finshed!")
                # print(f"{bcs_st =}...")
                self.data_path.put(os.path.join(bcs_st["Log Directory"], bcs_st["Last Filename"]))
                # print(f"{self.data_path.get() =}")
                self.done.put(True)
                self.busy.put(False)
                
        def check_value(*, old_value, value, **kwargs):
            "Return True when the acquisition is complete, False otherwise."
            return (value and not old_value)
        
        status = SubscriptionStatus(self.done, check_value)
        
        self.execute_scan.put(True)
        call_status = self._bcs.sc_single_motor_flying_scan(
            x_motor=self.motor_name.get(), 
            start=self.first_value.get(), 
            stop=self.last_value.get(), 
            increment=self.step_value.get(), 
            velocity_units=velocity_value, 
            number_of_scans=self.num_scans.get(),
            bidirect=self.bidirectional.get(),
            at_end_of_scan=self.final_move.get(), 
            description=self.memo.get(),
            file_pattern=self.filename_pattern.get(),
            )
        if not call_status["success"]:
            print(f"{call_status = }")
            # TODO: Raise exception; reset state?
        self.busy.put(True)
        self.done.put(False)
        # Give the scanner time to start
        time.sleep(1)  # TODO: Is this needed?
        self.execute_scan.put(False)
        acquire_task = asyncio.create_task(check_for_acquire_done(self))
        # acquire_task.add_done_callback()
        
        return status

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class BcsSigFlyScanFlyer(FlyerInterface, BcsSingleMotorFlyingScan):
    '''Example of BCS Single Motor Flying Scan accessed through Bluesky'''
    
    yield_array_events = Component(Signal, kind="config", value=True)

    def __init__(self, *args, **kwargs):
        self._acquiring = False
        self._paused = False

        super().__init__(*args, **kwargs)
        
    def stage(self):
        super().stage()
        # self.select_channels()

    def unstage(self):
        super().unstage()
        # self.select_channels()

    def read_configuration(self):
        return {}

    def describe_configuration(self):
        return {}

    def kickoff(self):
        """Start the scan."""
        self.stage()
        time.sleep(0.1)

        # Check for currently running scan
        bcs_st = self._bcs.scan_status()
        if bcs_st["running_scan"]:
            self.unstage()
            raise RuntimeError("Cannot start scan. Another scan is already running.")
        
        self.set(1)
        self._acquiring = True
        self._paused = False

        status = DeviceStatus(self)
        status.set_finished()  # means that kickoff was successful
        return status

    def complete(self):
        """Wait for sscan to complete."""
        logging.info("complete() starting")
        if not self._acquiring:
            raise RuntimeError("Not acquiring")

        # status = DeviceStatus(self)
        # cb_started = False

        def is_scan_complete(*, old_value, value, **kwargs):
            "Return True when the acquisition is complete, False otherwise."
            # value = bool(value)
            if self._acquiring and value and not old_value:
                logging.info("complete() ending")
                self.unstage()
                self._acquiring = False
                return True
            return False

        status = SubscriptionStatus(self.done, is_scan_complete)
        
        return status

    def describe_collect(self):
        """
        Provide schema & meta-data from collect().
        
        http://nsls-ii.github.io/ophyd/generated/ophyd.flyers.FlyerInterface.describe_collect.html
        """
        # TODO: Add hinted signals
        
        ai_names = self._bcs.list_ais()['names']
        
        scan_motor = self.motor_name.get()
        scan_channels = [
            "Time of Day", 
            "Time (s)", 
            f"{scan_motor}", 
            ] + ai_names
      
        # DataFrame enables convenience functions from 'bcs_events'
        data_df = pd.DataFrame(columns=scan_channels)

        sanitize_event_data_keys = {col: sanitize_key(col) 
            for col in data_df.columns[1:].values}

        descriptor_keys = get_descriptor_keys(
            data_df, 
            sanitize_event_data_keys,
            data_src="Inferred from AI List")
        
        if self.yield_array_events.get():
            # This is only for array events
            reading_size = 1 + (self.last_value.get() - self.first_value.get()) / self.step_value.get()
            array_descriptor_keys = {
                key: prepend_dimension_to_descriptor_key(
                    key=value, size=int(reading_size), dim_name="reading") 
                for (key, value) in descriptor_keys.items()
            }
            descriptor_keys = array_descriptor_keys
        
        event_stream_name = "primary"
        
        return {event_stream_name: descriptor_keys}

    def collect(self):
        """
        Retrieve all collected data (after complete()).
        
        Retrieve data from the flyer as proto-events.
        http://nsls-ii.github.io/ophyd/generated/ophyd.flyers.FlyerInterface.collect.html
        """
        if self._acquiring:
            raise RuntimeError("Acquisition still in progress. Call complete() first.")
        
        def get_data_from_scan(self):
            """Extract data from scan output file; return as PANDAS DataFrame."""
            file_path = self.data_path.get()
            if not file_path:
                raise RuntimeError("There is no scan data. Call kickoff() first.")
            
            file_text = self._bcs.get_text_file(file_path)['text']

            # Get date from BCS data file header
            with io.StringIO(file_text) as data_file:
                data_file_date_str = data_file.readline().strip().split("Date: ", 1)[1]
                data_date = datetime.strptime(data_file_date_str, "%m/%d/%Y").date()
                self._data_date = data_date
                
            data = io.StringIO(file_text)
            skip_rows = find_bcs_data(data)
            df = pd.read_csv(data, sep='\t', skiprows=skip_rows)
            return df

        def generate_scan_data_events(self):
            """Get the entire scan data and yield bluesky events."""
            data_df = get_data_from_scan(self)
            num_points = len(data_df)
            
            # Run info will not be used by fly() plan
            # ...enables convenience functions from 'bcs_events'
            run_bundle = event_model.compose_run()
            event_stream_name = 'primary'

            sanitize_event_data_keys = {col: sanitize_key(col) 
                for col in data_df.columns[1:].values}

            descriptor_keys = get_descriptor_keys(
                data_df, 
                sanitize_event_data_keys,
                data_src=self.data_path.get())
    
            stream_descriptor = run_bundle.compose_descriptor(
                data_keys=descriptor_keys,
                name=event_stream_name,
                )

            # Get date from BCS data file header
#             with io.StringIO(self.data_path.get()) as data_file:
#                 data_file_date_str = data_file.readline().strip().split("Date: ", 1)[1]
#                 data_date = datetime.strptime(data_file_date_str, "%m/%d/%Y").date()
            data_date = self._data_date
    
            if self.yield_array_events.get():
                # This is only for array events
                
                data_df = add_timestamps(data_df, data_date, inplace=True)
                data_df.drop('Time of Day', axis=1, inplace=True)  # Redundant; have timestamps
                timestamp_col="timestamp"
                
                timestamp = data_df.iloc[0][timestamp_col]
                data_df.drop(timestamp_col, axis=1, inplace=True, errors="ignore")
            
                event = make_array_event(
                    data_df, timestamp, sanitize_event_data_keys, stream_descriptor)

                # yield 'event', event
                yield event
        
            
            else:
                data_df = get_timestamps(data_df, data_date, inplace=True)

                def get_bundled_event(data_row):
                    return get_event(
                        data_row, sanitize_event_data_keys, stream_descriptor)

                # Pack events into an event_page
                events = data_df.apply(get_bundled_event, axis='columns').values

                # yield 'event_page', event_model.pack_event_page(*events)

                for event in events:
                    # yield 'event', event
                    yield dict(
                        seq_num=event["seq_num"],
                        time=event["time"],
                        data=event["data"],
                        timestamps=event["timestamps"],
                    )

        yield from generate_scan_data_events(self)
        self.unstage()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class BcsTrajectoryScan(Device):
    """
    BCS Trajectory Scan (uses input file) as an Ophyd device
        Can be Move-Delay-Acquire or Continuous Acquisition
    """
    
    # Signals to enable subscriptions for scan setup
    input_file_path = Component(Signal, kind="config", value='')
    delay_sec = Component(Signal, kind="config", value=0.)
    count_sec = Component(Signal, kind="config", value=1.)
    num_scans = Component(Signal, kind="config", value=1)
    skip_unchanged_motors = Component(Signal, kind="config", value=True)
    move_sequentially = Component(Signal, kind="config", value=False)
    shift_flying_data = Component(Signal, kind="config", value=False)
    final_move = Component(Signal, kind="config", value="Stay")
    final_trajectory_name = Component(Signal, kind="config", value="")
    memo = Component(Signal, kind="config", value='')
    filename_pattern = Component(Signal,kind="config", value="*.txt")
    
    # Signals to enable subscriptions/polling for scan status
    ready = Component(Signal, kind="omitted", value=False)
    busy = Component(Signal, kind="omitted", value=False)
    done = Component(Signal, kind="omitted", value=True)
    execute_scan = Component(Signal, kind="omitted", value=True)
    # More than one file can be output
    data_paths = Component(Signal, kind="normal", value='')
    # Motors controlled by this scan
    normal_motors = Component(Signal, kind="config", value=[])
    flying_motor = Component(Signal, kind="config", value='')
    stream_event_sizes = Component(Signal, kind="config", value=[])


    def scan_setup(
            self, *, 
            # server: str, port: int, 
            bcs_server: BCSz.BCSServer, 
            input_file_path: str, 
            delay: float = 0., count: float = 1., 
            num_scans: int = 1, 
            final: str = "Stay",
            skip_unchanged_motors: bool = True, 
            move_sequentially: bool = False, 
            shift_flying_data: bool = False, 
            final_trajectory_name: str = "",
            memo: str = '',
            filename_pattern: str = "*.txt", 
            subpath_replace_dict: dict = {}, 
            **kwargs):
        """Configure a BCS Single Motor Scan"""
        
        # TODO: Check that bcs_server is connected
        self._bcs = bcs_server
        
        self.input_file_path.put(input_file_path), 
        self.delay_sec.put(delay)
        self.count_sec.put(count)
        self.num_scans.put(num_scans)
        self.final_move.put(final)
        self.skip_unchanged_motors.put(skip_unchanged_motors),
        self.move_sequentially.put(move_sequentially),
        self.shift_flying_data.put(shift_flying_data),
        self.final_trajectory_name.put(final_trajectory_name),
        self.memo.put(memo)
        self.filename_pattern.put(filename_pattern)

        # Parse input file for controlled motors           
        # try:
        #     file_text = self._bcs.get_text_file(input_file_path)['text']
        #     print(f"{file_text = }")
        # except Exception as e:
        #     print(f"Scan input file <{input_file_path}> is not accessible")
        #     raise ValueError(f"Scan input file <{input_file_path}> is not accessible")

        # print(f"{file_text = }")
        # print(f"{self._bcs.get_text_file(input_file_path) = }")

        call_status = self._bcs.get_text_file(input_file_path)
        if call_status["success"]:
            file_text = call_status['text']
        else: 
            new_file_path = input_file_path
            for (old_subpath, new_subpath) in subpath_replace_dict.items():
                # print(f"(old_subpath, new_subpath): {(old_subpath, new_subpath)}")
                new_file_path = input_file_path.replace(old_subpath, new_subpath)
                # print(f"(input_file_path, new_file_path): {(input_file_path, new_file_path)}")
            try:
                with open(new_file_path, "r") as input_file:
                    file_text = input_file.read()
            except OSError as e:
                error_msg = f"Scan input file <{input_file_path}> is not accessible"
                print(error_msg)
                raise ValueError(error_msg)
        
        with io.StringIO(file_text) as input_file:
            # print(input_file.readline().strip().split("Flying ", 1))
            (input_file_motor_header,
             input_file_flying_header,
            ) = tuple(input_file.readline().strip().split("Flying ", 1))
            if input_file_flying_header:
                flying_motor = input_file_flying_header.split(" (", 1)[0]
                input_file_motor_header = input_file.readline().strip()
            else:
                flying_motor = ''
            normal_motors = []
            for motor in input_file_motor_header.split('\t'):
                normal_motors.append(motor)

            # Extract metadata needed for stream event descriptors
            stream_event_sizes = []  # Zero for 'file' command
            stream_event_size = 0    # Single event
            if flying_motor:
                for input_line in input_file.readlines():
                    # input_line.strip().lower().split("flying ", 1)
                    input_line = input_line.strip()
                    if not input_line:
                        # Skip blank lines
                        continue
                    if input_line.lower() == "file":    
                        # end of output file; possible new stream
                        # ignore repeated 'file' commands
                        if (stream_event_sizes and 
                                (stream_event_sizes[-1] == 0)
                                ):
                            continue
                        stream_event_sizes.append(0)
                        continue
                    input_line = input_line.split("(", 1)[1]
                    input_line = input_line.rsplit(")", 1)[0]
                    input_line = input_line.replace(" ", '')
                    (
                     first_value, 
                     last_value, 
                     step_value, 
                     velocity, 
                     ) = (
                        float(value) for value in input_line.split(',')
                        )
                    num_readings = 1 + int(abs(
                        round((last_value - first_value) / step_value)
                        ))
                    if (stream_event_size and 
                            (num_readings != stream_event_size)
                            ):
                        (error_msg) = (
                            f"{input_file_path}: flying() command in " +
                            f"Trajectory Scan has {num_readings} readings " +
                            f"and previous flying() command had " +
                            f"{stream_event_size} readings. " +
                            f"Separate these commands with the 'file' command."
                            )
                        warn(error_msg, RuntimeWarning)
                    stream_event_size = num_readings
                    stream_event_sizes.append(stream_event_size)
                else:
                    ...  # check/store size of final command?
            else:
                # Point-by-point (move-delay-acquire) scan
                # Extract number of readings in each output file
                for input_line in input_file.readlines():
                    input_line = input_line.strip()
                    if not input_line:
                        # Skip blank lines
                        continue
                    if input_line.lower() == "file":    
                        # end of output file; possible new stream
                        # ignore repeated 'file' commands
                        if (stream_event_sizes and 
                                (stream_event_sizes[-1] == 0)
                                ):
                            continue
                        # Include number of readings detected
                        stream_event_sizes.append(stream_event_size)
                        # Include 'file' separator'
                        stream_event_sizes.append(0)
                        # Reset number of readings
                        stream_event_size = 0
                        continue
                    stream_event_size += 1
                else:
                    # Include number of readings detected, if any
                    if stream_event_size:
                        stream_event_sizes.append(stream_event_size)

        self.normal_motors.put(normal_motors)
        self.flying_motor.put(flying_motor)
        self.stream_event_sizes.put(stream_event_sizes)
        
        # Scan is configured and ready to execute
        self.ready.put(True)


    def set(self, value, **kwargs):
        """interface to use bps.mv()"""
        if value != 1:
            return
        if self.ready.get():
            bcs_st = self._bcs.scan_status()
            self.busy.put(bcs_st["running_scan"])
        if (not self.ready.get()) or (self.busy.get()):
            # TODO: Raise Warning that scan is not ready
            return

        async def check_for_acquire_done(self):
            bcs_st = self._bcs.scan_status()
            # print(f"{bcs_st =}...")
            while bcs_st["running_scan"]:
                await asyncio.sleep(0.1)
                data_paths = list(self.data_paths.get())
                # print(f"{data_paths =}")
                if data_paths:
                    last_data_path = data_paths[-1]
                else:
                    last_data_path = ""
                data_path = os.path.join(bcs_st["Log Directory"], bcs_st["Last Filename"])
                if data_path != last_data_path:
                    print("Adding new data file")
                    data_paths.append(data_path)
                    self.data_paths.put(data_paths)
                bcs_st = self._bcs.scan_status()
            else:
                # bcs_st = self._bcs.scan_status()
                print("Scan is finshed!")
                # print(f"{bcs_st =}...")
                # self.data_paths.put(os.path.join(bcs_st["Log Directory"], bcs_st["Last Filename"]))
                # print(f"{self.data_path.get() =}")
                self.done.put(True)
                self.busy.put(False)
                
        def check_value(*, old_value, value, **kwargs):
            "Return True when the acquisition is complete, False otherwise."
            return (value and not old_value)
        
        status = SubscriptionStatus(self.done, check_value)
        
        self.data_paths.put([])
        self.execute_scan.put(True)
        call_status = self._bcs.sc_trajectory_scan(
            file_path=self.input_file_path.get(), 
            delay_after_move_s=self.delay_sec.get(), 
            count_time_s=self.count_sec.get(), 
            at_end=self.final_move.get(), 
            # number_of_scans=self.num_scans.get(),
            dont_repeat_motor_moves=self.skip_unchanged_motors.get(),
            move_motors_sequentially=self.move_sequentially.get(),
            shift_flying_data=self.shift_flying_data.get(),
            move_at_end_of_scan=self.final_trajectory_name.get(),
            description=self.memo.get(),
            file_pattern=self.filename_pattern.get(),
            )
        if not call_status["success"]:
            print(f"{call_status = }")
            # TODO: Raise exception; reset state?
        self.busy.put(True)
        self.done.put(False)
        # Give the scanner time to start
        time.sleep(1)  # TODO: Is this needed?
        self.execute_scan.put(False)
        acquire_task = asyncio.create_task(check_for_acquire_done(self))
        # acquire_task.add_done_callback()
        
        return status

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class BcsTrajScanFlyer(FlyerInterface, BcsTrajectoryScan):
    '''Example of BCS Trajectory Scan accessed through Bluesky'''
    
    yield_array_events = Component(Signal, kind="config", value=True)

    def __init__(self, *args, **kwargs):
        self._acquiring = False
        self._paused = False

        super().__init__(*args, **kwargs)
        
    def stage(self):
        super().stage()
        # self.select_channels()

    def unstage(self):
        super().unstage()
        # self.select_channels()

    def read_configuration(self):
        return {}

    def describe_configuration(self):
        return {}

    def kickoff(self):
        """Start the scan."""
        self.stage()
        time.sleep(0.1)

        # Check for currently running scan
        bcs_st = self._bcs.scan_status()
        if bcs_st["running_scan"]:
            self.unstage()
            raise RuntimeError("Cannot start scan. Another scan is already running.")
        
        if self.flying_motor.get():
            self.yield_array_events.put(True)
        else:
            self.yield_array_events.put(False)
        
        self.set(1)
        self._acquiring = True
        self._paused = False

        status = DeviceStatus(self)
        status.set_finished()  # means that kickoff was successful
        return status

    def complete(self):
        """Wait for sscan to complete."""
        logging.info("complete() starting")
        if not self._acquiring:
            raise RuntimeError("Not acquiring")

        # status = DeviceStatus(self)
        # cb_started = False

        def is_scan_complete(*, old_value, value, **kwargs):
            "Return True when the acquisition is complete, False otherwise."
            # value = bool(value)
            if self._acquiring and value and not old_value:
                logging.info("complete() ending")
                self.unstage()
                self._acquiring = False
                return True
            return False

        status = SubscriptionStatus(self.done, is_scan_complete)
        
        return status

    def describe_collect(self):
        """
        Provide schema & meta-data from collect().
        
        http://nsls-ii.github.io/ophyd/generated/ophyd.flyers.FlyerInterface.describe_collect.html
        """
        # TODO: Add hinted signals
        
        ai_names = self._bcs.list_ais()['names']
        
        scan_channels = [
            "Time of Day", 
            "Time (s)", 
            ] + [f"{motor} {suffix}" 
                    for motor in self.normal_motors.get()
                    for suffix in ["Goal", "Actual"]
                ]
        flying_motor = self.flying_motor.get()
        if flying_motor:
            # scan_channels += [flying_motor]
            # Workaround for current behavior of Trajectory Scan
            scan_channels += [f"{flying_motor} Actual"]
        scan_channels += ai_names
      
        # DataFrame enables convenience functions from 'bcs_events'
        data_df = pd.DataFrame(columns=scan_channels)

        sanitize_event_data_keys = {col: sanitize_key(col) 
            for col in data_df.columns[1:].values}

        descriptor_keys = get_descriptor_keys(
            data_df, 
            sanitize_event_data_keys,
            data_src="Inferred from AI List")

        def array_descriptor_keys(reading_size):
            return {
                key: prepend_dimension_to_descriptor_key(
                    key=value, 
                    size=int(reading_size), 
                    dim_name="reading") 
                for (key, value) in descriptor_keys.items()
                }
        
        def split_by_separator(value_sequence, separator):
            result = []
            for value in value_sequence:
                if value == separator:
                    yield result
                    result = []
                    continue
                result.append(value)
            yield result

        file_event_sizes = [
            event_size for event_size in split_by_separator(
                self.stream_event_sizes.get(),
                0  # Proxy for 'file' separator
                )
            ]
        
        def stream_name_generator(default_stream="primary"):
            yield default_stream
            for stream_number in it.count():
                yield f"aux_{stream_number:04d}"
        
        stream_names = stream_name_generator()

        event_streams = {}
        if self.yield_array_events.get():
            # This is only for array events
            stream_event_size = None
            for (output_path, event_sizes) in zip(
                    self.data_paths.get(), 
                    file_event_sizes, 
                    ):
                for event_size in event_sizes:
                    if (stream_event_size and 
                            (event_size == stream_event_size)
                            ):
                        # event discriptor already exists; skip
                        continue
                    # Add new descriptor
                    #   First event in stream, or...
                    #   Eeent size mismatch; start new stream
                    event_stream_name = next(stream_names)
                    print(f"Creating new stream {event_stream_name}")
                    print(f"...New {event_size = }")
                    print(f"...Previous {stream_event_size = }")
                    event_streams[
                        event_stream_name
                        ] = array_descriptor_keys(event_size)
                    stream_event_size = event_size
                else:
                    # Start new stream for next file?
                    # stream_event_size = None
                    ...
        else: 
            # Single-reading events
            event_stream_name = "primary"
            event_streams = {event_stream_name: descriptor_keys}
        
        return event_streams

    def collect(self):
        """
        Retrieve all collected data (after complete()).
        
        Retrieve data from the flyer as proto-events.
        http://nsls-ii.github.io/ophyd/generated/ophyd.flyers.FlyerInterface.collect.html
        """
        if self._acquiring:
            raise RuntimeError("Acquisition still in progress. Call complete() first.")
        
        def get_data_from_scan(self, file_path):
            """Extract data from scan output file; return as PANDAS DataFrame."""
            if not file_path:
                raise RuntimeError("There is no scan data. Call kickoff() first.")
            
            file_text = self._bcs.get_text_file(file_path)['text']

            # Get date from BCS data file header
            with io.StringIO(file_text) as data_file:
                data_file_date_str = data_file.readline().strip().split("Date: ", 1)[1]
                data_date = datetime.strptime(data_file_date_str, "%m/%d/%Y").date()
                self._data_date = data_date
                
            data = io.StringIO(file_text)
            skip_rows = find_bcs_data(data)
            df = pd.read_csv(data, sep='\t', skiprows=skip_rows)
            return df

        def generate_scan_data_events(self):
            """Get the entire scan data and yield bluesky events."""
        
            def split_by_separator(value_sequence, separator):
                result = []
                for value in value_sequence:
                    if value == separator:
                        yield result
                        result = []
                        continue
                    result.append(value)
                yield result

            file_event_sizes = [
                event_size for event_size in split_by_separator(
                    self.stream_event_sizes.get(),
                    0  # Proxy for 'file' separator
                    )
                ]

            for (output_path, event_sizes) in zip(
                    self.data_paths.get(), 
                    file_event_sizes, 
                    ):

                data_df = get_data_from_scan(self, output_path)
                num_points = len(data_df)
                
                # Run info will not be used by fly() plan
                # ...enables convenience functions from 'bcs_events'
                run_bundle = event_model.compose_run()
                event_stream_name = 'primary'

                sanitize_event_data_keys = {col: sanitize_key(col) 
                    for col in data_df.columns[1:].values}

                descriptor_keys = get_descriptor_keys(
                    data_df, 
                    sanitize_event_data_keys,
                    data_src=output_path)
        
                stream_descriptor = run_bundle.compose_descriptor(
                    data_keys=descriptor_keys,
                    name=event_stream_name,
                    )

                data_date = self._data_date

                event_stops = cumsum(event_sizes)
                event_starts = roll(event_stops, 1)
                event_starts[0] = 0
    
                if self.yield_array_events.get():
                    # This is only for array events
                
                    for (event_start, event_stop) in zip(
                            event_starts, event_stops):
                        
                        df = data_df.iloc[event_start:event_stop]
                    
                        df = add_timestamps(df, data_date, inplace=True)
                        df.drop(
                            'Time of Day', 
                            axis=1, 
                            inplace=True)  # Redundant; have timestamps
                        timestamp_col="timestamp"
                        
                        timestamp = df.iloc[0][timestamp_col]
                        df.drop(
                            timestamp_col, 
                            axis=1, 
                            inplace=True, 
                            errors="ignore")
                    
                        event = make_array_event(
                            df, 
                            timestamp, 
                            sanitize_event_data_keys, 
                            stream_descriptor)

                        # yield 'event', event
                        yield event
                
                else:
                    data_df = get_timestamps(
                        data_df, data_date, inplace=True)

                    def get_bundled_event(data_row):
                        return get_event(
                            data_row, sanitize_event_data_keys, stream_descriptor)

                    # Pack events into an event_page
                    events = data_df.apply(
                        get_bundled_event, axis='columns').values

                    # yield 'event_page', event_model.pack_event_page(*events)

                    for event in events:
                        # yield 'event', event
                        yield dict(
                            seq_num=event["seq_num"],
                            time=event["time"],
                            data=event["data"],
                            timestamps=event["timestamps"],
                        )

        yield from generate_scan_data_events(self)
        self.unstage()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import types
__all__ = [name for name, thing in globals().items()
            if not (
              name.startswith('_') or 
              isinstance(thing, types.ModuleType) or 
              # isinstance(thing, types.FunctionType) or 
              # isinstance(thing, type) or  # Class type
              isinstance(thing, dict) 
              )
            ]
del types
