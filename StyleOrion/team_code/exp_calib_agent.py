# ==============================================================================
# [Bench2Drive-Style] NEW FILE -- script
# Author: YinXuan1223   (carried over from the ORION_NYCUACM_version fork)
#
# One-shot calibration agent that finds the exposure values used by
# orion_b2d_agent_fixedexp.py.  It loads no model: it boots through the normal
# leaderboard pipeline, parks the ego at spawn and saves several front cameras that
# differ only in exposure_compensation, so one run sweeps the whole range.
# Calibration tool only -- never use it for real evaluation.
# ==============================================================================
"""Minimal exposure-calibration agent (NO model load).

Boots through the normal leaderboard pipeline (which launches CARLA reliably,
unlike a standalone server), parks the ego at spawn, and saves several front
cameras that are identical except for their manual exposure_compensation. One run
therefore sweeps the whole exposure range simultaneously, so we can pick the value
whose mean brightness matches the default-histogram reference (~110) with no blowout.

Used only to calibrate FIXED_EXPOSURE for orion_b2d_agent_fixedexp.py; not for real eval.
"""
import os
import numpy as np
import carla
from PIL import Image
from leaderboard.autoagents import autonomous_agent

SAVE = os.environ.get('CALIB_SAVE', '/mnt/HDD5/college_student/OurCode/exp_calib')
N_FRAMES = int(os.environ.get('CALIB_NFRAMES', '12'))
FSTOP = float(os.environ.get('CALIB_FSTOP', '5.6'))

# front-cam pose from the real agent
_POSE = dict(x=0.80, y=0.0, z=1.60, roll=0.0, pitch=0.0, yaw=0.0,
             width=1600, height=900, fov=70)

# one camera per exposure setting (same pose) -> full sweep in a single run
_SWEEP = [
    ('ref_hist', dict(exposure_mode='histogram')),                                  # reference "correct" daytime
    ('c03', dict(exposure_mode='manual', shutter_speed=200, iso=100, fstop=FSTOP, exposure_compensation=3)),
    ('c05', dict(exposure_mode='manual', shutter_speed=200, iso=100, fstop=FSTOP, exposure_compensation=5)),
    ('c07', dict(exposure_mode='manual', shutter_speed=200, iso=100, fstop=FSTOP, exposure_compensation=7)),
    ('c09', dict(exposure_mode='manual', shutter_speed=200, iso=100, fstop=FSTOP, exposure_compensation=9)),
    ('c11', dict(exposure_mode='manual', shutter_speed=200, iso=100, fstop=FSTOP, exposure_compensation=11)),
]


def get_entry_point():
    return 'ExpCalibAgent'


class ExpCalibAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.step = 0
        os.makedirs(SAVE, exist_ok=True)
        for cid, _ in _SWEEP:
            os.makedirs(os.path.join(SAVE, cid), exist_ok=True)

    def sensors(self):
        sensors = []
        for cid, attrs in _SWEEP:
            s = dict(type='sensor.camera.rgb', id=cid, **_POSE)
            s.update(attrs)
            sensors.append(s)
        # minimal non-camera sensors the pipeline expects
        sensors += [
            {'type': 'sensor.other.imu', 'x': -1.4, 'y': 0.0, 'z': 0.0,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'sensor_tick': 0.05, 'id': 'IMU'},
            {'type': 'sensor.other.gnss', 'x': -1.4, 'y': 0.0, 'z': 0.0,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'sensor_tick': 0.01, 'id': 'GPS'},
            {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'SPEED'},
        ]
        return sensors

    def run_step(self, input_data, timestamp):
        if self.step < N_FRAMES:
            for cid, _ in _SWEEP:
                if cid in input_data:
                    bgra = input_data[cid][1]
                    rgb = bgra[:, :, :3][:, :, ::-1]
                    Image.fromarray(rgb.copy()).save(
                        os.path.join(SAVE, cid, '%04d.png' % self.step))
        self.step += 1
        # stay parked at spawn
        return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)

    def destroy(self):
        pass
