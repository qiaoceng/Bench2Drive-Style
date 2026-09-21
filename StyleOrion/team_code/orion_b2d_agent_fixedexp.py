# ==============================================================================
# [Bench2Drive-Style] NEW FILE -- script
# Author: YinXuan1223   (carried over from the ORION_NYCUACM_version fork)
#
# OrionAgent subclass that pins every RGB camera to MANUAL exposure.  On some
# bright daytime routes CARLA's histogram auto-exposure converges to a wrong
# over-bright steady state and the frame stays blown out for the rest of the
# episode, non-deterministically between repetitions.  Only sensors() is overridden.
# The values are calibrated for ONE lighting condition (daytime, route 2715) and
# will render a night route black -- do not blanket-apply it to a mixed-lighting
# benchmark.
# ==============================================================================
"""
ORION agent variant with FIXED (manual) camera exposure.

Why this exists
---------------
On some daytime, high-dynamic-range routes (e.g. RouteScenario_2715 StaticCutIn:
sun_altitude=45, cloudiness=80, wet road), CARLA's default histogram auto-exposure
(UE4 "eye adaptation") sometimes converges to a wrong, over-bright steady state.
The whole frame blows out white with a green bloom smear and never recovers for the
rest of the episode. It is non-deterministic: the same route renders fine on some
repetitions and blows out on others (observed: 2715 rep0 partial, rep1 clean,
rep2 fully blown). See the brightness analysis that motivated this file.

This subclass keeps the base OrionAgent untouched and only overrides sensors() to
pin every RGB camera to MANUAL exposure, which disables eye-adaptation entirely and
removes the run-to-run drift.

IMPORTANT (domain note): a fixed exposure is tuned for ONE lighting condition. The
values below target daytime / bright-overcast (route 2715). They will make a NIGHT
route (e.g. 2709, sun_altitude=-90) render black. Do NOT blanket-apply this agent to
mixed-lighting benchmarks -- use it for the daytime StaticCutIn re-run, or make the
exposure lighting-dependent before wider use.
"""

from StyleOrion.team_code.orion_b2d_agent import OrionAgent as _BaseOrionAgent


def get_entry_point():
    return 'OrionAgentFixedExp'


# Manual exposure = UE4 physical-camera model, no auto eye-adaptation.
# Values calibrated against route 2715's weather via a one-shot exposure sweep
# (exp_calib_agent.py): at fstop=5.6, 1/200 s, ISO 100, exposure_compensation=+3
# the front camera renders mean ~125/255 with 0% saturation -- matching the correct
# auto-exposure reference (~109) with no blowout. Sweep (mean / sat%):
#   comp +3 -> 125 / 0% (chosen)   +5 -> 202 / 0%   +7 -> 244 / 56%   +9 -> 254 / 91%
# Tuning: exposure_compensation is in stops (higher = brighter); +5 starts to clip.
import os as _os


def _build_fixed_exposure():
    """Build the camera exposure attribute dict, overridable via env so the
    fix-and-retry loop can sweep values without editing code. Defaults reproduce
    the calibrated manual/+3 config."""
    mode = _os.environ.get('FIX_EXP_MODE', 'manual')
    d = dict(exposure_mode=mode)
    if mode == 'manual':
        d.update(
            shutter_speed=float(_os.environ.get('FIX_SHUTTER', '200')),
            iso=float(_os.environ.get('FIX_ISO', '100')),
            fstop=float(_os.environ.get('FIX_FSTOP', '5.6')),
            exposure_compensation=float(_os.environ.get('FIX_EXP_COMP', '3.0')),
        )
    else:  # histogram (auto), optionally pinned via min==max bright
        d['exposure_compensation'] = float(_os.environ.get('FIX_EXP_COMP', '0'))
        if 'FIX_MINBRIGHT' in _os.environ:
            d['exposure_min_bright'] = float(_os.environ['FIX_MINBRIGHT'])
        if 'FIX_MAXBRIGHT' in _os.environ:
            d['exposure_max_bright'] = float(_os.environ['FIX_MAXBRIGHT'])
    if 'FIX_BLOOM' in _os.environ:
        d['bloom_intensity'] = float(_os.environ['FIX_BLOOM'])
    if 'FIX_FLARE' in _os.environ:
        d['lens_flare_intensity'] = float(_os.environ['FIX_FLARE'])
    return d


FIXED_EXPOSURE = _build_fixed_exposure()


class OrionAgentFixedExp(_BaseOrionAgent):
    def sensors(self):
        specs = super().sensors()
        for spec in specs:
            if spec.get('type') == 'sensor.camera.rgb':
                spec.update(FIXED_EXPOSURE)
        return specs
