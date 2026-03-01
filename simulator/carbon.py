"""Synthetic carbon intensity model with realistic diurnal and renewable patterns."""
import numpy as np


class CarbonIntensityModel:
    """
    Simulates grid carbon intensity (gCO2eq/kWh) over a 24-hour period.
    Uses a combination of:
    - Diurnal solar generation curve (low carbon midday)
    - Evening demand peak (higher carbon)
    - Random noise for realism
    - Optional region preset
    """

    REGION_PRESETS = {
        "Texas (ERCOT)":      {"base": 420, "amplitude": 160, "phase_offset": 0.0},
        "California (CAISO)": {"base": 240, "amplitude": 110, "phase_offset": 0.3},
        "UK (National Grid)": {"base": 210, "amplitude": 90,  "phase_offset": 0.1},
        "Germany (ENTSO-E)":  {"base": 320, "amplitude": 130, "phase_offset": 0.2},
        "France (RTE)":       {"base": 85,  "amplitude": 40,  "phase_offset": 0.0},
        "Australia (NEM)":    {"base": 580, "amplitude": 180, "phase_offset": -0.4},
    }

    def __init__(self, region: str = "Texas (ERCOT)", seed: int = 42):
        self.region = region
        preset = self.REGION_PRESETS[region]
        self.base = preset["base"]
        self.amplitude = preset["amplitude"]
        self.phase_offset = preset["phase_offset"]
        self.rng = np.random.default_rng(seed)
        self._noise_cache = {}

    def intensity_at(self, time_minutes: float) -> float:
        """Return carbon intensity at a given simulation time (minutes)."""
        t_hours = (time_minutes % 1440) / 60.0  # wrap to 24h
        # Solar generation curve: dip centered at 13:00
        solar_dip = self.amplitude * np.exp(-0.5 * ((t_hours - 13.0 + self.phase_offset * 24) / 3.5) ** 2)
        # Evening demand peak: rise centered at 19:00
        demand_peak = (self.amplitude * 0.4) * np.exp(-0.5 * ((t_hours - 19.0) / 2.0) ** 2)
        # Noise (cached per 15-min slot for continuity)
        slot = int(time_minutes // 15)
        if slot not in self._noise_cache:
            self._noise_cache[slot] = self.rng.normal(0, self.amplitude * 0.08)
        noise = self._noise_cache[slot]
        intensity = self.base - solar_dip + demand_peak + noise
        return float(np.clip(intensity, 10, 1200))

    def get_forecast(self, start_minutes: float, horizon_minutes: float = 120, resolution: int = 15) -> tuple:
        """Return (times, intensities) forecast arrays."""
        times = np.arange(start_minutes, start_minutes + horizon_minutes, resolution)
        intensities = np.array([self.intensity_at(t) for t in times])
        return times, intensities

    def get_full_day_profile(self) -> tuple:
        """Return full 24h profile at 15-min resolution."""
        return self.get_forecast(0, 1440, 15)
