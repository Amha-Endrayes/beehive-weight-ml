# Research Paper Executive Summary

**Title:** Can Machine Learning Estimate Beehive Weight Without a Scale? An Empirical Investigation on 78 Honeybee Colonies in Germany (2019–2022)  
**Authors:** Amha Endrayes, Natnael Behailu  
**Institution:** Robox AI and Robotics Center, Addis Ababa, Ethiopia  

### Hypotheses & Verified Outcomes
1. **Hypothesis 1 (Zero-Shot Unidentifiability): CONFIRMED.** Models evaluating unseen hives with zero calibration fail completely ($R^2 = -1.421$, $	ext{MAE} = 11,009	ext{ g}$). Static hardware tare mass (17.7–49.8 kg) emits no thermal signature.
2. **Hypothesis 2 (Few-Shot Calibration): CONFIRMED.** A 24-hour scale calibration window recovers the tare offset, restoring predictive skill ($R^2 = 0.430$, $	ext{MAE} = 4,838	ext{ g}$), while 7-day calibration achieves $	ext{MAE} = 3,120	ext{ g}$ ($R^2 = 0.718$).
3. **Hypothesis 3 (Diurnal Biomass Tracking): CONFIRMED.** 1-hour relative weight change estimation achieves $	ext{MAE} = 35.2	ext{ g}$, accurately tracking worker bee flight departures and returns.
4. **Hypothesis 4 (Long-Horizon Error Drift): CONFIRMED.** 24-hour weight change estimation error increases to $373.7	ext{ g}$ because internal sensors cannot observe surrounding floral nectar density.
