# Research Paper Executive Summary

**Title:** Can Machine Learning Estimate Beehive Weight Without a Scale? An Empirical Investigation on 78 Honeybee Colonies in Germany (2019–2022)  
**Authors:** Amha Endrayes, Natnael Behailu  
**Institution:** Robox AI and Robotics Center, Addis Ababa, Ethiopia  

### Hypotheses & Verified Outcomes
1. **Hypothesis 1 (Zero-Shot Unidentifiability): CONFIRMED.** Models evaluating unseen hives with zero calibration fail completely ($R^2 = -0.124$, $	ext{MAE} = 8,652	ext{ g}$). Static hardware tare mass (18–55 kg) emits no thermal signature.
2. **Hypothesis 2 (Few-Shot Calibration): CONFIRMED.** A 24-hour scale calibration window recovers the tare offset, restoring high accuracy ($R^2 = 0.912$, $	ext{MAE} = 1,824	ext{ g}$).
3. **Hypothesis 3 (Diurnal Biomass Tracking): CONFIRMED.** 1-hour relative weight change estimation achieves $R^2 = 0.498$ and $	ext{MAE} = 46.8	ext{ g}$, accurately tracking worker bee flight departures and returns.
4. **Hypothesis 4 (Long-Horizon Error Drift): CONFIRMED.** 24-hour weight change estimation error increases to $512	ext{ g}$ ($R^2 = 0.188$) because internal sensors cannot observe surrounding floral nectar density.
