# Dataset Information

This repository utilizes the open-access dataset:
**"Weight, Temperature and Humidity Sensor Data of Honey Bee Colonies in Germany, 2019–2022"**
* Zenodo Record: https://doi.org/10.5281/zenodo.10407693
* Authors: Diren Senger, Clemens Gruber, Thorsten Kluss, Carolin Johannsen
* Publication: Data in Brief 52 (2024) 110015 (PMC10809063)

### Directory Contents
* `metadata/dataset_metadata.json`: Full technical schema, sensor hardware, and Zenodo checksums.
* `metadata/data_dictionary.md`: Definitions of all sensor channels (t_i_1..5, t_o, h, p, weight_kg).
* `metadata/colony_locations.csv`: Geographic coordinates and identifiers for all 78 colonies.

### Automated Download
The Google Colab notebook (`notebooks/Beehive_Weight_Estimation_Pipeline.ipynb`) includes automated resumable download logic for Zenodo record 10407693.
