# Data for: A modified Michaelis-Menten equation estimates growth from birth to 3 years in healthy babies in the US

<https://doi.org/10.5061/dryad.4j0zpc8jf>

## Description of the data and file structure

Data for this study include, per baby: sex, age in days, and, over time, weight in Kg and height in cm. Each baby had at least 5 visits. Our goal was to fit each baby’s data to a curve as described by a modified Michaelis-Menten equation, allowing interpolation of missing weight or height values. Among the subset of all infants who had 7 well-baby visits in the  first year of life, and 12 visits over 3 years, we further explored the minimum number of, and which, data points were necessary for good fit. Finally, among babies with 5 time points in year 1, and 2 in both year 2 and year 3, we examined whether weight or height data early in life could predict growth in later months.

To meet anonymization guidelines, we are providing only STARR data, including sex, age and jittered weight and height (for STORK data, STARR race/ethnicity information and STARR exact data, please contact us directly). We used half of the RMSE sd as a range for the jitter (+/- 0.075 kg for weight and +/- 0.2 cm for height). Attached files include:

(1) STARR weight data (starr_source_weight_data.txt): anon_id, sex (male or female), the baby's age in days (AgeBabyDays), jittered weight in kg (jit_weightkg); 
(2) STARR height data (starr_source_height_data.txt): anon_id, sex, AgeBabyDays, jittered height in cm (jit_heightcm).

## Sharing/Access information

All data (on Dryad) and code (via Zenodo: <https://doi.org/10.5281/zenodo.10537088>) are included here.

## Code/Software

The R code for fitting weight and/or height data with the MME equation is shown in the MME_growth_fitting.RMD file. The tab-delimited and anonymized source data for weights and heights (both jittered) are posted. These can be used with the R code-but the user will need to correct input and output filepaths used in the script. The HTML version of these files is available as well, in case viewing the scripts without opening them in R is desired.

The add_holdout script is a helper script for creating a holdout column used by the hbgd_holdout_tests, sitar_holdout_test, and locf_holdout_tests scripts. The simple_linear_model and MME_holdout_tests script have the same functionality built into the script.

R_sessionInfo.txt contains the R software version, as well as the versions of the packages included in the code.

See the methods section for the description of the starting parameters for the nls() function.
