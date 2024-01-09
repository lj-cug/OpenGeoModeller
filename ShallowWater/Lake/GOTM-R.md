# GOTM-R

## gotmtools

Tools for interacting with the [General Ocean Turbulence Model (GOTM)](http://gotm.net/ "General Ocean Turbulence Model's website") in R. `gotmtools` is inspired by glmtools and applies many of the same functions but adapted for handling GOTM output. It is focused on dealing with model output and diagnosing model performance compared to observed data.

## GOTMr

R package for basic [GOTM](http://gotm.net/) model running. `GOTMr` holds version 5.3 of the [lake branch](http://github.com/gotm-model/code/tree/lake) of the General Ocean Turbulence Model (GOTM) for windows 64bit platforms. This package does not contain the source code for the model, only the executable. Also, use `gotm_version()` to figure out what version of GOTM you are running. This package was inspired by [GOTMr](https://github.com/GLEON/GOTMr).