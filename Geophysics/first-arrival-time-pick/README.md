# first-arrival-time-pick

初至拾取的4个Python程序

## first_breaks_picking

This project is devoted to pick waves that are the first to be detected on a seismogram (first breaks, first arrivals).
Traditionally, this procedure is performed manually. When processing field data, the number of picks reaches hundreds of
thousands. Existing analytical methods allow you to automate picking only on high-quality data with a high
signal / noise ratio.

可以使用GUI界面的手动拾取：first-breaks-picking app

也可以编程脚本程序，具体参考first-arrival-time-pick/README.md

## first-arrival-picker

Python based (GUI) first-arrival picker for seismic refraction data.

## AIPycker-master

Pycker provides user-friendly routines to visualize seismic traces and pick first break arrival times. This package requires ObsPy.

## deeppick-main

This code is a Keras implementation of [PhaseNet](https://github.com/wayneweiqiang/PhaseNet), dedicated to *"Automatic arrival time picking for seismic inversion with unlabeled data"*. This version allows to deal with new datasets having different sizes and number of channels, especially, to implement transfer learning using the pretrained model from [NCEDC](https://ncedc.org/) data (Northern California Earthquake Data Center) and deal with small labeled datasets or unlabeled datasets using semi-supervised learning. Robust linear regression methods and SVR (Support Vector Regression) are used to correct labels in the pseudo-labeling (see `correct_label` directory), that helps significantly enhance the quality of pseudo labels.