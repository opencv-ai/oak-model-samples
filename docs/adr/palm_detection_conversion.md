# Palm Detection Model Conversion

## Status

Accepted

## Context

With default settings for OpenVINO Model Optimizer, the forward method of the OAK model returned garbage while worked fine for OpenVINO model.

## Decision

We decided to integrate the preprocessing directly into the model (it requires RGB [-1, 1] images) by using the following arguments:
`--data_type=FP16 --mean_values [127.5,127.5,127.5] --scale_values [127.5,127.5,127.5]`

## Consequences

After removing preprocessing from the code, inference started to work fine
