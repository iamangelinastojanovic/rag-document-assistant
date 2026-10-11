# Data logging

Process sensors are sampled once per second and written to the historian. Raw one-second data is retained for 90 days, after which it is downsampled to one-minute averages and kept for five years.

Each record stores the sensor tag, the timestamp in UTC, the measured value, and a quality flag. A quality flag of BAD means the reading was rejected by the acquisition system and must not be used in calculations; averaging over BAD values silently corrupts downstream reports.

Gaps in the series are left as gaps. The historian never interpolates missing values, because an interpolated reading is indistinguishable from a measured one once it is stored.
