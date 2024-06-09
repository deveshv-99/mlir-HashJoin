## Join_v1:

1. For 10M rows in both relations, with range of values (1 - 100k) uniformly distributed and Hash Table size = 100k
Result Count = 1B
Total GPU Memory used ~ 8.5 GB
Total Time taken for all kernels ~ 2 secs

2. For 100M rows in both relations, with range of values (1 - 1B) uniformly distributed values and Hash Table size = 1M
Result Count = 10M
Total GPU Memory used ~ 4.5 GB
Total Time taken for all kernels ~ 12 secs


## Join_v2:

1. For 10M rows in both relations, with range of values (1 - 100k) uniformly distributed and Hash Table size = 100k
Result Count = 1B
Total GPU Memory used ~ 8.5 GB
Total Time taken for all kernels ~ 1.5 secs

2. For 100M rows in both relations, with range of values (1 - 1B) uniformly distributed values and Hash Table size = 1M
Result Count = 10M
Total GPU Memory used ~ 4.5 GB
Total Time taken for all kernels ~ 12.5 secs