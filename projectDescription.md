## Summary

- Implementing Hash-join in MLIR, targeted entirely on GPUs
- Code + lowering pass


## ToDos:

- Compare join on TPC-H columns with RAPIDS
- Figuring out code-generator part
- Characterizing the implementations wrt:
    input sizes
    hash-table sizes
    thread block sizes
    selectivity of joins
    how expensive is compile vs running the query
- Look for crystal, heavyDB, HCHJ, Garuda timings


## Things left out:

- Comparing entire query instead of just Join
- Relations that don't fit on GPU 
- Partitioned Hash-Join 
- Joining skewed data
- Hash Table storing entire data instead of just rowIDs 
- Non-Integer Key
- Multi-columned key join
- Non-Equi Joins
- Non-materialized Tables
- Merge-sort join
- Comparing on AMD vs NVIDIA GPUs
