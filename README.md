# RobustInfoPOMDP
Data collection and figures for robust belief-reward partially observable Markov decision processes.

## Installation
These scripts are built for Julia 0.6. If not already installed, the suite can be cloned using

```julia
Pkg.clone("https://github.com/ajkeith/RobustInfoPOMDP")
```

These scripts also require `RPOMDPs`, `RPOMDPToolbox`, `RPOMDPModels`, and `SimpleProbabilitySets` which can be cloned.

```julia
Pkg.clone("https://github.com/ajkeith/RPOMDPs.jl/tree/ajk/robust")
Pkg.clone("https://github.com/ajkeith/RPOMDPToolbox.jl")
Pkg.clone("https://github.com/ajkeith/RPOMDPModels.jl")
Pkg.clone("https://github.com/ajkeith/SimpleProbabilitySets.jl")
```

Lastly, the scripts require several Julia packages: `DataFrames`, `Query`, `CSV`, `ProgressMeter`, `BenchmarkTools`, `StatsBase`, and `Plots`. These can be added with the standard package commands.

```julia
Pkg.add("Example.jl")
```

## Usage
Each script file produces a different set of robust belief-reward POMDP results. All files write to the data folder. Figures are stored in the `figures` subfolder, numeric data is stored in .csv files, and older versions of data are stored in the `archive` sub-folder.

The robust belief-reward point-based value iteration algorithm implementation and details are available at [RobustValueIteration](https://github.com/ajkeith/RobustValueIteration).

`collect_results.jl` produces results for the robust standard-reward POMDP tiger and baby problems.

`collect_results_info.jl` produces results for the robust belief-reward POMDP rock diagnosis problem.

`collect_assessment.jl` produces results for the cybersecurity assessment application.

`collect_simple.jl` and `info_reward.jl` are used for intermediate results and testing.

`figures.jl` produces figures that require benchmark data (included in the code).

## References
The robust POMDP environment in the components of these scripts are an extension of [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl), [POMDPModels.jl](https://github.com/JuliaPOMDP/POMDPModels.jl), and [POMDPModelTools.jl](https://github.com/JuliaPOMDP/POMDPModelTools.jl) to the robust setting.

If this code is useful to you, please star this package and consider citing the following papers.

Egorov, M., Sunberg, Z. N., Balaban, E., Wheeler, T. A., Gupta, J. K., & Kochenderfer, M. J. (2017). POMDPs.jl: A framework for sequential decision making under uncertainty. Journal of Machine Learning Research, 18(26), 1–5.

Osogami, T. (2015). Robust partially observable Markov decision process. In International Conference on Machine Learning (pp. 106–115).
