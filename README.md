# StableHLO.jl

Julia bindings for the MLIR HLO (High Level Operations) dialects found in [openxla/stablehlo](https://github.com/openxla/stablehlo).

> !WARNING
> **Disclaimer**: This library just provides the bindings for creating the HLO operations.
> You still need to provide a compatible backend (like XLA or IREE).
> Check out the [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) and [Coil.jl](https://github.com/pangoraw/Coil.jl) to generate StableHLO code from Julia.
