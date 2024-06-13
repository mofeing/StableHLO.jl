using Pkg
using BinaryBuilderBase: temp_prefix, PkgSpec, setup_dependencies, destdir
using LibGit2

llvm_version = v"17.0.6+3"
julia_version = v"1.12"

target_dialects = [
    ("CHLO.jl", ["ChloOps.td"]),
    ("StableHLO.jl", ["StablehloOps.td"]),
    ("VHLO.jl", ["VhloOps.td"]),
]

temp_prefix() do prefix
    # setup dependencies
    platform = Pkg.BinaryPlatforms.HostPlatform()
    platform["llvm_version"] = string(llvm_version.major)
    platform["julia_version"] = string(julia_version)

    dependencies = PkgSpec[
        PkgSpec(; name="mlir_jl_tblgen_jll"),
        PkgSpec(; name="LLVM_full_jll", version=llvm_version)
    ]

    artifact_paths = setup_dependencies(prefix, dependencies, platform; verbose=true)

    # fetch `stablehlo` repo
    repo = LibGit2.clone("https://github.com/openxla/stablehlo", joinpath(prefix, "stablehlo"))
    LibGit2.checkout!(repo, "14e2323f0ee3d308c1384fdb806dc6d0c98b16ca") # v1.1.2

    # generate bindings
    mlir_jl_tblgen = joinpath(destdir(prefix, platform), "bin", "mlir-jl-tblgen")
    include_dir = joinpath(destdir(prefix, platform), "include")
    dialect_dir = joinpath(joinpath(prefix, "stablehlo"), "stablehlo", "dialect")

    for (output, inputs) in target_dialects
        inputs = map(inputs) do x
            joinpath(dialect_dir, x)
        end
        run(`$mlir_jl_tblgen --external --generator=jl-op-defs -I$include_dir -I$(joinpath(prefix, "stablehlo")) -o $(joinpath(dirname(@__DIR__), "src", "Dialects", output)) $inputs`)
    end
end
