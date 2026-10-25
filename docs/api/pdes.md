# PDEs

Every shipped PDE is registered once in `pdes.registry`, and both the CLI and the dashboard read it from there. User-defined equations go through `CustomPDE` (see [Custom equations](../custom_equations.md)).

::: pdes.custom.CustomPDE

::: pdes.custom.Fields

::: pdes.symbolic.compile_expression

::: pdes.symbolic.CompiledExpression

::: pdes.catalog.from_catalog

::: pdes.catalog.CATALOG

::: pdes.registry.create_pde

::: pdes.registry.get_pde_spec

::: pdes.registry.PDESpec

::: pdes.pde_base.PDEConfig

::: pdes.pde_base.PDEBase

::: pdes.heat_equation.HeatEquation

::: pdes.wave_equation.WaveEquation

::: pdes.burgers_equation.BurgersEquation

::: pdes.kdv_equation.KdVEquation

::: pdes.convection_equation.ConvectionEquation

::: pdes.allen_cahn.AllenCahnEquation

::: pdes.cahn_hilliard.CahnHilliardEquation

::: pdes.black_scholes.BlackScholesEquation

::: pdes.pendulum_equation.PendulumEquation
