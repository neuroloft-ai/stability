"""
AppTest/modules — reusable pipeline building blocks.

Modules
-------
data_intake      : load_data()            — data loading, column selection, output-def building
test_generator   : get_available_tests()  — list of test names from registry
                   get_test_specs()       — DataFrame of all test specifications
                   generate_tests()       — DataFrame of generated stress variants
test_runner      : run_tests()            — run generated tests against any SUT
                   make_agent_sut()       — wrap a local agent (.extract(**fields))
                   make_llm_sut()         — wrap an OpenAI-compatible LLM
                   describe_run()         — print run summary
eval_output      : eval_output()          — score predictions vs ground truth
eval_kb          : eval_kb()              — KB behavioral compliance scoring (LLM judge)
eval_all         : eval_all()            — unified evaluator: routes per row + per field to correct strategy
                   summarize_measures()  — aggregate measurements into one row per category
compute_metrics  : compute_metrics()      — behavioural metrics (SR, CP, SS, CS, Osc)
                   MetricsResult          — result dataclass
display_metrics  : display_metrics()      — rich notebook display of MetricsResult
"""
from .data_intake      import load_data, DataIntakeResult
from .test_generator   import get_available_tests, get_test_specs, generate_tests
from .test_runner      import run_tests, make_agent_sut, make_llm_sut, describe_run
from .eval_output      import eval_output
from .eval_kb          import eval_kb
from .eval_all         import eval_all
from .compute_metrics  import compute_metrics, MetricsResult
from .display_metrics  import display_metrics
from .stress_metrics   import compute_stress_metrics, plot_stress_scatter, plot_score_stability_bars, compute_sr

__all__ = [
    "load_data", "DataIntakeResult",
    "get_available_tests", "get_test_specs", "generate_tests",
    "run_tests", "make_agent_sut", "make_llm_sut", "describe_run",
    "eval_output",
    "eval_kb",
    "eval_all",
    "eval_measure", "summarize_measures",
    "compute_metrics", "MetricsResult",
    "display_metrics",
    "compute_stress_metrics", "plot_stress_scatter", "plot_score_stability_bars", "compute_sr",
]
