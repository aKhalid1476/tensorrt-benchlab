"""Markdown report generator for benchmark runs."""
from datetime import datetime
from typing import List, Dict, Any
from contracts import RunRecord, EngineResult, EngineType


def generate_markdown_report(run: RunRecord) -> str:
    """
    Generate a comprehensive markdown report from a benchmark run.

    Args:
        run: Complete run record with results

    Returns:
        Formatted markdown report
    """
    lines = []

    # Header
    lines.append("# TensorRT BenchLab - Benchmark Report")
    lines.append("")
    lines.append(f"**Run ID:** `{run.run_id}`")
    lines.append(f"**Status:** {run.status.upper()}")
    lines.append(f"**Model:** {run.model_name}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **Engines:** {', '.join(run.engines)}")
    lines.append(f"- **Batch Sizes:** {', '.join(map(str, run.batch_sizes))}")
    lines.append(f"- **Iterations:** {run.num_iterations} (measurement) + {run.warmup_iterations} (warmup)")
    lines.append(f"- **Runner URL:** {run.runner_url}")
    lines.append("")

    # Results Summary
    if run.results and len(run.results) > 0:
        lines.append("## Results Summary")
        lines.append("")

        # Group results by engine
        results_by_engine: Dict[str, List[EngineResult]] = {}
        for result in run.results:
            engine = result.engine_type
            if engine not in results_by_engine:
                results_by_engine[engine] = []
            results_by_engine[engine].append(result)

        # Create table for each engine
        for engine_type in sorted(results_by_engine.keys()):
            engine_results = sorted(results_by_engine[engine_type], key=lambda r: r.batch_size)

            lines.append(f"### {engine_type}")
            lines.append("")
            lines.append("| Batch Size | p50 (ms) | p95 (ms) | Mean (ms) | Stddev (ms) | Throughput (img/s) |")
            lines.append("|------------|----------|----------|-----------|-------------|---------------------|")

            for result in engine_results:
                lines.append(
                    f"| {result.batch_size:>10} | "
                    f"{result.latency_ms_p50:>8.2f} | "
                    f"{result.latency_ms_p95:>8.2f} | "
                    f"{result.latency_ms_mean:>9.2f} | "
                    f"{result.latency_ms_stddev:>11.2f} | "
                    f"{result.throughput_img_per_sec:>19.1f} |"
                )

            lines.append("")

        # Speedup Analysis
        lines.append("## Speedup Analysis")
        lines.append("")

        # Calculate speedups vs PyTorch CPU
        cpu_results = {r.batch_size: r for r in run.results if r.engine_type == "pytorch_cpu"}

        if cpu_results:
            lines.append("### vs PyTorch CPU (p50 latency)")
            lines.append("")
            lines.append("| Engine | Batch Size | CPU p50 (ms) | Engine p50 (ms) | Speedup |")
            lines.append("|--------|------------|--------------|-----------------|---------|")

            for engine_type in sorted(results_by_engine.keys()):
                if engine_type == "pytorch_cpu":
                    continue

                for result in sorted(results_by_engine[engine_type], key=lambda r: r.batch_size):
                    if result.batch_size in cpu_results:
                        cpu_p50 = cpu_results[result.batch_size].latency_ms_p50
                        engine_p50 = result.latency_ms_p50
                        speedup = cpu_p50 / engine_p50 if engine_p50 > 0 else 0

                        lines.append(
                            f"| {engine_type:>16} | "
                            f"{result.batch_size:>10} | "
                            f"{cpu_p50:>12.2f} | "
                            f"{engine_p50:>15.2f} | "
                            f"**{speedup:>5.2f}x** |"
                        )

            lines.append("")

        # Best configuration analysis
        lines.append("## Best Configurations")
        lines.append("")

        # Find best throughput per engine
        best_throughput: Dict[str, EngineResult] = {}
        for engine_type, engine_results in results_by_engine.items():
            best = max(engine_results, key=lambda r: r.throughput_img_per_sec)
            best_throughput[engine_type] = best

        lines.append("**Best Throughput by Engine:**")
        lines.append("")
        for engine_type in sorted(best_throughput.keys()):
            best = best_throughput[engine_type]
            lines.append(
                f"- **{engine_type}**: {best.throughput_img_per_sec:.1f} img/s "
                f"(batch size {best.batch_size})"
            )
        lines.append("")

        # Find best latency (lowest p95) per engine
        best_latency: Dict[str, EngineResult] = {}
        for engine_type, engine_results in results_by_engine.items():
            best = min(engine_results, key=lambda r: r.latency_ms_p95)
            best_latency[engine_type] = best

        lines.append("**Best Latency (p95) by Engine:**")
        lines.append("")
        for engine_type in sorted(best_latency.keys()):
            best = best_latency[engine_type]
            lines.append(
                f"- **{engine_type}**: {best.latency_ms_p95:.2f} ms p95 "
                f"(batch size {best.batch_size})"
            )
        lines.append("")

    # Environment Metadata
    if run.environment:
        lines.append("## Environment")
        lines.append("")
        lines.append("| Component | Version/Details |")
        lines.append("|-----------|-----------------|")

        env = run.environment
        if env.gpu_name:
            lines.append(f"| GPU | {env.gpu_name} |")
        if env.gpu_driver_version:
            lines.append(f"| GPU Driver | {env.gpu_driver_version} |")
        if env.cuda_version:
            lines.append(f"| CUDA | {env.cuda_version} |")
        if env.torch_version:
            lines.append(f"| PyTorch | {env.torch_version} |")
        if env.tensorrt_version:
            lines.append(f"| TensorRT | {env.tensorrt_version} |")
        if env.python_version:
            lines.append(f"| Python | {env.python_version} |")
        if env.cpu_model:
            lines.append(f"| CPU | {env.cpu_model} |")
        if env.git_commit:
            lines.append(f"| Git Commit | `{env.git_commit}` |")

        lines.append("")

        # Sanity check
        if env.sanity_check_passed is not None:
            status = "✅ PASSED" if env.sanity_check_passed else "❌ FAILED"
            lines.append(f"**Sanity Check:** {status}")
            lines.append("")

    # Timing Breakdown
    if run.timing:
        lines.append("## Timing Breakdown")
        lines.append("")
        lines.append("| Phase | Duration |")
        lines.append("|-------|----------|")

        timing = run.timing
        lines.append(f"| Total Duration | {timing.total_duration_sec:.2f} s |")
        lines.append(f"| Model Loading | {timing.model_load_sec:.2f} s |")
        lines.append(f"| Warmup | {timing.warmup_duration_sec:.2f} s |")
        lines.append(f"| Measurement | {timing.measurement_duration_sec:.2f} s |")

        lines.append("")

    # Telemetry Summary
    if run.telemetry and run.telemetry.samples:
        lines.append("## Telemetry Summary")
        lines.append("")

        samples = run.telemetry.samples
        lines.append(f"- **Samples Collected:** {len(samples)}")
        lines.append(f"- **Duration:** {samples[-1].t_ms / 1000:.1f} seconds")

        # Calculate averages
        avg_gpu_util = sum(s.gpu_utilization_percent for s in samples if s.gpu_utilization_percent is not None) / len(samples)
        avg_mem = sum(s.memory_used_mb for s in samples if s.memory_used_mb is not None) / len(samples)
        avg_temp = sum(s.temperature_celsius for s in samples if s.temperature_celsius is not None) / len(samples)
        avg_power = sum(s.power_usage_watts for s in samples if s.power_usage_watts is not None) / len(samples)

        lines.append("")
        lines.append("**Averages:**")
        lines.append(f"- GPU Utilization: {avg_gpu_util:.1f}%")
        lines.append(f"- Memory Used: {avg_mem:.0f} MB")
        lines.append(f"- Temperature: {avg_temp:.1f}°C")
        lines.append(f"- Power Usage: {avg_power:.1f} W")
        lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("This benchmark follows rigorous methodology for reproducible results:")
    lines.append("")
    lines.append("- **Warmup:** Excluded from measurements to avoid initialization overhead")
    lines.append("- **CUDA Synchronization:** `torch.cuda.synchronize()` before/after forward pass")
    lines.append("- **Fixed Inputs:** Same random seed (42) and input tensors for all runs")
    lines.append("- **Statistics:** p50/p95 percentiles + mean/stddev for robustness")
    lines.append("- **Sanity Checks:** Cross-engine validation of predictions")
    lines.append("")
    lines.append("For detailed methodology, see [METHODOLOGY.md](https://github.com/your-repo/docs/METHODOLOGY.md)")
    lines.append("")

    # Errors (if failed)
    if run.status == "failed" and run.error_message:
        lines.append("## Error Details")
        lines.append("")
        lines.append(f"**Error Message:**")
        lines.append("```")
        lines.append(run.error_message)
        lines.append("```")
        lines.append("")

        if run.error_stack:
            lines.append("**Stack Trace:**")
            lines.append("```")
            lines.append(run.error_stack)
            lines.append("```")
            lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Generated by TensorRT BenchLab - Production-Quality Inference Benchmarking*")
    lines.append("")

    return "\n".join(lines)
