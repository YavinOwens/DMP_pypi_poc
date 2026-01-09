"""
CLI entry point for datamanagement_genai package
"""

def main():
    """CLI entry point - delegates to benchmark.main"""
    from ..benchmark import main as benchmark_main
    return benchmark_main()

__all__ = ["main"]
