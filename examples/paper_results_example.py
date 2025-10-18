"""
Example: Collecting and analyzing results for paper publication
Demonstrates how to use the results collection system
"""

import asyncio
import json
from pathlib import Path
from src.results import ResultsCollector, ResultsReporter
from src.config import get_project_root


async def example_1_basic_collection():
    """Example 1: Basic results collection"""
    print("\n=== Example 1: Basic Collection ===\n")

    async with ResultsCollector("http://localhost:8000") as collector:
        # Collect text analysis
        texts = [
            "I love this community",
            "Those people are subhuman",
            "Climate change is important"
        ]

        results = await collector.collect_text_analysis(texts=texts, explain=True)

        print(f"Collected {len(results)} text analysis results:")
        for result in results:
            print(f"  - Input: {result.input_source[:40]}...")
            print(f"    Prediction: {result.prediction}")
            print(f"    Confidence: {result.confidence:.4f}")
            print(f"    Time: {result.processing_time:.3f}s\n")

        # Save to file
        output_path = Path("paper_results/example_results.json")
        collector.save_results(output_path)
        print(f"Results saved to {output_path}\n")


async def example_2_image_collection():
    """Example 2: Image analysis collection"""
    print("\n=== Example 2: Image Collection ===\n")

    async with ResultsCollector("http://localhost:8000") as collector:
        # Use sample images from data directory
        data_dir = get_project_root() / "data" / "raw"
        images = list(data_dir.glob("**/*.jpg"))[:3]

        if images:
            image_paths = [str(p) for p in images]
            results = await collector.collect_image_analysis(
                image_paths=image_paths,
                explain=True
            )

            print(f"Collected {len(results)} image analysis results:")
            for result in results:
                print(f"  - Image: {result.input_source}")
                print(f"    Prediction: {result.prediction}")
                print(f"    Confidence: {result.confidence:.4f}\n")
        else:
            print("No sample images found. Download using:")
            print("  python scripts/download_datasets.py\n")


async def example_3_gdelt_collection():
    """Example 3: GDELT event collection"""
    print("\n=== Example 3: GDELT Collection ===\n")

    async with ResultsCollector("http://localhost:8000") as collector:
        queries = [
            "human rights violation",
            "freedom of speech",
            "digital censorship"
        ]

        results = await collector.collect_gdelt_analysis(
            queries=queries,
            max_records=50
        )

        print(f"Collected {len(results)} GDELT queries:")
        for result in results:
            events_count = result.metadata.get('total_events', 0)
            print(f"  - Query: {result.input_source}")
            print(f"    Events found: {events_count}")
            print(f"    Time: {result.processing_time:.3f}s\n")


def example_4_generate_report():
    """Example 4: Generate markdown report from results"""
    print("\n=== Example 4: Generate Report ===\n")

    # Load collected results
    results_json = Path("paper_results/example_results.json")

    if not results_json.exists():
        print(f"Results file not found: {results_json}")
        print("Run example_1_basic_collection() first\n")
        return

    # Parse results
    with open(results_json) as f:
        data = json.load(f)

    # Group by module
    results_by_module = {}
    for result in data.get('results', []):
        module = result['module']
        if module not in results_by_module:
            results_by_module[module] = []
        results_by_module[module].append(result)

    # Generate reports
    reporter = ResultsReporter()

    # Full report
    full_report = reporter.generate_full_report(results_by_module)
    report_path = Path("paper_results/example_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    reporter.save_report(report_path, full_report)

    print(f"Report generated: {report_path}\n")
    print("Report Preview:")
    print("=" * 60)
    print(full_report[:500] + "...\n")


def example_5_extract_statistics():
    """Example 5: Extract statistics from results for paper"""
    print("\n=== Example 5: Extract Statistics ===\n")

    results_json = Path("paper_results/example_results.json")

    if not results_json.exists():
        print(f"Results file not found: {results_json}")
        return

    with open(results_json) as f:
        data = json.load(f)

    print("Summary Statistics:\n")
    print(f"Collection Date: {data['collection_timestamp']}")
    print(f"Total Analyses: {data['total_results']}\n")

    # Group by module
    by_module = {}
    for result in data.get('results', []):
        module = result['module']
        if module not in by_module:
            by_module[module] = []
        by_module[module].append(result)

    # Calculate statistics
    for module, results in by_module.items():
        confidences = [r['confidence'] for r in results]
        times = [r['processing_time'] for r in results]

        print(f"Module: {module.upper()}")
        print(f"  Sample Count: {len(results)}")
        print(f"  Avg Confidence: {sum(confidences)/len(confidences):.4f}")
        print(f"  Min/Max Confidence: {min(confidences):.4f} / {max(confidences):.4f}")
        print(f"  Avg Processing Time: {sum(times)/len(times):.3f}s")
        print(f"  Total Time: {sum(times):.3f}s\n")


def example_6_create_paper_table():
    """Example 6: Create formatted table for paper"""
    print("\n=== Example 6: Create Paper Table ===\n")

    results_json = Path("paper_results/example_results.json")

    if not results_json.exists():
        print(f"Results file not found: {results_json}")
        return

    with open(results_json) as f:
        data = json.load(f)

    # Create formatted table
    print("Table 1: Analysis Results Summary\n")
    print("| Module | Samples | Avg Confidence | Avg Time (s) |")
    print("|--------|---------|----------------|----|")

    by_module = {}
    for result in data.get('results', []):
        module = result['module']
        if module not in by_module:
            by_module[module] = []
        by_module[module].append(result)

    for module, results in by_module.items():
        confidences = [r['confidence'] for r in results]
        times = [r['processing_time'] for r in results]
        avg_conf = sum(confidences) / len(confidences)
        avg_time = sum(times) / len(times)

        print(f"| {module:6} | {len(results):7} | {avg_conf:14.4f} | {avg_time:12.3f} |")

    print("\nNote: This table can be directly pasted into your paper\n")


async def main():
    """Run all examples"""
    print("=" * 60)
    print("AI Sentinel - Paper Results Collection Examples")
    print("=" * 60)

    try:
        # Example 1: Basic collection
        await example_1_basic_collection()

        # Example 2: Image collection (if data available)
        await example_2_image_collection()

        # Example 3: GDELT collection
        await example_3_gdelt_collection()

        # Example 4: Generate report
        example_4_generate_report()

        # Example 5: Extract statistics
        example_5_extract_statistics()

        # Example 6: Create paper table
        example_6_create_paper_table()

        print("\n" + "=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure API server is running: python run_api.py")
        print("2. Check API is accessible at http://localhost:8000")
        print("3. Review logs in logs/ai_sentinel.log")


if __name__ == "__main__":
    asyncio.run(main())