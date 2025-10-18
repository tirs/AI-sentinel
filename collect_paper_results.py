"""
Automated results collection script for paper.
Collects data from all analysis modules and generates markdown reports with tables.
Usage: python collect_paper_results.py
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any
import logging
import argparse
from datetime import datetime

from src.results import ResultsCollector, ResultsReporter
from src.config import get_project_root
from src.utils import get_logger

logger = get_logger(__name__)


class PaperResultsCollector:
    """Orchestrates automated collection of all analysis results for paper"""

    def __init__(self, output_dir: Path = None, api_url: str = "http://localhost:8000"):
        self.output_dir = output_dir or get_project_root() / "paper_results" / "analysis_data"
        self.api_url = api_url
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def collect_all_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """Main orchestration function"""
        logger.info("Starting automated results collection for paper...")
        all_results = {}

        try:
            async with ResultsCollector(self.api_url) as collector:
                # Text analysis
                logger.info("Collecting text analysis results...")
                text_samples = await self._get_text_samples()
                if text_samples:
                    text_results = await collector.collect_text_analysis(
                        texts=text_samples,
                        explain=True,
                        correlate=False
                    )
                    all_results['text'] = self._serialize_results(text_results)
                    logger.info(f"Collected {len(text_results)} text analysis results")

                # Image analysis
                logger.info("Collecting image analysis results...")
                image_paths = await self._get_image_samples()
                if image_paths:
                    image_results = await collector.collect_image_analysis(
                        image_paths=image_paths,
                        explain=True
                    )
                    all_results['image'] = self._serialize_results(image_results)
                    logger.info(f"Collected {len(image_results)} image analysis results")

                # Video analysis
                logger.info("Collecting video analysis results...")
                video_paths = await self._get_video_samples()
                if video_paths:
                    video_results = await collector.collect_video_analysis(
                        video_paths=video_paths,
                        frame_interval=10,
                        max_frames=30
                    )
                    all_results['video'] = self._serialize_results(video_results)
                    logger.info(f"Collected {len(video_results)} video analysis results")

                # GDELT analysis
                logger.info("Collecting GDELT analysis results...")
                gdelt_queries = self._get_gdelt_queries()
                if gdelt_queries:
                    gdelt_results = await collector.collect_gdelt_analysis(
                        queries=gdelt_queries,
                        max_records=100,
                        time_range_days=7
                    )
                    all_results['gdelt'] = self._serialize_results(gdelt_results)
                    logger.info(f"Collected {len(gdelt_results)} GDELT analysis results")

                # Save raw results
                results_json_path = self.output_dir / "analysis_results.json"
                collector.save_results(results_json_path)

        except Exception as e:
            logger.error(f"Error during results collection: {e}")
            raise

        return all_results

    @staticmethod
    def _serialize_results(results: List) -> List[Dict[str, Any]]:
        """Convert result objects to dictionaries"""
        serialized = []
        for r in results:
            data = {
                'module': r.module,
                'input_id': r.input_id,
                'input_source': r.input_source,
                'prediction': r.prediction,
                'confidence': r.confidence,
                'timestamp': r.timestamp.isoformat(),
                'metadata': r.metadata,
                'processing_time': r.processing_time
            }
            serialized.append(data)
        return serialized

    @staticmethod
    async def _get_text_samples() -> List[str]:
        """Get text samples for analysis"""
        return [
            "I love this community and respect all its members",
            "This group of people are criminals and should be eliminated",
            "I disagree with your political opinion",
            "Those people are subhuman and don't deserve rights",
            "Climate change is an important global issue",
            "We should welcome all immigrants to our country",
            "Wonderful day to celebrate diversity and inclusion",
            "They don't belong here and should leave"
        ]

    @staticmethod
    async def _get_image_samples() -> List[str]:
        """Get image paths for analysis"""
        sample_dir = get_project_root() / "data" / "raw"
        image_paths = []

        # Look for sample images
        for img_type in ["Celeb-real", "Celeb-synthesis"]:
            type_dir = sample_dir / img_type
            if type_dir.exists():
                images = list(type_dir.glob("**/*.jpg")) + list(type_dir.glob("**/*.png"))
                image_paths.extend([str(p) for p in images[:2]])  # Take first 2 from each

        return image_paths[:10]  # Limit to 10 images for paper

    @staticmethod
    async def _get_video_samples() -> List[str]:
        """Get video paths for analysis"""
        sample_dir = get_project_root() / "data" / "raw"
        video_paths = []

        if sample_dir.exists():
            videos = list(sample_dir.glob("**/*.mp4")) + list(sample_dir.glob("**/*.avi"))
            video_paths = [str(p) for p in videos[:3]]  # Take first 3 videos

        return video_paths

    @staticmethod
    def _get_gdelt_queries() -> List[str]:
        """Get GDELT search queries"""
        return [
            "human rights violation",
            "freedom of speech",
            "digital censorship",
            "online harassment",
            "misinformation",
            "deep fake",
            "fake news"
        ]

    def generate_reports(self, results_by_module: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Path]:
        """Generate markdown reports from results"""
        logger.info("Generating markdown reports...")
        reports = {}

        reporter = ResultsReporter()

        # Generate full report
        full_report = reporter.generate_full_report(results_by_module)
        full_report_path = self.output_dir / "PAPER_RESULTS.md"
        reporter.save_report(full_report_path, full_report)
        reports['full'] = full_report_path
        logger.info(f"Full report saved to {full_report_path}")

        # Generate module-specific reports
        if 'text' in results_by_module:
            text_report = reporter.generate_text_analysis_report(results_by_module['text'])
            text_path = self.output_dir / "TEXT_ANALYSIS.md"
            with open(text_path, 'w') as f:
                f.write("# Text Analysis Results\n\n" + text_report)
            reports['text'] = text_path
            logger.info(f"Text report saved to {text_path}")

        if 'image' in results_by_module:
            image_report = reporter.generate_image_analysis_report(results_by_module['image'])
            image_path = self.output_dir / "IMAGE_ANALYSIS.md"
            with open(image_path, 'w') as f:
                f.write("# Image Analysis Results\n\n" + image_report)
            reports['image'] = image_path
            logger.info(f"Image report saved to {image_path}")

        if 'video' in results_by_module:
            video_report = reporter.generate_video_analysis_report(results_by_module['video'])
            video_path = self.output_dir / "VIDEO_ANALYSIS.md"
            with open(video_path, 'w') as f:
                f.write("# Video Analysis Results\n\n" + video_report)
            reports['video'] = video_path
            logger.info(f"Video report saved to {video_path}")

        if 'gdelt' in results_by_module:
            gdelt_report = reporter.generate_gdelt_analysis_report(results_by_module['gdelt'])
            gdelt_path = self.output_dir / "GDELT_ANALYSIS.md"
            with open(gdelt_path, 'w') as f:
                f.write("# GDELT Analysis Results\n\n" + gdelt_report)
            reports['gdelt'] = gdelt_path
            logger.info(f"GDELT report saved to {gdelt_path}")

        return reports

    def generate_summary_json(self, results_by_module: Dict[str, List[Dict[str, Any]]]) -> Path:
        """Generate JSON summary of results"""
        summary = {
            "collection_date": datetime.now().isoformat(),
            "modules": {},
            "total_analyses": 0
        }

        for module, results in results_by_module.items():
            if results:
                confidences = [r['confidence'] for r in results]
                times = [r['processing_time'] for r in results]

                summary['modules'][module] = {
                    "sample_count": len(results),
                    "avg_confidence": sum(confidences) / len(confidences),
                    "min_confidence": min(confidences),
                    "max_confidence": max(confidences),
                    "avg_processing_time": sum(times) / len(times),
                    "results": results
                }
                summary['total_analyses'] += len(results)

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary JSON saved to {summary_path}")
        return summary_path

    async def run(self) -> Dict[str, Path]:
        """Run complete collection and reporting pipeline"""
        try:
            # Collect results
            results_by_module = await self.collect_all_results()

            # Generate reports
            reports = self.generate_reports(results_by_module)

            # Generate summary
            summary_path = self.generate_summary_json(results_by_module)
            reports['summary'] = summary_path

            logger.info("Results collection and report generation complete!")
            logger.info(f"Output directory: {self.output_dir}")

            return reports

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Collect paper results from AI Sentinel")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: paper_results/analysis_data)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)"
    )

    args = parser.parse_args()

    collector = PaperResultsCollector(
        output_dir=args.output_dir,
        api_url=args.api_url
    )

    reports = await collector.run()

    print("\nGenerated Reports:")
    for report_type, report_path in reports.items():
        print(f"  {report_type}: {report_path}")

    print(f"\nAll results saved to: {collector.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())