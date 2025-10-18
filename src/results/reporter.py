"""
Report generation system for creating markdown documents with tables and figures.
Generates publication-ready results from collected analysis data.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
from src.results.collector import AnalysisResult


class ResultsReporter:
    """Generates markdown reports from collected analysis results"""

    def __init__(self, results_data: Dict[str, Any] = None):
        self.results_data = results_data or {}
        self.timestamp = datetime.now()

    @staticmethod
    def _load_results(json_path: Path) -> Dict[str, Any]:
        """Load results from JSON file"""
        with open(json_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def _create_markdown_table(data: List[Dict[str, Any]], title: str = "") -> str:
        """Convert data to markdown table"""
        if not data:
            return f"No data available for {title}"

        df = pd.DataFrame(data)
        return df.to_markdown(index=False)

    def generate_text_analysis_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate markdown report for text analysis"""
        report = "## Text Detection Analysis\n\n"

        # Summary statistics
        if results:
            df = pd.DataFrame(results)
            report += "### Summary Statistics\n\n"
            report += f"- **Total Samples**: {len(results)}\n"
            report += f"- **Average Confidence**: {df['confidence'].mean():.4f}\n"
            report += f"- **Avg Processing Time**: {df['processing_time'].mean():.3f}s\n"
            report += "\n"

            # Prediction distribution
            if 'prediction' in df.columns:
                pred_counts = df['prediction'].value_counts().to_dict()
                report += "### Prediction Distribution\n\n"
                for pred, count in pred_counts.items():
                    percentage = (count / len(results)) * 100
                    report += f"- **{pred}**: {count} ({percentage:.1f}%)\n"
                report += "\n"

            # Detailed results table
            display_cols = ['input_id', 'input_source', 'prediction', 'confidence', 'processing_time']
            display_cols = [c for c in display_cols if c in df.columns]

            if display_cols:
                report += "### Detailed Results\n\n"
                display_data = df[display_cols].copy()
                report += self._create_markdown_table(
                    display_data.to_dict('records'),
                    "Text Analysis"
                )
                report += "\n"

        return report

    def generate_image_analysis_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate markdown report for image analysis"""
        report = "## Image Detection Analysis\n\n"

        if results:
            df = pd.DataFrame(results)
            report += "### Summary Statistics\n\n"
            report += f"- **Total Samples**: {len(results)}\n"
            report += f"- **Average Confidence**: {df['confidence'].mean():.4f}\n"
            report += f"- **Avg Processing Time**: {df['processing_time'].mean():.3f}s\n"
            report += "\n"

            # Prediction distribution
            if 'prediction' in df.columns:
                pred_counts = df['prediction'].value_counts().to_dict()
                report += "### Prediction Distribution\n\n"
                for pred, count in pred_counts.items():
                    percentage = (count / len(results)) * 100
                    report += f"- **{pred}**: {count} ({percentage:.1f}%)\n"
                report += "\n"

            # Real vs Fake confidence comparison
            report += "### Confidence Analysis\n\n"
            report += f"- **Min Confidence**: {df['confidence'].min():.4f}\n"
            report += f"- **Max Confidence**: {df['confidence'].max():.4f}\n"
            report += f"- **Std Dev**: {df['confidence'].std():.4f}\n"
            report += "\n"

            # Detailed results table
            display_cols = ['input_id', 'input_source', 'prediction', 'confidence', 'processing_time']
            display_cols = [c for c in display_cols if c in df.columns]

            if display_cols:
                report += "### Detailed Results\n\n"
                display_data = df[display_cols].copy()
                report += self._create_markdown_table(
                    display_data.to_dict('records'),
                    "Image Analysis"
                )
                report += "\n"

        return report

    def generate_video_analysis_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate markdown report for video analysis"""
        report = "## Video Detection Analysis\n\n"

        if results:
            df = pd.DataFrame(results)
            report += "### Summary Statistics\n\n"
            report += f"- **Total Videos**: {len(results)}\n"
            report += f"- **Average Confidence**: {df['confidence'].mean():.4f}\n"
            report += f"- **Avg Processing Time**: {df['processing_time'].mean():.3f}s\n"
            report += "\n"

            # Frame analysis statistics
            report += "### Frame Analysis\n\n"
            if 'metadata' in results[0]:
                total_frames = sum(r.get('metadata', {}).get('frames_analyzed', 0) for r in results)
                report += f"- **Total Frames Analyzed**: {total_frames}\n"
                report += f"- **Avg Frames per Video**: {total_frames / max(len(results), 1):.1f}\n"
                report += "\n"

            # Prediction distribution
            if 'prediction' in df.columns:
                pred_counts = df['prediction'].value_counts().to_dict()
                report += "### Prediction Distribution\n\n"
                for pred, count in pred_counts.items():
                    percentage = (count / len(results)) * 100
                    report += f"- **{pred}**: {count} ({percentage:.1f}%)\n"
                report += "\n"

            # Detailed results table
            display_cols = ['input_id', 'input_source', 'prediction', 'confidence', 'processing_time']
            display_cols = [c for c in display_cols if c in df.columns]

            if display_cols:
                report += "### Detailed Results\n\n"
                display_data = df[display_cols].copy()
                report += self._create_markdown_table(
                    display_data.to_dict('records'),
                    "Video Analysis"
                )
                report += "\n"

        return report

    def generate_gdelt_analysis_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate markdown report for GDELT analysis"""
        report = "## Global Events (GDELT) Analysis\n\n"

        if results:
            df = pd.DataFrame(results)
            report += "### Summary Statistics\n\n"
            report += f"- **Total Queries**: {len(results)}\n"
            report += f"- **Avg Query Time**: {df['processing_time'].mean():.3f}s\n"
            report += "\n"

            # Events coverage
            report += "### Event Coverage\n\n"
            if 'metadata' in results[0]:
                total_events = sum(r.get('metadata', {}).get('total_events', 0) for r in results)
                report += f"- **Total Events Retrieved**: {total_events}\n"
                report += f"- **Avg Events per Query**: {total_events / max(len(results), 1):.1f}\n"
                report += "\n"

            # Query details table
            query_data = []
            for r in results:
                query_data.append({
                    'Query': r['input_source'],
                    'Events Found': r.get('metadata', {}).get('total_events', 0),
                    'Processing Time (s)': f"{r['processing_time']:.3f}"
                })

            if query_data:
                report += "### Query Results\n\n"
                report += self._create_markdown_table(query_data, "GDELT Queries")
                report += "\n"

        return report

    def generate_multimodal_analysis_report(self, all_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate markdown report for multimodal fusion analysis"""
        report = "## Multimodal Fusion Analysis\n\n"

        report += "### Overview\n\n"
        report += "This section presents results from multimodal analysis combining text, image, and GDELT data.\n\n"

        # Module comparison
        report += "### Module Comparison\n\n"
        comparison_data = []
        for module, results in all_results.items():
            if results:
                df = pd.DataFrame(results)
                comparison_data.append({
                    'Module': module.capitalize(),
                    'Sample Count': len(results),
                    'Avg Confidence': f"{df['confidence'].mean():.4f}",
                    'Avg Processing Time': f"{df['processing_time'].mean():.3f}s"
                })

        if comparison_data:
            report += self._create_markdown_table(comparison_data, "Module Comparison")
            report += "\n"

        report += "### Cross-Module Insights\n\n"
        report += "- Multimodal analysis combines predictions from multiple modules\n"
        report += "- Higher confidence scores indicate agreement across modules\n"
        report += "- Processing time increases with number of active modules\n"
        report += "\n"

        return report

    def generate_full_report(self, results_by_module: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate complete markdown report"""
        report = "# AI Sentinel Analysis Results Report\n\n"
        report += f"**Generated**: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"**Report Version**: 1.0\n\n"

        report += "## Executive Summary\n\n"
        total_results = sum(len(r) for r in results_by_module.values())
        report += f"- **Total Analyses**: {total_results}\n"
        report += f"- **Modules Tested**: {len(results_by_module)}\n"
        report += f"- **Report Date**: {self.timestamp.strftime('%Y-%m-%d')}\n\n"

        # Module-specific reports
        report += "---\n\n"

        if 'text' in results_by_module:
            report += self.generate_text_analysis_report(results_by_module['text'])
            report += "---\n\n"

        if 'image' in results_by_module:
            report += self.generate_image_analysis_report(results_by_module['image'])
            report += "---\n\n"

        if 'video' in results_by_module:
            report += self.generate_video_analysis_report(results_by_module['video'])
            report += "---\n\n"

        if 'gdelt' in results_by_module:
            report += self.generate_gdelt_analysis_report(results_by_module['gdelt'])
            report += "---\n\n"

        # Multimodal summary
        report += self.generate_multimodal_analysis_report(results_by_module)

        report += "---\n\n"
        report += "## Methodology\n\n"
        report += "All analyses were performed using the AI Sentinel API with default parameters.\n"
        report += "Processing times include API communication overhead.\n"
        report += "Confidence scores represent model certainty in predictions (0.0 to 1.0).\n\n"

        return report

    def save_report(self, output_path: Path, report_content: str) -> None:
        """Save report to markdown file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(report_content)

        print(f"Report saved to {output_path}")