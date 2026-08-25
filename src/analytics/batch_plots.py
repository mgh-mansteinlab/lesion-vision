"""Seaborn dashboards for multi-GPU batch prediction."""

import os
from collections import defaultdict
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


class BatchPlotsMixin:
    def _generate_enhanced_plots(self, analytics_results: Dict, plots_dir: str, batch_dir: str):
        """Generate enhanced visualization plots including grid views and analytics."""
        
        print("Generating enhanced visualizations...")
        
        # 1. Create grid view of all overlays
        self._create_overlay_grid(plots_dir, batch_dir)
        
        # 2. Create comprehensive analytics plots
        self._create_analytics_plots(analytics_results, plots_dir)
        
        # 3. Create progression plots
        self._create_progression_plots(analytics_results, plots_dir)
        
        print(f"Enhanced plots saved to {plots_dir}")
    
    def _create_overlay_grid(self, plots_dir: str, batch_dir: str):
        """Create grid view of original and corrected overlays."""
        
        skip = {'analytics', 'plots', 'temp_ndpi_extract'}
        pred_dirs = sorted(
            d for d in os.listdir(batch_dir)
            if os.path.isdir(os.path.join(batch_dir, d)) and d not in skip
        )
        
        if not pred_dirs:
            print("No prediction directories found for grid visualization")
            return
        
        max_images = min(20, len(pred_dirs))
        pred_dirs = pred_dirs[:max_images]
        
        cols = min(4, max_images)
        rows = (max_images + cols - 1) // cols
        
        # --- original overlay grid ---
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Original Overlays Grid', fontsize=16, fontweight='bold')
        
        for idx, pred_dir in enumerate(pred_dirs):
            row, col = idx // cols, idx % cols
            base_name = pred_dir
            images_subdir = os.path.join(batch_dir, pred_dir, 'images')
            
            overlay_patterns = [
                os.path.join(images_subdir, f"{base_name}_analysis.png"),
                os.path.join(images_subdir, f"{base_name}_overlay.png"),
            ]
            
            overlay_found = False
            for overlay_path in overlay_patterns:
                if os.path.exists(overlay_path):
                    img = Image.open(overlay_path)
                    axes[row, col].imshow(img)
                    axes[row, col].set_title(base_name, fontsize=10)
                    overlay_found = True
                    break
            
            if not overlay_found:
                axes[row, col].text(0.5, 0.5, 'No overlay\nfound', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(base_name, fontsize=10)
            
            axes[row, col].axis('off')
        
        for idx in range(max_images, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'original_overlays_grid.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # --- corrected overlay grid ---
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Corrected Overlays Grid', fontsize=16, fontweight='bold')
        
        for idx, pred_dir in enumerate(pred_dirs):
            row, col = idx // cols, idx % cols
            base_name = pred_dir
            images_subdir = os.path.join(batch_dir, pred_dir, 'images')
            
            corrected_patterns = [
                os.path.join(images_subdir, f"{base_name}_corrected_overlay.png"),
                os.path.join(images_subdir, f"{base_name}_corrected.png"),
            ]
            
            overlay_found = False
            for corrected_path in corrected_patterns:
                if os.path.exists(corrected_path):
                    img = Image.open(corrected_path)
                    axes[row, col].imshow(img)
                    axes[row, col].set_title(base_name, fontsize=10)
                    overlay_found = True
                    break
            
            if not overlay_found:
                fallback_patterns = [
                    os.path.join(images_subdir, f"{base_name}_analysis.png"),
                    os.path.join(images_subdir, f"{base_name}_overlay.png"),
                ]
                for overlay_path in fallback_patterns:
                    if os.path.exists(overlay_path):
                        img = Image.open(overlay_path)
                        axes[row, col].imshow(img)
                        axes[row, col].set_title(f"{base_name} (orig)", fontsize=10)
                        overlay_found = True
                        break
            
            if not overlay_found:
                axes[row, col].text(0.5, 0.5, 'No overlay\nfound', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(base_name, fontsize=10)
            
            axes[row, col].axis('off')
        
        for idx in range(max_images, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'corrected_overlays_grid.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_analytics_plots(self, analytics_results: Dict, plots_dir: str):
        """Create comprehensive analytics visualization plots."""
        
        if not self.file_results:
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create analytics subdirectory
        analytics_plots_dir = os.path.join(plots_dir, 'analytics')
        os.makedirs(analytics_plots_dir, exist_ok=True)
        
        # 1. Per-file metrics comparison
        self._plot_per_file_metrics(analytics_plots_dir)
        
        # 2. Distribution analysis
        self._plot_distribution_analysis(analytics_plots_dir)
        
        # 3. Correlation analysis
        self._plot_correlation_analysis(analytics_plots_dir)
        
        # 4. Summary dashboard
        self._plot_summary_dashboard(analytics_results, analytics_plots_dir)
    
    def _plot_per_file_metrics(self, output_dir: str):
        """Plot per-file metrics comparison."""
        
        # Prepare data
        filenames = sorted(self.file_results.keys())
        ablation_means = []
        coagulation_means = []
        lesion_counts = []
        
        for filename in filenames:
            metrics = self.file_results[filename]
            if metrics:
                ablation_means.append(np.mean([m.ablation_diameter_um for m in metrics]))
                coagulation_means.append(np.mean([m.coagulation_width_um for m in metrics]))
                lesion_counts.append(len(metrics))
            else:
                ablation_means.append(0)
                coagulation_means.append(0)
                lesion_counts.append(0)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Per-File Metrics Analysis', fontsize=16, fontweight='bold')
        
        # Ablation diameter by file
        axes[0, 0].bar(range(len(filenames)), ablation_means, alpha=0.7, color='red')
        axes[0, 0].set_title('Mean Ablation Diameter by File')
        axes[0, 0].set_ylabel('Diameter (μm)')
        axes[0, 0].set_xticks(range(len(filenames)))
        axes[0, 0].set_xticklabels([f.replace('.tif', '') for f in filenames], rotation=45, ha='right')
        
        # Coagulation width by file
        axes[0, 1].bar(range(len(filenames)), coagulation_means, alpha=0.7, color='blue')
        axes[0, 1].set_title('Mean Coagulation Width by File')
        axes[0, 1].set_ylabel('Width (μm)')
        axes[0, 1].set_xticks(range(len(filenames)))
        axes[0, 1].set_xticklabels([f.replace('.tif', '') for f in filenames], rotation=45, ha='right')
        
        # Lesion count by file
        axes[1, 0].bar(range(len(filenames)), lesion_counts, alpha=0.7, color='green')
        axes[1, 0].set_title('Lesion Count by File')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xticks(range(len(filenames)))
        axes[1, 0].set_xticklabels([f.replace('.tif', '') for f in filenames], rotation=45, ha='right')
        
        # Combined scatter plot
        axes[1, 1].scatter(ablation_means, coagulation_means, s=[c*10 for c in lesion_counts], 
                          alpha=0.6, c=range(len(filenames)), cmap='viridis')
        axes[1, 1].set_xlabel('Mean Ablation Diameter (μm)')
        axes[1, 1].set_ylabel('Mean Coagulation Width (μm)')
        axes[1, 1].set_title('Ablation vs Coagulation (bubble size = lesion count)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_file_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distribution_analysis(self, output_dir: str):
        """Plot distribution analysis."""
        
        if not self.all_metrics:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Extract all metrics
        ablation_diameters = [m.ablation_diameter_um for m in self.all_metrics]
        coagulation_widths = [m.coagulation_width_um for m in self.all_metrics]
        ablation_areas = [m.ablation_area_um2 for m in self.all_metrics]
        
        # Histograms
        axes[0, 0].hist(ablation_diameters, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0, 0].set_title('Ablation Diameter Distribution')
        axes[0, 0].set_xlabel('Diameter (μm)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(ablation_diameters), color='darkred', linestyle='--', 
                          label=f'Mean: {np.mean(ablation_diameters):.1f}')
        axes[0, 0].legend()
        
        axes[0, 1].hist(coagulation_widths, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].set_title('Coagulation Width Distribution')
        axes[0, 1].set_xlabel('Width (μm)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(coagulation_widths), color='darkblue', linestyle='--',
                          label=f'Mean: {np.mean(coagulation_widths):.1f}')
        axes[0, 1].legend()
        
        axes[0, 2].hist(ablation_areas, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_title('Ablation Area Distribution')
        axes[0, 2].set_xlabel('Area (μm²)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(np.mean(ablation_areas), color='darkorange', linestyle='--',
                          label=f'Mean: {np.mean(ablation_areas):.0f}')
        axes[0, 2].legend()
        
        # Box plots
        axes[1, 0].boxplot(ablation_diameters, labels=['Ablation'])
        axes[1, 0].set_title('Ablation Diameter Box Plot')
        axes[1, 0].set_ylabel('Diameter (μm)')
        
        axes[1, 1].boxplot(coagulation_widths, labels=['Coagulation'])
        axes[1, 1].set_title('Coagulation Width Box Plot')
        axes[1, 1].set_ylabel('Width (μm)')
        
        axes[1, 2].boxplot(ablation_areas, labels=['Ablation'])
        axes[1, 2].set_title('Ablation Area Box Plot')
        axes[1, 2].set_ylabel('Area (μm²)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'distribution_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_analysis(self, output_dir: str):
        """Plot correlation analysis."""
        
        if not self.all_metrics:
            return
        
        # Create correlation data
        data = {
            'ablation_diameter': [m.ablation_diameter_um for m in self.all_metrics],
            'coagulation_width': [m.coagulation_width_um for m in self.all_metrics],
            'ablation_area': [m.ablation_area_um2 for m in self.all_metrics],
            'coagulation_area': [m.coagulation_area_um2 for m in self.all_metrics]
        }
        
        df = pd.DataFrame(data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Correlation heatmap
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[0, 0])
        axes[0, 0].set_title('Correlation Matrix')
        
        # Scatter plots
        axes[0, 1].scatter(data['ablation_diameter'], data['coagulation_width'], alpha=0.6)
        axes[0, 1].set_xlabel('Ablation Diameter (μm)')
        axes[0, 1].set_ylabel('Coagulation Width (μm)')
        axes[0, 1].set_title('Ablation Diameter vs Coagulation Width')
        
        axes[1, 0].scatter(data['ablation_area'], data['coagulation_area'], alpha=0.6)
        axes[1, 0].set_xlabel('Ablation Area (μm²)')
        axes[1, 0].set_ylabel('Coagulation Area (μm²)')
        axes[1, 0].set_title('Ablation Area vs Coagulation Area')
        
        # Pair plot style
        axes[1, 1].scatter(data['ablation_diameter'], data['ablation_area'], alpha=0.6)
        axes[1, 1].set_xlabel('Ablation Diameter (μm)')
        axes[1, 1].set_ylabel('Ablation Area (μm²)')
        axes[1, 1].set_title('Ablation Diameter vs Area')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_summary_dashboard(self, analytics_results: Dict, output_dir: str):
        """Create comprehensive summary dashboard."""
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Batch Processing Analytics Dashboard', fontsize=20, fontweight='bold')
        
        # Summary statistics text
        ax_text = fig.add_subplot(gs[0, :2])
        ax_text.axis('off')
        
        dataset_stats = analytics_results.get('dataset_analytics', {})
        file_stats = analytics_results.get('file_analytics', {})
        
        text_content = f"""
        Batch Processing Summary:
        • Total Files Processed: {len(file_stats)}
        • Total Lesions Detected: {dataset_stats.get('total_lesions_detected', 0)}
        • Mean Ablation Diameter: {dataset_stats.get('mean_ablation_diameter_um', 0):.1f} ± {dataset_stats.get('std_ablation_diameter_um', 0):.1f} μm
        • Mean Coagulation Width: {dataset_stats.get('mean_coagulation_width_um', 0):.1f} ± {dataset_stats.get('std_coagulation_width_um', 0):.1f} μm
        • Processing Timestamp: {analytics_results.get('processing_timestamp', 'N/A')}
        """
        
        ax_text.text(0.05, 0.95, text_content, transform=ax_text.transAxes, 
                    fontsize=12, verticalalignment='top', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        # File count and lesion distribution
        if file_stats:
            filenames = sorted(file_stats.keys())
            lesion_counts = [file_stats[f]['num_lesions'] for f in filenames]
            
            ax_files = fig.add_subplot(gs[0, 2:])
            ax_files.bar(range(len(filenames)), lesion_counts, alpha=0.7)
            ax_files.set_title('Lesions per File')
            ax_files.set_ylabel('Count')
            ax_files.set_xticks(range(0, len(filenames), max(1, len(filenames)//10)))
            ax_files.set_xticklabels([filenames[i].replace('.tif', '') 
                                    for i in range(0, len(filenames), max(1, len(filenames)//10))], 
                                   rotation=45, ha='right')
        
        # Distribution plots
        if self.all_metrics:
            ablation_data = [m.ablation_diameter_um for m in self.all_metrics]
            coagulation_data = [m.coagulation_width_um for m in self.all_metrics]
            
            ax_dist1 = fig.add_subplot(gs[1, :2])
            ax_dist1.hist(ablation_data, bins=30, alpha=0.7, color='red', edgecolor='black')
            ax_dist1.set_title('Ablation Diameter Distribution')
            ax_dist1.set_xlabel('Diameter (μm)')
            ax_dist1.set_ylabel('Frequency')
            
            ax_dist2 = fig.add_subplot(gs[1, 2:])
            ax_dist2.hist(coagulation_data, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax_dist2.set_title('Coagulation Width Distribution')
            ax_dist2.set_xlabel('Width (μm)')
            ax_dist2.set_ylabel('Frequency')
            
            # Box plots
            ax_box1 = fig.add_subplot(gs[2, 0])
            ax_box1.boxplot(ablation_data, labels=['Ablation'])
            ax_box1.set_title('Ablation Diameter')
            ax_box1.set_ylabel('Diameter (μm)')
            
            ax_box2 = fig.add_subplot(gs[2, 1])
            ax_box2.boxplot(coagulation_data, labels=['Coagulation'])
            ax_box2.set_title('Coagulation Width')
            ax_box2.set_ylabel('Width (μm)')
            
            # Scatter plot
            ax_scatter = fig.add_subplot(gs[2, 2:])
            ax_scatter.scatter(ablation_data, coagulation_data, alpha=0.6)
            ax_scatter.set_xlabel('Ablation Diameter (μm)')
            ax_scatter.set_ylabel('Coagulation Width (μm)')
            ax_scatter.set_title('Ablation vs Coagulation Relationship')
        
        # File progression plot
        if file_stats:
            filenames = sorted(file_stats.keys())
            ablation_means = [file_stats[f]['mean_ablation_diameter_um'] for f in filenames]
            coagulation_means = [file_stats[f]['mean_coagulation_width_um'] for f in filenames]
            
            ax_prog = fig.add_subplot(gs[3, :])
            x_pos = range(len(filenames))
            ax_prog.plot(x_pos, ablation_means, 'ro-', label='Ablation Diameter', alpha=0.7)
            ax_prog.plot(x_pos, coagulation_means, 'bo-', label='Coagulation Width', alpha=0.7)
            ax_prog.set_title('Metrics Progression Across Files')
            ax_prog.set_xlabel('File Index (sorted by name)')
            ax_prog.set_ylabel('Measurement (μm)')
            ax_prog.legend()
            ax_prog.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, 'summary_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_progression_plots(self, analytics_results: Dict, plots_dir: str):
        """Create progression plots sorted by image names."""
        
        if not self.file_results:
            return
        
        # Sort files by name
        filenames = sorted(self.file_results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Progression Analysis (Sorted by Filename)', fontsize=16, fontweight='bold')
        
        # Prepare data
        ablation_means = []
        coagulation_means = []
        lesion_counts = []
        total_areas = []
        
        for filename in filenames:
            metrics = self.file_results[filename]
            if metrics:
                ablation_means.append(np.mean([m.ablation_diameter_um for m in metrics]))
                coagulation_means.append(np.mean([m.coagulation_width_um for m in metrics]))
                lesion_counts.append(len(metrics))
                total_areas.append(sum([m.ablation_area_um2 for m in metrics]))
            else:
                ablation_means.append(0)
                coagulation_means.append(0)
                lesion_counts.append(0)
                total_areas.append(0)
        
        x_pos = range(len(filenames))
        
        # Ablation diameter progression
        axes[0, 0].plot(x_pos, ablation_means, 'ro-', alpha=0.7)
        axes[0, 0].set_title('Ablation Diameter Progression')
        axes[0, 0].set_ylabel('Mean Diameter (μm)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Coagulation width progression
        axes[0, 1].plot(x_pos, coagulation_means, 'bo-', alpha=0.7)
        axes[0, 1].set_title('Coagulation Width Progression')
        axes[0, 1].set_ylabel('Mean Width (μm)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Lesion count progression
        axes[1, 0].plot(x_pos, lesion_counts, 'go-', alpha=0.7)
        axes[1, 0].set_title('Lesion Count Progression')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xlabel('File Index')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Total area progression
        axes[1, 1].plot(x_pos, total_areas, 'mo-', alpha=0.7)
        axes[1, 1].set_title('Total Ablation Area Progression')
        axes[1, 1].set_ylabel('Total Area (μm²)')
        axes[1, 1].set_xlabel('File Index')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'progression_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_analytics_plots_in_analytics(self, analytics_results: Dict, output_dir: str):
        """Create analytics plots specifically for the analytics directory."""
        
        if not self.file_results:
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Per-file metrics comparison
        self._plot_per_file_metrics(output_dir)
        
        # 2. Distribution analysis
        self._plot_distribution_analysis(output_dir)
        
        # 3. Correlation analysis
        self._plot_correlation_analysis(output_dir)
        
        # 4. Summary dashboard
        self._plot_summary_dashboard(analytics_results, output_dir)
        
        # 5. Mean metrics compilation plot (sorted by name)
        self._plot_mean_metrics_compilation(output_dir)

    def _plot_mean_metrics_compilation(self, output_dir: str):
        """Create compilation plot of mean metrics for each file, sorted by name."""
        
        if not self.file_results:
            return
        
        # Sort files by name and calculate means
        filenames = sorted(self.file_results.keys())
        file_means = []
        
        for filename in filenames:
            metrics_list = self.file_results[filename]
            if metrics_list:
                ablation_mean = np.mean([m.ablation_diameter_um for m in metrics_list])
                coagulation_mean = np.mean([m.coagulation_width_um for m in metrics_list])
                ablation_area_mean = np.mean([m.ablation_area_um2 for m in metrics_list])
                coagulation_area_mean = np.mean([m.coagulation_area_um2 for m in metrics_list])
                lesion_count = len(metrics_list)
            else:
                ablation_mean = coagulation_mean = ablation_area_mean = coagulation_area_mean = lesion_count = 0
            
            file_means.append({
                'filename': filename.replace('.tif', ''),
                'ablation_diameter_um': ablation_mean,
                'coagulation_width_um': coagulation_mean,
                'ablation_area_um2': ablation_area_mean,
                'coagulation_area_um2': coagulation_area_mean,
                'lesion_count': lesion_count
            })
        
        # Create comprehensive compilation plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Mean Metrics Compilation (Sorted by Filename)', fontsize=16, fontweight='bold')
        
        x_pos = range(len(file_means))
        filenames_short = [f['filename'] for f in file_means]
        
        # Plot 1: Ablation Diameter
        axes[0, 0].bar(x_pos, [f['ablation_diameter_um'] for f in file_means], alpha=0.7, color='red')
        axes[0, 0].set_title('Mean Ablation Diameter')
        axes[0, 0].set_ylabel('Diameter (μm)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        if len(filenames_short) <= 10:
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(filenames_short, rotation=45, ha='right')
        
        # Plot 2: Coagulation Width
        axes[0, 1].bar(x_pos, [f['coagulation_width_um'] for f in file_means], alpha=0.7, color='blue')
        axes[0, 1].set_title('Mean Coagulation Width')
        axes[0, 1].set_ylabel('Width (μm)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        if len(filenames_short) <= 10:
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(filenames_short, rotation=45, ha='right')
        
        # Plot 3: Lesion Count
        axes[0, 2].bar(x_pos, [f['lesion_count'] for f in file_means], alpha=0.7, color='green')
        axes[0, 2].set_title('Lesion Count per File')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].tick_params(axis='x', rotation=45)
        if len(filenames_short) <= 10:
            axes[0, 2].set_xticks(x_pos)
            axes[0, 2].set_xticklabels(filenames_short, rotation=45, ha='right')
        
        # Plot 4: Ablation Area
        axes[1, 0].bar(x_pos, [f['ablation_area_um2'] for f in file_means], alpha=0.7, color='orange')
        axes[1, 0].set_title('Mean Ablation Area')
        axes[1, 0].set_ylabel('Area (μm²)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        if len(filenames_short) <= 10:
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(filenames_short, rotation=45, ha='right')
        
        # Plot 5: Coagulation Area
        axes[1, 1].bar(x_pos, [f['coagulation_area_um2'] for f in file_means], alpha=0.7, color='purple')
        axes[1, 1].set_title('Mean Coagulation Area')
        axes[1, 1].set_ylabel('Area (μm²)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        if len(filenames_short) <= 10:
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(filenames_short, rotation=45, ha='right')
        
        # Plot 6: Combined line plot
        axes[1, 2].plot(x_pos, [f['ablation_diameter_um'] for f in file_means], 'ro-', label='Ablation Diameter', alpha=0.7)
        axes[1, 2].plot(x_pos, [f['coagulation_width_um'] for f in file_means], 'bo-', label='Coagulation Width', alpha=0.7)
        axes[1, 2].set_title('Combined Metrics Trend')
        axes[1, 2].set_ylabel('Measurement (μm)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].tick_params(axis='x', rotation=45)
        if len(filenames_short) <= 10:
            axes[1, 2].set_xticks(x_pos)
            axes[1, 2].set_xticklabels(filenames_short, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mean_metrics_compilation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save the mean metrics data as CSV
        df_means = pd.DataFrame(file_means)
        df_means.to_csv(os.path.join(output_dir, 'mean_metrics_by_file.csv'), index=False)


