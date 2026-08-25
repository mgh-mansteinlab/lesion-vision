import os
import glob
import threading
import queue
import json
import re
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import cv2
import torch
import tempfile
import psutil
import gc
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from PIL import Image
import openslide
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set OpenCV to allow large images (for NDPI processing)
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))

from src.analytics.lesions import LesionMetrics
from src.infer.ndpi import NDPIExtractor
from src.analytics.batch_plots import BatchPlotsMixin
from src.infer.predictor import LesionPredictor


class MultiGPUBatchPredictor(BatchPlotsMixin):
    """
    Multi-GPU batch prediction class for processing directories of lesion images.
    Distributes work across multiple GPUs for faster processing.
    """
    
    def __init__(self, model_path: Optional[str] = None, num_gpus: Optional[int] = None, max_workers_per_gpu: int = 1):
        """
        Initialize the multi-GPU batch predictor.
        
        Args:
            model_path: Path to the trained model
            num_gpus: Number of GPUs to use (auto-detect if None)
            max_workers_per_gpu: Maximum number of worker threads per GPU (default: 1 to reduce memory usage)
        """
        self.model_path = model_path
        self.num_gpus = num_gpus or torch.cuda.device_count()
        if self.num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available")
        
        self.max_workers_per_gpu = max(1, max_workers_per_gpu)
        self.slide_results = defaultdict(list)
        self.all_metrics = []
        self.slide_aggregated_metrics = {}
        self.file_results = {}
        
        # Memory management
        self.max_tiles_in_memory = 1000  # Limit number of tiles in memory at once
        self.current_tiles_in_memory = 0
        self.memory_lock = threading.Lock()
        
        print(f"Initialized MultiGPUBatchPredictor with {self.num_gpus} GPUs and {self.max_workers_per_gpu} workers per GPU")
    
    def extract_slide_name(self, filename: str) -> str:
        """Extract slide name from filename."""
        base_name = Path(filename).stem
        match = re.match(r'([A-Za-z]+\d+)_\d+', base_name)
        if match:
            return match.group(1)
        else:
            parts = base_name.split('_')
            if len(parts) > 1:
                return '_'.join(parts[:-1])
            return base_name
    
    def worker_thread(self, gpu_id: int, image_queue: queue.Queue, results_queue: queue.Queue,
                     tile_size: int, overlap: int, summary_metrics: bool, figures: bool,
                     use_tta: bool = False):
        """
        Worker thread for a single GPU with memory management.
        Processes both regular images and extracted NDPI samples.
        """
        try:
            # Initialize predictor on this GPU
            device = f'cuda:{gpu_id}'
            predictor = LesionPredictor(self.model_path, device)
            
            while True:
                try:
                    # Check memory usage before getting next item
                    with self.memory_lock:
                        if self.current_tiles_in_memory >= self.max_tiles_in_memory:
                            time.sleep(1)  # Wait if too many tiles in memory
                            continue
                        
                    # Get next image from queue if memory allows
                    try:
                        item = image_queue.get(timeout=1)
                        if item is None:  # Sentinel value to stop
                            break
                            
                        image_path, pred_output_path, filename, slide_name = item
                        self.current_tiles_in_memory += 1
                    except queue.Empty:
                        continue
                    
                    try:
                        # Process image
                        predictor.save_prediction(
                            image_path, pred_output_path,
                            tile_size=tile_size, overlap=overlap,
                            summary_metrics=summary_metrics,
                            figures=figures,
                            use_tta=use_tta,
                        )
                        
                        # Get metrics
                        image_metrics = predictor.analyzer.metrics_list.copy()
                        
                        # Clear analyzer for next image
                        predictor.analyzer.metrics_list.clear()
                        predictor.analyzer.lesion_centers.clear()
                        
                        # Clear CUDA cache to free memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Send results back
                        results_queue.put((filename, slide_name, image_metrics))
                        
                    except Exception as e:
                        results_queue.put((None, None, f"Error processing {image_path}: {str(e)}"))
                    finally:
                        # Ensure we always decrement the counter
                        with self.memory_lock:
                            self.current_tiles_in_memory = max(0, self.current_tiles_in_memory - 1)
                    
                except Exception as e:
                    print(f"Error in worker {gpu_id}: {str(e)}")
                    time.sleep(1)  # Prevent tight loop on errors
                    
        except Exception as e:
            print(f"Worker {gpu_id} failed: {str(e)}")
        finally:
            # Clean up CUDA memory when done
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # Convert to MB

    def process_directory(self, input_dir: str, output_dir: str,
                         tile_size: int = 512, overlap: int = 128,
                         summary_metrics: bool = False,
                         figures: bool = False,
                         use_tta: bool = False) -> Dict:
        """
        Process all images using multiple GPUs with memory management.
        Handles both regular images and NDPI files with multiple samples.
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        analytics_dir = os.path.join(output_dir, 'analytics')
        plots_dir = os.path.join(output_dir, 'plots')
        
        for dir_path in [analytics_dir, plots_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Find all images (TIF/TIFF and NDPI)
        image_files = []
        ndpi_files = []
        
        # Separate NDPI files from regular images
        for ext in ['*.tif', '*.tiff', '*.ndpi']:
            for file in glob.glob(os.path.join(input_dir, ext)):
                if file.lower().endswith('.ndpi'):
                    ndpi_files.append(file)
                else:
                    image_files.append(file)
        
        # Process NDPI files first (if any)
        # NDPI extraction is I/O-bound, not GPU-bound, so we use the regular extractor
        if ndpi_files:
            print(f"Found {len(ndpi_files)} NDPI files to process")
            
            for ndpi_file in ndpi_files:
                try:
                    print(f"\nProcessing NDPI file: {ndpi_file}")
                    # Extract samples from NDPI to a temporary directory
                    temp_dir = os.path.join(output_dir, 'temp_ndpi_extract')
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # Extract samples (I/O-bound operation, no GPU needed)
                    ndpi_extractor = NDPIExtractor(input_file=ndpi_file, output_dir=temp_dir)
                    extracted_files = ndpi_extractor.extract_to_tif(
                        compression="lzw",
                        level=2,  # Same as training
                        debug=True
                    )
                    
                    # Add extracted files to image_files for processing
                    image_files.extend(extracted_files)
                    
                    # Clean up temporary directory if empty
                    try:
                        os.rmdir(temp_dir)
                    except OSError:
                        pass
                        
                except Exception as e:
                    print(f"Error processing NDPI file {ndpi_file}: {str(e)}")
                    continue
        
        if not image_files and not ndpi_files:
            raise ValueError(f"No .tif, .tiff, or .ndpi files found in {input_dir}")
        
        print(f"Found {len(image_files)} images to process with {self.num_gpus} GPUs")
        
        # Create queues
        image_queue = queue.Queue()
        results_queue = queue.Queue()
        
        # Populate image queue — save_prediction creates <file_id>/{images,csv,individual_lesions}
        for image_path in image_files:
            filename = os.path.basename(image_path)
            slide_name = self.extract_slide_name(filename)
            base_filename = os.path.splitext(filename)[0]
            pred_output_path = os.path.join(output_dir, base_filename)
            image_queue.put((image_path, pred_output_path, filename, slide_name))
        
        # Add sentinel values to stop workers
        for _ in range(self.num_gpus):
            image_queue.put(None)
        
        # Start worker threads with limited concurrency per GPU
        threads = []
        for gpu_id in range(self.num_gpus):
            for worker_num in range(self.max_workers_per_gpu):
                t = threading.Thread(
                    target=self.worker_thread, 
                    args=(gpu_id, image_queue, results_queue, 
                         tile_size, overlap, summary_metrics, figures, use_tta),
                    daemon=True  # Allow main thread to exit even if workers are running
                )
                t.start()
                threads.append(t)
                print(f"Started worker {worker_num + 1} for GPU {gpu_id}")
                
                # Small delay to stagger thread startup
                time.sleep(0.5)
        
        # Collect results with memory monitoring
        processed_count = 0
        last_memory_check = time.time()
        
        with tqdm(total=len(image_files), desc="Processing images") as pbar:
            while processed_count < len(image_files):
                try:
                    # Periodically check memory usage
                    current_time = time.time()
                    if current_time - last_memory_check > 30:  # Every 30 seconds
                        mem_usage = self.get_memory_usage()
                        print(f"\nCurrent memory usage: {mem_usage:.2f} MB")
                        if torch.cuda.is_available():
                            for i in range(self.num_gpus):
                                print(f"GPU {i} memory: {torch.cuda.memory_allocated(i) / (1024**2):.1f}MB / "
                                      f"{torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f}GB")
                        last_memory_check = current_time
                    
                    # Get result with shorter timeout for more responsive memory checks
                    result = results_queue.get(timeout=5)
                    filename, slide_name, metrics = result
                    
                    if filename is None:  # Error case
                        print(f"\nError: {metrics}")
                        processed_count += 1
                        pbar.update(1)
                        continue
                    
                    if isinstance(metrics, str):  # Error message
                        print(f"\nError processing {filename}: {metrics}")
                    else:
                        # Store results
                        self.slide_results[slide_name].extend(metrics)
                        self.all_metrics.extend(metrics)
                        self.file_results[filename] = metrics
                        
                        # Force garbage collection periodically
                        if processed_count % 10 == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    
                    processed_count += 1
                    pbar.update(1)
                    
                except queue.Empty:
                    # Just continue the loop to check memory usage again
                    continue
        
        # Wait for all threads to finish
        for t in threads:
            t.join()
        
        # Generate analytics (reuse from BatchPredictor)
        analytics_results = self._generate_analytics()
        
        # Save analytics
        self._save_analytics(analytics_results, analytics_dir)
        
        # Generate enhanced plots
        self._generate_enhanced_plots(analytics_results, plots_dir, output_dir)
        
        return analytics_results
    
    def _generate_analytics(self) -> Dict:
        """Generate comprehensive analytics (same as BatchPredictor)."""
        # Aggregate metrics by slide
        for slide_name, metrics_list in self.slide_results.items():
            if not metrics_list:
                continue
                
            ablation_diameters = [m.ablation_diameter_um for m in metrics_list]
            coagulation_widths = [m.coagulation_width_um for m in metrics_list]
            ablation_areas = [m.ablation_area_um2 for m in metrics_list]
            coagulation_areas = [m.coagulation_area_um2 for m in metrics_list]
            lesion_diameters = [m.lesion_diameter_um for m in metrics_list]
            
            self.slide_aggregated_metrics[slide_name] = {
                'num_lesions': len(metrics_list),
                'mean_ablation_diameter_um': np.mean(ablation_diameters),
                'std_ablation_diameter_um': np.std(ablation_diameters),
                'mean_coagulation_width_um': np.mean(coagulation_widths),
                'std_coagulation_width_um': np.std(coagulation_widths),
                'mean_lesion_diameter_um': np.mean(lesion_diameters),
                'std_lesion_diameter_um': np.std(lesion_diameters),
                'mean_ablation_area_um2': np.mean(ablation_areas),
                'std_ablation_area_um2': np.std(ablation_areas),
                'mean_coagulation_area_um2': np.mean(coagulation_areas),
                'std_coagulation_area_um2': np.std(coagulation_areas),
                'total_ablation_area_um2': np.sum(ablation_areas),
                'total_coagulation_area_um2': np.sum(coagulation_areas)
            }
        
        # Dataset-wide analytics
        if self.all_metrics:
            all_ablation_diameters = [m.ablation_diameter_um for m in self.all_metrics]
            all_coagulation_widths = [m.coagulation_width_um for m in self.all_metrics]
            all_ablation_areas = [m.ablation_area_um2 for m in self.all_metrics]
            all_coagulation_areas = [m.coagulation_area_um2 for m in self.all_metrics]
            all_lesion_diameters = [m.lesion_diameter_um for m in self.all_metrics]
            
            dataset_analytics = {
                'total_images_processed': len(self.slide_results),
                'total_lesions_detected': len(self.all_metrics),
                'mean_ablation_diameter_um': np.mean(all_ablation_diameters),
                'std_ablation_diameter_um': np.std(all_ablation_diameters),
                'median_ablation_diameter_um': np.median(all_ablation_diameters),
                'mean_coagulation_width_um': np.mean(all_coagulation_widths),
                'std_coagulation_width_um': np.std(all_coagulation_widths),
                'median_coagulation_width_um': np.median(all_coagulation_widths),
                'mean_lesion_diameter_um': np.mean(all_lesion_diameters),
                'std_lesion_diameter_um': np.std(all_lesion_diameters),
                'median_lesion_diameter_um': np.median(all_lesion_diameters),
                'mean_ablation_area_um2': np.mean(all_ablation_areas),
                'std_ablation_area_um2': np.std(all_ablation_areas),
                'mean_coagulation_area_um2': np.mean(all_coagulation_areas),
                'std_coagulation_area_um2': np.std(all_coagulation_areas),
                'total_ablation_area_um2': np.sum(all_ablation_areas),
                'total_coagulation_area_um2': np.sum(all_coagulation_areas)
            }
            
            slide_means_ablation = [metrics['mean_ablation_diameter_um'] 
                                  for metrics in self.slide_aggregated_metrics.values()]
            slide_means_coagulation = [metrics['mean_coagulation_width_um'] 
                                     for metrics in self.slide_aggregated_metrics.values()]
            
            slide_summary = {
                'num_slides': len(self.slide_aggregated_metrics),
                'mean_ablation_diameter_across_slides_um': np.mean(slide_means_ablation),
                'std_ablation_diameter_across_slides_um': np.std(slide_means_ablation),
                'mean_coagulation_width_across_slides_um': np.mean(slide_means_coagulation),
                'std_coagulation_width_across_slides_um': np.std(slide_means_coagulation),
                'lesions_per_slide_mean': np.mean([metrics['num_lesions'] 
                                                 for metrics in self.slide_aggregated_metrics.values()]),
                'lesions_per_slide_std': np.std([metrics['num_lesions'] 
                                               for metrics in self.slide_aggregated_metrics.values()])
            }
        else:
            dataset_analytics = {}
            slide_summary = {}
        
        # File-level analytics
        file_analytics = {}
        if self.file_results:
            for filename, metrics_list in self.file_results.items():
                if metrics_list:
                    ablation_diameters = [m.ablation_diameter_um for m in metrics_list]
                    coagulation_widths = [m.coagulation_width_um for m in metrics_list]
                    ablation_areas = [m.ablation_area_um2 for m in metrics_list]
                    coagulation_areas = [m.coagulation_area_um2 for m in metrics_list]
                    lesion_diameters = [m.lesion_diameter_um for m in metrics_list]
                    
                    file_analytics[filename] = {
                        'num_lesions': len(metrics_list),
                        'mean_ablation_diameter_um': np.mean(ablation_diameters),
                        'std_ablation_diameter_um': np.std(ablation_diameters),
                        'mean_coagulation_width_um': np.mean(coagulation_widths),
                        'std_coagulation_width_um': np.std(coagulation_widths),
                        'mean_lesion_diameter_um': np.mean(lesion_diameters),
                        'std_lesion_diameter_um': np.std(lesion_diameters),
                        'mean_ablation_area_um2': np.mean(ablation_areas),
                        'std_ablation_area_um2': np.std(ablation_areas),
                        'mean_coagulation_area_um2': np.mean(coagulation_areas),
                        'std_coagulation_area_um2': np.std(coagulation_areas),
                        'total_ablation_area_um2': np.sum(ablation_areas),
                        'total_coagulation_area_um2': np.sum(coagulation_areas)
                    }
        
        return {
            'dataset_analytics': dataset_analytics,
            'slide_summary': slide_summary,
            'slide_metrics': self.slide_aggregated_metrics,
            'file_analytics': file_analytics,
            'processing_timestamp': datetime.now().isoformat()
        }
    
    def _save_analytics(self, analytics_results: Dict, output_dir: str):
        """Save analytics results to files."""
        
        if self.slide_aggregated_metrics:
            slide_df = pd.DataFrame.from_dict(self.slide_aggregated_metrics, orient='index')
            slide_df.index.name = 'slide_name'
            slide_df.to_csv(os.path.join(output_dir, 'slide_metrics.csv'))
            
            lesion_data = []
            for slide_name, metrics_list in self.slide_results.items():
                for i, metric in enumerate(metrics_list):
                    lesion_data.append({
                        'slide_name': slide_name,
                        'lesion_id': f"{slide_name}_lesion_{i+1}",
                        'ablation_diameter_um': metric.ablation_diameter_um,
                        'coagulation_width_um': metric.coagulation_width_um,
                        'lesion_diameter_um': metric.lesion_diameter_um,
                        'ablation_area_um2': metric.ablation_area_um2,
                        'coagulation_area_um2': metric.coagulation_area_um2,
                        'center_x': metric.center[0],
                        'center_y': metric.center[1],
                        'touches_edge': bool(getattr(metric, 'touches_edge', False)),
                        'dist_to_edge_um': getattr(metric, 'dist_to_edge_um', float('nan')),
                    })
            
            lesion_df = pd.DataFrame(lesion_data)
            lesion_df.to_csv(os.path.join(output_dir, 'individual_lesions.csv'), index=False)
            
            if analytics_results.get('file_analytics'):
                file_df = pd.DataFrame.from_dict(analytics_results['file_analytics'], orient='index')
                file_df.index.name = 'filename'
                file_df.to_csv(os.path.join(output_dir, 'file_metrics.csv'))
                
                file_means = {
                    'total_files_processed': len(analytics_results['file_analytics']),
                    'total_lesions_all_files': sum([f['num_lesions'] for f in analytics_results['file_analytics'].values()]),
                    'mean_ablation_diameter_across_files_um': np.mean([f['mean_ablation_diameter_um'] for f in analytics_results['file_analytics'].values()]),
                    'std_ablation_diameter_across_files_um': np.std([f['mean_ablation_diameter_um'] for f in analytics_results['file_analytics'].values()]),
                    'mean_coagulation_width_across_files_um': np.mean([f['mean_coagulation_width_um'] for f in analytics_results['file_analytics'].values()]),
                    'std_coagulation_width_across_files_um': np.std([f['mean_coagulation_width_um'] for f in analytics_results['file_analytics'].values()]),
                    'mean_lesion_diameter_across_files_um': np.mean([f['mean_lesion_diameter_um'] for f in analytics_results['file_analytics'].values()]),
                    'std_lesion_diameter_across_files_um': np.std([f['mean_lesion_diameter_um'] for f in analytics_results['file_analytics'].values()]),
                    'mean_lesions_per_file': np.mean([f['num_lesions'] for f in analytics_results['file_analytics'].values()]),
                    'std_lesions_per_file': np.std([f['num_lesions'] for f in analytics_results['file_analytics'].values()]),
                    'total_ablation_area_all_files_um2': sum([f['total_ablation_area_um2'] for f in analytics_results['file_analytics'].values()]),
                    'total_coagulation_area_all_files_um2': sum([f['total_coagulation_area_um2'] for f in analytics_results['file_analytics'].values()])
                }
                
                with open(os.path.join(output_dir, 'overall_summary.json'), 'w') as f:
                    json.dump(file_means, f, indent=2)
                    
                summary_df = pd.DataFrame([file_means])
                summary_df.to_csv(os.path.join(output_dir, 'overall_summary.csv'), index=False)
        
        print(f"Analytics saved to {output_dir}")
        
        # Also save analytics plots in analytics directory
        analytics_plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(analytics_plots_dir, exist_ok=True)
        self._create_analytics_plots_in_analytics(analytics_results, analytics_plots_dir)
    
    def _save_per_file_summaries(self, output_dir: str):
        """Save individual lesion summary for each file."""
        summaries_dir = os.path.join(output_dir, 'file_summaries')
        os.makedirs(summaries_dir, exist_ok=True)
        
        for filename, metrics_list in self.file_results.items():
            if not metrics_list:
                continue
                
            lesion_data = []
            for i, metric in enumerate(metrics_list):
                lesion_data.append({
                    'lesion_id': f"{filename.replace('.tif', '')}_lesion_{i+1}",
                    'filename': filename,
                    'ablation_diameter_um': metric.ablation_diameter_um,
                    'coagulation_width_um': metric.coagulation_width_um,
                    'lesion_diameter_um': metric.lesion_diameter_um,
                    'ablation_area_um2': metric.ablation_area_um2,
                    'coagulation_area_um2': metric.coagulation_area_um2,
                    'center_x': metric.center[0],
                    'center_y': metric.center[1],
                    'touches_edge': bool(getattr(metric, 'touches_edge', False)),
                    'dist_to_edge_um': getattr(metric, 'dist_to_edge_um', float('nan')),
                })
            
            lesion_df = pd.DataFrame(lesion_data)
            summary_filename = f"{filename.replace('.tif', '')}_lesion_summary.csv"
            lesion_df.to_csv(os.path.join(summaries_dir, summary_filename), index=False)
            
            if lesion_data:
                ablation_diameters = [d['ablation_diameter_um'] for d in lesion_data]
                coagulation_widths = [d['coagulation_width_um'] for d in lesion_data]
                lesion_diameters = [d['lesion_diameter_um'] for d in lesion_data]
                ablation_areas = [d['ablation_area_um2'] for d in lesion_data]
                coagulation_areas = [d['coagulation_area_um2'] for d in lesion_data]
                
                file_stats = {
                    'filename': filename,
                    'num_lesions': len(lesion_data),
                    'mean_ablation_diameter_um': np.mean(ablation_diameters),
                    'std_ablation_diameter_um': np.std(ablation_diameters),
                    'mean_coagulation_width_um': np.mean(coagulation_widths),
                    'std_coagulation_width_um': np.std(coagulation_widths),
                    'mean_lesion_diameter_um': np.mean(lesion_diameters),
                    'std_lesion_diameter_um': np.std(lesion_diameters),
                    'mean_ablation_area_um2': np.mean(ablation_areas),
                    'std_ablation_area_um2': np.std(ablation_areas),
                    'mean_coagulation_area_um2': np.mean(coagulation_areas),
                    'std_coagulation_area_um2': np.std(coagulation_areas),
                    'total_ablation_area_um2': np.sum(ablation_areas),
                    'total_coagulation_area_um2': np.sum(coagulation_areas)
                }
                
                stats_filename = f"{filename.replace('.tif', '')}_statistics.json"
                with open(os.path.join(summaries_dir, stats_filename), 'w') as f:
                    json.dump(file_stats, f, indent=2)
        
        if self.file_results:
            master_summary = []
            for filename, metrics_list in self.file_results.items():
                if metrics_list:
                    ablation_diameters = [m.ablation_diameter_um for m in metrics_list]
                    coagulation_widths = [m.coagulation_width_um for m in metrics_list]
                    lesion_diameters = [m.lesion_diameter_um for m in metrics_list]
                    ablation_areas = [m.ablation_area_um2 for m in metrics_list]
                    coagulation_areas = [m.coagulation_area_um2 for m in metrics_list]
                    
                    master_summary.append({
                        'filename': filename,
                        'num_lesions': len(metrics_list),
                        'mean_ablation_diameter_um': np.mean(ablation_diameters),
                        'std_ablation_diameter_um': np.std(ablation_diameters),
                        'mean_coagulation_width_um': np.mean(coagulation_widths),
                        'std_coagulation_width_um': np.std(coagulation_widths),
                        'mean_lesion_diameter_um': np.mean(lesion_diameters),
                        'std_lesion_diameter_um': np.std(lesion_diameters),
                        'total_ablation_area_um2': np.sum(ablation_areas),
                        'total_coagulation_area_um2': np.sum(coagulation_areas)
                    })
            
            if master_summary:
                master_df = pd.DataFrame(master_summary)
                master_df.to_csv(os.path.join(summaries_dir, 'master_file_summary.csv'), index=False)
                
                overall_means = {
                    'total_files': len(master_summary),
                    'total_lesions': sum([s['num_lesions'] for s in master_summary]),
                    'mean_ablation_diameter_um': np.mean([s['mean_ablation_diameter_um'] for s in master_summary]),
                    'std_ablation_diameter_um': np.std([s['mean_ablation_diameter_um'] for s in master_summary]),
                    'mean_coagulation_width_um': np.mean([s['mean_coagulation_width_um'] for s in master_summary]),
                    'std_coagulation_width_um': np.std([s['mean_coagulation_width_um'] for s in master_summary]),
                    'mean_lesion_diameter_um': np.mean([s['mean_lesion_diameter_um'] for s in master_summary]),
                    'std_lesion_diameter_um': np.std([s['mean_lesion_diameter_um'] for s in master_summary]),
                    'mean_lesions_per_file': np.mean([s['num_lesions'] for s in master_summary]),
                    'total_ablation_area_um2': sum([s['total_ablation_area_um2'] for s in master_summary]),
                    'total_coagulation_area_um2': sum([s['total_coagulation_area_um2'] for s in master_summary])
                }
                
                with open(os.path.join(summaries_dir, 'overall_means.json'), 'w') as f:
                    json.dump(overall_means, f, indent=2)
                    
                means_df = pd.DataFrame([overall_means])
                means_df.to_csv(os.path.join(summaries_dir, 'overall_means.csv'), index=False)
        
        print(f"Per-file summaries saved to {summaries_dir}")
    

def main():
    """Example usage of the MultiGPUBatchPredictor class."""
    
    # Initialize multi-GPU batch predictor
    batch_predictor = MultiGPUBatchPredictor()
    
    # Process directory
    input_dir = "/path/to/your/images"
    output_dir = "/path/to/output"
    
    results = batch_predictor.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        tile_size=512,
        overlap=128,
        summary_metrics=True,
        figures=True,
    )
    
    print("Multi-GPU processing complete with enhanced visualizations!")
    print(f"Dataset analytics: {results['dataset_analytics']}")
    print(f"Slide summary: {results['slide_summary']}")


if __name__ == "__main__":
    main()
