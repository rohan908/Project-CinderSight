#!/usr/bin/env python3
"""
Fire Mask Temporal Sequence Reconstruction

This script analyzes fire masks across the entire Enhanced NDWS dataset to find matching
patterns and reconstruct temporal sequences of individual fires. This enables creating
longer temporal datasets for training attention mechanisms and long-term fire modeling.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Set
import tfrecord
from collections import defaultdict, deque
import hashlib

# Import shared configuration
from config import (
    ENHANCED_INPUT_FEATURES,
    OUTPUT_FEATURES,
    DEFAULT_DATA_SIZE
)

class FireSequenceReconstructor:
    """Reconstructs temporal fire sequences by matching fire patterns across the dataset."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize the sequence reconstructor.
        
        Args:
            data_dir: Directory containing TFRecord files.
        """
        self.data_dir = Path(data_dir)
        self.tfrecord_files = list(self.data_dir.glob("*.tfrecord"))
        print(f"Found {len(self.tfrecord_files)} TFRecord files")
        
    def _parse_tfrecord_example(self, record, sample_id: int):
        """Parse a single TFRecord example to extract fire masks with metadata.
        
        Args:
            record: A TFRecord example dictionary.
            sample_id: Unique identifier for this sample.
            
        Returns:
            Dictionary containing parsed fire masks and metadata.
        """
        parsed_data = {
            'sample_id': sample_id,
            'prev_fire_mask': None,
            'curr_fire_mask': None,
            'prev_fire_hash': None,
            'curr_fire_hash': None,
            'prev_fire_pixels': 0,
            'curr_fire_pixels': 0
        }
        
        # Extract fire masks
        for mask_type, key in [('prev', 'PrevFireMask'), ('curr', 'FireMask')]:
            if key in record:
                fire_mask = np.array(record[key]).reshape(DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE)
                
                # Count fire pixels (excluding invalid -1 values)
                valid_mask = fire_mask != -1
                fire_pixels = np.sum((fire_mask == 1) & valid_mask)
                
                # Create a hash of the fire pattern for quick matching
                # Only hash the fire pixels (1s) and ignore 0s and -1s for better matching
                fire_pattern = (fire_mask == 1).astype(np.uint8)
                pattern_hash = hashlib.md5(fire_pattern.tobytes()).hexdigest()
                
                parsed_data[f'{mask_type}_fire_mask'] = fire_mask
                parsed_data[f'{mask_type}_fire_hash'] = pattern_hash
                parsed_data[f'{mask_type}_fire_pixels'] = fire_pixels
            else:
                print(f"Warning: {key} not found in sample {sample_id}")
                
        return parsed_data
    
    def load_all_fire_data(self, max_samples: int = 2000) -> List[Dict]:
        """Load all fire mask data from TFRecord files.
        
        Args:
            max_samples: Maximum number of samples to load.
            
        Returns:
            List of dictionaries containing fire mask data with metadata.
        """
        print(f"Loading all fire mask data (max {max_samples} samples)...")
        
        all_fire_data = []
        samples_loaded = 0
        
        for tfrecord_file in self.tfrecord_files:
            if samples_loaded >= max_samples:
                break
                
            try:
                dataset = tfrecord.torch.TFRecordDataset(str(tfrecord_file), index_path=None)
                
                for record in dataset:
                    if samples_loaded >= max_samples:
                        break
                        
                    if samples_loaded % 200 == 0:
                        print(f"Processing sample {samples_loaded+1}/{max_samples}")
                    
                    try:
                        parsed = self._parse_tfrecord_example(record, samples_loaded)
                        # Only include samples with actual fire data
                        if parsed['prev_fire_pixels'] > 0 or parsed['curr_fire_pixels'] > 0:
                            all_fire_data.append(parsed)
                        samples_loaded += 1
                        
                    except Exception as e:
                        print(f"Error parsing record {samples_loaded}: {e}")
                        samples_loaded += 1
                        continue
                        
            except Exception as e:
                print(f"Error reading TFRecord file {tfrecord_file}: {e}")
                continue
        
        print(f"Loaded {len(all_fire_data)} samples with fire data out of {samples_loaded} total samples")
        return all_fire_data
    
    def find_exact_pattern_matches(self, fire_data: List[Dict], min_fire_pixels: int = 5) -> Dict:
        """Find exact matches between previous fire masks and current fire masks across all samples.
        
        Args:
            fire_data: List of fire data dictionaries.
            min_fire_pixels: Minimum number of fire pixels to consider for matching.
            
        Returns:
            Dictionary containing match results and sequence information.
        """
        print(f"Searching for exact pattern matches (min {min_fire_pixels} fire pixels)...")
        
        # Build hash tables for quick lookup
        prev_patterns = defaultdict(list)  # hash -> list of sample indices
        curr_patterns = defaultdict(list)  # hash -> list of sample indices
        
        # Index all patterns by their hash
        for i, sample in enumerate(fire_data):
            if sample['prev_fire_pixels'] >= min_fire_pixels:
                prev_patterns[sample['prev_fire_hash']].append(i)
            if sample['curr_fire_pixels'] >= min_fire_pixels:
                curr_patterns[sample['curr_fire_hash']].append(i)
        
        print(f"Indexed {len(prev_patterns)} unique previous fire patterns")
        print(f"Indexed {len(curr_patterns)} unique current fire patterns")
        
        # Find matches: prev_pattern -> curr_pattern
        matches = []
        matched_samples = set()
        
        for prev_hash, prev_indices in prev_patterns.items():
            if prev_hash in curr_patterns:
                curr_indices = curr_patterns[prev_hash]
                
                # Create all possible matches between prev and curr
                for prev_idx in prev_indices:
                    for curr_idx in curr_indices:
                        # Don't match a sample to itself
                        if prev_idx != curr_idx:
                            matches.append({
                                'prev_sample_id': fire_data[prev_idx]['sample_id'],
                                'curr_sample_id': fire_data[curr_idx]['sample_id'],
                                'prev_idx': prev_idx,
                                'curr_idx': curr_idx,
                                'pattern_hash': prev_hash,
                                'fire_pixels': fire_data[prev_idx]['prev_fire_pixels'],
                                'match_type': 'exact'
                            })
                            matched_samples.add(prev_idx)
                            matched_samples.add(curr_idx)
        
        results = {
            'total_samples': len(fire_data),
            'total_patterns_prev': len(prev_patterns),
            'total_patterns_curr': len(curr_patterns),
            'total_matches': len(matches),
            'matched_samples': len(matched_samples),
            'unmatched_samples': len(fire_data) - len(matched_samples),
            'matches': matches
        }
        
        return results
    
    def build_fire_sequences(self, matches: List[Dict], fire_data: List[Dict]) -> Dict:
        """Build temporal sequences from the matched patterns.
        
        Args:
            matches: List of pattern matches.
            fire_data: Original fire data.
            
        Returns:
            Dictionary containing sequence information.
        """
        print("Building temporal fire sequences...")
        
        # Build a graph of connections: sample_id -> [connected_sample_ids]
        connections = defaultdict(set)
        
        for match in matches:
            prev_id = match['prev_sample_id']
            curr_id = match['curr_sample_id']
            connections[prev_id].add(curr_id)
        
        # Find connected components (fire sequences)
        visited = set()
        sequences = []
        
        def dfs_sequence(start_id, current_sequence):
            """Depth-first search to build fire sequence."""
            if start_id in visited:
                return current_sequence
            
            visited.add(start_id)
            current_sequence.append(start_id)
            
            # Follow all connections
            for next_id in connections.get(start_id, []):
                if next_id not in visited:
                    dfs_sequence(next_id, current_sequence)
            
            return current_sequence
        
        # Build all sequences
        for sample_id in connections.keys():
            if sample_id not in visited:
                sequence = dfs_sequence(sample_id, [])
                if len(sequence) > 1:  # Only keep sequences with multiple steps
                    sequences.append(sequence)
        
        # Sort sequences by length (longest first)
        sequences.sort(key=len, reverse=True)
        
        sequence_stats = {
            'total_sequences': len(sequences),
            'longest_sequence': len(sequences[0]) if sequences else 0,
            'average_sequence_length': np.mean([len(seq) for seq in sequences]) if sequences else 0,
            'sequences': sequences[:20],  # Keep top 20 longest sequences
            'sequence_lengths': [len(seq) for seq in sequences]
        }
        
        return sequence_stats
    
    def analyze_sequence_patterns(self, matches: List[Dict], sequence_stats: Dict) -> Dict:
        """Analyze patterns in the reconstructed sequences.
        
        Args:
            matches: List of pattern matches.
            sequence_stats: Sequence statistics.
            
        Returns:
            Analysis results dictionary.
        """
        print("Analyzing sequence patterns...")
        
        # Group matches by pattern hash to find frequently occurring patterns
        pattern_frequency = defaultdict(int)
        pattern_sizes = defaultdict(list)
        
        for match in matches:
            pattern_hash = match['pattern_hash']
            pattern_frequency[pattern_hash] += 1
            pattern_sizes[pattern_hash].append(match['fire_pixels'])
        
        # Find most common patterns
        common_patterns = sorted(pattern_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        
        analysis = {
            'unique_patterns': len(pattern_frequency),
            'most_common_patterns': [
                {
                    'pattern_hash': pattern_hash[:8],  # Shortened hash for display
                    'frequency': freq,
                    'avg_size': np.mean(pattern_sizes[pattern_hash]),
                    'size_range': (min(pattern_sizes[pattern_hash]), max(pattern_sizes[pattern_hash]))
                }
                for pattern_hash, freq in common_patterns
            ],
            'reuse_statistics': {
                'patterns_used_once': sum(1 for freq in pattern_frequency.values() if freq == 1),
                'patterns_used_multiple': sum(1 for freq in pattern_frequency.values() if freq > 1),
                'max_pattern_reuse': max(pattern_frequency.values()) if pattern_frequency else 0
            }
        }
        
        return analysis
    
    def print_reconstruction_summary(self, match_results: Dict, sequence_stats: Dict, analysis: Dict):
        """Print a comprehensive summary of the sequence reconstruction.
        
        Args:
            match_results: Pattern matching results.
            sequence_stats: Sequence building statistics.
            analysis: Pattern analysis results.
        """
        print("\n" + "="*70)
        print("FIRE SEQUENCE RECONSTRUCTION SUMMARY")
        print("="*70)
        
        print(f"DATASET OVERVIEW:")
        print(f"  Total samples with fire: {match_results['total_samples']:,}")
        print(f"  Unique previous patterns: {match_results['total_patterns_prev']:,}")
        print(f"  Unique current patterns: {match_results['total_patterns_curr']:,}")
        
        print(f"\nPATTERN MATCHING:")
        print(f"  Total exact matches found: {match_results['total_matches']:,}")
        print(f"  Samples involved in matches: {match_results['matched_samples']:,} ({match_results['matched_samples']/match_results['total_samples']*100:.1f}%)")
        print(f"  Unmatched samples: {match_results['unmatched_samples']:,} ({match_results['unmatched_samples']/match_results['total_samples']*100:.1f}%)")
        
        print(f"\nTEMPORAL SEQUENCES:")
        print(f"  Total sequences reconstructed: {sequence_stats['total_sequences']:,}")
        print(f"  Longest sequence length: {sequence_stats['longest_sequence']:,} time steps")
        print(f"  Average sequence length: {sequence_stats['average_sequence_length']:.1f} time steps")
        
        if sequence_stats['sequences']:
            print(f"\nTOP LONGEST SEQUENCES:")
            for i, seq in enumerate(sequence_stats['sequences'][:5]):
                print(f"    Sequence {i+1}: {len(seq)} time steps (samples: {seq[:3]}{'...' if len(seq) > 3 else ''})")
        
        print(f"\nPATTERN ANALYSIS:")
        print(f"  Unique fire patterns: {analysis['unique_patterns']:,}")
        print(f"  Patterns used once: {analysis['reuse_statistics']['patterns_used_once']:,}")
        print(f"  Patterns reused: {analysis['reuse_statistics']['patterns_used_multiple']:,}")
        print(f"  Maximum pattern reuse: {analysis['reuse_statistics']['max_pattern_reuse']:,} times")
        
        if analysis['most_common_patterns']:
            print(f"\nMOST REUSED PATTERNS:")
            for i, pattern in enumerate(analysis['most_common_patterns'][:3]):
                print(f"    Pattern {pattern['pattern_hash']}: {pattern['frequency']} uses, "
                      f"avg {pattern['avg_size']:.0f} pixels, "
                      f"size range {pattern['size_range'][0]}-{pattern['size_range'][1]}")
        
        print("="*70)
    
    def visualize_sequence_reconstruction(self, match_results: Dict, sequence_stats: Dict, 
                                        analysis: Dict, save_path: str = None):
        """Create visualizations of the sequence reconstruction results.
        
        Args:
            match_results: Pattern matching results.
            sequence_stats: Sequence statistics.
            analysis: Pattern analysis results.
            save_path: Optional path to save the visualization.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Match success rate
        ax1 = axes[0, 0]
        labels = ['Matched\nSamples', 'Unmatched\nSamples']
        sizes = [match_results['matched_samples'], match_results['unmatched_samples']]
        colors = ['lightblue', 'lightcoral']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Sample Matching Success Rate')
        
        # Sequence length distribution
        ax2 = axes[0, 1]
        if sequence_stats['sequence_lengths']:
            ax2.hist(sequence_stats['sequence_lengths'], bins=min(20, len(sequence_stats['sequence_lengths'])), 
                    alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('Sequence Length (time steps)')
            ax2.set_ylabel('Number of Sequences')
            ax2.set_title('Distribution of Sequence Lengths')
        else:
            ax2.text(0.5, 0.5, 'No sequences found', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Distribution of Sequence Lengths')
        
        # Pattern reuse frequency
        ax3 = axes[1, 0]
        reuse_stats = analysis['reuse_statistics']
        reuse_labels = ['Used Once', 'Reused Multiple Times']
        reuse_counts = [reuse_stats['patterns_used_once'], reuse_stats['patterns_used_multiple']]
        
        ax3.bar(reuse_labels, reuse_counts, color=['orange', 'purple'], alpha=0.7)
        ax3.set_ylabel('Number of Patterns')
        ax3.set_title('Pattern Reuse Statistics')
        
        # Top pattern usage
        ax4 = axes[1, 1]
        if analysis['most_common_patterns']:
            top_patterns = analysis['most_common_patterns'][:8]
            pattern_names = [f"Pattern {p['pattern_hash']}" for p in top_patterns]
            frequencies = [p['frequency'] for p in top_patterns]
            
            ax4.barh(pattern_names, frequencies, color='teal', alpha=0.7)
            ax4.set_xlabel('Usage Frequency')
            ax4.set_title('Most Reused Fire Patterns')
        else:
            ax4.text(0.5, 0.5, 'No common patterns found', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Most Reused Fire Patterns')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()

def main():
    """Main function to run the fire sequence reconstruction."""
    print("Fire Mask Temporal Sequence Reconstruction")
    print("="*60)
    
    # Initialize reconstructor
    reconstructor = FireSequenceReconstructor()
    
    # Load all fire data
    fire_data = reconstructor.load_all_fire_data(max_samples=1000)
    
    if not fire_data:
        print("ERROR: No fire data loaded!")
        return
    
    # Find exact pattern matches across all samples
    match_results = reconstructor.find_exact_pattern_matches(fire_data, min_fire_pixels=10)
    
    # Build temporal sequences from matches
    sequence_stats = reconstructor.build_fire_sequences(match_results['matches'], fire_data)
    
    # Analyze sequence patterns
    analysis = reconstructor.analyze_sequence_patterns(match_results['matches'], sequence_stats)
    
    # Print comprehensive summary
    reconstructor.print_reconstruction_summary(match_results, sequence_stats, analysis)
    
    # Create visualizations
    reconstructor.visualize_sequence_reconstruction(
        match_results, sequence_stats, analysis, 
        save_path="visualizations/fire_sequence_reconstruction.png"
    )
    
    print(f"\nSequence reconstruction completed!")
    print(f"Check 'visualizations/fire_sequence_reconstruction.png' for visualizations.")
    
    # Save sequence data for future use
    if sequence_stats['sequences']:
        sequence_file = Path("visualizations/fire_sequences.txt")
        with open(sequence_file, 'w') as f:
            f.write("Fire Sequence Reconstruction Results\n")
            f.write("="*50 + "\n\n")
            for i, seq in enumerate(sequence_stats['sequences'][:10]):
                f.write(f"Sequence {i+1} ({len(seq)} steps): {seq}\n")
        print(f"Sequence data saved to: {sequence_file}")

if __name__ == "__main__":
    main() 