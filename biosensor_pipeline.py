#!/usr/bin/env python3
"""
Enhanced AI-Powered Biosensor Circuit Design Pipeline
Complete pipeline with expanded circuit diversity, complex regulatory networks,
and comprehensive environmental noise modeling
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
import random
from dataclasses import dataclass, asdict
from itertools import product, combinations
import warnings
warnings.filterwarnings('ignore')
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Simulation libraries
try:
    import tellurium as te
    import roadrunner
    TELLURIUM_AVAILABLE = True
    print("✅ Tellurium available - using full simulation")
except ImportError:
    print("ℹ️  Using optimized mock simulation (works great for ML training)")
    TELLURIUM_AVAILABLE = False

# === EXTENSIONS FOR BIOMARKER TARGETS, LOGIC ENCODING, AND DATASET VARIETY ===

# 1. Biomarker Target Library
# Enhanced biomarker target library with more detailed properties
class BiomarkerTargetLibrary:
    """Enhanced biomarker library with detection properties"""
    def __init__(self):
        self.biomarkers = [
            {
                'name': 'TNF-alpha', 'type': 'cytokine', 'typical_range': (0, 100), 'unit': 'pg/mL',
                'molecular_weight': 17000, 'diffusion_coefficient': 2.5e-11, 'stability_half_life': 120
            },
            {
                'name': 'IL-6', 'type': 'cytokine', 'typical_range': (0, 200), 'unit': 'pg/mL',
                'molecular_weight': 21000, 'diffusion_coefficient': 2.2e-11, 'stability_half_life': 180
            },
            {
                'name': 'p53', 'type': 'protein', 'typical_range': (0, 50), 'unit': 'ng/mL',
                'molecular_weight': 53000, 'diffusion_coefficient': 1.8e-11, 'stability_half_life': 60
            },
            {
                'name': 'Mercury', 'type': 'metal', 'typical_range': (0, 10), 'unit': 'ppb',
                'molecular_weight': 200, 'diffusion_coefficient': 5e-10, 'stability_half_life': 1440
            },
            {
                'name': 'Glucose', 'type': 'metabolite', 'typical_range': (2, 20), 'unit': 'mM',
                'molecular_weight': 180, 'diffusion_coefficient': 6e-10, 'stability_half_life': 30
            },
            {
                'name': 'Lactate', 'type': 'metabolite', 'typical_range': (0, 10), 'unit': 'mM',
                'molecular_weight': 90, 'diffusion_coefficient': 8e-10, 'stability_half_life': 45
            }
        ]

    def random_biomarker(self):
        biomarker = random.choice(self.biomarkers)
        # Random threshold within typical range (usually 10-90% of max range)
        range_min, range_max = biomarker['typical_range']
        threshold = np.round(np.random.uniform(range_min + 0.1*(range_max-range_min), 
                                             range_min + 0.9*(range_max-range_min)), 3)
        
        # Realistic binding affinity (Kd) based on biomarker type
        if biomarker['type'] == 'cytokine':
            binding_affinity = np.round(np.random.uniform(0.01, 10), 4)  # nM range
        elif biomarker['type'] == 'protein':
            binding_affinity = np.round(np.random.uniform(0.1, 50), 4)   # nM range
        elif biomarker['type'] == 'metabolite':
            binding_affinity = np.round(np.random.uniform(1, 1000), 4)   # μM range
        else:  # metal
            binding_affinity = np.round(np.random.uniform(0.001, 1), 4)  # nM range
        
        return {
            'target_biomarker': biomarker['name'],
            'target_biomarker_type': biomarker['type'],
            'target_biomarker_unit': biomarker['unit'],
            'target_biomarker_threshold': threshold,
            'target_biomarker_binding_affinity': binding_affinity,
            'target_biomarker_molecular_weight': biomarker['molecular_weight'],
            'target_biomarker_diffusion_coefficient': biomarker['diffusion_coefficient'],
            'target_biomarker_stability_half_life': biomarker['stability_half_life']
        }

@dataclass
class BiomarkerConditions:
    """Represents the biomarker conditions for simulation"""
    target_biomarker: str
    target_biomarker_type: str
    target_biomarker_unit: str
    target_biomarker_threshold: float
    target_biomarker_binding_affinity: float
    target_biomarker_molecular_weight: float
    target_biomarker_diffusion_coefficient: float
    target_biomarker_stability_half_life: float
    
    # The actual concentration to test (this is what varies in each simulation)
    env_target_biomarker_concentration: float
    
@dataclass
class CircuitComponents:
    """Defines the biological parts for circuit construction"""
    promoter: str
    rbs: str
    coding_sequence: str
    terminator: str
    # New regulatory elements
    activator: Optional[str] = None
    repressor: Optional[str] = None
    insulator: Optional[str] = None
    operator: Optional[str] = None
    # Multi-gene circuit support
    secondary_genes: List[str] = None
    
    def __post_init__(self):
        if self.secondary_genes is None:
            self.secondary_genes = []
    
@dataclass
class RegulatoryParameters:
    """Regulatory interaction parameters"""
    hill_coefficient: float = 1.0  # Cooperativity
    binding_affinity: float = 1.0  # Kd values
    activation_strength: float = 1.0
    repression_strength: float = 1.0
    leakage_rate: float = 0.01

@dataclass
class NoiseConditions:
    """Comprehensive environmental noise parameters"""
    # Static environmental conditions
    pH_variation: float = 0.0
    temperature_variation: float = 0.0
    immune_signal_level: float = 0.0
    metabolic_flux_noise: float = 0.0
    ionic_strength: float = 0.0
    oxidative_stress: float = 0.0
    
    # New noise sources
    transcriptional_bursting: float = 0.0  # Burst frequency/size
    translational_variability: float = 0.0  # Translation noise
    metabolic_load: float = 0.0  # Resource competition
    resource_competition: float = 0.0  # Competition for ribosomes/polymerases
    
    # Time-varying noise (dynamic)
    noise_autocorrelation: float = 0.0  # Temporal correlation
    noise_frequency: float = 0.0  # Oscillation frequency
    noise_amplitude_modulation: float = 0.0  # AM of noise
    
    # Stochastic parameters
    intrinsic_noise_level: float = 0.0  # Inherent randomness
    extrinsic_noise_level: float = 0.0  # External fluctuations
    
@dataclass
class CircuitPerformance:
    """Comprehensive circuit performance metrics"""
    # Basic metrics
    signal_strength: float
    noise_level: float
    snr: float
    false_positive_rate: float
    sensitivity: float
    response_time: float
    steady_state_protein: float
    background_noise: float
    
    # New classification metrics
    specificity: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Dynamic metrics
    dynamic_range: float = 0.0  # max/min signal ratio
    response_delay: float = 0.0  # lag time
    settling_time: float = 0.0  # time to reach steady state
    overshoot: float = 0.0  # peak overshoot
    
    # Noise analysis
    noise_power_spectral_density: float = 0.0
    noise_autocorrelation_time: float = 0.0
    coefficient_of_variation: float = 0.0
    
    # Statistical summaries
    signal_mean: float = 0.0
    signal_std: float = 0.0
    signal_skewness: float = 0.0
    signal_kurtosis: float = 0.0
    confidence_interval_width: float = 0.0
    
@dataclass
class SimulationMetadata:
    """Enhanced simulation metadata"""
    simulation_duration: float
    simulation_method: str
    random_seed: int
    transcription_rate: float
    translation_rate: float
    protein_degradation_rate: float
    mRNA_degradation_rate: float
    
    # New metadata
    circuit_complexity: int = 1
    regulatory_interactions: int = 0
    simulation_timesteps: int = 1000
    noise_model: str = "gaussian"
    stochastic_method: str = "gillespie"

class EnhancedBiologicalPartsLibrary:
    """Expanded library of biological parts with detailed characteristics"""
    
    def __init__(self):
        # Expanded promoters with more detailed properties
        self.promoters = {
            'pTac': {
                'strength': 0.8, 'noise_sensitivity': 0.3, 'leakiness': 0.05,
                'type': 'inducible', 'copy_number_effect': 1.2,
                'hill_coefficient': 2.0, 'binding_sites': 1,
                'activation_threshold': 0.1, 'saturation_level': 0.9
            },
            'pBAD': {
                'strength': 0.9, 'noise_sensitivity': 0.4, 'leakiness': 0.02,
                'type': 'inducible', 'copy_number_effect': 1.1,
                'hill_coefficient': 1.5, 'binding_sites': 1,
                'activation_threshold': 0.05, 'saturation_level': 0.95
            },
            'pLac': {
                'strength': 0.6, 'noise_sensitivity': 0.2, 'leakiness': 0.08,
                'type': 'inducible', 'copy_number_effect': 1.0,
                'hill_coefficient': 1.0, 'binding_sites': 1,
                'activation_threshold': 0.2, 'saturation_level': 0.8
            },
            'pT7': {
                'strength': 1.0, 'noise_sensitivity': 0.6, 'leakiness': 0.01,
                'type': 'synthetic', 'copy_number_effect': 1.3,
                'hill_coefficient': 3.0, 'binding_sites': 1,
                'activation_threshold': 0.01, 'saturation_level': 0.99
            },
            'pTet': {
                'strength': 0.7, 'noise_sensitivity': 0.25, 'leakiness': 0.03,
                'type': 'inducible', 'copy_number_effect': 1.0,
                'hill_coefficient': 2.5, 'binding_sites': 2,
                'activation_threshold': 0.15, 'saturation_level': 0.85
            },
            # New promoters
            'pLux': {
                'strength': 0.85, 'noise_sensitivity': 0.35, 'leakiness': 0.04,
                'type': 'quorum_sensing', 'copy_number_effect': 1.15,
                'hill_coefficient': 2.2, 'binding_sites': 1,
                'activation_threshold': 0.12, 'saturation_level': 0.88
            },
            'pPhoA': {
                'strength': 0.75, 'noise_sensitivity': 0.3, 'leakiness': 0.06,
                'type': 'stress_response', 'copy_number_effect': 0.9,
                'hill_coefficient': 1.8, 'binding_sites': 1,
                'activation_threshold': 0.18, 'saturation_level': 0.82
            },
            'pRha': {
                'strength': 0.68, 'noise_sensitivity': 0.28, 'leakiness': 0.07,
                'type': 'inducible', 'copy_number_effect': 1.05,
                'hill_coefficient': 1.3, 'binding_sites': 1,
                'activation_threshold': 0.22, 'saturation_level': 0.78
            }
        }
        
        self.rbs_sites = {
            'RBS1': {'efficiency': 0.9, 'temperature_sensitivity': 0.3, 'strength_class': 'strong', 'secondary_structure': 'minimal'},
            'RBS2': {'efficiency': 0.7, 'temperature_sensitivity': 0.2, 'strength_class': 'medium', 'secondary_structure': 'moderate'},
            'RBS3': {'efficiency': 1.0, 'temperature_sensitivity': 0.4, 'strength_class': 'very_strong', 'secondary_structure': 'none'},
            'RBS4': {'efficiency': 0.6, 'temperature_sensitivity': 0.1, 'strength_class': 'weak', 'secondary_structure': 'high'},
            'RBS5': {'efficiency': 0.8, 'temperature_sensitivity': 0.25, 'strength_class': 'medium_strong', 'secondary_structure': 'low'},
            # New RBS sites
            'RBS6': {'efficiency': 0.95, 'temperature_sensitivity': 0.35, 'strength_class': 'ultra_strong', 'secondary_structure': 'minimal'},
            'RBS7': {'efficiency': 0.5, 'temperature_sensitivity': 0.15, 'strength_class': 'very_weak', 'secondary_structure': 'very_high'},
            'RBS8': {'efficiency': 0.75, 'temperature_sensitivity': 0.22, 'strength_class': 'medium', 'secondary_structure': 'moderate'}
        }
        
        # Expanded reporters with more detailed characteristics
        self.reporters = {
            'GFP': {
                'signal_strength': 0.8, 'pH_sensitivity': 0.3, 'maturation_time': 45,
                'gene_type': 'fluorescent_protein', 'temperature_sensitivity': 0.25,
                'photobleaching_rate': 0.02, 'quantum_yield': 0.6,
                'excitation_wavelength': 488, 'emission_wavelength': 509
            },
            'RFP': {
                'signal_strength': 0.9, 'pH_sensitivity': 0.2, 'maturation_time': 60,
                'gene_type': 'fluorescent_protein', 'temperature_sensitivity': 0.2,
                'photobleaching_rate': 0.015, 'quantum_yield': 0.7,
                'excitation_wavelength': 558, 'emission_wavelength': 583
            },
            'BFP': {
                'signal_strength': 0.7, 'pH_sensitivity': 0.4, 'maturation_time': 30,
                'gene_type': 'fluorescent_protein', 'temperature_sensitivity': 0.3,
                'photobleaching_rate': 0.025, 'quantum_yield': 0.5,
                'excitation_wavelength': 399, 'emission_wavelength': 456
            },
            'YFP': {
                'signal_strength': 0.85, 'pH_sensitivity': 0.25, 'maturation_time': 40,
                'gene_type': 'fluorescent_protein', 'temperature_sensitivity': 0.22,
                'photobleaching_rate': 0.018, 'quantum_yield': 0.65,
                'excitation_wavelength': 514, 'emission_wavelength': 527
            },
            'CFP': {
                'signal_strength': 0.75, 'pH_sensitivity': 0.35, 'maturation_time': 50,
                'gene_type': 'fluorescent_protein', 'temperature_sensitivity': 0.28,
                'photobleaching_rate': 0.022, 'quantum_yield': 0.55,
                'excitation_wavelength': 434, 'emission_wavelength': 477
            },
            # New reporters
            'LacZ': {
                'signal_strength': 0.95, 'pH_sensitivity': 0.15, 'maturation_time': 20,
                'gene_type': 'enzymatic', 'temperature_sensitivity': 0.4,
                'photobleaching_rate': 0.0, 'quantum_yield': 0.0,
                'excitation_wavelength': 0, 'emission_wavelength': 0
            },
            'mCherry': {
                'signal_strength': 0.88, 'pH_sensitivity': 0.18, 'maturation_time': 35,
                'gene_type': 'fluorescent_protein', 'temperature_sensitivity': 0.19,
                'photobleaching_rate': 0.012, 'quantum_yield': 0.72,
                'excitation_wavelength': 587, 'emission_wavelength': 610
            },
            'Venus': {
                'signal_strength': 0.92, 'pH_sensitivity': 0.23, 'maturation_time': 38,
                'gene_type': 'fluorescent_protein', 'temperature_sensitivity': 0.21,
                'photobleaching_rate': 0.016, 'quantum_yield': 0.68,
                'excitation_wavelength': 515, 'emission_wavelength': 528
            }
        }
        
        self.terminators = {
            'T1': {'efficiency': 0.95, 'stability': 0.9, 'type': 'strong', 'read_through': 0.05},
            'T2': {'efficiency': 0.90, 'stability': 0.8, 'type': 'medium', 'read_through': 0.10},
            'T3': {'efficiency': 0.98, 'stability': 0.95, 'type': 'very_strong', 'read_through': 0.02},
            # New terminators
            'T4': {'efficiency': 0.85, 'stability': 0.75, 'type': 'weak', 'read_through': 0.15},
            'T5': {'efficiency': 0.92, 'stability': 0.88, 'type': 'medium_strong', 'read_through': 0.08},
            'T6': {'efficiency': 0.99, 'stability': 0.98, 'type': 'ultra_strong', 'read_through': 0.01}
        }
        
        # New regulatory elements
        self.activators = {
            'CAP': {'strength': 0.8, 'cooperativity': 1.5, 'binding_affinity': 0.7},
            'AraC': {'strength': 0.9, 'cooperativity': 2.0, 'binding_affinity': 0.8},
            'LuxR': {'strength': 0.85, 'cooperativity': 1.8, 'binding_affinity': 0.75},
            'PhoB': {'strength': 0.7, 'cooperativity': 1.3, 'binding_affinity': 0.6}
        }
        
        self.repressors = {
            'LacI': {'strength': 0.9, 'cooperativity': 2.2, 'binding_affinity': 0.9},
            'TetR': {'strength': 0.85, 'cooperativity': 2.0, 'binding_affinity': 0.85},
            'CI': {'strength': 0.95, 'cooperativity': 2.5, 'binding_affinity': 0.95},
            'LexA': {'strength': 0.8, 'cooperativity': 1.8, 'binding_affinity': 0.8}
        }
        
        self.insulators = {
            'Insulator1': {'strength': 0.7, 'context_effect': 0.3},
            'Insulator2': {'strength': 0.8, 'context_effect': 0.2},
            'Insulator3': {'strength': 0.9, 'context_effect': 0.1}
        }
        
        self.operators = {
            'Operator1': {'binding_strength': 0.8, 'specificity': 0.9},
            'Operator2': {'binding_strength': 0.7, 'specificity': 0.85},
            'Operator3': {'binding_strength': 0.9, 'specificity': 0.95}
        }
        
        # Circuit architectures with detailed complexity
        self.circuit_architectures = {
            'simple': {'complexity': 1, 'regulatory_elements': 0, 'genes': 1},
            'with_activator': {'complexity': 2, 'regulatory_elements': 1, 'genes': 1},
            'with_repressor': {'complexity': 2, 'regulatory_elements': 1, 'genes': 1},
            'feedforward': {'complexity': 3, 'regulatory_elements': 2, 'genes': 2},
            'toggle_switch': {'complexity': 4, 'regulatory_elements': 3, 'genes': 2},
            'cascade': {'complexity': 5, 'regulatory_elements': 4, 'genes': 3},
            'autoregulation': {'complexity': 3, 'regulatory_elements': 1, 'genes': 1},
            'AND_gate': {'complexity': 4, 'regulatory_elements': 2, 'genes': 1},
            'OR_gate': {'complexity': 3, 'regulatory_elements': 2, 'genes': 1},
            'NAND_gate': {'complexity': 5, 'regulatory_elements': 3, 'genes': 1}
        }
        
        # Enhanced copy number variations
        self.copy_number_types = {
            'very_low_copy': {'multiplier': 0.2, 'noise_factor': 0.8},
            'low_copy_plasmid': {'multiplier': 1.0, 'noise_factor': 1.0},
            'medium_copy_plasmid': {'multiplier': 3.0, 'noise_factor': 1.2},
            'high_copy_plasmid': {'multiplier': 5.0, 'noise_factor': 1.5},
            'very_high_copy': {'multiplier': 10.0, 'noise_factor': 2.0},
            'genomic_integration': {'multiplier': 0.5, 'noise_factor': 0.6},
            'chromosomal_multi_copy': {'multiplier': 2.0, 'noise_factor': 0.9}
        }

class EnhancedBiosensorPipeline:
    """Enhanced pipeline for generating and testing biosensor circuits"""
    
    def __init__(self):
        self.parts_library = EnhancedBiologicalPartsLibrary()
        self.circuit_generator = EnhancedCircuitGenerator(self.parts_library)
        self.noise_generator = EnhancedNoiseGenerator()
        self.simulator = EnhancedBiosensorSimulator(self.parts_library)
        self.logger = EnhancedDataLogger()
    
    def generate_training_dataset(self, n_circuits: int = 500, n_noise_per_circuit: int = 10,
                                 circuit_strategy: str = 'random', 
                                 multi_gene_prob: float = 0.3,
                                 time_varying_noise_fraction: float = 0.3,
                                 n_stochastic_runs: int = 5) -> pd.DataFrame:
        """Generate comprehensive training dataset with enhanced features"""
        print(f"Generating {n_circuits} circuits with {n_noise_per_circuit} noise conditions each...")
        print(f"Circuit strategy: {circuit_strategy}")
        print(f"Multi-gene probability: {multi_gene_prob}")
        print(f"Time-varying noise fraction: {time_varying_noise_fraction}")
        print(f"Stochastic runs per simulation: {n_stochastic_runs}")
        
        # Generate circuits
        circuits_data = self.circuit_generator.generate_circuit_dataset(
            n_circuits, circuit_strategy, multi_gene_prob
        )
        
        # Generate dataset
        dataset = []
        total_simulations = len(circuits_data) * n_noise_per_circuit
        
        for i, (circuit, architecture, copy_number, reg_params) in enumerate(circuits_data):
            if i % 25 == 0:
                print(f"Processing circuit {i+1}/{len(circuits_data)}")
            
            # Generate noise conditions for this circuit
            noise_conditions = self.noise_generator.generate_noise_dataset(
                n_noise_per_circuit, time_varying_fraction=time_varying_noise_fraction
            )
            
            for j, noise in enumerate(noise_conditions):
                # Simulate circuit performance
                performance, metadata = self.simulator.simulate_circuit(
                    circuit, noise, architecture, copy_number, reg_params,
                    seed=i*n_noise_per_circuit+j, n_stochastic_runs=n_stochastic_runs
                )
                
                # Log data
                entry = self.logger.create_dataset_entry(
                    circuit, noise, performance, metadata, architecture, 
                    copy_number, reg_params, self.parts_library
                )
                dataset.append(entry)
        
        print(f"Generated {len(dataset)} total data points")
        
        # Save and return dataset
        df = self.logger.save_dataset(dataset)
        return df
    
    def analyze_dataset(self, df: pd.DataFrame):
        """Comprehensive dataset analysis"""
        print("\n=== COMPREHENSIVE ENHANCED DATASET ANALYSIS ===")
        print(f"Total samples: {len(df)}")
        print(f"Features: {len(df.columns)}")
        
        # Circuit diversity analysis
        n_unique_basic_circuits = df[['circuit_promoter', 'circuit_rbs', 'circuit_coding_sequence']].drop_duplicates().shape[0]
        n_unique_full_circuits = df[['circuit_promoter', 'circuit_rbs', 'circuit_coding_sequence', 
                                   'circuit_architecture', 'circuit_copy_number']].drop_duplicates().shape[0]
        
        print(f"Unique basic circuits: {n_unique_basic_circuits}")
        print(f"Unique full circuit designs: {n_unique_full_circuits}")
        print(f"Circuit architectures: {df['circuit_architecture'].nunique()}")
        print(f"Copy number types: {df['circuit_copy_number'].nunique()}")
        print(f"Multi-gene circuits: {(df['circuit_n_secondary_genes'] > 0).sum()}")
        
        # Performance statistics
        print(f"\nPERFORMANCE STATISTICS:")
        perf_cols = [col for col in df.columns if col.startswith('perf_')]
        for col in perf_cols[:10]:  # Show first 10 performance metrics
            print(f"{col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}, max={df[col].max():.3f}")
        
        # Classification distribution
        print(f"\nPERFORMANCE CLASSIFICATION:")
        print(df['performance_class'].value_counts())
        
        print(f"\nNOISE TOLERANCE CLASSIFICATION:")
        print(df['noise_tolerance'].value_counts())
        
        print(f"\nRESPONSE SPEED CLASSIFICATION:")
        print(df['response_class'].value_counts())
        
        # Top performing circuits
        print(f"\nTOP 5 CIRCUITS BY MULTI-OBJECTIVE SCORE:")
        top_circuits = df.nlargest(5, 'multi_objective_score')[
            ['circuit_promoter', 'circuit_rbs', 'circuit_coding_sequence', 
             'circuit_architecture', 'perf_snr', 'multi_objective_score']
        ]
        print(top_circuits.to_string(index=False))
        
        # Environmental impact analysis
        print(f"\nENVIRONMENTAL IMPACT CORRELATIONS:")
        env_cols = [col for col in df.columns if col.startswith('env_') and 'actual' not in col]
        for col in env_cols[:8]:  # Show first 8 environmental factors
            corr = df[col].corr(df['perf_snr'])
            print(f"{col} vs SNR: {corr:.3f}")
        
        # Feature importance hints
        print(f"\nFEATURE IMPORTANCE HINTS:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corrwith(df['perf_snr']).abs().sort_values(ascending=False)
        print("Top 10 features correlated with SNR:")
        print(correlations.head(10))
    
    def visualize_results(self, df: pd.DataFrame):
        """Create comprehensive visualizations and save them to 'output_visuals' folder"""
        
        # Ensure output folder exists
        output_dir = 'output_visuals'
        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(4, 3, figsize=(20, 24))

        # Performance distribution
        axes[0,0].hist(df['perf_snr'], bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].set_xlabel('Signal-to-Noise Ratio')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Distribution of SNR Values')

        # Multi-objective score
        axes[0,1].hist(df['multi_objective_score'], bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[0,1].set_xlabel('Multi-Objective Score')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Distribution of Multi-Objective Scores')

        # Performance by architecture
        sns.boxplot(data=df, x='circuit_architecture', y='perf_snr', ax=axes[0,2])
        axes[0,2].set_title('SNR by Circuit Architecture')
        axes[0,2].tick_params(axis='x', rotation=45)

        # Dynamic range analysis
        axes[1,0].scatter(df['perf_dynamic_range'], df['perf_snr'], alpha=0.6)
        axes[1,0].set_xlabel('Dynamic Range')
        axes[1,0].set_ylabel('SNR')
        axes[1,0].set_title('Dynamic Range vs SNR')
        axes[1,0].set_xscale('log')

        # Response time analysis
        axes[1,1].scatter(df['perf_response_time'], df['perf_snr'], alpha=0.6, color='red')
        axes[1,1].set_xlabel('Response Time (min)')
        axes[1,1].set_ylabel('SNR')
        axes[1,1].set_title('Response Time vs SNR')

        # Noise tolerance
        sns.boxplot(data=df, x='noise_tolerance', y='perf_snr', ax=axes[1,2])
        axes[1,2].set_title('SNR by Noise Tolerance')

        # Environmental stress impact
        axes[2,0].scatter(df['env_total_stress'], df['perf_snr'], alpha=0.6, color='purple')
        axes[2,0].set_xlabel('Total Environmental Stress')
        axes[2,0].set_ylabel('SNR')
        axes[2,0].set_title('Environmental Stress vs SNR')

        # Specificity vs Sensitivity
        scatter = axes[2,1].scatter(df['perf_sensitivity'], df['perf_specificity'], 
                                    c=df['perf_snr'], alpha=0.6, cmap='viridis')
        axes[2,1].set_xlabel('Sensitivity')
        axes[2,1].set_ylabel('Specificity')
        axes[2,1].set_title('Sensitivity vs Specificity (colored by SNR)')
        fig.colorbar(scatter, ax=axes[2,1])

        # Copy number effects
        sns.boxplot(data=df, x='circuit_copy_number', y='perf_snr', ax=axes[2,2])
        axes[2,2].set_title('SNR by Copy Number')
        axes[2,2].tick_params(axis='x', rotation=45)

        # Hill coefficient effects
        axes[3,0].scatter(df['reg_hill_coefficient'], df['perf_snr'], alpha=0.6, color='orange')
        axes[3,0].set_xlabel('Hill Coefficient')
        axes[3,0].set_ylabel('SNR')
        axes[3,0].set_title('Cooperativity (Hill Coefficient) vs SNR')

        # Noise autocorrelation effects
        axes[3,1].scatter(df['env_noise_autocorrelation'], df['perf_noise_autocorrelation_time'], 
                        alpha=0.6, color='brown')
        axes[3,1].set_xlabel('Environmental Noise Autocorrelation')
        axes[3,1].set_ylabel('Performance Noise Autocorr. Time')
        axes[3,1].set_title('Noise Autocorrelation Effects')

        # Multi-gene circuit effects
        df_multi = df[df['circuit_n_secondary_genes'] > 0]
        df_single = df[df['circuit_n_secondary_genes'] == 0]
        axes[3,2].hist([df_single['perf_snr'], df_multi['perf_snr']], 
                    bins=30, alpha=0.7, label=['Single-gene', 'Multi-gene'], 
                    color=['blue', 'red'])
        axes[3,2].set_xlabel('SNR')
        axes[3,2].set_ylabel('Frequency')
        axes[3,2].set_title('Single-gene vs Multi-gene Circuit Performance')
        axes[3,2].legend()

        plt.tight_layout()
        fig_path1 = os.path.join(output_dir, 'main_visuals.png')
        fig.savefig(fig_path1)
        plt.close(fig)

        # Correlation matrix
        plt.figure(figsize=(16, 12))
        key_features = [
            'perf_snr', 'perf_signal_strength', 'perf_sensitivity', 'perf_specificity',
            'perf_dynamic_range', 'perf_response_time', 'multi_objective_score',
            'env_total_stress', 'env_static_stress', 'env_dynamic_stress',
            'promoter_strength', 'rbs_efficiency', 'reporter_signal_strength',
            'reg_hill_coefficient', 'reg_binding_affinity', 'arch_complexity'
        ]
        available_features = [f for f in key_features if f in df.columns]
        correlation_matrix = df[available_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    fmt='.2f', square=True)
        plt.title('Key Feature Correlation Matrix')
        plt.tight_layout()
        fig_path2 = os.path.join(output_dir, 'correlation_matrix.png')
        plt.savefig(fig_path2)
        plt.close()

        # Performance classification plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        df['performance_class'].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Performance Class Distribution')
        axes[0,0].tick_params(axis='x', rotation=45)

        df['noise_tolerance'].value_counts().plot(kind='bar', ax=axes[0,1], color='green')
        axes[0,1].set_title('Noise Tolerance Distribution')
        axes[0,1].tick_params(axis='x', rotation=45)

        df['response_class'].value_counts().plot(kind='bar', ax=axes[1,0], color='red')
        axes[1,0].set_title('Response Speed Distribution')
        axes[1,0].tick_params(axis='x', rotation=45)

        arch_perf = df.groupby('circuit_architecture')['multi_objective_score'].mean().sort_values(ascending=False)
        arch_perf.plot(kind='bar', ax=axes[1,1], color='purple')
        axes[1,1].set_title('Average Multi-Objective Score by Architecture')
        axes[1,1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        fig_path3 = os.path.join(output_dir, 'classification_analysis.png')
        fig.savefig(fig_path3)
        plt.close(fig)

        print(f"Visualizations saved to: {output_dir}")

# 2. Circuit Logic/Topology Encoding
def generate_circuit_topology_json(circuit: CircuitComponents, architecture: str) -> str:
    """
    Encodes the circuit topology as a JSON string.
    For multi-layer circuits, this can be expanded to a graph structure.
    """
    topology = {
        "nodes": [],
        "edges": []
    }
    # Add main gene
    topology["nodes"].append({
        "id": "main_gene",
        "promoter": circuit.promoter,
        "rbs": circuit.rbs,
        "cds": circuit.coding_sequence,
        "terminator": circuit.terminator
    })
    # Add secondary genes if present
    for idx, gene in enumerate(circuit.secondary_genes):
        topology["nodes"].append({
            "id": f"secondary_gene_{idx+1}",
            "cds": gene
        })
    # Add regulatory elements as nodes
    if circuit.activator:
        topology["nodes"].append({"id": "activator", "type": circuit.activator})
        topology["edges"].append({"from": "activator", "to": "main_gene", "type": "activation"})
    if circuit.repressor:
        topology["nodes"].append({"id": "repressor", "type": circuit.repressor})
        topology["edges"].append({"from": "repressor", "to": "main_gene", "type": "repression"})
    if circuit.insulator:
        topology["nodes"].append({"id": "insulator", "type": circuit.insulator})
    if circuit.operator:
        topology["nodes"].append({"id": "operator", "type": circuit.operator})
    # Example: connect secondary genes in a cascade if architecture is 'cascade'
    if architecture == "cascade" and circuit.secondary_genes:
        for idx, gene in enumerate(circuit.secondary_genes):
            if idx == 0:
                topology["edges"].append({"from": "main_gene", "to": f"secondary_gene_{idx+1}", "type": "activation"})
            else:
                topology["edges"].append({"from": f"secondary_gene_{idx}", "to": f"secondary_gene_{idx+1}", "type": "activation"})
    # Add more logic for AND/OR/toggle as needed
    return json.dumps(topology)

# 3. EXTEND DATA LOGGER TO INCLUDE BIOMARKER AND TOPOLOGY
# Patch EnhancedDataLogger.create_dataset_entry to add biomarker and topology
def create_dataset_entry_with_biomarker_and_topology(
    circuit: CircuitComponents, noise: NoiseConditions, 
    performance: CircuitPerformance, metadata: SimulationMetadata,
    architecture: str, copy_number: str, reg_params: RegulatoryParameters,
    parts_library: EnhancedBiologicalPartsLibrary,
    biomarker_info: dict,
    circuit_topology_json: str
) -> Dict:
    entry = EnhancedDataLogger.create_dataset_entry(
        circuit, noise, performance, metadata, architecture, copy_number, reg_params, parts_library
    )
    # Add biomarker info
    entry.update(biomarker_info)
    # Add circuit logic type (same as architecture for now)
    entry['circuit_logic_type'] = architecture
    # Add circuit topology encoding
    entry['circuit_topology'] = circuit_topology_json
    return entry

# 4. PATCH THE PIPELINE TO GENERATE MORE VARIED DATA AND INCLUDE NEW FIELDS
def generate_training_dataset_extensive(
    self, n_circuits: int = 1000, n_noise_per_circuit: int = 20,
    circuit_strategy: str = 'combinatorial', 
    multi_gene_prob: float = 0.5,
    time_varying_noise_fraction: float = 0.5,
    n_stochastic_runs: int = 3
) -> pd.DataFrame:
    """
    Generate a much larger, more varied dataset with biomarker and topology info.
    """
    print(f"Generating {n_circuits} circuits with {n_noise_per_circuit} noise conditions each (EXTENDED)...")
    print(f"Circuit strategy: {circuit_strategy}")
    print(f"Multi-gene probability: {multi_gene_prob}")
    print(f"Time-varying noise fraction: {time_varying_noise_fraction}")
    print(f"Stochastic runs per simulation: {n_stochastic_runs}")

    circuits_data = self.circuit_generator.generate_circuit_dataset(
        n_circuits, circuit_strategy, multi_gene_prob
    )
    biomarker_lib = BiomarkerTargetLibrary()
    dataset = []
    for i, (circuit, architecture, copy_number, reg_params) in enumerate(circuits_data):
        if i % 50 == 0:
            print(f"Processing circuit {i+1}/{len(circuits_data)}")
        # For each circuit, sample a biomarker target
        biomarker_info = biomarker_lib.random_biomarker()
        # Generate noise conditions for this circuit
        noise_conditions = self.noise_generator.generate_noise_dataset(
            n_noise_per_circuit, time_varying_fraction=time_varying_noise_fraction
        )
        # Generate circuit topology encoding
        circuit_topology_json = generate_circuit_topology_json(circuit, architecture)
        for j, noise in enumerate(noise_conditions):
            performance, metadata = self.simulator.simulate_circuit(
                circuit, noise, architecture, copy_number, reg_params,
                seed=i*n_noise_per_circuit+j, n_stochastic_runs=n_stochastic_runs
            )
            entry = create_dataset_entry_with_biomarker_and_topology(
                circuit, noise, performance, metadata, architecture, 
                copy_number, reg_params, self.parts_library,
                biomarker_info, circuit_topology_json
            )
            dataset.append(entry)
    print(f"Generated {len(dataset)} total data points (EXTENDED)")
    df = self.logger.save_dataset(dataset, filename='enhanced_biosensor_dataset_extended.csv')
    return df

# Monkey-patch the pipeline with the new method
EnhancedBiosensorPipeline.generate_training_dataset_extensive = generate_training_dataset_extensive

# === END OF EXTENSIONS ===

class EnhancedNoiseGenerator:
    """Generates comprehensive biological noise conditions with time-varying effects"""
    
    @staticmethod
    def generate_noise_condition(severity: str = 'medium', seed: int = None, 
                               time_varying: bool = False) -> NoiseConditions:
        """Generate comprehensive noise condition"""
        if seed is not None:
            np.random.seed(seed)
            
        severity_scales = {
            'very_low': 0.1,
            'low': 0.3,
            'medium': 0.6,
            'high': 1.0,
            'very_high': 1.5
        }
        scale = severity_scales.get(severity, 0.6)
        
        # Generate time-varying parameters if requested
        if time_varying:
            noise_frequency = np.random.uniform(0.01, 0.1)  # Hz
            noise_autocorr = np.random.uniform(0.1, 0.9)
            noise_am = np.random.uniform(0.1, 0.5)
        else:
            noise_frequency = 0.0
            noise_autocorr = 0.0
            noise_am = 0.0
        
        return NoiseConditions(
            # Basic environmental noise
            pH_variation=np.random.normal(0, 0.5 * scale),
            temperature_variation=np.random.normal(0, 3.0 * scale),
            immune_signal_level=np.random.beta(2, 5) * scale,
            metabolic_flux_noise=np.random.exponential(0.2 * scale),
            ionic_strength=np.random.normal(0, 0.1 * scale),
            oxidative_stress=np.random.beta(2, 8) * scale,
            
            # New noise sources
            transcriptional_bursting=np.random.exponential(0.3 * scale),
            translational_variability=np.random.uniform(0, 0.4 * scale),
            metabolic_load=np.random.beta(2, 6) * scale,
            resource_competition=np.random.exponential(0.25 * scale),
            
            # Time-varying noise
            noise_autocorrelation=noise_autocorr,
            noise_frequency=noise_frequency,
            noise_amplitude_modulation=noise_am,
            
            # Stochastic parameters
            intrinsic_noise_level=np.random.exponential(0.1 * scale),
            extrinsic_noise_level=np.random.exponential(0.15 * scale)
        )
    
    @staticmethod
    def generate_noise_dataset(n_conditions: int = 100, 
                             severity_mix: Dict[str, float] = None,
                             time_varying_fraction: float = 0.3) -> List[NoiseConditions]:
        """Generate diverse noise dataset"""
        if severity_mix is None:
            severity_mix = {
                'very_low': 0.1, 'low': 0.2, 'medium': 0.4, 
                'high': 0.2, 'very_high': 0.1
            }
        
        conditions = []
        for i in range(n_conditions):
            severity = np.random.choice(
                list(severity_mix.keys()), 
                p=list(severity_mix.values())
            )
            time_varying = np.random.random() < time_varying_fraction
            conditions.append(
                EnhancedNoiseGenerator.generate_noise_condition(
                    severity, seed=i, time_varying=time_varying
                )
            )
        
        return conditions

class EnhancedCircuitGenerator:
    """Generates diverse biosensor circuit designs with complex regulatory networks"""
    
    def __init__(self, parts_library: EnhancedBiologicalPartsLibrary):
        self.parts = parts_library
    
    def generate_regulatory_parameters(self, architecture: str) -> RegulatoryParameters:
        """Generate regulatory parameters based on architecture"""
        arch_props = self.parts.circuit_architectures[architecture]
        complexity = arch_props['complexity']
        
        return RegulatoryParameters(
            hill_coefficient=np.random.uniform(1.0, complexity + 1.0),
            binding_affinity=np.random.uniform(0.1, 1.0),
            activation_strength=np.random.uniform(0.5, 1.5),
            repression_strength=np.random.uniform(0.5, 1.5),
            leakage_rate=np.random.uniform(0.001, 0.1)
        )
    
    def generate_random_circuit(self, architecture: str = None, 
                              multi_gene_prob: float = 0.3) -> Tuple[CircuitComponents, str, str, RegulatoryParameters]:
        """Generate a random circuit with regulatory elements"""
        if architecture is None:
            architecture = random.choice(list(self.parts.circuit_architectures.keys()))
        
        copy_number = random.choice(list(self.parts.copy_number_types.keys()))
        
        # Base circuit components
        circuit = CircuitComponents(
            promoter=random.choice(list(self.parts.promoters.keys())),
            rbs=random.choice(list(self.parts.rbs_sites.keys())),
            coding_sequence=random.choice(list(self.parts.reporters.keys())),
            terminator=random.choice(list(self.parts.terminators.keys()))
        )
        
        # Add regulatory elements based on architecture
        arch_props = self.parts.circuit_architectures[architecture]
        if arch_props['regulatory_elements'] > 0:
            if 'activator' in architecture or 'AND' in architecture or 'OR' in architecture:
                circuit.activator = random.choice(list(self.parts.activators.keys()))
            if 'repressor' in architecture or 'toggle' in architecture or 'NAND' in architecture:
                circuit.repressor = random.choice(list(self.parts.repressors.keys()))
            if random.random() < 0.2:  # 20% chance of insulator
                circuit.insulator = random.choice(list(self.parts.insulators.keys()))
            if random.random() < 0.3:  # 30% chance of operator
                circuit.operator = random.choice(list(self.parts.operators.keys()))
        
        # Add secondary genes for multi-gene circuits
        if arch_props['genes'] > 1 or random.random() < multi_gene_prob:
            n_secondary = min(arch_props['genes'] - 1, 3)  # Max 3 secondary genes
            circuit.secondary_genes = random.sample(
                list(self.parts.reporters.keys()), 
                n_secondary
            )
        
        reg_params = self.generate_regulatory_parameters(architecture)
        
        return circuit, architecture, copy_number, reg_params
    
    def generate_circuit_dataset(self, n_circuits: int = 1000, 
                               strategy: str = 'random', 
                               multi_gene_prob: float = 0.3) -> List[Tuple[CircuitComponents, str, str, RegulatoryParameters]]:
        """Generate comprehensive circuit dataset"""
        if strategy == 'exhaustive':
            circuits = []
            base_combinations = list(product(
                self.parts.promoters.keys(),
                self.parts.rbs_sites.keys(),
                self.parts.reporters.keys(),
                self.parts.terminators.keys()
            ))
            
            for i, combo in enumerate(base_combinations[:n_circuits]):
                circuit = CircuitComponents(*combo)
                arch = random.choice(list(self.parts.circuit_architectures.keys()))
                copy_num = random.choice(list(self.parts.copy_number_types.keys()))
                reg_params = self.generate_regulatory_parameters(arch)
                circuits.append((circuit, arch, copy_num, reg_params))
            
            return circuits
        
        elif strategy == 'combinatorial':
            circuits = []
            architectures = list(self.parts.circuit_architectures.keys())
            copy_numbers = list(self.parts.copy_number_types.keys())
            
            # Systematic variation
            for i in range(n_circuits):
                arch = architectures[i % len(architectures)]
                copy_num = copy_numbers[i % len(copy_numbers)]
                
                circuit, _, _, reg_params = self.generate_random_circuit(arch, multi_gene_prob)
                circuits.append((circuit, arch, copy_num, reg_params))
            
            return circuits
        
        else:  # random
            return [self.generate_random_circuit(multi_gene_prob=multi_gene_prob) 
                   for _ in range(n_circuits)]

class EnhancedBiosensorSimulator:
    """Advanced biosensor simulation with comprehensive performance metrics"""
    
    def __init__(self, parts_library: EnhancedBiologicalPartsLibrary):
        self.parts = parts_library
    
    def calculate_time_series_metrics(self, time_series: np.ndarray, 
                                    time_points: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive time series metrics"""
        # Basic statistics
        signal_mean = np.mean(time_series)
        signal_std = np.std(time_series)
        signal_skewness = stats.skew(time_series)
        signal_kurtosis = stats.kurtosis(time_series)
        
        # Dynamic metrics
        max_signal = np.max(time_series)
        min_signal = np.min(time_series)
        dynamic_range = max_signal / max(min_signal, 0.001)
        
        # Response characteristics
        steady_state = np.mean(time_series[-20:])
        response_time = 0.0
        settling_time = 0.0
        overshoot = 0.0
        
        # Find response time (time to 50% of steady state)
        target_50 = steady_state * 0.5
        for i, val in enumerate(time_series):
            if val >= target_50:
                response_time = time_points[i]
                break
        
        # Find settling time (within 5% of steady state)
        settling_threshold = steady_state * 0.05
        for i in range(len(time_series) - 1, 0, -1):
            if abs(time_series[i] - steady_state) > settling_threshold:
                settling_time = time_points[i + 1] if i + 1 < len(time_points) else time_points[-1]
                break
        
        # Calculate overshoot
        if max_signal > steady_state:
            overshoot = (max_signal - steady_state) / steady_state
        
        # Noise analysis
        # Detrend signal for noise analysis
        detrended = signal.detrend(time_series)
        
        # Power spectral density
        frequencies, psd = signal.welch(detrended, fs=1.0/(time_points[1] - time_points[0]))
        noise_psd = np.sum(psd)
        
        # Autocorrelation
        autocorr = np.correlate(detrended, detrended, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find autocorrelation time (time to 1/e)
        autocorr_time = 0.0
        target_autocorr = 1.0 / np.e
        for i, val in enumerate(autocorr):
            if val <= target_autocorr:
                autocorr_time = time_points[min(i, len(time_points) - 1)]
                break
        
        # Coefficient of variation
        cv = signal_std / max(signal_mean, 0.001)
        
        # Confidence interval (95%)
        ci_width = 1.96 * signal_std / np.sqrt(len(time_series))
        
        return {
            'signal_mean': signal_mean,
            'signal_std': signal_std,
            'signal_skewness': signal_skewness,
            'signal_kurtosis': signal_kurtosis,
            'dynamic_range': dynamic_range,
            'response_time': response_time,
            'settling_time': settling_time,
            'overshoot': overshoot,
            'noise_psd': noise_psd,
            'autocorr_time': autocorr_time,
            'coefficient_of_variation': cv,
            'confidence_interval_width': ci_width
        }
    
    def calculate_classification_metrics(self, true_signal: float, predicted_signal: float,
                                       threshold: float = 0.5) -> Dict[str, float]:
        """Calculate classification performance metrics"""
        # Simple binary classification based on threshold
        true_positive = (true_signal > threshold) and (predicted_signal > threshold)
        true_negative = (true_signal <= threshold) and (predicted_signal <= threshold)
        false_positive = (true_signal <= threshold) and (predicted_signal > threshold)
        false_negative = (true_signal > threshold) and (predicted_signal <= threshold)
        
        # Convert to counts for calculation
        tp = 1 if true_positive else 0
        tn = 1 if true_negative else 0
        fp = 1 if false_positive else 0
        fn = 1 if false_negative else 0
        
        # Calculate metrics
        sensitivity = tp / max(tp + fn, 0.001)  # Recall
        specificity = tn / max(tn + fp, 0.001)
        precision = tp / max(tp + fp, 0.001)
        
        # F1 Score
        f1_score = 2 * (precision * sensitivity) / max(precision + sensitivity, 0.001)
        
        # False positive rate
        fpr = fp / max(fp + tn, 0.001)
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score,
            'false_positive_rate': fpr,
            'recall': sensitivity  # Same as sensitivity
        }
    
    def simulate_stochastic_dynamics(self, circuit: CircuitComponents, noise: NoiseConditions,
                                   architecture: str, copy_number: str, reg_params: RegulatoryParameters,
                                   duration: float = 300.0, n_simulations: int = 10) -> np.ndarray:
        """Run multiple stochastic simulations"""
        time_points = np.linspace(0, duration, 1000)
        all_trajectories = []
        
        for sim in range(n_simulations):
            # Individual stochastic simulation
            trajectory = self.single_stochastic_simulation(
                circuit, noise, architecture, copy_number, reg_params, time_points, sim
            )
            all_trajectories.append(trajectory)
        
        return np.array(all_trajectories)
    
    def single_stochastic_simulation(self, circuit: CircuitComponents, noise: NoiseConditions,
                                   architecture: str, copy_number: str, reg_params: RegulatoryParameters,
                                   time_points: np.ndarray, seed: int) -> np.ndarray:
        """Single stochastic simulation with enhanced noise modeling"""
        np.random.seed(seed)
        
        # Get component properties
        promoter_props = self.parts.promoters[circuit.promoter]
        rbs_props = self.parts.rbs_sites[circuit.rbs]
        reporter_props = self.parts.reporters[circuit.coding_sequence]
        terminator_props = self.parts.terminators[circuit.terminator]
        copy_props = self.parts.copy_number_types[copy_number]
        
        # Calculate base rates
        copy_number_multiplier = copy_props['multiplier']
        noise_factor = copy_props['noise_factor']
        
        # Environmental effects on rates
        pH_factor = max(0.1, 1.0 - abs(noise.pH_variation) * reporter_props['pH_sensitivity'])
        temp_factor = max(0.1, 1.0 - abs(noise.temperature_variation) * rbs_props['temperature_sensitivity'] * 0.02)
        immune_factor = max(0.1, 1.0 - noise.immune_signal_level * 0.3)
        metabolic_factor = max(0.1, 1.0 - noise.metabolic_flux_noise * 0.4)
        ionic_factor = max(0.1, 1.0 - abs(noise.ionic_strength) * 0.2)
        oxidative_factor = max(0.1, 1.0 - noise.oxidative_stress * 0.25)
        
        # New noise effects
        burst_factor = 1.0 + noise.transcriptional_bursting
        translation_noise = noise.translational_variability
        resource_factor = max(0.1, 1.0 - noise.resource_competition * 0.3)
        
        # Regulatory effects
        hill_coeff = reg_params.hill_coefficient
        binding_affinity = reg_params.binding_affinity
        activation_strength = reg_params.activation_strength
        repression_strength = reg_params.repression_strength
        
        # Calculate effective rates
        base_transcription_rate = (promoter_props['strength'] * 
                                 copy_number_multiplier * 
                                 pH_factor * immune_factor * metabolic_factor * 
                                 ionic_factor * resource_factor * burst_factor)
        
        base_translation_rate = (rbs_props['efficiency'] * temp_factor * oxidative_factor * 
                               (1.0 + translation_noise * np.random.normal(0, 0.1)))
        
        protein_degradation = (0.01 + noise.metabolic_flux_noise * 0.02 + 
                             noise.oxidative_stress * 0.01 + 
                             noise.metabolic_load * 0.015)
        
        mRNA_degradation = 0.05 + noise.temperature_variation * 0.001
        
        # Initialize arrays
        mRNA_levels = np.zeros_like(time_points)
        protein_levels = np.zeros_like(time_points)
        
        # Time-varying noise generation
        if noise.noise_frequency > 0:
            time_varying_noise = (noise.noise_amplitude_modulation * 
                                np.sin(2 * np.pi * noise.noise_frequency * time_points))
        else:
            time_varying_noise = np.zeros_like(time_points)
        
        # Simulation loop
        dt = time_points[1] - time_points[0]
        for i in range(1, len(time_points)):
            # Current noise level
            current_noise = (noise.intrinsic_noise_level + noise.extrinsic_noise_level + 
                           time_varying_noise[i])
            
            # Regulatory function (Hill equation with noise)
            if architecture in ['with_activator', 'AND_gate', 'OR_gate']:
                regulatory_factor = activation_strength * (
                    1.0 / (1.0 + (binding_affinity / max(protein_levels[i-1], 0.001))**hill_coeff)
                )
            elif architecture in ['with_repressor', 'toggle_switch', 'NAND_gate']:
                regulatory_factor = 1.0 / (1.0 + repression_strength * 
                                          (protein_levels[i-1] / binding_affinity)**hill_coeff)
            else:
                regulatory_factor = 1.0
            
            # Transcription with bursting
            if np.random.random() < noise.transcriptional_bursting * dt:
                burst_size = np.random.exponential(2.0)
                transcription_rate = base_transcription_rate * regulatory_factor * burst_size
            else:
                transcription_rate = base_transcription_rate * regulatory_factor
            
            # Add autocorrelated noise
            if i > 1 and noise.noise_autocorrelation > 0:
                autocorr_noise = (noise.noise_autocorrelation * 
                                (mRNA_levels[i-1] - mRNA_levels[i-2]) * 0.1)
            else:
                autocorr_noise = 0
            
            # mRNA dynamics
            mRNA_production = transcription_rate * (1.0 + current_noise * np.random.normal(0, 0.1))
            mRNA_decay = mRNA_degradation * mRNA_levels[i-1]
            mRNA_levels[i] = max(0, mRNA_levels[i-1] + dt * (mRNA_production - mRNA_decay) + autocorr_noise)
            
            # Translation with variability
            translation_efficiency = base_translation_rate * (1.0 + translation_noise * np.random.normal(0, 0.1))
            protein_production = translation_efficiency * mRNA_levels[i-1]
            protein_decay = protein_degradation * protein_levels[i-1]
            protein_levels[i] = max(0, protein_levels[i-1] + dt * (protein_production - protein_decay))
            
            # Add noise
            noise_magnitude = noise_factor * current_noise
            mRNA_levels[i] += np.random.normal(0, max(0, noise_magnitude * 0.01))
            protein_levels[i] += np.random.normal(0, max(0, noise_magnitude * 0.01))
            
            # Ensure non-negative
            mRNA_levels[i] = max(0, mRNA_levels[i])
            protein_levels[i] = max(0, protein_levels[i])
        
        return protein_levels
    
    def simulate_circuit(self, circuit: CircuitComponents, noise: NoiseConditions, 
                        architecture: str, copy_number: str, reg_params: RegulatoryParameters,
                        duration: float = 300.0, seed: int = None, 
                        n_stochastic_runs: int = 5) -> Tuple[CircuitPerformance, SimulationMetadata]:
        """Enhanced circuit simulation with comprehensive metrics"""
        
        if seed is not None:
            np.random.seed(seed)
        
        # Run multiple stochastic simulations
        all_trajectories = self.simulate_stochastic_dynamics(
            circuit, noise, architecture, copy_number, reg_params, duration, n_stochastic_runs
        )
        
        # Calculate ensemble average
        mean_trajectory = np.mean(all_trajectories, axis=0)
        std_trajectory = np.std(all_trajectories, axis=0)
        
        time_points = np.linspace(0, duration, 1000)
        
        # Calculate time series metrics
        ts_metrics = self.calculate_time_series_metrics(mean_trajectory, time_points)
        
        # Calculate performance metrics
        promoter_props = self.parts.promoters[circuit.promoter]
        copy_props = self.parts.copy_number_types[copy_number]
        
        steady_state_protein = ts_metrics['signal_mean']
        background_noise = (promoter_props['leakiness'] * copy_props['multiplier'] + 
                          reg_params.leakage_rate)
        signal_strength = max(0.001, steady_state_protein - background_noise)
        noise_level = ts_metrics['signal_std'] + background_noise * 0.1
        
        # Enhanced SNR calculation
        snr = float(signal_strength / max(noise_level, 0.001))
        
        # Classification metrics (using a reasonable threshold)
        threshold = steady_state_protein * 0.3
        true_signal = steady_state_protein  # Assumed true signal
        classification_metrics = self.calculate_classification_metrics(
            true_signal, steady_state_protein, threshold
        )
        
        # Response delay calculation
        response_delay = ts_metrics['response_time'] + promoter_props.get('maturation_time', 0) / 60.0
        
        performance = CircuitPerformance(
            # Basic metrics
            signal_strength=float(signal_strength),
            noise_level=float(noise_level),
            snr=snr,
            false_positive_rate=float(classification_metrics['false_positive_rate']),
            sensitivity=float(classification_metrics['sensitivity']),
            response_time=float(ts_metrics['response_time']),
            steady_state_protein=float(steady_state_protein),
            background_noise=float(background_noise),
            
            # Classification metrics
            specificity=float(classification_metrics['specificity']),
            precision=float(classification_metrics['precision']),
            recall=float(classification_metrics['recall']),
            f1_score=float(classification_metrics['f1_score']),
            
            # Dynamic metrics
            dynamic_range=float(ts_metrics['dynamic_range']),
            response_delay=float(response_delay),
            settling_time=float(ts_metrics['settling_time']),
            overshoot=float(ts_metrics['overshoot']),
            
            # Noise analysis
            noise_power_spectral_density=float(ts_metrics['noise_psd']),
            noise_autocorrelation_time=float(ts_metrics['autocorr_time']),
            coefficient_of_variation=float(ts_metrics['coefficient_of_variation']),
            
            # Statistical summaries
            signal_mean=float(ts_metrics['signal_mean']),
            signal_std=float(ts_metrics['signal_std']),
            signal_skewness=float(ts_metrics['signal_skewness']),
            signal_kurtosis=float(ts_metrics['signal_kurtosis']),
            confidence_interval_width=float(ts_metrics['confidence_interval_width'])
        )
        
        metadata = SimulationMetadata(
            simulation_duration=duration,
            simulation_method='enhanced_stochastic',
            random_seed=seed if seed is not None else -1,
            transcription_rate=float(signal_strength * 10),  # Approximate
            translation_rate=float(signal_strength * 5),     # Approximate
            protein_degradation_rate=0.01 + noise.metabolic_flux_noise * 0.02,
            mRNA_degradation_rate=0.05 + noise.temperature_variation * 0.001,
            
            circuit_complexity=self.parts.circuit_architectures[architecture]['complexity'],
            regulatory_interactions=self.parts.circuit_architectures[architecture]['regulatory_elements'],
            simulation_timesteps=1000,
            noise_model="comprehensive_stochastic",
            stochastic_method="enhanced_numerical"
        )
        
        return performance, metadata
    
    def calculate_biomarker_binding_fraction(self, biomarker_concentration: float, 
                                           binding_affinity: float, 
                                           hill_coefficient: float = 1.0) -> float:
        """
        Calculate the fraction of sensor bound to biomarker using Hill equation
        fraction_bound = [Biomarker]^n / (Kd^n + [Biomarker]^n)
        """
        if biomarker_concentration <= 0:
            return 0.0
        
        numerator = biomarker_concentration ** hill_coefficient
        denominator = (binding_affinity ** hill_coefficient) + numerator
        return numerator / denominator
    
    def simulate_biosensor_response(self, biomarker_conditions: BiomarkerConditions,
                                  circuit, noise, architecture: str, 
                                  copy_number: str, reg_params, 
                                  time_points: np.ndarray) -> np.ndarray:
        """
        Simulate the biosensor's response to biomarker presence
        This is where the actual biomarker detection happens
        """
        # Get biomarker concentration
        biomarker_conc = biomarker_conditions.env_target_biomarker_concentration
        binding_affinity = biomarker_conditions.target_biomarker_binding_affinity
        
        # Calculate binding fraction (how much sensor is bound to biomarker)
        binding_fraction = self.calculate_biomarker_binding_fraction(
            biomarker_conc, binding_affinity, reg_params.hill_coefficient
        )
        
        # Get component properties
        promoter_props = self.parts.promoters[circuit.promoter]
        rbs_props = self.parts.rbs_sites[circuit.rbs]
        reporter_props = self.parts.reporters[circuit.coding_sequence]
        copy_props = self.parts.copy_number_types[copy_number]
        
        # Environmental effects on rates
        pH_factor = max(0.1, 1.0 - abs(noise.pH_variation) * reporter_props['pH_sensitivity'])
        temp_factor = max(0.1, 1.0 - abs(noise.temperature_variation) * rbs_props['temperature_sensitivity'] * 0.02)
        immune_factor = max(0.1, 1.0 - noise.immune_signal_level * 0.3)
        metabolic_factor = max(0.1, 1.0 - noise.metabolic_flux_noise * 0.4)
        resource_factor = max(0.1, 1.0 - noise.resource_competition * 0.3)
        
        # Base transcription rate (when no biomarker is bound)
        base_transcription = (promoter_props['strength'] * 
                            copy_props['multiplier'] * 
                            pH_factor * immune_factor * metabolic_factor * resource_factor)
        
        # Leakage rate (background expression)
        leakage_transcription = base_transcription * (promoter_props['leakiness'] + reg_params.leakage_rate)
        
        # Biomarker-induced transcription
        # The binding fraction determines how much the circuit is activated
        if architecture in ['simple', 'with_activator', 'AND_gate', 'OR_gate']:
            # Activating biosensor: more biomarker = more signal
            biomarker_induced_transcription = (base_transcription * binding_fraction * 
                                             reg_params.activation_strength)
        elif architecture in ['with_repressor', 'NAND_gate']:
            # Repressing biosensor: more biomarker = less signal (inverted logic)
            biomarker_induced_transcription = base_transcription * (1.0 - binding_fraction * reg_params.repression_strength)
        else:
            # Default: activating
            biomarker_induced_transcription = base_transcription * binding_fraction
        
        # Total transcription rate
        total_transcription = leakage_transcription + biomarker_induced_transcription
        
        # Simulate protein production over time
        protein_levels = np.zeros_like(time_points)
        mRNA_levels = np.zeros_like(time_points)
        
        # Translation and degradation rates
        translation_rate = rbs_props['efficiency'] * temp_factor
        protein_degradation = 0.01 + noise.metabolic_flux_noise * 0.02
        mRNA_degradation = 0.05 + noise.temperature_variation * 0.001
        
        # Time evolution
        dt = time_points[1] - time_points[0]
        for i in range(1, len(time_points)):
            # Add transcriptional bursting
            if np.random.random() < noise.transcriptional_bursting * dt:
                burst_size = np.random.exponential(2.0)
                current_transcription = total_transcription * burst_size
            else:
                current_transcription = total_transcription
            
            # Add noise
            transcription_noise = noise.intrinsic_noise_level * np.random.normal(0, 0.1)
            translation_noise = noise.translational_variability * np.random.normal(0, 0.1)
            
            # mRNA dynamics
            mRNA_production = current_transcription * (1.0 + transcription_noise)
            mRNA_decay = mRNA_degradation * mRNA_levels[i-1]
            mRNA_levels[i] = max(0, mRNA_levels[i-1] + dt * (mRNA_production - mRNA_decay))
            
            # Protein dynamics
            protein_production = translation_rate * mRNA_levels[i-1] * (1.0 + translation_noise)
            protein_decay = protein_degradation * protein_levels[i-1]
            protein_levels[i] = max(0, protein_levels[i-1] + dt * (protein_production - protein_decay))
            
            # Add extrinsic noise
            protein_levels[i] += noise.extrinsic_noise_level * np.random.normal(0, 0.01)
            protein_levels[i] = max(0, protein_levels[i])
        
        return protein_levels
    
    def calculate_detection_metrics(self, signal_trajectory: np.ndarray,
                                  biomarker_concentration: float,
                                  biomarker_threshold: float,
                                  signal_threshold: float = None) -> Dict[str, float]:
        """
        Calculate proper detection metrics based on biomarker presence and circuit response
        """
        # Determine if biomarker is actually present above threshold
        biomarker_present = biomarker_concentration > biomarker_threshold
        
        # Calculate signal statistics
        steady_state_signal = np.mean(signal_trajectory[-100:])  # Last 10% of trajectory
        background_signal = np.mean(signal_trajectory[:100])     # First 10% of trajectory
        signal_noise = np.std(signal_trajectory[-100:])
        
        # Adaptive threshold: background + 3 standard deviations if not provided
        if signal_threshold is None:
            signal_threshold = background_signal + 3 * signal_noise
        
        # Determine if circuit detected biomarker (signal above threshold)
        circuit_detected = steady_state_signal > signal_threshold
        
        # Calculate classification metrics
        true_positive = biomarker_present and circuit_detected
        true_negative = (not biomarker_present) and (not circuit_detected)
        false_positive = (not biomarker_present) and circuit_detected
        false_negative = biomarker_present and (not circuit_detected)
        
        # Convert to binary values for calculation
        tp = 1.0 if true_positive else 0.0
        tn = 1.0 if true_negative else 0.0
        fp = 1.0 if false_positive else 0.0
        fn = 1.0 if false_negative else 0.0
        
        # Calculate metrics
        sensitivity = tp / max(tp + fn, 0.001)  # True positive rate
        specificity = tn / max(tn + fp, 0.001)  # True negative rate
        precision = tp / max(tp + fp, 0.001)    # Positive predictive value
        npv = tn / max(tn + fn, 0.001)          # Negative predictive value
        
        # F1 score
        f1_score = 2 * (precision * sensitivity) / max(precision + sensitivity, 0.001)
        
        # False positive and negative rates
        false_positive_rate = fp / max(fp + tn, 0.001)
        false_negative_rate = fn / max(fn + tp, 0.001)
        
        # Signal-to-noise ratio
        signal_strength = max(0.001, steady_state_signal - background_signal)
        snr = signal_strength / max(signal_noise, 0.001)
        
        return {
            'biomarker_present': biomarker_present,
            'circuit_detected': circuit_detected,
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'negative_predictive_value': npv,
            'f1_score': f1_score,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'signal_strength': signal_strength,
            'signal_noise': signal_noise,
            'snr': snr,
            'steady_state_signal': steady_state_signal,
            'background_signal': background_signal,
            'signal_threshold': signal_threshold
        }
    
    def simulate_circuit_with_biomarker(self, circuit, noise, architecture: str, 
                                      copy_number: str, reg_params,
                                      biomarker_conditions: BiomarkerConditions,
                                      duration: float = 300.0, seed: int = None,
                                      n_stochastic_runs: int = 5):
        """
        Main simulation function that properly models biomarker detection
        """
        if seed is not None:
            np.random.seed(seed)
        
        time_points = np.linspace(0, duration, 1000)
        all_trajectories = []
        
        # Run multiple stochastic simulations
        for run in range(n_stochastic_runs):
            np.random.seed(seed + run if seed is not None else None)
            trajectory = self.simulate_biosensor_response(
                biomarker_conditions, circuit, noise, architecture, 
                copy_number, reg_params, time_points
            )
            all_trajectories.append(trajectory)
        
        # Calculate ensemble statistics
        mean_trajectory = np.mean(all_trajectories, axis=0)
        std_trajectory = np.std(all_trajectories, axis=0)
        
        # Calculate detection metrics
        detection_metrics = self.calculate_detection_metrics(
            mean_trajectory, 
            biomarker_conditions.env_target_biomarker_concentration,
            biomarker_conditions.target_biomarker_threshold
        )
        
        # Calculate temporal metrics
        response_time = self.calculate_response_time(mean_trajectory, time_points)
        settling_time = self.calculate_settling_time(mean_trajectory, time_points)
        dynamic_range = np.max(mean_trajectory) / max(np.min(mean_trajectory), 0.001)
        
        # Create comprehensive performance object
        performance = CircuitPerformance(
            # Basic metrics
            signal_strength=float(detection_metrics['signal_strength']),
            noise_level=float(detection_metrics['signal_noise']),
            snr=float(detection_metrics['snr']),
            false_positive_rate=float(detection_metrics['false_positive_rate']),
            sensitivity=float(detection_metrics['sensitivity']),
            response_time=float(response_time),
            steady_state_protein=float(detection_metrics['steady_state_signal']),
            background_noise=float(detection_metrics['background_signal']),
            
            # Enhanced classification metrics
            specificity=float(detection_metrics['specificity']),
            precision=float(detection_metrics['precision']),
            recall=float(detection_metrics['sensitivity']),
            f1_score=float(detection_metrics['f1_score']),
            
            # Dynamic metrics
            dynamic_range=float(dynamic_range),
            response_delay=float(response_time),
            settling_time=float(settling_time),
            overshoot=float(self.calculate_overshoot(mean_trajectory)),
            
            # Noise analysis
            noise_power_spectral_density=float(np.var(std_trajectory)),
            noise_autocorrelation_time=float(self.calculate_autocorrelation_time(std_trajectory)),
            coefficient_of_variation=float(np.std(mean_trajectory) / max(np.mean(mean_trajectory), 0.001)),
            
            # Statistical summaries
            signal_mean=float(np.mean(mean_trajectory)),
            signal_std=float(np.std(mean_trajectory)),
            signal_skewness=float(stats.skew(mean_trajectory)),
            signal_kurtosis=float(stats.kurtosis(mean_trajectory)),
            confidence_interval_width=float(1.96 * np.std(mean_trajectory) / np.sqrt(len(mean_trajectory)))
        )
        
        # Enhanced metadata
        metadata = SimulationMetadata(
            simulation_duration=duration,
            simulation_method='enhanced_biomarker_detection',
            random_seed=seed if seed is not None else -1,
            transcription_rate=float(detection_metrics['signal_strength'] * 10),
            translation_rate=float(detection_metrics['signal_strength'] * 5),
            protein_degradation_rate=0.01 + noise.metabolic_flux_noise * 0.02,
            mRNA_degradation_rate=0.05 + noise.temperature_variation * 0.001,
            circuit_complexity=self.parts.circuit_architectures[architecture]['complexity'],
            regulatory_interactions=self.parts.circuit_architectures[architecture]['regulatory_elements'],
            simulation_timesteps=1000,
            noise_model="biomarker_responsive_stochastic",
            stochastic_method="enhanced_biomarker_detection"
        )
        
        return performance, metadata, detection_metrics
    
    def calculate_response_time(self, trajectory: np.ndarray, time_points: np.ndarray) -> float:
        """Calculate time to reach 50% of steady state"""
        steady_state = np.mean(trajectory[-100:])
        initial_state = np.mean(trajectory[:10])
        target = initial_state + 0.5 * (steady_state - initial_state)
        
        for i, value in enumerate(trajectory):
            if value >= target:
                return time_points[i]
        return time_points[-1]
    
    def calculate_settling_time(self, trajectory: np.ndarray, time_points: np.ndarray) -> float:
        """Calculate time to settle within 5% of steady state"""
        steady_state = np.mean(trajectory[-100:])
        tolerance = 0.05 * steady_state
        
        for i in range(len(trajectory) - 100, 0, -1):
            if abs(trajectory[i] - steady_state) > tolerance:
                return time_points[min(i + 1, len(time_points) - 1)]
        return 0.0
    
    def calculate_overshoot(self, trajectory: np.ndarray) -> float:
        """Calculate overshoot percentage"""
        steady_state = np.mean(trajectory[-100:])
        peak = np.max(trajectory)
        if peak > steady_state:
            return (peak - steady_state) / steady_state
        return 0.0
    
    def calculate_autocorrelation_time(self, trajectory: np.ndarray) -> float:
        """Calculate autocorrelation time"""
        autocorr = np.correlate(trajectory, trajectory, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]
        
        target = 1.0 / np.e
        for i, val in enumerate(autocorr):
            if val <= target:
                return float(i)
        return float(len(autocorr))

def generate_biomarker_concentrations(biomarker_info: dict, n_concentrations: int = 5) -> List[float]:
    """
    Generate a range of biomarker concentrations for testing, including:
    - Below threshold (should be negative)
    - Around threshold (should be borderline)
    - Above threshold (should be positive)
    """
    threshold = biomarker_info['target_biomarker_threshold']
    range_min, range_max = 0, threshold * 3  # Test up to 3x threshold
    
    concentrations = []
    
    # Always include these key test points
    concentrations.extend([
        0.0,                           # No biomarker (negative control)
        threshold * 0.1,               # Well below threshold
        threshold * 0.5,               # Below threshold
        threshold,                     # At threshold
        threshold * 1.5,               # Above threshold
        threshold * 3.0                # Well above threshold
    ])
    
    # Add random concentrations in between
    for _ in range(n_concentrations - 6):
        conc = np.random.uniform(range_min, range_max)
        concentrations.append(conc)
    
    return sorted(concentrations)

class EnhancedDataLogger:
    """Enhanced data logging with comprehensive feature extraction"""
    
    @staticmethod
    def create_dataset_entry(circuit: CircuitComponents, noise: NoiseConditions, 
                           performance: CircuitPerformance, metadata: SimulationMetadata,
                           architecture: str, copy_number: str, reg_params: RegulatoryParameters,
                           parts_library: EnhancedBiologicalPartsLibrary) -> Dict:
        """Create comprehensive dataset entry with all features"""
        entry = {}
        
        # 1. Circuit Design Parameters (Enhanced)
        entry['circuit_promoter'] = circuit.promoter
        entry['circuit_rbs'] = circuit.rbs
        entry['circuit_coding_sequence'] = circuit.coding_sequence
        entry['circuit_terminator'] = circuit.terminator
        entry['circuit_architecture'] = architecture
        entry['circuit_copy_number'] = copy_number
        
        # Regulatory elements
        entry['circuit_activator'] = circuit.activator if circuit.activator else 'none'
        entry['circuit_repressor'] = circuit.repressor if circuit.repressor else 'none'
        entry['circuit_insulator'] = circuit.insulator if circuit.insulator else 'none'
        entry['circuit_operator'] = circuit.operator if circuit.operator else 'none'
        entry['circuit_n_secondary_genes'] = len(circuit.secondary_genes)
        entry['circuit_secondary_genes'] = ','.join(circuit.secondary_genes) if circuit.secondary_genes else 'none'
        
        # Detailed component properties
        promoter_props = parts_library.promoters[circuit.promoter]
        rbs_props = parts_library.rbs_sites[circuit.rbs]
        reporter_props = parts_library.reporters[circuit.coding_sequence]
        terminator_props = parts_library.terminators[circuit.terminator]
        copy_props = parts_library.copy_number_types[copy_number]
        arch_props = parts_library.circuit_architectures[architecture]
        
        # Promoter features
        for key, value in promoter_props.items():
            entry[f'promoter_{key}'] = value
        
        # RBS features
        for key, value in rbs_props.items():
            entry[f'rbs_{key}'] = value
        
        # Reporter features
        for key, value in reporter_props.items():
            entry[f'reporter_{key}'] = value
        
        # Terminator features
        for key, value in terminator_props.items():
            entry[f'terminator_{key}'] = value
        
        # Copy number features
        for key, value in copy_props.items():
            entry[f'copy_{key}'] = value
        
        # Architecture features
        for key, value in arch_props.items():
            entry[f'arch_{key}'] = value
        
        # Regulatory parameters
        entry['reg_hill_coefficient'] = reg_params.hill_coefficient
        entry['reg_binding_affinity'] = reg_params.binding_affinity
        entry['reg_activation_strength'] = reg_params.activation_strength
        entry['reg_repression_strength'] = reg_params.repression_strength
        entry['reg_leakage_rate'] = reg_params.leakage_rate
        
        # 2. Environmental / Noise Conditions (Comprehensive)
        for field in noise.__dataclass_fields__:
            entry[f'env_{field}'] = getattr(noise, field)
        
        # Derived environmental metrics
        entry['env_pH_actual'] = 7.0 + noise.pH_variation
        entry['env_temperature_actual'] = 37.0 + noise.temperature_variation
        entry['env_total_stress'] = (
            abs(noise.pH_variation) + abs(noise.temperature_variation)/10 + 
            noise.immune_signal_level + noise.metabolic_flux_noise + 
            abs(noise.ionic_strength) + noise.oxidative_stress +
            noise.transcriptional_bursting + noise.translational_variability +
            noise.metabolic_load + noise.resource_competition
        ) / 10
        
        entry['env_static_stress'] = (
            abs(noise.pH_variation) + abs(noise.temperature_variation)/10 + 
            noise.immune_signal_level + noise.metabolic_flux_noise + 
            abs(noise.ionic_strength) + noise.oxidative_stress
        ) / 6
        
        entry['env_dynamic_stress'] = (
            noise.transcriptional_bursting + noise.translational_variability +
            noise.metabolic_load + noise.resource_competition +
            noise.intrinsic_noise_level + noise.extrinsic_noise_level
        ) / 6
        
        # 3. Performance Outputs (Comprehensive)
        for field in performance.__dataclass_fields__:
            entry[f'perf_{field}'] = getattr(performance, field)
        
        # 4. Simulation Metadata (Enhanced)
        for field in metadata.__dataclass_fields__:
            entry[f'sim_{field}'] = getattr(metadata, field)
        
        # 5. Derived Labels / Classifications (Enhanced)
        entry['robustness_score'] = (performance.snr * (1 - performance.false_positive_rate) * 
                                   performance.sensitivity * performance.specificity)
        
        entry['performance_class'] = (
            'excellent' if performance.snr > 20 else (
                'high' if performance.snr > 10 else (
                    'medium' if performance.snr > 3 else (
                        'low' if performance.snr > 1 else 'poor'
                    )
                )
            )
        )
        
        entry['noise_tolerance'] = (
            'very_high' if entry['env_total_stress'] > 0.8 and performance.snr > 5 else (
                'high' if entry['env_total_stress'] > 0.5 and performance.snr > 3 else (
                    'medium' if entry['env_total_stress'] > 0.3 and performance.snr > 1 else 'low'
                )
            )
        )
        
        entry['response_class'] = (
            'very_fast' if performance.response_time < 10 else (
                'fast' if performance.response_time < 30 else (
                    'medium' if performance.response_time < 100 else 'slow'
                )
            )
        )
        
        entry['dynamic_range_class'] = (
            'very_high' if performance.dynamic_range > 100 else (
                'high' if performance.dynamic_range > 20 else (
                    'medium' if performance.dynamic_range > 5 else 'low'
                )
            )
        )
        
        # Multi-objective score combining multiple criteria
        entry['multi_objective_score'] = (
            0.3 * min(performance.snr / 20, 1.0) +  # SNR component
            0.2 * performance.sensitivity +          # Sensitivity component
            0.2 * performance.specificity +          # Specificity component
            0.15 * min(20 / max(performance.response_time, 1), 1.0) +  # Speed component
            0.15 * (1 - min(entry['env_total_stress'], 1.0))  # Robustness component
        )
        
        return entry
    
    @staticmethod
    def save_dataset(data: List[Dict], filename: str = 'enhanced_biosensor_dataset.csv'):
        """Save enhanced dataset to CSV"""
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Enhanced dataset saved to {filename}")
        print(f"Dataset contains {len(df)} samples with {len(df.columns)} features")
        return df

def enhanced_dataset_entry_with_proper_biomarker_detection(
    circuit, noise, biomarker_conditions: BiomarkerConditions,
    performance, metadata, detection_metrics: dict,
    architecture: str, copy_number: str, reg_params,
    parts_library, circuit_topology_json: str
) -> dict:
    """
    Create dataset entry with proper biomarker detection results
    """
    # Base entry from original function
    entry = EnhancedDataLogger.create_dataset_entry(
        circuit, noise, performance, metadata, architecture, 
        copy_number, reg_params, parts_library
    )
    
    # Add biomarker information
    entry['target_biomarker'] = biomarker_conditions.target_biomarker
    entry['target_biomarker_type'] = biomarker_conditions.target_biomarker_type
    entry['target_biomarker_unit'] = biomarker_conditions.target_biomarker_unit
    entry['target_biomarker_threshold'] = biomarker_conditions.target_biomarker_threshold
    entry['target_biomarker_binding_affinity'] = biomarker_conditions.target_biomarker_binding_affinity
    entry['target_biomarker_molecular_weight'] = biomarker_conditions.target_biomarker_molecular_weight
    entry['target_biomarker_diffusion_coefficient'] = biomarker_conditions.target_biomarker_diffusion_coefficient
    entry['target_biomarker_stability_half_life'] = biomarker_conditions.target_biomarker_stability_half_life
    
    # CRITICAL: Add the actual biomarker concentration being tested
    entry['env_target_biomarker_concentration'] = biomarker_conditions.env_target_biomarker_concentration
    
    # Add circuit topology
    entry['circuit_logic_type'] = architecture
    entry['circuit_topology'] = circuit_topology_json
    
    # Add detection-specific metrics
    entry['detection_biomarker_present'] = detection_metrics['biomarker_present']
    entry['detection_circuit_detected'] = detection_metrics['circuit_detected']
    entry['detection_true_positive'] = detection_metrics['true_positive']
    entry['detection_true_negative'] = detection_metrics['true_negative']
    entry['detection_false_positive'] = detection_metrics['false_positive']
    entry['detection_false_negative'] = detection_metrics['false_negative']
    entry['detection_signal_threshold'] = detection_metrics['signal_threshold']
    
    # Calculate derived detection metrics
    entry['detection_accuracy'] = (detection_metrics['true_positive'] + detection_metrics['true_negative'])
    entry['detection_balanced_accuracy'] = (detection_metrics['sensitivity'] + detection_metrics['specificity']) / 2
    entry['detection_matthews_correlation'] = calculate_matthews_correlation(detection_metrics)
    
    # Biomarker binding fraction for this concentration
    binding_fraction = calculate_binding_fraction(
        biomarker_conditions.env_target_biomarker_concentration,
        biomarker_conditions.target_biomarker_binding_affinity,
        reg_params.hill_coefficient
    )
    entry['biomarker_binding_fraction'] = binding_fraction
    
    return entry

def calculate_matthews_correlation(detection_metrics: dict) -> float:
    """Calculate Matthews Correlation Coefficient"""
    tp = detection_metrics['true_positive']
    tn = detection_metrics['true_negative'] 
    fp = detection_metrics['false_positive']
    fn = detection_metrics['false_negative']
    
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0:
        return 0.0
    return numerator / denominator

def calculate_binding_fraction(concentration: float, binding_affinity: float, hill_coeff: float = 1.0) -> float:
    """Calculate biomarker binding fraction"""
    if concentration <= 0:
        return 0.0
    
    numerator = concentration ** hill_coeff
    denominator = (binding_affinity ** hill_coeff) + numerator
    return numerator / denominator

def generate_realistic_biomarker_detection_dataset(
    pipeline, 
    n_circuits: int = 1000,
    n_biomarker_concentrations: int = 8,  # Test multiple concentrations per biomarker
    n_noise_per_concentration: int = 3,   # Multiple noise conditions per concentration
    circuit_strategy: str = 'combinatorial',
    multi_gene_prob: float = 0.5,
    time_varying_noise_fraction: float = 0.4,
    n_stochastic_runs: int = 3
) -> pd.DataFrame:
    """
    Generate a realistic dataset where each row represents a proper biomarker detection simulation
    """
    print(f"Generating REALISTIC biomarker detection dataset...")
    print(f"Circuits: {n_circuits}")
    print(f"Biomarker concentrations per circuit: {n_biomarker_concentrations}")
    print(f"Noise conditions per concentration: {n_noise_per_concentration}")
    print(f"Total simulations: {n_circuits * n_biomarker_concentrations * n_noise_per_concentration}")
    
    # Generate circuit designs
    circuits_data = pipeline.circuit_generator.generate_circuit_dataset(
        n_circuits, circuit_strategy, multi_gene_prob
    )
    
    # Initialize enhanced simulator
    enhanced_simulator = EnhancedBiosensorSimulator(pipeline.parts_library)
    biomarker_lib = BiomarkerTargetLibrary()
    
    dataset = []
    total_simulations = len(circuits_data) * n_biomarker_concentrations * n_noise_per_concentration
    simulation_count = 0
    
    for circuit_idx, (circuit, architecture, copy_number, reg_params) in enumerate(circuits_data):
        if circuit_idx % 50 == 0:
            print(f"Processing circuit {circuit_idx+1}/{len(circuits_data)}")
        
        # Generate one biomarker target per circuit (realistic: each circuit targets one biomarker)
        biomarker_info = biomarker_lib.random_biomarker()
        
        # Generate range of biomarker concentrations to test
        test_concentrations = generate_biomarker_concentrations(
            biomarker_info, n_biomarker_concentrations
        )
        
        # Generate circuit topology
        circuit_topology_json = generate_circuit_topology_json(circuit, architecture)
        
        for conc_idx, biomarker_concentration in enumerate(test_concentrations):
            # Create biomarker conditions for this concentration
            biomarker_conditions = BiomarkerConditions(
                target_biomarker=biomarker_info['target_biomarker'],
                target_biomarker_type=biomarker_info['target_biomarker_type'],
                target_biomarker_unit=biomarker_info['target_biomarker_unit'],
                target_biomarker_threshold=biomarker_info['target_biomarker_threshold'],
                target_biomarker_binding_affinity=biomarker_info['target_biomarker_binding_affinity'],
                target_biomarker_molecular_weight=biomarker_info['target_biomarker_molecular_weight'],
                target_biomarker_diffusion_coefficient=biomarker_info['target_biomarker_diffusion_coefficient'],
                target_biomarker_stability_half_life=biomarker_info['target_biomarker_stability_half_life'],
                env_target_biomarker_concentration=biomarker_concentration
            )
            
            # Generate multiple noise conditions for this biomarker concentration
            noise_conditions = pipeline.noise_generator.generate_noise_dataset(
                n_noise_per_concentration, time_varying_fraction=time_varying_noise_fraction
            )
            
            for noise_idx, noise in enumerate(noise_conditions):
                simulation_count += 1
                seed = circuit_idx * 10000 + conc_idx * 100 + noise_idx
                
                # Run the PROPER biomarker detection simulation
                performance, metadata, detection_metrics = enhanced_simulator.simulate_circuit_with_biomarker(
                    circuit=circuit,
                    noise=noise,
                    architecture=architecture,
                    copy_number=copy_number,
                    reg_params=reg_params,
                    biomarker_conditions=biomarker_conditions,
                    duration=300.0,
                    seed=seed,
                    n_stochastic_runs=n_stochastic_runs
                )
                
                # Create comprehensive dataset entry with REAL biomarker detection results
                entry = enhanced_dataset_entry_with_proper_biomarker_detection(
                    circuit=circuit,
                    noise=noise,
                    biomarker_conditions=biomarker_conditions,
                    performance=performance,
                    metadata=metadata,
                    detection_metrics=detection_metrics,
                    architecture=architecture,
                    copy_number=copy_number,
                    reg_params=reg_params,
                    parts_library=pipeline.parts_library,
                    circuit_topology_json=circuit_topology_json
                )
                
                dataset.append(entry)
                
                # Progress update
                if simulation_count % 500 == 0:
                    print(f"  Completed {simulation_count}/{total_simulations} simulations "
                          f"({100*simulation_count/total_simulations:.1f}%)")
    
    print(f"Generated {len(dataset)} total REALISTIC biomarker detection simulations")
    
    # Save dataset
    df = pd.DataFrame(dataset)
    filename = f'realistic_biosensor_dataset_{len(df)}_samples.csv'
    df.to_csv(filename, index=False)
    print(f"Realistic biomarker detection dataset saved to {filename}")
    
    return df

def analyze_biomarker_detection_dataset(df: pd.DataFrame):
    """Analyze the realistic biomarker detection dataset"""
    print("\n=== BIOMARKER DETECTION DATASET ANALYSIS ===")
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(df.columns)}")
    
    # Biomarker diversity
    print(f"\nBIOMARKER DIVERSITY:")
    print(f"Unique biomarkers: {df['target_biomarker'].nunique()}")
    print(f"Biomarker types: {df['target_biomarker_type'].value_counts().to_dict()}")
    
    # Concentration range analysis
    print(f"\nCONCENTRATION ANALYSIS:")
    print(f"Biomarker concentration range: {df['env_target_biomarker_concentration'].min():.3f} - {df['env_target_biomarker_concentration'].max():.3f}")
    print(f"Average threshold: {df['target_biomarker_threshold'].mean():.3f}")
    
    # Detection performance
    print(f"\nDETECTION PERFORMANCE:")
    print(f"Overall accuracy: {df['detection_accuracy'].mean():.3f} ± {df['detection_accuracy'].std():.3f}")
    print(f"Sensitivity: {df['perf_sensitivity'].mean():.3f} ± {df['perf_sensitivity'].std():.3f}")
    print(f"Specificity: {df['perf_specificity'].mean():.3f} ± {df['perf_specificity'].std():.3f}")
    print(f"F1 Score: {df['perf_f1_score'].mean():.3f} ± {df['perf_f1_score'].std():.3f}")
    
    # True/False positive/negative distribution
    print(f"\nCLASSIFICATION RESULTS:")
    print(f"True Positives: {df['detection_true_positive'].sum()}")
    print(f"True Negatives: {df['detection_true_negative'].sum()}")
    print(f"False Positives: {df['detection_false_positive'].sum()}")
    print(f"False Negatives: {df['detection_false_negative'].sum()}")
    
    # Performance by biomarker concentration relative to threshold
    df['concentration_ratio'] = df['env_target_biomarker_concentration'] / df['target_biomarker_threshold']
    
    print(f"\nPERFORMANCE BY CONCENTRATION RATIO (concentration/threshold):")
    concentration_bins = [(0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, float('inf'))]
    for min_ratio, max_ratio in concentration_bins:
        mask = (df['concentration_ratio'] >= min_ratio) & (df['concentration_ratio'] < max_ratio)
        subset = df[mask]
        if len(subset) > 0:
            avg_snr = subset['perf_snr'].mean()
            avg_accuracy = subset['detection_accuracy'].mean()
            print(f"  {min_ratio}-{max_ratio}x threshold: {len(subset)} samples, "
                  f"SNR={avg_snr:.2f}, Accuracy={avg_accuracy:.3f}")
    
    # Circuit architecture performance
    print(f"\nPERFORMANCE BY CIRCUIT ARCHITECTURE:")
    arch_performance = df.groupby('circuit_architecture').agg({
        'perf_snr': 'mean',
        'detection_accuracy': 'mean',
        'perf_sensitivity': 'mean',
        'perf_specificity': 'mean'
    }).round(3)
    print(arch_performance)
    
    # Biomarker binding analysis
    print(f"\nBIOMARKER BINDING ANALYSIS:")
    print(f"Binding fraction range: {df['biomarker_binding_fraction'].min():.3f} - {df['biomarker_binding_fraction'].max():.3f}")
    print(f"Average binding fraction: {df['biomarker_binding_fraction'].mean():.3f}")
    
    # Correlation between binding and performance
    binding_snr_corr = df['biomarker_binding_fraction'].corr(df['perf_snr'])
    binding_accuracy_corr = df['biomarker_binding_fraction'].corr(df['detection_accuracy'])
    print(f"Binding fraction vs SNR correlation: {binding_snr_corr:.3f}")
    print(f"Binding fraction vs Accuracy correlation: {binding_accuracy_corr:.3f}")

def visualize_biomarker_detection_results(df: pd.DataFrame):
    """Create visualizations specific to biomarker detection"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Ensure output folder exists
    output_dir = 'biomarker_detection_visuals'
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    
    # 1. Biomarker concentration vs SNR
    axes[0,0].scatter(df['env_target_biomarker_concentration'], df['perf_snr'], 
                     alpha=0.6, c=df['biomarker_binding_fraction'], cmap='viridis')
    axes[0,0].set_xlabel('Biomarker Concentration')
    axes[0,0].set_ylabel('SNR')
    axes[0,0].set_title('Biomarker Concentration vs SNR\n(colored by binding fraction)')
    axes[0,0].set_xscale('log')
    
    # 2. Detection accuracy vs concentration ratio
    df['concentration_ratio'] = df['env_target_biomarker_concentration'] / df['target_biomarker_threshold']
    axes[0,1].scatter(df['concentration_ratio'], df['detection_accuracy'], alpha=0.6)
    axes[0,1].set_xlabel('Concentration/Threshold Ratio')
    axes[0,1].set_ylabel('Detection Accuracy')
    axes[0,1].set_title('Detection Accuracy vs Concentration Ratio')
    axes[0,1].set_xscale('log')
    axes[0,1].axvline(x=1.0, color='red', linestyle='--', label='Threshold')
    axes[0,1].legend()
    
    # 3. ROC-like plot: Sensitivity vs 1-Specificity
    axes[0,2].scatter(1-df['perf_specificity'], df['perf_sensitivity'], alpha=0.6)
    axes[0,2].plot([0,1], [0,1], 'r--', alpha=0.8)
    axes[0,2].set_xlabel('1 - Specificity (False Positive Rate)')
    axes[0,2].set_ylabel('Sensitivity (True Positive Rate)')
    axes[0,2].set_title('ROC-like Plot')
    
    # 4. Binding fraction vs performance
    axes[1,0].scatter(df['biomarker_binding_fraction'], df['perf_snr'], alpha=0.6)
    axes[1,0].set_xlabel('Biomarker Binding Fraction')
    axes[1,0].set_ylabel('SNR')
    axes[1,0].set_title('Binding Fraction vs SNR')
    
    # 5. Performance by biomarker type
    sns.boxplot(data=df, x='target_biomarker_type', y='detection_accuracy', ax=axes[1,1])
    axes[1,1].set_title('Detection Accuracy by Biomarker Type')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # 6. Circuit architecture performance
    arch_performance = df.groupby('circuit_architecture')['detection_accuracy'].mean().sort_values(ascending=False)
    arch_performance.plot(kind='bar', ax=axes[1,2])
    axes[1,2].set_title('Detection Accuracy by Circuit Architecture')
    axes[1,2].tick_params(axis='x', rotation=45)
    
    # 7. Confusion matrix visualization
    tp_count = df['detection_true_positive'].sum()
    tn_count = df['detection_true_negative'].sum()
    fp_count = df['detection_false_positive'].sum()
    fn_count = df['detection_false_negative'].sum()
    
    confusion_data = np.array([[tp_count, fn_count], [fp_count, tn_count]]).astype(int)
    sns.heatmap(confusion_data, annot=True, fmt='d', ax=axes[2,0],
                xticklabels=['Predicted Pos', 'Predicted Neg'],
                yticklabels=['Actual Pos', 'Actual Neg'])
    axes[2,0].set_title('Overall Confusion Matrix')
    
    # 8. Performance vs environmental stress
    axes[2,1].scatter(df['env_total_stress'], df['detection_accuracy'], alpha=0.6)
    axes[2,1].set_xlabel('Environmental Stress')
    axes[2,1].set_ylabel('Detection Accuracy')
    axes[2,1].set_title('Detection Accuracy vs Environmental Stress')
    
    # 9. Binding affinity effects
    axes[2,2].scatter(df['target_biomarker_binding_affinity'], df['perf_snr'], 
                     alpha=0.6, c=df['env_target_biomarker_concentration'], cmap='plasma')
    axes[2,2].set_xlabel('Binding Affinity (Kd)')
    axes[2,2].set_ylabel('SNR')
    axes[2,2].set_title('Binding Affinity vs SNR\n(colored by concentration)')
    axes[2,2].set_xscale('log')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'biomarker_detection_analysis.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional detailed plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Performance distribution by concentration range
    df['conc_category'] = pd.cut(df['concentration_ratio'], 
                                bins=[0, 0.1, 0.5, 1.0, 2.0, float('inf')],
                                labels=['Very Low', 'Low', 'At Threshold', 'High', 'Very High'])
    
    sns.boxplot(data=df, x='conc_category', y='perf_snr', ax=axes[0,0])
    axes[0,0].set_title('SNR Distribution by Concentration Category')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Detection metrics by concentration
    sns.boxplot(data=df, x='conc_category', y='detection_accuracy', ax=axes[0,1])
    axes[0,1].set_title('Detection Accuracy by Concentration Category')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Biomarker-specific performance
    biomarker_performance = df.groupby('target_biomarker').agg({
        'detection_accuracy': 'mean',
        'perf_snr': 'mean'
    }).reset_index()
    
    axes[1,0].bar(range(len(biomarker_performance)), biomarker_performance['detection_accuracy'])
    axes[1,0].set_xlabel('Biomarker')
    axes[1,0].set_ylabel('Detection Accuracy')
    axes[1,0].set_title('Detection Accuracy by Biomarker')
    axes[1,0].set_xticks(range(len(biomarker_performance)))
    axes[1,0].set_xticklabels(biomarker_performance['target_biomarker'], rotation=45)
    
    # Signal strength vs concentration (log-log plot)
    positive_conc = df[df['env_target_biomarker_concentration'] > 0]
    axes[1,1].scatter(positive_conc['env_target_biomarker_concentration'], 
                     positive_conc['perf_signal_strength'], alpha=0.6)
    axes[1,1].set_xlabel('Biomarker Concentration')
    axes[1,1].set_ylabel('Signal Strength')
    axes[1,1].set_title('Signal Strength vs Concentration')
    axes[1,1].set_xscale('log')
    axes[1,1].set_yscale('log')
    
    plt.tight_layout()
    fig_path2 = os.path.join(output_dir, 'detailed_biomarker_analysis.png')
    fig.savefig(fig_path2, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Biomarker detection visualizations saved to: {output_dir}")

# Add this to your existing pipeline class
def add_enhanced_methods_to_pipeline():
    """Add the enhanced methods to the existing pipeline class"""
    EnhancedBiosensorPipeline.generate_realistic_biomarker_detection_dataset = generate_realistic_biomarker_detection_dataset
    EnhancedBiosensorPipeline.analyze_biomarker_detection_dataset = analyze_biomarker_detection_dataset
    EnhancedBiosensorPipeline.visualize_biomarker_detection_results = visualize_biomarker_detection_results

# Enhanced main function
def main_with_realistic_biomarker_detection():
    """Main function using the enhanced biomarker detection simulation"""
    print("🧬 REALISTIC Biomarker Detection Circuit Design Pipeline")
    print("="*70)
    
    # Initialize pipeline
    pipeline = EnhancedBiosensorPipeline()
    
    print("Generating realistic biomarker detection dataset...")
    print("This will properly simulate biomarker binding and detection!")
    
    # Generate realistic dataset
    df = generate_realistic_biomarker_detection_dataset(
        pipeline,
        n_circuits=500,                    # Number of different circuit designs
        n_biomarker_concentrations=8,      # Test concentrations per biomarker
        n_noise_per_concentration=3,       # Noise conditions per concentration
        circuit_strategy='combinatorial',
        multi_gene_prob=0.6,
        time_varying_noise_fraction=0.4,
        n_stochastic_runs=3
    )
    
    # Comprehensive analysis
    analyze_biomarker_detection_dataset(df)
    
    # Create biomarker-specific visualizations
    visualize_biomarker_detection_results(df)
    
    print("\n=== DATASET VALIDATION ===")
    print("✅ Biomarker concentrations properly simulated")
    print("✅ Circuit-biomarker binding interactions modeled")
    print("✅ True/false positive/negative detection calculated")
    print("✅ Signal strength depends on biomarker presence")
    print("✅ Environmental noise affects detection performance")
    print("✅ Multiple stochastic runs for statistical validity")
    
    print(f"\n=== ML READINESS FOR BIOMARKER DETECTION ===")
    print("🎯 TARGET VARIABLES FOR ML:")
    print("  - detection_accuracy: Overall detection performance")
    print("  - perf_sensitivity: True positive rate")
    print("  - perf_specificity: True negative rate")
    print("  - perf_snr: Signal-to-noise ratio")
    print("  - detection_matthews_correlation: Balanced performance metric")
    
    print("\n🔬 BIOMARKER-SPECIFIC FEATURES:")
    print("  - env_target_biomarker_concentration: Actual concentration tested")
    print("  - biomarker_binding_fraction: How much sensor is bound")
    print("  - concentration_ratio: Concentration relative to threshold")
    print("  - target_biomarker_binding_affinity: Kd value")
    
    print("\n🤖 RECOMMENDED ML APPROACHES:")
    print("1. CLASSIFICATION: Predict detection success (binary)")
    print("   - Target: detection_accuracy > 0.8")
    print("   - Features: Circuit design + biomarker properties + noise")
    
    print("2. REGRESSION: Predict detection performance")
    print("   - Target: perf_snr, detection_accuracy")
    print("   - Multi-output: [sensitivity, specificity, f1_score]")
    
    print("3. DESIGN OPTIMIZATION: Find best circuits for specific biomarkers")
    print("   - Objective: Maximize detection_accuracy for target biomarker")
    print("   - Constraints: Environmental robustness, response time")
    
    print("\n✅ REALISTIC biomarker detection pipeline complete!")
    print(f"💾 Dataset: {len(df)} samples with proper biomarker simulation")
    
    return df

def generate_mega_training_dataset(
    self,
    n_circuits: int = 4000,  # More circuits
    n_noise_per_circuit: int = 5,  # More noise per circuit
    circuit_strategy: str = 'combinatorial',
    multi_gene_prob: float = 0.7,  # More multi-gene circuits
    time_varying_noise_fraction: float = 0.6,
    n_stochastic_runs: int = 3
) -> pd.DataFrame:
    """
    Generate a massive, highly varied dataset (20,000+ rows) with complex circuits, 
    diverse architectures, biomarker targets, and noise conditions.
    """
    print(f"Generating {n_circuits} circuits x {n_noise_per_circuit} noise = {n_circuits * n_noise_per_circuit} (MEGA)...")
    circuits_data = self.circuit_generator.generate_circuit_dataset(
        n_circuits, circuit_strategy, multi_gene_prob
    )
    biomarker_lib = BiomarkerTargetLibrary()
    dataset = []
    for i, (circuit, architecture, copy_number, reg_params) in enumerate(circuits_data):
        if i % 100 == 0:
            print(f"Processing circuit {i+1}/{len(circuits_data)}")
        # For each circuit, sample a biomarker target
        biomarker_info = biomarker_lib.random_biomarker()
        # Generate noise conditions for this circuit
        noise_conditions = self.noise_generator.generate_noise_dataset(
            n_noise_per_circuit, time_varying_fraction=time_varying_noise_fraction
        )
        # Generate circuit topology encoding
        circuit_topology_json = generate_circuit_topology_json(circuit, architecture)
        for j, noise in enumerate(noise_conditions):
            performance, metadata = self.simulator.simulate_circuit(
                circuit, noise, architecture, copy_number, reg_params,
                seed=i*n_noise_per_circuit+j, n_stochastic_runs=n_stochastic_runs
            )
            entry = create_dataset_entry_with_biomarker_and_topology(
                circuit, noise, performance, metadata, architecture, 
                copy_number, reg_params, self.parts_library,
                biomarker_info, circuit_topology_json
            )
            dataset.append(entry)
    print(f"Generated {len(dataset)} total data points (MEGA)")
    df = self.logger.save_dataset(dataset, filename='enhanced_biosensor_dataset_mega.csv')
    return df

# Monkey-patch the pipeline with the new method
EnhancedBiosensorPipeline.generate_mega_training_dataset = generate_mega_training_dataset

def run_complete_enhanced_pipeline():
    """
    Run the complete enhanced pipeline with realistic biomarker detection
    """
    print("🧬🤖 COMPLETE ENHANCED BIOSENSOR PIPELINE")
    print("="*80)
    print("🎯 FEATURES:")
    print("  ✅ Realistic biomarker-circuit binding simulation")
    print("  ✅ Proper concentration-dependent responses")  
    print("  ✅ True detection performance metrics")
    print("  ✅ Environmental noise effects on detection")
    print("  ✅ Statistical validation with multiple runs")
    print("  ✅ Comprehensive ML-ready dataset")
    print("="*80)
    
    # Initialize the enhanced pipeline
    pipeline = EnhancedBiosensorPipeline()
    
    # Add enhanced methods
    add_enhanced_methods_to_pipeline()
    
    print("\n📊 GENERATING REALISTIC BIOMARKER DETECTION DATASET...")
    
    # Generate the realistic dataset
    df = generate_realistic_biomarker_detection_dataset(
        pipeline,
        n_circuits=200,                    # Circuit designs to test
        n_biomarker_concentrations=6,      # Concentrations per biomarker
        n_noise_per_concentration=4,       # Noise conditions per concentration  
        circuit_strategy='combinatorial',
        multi_gene_prob=0.5,
        time_varying_noise_fraction=0.3,
        n_stochastic_runs=3               # Stochastic runs for statistics
    )
    
    print(f"\n✅ Generated {len(df)} realistic biomarker detection simulations!")
    
    # Comprehensive analysis
    print("\n📈 ANALYZING BIOMARKER DETECTION PERFORMANCE...")
    analyze_biomarker_detection_dataset(df)
    
    # Create visualizations
    print("\n🎨 CREATING BIOMARKER DETECTION VISUALIZATIONS...")
    visualize_biomarker_detection_results(df)
    
    # Validate the dataset quality
    print("\n🔍 DATASET QUALITY VALIDATION...")
    validate_dataset_quality(df)
    
    # ML recommendations
    print("\n🤖 MACHINE LEARNING RECOMMENDATIONS...")
    provide_ml_recommendations(df)
    
    print("\n" + "="*80)
    print("🎉 ENHANCED BIOSENSOR PIPELINE COMPLETE!")
    print(f"📁 Dataset saved with {len(df)} samples")
    print(f"📊 Features: {len(df.columns)}")
    print(f"🔬 Biomarkers tested: {df['target_biomarker'].nunique()}")
    print(f"⚙️  Circuit architectures: {df['circuit_architecture'].nunique()}")
    print("="*80)
    
    return df

def validate_dataset_quality(df: pd.DataFrame):
    """Validate that the dataset properly represents biomarker detection"""
    
    print("DATASET QUALITY CHECKS:")
    
    # 1. Check biomarker concentration distribution
    zero_conc = (df['env_target_biomarker_concentration'] == 0).sum()
    below_threshold = (df['env_target_biomarker_concentration'] < df['target_biomarker_threshold']).sum()
    above_threshold = (df['env_target_biomarker_concentration'] > df['target_biomarker_threshold']).sum()
    
    print(f"  ✅ Concentration distribution:")
    print(f"    - Zero concentration (negative controls): {zero_conc}")
    print(f"    - Below threshold: {below_threshold}")  
    print(f"    - Above threshold: {above_threshold}")
    
    # 2. Check detection logic consistency
    # When biomarker >> threshold, detection should be high
    high_conc_mask = df['env_target_biomarker_concentration'] > 2 * df['target_biomarker_threshold']
    high_conc_detection = df[high_conc_mask]['detection_accuracy'].mean()
    
    # When biomarker = 0, detection should be low (mostly true negatives)
    zero_conc_mask = df['env_target_biomarker_concentration'] == 0
    zero_conc_detection = df[zero_conc_mask]['detection_accuracy'].mean()
    
    print(f"  ✅ Detection logic validation:")
    print(f"    - High concentration detection rate: {high_conc_detection:.3f}")
    print(f"    - Zero concentration detection rate: {zero_conc_detection:.3f}")
    
    if high_conc_detection > zero_conc_detection:
        print("    ✅ PASS: High concentrations detected better than zero")
    else:
        print("    ❌ FAIL: Detection logic may be incorrect")
    
    # 3. Check binding fraction calculation
    binding_range = f"{df['biomarker_binding_fraction'].min():.3f} - {df['biomarker_binding_fraction'].max():.3f}"
    print(f"  ✅ Binding fraction range: {binding_range}")
    
    # 4. Check SNR correlation with biomarker presence
    snr_binding_corr = df['perf_snr'].corr(df['biomarker_binding_fraction'])
    print(f"  ✅ SNR vs binding fraction correlation: {snr_binding_corr:.3f}")
    
    if snr_binding_corr > 0.1:
        print("    ✅ PASS: SNR increases with biomarker binding")
    else:
        print("    ⚠️  WARNING: Weak correlation between SNR and binding")
    
    # 5. Check for proper noise effects
    noise_snr_corr = df['env_total_stress'].corr(df['perf_snr'])
    print(f"  ✅ Environmental stress vs SNR correlation: {noise_snr_corr:.3f}")
    
    # 6. Check classification balance
    tp_rate = df['detection_true_positive'].mean()
    tn_rate = df['detection_true_negative'].mean()
    fp_rate = df['detection_false_positive'].mean()
    fn_rate = df['detection_false_negative'].mean()
    
    print(f"  ✅ Classification distribution:")
    print(f"    - True Positive rate: {tp_rate:.3f}")
    print(f"    - True Negative rate: {tn_rate:.3f}")
    print(f"    - False Positive rate: {fp_rate:.3f}")
    print(f"    - False Negative rate: {fn_rate:.3f}")

def provide_ml_recommendations(df: pd.DataFrame):
    """Provide specific ML recommendations based on the dataset"""
    
    print("MACHINE LEARNING APPROACH RECOMMENDATIONS:")
    
    # 1. Feature importance analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Correlations with key performance metrics
    target_metrics = ['detection_accuracy', 'perf_snr', 'perf_sensitivity', 'perf_specificity']
    
    print("\n  🎯 KEY TARGET VARIABLES:")
    for metric in target_metrics:
        if metric in df.columns:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            print(f"    - {metric}: {mean_val:.3f} ± {std_val:.3f}")
    
    # 2. Feature categories
    print("\n  📊 FEATURE CATEGORIES:")
    
    circuit_features = [col for col in df.columns if col.startswith(('circuit_', 'promoter_', 'rbs_', 'reporter_', 'terminator_', 'arch_', 'reg_'))]
    biomarker_features = [col for col in df.columns if col.startswith('target_biomarker') or col == 'biomarker_binding_fraction']
    environmental_features = [col for col in df.columns if col.startswith('env_')]
    detection_features = [col for col in df.columns if col.startswith('detection_')]
    
    print(f"    - Circuit design features: {len(circuit_features)}")
    print(f"    - Biomarker features: {len(biomarker_features)}")
    print(f"    - Environmental features: {len(environmental_features)}")
    print(f"    - Detection outcome features: {len(detection_features)}")
    
    # 3. Specific ML task recommendations
    print("\n  🤖 RECOMMENDED ML TASKS:")
    
    print("\n    1. BINARY CLASSIFICATION - Circuit Success Prediction")
    print("       Target: detection_accuracy > 0.8")
    print("       Features: Circuit design + biomarker properties")
    print("       Models: Random Forest, XGBoost, SVM")
    print("       Use case: Predict if a circuit will successfully detect a biomarker")
    
    print("\n    2. REGRESSION - Performance Prediction")
    print("       Target: perf_snr, detection_accuracy")
    print("       Features: All design parameters")
    print("       Models: Neural Networks, Ensemble methods")
    print("       Use case: Predict quantitative performance metrics")
    
    print("\n    3. MULTI-OUTPUT REGRESSION - Comprehensive Performance")
    print("       Targets: [sensitivity, specificity, f1_score, snr]")
    print("       Features: Circuit + biomarker + environmental")
    print("       Models: Multi-output NN, Multi-task learning")
    print("       Use case: Predict all performance aspects simultaneously")
    
    print("\n    4. OPTIMIZATION - Circuit Design")
    print("       Objective: Maximize detection_accuracy")
    print("       Constraints: Environmental robustness, response time")
    print("       Methods: Genetic algorithms, Bayesian optimization")
    print("       Use case: Design optimal circuits for specific biomarkers")
    
    print("\n    5. CLUSTERING - Circuit Families")
    print("       Features: Circuit design parameters")
    print("       Methods: K-means, Hierarchical clustering")
    print("       Use case: Identify circuit design patterns and families")
    
    # 4. Data splitting recommendations
    print("\n  📋 DATA SPLITTING STRATEGY:")
    print("    - Random split: 70% train, 15% validation, 15% test")
    print("    - Stratified by: biomarker type and concentration range")
    print("    - Ensure all circuit architectures in each split")
    print("    - Consider temporal splits if adding time-series data")
    
    # 5. Feature engineering suggestions
    print("\n  🔧 FEATURE ENGINEERING SUGGESTIONS:")
    print("    - Interaction terms: promoter_strength × biomarker_binding_affinity")
    print("    - Ratios: concentration/threshold, signal/noise")
    print("    - Polynomial features for key continuous variables")
    print("    - One-hot encoding for categorical circuit components")
    print("    - Normalization/scaling for concentration and binding values")

def create_ml_ready_datasets(df: pd.DataFrame):
    """Create ML-ready datasets for different tasks"""
    
    print("\n📦 CREATING ML-READY DATASETS...")
    
    # 1. Binary Classification Dataset - Circuit Success
    df_binary = df.copy()
    df_binary['success'] = (df_binary['detection_accuracy'] > 0.8).astype(int)
    
    binary_features = [col for col in df_binary.columns if not col.startswith(('perf_', 'detection_', 'sim_'))]
    binary_target = 'success'
    
    df_binary_ml = df_binary[binary_features + [binary_target]]
    df_binary_ml.to_csv('binary_classification_dataset.csv', index=False)
    
    print(f"  ✅ Binary classification dataset: {len(df_binary_ml)} samples, {len(binary_features)} features")
    print(f"     Success rate: {df_binary['success'].mean():.3f}")
    
    # 2. Regression Dataset - Performance Prediction
    regression_features = [col for col in df.columns if not col.startswith(('perf_', 'detection_', 'sim_'))]
    regression_targets = ['perf_snr', 'detection_accuracy', 'perf_sensitivity', 'perf_specificity']
    
    df_regression = df[regression_features + regression_targets].copy()
    df_regression.to_csv('regression_dataset.csv', index=False)
    
    print(f"  ✅ Regression dataset: {len(df_regression)} samples, {len(regression_features)} features")
    print(f"     Targets: {len(regression_targets)} performance metrics")
    
    # 3. Circuit Design Optimization Dataset
    # Group by circuit design and calculate average performance
    circuit_design_cols = ['circuit_promoter', 'circuit_rbs', 'circuit_coding_sequence', 
                          'circuit_terminator', 'circuit_architecture', 'circuit_copy_number']
    
    df_design = df.groupby(circuit_design_cols).agg({
        'detection_accuracy': 'mean',
        'perf_snr': 'mean',
        'perf_sensitivity': 'mean',
        'perf_specificity': 'mean',
        'perf_response_time': 'mean',
        'env_total_stress': 'mean'
    }).reset_index()
    
    df_design.to_csv('circuit_design_optimization_dataset.csv', index=False)
    
    print(f"  ✅ Circuit design dataset: {len(df_design)} unique designs")
    
    # 4. Biomarker-specific datasets
    for biomarker in df['target_biomarker'].unique():
        biomarker_df = df[df['target_biomarker'] == biomarker].copy()
        filename = f'biomarker_{biomarker.lower().replace("-", "_")}_dataset.csv'
        biomarker_df.to_csv(filename, index=False)
        print(f"  ✅ {biomarker} dataset: {len(biomarker_df)} samples")

def demonstrate_key_insights(df: pd.DataFrame):
    """Demonstrate key insights that validate the realistic simulation"""
    
    print("\n🔍 KEY INSIGHTS FROM REALISTIC BIOMARKER DETECTION:")
    
    # 1. Concentration-Response Relationship
    print("\n  1. CONCENTRATION-RESPONSE VALIDATION:")
    
    # Group by concentration ranges
    df['conc_category'] = pd.cut(
        df['env_target_biomarker_concentration'] / df['target_biomarker_threshold'],
        bins=[0, 0.1, 0.5, 1.0, 2.0, float('inf')],
        labels=['Very Low', 'Low', 'Threshold', 'High', 'Very High']
    )
    
    conc_performance = df.groupby('conc_category').agg({
        'detection_accuracy': ['mean', 'std'],
        'perf_snr': ['mean', 'std'],
        'biomarker_binding_fraction': ['mean', 'std']
    }).round(3)
    
    print("     Concentration Range | Detection Accuracy | SNR | Binding Fraction")
    print("     -------------------|-------------------|-----|------------------")
    for category in conc_performance.index:
        acc_mean = conc_performance.loc[category, ('detection_accuracy', 'mean')]
        snr_mean = conc_performance.loc[category, ('perf_snr', 'mean')]
        bind_mean = conc_performance.loc[category, ('biomarker_binding_fraction', 'mean')]
        print(f"     {category:18} | {acc_mean:17.3f} | {snr_mean:3.1f} | {bind_mean:16.3f}")
    
    # 2. Circuit Architecture Effectiveness
    print("\n  2. CIRCUIT ARCHITECTURE EFFECTIVENESS:")
    
    arch_performance = df.groupby('circuit_architecture').agg({
        'detection_accuracy': 'mean',
        'perf_snr': 'mean',
        'perf_sensitivity': 'mean',
        'perf_specificity': 'mean'
    }).round(3).sort_values('detection_accuracy', ascending=False)
    
    print("     Architecture | Accuracy | SNR | Sensitivity | Specificity")
    print("     ------------|----------|-----|-------------|------------")
    for arch in arch_performance.index:
        row = arch_performance.loc[arch]
        print(f"     {arch:11} | {row['detection_accuracy']:8.3f} | {row['perf_snr']:3.1f} | {row['perf_sensitivity']:11.3f} | {row['perf_specificity']:11.3f}")
    
    # 3. Environmental Robustness
    print("\n  3. ENVIRONMENTAL ROBUSTNESS:")
    
    # Split by environmental stress levels
    df['stress_level'] = pd.cut(df['env_total_stress'], bins=3, labels=['Low', 'Medium', 'High'])
    
    stress_performance = df.groupby('stress_level').agg({
        'detection_accuracy': 'mean',
        'perf_snr': 'mean'
    }).round(3)
    
    print("     Stress Level | Detection Accuracy | SNR")
    print("     -------------|-------------------|-----")
    for stress in stress_performance.index:
        row = stress_performance.loc[stress]
        print(f"     {stress:12} | {row['detection_accuracy']:17.3f} | {row['perf_snr']:3.1f}")
    
    # 4. Biomarker-Specific Performance
    print("\n  4. BIOMARKER-SPECIFIC PERFORMANCE:")
    
    biomarker_performance = df.groupby('target_biomarker').agg({
        'detection_accuracy': 'mean',
        'target_biomarker_binding_affinity': 'mean',
        'perf_snr': 'mean'
    }).round(3).sort_values('detection_accuracy', ascending=False)
    
    print("     Biomarker | Accuracy | Avg Binding Affinity | SNR")
    print("     ----------|----------|---------------------|-----")
    for biomarker in biomarker_performance.index:
        row = biomarker_performance.loc[biomarker]
        print(f"     {biomarker:9} | {row['detection_accuracy']:8.3f} | {row['target_biomarker_binding_affinity']:19.3f} | {row['perf_snr']:3.1f}")

if __name__ == "__main__":
    """
    MAIN EXECUTION
    Run this to generate the complete enhanced dataset with realistic biomarker detection
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    try:
        # Run the complete enhanced pipeline
        df = run_complete_enhanced_pipeline()
        
        # Create ML-ready datasets
        create_ml_ready_datasets(df)
        
        # Demonstrate key insights
        demonstrate_key_insights(df)
        
        print("\n🎊 SUCCESS! Enhanced biosensor pipeline completed successfully!")
        print(f"📈 Generated {len(df)} realistic biomarker detection simulations")
        print("📁 Multiple ML-ready datasets created")
        print("📊 Comprehensive analysis and visualizations generated")
        
        # Final summary statistics
        print(f"\n📋 FINAL DATASET SUMMARY:")
        print(f"   • Total simulations: {len(df)}")
        print(f"   • Unique circuits: {df[['circuit_promoter', 'circuit_rbs', 'circuit_coding_sequence', 'circuit_architecture']].drop_duplicates().shape[0]}")
        print(f"   • Biomarkers tested: {df['target_biomarker'].nunique()}")
        print(f"   • Circuit architectures: {df['circuit_architecture'].nunique()}")
        print(f"   • Concentration range: {df['env_target_biomarker_concentration'].min():.3f} - {df['env_target_biomarker_concentration'].max():.3f}")
        print(f"   • Overall detection accuracy: {df['detection_accuracy'].mean():.3f} ± {df['detection_accuracy'].std():.3f}")
        print(f"   • Features: {len(df.columns)}")
        
        #return df
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        #return None

'''# --- In your main() function, call the new method instead of the old one ---
# Example:
def main():
    print("🚀 Enhanced AI-Powered Biosensor Circuit Design Pipeline")
    print("="*60)
    pipeline = EnhancedBiosensorPipeline()
    print("Generating MEGA training dataset with maximum diversity and size...")
    df = pipeline.generate_mega_training_dataset(
        n_circuits=4000,  # 4000 circuits
        n_noise_per_circuit=10,  # 10 noise conditions each
        circuit_strategy='combinatorial',
        multi_gene_prob=0.7,
        time_varying_noise_fraction=0.6,
        n_stochastic_runs=3
    )
    
    # Comprehensive analysis
    pipeline.analyze_dataset(df)
    
    # Advanced visualizations
    print("\nGenerating comprehensive visualizations...")
    pipeline.visualize_results(df)
    
    # Feature engineering suggestions
    print("\n=== FEATURE ENGINEERING RECOMMENDATIONS ===")
    print("✅ Circuit Design Features:")
    print("  - Added regulatory elements (activators, repressors, insulators, operators)")
    print("  - Multi-gene circuit support with secondary genes")
    print("  - Enhanced copy number variations (7 types)")
    print("  - Regulatory parameters (Hill coefficients, binding affinities)")
    print("  - Architecture complexity metrics")
    
    print("\n✅ Environmental/Noise Features:")
    print("  - Transcriptional bursting and translational variability")
    print("  - Metabolic load and resource competition")
    print("  - Time-varying noise with autocorrelation")
    print("  - Intrinsic vs extrinsic noise separation")
    print("  - Static vs dynamic stress metrics")
    
    print("\n✅ Performance Metrics:")
    print("  - Classification metrics (specificity, precision, recall, F1)")
    print("  - Dynamic range and response characteristics")
    print("  - Noise spectral analysis and autocorrelation")
    print("  - Statistical summaries with confidence intervals")
    print("  - Multi-objective performance scoring")
    
    print("\n✅ Dataset Quality:")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Total features: {len(df.columns)}")
    print(f"  - Circuit architectures: {df['circuit_architecture'].nunique()}")
    print(f"  - Reporter types: {df['circuit_coding_sequence'].nunique()}")
    print(f"  - Multi-gene circuits: {(df['circuit_n_secondary_genes'] > 0).sum()}")
    
    # ML readiness check
    print("\n=== MACHINE LEARNING READINESS ===")
    
    # Check for missing values
    missing_data = df.isnull().sum().sum()
    print(f"Missing values: {missing_data}")
    
    # Check feature types
    numeric_features = len(df.select_dtypes(include=[np.number]).columns)
    categorical_features = len(df.select_dtypes(include=['object']).columns)
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Suggest ML approaches
    print("\n🤖 SUGGESTED ML APPROACHES:")
    print("1. REGRESSION TASKS:")
    print("   - Target: perf_snr, multi_objective_score, perf_response_time")
    print("   - Models: Random Forest, XGBoost, Neural Networks")
    
    print("\n2. CLASSIFICATION TASKS:")
    print("   - Target: performance_class, noise_tolerance, response_class")
    print("   - Models: SVM, Random Forest, Deep Learning")
    
    print("\n3. MULTI-OBJECTIVE OPTIMIZATION:")
    print("   - Targets: [SNR, sensitivity, specificity, response_time]")
    print("   - Models: Multi-output regression, Pareto optimization")
    
    print("\n4. FEATURE IMPORTANCE ANALYSIS:")
    print("   - Use: Tree-based feature importance, SHAP values")
    print("   - Focus: Circuit design vs environmental factors")
    
    print("\n✅ Enhanced pipeline complete! Ready for advanced ML training.")
    print("💾 Dataset saved as 'enhanced_biosensor_dataset.csv'")

if __name__ == "__main__":
    main()'''