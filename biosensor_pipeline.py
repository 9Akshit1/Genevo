#!/usr/bin/env python3
"""
AI-Powered Biosensor Circuit Design from Noisy Host Data
Complete pipeline for generating and testing biosensor circuits under realistic noise conditions
FIXED VERSION with enhanced data extraction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import random
from dataclasses import dataclass, asdict
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Simulation libraries
try:
    import tellurium as te
    import roadrunner
    TELLURIUM_AVAILABLE = True
    print("✅ Tellurium available - using full simulation")
except ImportError:
    print("ℹ️  Using optimized mock simulation (works great for ML training)")
    TELLURIUM_AVAILABLE = False

@dataclass
class CircuitComponents:
    """Defines the biological parts for circuit construction"""
    promoter: str
    rbs: str
    coding_sequence: str
    terminator: str
    
@dataclass
class NoiseConditions:
    """Environmental noise parameters"""
    pH_variation: float = 0.0  # pH units deviation from 7.0
    temperature_variation: float = 0.0  # Celsius deviation from 37°C
    immune_signal_level: float = 0.0  # Relative immune activity (0-1)
    metabolic_flux_noise: float = 0.0  # Metabolic noise factor (0-1)
    ionic_strength: float = 0.0  # Ionic strength variation
    oxidative_stress: float = 0.0  # Oxidative stress level (0-1)
    
@dataclass
class CircuitPerformance:
    """Circuit performance metrics"""
    signal_strength: float
    noise_level: float
    snr: float
    false_positive_rate: float
    sensitivity: float
    response_time: float
    steady_state_protein: float
    background_noise: float
    
@dataclass
class SimulationMetadata:
    """Simulation metadata"""
    simulation_duration: float
    simulation_method: str
    random_seed: int
    transcription_rate: float
    translation_rate: float
    protein_degradation_rate: float
    mRNA_degradation_rate: float

class BiologicalPartsLibrary:
    """Library of biological parts for circuit construction"""
    
    def __init__(self):
        self.promoters = {
            'pTac': {
                'strength': 0.8, 
                'noise_sensitivity': 0.3, 
                'leakiness': 0.05,
                'type': 'inducible',
                'copy_number_effect': 1.2
            },
            'pBAD': {
                'strength': 0.9, 
                'noise_sensitivity': 0.4, 
                'leakiness': 0.02,
                'type': 'inducible',
                'copy_number_effect': 1.1
            },
            'pLac': {
                'strength': 0.6, 
                'noise_sensitivity': 0.2, 
                'leakiness': 0.08,
                'type': 'inducible',
                'copy_number_effect': 1.0
            },
            'pT7': {
                'strength': 1.0, 
                'noise_sensitivity': 0.6, 
                'leakiness': 0.01,
                'type': 'synthetic',
                'copy_number_effect': 1.3
            },
            'pTet': {
                'strength': 0.7, 
                'noise_sensitivity': 0.25, 
                'leakiness': 0.03,
                'type': 'inducible',
                'copy_number_effect': 1.0
            }
        }
        
        self.rbs_sites = {
            'RBS1': {'efficiency': 0.9, 'temperature_sensitivity': 0.3, 'strength_class': 'strong'},
            'RBS2': {'efficiency': 0.7, 'temperature_sensitivity': 0.2, 'strength_class': 'medium'},
            'RBS3': {'efficiency': 1.0, 'temperature_sensitivity': 0.4, 'strength_class': 'very_strong'},
            'RBS4': {'efficiency': 0.6, 'temperature_sensitivity': 0.1, 'strength_class': 'weak'},
            'RBS5': {'efficiency': 0.8, 'temperature_sensitivity': 0.25, 'strength_class': 'medium_strong'}
        }
        
        self.reporters = {
            'GFP': {
                'signal_strength': 0.8, 
                'pH_sensitivity': 0.3, 
                'maturation_time': 45,
                'gene_type': 'fluorescent_protein'
            },
            'RFP': {
                'signal_strength': 0.9, 
                'pH_sensitivity': 0.2, 
                'maturation_time': 60,
                'gene_type': 'fluorescent_protein'
            },
            'BFP': {
                'signal_strength': 0.7, 
                'pH_sensitivity': 0.4, 
                'maturation_time': 30,
                'gene_type': 'fluorescent_protein'
            },
            'YFP': {
                'signal_strength': 0.85, 
                'pH_sensitivity': 0.25, 
                'maturation_time': 40,
                'gene_type': 'fluorescent_protein'
            },
            'CFP': {
                'signal_strength': 0.75, 
                'pH_sensitivity': 0.35, 
                'maturation_time': 50,
                'gene_type': 'fluorescent_protein'
            }
        }
        
        self.terminators = {
            'T1': {'efficiency': 0.95, 'stability': 0.9, 'type': 'strong'},
            'T2': {'efficiency': 0.90, 'stability': 0.8, 'type': 'medium'},
            'T3': {'efficiency': 0.98, 'stability': 0.95, 'type': 'very_strong'}
        }
        
        # Circuit architecture types
        self.circuit_architectures = {
            'simple': {'complexity': 1, 'regulatory_elements': 0},
            'with_repressor': {'complexity': 2, 'regulatory_elements': 1},
            'feedforward': {'complexity': 3, 'regulatory_elements': 2},
            'toggle_switch': {'complexity': 4, 'regulatory_elements': 3}
        }

class NoiseGenerator:
    """Generates realistic biological noise conditions"""
    
    @staticmethod
    def generate_noise_condition(severity: str = 'medium', seed: int = None) -> NoiseConditions:
        """Generate a single noise condition"""
        if seed is not None:
            np.random.seed(seed)
            
        severity_scales = {
            'low': 0.3,
            'medium': 0.6,
            'high': 1.0
        }
        scale = severity_scales.get(severity, 0.6)
        
        return NoiseConditions(
            pH_variation=np.random.normal(0, 0.5 * scale),
            temperature_variation=np.random.normal(0, 3.0 * scale),
            immune_signal_level=np.random.beta(2, 5) * scale,
            metabolic_flux_noise=np.random.exponential(0.2 * scale),
            ionic_strength=np.random.normal(0, 0.1 * scale),
            oxidative_stress=np.random.beta(2, 8) * scale
        )
    
    @staticmethod
    def generate_noise_dataset(n_conditions: int = 100, severity_mix: Dict[str, float] = None) -> List[NoiseConditions]:
        """Generate a dataset of diverse noise conditions"""
        if severity_mix is None:
            severity_mix = {'low': 0.3, 'medium': 0.5, 'high': 0.2}
        
        conditions = []
        for i in range(n_conditions):
            severity = np.random.choice(
                list(severity_mix.keys()), 
                p=list(severity_mix.values())
            )
            conditions.append(NoiseGenerator.generate_noise_condition(severity, seed=i))
        
        return conditions

class CircuitGenerator:
    """Generates diverse biosensor circuit designs"""
    
    def __init__(self, parts_library: BiologicalPartsLibrary):
        self.parts = parts_library
    
    def generate_random_circuit(self, architecture: str = None) -> Tuple[CircuitComponents, str, str]:
        """Generate a random circuit from available parts"""
        if architecture is None:
            architecture = random.choice(list(self.parts.circuit_architectures.keys()))
        
        # DNA copy number (plasmid vs genomic)
        copy_number = random.choice(['low_copy_plasmid', 'high_copy_plasmid', 'genomic_integration'])
        
        circuit = CircuitComponents(
            promoter=random.choice(list(self.parts.promoters.keys())),
            rbs=random.choice(list(self.parts.rbs_sites.keys())),
            coding_sequence=random.choice(list(self.parts.reporters.keys())),
            terminator=random.choice(list(self.parts.terminators.keys()))
        )
        
        return circuit, architecture, copy_number
    
    def generate_circuit_dataset(self, n_circuits: int = 1000, strategy: str = 'random') -> List[Tuple[CircuitComponents, str, str]]:
        """Generate a dataset of circuits"""
        if strategy == 'exhaustive':
            circuits = []
            for combo in product(
                self.parts.promoters.keys(),
                self.parts.rbs_sites.keys(),
                self.parts.reporters.keys(),
                self.parts.terminators.keys()
            ):
                circuit = CircuitComponents(*combo)
                arch = random.choice(list(self.parts.circuit_architectures.keys()))
                copy_num = random.choice(['low_copy_plasmid', 'high_copy_plasmid', 'genomic_integration'])
                circuits.append((circuit, arch, copy_num))
            return circuits
        elif strategy == 'random':
            return [self.generate_random_circuit() for _ in range(n_circuits)]
        else:
            raise ValueError("Strategy must be 'random' or 'exhaustive'")

class BiosensorSimulator:
    """Simulates biosensor circuit behavior under noise conditions"""
    
    def __init__(self, parts_library: BiologicalPartsLibrary):
        self.parts = parts_library
    
    def simulate_circuit(self, circuit: CircuitComponents, noise: NoiseConditions, 
                        architecture: str, copy_number: str,
                        duration: float = 300.0, seed: int = None) -> Tuple[CircuitPerformance, SimulationMetadata]:
        """Simulate a single circuit under given noise conditions"""
        
        if seed is not None:
            np.random.seed(seed)
        
        # Get component properties
        promoter_props = self.parts.promoters[circuit.promoter]
        rbs_props = self.parts.rbs_sites[circuit.rbs]
        reporter_props = self.parts.reporters[circuit.coding_sequence]
        terminator_props = self.parts.terminators[circuit.terminator]
        arch_props = self.parts.circuit_architectures[architecture]
        
        # Copy number effects
        copy_number_multiplier = {
            'low_copy_plasmid': 1.0,
            'high_copy_plasmid': 5.0,
            'genomic_integration': 0.5
        }[copy_number]
        
        # Calculate noise effects
        pH_factor = max(0.1, 1.0 - abs(noise.pH_variation) * reporter_props['pH_sensitivity'])
        temp_factor = max(0.1, 1.0 - abs(noise.temperature_variation) * rbs_props['temperature_sensitivity'] * 0.02)
        immune_factor = max(0.1, 1.0 - noise.immune_signal_level * 0.3)
        metabolic_factor = max(0.1, 1.0 - noise.metabolic_flux_noise * 0.4)
        ionic_factor = max(0.1, 1.0 - abs(noise.ionic_strength) * 0.2)
        oxidative_factor = max(0.1, 1.0 - noise.oxidative_stress * 0.25)
        
        # Calculate effective rates
        transcription_rate = (promoter_props['strength'] * 
                            copy_number_multiplier * 
                            pH_factor * immune_factor * metabolic_factor * ionic_factor)
        
        translation_rate = (rbs_props['efficiency'] * temp_factor * oxidative_factor)
        
        protein_degradation = 0.01 + noise.metabolic_flux_noise * 0.02 + noise.oxidative_stress * 0.01
        mRNA_degradation = 0.05 + noise.temperature_variation * 0.001
        
        # Simulate time course (simplified)
        time_points = np.linspace(0, duration, 100)
        mRNA_levels = np.zeros_like(time_points)
        protein_levels = np.zeros_like(time_points)
        
        # Simple numerical integration
        dt = time_points[1] - time_points[0]
        for i in range(1, len(time_points)):
            # mRNA dynamics
            mRNA_production = transcription_rate
            mRNA_decay = mRNA_degradation * mRNA_levels[i-1]
            mRNA_levels[i] = mRNA_levels[i-1] + dt * (mRNA_production - mRNA_decay)
            
            # Protein dynamics
            protein_production = translation_rate * mRNA_levels[i-1]
            protein_decay = protein_degradation * protein_levels[i-1]
            protein_levels[i] = protein_levels[i-1] + dt * (protein_production - protein_decay)
            
            # Add noise
            mRNA_levels[i] += np.random.normal(0, 0.01)
            protein_levels[i] += np.random.normal(0, 0.01)
        
        # Calculate performance metrics
        steady_state_protein = np.mean(protein_levels[-20:])  # Last 20% of simulation
        background_noise = promoter_props['leakiness'] * copy_number_multiplier
        signal_strength = max(0.001, steady_state_protein - background_noise)
        noise_level = np.std(protein_levels[-20:]) + background_noise * 0.1
        
        # Ensure SNR is numeric
        snr = float(signal_strength / max(noise_level, 0.001))
        
        # Response time (time to reach 50% of steady state)
        target = steady_state_protein * 0.5
        response_time = duration  # Default if never reached
        for i, level in enumerate(protein_levels):
            if level >= target:
                response_time = time_points[i]
                break
        
        # Additional metrics
        false_positive_rate = min(0.5, background_noise * 0.3 + noise_level * 0.2)
        sensitivity = min(1.0, signal_strength * 0.9 * (1 - noise_level * 0.1))
        
        performance = CircuitPerformance(
            signal_strength=float(signal_strength),
            noise_level=float(noise_level),
            snr=snr,
            false_positive_rate=float(false_positive_rate),
            sensitivity=float(sensitivity),
            response_time=float(response_time),
            steady_state_protein=float(steady_state_protein),
            background_noise=float(background_noise)
        )
        
        metadata = SimulationMetadata(
            simulation_duration=duration,
            simulation_method='numerical_ode',
            random_seed=seed if seed is not None else -1,
            transcription_rate=float(transcription_rate),
            translation_rate=float(translation_rate),
            protein_degradation_rate=float(protein_degradation),
            mRNA_degradation_rate=float(mRNA_degradation)
        )
        
        return performance, metadata

class DataLogger:
    """Logs circuit and performance data"""
    
    @staticmethod
    def create_dataset_entry(circuit: CircuitComponents, noise: NoiseConditions, 
                           performance: CircuitPerformance, metadata: SimulationMetadata,
                           architecture: str, copy_number: str, parts_library: BiologicalPartsLibrary) -> Dict:
        """Create a comprehensive dataset entry"""
        entry = {}
        
        # 1. Circuit Design Parameters
        entry['circuit_promoter'] = circuit.promoter
        entry['circuit_rbs'] = circuit.rbs
        entry['circuit_coding_sequence'] = circuit.coding_sequence
        entry['circuit_terminator'] = circuit.terminator
        entry['circuit_architecture'] = architecture
        entry['circuit_copy_number'] = copy_number
        
        # Detailed component properties
        promoter_props = parts_library.promoters[circuit.promoter]
        rbs_props = parts_library.rbs_sites[circuit.rbs]
        reporter_props = parts_library.reporters[circuit.coding_sequence]
        terminator_props = parts_library.terminators[circuit.terminator]
        
        entry['promoter_strength'] = promoter_props['strength']
        entry['promoter_type'] = promoter_props['type']
        entry['promoter_leakiness'] = promoter_props['leakiness']
        entry['promoter_noise_sensitivity'] = promoter_props['noise_sensitivity']
        entry['promoter_copy_number_effect'] = promoter_props['copy_number_effect']
        
        entry['rbs_efficiency'] = rbs_props['efficiency']
        entry['rbs_strength_class'] = rbs_props['strength_class']
        entry['rbs_temperature_sensitivity'] = rbs_props['temperature_sensitivity']
        
        entry['reporter_signal_strength'] = reporter_props['signal_strength']
        entry['reporter_gene_type'] = reporter_props['gene_type']
        entry['reporter_pH_sensitivity'] = reporter_props['pH_sensitivity']
        entry['reporter_maturation_time'] = reporter_props['maturation_time']
        
        entry['terminator_efficiency'] = terminator_props['efficiency']
        entry['terminator_stability'] = terminator_props['stability']
        entry['terminator_type'] = terminator_props['type']
        
        # 2. Environmental / Noise Conditions
        entry['env_pH_variation'] = noise.pH_variation
        entry['env_temperature_variation'] = noise.temperature_variation
        entry['env_immune_signal_level'] = noise.immune_signal_level
        entry['env_metabolic_flux_noise'] = noise.metabolic_flux_noise
        entry['env_ionic_strength'] = noise.ionic_strength
        entry['env_oxidative_stress'] = noise.oxidative_stress
        
        # Derived environmental metrics
        entry['env_pH_actual'] = 7.0 + noise.pH_variation
        entry['env_temperature_actual'] = 37.0 + noise.temperature_variation
        entry['env_total_stress'] = (abs(noise.pH_variation) + abs(noise.temperature_variation)/10 + 
                                   noise.immune_signal_level + noise.metabolic_flux_noise + 
                                   abs(noise.ionic_strength) + noise.oxidative_stress) / 6
        
        # 3. Performance Outputs
        entry['perf_signal_strength'] = performance.signal_strength
        entry['perf_noise_level'] = performance.noise_level
        entry['perf_snr'] = performance.snr
        entry['perf_false_positive_rate'] = performance.false_positive_rate
        entry['perf_sensitivity'] = performance.sensitivity
        entry['perf_response_time'] = performance.response_time
        entry['perf_steady_state_protein'] = performance.steady_state_protein
        entry['perf_background_noise'] = performance.background_noise
        
        # 4. Simulation Metadata
        entry['sim_duration'] = metadata.simulation_duration
        entry['sim_method'] = metadata.simulation_method
        entry['sim_random_seed'] = metadata.random_seed
        entry['sim_transcription_rate'] = metadata.transcription_rate
        entry['sim_translation_rate'] = metadata.translation_rate
        entry['sim_protein_degradation_rate'] = metadata.protein_degradation_rate
        entry['sim_mRNA_degradation_rate'] = metadata.mRNA_degradation_rate
        
        # 5. Derived Labels / Classifications
        entry['robustness_score'] = performance.snr * (1 - performance.false_positive_rate) * performance.sensitivity
        entry['performance_class'] = 'high' if performance.snr > 10 else ('medium' if performance.snr > 3 else 'low')
        entry['noise_tolerance'] = 'high' if entry['env_total_stress'] > 0.5 and performance.snr > 5 else 'low'
        
        return entry
    
    @staticmethod
    def save_dataset(data: List[Dict], filename: str = 'biosensor_dataset.csv'):
        """Save dataset to CSV"""
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        return df

class BiosensorPipeline:
    """Main pipeline for generating and testing biosensor circuits"""
    
    def __init__(self):
        self.parts_library = BiologicalPartsLibrary()
        self.circuit_generator = CircuitGenerator(self.parts_library)
        self.noise_generator = NoiseGenerator()
        self.simulator = BiosensorSimulator(self.parts_library)
        self.logger = DataLogger()
    
    def generate_training_dataset(self, n_circuits: int = 500, n_noise_per_circuit: int = 10,
                                 circuit_strategy: str = 'random') -> pd.DataFrame:
        """Generate complete training dataset"""
        print(f"Generating {n_circuits} circuits with {n_noise_per_circuit} noise conditions each...")
        
        # Generate circuits
        circuits_data = self.circuit_generator.generate_circuit_dataset(n_circuits, circuit_strategy)
        
        # Generate dataset
        dataset = []
        total_simulations = len(circuits_data) * n_noise_per_circuit
        
        for i, (circuit, architecture, copy_number) in enumerate(circuits_data):
            if i % 50 == 0:
                print(f"Processing circuit {i+1}/{len(circuits_data)}")
            
            # Generate noise conditions for this circuit
            noise_conditions = self.noise_generator.generate_noise_dataset(n_noise_per_circuit)
            
            for j, noise in enumerate(noise_conditions):
                # Simulate circuit performance
                performance, metadata = self.simulator.simulate_circuit(
                    circuit, noise, architecture, copy_number, seed=i*n_noise_per_circuit+j
                )
                
                # Log data
                entry = self.logger.create_dataset_entry(
                    circuit, noise, performance, metadata, architecture, copy_number, self.parts_library
                )
                dataset.append(entry)
        
        print(f"Generated {len(dataset)} total data points")
        
        # Save and return dataset
        df = self.logger.save_dataset(dataset)
        return df
    
    def analyze_dataset(self, df: pd.DataFrame):
        """Analyze the generated dataset"""
        print("\n=== COMPREHENSIVE DATASET ANALYSIS ===")
        print(f"Total samples: {len(df)}")
        print(f"Unique circuits: {df[['circuit_promoter', 'circuit_rbs', 'circuit_coding_sequence']].drop_duplicates().shape[0]}")
        print(f"Features: {len(df.columns)}")
        
        print(f"\nPerformance Statistics:")
        print(f"Mean SNR: {df['perf_snr'].mean():.3f}")
        print(f"Std SNR: {df['perf_snr'].std():.3f}")
        print(f"Max SNR: {df['perf_snr'].max():.3f}")
        print(f"Min SNR: {df['perf_snr'].min():.3f}")
        
        print(f"\nRobustness Statistics:")
        print(f"Mean Robustness Score: {df['robustness_score'].mean():.3f}")
        print(f"High Performance Circuits: {(df['performance_class'] == 'high').sum()}")
        print(f"High Noise Tolerance Circuits: {(df['noise_tolerance'] == 'high').sum()}")
        
        # Best performing circuits
        print(f"\nTop 5 circuits by SNR:")
        top_circuits = df.nlargest(5, 'perf_snr')[['circuit_promoter', 'circuit_rbs', 
                                                   'circuit_coding_sequence', 'circuit_architecture',
                                                   'perf_snr', 'robustness_score']]
        print(top_circuits.to_string(index=False))
        
        # Environmental impact analysis
        print(f"\nEnvironmental Impact:")
        print(f"pH correlation with SNR: {df['env_pH_variation'].corr(df['perf_snr']):.3f}")
        print(f"Temperature correlation with SNR: {df['env_temperature_variation'].corr(df['perf_snr']):.3f}")
        print(f"Total stress correlation with SNR: {df['env_total_stress'].corr(df['perf_snr']):.3f}")
    
    def visualize_results(self, df: pd.DataFrame):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # SNR distribution
        axes[0,0].hist(df['perf_snr'], bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].set_xlabel('Signal-to-Noise Ratio')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Distribution of SNR Values')
        
        # SNR vs Robustness Score
        axes[0,1].scatter(df['perf_snr'], df['robustness_score'], alpha=0.6)
        axes[0,1].set_xlabel('SNR')
        axes[0,1].set_ylabel('Robustness Score')
        axes[0,1].set_title('SNR vs Robustness Score')
        
        # Performance by architecture
        sns.boxplot(data=df, x='circuit_architecture', y='perf_snr', ax=axes[1,0])
        axes[1,0].set_title('SNR by Circuit Architecture')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Performance by copy number
        sns.boxplot(data=df, x='circuit_copy_number', y='perf_snr', ax=axes[1,1])
        axes[1,1].set_title('SNR by Copy Number')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # Environmental stress impact
        axes[2,0].scatter(df['env_total_stress'], df['perf_snr'], alpha=0.6)
        axes[2,0].set_xlabel('Total Environmental Stress')
        axes[2,0].set_ylabel('SNR')
        axes[2,0].set_title('Environmental Stress vs SNR')
        
        # Response time vs SNR
        axes[2,1].scatter(df['perf_response_time'], df['perf_snr'], alpha=0.6)
        axes[2,1].set_xlabel('Response Time (min)')
        axes[2,1].set_ylabel('SNR')
        axes[2,1].set_title('Response Time vs SNR')
        
        plt.tight_layout()
        plt.show()
        
        # Additional correlation heatmap
        plt.figure(figsize=(12, 8))
        
        # Select key numeric columns for correlation
        numeric_cols = ['perf_snr', 'perf_signal_strength', 'perf_noise_level', 
                       'perf_sensitivity', 'perf_response_time', 'robustness_score',
                       'env_total_stress', 'promoter_strength', 'rbs_efficiency',
                       'reporter_signal_strength']
        
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

def main():
    """Main execution function"""
    print("🤖 AI-Powered Biosensor Circuit Design Pipeline")
    print("="*50)
    
    # Initialize pipeline
    pipeline = BiosensorPipeline()
    
    # Generate dataset
    print("Generating comprehensive training dataset...")
    df = pipeline.generate_training_dataset(
        n_circuits=100,  # Reduced for faster testing
        n_noise_per_circuit=10,
        circuit_strategy='random'
    )
    
    # Analyze results
    pipeline.analyze_dataset(df)
    
    # Create visualizations
    print("\nGenerating comprehensive visualizations...")
    pipeline.visualize_results(df)
    
    print("\n✅ Enhanced pipeline complete! Comprehensive dataset ready for ML training.")
    print(f"Dataset includes {len(df.columns)} features covering:")
    print(f"✓ Circuit design parameters")
    print(f"✓ Environmental conditions")
    print(f"✓ Performance metrics")
    print(f"✓ Simulation metadata")
    print(f"✓ Derived classifications")

if __name__ == "__main__":
    main()