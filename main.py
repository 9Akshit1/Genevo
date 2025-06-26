#!/usr/bin/env python3
"""
AI-Powered Biosensor Circuit Design from Noisy Host Data
Complete pipeline for generating and testing biosensor circuits under realistic noise conditions
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

# Simulation libraries (install with: pip install tellurium bioscrape)
try:
    import tellurium as te
    import roadrunner
    TELLURIUM_AVAILABLE = True
except ImportError:
    print("Warning: Tellurium not available. Using mock simulation.")
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
    
@dataclass
class CircuitPerformance:
    """Circuit performance metrics"""
    signal_strength: float
    noise_level: float
    snr: float
    false_positive_rate: float
    sensitivity: float
    
class BiologicalPartsLibrary:
    """Library of biological parts for circuit construction"""
    
    def __init__(self):
        self.promoters = {
            'pTac': {'strength': 0.8, 'noise_sensitivity': 0.3, 'leakiness': 0.05},
            'pBAD': {'strength': 0.9, 'noise_sensitivity': 0.4, 'leakiness': 0.02},
            'pLac': {'strength': 0.6, 'noise_sensitivity': 0.2, 'leakiness': 0.08},
            'pT7': {'strength': 1.0, 'noise_sensitivity': 0.6, 'leakiness': 0.01},
            'pTet': {'strength': 0.7, 'noise_sensitivity': 0.25, 'leakiness': 0.03}
        }
        
        self.rbs_sites = {
            'RBS1': {'efficiency': 0.9, 'temperature_sensitivity': 0.3},
            'RBS2': {'efficiency': 0.7, 'temperature_sensitivity': 0.2},
            'RBS3': {'efficiency': 1.0, 'temperature_sensitivity': 0.4},
            'RBS4': {'efficiency': 0.6, 'temperature_sensitivity': 0.1},
            'RBS5': {'efficiency': 0.8, 'temperature_sensitivity': 0.25}
        }
        
        self.reporters = {
            'GFP': {'signal_strength': 0.8, 'pH_sensitivity': 0.3, 'maturation_time': 45},
            'RFP': {'signal_strength': 0.9, 'pH_sensitivity': 0.2, 'maturation_time': 60},
            'BFP': {'signal_strength': 0.7, 'pH_sensitivity': 0.4, 'maturation_time': 30},
            'YFP': {'signal_strength': 0.85, 'pH_sensitivity': 0.25, 'maturation_time': 40},
            'CFP': {'signal_strength': 0.75, 'pH_sensitivity': 0.35, 'maturation_time': 50}
        }
        
        self.terminators = {
            'T1': {'efficiency': 0.95, 'stability': 0.9},
            'T2': {'efficiency': 0.90, 'stability': 0.8},
            'T3': {'efficiency': 0.98, 'stability': 0.95}
        }

class NoiseGenerator:
    """Generates realistic biological noise conditions"""
    
    @staticmethod
    def generate_noise_condition(severity: str = 'medium') -> NoiseConditions:
        """Generate a single noise condition"""
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
            metabolic_flux_noise=np.random.exponential(0.2 * scale)
        )
    
    @staticmethod
    def generate_noise_dataset(n_conditions: int = 100, severity_mix: Dict[str, float] = None) -> List[NoiseConditions]:
        """Generate a dataset of diverse noise conditions"""
        if severity_mix is None:
            severity_mix = {'low': 0.3, 'medium': 0.5, 'high': 0.2}
        
        conditions = []
        for _ in range(n_conditions):
            severity = np.random.choice(
                list(severity_mix.keys()), 
                p=list(severity_mix.values())
            )
            conditions.append(NoiseGenerator.generate_noise_condition(severity))
        
        return conditions

class CircuitGenerator:
    """Generates diverse biosensor circuit designs"""
    
    def __init__(self, parts_library: BiologicalPartsLibrary):
        self.parts = parts_library
    
    def generate_random_circuit(self) -> CircuitComponents:
        """Generate a random circuit from available parts"""
        return CircuitComponents(
            promoter=random.choice(list(self.parts.promoters.keys())),
            rbs=random.choice(list(self.parts.rbs_sites.keys())),
            coding_sequence=random.choice(list(self.parts.reporters.keys())),
            terminator=random.choice(list(self.parts.terminators.keys()))
        )
    
    def generate_all_combinations(self) -> List[CircuitComponents]:
        """Generate all possible circuit combinations"""
        combinations = []
        for combo in product(
            self.parts.promoters.keys(),
            self.parts.rbs_sites.keys(),
            self.parts.reporters.keys(),
            self.parts.terminators.keys()
        ):
            combinations.append(CircuitComponents(*combo))
        return combinations
    
    def generate_circuit_dataset(self, n_circuits: int = 1000, strategy: str = 'random') -> List[CircuitComponents]:
        """Generate a dataset of circuits"""
        if strategy == 'exhaustive':
            return self.generate_all_combinations()
        elif strategy == 'random':
            return [self.generate_random_circuit() for _ in range(n_circuits)]
        else:
            raise ValueError("Strategy must be 'random' or 'exhaustive'")

class BiosensorSimulator:
    """Simulates biosensor circuit behavior under noise conditions"""
    
    def __init__(self, parts_library: BiologicalPartsLibrary):
        self.parts = parts_library
    
    def _create_sbml_model(self, circuit: CircuitComponents, noise: NoiseConditions) -> str:
        """Create SBML model string for the circuit"""
        promoter_props = self.parts.promoters[circuit.promoter]
        rbs_props = self.parts.rbs_sites[circuit.rbs]
        reporter_props = self.parts.reporters[circuit.coding_sequence]
        terminator_props = self.parts.terminators[circuit.terminator]
        
        # Adjust parameters based on noise conditions
        pH_factor = 1.0 - abs(noise.pH_variation) * 0.1
        temp_factor = 1.0 - abs(noise.temperature_variation) * 0.02
        immune_factor = 1.0 - noise.immune_signal_level * 0.2
        metabolic_factor = 1.0 - noise.metabolic_flux_noise * 0.3
        
        # Effective parameters
        transcription_rate = (promoter_props['strength'] * 
                            pH_factor * immune_factor * metabolic_factor)
        translation_rate = (rbs_props['efficiency'] * temp_factor)
        protein_degradation = 0.01 + noise.metabolic_flux_noise * 0.02
        
        # SBML model
        model = f"""
        model biosensor_circuit
            // Species
            species mRNA, Protein, Signal
            
            // Parameters
            transcription_rate = {transcription_rate:.4f}
            translation_rate = {translation_rate:.4f}
            mRNA_degradation = 0.05
            protein_degradation = {protein_degradation:.4f}
            signal_input = 1.0
            
            // Reactions
            transcription: Signal -> Signal + mRNA; transcription_rate * Signal
            translation: mRNA -> mRNA + Protein; translation_rate * mRNA
            mRNA_decay: mRNA -> ; mRNA_degradation * mRNA
            protein_decay: Protein -> ; protein_degradation * Protein
            
            // Initial conditions
            mRNA = 0
            Protein = 0
            Signal = 1.0
        end
        """
        return model
    
    def simulate_circuit(self, circuit: CircuitComponents, noise: NoiseConditions, 
                        duration: float = 300.0) -> CircuitPerformance:
        """Simulate a single circuit under given noise conditions"""
        
        if TELLURIUM_AVAILABLE:
            try:
                # Create and simulate model
                model_str = self._create_sbml_model(circuit, noise)
                r = te.loada(model_str)
                
                # Run simulation
                result = r.simulate(0, duration, 100)
                
                # Extract protein levels (our signal)
                protein_levels = result[:, -1]  # Last column is Protein
                
                # Calculate performance metrics
                signal_strength = np.mean(protein_levels[-20:])  # Steady state
                noise_level = np.std(protein_levels[-20:])
                snr = signal_strength / max(noise_level, 0.001)
                
                # Mock additional metrics
                false_positive_rate = max(0, noise_level * 0.1)
                sensitivity = min(1.0, signal_strength * 0.8)
                
            except Exception as e:
                print(f"Simulation error: {e}")
                return self._mock_simulation(circuit, noise)
        else:
            return self._mock_simulation(circuit, noise)
        
        return CircuitPerformance(
            signal_strength=signal_strength,
            noise_level=noise_level,
            snr=snr,
            false_positive_rate=false_positive_rate,
            sensitivity=sensitivity
        )
    
    def _mock_simulation(self, circuit: CircuitComponents, noise: NoiseConditions) -> CircuitPerformance:
        """Mock simulation for when Tellurium is not available"""
        # Get component properties
        promoter_props = self.parts.promoters[circuit.promoter]
        rbs_props = self.parts.rbs_sites[circuit.rbs]
        reporter_props = self.parts.reporters[circuit.coding_sequence]
        
        # Calculate base signal strength
        base_signal = (promoter_props['strength'] * 
                      rbs_props['efficiency'] * 
                      reporter_props['signal_strength'])
        
        # Apply noise effects
        pH_penalty = abs(noise.pH_variation) * reporter_props['pH_sensitivity']
        temp_penalty = abs(noise.temperature_variation) * rbs_props['temperature_sensitivity'] * 0.01
        immune_penalty = noise.immune_signal_level * 0.2
        metabolic_penalty = noise.metabolic_flux_noise * 0.3
        
        # Final signal strength
        signal_strength = max(0.1, base_signal - pH_penalty - temp_penalty - 
                            immune_penalty - metabolic_penalty)
        
        # Noise level increases with environmental stress
        noise_level = (abs(noise.pH_variation) * 0.1 + 
                      abs(noise.temperature_variation) * 0.01 +
                      noise.immune_signal_level * 0.15 +
                      noise.metabolic_flux_noise * 0.2 +
                      promoter_props['noise_sensitivity'] * 0.1)
        
        snr = signal_strength / max(noise_level, 0.001)
        false_positive_rate = min(0.5, noise_level * 0.2)
        sensitivity = min(1.0, signal_strength * 0.9)
        
        return CircuitPerformance(
            signal_strength=signal_strength,
            noise_level=noise_level,
            snr=snr,
            false_positive_rate=false_positive_rate,
            sensitivity=sensitivity
        )

class DataLogger:
    """Logs circuit and performance data"""
    
    @staticmethod
    def create_dataset_entry(circuit: CircuitComponents, noise: NoiseConditions, 
                           performance: CircuitPerformance) -> Dict:
        """Create a single dataset entry"""
        entry = {}
        
        # Circuit features
        entry.update({f'circuit_{k}': v for k, v in asdict(circuit).items()})
        
        # Noise features
        entry.update({f'noise_{k}': v for k, v in asdict(noise).items()})
        
        # Performance targets
        entry.update({f'perf_{k}': v for k, v in asdict(performance).items()})
        
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
        circuits = self.circuit_generator.generate_circuit_dataset(n_circuits, circuit_strategy)
        
        # Generate dataset
        dataset = []
        total_simulations = len(circuits) * n_noise_per_circuit
        
        for i, circuit in enumerate(circuits):
            if i % 50 == 0:
                print(f"Processing circuit {i+1}/{len(circuits)}")
            
            # Generate noise conditions for this circuit
            noise_conditions = self.noise_generator.generate_noise_dataset(n_noise_per_circuit)
            
            for noise in noise_conditions:
                # Simulate circuit performance
                performance = self.simulator.simulate_circuit(circuit, noise)
                
                # Log data
                entry = self.logger.create_dataset_entry(circuit, noise, performance)
                dataset.append(entry)
        
        print(f"Generated {len(dataset)} total data points")
        
        # Save and return dataset
        df = self.logger.save_dataset(dataset)
        return df
    
    def analyze_dataset(self, df: pd.DataFrame):
        """Analyze the generated dataset"""
        print("\n=== DATASET ANALYSIS ===")
        print(f"Total samples: {len(df)}")
        print(f"Unique circuits: {df[['circuit_promoter', 'circuit_rbs', 'circuit_coding_sequence']].drop_duplicates().shape[0]}")
        
        print(f"\nSNR Statistics:")
        print(f"Mean SNR: {df['perf_snr'].mean():.3f}")
        print(f"Std SNR: {df['perf_snr'].std():.3f}")
        print(f"Max SNR: {df['perf_snr'].max():.3f}")
        print(f"Min SNR: {df['perf_snr'].min():.3f}")
        
        # Best performing circuits
        print(f"\nTop 5 circuits by SNR:")
        top_circuits = df.nlargest(5, 'perf_snr')[['circuit_promoter', 'circuit_rbs', 
                                                   'circuit_coding_sequence', 'perf_snr']]
        print(top_circuits.to_string(index=False))
    
    def visualize_results(self, df: pd.DataFrame):
        """Create visualizations of the results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # SNR distribution
        axes[0,0].hist(df['perf_snr'], bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].set_xlabel('Signal-to-Noise Ratio')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Distribution of SNR Values')
        
        # SNR vs Noise Level
        axes[0,1].scatter(df['perf_noise_level'], df['perf_snr'], alpha=0.6)
        axes[0,1].set_xlabel('Noise Level')
        axes[0,1].set_ylabel('SNR')
        axes[0,1].set_title('SNR vs Noise Level')
        
        # Performance by promoter
        sns.boxplot(data=df, x='circuit_promoter', y='perf_snr', ax=axes[1,0])
        axes[1,0].set_title('SNR by Promoter Type')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Performance by reporter
        sns.boxplot(data=df, x='circuit_coding_sequence', y='perf_snr', ax=axes[1,1])
        axes[1,1].set_title('SNR by Reporter Type')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main execution function"""
    print("🤖 AI-Powered Biosensor Circuit Design Pipeline")
    print("="*50)
    
    # Initialize pipeline
    pipeline = BiosensorPipeline()
    
    # Generate dataset
    print("Generating training dataset...")
    df = pipeline.generate_training_dataset(
        n_circuits=200,  # Start small for testing
        n_noise_per_circuit=5,
        circuit_strategy='random'
    )
    
    # Analyze results
    pipeline.analyze_dataset(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    pipeline.visualize_results(df)
    
    print("\n✅ Pipeline complete! Dataset ready for ML training.")
    print(f"Next steps:")
    print(f"1. Train RL agent on this dataset")
    print(f"2. Use agent to design optimized circuits")
    print(f"3. Validate designs experimentally")

if __name__ == "__main__":
    main()