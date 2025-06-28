import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class CircuitDesignPredictor:
    """
    Predicts circuit performance metrics given design parameters and environmental conditions
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.target_metrics = [
            'perf_signal_strength', 'perf_noise_level', 'perf_snr', 
            'perf_false_positive_rate', 'perf_response_time', 
            'perf_steady_state_protein', 'robustness_score'
        ]
        
    def prepare_features(self, df, is_training=True):
        """Prepare features for model training or prediction"""
        # Categorical features to encode
        categorical_features = [
            'circuit_promoter', 'circuit_rbs', 'circuit_coding_sequence', 
            'circuit_terminator', 'circuit_architecture', 'circuit_copy_number',
            'promoter_type', 'rbs_strength_class', 'reporter_gene_type',
            'terminator_type', 'performance_class', 'noise_tolerance', 'sim_method'
        ]
        
        # Numerical features
        numerical_features = [
            'promoter_strength', 'promoter_leakiness', 'promoter_noise_sensitivity',
            'promoter_copy_number_effect', 'rbs_efficiency', 'rbs_temperature_sensitivity',
            'reporter_signal_strength', 'reporter_pH_sensitivity', 'reporter_maturation_time',
            'terminator_efficiency', 'terminator_stability', 'env_pH_variation',
            'env_temperature_variation', 'env_immune_signal_level', 'env_metabolic_flux_noise',
            'env_ionic_strength', 'env_oxidative_stress', 'env_pH_actual',
            'env_temperature_actual', 'env_total_stress', 'sim_transcription_rate',
            'sim_translation_rate', 'sim_protein_degradation_rate', 'sim_mRNA_degradation_rate'
        ]
        
        # Make a copy of the dataframe
        df_encoded = df.copy()
        
        if is_training:
            # During training, store which features are available
            available_features = list(df.columns)
            self.available_categorical = [f for f in categorical_features if f in available_features]
            self.available_numerical = [f for f in numerical_features if f in available_features]
            
            # Encode categorical features
            for feature in self.available_categorical:
                if feature not in self.encoders:
                    self.encoders[feature] = LabelEncoder()
                df_encoded[feature] = self.encoders[feature].fit_transform(df[feature].astype(str))
            
            # Store feature columns for consistency
            self.feature_columns = self.available_numerical + self.available_categorical
        else:
            # During prediction, ensure all required features are present
            # Add missing features with default values
            for feature in self.feature_columns:
                if feature not in df_encoded.columns:
                    if feature in categorical_features:
                        # Use the most common category as default
                        if feature in self.encoders:
                            df_encoded[feature] = 0  # Use first encoded value as default
                    else:
                        # Use median value as default for numerical features
                        df_encoded[feature] = 0.5  # Reasonable default
            
            # Encode categorical features using existing encoders
            for feature in self.available_categorical:
                if feature in df_encoded.columns:
                    # Handle unseen categories by using the first category as default
                    unique_vals = df_encoded[feature].astype(str).unique()
                    encoded_vals = []
                    for val in df_encoded[feature].astype(str):
                        if val in self.encoders[feature].classes_:
                            encoded_vals.append(self.encoders[feature].transform([val])[0])
                        else:
                            # Use first category for unknown values
                            encoded_vals.append(0)
                    df_encoded[feature] = encoded_vals
        
        return df_encoded[self.feature_columns]
    
    def train(self, df):
        """Train models to predict circuit performance metrics"""
        print("Training circuit performance prediction models...")
        
        # Prepare features
        X = self.prepare_features(df, is_training=True)
        
        # Train a model for each target metric
        for metric in self.target_metrics:
            if metric in df.columns:
                print(f"Training model for {metric}...")
                
                y = df[metric]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                print(f"  {metric}: R² = {r2:.3f}, RMSE = {rmse:.3f}")
                
                # Store model and scaler
                self.models[metric] = model
                self.scalers[metric] = scaler
    
    def predict(self, circuit_design):
        """
        Predict performance metrics for a given circuit design
        
        Args:
            circuit_design: dict with circuit parameters
            
        Returns:
            dict with predicted performance metrics
        """
        # Convert to DataFrame
        df_design = pd.DataFrame([circuit_design])
        
        # Prepare features
        X = self.prepare_features(df_design, is_training=False)
        
        predictions = {}
        for metric in self.target_metrics:
            if metric in self.models:
                # Scale features
                X_scaled = self.scalers[metric].transform(X)
                
                # Make prediction
                pred = self.models[metric].predict(X_scaled)[0]
                predictions[metric] = pred
        
        return predictions


class CircuitOptimizer:
    """
    Optimizes circuit designs using genetic algorithm approach
    """
    
    def __init__(self, predictor, design_space):
        self.predictor = predictor
        self.design_space = design_space
        
    def create_random_design(self):
        """Create a random circuit design within the design space"""
        design = {}
        for param, options in self.design_space.items():
            if isinstance(options, list):
                design[param] = np.random.choice(options)
            elif isinstance(options, tuple) and len(options) == 2:
                # Numerical range (min, max)
                design[param] = np.random.uniform(options[0], options[1])
        
        # Add default values for features that might be missing
        default_values = {
            'promoter_copy_number_effect': 1.0,
            'env_pH_variation': 1.0,
            'env_temperature_variation': 1.0,
            'env_total_stress': 0.2,
            'sim_duration': 300.0,
            'sim_random_seed': np.random.randint(0, 1000),
            'performance_class': 'medium',
            'noise_tolerance': 'medium',
            'sim_method': 'numerical_ode',
            'rbs_temperature_sensitivity': 0.2,
            'reporter_pH_sensitivity': 0.2,
            'terminator_type': 'strong'
        }
        
        for param, value in default_values.items():
            if param not in design:
                design[param] = value
                
        return design
    
    def mutate_design(self, design, mutation_rate=0.3):
        """Mutate a circuit design"""
        mutated = design.copy()
        
        for param in design:
            if np.random.random() < mutation_rate:
                # Only mutate if parameter is in design space
                if param in self.design_space:
                    options = self.design_space[param]
                    if isinstance(options, list):
                        mutated[param] = np.random.choice(options)
                    elif isinstance(options, tuple):
                        mutated[param] = np.random.uniform(options[0], options[1])
                # If parameter not in design space, keep original value
        
        return mutated
    
    def crossover_designs(self, parent1, parent2):
        """Create offspring from two parent designs"""
        offspring = {}
        
        # Get all unique parameters from both parents
        all_params = set(parent1.keys()) | set(parent2.keys())
        
        for param in all_params:
            if np.random.random() < 0.5:
                # Take from parent1 if available, otherwise from parent2
                if param in parent1:
                    offspring[param] = parent1[param]
                else:
                    offspring[param] = parent2[param]
            else:
                # Take from parent2 if available, otherwise from parent1
                if param in parent2:
                    offspring[param] = parent2[param]
                else:
                    offspring[param] = parent1[param]
                    
        return offspring
    
    def calculate_fitness(self, predictions, objectives):
        """
        Calculate fitness score based on multiple objectives
        
        Args:
            predictions: dict of predicted performance metrics
            objectives: dict of objective weights and targets
        """
        fitness = 0
        
        for metric, config in objectives.items():
            if metric in predictions:
                value = predictions[metric]
                target = config.get('target', 0)
                weight = config.get('weight', 1)
                maximize = config.get('maximize', True)
                
                if maximize:
                    # Higher is better
                    score = max(0, value - target)
                else:
                    # Lower is better
                    score = max(0, target - value)
                
                fitness += weight * score
        
        return fitness
    
    def optimize(self, objectives, population_size=50, generations=100):
        """
        Optimize circuit design using genetic algorithm
        
        Args:
            objectives: dict defining optimization objectives
            population_size: number of designs in population
            generations: number of optimization iterations
        """
        print(f"Optimizing circuit design over {generations} generations...")
        
        # Initialize population
        population = [self.create_random_design() for _ in range(population_size)]
        
        best_fitness_history = []
        best_design = None
        best_fitness = -np.inf
        
        for generation in range(generations):
            # Evaluate fitness for all designs
            fitness_scores = []
            
            for design in population:
                predictions = self.predictor.predict(design)
                fitness = self.calculate_fitness(predictions, objectives)
                fitness_scores.append(fitness)
                
                # Track best design
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_design = design.copy()
            
            best_fitness_history.append(best_fitness)
            
            if generation % 10 == 0:
                avg_fitness = np.mean(fitness_scores)
                print(f"Generation {generation}: Best fitness = {best_fitness:.3f}, Avg fitness = {avg_fitness:.3f}")
            
            # Selection and reproduction
            fitness_scores = np.array(fitness_scores)
            # Prevent negative fitness from causing issues
            fitness_scores = fitness_scores - np.min(fitness_scores) + 1e-6
            
            # Tournament selection
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament_size = 3
                tournament_indices = np.random.choice(len(population), tournament_size)
                tournament_fitness = fitness_scores[tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                
                # Mutate winner
                new_design = self.mutate_design(population[winner_idx])
                new_population.append(new_design)
            
            # Add some crossover
            for i in range(population_size // 4):
                parent1_idx = np.random.choice(len(population))
                parent2_idx = np.random.choice(len(population))
                offspring = self.crossover_designs(population[parent1_idx], population[parent2_idx])
                new_population[i] = offspring
            
            population = new_population
        
        print(f"Optimization complete! Best fitness: {best_fitness:.3f}")
        
        return {
            'best_design': best_design,
            'best_fitness': best_fitness,
            'fitness_history': best_fitness_history,
            'best_predictions': self.predictor.predict(best_design)
        }


class BiosensorDesignAI:
    """
    Main class for biosensor circuit design AI system
    """
    
    def __init__(self):
        self.predictor = CircuitDesignPredictor()
        self.optimizer = None
        
        # Default design space
        self.design_space = {
            'circuit_promoter': ['pTet', 'pLac', 'pBAD', 'pLux'],
            'circuit_rbs': ['RBS1', 'RBS2', 'RBS3'],
            'circuit_coding_sequence': ['GFP', 'RFP', 'YFP', 'CFP'],
            'circuit_terminator': ['T1', 'T2', 'T3', 'T7'],
            'circuit_architecture': ['simple', 'feedback', 'cascade'],
            'circuit_copy_number': ['low_copy_plasmid', 'medium_copy_plasmid', 'high_copy_plasmid'],
            'promoter_strength': (0.1, 1.0),
            'promoter_type': ['inducible', 'constitutive'],
            'promoter_leakiness': (0.01, 0.1),
            'promoter_noise_sensitivity': (0.1, 0.5),
            'promoter_copy_number_effect': (0.8, 1.2),
            'rbs_efficiency': (0.3, 1.0),
            'rbs_strength_class': ['weak', 'medium', 'strong'],
            'rbs_temperature_sensitivity': (0.1, 0.3),
            'reporter_signal_strength': (0.5, 1.0),
            'reporter_gene_type': ['fluorescent_protein', 'enzyme'],
            'reporter_pH_sensitivity': (0.1, 0.3),
            'reporter_maturation_time': (30, 120),
            'terminator_efficiency': (0.8, 1.0),
            'terminator_stability': (0.8, 1.0),
            'terminator_type': ['weak', 'medium', 'strong', 'very_strong'],
            'env_pH_variation': (0.5, 2.0),
            'env_temperature_variation': (0.8, 2.5),
            'env_pH_actual': (6.5, 8.0),
            'env_temperature_actual': (30, 42),
            'env_immune_signal_level': (0.0, 0.5),
            'env_metabolic_flux_noise': (0.05, 0.3),
            'env_ionic_strength': (0.1, 0.3),
            'env_oxidative_stress': (0.05, 0.2),
            'env_total_stress': (0.1, 0.5),
            'sim_transcription_rate': (2.0, 4.0),
            'sim_translation_rate': (0.6, 0.8),
            'sim_protein_degradation_rate': (0.01, 0.02),
            'sim_mRNA_degradation_rate': (0.04, 0.06),
            'performance_class': ['low', 'medium', 'high'],
            'noise_tolerance': ['low', 'medium', 'high']
        }
    
    def train_from_data(self, df):
        """Train the AI system from experimental/simulation data"""
        print("Training biosensor design AI...")
        self.predictor.train(df)
        self.optimizer = CircuitOptimizer(self.predictor, self.design_space)
        print("Training complete!")
    
    def predict_performance(self, circuit_design):
        """Predict performance for a given circuit design"""
        return self.predictor.predict(circuit_design)
    
    def design_robust_circuit(self, target_performance, constraints=None):
        """
        Design a robust biosensor circuit to meet target performance
        
        Args:
            target_performance: dict with target metrics and objectives
            constraints: dict with design constraints (optional)
        """
        if self.optimizer is None:
            raise ValueError("AI system must be trained first using train_from_data()")
        
        # Define optimization objectives
        objectives = {}
        
        # Default objectives for robust design
        default_objectives = {
            'perf_snr': {'target': 100, 'weight': 2.0, 'maximize': True},
            'perf_false_positive_rate': {'target': 0.05, 'weight': 1.5, 'maximize': False},
            'perf_response_time': {'target': 300, 'weight': 1.0, 'maximize': False},
            'robustness_score': {'target': 0, 'weight': 1.5, 'maximize': True},
            'perf_signal_strength': {'target': 1000, 'weight': 1.0, 'maximize': True}
        }
        
        # Update with user-specified targets
        for metric, config in default_objectives.items():
            if metric in target_performance:
                config.update(target_performance[metric])
            objectives[metric] = config
        
        # Run optimization
        result = self.optimizer.optimize(objectives, population_size=50, generations=100)
        
        return result
    
    def analyze_design_tradeoffs(self, designs):
        """Analyze trade-offs between multiple design options"""
        results = []
        
        for i, design in enumerate(designs):
            predictions = self.predict_performance(design)
            
            analysis = {
                'design_id': i,
                'design': design,
                'predictions': predictions,
                'snr_score': predictions.get('perf_snr', 0),
                'robustness_score': predictions.get('robustness_score', 0),
                'false_positive_rate': predictions.get('perf_false_positive_rate', 1),
                'response_time': predictions.get('perf_response_time', 1000),
                'signal_strength': predictions.get('perf_signal_strength', 0)
            }
            
            results.append(analysis)
        
        return sorted(results, key=lambda x: x['snr_score'], reverse=True)


# Example usage and demonstration
def demonstrate_biosensor_ai():
    """Demonstrate the biosensor design AI system"""
    
    # Create sample data (in practice, this would be your experimental data)
    np.random.seed(42)
    
    sample_data = []
    design_options = {
        'circuit_promoter': ['pTet', 'pLac', 'pBAD'],
        'circuit_rbs': ['RBS1', 'RBS2', 'RBS3'],
        'circuit_coding_sequence': ['GFP', 'RFP', 'YFP'],
        'circuit_terminator': ['T1', 'T2', 'T3'],
        'circuit_architecture': ['simple', 'feedback'],
        'circuit_copy_number': ['low_copy_plasmid', 'high_copy_plasmid']
    }
    
    for _ in range(200):  # Generate synthetic data
        design = {}
        for param, options in design_options.items():
            design[param] = np.random.choice(options)
        
        # Add numerical parameters
        design.update({
            'promoter_strength': np.random.uniform(0.3, 1.0),
            'promoter_type': np.random.choice(['inducible', 'constitutive']),
            'promoter_leakiness': np.random.uniform(0.01, 0.08),
            'promoter_noise_sensitivity': np.random.uniform(0.1, 0.4),
            'promoter_copy_number_effect': 1.0,
            'rbs_efficiency': np.random.uniform(0.4, 0.9),
            'rbs_strength_class': np.random.choice(['weak', 'medium', 'strong']),
            'rbs_temperature_sensitivity': np.random.uniform(0.1, 0.3),
            'reporter_signal_strength': np.random.uniform(0.6, 1.0),
            'reporter_gene_type': 'fluorescent_protein',
            'reporter_pH_sensitivity': np.random.uniform(0.1, 0.3),
            'reporter_maturation_time': np.random.uniform(40, 100),
            'terminator_efficiency': np.random.uniform(0.85, 1.0),
            'terminator_stability': np.random.uniform(0.85, 1.0),
            'terminator_type': 'strong',
            'env_pH_variation': np.random.uniform(0.5, 1.5),
            'env_temperature_variation': np.random.uniform(0.8, 2.0),
            'env_immune_signal_level': np.random.uniform(0.1, 0.4),
            'env_metabolic_flux_noise': np.random.uniform(0.05, 0.25),
            'env_ionic_strength': np.random.uniform(0.1, 0.25),
            'env_oxidative_stress': np.random.uniform(0.05, 0.15),
            'env_pH_actual': np.random.uniform(7.0, 7.8),
            'env_temperature_actual': np.random.uniform(35, 40),
            'env_total_stress': np.random.uniform(0.1, 0.4),
            'sim_duration': 300.0,
            'sim_method': 'numerical_ode',
            'sim_random_seed': np.random.randint(0, 1000),
            'sim_transcription_rate': np.random.uniform(2.0, 4.0),
            'sim_translation_rate': np.random.uniform(0.6, 0.8),
            'sim_protein_degradation_rate': np.random.uniform(0.01, 0.02),
            'sim_mRNA_degradation_rate': np.random.uniform(0.04, 0.06),
            'performance_class': np.random.choice(['low', 'medium', 'high']),
            'noise_tolerance': np.random.choice(['low', 'medium', 'high'])
        })
        
        # Simulate performance metrics (simplified relationships)
        snr_base = design['promoter_strength'] * design['rbs_efficiency'] * design['reporter_signal_strength'] * 150
        noise_factor = design['env_metabolic_flux_noise'] + design['promoter_leakiness']
        
        design['perf_signal_strength'] = snr_base * (1 + np.random.normal(0, 0.1))
        design['perf_noise_level'] = noise_factor * 100 * (1 + np.random.normal(0, 0.2))
        design['perf_snr'] = design['perf_signal_strength'] / max(design['perf_noise_level'], 1)
        design['perf_false_positive_rate'] = design['promoter_leakiness'] * (1 + np.random.normal(0, 0.3))
        design['perf_response_time'] = design['reporter_maturation_time'] * (2 + np.random.normal(0, 0.5))
        design['perf_steady_state_protein'] = design['perf_signal_strength'] * 2
        design['perf_background_noise'] = design['perf_noise_level'] * 0.3
        design['robustness_score'] = design['perf_snr'] / 100 - design['env_total_stress']
        
        sample_data.append(design)
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Initialize and train AI system
    ai_system = BiosensorDesignAI()
    ai_system.train_from_data(df)
    
    print("\n" + "="*60)
    print("BIOSENSOR DESIGN AI DEMONSTRATION")
    print("="*60)
    
    # Example 1: Predict performance for a specific design
    print("\n1. PERFORMANCE PREDICTION")
    print("-" * 30)
    
    test_design = {
        'circuit_promoter': 'pTet',
        'circuit_rbs': 'RBS2',
        'circuit_coding_sequence': 'GFP',
        'circuit_terminator': 'T1',
        'circuit_architecture': 'simple',
        'circuit_copy_number': 'high_copy_plasmid',
        'promoter_strength': 0.8,
        'promoter_type': 'inducible',
        'promoter_leakiness': 0.02,
        'promoter_noise_sensitivity': 0.15,
        'promoter_copy_number_effect': 1.0,
        'rbs_efficiency': 0.7,
        'rbs_strength_class': 'strong',
        'rbs_temperature_sensitivity': 0.2,
        'reporter_signal_strength': 0.9,
        'reporter_gene_type': 'fluorescent_protein',
        'reporter_pH_sensitivity': 0.2,
        'reporter_maturation_time': 60,
        'terminator_efficiency': 0.95,
        'terminator_stability': 0.9,
        'terminator_type': 'strong',
        'env_pH_variation': 1.0,
        'env_temperature_variation': 1.2,
        'env_pH_actual': 7.4,
        'env_temperature_actual': 37,
        'env_immune_signal_level': 0.2,
        'env_metabolic_flux_noise': 0.1,
        'env_ionic_strength': 0.15,
        'env_oxidative_stress': 0.08,
        'env_total_stress': 0.25,
        'sim_duration': 300.0,
        'sim_method': 'numerical_ode',
        'sim_random_seed': 42,
        'sim_transcription_rate': 3.0,
        'sim_translation_rate': 0.7,
        'sim_protein_degradation_rate': 0.015,
        'sim_mRNA_degradation_rate': 0.05,
        'performance_class': 'high',
        'noise_tolerance': 'medium'
    }
    
    predictions = ai_system.predict_performance(test_design)
    
    print("Circuit Design:")
    for key, value in test_design.items():
        print(f"  {key}: {value}")
    
    print("\nPredicted Performance:")
    for metric, value in predictions.items():
        print(f"  {metric}: {value:.3f}")
    
    # Example 2: Optimize circuit design
    print("\n\n2. CIRCUIT OPTIMIZATION")
    print("-" * 30)
    
    target_performance = {
        'perf_snr': {'target': 120, 'weight': 2.0, 'maximize': True},
        'perf_false_positive_rate': {'target': 0.03, 'weight': 1.5, 'maximize': False},
        'perf_response_time': {'target': 200, 'weight': 1.0, 'maximize': False}
    }
    
    optimization_result = ai_system.design_robust_circuit(target_performance)
    
    print("Optimization Results:")
    print(f"Best fitness score: {optimization_result['best_fitness']:.3f}")
    print("\nOptimal Design:")
    for key, value in optimization_result['best_design'].items():
        print(f"  {key}: {value}")
    
    print("\nPredicted Performance:")
    best_predictions = optimization_result['best_predictions']
    for metric, value in best_predictions.items():
        print(f"  {metric}: {value:.3f}")
    
    # Example 3: Analyze design trade-offs
    print("\n\n3. DESIGN TRADE-OFF ANALYSIS")
    print("-" * 30)
    
    # Create multiple design candidates
    candidate_designs = [
        optimization_result['best_design'],  # Optimized design
        test_design,  # Manual design
        ai_system.optimizer.create_random_design(),  # Random design 1
        ai_system.optimizer.create_random_design()   # Random design 2
    ]
    
    tradeoff_analysis = ai_system.analyze_design_tradeoffs(candidate_designs)
    
    print("Design Comparison (ranked by SNR):")
    for i, result in enumerate(tradeoff_analysis):
        print(f"\nRank {i+1} (Design {result['design_id']}):")
        print(f"  SNR: {result['snr_score']:.1f}")
        print(f"  False Positive Rate: {result['false_positive_rate']:.3f}")
        print(f"  Response Time: {result['response_time']:.1f}s")
        print(f"  Robustness Score: {result['robustness_score']:.3f}")
        print(f"  Signal Strength: {result['signal_strength']:.1f}")
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    
    return ai_system

# Run demonstration
if __name__ == "__main__":
    ai_system = demonstrate_biosensor_ai()