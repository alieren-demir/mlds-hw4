import matplotlib
matplotlib.use('Agg') 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from bayes_opt import BayesianOptimization

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_data():
    print("Loading Federated EMNIST data...")
    try:
        train_data_obj = np.load('train_data.npy', allow_pickle=True)
        test_data_obj = np.load('test_data.npy', allow_pickle=True)
    except FileNotFoundError:
        print("Error: 'train_data.npy' or 'test_data.npy' not found.")
        return None, None, None, None, None, None

    def aggregate_clients(data_obj):
        all_images = []
        all_labels = []
        if data_obj.ndim == 0: 
            data_obj = data_obj.item()
            
        for client_data in data_obj:
            all_images.append(client_data['images'])
            all_labels.append(client_data['labels'])
            
        X_agg = np.concatenate(all_images, axis=0)
        y_agg = np.concatenate(all_labels, axis=0)
        return X_agg, y_agg

    print("Aggregating clients...")
    X_train_full, y_train_full = aggregate_clients(train_data_obj)
    X_test_full, y_test_full = aggregate_clients(test_data_obj)

    # --- FILTER FOR DIGITS (0-9) ONLY ---
    print("Filtering dataset for digits (labels 0-9) only...")
    train_mask = y_train_full < 10
    X_train_full = X_train_full[train_mask]
    y_train_full = y_train_full[train_mask]
    
    test_mask = y_test_full < 10
    X_test_full = X_test_full[test_mask]
    y_test_full = y_test_full[test_mask]

    # --- SHAPE FIX & NORMALIZATION ---
    if len(X_train_full.shape) == 3:
        X_train_full = X_train_full.reshape(X_train_full.shape[0], -1)
        X_test_full = X_test_full.reshape(X_test_full.shape[0], -1)

    X_train_full = X_train_full.astype(np.float32)
    X_test_full = X_test_full.astype(np.float32)

    if X_train_full.max() > 1.0:
        X_train_full /= 255.0
        X_test_full /= 255.0
    else:
        print("Data appears already normalized. Skipping division.")

    y_train_full = y_train_full.astype(np.int64)
    y_test_full = y_test_full.astype(np.int64)

    X_train_tensor = torch.tensor(X_train_full)
    y_train_tensor = torch.tensor(y_train_full).long()
    X_test_tensor = torch.tensor(X_test_full)
    y_test_tensor = torch.tensor(y_test_full).long()

    # Split 80% Train, 20% Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_tensor, y_train_tensor, test_size=0.2, random_state=42, stratify=y_train_tensor
    )
    
    print(f"Data Loaded: Train {X_train.shape}, Val {X_val.shape}, Test {X_test_tensor.shape}")
    return X_train, y_train, X_val, y_val, X_test_tensor, y_test_tensor

# =============================================================================
# NEURAL NETWORK & TRAINING LOOP
# =============================================================================

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation_name):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512) 
        
        if activation_name == 'relu':
            self.activation = nn.ReLU()
        elif activation_name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_name == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
            
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out

def train_and_evaluate(batch_size, activation_name, X_train, y_train, X_val, y_val, epochs=5, verbose=False):
    batch_size = int(batch_size)
    batch_size = max(16, min(1024, batch_size))
    
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    input_dim = X_train.shape[1]
    model = SimpleNet(input_dim, 512, 10, activation_name)
    
    criterion = nn.CrossEntropyLoss()
    
    # --- LOWER BASE LR ---
    ref_batch = 256
    scaled_lr = 0.05 * (batch_size / ref_batch)
    
    scaled_lr = max(0.001, min(0.2, scaled_lr))
    
    optimizer = optim.SGD(model.parameters(), lr=scaled_lr, momentum=0.9)
    
    # --- SCHEDULER ---
    # Decay LR by factor of 0.5 every 25 epochs.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    
    history_f1 = []
    
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        if verbose:
            model.eval()
            with torch.no_grad():
                train_preds = torch.argmax(model(X_train), dim=1)
                train_f1 = f1_score(y_train, train_preds, average='macro')
                history_f1.append(train_f1)
            model.train()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_preds = torch.argmax(val_outputs, dim=1)
        fitness = f1_score(y_val, val_preds, average='macro')
        
    if verbose:
        return fitness, history_f1, model
    return fitness

# =============================================================================
# GENETIC ALGORITHM
# =============================================================================

class GeneticAlgorithm:
    def __init__(self, X_train, y_train, X_val, y_val, pop_size=20, generations=20, mutation_rate=0.1, search_epochs=10):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.search_epochs = search_epochs
        
        self.chromosome_len = 12 
        self.population = [] 
        self.history_avg = []
        self.history_best = []

    def decode(self, chromosome):
        b_val = int("".join(map(str, chromosome[:10])), 2)
        batch_size = 16 + int(b_val * ((1024 - 16) / 1023))
        
        a_val = int("".join(map(str, chromosome[10:])), 2)
        activations = ['relu', 'sigmoid', 'tanh', 'relu'] 
        return batch_size, activations[a_val]

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            genes = [random.randint(0, 1) for _ in range(self.chromosome_len)]
            self.population.append({'genes': genes, 'fitness': None, 'age': 0})

    def roulette_selection(self):
        total_fit = sum(ind['fitness'] for ind in self.population)
        if total_fit == 0: return random.choice(self.population)
        pick = random.uniform(0, total_fit)
        current = 0
        for ind in self.population:
            current += ind['fitness']
            if current > pick: return ind
        return self.population[-1]

    def run(self):
        self.initialize_population()
        
        for ind in self.population:
            if ind['fitness'] is None:
                b, a = self.decode(ind['genes'])
                ind['fitness'] = train_and_evaluate(b, a, self.X_train, self.y_train, self.X_val, self.y_val, epochs=self.search_epochs)

        for gen in range(self.generations):
            best_individual = max(self.population, key=lambda x: x['fitness'])
            elite_copy = best_individual.copy() 
            elite_copy['age'] = 0 

            fits = [ind['fitness'] for ind in self.population]
            self.history_avg.append(sum(fits)/len(fits))
            self.history_best.append(max(fits))
            print(f"  > GA Gen {gen+1}: Max Fitness {max(fits):.4f} (Avg: {sum(fits)/len(fits):.4f})")

            # Increase Age
            for ind in self.population: ind['age'] += 1
            
            # Offspring
            num_offspring = int(self.pop_size * 0.4)
            offsprings = []
            while len(offsprings) < num_offspring:
                p1 = self.roulette_selection()
                p2 = self.roulette_selection()
                
                point = random.randint(1, self.chromosome_len-1)
                c1_genes = p1['genes'][:point] + p2['genes'][point:]
                c2_genes = p2['genes'][:point] + p1['genes'][point:]
                
                for i in range(len(c1_genes)):
                    if random.random() < self.mutation_rate: c1_genes[i] = 1 - c1_genes[i]
                for i in range(len(c2_genes)):
                    if random.random() < self.mutation_rate: c2_genes[i] = 1 - c2_genes[i]

                offsprings.append({'genes': c1_genes, 'fitness': None, 'age': 0})
                if len(offsprings) < num_offspring:
                    offsprings.append({'genes': c2_genes, 'fitness': None, 'age': 0})
            
            for ind in offsprings:
                b, a = self.decode(ind['genes'])
                ind['fitness'] = train_and_evaluate(b, a, self.X_train, self.y_train, self.X_val, self.y_val, epochs=self.search_epochs)

            # Survivor Selection (Age-based)
            self.population.sort(key=lambda x: x['age'], reverse=True)
            self.population = self.population[num_offspring:] + offsprings
            
            current_best = max(self.population, key=lambda x: x['fitness'])
            if current_best['fitness'] < elite_copy['fitness']:
                self.population.sort(key=lambda x: x['fitness']) 
                self.population[0] = elite_copy

        best_ind = max(self.population, key=lambda x: x['fitness'])
        return best_ind, self.history_avg, self.history_best

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("--- AUTO HYPERPARAMETER TUNING ---")
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    if X_train is None: exit()

    POP_SIZE = 50        
    GENERATIONS = 30     
    SEARCH_EPOCHS = 25
    FINAL_EPOCHS = 100 

    # ---------------------------------------------------------
    # PART 1: GENETIC ALGORITHM
    # ---------------------------------------------------------
    print("\n--- PART 1: GENETIC ALGORITHM ---")
    print(f"Settings: Pop={POP_SIZE}, Gens={GENERATIONS}, Search Epochs={SEARCH_EPOCHS}")
    
    ga = GeneticAlgorithm(
        X_train, y_train, X_val, y_val, 
        pop_size=POP_SIZE, 
        generations=GENERATIONS,
        search_epochs=SEARCH_EPOCHS
    )
    
    best_ind_ga, ga_avg_hist, ga_best_hist = ga.run()
    
    ga_batch, ga_act = ga.decode(best_ind_ga['genes'])
    print(f"GA Result -> Batch: {ga_batch}, Activation: {ga_act}")

    # Plot Stats
    plt.figure(figsize=(10, 4))
    gens_ran = len(ga_avg_hist)
    plt.plot(range(1, gens_ran + 1), ga_avg_hist, label='Avg Fitness')
    plt.plot(range(1, gens_ran + 1), ga_best_hist, label='Highest Fitness')
    plt.title('GA: Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('ga_fitness_history.png')
    plt.close()
    print("Saved plot: ga_fitness_history.png")

    # Final GA Training
    print("Merging Train and Validation sets for final evaluation...")
    X_final_train = torch.cat((X_train, X_val), 0)
    y_final_train = torch.cat((y_train, y_val), 0)

    print(f"Training final GA model ({FINAL_EPOCHS} Epochs)...")
    ga_final_f1, ga_train_hist, ga_model = train_and_evaluate(
        ga_batch, ga_act, X_final_train, y_final_train, X_val, y_val, epochs=FINAL_EPOCHS, verbose=True
    )
    
    ga_model.eval()
    with torch.no_grad():
        test_preds = torch.argmax(ga_model(X_test), dim=1)
        ga_test_f1 = f1_score(y_test, test_preds, average='macro')
    
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, FINAL_EPOCHS + 1), ga_train_hist, marker='o', label='Training F1')
    plt.title(f'GA Final Training (Batch={ga_batch}, Act={ga_act})')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.savefig('ga_training_curve.png')
    plt.close()
    print("Saved plot: ga_training_curve.png")

    # ---------------------------------------------------------
    # PART 2: BAYESIAN OPTIMIZATION
    # ---------------------------------------------------------
    print("\n--- PART 2: BAYESIAN OPTIMIZATION ---")

    def bo_objective(batch_size, activation_code):
        b_size = int(batch_size)
        act_idx = min(int(activation_code), 2)
        act_name = ['relu', 'sigmoid', 'tanh'][act_idx]
        return train_and_evaluate(b_size, act_name, X_train, y_train, X_val, y_val, epochs=SEARCH_EPOCHS, verbose=False)

    pbounds = {'batch_size': (16, 1024), 'activation_code': (0, 2.999)}
    
    print(f"Settings: 100 Iterations, Search Epochs={SEARCH_EPOCHS}")
    optimizer = BayesianOptimization(f=bo_objective, pbounds=pbounds, random_state=SEED, verbose=2)
    optimizer.maximize(init_points=20, n_iter=80)

    bo_best_params = optimizer.max['params']
    bo_batch = int(bo_best_params['batch_size'])
    bo_act_idx = min(int(bo_best_params['activation_code']), 2)
    bo_act = ['relu', 'sigmoid', 'tanh'][bo_act_idx]
    
    print(f"BO Result -> Batch: {bo_batch}, Activation: {bo_act}")

    # Train Final BO Model
    print(f"Training final BO model ({FINAL_EPOCHS} Epochs)...")
    bo_final_f1, bo_train_hist, bo_model = train_and_evaluate(
        bo_batch, bo_act, X_final_train, y_final_train, X_val, y_val, epochs=FINAL_EPOCHS, verbose=True
    )

    bo_model.eval()
    with torch.no_grad():
        test_preds = torch.argmax(bo_model(X_test), dim=1)
        bo_test_f1 = f1_score(y_test, test_preds, average='macro')

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, FINAL_EPOCHS + 1), bo_train_hist, marker='o', color='green', label='Training F1')
    plt.title(f'BO Final Training (Batch={bo_batch}, Act={bo_act})')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.savefig('bo_training_curve.png')
    plt.close()
    print("Saved plot: bo_training_curve.png")

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("FINAL RESULTS SUMMARY (SERVER/STABLE)")
    print("="*40)
    print(f"{'Metric':<20} | {'Genetic Algo':<15} | {'Bayesian Opt':<15}")
    print("-" * 56)
    print(f"{'Best Batch Size':<20} | {ga_batch:<15} | {bo_batch:<15}")
    print(f"{'Best Activation':<20} | {ga_act:<15} | {bo_act:<15}")
    print(f"{'Test F1 Score':<20} | {ga_test_f1:.4f}           | {bo_test_f1:.4f}")
    print("="*40)
    print("\nExecution complete.")
