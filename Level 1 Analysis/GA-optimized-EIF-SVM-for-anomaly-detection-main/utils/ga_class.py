import numpy as np
import random
from eif.eif_class import iForest

def init_population(pop_size, feature_dim, n_obs):
    population = []
    for _ in range(pop_size):
        chromosome = {
            'ntrees': random.randint(50, 300),
            'sample_size': random.randint(int(n_obs*0.5), n_obs),
            'contamination': round(random.uniform(0.01, 0.2), 3),
            'exlevel': random.randint(0, feature_dim-1)
        }
        population.append(chromosome)
    return population

def fitness(chromosome, X, labels=None):
    forest = iForest(X, ntrees=chromosome['ntrees'], 
                     sample_size=chromosome['sample_size'], 
                     ExtensionLevel=chromosome['exlevel'])
    scores = forest.compute_paths(X)
    
    if labels is not None:
        from sklearn.metrics import f1_score
        preds = (scores > np.percentile(scores, 100*chromosome['contamination'])).astype(int)
        return f1_score(labels, preds)
    else:
        return np.var(scores)

def select_parents(population, fitnesses, num_parents):
    idx = np.argsort(fitnesses)[-num_parents:]
    return [population[i] for i in idx]

def crossover(parent1, parent2):
    child = {}
    for key in parent1.keys():
        child[key] = random.choice([parent1[key], parent2[key]])
    return child

def mutate(chromosome, feature_dim, n_obs, mutation_rate=0.2):
    if random.random() < mutation_rate:
        key = random.choice(list(chromosome.keys()))
        if key == 'ntrees':
            chromosome[key] = random.randint(50, 300)
        elif key == 'sample_size':
            chromosome[key] = random.randint(int(n_obs*0.5), n_obs)
        elif key == 'contamination':
            chromosome[key] = round(random.uniform(0.01, 0.2), 3)
        elif key == 'exlevel':
            chromosome[key] = random.randint(0, feature_dim-1)
    return chromosome

def ga_eif(X, labels=None, pop_size=20, generations=10):
    n_obs, feature_dim = X.shape
    population = init_population(pop_size, feature_dim, n_obs)
    
    best_overall = None
    best_fitness_overall = -np.inf
    
    for gen in range(generations):
        fitnesses = [fitness(ch, X, labels) for ch in population]
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fitness = fitnesses[gen_best_idx]
        gen_best_chrom = population[gen_best_idx]
        
        print(f"Gen {gen} best fitness: {gen_best_fitness:.4f}")
        print(f"Gen {gen} best params: {gen_best_chrom}")
        
        if gen_best_fitness > best_fitness_overall:
            best_fitness_overall = gen_best_fitness
            best_overall = gen_best_chrom.copy()
        
        parents = select_parents(population, fitnesses, num_parents=pop_size//2)
        next_pop = parents.copy()
        
        while len(next_pop) < pop_size:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = mutate(child, feature_dim, n_obs)
            next_pop.append(child)
        
        population = next_pop
    
    print("\nGA finished.")
    print("Best overall fitness:", best_fitness_overall)
    print("Best overall EIF params:", best_overall)
    return best_overall

#X defined as orderbook features. For more information visit data/ folder. 
#best_chrom = ga_eif(X, pop_size=20, generations=10)
