import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# función fitness
def fitness(individual, food_table, daily_requirements):
    nutrients = [
        "energy",
        "protein",
        "fat",
        "calcium",
        "iron",
        "vitamin_a",
        "thiamine",
        "riboflavin",
        "niacin",
        "folate",
        "vitamin_c",
    ]
    total_nutrients = {nutrient: 0 for nutrient in nutrients}
    total_weight = 0

    # Calcula el total de nutrientes y peso
    for food_index, food_quantity in individual:
        food = food_table[food_index]
        total_weight += food_quantity * 100
        for i, nutrient in enumerate(nutrients, 2):
            total_nutrients[nutrient] += food[i] * food_quantity

    # Define las metas para cada nutriente y la penalización
    weight_limit = 3000
    excess_penalty = 10
    margin = 1.2
    nutrient_diff = 0

    # Ajuste de ponderación para nutrientes especificados
    importance_factor = 10

    # Calcula la distancia a los valores ideales de cada nutriente
    for nutrient in nutrients:
        ideal_nutrient = daily_requirements[nutrient]
        if ideal_nutrient != 0:
            nutrient_diff_current = (
                total_nutrients[nutrient] - ideal_nutrient
            ) / ideal_nutrient
            nutrient_diff += abs(nutrient_diff_current) * importance_factor

            if total_nutrients[nutrient] > ideal_nutrient * margin:
                nutrient_diff += (nutrient_diff_current**2) * excess_penalty

    # Penalizar fuertemente si el peso total supera el límite
    if total_weight > weight_limit:
        nutrient_diff += (total_weight - weight_limit) * excess_penalty

    return 10 / (nutrient_diff + 1)


class GeneticAlgorithm:
    def __init__(
        self,
        food_table,
        daily_requirements,
        population_size,
        num_generations,
        crossover_prob,
        mutation_prob,
        tournament_size,
    ):
        self.food_table = food_table
        self.daily_requirements = daily_requirements
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.best_fitnesses = []
        self.avg_fitnesses = []
        self.worst_fitnesses = []

        self.population = []

        # Define las proporciones de cada tipo de individuo
        single_food_ratio = 0.2  # Porcentaje de individuos con un solo tipo de alimento
        balanced_ratio = 0.4  # Porcentaje de individuos equilibrados

        # Calcula el número de individuos de cada tipo
        num_single_food = int(self.population_size * single_food_ratio)
        num_balanced = int(self.population_size * balanced_ratio)
        num_random_combinations = self.population_size - num_single_food - num_balanced

        # Agregar individuos con un solo tipo de alimento
        for _ in range(num_single_food):
            food_index = random.randint(0, len(food_table) - 1)
            self.population.append([(food_index, random.randint(1, 4))])

        # Añadir individuos equilibrados entre todos los tipos de alimentos
        for _ in range(num_balanced):
            balanced_individual = [
                (food_index, random.randint(1, 2))
                for food_index in range(len(food_table))
            ]
            self.population.append(balanced_individual)

        # Rellenar el resto de la población con combinaciones aleatorias
        for _ in range(num_random_combinations):
            self.population.append(
                [
                    (food_index, random.randint(1, 3))
                    for food_index in random.sample(
                        range(len(food_table)), random.randint(1, len(food_table))
                    )
                ]
            )

    def tournament_selection(self):
        # Selecciona 'tournament_size' individuos aleatorios de la población
        tournament = random.sample(self.population, self.tournament_size)
        # Evalúa la aptitud de cada individuo
        fitnesses = [
            fitness(ind, self.food_table, self.daily_requirements) for ind in tournament
        ]
        # Devuelve el individuo con mayor aptitud
        return tournament[np.argmax(fitnesses)]

    def select_parents(self):
        # Utiliza la selección de torneo para seleccionar los padres
        return self.tournament_selection(), self.tournament_selection()

    def ordered_crossover(self, parents):
        if random.random() < self.crossover_prob:
            size1, size2 = len(parents[0]), len(parents[1])

            if size1 == 1 and size2 == 1:
                return parents[0] if random.random() < 0.5 else parents[1]
            elif size1 == 1:  # Si el primer padre tiene solo un gen
                child = [parents[0][0]] + [
                    item for item in parents[1] if item != parents[0][0]
                ]
                return child
            elif size2 == 1:  # Si el segundo padre tiene solo un gen
                child = [parents[1][0]] + [
                    item for item in parents[0] if item != parents[1][0]
                ]
                return child
            else:
                # Genera dos índices aleatorios
                gene1 = random.randint(0, min(size1, size2))
                gene2 = random.randint(0, min(size1, size2))
                # Ordena los índices
                startGene = min(gene1, gene2)
                endGene = max(gene1, gene2)
                # Obtiene las partes de los padres que van a intercambiarse
                child1 = parents[0][startGene:endGene]
                child2 = parents[1][startGene:endGene]

                # Guarda los genes de los padres que aún no se han añadido a los hijos
                remaining1 = [item for item in parents[0] if item not in child2]
                remaining2 = [item for item in parents[1] if item not in child1]

                # Completa los hijos con los genes restantes de los padres
                child1 = child1 + remaining2
                child2 = child2 + remaining1

                # Devuelve los hijos
                return child1 if random.random() < 0.5 else child2
        else:
            # Si no se hace cruce, se devuelve un progenitor
            return parents[0] if random.random() < 0.5 else parents[1]

    def mutate(self, individual):
        current_foods = [food[0] for food in individual]
        all_foods = set(range(len(self.food_table)))

        if len(individual) > 1:
            # Si todos los alimentos ya están en el individuo, no permitimos la mutación "add"
            possible_mutations = (
                ["change", "add", "remove"]
                if len(current_foods) != len(all_foods)
                else ["change", "remove"]
            )
        else:
            # Si el individuo sólo tiene un alimento, no permitimos la mutación "remove"
            possible_mutations = ["change", "add"]

        if random.random() < self.mutation_prob:
            mutation_type = random.choice(possible_mutations)

            if mutation_type == "change":
                # Selecciona un alimento aleatorio y cambia su cantidad
                food_index = random.choice(range(len(individual)))
                individual[food_index] = (
                    individual[food_index][0],
                    random.randint(1, 4),
                )
            elif mutation_type == "add":
                # Agrega un alimento aleatorio con una cantidad aleatoria, asegurándose de que no esté ya en la dieta
                available_foods = list(all_foods - set(current_foods))
                if available_foods:
                    food_index = random.choice(available_foods)
                    individual.append((food_index, random.randint(1, 4)))
            # Asegura que siempre haya al menos un alimento
            elif mutation_type == "remove" and len(individual) > 1:
                # Elimina un alimento aleatorio de la dieta
                food_index = random.choice(range(len(individual)))
                individual.pop(food_index)
        return individual

    def run(self):
        best_individual = None
        best_fitness = -float("inf")

        # Porcentaje de individuos elitistas
        elitism_ratio = 0.2
        num_elites = int(self.population_size * elitism_ratio)

        for gen in range(self.num_generations):
            new_population = []
            generation_fitnesses = []
            generation_individuals = list(
                zip(
                    self.population,
                    [
                        fitness(ind, self.food_table, self.daily_requirements)
                        for ind in self.population
                    ],
                )
            )

            # Ordenar los individuos en función de su aptitud
            sorted_individuals = sorted(
                generation_individuals, key=lambda x: x[1], reverse=True
            )

            # Añadir los mejores individuos a la nueva población (Elitismo)
            elites = [ind for ind, fit in sorted_individuals[:num_elites]]
            new_population.extend(elites)

            for _ in range(self.population_size - num_elites):
                parents = self.select_parents()
                child = self.ordered_crossover(parents)
                child = self.mutate(child)

                child_fitness = fitness(child, self.food_table, self.daily_requirements)
                # Almacenamos el fitness del niño en la lista
                generation_fitnesses.append(child_fitness)

                if child_fitness > best_fitness:
                    best_individual = child
                    best_fitness = child_fitness

                new_population.append(child)

            # Registramos el mejor, promedio y peor fitness de esta generación
            self.best_fitnesses.append(max(generation_fitnesses))
            self.avg_fitnesses.append(
                sum(generation_fitnesses) / len(generation_fitnesses)
            )
            self.worst_fitnesses.append(min(generation_fitnesses))

            self.population = new_population

        return best_individual, best_fitness


def plot_results(ga):
    generations = range(1, len(ga.best_fitnesses) + 1)

    plt.figure()
    plt.plot(generations, ga.best_fitnesses, label="Mejor Fitness")
    plt.plot(generations, ga.avg_fitnesses, label="Fitness Promedio", linestyle="-.")
    plt.plot(generations, ga.worst_fitnesses, label="Peor Fitness", linestyle="--")

    plt.xlabel("Generaciones")
    plt.ylabel("Fitness")
    plt.title("Evolución de Fitness en el Algoritmo Genético")
    plt.legend()
    plt.savefig("graphs/fitness_plot.png")


def merge(diet):
    sums = {}

    for tuple in diet:
        if tuple[0] in sums:
            sums[tuple[0]] += tuple[1]
        else:
            sums[tuple[0]] = tuple[1]
    new_diet = [(key, value) for key, value in sums.items()]

    return new_diet


def execute(
    food_table,
    daily_requirements,
    population_size,
    num_generations,
    crossover_prob,
    mutation_prob,
    tournament_size,
):
    ga = GeneticAlgorithm(
        food_table,
        daily_requirements,
        population_size,
        num_generations,
        crossover_prob,
        mutation_prob,
        tournament_size,
    )
    best_diet_unchecked, best_fitness = ga.run()
    best_diet = merge(best_diet_unchecked)
    plot_results(ga)
    print(best_diet)
    return best_diet
