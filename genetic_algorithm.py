import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# función fitness mejorada
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
        if food_index >= len(food_table):  # Validación de índice
            continue
        food = food_table[food_index]
        total_weight += food_quantity * 100
        for i, nutrient in enumerate(nutrients, 2):
            if i < len(food):  # Validación de columnas
                total_nutrients[nutrient] += food[i] * food_quantity

    # Define las metas para cada nutriente y la penalización
    weight_limit = 3000
    excess_penalty = 15  # Incrementado para penalizar más los excesos
    margin = 1.15  # Reducido para ser más estricto
    nutrient_diff = 0
    deficit_penalty = 5  # Nueva penalización por deficiencia

    # Calcular pesos adaptativos para nutrientes críticos
    critical_nutrients = ["energy", "protein", "vitamin_c", "calcium", "iron"]

    # Calcula la distancia a los valores ideales de cada nutriente
    for nutrient in nutrients:
        ideal_nutrient = daily_requirements[nutrient]
        if ideal_nutrient > 0:
            current_nutrient = total_nutrients[nutrient]

            # Factor de importancia adaptativo
            importance_factor = 15 if nutrient in critical_nutrients else 10

            # Diferencia relativa
            if current_nutrient > 0:
                nutrient_diff_current = (
                    current_nutrient - ideal_nutrient
                ) / ideal_nutrient
            else:
                # Penalización severa por nutriente completamente ausente
                nutrient_diff += 50 * importance_factor
                continue

            # Penalización base por desviación
            base_penalty = abs(nutrient_diff_current) * importance_factor

            # Penalización adicional por exceso
            if current_nutrient > ideal_nutrient * margin:
                excess_factor = (nutrient_diff_current**2) * excess_penalty
                base_penalty += excess_factor

            # Penalización adicional por deficiencia severa (menos del 70%)
            elif current_nutrient < ideal_nutrient * 0.7:
                deficit_factor = (
                    (0.7 - (current_nutrient / ideal_nutrient)) ** 2
                ) * deficit_penalty
                base_penalty += deficit_factor

            nutrient_diff += base_penalty

    # Penalización progresiva por peso excesivo
    if total_weight > weight_limit:
        weight_excess = (total_weight - weight_limit) / weight_limit
        nutrient_diff += (weight_excess**1.5) * excess_penalty * 2

    # Penalización por dieta muy pequeña (menos de 1000g)
    if total_weight < 1000:
        weight_deficit = (1000 - total_weight) / 1000
        nutrient_diff += weight_deficit * 10

    # Función de fitness con mejor escalado
    if nutrient_diff == 0:
        return 100  # Fitness perfecto
    else:
        return max(0.1, 100 / (1 + nutrient_diff))  # Evitar fitness 0


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

        # Estrategia de inicialización mejorada
        single_food_ratio = 0.15  # Reducido para menos individuos monótonos
        balanced_ratio = 0.3  # Reducido
        targeted_ratio = (
            0.25  # Nueva categoría: individuos dirigidos a nutrientes específicos
        )
        smart_random_ratio = 0.3  # Combinaciones aleatorias inteligentes

        # Calcula el número de individuos de cada tipo
        num_single_food = int(self.population_size * single_food_ratio)
        num_balanced = int(self.population_size * balanced_ratio)
        num_targeted = int(self.population_size * targeted_ratio)
        num_smart_random = (
            self.population_size - num_single_food - num_balanced - num_targeted
        )

        # Agregar individuos con un solo tipo de alimento (cantidad variable)
        for _ in range(num_single_food):
            food_index = random.randint(0, len(food_table) - 1)
            quantity = random.randint(1, 6)  # Rango ampliado
            self.population.append([(food_index, quantity)])

        # Añadir individuos equilibrados entre todos los tipos de alimentos
        for _ in range(num_balanced):
            balanced_individual = [
                (food_index, random.randint(1, 3))  # Cantidades más conservadoras
                for food_index in range(len(food_table))
            ]
            self.population.append(balanced_individual)

        # Nuevos individuos dirigidos a cubrir nutrientes específicos
        critical_nutrients = ["energy", "protein", "vitamin_c", "calcium", "iron"]
        for _ in range(num_targeted):
            target_nutrient_idx = random.randint(
                2, 12
            )  # Índice de nutriente en la tabla
            # Seleccionar alimentos ricos en el nutriente objetivo
            rich_foods = []
            for i, food in enumerate(food_table):
                if len(food) > target_nutrient_idx and food[target_nutrient_idx] > 0:
                    rich_foods.append(i)

            if rich_foods:
                # Crear individuo con 2-4 alimentos ricos en el nutriente objetivo
                num_foods = random.randint(2, min(4, len(rich_foods)))
                selected_foods = random.sample(rich_foods, num_foods)
                targeted_individual = [
                    (food_idx, random.randint(1, 4)) for food_idx in selected_foods
                ]
                self.population.append(targeted_individual)
            else:
                # Fallback a individuo aleatorio
                num_foods = random.randint(2, 5)
                selected_foods = random.sample(range(len(food_table)), num_foods)
                fallback_individual = [
                    (food_idx, random.randint(1, 3)) for food_idx in selected_foods
                ]
                self.population.append(fallback_individual)

        # Rellenar el resto con combinaciones aleatorias inteligentes
        for _ in range(num_smart_random):
            # Número variable de alimentos (2-6)
            num_foods = random.randint(2, min(6, len(food_table)))
            selected_foods = random.sample(range(len(food_table)), num_foods)

            # Cantidades ponderadas (más probabilidad de cantidades menores)
            smart_individual = []
            for food_idx in selected_foods:
                # Distribución ponderada hacia cantidades menores
                if random.random() < 0.6:
                    quantity = random.randint(1, 2)
                elif random.random() < 0.8:
                    quantity = random.randint(2, 3)
                else:
                    quantity = random.randint(3, 4)
                smart_individual.append((food_idx, quantity))

            self.population.append(smart_individual)

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

        # Mutación adaptativa basada en el tamaño del individuo
        if len(individual) > 6:
            # Individuos grandes: favorecer eliminación y cambio
            possible_mutations = (
                ["change", "remove"] if len(individual) > 1 else ["change", "add"]
            )
        elif len(individual) < 3:
            # Individuos pequeños: favorecer adición y cambio
            possible_mutations = (
                ["change", "add"]
                if len(current_foods) != len(all_foods)
                else ["change"]
            )
        else:
            # Tamaño medio: todas las operaciones disponibles
            possible_mutations = (
                ["change", "add", "remove"]
                if len(current_foods) != len(all_foods) and len(individual) > 1
                else (
                    ["change", "add"] if len(individual) == 1 else ["change", "remove"]
                )
            )

        if random.random() < self.mutation_prob:
            mutation_type = random.choice(possible_mutations)

            if mutation_type == "change":
                # Mutación inteligente de cantidad
                food_index = random.choice(range(len(individual)))
                current_quantity = individual[food_index][1]

                # Probabilidad de incrementar vs decrementar basada en cantidad actual
                if current_quantity <= 2:
                    # Cantidades bajas: más probabilidad de incrementar
                    if random.random() < 0.7:
                        new_quantity = min(6, current_quantity + random.randint(1, 2))
                    else:
                        new_quantity = max(1, current_quantity - 1)
                else:
                    # Cantidades altas: más probabilidad de decrementar
                    if random.random() < 0.6:
                        new_quantity = max(1, current_quantity - random.randint(1, 2))
                    else:
                        new_quantity = min(6, current_quantity + 1)

                individual[food_index] = (individual[food_index][0], new_quantity)

            elif mutation_type == "add":
                # Adición inteligente de alimentos
                available_foods = list(all_foods - set(current_foods))
                if available_foods:
                    # Evaluar déficits nutricionales actuales
                    current_nutrients = self._calculate_nutrients(individual)

                    # Seleccionar alimento que pueda ayudar con déficits
                    best_food = None
                    best_score = -1

                    for food_idx in random.sample(
                        available_foods, min(5, len(available_foods))
                    ):
                        score = self._evaluate_food_addition(
                            food_idx, current_nutrients
                        )
                        if score > best_score:
                            best_score = score
                            best_food = food_idx

                    if best_food is not None:
                        # Cantidad conservadora para nuevos alimentos
                        quantity = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
                        individual.append((best_food, quantity))
                    else:
                        # Fallback a selección aleatoria
                        food_index = random.choice(available_foods)
                        individual.append((food_index, random.randint(1, 3)))

            elif mutation_type == "remove" and len(individual) > 1:
                # Eliminación inteligente (remover alimentos menos contributivos)
                if len(individual) > 2:
                    current_nutrients = self._calculate_nutrients(individual)
                    worst_contribution = float("inf")
                    worst_idx = 0

                    for i, (food_idx, quantity) in enumerate(individual):
                        # Calcular contribución relativa de este alimento
                        contribution = self._evaluate_food_contribution(
                            food_idx, quantity, current_nutrients
                        )
                        if contribution < worst_contribution:
                            worst_contribution = contribution
                            worst_idx = i

                    individual.pop(worst_idx)
                else:
                    # Eliminación aleatoria si solo hay 2 alimentos
                    food_index = random.choice(range(len(individual)))
                    individual.pop(food_index)

        return individual

    def _calculate_nutrients(self, individual):
        """Calcula los nutrientes totales de un individuo"""
        nutrients = {
            "energy": 0,
            "protein": 0,
            "fat": 0,
            "calcium": 0,
            "iron": 0,
            "vitamin_a": 0,
            "thiamine": 0,
            "riboflavin": 0,
            "niacin": 0,
            "folate": 0,
            "vitamin_c": 0,
        }

        for food_index, food_quantity in individual:
            if food_index < len(self.food_table):
                food = self.food_table[food_index]
                for i, nutrient in enumerate(nutrients.keys(), 2):
                    if i < len(food):
                        nutrients[nutrient] += food[i] * food_quantity

        return nutrients

    def _evaluate_food_addition(self, food_idx, current_nutrients):
        """Evalúa qué tan beneficioso sería agregar un alimento específico"""
        if food_idx >= len(self.food_table):
            return 0

        food = self.food_table[food_idx]
        score = 0

        nutrient_names = list(current_nutrients.keys())
        for i, nutrient in enumerate(nutrient_names, 2):
            if i < len(food) and nutrient in self.daily_requirements:
                required = self.daily_requirements[nutrient]
                current = current_nutrients[nutrient]
                food_contribution = food[i]

                if required > 0 and current < required:
                    # Puntuar positivamente si ayuda a cubrir déficit
                    deficit = required - current
                    score += min(food_contribution, deficit) / required

        return score

    def _evaluate_food_contribution(self, food_idx, quantity, current_nutrients):
        """Evalúa la contribución actual de un alimento a la dieta"""
        if food_idx >= len(self.food_table):
            return 0

        food = self.food_table[food_idx]
        contribution = 0

        nutrient_names = list(current_nutrients.keys())
        for i, nutrient in enumerate(nutrient_names, 2):
            if i < len(food) and nutrient in self.daily_requirements:
                required = self.daily_requirements[nutrient]
                food_contribution = food[i] * quantity

                if required > 0:
                    # Contribución relativa al requerimiento
                    contribution += food_contribution / required

        return contribution

    def run(self):
        best_individual = None
        best_fitness = -float("inf")
        stagnation_counter = 0
        max_stagnation = 15  # Generaciones sin mejora antes de aplicar diversificación

        # Parámetros adaptativos
        initial_mutation_rate = self.mutation_prob
        elitism_ratio = 0.25  # Incrementado para mejor preservación
        num_elites = int(self.population_size * elitism_ratio)

        for gen in range(self.num_generations):
            new_population = []
            generation_fitnesses = []

            # Calcular fitness de toda la población
            generation_individuals = []
            for ind in self.population:
                fit = fitness(ind, self.food_table, self.daily_requirements)
                generation_individuals.append((ind, fit))
                generation_fitnesses.append(fit)

            # Ordenar los individuos en función de su aptitud
            sorted_individuals = sorted(
                generation_individuals, key=lambda x: x[1], reverse=True
            )

            # Verificar mejora
            current_best_fitness = sorted_individuals[0][1]
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = sorted_individuals[0][0][:]  # Copia profunda
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Mutación adaptativa
            if stagnation_counter > 5:
                # Incrementar tasa de mutación gradualmente
                self.mutation_prob = min(
                    0.3, initial_mutation_rate * (1 + stagnation_counter * 0.1)
                )
            else:
                self.mutation_prob = initial_mutation_rate

            # Diversificación cuando hay estancamiento
            if stagnation_counter >= max_stagnation:
                # Reinicializar parte de la población
                num_to_replace = int(self.population_size * 0.4)
                self._diversify_population(sorted_individuals, num_to_replace)
                stagnation_counter = 0

            # Añadir los mejores individuos a la nueva población (Elitismo)
            elites = [ind for ind, fit in sorted_individuals[:num_elites]]
            new_population.extend(elites)

            # Generar resto de la población
            while len(new_population) < self.population_size:
                parents = self.select_parents()
                child = self.ordered_crossover(parents)
                child = self.mutate(child)

                # Validar que el hijo no esté vacío
                if not child:
                    child = [
                        (
                            random.randint(0, len(self.food_table) - 1),
                            random.randint(1, 3),
                        )
                    ]

                new_population.append(child)

            # Actualizar estadísticas
            self.best_fitnesses.append(max(generation_fitnesses))
            self.avg_fitnesses.append(
                sum(generation_fitnesses) / len(generation_fitnesses)
            )
            self.worst_fitnesses.append(min(generation_fitnesses))

            self.population = new_population

            # Criterio de parada temprana si se alcanza fitness muy alto
            if best_fitness > 95:
                print(f"Convergencia alcanzada en generación {gen + 1}")
                break

        return best_individual, best_fitness

    def _diversify_population(self, sorted_individuals, num_to_replace):
        """Diversifica la población reemplazando individuos similares"""
        # Mantener los mejores individuos
        num_keep = len(sorted_individuals) - num_to_replace
        kept_individuals = [ind for ind, fit in sorted_individuals[:num_keep]]

        # Generar nuevos individuos diversos
        for _ in range(num_to_replace):
            # Crear individuuo completamente nuevo con estrategia aleatoria
            num_foods = random.randint(2, min(6, len(self.food_table)))
            selected_foods = random.sample(range(len(self.food_table)), num_foods)
            new_individual = [
                (food_idx, random.randint(1, 4)) for food_idx in selected_foods
            ]
            kept_individuals.append(new_individual)

        self.population = kept_individuals


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
