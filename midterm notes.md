# Module 1: Rational Agents and Task Environments
## Module 1 Definitions
* **Agent** - anything that can perceive and act in an environment
* **Percept** - an agent's sensors / input 
* **Percept sequence** - history of percept inputs
* **Rational agent** - agent that maximizes its performance measure based on the percept sequence
* **Agent function** - function that takes in percept sequence and outputs an action
* **Agent program** - internal implementation of agent function
* **Simple reflex agent** - Acts on the current percept, not the percept history (percept sequence)
* **Search problem** - problem of looking for a sequence of actions that reaches a goal
* **Fully observable** - agent is capable of seeing whole environement (i.e. chess)
* **Partially observable** - agent is capable of seeing part of the environment (i.e. driving)
* **Unobservable** - agent is not capable of seeing the environment
* **Deterministic** - an action given the current state produces the same next state every time (i.e. )
* **Nondeterministic** - an action given the current state can produce different state outcomes (stochastic is non-deterministic, but not the other way around)
* **Stochastic** - randomness and probablities are associated with the output of the next state
* **Episodic** - agent's actions are not dependent on previous actions / states
* **Sequential** - agent's actions are affected by previous actions
* **Dynamic** - environment changes when agent is in the process of responding to percept sequence (i.e. taxi driving)
* **Semidynamic** - environment does not change with time but agent's performance score does (i.e. chess with clock)
* **Static** - environment does not change when agent is in the process of responding to a percept sequence (i.e. poker)
* **Discrete** - environment consists of finite number of actions
* **Continuous** - not descrete; actions are not numbered
* **Single agent** - one agent in the environemnt
* **Multi agent** - multiple agents in the environment


## Module 1 Figures
<img src="https://www.cis.upenn.edu/~ccb/data/smart-textbook/AIMA_figures/figure_2.5.jpg" alt="Figure 2.5" width="75%"/>

* **Figure 2.5** -  


# Module 2: Python
## Module 2 Lecture Notes

### Collections:
```
from collections import defaultdict
d = defaultdict(str)
d['a'] #returns None

from collections import Counter
d = Counter()
d['a'] #returns 0
d['dog'] += 10
d['dog'] #return 10
```


### Args & Kwargs:
```
def print_everything(*args):
    # args is a tuple of arguments passed to the fn
    print(args)
print_everything('a', 'b', 'c’) # prints ('a', 'b', 'c’)
lst = ['a', 'b', 'c’]
print_everything(*lst) # prints ('a', 'b', 'c’)

def print_keyword_args(**kwargs):
    # kwargs is a dict of the keyword args passed to the fn
    print(kwargs)
print_keyword_args(first_name="John", last_name="Doe")
```

### Inheritence:
```
class A(object):
    def foo(self):
        print('Foo!')
class B(object):
    def foo(self):
        print('Foo?')
    def bar(self):
        print('Bar!')
class C(A, B):
    def foobar(self):
        super().foo() # Foo!
        super().bar() # Bar!
```

### Default Arguments:
* the function uses the same default object, initialized on execution


### Iterators:
* defines __next__, and __iter__ functions
* use memory more efficiently (loads one at a time)
```
class Reverse:
    "Iterator for looping over a sequence backwards”
    def __init__(self, data):
        self.data = data
        self.index = len(data)
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index – 1
        return self.data[self.index]
    def __iter__(self):
        return self
```

### Generators:
* use yield, can call next(iter)

### Misc:
* dictionary keys are immutable, d.items() gives tuples
* 


## Module 2 Definitions
* **Mutability** - can change
* **First class object** - have all the rights other variables in the programming language have, i.e. dynamically created, destroyed, returned
* **Magic methods** - Allow user defined classes to behave like built in types (i.e. "def __str__()" )
* **Duck typing** - You don't care about type or class of object, only care about methods defined


# Module 3: Search Problems
## Module 3 Lecture Notes
### Search Problem:
* Formulation: 
    * States (and initial state)
    * Actions (set A)
    * Transition Model
    * Performance Measure / Path Cost (must be additive)
    * Goal test - can be implicit (i.e. checkmate)

### Time  / Space Complexity:
* b = maximum branching factor of search tree
* d = depth of shallowest goal node
* m = maximum length of any path in state space (potentially infinity)

### BFS:
* Complete -> Yes
* Optimal -> Yes, if cost = 1 per step (not in general)
* Time Complexity -> $O(b^d)$ = O(V + E)
* Space Complexity -> $O(b^d)$ = O(V) [every node in memory]

```
def bfs(): #O(V + E) time, O(V) space
    while frontier is not empty:
        node = frontier.get()
        for neighbor in neighbors(node):
            if neighbor not in visited:
                frontier.pushToBack(neighbor)
                visited.add(neighbor)
```

### DFS:
* Complete -> No. Fails in infinite-depth spaces or with loops. If m is bounded, then complete
* Optimal -> No
* Time Complexity -> $O(b^m)$
    * Worse than BFS if $m >>> d$, better if solutions are dense
* Space Complexity --> $O(b * m)$ (linear space)

```
# DFS: O(V + E) time, O(V) space
def dfs():
    while frontier is not empty:
        node = frontier.get()
        for neighbor in neighbors(node):
            if neighbor not in visited:
                frontier.putOnTop(neighbor)
                visited.add(neighbor)
```

### DFS vs BFS:
(these benefits / loop problem only exist if you don't track visited)
* DFS is better:
    * Space is restricted
    * Lots of solutions with long paths, wrong paths are short / terminated quickly
    * Search can be fine-tuned quickly
* BFS is Better:
    * Possible infinite paths / loops
    * Some solutions have short paths
    * Can quickly discard unlikely paths


### Iterative Deepening Search:
* Procedure: depth first search with depth limit
* Optimal --> Yes, if step 
* Complete --> Yes, no infinite paths
* Time Complexity --> $O(b^d)$
* Space Complexity --> $O(b * d)$


### Summary:
| Criterion | BreadthFirst | DepthFirst | Depthlimited | Iterative-deepening |
| ----- | ----- | ----- | ----- | ----- |
| **Complete?** | YES | NO | NO | YES | 
| **Optimal?** | YES | NO | NO | YES |
| **Time** | $b^d$ | $b^m$ | $b^l$ | $b^d$ | 
| **Space** | $b^d$ | $b^m$ | $b \cdot l$ | $d$ | 

* b = maximum branching factor of search tree
* d = depth of shallowest goal node
* m = maximum length of any path in state space (potentially infinity)


## Module 3 Definitions
* **Problem-solving agent** - an agent that can formulate a plan via search
* **Search problem** - states, actions, performance measure
* **Tree search** - enumerate in some order all the possible paths from the initial state
* **Graph search** - iteration of tree search where we limit the paths to nodes we haven't seen yet
* **Completeness** - Whether you always find a solution
* **Optimality** - Finds a least cost solution or lowest path cost first
* **Time complexity** - nodes generated in worst case
* **Frontier** - set of leaf nodes that can be expanded at any given point
* **Uniform cost search** - expands the node with the lowest path cost g(n), optimal and complete
* **Iddfs** - Iterative Deepening Depth First Search


# Module 4: Informed Search
## Module 4 Lecture Notes
### Uniform Cost Search(UCS): uses g(n)
* all moves equal in cost, g(N) = depth(N) in tree
* expand node (remove and use node) with lowest path cost
* priority queue ordered by path cost
* diff w/ bfs: tests if node is goal state when selected, not when added to frontier

### Greedy Best-first Search: uses h(n)
* select node for expansion that is estimated to be closest to goal
* f(n) *includes* h(n) heuristic distance to goal
* sorted by priority queue, f(n) replaces g(n) and g(n) ignored
* **Not Complete** --> can get stuck in loops


```
def greedy_best_first():
    while frontier is not empty:
        node = frontier.get()
        if node == goal:
            break

        for neighbor in neighbors(node):
            if neighbor not in discovered:
                priority = heuristic(neighbor --> goal)
                frontier.put(priority, neighbor)
                came_from[neighbor] = node
```

### A Star Search: uses g(n) + h(n)
* f(n) = g(n) + h(n); total cost = depth/actual cost so far + estimated heuristic
* implemented with priority queue
* uses an admissable heuristic
* explores more nodes than greedy best-first, but it will always find the shortest path to the goal first
* Complete --> Yes
* Optimal --> Yes
* Time Complexity --> O(b * d)

```
def a_star(): # O(b * d) time, where b is branching factor and d = depth/distance to goal
    while frontier/queue is not empty:
        if node == goal:
            break

        for neighbor in neighbors(node):
            new_cost = cost_so_far[node] + actual_cost(node --> neighbor)

            if neighbor not discovered or new_cost < cost_so_far[node]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor --> goal)
                frontier.put(priority, neighbor)
                came_from[neighbor] = node
```

### Dominance: metric on better heuristics
* $h_2(n) \geq h_1(n) \implies h_2$ dominates $h_1$
* $h_2$ is optimistic but more accurate than $h_1 \implies h_2$ is better 

### Admissible Heuristics:
* $h(n)$ is admissible $\iff 0 \leq h(n) \leq h^*(n)$, where $h^*(n)$ is true cost to goal



## Module 4 Definitions
* **Admissible heuristic** - heuristic (of distance) that cannot be an overestimate (optimistic)
* **Dominance** - metric on better heuristics
* **Uniform-cost search (ucs)** - expand node with lowest path cost g(n)
* **Beam search** - best-first search optimization that reduces memory and size of frontier --> not optimal or complete
* **Sma (simplified memory bounded a*)** - More efficient memory bounded A*
* **Rbfs (recursive best first search)** - recursive best first search, more efficient than A*
* **Ida (iterative deepening a*)** - A* with limited depth, more efficient than A*


# Module 5: Adversarial Search and Games
## Module 5 Lecture Notes
### Minimax:
* Rules: Make best moves for MAX, assuming MIN moves with best for MIN
* Procedure: (until game is over)
    1. Start with current position as MAX node
    2. Expand game tree a fixed number of **ply**,
    2. Apply evaluation function to leaves
    3. Calculate back up values bottom-up
    4. Pick move assiged to MAX at the root
    5. Wait for MIN node to respond

```
def value(state):
    if state is terminal state:
        return state's utility
    if next agent is MAX:
        return max_value(state)
    if next agent is MIN:
        return min_value(state)

def max_value(state):
    if state is terminal state: #leaf node
        return state's utility 
    initialize v = -infinity
    for each successor of state:
        v = max(v, min_value(successor))
    return v

def min_value(state):
    if state is terminal state: #leaf node
        return state's utility
    initialize v = +inifinity
    for each successor of state:
        v = min(v, max_value(successor))
    return v
```

### Alpha Beta Pruning:
* Alpha: MAX's current lower bound on MAX's outcome ("at least")
* Beta: MIN's current upper bound on MIN's outcome ("at most")
* On MAX node: if $v \geq beta$ is backed-up $\implies$ MIN will never select that MAX, stop branch
* On MIN node: if $v \leq alpha$ is found $\implies$ MAX will never select that MIN, stops branch

```
def max_value(state, alpha, beta):
    if terminal state:
        return state's utility

    v = - infinity
    for each successor of state:
        v' = min_value(successor, alpha, beta)
        if v' > v:
            v = v'
        if v' >= beta:
            return v
        if v' > alpha:
            alpha = v'
    return v

def min_value(state, alpha, beta):
    if terminal state:
        return state's utility
    
    v = + infinity
    for each successor of state:
        v' = max_value(successor, alpha, beta)
        if v' < v:
            v = v'
        if v' <= alpha:
            return v
        if v' < beta:
            beta = v'
    return v
```

### Expectimax:
* for explicit randomness, unpredictable opponents, actions that fail (wheel stuck)
* Uncertain outcomes controlled by chance and not adversary, compute average score under uptimal play
* MAX nodes and Chance nodes (similar to min nodes, but outcome is uncertain)
* Calculate their expected values

```
def value(state):
    if state is terminal:
        return state's utility
    if next agent is MAX:
        return max_value(state)
    if next agent is EXP:
        return exp_value(state)

def max_value(state):
    v = - infinity
    for each successor of state:
        v = max(v, value(successor))
    return v

def exp_value(state):
    v = 0
    for each successor of state:
        v += probability(successor) * value(successor)
    return v
```


## Module 5 Definitions
* **Ply** - turn taken by player
* **Backed-up value** - MAX node: *maximum* value of its children, MIN node: *minimum* value of its children
* **Forward pruning** - prunes moves that appear to be bad move, i.e. beam search


# Module 6: Constraint Satisfaction Problems
## Module 6 Lecture Notes
### Constraint Satisfaction Problems (CSP):
* Applications: map coloring, scheduling, planning, sudoku
* Solutions to CSP are complete and consistent assigments (def. below)
* Consists of variables, domains, constraints
    * i.e. Mapping: countries, color set, adjacent countries have different colors
* General purpose algorithm, given only the formal specification of the CSP
* Prune branches that violate constraints
* Problems are **commutative** (can choose any empty variable to assign at any node)

### CSP as a Search Problem:
* Uses depth first search & backtracking
* Initial State: empty assignment
* Successor Function: assign value to any unassigned variable (with no constraint conflict)
* Goal Test: Current assigment is complete 
* Path Cost: constant cost for every step
* Branch Factor: num variables left * domain size ($n \cdot d$)
* Num Leaves: $n! \cdot d^n$

### CSP Heuristics: (General Purpose methods and heuristics can give huge gains in speed on average)
* Which variable should be assigned next
    1. Most constrain*ed* variable (fast fail)
    2. (if ties) Most Constrain*ing* variable on other variables
* What order should that variable's values be tried:
    3. Least Constraining **value** on other variables
* Can we detect inevitable failure early?
    4. Forward Checking: 
        * track the remaining legal values for *unassigned variables*
        * terminate search when any unnasigned variable has no remaining legal values

### Arc Consistency:
* CSP Solver: Search + Inference/Constraint Propogation)
* Arc Digraph: edges are constraints and nodes are variables
* Propogation & Arc Consistency: make each arc consistent 
    * X &rarr; Y is consistent (X is tail, Y is head), iff for every value x of X there is some allowed y
    * Check if Y has values allowed by the values of X. If not, remove possible assignments from head
    * If X loses a value, recheck neighbors of X 
    * Formally: given arc(i, j), to make the arc consistent: # With d values, arc(i, j) takes $O(d^2)$ time
        * for each value v on $X_i$:
            * if there is no label on $Y_j$ consistent with v:
                * remove v from $X_i$

### AC3:
* Derived from waltz algorithm, **search** to add labels for one solution, then constraint propogation to eliminate labels to find all solutions simultaneously
* time complexity: $O(n^2 d^3)$
    * Fully connected digraph has $2n(n-1)$ arcs (edges) = $O(n^2)$ [sudoku would be O(n * sqrt(n)) edges]
    * d values, checking arc(i, j) --> $O(d^2) time
    * each arc(i, j) can be inserted into queue at most $d$ times = $O(d)$

```
def AC3(CSP problem): #return CSP, possibly with reduced domains
    queue = all arcs of the CSP problem
    while queue is not empty:
        X_i, X_j = queue.get()
        if REMOVE_INCONSISTENT_VALUES(X_i, X_j):
            for each X_k in neighbors(X_i) w/o X_j:
                queue.add( (X_k, X_i) )

def REMOVE_INCONSISTENT_VALUES(X_i, X_j): #returns true iff we remove a value
    removed = false
    for each x in domain(X_i):
        if no value y in domain(X_j) allows (x, y) to satisfy the constraint between X_i and X_j:
            delete x from domain(X_i)
            removed = true
    return removed
```



## Module 6 Definitions
* **Consistent assignment** - Assigment of variable values that do not violate constraints
* **Complete assignment** - Every Variable is assigned a value
* **Nodes** - Variables
* **Arcs** - Binary Constraints
* **Unary constraints** - Constraints of one variable (i.e. region != green)
* **Binary constraints** - Constraints of two variables (i.e. region1 != region2)
* **Higher order constraints** - Constraints that involve 3 or more variables
* **Soft constraints / preference** - Cost or Rule for what is better (i.e. red is better than green)
* **Backtracking search** - DFS chooses values for one variable and backtracks when a variable has no legal values left to assign
* **Constraint propogation (inference)** - Elminate possible values if the value would violate local consistency
* **Node consistency** - Local Consistency that satisfying unary constraints
* **Arc consistency** - Local Consistency that satisfies binary constraints


# Module 7: Logical Agents
## Module 7 Lecture Notes
### Knowledge-based Agents (KB Agents):
* Knowledge representation language: i.e. "if planit is far from sun, then it iscold" becomes planet(x) and sun(y) and far_from(x,y) $\implies$ cold(x)
* two types of sentences: axioms, derived sentences

### Define World: (i.e. Wampa)
* environement, performance measuere, actuators (define actions and results), information sensors

***Model*** *example*: if alpha is "there is no pit in [2, 2]", then M(alpha) is the set of all Wampa Worlds where [2, 2] doesn't have a pit, and $m \in M(alpha)$

### Logical Entailment:
* **Logical Reasoning** involves entailment relationship between sentence
* alpha entails beta --> alpha ⊨ beta
* **Entailment Definition**: alpha ⊨ beta $\iff M(alpha) \subseteq M(beta)$, where $M(x)$ is the set of all models of $x$
    * alpha is more specific than beta --> i.e. alpha is "I am a human", beta is "I am a mammal"

### Logical Inference:
* deriving conclusions by applying entailment
* design inference algorithm ($i$) to enumerate n the sentences, $KB ⊨_i a$ --> "$a$ is derived from $KB$ by $i$"

### Propositional Logic:
* $\lor$ = or
* $\land$ = and
* ¬ = not
* $\implies$ = implies / conclusion || i.e. antecedent $\implies$ conclusion
* $\iff$ = if and only iff / biconditional

### Knowledge Bases:
* Construct representation for environment (i.e. grid and symbols for Wampa World)
* A entails B <--> A must be a subset of B

### Logical Equivalence:
* (a ⋀ β) ≡ (β ⋀ a) **Commutativity of ⋀**
* (a ⋁ β) ≡ (β ⋁ a) **Commutativity of ⋁**
* ((a ⋀ β) ⋀ ɣ) ≡ (a ⋀ (β ⋀ ɣ)) **Associativity of ⋀**
* ((a ⋁ β) ⋁ ɣ) ≡ (a ⋁ (β ⋁ ɣ)) **Associativity of ⋁**
* ¬ (¬ a) ≡ a **Double-negation elimination**
* (a ⟹ β) ≡ (¬β ⟹ ¬a) **Contraposition**
* (a ⟹ β) ≡ (¬a ⋁ β) **Implication elimination**
* (a ⇔ β) ≡ ((a ⟹ β) ⋀ (β ⟹a)) **Biconditional elimination**
* ¬(a ⋀ β) ≡ (¬a ⋁ ¬β) **De Morgan**
* ¬(a ⋁ β) ≡ (¬a ⋀ ¬β) **De Morgan**
* ((a ⋀ β) ⋁ ɣ) ≡ ((a ⋀ β) ⋁ (a ⋀ ɣ))) **Distributivity of ⋀ over ⋁**
* ((a ⋁ β) ⋀ ɣ) ≡ ((a ⋁ β) ⋀ (a ⋁ ɣ))) **Distributivity of ⋁ over ⋀**

### Theorem Proving:
* use **inference rules** to derive proofs (a $\implies$ β, a) / β
    * numerator is given, denomitor is inferred
* Proof by resolution and Conjunctive Normal Form

### Conjunctive Normal Form (CNF): Conjunction of Disjoints (ANDs of ORs)
1. Eliminate bidirectionals ($\iff$): (a ⇔ β) ≡ ((a ⟹ β) ⋀ (β ⟹a))
2. Eliminate implications ($\implies$): (a ⟹ β) ≡ (¬a ⋁ β)
3. Move ¬ inwards (eliminated double negation and apply De Morgan)
4. Apply distributivity laws for ⋀ and ⋁
    
### Resolution Algorithm:
* Proof by contradiction: Prove that KB ⊨ alpha by showing that KB ⋀ ¬alpha is unsatisfiable
1. Convert (KB ⋀ ¬a) into CNF
2. Apply resolution rule
3. Resolve Every pair that contains complementary literals to produce a new clause
4. If new clause isn't already present, add it to the set
5. Continue until: 
    * there are no new classes to be added, then KB does not entail a
    * two clauses resolve to yield the empty clause, then KB does entail a

### Horn and Definite Clauses:
* More efficient inference algorithm (now linear time) with restriction than Resolution Algorithm
* Resolution applied to a horn clause gets back a horn clause
* **Forward Chaining**: An agent can start with known data, adding premises as new percepts come in, and then incrementally apply the forward chaining algorithm to derive new conclusions. It doesn’t have to have a specific query in mind to do so
* **Backward Chaining**: works backwards from a query. If the query is known to be true,
then no work is needed. Otherwise, the algorithm finds implications in the KB whose
conclusions is the query. If all the premises of one of those can be proved to be true
(via backward chaining), then the query is true.


## Module 7 Definitions
* **Knowledge base (kb)** - Set of sentences with some assertion about the world written in a knowledge representation language
* **Knowledge base agents** - Use process of reasoning over internal representation of knowledge to decide what action to take
* **Axioms (sentence)** - Sentence that is given
* **Derived sentences** - New sentence that is derived from others sentences
* **Inference** - Process of deriving a new sentence (axioms --> derived sentences)
* **Logic** - defines truth of each sentence w.r.t. a possible world (called a model)
* **Models** - "Possible World": Mathematical abstractions that have a fixed set of truth values {true, false} for each sentence
* **Entailment** - Idea that a sentence follows logically from another sentence
* **Logical inference** - Deriving conclusions (by applying entailment)
* **Atomic sentences** - sentences represented with a signle propositional symbol [i.e. $W_{1, 3}$ = Wompa in posiiton (1, 3)]
* **Propositional symbols** - Symbols that stand for a stament that can be true or false
* **Complex sentences** - Simple sentences connected with logical connectives
* **Model checking** - Process of enumerating all the possible models that are compatable with the Knowledge Base (KB). $M(KB) \subseteq M(alpha_1)$
* **Sound (soundness)** - An inference algorithm should only derive entailed sentences
* **Complete (completeness)** - An inference algorithm can derive all sentences that are entailed
* **Monotonicity** - if KB ⊨ alpha, then KB ⋀ β ⊨ alpha; we can add to KB w/o invalidating what we inferred
* **Resolution** - a single valid inference rule that produces a new clause implied by two clauses containing complementary literals
* **Disjunction** - Or statement: ⋁
* **Conjunction** - And Statement: ⋀
* **Conjunctive normal form (cnf)** - a conjunction of one or more clauses, where a clause is a disjunction of literals
* **Definite clause** - disjunction of literals where exactly one is positive
* **Horn clause** - disjunction of literals where at least one is positive, including definite clauses and clauses with no positive literals


## Module 7 Figures
<img src="https://www.cis.upenn.edu/~ccb/data/smart-textbook/AIMA_figures/figure_7.6.jpg" alt="Figure 7.6" width="75%"/>

* **Figure 7.6** -  
<img src="https://www.cis.upenn.edu/~ccb/data/smart-textbook/AIMA_figures/figure_7.7.jpg" alt="Figure 7.7" width="75%"/>

* **Figure 7.7** -  
<img src="https://www.cis.upenn.edu/~ccb/data/smart-textbook/AIMA_figures/figure_7.8.jpg" alt="Figure 7.8" width="75%"/>

* **Figure 7.8** -  
<img src="https://www.cis.upenn.edu/~ccb/data/smart-textbook/AIMA_figures/figure_7.12.jpg" alt="Figure 7.12" width="75%"/>

* **Figure 7.12** -  
