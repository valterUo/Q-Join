# Quantum computing methods for join order optimization in relational databases

## Plan

**Input:** Query as a query graph: nodes are tables, edges are joins. Graph can be a multigraph so that different edges represent different type of joins. Every node contains information of the size of the table. Every edge describes the selectivity of the join. Rational: The selectivity can be computed/estimated easily (ideally).
**Output:** High-order unconstrained binary optimization problem

### Algorithm 1

Binary variables: (join id, level in join tree). This means that we have $m*(m-1)$ binary variables where $m$ is number of joins i.e. the number of edges in the query graph. Level in join trees describes precisly when the join should be executed. With this information we can also build the join tree itself. Let the notation be the following. Let the database consists of $n$ tables. Tables are $T_i$ for $i \in \left\{0, \ldots, n - 1\right\}$. The joins are binary join of type $T_i \bowtie_{t} T_j$ which we encode as a variable $x_{ijt}$. The variable $t$ encodes the type of the performed join and for simplicity we drop it from the current discussion and add it later. Besides, the join happening at level $l$, we use the notation $x_{ij}^{l}$. If $x_{ij}^{l} = 1$, then $T_i \bowtie T_j$ should be performed at level $l$ in the join tree. We assume that the join is commutative $T_i \bowtie T_j = T_j \bowtie T_i$ with the same cost and that there is a natural order among the tables so that $i \leq j$.

There is a misuse in this notation: the joins are considered operations between tables with suitable signature rather than operations between tables of fixed relations. Although we write $T_i \bowtie T_j$, we do not exactly mean tables $T_i$ and $T_j$ as they are stored in the database but we mean tables that have the same columns. This means that one of $T_i$ or $T_j$ or both can be results of previously performed joins so that only the signature of the table matches and we are technically able to perform the join. In other words, joins are rather considered to be functions $\bowtie(T_i, T_j)$ depending only on signature of the tables than operations between fixed tables.

Claim: Now any sequence of joins defines a join tree. This can be approached so that every sequence of joins defines a contraction/transformation of the query graph into a single node.

Now the constraints are really simple:

**Hard constraint encoding the validity of the solution:**
Every initial table needs to be part of exactly one join at exactly one level: This can be encoded so that we group the joins into sets $\left\{ x_{ij}^{l} \mid T_i \right\}$ and use the standard 1-combinations constraint for each group. This forces as to select only one join at one level for each table $T_{i}$.

**Note that we do not force:**
- Not every join needs to be used because it is possible that we have multiple different joins between the tables
- Not every level needs to have joins because we might find necessarily a bushy join tree whose hight is lower than left-deep join trees which are the longest possible join trees

**Soft constraint encoding the cost:**
Total cost should be minimized.

We know that at each step the cardinality of the join tree can be calculated recursively
$$
|T| = \begin{cases}
|T_i| &\text{if } T = T_i \text{ for some } i \in [n], \\
\prod_{T_i \in T_1, T_j \in T_2} f_{ij} |T_1||T_2| &\text{if } T = T_1 \bowtie T_2.
\end{cases}
$$

Besides, given a join tree $T$, the cost of performing the join is
$$
C(T) = \begin{cases}
0 &\text{if } T = T_i \text{ for some } i \in [n], \\
|T| + C(T_1) + C(T_2)  &\text{if } T = T_1 \bowtie T_2.
\end{cases}
$$

The simple idea is to use the two previous formulations to encode the cost function using the binary variables. Although the cost function becomes non-linear and classically complex, the idea is that with quantum computing we have native methods to minimize it and our claim is that quantum computing is excellent in encoding this type of non-linear complicated functions.

Let's assume that we are performing the join $T_i \bowtie T_j$ at level $1$. Then we are asked if we should set the variable $x_{ij}^{1} = 1$ or not. If we decide to set $x_{ij}^{1} = 1$, then we will the add the corresponding cost of performing this join at the level $1$ to the total cost. In other words, this would be encoded simply $c x_{ij}^{1}$. But the question is, what is the cost $c$? This cost clearly depends on the joins that we have performed at level $0$, i.e., the joins that we have performed before. So in reality, the cost $c$ is a function that depends on the variables at level $0$.

Now the recursive definitions help us to formulate this cost $c$. To calculate the cost $c$, we do not need to consider every possible variable $x_{tp}^{0}$ at level $0$ but only those variables which are adjacent to the tables $T_i$ and $T_j$ i.e. variables $x_{ik}^{0}$ and $x_{pj}^{0}$. Thus the cost $c$ becomes
$$
c = f_{ij} |T_i||T_j| + \sum_{T_i \bowtie T_k} c_{ik}^{0}x_{ik}^{0} + \sum_{T_k \bowtie T_j} c_{kj}^{0}x_{kj}^{0}.
$$
This formulate clearly relates to the previous one:
$$
c_{ij}^{l} = \begin{cases}
f_{ij} |T_i||T_j| &\text{ if } l = 0,\\
f_{ij} \sum_{T_i \bowtie T_k} c_{ik}^{l-1}x_{ik}^{l-1}\sum_{T_k \bowtie T_j} c_{kj}^{l-1}x_{kj}^{l-1} + \sum_{T_i \bowtie T_k} c_{ik}^{l-1}x_{ik}^{l-1} + \sum_{T_k \bowtie T_j} c_{kj}^{l-1}x_{kj}^{l-1}  &\text{if } l > 0.
\end{cases}
$$