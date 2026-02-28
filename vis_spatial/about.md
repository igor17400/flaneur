### The Data

**Gowalla** is a location-based social network where users *check in* at places (restaurants,
parks, shops, ...). Each check-in is an implicit signal: the user visited that venue, so they
probably like it. We strip away timestamps and geography and keep only the binary relation

$$
\mathcal{R} \;\subseteq\; \mathcal{U} \times \mathcal{I}
$$

where $\mathcal{U}$ is the set of users and $\mathcal{I}$ the set of items (venues).
A pair $(u, i) \in \mathcal{R}$ means "user $u$ interacted with item $i$".

> **Implicit feedback.** Unlike star ratings (explicit feedback), we only observe *positive*
> signals — a user checked in. Absence of a check-in does **not** mean dislike; it may simply
> mean the user hasn't discovered the venue yet. This asymmetry is the central challenge.

---

### How the files are organized

Each line in `train.txt` / `test.txt` follows the format:

```
userID  item_1  item_2  ...  item_k
```

So user $u$'s interaction set is the list $\{i_1, i_2, \dots, i_k\}$. Concatenating all
users gives us the interaction matrix:

$$
\mathbf{R} \in \{0,1\}^{|\mathcal{U}| \times |\mathcal{I}|}
\quad\text{where}\quad
R_{ui} =
\begin{cases}
1 & (u,i) \in \mathcal{R}\\
0 & \text{otherwise}
\end{cases}
$$

This matrix is extremely **sparse** (density $< 0.1\%$), which is typical of real-world
recommender datasets.

---

### Train / Test split

The dataset is pre-split per user. For every user $u$ with interactions $\mathcal{I}_u$:

- **Train set** $\mathcal{I}_u^{\text{train}}$: items the model can learn from.
- **Test set** $\mathcal{I}_u^{\text{test}}$: held-out items used only for evaluation.

The goal is to **rank all items** for each user and check whether the held-out items
appear near the top. We measure this with:

| Metric | Intuition |
|--------|-----------|
| **Recall@$K$** | Of the items the user *actually* liked, what fraction appears in the top-$K$? |
| **NDCG@$K$** | Same idea, but items ranked higher get more credit (position-aware). |

---

### The bipartite graph view

LightGCN treats the interaction matrix as a **bipartite graph**:

$$
\mathcal{G} = (\mathcal{U} \cup \mathcal{I},\; \mathcal{E})
\qquad
\mathcal{E} = \{(u, i) : R_{ui} = 1\}
$$

The adjacency matrix of this graph is the symmetric block matrix:

$$
\mathbf{A} =
\begin{pmatrix}
\mathbf{0} & \mathbf{R} \\
\mathbf{R}^\top & \mathbf{0}
\end{pmatrix}
\in \mathbb{R}^{(|\mathcal{U}|+|\mathcal{I}|) \times (|\mathcal{U}|+|\mathcal{I}|)}
$$

LightGCN propagates embeddings over this graph using the **normalized adjacency**
$\tilde{\mathbf{A}} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$, where $\mathbf{D}$
is the degree matrix. This is what makes a user's representation absorb information from
items they liked (and transitively, from other users who liked the same items).

---

### What to pay attention to in these tabs

| Tab | Key questions |
|-----|---------------|
| **Dataset Overview** | How sparse is the matrix? Is the item popularity distribution power-law? Are there users with very few interactions (hard to recommend for)? |
| **Graph Structure** | Does the degree distribution follow a power law? How does the 2-hop neighborhood look — are there hub items connecting many users? |
| **Dataloader Inspector** | Are negative samples truly items the user hasn't seen? How does negative item popularity compare to positive — is the sampler biased toward popular negatives? |
| **Train/Test Split** | Is the split ratio consistent across users? Are there cold-start users/items (appear only in test)? Do train and test share items per user (they shouldn't in a clean split)? |
