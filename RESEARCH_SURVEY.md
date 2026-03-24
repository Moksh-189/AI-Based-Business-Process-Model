# Comprehensive Research Survey: AI-Driven Process Intelligence

## 1. Introduction
This survey examines the intersection of **Process Mining (PM)**, **Graph Representation Learning**, and **Deep Reinforcement Learning (DRL)**. The goal is to contextualize the **AI-Based Business Process Intelligence (AI-BPI)** system within the current state-of-the-art, highlighting how it synthesizes distinct research streams into a unified "Perception-Action" architecture.

---

## 2. Predictive Process Monitoring (The "Perception" Layer)
This domain focuses on predicting the future state of a running process case (e.g., next activity, remaining time, outcome).

### A. Sequence Modeling with LSTMs
**Key Reference:** *Tax, N. et al. (2017). Predictive Business Process Monitoring with LSTM Neural Networks.* ([arXiv Link](https://arxiv.org/abs/1612.02130))

*   **Contribution:** Established the foundational baseline for Deep Learning in BPM. They treated event logs as sequential text and used Long Short-Term Memory (LSTM) networks to predict the next event suffix.
*   **Limitation:** LSTMs are purely sequential. They struggle to capture complex non-linear dependencies, such as parallel branches, loops, and "long-term" dependencies where an event at the start of a process impacts an outcome much later, without direct adjacency.
*   **Relevance:** This represents the "Standard Baseline" against which graph-based approaches are compared.

### B. Graph Representation Learning (GNNs)
### B. Graph Representation Learning (GNNs)
**Key Reference:** *Lischka, A. et al. (2025). Directly-Follows Graphs Go Predictive Process Monitoring With Graph Neural Networks.* ([arXiv Link](https://arxiv.org/abs/2503.03197))

*   **Contribution:** Proposed a novel mapping of event logs to **Directly-Follows Graphs (DFGs)** enriched with node features. They evaluated various GNN architectures (GCN, GAT, GraphSAGE) and found that GNNs successfully capture the topological structure of business processes, offering a competitive alternative to sequence-based models (LSTMs) especially in processes with complex control flows.
*   **Limitation:** The approach focuses on **Next-Activity Prediction**. It provides a "State Estimation" (where are we?) and "Future Forecasting" (where are we going?), but it remains an observer. It lacks the decision-making layer found in Prescriptive Process Monitoring.
*   **Relevance:** The AI-BPI system aligns with this architecture's "Perception" module. We use a similar DFG-to-GNN pipeline to extract a rich state representation. However, we extend this work by feeding this representation into a **Policy Network (RL Agent)**, answering the "So What?" question that purely predictive papers leave open.

---

## 3. Prescriptive Process Optimization (The "Action" Layer)
This domain asks: "Given the current state, what is the optimal action to maximize business value?"

### A. Tabular & Deep RL for Resource Allocation
**Key References:**
1.  *Silvander, J. (2019). Business Process Optimization with Reinforcement Learning.* ([ResearchGate Link](https://www.researchgate.net/publication/334112046_Business_Process_Optimization_with_Reinforcement_Learning))
2.  *Żbikowski, K. et al. (2021). Deep Reinforcement Learning for Resource Allocation in Business Processes.* ([arXiv Link](https://arxiv.org/abs/2103.13603))

*   **Contribution:** Applied RL (Q-Learning, DQN) to assign resources to tasks dynamically. They shifted the focus from static rules (FIFO) to learned policies that minimize cycle time.
*   **Limitation (The "Topology Gap"):** These agents typically viewed the process state as a flat vector of numbers (e.g., "Queue Length = 50"). They lacked **structural awareness**. The agent didn't "know" that Task A is a prerequisite for Task B; it just learned correlations blindly.
*   **Relevance:** The AI-BPI system bridges this gap. By integrating the GNN embeddings (from Section 2) into the RL observation space, our **PPO Agent** is "Topology-Aware." It learns to avoid actions that create downstream bottlenecks because it "sees" the graph structure.

---

## 4. Generative AI & Digital Twins (Validation & Interaction)

### A. Digital Twins for Resilience
**Key Reference:** *Ivanov, D. (2020). A digital supply chain twin for managing the disruption risks and resilience.* ([ResearchGate Link](https://www.researchgate.net/publication/340643209_A_digital_supply_chain_twin_for_managing_the_disruption_risks_and_resilience_in_the_era_of_Industry_4_0))

*   **Contribution:** Defined the Digital Twin as a simulation replica used for stress-testing supply chains. Validation typically involved running heavy "What-If" Monte Carlo simulations.
*   **Limitation:** Feedback loops were slow. Optimization required running thousands of simulations *per decision*.
*   **Relevance:** We use the Digital Twin (SimPy) differently: as a **Training Environment** (Gym). This allows the RL agent to learn *offline*. In production, the agent executes in milliseconds, providing real-time resilience without the computational cost of live simulation.

### B. LLMs in BPM
**Key Reference:** *Mendling, J. et al. (2023). Large Process Models: Business Process Management in the Age of Generative AI.* ([arXiv Link](https://arxiv.org/abs/2309.00900))

*   **Contribution:** Explored LLMs for process discovery and declaratively generating process models.
*   **Limitation:** Most applications are "Chat with PDF" style—disconnected from the runtime operational data.
*   **Relevance:** Our **RAG-based Chatbot** connects the LLM directly to the live quantitative metrics (GNN and RL outputs), making it an "Operational Analyst" rather than just a document summarizer.

---

## 5. Methodology: The "Multi-Perspective Semi-Synthetic" Approach

A critical challenge in OCPM (Object-Centric Process Mining) research is obtaining datasets that contain *both* hard operational logs (SAP) *and* soft collaborative context (Jira/Teams) with ground-truth links.

**Proposed Framework:** *Augmenting Real-World Event Logs with Synthetic Context.*

To validate our multi-modal system, we utilized a **Semantically Coupled Data Augmentation** strategy:

1.  **The "Skeleton" (Real):** We utilized a simplified version of the **SAP P2P (Procure-to-Pay)** dataset as the immutable ground truth. This provides the realistic temporal distribution, case variants, and "happy/unhappy" paths found in industry.
2.  **The "Flesh" (Synthetic-Correlated):**
    *   **Jira Tickets:** Generated using a probabilistic Mapping where `SAP_Value` $\propto$ `Jira_Priority` and `SAP_SpendArea` $\rightarrow$ `Jira_Domain`. This tests the system's ability to recover latent rules.
    *   **Teams Chatter:** Generated using a complexity-based trigger. `SAP_Event_Count > Threshold` $\rightarrow$ `Teams_Sentiment < 0.5`. This tests the system's ability to correlate "operational friction" (high event count) with "social friction" (negative sentiment).

**Scientific Validity:**
This approach aligns with the **"Synthetic Real"** evaluation standards in Process Mining (e.g., standardizing benchmarks like the *BPI Challenge* datasets). It allows for **Controlled Experimentation**: we *know* the correlation exists because we injected it, allowing us to mathematically prove (via Correlation Coefficient & MAE) that the GNN properly "discovered" the hidden relationship.
