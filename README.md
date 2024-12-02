# 📊 Social Network Analysis Course Repository

> This repository contains materials, assignments, and projects from my Social Network Analysis course as a fourth-year Data Science student.

## 📚 Table of Contents

- [Introduction](#-introduction)
- [Repository Structure](#-repository-structure)
- [Technologies Used](#-technologies-used)
- [Installation &amp; Usage](#-installation--usage)
- [Course Content](#-course-content)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 🌟 Introduction

This repository showcases various aspects of Social Network Analysis, including:

- Network structure analysis
- Key metrics calculation (PageRank, HITS, Centrality)
- Community detection
- Text analysis and topic modeling
- Real-world SNA applications

The code and materials here demonstrate both theoretical understanding and practical implementation of SNA concepts.

## 📂 Repository Structure

```
social-network-analysis/
│
├── data/                    # Sample data and datasets
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code
│   ├── page_rank_hits_demo.py
│   ├── tf_idf_lda_demo.py
│   └── overview.py
├── docs/                   # Course materials
│   └── 01. Introduction to Social Networks.pdf
├── results/                # Analysis outputs
├── requirements.txt        # Project dependencies
└── README.md
```

## 🔧 Technologies Used

- **Core Technologies:**

  - Python 3.8+
  - NetworkX for graph analysis
  - Scikit-learn for machine learning
  - Pandas for data manipulation
  - NumPy for numerical computations
  - Matplotlib/Plotly for visualization
- **Key Libraries:**

```python
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
```

## 💻 Installation & Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/social-network-analysis.git
cd social-network-analysis
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 📖 Course Content

### 1. Network Fundamentals

- Graph theory basics
- Network types and properties
- Social network characteristics
- Data collection methods

### 2. Network Analysis Techniques

- **Centrality Measures:**

  - Degree centrality
  - Betweenness centrality
  - Closeness centrality
  - Eigenvector centrality
- **Ranking Algorithms:**

  - PageRank
  - HITS (Hyperlink-Induced Topic Search)
  - Preferential Attachment

### 3. Community Detection

- Louvain method
- Girvan-Newman algorithm
- Modularity optimization
- Clique percolation

### 4. Text Analysis in Social Networks

- TF-IDF implementation
- Topic modeling with LDA
- Sentiment analysis
- Network text visualization

### 5. Practical Applications

- Friend network analysis
- Influence measurement
- Information diffusion
- Recommendation systems

## 🔍 Example Code

### PageRank and HITS Implementation

```python
import networkx as nx

# Create sample graph
G = nx.DiGraph()
G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])

# Calculate PageRank
pagerank = nx.pagerank(G, alpha=0.85)
print("PageRank:", pagerank)

# Calculate HITS
hubs, authorities = nx.hits(G)
print("Hubs:", hubs)
print("Authorities:", authorities)
```

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📫 Contact
