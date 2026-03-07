from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import warnings

# Sklearn imports
from sklearn.datasets import make_circles, make_moons, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import uvicorn
import os

warnings.filterwarnings('ignore')

app = FastAPI()

# Servir el archivo index.html en la ruta raíz
@app.get("/", response_class=HTMLResponse)
async def read_index():
    # Intenta leer el archivo index.html local
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html no encontrado en el servidor.</h1>"

def make_spirals(n_samples, noise):
    n = n_samples // 2
    theta = np.sqrt(np.random.rand(n)) * 2 * np.pi
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + noise * np.random.randn(n, 2)
    
    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + noise * np.random.randn(n, 2)
    
    res_a = np.append(x_a, np.zeros((n, 1)), axis=1)
    res_b = np.append(x_b, np.ones((n, 1)), axis=1)
    res = np.vstack((res_a, res_b))
    np.random.shuffle(res)
    return res[:, :2], res[:, 2]

class DataReq(BaseModel):
    dataset: str
    n_samples: int
    noise: float

@app.post("/api/generate_data")
def generate_data(req: DataReq):
    if req.dataset == 'circles':
        X, y = make_circles(n_samples=req.n_samples, noise=req.noise, factor=0.5)
    elif req.dataset == 'moons':
        X, y = make_moons(n_samples=req.n_samples, noise=req.noise)
    elif req.dataset == 'spirals':
        X, y = make_spirals(req.n_samples, req.noise)
    else: # lineal (blobs)
        X, y = make_classification(n_samples=req.n_samples, n_features=2, n_redundant=0, n_informative=2,
                                   n_clusters_per_class=1, class_sep=2.0 - req.noise*1.5)
    
    # Scale coordinates to fit roughly in canvas [-2.5, 2.5]
    if len(X) > 0:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
    ds = [{"x": [float(X[i,0]), float(X[i,1])], "y": int(y[i])} for i in range(len(y))]
    return {"dataset": ds}

class GridP(BaseModel):
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    res_x: int
    res_y: int

class TrainReq(BaseModel):
    X: List[List[float]]
    y: List[int]
    algo: str
    params: dict
    grid: GridP

@app.post("/api/train")
def train_model(req: TrainReq):
    X = np.array(req.X)
    y = np.array(req.y)
    
    if len(np.unique(y)) < 2:
        return {"error": "Dataset must have 2 classes"}

    # Initialize model
    if req.algo == 'lr':
        model = LogisticRegression(max_iter=200)
    elif req.algo == 'knn':
        model = KNeighborsClassifier(n_neighbors=int(req.params.get('knn_k', 5)))
    elif req.algo == 'dt':
        model = DecisionTreeClassifier(max_depth=int(req.params.get('dt_depth', 5)))
    elif req.algo == 'rf':
        model = RandomForestClassifier(n_estimators=int(req.params.get('rf_trees', 50)),
                                       max_depth=int(req.params.get('rf_depth', 5)))
    elif req.algo == 'svm':
        model = SVC(gamma=float(req.params.get('svm_gamma', 1.0)), probability=True)
    elif req.algo == 'nn':
        layers = int(req.params.get('nn_layers', 2))
        neurons = int(req.params.get('nn_neurons', 8))
        lr = float(req.params.get('nn_lr', 0.01))
        model = MLPClassifier(hidden_layer_sizes=tuple([neurons]*layers),
                              learning_rate_init=lr, max_iter=200)
    elif req.algo == 'nb':
        model = GaussianNB()
    elif req.algo == 'gb':
        model = GradientBoostingClassifier(n_estimators=int(req.params.get('gb_trees', 100)),
                                           learning_rate=float(req.params.get('gb_lr', 0.1)),
                                           max_depth=int(req.params.get('gb_depth', 3)))
    else:
        model = LogisticRegression()
    
    # Train
    model.fit(X, y)
    
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else preds
    
    tp, fp, fn, tn = 0, 0, 0, 0
    misclassified = []
    
    for i, (true_y, pred_y) in enumerate(zip(y, preds)):
        if true_y == 1 and pred_y == 1:
            tp += 1
        elif true_y == 0 and pred_y == 1:
            fp += 1
            misclassified.append(i)
        elif true_y == 1 and pred_y == 0:
            fn += 1
            misclassified.append(i)
        elif true_y == 0 and pred_y == 0:
            tn += 1

    acc = accuracy_score(y, preds)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    
    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = 0.50
    
    # Calculate grid for background colors (decision boundary)
    g = req.grid
    yy = np.linspace(g.y_max, g.y_min, g.res_y)
    xx = np.linspace(g.x_min, g.x_max, g.res_x)
    XX, YY = np.meshgrid(xx, yy)
    
    grid_points = np.c_[XX.ravel(), YY.ravel()]
    grid_probs = model.predict_proba(grid_points)[:, 1] if hasattr(model, "predict_proba") else model.predict(grid_points)
    
    return {
        "grid_probs": grid_probs.tolist(),
        "metrics": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "acc": float(acc), "rec": float(rec), "f1": float(f1), "auc": float(auc)
        },
        "misclassified_indices": misclassified
    }

@app.get("/")
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
