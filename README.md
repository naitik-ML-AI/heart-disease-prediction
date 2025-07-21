# Heart Disease Prediction

A compact overview of the repository, focused on the essentials.

## Project in One Line
Python pipeline that downloads the UCI Heart-Disease dataset, trains six classic ML models, compares their accuracy, and lets you predict heart-disease risk from new patient data.

## Quick Start

```bash
# 1. clone and enter the repo
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# 2. install requirements
pip install -r requirements.txt          # or: pip install pandas numpy matplotlib seaborn scikit-learn

# 3. run the script
python heart_disease_prediction.py       # generates plots & prints model metrics
```

## Dataset
UCI Heart-Disease dataset (Cleveland clinic subset).  
Direct CSV link: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data 

## Models & Typical Accuracy
| Model               | Typical Range |
|---------------------|---------------|
| Random Forest       | 85-95% |
| Logistic Regression | 80-90% |
| SVM (RBF)           | 82-92% |
| K-Nearest Neighbors | 75-85% |
| Decision Tree       | 75-85% |
| Naïve Bayes         | 80-88% |

*(Exact results vary with seed and hyper-parameters.)*

## File Structure
```
heart_disease_prediction.py   # main script
heart_disease_prediction.ipynb # optional notebook
README.md                     # this file
requirements.txt              # libs
```

## Contact
- **Name:** Naitik Sharma  
- **Email:** naitik28sharma@gmail.com  
- **LinkedIn:** www.linkedin.com/in/naitik-sharma-54627335a  

### License
MIT — feel free to fork, star, and contribute.
