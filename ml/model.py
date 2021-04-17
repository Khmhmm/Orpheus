from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib

n_epoch = 100
path = "tree.model"

''' .csv file example:

Companies,Employers,Vacancies,Graduaters,GRP,Class
0.0217,0.0167,0.1199,0.0574,0.0213,0
'''


def learn_and_save(csv_path):
    tree = DecisionTreeClassifier()
    dataset = pd.read_csv(csv_path, sep=",", names=["Companies","Employers","Vacancies","Graduaters","GRP","Class"], header=0)

    data = list()
    target = list()
    for (com, empl, vac, grad, grp, cl) in zip(dataset["Companies"], dataset["Employers"], dataset["Vacancies"], dataset["Graduaters"], dataset["GRP"], dataset["Class"]):
        data.append([com,empl,vac,grad,grp])
        target.append(cl)

    for _ in range(n_epoch):
        tree.fit( data, target )

    joblib.dump(tree, path)
    return

def load() -> DecisionTreeClassifier:
    return joblib.load(path)
