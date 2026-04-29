## 1 application de l'algorithme de gene cas reel (croisement :"technique de wright"  mutation:"mutation Muchalwicz" Selection = "Davi" )

 ### 1.1 importation du jeux de donnees from kaggle voir le lien ci dessous . 
 <a>https://www.kaggle.com/datasets/yasserh/advertising-sales-dataset/data</a>



```python
import numpy as np
import pandas as pd
DataSet = pd.read_csv(r"C:\Users\PC\Downloads\Advertising Budget and Sales.csv")
DataSet = DataSet.drop(columns='Unnamed: 0')
```


```python
DataSet.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TV Ad Budget ($)</th>
      <th>Radio Ad Budget ($)</th>
      <th>Newspaper Ad Budget ($)</th>
      <th>Sales ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>200.000000</td>
      <td>200.000000</td>
      <td>200.000000</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>147.042500</td>
      <td>23.264000</td>
      <td>30.554000</td>
      <td>14.022500</td>
    </tr>
    <tr>
      <th>std</th>
      <td>85.854236</td>
      <td>14.846809</td>
      <td>21.778621</td>
      <td>5.217457</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.700000</td>
      <td>0.000000</td>
      <td>0.300000</td>
      <td>1.600000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>74.375000</td>
      <td>9.975000</td>
      <td>12.750000</td>
      <td>10.375000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>149.750000</td>
      <td>22.900000</td>
      <td>25.750000</td>
      <td>12.900000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>218.825000</td>
      <td>36.525000</td>
      <td>45.100000</td>
      <td>17.400000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>296.400000</td>
      <td>49.600000</td>
      <td>114.000000</td>
      <td>27.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
DataSet.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TV Ad Budget ($)</th>
      <th>Radio Ad Budget ($)</th>
      <th>Newspaper Ad Budget ($)</th>
      <th>Sales ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>230.1</td>
      <td>37.8</td>
      <td>69.2</td>
      <td>22.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44.5</td>
      <td>39.3</td>
      <td>45.1</td>
      <td>10.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17.2</td>
      <td>45.9</td>
      <td>69.3</td>
      <td>9.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>151.5</td>
      <td>41.3</td>
      <td>58.5</td>
      <td>18.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>180.8</td>
      <td>10.8</td>
      <td>58.4</td>
      <td>12.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = DataSet.iloc[:, :3].values  # 3 colonnes indépendantes
y = DataSet.iloc[:, 3].values   # colonne dépendante
X
```




    array([[230.1,  37.8,  69.2],
           [ 44.5,  39.3,  45.1],
           [ 17.2,  45.9,  69.3],
           [151.5,  41.3,  58.5],
           [180.8,  10.8,  58.4],
           [  8.7,  48.9,  75. ],
           [ 57.5,  32.8,  23.5],
           [120.2,  19.6,  11.6],
           [  8.6,   2.1,   1. ],
           [199.8,   2.6,  21.2],
           [ 66.1,   5.8,  24.2],
           [214.7,  24. ,   4. ],
           [ 23.8,  35.1,  65.9],
           [ 97.5,   7.6,   7.2],
           [204.1,  32.9,  46. ],
           [195.4,  47.7,  52.9],
           [ 67.8,  36.6, 114. ],
           [281.4,  39.6,  55.8],
           [ 69.2,  20.5,  18.3],
           [147.3,  23.9,  19.1],
           [218.4,  27.7,  53.4],
           [237.4,   5.1,  23.5],
           [ 13.2,  15.9,  49.6],
           [228.3,  16.9,  26.2],
           [ 62.3,  12.6,  18.3],
           [262.9,   3.5,  19.5],
           [142.9,  29.3,  12.6],
           [240.1,  16.7,  22.9],
           [248.8,  27.1,  22.9],
           [ 70.6,  16. ,  40.8],
           [292.9,  28.3,  43.2],
           [112.9,  17.4,  38.6],
           [ 97.2,   1.5,  30. ],
           [265.6,  20. ,   0.3],
           [ 95.7,   1.4,   7.4],
           [290.7,   4.1,   8.5],
           [266.9,  43.8,   5. ],
           [ 74.7,  49.4,  45.7],
           [ 43.1,  26.7,  35.1],
           [228. ,  37.7,  32. ],
           [202.5,  22.3,  31.6],
           [177. ,  33.4,  38.7],
           [293.6,  27.7,   1.8],
           [206.9,   8.4,  26.4],
           [ 25.1,  25.7,  43.3],
           [175.1,  22.5,  31.5],
           [ 89.7,   9.9,  35.7],
           [239.9,  41.5,  18.5],
           [227.2,  15.8,  49.9],
           [ 66.9,  11.7,  36.8],
           [199.8,   3.1,  34.6],
           [100.4,   9.6,   3.6],
           [216.4,  41.7,  39.6],
           [182.6,  46.2,  58.7],
           [262.7,  28.8,  15.9],
           [198.9,  49.4,  60. ],
           [  7.3,  28.1,  41.4],
           [136.2,  19.2,  16.6],
           [210.8,  49.6,  37.7],
           [210.7,  29.5,   9.3],
           [ 53.5,   2. ,  21.4],
           [261.3,  42.7,  54.7],
           [239.3,  15.5,  27.3],
           [102.7,  29.6,   8.4],
           [131.1,  42.8,  28.9],
           [ 69. ,   9.3,   0.9],
           [ 31.5,  24.6,   2.2],
           [139.3,  14.5,  10.2],
           [237.4,  27.5,  11. ],
           [216.8,  43.9,  27.2],
           [199.1,  30.6,  38.7],
           [109.8,  14.3,  31.7],
           [ 26.8,  33. ,  19.3],
           [129.4,   5.7,  31.3],
           [213.4,  24.6,  13.1],
           [ 16.9,  43.7,  89.4],
           [ 27.5,   1.6,  20.7],
           [120.5,  28.5,  14.2],
           [  5.4,  29.9,   9.4],
           [116. ,   7.7,  23.1],
           [ 76.4,  26.7,  22.3],
           [239.8,   4.1,  36.9],
           [ 75.3,  20.3,  32.5],
           [ 68.4,  44.5,  35.6],
           [213.5,  43. ,  33.8],
           [193.2,  18.4,  65.7],
           [ 76.3,  27.5,  16. ],
           [110.7,  40.6,  63.2],
           [ 88.3,  25.5,  73.4],
           [109.8,  47.8,  51.4],
           [134.3,   4.9,   9.3],
           [ 28.6,   1.5,  33. ],
           [217.7,  33.5,  59. ],
           [250.9,  36.5,  72.3],
           [107.4,  14. ,  10.9],
           [163.3,  31.6,  52.9],
           [197.6,   3.5,   5.9],
           [184.9,  21. ,  22. ],
           [289.7,  42.3,  51.2],
           [135.2,  41.7,  45.9],
           [222.4,   4.3,  49.8],
           [296.4,  36.3, 100.9],
           [280.2,  10.1,  21.4],
           [187.9,  17.2,  17.9],
           [238.2,  34.3,   5.3],
           [137.9,  46.4,  59. ],
           [ 25. ,  11. ,  29.7],
           [ 90.4,   0.3,  23.2],
           [ 13.1,   0.4,  25.6],
           [255.4,  26.9,   5.5],
           [225.8,   8.2,  56.5],
           [241.7,  38. ,  23.2],
           [175.7,  15.4,   2.4],
           [209.6,  20.6,  10.7],
           [ 78.2,  46.8,  34.5],
           [ 75.1,  35. ,  52.7],
           [139.2,  14.3,  25.6],
           [ 76.4,   0.8,  14.8],
           [125.7,  36.9,  79.2],
           [ 19.4,  16. ,  22.3],
           [141.3,  26.8,  46.2],
           [ 18.8,  21.7,  50.4],
           [224. ,   2.4,  15.6],
           [123.1,  34.6,  12.4],
           [229.5,  32.3,  74.2],
           [ 87.2,  11.8,  25.9],
           [  7.8,  38.9,  50.6],
           [ 80.2,   0. ,   9.2],
           [220.3,  49. ,   3.2],
           [ 59.6,  12. ,  43.1],
           [  0.7,  39.6,   8.7],
           [265.2,   2.9,  43. ],
           [  8.4,  27.2,   2.1],
           [219.8,  33.5,  45.1],
           [ 36.9,  38.6,  65.6],
           [ 48.3,  47. ,   8.5],
           [ 25.6,  39. ,   9.3],
           [273.7,  28.9,  59.7],
           [ 43. ,  25.9,  20.5],
           [184.9,  43.9,   1.7],
           [ 73.4,  17. ,  12.9],
           [193.7,  35.4,  75.6],
           [220.5,  33.2,  37.9],
           [104.6,   5.7,  34.4],
           [ 96.2,  14.8,  38.9],
           [140.3,   1.9,   9. ],
           [240.1,   7.3,   8.7],
           [243.2,  49. ,  44.3],
           [ 38. ,  40.3,  11.9],
           [ 44.7,  25.8,  20.6],
           [280.7,  13.9,  37. ],
           [121. ,   8.4,  48.7],
           [197.6,  23.3,  14.2],
           [171.3,  39.7,  37.7],
           [187.8,  21.1,   9.5],
           [  4.1,  11.6,   5.7],
           [ 93.9,  43.5,  50.5],
           [149.8,   1.3,  24.3],
           [ 11.7,  36.9,  45.2],
           [131.7,  18.4,  34.6],
           [172.5,  18.1,  30.7],
           [ 85.7,  35.8,  49.3],
           [188.4,  18.1,  25.6],
           [163.5,  36.8,   7.4],
           [117.2,  14.7,   5.4],
           [234.5,   3.4,  84.8],
           [ 17.9,  37.6,  21.6],
           [206.8,   5.2,  19.4],
           [215.4,  23.6,  57.6],
           [284.3,  10.6,   6.4],
           [ 50. ,  11.6,  18.4],
           [164.5,  20.9,  47.4],
           [ 19.6,  20.1,  17. ],
           [168.4,   7.1,  12.8],
           [222.4,   3.4,  13.1],
           [276.9,  48.9,  41.8],
           [248.4,  30.2,  20.3],
           [170.2,   7.8,  35.2],
           [276.7,   2.3,  23.7],
           [165.6,  10. ,  17.6],
           [156.6,   2.6,   8.3],
           [218.5,   5.4,  27.4],
           [ 56.2,   5.7,  29.7],
           [287.6,  43. ,  71.8],
           [253.8,  21.3,  30. ],
           [205. ,  45.1,  19.6],
           [139.5,   2.1,  26.6],
           [191.1,  28.7,  18.2],
           [286. ,  13.9,   3.7],
           [ 18.7,  12.1,  23.4],
           [ 39.5,  41.1,   5.8],
           [ 75.5,  10.8,   6. ],
           [ 17.2,   4.1,  31.6],
           [166.8,  42. ,   3.6],
           [149.7,  35.6,   6. ],
           [ 38.2,   3.7,  13.8],
           [ 94.2,   4.9,   8.1],
           [177. ,   9.3,   6.4],
           [283.6,  42. ,  66.2],
           [232.1,   8.6,   8.7]])



#### Etape 1 : Initialisation de la population (les individus represente un vecteur aleatoire entre 10 et -10 qui est la pente de la droite de regression w)


```python
borne_max=10
borne_min=-10
taille_population=30
nb_variables=3 #Puisque ona un vecteur de 3 element et en traite la regression multiple avec trois variable 
population = np.random.uniform(borne_min, borne_max, (taille_population, nb_variables))
population 
```




    array([[-7.58728258, -0.78442464, -5.87332563],
           [-2.71460278,  0.06834542,  3.80789657],
           [-9.2137572 ,  5.98820798,  2.55800779],
           [-8.36481936,  7.47157248,  8.41744801],
           [-8.7784408 , -4.46244704,  6.1240256 ],
           [ 4.96519381, -6.30957961, -5.81301353],
           [-2.59055794, -0.3095403 ,  2.36509543],
           [-2.62172721, -0.74930568,  4.94941876],
           [-9.26633594, -4.95126111,  4.26699172],
           [ 7.90413675,  0.23354884,  0.64226971],
           [-7.85655977, -1.05175266,  0.65234533],
           [-5.15058993, -4.61513538, -2.45431674],
           [-9.59857604, -3.55841669, -5.77103986],
           [-3.45005296, -7.60475736,  7.81054561],
           [ 1.87184907,  3.58204638,  5.78342477],
           [-0.03115602, -8.26159424,  0.74213084],
           [ 1.73682236,  4.90878948, -1.36680908],
           [-7.44839394, -4.32448188, -2.73835407],
           [ 2.91834483,  1.41556609, -2.87806548],
           [ 9.73030498,  2.11549639, -5.25546417],
           [-7.96435055, -6.94281722, -5.08084543],
           [-6.78637253, -6.26865952, -4.29809663],
           [-6.53252809,  7.93530849, -8.39532509],
           [ 0.49022779, -1.79206346,  9.64757234],
           [-7.75922196, -2.04288802,  9.38940867],
           [ 7.31014252,  6.34144142, -4.84194346],
           [-6.58224825,  3.3728644 ,  8.58751978],
           [ 1.13525786,  1.43225379, -4.40041813],
           [ 5.38985866, -6.25912503, -3.52641527],
           [-1.49127123,  0.15220757, -5.15180535]])



### Etape 2 : Selection des individus pour faire le croisement en utilisant la methode Davis 

#### Etape 2 : 1 implimentation de la fonction fitness 


```python
def fitness_population(population, X, y,N_individu):

    
    fitness = np.zeros(N_individu)

    for i in range(N_individu):
        W = population[i]
        y_pred = X @ W  # produit scalaire (prédiction)
        erreur = np.mean(np.abs(y_pred - y))  # MAE
        fitness[i] = -erreur  # fitness = -erreur en prend l'inverse pour  On transforme le problème de minimisation en problème de maximisation 
    
    return fitness
```

#### Etape 2 : 2 implimentation de la fonction selection par davis


```python
def selection_davis(population, X, y, alpha=1.3, nb_selection=None):

    N = population.shape[0]
    if nb_selection is None:
        nb_selection = N

    # Étape 1 : calculer la fitness de chaque individu
    fitness = fitness_population(population, X, y,30)

    # Étape 2 : trier les individus par fitness décroissante
    indices_tries = np.argsort(fitness)[::-1]
    population_triee = population[indices_tries]

    # Étape 3 : calcul des poids relatifs Rw_i = (n - i)^alpha
    rangs = np.arange(N)
    Rw = (N - rangs) ** alpha

    # Étape 4 : probabilités de reproduction
    Pr = Rw / np.sum(Rw)

    # Étape 5 : sélection pondérée
    indices_selectionnes = np.random.choice(N, size=nb_selection, p=Pr)
    population_selectionnee = population_triee[indices_selectionnes]

    return population_selectionnee

# Sélection selon Davis
population_sel = selection_davis(population, X, y, alpha=1.3)
print("Nouvelle population sélectionnée :\n", population_sel)
```

    Nouvelle population sélectionnée :
     [[-0.03115602 -8.26159424  0.74213084]
     [ 4.96519381 -6.30957961 -5.81301353]
     [ 1.73682236  4.90878948 -1.36680908]
     [-3.45005296 -7.60475736  7.81054561]
     [-2.62172721 -0.74930568  4.94941876]
     [ 2.91834483  1.41556609 -2.87806548]
     [-1.49127123  0.15220757 -5.15180535]
     [ 1.13525786  1.43225379 -4.40041813]
     [ 1.73682236  4.90878948 -1.36680908]
     [-0.03115602 -8.26159424  0.74213084]
     [ 1.13525786  1.43225379 -4.40041813]
     [-7.58728258 -0.78442464 -5.87332563]
     [ 1.73682236  4.90878948 -1.36680908]
     [-5.15058993 -4.61513538 -2.45431674]
     [ 0.49022779 -1.79206346  9.64757234]
     [ 1.87184907  3.58204638  5.78342477]
     [-8.36481936  7.47157248  8.41744801]
     [-3.45005296 -7.60475736  7.81054561]
     [ 2.91834483  1.41556609 -2.87806548]
     [-2.59055794 -0.3095403   2.36509543]
     [-2.71460278  0.06834542  3.80789657]
     [-7.85655977 -1.05175266  0.65234533]
     [-7.75922196 -2.04288802  9.38940867]
     [-6.78637253 -6.26865952 -4.29809663]
     [-0.03115602 -8.26159424  0.74213084]
     [-6.58224825  3.3728644   8.58751978]
     [-9.2137572   5.98820798  2.55800779]
     [-2.62172721 -0.74930568  4.94941876]
     [ 1.13525786  1.43225379 -4.40041813]
     [-6.58224825  3.3728644   8.58751978]]
    

### Etape 3 : Croisement des individus selectionner par la technique de wright 


```python

def croisement_wright(population, pc=0.8):

    N, d = population.shape
    nouvelle_population = []

    # Mélanger la population pour former des paires aléatoires
    indices = np.random.permutation(N)

    for i in range(0, N - 1, 2):
        parent1 = population[indices[i]]
        parent2 = population[indices[i+1]]

        if np.random.rand() < pc:  # verification de la condition de croisement 
            # Croisement de Wright
            beta = np.random.rand()
            enfant1 = beta * parent1 + (1 - beta) * parent2
            enfant2 = beta * parent2 + (1 - beta) * parent1
        else:
            # Pas de croisement : on garde les parents
            enfant1 = parent1.copy()
            enfant2 = parent2.copy()

        nouvelle_population.append(enfant1)
        nouvelle_population.append(enfant2)

    # Si population impaire, ajouter le dernier non croisé
    if N % 2 != 0:
        nouvelle_population.append(population[indices[-1]])

    return np.array(nouvelle_population)
    
# Appliquer le croisement sur la population sélectionnée
population_apres_croisement = croisement_wright(population_sel, pc=0.8)
population_apres_croisement
```




    array([[-4.62344037e+00,  3.55755912e+00,  8.93719243e+00],
           [-3.25115120e+00,  2.12194991e+00,  9.12782792e+00],
           [-4.62919421e+00, -3.71160277e+00, -3.28349277e+00],
           [-2.98924439e+00, -5.33441611e+00, -1.84770202e+00],
           [-7.41489015e+00, -1.92741393e+00,  8.92145515e+00],
           [-2.93488975e+00, -4.25014383e-01,  2.83304894e+00],
           [ 7.02738981e-03, -2.70808099e+00,  2.00824045e+00],
           [-5.38735520e-01, -3.48111028e+00,  2.92423968e+00],
           [-6.64359196e+00,  4.04589492e+00, -7.93973011e-03],
           [-4.06143647e+00,  2.09452063e+00, -2.58585783e+00],
           [ 2.30829981e+00, -9.38933468e-01, -4.83307123e+00],
           [ 3.79215186e+00, -3.93839236e+00, -5.38036043e+00],
           [ 1.64087966e+00,  2.14457324e+00,  5.17155847e+00],
           [ 1.99813390e-01, -6.82412110e+00,  1.35399714e+00],
           [-4.13982576e-01,  4.51169281e+00,  1.20677404e+00],
           [-4.43144332e+00,  3.76996107e+00,  6.01393667e+00],
           [-3.45775113e+00, -7.60167447e+00,  7.78260633e+00],
           [-6.77867436e+00, -6.27174241e+00, -4.27015734e+00],
           [ 1.34485425e-01,  1.98909274e+00,  4.81575024e-01],
           [-3.79838885e+00,  2.79933775e+00,  5.22787928e+00],
           [-2.33052645e+00, -1.12390207e+00,  3.36909970e+00],
           [-4.15232350e-01, -7.06934676e+00,  1.18092771e+00],
           [-2.32536193e-01,  2.18555729e+00, -1.67776658e+00],
           [-3.18123137e+00, -1.89190318e+00, -2.14335924e+00],
           [-1.55229297e+00,  6.38991190e-01,  3.39963936e+00],
           [ 6.67388118e-01,  3.52049262e+00,  1.82970325e-01],
           [ 1.13525786e+00,  1.43225379e+00, -4.40041813e+00],
           [ 1.13525786e+00,  1.43225379e+00, -4.40041813e+00],
           [-5.46424014e+00, -9.13534325e-01,  2.61610883e+00],
           [-5.01404684e+00, -8.87524016e-01,  2.98565526e+00]])



#### Etape 4 : Mutation en utilisant la Mutation de Michalewicz pour assurer la diversity de la ppopulation 


```python
def mutation_michalewicz(population, t, T, xmin, xmax, pm=0.1, b=3):

    N, d = population.shape
    nouvelle_population = population.copy()

    for i in range(N):
        for j in range(d):
            if np.random.rand() < pm:
                r = np.random.rand()
                direction = np.random.choice(["pile", "face"])
                y = xmax[j] - population[i, j] if direction == "pile" else population[i, j] - xmin[j]
                delta = r * y * ((1 - t / T) ** b)
                if direction == "pile":
                    nouvelle_population[i, j] += delta
                else:
                    nouvelle_population[i, j] -= delta

                # Clamping to bounds
                nouvelle_population[i, j] = np.clip(nouvelle_population[i, j], xmin[j], xmax[j])

    return nouvelle_population
# Supposons que nous sommes à la génération 5/50
population_mutée = mutation_michalewicz(population_apres_croisement,
                                         t=5, T=100,
                                         xmin=np.array([-10, -10, -10]),
                                         xmax=np.array([10, 10, 10]),
                                         pm=0.1, b=3)
population_mutée
```




    array([[-4.62344037e+00,  3.55755912e+00,  8.93719243e+00],
           [-3.25115120e+00,  2.12194991e+00,  9.12782792e+00],
           [-7.38489994e+00, -3.71160277e+00, -3.28349277e+00],
           [-2.98924439e+00, -5.33441611e+00, -1.84770202e+00],
           [-7.41489015e+00, -1.92741393e+00,  8.92145515e+00],
           [-2.93488975e+00, -4.25014383e-01,  2.83304894e+00],
           [ 7.02738981e-03, -2.70808099e+00, -6.97076808e+00],
           [-5.38735520e-01, -3.48111028e+00,  2.92423968e+00],
           [ 2.65128764e+00,  4.04589492e+00, -7.93973011e-03],
           [-4.06143647e+00,  2.09452063e+00, -2.58585783e+00],
           [ 2.30829981e+00, -9.38933468e-01, -4.83307123e+00],
           [ 3.79215186e+00, -3.93839236e+00, -5.38036043e+00],
           [ 1.64087966e+00,  2.14457324e+00,  5.17155847e+00],
           [ 1.99813390e-01, -6.82412110e+00, -1.90902509e+00],
           [-4.13982576e-01,  4.51169281e+00,  1.20677404e+00],
           [-4.43144332e+00,  3.76996107e+00,  6.01393667e+00],
           [-3.45775113e+00, -7.60167447e+00,  7.78260633e+00],
           [-6.77867436e+00, -9.18935962e+00, -4.27015734e+00],
           [ 1.34485425e-01,  1.98909274e+00,  4.81575024e-01],
           [-3.79838885e+00,  2.79933775e+00,  5.22787928e+00],
           [ 4.68431442e+00, -1.12390207e+00,  3.36909970e+00],
           [-4.15232350e-01, -7.06934676e+00,  1.18092771e+00],
           [-2.32536193e-01,  2.18555729e+00, -1.67776658e+00],
           [-3.18123137e+00, -1.89190318e+00, -2.14335924e+00],
           [-1.55229297e+00,  6.38991190e-01,  3.39963936e+00],
           [ 6.67388118e-01,  3.52049262e+00,  1.82970325e-01],
           [ 1.13525786e+00,  1.43225379e+00, -4.40041813e+00],
           [ 1.13525786e+00,  1.43225379e+00, -4.40041813e+00],
           [-5.46424014e+00, -9.13534325e-01,  2.61610883e+00],
           [-8.23404758e+00, -8.87524016e-01,  2.98565526e+00]])



#### Etape 5 : Applique toutes les étapes de l’algorithme (initialisation → sélection → croisement → mutation)


```python
def algorithme_genetique_reel(X, y, population,n_generations=100, population_size=30, pc=0.8, pm=0.1, alpha=1.3, b=3):
    
    N, d = X.shape

    # Définir les bornes (à adapter si besoin)
    xmin = np.full(d, -10)
    xmax = np.full(d, 10)

    for t in range(n_generations):
        # Étape 2 : Sélection (Davis)
        population_sel = selection_davis(population, X, y, alpha=alpha)

        # Étape 3 : Croisement (Wright)
        population_cross = croisement_wright(population_sel, pc=pc)

        # Étape 4 : Mutation (Michalewicz)
        population = mutation_michalewicz(population_cross, t, n_generations, xmin, xmax, pm=pm, b=b)

        # Évaluation
        fitness = fitness_population(population, X, y,30)
        best_index = np.argmax(fitness)
        best_fitness = fitness[best_index]
        best_individu = population[best_index]

        print(f"Génération {t+1}/{n_generations} : meilleure fitness = {best_fitness:.6f}, W = {best_individu}")

    # Résultat final
    print("\n✅ Meilleure solution approchée trouvée :")
    print(f"W = {best_individu}")
    print(f"Fitness = {best_fitness:.6f}")


algorithme_genetique_reel(X, y,population)

```

    Génération 1/100 : meilleure fitness = -200.350991, W = [ 1.84179184  1.42564142 -3.79719804]
    Génération 2/100 : meilleure fitness = -48.026335, W = [ 0.51724666  0.90710686 -1.95568361]
    Génération 3/100 : meilleure fitness = -48.026335, W = [ 0.51724666  0.90710686 -1.95568361]
    Génération 4/100 : meilleure fitness = -41.553817, W = [-0.34861082  2.47342274 -0.4222883 ]
    Génération 5/100 : meilleure fitness = -38.478664, W = [ 0.30974377  1.03651988 -2.1162566 ]
    Génération 6/100 : meilleure fitness = -23.655125, W = [ 0.20172879 -0.20492201 -0.90943963]
    Génération 7/100 : meilleure fitness = -11.213937, W = [ 0.05980321 -0.22694769  0.5929761 ]
    Génération 8/100 : meilleure fitness = -11.213937, W = [ 0.05980321 -0.22694769  0.5929761 ]
    Génération 9/100 : meilleure fitness = -7.293024, W = [ 0.05857758 -0.25622052  0.40548015]
    Génération 10/100 : meilleure fitness = -6.220228, W = [ 0.0646713  -0.26414306  0.27100851]
    Génération 11/100 : meilleure fitness = -5.797201, W = [ 0.07215273 -0.23581346  0.24588486]
    Génération 12/100 : meilleure fitness = -5.608445, W = [ 0.0829425  -0.22106955  0.16828064]
    Génération 13/100 : meilleure fitness = -5.764423, W = [ 0.08470854 -0.2254853   0.22709073]
    Génération 14/100 : meilleure fitness = -5.579461, W = [ 0.08189933 -0.21983111  0.21094316]
    Génération 15/100 : meilleure fitness = -5.432815, W = [ 0.07501023 -0.20883605  0.20859539]
    Génération 16/100 : meilleure fitness = -5.406176, W = [ 0.08244409 -0.18643514  0.13106666]
    Génération 17/100 : meilleure fitness = -5.406176, W = [ 0.08244409 -0.18643514  0.13106666]
    Génération 18/100 : meilleure fitness = -4.654527, W = [0.08601062 0.19075053 0.0369647 ]
    Génération 19/100 : meilleure fitness = -3.384336, W = [0.08540513 0.05369836 0.04746789]
    Génération 20/100 : meilleure fitness = -3.384336, W = [0.08540513 0.05369836 0.04746789]
    Génération 21/100 : meilleure fitness = -3.387877, W = [0.08548538 0.06584257 0.04839906]
    Génération 22/100 : meilleure fitness = -3.463628, W = [0.08801494 0.07889456 0.01312494]
    Génération 23/100 : meilleure fitness = -5.050291, W = [ 0.08636601 -0.09721224  0.03608765]
    Génération 24/100 : meilleure fitness = -5.069538, W = [ 0.08641465 -0.09779477  0.03541035]
    Génération 25/100 : meilleure fitness = -5.071854, W = [ 0.0864205  -0.09786486  0.03532886]
    Génération 26/100 : meilleure fitness = -5.071854, W = [ 0.0864205  -0.09786486  0.03532886]
    Génération 27/100 : meilleure fitness = -5.071854, W = [ 0.0864205  -0.09786486  0.03532886]
    Génération 28/100 : meilleure fitness = -4.999056, W = [0.08723482 0.21267625 0.02632004]
    Génération 29/100 : meilleure fitness = -3.471525, W = [0.08680997 0.04892077 0.03103634]
    Génération 30/100 : meilleure fitness = -4.472854, W = [ 0.08647053 -0.06131374  0.0361172 ]
    Génération 31/100 : meilleure fitness = -4.472854, W = [ 0.08647053 -0.06131374  0.0361172 ]
    Génération 32/100 : meilleure fitness = -4.480374, W = [ 0.08647109 -0.06180017  0.0360963 ]
    Génération 33/100 : meilleure fitness = -4.513044, W = [ 0.08647067 -0.06393983  0.03603792]
    Génération 34/100 : meilleure fitness = -4.567801, W = [ 0.08646997 -0.06750637  0.0359406 ]
    Génération 35/100 : meilleure fitness = -4.597222, W = [ 0.0864696  -0.06937935  0.03588949]
    Génération 36/100 : meilleure fitness = -3.418361, W = [0.08644069 0.08867334 0.03599869]
    Génération 37/100 : meilleure fitness = -3.402071, W = [0.08646727 0.06822925 0.0359029 ]
    Génération 38/100 : meilleure fitness = -3.398096, W = [0.08645961 0.0741232  0.03593051]
    Génération 39/100 : meilleure fitness = -3.396925, W = [0.08644024 0.07289829 0.03599463]
    Génération 40/100 : meilleure fitness = -3.400905, W = [0.08644092 0.06790292 0.03602753]
    Génération 41/100 : meilleure fitness = -3.396848, W = [0.08643893 0.07267389 0.03599828]
    Génération 42/100 : meilleure fitness = -3.396848, W = [0.08643893 0.07267389 0.03599828]
    Génération 43/100 : meilleure fitness = -3.396848, W = [0.08643893 0.07267389 0.03599828]
    Génération 44/100 : meilleure fitness = -3.398551, W = [0.08644012 0.06980578 0.03601586]
    Génération 45/100 : meilleure fitness = -3.398632, W = [0.08644016 0.06972085 0.03601638]
    Génération 46/100 : meilleure fitness = -3.399646, W = [0.08644038 0.06879609 0.03601937]
    Génération 47/100 : meilleure fitness = -3.403211, W = [0.08644041 0.06623086 0.03601887]
    Génération 48/100 : meilleure fitness = -3.403670, W = [0.08644036 0.0659034  0.03601901]
    Génération 49/100 : meilleure fitness = -3.404192, W = [0.08644037 0.06553377 0.03601885]
    Génération 50/100 : meilleure fitness = -3.396528, W = [0.08644029 0.06398444 0.02675456]
    Génération 51/100 : meilleure fitness = -3.172512, W = [0.06819563 0.064312   0.0192312 ]
    Génération 52/100 : meilleure fitness = -3.054297, W = [0.07016831 0.06441859 0.02104631]
    Génération 53/100 : meilleure fitness = -2.952903, W = [0.07539585 0.0645307  0.02343354]
    Génération 54/100 : meilleure fitness = -2.958929, W = [0.07655829 0.06456236 0.02441434]
    Génération 55/100 : meilleure fitness = -2.854077, W = [0.07926965 0.1319392  0.02617382]
    Génération 56/100 : meilleure fitness = -2.840995, W = [0.07939478 0.12489131 0.02623416]
    Génération 57/100 : meilleure fitness = -2.900106, W = [0.08001183 0.11744949 0.02992955]
    Génération 58/100 : meilleure fitness = -2.897467, W = [0.08001439 0.11646877 0.02985988]
    Génération 59/100 : meilleure fitness = -2.808218, W = [0.07025516 0.06926286 0.03209879]
    Génération 60/100 : meilleure fitness = -2.744849, W = [0.07305056 0.06589939 0.0433285 ]
    Génération 61/100 : meilleure fitness = -2.703264, W = [0.07134646 0.07760898 0.03074705]
    Génération 62/100 : meilleure fitness = -2.744849, W = [0.07305056 0.06589939 0.0433285 ]
    Génération 63/100 : meilleure fitness = -2.744849, W = [0.07305056 0.06589939 0.0433285 ]
    Génération 64/100 : meilleure fitness = -2.635806, W = [0.0738506  0.1593341  0.03321252]
    Génération 65/100 : meilleure fitness = -2.454221, W = [ 0.07469483  0.17292621 -0.03183348]
    Génération 66/100 : meilleure fitness = -2.401618, W = [ 0.07393914  0.19063592 -0.03697795]
    Génération 67/100 : meilleure fitness = -2.409881, W = [ 0.0741943   0.17855431 -0.03156451]
    Génération 68/100 : meilleure fitness = -2.393847, W = [ 0.07424156  0.16426434 -0.0145308 ]
    Génération 69/100 : meilleure fitness = -2.393847, W = [ 0.07424156  0.16426434 -0.0145308 ]
    Génération 70/100 : meilleure fitness = -2.393847, W = [ 0.07424156  0.16426434 -0.0145308 ]
    Génération 71/100 : meilleure fitness = -2.229497, W = [0.04815633 0.17190759 0.1061253 ]
    Génération 72/100 : meilleure fitness = -1.847765, W = [0.06191212 0.16928079 0.04031694]
    Génération 73/100 : meilleure fitness = -1.836515, W = [0.06037127 0.17040695 0.04602155]
    Génération 74/100 : meilleure fitness = -1.831142, W = [0.06191943 0.17273039 0.03727318]
    Génération 75/100 : meilleure fitness = -1.849398, W = [0.06327079 0.17360785 0.03009248]
    Génération 76/100 : meilleure fitness = -1.950834, W = [0.06686073 0.178888   0.00879062]
    Génération 77/100 : meilleure fitness = -1.952408, W = [0.06689381 0.1787661  0.00866366]
    Génération 78/100 : meilleure fitness = -1.952408, W = [0.06689381 0.1787661  0.00866366]
    Génération 79/100 : meilleure fitness = -1.939960, W = [ 0.06707332  0.18921125 -0.00604176]
    Génération 80/100 : meilleure fitness = -1.934591, W = [0.0625647  0.17092721 0.00058595]
    Génération 81/100 : meilleure fitness = -1.831599, W = [0.06444378 0.18138323 0.00557373]
    Génération 82/100 : meilleure fitness = -1.625481, W = [0.05983473 0.20542385 0.00455468]
    Génération 83/100 : meilleure fitness = -1.764134, W = [0.06272158 0.18814895 0.0024404 ]
    Génération 84/100 : meilleure fitness = -1.764134, W = [0.06272158 0.18814895 0.0024404 ]
    Génération 85/100 : meilleure fitness = -1.757149, W = [0.06284804 0.19057747 0.00202154]
    Génération 86/100 : meilleure fitness = -1.757149, W = [0.06284804 0.19057747 0.00202154]
    Génération 87/100 : meilleure fitness = -1.757149, W = [0.06284804 0.19057747 0.00202154]
    Génération 88/100 : meilleure fitness = -1.757149, W = [0.06284804 0.19057747 0.00202154]
    Génération 89/100 : meilleure fitness = -1.757149, W = [0.06284804 0.19057747 0.00202154]
    Génération 90/100 : meilleure fitness = -1.731352, W = [0.06138765 0.19099159 0.00215911]
    Génération 91/100 : meilleure fitness = -1.731352, W = [0.06138765 0.19099159 0.00215911]
    Génération 92/100 : meilleure fitness = -1.731631, W = [0.06145057 0.19098083 0.00217505]
    Génération 93/100 : meilleure fitness = -1.696745, W = [0.05855621 0.19455601 0.00608161]
    Génération 94/100 : meilleure fitness = -1.694388, W = [0.05925757 0.19375466 0.00574052]
    Génération 95/100 : meilleure fitness = -1.696425, W = [0.06000038 0.19290593 0.00537927]
    Génération 96/100 : meilleure fitness = -1.694817, W = [0.06001049 0.193141   0.00534129]
    Génération 97/100 : meilleure fitness = -1.694817, W = [0.06001049 0.193141   0.00534129]
    Génération 98/100 : meilleure fitness = -1.696224, W = [0.06000217 0.19295198 0.00535542]
    Génération 99/100 : meilleure fitness = -1.696614, W = [0.06000575 0.19306949 0.0051644 ]
    Génération 100/100 : meilleure fitness = -1.697555, W = [0.06005339 0.19300142 0.00506805]
    
    ✅ Meilleure solution approchée trouvée :
    W = [0.06005339 0.19300142 0.00506805]
    Fitness = -1.697555
    

## 2 application de l'algorithme de gene cas reel Methode simple comme l'Exercice 2 Examen 


```python
import numpy as np

# ---- Fonction d'évaluation (fitness) : erreur absolue ----
def fitness(w, x, t):
    y_pred = np.dot(x, w)
    return np.abs(y_pred - t).sum()  # Erreur absolue totale

# ---- Initialisation de la population ----
def initialiser_population(taille_pop, dimension, borne_min=-5, borne_max=5):
    return np.random.randint(borne_min, borne_max + 1, size=(taille_pop, dimension))

# ---- Sélection : garder les meilleurs individus (plus faible erreur) ----
def selection(population, x, t, nb_selection):
    scores = np.array([fitness(ind, x, t) for ind in population])
    indices = np.argsort(scores)
    return population[indices[:nb_selection]]

# ---- Croisement par pivot ----
def croisement(parent1, parent2):
    point_pivot = np.random.randint(1, len(parent1))
    enfant1 = np.concatenate((parent1[:point_pivot], parent2[point_pivot:]))
    enfant2 = np.concatenate((parent2[:point_pivot], parent1[point_pivot:]))
    return enfant1, enfant2

# ---- Mutation aléatoire ----
def mutation(individu, prob_mutation=0.1, borne_min=-5, borne_max=5):
    for i in range(len(individu)):
        if np.random.rand() < prob_mutation:
            individu[i] = np.random.randint(borne_min, borne_max + 1)
    return individu

# ---- Algorithme génétique principal ----
def algorithme_genetique(x, t, generations=20, taille_pop=10, nb_selection=4, prob_mutation=0.1):
    dimension = x.shape[1]
    population = initialiser_population(taille_pop, dimension)

    for gen in range(generations):
        print(f"\n🧬 Génération {gen + 1}")
        
        # Évaluer la population
        scores = [fitness(ind, x, t) for ind in population]
        for i, (ind, score) in enumerate(zip(population, scores)):
            print(f" Individu {i} : {ind} -> Erreur = {score:.4f}")
        
        # Sélection des meilleurs
        parents = selection(population, x, t, nb_selection)

        # Croisement
        enfants = []
        while len(enfants) < taille_pop - nb_selection:
            p1, p2 = parents[np.random.randint(0, nb_selection)], parents[np.random.randint(0, nb_selection)]
            e1, e2 = croisement(p1, p2)
            enfants.append(e1)
            if len(enfants) < taille_pop - nb_selection:
                enfants.append(e2)

        # Mutation
        enfants = [mutation(ind, prob_mutation) for ind in enfants]

        # Nouvelle population = parents + enfants
        population = np.vstack((parents, enfants))

    # Évaluer la population finale
    scores = [fitness(ind, x, t) for ind in population]
    meilleur_indice = np.argmin(scores)
    meilleur_individu = population[meilleur_indice]
    print(f"\n✅ Meilleure solution trouvée : {meilleur_individu}")
    print(f"   Erreur totale : {scores[meilleur_indice]:.4f}")
    return meilleur_individu
# Exemple fictif
x = np.array([[4, -2, 7, 7, 11, 1]])
t = np.array([64.2])

# Appel de l'algorithme
w_opt = algorithme_genetique(x, t, generations=10)
```

    
    🧬 Génération 1
     Individu 0 : [ 2 -4 -1 -4 -1  1] -> Erreur = 93.2000
     Individu 1 : [ 0 -1 -2  2  5  2] -> Erreur = 5.2000
     Individu 2 : [-5  4 -4 -3 -5  5] -> Erreur = 191.2000
     Individu 3 : [-3  4  2 -2  1  1] -> Erreur = 72.2000
     Individu 4 : [ 5 -1 -5  4 -2  1] -> Erreur = 70.2000
     Individu 5 : [-1 -4  2 -4  5  2] -> Erreur = 17.2000
     Individu 6 : [ 5 -4  5 -1 -2 -5] -> Erreur = 35.2000
     Individu 7 : [ 2  4 -3 -2 -2 -1] -> Erreur = 122.2000
     Individu 8 : [-3  3 -2  4  5  5] -> Erreur = 8.2000
     Individu 9 : [-2  0  2  1 -2 -2] -> Erreur = 75.2000
    
    🧬 Génération 2
     Individu 0 : [ 0 -1 -2  2  5  2] -> Erreur = 5.2000
     Individu 1 : [-3  3 -2  4  5  5] -> Erreur = 8.2000
     Individu 2 : [-1 -4  2 -4  5  2] -> Erreur = 17.2000
     Individu 3 : [ 5 -4  5 -1 -2 -5] -> Erreur = 35.2000
     Individu 4 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 5 : [-3  3 -2  4  5  2] -> Erreur = 11.2000
     Individu 6 : [ 5 -5  5  4 -2  0] -> Erreur = 6.8000
     Individu 7 : [ 0 -1 -2 -2  5 -5] -> Erreur = 40.2000
     Individu 8 : [ 5 -4  5 -1 -2 -5] -> Erreur = 35.2000
     Individu 9 : [-3  3 -2  4  5 -5] -> Erreur = 18.2000
    
    🧬 Génération 3
     Individu 0 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 1 : [ 0 -1 -2  2  5  2] -> Erreur = 5.2000
     Individu 2 : [ 5 -5  5  4 -2  0] -> Erreur = 6.8000
     Individu 3 : [-3  3 -2  4  5  5] -> Erreur = 8.2000
     Individu 4 : [-3  3 -2  4  5 -3] -> Erreur = 16.2000
     Individu 5 : [-3  3 -2  4  5  5] -> Erreur = 8.2000
     Individu 6 : [ 0 -1 -2  2  5  2] -> Erreur = 5.2000
     Individu 7 : [ 0 -1 -2  2  5  2] -> Erreur = 5.2000
     Individu 8 : [ 0 -1 -2  2  5  2] -> Erreur = 5.2000
     Individu 9 : [ 0 -1 -2 -2  5 -4] -> Erreur = 39.2000
    
    🧬 Génération 4
     Individu 0 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 1 : [ 0 -1 -2  2  5  2] -> Erreur = 5.2000
     Individu 2 : [ 0 -1 -2  2  5  2] -> Erreur = 5.2000
     Individu 3 : [ 0 -1 -2  2  5  2] -> Erreur = 5.2000
     Individu 4 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 5 : [ 0 -1 -2  2  3  5] -> Erreur = 24.2000
     Individu 6 : [ 0 -1 -2  2  1  2] -> Erreur = 49.2000
     Individu 7 : [ 0 -1 -2  2  5  2] -> Erreur = 5.2000
     Individu 8 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 9 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
    
    🧬 Génération 5
     Individu 0 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 1 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 2 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 3 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 4 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
     Individu 5 : [ 0 -1  2  2  5  5] -> Erreur = 25.8000
     Individu 6 : [ 0 -1 -2  2 -1  5] -> Erreur = 68.2000
     Individu 7 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 8 : [ 0 -1 -2  2  5  1] -> Erreur = 6.2000
     Individu 9 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
    
    🧬 Génération 6
     Individu 0 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
     Individu 1 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 2 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 3 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 4 : [ 0 -1 -2  2  5  4] -> Erreur = 3.2000
     Individu 5 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 6 : [ 0 -1 -2  2  5  1] -> Erreur = 6.2000
     Individu 7 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 8 : [ 4  3 -2  2  5  5] -> Erreur = 5.8000
     Individu 9 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
    
    🧬 Génération 7
     Individu 0 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
     Individu 1 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 2 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 3 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 4 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 5 : [ 4 -1 -2  2  5 -1] -> Erreur = 7.8000
     Individu 6 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 7 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
     Individu 8 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 9 : [ 0 -2 -2  2 -5  5] -> Erreur = 110.2000
    
    🧬 Génération 8
     Individu 0 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
     Individu 1 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
     Individu 2 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 3 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 4 : [ 0 -2 -2  2  5 -1] -> Erreur = 6.2000
     Individu 5 : [ 0 -2 -2  2  0  5] -> Erreur = 55.2000
     Individu 6 : [ 0 -2 -2  2  5 -4] -> Erreur = 9.2000
     Individu 7 : [ 0 -2 -2  2  5  3] -> Erreur = 2.2000
     Individu 8 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
     Individu 9 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
    
    🧬 Génération 9
     Individu 0 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
     Individu 1 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
     Individu 2 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
     Individu 3 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 4 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
     Individu 5 : [ 0 -2 -2  2  5  1] -> Erreur = 4.2000
     Individu 6 : [ 0 -2 -2  2  5 -1] -> Erreur = 6.2000
     Individu 7 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 8 : [ 0 -1 -2  2  5  5] -> Erreur = 2.2000
     Individu 9 : [ 0 -4 -2  2  5  5] -> Erreur = 3.8000
    
    🧬 Génération 10
     Individu 0 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
     Individu 1 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
     Individu 2 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
     Individu 3 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
     Individu 4 : [ 0 -2  3  2  5  5] -> Erreur = 34.8000
     Individu 5 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
     Individu 6 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
     Individu 7 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
     Individu 8 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
     Individu 9 : [ 0 -2 -2  2  5  5] -> Erreur = 0.2000
    
    ✅ Meilleure solution trouvée : [ 0 -2 -2  2  5  5]
       Erreur totale : 0.2000
    

#### Conclusion 

Méthode 1 (Wright + Michalewicz + Davi) est clairement plus robuste, surtout pour les problèmes réels à variables continues. La descente de fitness montre une bonne exploitation après une phase d’exploration. La stagnation est naturelle vers la fin (proche d’un optimum local ou global).

Méthode 2 montre que l’algorithme génétique est très efficace même en version simple, à condition que :
Le codage soit adapté.


```python
