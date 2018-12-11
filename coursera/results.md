<p>Ik heb de QUIZ "Octave/Matlab Tutorial" van WEEK 2 niet gedaan omdat het niet hoefde maar ik heb wel gespeeld met de COST FUNCTION in PYTHON, zie script helemaal onderaan deze pagina.</p>

<p>Week 1</p>
<img src="https://github.com/ciCciC/Aphasia-portfolio/blob/master/coursera/week1.png"
alt="drawing" width="600" height="400"/>

<p>Week 2</p>
<img src="https://github.com/ciCciC/Aphasia-portfolio/blob/master/coursera/week2.png"
alt="drawing" width="600" height="400"/>

<p>Week 3</p>
<img src="https://github.com/ciCciC/Aphasia-portfolio/blob/master/coursera/week3.png"
alt="drawing" width="600" height="400"/>

<p>Week 6</p>
<img src="https://github.com/ciCciC/Aphasia-portfolio/blob/master/coursera/week6.png"
alt="drawing" width="600" height="400"/>

<p>Script Cost Function</p>

```javascript
X = np.array([[1,1],[1,2],[1,3],[1,4]])
print('X: ' + str(X))

y = np.array([1,2,3,4])
print('y: ' + str(y))

theta = np.array([0, 1])
print('theta: ' + str(theta))


def h(theta, X):
    prediction = np.matmul(X, theta)
    return np.transpose(prediction)


def costfunction(X, y, theta):
    # Dit is hetzelfde als costfunctionAnders maar anders geprogrammeerd
    m = len(X)
    sqrtError = (h(theta, X)-y)**2

    j = 1 / (2 * m) * np.sum(sqrtError)
    return j

def costfunctionAnders(X, y, theta):
    # Dit is hetzelfde als cost function maar anders geprogrammeerd
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp, 2)) / (2*len(X))


print('costfunction()')
print(costfunction(X, y, theta))
resultaat = 0.0, omdat de linear lijn goed door de data punten heen gaat, dus er zit geen verschil met de Y
print('costfunctionAnders()')
print(costfunctionAnders(X, y, theta))
resultaat = 0.0, omdat de linear lijn goed door de data punten heen gaat, dus er zit geen verschil met de Y
```
