#perceptron neural network for AND example without library - Shima Shahraeini

class Perceptron:
    def __init__(self, input, alpha=0.1):
		# initialize the weight matrix and store the learning rate
        self.W = [0, 0, 0]
        self.alpha = alpha
	
    def step(self, x):
        # apply the step function
        return 1 if x > 0.1 else -1

    def fit(self, X, y, epochs=10):
        # insert a column of 1's as the last entry in the feature
        # matrix -- this little trick allows us to treat the bias
        # as a trainable parameter within the weight matrix
        for i in range(count_input(X)):
            X[i].extend([1])

        for epoch in range(epochs):
			# loop over each individual data point
                for (x, target) in zip(X, y):
                    p = self.step(dot(x, self.W))
                
                    if p != int_convertor(target):
                    # update the weight matrix
                        change_w = [(item * self.alpha * int_convertor(target)) for item in x]
                        self.W =  [x + y for x, y in zip(self.W, change_w)]

                        
    def predict(self, X, addBias=True):
        return self.step(dot(X, self.W))

def count_input(list):
    count = 0
    for item in list:
        count+=1
    return count

def int_convertor(list):
    new_list = [str(current_integer) for current_integer in list]
    string_value = "".join(new_list)
    return int(string_value)


def dot(v1, v2):
    return sum(x*y for x, y in zip(v1, v2))

# construct the AND dataset
X = [[-1, -1], [1, -1], [-1, 1], [1, 1]]
y = [[-1], [-1], [-1], [1]]

# define our perceptron and train it
print("[INFO] training perceptron...")
p = Perceptron(count_input(X), alpha=1)
p.fit(X, y, epochs=25)

# now that our perceptron is trained we can evaluate it
print("[INFO] testing perceptron...")
print(f'weight = {p.W}')
for (x, target) in zip(X, y):
	pred = p.predict(x)
	print("[INFO] data={}, ground-truth={}, pred={}".format(
		x, target[0], pred))
