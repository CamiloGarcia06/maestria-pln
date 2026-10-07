---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: Python 3
    name: python3
---

```python colab={"base_uri": "https://localhost:8080/"} id="_-sM0ZZlwoiv" executionInfo={"status": "ok", "timestamp": 1790779653886, "user_tz": 300, "elapsed": 3423, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}} outputId="ce252355-097f-4553-b3db-80db30b51ba5"
import nltk
nltk.download('punkt_tab')
```

```python id="oQhVmSRIvtBG" executionInfo={"status": "ok", "timestamp": 1790779653906, "user_tz": 300, "elapsed": 22, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}}
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter, defaultdict
import numpy as np
from scipy import linalg
```

<!-- #region id="eNwQQ2uCv6Xm" -->
# Funciones de utilidad
<!-- #endregion -->

```python id="TukX8a82v4zI" executionInfo={"status": "ok", "timestamp": 1790779653918, "user_tz": 300, "elapsed": 11, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}}
def sigmoid(z):
    # sigmoid function
    return 1.0 / (1.0 + np.exp(-z))
```

```python id="3J0wpxcBv_GW" executionInfo={"status": "ok", "timestamp": 1790779653937, "user_tz": 300, "elapsed": 18, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}}
def get_idx(words, word2Ind):
    idx = []
    for word in words:
        idx = idx + [word2Ind[word]]
    return idx
```

```python id="f04kxwf_v-27" executionInfo={"status": "ok", "timestamp": 1790779653965, "user_tz": 300, "elapsed": 2, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}}
def pack_idx_with_frequency(context_words, word2Ind):
    freq_dict = defaultdict(int)
    for word in context_words:
        freq_dict[word] += 1
    idxs = get_idx(context_words, word2Ind)
    packed = []
    for i in range(len(idxs)):
        idx = idxs[i]
        freq = freq_dict[context_words[i]]
        packed.append((idx, freq))
    return packed
```

```python id="gten7CnLwDOF" executionInfo={"status": "ok", "timestamp": 1790779654050, "user_tz": 300, "elapsed": 79, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}}
def get_vectors(data, word2Ind, V, C):
    i = C
    while True:
        y = np.zeros(V)
        x = np.zeros(V)
        center_word = data[i]
        y[word2Ind[center_word]] = 1
        context_words = data[(i - C) : i] + data[(i + 1) : (i + C + 1)]
        num_ctx_words = len(context_words)
        for idx, freq in pack_idx_with_frequency(context_words, word2Ind):
            x[idx] = freq / num_ctx_words
        yield x, y
        i += 1
        if i >= len(data) - C:
            print("i is being set to", C)
            i = C
```

```python id="CCwh6-jHwEtJ" executionInfo={"status": "ok", "timestamp": 1790779654052, "user_tz": 300, "elapsed": 2, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}}
def get_batches(data, word2Ind, V, C, batch_size):
    batch_x = []
    batch_y = []
    for x, y in get_vectors(data, word2Ind, V, C):
        if len(batch_x) < batch_size:
            batch_x.append(x)
            batch_y.append(y)
        else:
            yield np.array(batch_x).T, np.array(batch_y).T
            batch_x = []
            batch_y = []
```

```python id="yMIK-e5vwGcU" executionInfo={"status": "ok", "timestamp": 1790779654081, "user_tz": 300, "elapsed": 28, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}}
def compute_pca(data, n_components=2):
    """
    Input:
        data: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output:
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """

    m, n = data.shape

    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = linalg.eigh(R)
    # sort eigenvalue in decreasing order
    # this returns the corresponding indices of evals and evecs
    idx = np.argsort(evals)[::-1]

    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :n_components]
    return np.dot(evecs.T, data.T).T
```

```python id="SHDKQnWWwKV8" executionInfo={"status": "ok", "timestamp": 1790779654115, "user_tz": 300, "elapsed": 2, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}}
def get_dict(data):
    """
    Input:
        K: the number of negative samples
        data: the data you want to pull from
        indices: a list of word indices
    Output:
        word_dict: a dictionary with the weighted probabilities of each word
        word2Ind: returns dictionary mapping the word to its index
        Ind2Word: returns dictionary mapping the index to its word
    """
    #
    #     words = nltk.word_tokenize(data)
    words = sorted(list(set(data)))
    n = len(words)
    idx = 0
    # return these correctly
    word2Ind = {}
    Ind2word = {}
    for k in words:
        word2Ind[k] = idx
        Ind2word[idx] = k
        idx += 1
    return word2Ind, Ind2word
```

<!-- #region id="IB3rq36OFWqU" -->
# Fin de las funciones de utilidad
<!-- #endregion -->

```python id="2_Yy6n-cwMt6" executionInfo={"status": "ok", "timestamp": 1790779654117, "user_tz": 300, "elapsed": 1, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}}
# Download sentence tokenizer
nltk.data.path.append('.')
```

```python colab={"base_uri": "https://localhost:8080/"} id="UbIe8yz5wPCz" executionInfo={"status": "ok", "timestamp": 1790779655488, "user_tz": 300, "elapsed": 1371, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}} outputId="7f1e6881-31ee-41ed-fcc0-e6d0edd34639"
# Load, tokenize and process the data
import re                                                           #  Load the Regex-modul
with open('./shakespeare.txt') as f:
    data = f.read()                                                 #  Read in the data
data = re.sub(r'[,!?;-]', '.',data)                                 #  Punktuations are replaced by .
data = nltk.word_tokenize(data)                                     #  Tokenize string to words
data = [ ch.lower() for ch in data if ch.isalpha() or ch == '.']    #  Lower case and drop non-alphabetical tokens
print("Number of tokens:", len(data),'\n', data[:15])               #  print data sample
```

```python colab={"base_uri": "https://localhost:8080/"} id="-TiUnj7Owgqj" executionInfo={"status": "ok", "timestamp": 1790779655550, "user_tz": 300, "elapsed": 61, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}} outputId="8f210d27-7067-4bf9-d1d5-6db445ea8f06"
# Compute the frequency distribution of the words in the dataset (vocabulary)
fdist = nltk.FreqDist(word for word in data)
print("Size of vocabulary: ",len(fdist) )
print("Most frequent tokens: ",fdist.most_common(20) ) # print the 20 most frequent words and their freq.
```

```python colab={"base_uri": "https://localhost:8080/"} id="N1UOWK2Swy4u" executionInfo={"status": "ok", "timestamp": 1790779655586, "user_tz": 300, "elapsed": 32, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}} outputId="24a16711-262a-434e-c35d-892008fce604"
# get_dict creates two dictionaries, converting words to indices and viceversa.
word2Ind, Ind2word = get_dict(data)
V = len(word2Ind)
print("Size of vocabulary: ", V)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Z9WOOH5Hw0fv" executionInfo={"status": "ok", "timestamp": 1790779655609, "user_tz": 300, "elapsed": 20, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}} outputId="d6638c6d-972c-4a79-a2fe-9fb213a7c61e"
# example of word to index mapping
print("Index of the word 'king' :  ",word2Ind['king'] )
print("Word which has index 2747:  ",Ind2word[2743] )
```

```python id="ju9vDQJYw6M7" executionInfo={"status": "ok", "timestamp": 1790779655622, "user_tz": 300, "elapsed": 2, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}}
def initialize_model(N,V, random_seed=1):
    '''
    Inputs:
        N:  dimension of hidden vector
        V:  dimension of vocabulary
        random_seed: random seed for consistent results in the unit tests
     Outputs:
        W1, W2, b1, b2: initialized weights and biases
    '''
    np.random.seed(random_seed)
    # W1 has shape (N,V)
    W1 = np.random.rand(N, V)

    # W2 has shape (V,N)
    W2 = np.random.rand(V, N)

    # b1 has shape (N,1)
    b1 = np.random.rand(N, 1)

    # b2 has shape (V,1)
    b2 = np.random.rand(V, 1)

    return W1, W2, b1, b2
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ykw73ZbixG3z" executionInfo={"status": "ok", "timestamp": 1790779655678, "user_tz": 300, "elapsed": 54, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}} outputId="18b5b9ed-8045-4ca2-eaf7-db85f7313439"
# Test your function example.
tmp_N = 4
tmp_V = 10
tmp_W1, tmp_W2, tmp_b1, tmp_b2 = initialize_model(tmp_N,tmp_V)
assert tmp_W1.shape == ((tmp_N,tmp_V))
assert tmp_W2.shape == ((tmp_V,tmp_N))
print(f"tmp_W1.shape: {tmp_W1.shape}")
print(f"tmp_W2.shape: {tmp_W2.shape}")
print(f"tmp_b1.shape: {tmp_b1.shape}")
print(f"tmp_b2.shape: {tmp_b2.shape}")
```

```python id="c_N53qIUxI9L" executionInfo={"status": "ok", "timestamp": 1790779655692, "user_tz": 300, "elapsed": 4, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}}
def softmax(z):
    '''
    Inputs:
        z: output scores from the hidden layer
    Outputs:
        yhat: prediction (estimate of y)
    '''
    # Calculate yhat (softmax)
    yhat = np.exp(z)/ np.sum(np.exp(z), axis = 0)
    return yhat
```

```python colab={"base_uri": "https://localhost:8080/", "height": 52} id="qxMghX_ExOxu" executionInfo={"status": "ok", "timestamp": 1790779655746, "user_tz": 300, "elapsed": 31, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}} outputId="0b7a0473-7b56-4ebc-8599-64c3fa17d188"
# Test the function
tmp = np.array([[1,2,3],
                [1,1,1]
               ])
tmp_sm = softmax(tmp)
display(tmp_sm)
```

```python id="fZ-26IaRxQpo" executionInfo={"status": "ok", "timestamp": 1790779655748, "user_tz": 300, "elapsed": 4, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}}
def forward_prop(x, W1, W2, b1, b2):
    '''
    Inputs:
        x:  average one hot vector for the context
        W1, W2, b1, b2:  matrices and biases to be learned
     Outputs:
        z:  output score vector
    '''
    # Calculate h
    h = np.dot(W1, x) + b1

    # Apply the relu on h,
    # store the relu in h
    h = np.maximum(0, h)

    # Calculate z
    z = np.dot(W2, h) + b2

    return z, h
```

```python colab={"base_uri": "https://localhost:8080/"} id="Fx0hg7bxxWH-" executionInfo={"status": "ok", "timestamp": 1790779655787, "user_tz": 300, "elapsed": 37, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}} outputId="66373c53-9b47-40b6-d7f2-c59311ca41d7"
# Test the function

# Create some inputs
tmp_N = 2
tmp_V = 3
tmp_x = np.array([[0,1,0]]).T
#print(tmp_x)
tmp_W1, tmp_W2, tmp_b1, tmp_b2 = initialize_model(N=tmp_N,V=tmp_V, random_seed=1)

print(f"x has shape {tmp_x.shape}")
print(f"N is {tmp_N} and vocabulary size V is {tmp_V}")

# call function
tmp_z, tmp_h = forward_prop(tmp_x, tmp_W1, tmp_W2, tmp_b1, tmp_b2)
print("call forward_prop")
print()
# Look at output
print(f"z has shape {tmp_z.shape}")
print("z has values:")
print(tmp_z)

print()

print(f"h has shape {tmp_h.shape}")
print("h has values:")
print(tmp_h)
```

```python id="jOCRSwYgxYWr" executionInfo={"status": "ok", "timestamp": 1790779655787, "user_tz": 300, "elapsed": 10, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}}
# compute_cost: cross-entropy cost function
def compute_cost(y, yhat, batch_size):

    # cost function
    logprobs = np.multiply(np.log(yhat),y)
    cost = - 1/batch_size * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost
```

```python colab={"base_uri": "https://localhost:8080/"} id="ORE0pKFdxe87" executionInfo={"status": "ok", "timestamp": 1790779655919, "user_tz": 300, "elapsed": 140, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}} outputId="1a450e3d-1100-4c46-edc8-a8c44b862a3d"
# Test the function
tmp_C = 2
tmp_N = 50
tmp_batch_size = 4
tmp_word2Ind, tmp_Ind2word = get_dict(data)
tmp_V = len(word2Ind)

tmp_x, tmp_y = next(get_batches(data, tmp_word2Ind, tmp_V, tmp_C, tmp_batch_size))

print(f"tmp_x.shape {tmp_x.shape}")
print(f"tmp_y.shape {tmp_y.shape}")

tmp_W1, tmp_W2, tmp_b1, tmp_b2 = initialize_model(tmp_N,tmp_V)

print(f"tmp_W1.shape {tmp_W1.shape}")
print(f"tmp_W2.shape {tmp_W2.shape}")
print(f"tmp_b1.shape {tmp_b1.shape}")
print(f"tmp_b2.shape {tmp_b2.shape}")

tmp_z, tmp_h = forward_prop(tmp_x, tmp_W1, tmp_W2, tmp_b1, tmp_b2)
print(f"tmp_z.shape: {tmp_z.shape}")
print(f"tmp_h.shape: {tmp_h.shape}")

tmp_yhat = softmax(tmp_z)
print(f"tmp_yhat.shape: {tmp_yhat.shape}")

tmp_cost = compute_cost(tmp_y, tmp_yhat, tmp_batch_size)
print("call compute_cost")
print(f"tmp_cost {tmp_cost:.4f}")
```

```python id="-YzHubd2xhVA" executionInfo={"status": "ok", "timestamp": 1790779655964, "user_tz": 300, "elapsed": 44, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}}
def back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size):
    '''
    Inputs:
        x:  average one hot vector for the context
        yhat: prediction (estimate of y)
        y:  target vector
        h:  hidden vector (see eq. 1)
        W1, W2, b1, b2:  matrices and biases
        batch_size: batch size
     Outputs:
        grad_W1, grad_W2, grad_b1, grad_b2:  gradients of matrices and biases
    '''

    # Compute z1 as "W1⋅x + b1"
    z1 = np.dot(W1, x) + b1

    # Compute l1 as W2^T (Yhat - Y)
    l1 = np.dot(W2.T, (yhat - y))

    # if z1 < 0, then l1 = 0
    # otherwise l1 = l1
    # (this is already implemented for you)

    l1[z1 < 0] = 0 # use "l1" to compute gradients below

    # compute the gradient for W1
    grad_W1 =  (1/ batch_size) * np.dot(l1, x.T)

    # Compute gradient of W2
    grad_W2 = (1/ batch_size) * np.dot((yhat - y), h.T)

    # compute gradient for b1
    grad_b1 = (1/ batch_size) * np.sum(l1, axis = 1, keepdims = True)

    # compute gradient for b2
    grad_b2 = (1/ batch_size) * np.sum((yhat - y), axis = 1, keepdims = True)

    return grad_W1, grad_W2, grad_b1, grad_b2
```

```python colab={"base_uri": "https://localhost:8080/"} id="p0yCeI7Bxm_H" executionInfo={"status": "ok", "timestamp": 1790779656140, "user_tz": 300, "elapsed": 174, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}} outputId="f7d6652a-4ddf-4fe5-bce1-5ab04c016371"
# Test the function
tmp_C = 2
tmp_N = 50
tmp_batch_size = 4
tmp_word2Ind, tmp_Ind2word = get_dict(data)
tmp_V = len(word2Ind)


# get a batch of data
tmp_x, tmp_y = next(get_batches(data, tmp_word2Ind, tmp_V,tmp_C, tmp_batch_size))

print("get a batch of data")
print(f"tmp_x.shape {tmp_x.shape}")
print(f"tmp_y.shape {tmp_y.shape}")

print()
print("Initialize weights and biases")
tmp_W1, tmp_W2, tmp_b1, tmp_b2 = initialize_model(tmp_N,tmp_V)

print(f"tmp_W1.shape {tmp_W1.shape}")
print(f"tmp_W2.shape {tmp_W2.shape}")
print(f"tmp_b1.shape {tmp_b1.shape}")
print(f"tmp_b2.shape {tmp_b2.shape}")

print()
print("Forwad prop to get z and h")
tmp_z, tmp_h = forward_prop(tmp_x, tmp_W1, tmp_W2, tmp_b1, tmp_b2)
print(f"tmp_z.shape: {tmp_z.shape}")
print(f"tmp_h.shape: {tmp_h.shape}")

print()
print("Get yhat by calling softmax")
tmp_yhat = softmax(tmp_z)
print(f"tmp_yhat.shape: {tmp_yhat.shape}")

tmp_m = (2*tmp_C)
tmp_grad_W1, tmp_grad_W2, tmp_grad_b1, tmp_grad_b2 = back_prop(tmp_x, tmp_yhat, tmp_y, tmp_h, tmp_W1, tmp_W2, tmp_b1, tmp_b2, tmp_batch_size)

print()
print("call back_prop")
print(f"tmp_grad_W1.shape {tmp_grad_W1.shape}")
print(f"tmp_grad_W2.shape {tmp_grad_W2.shape}")
print(f"tmp_grad_b1.shape {tmp_grad_b1.shape}")
print(f"tmp_grad_b2.shape {tmp_grad_b2.shape}")
```

```python id="E1kZz1nyxo0P" executionInfo={"status": "ok", "timestamp": 1790779656192, "user_tz": 300, "elapsed": 54, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}}
def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03,
                     random_seed=282, initialize_model=initialize_model,
                     get_batches=get_batches, forward_prop=forward_prop,
                     softmax=softmax, compute_cost=compute_cost,
                     back_prop=back_prop):

    '''
    This is the gradient_descent function

      Inputs:
        data:      text
        word2Ind:  words to Indices
        N:         dimension of hidden vector
        V:         dimension of vocabulary
        num_iters: number of iterations
        random_seed: random seed to initialize the model's matrices and vectors
        initialize_model: your implementation of the function to initialize the model
        get_batches: function to get the data in batches
        forward_prop: your implementation of the function to perform forward propagation
        softmax: your implementation of the softmax function
        compute_cost: cost function (Cross entropy)
        back_prop: your implementation of the function to perform backward propagation
     Outputs:
        W1, W2, b1, b2:  updated matrices and biases after num_iters iterations

    '''
    W1, W2, b1, b2 = initialize_model(N,V, random_seed=random_seed) #W1=(N,V) and W2=(V,N)

    batch_size = 128
#    batch_size = 512
    iters = 0
    C = 2

    for x, y in get_batches(data, word2Ind, V, C, batch_size):

        # get z and h
        z, h = forward_prop(x, W1, W2, b1, b2)

        # get yhat
        yhat = softmax(z)

        # get cost
        cost = compute_cost(y, yhat, batch_size)
        if ( (iters+1) % 10 == 0):
            print(f"iters: {iters + 1} cost: {cost:.6f}")

        # get gradients
        grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size)

        # update weights and biases
        W1 = W1 - alpha * grad_W1
        W2 = W2 - alpha * grad_W2
        b1 = b1 - alpha * grad_b1
        b2 = b2 - alpha * grad_b2

        iters +=1
        if iters == num_iters:
            break
        if iters % 100 == 0:
            alpha *= 0.66

    return W1, W2, b1, b2
```

```python colab={"base_uri": "https://localhost:8080/"} id="Kn33AADFxxzn" executionInfo={"status": "ok", "timestamp": 1790779677726, "user_tz": 300, "elapsed": 21526, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}} outputId="65a8a19d-dc09-4a38-d4c0-6f51b3d0f7ce"
# test your function

C = 2
N = 50
word2Ind, Ind2word = get_dict(data)
V = len(word2Ind)
num_iters = 150
print("Call gradient_descent")
W1, W2, b1, b2 = gradient_descent(data, word2Ind, N, V, num_iters)
```

```python colab={"base_uri": "https://localhost:8080/"} id="33uBYZ61xz4j" executionInfo={"status": "ok", "timestamp": 1790779677753, "user_tz": 300, "elapsed": 26, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}} outputId="d9ab0c46-579c-42bc-c04a-f2978a3cedfe"
# visualizing the word vectors here
from matplotlib import pyplot
%config InlineBackend.figure_format = 'svg'
words = ['king', 'queen','lord','man', 'woman','dog','wolf',
         'rich','happy','sad']

embs = (W1.T + W2)/2.0

# given a list of words and the embeddings, it returns a matrix with all the embeddings
idx = [word2Ind[word] for word in words]
X = embs[idx, :]
print(X.shape, idx)  # X.shape:  Number of words of dimension N each
```

```python colab={"base_uri": "https://localhost:8080/", "height": 418} id="rOyA16Mjx6o4" executionInfo={"status": "ok", "timestamp": 1790779678067, "user_tz": 300, "elapsed": 313, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}} outputId="457ad395-d579-4917-9935-81ec08e63a8c"
result= compute_pca(X, 2)
pyplot.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 418} id="jfIO8Ry4x8bU" executionInfo={"status": "ok", "timestamp": 1790779678187, "user_tz": 300, "elapsed": 118, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}} outputId="8b570ed2-5f97-4102-856b-3c34ccb372ec"
result= compute_pca(X, 4)
pyplot.scatter(result[:, 3], result[:, 1])
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 3], result[i, 1]))
pyplot.show()
```

```python id="4kOnOyAIyAHX" executionInfo={"status": "ok", "timestamp": 1790779678190, "user_tz": 300, "elapsed": 1, "user": {"displayName": "Juan Pablo Pajaro", "userId": "06558277647828565034"}}

```
