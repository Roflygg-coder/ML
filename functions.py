def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Not vectorized implementation.
    """
    ans = 0;
    for i in range (min(x.shape)):
        if (x[i][i] != 0):
            if (ans == 0):
                ans = x[i][i]
            else:
                ans *= x[i][i]
    return ans


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Not vectorized implementation.
    """
    if len(x) != len(y):
        return False
        
    sorted_x = sorted(x)
    sorted_y = sorted(y)
    for i in range(len(sorted_x)):
        if sorted_x[i] != sorted_y[i]:
            return False
    return True
    
def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Not vectorized implementation.
    """
    maxel = 0
    for i in range(1, x.shape[0]):
        if (x[i - 1] == 0):
            maxel = max(maxel, x[i])
    return maxel


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Not vectorized implementation.
    """
    height = len(img)
    width = len(img[0])
    num_channels = len(coefs)
    result = [[0.0 for _ in range(width)] for _ in range(height)]
    
    for h in range(height):
        for w in range(width):
            pixel_sum = 0.0
            for c in range(num_channels):
                pixel_sum += img[h][w][c] * coefs[c]
            result[h][w] = pixel_sum
    return result


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Not vectorized implementation.
    """
    if len(x) == 0:
        return [], []
    values = []
    cnt = []
    cur_val = x[0]
    cur_cnt = 1
    for i in range(1, len(x)):
        if x[i] == cur_val:
            cur_cnt += 1
        else:
            values.append(cur_val)
            cnt.append(cur_cnt)
            cur_val = x[i]
            cur_cnt = 1
            
    values.append(cur_val)
    cnt.append(cur_cnt)
    return values, cnt


def pairwise_distance(X, Y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Not vectorized implementation.
    """
    n = len(X)
    m = len(Y)
    dim = len(X[0])
    
    result = [[0.0 for _ in range(m)] for _ in range(n)]
    
    for i in range(n):
        for j in range(m):
            sum_sq = 0.0
            for d in range(dim):
                diff = X[i][d] - Y[j][d]
                sum_sq += diff * diff
            result[i][j] = (sum_sq ** 0.5)
            
    return result

