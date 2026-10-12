def shape(a):
    result = []

    while isinstance(a, list):
        result.append(len(a))

        if len(a) == 0:
            break

        a = a[0]

    return tuple(result)
    
def transpose(a):

    #2d
    rows = len(a)
    cols = len(a[0])

    result = []
    for j in range(cols):
        new_row = []
        for i in range(rows):
            new_row.append(a[i][j])
        result.append(new_row)

    return result

a = [[1, 2],
     [3, 4]]
b = [[1, 2, 3, 4]]

print(transpose(a))
print(transpose(b))