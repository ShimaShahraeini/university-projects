# tamrin3 - mabani bazybi etela'at va jostojo web - Shima Shahraeini

def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[i-1] == word2[j-1]:
                dp[i][j] = min(dp[i-1][j] +1 , dp[i][j-1] +1 , dp[i-1][j-1])
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]

# Example usage:
word1 = "kitten"
word2 = "sitting"
print(f"The edit distance between '{word1}' and '{word2}' is {edit_distance(word1, word2)}.")

# Example from class:
word1_1 = "bbcab"
word1_2 = "abab"
print(f"The edit distance between '{word1_1}' and '{word1_2}' is {edit_distance(word1_1, word1_2)}.")


'''
output: 
The edit distance between 'kitten' and 'sitting' is 3.
The edit distance between 'bbcab' and 'abab' is 2.
'''