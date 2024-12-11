[**Leetcode Interview Questions 150**](https://leetcode.com/studyplan/top-interview-150/)

- [88. Merge Sorted Array](#88-merge-sorted-array)
- [27. Remove Element](#27-remove-element)
- [26. Remove Duplicates from Sorted Array](#26-remove-duplicates-from-sorted-array)
- [80. Remove Duplicates from Sorted Array II](#80-remove-duplicates-from-sorted-array-ii)
- [169. Majority Element](#169-majority-element)
- [189. Rotate Array](#189-rotate-array)
- [121. Best Time to Buy and Sell Stock](#121-best-time-to-buy-and-sell-stock)
- [122. Best Time to Buy and Sell Stock II](#122-best-time-to-buy-and-sell-stock-ii)
- [55. Jump Game](#55-jump-game)
- [45. Jump Game II](#45-jump-game-ii)
- [274. H-Index](#274-h-index)
- [380. Insert Delete GetRandom O(1)](#380-insert-delete-getrandom-o1)
- [238. Product of Array Except Self](#238-product-of-array-except-self)
- [134. Gas Station](#134-gas-station)
- [135. Candy (Hard)](#135-candy-hard)
- [13. Roman to Integer](#13-roman-to-integer)
- [12. Integer to Roman](#12-integer-to-roman)
- [58. Length of Last Word](#58-length-of-last-word)
- [14. Longest Common Prefix](#14-longest-common-prefix)
- [151. Reverse Words in a String](#151-reverse-words-in-a-string)
- [28. Find the Index of the First Occurrence in a String](#28-find-the-index-of-the-first-occurrence-in-a-string)
- [125. Valid Palindrome](#125-valid-palindrome)
- [392. Is Subsequence](#392-is-subsequence)
- [167. Two Sum II - Input Array Is Sorted](#167-two-sum-ii---input-array-is-sorted)
- [11. Container With Most Water](#11-container-with-most-water)
- [15. 3 Sum](#15-3-sum)
- [209. Minimum Size Subarray Sum](#209-minimum-size-subarray-sum)
- [3. Longest Substring Without Repeating Characters](#3-longest-substring-without-repeating-characters)
- [36. Valid Sudoku](#36-valid-sudoku)
- [54. Spiral Matrix](#54-spiral-matrix)
- [48. Rotate Image](#48-rotate-image)
- [73. Set Matrix Zeroes](#73-set-matrix-zeroes)
- [289. Game of Life](#289-game-of-life)
- [383. Ransom Note](#383-ransom-note)
- [205. Isomorphic Strings](#205-isomorphic-strings)
- [290. Word Pattern](#290-word-pattern)
- [242. Valid Anagram](#242-valid-anagram)
- [1. Two Sum](#1-two-sum)
- [202. Happy Number (Cycle Detection)](#202-happy-number-cycle-detection)
- [219. Contains Duplicate II](#219-contains-duplicate-ii)
- [128. Longest Consecutive Sequence](#128-longest-consecutive-sequence)
- [228. Summary Ranges](#228-summary-ranges)
- [56. Merge Intervals (Greedy Algorithm)](#56-merge-intervals-greedy-algorithm)
- [57. Insert Interval](#57-insert-interval)
- [20. Valid Parentheses](#20-valid-parentheses)
- [71. Simplify Path](#71-simplify-path)
- [155. Min Stack](#155-min-stack)
- [150. Evaluate Reverse Polish Notation](#150-evaluate-reverse-polish-notation)
- [141. Linked List Cycle](#141-linked-list-cycle)


# 88. Merge Sorted Array

- Input: nums1 = [1, 2, 3, 0, 0, 0], m = 3, nums2 = [2, 5, 6], n = 3
- Output: [1, 2, 2, 3, 5, 6]  # sorted in **non-decreasing** order

```python
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        
        midx = m - 1
        nidx = n - 1

        right = m + n - 1

        while nidx >= 0:
            
            if midx >= 0 and nums1[midx] > nums2[nidx]:
                
                nums1[right] = nums1[midx]

                midx -= 1

            else:
                nums1[right] = nums2[nidx]

                nidx -= 1

            right -= 1
```

# 27. Remove Element

有機會可以用在其他地方。當想要移除數列中的某個數值（多個 duplicates）而位置不重要的時候。

關於移除某個單個unique的數值，可以參考 [380. Insert Delete GetRandom O(1)](#380-insert-delete-getrandom-o1)

- Input: nums = [3, 2, 2, 3], val = 3
- Output: 2, nums = [2, 2, _, _]

```python
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """

        k = 0

        for i in range(len(nums)):

            if nums[i] != val:
                
                nums[k] = nums[i]

                k += 1

        # [1, 2, 3] remove 2, --> [1, 3, _] --> k = 2
        # [2, 3, 3, 2] remove 2, --> [3, 3, _, _] --> k = 2

        # _ denote the values we dont care

        return k
```

# 26. Remove Duplicates from Sorted Array

和上一題的分別是：這一題是移除重複的值。

上一題：

[2, 3, 3, 2] 移除 2 --> [3, 3, _, _]

這一題：

[1, 1, 2] --> [1, 2, _]  k = 2

[0, 0, 1, 1, 1, 2, 2, 3, 3, 4] --> [0, 1, 2, 3, 4, _, _, _, _, _] --> k = 5


```python
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        k = 1
        
        for i in range(1, len(nums)):

            if nums[i] != nums[i - 1]:

                nums[k] = nums[i]

                k += 1

        return k
```

# 80. Remove Duplicates from Sorted Array II

Given an integer array nums sorted in non-decreasing order, remove some duplicates in-place such that each unique element appears **at most twice**. The relative order of the elements should be kept the same.

[1, 1, 1, 2, 2, 3] --> [1, 1, 2, 2, 3, _]

```python
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        k = 2

        for i in range(2, len(nums)):

            if nums[i] != nums[k - 2]:
                
                nums[k] = nums[i]
        
                k += 1

        return k
```

# 169. Majority Element

**Hash map**

```python
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        a = {}

        count = 0
        majority = 0

        for num in nums:

            a[num] = a.get(num, 0) + 1

            if a[num] > count:
                
                majority = num
                count = a[num]

        return majority
```

# 189. Rotate Array

- Input: nums = [1, 2, 3, 4, 5, 6, 7], k = 3
- Output: [5, 6, 7, 1, 2, 3, 4]

```python
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """

        k = k % len(nums)

        nums[:] = nums[-k:] + nums[:-k]
```

# 121. Best Time to Buy and Sell Stock

- Input: prices = [7, 1, 5, 3, 6, 4]
- Output: 5  # Best profit

```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        
        buy_price = prices[0]

        profit = 0

        for price in prices:
            
            if price - buy_price < 0:
                
                buy_price = price

            else:
                if price - buy_price > profit:
                    profit = price - buy_price

        return profit
```

# 122. Best Time to Buy and Sell Stock II

- Input: prices = [7, 1, 5, 3, 6, 4]
- Output: 7  # (5-1) + (6-3) Total profit

```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """

        profit = 0

        for i in range(1, len(prices)):

            if prices[i] > prices[i - 1]:

                profit += (prices[i] - prices[i - 1])

        return profit
```

# 55. Jump Game

Each element represents the **maximum** jump length. 

```python
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        
        goal = len(nums) - 1
        
        for i in range(len(nums)-2, -1, -1):
            
            if nums[i] + i >= goal:  # This line

                goal = i

        if goal == 0:
            return True
        
        else:
            return False
```

# 45. Jump Game II

Jump game I:

- Input: nums = [2, 3, 1, 1, 4]
- Output: true

Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.


Jump game II:

- Input: nums = [2, 3, 1, 1, 4]
- Output: 2

Explanation: The **minimum** number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.

```python
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        n = len(nums) - 1
        
        jumps = 0
        i_lower = 0
        i_upper = 0

        while i_upper < n:

            farthest = 0

            for i in range(i_lower, i_upper + 1):

                farthest = max(farthest, i + nums[i])

            i_lower = i_upper + 1
            i_upper = farthest

            jumps += 1

        return jumps
```

# 274. H-Index

The **h-index** is defined as the maximum value of `h` such that the given researcher has published at least `h` papers that have each been cited at least `h` times.

這一題確實有點難理解。

```python
class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """

        n = len(citations)
        citations.sort()

        for i,v in enumerate(citations):
            if n - i <= v:
                return n - i
        return 0
```

# 380. Insert Delete GetRandom O(1)

```python
class RandomizedSet(object):

    def __init__(self):
        
        self.lst = []
        self.idx_map = {}

    def is_exist(self, val):
        return val in self.idx_map

    def insert(self, val):
        """
        :type val: int
        :rtype: bool
        """
        if self.is_exist(val):
            return False
        
        self.lst.append(val)
        self.idx_map[val] = len(self.lst) - 1
        return True

    def remove(self, val):
        """
        :type val: int
        :rtype: bool
        """
        if not self.is_exist(val):
            return False
        
        idx = self.idx_map[val]
       
        # 在需要被remove的item的位置上複製最後一個位置的值
        self.lst[idx] = self.lst[-1]
        # 同時記下更改位置後 原本最後一個值的位置
        self.idx_map[self.lst[-1]] = idx

        # pop走最後一個值
        self.lst.pop()
        del self.idx_map[val]
        return True

    def getRandom(self):
        """
        :rtype: int
        """
        import random
        return random.choice(self.lst)
        
# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
```

# 238. Product of Array Except Self

- Input: nums = [1, 2, 3, 4]
- Output: [24, 12, 8, 6]

```python
class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        n = len(nums)

        result = [1 for _ in range(n)]

        to_right = 1

        for i in range(n):
            result[i] *= to_right
            to_right *= nums[i]

        to_left = 1

        for i in range(n-1, -1, -1):
            result[i] *= to_left
            to_left *= nums[i]

        return result
```

# 134. Gas Station

移動的順序是跟隨sequence的。

- Input: gas = [1, 2, 3, 4, 5], cost = [3, 4, 5, 1, 2]
- Output: 3  # 3 是起點

```python
class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        if sum(gas) < sum(cost):
            return -1
                
        curernt_gas = 0
        start = 0
        for i in range(len(gas)):
            curernt_gas += gas[i] - cost[i]
            if curernt_gas < 0:
                curernt_gas = 0
                start = i + 1

        return start
```

# 135. Candy (Hard)

貪婪演算法（英語：greedy algorithm），又稱貪心演算法，是一種在每一步選擇中都採取在當前狀態下最好或最佳（即最有利）的選擇，從而希望導致結果是最好或最佳的演算法。比如在旅行推銷員問題中，如果旅行員每次都選擇最近的城市，那這就是一種貪婪演算法。

The Nuts and Bolts of the Two-Pass Method

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        candies = [1] * n 

        for i in range(1, n):
            if ratings[i] > ratings[i-1]:
                candies[i] = candies[i-1] + 1

        for i in range(n-2, -1, -1):
            if ratings[i] > ratings[i+1]:
                candies[i] = max(candies[i], candies[i+1] + 1)
        
        return sum(candies)
```

One-Pass Greedy Algorithm: Up-Down-Peak Method

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        if not ratings:
            return 0
        
        ret, up, down, peak = 1, 0, 0, 0
        
        for prev, curr in zip(ratings[:-1], ratings[1:]):
            if prev < curr:
                up, down, peak = up + 1, 0, up + 1
                ret += 1 + up
            elif prev == curr:
                up = down = peak = 0
                ret += 1
            else:
                up, down = 0, down + 1
                ret += 1 + down - int(peak >= down)
        
        return ret
```

# 13. Roman to Integer

```python
class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """

        res = 0
        
        roman = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000,
        }

        # e.g. s = "LVIII"
        # print:
        # L, V
        # V, I
        # I, I
        # I, I

        for a, b in zip(s, s[1:]):
            
            if roman[a] < roman[b]:
                res -= roman[a]
            else:
                res += roman[a]

        return res + roman[s[-1]]
```

# 12. Integer to Roman

重點在於`value_symbols`

```python
class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        
        value_symbols = [
            (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
            (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
            (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'),
            (1, 'I'),
        ]
        
        res = []

        # 3749 // 1000 = 3 --> MMM
        # 749 // 500 = 1 --> D
        # 249 // 100 = 2 --> CC
        # 49 // 40 = 1 --> XL
        # 9 // 9 = 1 --> IX
        # 3749 --> MMMDCCXLIX

        for value, symbol in value_symbols:
            if num == 0:
                break
            count = num // value
            res.append(symbol * count)
            num -= count * value

        return ''.join(res)  
```

# 58. Length of Last Word

```python
class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        s = s.rstrip(" ").split(" ")
        return len(s[-1])
```

# 14. Longest Common Prefix

```python
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        
        prefix = strs[0]
        prefix_len = len(prefix)

        for s in strs[1:]:
            
            while s[:prefix_len] != prefix[:prefix_len]:
                    
                prefix_len -= 1

                if prefix_len == 0:
                    return ""

        return prefix[:prefix_len]
```

# 151. Reverse Words in a String

Note: if " " --> False

"".join(r)  # 這個寫法可以記一下

```python
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """

        s = s.strip(" ").split(" ")
        
        r = []

        for i in range(len(s)-1, -1, -1):
            if s[i]:
                if i != 0:
                    temp = s[i] + " "
                else:
                    temp = s[i]

                r.append(temp)

        return "".join(r)
```

# 28. Find the Index of the First Occurrence in a String

- Input: haystack = "sadbutsad", needle = "sad"
- Output: 0

```python
class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        needle_len = len(needle)

        flag = False

        for i in range(0, len(haystack)):
            if haystack[i: i + needle_len] == needle:
                flag = True
                break

        if flag:
            return i

        else:
            return -1 
```

# 125. Valid Palindrome

Note: Remove all non-alphanumeric characters. Alphanumeric characters include letters and numbers.

"A ma," --> "ama" is poalindrome 

```python
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """

        s = s.lower()
        s = ''.join(char for char in s if char.isalnum())
        
        s_len = len(s)

        for i in range(s_len // 2):
            if s[i] != s[s_len-1-i]:
                return False

        return True
```

# 392. Is Subsequence

Need revision on this. ...this question is actually easy.

(1) While loop, two pointers

(2) return the ==  # 這兩個寫法都可以記一下

- Input: s = "abc", t = "ahbgdc"
- Output: true

```python
class Solution(object):
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """

        sp = tp = 0

        while sp < len(s) and tp < len(t):
            if s[sp] == t[tp]:
                sp += 1
            tp += 1
        
        return sp == len(s)
```

# 167. Two Sum II - Input Array Is Sorted

The last four lines worth remember. The point is, this array is **sorted**.

Two pointers move from (1) left to right, and (2) right to left.

- Input: numbers = [2, 7, 11, 15], target = 9
- Output: [1, 2]

```python
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        
        left = 0
        right = len(numbers) - 1

        while right > left:

            total = numbers[left] + numbers[right]
            
            if total == target:
                return [left + 1, right +1]

            elif total > target:
                right -= 1

            else:
                left += 1
                
        return False
```

# 11. Container With Most Water

I use the similar logic in the previous question [167. Two Sum II - Input Array Is Sorted](#167-two-sum-ii---input-array-is-sorted)

Two pointers, left to right, right to left.

```python
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        
        left = 0
        right = len(height) - 1

        max_area = 0

        while right > left:
            
            width = right - left

            if height[left] > height[right]:

                area = height[right] * width

                right -= 1
            
            else:
                area = height[left] * width

                left += 1

            if area > max_area:
                max_area = area

        return max_area
```

# 15. 3 Sum

My solution. The time complexity is O(N^2).

Find all possible sums of three numbers equal to 0

- Input: nums = [-1, 0, 1, 2, -1, -4]
- Output: [[-1, -1, 2],[-1, 0, 1]]

```python
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        nums.sort()

        res = []

        for i in range(len(nums)):
            
            j = i + 1
            k = len(nums) - 1

            while j < k:
                
                total = nums[i] + nums[j] + nums[k]

                if total == 0:
                    ans = [nums[i], nums[j], nums[k]]
                    if ans not in res:
                        res.append(ans)
                    j += 1
                
                elif total > 0:
                    k -= 1

                elif total < 0:
                    j += 1

        return res
```

# 209. Minimum Size Subarray Sum

題目在於找出最小的長度的滑動窗口，其中的所有數值總和sum等於target。而不是找出任意間隔最小的兩個數字。

這個解法的time complexity是 O(N)

```python
class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        min_len = float("inf")
        left = 0
        cur_sum = 0

        for right in range(len(nums)):
            cur_sum += nums[right]

            while cur_sum >= target:
                if right - left + 1 < min_len:
                    min_len = right - left + 1
                cur_sum -= nums[left]
                left += 1
        
        return min_len if min_len != float("inf") else 0
```

# 3. Longest Substring Without Repeating Characters

Beat 20% only, can do better.

```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        left = 0

        ret = 0  # return the window length

        for right in range(len(s)):

            if len(s[left : right + 1]) > len(set(s[left : right + 1])):
                left += 1
            
            if len(s[left : right + 1]) == len(set(s[left : right + 1])):
                if right - left + 1 > ret:
                    ret = right - left + 1

        return ret
```

This is generated from Copilot, beat 30%. looks much prettier.

```python
class Solution:
    def lengthOfLongestSubstring(self, s):
        char_index_map = {}
        max_length = 0
        start = 0
        
        for end in range(len(s)):
            if s[end] in char_index_map:
                start = max(start, char_index_map[s[end]] + 1)
            
            char_index_map[s[end]] = end
            max_length = max(max_length, end - start + 1)
        
        return max_length
```

# 36. Valid Sudoku

Beats 50%. Can do better.

Note: learn about the definition of `grid_index`. It could be useful in webpage layout design.

```python
class Solution(object):
    def isValidSudoku(self, board):
        # Initialize empty sets to keep track of seen numbers in rows, columns, and subgrids
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        grids = [set() for _ in range(9)]

        for r in range(9):
            for c in range(9):
                num = board[r][c]

                if num == ".":
                    continue

                # Calculate the index for the 3x3 subgrid
                # === # === # === #
                #  0     1     2  #
                #  3     4     5  #
                #  6     7     8  #
                # === # === # === #

                grid_index = (r // 3) * 3 + c // 3

                # Check if the number is already seen in the row, column, or subgrid
                if num in rows[r] or num in cols[c] or num in grids[grid_index]:
                    return False

                # Add the number to the respective sets
                rows[r].add(num)
                cols[c].add(num)
                grids[grid_index].add(num)

        return True
```

# 54. Spiral Matrix

加入兩個IF Condition 考慮矩陣不是正方形的時候的情況

- Input: matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
- Output: [1, 2, 3, 6, 9, 8, 7, 4, 5]

```python
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        ret = []

        top, bottom = 0,len(matrix) - 1
        left, right = 0, len(matrix[0]) - 1

        while top <= bottom and left <= right:
            
            # Left to Right
            for i in range(left, right + 1):
                ret.append(matrix[top][i])
            top += 1

            # Top to Bottom
            for i in range(top, bottom + 1):
                ret.append(matrix[i][right])
            right -= 1

            # 需要這兩個 IF condition 的原因是given matrix 可能不是正方形。
            # Right to Left
            if top <= bottom:  # top <= bottom 的情況下才有需要做反向的右到左
                for i in range(right, left - 1, -1):
                    ret.append(matrix[bottom][i])
                bottom -= 1

            # Bottom to Top
            if left <= right:  # left <= right 的情況下才有需要做反向的下到上
                for i in range(bottom, top - 1, -1):
                    ret.append(matrix[i][left])
                left += 1

        return ret
```

# 48. Rotate Image

You are given an `n x n` 2D matrix, rotate the matrix by 90 degrees (clockwise).

Example matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

Output: [[7, 4, 1], [8, 5, 2], [9, 6, 3]]

Note: The matrix should be N x N. can do (1) Transpose (2) Reverse

```python
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        
        # Note: the matrix is n x n

        # Step 1. Transpose the matrix

        for i in range(len(matrix)):
            for j in range(i+ 1, len(matrix[i])):  
                # 因為考慮 upper triangle 和 lower triangle 互換, j 由 i + 1 開始

                temp = matrix[i][j]

                matrix[i][j] = matrix[j][i]
                
                matrix[j][i] = temp

        # Step 2. Reverse each row

        for i in range(len(matrix)):
            matrix[i].reverse()
```

# 73. Set Matrix Zeroes

- Input: matrix = [[1, 1, 1],[1, 0, 1],[1, 1, 1]]
- Output: [[1, 0, 1],[0, 0, 0],[1, 0, 1]]

```python
class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        
        row = []
        col = []
        
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):

                if matrix[i][j] == 0:

                    row.append(i)
                    col.append(j)

        for r in row:
            for i in range(len(matrix[r])):

                matrix[r][i] = 0

        for c in col:
            for i in range(len(matrix)):
                
                matrix[i][c] = 0
```

# 289. Game of Life

Note: Check the boundaries with the IF condition.

- Input: board = [[0, 1, 0],[0, 0, 1],[1, 1, 1],[0, 0, 0]]
- Output: [[0, 0, 0],[1, 0, 1],[0, 1, 1],[0, 1, 0]]

```python
class Solution(object):
    def gameOfLife(self, board):
        """
        :type board: List[List[int]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        
        m, n = len(board), len(board[0])
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        copy_board = [[board[i][j] for j in range(n)] for i in range(m)]
        
        for i in range(m):
            for j in range(n):
                live_neighbors = 0
                
                for direction in directions:
                    r, c = i + direction[0], j + direction[1]

                    # Check the boundaries here.
                    if (0 <= r < m) and (0 <= c < n) and (copy_board[r][c] == 1):
                        live_neighbors += 1
                
                if copy_board[i][j] == 1:
                    if live_neighbors < 2 or live_neighbors > 3:
                        board[i][j] = 0  # Cell dies
                else:
                    if live_neighbors == 3:
                        board[i][j] = 1  # Cell becomes alive
```

# 383. Ransom Note

Learn solution 2.

Syntax: string.replace(oldvalue, newvalue, count)

Note: count (Optional.) A number specifying how many occurrences of the old value you want to replace. Default is all occurrences

Beats 38%

```python
class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        magazine_dict = {}

        for i in magazine:
            magazine_dict[i] = magazine_dict.get(i, 0) + 1

        for i in ransomNote:
            if not magazine_dict.get(i, False):
                return False
            
            magazine_dict[i] -= 1

            if magazine_dict[i] < 0:
                return False
        
        return True
```

Beats 97%

```python
class Solution:
    def canConstruct(self, ransomNote, magazine):
        
        for element in ransomNote: 
            if element not in magazine:
                return False
            else:
                magazine = magazine.replace(element, "", 1)
        
                # For example, "aab".replace("a", "", 1) -> "ab"

        return True
```

# 205. Isomorphic Strings

38.25%, can do better.

- Input: s = "egg", t = "add"
- Output: true

Learn the IF conditon:

if (s_char in s_to_t and s_to_t[s_char] != t_char) or (t_char in t_to_s and t_to_s[t_char] != s_char): return False

Instead of counting number, map the characters.

```python
class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """

        if len(s) != len(t):
            return False
        
        s_to_t = {}
        t_to_s = {}

        for s_char, t_char in zip(s, t):
            if (s_char in s_to_t and s_to_t[s_char] != t_char) or (t_char in t_to_s and t_to_s[t_char] != s_char): 
                return False
            
            s_to_t[s_char] = t_char
            t_to_s[t_char] = s_char
        
        return True
```

# 290. Word Pattern

- Input: pattern = "abba", s = "dog cat cat dog"
- Output: true

11%, solution one uses two hashmap.

Learn solution two, it uses only one hashmap.

```python
class Solution(object):
    def wordPattern(self, pattern, s):
        """
        :type pattern: str
        :type s: str
        :rtype: bool
        """
        
        s = s.split(" ")

        if len(s) != len(pattern):
            return False

        p_dict = {}
        s_dict = {}

        for i, j in zip(pattern, s):

            if (i in p_dict and p_dict[i] != j) or (j in s_dict and s_dict[j] != i):
                return False

            p_dict[i] = j
            s_dict[j] = i

        return True
```

! Beats 100%; Uses single hashmap.

```python
class Solution(object):
    def wordPattern(self, p, s):
    
        words, w_to_p = s.split(' '), dict()

        if len(p) != len(words): return False

        # for the case w = ['dog', 'cat'] and p = 'aa'
        if len(set(p)) != len(set(words)): return False  

        for i in range(len(words)):
            
            if words[i] not in w_to_p: 
                w_to_p[words[i]] = p[i]

            elif w_to_p[words[i]] != p[i]: 
                return False

        return True
```

# 242. Valid Anagram

Anagram example: 'listen' ~= 'silent'

Learn: XOR `^` **Bitwise** operation; In Python, XOR is a bitwise operator that is also known as **Exclusive OR**.

- ( a ^ a = 0 ) (XOR-ing a number with itself results in 0).
- ( a ^ 0 = a ) (XOR-ing a number with 0 leaves it unchanged).

If two strings are anagrams, XOR-ing all their characters will result in 0 because every character in one string will "cancel out" the same character in the other.

1. XOR all characters in both strings.
2. If the result is 0, the strings are anagrams.
3. Additionally, count the total character frequency to handle cases where strings of the same length have different characters but still result in XOR 0.

Learn `ord()` in Python

- ord('a') -> 97
- ord('b') -> 98

Note that the Beats percentage in Leetcode consider the language, Pyhton is lower than C++

```python
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        
        if len(s) != len(t):
            return False
        
        xor_sum = 0
        frequency = [0] * 26
        
        for i in range(len(s)):
            xor_sum ^= ord(s[i]) ^ ord(t[i]) # XOR both characters
            frequency[ord(s[i]) - ord('a')] += 1 # Increment for s
            frequency[ord(t[i]) - ord('a')] -= 1 # Decrement for t
        
        if xor_sum != 0: # If XOR isn't zero, they're not anagrams
            return False
        
        if any(freq != 0 for freq in frequency): # Check frequency counts
            return False
        
        return True
```

# 1. Two Sum

This is really smart.

- Input: nums = [2, 7, 11, 15], target = 9
- Output: [0, 1]


Copilot explaination:

This function uses a hash map (dictionary) to store the indices of the numbers we've seen so far. As we iterate through the list, for each number, we calculate its complement (the number that, when added to the current number, equals the target). If the complement is already in our hash map, it means we’ve found the pair of numbers that add up to the target, and we return their indices. If not, we store the current number and its index in the hash map.

```python
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """

        num_to_index = {}

        for i, num in enumerate(nums):

            complement = target - num

            if complement in num_to_index:
                return [num_to_index[complement], i]

            num_to_index[num] = i
        return []
```

# 202. Happy Number (Cycle Detection)

做到有點迷。之後可以再重新做一次這一題。Cycle Detection Algorithm.

- Input: n = 19
- Output: true
```
1^2 + 9^2 = 82
8^2 + 2^2 = 68
6^2 + 8^2 = 100
1^2 + 0^2 + 0^2 = 1
```

At the end of process, result is 1, so 19 is a happy number.

This function uses the **Floyd's Cycle-Finding Algorithm** (also known as the tortoise and the hare) to detect cycles in the sequence of numbers generated by summing the squares of the digits. It uses two pointers (slow and fast) that move through the sequence at different speeds. If they ever meet before reaching 1, it means there is a cycle, and the number is not happy. Otherwise, if fast reaches 1, the number is happy.

```python
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """

        def get_next(number):
            return sum(int(digit) ** 2 for digit in str(number))
        
        slow = n
        fast = get_next(n)
        
        while fast != 1 and slow != fast:
            slow = get_next(slow)
            fast = get_next(get_next(fast))
        
        return fast == 1
```

# 219. Contains Duplicate II

- Input: nums = [1, 2, 3, 1], k = 3
- Output: true

```python
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """

        num_hash = {}

        for i, n in enumerate(nums):

            if n in num_hash and i - num_hash[n] <= k:
                return True

            num_hash[n] = i
        return False
```

# 128. Longest Consecutive Sequence

- Input: nums = [100, 4, 200, 1, 3, 2]
- Output: 4
- Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.

Note: The point is to find the starting num. `if num - 1 not in num_set`

```python
class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        
        num_set = set(nums)
        longest_streak = 0
        
        for num in num_set:
            if num - 1 not in num_set:  # find start num of consecutive sequences
                current_num = num
                current_streak = 1
                
                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1
                
                longest_streak = max(longest_streak, current_streak)
        
        return longest_streak
```

# 228. Summary Ranges

- Input: nums = [0, 1, 2, 4, 5, 7]
- Output: ["0->2", "4->5", "7"]

```python
class Solution(object):
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        nums = set(nums)
        nums = sorted(nums)

        ret = []

        for n in nums:
            temp = ""
            if n - 1 not in nums:  # This is a start
                _n = n
                temp += "{}".format(_n)
                
                c = 0
                while True:
                    _n += 1
                    c += 1

                    if _n not in nums:
                        if c > 1:
                            temp += "->{}".format(_n-1)
                        break

                ret.append(temp)
        return ret
```

# 56. Merge Intervals (Greedy Algorithm)

- Input: intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
- Output: [[1, 6], [8, 10], [15, 18]]

重點在於：
1. Sort
2. 活用 Python `Max()`


```python
class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        # Sort the intervals based on the starting point
        intervals.sort(key=lambda x: x[0])

        merged = []
        
        for interval in intervals:
            # If the merged list is empty or the current interval does not overlap with the previous one
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                # There is overlap, so we merge the current interval with the previous one
                merged[-1][1] = max(merged[-1][1], interval[1])
        
        return merged
```

# 57. Insert Interval

我的解法和上一題一樣，但是可能有更好的解法。

- Input: intervals = [[1, 3], [6, 9]], newInterval = [2, 5]
- Output: [[1, 5], [6, 9]]

```python
class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """

        intervals.append(newInterval)

        # Sort the intervals based on the starting point
        intervals.sort(key=lambda x: x[0])

        merged = []
        
        for interval in intervals:
            # If the merged list is empty or the current interval does not overlap with the previous one
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                # There is overlap, so we merge the current interval with the previous one
                merged[-1][1] = max(merged[-1][1], interval[1])
        
        return merged
```

# 20. Valid Parentheses

這個寫法只能學。確實是想不出來可以這樣寫。

- Input: s = "()[]{}"
- Output: true
<br/><br/>
- Input: s = "(]"
- Output: false

```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        bracket_map = {')': '(', '}': '{', ']': '['}
        
        for char in s:
            if char in bracket_map:
                top_element = stack.pop() if stack else '#'
                if bracket_map[char] != top_element:
                    return False
            else:
                stack.append(char)
        
        return not stack
```

# 71. Simplify Path

! Important, Learn the data structure of **Stack**.

Definition: A stack is a linear data structure in which the insertions and deletions are allowed only at the end, called the top of the stack.

Implementation in Python:

- add data to stack: append
- access data in stack: pop
<br/><br/>
- Input: path = "/.../a/../b/c/../d/./"
- Output: "/.../b/d"

```python
class Solution(object):
    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        stack = []
        components = path.split('/')
        
        for component in components:
            if component == '..':
                if stack:
                    stack.pop()
            elif component and component != '.':
                stack.append(component)
        
        simplified_path = '/' + '/'.join(stack)
        return simplified_path
```

# 155. Min Stack

Time complexity: O(1)!

重點在於每一次加入新的element的時候，都保持追蹤目前最小值，並把目前最小值放在最頂。

在數據結構上，設計兩條並列的Stack。


```python
class MinStack(object):

    def __init__(self):
        self.stack = []

    def push(self, val):
        """
        :type val: int
        :rtype: None
        """
        current_min = self.getMin()
        if current_min == None or val < current_min:
            current_min = val

        self.stack.append([val, current_min])
        return None

    def pop(self):
        """
        :rtype: None
        """
        self.stack.pop()
        return None
        
    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1][0] if self.stack else None
      
    def getMin(self):
        """
        :rtype: int
        """
        return self.stack[-1][1] if self.stack else None

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```

# 150. Evaluate Reverse Polish Notation

- Input: tokens = ["2", "1", "+", "3", "*"]
- Output: 9
- Explanation: ((2 + 1) * 3) = 9

不知道為什麼用 Python 編譯出的答案是錯的。要用 Python3 才可以。

從這一題開始轉用 Python3。

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
 
        stack = []
 
        for token in tokens:
            if token in ("+", "-", "*", "/"):
 
                b = stack.pop()
                a = stack.pop()
 
                if token == "+":
                    stack.append(a + b)
                elif token == "-":
                    stack.append(a - b)
                elif token == "*":
                    stack.append(a * b)
                elif token == "/":
                    stack.append(int(a / b))
 
            else:
                stack.append(int(token))
 
        return stack[0]
```

# 141. Linked List Cycle

**Floyd's Cycle-Finding Algorithm**

Two pointers, one slow (one step), one fast (two steps). If they meet each other, there is a circle in the list.

```python
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        fast = head
        slow = head

        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

            if fast == slow:
                return True

        return False
```


