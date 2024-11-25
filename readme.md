
[274. H-Index](#274-h-index)


# 274. H-Index

[link](https://leetcode.com/problems/h-index/?envType=study-plan-v2&envId=top-interview-150)

```
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