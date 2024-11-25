
[274. H-Index](#274-h-index)


# 274. H-Index
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