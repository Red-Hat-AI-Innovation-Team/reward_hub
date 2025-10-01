"""Mock model objects for testing"""


class MockVLLMOutput:
    """Mock VLLM output object"""
    def __init__(self, scores):
        self.outputs = MockOutputs(scores)


class MockOutputs:
    """Mock outputs container"""
    def __init__(self, scores):
        self.data = [MockData(score) for score in scores]


class MockData:
    """Mock data object with score (acts like a list)"""
    def __init__(self, score):
        self._scores = [score]

    def __getitem__(self, index):
        """Allow indexing like data[-1]"""
        return MockScoreItem(self._scores[index])


class MockScoreItem:
    """Mock score item that can call .item()"""
    def __init__(self, score):
        self._score = score

    def item(self):
        return self._score
