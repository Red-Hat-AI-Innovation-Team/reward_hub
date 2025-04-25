import unittest
from reward_hub import AutoRM
from reward_hub.base import PRMResult


class TestHuggingFaceProcessRewardModels(unittest.TestCase):
    def test_llama3_prm(self):
        """Test the RLHFlow/Llama3.1-8B-PRM-Deepseek-Data model."""
        model = AutoRM.load("RLHFlow/Llama3.1-8B-PRM-Deepseek-Data", "hf", device=0)
        
        question = [{"role": "user", "content": "What is 2+2?"}]
        
        output1 = """Let me solve this step by step:
1) First, I need to add 2 and 2
2) 2 + 2 = 4
Therefore, the answer is 4."""
        
        output2 = """Let me solve this step by step:
1) First, I need to add 2 and 2
2) 2 + 2 = 5
Therefore, the answer is 5."""
        
        messages1 = question + [{"role": "assistant", "content": output1}]
        messages2 = question + [{"role": "assistant", "content": output2}]
        
        # Test with return_full_prm_result=True
        results = model.score([messages1, messages2], return_full_prm_result=True)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], PRMResult)
        self.assertIsInstance(results[1], PRMResult)
        
        # Correct answer should have higher score
        self.assertGreater(results[0].score, results[1].score)
        
        # Test with return_full_prm_result=False
        scores = model.score([messages1, messages2], return_full_prm_result=False)
        
        self.assertIsInstance(scores, list)
        self.assertEqual(len(scores), 2)
        self.assertIsInstance(scores[0], float)
        self.assertIsInstance(scores[1], float)
        
        # Correct answer should have higher score
        self.assertGreater(scores[0], scores[1])
    
    def test_qwen_prm(self):
        """Test the Qwen/Qwen2.5-Math-PRM-7B model."""
        model = AutoRM.load("Qwen/Qwen2.5-Math-PRM-7B", "hf", device=0)
        
        question = [{"role": "user", "content": "Solve the equation: 3x + 5 = 14"}]
        
        output1 = """Let me solve this step by step:
1) First, I'll subtract 5 from both sides
   3x + 5 - 5 = 14 - 5
   3x = 9
2) Then, I'll divide both sides by 3
   3x/3 = 9/3
   x = 3
Therefore, x = 3."""
        
        output2 = """Let me solve this step by step:
1) First, I'll subtract 5 from both sides
   3x + 5 - 5 = 14 - 5
   3x = 9
2) Then, I'll divide both sides by 3
   3x/3 = 9/3
   x = 4
Therefore, x = 4."""
        
        messages1 = question + [{"role": "assistant", "content": output1}]
        messages2 = question + [{"role": "assistant", "content": output2}]
        
        # Test with return_full_prm_result=True
        results = model.score([messages1, messages2], return_full_prm_result=True)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], PRMResult)
        self.assertIsInstance(results[1], PRMResult)
        
        # Correct answer should have higher score
        self.assertGreater(results[0].score, results[1].score)
        
        # Test with return_full_prm_result=False
        scores = model.score([messages1, messages2], return_full_prm_result=False)
        
        self.assertIsInstance(scores, list)
        self.assertEqual(len(scores), 2)
        self.assertIsInstance(scores[0], float)
        self.assertIsInstance(scores[1], float)
        
        # Correct answer should have higher score
        self.assertGreater(scores[0], scores[1])
