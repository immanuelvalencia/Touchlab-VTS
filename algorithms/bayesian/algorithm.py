import json
import os

class BayesianShapePredictor:
    def __init__(self):
        # Bayesian Shape State
        self.config_path = os.path.join(os.path.dirname(__file__), "config.json")
        self.load_config()
        self.reset()
        
    def load_config(self):
        default_likelihoods = {
            "planar": {"cube": 0.8, "sphere": 0.01, "cylinder": 0.4, "cone": 0.2, "square_pyramid": 0.8},
            "single_curvature": {"cube": 0.01, "sphere": 0.01, "cylinder": 0.9, "cone": 0.9, "square_pyramid": 0.01},
            "double_curvature": {"cube": 0.01, "sphere": 0.9, "cylinder": 0.01, "cone": 0.01, "square_pyramid": 0.01},
            "straight_edge": {"cube": 0.9, "sphere": 0.01, "cylinder": 0.01, "cone": 0.01, "square_pyramid": 0.9},
            "circular_rim": {"cube": 0.01, "sphere": 0.01, "cylinder": 0.9, "cone": 0.9, "square_pyramid": 0.01},
            "multi_face_vertex": {"cube": 0.9, "sphere": 0.01, "cylinder": 0.01, "cone": 0.01, "square_pyramid": 0.6},
            "sharp_apex": {"cube": 0.01, "sphere": 0.01, "cylinder": 0.01, "cone": 0.9, "square_pyramid": 0.9},
        }
        
        if not os.path.exists(self.config_path):
            with open(self.config_path, "w") as f:
                json.dump(default_likelihoods, f, indent=4)
            self.SHAPE_LIKELIHOODS = default_likelihoods
        else:
            with open(self.config_path, "r") as f:
                self.SHAPE_LIKELIHOODS = json.load(f)
            for feature, likelihoods in default_likelihoods.items():
                self.SHAPE_LIKELIHOODS.setdefault(feature, likelihoods)
        
    def reset(self):
        """Resets the probabilities to a flat prior (20% for 5 shapes)."""
        self.shape_probs = {shape: 0.2 for shape in ["cube", "sphere", "cylinder", "cone", "square_pyramid"]}

    def update(self, predicted_feature: str) -> dict:
        """
        Updates the shape probabilities using Bayes' rule.
        Returns a copy of the updated probabilities dictionary.
        """
        predicted_feature = predicted_feature.lower()
        if predicted_feature == "multiface_vertex":
            predicted_feature = "multi_face_vertex"
            
        if predicted_feature in self.SHAPE_LIKELIHOODS:
            likelihoods = self.SHAPE_LIKELIHOODS[predicted_feature]
            total_prob = 0
            
            # P(Shape | Feature) = P(Feature | Shape) * P(Shape)
            for shape in self.shape_probs:
                self.shape_probs[shape] *= likelihoods.get(shape, 0.01)
                total_prob += self.shape_probs[shape]
                
            # Normalize so they sum to 1.0
            if total_prob > 0:
                for shape in self.shape_probs:
                    self.shape_probs[shape] /= total_prob
                    
        return self.shape_probs.copy()
