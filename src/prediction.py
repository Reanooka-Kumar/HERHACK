import os
import joblib
import json

try:
    from data_preprocessing import preprocess_single_student
except ImportError:
    from src.data_preprocessing import preprocess_single_student

class PlacementPredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.classifier = None
        self.regressor = None
        self.scaler = None
        self.feature_names = None
        self.metrics = {}
        self.load_assets()
        
    def load_assets(self):
        c_path = os.path.join(self.models_dir, "best_classifier.joblib")
        r_path = os.path.join(self.models_dir, "salary_regressor.joblib")
        s_path = os.path.join(self.models_dir, "scaler.joblib")
        f_path = os.path.join(self.models_dir, "feature_names.joblib")
        m_path = os.path.join(self.models_dir, "model_metrics.json")
        
        if os.path.exists(c_path): self.classifier = joblib.load(c_path)
        if os.path.exists(r_path): self.regressor = joblib.load(r_path)
        if os.path.exists(s_path): self.scaler = joblib.load(s_path)
        if os.path.exists(f_path): self.feature_names = joblib.load(f_path)
        if os.path.exists(m_path):
            with open(m_path, "r") as f: self.metrics = json.load(f)
                
    def is_ready(self):
        """Checks if all required model assets are loaded and attempts auto-reload."""
        if self.classifier is None or self.regressor is None or self.scaler is None or self.feature_names is None:
            self.load_assets()
        return (self.classifier is not None and 
                self.regressor is not None and 
                self.scaler is not None and 
                self.feature_names is not None)

    def predict_student(self, student_dict):
        if not self.is_ready():
            self.load_assets()
            if not self.is_ready():
                return {'error': 'Models not trained. Please run model_training.py first.'}
                
        X_scaled, mapped_features, feature_names = preprocess_single_student(student_dict, self.scaler)
        
        pred_class = self.classifier.predict(X_scaled)[0]
        placed_prob = float(self.classifier.predict_proba(X_scaled)[0][1])
        status = "Placed" if pred_class == 1 else "Not Placed"
        expected_salary = float(self.regressor.predict(X_scaled)[0])
        
        residual_std = self.metrics.get('salary_metrics', {}).get('residual_std', 1.2)
        margin = 1.96 * residual_std
        
        return {
            'placement_status': status,
            'placement_probability': placed_prob * 100.0,
            'confidence_score': float(max(placed_prob, 1.0 - placed_prob) * 100.0),
            'expected_salary': expected_salary,
            'salary_range_min': max(2.0, expected_salary - margin),
            'salary_range_max': expected_salary + margin,
            'mapped_features': mapped_features,
            'scaled_features_array': X_scaled,
            'feature_names': feature_names
        }
