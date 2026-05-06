import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class RiskModel:

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=200)

        # Simulated realistic rural malaria dataset
        np.random.seed(42)
        X = np.random.rand(1000,7)

        # Simulated labels (0=Low,1=Moderate,2=High)
        y = []
        for row in X:
            score = row[0] + row[1] + row[6]
            if score > 1.8:
                y.append(2)
            elif score > 1.2:
                y.append(1)
            else:
                y.append(0)

        y = np.array(y)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, features):
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        return prediction