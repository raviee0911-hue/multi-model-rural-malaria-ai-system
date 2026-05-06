import numpy as np

species_names = ["Uninfected", "Parasitized"]
risk_levels = ["Low", "Medium", "High"]

def final_decision(image_prob, risk_probs):

    species = species_names[int(image_prob > 0.5)]
    infection_conf = float(image_prob)

    risk_index = np.argmax(risk_probs)
    risk_level = risk_levels[risk_index]
    risk_conf = float(np.max(risk_probs))

    # Final overall severity logic
    if species == "Parasitized" and risk_level == "High":
        final_status = "🚨 CRITICAL"
    elif species == "Parasitized":
        final_status = "⚠ MODERATE"
    else:
        final_status = "✅ SAFE"

    return {
        "species": species,
        "infection_confidence": infection_conf,
        "risk_level": risk_level,
        "risk_confidence": risk_conf,
        "final_status": final_status
    }