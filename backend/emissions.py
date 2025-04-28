# Estimate CO₂ emitted vs saved

def calculate_emissions(ch4_percent, co2_percent, h2_yield):
    co2_emitted = h2_yield * (co2_percent/100) * 2.75
    co2_saved   = h2_yield * (ch4_percent/100) * 2.75
    return {'co2_emitted':co2_emitted,'co2_saved':co2_saved}
