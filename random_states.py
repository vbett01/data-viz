import pandas as pd
import random
import numpy as np

# A larger set of states with some counties (not necessarily accurate, just for testing purposes)
states = [
    "California", "Texas", "New York", "Florida", "Illinois",
    "Pennsylvania", "Ohio", "Georgia", "North Carolina", "Michigan",
    "New Jersey", "Virginia", "Washington", "Arizona", "Massachusetts",
    "Tennessee", "Indiana", "Missouri", "Maryland", "Wisconsin"
]

counties_by_state = {
    "California": ["Los Angeles", "San Diego", "San Francisco", "Orange", "Sacramento", "Alameda", "Fresno", "Kern", "Ventura", "San Mateo"],
    "Texas": ["Harris", "Dallas", "Tarrant", "Travis", "Bexar", "Collin", "El Paso", "Fort Bend", "Hidalgo", "Montgomery"],
    "New York": ["Kings", "Queens", "Albany", "Suffolk", "Bronx", "Nassau", "Westchester", "Erie", "Monroe", "Onondaga"],
    "Florida": ["Miami-Dade", "Broward", "Palm Beach", "Orange", "Hillsborough", "Duval", "Pinellas", "Leon", "Sarasota", "Lee"],
    "Illinois": ["Cook", "DuPage", "Lake", "Will", "Kane", "McHenry", "Winnebago", "Madison", "Peoria", "Sangamon"],
    "Pennsylvania": ["Philadelphia", "Allegheny", "Montgomery", "Bucks", "Delaware", "Lancaster", "Chester", "York", "Berks", "Lehigh"],
    "Ohio": ["Cuyahoga", "Franklin", "Hamilton", "Summit", "Montgomery", "Lucas", "Stark", "Butler", "Lorain", "Mahoning"],
    "Georgia": ["Fulton", "Gwinnett", "Cobb", "DeKalb", "Clayton", "Chatham", "Cherokee", "Forsyth", "Henry", "Hall"],
    "North Carolina": ["Mecklenburg", "Wake", "Guilford", "Forsyth", "Cumberland", "Durham", "Buncombe", "Union", "New Hanover", "Gaston"],
    "Michigan": ["Wayne", "Oakland", "Macomb", "Kent", "Genesee", "Washtenaw", "Ingham", "Ottawa", "Kalamazoo", "Saginaw"],
    "New Jersey": ["Bergen", "Middlesex", "Essex", "Hudson", "Monmouth", "Ocean", "Union", "Camden", "Passaic", "Morris"],
    "Virginia": ["Fairfax", "Prince William", "Loudoun", "Chesterfield", "Henrico", "Arlington", "Stafford", "Spotsylvania", "Albemarle", "Hampton"],
    "Washington": ["King", "Pierce", "Snohomish", "Spokane", "Clark", "Thurston", "Kitsap", "Yakima", "Whatcom", "Benton"],
    "Arizona": ["Maricopa", "Pima", "Pinal", "Yavapai", "Yuma", "Mohave", "Coconino", "Cochise", "Navajo", "Apache"],
    "Massachusetts": ["Middlesex", "Worcester", "Suffolk", "Essex", "Norfolk", "Plymouth", "Bristol", "Barnstable", "Hampden", "Hampshire"],
    "Tennessee": ["Shelby", "Davidson", "Knox", "Hamilton", "Rutherford", "Williamson", "Montgomery", "Sumner", "Sullivan", "Blount"],
    "Indiana": ["Marion", "Lake", "Allen", "Hamilton", "St. Joseph", "Elkhart", "Tippecanoe", "Vanderburgh", "Porter", "Hendricks"],
    "Missouri": ["St. Louis", "Jackson", "St. Charles", "Greene", "Clay", "Jefferson", "Boone", "Jasper", "Cole", "Franklin"],
    "Maryland": ["Montgomery", "Prince George's", "Baltimore", "Anne Arundel", "Howard", "Harford", "Frederick", "Carroll", "Charles", "Washington"],
    "Wisconsin": ["Milwaukee", "Dane", "Waukesha", "Brown", "Racine", "Outagamie", "Winnebago", "Kenosha", "Rock", "Marathon"]
}

municipalities = ["City", "Borough", "Township", "Village"]
districts = ["North", "East", "South", "West"]
risk_indicators = ["Flood Risk", "Earthquake Risk", "Fire Risk", "Crime Rate", "Hurricane Risk"]

num_records = 2000  # number of records to generate, adjust as needed

rows = []
for _ in range(num_records):
    state = random.choice(states)
    county = random.choice(counties_by_state[state])
    municipality = random.choice(municipalities)
    district = random.choice(districts)
    risk_indicator = random.choice(risk_indicators)
    # Generate a threshold and risk value
    threshold = round(random.uniform(50, 150), 2)
    risk_value = round(random.uniform(30, 200), 2)
    rows.append([state, county, municipality, district, risk_indicator, threshold, risk_value])

df = pd.DataFrame(rows, columns=["State", "County", "Municipality", "District", "Key Risk Indicator", "Threshold", "Risk Value"])
df.to_csv("extended_test_data.csv", index=False)

print("extended_test_data.csv created with", num_records, "rows.")
