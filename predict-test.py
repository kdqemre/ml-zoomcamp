#!/usr/bin/env python
# coding: utf-8


import requests



url = 'http://localhost:9696/predict'




pump = {
  "amount_tsh": 500.0,
  "gps_height": 1743,
  "longitude": 38.21381371,
  "latitude": -4.5568768,
  "basin": "pangani",
  "population": 300,
  "public_meeting": "true",
  "scheme_management": "vwc",
  "permit": "true",
  "construction_year": 2012,
  "extraction_type": "gravity",
  "extraction_type_class": "gravity",
  "management_group": "user-group",
  "payment_type": "monthly",
  "water_quality": "soft",
  "quantity": "enough",
  "source": "spring",
  "source_class": "groundwater",
  "waterpoint_type": "communal_standpipe"
}




response = requests.post(url, json=pump).json()
print(response,'\n')








