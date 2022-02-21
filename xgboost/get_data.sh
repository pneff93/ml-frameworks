curl --get 'https://data.cityofchicago.org/resource/wrvz-psew.csv' --data-urlencode '$limit=1000' --data-urlencode '$select=tips,trip_seconds,trip_miles,pickup_community_area,dropoff_community_area,fare,tolls,extras,trip_total' --data-urlencode '$where=trip_start_timestamp >= "2019-01-01" AND trip_start_timestamp < "2019-02-01"' | tr -d '"' > "data/data.csv"  # Removing unneeded quotes around all numbers