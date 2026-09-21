"""Airline data analysis package."""
from airline.data.loading import load_data

def main():

    file_path = r"F:\October Plan Phases 1-7\__Projects__\U.S. Bureau of Transportation Statistics\U.S. Bureau of Transportation poject\AIRLINE_BTS.csv"

    df = load_data(file_path)

    required_columns = ['YEAR', 'QUARTER', 'MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'ORIGIN', 'ORIGIN_CITY_NAME',
                         'ORIGIN_STATE_ABR', 'DEST', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'CRS_DEP_TIME',
                         'DEP_TIME', 'DEP_DELAY', 'DEP_DELAY_NEW', 'TAXI_OUT', 'TAXI_IN', 'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY',
                         'ARR_DELAY_NEW', 'ARR_DEL15', 'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED', 'ACTUAL_ELAPSED_TIME',
                         'AIR_TIME', 'DISTANCE', 'CARRIER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']

    expected_types = {
                "YEAR": "int64", "QUARTER": "int64", "MONTH": "int64", "DAY_OF_WEEK": "int64", "FL_DATE": "object", "OP_UNIQUE_CARRIER": "object",
                "TAIL_NUM": "object","ORIGIN": "object", "ORIGIN_CITY_NAME": "object", "ORIGIN_STATE_ABR": "object", "DEST": "object", "DEST_CITY_NAME": "object",
                "DEST_STATE_ABR": "object", "CRS_DEP_TIME": "int64", "DEP_TIME": "float64",  "DEP_DELAY": "float64", "DEP_DELAY_NEW": "float64", "TAXI_OUT": "float64",
                "TAXI_IN": "float64", "CRS_ARR_TIME": "int64", "ARR_TIME": "float64", "ARR_DELAY": "float64", "ARR_DELAY_NEW": "float64", "ARR_DEL15": "float64",
                "CANCELLED": "float64", "CANCELLATION_CODE": "object", "DIVERTED": "float64", "ACTUAL_ELAPSED_TIME": "float64", "AIR_TIME": "float64", "DISTANCE": "float64",
                "CARRIER_DELAY": "float64", "NAS_DELAY": "float64", "SECURITY_DELAY": "float64",  "LATE_AIRCRAFT_DELAY": "float64"
                }

if __name__ == "__main__":

    main()

    